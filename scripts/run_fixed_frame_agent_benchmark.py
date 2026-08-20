#!/usr/bin/env python3
"""Run repeat-aware SAM3-agent first passes on a fixed frame manifest.

The expensive agent runner accepts a video, not an arbitrary frame.  This
wrapper extracts each fixed frame into a one-frame MP4, invokes the existing
runner once per frame/repeat, and records enough metadata to resume safely.
It deliberately does not score the results; visual/scored analysis is a
separate step.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--manifest",
        default="configs/seatube_meagan_five_frames_v1.json",
    )
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--model", default="claude-sonnet-4-6")
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--prompt", default="all visible marine life")
    parser.add_argument("--prompt-profile", default="underwater")
    parser.add_argument("--max-completion-tokens", type=int, default=2048)
    parser.add_argument("--max-generations", type=int, default=20)
    parser.add_argument("--confidence-threshold", type=float, default=0.40)
    parser.add_argument(
        "--proposal-prompt",
        action="append",
        default=[],
        help=(
            "Override the per-frame deterministic proposal bank with this phrase; "
            "may be repeated."
        ),
    )
    parser.add_argument(
        "--proposal-exclude-region",
        action="append",
        default=[],
        metavar="X1,Y1,X2,Y2",
        help=(
            "Override per-frame normalized logo/overlay exclusions; may be repeated."
        ),
    )
    parser.add_argument("--proposal-exclusion-overlap", type=float, default=0.80)
    parser.add_argument("--proposal-iom-threshold", type=float, default=0.30)
    parser.add_argument(
        "--proposal-fragment-merge-prompt",
        action="append",
        default=[],
        help="Override per-frame evidence-backed fragment-stitch prompt; may repeat.",
    )
    parser.add_argument(
        "--proposal-fragment-bbox-iom-threshold", type=float, default=0.15
    )
    parser.add_argument("--temporal-context-offset-seconds", type=float, default=0.50)
    parser.add_argument(
        "--no-temporal-context",
        action="store_true",
        help="Do not include adjacent source-video frames in the verification request.",
    )
    parser.add_argument(
        "--frame-id",
        action="append",
        default=[],
        help="Run only this frame id; repeat for multiple ids.",
    )
    parser.add_argument("--resume", action="store_true")
    parser.add_argument(
        "--prepare-only",
        action="store_true",
        help="Create one-frame inputs and metadata without making API calls.",
    )
    return parser.parse_args()


def read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(value, handle, indent=2)
        handle.write("\n")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def git_state(repo_root: Path) -> dict[str, Any]:
    def run(*args: str) -> str:
        proc = subprocess.run(
            ["git", *args],
            cwd=repo_root,
            check=False,
            capture_output=True,
            text=True,
        )
        return proc.stdout.strip() if proc.returncode == 0 else "unknown"

    status = run("status", "--porcelain")
    return {
        "sha": run("rev-parse", "HEAD"),
        "dirty": bool(status and status != "unknown"),
    }


def select_records(
    manifest: dict[str, Any], requested: set[str]
) -> list[dict[str, Any]]:
    records = [
        row
        for row in manifest.get("frames", [])
        if not requested or str(row["id"]) in requested
    ]
    present = {str(row["id"]) for row in records}
    missing = requested - present
    if missing:
        raise SystemExit(f"frame ids not found in manifest: {sorted(missing)}")
    if not records:
        raise SystemExit("manifest selected no frames")
    return records


def materialize_one_frame_video(record: dict[str, Any], output: Path) -> dict[str, Any]:
    import cv2

    source = Path(str(record["video"]))
    if not source.is_file():
        raise FileNotFoundError(source)
    capture = cv2.VideoCapture(str(source))
    if not capture.isOpened():
        raise RuntimeError(f"could not open {source}")
    fps = float(capture.get(cv2.CAP_PROP_FPS) or 0.0)
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
    frame_index = int(record["frame_index"])
    capture.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
    ok, frame = capture.read()
    capture.release()
    if not ok or fps <= 0 or width <= 0 or height <= 0:
        raise RuntimeError(f"could not decode frame {frame_index} from {source}")
    decoded_seconds = frame_index / fps
    expected_seconds = float(record["time_seconds"])
    if abs(decoded_seconds - expected_seconds) > 0.05:
        raise ValueError(
            f"time mismatch for {record['id']}: decoded={decoded_seconds:.6f}, "
            f"manifest={expected_seconds:.6f}"
        )

    output.parent.mkdir(parents=True, exist_ok=True)
    if not output.exists():
        writer = cv2.VideoWriter(
            str(output),
            cv2.VideoWriter_fourcc(*"mp4v"),
            fps,
            (width, height),
        )
        if not writer.isOpened():
            raise RuntimeError(f"could not create {output}")
        writer.write(frame)
        writer.release()
    return {
        "source_video": str(source),
        "source_frame_index": frame_index,
        "source_fps": fps,
        "decoded_seconds": decoded_seconds,
        "frame_size_hw": [height, width],
        "input_video": str(output),
        "input_sha256": sha256(output),
    }


def materialize_temporal_context(
    record: dict[str, Any], output: Path, offset_seconds: float
) -> dict[str, Any]:
    """Create a vertical before/after strip for one fixed-frame review."""
    import cv2
    import numpy as np

    if offset_seconds <= 0:
        raise ValueError("temporal context offset must be positive")
    source = Path(str(record["video"]))
    capture = cv2.VideoCapture(str(source))
    if not capture.isOpened():
        raise RuntimeError(f"could not open {source}")
    fps = float(capture.get(cv2.CAP_PROP_FPS) or 0.0)
    total_frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    center_index = int(record["frame_index"])
    delta = max(1, int(round(offset_seconds * fps)))
    indices = [max(0, center_index - delta), min(total_frames - 1, center_index + delta)]
    panels = []
    decoded_indices = []
    for label, frame_index in zip(("BEFORE", "AFTER"), indices):
        capture.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
        ok, frame = capture.read()
        if not ok:
            capture.release()
            raise RuntimeError(f"could not decode frame {frame_index} from {source}")
        scale = min(1.0, 768.0 / float(frame.shape[1]))
        if scale < 1.0:
            frame = cv2.resize(
                frame,
                (int(round(frame.shape[1] * scale)), int(round(frame.shape[0] * scale))),
                interpolation=cv2.INTER_AREA,
            )
        cv2.rectangle(frame, (0, 0), (210, 42), (0, 0, 0), -1)
        cv2.putText(
            frame,
            f"{label}  t={frame_index / fps:.2f}s",
            (10, 29),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )
        panels.append(frame)
        decoded_indices.append(frame_index)
    capture.release()
    width = min(panel.shape[1] for panel in panels)
    panels = [panel[:, :width] for panel in panels]
    collage = np.vstack(panels)
    output.parent.mkdir(parents=True, exist_ok=True)
    if not output.exists() and not cv2.imwrite(str(output), collage):
        raise RuntimeError(f"could not create {output}")
    return {
        "temporal_context_image": str(output),
        "temporal_context_sha256": sha256(output),
        "temporal_context_frame_indices": decoded_indices,
        "temporal_context_offset_seconds": float(offset_seconds),
    }


def valid_cached_run(path: Path, model: str) -> bool:
    summary_path = path / "summary.json"
    outputs_path = path / "frame_outputs_rle.json"
    if not summary_path.is_file() or not outputs_path.is_file():
        return False
    try:
        summary = read_json(summary_path)
    except (OSError, ValueError, TypeError):
        return False
    return (
        str(summary.get("model")) == model
        and int(summary.get("processed_frames", 0)) == 1
        and int(summary.get("error_count", 0)) == 0
    )


def main() -> int:
    args = parse_args()
    if args.repeats < 1:
        raise SystemExit("--repeats must be at least 1")
    repo_root = Path(__file__).resolve().parents[1]
    # Secrets stay on WSL.  Loading here lets child agent processes inherit the
    # key without placing it in a command line, log, or run metadata.
    from dotenv import load_dotenv

    load_dotenv(repo_root / ".env")
    manifest_path = Path(args.manifest)
    if not manifest_path.is_absolute():
        manifest_path = repo_root / manifest_path
    manifest = read_json(manifest_path.resolve())
    records = select_records(manifest, set(args.frame_id))
    output_root = Path(args.output_dir)
    if not output_root.is_absolute():
        output_root = repo_root / output_root
    output_root = output_root.resolve()
    output_root.mkdir(parents=True, exist_ok=True)

    inputs: dict[str, dict[str, Any]] = {}
    for record in records:
        frame_id = str(record["id"])
        inputs[frame_id] = materialize_one_frame_video(
            record, output_root / "inputs" / f"{frame_id}.mp4"
        )
        if not args.no_temporal_context:
            inputs[frame_id].update(
                materialize_temporal_context(
                    record,
                    output_root / "inputs" / f"{frame_id}_temporal_context.jpg",
                    float(args.temporal_context_offset_seconds),
                )
            )

    proposal_prompts_by_frame: dict[str, list[str]] = {}
    proposal_exclusions_by_frame: dict[str, list[str]] = {}
    fragment_merge_prompts_by_frame: dict[str, list[str]] = {}
    for record in records:
        frame_id = str(record["id"])
        proposal_prompts_by_frame[frame_id] = list(
            dict.fromkeys(
                str(value).strip()
                for value in (
                    args.proposal_prompt
                    or record.get("sam3_proposal_prompts", [])
                )
                if str(value).strip()
            )
        )
        if args.proposal_exclude_region:
            proposal_exclusions_by_frame[frame_id] = list(
                dict.fromkeys(args.proposal_exclude_region)
            )
        else:
            proposal_exclusions_by_frame[frame_id] = [
                ",".join(str(coord) for coord in region)
                for region in record.get(
                    "proposal_exclusion_regions_xyxy_normalized", []
                )
            ]
        if not proposal_prompts_by_frame[frame_id]:
            raise ValueError(
                f"no deterministic proposal prompts configured for {frame_id}"
            )
        fragment_merge_prompts_by_frame[frame_id] = list(
            dict.fromkeys(
                str(value).strip()
                for value in (
                    args.proposal_fragment_merge_prompt
                    or record.get("sam3_fragment_merge_prompts", [])
                )
                if str(value).strip()
            )
        )

    runner = repo_root / "scripts" / "run_sam3_agent_every_frame_video.py"
    metadata: dict[str, Any] = {
        "schema_version": 1,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "benchmark_id": manifest.get("benchmark_id"),
        "exploratory_not_scored": True,
        "reason_not_scored": "The inherited source checkout is not committed cleanly.",
        "git": git_state(repo_root),
        "source_hashes": {
            str(runner.relative_to(repo_root)): sha256(runner),
            str(Path(__file__).resolve().relative_to(repo_root)): sha256(
                Path(__file__).resolve()
            ),
            "sam3/agent/agent_core.py": sha256(
                repo_root / "sam3" / "agent" / "agent_core.py"
            ),
            "sam3/agent/proposal_bank.py": sha256(
                repo_root / "sam3" / "agent" / "proposal_bank.py"
            ),
            "sam3/agent/system_prompts/system_prompt_proposal_verification.txt": sha256(
                repo_root
                / "sam3"
                / "agent"
                / "system_prompts"
                / "system_prompt_proposal_verification.txt"
            ),
            str(manifest_path.resolve()): sha256(manifest_path.resolve()),
        },
        "configuration": {
            "model": args.model,
            "repeats": args.repeats,
            "prompt": args.prompt,
            "prompt_profile": args.prompt_profile,
            "max_completion_tokens": args.max_completion_tokens,
            "max_generations": args.max_generations,
            "confidence_threshold": args.confidence_threshold,
            "persistent_masks": True,
            "proposal_prompts_by_frame": proposal_prompts_by_frame,
            "proposal_exclusions_by_frame": proposal_exclusions_by_frame,
            "proposal_exclusion_overlap": args.proposal_exclusion_overlap,
            "proposal_iom_threshold": args.proposal_iom_threshold,
            "fragment_merge_prompts_by_frame": fragment_merge_prompts_by_frame,
            "proposal_fragment_bbox_iom_threshold": (
                args.proposal_fragment_bbox_iom_threshold
            ),
            "verification_only": True,
            "temporal_context": not args.no_temporal_context,
            "temporal_context_offset_seconds": args.temporal_context_offset_seconds,
        },
        "inputs": inputs,
        "runs": [],
    }
    write_json(output_root / "benchmark_metadata.json", metadata)
    if args.prepare_only:
        print(f"Prepared {len(records)} frame input(s) in {output_root}")
        return 0

    for repeat in range(1, args.repeats + 1):
        for record in records:
            frame_id = str(record["id"])
            frame_output = output_root / f"repeat_{repeat}" / frame_id
            if args.resume and valid_cached_run(frame_output, args.model):
                print(f"[repeat {repeat}] [{frame_id}] cached", flush=True)
                metadata["runs"].append(
                    {"repeat": repeat, "frame_id": frame_id, "status": "cached"}
                )
                write_json(output_root / "benchmark_metadata.json", metadata)
                continue

            command = [
                sys.executable,
                str(runner),
                inputs[frame_id]["input_video"],
                "--llm-provider",
                "claude",
                "--claude-model",
                args.model,
                "--prompt",
                args.prompt,
                "--prompt-profile",
                args.prompt_profile,
                "--max-completion-tokens",
                str(args.max_completion_tokens),
                "--max-generations",
                str(args.max_generations),
                "--confidence-threshold",
                str(args.confidence_threshold),
                "--image-detail",
                "high",
                "--max-frames",
                "1",
                "--keep-artifacts",
                "--verification-only",
                "--proposal-exclusion-overlap",
                str(args.proposal_exclusion_overlap),
                "--proposal-iom-threshold",
                str(args.proposal_iom_threshold),
                "--proposal-fragment-bbox-iom-threshold",
                str(args.proposal_fragment_bbox_iom_threshold),
                "--output-dir",
                str(frame_output),
            ]
            for proposal_prompt in proposal_prompts_by_frame[frame_id]:
                command.extend(["--proposal-prompt", proposal_prompt])
            for exclusion_region in proposal_exclusions_by_frame[frame_id]:
                command.extend(["--proposal-exclude-region", exclusion_region])
            for fragment_prompt in fragment_merge_prompts_by_frame[frame_id]:
                command.extend(
                    ["--proposal-fragment-merge-prompt", fragment_prompt]
                )
            context_image = inputs[frame_id].get("temporal_context_image")
            if context_image:
                command.extend(["--verification-context-image", str(context_image)])
            print(f"[repeat {repeat}] [{frame_id}] running", flush=True)
            started = datetime.now(timezone.utc).isoformat()
            proc = subprocess.run(command, cwd=repo_root, check=False)
            status = "complete" if proc.returncode == 0 else "failed"
            metadata["runs"].append(
                {
                    "repeat": repeat,
                    "frame_id": frame_id,
                    "status": status,
                    "returncode": proc.returncode,
                    "started_utc": started,
                    "finished_utc": datetime.now(timezone.utc).isoformat(),
                    "output_dir": str(frame_output),
                }
            )
            write_json(output_root / "benchmark_metadata.json", metadata)
            if proc.returncode != 0 or not valid_cached_run(frame_output, args.model):
                raise RuntimeError(
                    f"agent run failed validation for repeat {repeat}, {frame_id}"
                )

    print(f"Wrote {output_root / 'benchmark_metadata.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
