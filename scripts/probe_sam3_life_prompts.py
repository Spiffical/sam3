#!/usr/bin/env python3
"""Probe SAM3 text retrieval for fauna, plant-like, and WoRMS phrases.

This diagnostic bypasses the MLLM agent entirely.  It encodes each fixed frame
once, applies a controlled prompt bank, and records detections at several score
thresholds.  Per-clip WoRMS labels are used only as SAM3 text prompts; they are
not treated as ground truth and are never assigned to returned masks.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import html
import json
import os
import re
import subprocess
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


GENERIC_PROMPT_GROUPS: dict[str, tuple[str, ...]] = {
    "generic_creature": (
        "small creatures",
        "creatures",
        "animals",
        "marine animals",
    ),
    "generic_life": (
        "marine life",
        "underwater life",
        "living organisms",
        "marine organisms",
        "benthic organisms",
    ),
    "plant_language": (
        "plants",
        "underwater plants",
        "marine plants",
        "aquatic plants",
        "vegetation",
        "seaweed",
        "algae",
    ),
    "sessile_morphology": (
        "coral",
        "coral colony",
        "branching coral",
        "sea fan",
        "gorgonian",
        "sponge",
        "anemone",
        "tube worm",
    ),
}

# Keep the production fallback deliberately small.  The larger matrix above is
# useful for diagnosis, but most of its phrases produced no additional masks on
# the five SeaTube presentation frames.  These five cover the one historically
# strong generic prompt plus the sessile morphologies that matter in the newer
# clips.
PRODUCTION_GENERIC_PROMPTS: tuple[str, ...] = (
    "small creatures",
    "coral",
    "branching coral",
    "sponge",
    "anemone",
)


@dataclass(frozen=True)
class PromptSpec:
    text: str
    group: str
    source_taxon: str = ""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--manifest", default="configs/seatube_meagan_five_frames_v1.json"
    )
    parser.add_argument(
        "--selection-manifest",
        default=(
            "/home/sbialek/ONC/seatube-downloader/downloads/"
            "sam3_seatube_matching_v1/meagan_five_clips_v1/selection_manifest.json"
        ),
    )
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--frame-id", action="append", default=[])
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--checkpoint-path", default="")
    parser.add_argument("--min-threshold", type=float, default=0.20)
    parser.add_argument("--report-thresholds", default="0.20,0.30,0.40")
    parser.add_argument("--render-threshold", type=float, default=0.40)
    parser.add_argument(
        "--extra-prompt",
        action="append",
        default=[],
        help="Additional SAM3 text prompt to test; may be repeated.",
    )
    parser.add_argument(
        "--only-extra-prompts",
        action="store_true",
        help="Test only --extra-prompt values instead of the built-in matrix.",
    )
    parser.add_argument("--resume", action="store_true")
    return parser.parse_args()


def read_json(path: Path) -> Any:
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

    return {
        "sha": run("rev-parse", "HEAD"),
        "dirty": bool(run("status", "--porcelain")),
    }


def clean_taxon_label(value: str) -> str:
    return value.split(" | ID:", 1)[0].strip()


def expand_taxon_prompts(label: str) -> list[str]:
    """Return exact, scientific, and parenthetical common-name variants."""
    clean = clean_taxon_label(label)
    if not clean:
        return []
    values = [clean]
    match = re.fullmatch(r"(.+?)\s*\((.+)\)", clean)
    if match:
        values.append(match.group(1).strip())
        values.extend(part.strip() for part in match.group(2).split(";") if part.strip())
    return dedupe_strings(values)


def dedupe_strings(values: list[str]) -> list[str]:
    seen: set[str] = set()
    result: list[str] = []
    for value in values:
        normalized = " ".join(value.split())
        key = normalized.casefold()
        if normalized and key not in seen:
            seen.add(key)
            result.append(normalized)
    return result


def prompt_bank(taxon_labels: list[str]) -> list[PromptSpec]:
    specs: list[PromptSpec] = []
    for group, prompts in GENERIC_PROMPT_GROUPS.items():
        specs.extend(PromptSpec(text=prompt, group=group) for prompt in prompts)
    for label in dedupe_strings(taxon_labels):
        clean = clean_taxon_label(label)
        specs.extend(
            PromptSpec(text=prompt, group="worms_taxon", source_taxon=clean)
            for prompt in expand_taxon_prompts(label)
        )

    seen: set[str] = set()
    unique: list[PromptSpec] = []
    for spec in specs:
        key = spec.text.casefold()
        if key not in seen:
            seen.add(key)
            unique.append(spec)
    return unique


def add_extra_prompts(
    specs: list[PromptSpec], extra_prompts: list[str], only_extra: bool = False
) -> list[PromptSpec]:
    """Append ad-hoc diagnostic prompts while preserving casefolded uniqueness."""
    combined = [] if only_extra else list(specs)
    combined.extend(
        PromptSpec(text=prompt, group="extra_diagnostic")
        for prompt in dedupe_strings(extra_prompts)
    )
    seen: set[str] = set()
    unique: list[PromptSpec] = []
    for spec in combined:
        key = spec.text.casefold()
        if key not in seen:
            seen.add(key)
            unique.append(spec)
    return unique


def production_prompt_bank(
    taxon_labels: list[str], mode: str = "compact"
) -> list[PromptSpec]:
    """Build the taxon-first prompt bank used before click recovery.

    ``compact`` uses each annotation's scientific name and its parenthetical
    common-name variants, followed by five proven generic/morphology fallbacks.
    The original full display string is excluded because it is redundant with
    those cleaner variants.  ``broad`` retains the same taxon-first ordering but
    appends the complete diagnostic generic bank.
    """
    if mode not in {"compact", "broad"}:
        raise ValueError(f"unknown production prompt mode: {mode}")

    specs: list[PromptSpec] = []
    for label in dedupe_strings(taxon_labels):
        clean = clean_taxon_label(label)
        expanded = expand_taxon_prompts(label)
        variants = expanded[1:] if len(expanded) > 1 else expanded
        for index, prompt in enumerate(variants):
            group = (
                "worms_taxon_scientific" if index == 0
                else "worms_taxon_common"
            )
            specs.append(
                PromptSpec(text=prompt, group=group, source_taxon=clean)
            )

    if mode == "compact":
        specs.extend(
            PromptSpec(text=prompt, group="generic_production")
            for prompt in PRODUCTION_GENERIC_PROMPTS
        )
    else:
        specs.extend(
            PromptSpec(text=prompt, group=group)
            for group, prompts in GENERIC_PROMPT_GROUPS.items()
            for prompt in prompts
        )

    seen: set[str] = set()
    unique: list[PromptSpec] = []
    for spec in specs:
        key = spec.text.casefold()
        if key not in seen:
            seen.add(key)
            unique.append(spec)
    return unique


def safe_slug(value: str) -> str:
    slug = re.sub(r"[^a-z0-9]+", "_", value.casefold()).strip("_")
    return slug[:80] or "prompt"


def resolve_path(repo_root: Path, value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else (repo_root / path).resolve()


def find_bpe_path(repo_root: Path) -> str:
    env_path = os.environ.get("SAM3_BPE_PATH")
    if env_path and Path(env_path).is_file():
        return env_path
    for path in (
        repo_root / "assets" / "bpe_simple_vocab_16e6.txt.gz",
        repo_root / "sam3" / "assets" / "bpe_simple_vocab_16e6.txt.gz",
    ):
        if path.is_file():
            return str(path)
    raise FileNotFoundError("Could not find SAM3 BPE vocabulary")


def decode_fixed_frame(record: dict[str, Any], output_path: Path) -> dict[str, Any]:
    import cv2

    video = Path(str(record["video"]))
    capture = cv2.VideoCapture(str(video))
    if not capture.isOpened():
        raise RuntimeError(f"could not open {video}")
    fps = float(capture.get(cv2.CAP_PROP_FPS) or 0.0)
    frame_index = int(record["frame_index"])
    capture.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
    ok, frame = capture.read()
    capture.release()
    if not ok or fps <= 0:
        raise RuntimeError(f"could not decode frame {frame_index} from {video}")
    decoded_seconds = frame_index / fps
    expected_seconds = float(record["time_seconds"])
    if abs(decoded_seconds - expected_seconds) > 0.05:
        raise ValueError(
            f"frame time mismatch: decoded={decoded_seconds}, expected={expected_seconds}"
        )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if not output_path.exists() and not cv2.imwrite(str(output_path), frame):
        raise RuntimeError(f"could not write {output_path}")
    return {
        "video": str(video),
        "frame_index": frame_index,
        "fps": fps,
        "decoded_seconds": decoded_seconds,
        "frame_size_hw": list(frame.shape[:2]),
    }


def serialize_state(state: dict[str, Any], width: int, height: int) -> dict[str, Any]:
    import torch
    from sam3.model.box_ops import box_xyxy_to_xywh
    from sam3.train.masks_ops import rle_encode

    boxes_xyxy = state["boxes"] / torch.tensor(
        [width, height, width, height], device=state["boxes"].device
    )
    return {
        "orig_img_h": height,
        "orig_img_w": width,
        "pred_boxes": box_xyxy_to_xywh(boxes_xyxy).tolist(),
        "pred_masks": [row["counts"] for row in rle_encode(state["masks"].squeeze(1))],
        "pred_scores": [float(value) for value in state["scores"].tolist()],
    }


def filter_outputs(outputs: dict[str, Any], threshold: float) -> dict[str, Any]:
    keep = [
        index
        for index, score in enumerate(outputs.get("pred_scores", []))
        if float(score) > threshold
    ]
    filtered = dict(outputs)
    for key in ("pred_boxes", "pred_masks", "pred_scores"):
        filtered[key] = [outputs[key][index] for index in keep]
    return filtered


def render_overlay(
    image_path: Path,
    outputs: dict[str, Any],
    threshold: float,
    output_path: Path,
) -> None:
    from PIL import Image
    from sam3.agent.viz import visualize

    filtered = filter_outputs(outputs, threshold)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if filtered["pred_masks"]:
        payload = {"original_image_path": str(image_path), **filtered}
        visualize(payload, mask_alpha=0.28).save(output_path)
    else:
        Image.open(image_path).convert("RGB").save(output_path)


def threshold_key(value: float) -> str:
    return f"count_at_{value:.2f}".replace(".", "p")


def build_taxa_by_clip(selection: dict[str, Any]) -> dict[str, list[str]]:
    result: dict[str, list[str]] = {}
    for clip in selection.get("clips", []):
        result[str(clip["clip_id"])] = dedupe_strings(
            [
                str(annotation.get("taxon_display_text") or "")
                for annotation in clip.get("annotations", [])
            ]
        )
    return result


def write_csv(path: Path, rows: list[dict[str, Any]], thresholds: list[float]) -> None:
    fields = [
        "frame_id",
        "prompt_index",
        "prompt",
        "group",
        "source_taxon",
        *[threshold_key(value) for value in thresholds],
        "top_score",
        "overlay",
        "error",
    ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows({key: row.get(key, "") for key in fields} for row in rows)


def write_gallery(path: Path, rows: list[dict[str, Any]], thresholds: list[float]) -> None:
    parts = [
        "<!doctype html><meta charset='utf-8'>",
        "<title>SAM3 life-prompt probe</title>",
        "<style>body{font:14px system-ui;background:#111;color:#eee}"
        "table{border-collapse:collapse;margin-bottom:36px}td,th{border:1px solid #555;"
        "padding:5px;vertical-align:top}img{width:360px;height:auto}code{color:#9ef}</style>",
        "<h1>SAM3 direct life-prompt probe</h1>",
        "<p>No MLLM calls. Counts are detections, not ground truth.</p>",
    ]
    for frame_id in sorted({str(row["frame_id"]) for row in rows}):
        parts.append(f"<h2>{html.escape(frame_id)}</h2><table><tr>")
        parts.extend(["<th>group / prompt</th>"])
        parts.extend(f"<th>{value:.2f}</th>" for value in thresholds)
        parts.append("<th>overlay at render threshold</th></tr>")
        for row in [item for item in rows if item["frame_id"] == frame_id]:
            parts.append("<tr><td>")
            parts.append(
                f"{html.escape(str(row['group']))}<br><code>"
                f"{html.escape(str(row['prompt']))}</code>"
            )
            if row.get("error"):
                parts.append(f"<br>{html.escape(str(row['error']))}")
            parts.append("</td>")
            parts.extend(
                f"<td>{row.get(threshold_key(value), '')}</td>" for value in thresholds
            )
            overlay = html.escape(str(row.get("overlay") or ""))
            parts.append(f"<td><img loading='lazy' src='{overlay}'></td></tr>")
        parts.append("</table>")
    path.write_text("".join(parts), encoding="utf-8")


def main() -> int:
    args = parse_args()
    repo_root = Path(__file__).resolve().parents[1]
    manifest_path = resolve_path(repo_root, args.manifest)
    selection_path = resolve_path(repo_root, args.selection_manifest)
    output_root = resolve_path(repo_root, args.output_dir)
    output_root.mkdir(parents=True, exist_ok=True)

    thresholds = sorted(
        {float(value.strip()) for value in args.report_thresholds.split(",") if value.strip()}
    )
    if not thresholds or args.min_threshold > min(thresholds):
        raise SystemExit("--min-threshold must be <= every report threshold")
    if args.render_threshold < args.min_threshold:
        raise SystemExit("--render-threshold must be >= --min-threshold")

    manifest = read_json(manifest_path)
    requested = set(args.frame_id)
    records = [
        row
        for row in manifest.get("frames", [])
        if not requested or str(row["id"]) in requested
    ]
    missing = requested - {str(row["id"]) for row in records}
    if missing:
        raise SystemExit(f"unknown frame ids: {sorted(missing)}")
    taxa_by_clip = build_taxa_by_clip(read_json(selection_path))

    frame_inputs: dict[str, dict[str, Any]] = {}
    specs_by_frame: dict[str, list[PromptSpec]] = {}
    for record in records:
        frame_id = str(record["id"])
        image_path = output_root / "frames" / f"{frame_id}.jpg"
        frame_inputs[frame_id] = {
            **decode_fixed_frame(record, image_path),
            "image": str(image_path),
        }
        specs_by_frame[frame_id] = add_extra_prompts(
            prompt_bank(taxa_by_clip.get(frame_id, [])),
            args.extra_prompt,
            only_extra=args.only_extra_prompts,
        )

    cache_paths = [
        output_root / "results" / frame_id / f"{index:03d}_{safe_slug(spec.text)}.json"
        for frame_id, specs in specs_by_frame.items()
        for index, spec in enumerate(specs, start=1)
    ]
    needs_model = not args.resume or any(not path.is_file() for path in cache_paths)

    processor = None
    if needs_model:
        from sam3.model.sam3_image_processor import Sam3Processor
        from sam3.model_builder import build_sam3_image_model

        model = build_sam3_image_model(
            bpe_path=find_bpe_path(repo_root),
            device=args.device,
            checkpoint_path=(args.checkpoint_path or None),
            compile=False,
        )
        processor = Sam3Processor(
            model,
            device=args.device,
            confidence_threshold=args.min_threshold,
        )

    rows: list[dict[str, Any]] = []
    for record in records:
        frame_id = str(record["id"])
        image_path = Path(frame_inputs[frame_id]["image"])
        from PIL import Image

        image = Image.open(image_path).convert("RGB")
        state = processor.set_image(image) if processor is not None else None
        for index, spec in enumerate(specs_by_frame[frame_id], start=1):
            stem = f"{index:03d}_{safe_slug(spec.text)}"
            result_path = output_root / "results" / frame_id / f"{stem}.json"
            overlay_path = output_root / "overlays" / frame_id / f"{stem}.png"
            cached = read_json(result_path) if args.resume and result_path.is_file() else None
            if cached is None:
                if processor is None or state is None:
                    raise RuntimeError("SAM3 processor was not initialized")
                try:
                    processor.reset_all_prompts(state)
                    state = processor.set_text_prompt(prompt=spec.text, state=state)
                    raw_outputs = serialize_state(state, image.width, image.height)
                    from sam3.agent.client_sam3 import remove_overlapping_masks

                    outputs = remove_overlapping_masks(raw_outputs)
                    order = sorted(
                        range(len(outputs["pred_scores"])),
                        key=lambda item: outputs["pred_scores"][item],
                        reverse=True,
                    )
                    for key in ("pred_boxes", "pred_masks", "pred_scores"):
                        outputs[key] = [outputs[key][item] for item in order]
                    cached = {
                        "frame_id": frame_id,
                        "prompt_index": index,
                        "prompt": asdict(spec),
                        "min_threshold": args.min_threshold,
                        "outputs": outputs,
                        "error": "",
                    }
                except Exception as exc:  # keep the matrix resumable
                    cached = {
                        "frame_id": frame_id,
                        "prompt_index": index,
                        "prompt": asdict(spec),
                        "min_threshold": args.min_threshold,
                        "outputs": {
                            "orig_img_h": image.height,
                            "orig_img_w": image.width,
                            "pred_boxes": [],
                            "pred_masks": [],
                            "pred_scores": [],
                        },
                        "error": f"{type(exc).__name__}: {exc}",
                    }
                write_json(result_path, cached)

            outputs = cached["outputs"]
            if not overlay_path.is_file():
                render_overlay(
                    image_path, outputs, args.render_threshold, overlay_path
                )
            row = {
                "frame_id": frame_id,
                "prompt_index": index,
                "prompt": spec.text,
                "group": spec.group,
                "source_taxon": spec.source_taxon,
                "top_score": max(outputs.get("pred_scores", []) or [0.0]),
                "overlay": str(overlay_path.relative_to(output_root)),
                "error": cached.get("error", ""),
            }
            for threshold in thresholds:
                row[threshold_key(threshold)] = sum(
                    float(score) > threshold
                    for score in outputs.get("pred_scores", [])
                )
            rows.append(row)
            print(
                f"[{frame_id}] {spec.group}: {spec.text!r} -> "
                + ", ".join(
                    f"{threshold:.2f}={row[threshold_key(threshold)]}"
                    for threshold in thresholds
                ),
                flush=True,
            )

    write_json(
        output_root / "summary.json",
        {
            "schema_version": 1,
            "created_utc": datetime.now(timezone.utc).isoformat(),
            "exploratory_not_scored": True,
            "git": git_state(repo_root),
            "source_hash": sha256(Path(__file__)),
            "manifest": str(manifest_path),
            "selection_manifest": str(selection_path),
            "configuration": {
                "device": args.device,
                "min_threshold": args.min_threshold,
                "report_thresholds": thresholds,
                "render_threshold": args.render_threshold,
                "extra_prompts": args.extra_prompt,
                "only_extra_prompts": args.only_extra_prompts,
                "no_mllm_calls": True,
            },
            "frame_inputs": frame_inputs,
            "rows": rows,
        },
    )
    write_csv(output_root / "summary.csv", rows, thresholds)
    write_gallery(output_root / "index.html", rows, thresholds)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
