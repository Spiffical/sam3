#!/usr/bin/env python3
from __future__ import annotations

import argparse
import copy
import importlib
import json
import math
import os
import re
import sys
import time
from collections import Counter, deque
from functools import partial
from pathlib import Path
from types import SimpleNamespace
from typing import Any

try:
    from tqdm import tqdm
except ImportError:
    tqdm = None


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_PROMPT_PROFILE = "underwater"
DEFAULT_CODEC = "mp4v"
DEPRECATED_MISSED_CREATURES_MESSAGE = (
    "The MLLM missed-creature discovery stage is deprecated because it proved "
    "unreliable in practice. Keep the code for reference, but do not enable "
    "'missed_creatures' or '--find-missed-creatures' in production runs."
)

cv2 = None
np = None
torch = None
send_generate_request_orig = None
read_video_frame = None
decode_rle_to_mask = None
encode_binary_mask_to_rle = None
frame_object_metadata = None
iter_output_masks_with_ids = None
PredictorBackend = None


def ensure_runtime_deps() -> None:
    global cv2, np, torch, send_generate_request_orig
    global read_video_frame, decode_rle_to_mask, encode_binary_mask_to_rle
    global frame_object_metadata, iter_output_masks_with_ids, PredictorBackend
    if (
        cv2 is not None
        and np is not None
        and torch is not None
        and send_generate_request_orig is not None
    ):
        return

    missing: list[str] = []
    try:
        cv2 = importlib.import_module("cv2")
    except ImportError:
        missing.append("opencv-python")
    try:
        np = importlib.import_module("numpy")
    except ImportError:
        missing.append("numpy")
    try:
        torch = importlib.import_module("torch")
    except ImportError:
        missing.append("torch")

    try:
        repo_root_str = str(REPO_ROOT)
        nibi_root_str = str(REPO_ROOT / "nibi_model_compare")
        if repo_root_str not in sys.path:
            sys.path.insert(0, repo_root_str)
        if nibi_root_str not in sys.path:
            sys.path.insert(0, nibi_root_str)
        from sam3.agent.client_llm import (
            send_generate_request as _send_generate_request_orig,
        )
        from sam3.apps.interactive_video.backend import PredictorBackend as _PredictorBackend
        from frame_output_utils import (
            encode_binary_mask_to_rle as _encode_binary_mask_to_rle,
            decode_rle_to_mask as _decode_rle_to_mask,
            frame_object_metadata as _frame_object_metadata,
            iter_output_masks_with_ids as _iter_output_masks_with_ids,
            read_video_frame as _read_video_frame,
        )
    except ImportError as exc:
        missing.append(str(exc))
    else:
        send_generate_request_orig = _send_generate_request_orig
        PredictorBackend = _PredictorBackend
        encode_binary_mask_to_rle = _encode_binary_mask_to_rle
        decode_rle_to_mask = _decode_rle_to_mask
        frame_object_metadata = _frame_object_metadata
        iter_output_masks_with_ids = _iter_output_masks_with_ids
        read_video_frame = _read_video_frame

    if missing:
        raise RuntimeError(
            "Missing runtime dependencies: "
            + ", ".join(missing)
            + ". Activate the SAM3 environment before running this script."
        )


def log(message: str) -> None:
    timestamp = time.strftime("%H:%M:%S")
    print(f"[{timestamp}] {message}", flush=True)


class ProgressReporter:
    def __init__(self, total: int, desc: str) -> None:
        self.total = max(0, int(total))
        self.desc = desc
        self.start_time = time.time()
        self.current = 0
        self._bar = (
            tqdm(total=self.total, desc=self.desc, unit="window", dynamic_ncols=True)
            if tqdm is not None
            else None
        )

    def update(self, count: int = 1, *, postfix: dict[str, Any] | None = None) -> None:
        self.current += int(count)
        if self._bar is not None:
            self._bar.update(count)
            if postfix:
                self._bar.set_postfix(postfix, refresh=False)
            return

        should_print = (
            self.current == self.total
            or self.current == 1
            or self.current % 5 == 0
        )
        if should_print:
            elapsed = max(1e-6, time.time() - self.start_time)
            rate = self.current / elapsed
            suffix = ""
            if postfix:
                suffix = " | " + ", ".join(f"{k}={v}" for k, v in postfix.items())
            print(
                f"\r{self.desc}: {self.current}/{self.total} ({rate:.2f} windows/s){suffix}",
                end="" if self.current < self.total else "\n",
                flush=True,
            )

    def close(self) -> None:
        if self._bar is not None:
            self._bar.close()


def sanitize_name(value: str) -> str:
    safe = re.sub(r"[^A-Za-z0-9._-]+", "_", str(value or "")).strip("._-")
    return safe or "run"


def default_prompt_path(prompt_profile: str) -> str:
    base = REPO_ROOT / "sam3" / "agent" / "system_prompts"
    profile = (prompt_profile or "").strip().lower()
    if profile == "underwater":
        return str(base / "system_prompt_id_reassignment_underwater.txt")
    return str(base / "system_prompt_id_reassignment_general.txt")


def default_missing_mask_prompt_path(prompt_profile: str) -> str:
    base = REPO_ROOT / "sam3" / "agent" / "system_prompts"
    profile = (prompt_profile or "").strip().lower()
    if profile == "underwater":
        return str(base / "system_prompt_missing_mask_fill_underwater.txt")
    return str(base / "system_prompt_missing_mask_fill_general.txt")


def default_gap_fill_verify_prompt_path(prompt_profile: str) -> str:
    base = REPO_ROOT / "sam3" / "agent" / "system_prompts"
    profile = (prompt_profile or "").strip().lower()
    if profile == "underwater":
        return str(base / "system_prompt_missing_mask_verify_underwater.txt")
    return str(base / "system_prompt_missing_mask_verify_general.txt")


def default_missed_creatures_prompt_path(prompt_profile: str) -> str:
    base = REPO_ROOT / "sam3" / "agent" / "system_prompts"
    profile = (prompt_profile or "").strip().lower()
    if profile == "underwater":
        return str(base / "system_prompt_missed_creature_discovery_underwater.txt")
    return str(base / "system_prompt_missed_creature_discovery_general.txt")


def default_missed_creatures_verify_prompt_path(prompt_profile: str) -> str:
    base = REPO_ROOT / "sam3" / "agent" / "system_prompts"
    profile = (prompt_profile or "").strip().lower()
    if profile == "underwater":
        return str(base / "system_prompt_missed_creature_verify_underwater.txt")
    return str(base / "system_prompt_missed_creature_verify_general.txt")


def default_outlier_mask_prompt_path(prompt_profile: str) -> str:
    base = REPO_ROOT / "sam3" / "agent" / "system_prompts"
    profile = (prompt_profile or "").strip().lower()
    if profile == "underwater":
        return str(base / "system_prompt_outlier_mask_filter_underwater.txt")
    return str(base / "system_prompt_outlier_mask_filter_general.txt")


def resolve_system_prompt(prompt_profile: str, prompt_path: str | None) -> str:
    candidate = prompt_path or default_prompt_path(prompt_profile)
    if candidate and os.path.exists(candidate):
        with open(candidate, "r", encoding="utf-8") as handle:
            return handle.read().strip()
    return (
        "Assign temporally consistent IDs to the same creature across frames and "
        'return strict JSON only with schema {"frames":[{"frame_index":int,'
        '"assignments":[{"local_id":int,"track_label":str}]}]}.'
    )


def resolve_missing_mask_prompt(prompt_profile: str, prompt_path: str | None) -> str:
    candidate = prompt_path or default_missing_mask_prompt_path(prompt_profile)
    if candidate and os.path.exists(candidate):
        with open(candidate, "r", encoding="utf-8") as handle:
            return handle.read().strip()
    return (
        "Find clearly visible creatures that are missing segmentation masks in some "
        "frames even though the same creature is already segmented in nearby frames. "
        'Return strict JSON only with schema {"issues":[{"target_frame_index":int,'
        '"reference_mask":{"frame_index":int,"local_id":int},"description":str,'
        '"confidence":float}]}. Only report clear, fixable gaps.'
    )


def resolve_gap_fill_verify_prompt(prompt_profile: str, prompt_path: str | None) -> str:
    candidate = prompt_path or default_gap_fill_verify_prompt_path(prompt_profile)
    if candidate and os.path.exists(candidate):
        with open(candidate, "r", encoding="utf-8") as handle:
            return handle.read().strip()
    return (
        "Judge whether a SAM3 point-prompt candidate mask correctly fills a missing "
        "creature on the target frame. Return strict JSON only with schema "
        '{"decision":"accept|retry|reject","reason":str,"suggested_point":[x,y]|null}.'
    )


def resolve_missed_creatures_prompt(prompt_profile: str, prompt_path: str | None) -> str:
    candidate = prompt_path or default_missed_creatures_prompt_path(prompt_profile)
    if candidate and os.path.exists(candidate):
        with open(candidate, "r", encoding="utf-8") as handle:
            return handle.read().strip()
    return (
        "Review chronological video frames and find clearly visible creatures that are "
        "present in the raw frames but not covered by any current segmentation mask. "
        'Return strict JSON only with schema {"issues":[{"target_frame_index":int,'
        '"description":str,"click_points":[[x,y]],"confidence":float}]}.'
    )


def resolve_missed_creatures_verify_prompt(prompt_profile: str, prompt_path: str | None) -> str:
    candidate = prompt_path or default_missed_creatures_verify_prompt_path(prompt_profile)
    if candidate and os.path.exists(candidate):
        with open(candidate, "r", encoding="utf-8") as handle:
            return handle.read().strip()
    return (
        "Verify whether a candidate mask adds a real missed creature. "
        'Return strict JSON only with schema {"decision":"accept|retry|reject","reason":str,"additional_points":[[x,y]],"confidence":float}.'
    )


def resolve_outlier_mask_prompt(prompt_profile: str, prompt_path: str | None) -> str:
    candidate = prompt_path or default_outlier_mask_prompt_path(prompt_profile)
    if candidate and os.path.exists(candidate):
        with open(candidate, "r", encoding="utf-8") as handle:
            return handle.read().strip()
    return (
        "Review a proposed gap-fill mask and decide whether it should stay or be removed. "
        "Image 1 is the raw target frame, image 2 is the candidate mask overlay, and "
        "image 3 shows nearby reference masks for the same creature. Remove the mask only "
        "if it clearly segments empty background, sediment, rocks, noise, or another obvious "
        "non-creature region instead of the intended animal. Return strict JSON only with schema "
        '{"decision":"keep|remove","reason":str}.'
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run an MLLM pass over one or more framewise SAM3 output folders to "
            "reassign mask IDs consistently over time."
        )
    )
    parser.add_argument(
        "input_dirs",
        nargs="+",
        help=(
            "One or more framewise SAM3 run directories. Each directory must contain "
            "summary.json and frame_outputs_rle.json."
        ),
    )
    parser.add_argument(
        "--server-url",
        default=os.environ.get("OPENAI_BASE_URL", "http://127.0.0.1:8000/v1"),
        help="OpenAI-compatible server URL for the MLLM.",
    )
    parser.add_argument(
        "--model",
        default=os.environ.get("OPENAI_MODEL", "Qwen/Qwen3.5-27B"),
        help="OpenAI-compatible model ID for the MLLM.",
    )
    parser.add_argument(
        "--api-key",
        default="",
        help="Optional API key. Defaults to OPENAI_API_KEY or VLLM_API_KEY if set.",
    )
    parser.add_argument(
        "--prompt-profile",
        default=DEFAULT_PROMPT_PROFILE,
        help=f"Prompt profile for the ID-reassignment pass. Default: {DEFAULT_PROMPT_PROFILE}",
    )
    parser.add_argument(
        "--prompt-path",
        default="",
        help="Optional override system prompt file for the ID-reassignment pass.",
    )
    parser.add_argument(
        "--missing-mask-prompt-path",
        default="",
        help="Optional override system prompt file for missing-mask detection.",
    )
    parser.add_argument(
        "--verify-gap-fill-prompt-path",
        default="",
        help="Optional override system prompt file for gap-fill candidate verification.",
    )
    parser.add_argument(
        "--missed-creatures-prompt-path",
        default="",
        help="Deprecated compatibility option for the disabled missed-creature stage.",
    )
    parser.add_argument(
        "--verify-missed-creatures-prompt-path",
        default="",
        help="Deprecated compatibility option for the disabled missed-creature stage.",
    )
    parser.add_argument(
        "--outlier-mask-prompt-path",
        default="",
        help="Optional override system prompt file for gap-fill outlier-mask review.",
    )
    parser.add_argument(
        "--stage",
        action="append",
        choices=["missed_creatures", "gap_fill", "outlier_filter", "id_reassign"],
        default=[],
        help=(
            "Ordered post-processing stage to run. May be repeated, and order matters. "
            "If omitted, legacy stage flags determine the default pipeline. "
            "'missed_creatures' is deprecated and will exit immediately if requested."
        ),
    )
    parser.add_argument(
        "--window-size",
        type=int,
        default=10,
        help="Number of valid/analyzable frames per MLLM window. Default: 10",
    )
    parser.add_argument(
        "--window-stride",
        type=int,
        default=8,
        help="Stride across valid/analyzable frames. Default: 8",
    )
    parser.add_argument(
        "--max-completion-tokens",
        type=int,
        default=1024,
        help="Maximum tokens per MLLM completion. Default: 1024",
    )
    parser.add_argument(
        "--assignment-history-frames",
        type=int,
        default=8,
        help=(
            "How many already-processed prior frames to consult when reusing an "
            "existing global ID during assignment resolution. Default: 8"
        ),
    )
    parser.add_argument(
        "--assignment-heuristic-min-score",
        type=float,
        default=0.85,
        help="Minimum heuristic match score needed to reuse an existing global ID. Default: 0.85",
    )
    parser.add_argument(
        "--max-json-retries",
        type=int,
        default=2,
        help="Maximum JSON repair retries per window. Default: 2",
    )
    parser.add_argument(
        "--max-gap-issues-per-window",
        type=int,
        default=8,
        help="Maximum missing-mask issues to attempt per window. Default: 8",
    )
    parser.add_argument(
        "--missed-creatures-window-size",
        type=int,
        default=20,
        help="Number of valid frames per missed-creature discovery window. Default: 20",
    )
    parser.add_argument(
        "--missed-creatures-window-stride",
        type=int,
        default=10,
        help="Stride across valid frames for missed-creature discovery. Default: 10",
    )
    parser.add_argument(
        "--max-missed-creature-issues-per-window",
        type=int,
        default=4,
        help="Maximum missed-creature issues to attempt per detection window. Default: 4",
    )
    parser.add_argument(
        "--missed-creatures-max-rounds",
        type=int,
        default=10,
        help="Maximum full-video missed-creature discovery rounds. Default: 10",
    )
    parser.add_argument(
        "--missed-creatures-max-attempts",
        type=int,
        default=10,
        help="Maximum click-refinement attempts per missed-creature issue. Default: 10",
    )
    parser.add_argument(
        "--missed-creatures-max-images-per-request",
        type=int,
        default=20,
        help="Maximum images to include in a missed-creature discovery request. Default: 20",
    )
    parser.add_argument(
        "--missed-creatures-duplicate-iou-threshold",
        type=float,
        default=0.80,
        help="Reject missed-creature masks that overlap an existing mask above this IoU. Default: 0.80",
    )
    parser.add_argument(
        "--gap-fill-max-attempts",
        type=int,
        default=4,
        help="Maximum point-prompt attempts per missing-mask issue. Default: 4",
    )
    parser.add_argument(
        "--gap-fill-point-candidates",
        type=int,
        default=6,
        help="Maximum heuristic point candidates per issue before giving up. Default: 6",
    )
    parser.add_argument(
        "--max-outlier-checks-per-window",
        type=int,
        default=12,
        help=(
            "Maximum accepted gap-fill masks to re-review per window for one-frame outliers. "
            "Default: 12"
        ),
    )
    parser.add_argument(
        "--image-detail",
        choices=["low", "high"],
        default=os.environ.get("SAM3_IMAGE_DETAIL", "high"),
        help="Multimodal image detail setting for MLLM requests. Default: high",
    )
    parser.add_argument(
        "--max-images-per-request",
        type=int,
        default=int(os.environ.get("SAM3_MAX_IMAGES_PER_REQUEST", "3")),
        help="Maximum images to keep in one MLLM request. Default: 3",
    )
    parser.add_argument(
        "--image-max-edge",
        type=int,
        default=768,
        help="Maximum image edge for collage requests before downscaling. Default: 768",
    )
    parser.add_argument(
        "--image-min-edge",
        type=int,
        default=384,
        help="Minimum image edge to back off to on context overflow. Default: 384",
    )
    parser.add_argument(
        "--collage-cols",
        type=int,
        default=2,
        help="Number of columns in raw/overlay collages. Default: 2",
    )
    parser.add_argument(
        "--collage-tile-max-edge",
        type=int,
        default=320,
        help="Max edge for each collage tile. Default: 320",
    )
    parser.add_argument(
        "--output-subdir",
        default="consistent_ids_mllm",
        help="Per-run output subdirectory. Default: consistent_ids_mllm",
    )
    parser.add_argument(
        "--output-video-name",
        default="overlay_consistent_ids.mp4",
        help="Filename for the relabeled overlay video. Default: overlay_consistent_ids.mp4",
    )
    parser.add_argument(
        "--sam3-gpu-ids",
        default=os.environ.get("SAM3_GAP_FILL_GPU_IDS", "0"),
        help="Comma-separated GPU ids for SAM3 point-prompt repair inside the runner process. Default: 0",
    )
    parser.add_argument(
        "--sam3-image-size",
        type=int,
        default=1008,
        help="SAM3 video predictor image size for gap filling. Default: 1008",
    )
    parser.add_argument(
        "--sam3-offload-video-to-cpu",
        action="store_true",
        help="Offload SAM3 video frames to CPU memory during gap-fill repair.",
    )
    parser.add_argument(
        "--fill-missing-masks",
        dest="fill_missing_masks",
        action="store_true",
        help="Run MLLM-guided SAM3 point-prompt repair before ID reassignment. Enabled by default.",
    )
    parser.add_argument(
        "--no-fill-missing-masks",
        dest="fill_missing_masks",
        action="store_false",
        help="Skip the missing-mask repair stage and only do ID reassignment.",
    )
    parser.add_argument(
        "--find-missed-creatures",
        dest="find_missed_creatures",
        action="store_true",
        help=(
            "Deprecated and disabled: the MLLM-guided missed-creature discovery pass "
            "proved unreliable."
        ),
    )
    parser.add_argument(
        "--no-find-missed-creatures",
        dest="find_missed_creatures",
        action="store_false",
        help="Leave the deprecated missed-creature discovery stage disabled.",
    )
    parser.add_argument(
        "--allow-drop-assignments",
        action="store_true",
        help=(
            "Allow the MLLM ID reassignment stage to emit drop labels that remove masks. "
            "Disabled by default to preserve masks unless explicitly requested."
        ),
    )
    parser.add_argument(
        "--filter-outlier-masks",
        dest="filter_outlier_masks",
        action="store_true",
        help=(
            "Review accepted gap-fill masks for obvious one-frame hallucinations before "
            "ID reassignment. Enabled by default."
        ),
    )
    parser.add_argument(
        "--no-filter-outlier-masks",
        dest="filter_outlier_masks",
        action="store_false",
        help="Skip the outlier-mask cleanup stage after gap fill.",
    )
    parser.add_argument(
        "--codec",
        default=DEFAULT_CODEC,
        help=f"OpenCV fourcc codec. Default: {DEFAULT_CODEC}",
    )
    parser.add_argument(
        "--render-video",
        action="store_true",
        help="Render a relabeled overlay video. Enabled by default.",
    )
    parser.add_argument(
        "--no-render-video",
        dest="render_video",
        action="store_false",
        help="Skip overlay video rendering.",
    )
    parser.set_defaults(render_video=True)
    parser.add_argument(
        "--continue-on-error",
        action="store_true",
        help="Continue processing later input dirs if one run fails.",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Keep extra window-level debug JSON files.",
    )
    parser.set_defaults(
        fill_missing_masks=True,
        filter_outlier_masks=True,
        find_missed_creatures=False,
    )
    return parser.parse_args()


def resolve_run_dir(path_str: str) -> Path:
    path = Path(path_str).expanduser().resolve()
    if path.is_file():
        if path.name in {"summary.json", "frame_outputs_rle.json"}:
            return path.parent
        raise FileNotFoundError(f"Unsupported input file path: {path}")

    if not path.is_dir():
        raise FileNotFoundError(f"Input directory does not exist: {path}")

    if (path / "summary.json").exists():
        return path

    candidates = sorted(path.glob("**/summary.json"))
    if len(candidates) == 1:
        return candidates[0].parent
    if len(candidates) > 1:
        raise RuntimeError(
            f"Input path {path} contains multiple summary.json files; pass a specific run directory."
        )
    raise FileNotFoundError(
        f"Could not find summary.json under input path: {path}"
    )


def read_json(path: str | Path) -> Any:
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def write_json(path: str | Path, payload: Any) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)
        handle.write("\n")


def extract_json_object(text: str) -> dict[str, Any] | None:
    if not text:
        return None

    fenced = re.findall(r"```(?:json)?\s*(.*?)```", text, flags=re.IGNORECASE | re.DOTALL)
    for block in fenced:
        block = block.strip()
        if not block:
            continue
        try:
            parsed = json.loads(block)
        except Exception:
            continue
        if isinstance(parsed, dict):
            return parsed

    start_positions = [idx for idx, ch in enumerate(text) if ch == "{"]
    for start in start_positions:
        depth = 0
        in_string = False
        escape = False
        for idx in range(start, len(text)):
            ch = text[idx]
            if in_string:
                if escape:
                    escape = False
                elif ch == "\\":
                    escape = True
                elif ch == '"':
                    in_string = False
                continue
            if ch == '"':
                in_string = True
                continue
            if ch == "{":
                depth += 1
            elif ch == "}":
                depth -= 1
                if depth == 0:
                    candidate = text[start : idx + 1]
                    try:
                        parsed = json.loads(candidate)
                    except Exception:
                        break
                    if isinstance(parsed, dict):
                        return parsed
                    break
    return None


def build_index_windows(indices: list[int], *, window_size: int, stride: int) -> list[list[int]]:
    ordered = [int(x) for x in indices]
    if not ordered:
        return []

    window_size = max(1, int(window_size))
    stride = max(1, int(stride))
    windows: list[list[int]] = []
    for start in range(0, len(ordered), stride):
        window = ordered[start : start + window_size]
        if window:
            windows.append(window)

    tail = ordered[max(0, len(ordered) - window_size) :]
    if tail and (not windows or windows[-1] != tail):
        windows.append(tail)
    return windows


def rescale_frame(frame_bgr: Any, max_edge: int) -> Any:
    h, w = frame_bgr.shape[:2]
    longest = max(h, w)
    if longest <= max_edge:
        return frame_bgr
    scale = float(max_edge) / float(longest)
    new_w = max(1, int(round(w * scale)))
    new_h = max(1, int(round(h * scale)))
    return cv2.resize(frame_bgr, (new_w, new_h), interpolation=cv2.INTER_AREA)


def build_collage(
    frame_items: list[tuple[int, Any]],
    output_path: str,
    *,
    cols: int,
    tile_max_edge: int,
) -> tuple[int, int]:
    assert frame_items
    cols = max(1, int(cols))
    rows = int(math.ceil(len(frame_items) / float(cols)))
    resized: list[tuple[int, Any]] = []
    for frame_idx, frame in frame_items:
        tile = rescale_frame(frame, max_edge=tile_max_edge).copy()
        cv2.putText(
            tile,
            f"f={frame_idx}",
            (12, 28),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )
        cv2.putText(
            tile,
            f"f={frame_idx}",
            (12, 28),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 0, 0),
            1,
            cv2.LINE_AA,
        )
        resized.append((frame_idx, tile))

    tile_h = max(tile.shape[0] for _, tile in resized)
    tile_w = max(tile.shape[1] for _, tile in resized)
    collage = np.zeros((rows * tile_h, cols * tile_w, 3), dtype=np.uint8)
    for i, (_frame_idx, tile) in enumerate(resized):
        r = i // cols
        c = i % cols
        y0 = r * tile_h
        x0 = c * tile_w
        h, w = tile.shape[:2]
        collage[y0 : y0 + h, x0 : x0 + w] = tile

    cv2.imwrite(output_path, collage)
    return rows, cols


def object_color(obj_id: int) -> tuple[int, int, int]:
    return (
        int((obj_id * 47) % 255),
        int((obj_id * 89 + 37) % 255),
        int((obj_id * 131 + 73) % 255),
    )


def normalize_rle_mask(mask_rle: Any, frame_h: int, frame_w: int) -> dict[str, Any]:
    if isinstance(mask_rle, dict):
        counts = mask_rle.get("counts", "")
        size = mask_rle.get("size") or [frame_h, frame_w]
        if isinstance(counts, bytes):
            counts = counts.decode("utf-8")
        return {"size": [int(size[0]), int(size[1])], "counts": str(counts)}
    if isinstance(mask_rle, str):
        return {"size": [int(frame_h), int(frame_w)], "counts": mask_rle}
    raise ValueError(f"Unsupported mask RLE type: {type(mask_rle)}")


def frame_row_local_ids(frame_row: dict[str, Any]) -> list[int]:
    raw_ids = frame_row.get("out_obj_ids") or []
    ids: list[int] = []
    for raw in raw_ids:
        try:
            ids.append(int(raw))
        except Exception:
            continue
    if ids:
        return ids
    mask_count = len(frame_row.get("out_binary_masks_rle") or [])
    return list(range(1, mask_count + 1))


def decode_frame_row_masks(
    frame_row: dict[str, Any],
    frame_h: int,
    frame_w: int,
) -> list[dict[str, Any]]:
    raw_ids = frame_row_local_ids(frame_row)
    raw_masks = list(frame_row.get("out_binary_masks_rle") or [])
    raw_boxes = list(frame_row.get("out_boxes_xywh") or [])
    raw_probs = list(frame_row.get("out_probs") or [])

    items: list[dict[str, Any]] = []
    for idx, mask_rle in enumerate(raw_masks):
        local_id = raw_ids[idx] if idx < len(raw_ids) else idx + 1
        normalized = normalize_rle_mask(mask_rle, frame_h=frame_h, frame_w=frame_w)
        mask = decode_rle_to_mask(normalized, frame_h, frame_w).astype(bool)
        ys, xs = np.where(mask)
        if len(xs) == 0 or len(ys) == 0:
            bbox_xyxy = None
            centroid = None
            area = 0
        else:
            x1 = int(xs.min())
            y1 = int(ys.min())
            x2 = int(xs.max())
            y2 = int(ys.max())
            bbox_xyxy = (x1, y1, x2, y2)
            centroid = (int(np.round(xs.mean())), int(np.round(ys.mean())))
            area = int(mask.sum())
        box_xywh = raw_boxes[idx] if idx < len(raw_boxes) else None
        prob = raw_probs[idx] if idx < len(raw_probs) else None
        try:
            score = float(prob)
        except Exception:
            score = None
        items.append(
            {
                "local_id": int(local_id),
                "mask": mask,
                "mask_rle": normalized,
                "bbox_xyxy": bbox_xyxy,
                "box_xywh": box_xywh,
                "score": score,
                "centroid": centroid,
                "area": area,
            }
        )
    return items


def draw_overlay_with_labels(
    frame_bgr: Any,
    mask_items: list[dict[str, Any]],
    *,
    assigned_global_ids: dict[int, int] | None = None,
    show_local_id: bool = True,
) -> Any:
    output = frame_bgr.copy()
    overlay = np.zeros_like(output)
    assigned_global_ids = assigned_global_ids or {}

    for item in mask_items:
        local_id = int(item["local_id"])
        color_seed = assigned_global_ids.get(local_id, local_id)
        color = object_color(int(color_seed))
        mask = item["mask"]
        if mask.shape != output.shape[:2]:
            continue
        overlay[mask] = color
        mask_u8 = (mask.astype(np.uint8) * 255)
        contours, _ = cv2.findContours(mask_u8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(output, contours, -1, color, 2)

    output = cv2.addWeighted(output, 1.0, overlay, 0.35, 0.0)

    for item in mask_items:
        local_id = int(item["local_id"])
        bbox = item.get("bbox_xyxy")
        centroid = item.get("centroid")
        global_id = assigned_global_ids.get(local_id)
        if global_id is None:
            label = f"l{local_id}" if show_local_id else f"id{local_id}"
        elif show_local_id:
            label = f"g{global_id}/l{local_id}"
        else:
            label = f"g{global_id}"
        if centroid is not None:
            x, y = centroid
        elif bbox is not None:
            x = int((bbox[0] + bbox[2]) / 2)
            y = int((bbox[1] + bbox[3]) / 2)
        else:
            x, y = 12, 24
        x = max(4, min(output.shape[1] - 120, int(x)))
        y = max(18, min(output.shape[0] - 8, int(y)))
        cv2.putText(
            output,
            label,
            (x, y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )
        cv2.putText(
            output,
            label,
            (x, y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (0, 0, 0),
            1,
            cv2.LINE_AA,
        )
    return output


def binary_mask_iou(mask_a: Any, mask_b: Any) -> float:
    a = np.asarray(mask_a).astype(bool)
    b = np.asarray(mask_b).astype(bool)
    inter = float(np.logical_and(a, b).sum())
    if inter <= 0.0:
        return 0.0
    union = float(np.logical_or(a, b).sum())
    if union <= 0.0:
        return 0.0
    return inter / union


def mask_match_support_score(current_item: dict[str, Any], prior_item: dict[str, Any]) -> float:
    area = max(1.0, float(current_item.get("area") or 0.0))
    centroid = current_item.get("centroid")
    iou = binary_mask_iou(current_item["mask"], prior_item["mask"])
    area_prior = max(1.0, float(prior_item.get("area") or 0.0))
    area_ratio = min(area, area_prior) / max(area, area_prior)
    dist_score = 0.0
    if centroid is not None and prior_item.get("centroid") is not None:
        dx = float(centroid[0] - prior_item["centroid"][0])
        dy = float(centroid[1] - prior_item["centroid"][1])
        dist = math.sqrt(dx * dx + dy * dy)
        dist_score = max(0.0, 1.0 - dist / 150.0)
    return (2.0 * iou) + (0.5 * area_ratio) + (0.5 * dist_score)


def heuristic_match_global_id(
    *,
    current_item: dict[str, Any],
    prior_items: list[dict[str, Any]],
    prior_assignments: dict[int, int],
    disallowed_global_ids: set[int],
    min_score: float = 0.85,
) -> int | None:
    best_global_id: int | None = None
    best_score = 0.0

    for prior_item in prior_items:
        local_id = int(prior_item["local_id"])
        global_id = prior_assignments.get(local_id)
        if global_id is None or global_id in disallowed_global_ids:
            continue
        score = mask_match_support_score(current_item, prior_item)
        if score > best_score:
            best_score = score
            best_global_id = int(global_id)

    if best_score >= float(min_score):
        return best_global_id
    return None


def collect_recent_assignment_history(
    *,
    frame_index: int,
    resolved_window_assignments: dict[int, dict[int, int]],
    existing_frame_assignments: dict[int, dict[int, int]],
    mask_items_by_frame: dict[int, list[dict[str, Any]]],
    working_frame_rows_by_index: dict[int, dict[str, Any]],
    frame_h: int,
    frame_w: int,
    history_frame_budget: int,
) -> list[tuple[list[dict[str, Any]], dict[int, int]]]:
    budget = max(0, int(history_frame_budget))
    if budget <= 0:
        return []

    ordered_frames: list[int] = []
    seen_frames: set[int] = set()

    for prior_frame in sorted(
        (int(fi) for fi in resolved_window_assignments.keys() if int(fi) < int(frame_index)),
        reverse=True,
    ):
        if prior_frame in seen_frames:
            continue
        ordered_frames.append(prior_frame)
        seen_frames.add(prior_frame)
        if len(ordered_frames) >= budget:
            break

    if len(ordered_frames) < budget:
        for prior_frame in sorted(
            (int(fi) for fi in existing_frame_assignments.keys() if int(fi) < int(frame_index)),
            reverse=True,
        ):
            if prior_frame in seen_frames:
                continue
            ordered_frames.append(prior_frame)
            seen_frames.add(prior_frame)
            if len(ordered_frames) >= budget:
                break

    history_pairs: list[tuple[list[dict[str, Any]], dict[int, int]]] = []
    for prior_frame in ordered_frames:
        prior_mapping = resolved_window_assignments.get(prior_frame) or existing_frame_assignments.get(prior_frame) or {}
        if not prior_mapping:
            continue

        prior_items = mask_items_by_frame.get(prior_frame)
        if prior_items is None:
            prior_row = working_frame_rows_by_index.get(prior_frame)
            if prior_row is None:
                continue
            prior_items = decode_frame_row_masks(prior_row, frame_h, frame_w)

        if prior_items:
            history_pairs.append(
                (
                    prior_items,
                    {int(k): int(v) for k, v in prior_mapping.items()},
                )
            )

    return history_pairs


def normalize_track_label(value: Any) -> str | None:
    if value is None:
        return None
    if isinstance(value, (int, float)):
        return f"g{int(value)}"
    label = str(value).strip()
    if not label:
        return None
    return label


def sanitize_assignment_response(
    parsed: dict[str, Any] | None,
    *,
    window_frame_indices: list[int],
    local_ids_by_frame: dict[int, list[int]],
) -> dict[int, dict[int, str | None]]:
    sanitized: dict[int, dict[int, str | None]] = {}
    if not isinstance(parsed, dict):
        return sanitized

    raw_frames = parsed.get("frames", [])
    if not isinstance(raw_frames, list):
        return sanitized

    allowed_frames = set(int(x) for x in window_frame_indices)
    for frame_row in raw_frames:
        if not isinstance(frame_row, dict):
            continue
        try:
            frame_index = int(frame_row.get("frame_index"))
        except Exception:
            continue
        if frame_index not in allowed_frames:
            continue

        assignments = frame_row.get("assignments", [])
        if not isinstance(assignments, list):
            continue
        per_frame: dict[int, str | None] = {}
        allowed_local = set(int(x) for x in local_ids_by_frame.get(frame_index, []))
        for item in assignments:
            if not isinstance(item, dict):
                continue
            try:
                local_id = int(item.get("local_id"))
            except Exception:
                continue
            if local_id not in allowed_local:
                continue
            per_frame[local_id] = normalize_track_label(item.get("track_label"))
        sanitized[frame_index] = per_frame
    return sanitized


def build_anchor_text(
    *,
    frame_assignments: dict[int, dict[int, int]],
    window_frame_indices: list[int],
) -> str:
    lines: list[str] = []
    for frame_index in window_frame_indices:
        mapping = frame_assignments.get(frame_index, {})
        if not mapping:
            continue
        ordered = ", ".join(
            f"l{int(local_id)}->g{int(global_id)}"
            for local_id, global_id in sorted(mapping.items())
        )
        lines.append(f"frame {frame_index}: {ordered}")
    if not lines:
        return "No fixed overlap global IDs yet in this window."
    return "Fixed overlap assignments:\n" + "\n".join(lines)


def build_inventory_text(
    *,
    window_frame_indices: list[int],
    local_ids_by_frame: dict[int, list[int]],
) -> str:
    lines = []
    for frame_index in window_frame_indices:
        local_ids = local_ids_by_frame.get(frame_index, [])
        ids_text = ", ".join(f"l{int(local_id)}" for local_id in local_ids) or "(none)"
        lines.append(f"frame {frame_index}: {ids_text}")
    return "Local masks per frame:\n" + "\n".join(lines)


def request_window_assignment(
    *,
    send_generate_request_fn: Any,
    system_prompt: str,
    raw_collage_path: str,
    overlay_collage_path: str,
    window_frame_indices: list[int],
    anchor_text: str,
    inventory_text: str,
    next_global_id: int,
    max_json_retries: int,
    allow_drop_assignments: bool,
) -> tuple[dict[str, Any] | None, str | None]:
    messages = [
        {"role": "system", "content": system_prompt},
        {
            "role": "user",
            "content": [
                {"type": "image", "image": raw_collage_path},
                {"type": "image", "image": overlay_collage_path},
                {
                    "type": "text",
                    "text": (
                        f"Frames in this chronological window: {window_frame_indices}\n\n"
                        f"{anchor_text}\n\n"
                        f"{inventory_text}\n\n"
                        f"Use existing anchored labels exactly as given. "
                        f"Be conservative about creating new track labels: only use temporary labels like new_a, new_b "
                        f"when a creature clearly does not match any already-anchored creature continuing through the overlap frames. "
                        f"If a creature plausibly continues from an anchored overlap creature, reuse that anchored g-label instead of inventing a new_* label. "
                        f"The next available permanent global id after this window starts at g{int(next_global_id)}. "
                        + (
                            "Preserve every existing mask and do not use drop labels. "
                            if not allow_drop_assignments
                            else "Only use drop for clearly spurious masks that should be removed. "
                        )
                        + "Return strict JSON only."
                    ),
                },
            ],
        },
    ]

    last_text: str | None = None
    for attempt in range(max(0, int(max_json_retries)) + 1):
        last_text = send_generate_request_fn(messages)
        parsed = extract_json_object(last_text or "")
        if isinstance(parsed, dict):
            return parsed, last_text

        messages.append(
            {
                "role": "assistant",
                "content": [{"type": "text", "text": str(last_text or "")}],
            }
        )
        messages.append(
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": (
                            "Your previous response did not contain valid JSON. "
                            "Return strict JSON only with schema "
                            '{"frames":[{"frame_index":int,"assignments":[{"local_id":int,"track_label":str}]}]}.'
                        ),
                    }
                ],
            }
        )
    return None, last_text


def parse_gpu_ids(raw: str) -> list[int]:
    values: list[int] = []
    for part in str(raw or "").split(","):
        part = part.strip()
        if not part:
            continue
        values.append(int(part))
    if not values:
        return [0]
    return values


def frame_row_mask_count(frame_row: dict[str, Any]) -> int:
    return len(frame_row.get("out_binary_masks_rle") or [])


def next_local_id_for_frame(frame_row: dict[str, Any]) -> int:
    local_ids = frame_row_local_ids(frame_row)
    return max(local_ids) + 1 if local_ids else 1


def _to_serializable_list(value: Any) -> list[Any]:
    if value is None:
        return []
    if torch is not None and isinstance(value, torch.Tensor):
        value = value.detach().cpu().numpy()
    if np is not None and isinstance(value, np.ndarray):
        return value.tolist()
    if hasattr(value, "tolist"):
        try:
            return value.tolist()
        except Exception:
            pass
    if isinstance(value, list):
        return value
    try:
        return list(value)
    except Exception:
        return []


def normalize_point_xy(point: Any, frame_w: int, frame_h: int) -> tuple[int, int] | None:
    if not isinstance(point, (list, tuple)) or len(point) < 2:
        return None
    try:
        x = int(round(float(point[0])))
        y = int(round(float(point[1])))
    except Exception:
        return None
    x = max(0, min(int(frame_w) - 1, x))
    y = max(0, min(int(frame_h) - 1, y))
    return (x, y)


def dedupe_points(
    points: list[tuple[int, int]],
    *,
    min_distance: float = 6.0,
) -> list[tuple[int, int]]:
    kept: list[tuple[int, int]] = []
    for point in points:
        if any(
            math.hypot(float(point[0] - other[0]), float(point[1] - other[1])) < min_distance
            for other in kept
        ):
            continue
        kept.append(point)
    return kept


def bbox_xywh_from_xyxy(
    bbox_xyxy: tuple[int, int, int, int] | None,
    *,
    frame_h: int,
    frame_w: int,
) -> list[float] | None:
    if bbox_xyxy is None:
        return None
    x1, y1, x2, y2 = bbox_xyxy
    if x2 <= x1 or y2 <= y1 or frame_w <= 0 or frame_h <= 0:
        return None
    return [
        float(x1) / float(frame_w),
        float(y1) / float(frame_h),
        float(x2 - x1) / float(frame_w),
        float(y2 - y1) / float(frame_h),
    ]


def bbox_dims_xyxy(bbox_xyxy: tuple[int, int, int, int] | None) -> tuple[int, int]:
    if bbox_xyxy is None:
        return (0, 0)
    return (
        max(0, int(bbox_xyxy[2]) - int(bbox_xyxy[0])),
        max(0, int(bbox_xyxy[3]) - int(bbox_xyxy[1])),
    )


def bbox_center_xyxy(bbox_xyxy: tuple[int, int, int, int] | None) -> tuple[float, float] | None:
    if bbox_xyxy is None:
        return None
    return (
        (float(bbox_xyxy[0]) + float(bbox_xyxy[2])) / 2.0,
        (float(bbox_xyxy[1]) + float(bbox_xyxy[3])) / 2.0,
    )


def bbox_iou_xyxy(
    bbox_a: tuple[int, int, int, int] | None,
    bbox_b: tuple[int, int, int, int] | None,
) -> float:
    if bbox_a is None or bbox_b is None:
        return 0.0
    ax1, ay1, ax2, ay2 = bbox_a
    bx1, by1, bx2, by2 = bbox_b
    inter_x1 = max(int(ax1), int(bx1))
    inter_y1 = max(int(ay1), int(by1))
    inter_x2 = min(int(ax2), int(bx2))
    inter_y2 = min(int(ay2), int(by2))
    inter_w = max(0, inter_x2 - inter_x1)
    inter_h = max(0, inter_y2 - inter_y1)
    inter = float(inter_w * inter_h)
    if inter <= 0.0:
        return 0.0
    area_a = float(max(0, int(ax2) - int(ax1)) * max(0, int(ay2) - int(ay1)))
    area_b = float(max(0, int(bx2) - int(bx1)) * max(0, int(by2) - int(by1)))
    union = area_a + area_b - inter
    if union <= 0.0:
        return 0.0
    return inter / union


def sanitize_missing_issue_response(
    parsed: dict[str, Any] | None,
    *,
    window_frame_indices: list[int],
    local_ids_by_frame: dict[int, list[int]],
    max_issues: int,
) -> list[dict[str, Any]]:
    if not isinstance(parsed, dict):
        return []
    raw_issues = parsed.get("issues", [])
    if not isinstance(raw_issues, list):
        return []

    allowed_frames = set(int(x) for x in window_frame_indices)
    issues: list[dict[str, Any]] = []
    seen: set[tuple[int, tuple[tuple[int, int], ...]]] = set()
    for raw_issue in raw_issues:
        if not isinstance(raw_issue, dict):
            continue
        try:
            target_frame_index = int(raw_issue.get("target_frame_index"))
        except Exception:
            continue
        if target_frame_index not in allowed_frames:
            continue

        raw_refs = raw_issue.get("reference_masks", [])
        if isinstance(raw_issue.get("reference_mask"), dict):
            raw_refs = [raw_issue.get("reference_mask")] + (
                raw_refs if isinstance(raw_refs, list) else []
            )
        if not isinstance(raw_refs, list):
            continue
        ref_pairs: list[dict[str, int]] = []
        seen_refs: set[tuple[int, int]] = set()
        for raw_ref in raw_refs:
            if not isinstance(raw_ref, dict):
                continue
            try:
                frame_index = int(raw_ref.get("frame_index"))
                local_id = int(raw_ref.get("local_id"))
            except Exception:
                continue
            if frame_index not in allowed_frames or frame_index == target_frame_index:
                continue
            if local_id not in set(int(x) for x in local_ids_by_frame.get(frame_index, [])):
                continue
            key = (frame_index, local_id)
            if key in seen_refs:
                continue
            seen_refs.add(key)
            ref_pairs.append({"frame_index": frame_index, "local_id": local_id})

        if not ref_pairs:
            continue

        ref_pairs.sort(
            key=lambda ref: (
                abs(int(ref["frame_index"]) - target_frame_index),
                int(ref["frame_index"]),
                int(ref["local_id"]),
            )
        )
        try:
            confidence = float(raw_issue.get("confidence", 0.0))
        except Exception:
            confidence = 0.0

        description = str(raw_issue.get("description", "")).strip()
        for split_index, ref_pair in enumerate(ref_pairs):
            issue_key = (target_frame_index, int(ref_pair["frame_index"]), int(ref_pair["local_id"]))
            if issue_key in seen:
                continue
            seen.add(issue_key)
            issues.append(
                {
                    "target_frame_index": int(target_frame_index),
                    "reference_masks": [ref_pair],
                    "description": description,
                    "confidence": confidence,
                    "source_reference_count": len(ref_pairs),
                    "source_reference_split_index": int(split_index),
                }
            )
            if len(issues) >= max(1, int(max_issues)):
                break
        if len(issues) >= max(1, int(max_issues)):
            break
    return issues


def request_missing_mask_issues(
    *,
    send_generate_request_fn: Any,
    system_prompt: str,
    raw_collage_path: str,
    overlay_collage_path: str,
    window_frame_indices: list[int],
    inventory_text: str,
    max_issues: int,
    max_json_retries: int,
) -> tuple[dict[str, Any] | None, str | None]:
    messages = [
        {"role": "system", "content": system_prompt},
        {
            "role": "user",
            "content": [
                {"type": "image", "image": raw_collage_path},
                {"type": "image", "image": overlay_collage_path},
                {
                    "type": "text",
                    "text": (
                        f"Frames in this chronological window: {window_frame_indices}\n\n"
                        f"{inventory_text}\n\n"
                        f"Find at most {int(max_issues)} clearly visible missing-mask cases. "
                        "Each reported issue must describe exactly one missing creature, not a bundle. "
                        "Each issue must include exactly one reference_mask showing that same creature in a nearby frame. "
                        "A valid issue means the same creature is already segmented in nearby frame(s) "
                        "inside this window, but it is visibly present and unsegmented on the target frame. "
                        "Do not report ambiguous, occluded, tiny, or low-confidence cases. "
                        "Return strict JSON only."
                    ),
                },
            ],
        },
    ]

    last_text: str | None = None
    for _attempt in range(max(0, int(max_json_retries)) + 1):
        last_text = send_generate_request_fn(messages)
        parsed = extract_json_object(last_text or "")
        if isinstance(parsed, dict):
            return parsed, last_text
        messages.append(
            {
                "role": "assistant",
                "content": [{"type": "text", "text": str(last_text or "")}],
            }
        )
        messages.append(
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": (
                            "Your previous response did not contain valid JSON. "
                            "Return strict JSON only with schema "
                            '{"issues":[{"target_frame_index":int,"reference_mask":{"frame_index":int,"local_id":int},"description":str,"confidence":float}]}.'
                        ),
                    }
                ],
            }
        )
    return None, last_text


def request_gap_fill_verdict(
    *,
    send_generate_request_fn: Any,
    system_prompt: str,
    raw_target_frame_path: str,
    candidate_overlay_path: str,
    reference_collage_path: str,
    target_frame_index: int,
    issue_description: str,
    attempt_index: int,
    positive_points: list[tuple[int, int]],
    max_json_retries: int,
) -> tuple[dict[str, Any] | None, str | None]:
    messages = [
        {"role": "system", "content": system_prompt},
        {
            "role": "user",
            "content": [
                {"type": "image", "image": raw_target_frame_path},
                {"type": "image", "image": candidate_overlay_path},
                {"type": "image", "image": reference_collage_path},
                {
                    "type": "text",
                    "text": (
                        f"Target frame: {int(target_frame_index)}\n"
                        f"Attempt: {int(attempt_index)}\n"
                        f"Positive clicks used: {[list(point) for point in positive_points]}\n"
                        f"Issue description: {issue_description or '(none)'}\n\n"
                        "Image 1 is the raw target frame. "
                        "Image 2 is the candidate mask overlay on the target frame. "
                        "Image 3 shows nearby reference masks for the same creature. "
                        "Accept only if the candidate clearly segments the same missing creature and does not just duplicate an existing mask. "
                        "If the creature is visible but the candidate is incomplete or wrong, return retry and suggest one additional click point [x,y] on an uncovered part of the same creature. "
                        "If the issue is not repairable from this frame, return reject. "
                        "Return strict JSON only."
                    ),
                },
            ],
        },
    ]

    last_text: str | None = None
    for _attempt in range(max(0, int(max_json_retries)) + 1):
        last_text = send_generate_request_fn(messages)
        parsed = extract_json_object(last_text or "")
        if isinstance(parsed, dict):
            return parsed, last_text
        messages.append(
            {
                "role": "assistant",
                "content": [{"type": "text", "text": str(last_text or "")}],
            }
        )
        messages.append(
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": (
                            "Your previous response did not contain valid JSON. "
                            'Return strict JSON only with schema {"decision":"accept|retry|reject","reason":str,"suggested_point":[x,y]|null}.'
                        ),
                    }
                ],
            }
        )
    return None, last_text


def request_outlier_mask_verdict(
    *,
    send_generate_request_fn: Any,
    system_prompt: str,
    raw_target_frame_path: str,
    candidate_overlay_path: str,
    reference_collage_path: str,
    target_frame_index: int,
    local_id: int,
    issue_description: str,
    max_json_retries: int,
) -> tuple[dict[str, Any] | None, str | None]:
    messages = [
        {"role": "system", "content": system_prompt},
        {
            "role": "user",
            "content": [
                {"type": "image", "image": raw_target_frame_path},
                {"type": "image", "image": candidate_overlay_path},
                {"type": "image", "image": reference_collage_path},
                {
                    "type": "text",
                    "text": (
                        f"Target frame: {int(target_frame_index)}\n"
                        f"Candidate local id: l{int(local_id)}\n"
                        f"Issue description: {issue_description or '(none)'}\n\n"
                        "Image 1 is the raw target frame. "
                        "Image 2 highlights the candidate mask that was added automatically. "
                        "Image 3 shows nearby reference masks for the intended creature. "
                        "Keep the mask only if it clearly covers a real creature matching the references. "
                        "Remove it if it is clearly an empty-background or non-creature hallucination. "
                        "When uncertain, keep. Return strict JSON only."
                    ),
                },
            ],
        },
    ]

    last_text: str | None = None
    for _attempt in range(max(0, int(max_json_retries)) + 1):
        last_text = send_generate_request_fn(messages)
        parsed = extract_json_object(last_text or "")
        if isinstance(parsed, dict):
            return parsed, last_text
        messages.append(
            {
                "role": "assistant",
                "content": [{"type": "text", "text": str(last_text or "")}],
            }
        )
        messages.append(
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": (
                            "Your previous response did not contain valid JSON. "
                            'Return strict JSON only with schema {"decision":"keep|remove","reason":str}.'
                        ),
                    }
                ],
            }
        )
    return None, last_text


def find_mask_item(
    mask_items_by_frame: dict[int, list[dict[str, Any]]],
    *,
    frame_index: int,
    local_id: int,
) -> dict[str, Any] | None:
    for item in mask_items_by_frame.get(int(frame_index), []):
        if int(item["local_id"]) == int(local_id):
            return item
    return None


def find_existing_issue_match(
    *,
    target_frame_index: int,
    reference_masks: list[dict[str, int]],
    mask_items_by_frame: dict[int, list[dict[str, Any]]],
    hint_bbox_xyxy: tuple[int, int, int, int] | None,
    candidate_item: dict[str, Any] | None = None,
    min_support_score: float = 0.9,
    min_hint_iou: float = 0.55,
) -> dict[str, Any] | None:
    existing_items = list(mask_items_by_frame.get(int(target_frame_index), []) or [])
    if not existing_items:
        return None

    reference_items: list[dict[str, Any]] = []
    for ref in reference_masks:
        try:
            ref_frame_index = int(ref["frame_index"])
            ref_local_id = int(ref["local_id"])
        except Exception:
            continue
        ref_item = find_mask_item(
            mask_items_by_frame,
            frame_index=ref_frame_index,
            local_id=ref_local_id,
        )
        if ref_item is not None:
            reference_items.append(ref_item)
    if not reference_items:
        return None

    best_match: dict[str, Any] | None = None
    best_score = float("-inf")
    for existing_item in existing_items:
        support_scores = [mask_match_support_score(existing_item, ref_item) for ref_item in reference_items]
        support_score = float(max(support_scores, default=0.0))
        hint_iou = float(bbox_iou_xyxy(existing_item.get("bbox_xyxy"), hint_bbox_xyxy))
        duplicate_iou = 0.0
        if candidate_item is not None:
            duplicate_iou = float(binary_mask_iou(existing_item["mask"], candidate_item["mask"]))
        combined_score = support_score + (0.75 * hint_iou) + (1.25 * duplicate_iou)
        if combined_score <= best_score:
            continue
        best_score = combined_score
        best_match = {
            "local_id": int(existing_item["local_id"]),
            "support_score": support_score,
            "hint_iou": hint_iou,
            "duplicate_iou": duplicate_iou,
            "combined_score": combined_score,
            "bbox_xyxy": (
                list(existing_item["bbox_xyxy"]) if existing_item.get("bbox_xyxy") is not None else None
            ),
            "reference_support_scores": [float(score) for score in support_scores],
        }

    if best_match is None:
        return None

    support_ok = float(best_match["support_score"]) >= float(min_support_score)
    hint_ok = float(best_match["hint_iou"]) >= float(min_hint_iou)
    duplicate_ok = candidate_item is not None and float(best_match["duplicate_iou"]) >= 0.80
    if duplicate_ok and float(best_match["support_score"]) >= 0.60:
        return best_match
    if support_ok and (hint_ok or candidate_item is None):
        return best_match
    return None


def estimate_target_hint_bbox_xyxy(
    *,
    target_frame_index: int,
    reference_masks: list[dict[str, int]],
    mask_items_by_frame: dict[int, list[dict[str, Any]]],
    frame_w: int,
    frame_h: int,
) -> tuple[int, int, int, int] | None:
    references: list[tuple[int, tuple[int, int, int, int]]] = []
    for ref in reference_masks:
        frame_index = int(ref["frame_index"])
        local_id = int(ref["local_id"])
        item = find_mask_item(mask_items_by_frame, frame_index=frame_index, local_id=local_id)
        bbox_xyxy = item.get("bbox_xyxy") if item is not None else None
        if bbox_xyxy is None:
            continue
        references.append((frame_index, bbox_xyxy))

    if not references:
        return None

    references.sort(key=lambda pair: pair[0])
    before = [pair for pair in references if pair[0] < int(target_frame_index)]
    after = [pair for pair in references if pair[0] > int(target_frame_index)]

    if before and after:
        before_idx, (bx1, by1, bx2, by2) = before[-1]
        after_idx, (ax1, ay1, ax2, ay2) = after[0]
        span = max(1, int(after_idx) - int(before_idx))
        alpha = float(int(target_frame_index) - int(before_idx)) / float(span)
        x1 = int(round((1.0 - alpha) * bx1 + alpha * ax1))
        y1 = int(round((1.0 - alpha) * by1 + alpha * ay1))
        x2 = int(round((1.0 - alpha) * bx2 + alpha * ax2))
        y2 = int(round((1.0 - alpha) * by2 + alpha * ay2))
    else:
        _frame_index, (x1, y1, x2, y2) = min(
            references,
            key=lambda pair: abs(int(pair[0]) - int(target_frame_index)),
        )

    x1 = max(0, min(int(frame_w) - 1, int(x1)))
    y1 = max(0, min(int(frame_h) - 1, int(y1)))
    x2 = max(0, min(int(frame_w) - 1, int(x2)))
    y2 = max(0, min(int(frame_h) - 1, int(y2)))
    if x2 <= x1 or y2 <= y1:
        return None
    return (x1, y1, x2, y2)


def build_point_candidates(
    *,
    target_frame_index: int,
    reference_masks: list[dict[str, int]],
    mask_items_by_frame: dict[int, list[dict[str, Any]]],
    frame_w: int,
    frame_h: int,
    max_candidates: int,
) -> list[tuple[int, int]]:
    references: list[tuple[int, dict[str, Any]]] = []
    for ref in reference_masks:
        frame_index = int(ref["frame_index"])
        local_id = int(ref["local_id"])
        item = find_mask_item(mask_items_by_frame, frame_index=frame_index, local_id=local_id)
        if item is None or item.get("centroid") is None:
            continue
        references.append((frame_index, item))

    if not references:
        return []

    references.sort(key=lambda pair: pair[0])
    candidate_points: list[tuple[int, int]] = []

    hint_bbox_xyxy = estimate_target_hint_bbox_xyxy(
        target_frame_index=target_frame_index,
        reference_masks=reference_masks,
        mask_items_by_frame=mask_items_by_frame,
        frame_w=frame_w,
        frame_h=frame_h,
    )
    if hint_bbox_xyxy is not None:
        x1, y1, x2, y2 = hint_bbox_xyxy
        cx = int(round((x1 + x2) / 2.0))
        cy = int(round((y1 + y2) / 2.0))
        candidate_points.extend(
            [
                (cx, cy),
                (int(round((x1 + cx) / 2.0)), cy),
                (int(round((x2 + cx) / 2.0)), cy),
                (cx, int(round((y1 + cy) / 2.0))),
                (cx, int(round((y2 + cy) / 2.0))),
            ]
        )

    before = [pair for pair in references if pair[0] < int(target_frame_index)]
    after = [pair for pair in references if pair[0] > int(target_frame_index)]
    if before and after:
        before_idx, before_item = before[-1]
        after_idx, after_item = after[0]
        if before_item.get("centroid") is not None and after_item.get("centroid") is not None:
            span = max(1, int(after_idx) - int(before_idx))
            alpha = float(int(target_frame_index) - int(before_idx)) / float(span)
            interp_x = int(round((1.0 - alpha) * before_item["centroid"][0] + alpha * after_item["centroid"][0]))
            interp_y = int(round((1.0 - alpha) * before_item["centroid"][1] + alpha * after_item["centroid"][1]))
            candidate_points.append((interp_x, interp_y))

        if before_item.get("bbox_xyxy") is not None and after_item.get("bbox_xyxy") is not None:
            bx1, by1, bx2, by2 = before_item["bbox_xyxy"]
            ax1, ay1, ax2, ay2 = after_item["bbox_xyxy"]
            span = max(1, int(after_idx) - int(before_idx))
            alpha = float(int(target_frame_index) - int(before_idx)) / float(span)
            center_x = int(round((1.0 - alpha) * ((bx1 + bx2) / 2.0) + alpha * ((ax1 + ax2) / 2.0)))
            center_y = int(round((1.0 - alpha) * ((by1 + by2) / 2.0) + alpha * ((ay1 + ay2) / 2.0)))
            candidate_points.append((center_x, center_y))

    centroids = [item["centroid"] for _, item in references if item.get("centroid") is not None]
    if centroids:
        avg_x = int(round(sum(pt[0] for pt in centroids) / float(len(centroids))))
        avg_y = int(round(sum(pt[1] for pt in centroids) / float(len(centroids))))
        candidate_points.append((avg_x, avg_y))

    references_by_distance = sorted(
        references,
        key=lambda pair: abs(int(pair[0]) - int(target_frame_index)),
    )
    for _frame_index, item in references_by_distance:
        if item.get("centroid") is not None:
            candidate_points.append(tuple(int(v) for v in item["centroid"]))
        if item.get("bbox_xyxy") is not None:
            x1, y1, x2, y2 = item["bbox_xyxy"]
            candidate_points.append((int(round((x1 + x2) / 2.0)), int(round((y1 + y2) / 2.0))))

    widths = []
    heights = []
    for _frame_index, item in references:
        bbox = item.get("bbox_xyxy")
        if bbox is None:
            continue
        widths.append(max(1, int(bbox[2] - bbox[0])))
        heights.append(max(1, int(bbox[3] - bbox[1])))
    jitter_x = max(4, int(round((sum(widths) / float(len(widths))) * 0.15))) if widths else 6
    jitter_y = max(4, int(round((sum(heights) / float(len(heights))) * 0.15))) if heights else 6

    expanded: list[tuple[int, int]] = []
    for point in candidate_points:
        normalized = normalize_point_xy(point, frame_w=frame_w, frame_h=frame_h)
        if normalized is None:
            continue
        expanded.append(normalized)
        expanded.append(normalize_point_xy((normalized[0] + jitter_x, normalized[1]), frame_w, frame_h) or normalized)
        expanded.append(normalize_point_xy((normalized[0] - jitter_x, normalized[1]), frame_w, frame_h) or normalized)
        expanded.append(normalize_point_xy((normalized[0], normalized[1] + jitter_y), frame_w, frame_h) or normalized)
        expanded.append(normalize_point_xy((normalized[0], normalized[1] - jitter_y), frame_w, frame_h) or normalized)

    deduped = dedupe_points([pt for pt in expanded if pt is not None])
    return deduped[: max(1, int(max_candidates))]


def propose_refinement_point(
    *,
    candidate_item: dict[str, Any],
    hint_bbox_xyxy: tuple[int, int, int, int] | None,
    reference_item: dict[str, Any] | None,
    positive_points: list[tuple[int, int]],
    frame_w: int,
    frame_h: int,
) -> tuple[int, int] | None:
    candidate_mask = candidate_item.get("mask")
    candidate_bbox = candidate_item.get("bbox_xyxy")
    anchor_bbox = hint_bbox_xyxy or (reference_item.get("bbox_xyxy") if reference_item else None)
    if candidate_mask is None or candidate_bbox is None or anchor_bbox is None:
        return None

    ax1, ay1, ax2, ay2 = anchor_bbox
    cx1, cy1, cx2, cy2 = candidate_bbox
    anchor_w, anchor_h = bbox_dims_xyxy(anchor_bbox)
    if anchor_w <= 0 or anchor_h <= 0:
        return None

    orientation = "horizontal" if anchor_w >= anchor_h else "vertical"
    positive_set = {
        normalize_point_xy(point, frame_w=frame_w, frame_h=frame_h)
        for point in positive_points
    }

    candidates: list[tuple[int, int]] = []
    if orientation == "horizontal":
        anchor_mid_y = int(round((ay1 + ay2) / 2.0))
        left_gap = max(0, int(cx1) - int(ax1))
        right_gap = max(0, int(ax2) - int(cx2))
        if left_gap >= right_gap and left_gap > max(6, int(round(anchor_w * 0.08))):
            candidates.append((int(round((ax1 + cx1) / 2.0)), anchor_mid_y))
        if right_gap > 0:
            candidates.append((int(round((cx2 + ax2) / 2.0)), anchor_mid_y))
        for frac in (0.2, 0.35, 0.65, 0.8):
            candidates.append((int(round(ax1 + frac * anchor_w)), anchor_mid_y))
    else:
        anchor_mid_x = int(round((ax1 + ax2) / 2.0))
        top_gap = max(0, int(cy1) - int(ay1))
        bottom_gap = max(0, int(ay2) - int(cy2))
        if top_gap >= bottom_gap and top_gap > max(6, int(round(anchor_h * 0.08))):
            candidates.append((anchor_mid_x, int(round((ay1 + cy1) / 2.0))))
        if bottom_gap > 0:
            candidates.append((anchor_mid_x, int(round((cy2 + ay2) / 2.0))))
        for frac in (0.2, 0.35, 0.65, 0.8):
            candidates.append((anchor_mid_x, int(round(ay1 + frac * anchor_h))))

    reference_centroid = reference_item.get("centroid") if reference_item else None
    if reference_centroid is not None:
        candidates.insert(0, (int(reference_centroid[0]), int(reference_centroid[1])))

    for point in candidates:
        normalized = normalize_point_xy(point, frame_w=frame_w, frame_h=frame_h)
        if normalized is None or normalized in positive_set:
            continue
        px, py = normalized
        if 0 <= py < candidate_mask.shape[0] and 0 <= px < candidate_mask.shape[1]:
            if bool(candidate_mask[py, px]):
                continue
        return normalized
    return None


def assess_gap_fill_candidate_geometry(
    *,
    candidate_item: dict[str, Any],
    reference_item: dict[str, Any] | None,
    hint_bbox_xyxy: tuple[int, int, int, int] | None,
) -> dict[str, Any]:
    candidate_bbox = candidate_item.get("bbox_xyxy")
    candidate_area = float(candidate_item.get("area") or 0.0)
    anchor_bbox = hint_bbox_xyxy or (reference_item.get("bbox_xyxy") if reference_item else None)
    reference_bbox = reference_item.get("bbox_xyxy") if reference_item else None
    reference_area = float(reference_item.get("area") or 0.0) if reference_item else 0.0

    cand_w, cand_h = bbox_dims_xyxy(candidate_bbox)
    ref_w, ref_h = bbox_dims_xyxy(reference_bbox)
    anchor_w, anchor_h = bbox_dims_xyxy(anchor_bbox)
    cand_center = bbox_center_xyxy(candidate_bbox)
    ref_center = bbox_center_xyxy(reference_bbox)

    area_ratio = (candidate_area / reference_area) if reference_area > 0 else None
    width_ratio = (float(cand_w) / float(ref_w)) if ref_w > 0 else None
    height_ratio = (float(cand_h) / float(ref_h)) if ref_h > 0 else None
    bbox_iou = bbox_iou_xyxy(candidate_bbox, anchor_bbox)
    center_dist = None
    if cand_center is not None and ref_center is not None:
        center_dist = math.hypot(cand_center[0] - ref_center[0], cand_center[1] - ref_center[1])

    status = "ok"
    reason = ""

    if area_ratio is not None and reference_bbox is not None:
        aspect_ref = float(ref_w + 1) / float(ref_h + 1)
        aspect_cand = float(cand_w + 1) / float(cand_h + 1)
        aspect_ratio_delta = max(aspect_cand / aspect_ref, aspect_ref / aspect_cand)
        major_axis_ratio = max(
            (float(cand_w) / float(ref_w)) if ref_w > 0 else 0.0,
            (float(cand_h) / float(ref_h)) if ref_h > 0 else 0.0,
        )
        minor_axis_ratio = min(
            (float(cand_w) / float(ref_w)) if ref_w > 0 else 0.0,
            (float(cand_h) / float(ref_h)) if ref_h > 0 else 0.0,
        )

        if (
            area_ratio >= 1.75
            and (bbox_iou <= 0.35 or aspect_ratio_delta >= 2.5)
        ):
            status = "reject"
            reason = "candidate_geometry_wildly_larger_than_reference"
        elif center_dist is not None and max(ref_w, ref_h) > 0 and center_dist > (0.85 * max(ref_w, ref_h)) and bbox_iou <= 0.2:
            status = "reject"
            reason = "candidate_geometry_far_from_reference"
        elif area_ratio <= 0.35 or minor_axis_ratio <= 0.4 or (major_axis_ratio <= 0.6 and bbox_iou <= 0.55):
            status = "partial"
            reason = "candidate_geometry_partial_coverage"
        return {
            "status": status,
            "reason": reason,
            "area_ratio": area_ratio,
            "width_ratio": width_ratio,
            "height_ratio": height_ratio,
            "bbox_iou": bbox_iou,
            "center_distance": center_dist,
            "reference_bbox_xyxy": list(reference_bbox) if reference_bbox is not None else None,
            "anchor_bbox_xyxy": list(anchor_bbox) if anchor_bbox is not None else None,
            "candidate_bbox_xyxy": list(candidate_bbox) if candidate_bbox is not None else None,
        }

    if anchor_bbox is not None and anchor_w > 0 and anchor_h > 0:
        anchor_iou = bbox_iou_xyxy(candidate_bbox, anchor_bbox)
        if anchor_iou <= 0.1 and candidate_area > float(anchor_w * anchor_h) * 0.75:
            status = "reject"
            reason = "candidate_geometry_outside_anchor"

    return {
        "status": status,
        "reason": reason,
        "area_ratio": area_ratio,
        "width_ratio": width_ratio,
        "height_ratio": height_ratio,
        "bbox_iou": bbox_iou,
        "center_distance": center_dist,
        "reference_bbox_xyxy": list(reference_bbox) if reference_bbox is not None else None,
        "anchor_bbox_xyxy": list(anchor_bbox) if anchor_bbox is not None else None,
        "candidate_bbox_xyxy": list(candidate_bbox) if candidate_bbox is not None else None,
    }


def build_point_prompt_points(
    *,
    positive_points: list[tuple[int, int]],
    hint_bbox_xyxy: tuple[int, int, int, int] | None,
    frame_w: int,
    frame_h: int,
) -> list[tuple[int, int, int]]:
    points: list[tuple[int, int, int]] = []

    def _append(point: tuple[int, int] | None, label: int) -> None:
        if point is None:
            return
        normalized = normalize_point_xy(point, frame_w=frame_w, frame_h=frame_h)
        if normalized is None:
            return
        entry = (int(normalized[0]), int(normalized[1]), int(label))
        if entry not in points:
            points.append(entry)

    normalized_positive = [
        normalize_point_xy(point, frame_w=frame_w, frame_h=frame_h)
        for point in positive_points
    ]
    normalized_positive = [point for point in normalized_positive if point is not None]
    if not normalized_positive:
        return []
    deduped_positive = dedupe_points([(int(x), int(y)) for x, y in normalized_positive], min_distance=4.0)
    for point in deduped_positive:
        _append(point, 1)

    if hint_bbox_xyxy is not None:
        x1, y1, x2, y2 = hint_bbox_xyxy
        width = max(4, int(x2 - x1))
        height = max(4, int(y2 - y1))
        dx = max(2, int(round(width * 0.12)))
        dy = max(2, int(round(height * 0.12)))
        for cx, cy in deduped_positive:
            for offset in ((dx, 0), (-dx, 0), (0, dy), (0, -dy)):
                _append((cx + offset[0], cy + offset[1]), 1)
        center_x = int(round((x1 + x2) / 2.0))
        center_y = int(round((y1 + y2) / 2.0))
        _append((center_x, center_y), 1)
        border = max(3, int(round(min(width, height) * 0.08)))
        for negative in (
            (x1 - border, center_y),
            (x2 + border, center_y),
            (center_x, y1 - border),
            (center_x, y2 + border),
        ):
            _append(negative, 0)

    return points


def render_point_prompt_debug(
    frame_bgr: Any,
    *,
    prompt_points: list[tuple[int, int, int]],
    hint_bbox_xyxy: tuple[int, int, int, int] | None = None,
    existing_items: list[dict[str, Any]] | None = None,
) -> Any:
    output = frame_bgr.copy()
    if existing_items:
        output = draw_mask_focus(output, focus_items=[], existing_items=existing_items)
    if hint_bbox_xyxy is not None:
        x1, y1, x2, y2 = hint_bbox_xyxy
        cv2.rectangle(output, (int(x1), int(y1)), (int(x2), int(y2)), (255, 220, 40), 2)
    for idx, (x, y, label) in enumerate(prompt_points, start=1):
        color = (60, 220, 60) if int(label) > 0 else (40, 60, 220)
        cv2.circle(output, (int(x), int(y)), 6, color, -1)
        cv2.circle(output, (int(x), int(y)), 9, (255, 255, 255), 1)
        cv2.putText(
            output,
            f"{'+' if int(label) > 0 else '-'}{idx}",
            (int(x) + 8, int(y) - 6),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )
        cv2.putText(
            output,
            f"{'+' if int(label) > 0 else '-'}{idx}",
            (int(x) + 8, int(y) - 6),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            (0, 0, 0),
            1,
            cv2.LINE_AA,
        )
    return output


def serialize_backend_outputs(
    *,
    frame_index: int,
    outputs: dict[str, Any],
    frame_h: int,
    frame_w: int,
) -> dict[str, Any]:
    masks_with_ids = iter_output_masks_with_ids(
        outputs,
        frame_h=frame_h,
        frame_w=frame_w,
    )
    out_obj_ids = _to_serializable_list(outputs.get("out_obj_ids"))
    out_probs = _to_serializable_list(outputs.get("out_probs"))
    out_tracker_probs = _to_serializable_list(outputs.get("out_tracker_probs"))
    out_boxes_xywh = _to_serializable_list(outputs.get("out_boxes_xywh"))

    rle_masks: list[dict[str, Any]] = []
    obj_ids: list[int] = []
    for obj_id, mask in masks_with_ids:
        if mask.shape != (frame_h, frame_w):
            mask = cv2.resize(
                mask.astype(np.float32),
                (int(frame_w), int(frame_h)),
                interpolation=cv2.INTER_NEAREST,
            ) > 0.5
        obj_ids.append(int(obj_id))
        rle_masks.append(encode_binary_mask_to_rle(mask))

    return {
        "frame_index": int(frame_index),
        "out_obj_ids": out_obj_ids if out_obj_ids else obj_ids,
        "out_probs": out_probs,
        "out_tracker_probs": out_tracker_probs,
        "out_boxes_xywh": out_boxes_xywh,
        "out_binary_masks_rle": rle_masks,
    }


def unwrap_backend_outputs(response: Any) -> dict[str, Any]:
    """Accept both backend response shapes: flat outputs or {'outputs': ...}."""
    if not isinstance(response, dict):
        return {}
    nested = response.get("outputs")
    if isinstance(nested, dict):
        return nested
    return response


def summarize_backend_outputs(
    *,
    frame_index: int,
    outputs: dict[str, Any],
    frame_h: int,
    frame_w: int,
) -> dict[str, Any]:
    serialized = serialize_backend_outputs(
        frame_index=frame_index,
        outputs=outputs,
        frame_h=frame_h,
        frame_w=frame_w,
    )
    return {
        "frame_index": int(frame_index),
        "num_masks": len(serialized.get("out_binary_masks_rle") or []),
        "out_obj_ids": [int(x) for x in serialized.get("out_obj_ids") or []],
        "out_probs": list(serialized.get("out_probs") or []),
        "out_boxes_xywh": list(serialized.get("out_boxes_xywh") or []),
    }


def select_point_prompt_candidate(
    *,
    frame_index: int,
    outputs: dict[str, Any],
    requested_obj_id: int,
    frame_h: int,
    frame_w: int,
) -> dict[str, Any] | None:
    serialized = serialize_backend_outputs(
        frame_index=frame_index,
        outputs=outputs,
        frame_h=frame_h,
        frame_w=frame_w,
    )
    items = decode_frame_row_masks(serialized, frame_h=frame_h, frame_w=frame_w)
    if not items:
        return None
    obj_ids = [int(x) for x in serialized.get("out_obj_ids", [])]
    if requested_obj_id in obj_ids:
        try:
            idx = obj_ids.index(int(requested_obj_id))
            if idx < len(items):
                return items[idx]
        except Exception:
            pass
    items.sort(key=lambda item: (float(item.get("score") or 0.0), float(item.get("area") or 0.0)), reverse=True)
    return items[0]


def draw_mask_focus(
    frame_bgr: Any,
    *,
    focus_items: list[dict[str, Any]],
    existing_items: list[dict[str, Any]] | None = None,
    focus_label_prefix: str = "",
) -> Any:
    output = frame_bgr.copy()
    if existing_items:
        muted = np.zeros_like(output)
        for item in existing_items:
            mask = item.get("mask")
            if mask is None or mask.shape != output.shape[:2]:
                continue
            muted[mask] = (80, 80, 80)
        output = cv2.addWeighted(output, 1.0, muted, 0.18, 0.0)

    overlay = np.zeros_like(output)
    for idx, item in enumerate(focus_items, start=1):
        mask = item.get("mask")
        if mask is None or mask.shape != output.shape[:2]:
            continue
        color = object_color(200 + idx)
        overlay[mask] = color
        contours, _ = cv2.findContours((mask.astype(np.uint8) * 255), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(output, contours, -1, color, 2)
    output = cv2.addWeighted(output, 1.0, overlay, 0.40, 0.0)

    for idx, item in enumerate(focus_items, start=1):
        centroid = item.get("centroid")
        bbox = item.get("bbox_xyxy")
        label = f"{focus_label_prefix}{idx}"
        local_id = item.get("local_id")
        if local_id is not None:
            label = f"{focus_label_prefix}l{int(local_id)}"
        if centroid is not None:
            x, y = centroid
        elif bbox is not None:
            x = int(round((bbox[0] + bbox[2]) / 2.0))
            y = int(round((bbox[1] + bbox[3]) / 2.0))
        else:
            x, y = 12, 24
        x = max(4, min(output.shape[1] - 120, int(x)))
        y = max(18, min(output.shape[0] - 8, int(y)))
        cv2.putText(output, label, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2, cv2.LINE_AA)
        cv2.putText(output, label, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 1, cv2.LINE_AA)
    return output


def append_mask_to_frame_row(
    frame_row: dict[str, Any],
    *,
    frame_h: int,
    frame_w: int,
    mask_item: dict[str, Any],
) -> int:
    new_local_id = next_local_id_for_frame(frame_row)
    out_obj_ids = list(frame_row.get("out_obj_ids") or [])
    out_probs = list(frame_row.get("out_probs") or [])
    out_tracker_probs = list(frame_row.get("out_tracker_probs") or [])
    out_boxes_xywh = list(frame_row.get("out_boxes_xywh") or [])
    out_binary_masks_rle = list(frame_row.get("out_binary_masks_rle") or [])

    out_obj_ids.append(int(new_local_id))
    out_probs.append(mask_item.get("score"))
    if out_tracker_probs:
        out_tracker_probs.append(None)
    box_xywh = mask_item.get("box_xywh")
    if box_xywh is None:
        box_xywh = bbox_xywh_from_xyxy(mask_item.get("bbox_xyxy"), frame_h=frame_h, frame_w=frame_w)
    out_boxes_xywh.append(box_xywh)
    out_binary_masks_rle.append(
        normalize_rle_mask(mask_item.get("mask_rle"), frame_h=frame_h, frame_w=frame_w)
    )

    frame_row["out_obj_ids"] = out_obj_ids
    frame_row["out_probs"] = out_probs
    frame_row["out_tracker_probs"] = out_tracker_probs
    frame_row["out_boxes_xywh"] = out_boxes_xywh
    frame_row["out_binary_masks_rle"] = out_binary_masks_rle
    return int(new_local_id)


def remove_local_id_from_frame_row(frame_row: dict[str, Any], *, local_id: int) -> bool:
    target_local_id = int(local_id)
    raw_local_ids = frame_row_local_ids(frame_row)
    raw_masks = list(frame_row.get("out_binary_masks_rle") or [])
    raw_boxes = list(frame_row.get("out_boxes_xywh") or [])
    raw_probs = list(frame_row.get("out_probs") or [])
    raw_tracker_probs = list(frame_row.get("out_tracker_probs") or [])

    kept_obj_ids: list[int] = []
    kept_masks: list[Any] = []
    kept_boxes: list[Any] = []
    kept_probs: list[Any] = []
    kept_tracker_probs: list[Any] = []
    removed = False

    for idx, mask_rle in enumerate(raw_masks):
        existing_local_id = raw_local_ids[idx] if idx < len(raw_local_ids) else idx + 1
        if int(existing_local_id) == target_local_id:
            removed = True
            continue
        kept_obj_ids.append(int(existing_local_id))
        kept_masks.append(mask_rle)
        if idx < len(raw_boxes):
            kept_boxes.append(raw_boxes[idx])
        if idx < len(raw_probs):
            kept_probs.append(raw_probs[idx])
        if idx < len(raw_tracker_probs):
            kept_tracker_probs.append(raw_tracker_probs[idx])

    if removed:
        frame_row["out_obj_ids"] = kept_obj_ids
        frame_row["out_binary_masks_rle"] = kept_masks
        frame_row["out_boxes_xywh"] = kept_boxes
        frame_row["out_probs"] = kept_probs
        frame_row["out_tracker_probs"] = kept_tracker_probs
    return removed


def build_stage_helpers() -> SimpleNamespace:
    return SimpleNamespace(
        cv2=cv2,
        np=np,
        read_video_frame=read_video_frame,
        write_json=write_json,
        extract_json_object=extract_json_object,
        build_index_windows=build_index_windows,
        build_collage=build_collage,
        decode_frame_row_masks=decode_frame_row_masks,
        draw_overlay_with_labels=draw_overlay_with_labels,
        binary_mask_iou=binary_mask_iou,
        mask_match_support_score=mask_match_support_score,
        normalize_point_xy=normalize_point_xy,
        dedupe_points=dedupe_points,
        request_missing_mask_issues=request_missing_mask_issues,
        sanitize_missing_issue_response=sanitize_missing_issue_response,
        build_inventory_text=build_inventory_text,
        find_mask_item=find_mask_item,
        draw_mask_focus=draw_mask_focus,
        estimate_target_hint_bbox_xyxy=estimate_target_hint_bbox_xyxy,
        find_existing_issue_match=find_existing_issue_match,
        build_point_candidates=build_point_candidates,
        build_point_prompt_points=build_point_prompt_points,
        render_point_prompt_debug=render_point_prompt_debug,
        unwrap_backend_outputs=unwrap_backend_outputs,
        summarize_backend_outputs=summarize_backend_outputs,
        select_point_prompt_candidate=select_point_prompt_candidate,
        assess_gap_fill_candidate_geometry=assess_gap_fill_candidate_geometry,
        propose_refinement_point=propose_refinement_point,
        request_gap_fill_verdict=request_gap_fill_verdict,
        append_mask_to_frame_row=append_mask_to_frame_row,
        request_outlier_mask_verdict=request_outlier_mask_verdict,
        remove_local_id_from_frame_row=remove_local_id_from_frame_row,
    )


def resolve_enabled_stages(args: argparse.Namespace) -> list[str]:
    explicit = [str(stage).strip() for stage in (args.stage or []) if str(stage).strip()]
    if "missed_creatures" in explicit:
        raise SystemExit(DEPRECATED_MISSED_CREATURES_MESSAGE)
    if explicit:
        return explicit

    stages: list[str] = []
    if bool(args.find_missed_creatures):
        raise SystemExit(DEPRECATED_MISSED_CREATURES_MESSAGE)
    if bool(args.fill_missing_masks):
        stages.append("gap_fill")
    if bool(args.filter_outlier_masks) and bool(args.fill_missing_masks):
        stages.append("outlier_filter")
    stages.append("id_reassign")
    return stages


def repair_missing_masks_in_window(
    *,
    args: argparse.Namespace,
    send_generate_request_fn: Any,
    missing_system_prompt: str,
    verify_system_prompt: str,
    backend: Any,
    session_id: str,
    video_path: str,
    frame_h: int,
    frame_w: int,
    window_dir: Path,
    window_frame_indices: list[int],
    working_frame_rows_by_index: dict[int, dict[str, Any]],
    local_ids_by_frame: dict[int, list[int]],
    mask_items_by_frame: dict[int, list[dict[str, Any]]],
    raw_collage_path: str,
    overlay_before_gap_fill_path: str,
    max_json_retries: int,
) -> tuple[dict[str, Any], dict[int, list[int]], dict[int, list[dict[str, Any]]]]:
    from postprocess_stage_gap_fill import repair_missing_masks_in_window_stage

    return repair_missing_masks_in_window_stage(
        args=args,
        send_generate_request_fn=send_generate_request_fn,
        missing_system_prompt=missing_system_prompt,
        verify_system_prompt=verify_system_prompt,
        backend=backend,
        session_id=session_id,
        video_path=video_path,
        frame_h=frame_h,
        frame_w=frame_w,
        window_dir=window_dir,
        window_frame_indices=window_frame_indices,
        working_frame_rows_by_index=working_frame_rows_by_index,
        local_ids_by_frame=local_ids_by_frame,
        mask_items_by_frame=mask_items_by_frame,
        raw_collage_path=raw_collage_path,
        overlay_before_gap_fill_path=overlay_before_gap_fill_path,
        max_json_retries=max_json_retries,
        helpers=build_stage_helpers(),
    )


def run_missed_creatures_stage(
    *,
    args: argparse.Namespace,
    send_generate_request_fn: Any,
    detection_system_prompt: str,
    verify_system_prompt: str,
    backend: Any,
    session_id: str,
    video_path: str,
    frame_h: int,
    frame_w: int,
    valid_frame_indices: list[int],
    invalid_frame_indices: set[int],
    working_frame_rows_by_index: dict[int, dict[str, Any]],
    output_dir: Path,
    max_json_retries: int,
) -> dict[str, Any]:
    from postprocess_stage_missed_creatures import discover_missed_creatures_stage

    return discover_missed_creatures_stage(
        args=args,
        send_generate_request_fn=send_generate_request_fn,
        detection_system_prompt=detection_system_prompt,
        verify_system_prompt=verify_system_prompt,
        backend=backend,
        session_id=session_id,
        video_path=video_path,
        frame_h=frame_h,
        frame_w=frame_w,
        valid_frame_indices=valid_frame_indices,
        invalid_frame_indices=invalid_frame_indices,
        working_frame_rows_by_index=working_frame_rows_by_index,
        output_dir=output_dir,
        max_json_retries=max_json_retries,
        helpers=build_stage_helpers(),
    )


def filter_outlier_gap_fill_masks(
    *,
    args: argparse.Namespace,
    send_generate_request_fn: Any,
    outlier_system_prompt: str,
    video_path: str,
    frame_h: int,
    frame_w: int,
    window_dir: Path,
    working_frame_rows_by_index: dict[int, dict[str, Any]],
    local_ids_by_frame: dict[int, list[int]],
    mask_items_by_frame: dict[int, list[dict[str, Any]]],
    gap_fill_report: dict[str, Any] | None,
    max_json_retries: int,
) -> tuple[dict[str, Any], dict[int, list[int]], dict[int, list[dict[str, Any]]]]:
    from postprocess_stage_outlier_filter import filter_outlier_gap_fill_masks_stage

    return filter_outlier_gap_fill_masks_stage(
        args=args,
        send_generate_request_fn=send_generate_request_fn,
        outlier_system_prompt=outlier_system_prompt,
        video_path=video_path,
        frame_h=frame_h,
        frame_w=frame_w,
        window_dir=window_dir,
        working_frame_rows_by_index=working_frame_rows_by_index,
        local_ids_by_frame=local_ids_by_frame,
        mask_items_by_frame=mask_items_by_frame,
        gap_fill_report=gap_fill_report,
        max_json_retries=max_json_retries,
        helpers=build_stage_helpers(),
    )


def apply_window_assignments(
    *,
    window_frame_indices: list[int],
    mask_items_by_frame: dict[int, list[dict[str, Any]]],
    working_frame_rows_by_index: dict[int, dict[str, Any]],
    frame_h: int,
    frame_w: int,
    existing_frame_assignments: dict[int, dict[int, int]],
    parsed_assignments: dict[int, dict[int, str | None]],
    next_global_id: int,
    allow_drop_assignments: bool,
    assignment_history_frames: int,
    assignment_heuristic_min_score: float,
) -> tuple[dict[int, dict[int, int]], dict[int, list[int]], dict[int, list[int]], int]:
    resolved: dict[int, dict[int, int]] = {
        int(frame_idx): {int(k): int(v) for k, v in mapping.items()}
        for frame_idx, mapping in existing_frame_assignments.items()
        if frame_idx in window_frame_indices
    }
    dropped_by_frame: dict[int, list[int]] = {}
    ignored_drops_by_frame: dict[int, list[int]] = {}
    temp_label_to_gid: dict[str, int] = {}
    anchor_global_ids = {
        int(global_id)
        for mapping in resolved.values()
        for global_id in mapping.values()
    }

    for pos, frame_index in enumerate(window_frame_indices):
        mask_items = mask_items_by_frame.get(frame_index, [])
        if not mask_items:
            resolved.setdefault(int(frame_index), {})
            continue

        frame_fixed = resolved.setdefault(int(frame_index), {})
        frame_model_map = parsed_assignments.get(frame_index, {})
        assigned_global_ids = set(int(gid) for gid in frame_fixed.values())
        prior_pairs = collect_recent_assignment_history(
            frame_index=frame_index,
            resolved_window_assignments=resolved,
            existing_frame_assignments=existing_frame_assignments,
            mask_items_by_frame=mask_items_by_frame,
            working_frame_rows_by_index=working_frame_rows_by_index,
            frame_h=frame_h,
            frame_w=frame_w,
            history_frame_budget=assignment_history_frames,
        )

        for item in mask_items:
            local_id = int(item["local_id"])
            if local_id in frame_fixed:
                continue

            model_label = frame_model_map.get(local_id)
            normalized = normalize_track_label(model_label)
            chosen_global_id: int | None = None
            heuristic_gid: int | None = None

            for prior_items, prior_mapping in prior_pairs:
                heuristic_gid = heuristic_match_global_id(
                    current_item=item,
                    prior_items=prior_items,
                    prior_assignments=prior_mapping,
                    disallowed_global_ids=assigned_global_ids,
                    min_score=assignment_heuristic_min_score,
                )
                if heuristic_gid is not None:
                    break

            if normalized is None:
                chosen_global_id = heuristic_gid
            elif normalized.lower() == "drop":
                if bool(allow_drop_assignments):
                    dropped_by_frame.setdefault(int(frame_index), []).append(local_id)
                    continue
                ignored_drops_by_frame.setdefault(int(frame_index), []).append(local_id)
                chosen_global_id = heuristic_gid
            elif re.fullmatch(r"g\d+", normalized.lower()):
                requested_gid = int(normalized[1:])
                if requested_gid in assigned_global_ids:
                    chosen_global_id = None
                elif requested_gid in anchor_global_ids:
                    chosen_global_id = requested_gid
                elif heuristic_gid == requested_gid:
                    chosen_global_id = requested_gid
                else:
                    chosen_global_id = heuristic_gid
            else:
                temp_key = normalized.lower()
                chosen_global_id = temp_label_to_gid.get(temp_key)
                if chosen_global_id in assigned_global_ids:
                    chosen_global_id = None
                if chosen_global_id is None and heuristic_gid is not None:
                    chosen_global_id = heuristic_gid
                    temp_label_to_gid[temp_key] = int(chosen_global_id)
                if chosen_global_id is None:
                    chosen_global_id = int(next_global_id)
                    next_global_id += 1
                    temp_label_to_gid[temp_key] = chosen_global_id

            if chosen_global_id in assigned_global_ids:
                chosen_global_id = None

            if chosen_global_id is None:
                if heuristic_gid is not None:
                    chosen_global_id = heuristic_gid

            if chosen_global_id is None:
                chosen_global_id = int(next_global_id)
                next_global_id += 1

            frame_fixed[local_id] = int(chosen_global_id)
            assigned_global_ids.add(int(chosen_global_id))

    return resolved, dropped_by_frame, ignored_drops_by_frame, next_global_id


def relabel_frame_row(
    frame_row: dict[str, Any],
    *,
    frame_h: int,
    frame_w: int,
    frame_assignment: dict[int, int],
    dropped_local_ids: set[int],
) -> dict[str, Any]:
    local_ids = frame_row_local_ids(frame_row)
    raw_masks = list(frame_row.get("out_binary_masks_rle") or [])
    raw_boxes = list(frame_row.get("out_boxes_xywh") or [])
    raw_probs = list(frame_row.get("out_probs") or [])
    raw_tracker_probs = list(frame_row.get("out_tracker_probs") or [])

    kept_obj_ids: list[int] = []
    kept_masks: list[dict[str, Any]] = []
    kept_boxes: list[Any] = []
    kept_probs: list[Any] = []
    kept_tracker_probs: list[Any] = []

    for idx, mask_rle in enumerate(raw_masks):
        local_id = local_ids[idx] if idx < len(local_ids) else idx + 1
        if int(local_id) in dropped_local_ids:
            continue
        global_id = frame_assignment.get(int(local_id))
        if global_id is None:
            continue
        kept_obj_ids.append(int(global_id))
        kept_masks.append(normalize_rle_mask(mask_rle, frame_h=frame_h, frame_w=frame_w))
        if idx < len(raw_boxes):
            kept_boxes.append(raw_boxes[idx])
        if idx < len(raw_probs):
            kept_probs.append(raw_probs[idx])
        if idx < len(raw_tracker_probs):
            kept_tracker_probs.append(raw_tracker_probs[idx])

    return {
        "frame_index": int(frame_row.get("frame_index", 0)),
        "out_obj_ids": kept_obj_ids,
        "out_probs": kept_probs,
        "out_tracker_probs": kept_tracker_probs,
        "out_boxes_xywh": kept_boxes,
        "out_binary_masks_rle": kept_masks,
    }


def render_overlay_video(
    *,
    video_path: str,
    output_path: str,
    frame_rows_by_index: dict[int, dict[str, Any]],
    frame_size_hw: tuple[int, int],
    fps: float,
    codec: str,
) -> None:
    frame_h, frame_w = frame_size_hw
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video for overlay rendering: {video_path}")

    writer = cv2.VideoWriter(
        str(output_path),
        cv2.VideoWriter_fourcc(*str(codec)),
        float(fps),
        (int(frame_w), int(frame_h)),
    )
    if not writer.isOpened():
        cap.release()
        raise RuntimeError(f"Could not create output video: {output_path}")

    try:
        frame_index = 0
        while True:
            ret, frame_bgr = cap.read()
            if not ret:
                break
            frame_row = frame_rows_by_index.get(frame_index)
            if frame_row:
                mask_items = decode_frame_row_masks(frame_row, frame_h, frame_w)
                assignment = {
                    int(item["local_id"]): int(obj_id)
                    for item, obj_id in zip(mask_items, frame_row.get("out_obj_ids", []))
                }
                overlaid = draw_overlay_with_labels(
                    frame_bgr,
                    mask_items,
                    assigned_global_ids=assignment,
                    show_local_id=False,
                )
                writer.write(overlaid)
            else:
                writer.write(frame_bgr)
            frame_index += 1
    finally:
        cap.release()
        writer.release()


def process_run_dir(args: argparse.Namespace, run_dir: Path, send_req: Any) -> dict[str, Any]:
    summary_path = run_dir / "summary.json"
    summary = read_json(summary_path)
    frame_outputs_path = Path(
        summary.get("frame_outputs_path") or (run_dir / "frame_outputs_rle.json")
    )
    if not frame_outputs_path.exists():
        fallback_frame_outputs_path = run_dir / "frame_outputs_rle.json"
        if fallback_frame_outputs_path.exists():
            frame_outputs_path = fallback_frame_outputs_path
    if not frame_outputs_path.exists():
        raise FileNotFoundError(
            f"Missing frame_outputs_rle.json for {run_dir}. "
            "Re-run the per-frame SAM3 agent job after pulling the latest branch so it writes this file."
        )

    frame_outputs_payload = read_json(frame_outputs_path)
    video_path = str(summary.get("video_path") or "")
    if not video_path or not os.path.exists(video_path):
        raise FileNotFoundError(
            f"Original video path is missing or unreadable for {run_dir}: {video_path}"
        )

    frame_size = frame_outputs_payload.get("frame_size_hw") or summary.get("frame_size_hw") or [0, 0]
    frame_h = int(frame_size[0] or 0)
    frame_w = int(frame_size[1] or 0)
    if frame_h <= 0 or frame_w <= 0:
        raise RuntimeError(f"Could not determine frame size for run: {run_dir}")

    total_video_frames = int(
        frame_outputs_payload.get("total_video_frames") or summary.get("processed_frames") or 0
    )
    fps = float(summary.get("fps") or 30.0)
    invalid_frame_indices = set(
        int(x) for x in frame_outputs_payload.get("invalid_frame_indices", []) or []
    )
    frame_rows = [row for row in frame_outputs_payload.get("frames", []) if isinstance(row, dict)]
    frame_rows_by_index = {
        int(row.get("frame_index", -1)): row for row in frame_rows if "frame_index" in row
    }
    working_frame_rows_by_index = {
        int(frame_idx): copy.deepcopy(row)
        for frame_idx, row in frame_rows_by_index.items()
    }

    output_dir = run_dir / str(args.output_subdir)
    windows_dir = output_dir / "windows"
    output_dir.mkdir(parents=True, exist_ok=True)
    windows_dir.mkdir(parents=True, exist_ok=True)

    system_prompt = resolve_system_prompt(
        str(args.prompt_profile),
        str(Path(args.prompt_path).resolve()) if args.prompt_path else None,
    )
    missing_mask_system_prompt = resolve_missing_mask_prompt(
        str(args.prompt_profile),
        str(Path(args.missing_mask_prompt_path).resolve())
        if args.missing_mask_prompt_path
        else None,
    )
    verify_gap_fill_system_prompt = resolve_gap_fill_verify_prompt(
        str(args.prompt_profile),
        str(Path(args.verify_gap_fill_prompt_path).resolve())
        if args.verify_gap_fill_prompt_path
        else None,
    )
    missed_creatures_system_prompt = resolve_missed_creatures_prompt(
        str(args.prompt_profile),
        str(Path(args.missed_creatures_prompt_path).resolve())
        if args.missed_creatures_prompt_path
        else None,
    )
    verify_missed_creatures_system_prompt = resolve_missed_creatures_verify_prompt(
        str(args.prompt_profile),
        str(Path(args.verify_missed_creatures_prompt_path).resolve())
        if args.verify_missed_creatures_prompt_path
        else None,
    )
    outlier_mask_system_prompt = resolve_outlier_mask_prompt(
        str(args.prompt_profile),
        str(Path(args.outlier_mask_prompt_path).resolve())
        if args.outlier_mask_prompt_path
        else None,
    )

    stage_names = resolve_enabled_stages(args)
    run_missed_creatures = "missed_creatures" in stage_names
    run_gap_fill = "gap_fill" in stage_names
    run_outlier_filter = "outlier_filter" in stage_names
    run_id_reassign = "id_reassign" in stage_names

    frame_assignments: dict[int, dict[int, int]] = {}
    dropped_local_ids_by_frame: dict[int, set[int]] = {}
    ignored_drop_labels_by_frame: dict[int, set[int]] = {}
    next_global_id = 1
    window_reports: list[dict[str, Any]] = []
    missed_creatures_report: dict[str, Any] | None = None
    raw_response_failures = 0
    gap_fill_detection_failures = 0
    gap_fill_accepted_issue_count = 0
    gap_fill_already_present_issue_count = 0
    gap_fill_unresolved_issue_count = 0
    outlier_candidate_issue_count = 0
    outlier_reviewed_issue_count = 0
    outlier_removed_mask_count = 0
    outlier_kept_mask_count = 0

    backend = None
    session_id = None
    if run_missed_creatures or run_gap_fill:
        backend = PredictorBackend(gpu_ids=parse_gpu_ids(str(args.sam3_gpu_ids)))
        session_id = backend.start_session(
            resource_path=video_path,
            image_size=int(args.sam3_image_size),
            offload_video_to_cpu=bool(args.sam3_offload_video_to_cpu),
        )

    valid_frame_indices = [
        int(frame_idx)
        for frame_idx in sorted(working_frame_rows_by_index.keys())
        if frame_idx not in invalid_frame_indices
    ]

    if run_missed_creatures and backend is not None and session_id is not None:
        missed_creatures_report = run_missed_creatures_stage(
            args=args,
            send_generate_request_fn=send_req,
            detection_system_prompt=missed_creatures_system_prompt,
            verify_system_prompt=verify_missed_creatures_system_prompt,
            backend=backend,
            session_id=session_id,
            video_path=video_path,
            frame_h=frame_h,
            frame_w=frame_w,
            valid_frame_indices=valid_frame_indices,
            invalid_frame_indices=invalid_frame_indices,
            working_frame_rows_by_index=working_frame_rows_by_index,
            output_dir=output_dir / "missed_creatures",
            max_json_retries=int(args.max_json_retries),
        )
        valid_frame_indices = [
            int(frame_idx)
            for frame_idx in sorted(working_frame_rows_by_index.keys())
            if frame_idx not in invalid_frame_indices
        ]

    windows = build_index_windows(
        valid_frame_indices,
        window_size=int(args.window_size),
        stride=int(args.window_stride),
    )
    window_stage_requested = any(stage in {"gap_fill", "outlier_filter", "id_reassign"} for stage in stage_names)
    progress = (
        ProgressReporter(total=len(windows), desc=f"Post-process {run_dir.name}")
        if window_stage_requested
        else None
    )

    try:
        for window_index, window_frame_indices in enumerate(windows if window_stage_requested else []):
            window_dir = windows_dir / f"window_{window_index:04d}"
            window_dir.mkdir(parents=True, exist_ok=True)

            raw_tiles: list[tuple[int, Any]] = []
            overlay_tiles_pre_gap: list[tuple[int, Any]] = []
            local_ids_by_frame: dict[int, list[int]] = {}
            mask_items_by_frame: dict[int, list[dict[str, Any]]] = {}
            frame_bgr_by_index: dict[int, Any] = {}

            for frame_index in window_frame_indices:
                frame_bgr = read_video_frame(video_path, frame_index)
                if frame_bgr is None:
                    continue
                frame_bgr_by_index[int(frame_index)] = frame_bgr
                raw_tiles.append((frame_index, frame_bgr))
                frame_row = working_frame_rows_by_index.get(
                    frame_index, {"frame_index": int(frame_index), "out_obj_ids": [], "out_probs": [], "out_tracker_probs": [], "out_boxes_xywh": [], "out_binary_masks_rle": []}
                )
                working_frame_rows_by_index.setdefault(int(frame_index), frame_row)
                mask_items = decode_frame_row_masks(frame_row, frame_h, frame_w)
                local_ids_by_frame[frame_index] = [int(item["local_id"]) for item in mask_items]
                mask_items_by_frame[frame_index] = mask_items
                overlay_tiles_pre_gap.append(
                    (
                        frame_index,
                        draw_overlay_with_labels(
                            frame_bgr,
                            mask_items,
                            assigned_global_ids=frame_assignments.get(frame_index, {}),
                        ),
                    )
                )

            if not raw_tiles:
                if progress is not None:
                    progress.update(1, postfix={"windows": len(window_reports), "global_ids": next_global_id - 1})
                continue

            raw_collage_path = str(window_dir / "raw_collage.jpg")
            overlay_before_gap_fill_path = str(window_dir / "overlay_before_gap_fill.jpg")
            build_collage(
                raw_tiles,
                raw_collage_path,
                cols=int(args.collage_cols),
                tile_max_edge=int(args.collage_tile_max_edge),
            )
            build_collage(
                overlay_tiles_pre_gap,
                overlay_before_gap_fill_path,
                cols=int(args.collage_cols),
                tile_max_edge=int(args.collage_tile_max_edge),
            )

            gap_fill_report: dict[str, Any] | None = None
            outlier_filter_report: dict[str, Any] | None = None
            if run_gap_fill and backend is not None and session_id is not None:
                gap_fill_report, local_ids_by_frame, mask_items_by_frame = repair_missing_masks_in_window(
                    args=args,
                    send_generate_request_fn=send_req,
                    missing_system_prompt=missing_mask_system_prompt,
                    verify_system_prompt=verify_gap_fill_system_prompt,
                    backend=backend,
                    session_id=session_id,
                    video_path=video_path,
                    frame_h=frame_h,
                    frame_w=frame_w,
                    window_dir=window_dir,
                    window_frame_indices=window_frame_indices,
                    working_frame_rows_by_index=working_frame_rows_by_index,
                    local_ids_by_frame=local_ids_by_frame,
                    mask_items_by_frame=mask_items_by_frame,
                    raw_collage_path=raw_collage_path,
                    overlay_before_gap_fill_path=overlay_before_gap_fill_path,
                    max_json_retries=int(args.max_json_retries),
                )
                if gap_fill_report.get("raw_detection_response") and not gap_fill_report.get("issue_candidates"):
                    detection_parsed = extract_json_object(str(gap_fill_report.get("raw_detection_response") or ""))
                    if detection_parsed is None:
                        gap_fill_detection_failures += 1
                gap_fill_accepted_issue_count += int(gap_fill_report.get("accepted_issue_count", 0))
                gap_fill_already_present_issue_count += int(
                    gap_fill_report.get("already_present_issue_count", 0)
                )
                gap_fill_unresolved_issue_count += int(gap_fill_report.get("unresolved_issue_count", 0))
                if run_outlier_filter:
                    outlier_filter_report, local_ids_by_frame, mask_items_by_frame = filter_outlier_gap_fill_masks(
                        args=args,
                        send_generate_request_fn=send_req,
                        outlier_system_prompt=outlier_mask_system_prompt,
                        video_path=video_path,
                        frame_h=frame_h,
                        frame_w=frame_w,
                        window_dir=window_dir,
                        working_frame_rows_by_index=working_frame_rows_by_index,
                        local_ids_by_frame=local_ids_by_frame,
                        mask_items_by_frame=mask_items_by_frame,
                        gap_fill_report=gap_fill_report,
                        max_json_retries=int(args.max_json_retries),
                    )
                    outlier_candidate_issue_count += int(outlier_filter_report.get("candidate_issue_count", 0))
                    outlier_reviewed_issue_count += int(outlier_filter_report.get("reviewed_issue_count", 0))
                    outlier_removed_mask_count += int(outlier_filter_report.get("removed_mask_count", 0))
                    outlier_kept_mask_count += int(outlier_filter_report.get("kept_mask_count", 0))

            overlay_tiles: list[tuple[int, Any]] = []
            for frame_index in window_frame_indices:
                frame_bgr = frame_bgr_by_index.get(frame_index)
                if frame_bgr is None:
                    continue
                overlay_tiles.append(
                    (
                        frame_index,
                        draw_overlay_with_labels(
                            frame_bgr,
                            mask_items_by_frame.get(frame_index, []),
                            assigned_global_ids=frame_assignments.get(frame_index, {}),
                        ),
                    )
                )

            overlay_collage_path = str(window_dir / "overlay_collage.jpg")
            build_collage(
                overlay_tiles,
                overlay_collage_path,
                cols=int(args.collage_cols),
                tile_max_edge=int(args.collage_tile_max_edge),
            )

            anchor_text = build_anchor_text(
                frame_assignments=frame_assignments,
                window_frame_indices=window_frame_indices,
            )
            inventory_text = build_inventory_text(
                window_frame_indices=window_frame_indices,
                local_ids_by_frame=local_ids_by_frame,
            )
            parsed = None
            raw_text = None
            sanitized_assignments: dict[int, dict[int, str | None]] = {}
            resolved_window_assignments: dict[int, dict[int, int]] = {}
            dropped_by_frame: dict[int, list[int]] = {}
            ignored_drops_by_frame: dict[int, list[int]] = {}
            if run_id_reassign:
                parsed, raw_text = request_window_assignment(
                    send_generate_request_fn=send_req,
                    system_prompt=system_prompt,
                    raw_collage_path=raw_collage_path,
                    overlay_collage_path=overlay_collage_path,
                    window_frame_indices=window_frame_indices,
                    anchor_text=anchor_text,
                    inventory_text=inventory_text,
                    next_global_id=next_global_id,
                    max_json_retries=int(args.max_json_retries),
                    allow_drop_assignments=bool(args.allow_drop_assignments),
                )
                if parsed is None:
                    raw_response_failures += 1

                sanitized_assignments = sanitize_assignment_response(
                    parsed,
                    window_frame_indices=window_frame_indices,
                    local_ids_by_frame=local_ids_by_frame,
                )

                (
                    resolved_window_assignments,
                    dropped_by_frame,
                    ignored_drops_by_frame,
                    next_global_id,
                ) = apply_window_assignments(
                    window_frame_indices=window_frame_indices,
                    mask_items_by_frame=mask_items_by_frame,
                    working_frame_rows_by_index=working_frame_rows_by_index,
                    frame_h=frame_h,
                    frame_w=frame_w,
                    existing_frame_assignments=frame_assignments,
                    parsed_assignments=sanitized_assignments,
                    next_global_id=next_global_id,
                    allow_drop_assignments=bool(args.allow_drop_assignments),
                    assignment_history_frames=int(args.assignment_history_frames),
                    assignment_heuristic_min_score=float(args.assignment_heuristic_min_score),
                )

            for frame_index, mapping in resolved_window_assignments.items():
                frame_assignments.setdefault(int(frame_index), {}).update(
                    {int(k): int(v) for k, v in mapping.items()}
                )
            for frame_index, local_ids in dropped_by_frame.items():
                dropped_local_ids_by_frame.setdefault(int(frame_index), set()).update(
                    int(local_id) for local_id in local_ids
                )
            for frame_index, local_ids in ignored_drops_by_frame.items():
                ignored_drop_labels_by_frame.setdefault(int(frame_index), set()).update(
                    int(local_id) for local_id in local_ids
                )

            window_report = {
                "window_index": int(window_index),
                "frame_indices": [int(x) for x in window_frame_indices],
                "raw_collage_path": raw_collage_path,
                "overlay_before_gap_fill_path": overlay_before_gap_fill_path,
                "overlay_collage_path": overlay_collage_path,
                "anchor_text": anchor_text,
                "inventory_text": inventory_text,
                "gap_fill_report": gap_fill_report,
                "outlier_filter_report": outlier_filter_report,
                "raw_response_text": raw_text,
                "parsed_assignments": sanitized_assignments,
                "resolved_assignments": {
                    str(frame_index): {
                        str(local_id): int(global_id)
                        for local_id, global_id in sorted(mapping.items())
                    }
                    for frame_index, mapping in sorted(resolved_window_assignments.items())
                },
                "dropped_local_ids": {
                    str(frame_index): [int(x) for x in sorted(local_ids)]
                    for frame_index, local_ids in sorted(dropped_by_frame.items())
                },
                "ignored_drop_labels": {
                    str(frame_index): [int(x) for x in sorted(local_ids)]
                    for frame_index, local_ids in sorted(ignored_drops_by_frame.items())
                },
            }
            window_reports.append(window_report)
            if args.debug:
                write_json(window_dir / "window_report.json", window_report)

            if progress is not None:
                progress.update(
                    1,
                    postfix={
                        "windows": len(window_reports),
                        "global_ids": next_global_id - 1,
                        "json_failures": raw_response_failures,
                        "gap_fills": gap_fill_accepted_issue_count,
                    },
                )
    finally:
        if progress is not None:
            progress.close()
        if backend is not None and session_id is not None:
            try:
                backend.close_session(session_id)
            except Exception:
                pass
    consistent_frame_rows: list[dict[str, Any]] = []
    changed_frames = 0
    gap_filled_frame_count = 0
    relabeled_mask_count = 0
    for frame_index in sorted(working_frame_rows_by_index.keys()):
        original_row = frame_rows_by_index.get(
            frame_index,
            {"frame_index": int(frame_index), "out_obj_ids": [], "out_probs": [], "out_tracker_probs": [], "out_boxes_xywh": [], "out_binary_masks_rle": []},
        )
        working_row = working_frame_rows_by_index[frame_index]
        frame_assignment = frame_assignments.get(frame_index, {})
        if not run_id_reassign:
            frame_assignment = {
                int(local_id): int(local_id)
                for local_id in frame_row_local_ids(working_row)
            }
        dropped_local_ids = dropped_local_ids_by_frame.get(frame_index, set())
        relabeled_row = relabel_frame_row(
            working_row,
            frame_h=frame_h,
            frame_w=frame_w,
            frame_assignment=frame_assignment,
            dropped_local_ids=dropped_local_ids,
        )
        consistent_frame_rows.append(relabeled_row)
        original_ids = [int(x) for x in frame_row_local_ids(original_row)]
        working_ids = [int(x) for x in frame_row_local_ids(working_row)]
        new_ids = [int(x) for x in relabeled_row.get("out_obj_ids", [])]
        if len(working_ids) > len(original_ids):
            gap_filled_frame_count += 1
        if original_ids != new_ids or working_ids != original_ids or dropped_local_ids:
            changed_frames += 1
        relabeled_mask_count += len(new_ids)

    gap_fill_failure_reason_counts = Counter()
    gap_fill_attempt_failure_reason_counts = Counter()
    gap_fill_verification_status_counts = Counter()
    windows_with_detected_gap_fill_issues = 0
    outlier_verdict_counts = Counter()
    outlier_failure_reason_counts = Counter()
    windows_with_outlier_reviews = 0
    for window_report in window_reports:
        gap_fill_report = window_report.get("gap_fill_report") or {}
        issues = gap_fill_report.get("issues") or []
        if issues:
            windows_with_detected_gap_fill_issues += 1
        gap_fill_failure_reason_counts.update(gap_fill_report.get("failure_reason_counts") or {})
        gap_fill_attempt_failure_reason_counts.update(
            gap_fill_report.get("attempt_failure_reason_counts") or {}
        )
        gap_fill_verification_status_counts.update(
            gap_fill_report.get("verification_status_counts") or {}
        )
        outlier_filter_report = window_report.get("outlier_filter_report") or {}
        if outlier_filter_report.get("reviewed_issue_count"):
            windows_with_outlier_reviews += 1
        outlier_verdict_counts.update(outlier_filter_report.get("verdict_counts") or {})
        outlier_failure_reason_counts.update(outlier_filter_report.get("failure_reason_counts") or {})

    gap_fill_retained_issue_count = max(0, int(gap_fill_accepted_issue_count) - int(outlier_removed_mask_count))

    consistent_payload = {
        "format_version": int(frame_outputs_payload.get("format_version", 2)),
        "source": "sam3_agent_every_frame_consistent_ids_mllm",
        "frame_size_hw": [int(frame_h), int(frame_w)],
        "total_video_frames": int(total_video_frames),
        "num_frames_with_outputs": len(consistent_frame_rows),
        "invalid_frame_indices": sorted(int(x) for x in invalid_frame_indices),
        "keyframe_indices": list(frame_outputs_payload.get("keyframe_indices", []) or []),
        "frames": consistent_frame_rows,
    }
    consistent_frame_outputs_path = output_dir / "frame_outputs_consistent_ids.json"
    write_json(consistent_frame_outputs_path, consistent_payload)

    overlay_output_path = output_dir / str(args.output_video_name)
    if args.render_video:
        render_overlay_video(
            video_path=video_path,
            output_path=overlay_output_path,
            frame_rows_by_index={
                int(row["frame_index"]): row for row in consistent_frame_rows
            },
            frame_size_hw=(frame_h, frame_w),
            fps=fps,
            codec=str(args.codec),
        )

    frame_assignment_rows = []
    for frame_index in sorted(frame_assignments.keys()):
        frame_assignment_rows.append(
            {
                "frame_index": int(frame_index),
                "assignments": [
                    {"local_id": int(local_id), "global_id": int(global_id)}
                    for local_id, global_id in sorted(frame_assignments[frame_index].items())
                ],
                "dropped_local_ids": [
                    int(x) for x in sorted(dropped_local_ids_by_frame.get(frame_index, set()))
                ],
                "ignored_drop_labels": [
                    int(x) for x in sorted(ignored_drop_labels_by_frame.get(frame_index, set()))
                ],
            }
        )

    report = {
        "input_run_dir": str(run_dir),
        "video_path": video_path,
        "summary_path": str(summary_path),
        "frame_outputs_path": str(frame_outputs_path),
        "output_dir": str(output_dir),
        "consistent_frame_outputs_path": str(consistent_frame_outputs_path),
        "overlay_output_path": str(overlay_output_path) if args.render_video else "",
        "prompt_profile": str(args.prompt_profile),
        "stages": list(stage_names),
        "find_missed_creatures": bool(run_missed_creatures),
        "fill_missing_masks": bool(run_gap_fill),
        "filter_outlier_masks": bool(run_outlier_filter),
        "run_id_reassign": bool(run_id_reassign),
        "allow_drop_assignments": bool(args.allow_drop_assignments),
        "window_size": int(args.window_size),
        "window_stride": int(args.window_stride),
        "window_count": len(windows),
        "missed_creatures_report_path": (
            str(output_dir / "missed_creatures" / "missed_creatures_report.json")
            if missed_creatures_report is not None
            else ""
        ),
        "missed_creatures_accepted_issue_count": int(
            (missed_creatures_report or {}).get("accepted_issue_count", 0)
        ),
        "missed_creatures_unresolved_issue_count": int(
            (missed_creatures_report or {}).get("unresolved_issue_count", 0)
        ),
        "missed_creatures_failure_reason_counts": dict(
            (missed_creatures_report or {}).get("failure_reason_counts") or {}
        ),
        "raw_response_failure_count": int(raw_response_failures),
        "gap_fill_detection_failure_count": int(gap_fill_detection_failures),
        "gap_fill_accepted_issue_count": int(gap_fill_accepted_issue_count),
        "gap_fill_already_present_issue_count": int(gap_fill_already_present_issue_count),
        "gap_fill_retained_issue_count": int(gap_fill_retained_issue_count),
        "gap_fill_unresolved_issue_count": int(gap_fill_unresolved_issue_count),
        "gap_filled_frame_count": int(gap_filled_frame_count),
        "windows_with_detected_gap_fill_issues": int(windows_with_detected_gap_fill_issues),
        "gap_fill_failure_reason_counts": dict(gap_fill_failure_reason_counts),
        "gap_fill_attempt_failure_reason_counts": dict(gap_fill_attempt_failure_reason_counts),
        "gap_fill_verification_status_counts": dict(gap_fill_verification_status_counts),
        "outlier_candidate_issue_count": int(outlier_candidate_issue_count),
        "outlier_reviewed_issue_count": int(outlier_reviewed_issue_count),
        "outlier_removed_mask_count": int(outlier_removed_mask_count),
        "outlier_kept_mask_count": int(outlier_kept_mask_count),
        "windows_with_outlier_reviews": int(windows_with_outlier_reviews),
        "outlier_verdict_counts": dict(outlier_verdict_counts),
        "outlier_failure_reason_counts": dict(outlier_failure_reason_counts),
        "ignored_drop_label_count": int(sum(len(v) for v in ignored_drop_labels_by_frame.values())),
        "total_video_frames": int(total_video_frames),
        "valid_window_frames": len(valid_frame_indices),
        "changed_frame_count": int(changed_frames),
        "relabeled_mask_count": int(relabeled_mask_count),
        "num_global_ids": int(max(0, next_global_id - 1)) if run_id_reassign else 0,
        "window_reports_path": str(output_dir / "id_reassignment_report.json"),
        "frame_assignments": frame_assignment_rows,
        "missed_creatures_report": missed_creatures_report,
        "windows": window_reports,
        "finished_at_local": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    write_json(output_dir / "id_reassignment_report.json", report)

    summary_out = {
        "input_run_dir": str(run_dir),
        "input_video_path": video_path,
        "frame_outputs_path": str(frame_outputs_path),
        "consistent_frame_outputs_path": str(consistent_frame_outputs_path),
        "overlay_output_path": str(overlay_output_path) if args.render_video else "",
        "window_count": int(len(windows)),
        "valid_frame_count": int(len(valid_frame_indices)),
        "stages": list(stage_names),
        "find_missed_creatures": bool(run_missed_creatures),
        "fill_missing_masks": bool(run_gap_fill),
        "filter_outlier_masks": bool(run_outlier_filter),
        "run_id_reassign": bool(run_id_reassign),
        "allow_drop_assignments": bool(args.allow_drop_assignments),
        "missed_creatures_accepted_issue_count": int(
            (missed_creatures_report or {}).get("accepted_issue_count", 0)
        ),
        "missed_creatures_unresolved_issue_count": int(
            (missed_creatures_report or {}).get("unresolved_issue_count", 0)
        ),
        "missed_creatures_failure_reason_counts": dict(
            (missed_creatures_report or {}).get("failure_reason_counts") or {}
        ),
        "gap_fill_detection_failure_count": int(gap_fill_detection_failures),
        "gap_fill_accepted_issue_count": int(gap_fill_accepted_issue_count),
        "gap_fill_already_present_issue_count": int(gap_fill_already_present_issue_count),
        "gap_fill_retained_issue_count": int(gap_fill_retained_issue_count),
        "gap_fill_unresolved_issue_count": int(gap_fill_unresolved_issue_count),
        "gap_filled_frame_count": int(gap_filled_frame_count),
        "windows_with_detected_gap_fill_issues": int(windows_with_detected_gap_fill_issues),
        "gap_fill_failure_reason_counts": dict(gap_fill_failure_reason_counts),
        "gap_fill_attempt_failure_reason_counts": dict(gap_fill_attempt_failure_reason_counts),
        "gap_fill_verification_status_counts": dict(gap_fill_verification_status_counts),
        "outlier_candidate_issue_count": int(outlier_candidate_issue_count),
        "outlier_reviewed_issue_count": int(outlier_reviewed_issue_count),
        "outlier_removed_mask_count": int(outlier_removed_mask_count),
        "outlier_kept_mask_count": int(outlier_kept_mask_count),
        "windows_with_outlier_reviews": int(windows_with_outlier_reviews),
        "outlier_verdict_counts": dict(outlier_verdict_counts),
        "outlier_failure_reason_counts": dict(outlier_failure_reason_counts),
        "ignored_drop_label_count": int(sum(len(v) for v in ignored_drop_labels_by_frame.values())),
        "changed_frame_count": int(changed_frames),
        "relabeled_mask_count": int(relabeled_mask_count),
        "num_global_ids": int(max(0, next_global_id - 1)) if run_id_reassign else 0,
        "raw_response_failure_count": int(raw_response_failures),
        "finished_at_local": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    write_json(output_dir / "summary.json", summary_out)
    return summary_out


def main() -> int:
    args = parse_args()
    ensure_runtime_deps()

    stage_names = resolve_enabled_stages(args)
    os.environ["SAM3_IMAGE_DETAIL"] = str(args.image_detail)
    min_images = 1
    if "gap_fill" in stage_names:
        min_images = max(min_images, 3)
    if "missed_creatures" in stage_names:
        min_images = max(min_images, int(args.missed_creatures_max_images_per_request))
    os.environ["SAM3_MAX_IMAGES_PER_REQUEST"] = str(
        max(min_images, int(args.max_images_per_request))
    )
    os.environ["SAM3_AGENT_IMAGE_MAX_EDGE"] = str(max(128, int(args.image_max_edge)))
    os.environ["SAM3_AGENT_IMAGE_MIN_EDGE"] = str(max(128, int(args.image_min_edge)))

    api_key = (
        args.api_key
        or os.environ.get("OPENAI_API_KEY")
        or os.environ.get("VLLM_API_KEY")
        or "DUMMY_API_KEY"
    )
    send_req = partial(
        send_generate_request_orig,
        server_url=args.server_url,
        model=args.model,
        api_key=api_key,
        max_tokens=int(args.max_completion_tokens),
    )

    summaries: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []

    for raw_input in args.input_dirs:
        try:
            run_dir = resolve_run_dir(raw_input)
            log(f"Running post-process stages {stage_names} for: {run_dir}")
            summary = process_run_dir(args, run_dir, send_req)
            summaries.append(summary)
            log(
                f"Finished {run_dir.name}: "
                f"{summary['num_global_ids']} global ids, "
                f"{summary['changed_frame_count']} changed frames, "
                f"{summary.get('gap_fill_retained_issue_count', summary.get('gap_fill_accepted_issue_count', 0))} gap fills retained, "
                f"{summary.get('missed_creatures_accepted_issue_count', 0)} missed-creature seeds accepted."
            )
        except Exception as exc:
            failure = {
                "input": raw_input,
                "error_type": type(exc).__name__,
                "error": str(exc),
            }
            failures.append(failure)
            log(f"Failed for {raw_input}: {failure['error_type']}: {failure['error']}")
            if not args.continue_on_error:
                raise

    if failures and not summaries:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
