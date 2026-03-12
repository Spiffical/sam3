#!/usr/bin/env python3
from __future__ import annotations

import argparse
import importlib
import json
import math
import os
import re
import sys
import time
from functools import partial
from pathlib import Path
from typing import Any

try:
    from tqdm import tqdm
except ImportError:
    tqdm = None


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_PROMPT_PROFILE = "underwater"
DEFAULT_CODEC = "mp4v"

cv2 = None
np = None
send_generate_request_orig = None
read_video_frame = None
decode_rle_to_mask = None


def ensure_runtime_deps() -> None:
    global cv2, np, send_generate_request_orig, read_video_frame, decode_rle_to_mask
    if cv2 is not None and np is not None and send_generate_request_orig is not None:
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
        repo_root_str = str(REPO_ROOT)
        nibi_root_str = str(REPO_ROOT / "nibi_model_compare")
        if repo_root_str not in sys.path:
            sys.path.insert(0, repo_root_str)
        if nibi_root_str not in sys.path:
            sys.path.insert(0, nibi_root_str)
        from sam3.agent.client_llm import (
            send_generate_request as _send_generate_request_orig,
        )
        from frame_output_utils import (
            decode_rle_to_mask as _decode_rle_to_mask,
            read_video_frame as _read_video_frame,
        )
    except ImportError as exc:
        missing.append(str(exc))
    else:
        send_generate_request_orig = _send_generate_request_orig
        decode_rle_to_mask = _decode_rle_to_mask
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
        "--max-json-retries",
        type=int,
        default=2,
        help="Maximum JSON repair retries per window. Default: 2",
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


def heuristic_match_global_id(
    *,
    current_item: dict[str, Any],
    prior_items: list[dict[str, Any]],
    prior_assignments: dict[int, int],
    disallowed_global_ids: set[int],
) -> int | None:
    best_global_id: int | None = None
    best_score = 0.0
    area = max(1.0, float(current_item.get("area") or 0.0))
    centroid = current_item.get("centroid")

    for prior_item in prior_items:
        local_id = int(prior_item["local_id"])
        global_id = prior_assignments.get(local_id)
        if global_id is None or global_id in disallowed_global_ids:
            continue
        iou = binary_mask_iou(current_item["mask"], prior_item["mask"])
        area_prior = max(1.0, float(prior_item.get("area") or 0.0))
        area_ratio = min(area, area_prior) / max(area, area_prior)
        dist_score = 0.0
        if centroid is not None and prior_item.get("centroid") is not None:
            dx = float(centroid[0] - prior_item["centroid"][0])
            dy = float(centroid[1] - prior_item["centroid"][1])
            dist = math.sqrt(dx * dx + dy * dy)
            dist_score = max(0.0, 1.0 - dist / 150.0)
        score = (2.0 * iou) + (0.5 * area_ratio) + (0.5 * dist_score)
        if score > best_score:
            best_score = score
            best_global_id = int(global_id)

    if best_score >= 0.85:
        return best_global_id
    return None


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
                        f"For newly appearing creatures, start with temporary labels like new_a, new_b. "
                        f"The next available permanent global id after this window starts at g{int(next_global_id)}. "
                        "Return strict JSON only."
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


def apply_window_assignments(
    *,
    window_frame_indices: list[int],
    mask_items_by_frame: dict[int, list[dict[str, Any]]],
    existing_frame_assignments: dict[int, dict[int, int]],
    parsed_assignments: dict[int, dict[int, str | None]],
    next_global_id: int,
) -> tuple[dict[int, dict[int, int]], dict[int, list[int]], int]:
    resolved: dict[int, dict[int, int]] = {
        int(frame_idx): {int(k): int(v) for k, v in mapping.items()}
        for frame_idx, mapping in existing_frame_assignments.items()
        if frame_idx in window_frame_indices
    }
    dropped_by_frame: dict[int, list[int]] = {}
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
        prior_frames = [
            window_frame_indices[idx]
            for idx in range(max(0, pos - 2), pos)
            if window_frame_indices[idx] in resolved
        ]
        prior_pairs = [
            (
                mask_items_by_frame.get(prior_frame, []),
                resolved.get(prior_frame, {}),
            )
            for prior_frame in prior_frames
        ]

        for item in mask_items:
            local_id = int(item["local_id"])
            if local_id in frame_fixed:
                continue

            model_label = frame_model_map.get(local_id)
            normalized = normalize_track_label(model_label)
            chosen_global_id: int | None = None

            if normalized is None:
                chosen_global_id = None
            elif normalized.lower() == "drop":
                dropped_by_frame.setdefault(int(frame_index), []).append(local_id)
                continue
            elif re.fullmatch(r"g\d+", normalized.lower()):
                requested_gid = int(normalized[1:])
                if requested_gid in anchor_global_ids and requested_gid not in assigned_global_ids:
                    chosen_global_id = requested_gid
                else:
                    temp_key = f"model_{normalized.lower()}"
                    chosen_global_id = temp_label_to_gid.get(temp_key)
                    if chosen_global_id is None:
                        chosen_global_id = int(next_global_id)
                        next_global_id += 1
                        temp_label_to_gid[temp_key] = chosen_global_id
            else:
                temp_key = normalized.lower()
                chosen_global_id = temp_label_to_gid.get(temp_key)
                if chosen_global_id is None:
                    chosen_global_id = int(next_global_id)
                    next_global_id += 1
                    temp_label_to_gid[temp_key] = chosen_global_id

            if chosen_global_id in assigned_global_ids:
                chosen_global_id = None

            if chosen_global_id is None:
                heuristic_gid = None
                for prior_items, prior_mapping in reversed(prior_pairs):
                    heuristic_gid = heuristic_match_global_id(
                        current_item=item,
                        prior_items=prior_items,
                        prior_assignments=prior_mapping,
                        disallowed_global_ids=assigned_global_ids,
                    )
                    if heuristic_gid is not None:
                        break
                if heuristic_gid is not None:
                    chosen_global_id = heuristic_gid

            if chosen_global_id is None:
                chosen_global_id = int(next_global_id)
                next_global_id += 1

            frame_fixed[local_id] = int(chosen_global_id)
            assigned_global_ids.add(int(chosen_global_id))

    return resolved, dropped_by_frame, next_global_id


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
    frame_rows = [
        row
        for row in frame_outputs_payload.get("frames", [])
        if isinstance(row, dict)
    ]
    frame_rows_by_index = {
        int(row.get("frame_index", -1)): row for row in frame_rows if "frame_index" in row
    }

    valid_frame_indices = [
        int(frame_idx)
        for frame_idx in sorted(frame_rows_by_index.keys())
        if frame_idx not in invalid_frame_indices
    ]
    valid_frame_indices = [
        frame_idx
        for frame_idx in valid_frame_indices
        if len(frame_rows_by_index[frame_idx].get("out_binary_masks_rle") or []) > 0
    ]

    output_dir = run_dir / str(args.output_subdir)
    windows_dir = output_dir / "windows"
    output_dir.mkdir(parents=True, exist_ok=True)
    windows_dir.mkdir(parents=True, exist_ok=True)

    system_prompt = resolve_system_prompt(
        str(args.prompt_profile),
        str(Path(args.prompt_path).resolve()) if args.prompt_path else None,
    )

    windows = build_index_windows(
        valid_frame_indices,
        window_size=int(args.window_size),
        stride=int(args.window_stride),
    )
    progress = ProgressReporter(total=len(windows), desc=f"ID reassignment {run_dir.name}")

    frame_assignments: dict[int, dict[int, int]] = {}
    dropped_local_ids_by_frame: dict[int, set[int]] = {}
    next_global_id = 1
    window_reports: list[dict[str, Any]] = []
    raw_response_failures = 0

    for window_index, window_frame_indices in enumerate(windows):
        window_dir = windows_dir / f"window_{window_index:04d}"
        window_dir.mkdir(parents=True, exist_ok=True)

        raw_tiles: list[tuple[int, Any]] = []
        overlay_tiles: list[tuple[int, Any]] = []
        local_ids_by_frame: dict[int, list[int]] = {}
        mask_items_by_frame: dict[int, list[dict[str, Any]]] = {}

        for frame_index in window_frame_indices:
            frame_bgr = read_video_frame(video_path, frame_index)
            if frame_bgr is None:
                continue
            raw_tiles.append((frame_index, frame_bgr))
            frame_row = frame_rows_by_index.get(frame_index, {})
            mask_items = decode_frame_row_masks(frame_row, frame_h, frame_w)
            local_ids_by_frame[frame_index] = [int(item["local_id"]) for item in mask_items]
            mask_items_by_frame[frame_index] = mask_items
            overlay_tiles.append(
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
            progress.update(1, postfix={"windows": len(window_reports), "global_ids": next_global_id - 1})
            continue

        raw_collage_path = str(window_dir / "raw_collage.jpg")
        overlay_collage_path = str(window_dir / "overlay_collage.jpg")
        build_collage(
            raw_tiles,
            raw_collage_path,
            cols=int(args.collage_cols),
            tile_max_edge=int(args.collage_tile_max_edge),
        )
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
        )
        if parsed is None:
            raw_response_failures += 1

        sanitized_assignments = sanitize_assignment_response(
            parsed,
            window_frame_indices=window_frame_indices,
            local_ids_by_frame=local_ids_by_frame,
        )

        resolved_window_assignments, dropped_by_frame, next_global_id = apply_window_assignments(
            window_frame_indices=window_frame_indices,
            mask_items_by_frame=mask_items_by_frame,
            existing_frame_assignments=frame_assignments,
            parsed_assignments=sanitized_assignments,
            next_global_id=next_global_id,
        )

        for frame_index, mapping in resolved_window_assignments.items():
            frame_assignments.setdefault(int(frame_index), {}).update(
                {int(k): int(v) for k, v in mapping.items()}
            )
        for frame_index, local_ids in dropped_by_frame.items():
            dropped_local_ids_by_frame.setdefault(int(frame_index), set()).update(
                int(local_id) for local_id in local_ids
            )

        window_report = {
            "window_index": int(window_index),
            "frame_indices": [int(x) for x in window_frame_indices],
            "raw_collage_path": raw_collage_path,
            "overlay_collage_path": overlay_collage_path,
            "anchor_text": anchor_text,
            "inventory_text": inventory_text,
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
        }
        window_reports.append(window_report)
        if args.debug:
            write_json(window_dir / "window_report.json", window_report)

        progress.update(
            1,
            postfix={
                "windows": len(window_reports),
                "global_ids": next_global_id - 1,
                "json_failures": raw_response_failures,
            },
        )

    progress.close()

    consistent_frame_rows: list[dict[str, Any]] = []
    changed_frames = 0
    relabeled_mask_count = 0
    for frame_index in sorted(frame_rows_by_index.keys()):
        original_row = frame_rows_by_index[frame_index]
        frame_assignment = frame_assignments.get(frame_index, {})
        dropped_local_ids = dropped_local_ids_by_frame.get(frame_index, set())
        relabeled_row = relabel_frame_row(
            original_row,
            frame_h=frame_h,
            frame_w=frame_w,
            frame_assignment=frame_assignment,
            dropped_local_ids=dropped_local_ids,
        )
        consistent_frame_rows.append(relabeled_row)
        original_ids = [int(x) for x in frame_row_local_ids(original_row)]
        new_ids = [int(x) for x in relabeled_row.get("out_obj_ids", [])]
        if original_ids != new_ids or dropped_local_ids:
            changed_frames += 1
        relabeled_mask_count += len(new_ids)

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
        "window_size": int(args.window_size),
        "window_stride": int(args.window_stride),
        "window_count": len(windows),
        "raw_response_failure_count": int(raw_response_failures),
        "total_video_frames": int(total_video_frames),
        "valid_window_frames": len(valid_frame_indices),
        "changed_frame_count": int(changed_frames),
        "relabeled_mask_count": int(relabeled_mask_count),
        "num_global_ids": int(max(0, next_global_id - 1)),
        "window_reports_path": str(output_dir / "id_reassignment_report.json"),
        "frame_assignments": frame_assignment_rows,
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
        "changed_frame_count": int(changed_frames),
        "relabeled_mask_count": int(relabeled_mask_count),
        "num_global_ids": int(max(0, next_global_id - 1)),
        "raw_response_failure_count": int(raw_response_failures),
        "finished_at_local": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    write_json(output_dir / "summary.json", summary_out)
    return summary_out


def main() -> int:
    args = parse_args()
    ensure_runtime_deps()

    os.environ["SAM3_IMAGE_DETAIL"] = str(args.image_detail)
    os.environ["SAM3_MAX_IMAGES_PER_REQUEST"] = str(max(1, int(args.max_images_per_request)))
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
            log(f"Reassigning IDs for run: {run_dir}")
            summary = process_run_dir(args, run_dir, send_req)
            summaries.append(summary)
            log(
                f"Finished {run_dir.name}: "
                f"{summary['num_global_ids']} global ids, "
                f"{summary['changed_frame_count']} changed frames."
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
