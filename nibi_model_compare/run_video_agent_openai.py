#!/usr/bin/env python3
from __future__ import annotations
"""
Run SAM3 video agent workflow with any OpenAI-compatible VLM backend.

This script mirrors the Gemini-based workflow in sam3/apps/gemini_video_agent.py
but swaps model calls to sam3.agent.client_llm.send_generate_request.
"""

import argparse
import json
import os
import sys
import time
import traceback
from importlib import resources as importlib_resources
from functools import partial
from typing import Any

try:
    from dotenv import load_dotenv
except ImportError:
    load_dotenv = None

try:
    import cv2
except ImportError:
    cv2 = None

try:
    import numpy as np
except ImportError:
    np = None

try:
    import torch
except ImportError:
    torch = None

try:
    from PIL import Image
except ImportError:
    Image = None

# Ensure repo root importability when running from copied folder.
SCRIPT_DIR = os.path.abspath(os.path.dirname(__file__))
REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if REPO_ROOT not in sys.path:
    sys.path.append(REPO_ROOT)
if SCRIPT_DIR not in sys.path:
    sys.path.append(SCRIPT_DIR)
if load_dotenv is not None:
    load_dotenv(os.path.join(REPO_ROOT, ".env"))


def configure_job_cache_defaults() -> None:
    """Default Hugging Face/PyTorch caches to job-local storage on Slurm."""
    if os.environ.get("SAM3_DISABLE_AUTO_CACHE_SETUP") == "1":
        return

    slurm_tmpdir = os.environ.get("SLURM_TMPDIR")
    if not slurm_tmpdir:
        return

    # Respect explicit user configuration from environment or .env.
    if any(
        os.environ.get(key)
        for key in (
            "HF_HOME",
            "HF_HUB_CACHE",
            "HUGGINGFACE_HUB_CACHE",
            "TRANSFORMERS_CACHE",
            "TORCH_HOME",
            "XDG_CACHE_HOME",
        )
    ):
        return

    cache_root = os.environ.get(
        "SAM3_JOB_CACHE_ROOT", os.path.join(slurm_tmpdir, "hf-cache")
    )
    hf_home = os.path.join(cache_root, "hf")
    hf_hub_cache = os.path.join(hf_home, "hub")
    hf_xet_cache = os.path.join(hf_home, "xet")
    torch_home = os.path.join(hf_home, "torch")
    xdg_cache_home = os.path.join(hf_home, "xdg")
    tmpdir = os.path.join(slurm_tmpdir, "tmp")

    os.environ.setdefault("HF_HOME", hf_home)
    os.environ.setdefault("HF_HUB_CACHE", hf_hub_cache)
    os.environ.setdefault("HUGGINGFACE_HUB_CACHE", hf_hub_cache)
    os.environ.setdefault("HF_XET_CACHE", hf_xet_cache)
    os.environ.setdefault("TORCH_HOME", torch_home)
    os.environ.setdefault("XDG_CACHE_HOME", xdg_cache_home)
    os.environ.setdefault("TMPDIR", tmpdir)
    os.environ.setdefault("HF_HUB_DISABLE_XET", "1")
    os.environ.setdefault("TRANSFORMERS_CACHE", hf_hub_cache)

    for path in (
        os.environ["HF_HOME"],
        os.environ["HF_HUB_CACHE"],
        os.environ["HF_XET_CACHE"],
        os.environ["TORCH_HOME"],
        os.environ["XDG_CACHE_HOME"],
        os.environ["TMPDIR"],
    ):
        os.makedirs(path, exist_ok=True)


configure_job_cache_defaults()


def find_bpe_path() -> str:
    env_path = os.environ.get("SAM3_BPE_PATH")
    if env_path and os.path.exists(env_path):
        return env_path

    candidates = [
        os.path.join(REPO_ROOT, "assets/bpe_simple_vocab_16e6.txt.gz"),
        os.path.join(REPO_ROOT, "sam3/assets/bpe_simple_vocab_16e6.txt.gz"),
        "assets/bpe_simple_vocab_16e6.txt.gz",
        "sam3/assets/bpe_simple_vocab_16e6.txt.gz",
    ]
    for path in candidates:
        if os.path.exists(path):
            return path

    # Fallback to packaged resource path when installed/editable.
    try:
        resource_path = str(
            importlib_resources.files("sam3").joinpath(
                "assets/bpe_simple_vocab_16e6.txt.gz"
            )
        )
        if os.path.exists(resource_path):
            return resource_path
    except Exception:
        pass

    raise FileNotFoundError(
        f"Could not find bpe_simple_vocab_16e6.txt.gz in: {candidates}"
    )


def decode_rle_to_mask(rle: Any, height: int, width: int) -> np.ndarray:
    from pycocotools import mask as mask_util

    if isinstance(rle, str):
        rle = {"counts": rle.encode("utf-8"), "size": [height, width]}
    elif isinstance(rle, dict) and "counts" in rle and isinstance(rle["counts"], str):
        rle = dict(rle)
        rle["counts"] = rle["counts"].encode("utf-8")

    decoded = mask_util.decode([rle])
    if decoded.ndim == 3:
        return decoded[:, :, 0]
    return decoded


def get_center_point(mask: np.ndarray) -> tuple[int, int] | None:
    ys, xs = np.where(mask > 0)
    if len(ys) == 0:
        return None
    idx = len(ys) // 2
    return int(xs[idx]), int(ys[idx])


def binary_mask_iou(mask_a: np.ndarray, mask_b: np.ndarray) -> float:
    a = np.asarray(mask_a).astype(bool)
    b = np.asarray(mask_b).astype(bool)
    inter = float(np.logical_and(a, b).sum())
    if inter <= 0.0:
        return 0.0
    union = float(np.logical_or(a, b).sum())
    if union <= 0.0:
        return 0.0
    return inter / union


def deduplicate_masks_by_iou(
    masks: list[np.ndarray], iou_threshold: float
) -> tuple[list[np.ndarray], list[int], list[dict[str, Any]]]:
    """
    Remove near-duplicate masks within one keyframe before ID assignment.
    Keeps first occurrence order and drops masks with IoU >= threshold to a kept mask.
    Returns: (kept_masks, kept_original_indices, dropped_rows)
    """
    if len(masks) <= 1:
        return masks, list(range(len(masks))), []

    kept_masks: list[np.ndarray] = []
    kept_indices: list[int] = []
    dropped_rows: list[dict[str, Any]] = []

    for new_idx, candidate_mask in enumerate(masks):
        matched_kept_idx: int | None = None
        best_iou = 0.0
        is_duplicate = False
        for kept_pos, kept_mask in enumerate(kept_masks):
            iou = binary_mask_iou(candidate_mask, kept_mask)
            if iou > best_iou:
                best_iou = float(iou)
                matched_kept_idx = kept_indices[kept_pos]
            if iou >= iou_threshold:
                is_duplicate = True
                break
        if is_duplicate:
            dropped_rows.append(
                {
                    "dropped_mask_index": int(new_idx),
                    "matched_kept_mask_index": (
                        int(matched_kept_idx) if matched_kept_idx is not None else None
                    ),
                    "best_iou": float(best_iou),
                }
            )
            continue
        kept_masks.append(candidate_mask)
        kept_indices.append(new_idx)

    return kept_masks, kept_indices, dropped_rows


def read_video_frame(video_path: str, frame_idx: int) -> np.ndarray | None:
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return None
    cap.set(cv2.CAP_PROP_POS_FRAMES, int(frame_idx))
    ret, frame = cap.read()
    cap.release()
    if not ret:
        return None
    return frame


def mask_list_from_outputs(outputs: dict[str, Any]) -> list[np.ndarray]:
    if not isinstance(outputs, dict):
        return []
    masks = outputs.get("pred_masks")
    if masks is None:
        masks = outputs.get("video_res_masks")
    if masks is None:
        # Video propagation path returns post-processed masks here.
        masks = outputs.get("out_binary_masks")
    if masks is None and isinstance(outputs.get("obj_id_to_mask"), dict):
        # Some code paths may expose raw object-id keyed masks.
        masks = list(outputs["obj_id_to_mask"].values())
    if masks is None:
        return []
    if isinstance(masks, dict):
        masks = list(masks.values())
    if isinstance(masks, torch.Tensor):
        masks = masks.detach().cpu().numpy()
    parsed: list[np.ndarray] = []
    for mask in masks:
        arr = np.asarray(mask)
        if arr.ndim == 4:
            arr = arr[0]
        if arr.ndim == 3:
            arr = arr[0]
        parsed.append(arr > 0)
    return parsed


def object_color(obj_id: int) -> tuple[int, int, int]:
    # Deterministic BGR color per object id.
    return (
        int((obj_id * 47) % 255),
        int((obj_id * 89 + 37) % 255),
        int((obj_id * 131 + 73) % 255),
    )


def iter_output_masks_with_ids(
    outputs: dict[str, Any], frame_h: int, frame_w: int
) -> list[tuple[int, np.ndarray]]:
    if not isinstance(outputs, dict):
        return []

    if "out_binary_masks" in outputs:
        raw_masks = outputs.get("out_binary_masks")
        raw_ids = outputs.get("out_obj_ids")
        if isinstance(raw_masks, torch.Tensor):
            raw_masks = raw_masks.detach().cpu().numpy()
        if isinstance(raw_ids, torch.Tensor):
            raw_ids = raw_ids.detach().cpu().numpy()
        if raw_masks is None:
            return []

        out: list[tuple[int, np.ndarray]] = []
        for i, raw_mask in enumerate(raw_masks):
            arr = np.asarray(raw_mask)
            while arr.ndim > 2:
                arr = arr[0]
            if arr.shape != (frame_h, frame_w):
                arr = cv2.resize(
                    arr.astype(np.float32),
                    (frame_w, frame_h),
                    interpolation=cv2.INTER_NEAREST,
                )
            mask = arr > 0.5
            if raw_ids is not None and len(raw_ids) > i:
                obj_id = int(raw_ids[i])
            else:
                obj_id = i + 1
            out.append((obj_id, mask))
        return out

    # Fallback for legacy outputs without object ids.
    return [(i + 1, m) for i, m in enumerate(mask_list_from_outputs(outputs))]


def _coerce_to_list(value: Any) -> list[Any]:
    if value is None:
        return []
    if isinstance(value, torch.Tensor):
        value = value.detach().cpu().numpy()
    if isinstance(value, np.ndarray):
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


def _frame_object_metadata(
    outputs: dict[str, Any], frame_h: int, frame_w: int
) -> dict[int, dict[str, Any]]:
    obj_ids = _coerce_to_list(outputs.get("out_obj_ids"))
    out_boxes_xywh = _coerce_to_list(outputs.get("out_boxes_xywh"))
    out_probs = _coerce_to_list(outputs.get("out_probs"))

    metadata: dict[int, dict[str, Any]] = {}
    for i, raw_obj_id in enumerate(obj_ids):
        try:
            obj_id = int(raw_obj_id)
        except Exception:
            continue

        box_xyxy: tuple[int, int, int, int] | None = None
        if i < len(out_boxes_xywh):
            raw_box = out_boxes_xywh[i]
            if isinstance(raw_box, (list, tuple)) and len(raw_box) >= 4:
                try:
                    x, y, w, h = [float(raw_box[j]) for j in range(4)]
                    x1 = int(round(x))
                    y1 = int(round(y))
                    x2 = int(round(x + max(0.0, w)))
                    y2 = int(round(y + max(0.0, h)))
                    x1 = max(0, min(frame_w - 1, x1))
                    y1 = max(0, min(frame_h - 1, y1))
                    x2 = max(0, min(frame_w - 1, x2))
                    y2 = max(0, min(frame_h - 1, y2))
                    if x2 > x1 and y2 > y1:
                        box_xyxy = (x1, y1, x2, y2)
                except Exception:
                    box_xyxy = None

        conf: float | None = None
        if i < len(out_probs):
            raw_prob = out_probs[i]
            if isinstance(raw_prob, (list, tuple)) and len(raw_prob) > 0:
                raw_prob = raw_prob[0]
            try:
                conf = float(raw_prob)
            except Exception:
                conf = None

        metadata[obj_id] = {"box_xyxy": box_xyxy, "confidence": conf}

    return metadata


def overlay_masks_on_frame(video_frame: np.ndarray, outputs: dict[str, Any]) -> np.ndarray:
    frame_h, frame_w = video_frame.shape[:2]
    masks_with_ids = iter_output_masks_with_ids(outputs, frame_h, frame_w)
    if not masks_with_ids:
        return video_frame
    metadata_by_obj = _frame_object_metadata(outputs, frame_h, frame_w)

    max_area_ratio = float(os.environ.get("SAM3_OVERLAY_MAX_MASK_AREA_RATIO", "0.95"))
    alpha = float(os.environ.get("SAM3_OVERLAY_ALPHA", "0.35"))

    overlay = np.zeros_like(video_frame)
    drawn_any = False

    for obj_id, mask in masks_with_ids:
        if mask.shape != (frame_h, frame_w):
            continue
        area_ratio = float(mask.mean())
        # Guard against runaway masks that can wash out the whole frame.
        if area_ratio <= 0.0 or area_ratio > max_area_ratio:
            continue

        color = object_color(obj_id)
        overlay[mask] = color

        # Draw crisp contours for readability.
        mask_u8 = (mask.astype(np.uint8) * 255)
        contours, _ = cv2.findContours(mask_u8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(video_frame, contours, -1, color, 2)

        meta = metadata_by_obj.get(int(obj_id), {})
        box_xyxy = meta.get("box_xyxy")
        confidence = meta.get("confidence")

        if box_xyxy is None:
            ys, xs = np.where(mask)
            if len(xs) > 0 and len(ys) > 0:
                x1 = int(xs.min())
                y1 = int(ys.min())
                x2 = int(xs.max())
                y2 = int(ys.max())
                if x2 > x1 and y2 > y1:
                    box_xyxy = (x1, y1, x2, y2)

        if box_xyxy is not None:
            x1, y1, x2, y2 = box_xyxy
            cv2.rectangle(video_frame, (x1, y1), (x2, y2), color, 2)

            label = f"id {int(obj_id)}"
            if isinstance(confidence, (float, int)):
                label += f" p {float(confidence):.1f}"

            (text_w, text_h), baseline = cv2.getTextSize(
                label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1
            )
            text_x = x1
            text_y = max(text_h + 2, y1 - 4)
            bg_tl = (text_x, text_y - text_h - baseline - 2)
            bg_br = (text_x + text_w + 4, text_y + 2)
            cv2.rectangle(video_frame, bg_tl, bg_br, color, -1)
            cv2.putText(
                video_frame,
                label,
                (text_x + 2, text_y - 2),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 0, 0),
                1,
                cv2.LINE_AA,
            )
        drawn_any = True

    if not drawn_any:
        return video_frame

    return cv2.addWeighted(video_frame, 1.0, overlay, alpha, 0.0)


def _to_serializable_list(value: Any) -> list[Any]:
    if value is None:
        return []
    if isinstance(value, torch.Tensor):
        value = value.detach().cpu().numpy()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if hasattr(value, "tolist"):
        try:
            return value.tolist()
        except Exception:
            pass
    if isinstance(value, list):
        return value
    return list(value)


def _encode_binary_mask_to_rle(mask: np.ndarray) -> dict[str, Any]:
    from pycocotools import mask as mask_util

    arr = np.asarray(mask)
    while arr.ndim > 2:
        arr = arr[0]
    arr = (arr > 0).astype(np.uint8)
    rle = mask_util.encode(np.asfortranarray(arr))
    counts = rle.get("counts")
    if isinstance(counts, bytes):
        rle["counts"] = counts.decode("utf-8")
    return {"size": list(rle.get("size", arr.shape)), "counts": rle["counts"]}


def serialize_frame_output(frame_index: int, outputs: dict[str, Any]) -> dict[str, Any]:
    masks_with_ids = iter_output_masks_with_ids(
        outputs, frame_h=outputs.get("_frame_h", 0), frame_w=outputs.get("_frame_w", 0)
    )
    # If caller did not provide helper dimensions, derive from first mask.
    if masks_with_ids and (outputs.get("_frame_h", 0) == 0 or outputs.get("_frame_w", 0) == 0):
        sample_mask = masks_with_ids[0][1]
        frame_h, frame_w = int(sample_mask.shape[0]), int(sample_mask.shape[1])
    else:
        frame_h = int(outputs.get("_frame_h", 0))
        frame_w = int(outputs.get("_frame_w", 0))

    out_obj_ids = _to_serializable_list(outputs.get("out_obj_ids"))
    out_probs = _to_serializable_list(outputs.get("out_probs"))
    out_boxes_xywh = _to_serializable_list(outputs.get("out_boxes_xywh"))

    rle_masks: list[dict[str, Any]] = []
    obj_ids: list[int] = []
    for obj_id, mask in masks_with_ids:
        if frame_h > 0 and frame_w > 0 and mask.shape != (frame_h, frame_w):
            mask = cv2.resize(
                mask.astype(np.float32),
                (frame_w, frame_h),
                interpolation=cv2.INTER_NEAREST,
            ) > 0.5
        obj_ids.append(int(obj_id))
        rle_masks.append(_encode_binary_mask_to_rle(mask))

    return {
        "frame_index": int(frame_index),
        "out_obj_ids": out_obj_ids if out_obj_ids else obj_ids,
        "out_probs": out_probs,
        "out_boxes_xywh": out_boxes_xywh,
        "out_binary_masks_rle": rle_masks,
    }


def save_frame_outputs_json(
    output_path: str,
    results_by_frame: dict[int, dict[str, Any]],
    frame_h: int,
    frame_w: int,
    *,
    total_video_frames: int | None = None,
    invalid_frame_indices: list[int] | None = None,
    keyframe_indices: list[int] | None = None,
) -> None:
    frames_payload: list[dict[str, Any]] = []
    for frame_index in sorted(results_by_frame.keys()):
        frame_outputs = dict(results_by_frame[frame_index])
        frame_outputs["_frame_h"] = frame_h
        frame_outputs["_frame_w"] = frame_w
        frames_payload.append(serialize_frame_output(frame_index, frame_outputs))
    payload = {
        "format_version": 1,
        "frame_size_hw": [int(frame_h), int(frame_w)],
        "total_video_frames": int(total_video_frames or 0),
        "num_frames_with_outputs": len(frames_payload),
        "invalid_frame_indices": sorted(set(invalid_frame_indices or [])),
        "keyframe_indices": sorted(set(keyframe_indices or [])),
        "frames": frames_payload,
    }
    write_json(output_path, payload)


def write_json(path: str, payload: dict[str, Any]) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)


def write_run_documentation(output_dir: str, metrics: dict[str, Any]) -> None:
    """Write compact per-run docs for traceability on shared storage."""
    manifest = dict(metrics)
    manifest["recorded_env"] = {
        "slurm_job_id": os.environ.get("SLURM_JOB_ID", ""),
        "slurm_array_task_id": os.environ.get("SLURM_ARRAY_TASK_ID", ""),
        "slurm_job_nodelist": os.environ.get("SLURM_JOB_NODELIST", ""),
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES", ""),
        "hf_home": os.environ.get("HF_HOME", ""),
        "transformers_cache": os.environ.get("TRANSFORMERS_CACHE", ""),
    }
    write_json(os.path.join(output_dir, "manifest.json"), manifest)

    lines = [
        "# Run README",
        "",
        "## What This Run Is",
        "",
        "- SAM3 agent-based segmentation/tracking run on one video.",
        "- This folder is self-documenting and safe to archive.",
        "",
        "## Key Inputs",
        "",
        f"- Model: `{metrics.get('model', '')}`",
        f"- Prompt: `{metrics.get('prompt', '')}`",
        f"- Video: `{metrics.get('video_path', '')}`",
        f"- Server URL: `{metrics.get('server_url', '')}`",
        f"- GPUs: `{metrics.get('gpus', '')}`",
        "",
        "## Key Outputs",
        "",
        f"- Metrics JSON: `run_metrics.json`",
        f"- Manifest JSON: `manifest.json`",
        f"- Overlay video: `{metrics.get('output_video_path', '')}`",
        f"- Generated prompts: `{metrics.get('generated_prompts_path', '')}`",
        "",
        "## Runtime",
        "",
        f"- Status: `{metrics.get('status', '')}`",
        f"- Runtime sec: `{metrics.get('runtime_sec', '')}`",
        f"- Finished UTC: `{metrics.get('finished_at_utc', '')}`",
        "",
        "## Slurm Context",
        "",
        f"- SLURM_JOB_ID: `{os.environ.get('SLURM_JOB_ID', '')}`",
        f"- SLURM_ARRAY_TASK_ID: `{os.environ.get('SLURM_ARRAY_TASK_ID', '')}`",
        f"- SLURM_JOB_NODELIST: `{os.environ.get('SLURM_JOB_NODELIST', '')}`",
        "",
    ]
    with open(os.path.join(output_dir, "README.md"), "w", encoding="utf-8") as handle:
        handle.write("\n".join(lines))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--video_path", required=True, type=str)
    parser.add_argument(
        "--prompt",
        default="Identify and segment any biological creatures.",
        type=str,
    )
    parser.add_argument("--server_url", required=True, type=str)
    parser.add_argument("--model", required=True, type=str)
    parser.add_argument("--api_key", type=str, default=None)
    parser.add_argument(
        "--prompt-profile",
        default=os.environ.get("SAM3_AGENT_PROMPT_PROFILE", "general"),
        type=str,
        help="Agent system-prompt profile (e.g., general, underwater).",
    )
    parser.add_argument("--output_dir", default="nibi_model_compare/run_out", type=str)
    parser.add_argument("--gpus", default="0", type=str)
    parser.add_argument(
        "--image_size",
        default=1008,
        type=int,
        help="Video predictor processing size. For current SAM3 checkpoint, use 1008.",
    )
    parser.add_argument("--max_completion_tokens", default=8000, type=int)
    parser.add_argument(
        "--save_frame_outputs_json",
        action="store_true",
        help="Save propagated per-frame outputs (obj IDs, boxes, masks as RLE) to JSON.",
    )
    parser.add_argument(
        "--frame_outputs_json_path",
        default="",
        type=str,
        help="Optional explicit path for per-frame outputs JSON.",
    )
    parser.add_argument("--save_prompts", action="store_true")
    parser.add_argument("--debug", action="store_true")
    parser.add_argument(
        "--temporal_keyframe_pipeline",
        action="store_true",
        help=(
            "Enable multi-keyframe pipeline: frame-quality scan, keyframe discovery, "
            "iterative agent prompting, and incremental propagation."
        ),
    )
    parser.add_argument(
        "--max_keyframes",
        default=6,
        type=int,
        help="Maximum number of keyframes to analyze in temporal mode.",
    )
    parser.add_argument(
        "--min_keyframe_gap",
        default=24,
        type=int,
        help="Minimum frame gap between selected keyframes in temporal mode.",
    )
    parser.add_argument(
        "--keyframe_motion_threshold",
        default=0.03,
        type=float,
        help="Minimum normalized motion score required to consider a keyframe event.",
    )
    parser.add_argument(
        "--id_match_iou_threshold",
        default=0.30,
        type=float,
        help="IoU threshold to reuse existing object ID at a keyframe.",
    )
    parser.add_argument(
        "--intra_keyframe_dedup_iou_threshold",
        default=0.85,
        type=float,
        help=(
            "IoU threshold to drop near-duplicate masks selected on the same keyframe "
            "before object-ID assignment."
        ),
    )
    parser.add_argument(
        "--drop_invalid_frames",
        action="store_true",
        help=(
            "Skip invalid/corrupt frames during final video rendering. "
            "Useful for black/white/corrupt frame removal."
        ),
    )
    parser.add_argument(
        "--invalid_frame_source",
        default="hybrid",
        choices=["heuristic", "mllm", "hybrid"],
        help=(
            "Invalid-frame detection source: heuristic-only, mllm-only, "
            "or hybrid union (default)."
        ),
    )
    parser.add_argument(
        "--invalid_frame_black_mean_threshold",
        default=8.0,
        type=float,
        help="Frame-quality threshold: near-black mean cutoff.",
    )
    parser.add_argument(
        "--invalid_frame_white_mean_threshold",
        default=247.0,
        type=float,
        help="Frame-quality threshold: near-white mean cutoff.",
    )
    parser.add_argument(
        "--invalid_frame_low_std_threshold",
        default=2.5,
        type=float,
        help="Frame-quality threshold: low grayscale std cutoff.",
    )
    parser.add_argument(
        "--invalid_frame_low_entropy_threshold",
        default=0.08,
        type=float,
        help="Frame-quality threshold: low normalized entropy cutoff.",
    )
    parser.add_argument(
        "--mllm_invalid_window_size",
        default=4,
        type=int,
        help="Number of frames per MLLM invalid-frame classification window.",
    )
    parser.add_argument(
        "--mllm_invalid_window_stride",
        default=4,
        type=int,
        help=(
            "Frame stride between MLLM invalid-frame windows. "
            "Use <= window_size for full-video coverage."
        ),
    )
    parser.add_argument(
        "--mllm_invalid_max_completion_tokens",
        default=512,
        type=int,
        help="Completion token budget for MLLM invalid-frame classification calls.",
    )
    parser.add_argument(
        "--mllm_invalid_use_collage",
        dest="mllm_invalid_use_collage",
        action="store_true",
        help="Use a collage image per invalid-frame classification window.",
    )
    parser.add_argument(
        "--no_mllm_invalid_use_collage",
        dest="mllm_invalid_use_collage",
        action="store_false",
        help="Disable collage mode for MLLM invalid-frame classification windows.",
    )
    parser.set_defaults(mllm_invalid_use_collage=True)
    parser.add_argument(
        "--mllm_invalid_collage_cols",
        default=2,
        type=int,
        help="Number of columns in invalid-frame classification collages.",
    )
    parser.add_argument(
        "--mllm_invalid_collage_tile_max_edge",
        default=512,
        type=int,
        help="Max edge (px) for each tile in invalid-frame classification collages.",
    )
    parser.add_argument(
        "--mllm_invalid_prompt_path",
        default="",
        type=str,
        help=(
            "Optional path to frame-validity MLLM system prompt template. "
            "If unset, uses profile-specific default."
        ),
    )
    parser.add_argument(
        "--mllm_invalid_max_json_retries",
        default=2,
        type=int,
        help="Retries per invalid-frame window when output is not valid strict JSON.",
    )
    parser.add_argument(
        "--mllm_invalid_fill_missing_with_heuristic",
        dest="mllm_invalid_fill_missing_with_heuristic",
        action="store_true",
        help=(
            "For frames without MLLM vote, use heuristic quality fallback to ensure "
            "every frame has a validity label."
        ),
    )
    parser.add_argument(
        "--no_mllm_invalid_fill_missing_with_heuristic",
        dest="mllm_invalid_fill_missing_with_heuristic",
        action="store_false",
        help="Do not use heuristic fallback for frames missing MLLM votes.",
    )
    parser.set_defaults(mllm_invalid_fill_missing_with_heuristic=True)
    parser.add_argument(
        "--discovery_mode",
        default="hybrid",
        choices=["motion", "mllm", "hybrid"],
        help=(
            "Keyframe discovery mode in temporal pipeline: "
            "motion-only, mllm-only, or hybrid (mllm with motion fallback)."
        ),
    )
    parser.add_argument(
        "--mllm_discovery_window_size",
        default=4,
        type=int,
        help="Number of frames per MLLM temporal discovery window.",
    )
    parser.add_argument(
        "--mllm_discovery_window_stride",
        default=24,
        type=int,
        help="Frame stride between MLLM temporal discovery windows.",
    )
    parser.add_argument(
        "--mllm_discovery_min_confidence",
        default=0.45,
        type=float,
        help="Minimum confidence for MLLM-discovered new-creature events.",
    )
    parser.add_argument(
        "--mllm_discovery_max_events",
        default=10,
        type=int,
        help="Maximum number of MLLM-discovered events kept before keyframe selection.",
    )
    parser.add_argument(
        "--mllm_discovery_max_completion_tokens",
        default=768,
        type=int,
        help="Completion token budget for MLLM temporal discovery calls.",
    )
    parser.add_argument(
        "--mllm_discovery_use_collage",
        dest="mllm_discovery_use_collage",
        action="store_true",
        help=(
            "Use a single temporal collage image per discovery window. "
            "Recommended when server allows only one image per request."
        ),
    )
    parser.add_argument(
        "--no_mllm_discovery_use_collage",
        dest="mllm_discovery_use_collage",
        action="store_false",
        help="Disable collage mode and send multiple images per discovery window.",
    )
    parser.set_defaults(mllm_discovery_use_collage=True)
    parser.add_argument(
        "--mllm_discovery_collage_cols",
        default=2,
        type=int,
        help="Number of columns in temporal discovery collage layout.",
    )
    parser.add_argument(
        "--mllm_discovery_collage_tile_max_edge",
        default=512,
        type=int,
        help="Max edge (px) for each tile in temporal discovery collage.",
    )
    parser.add_argument(
        "--mllm_discovery_prompt_path",
        default="",
        type=str,
        help=(
            "Optional path to temporal discovery system prompt template. "
            "If unset, uses profile-specific default."
        ),
    )
    parser.add_argument(
        "--mllm_discovery_max_json_retries",
        default=2,
        type=int,
        help="Retries per discovery window when model output is not valid strict JSON.",
    )
    return parser.parse_args()


def run() -> int:
    args = parse_args()
    os.environ["SAM3_AGENT_PROMPT_PROFILE"] = args.prompt_profile
    os.makedirs(args.output_dir, exist_ok=True)
    start_ts = time.time()
    metrics: dict[str, Any] = {
        "status": "started",
        "video_path": os.path.abspath(args.video_path),
        "prompt": args.prompt,
        "prompt_profile": args.prompt_profile,
        "server_url": args.server_url,
        "model": args.model,
        "gpus": args.gpus,
    }

    backend: Any = None
    cap = None

    try:
        if cv2 is None or np is None or torch is None or Image is None:
            raise RuntimeError(
                "Missing required packages. Install dependencies including "
                "opencv-python, numpy, torch, and pillow in this environment."
            )

        from frame_quality import scan_video_frame_quality
        from frame_quality_mllm import discover_invalid_frames_with_mllm
        from keyframe_discovery import discover_keyframes_from_motion
        from keyframe_discovery_mllm import discover_keyframes_with_mllm
        from track_id_matching import assign_object_ids_by_iou

        from sam3.agent.agent_core import agent_inference
        from sam3.agent.client_llm import (
            send_generate_request as send_generate_request_orig,
        )
        from sam3.agent.client_sam3 import remove_overlapping_masks, sam3_inference
        from sam3.agent.viz import visualize
        from sam3.apps.interactive_video.backend import PredictorBackend
        from sam3.model.sam3_image_processor import Sam3Processor
        from sam3.model_builder import build_sam3_image_model

        class LocalSam3Service:
            """Local adapter for agent_core tool-calling path."""

            def __init__(self, processor: Any, output_dir: str) -> None:
                self.processor = processor
                self.output_dir = output_dir
                os.makedirs(output_dir, exist_ok=True)

            def call_service(
                self,
                image_path: str,
                text_prompt: str,
                output_folder_path: str | None = None,
            ) -> str:
                if output_folder_path is None:
                    output_folder_path = self.output_dir
                os.makedirs(output_folder_path, exist_ok=True)

                outputs = sam3_inference(self.processor, image_path, text_prompt)
                outputs = remove_overlapping_masks(outputs)

                safe_prompt = text_prompt.replace("/", "_").replace(" ", "_")
                out_name = f"{os.path.basename(image_path)}_{safe_prompt}"
                output_image_path = os.path.join(output_folder_path, f"{out_name}.png")
                output_json_path = os.path.join(output_folder_path, f"{out_name}.json")

                outputs = {
                    "original_image_path": image_path,
                    "output_image_path": output_image_path,
                    **outputs,
                }

                if "pred_scores" in outputs and outputs["pred_scores"]:
                    order = sorted(
                        range(len(outputs["pred_scores"])),
                        key=lambda i: outputs["pred_scores"][i],
                        reverse=True,
                    )
                    outputs["pred_scores"] = [outputs["pred_scores"][i] for i in order]
                    outputs["pred_boxes"] = [outputs["pred_boxes"][i] for i in order]
                    outputs["pred_masks"] = [outputs["pred_masks"][i] for i in order]

                valid_masks: list[Any] = []
                valid_boxes: list[Any] = []
                valid_scores: list[Any] = []
                for i, rle in enumerate(outputs.get("pred_masks", [])):
                    if len(rle) > 4:
                        valid_masks.append(rle)
                        valid_boxes.append(outputs["pred_boxes"][i])
                        valid_scores.append(outputs["pred_scores"][i])
                outputs["pred_masks"] = valid_masks
                outputs["pred_boxes"] = valid_boxes
                outputs["pred_scores"] = valid_scores

                with open(output_json_path, "w", encoding="utf-8") as handle:
                    json.dump(outputs, handle, indent=2)
                visualize(outputs).save(output_image_path)
                return output_json_path

        bpe_path = find_bpe_path()
        image_model = build_sam3_image_model(bpe_path=bpe_path)
        image_processor = Sam3Processor(image_model, confidence_threshold=0.4)
        local_service = LocalSam3Service(
            image_processor, os.path.join(args.output_dir, "sam_service")
        )

        cap = cv2.VideoCapture(args.video_path)
        if not cap.isOpened():
            raise RuntimeError(f"Could not open video: {args.video_path}")

        ret, frame = cap.read()
        if not ret:
            raise RuntimeError("Could not read frame 0 from video.")

        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = float(cap.get(cv2.CAP_PROP_FPS) or 30.0)
        frame_h, frame_w = frame.shape[:2]
        cap.release()
        cap = None

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
            max_tokens=args.max_completion_tokens,
        )
        send_req_discovery = partial(
            send_generate_request_orig,
            server_url=args.server_url,
            model=args.model,
            api_key=api_key,
            max_tokens=args.mllm_discovery_max_completion_tokens,
        )
        send_req_invalid_frames = partial(
            send_generate_request_orig,
            server_url=args.server_url,
            model=args.model,
            api_key=api_key,
            max_tokens=args.mllm_invalid_max_completion_tokens,
        )
        # Keep frame_0 extraction for compatibility and debugging.
        frame_0_path = os.path.join(args.output_dir, "frame_0.jpg")
        Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)).save(frame_0_path)

        invalid_frame_indices: list[int] = []
        keyframe_indices: list[int] = [0]
        keyframe_event_candidates: list[dict[str, Any]] = []
        heuristic_invalid_frame_indices: list[int] = []
        mllm_invalid_frame_indices: list[int] = []

        should_scan_frame_quality = (
            args.temporal_keyframe_pipeline or args.drop_invalid_frames
        )
        if should_scan_frame_quality:
            if args.invalid_frame_source in {"mllm", "hybrid"}:
                mllm_quality = discover_invalid_frames_with_mllm(
                    video_path=args.video_path,
                    send_generate_request_fn=send_req_invalid_frames,
                    initial_text_prompt=args.prompt,
                    total_frames=total_frames,
                    output_dir=os.path.join(args.output_dir, "mllm_frame_validity"),
                    window_size=args.mllm_invalid_window_size,
                    window_stride=args.mllm_invalid_window_stride,
                    use_collage=args.mllm_invalid_use_collage,
                    collage_cols=args.mllm_invalid_collage_cols,
                    collage_tile_max_edge=args.mllm_invalid_collage_tile_max_edge,
                    prompt_profile=args.prompt_profile,
                    prompt_template_path=(
                        args.mllm_invalid_prompt_path.strip() or None
                    ),
                    max_json_retries=args.mllm_invalid_max_json_retries,
                    fill_missing_with_heuristic=(
                        args.mllm_invalid_fill_missing_with_heuristic
                    ),
                )
                mllm_invalid_frame_indices = sorted(
                    set(mllm_quality.get("invalid_frame_indices", []))
                )
                write_json(
                    os.path.join(args.output_dir, "frame_quality_scan_mllm.json"),
                    mllm_quality,
                )
                metrics["mllm_invalid_frame_count"] = len(mllm_invalid_frame_indices)
                metrics["mllm_invalid_frame_ratio"] = float(
                    mllm_quality.get("invalid_frame_ratio", 0.0)
                )

            if args.invalid_frame_source in {"heuristic", "hybrid"}:
                heuristic_quality = scan_video_frame_quality(
                    args.video_path,
                    black_mean_threshold=args.invalid_frame_black_mean_threshold,
                    white_mean_threshold=args.invalid_frame_white_mean_threshold,
                    low_std_threshold=args.invalid_frame_low_std_threshold,
                    low_entropy_threshold=args.invalid_frame_low_entropy_threshold,
                )
                heuristic_invalid_frame_indices = sorted(
                    set(heuristic_quality.get("invalid_frame_indices", []))
                )
                write_json(
                    os.path.join(args.output_dir, "frame_quality_scan_heuristic.json"),
                    heuristic_quality,
                )
                metrics["heuristic_invalid_frame_count"] = len(
                    heuristic_invalid_frame_indices
                )
                metrics["heuristic_invalid_frame_ratio"] = float(
                    heuristic_quality.get("invalid_frame_ratio", 0.0)
                )

            if args.invalid_frame_source == "heuristic":
                invalid_frame_indices = list(heuristic_invalid_frame_indices)
            elif args.invalid_frame_source == "mllm":
                invalid_frame_indices = list(mllm_invalid_frame_indices)
            else:
                invalid_frame_indices = sorted(
                    set(heuristic_invalid_frame_indices)
                    | set(mllm_invalid_frame_indices)
                )

            invalid_ratio = (
                len(invalid_frame_indices) / float(total_frames)
                if total_frames > 0
                else 0.0
            )
            metrics["invalid_frame_source"] = args.invalid_frame_source
            metrics["invalid_frame_count"] = len(invalid_frame_indices)
            metrics["invalid_frame_ratio"] = invalid_ratio
            if len(invalid_frame_indices) > 0:
                metrics["invalid_frame_sample"] = invalid_frame_indices[:20]

            write_json(
                os.path.join(args.output_dir, "frame_quality_scan.json"),
                {
                    "mode": args.invalid_frame_source,
                    "total_video_frames": int(total_frames),
                    "invalid_frame_indices": invalid_frame_indices,
                    "invalid_frame_ratio": float(invalid_ratio),
                    "heuristic_invalid_frame_indices": heuristic_invalid_frame_indices,
                    "mllm_invalid_frame_indices": mllm_invalid_frame_indices,
                },
            )

        if args.temporal_keyframe_pipeline:
            invalid_set = set(invalid_frame_indices)
            valid_frame_indices = [
                frame_idx for frame_idx in range(total_frames) if frame_idx not in invalid_set
            ]
            if len(valid_frame_indices) == 0:
                raise RuntimeError(
                    "No valid frames remain after quality filtering; cannot run temporal pipeline."
                )

            motion_plan = discover_keyframes_from_motion(
                args.video_path,
                invalid_frame_indices=invalid_set,
                max_keyframes=args.max_keyframes,
                min_keyframe_gap=args.min_keyframe_gap,
                motion_threshold=args.keyframe_motion_threshold,
            )
            write_json(
                os.path.join(args.output_dir, "keyframe_discovery_motion.json"),
                motion_plan,
            )

            motion_keyframes = list(motion_plan.get("keyframes", []))
            motion_events = list(motion_plan.get("event_candidates", []))
            mllm_plan: dict[str, Any] = {}
            mllm_keyframes: list[int] = []
            mllm_events: list[dict[str, Any]] = []

            if args.discovery_mode in {"mllm", "hybrid"}:
                mllm_plan = discover_keyframes_with_mllm(
                    video_path=args.video_path,
                    send_generate_request_fn=send_req_discovery,
                    initial_text_prompt=args.prompt,
                    valid_frame_indices=valid_frame_indices,
                    output_dir=os.path.join(args.output_dir, "mllm_discovery"),
                    max_keyframes=args.max_keyframes,
                    min_keyframe_gap=args.min_keyframe_gap,
                    window_size=args.mllm_discovery_window_size,
                    window_stride=args.mllm_discovery_window_stride,
                    min_confidence=args.mllm_discovery_min_confidence,
                    max_events=args.mllm_discovery_max_events,
                    use_collage=args.mllm_discovery_use_collage,
                    collage_cols=args.mllm_discovery_collage_cols,
                    collage_tile_max_edge=args.mllm_discovery_collage_tile_max_edge,
                    prompt_profile=args.prompt_profile,
                    prompt_template_path=(
                        args.mllm_discovery_prompt_path.strip() or None
                    ),
                    max_json_retries=args.mllm_discovery_max_json_retries,
                )
                write_json(
                    os.path.join(args.output_dir, "keyframe_discovery_mllm.json"),
                    mllm_plan,
                )
                mllm_keyframes = list(mllm_plan.get("keyframes", []))
                mllm_events = list(mllm_plan.get("event_candidates", []))

            if args.discovery_mode == "motion":
                keyframe_indices = motion_keyframes
                keyframe_event_candidates = motion_events
            elif args.discovery_mode == "mllm":
                keyframe_indices = mllm_keyframes if mllm_keyframes else motion_keyframes
                keyframe_event_candidates = mllm_events if mllm_events else motion_events
            else:
                # hybrid: prefer MLLM keyframes/events, then fill from motion if needed.
                keyframe_indices = list(mllm_keyframes)
                keyframe_event_candidates = list(mllm_events)
                for mk in motion_keyframes:
                    if len(keyframe_indices) >= args.max_keyframes:
                        break
                    if all(
                        abs(int(mk) - int(existing)) >= args.min_keyframe_gap
                        for existing in keyframe_indices
                    ):
                        keyframe_indices.append(int(mk))
                keyframe_indices = sorted(set(keyframe_indices))
                if len(keyframe_event_candidates) == 0:
                    keyframe_event_candidates = motion_events

            if (0 not in invalid_frame_indices) and (0 not in keyframe_indices):
                keyframe_indices = [0] + keyframe_indices
            if not keyframe_indices:
                keyframe_indices = [valid_frame_indices[0]]
            keyframe_indices = sorted(set(int(x) for x in keyframe_indices))
            if len(keyframe_indices) > args.max_keyframes:
                keyframe_indices = keyframe_indices[: args.max_keyframes]

            keyframe_plan = {
                "mode": args.discovery_mode,
                "keyframes": keyframe_indices,
                "event_candidates": keyframe_event_candidates,
                "motion_fallback_keyframes": motion_keyframes,
                "motion_fallback_events": motion_events,
                "mllm_keyframes": mllm_keyframes,
                "mllm_events": mllm_events,
            }
            write_json(
                os.path.join(args.output_dir, "keyframe_discovery.json"),
                keyframe_plan,
            )

        metrics["keyframe_indices"] = keyframe_indices
        metrics["keyframe_event_candidates"] = keyframe_event_candidates
        metrics["discovery_mode"] = args.discovery_mode
        metrics["mllm_discovery_config"] = {
            "window_size": int(args.mllm_discovery_window_size),
            "window_stride": int(args.mllm_discovery_window_stride),
            "min_confidence": float(args.mllm_discovery_min_confidence),
            "max_events": int(args.mllm_discovery_max_events),
            "max_completion_tokens": int(args.mllm_discovery_max_completion_tokens),
            "use_collage": bool(args.mllm_discovery_use_collage),
            "collage_cols": int(args.mllm_discovery_collage_cols),
            "collage_tile_max_edge": int(args.mllm_discovery_collage_tile_max_edge),
            "max_json_retries": int(args.mllm_discovery_max_json_retries),
            "prompt_path": args.mllm_discovery_prompt_path,
        }
        metrics["mllm_invalid_frame_config"] = {
            "source": args.invalid_frame_source,
            "window_size": int(args.mllm_invalid_window_size),
            "window_stride": int(args.mllm_invalid_window_stride),
            "max_completion_tokens": int(args.mllm_invalid_max_completion_tokens),
            "use_collage": bool(args.mllm_invalid_use_collage),
            "collage_cols": int(args.mllm_invalid_collage_cols),
            "collage_tile_max_edge": int(args.mllm_invalid_collage_tile_max_edge),
            "max_json_retries": int(args.mllm_invalid_max_json_retries),
            "prompt_path": args.mllm_invalid_prompt_path,
            "fill_missing_with_heuristic": bool(
                args.mllm_invalid_fill_missing_with_heuristic
            ),
        }
        metrics["total_video_frames"] = total_frames

        generated_prompts: list[dict[str, Any]] = []
        agent_runs: list[dict[str, Any]] = []
        failed_agent_keyframes: list[dict[str, Any]] = []
        session_id: str | None = None
        results_by_frame: dict[int, dict[str, Any]] = {}

        if not args.save_prompts:
            gpu_ids = [int(x.strip()) for x in args.gpus.split(",") if x.strip()]
            backend = PredictorBackend(gpu_ids=gpu_ids)
            session_id = backend.start_session(
                resource_path=args.video_path,
                image_size=args.image_size,
            )
            metrics["video_session_id"] = session_id

        next_obj_id = 1
        total_agent_masks_raw = 0
        total_agent_masks_after_keyframe_dedup = 0
        total_history_len = 0
        keyframe_dedup_events: list[dict[str, Any]] = []

        for keyframe_idx in keyframe_indices:
            if keyframe_idx in invalid_frame_indices:
                print(
                    f"[warn] keyframe {keyframe_idx} marked invalid by quality scan; skipping."
                )
                continue

            frame_k = read_video_frame(args.video_path, keyframe_idx)
            if frame_k is None:
                print(
                    f"[warn] could not decode keyframe {keyframe_idx}; skipping keyframe."
                )
                continue

            keyframe_path = os.path.join(args.output_dir, f"frame_{keyframe_idx}.jpg")
            Image.fromarray(cv2.cvtColor(frame_k, cv2.COLOR_BGR2RGB)).save(keyframe_path)

            try:
                history, final_outputs, _ = agent_inference(
                    img_path=keyframe_path,
                    initial_text_prompt=args.prompt,
                    send_generate_request=send_req,
                    call_sam_service=local_service.call_service,
                    output_dir=os.path.join(args.output_dir, "agent_out"),
                    debug=args.debug,
                )
            except Exception as exc:
                err_msg = str(exc).strip() or repr(exc) or type(exc).__name__
                print(
                    f"[warn] agent_inference failed on keyframe {keyframe_idx}: "
                    f"{type(exc).__name__}: {err_msg}"
                )
                failed_agent_keyframes.append(
                    {
                        "frame_idx": int(keyframe_idx),
                        "frame_path": keyframe_path,
                        "error_type": type(exc).__name__,
                        "error": err_msg,
                    }
                )
                continue

            selected_masks_rle = list(final_outputs.get("pred_masks", []))
            raw_mask_count = len(selected_masks_rle)
            total_agent_masks_raw += raw_mask_count
            total_history_len += len(history)
            agent_runs.append(
                {
                    "frame_idx": int(keyframe_idx),
                    "frame_path": keyframe_path,
                    "history_len": int(len(history)),
                    "num_masks": int(raw_mask_count),
                }
            )

            decoded_masks: list[np.ndarray] = []
            for rle_idx, rle in enumerate(selected_masks_rle):
                try:
                    decoded_masks.append(
                        decode_rle_to_mask(rle, frame_h, frame_w).astype(bool)
                    )
                except Exception as exc:
                    print(
                        f"[warn] failed to decode RLE {rle_idx} on frame {keyframe_idx}: {exc}"
                    )

            if not decoded_masks:
                continue

            decoded_masks, _kept_mask_indices, dropped_rows = deduplicate_masks_by_iou(
                decoded_masks,
                iou_threshold=float(args.intra_keyframe_dedup_iou_threshold),
            )
            if dropped_rows:
                dropped_count = len(dropped_rows)
                print(
                    f"[warn] keyframe {keyframe_idx}: dropped {dropped_count} near-duplicate "
                    "mask(s) before ID assignment."
                )
                keyframe_dedup_events.append(
                    {
                        "frame_idx": int(keyframe_idx),
                        "raw_mask_count": int(raw_mask_count),
                        "kept_mask_count": int(len(decoded_masks)),
                        "dropped_mask_count": int(dropped_count),
                        "dropped_rows": dropped_rows,
                    }
                )
            if not decoded_masks:
                continue
            total_agent_masks_after_keyframe_dedup += len(decoded_masks)

            existing_masks_with_ids: list[tuple[int, np.ndarray]] = []
            if keyframe_idx in results_by_frame:
                existing_masks_with_ids = iter_output_masks_with_ids(
                    results_by_frame[keyframe_idx], frame_h, frame_w
                )

            assignment = assign_object_ids_by_iou(
                existing_masks_with_ids,
                decoded_masks,
                next_obj_id=next_obj_id,
                iou_match_threshold=args.id_match_iou_threshold,
            )
            assigned_ids: list[int] = list(assignment["assigned_ids"])
            next_obj_id = int(assignment["next_obj_id"])
            assignment_rows: list[dict[str, Any]] = list(assignment["assignments"])

            prompts_added_this_frame = 0
            for mask_idx, (mask, obj_id) in enumerate(zip(decoded_masks, assigned_ids)):
                point = get_center_point(mask)
                if point is None:
                    continue

                meta_row = assignment_rows[mask_idx]
                prompt_data = {
                    "frame_idx": int(keyframe_idx),
                    "obj_id": int(obj_id),
                    "points": [[float(point[0]), float(point[1]), 1]],
                    "label": 1,
                    "source": (
                        "agent_center_point_temporal"
                        if args.temporal_keyframe_pipeline
                        else "agent_center_point"
                    ),
                    "best_iou_at_assignment": float(meta_row["best_iou"]),
                    "matched_existing_obj_id": meta_row["matched_existing_obj_id"],
                    "is_new_obj_id": bool(meta_row["is_new_obj_id"]),
                }
                generated_prompts.append(prompt_data)
                prompts_added_this_frame += 1

                if backend is not None and session_id is not None:
                    backend.add_point_prompt(
                        session_id=session_id,
                        frame_idx=int(keyframe_idx),
                        obj_id=int(obj_id),
                        points=[(float(point[0]), float(point[1]), 1)],
                        frame_size=(frame_w, frame_h),
                    )

            if (
                backend is not None
                and session_id is not None
                and prompts_added_this_frame > 0
                and not args.save_prompts
            ):
                try:
                    req = {
                        "session_id": session_id,
                        "type": "propagate_in_video",
                        "start_frame_index": int(keyframe_idx),
                        "propagation_direction": "both",
                    }
                    for output in backend.propagate(req):
                        frame_index = int(output["frame_index"])
                        results_by_frame[frame_index] = output["outputs"]
                except Exception as exc:
                    err_msg = str(exc).strip() or repr(exc) or type(exc).__name__
                    raise RuntimeError(
                        f"propagate_in_video failed ({type(exc).__name__}): {err_msg}"
                    ) from exc

        metrics["agent_runs"] = agent_runs
        metrics["failed_agent_keyframes"] = failed_agent_keyframes
        metrics["agent_history_len"] = total_history_len
        metrics["num_agent_masks"] = total_agent_masks_after_keyframe_dedup
        metrics["num_agent_masks_raw"] = total_agent_masks_raw
        metrics["num_agent_masks_after_keyframe_dedup"] = (
            total_agent_masks_after_keyframe_dedup
        )
        metrics["keyframe_dedup_event_count"] = len(keyframe_dedup_events)
        if keyframe_dedup_events:
            metrics["keyframe_dedup_events"] = keyframe_dedup_events

        prompts_path = os.path.join(args.output_dir, "generated_prompts.json")
        write_json(
            prompts_path,
            {
                "prompts": generated_prompts,
                "keyframe_indices": keyframe_indices,
                "invalid_frame_indices": invalid_frame_indices,
            },
        )
        metrics["num_generated_prompts"] = len(generated_prompts)
        metrics["generated_prompts_path"] = prompts_path

        output_video_path = ""
        frame_outputs_json_path = ""

        if args.save_prompts:
            metrics["status"] = "success_prompts_only"
        elif backend is None or session_id is None:
            metrics["status"] = "success_no_propagation"
        else:
            if len(results_by_frame) == 0:
                metrics["status"] = "success_no_propagation"
                metrics["invalid_frame_indices"] = invalid_frame_indices
                metrics["output_video_path"] = ""
                metrics["output_dir"] = os.path.abspath(args.output_dir)
                return_code = 0
                return return_code

            out_path = os.path.join(args.output_dir, "output_video.mp4")
            should_save_frame_outputs = args.save_frame_outputs_json or os.environ.get(
                "SAM3_SAVE_FRAME_OUTPUTS_JSON", "1"
            ) == "1"
            if should_save_frame_outputs:
                frame_outputs_json_path = (
                    args.frame_outputs_json_path
                    or os.path.join(args.output_dir, "frame_outputs_rle.json")
                )
                save_frame_outputs_json(
                    frame_outputs_json_path,
                    results_by_frame,
                    frame_h=frame_h,
                    frame_w=frame_w,
                    total_video_frames=total_frames,
                    invalid_frame_indices=invalid_frame_indices,
                    keyframe_indices=keyframe_indices,
                )
                metrics["frame_outputs_json_path"] = frame_outputs_json_path

            cap = cv2.VideoCapture(args.video_path)
            writer = cv2.VideoWriter(
                out_path,
                cv2.VideoWriter_fourcc(*"mp4v"),
                fps,
                (frame_w, frame_h),
            )

            frame_index = 0
            dropped_frame_count = 0
            while True:
                ret, video_frame = cap.read()
                if not ret:
                    break
                if args.drop_invalid_frames and frame_index in invalid_frame_indices:
                    dropped_frame_count += 1
                    frame_index += 1
                    continue
                output = results_by_frame.get(frame_index)
                if output:
                    video_frame = overlay_masks_on_frame(video_frame, output)
                writer.write(video_frame)
                frame_index += 1

            writer.release()
            cap.release()
            cap = None

            output_video_path = out_path
            metrics["status"] = "success"
            metrics["frames_with_outputs"] = len(results_by_frame)
            metrics["invalid_frame_indices"] = invalid_frame_indices
            metrics["dropped_frame_count"] = dropped_frame_count
            if total_frames > 0:
                metrics["frame_output_fraction"] = len(results_by_frame) / total_frames

        metrics["output_video_path"] = output_video_path
        metrics["output_dir"] = os.path.abspath(args.output_dir)
        return_code = 0

    except Exception as exc:
        metrics["status"] = "failed"
        metrics["error"] = str(exc).strip() or repr(exc) or type(exc).__name__
        metrics["error_type"] = type(exc).__name__
        metrics["traceback"] = traceback.format_exc()
        return_code = 1
        print(f"[error] {metrics['error']}")
        print(metrics["traceback"])

    finally:
        if backend is not None:
            try:
                backend.shutdown()
            except Exception as exc:
                print(f"[warn] backend shutdown failed: {exc}")
        if cap is not None:
            cap.release()
        metrics["runtime_sec"] = round(time.time() - start_ts, 3)
        metrics["finished_at_utc"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
        write_json(os.path.join(args.output_dir, "run_metrics.json"), metrics)
        write_run_documentation(args.output_dir, metrics)
        print(json.dumps(metrics, indent=2))

    return return_code


if __name__ == "__main__":
    raise SystemExit(run())
