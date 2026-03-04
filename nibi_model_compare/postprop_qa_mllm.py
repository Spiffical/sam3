from __future__ import annotations

import json
import math
import os
import re
from collections import deque
from typing import Any, Callable

import cv2
import numpy as np

from frame_quality import analyze_frame_quality


def _repo_root() -> str:
    return os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))


def _default_prompt_path(prompt_profile: str) -> str:
    base = os.path.join(_repo_root(), "sam3", "agent", "system_prompts")
    profile = (prompt_profile or "").strip().lower()
    if profile == "underwater":
        return os.path.join(base, "system_prompt_postprop_qa_underwater.txt")
    return os.path.join(base, "system_prompt_postprop_qa_general.txt")


def _read_prompt_file(path: str) -> str:
    with open(path, "r", encoding="utf-8") as handle:
        return handle.read().strip()


def _resolve_system_prompt(
    prompt_profile: str,
    prompt_template_path: str | None,
) -> str:
    if prompt_template_path:
        if not os.path.exists(prompt_template_path):
            raise FileNotFoundError(f"Post-prop QA prompt template not found: {prompt_template_path}")
        return _read_prompt_file(prompt_template_path)

    candidate = _default_prompt_path(prompt_profile)
    if os.path.exists(candidate):
        return _read_prompt_file(candidate)

    return (
        "Review segmentation quality and return STRICT JSON only with schema "
        '{"frame_index":int,"frame_validity":"valid|invalid","reasons":[str],'
        '"bad_object_ids":[int],"overlap_pairs":[[int,int]],'
        '"missing_creatures":bool,"missing_creature_reason":str,"needs_rerun":bool,'
        '"confidence":float}.'
    )


def _extract_json_object(text: str) -> dict[str, Any] | None:
    if not text:
        return None

    fenced = re.findall(
        r"```(?:json)?\s*(.*?)```", text, flags=re.IGNORECASE | re.DOTALL
    )
    for block in fenced:
        block = block.strip()
        if not block:
            continue
        try:
            parsed = json.loads(block)
            if isinstance(parsed, dict):
                return parsed
        except Exception:
            pass

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


def _rescale_frame_bgr(frame_bgr: np.ndarray, max_edge: int) -> np.ndarray:
    h, w = frame_bgr.shape[:2]
    longest = max(h, w)
    if longest <= max_edge:
        return frame_bgr
    scale = float(max_edge) / float(longest)
    new_w = max(1, int(round(w * scale)))
    new_h = max(1, int(round(h * scale)))
    return cv2.resize(frame_bgr, (new_w, new_h), interpolation=cv2.INTER_AREA)


def _draw_tile_label(tile: np.ndarray, label: str) -> np.ndarray:
    out = tile.copy()
    (tw, th), baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.52, 1)
    pad_x = 8
    pad_y = 6
    x1, y1 = 6, 6
    x2 = min(out.shape[1] - 1, x1 + tw + (2 * pad_x))
    y2 = min(out.shape[0] - 1, y1 + th + baseline + (2 * pad_y))
    cv2.rectangle(out, (x1, y1), (x2, y2), (30, 30, 30), -1)
    cv2.rectangle(out, (x1, y1), (x2, y2), (200, 200, 200), 1)
    cv2.putText(
        out,
        label,
        (x1 + pad_x, y2 - baseline - pad_y + 1),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.52,
        (255, 255, 255),
        1,
        cv2.LINE_AA,
    )
    return out


def _build_labeled_collage(
    tiles: list[tuple[str, np.ndarray]],
    output_path: str,
    *,
    cols: int = 4,
    tile_max_edge: int = 320,
) -> tuple[int, int]:
    assert len(tiles) > 0
    cols = max(1, int(cols))
    rows = int(math.ceil(len(tiles) / float(cols)))

    rendered: list[np.ndarray] = []
    max_h = 0
    max_w = 0
    for label, frame in tiles:
        tile = _rescale_frame_bgr(frame, tile_max_edge)
        tile = _draw_tile_label(tile, label)
        rendered.append(tile)
        max_h = max(max_h, tile.shape[0])
        max_w = max(max_w, tile.shape[1])

    collage = np.zeros((rows * max_h, cols * max_w, 3), dtype=np.uint8)
    for i, tile in enumerate(rendered):
        r = i // cols
        c = i % cols
        y0 = r * max_h
        x0 = c * max_w
        h, w = tile.shape[:2]
        collage[y0 : y0 + h, x0 : x0 + w] = tile

    cv2.imwrite(output_path, collage)
    return rows, cols


def _to_list(value: Any) -> list[Any]:
    if value is None:
        return []
    if isinstance(value, np.ndarray):
        return value.tolist()
    if hasattr(value, "detach"):
        try:
            value = value.detach().cpu().numpy()
            return value.tolist()
        except Exception:
            pass
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


def _iter_output_masks_with_ids(
    outputs: dict[str, Any], frame_h: int, frame_w: int
) -> list[tuple[int, np.ndarray]]:
    if not isinstance(outputs, dict):
        return []

    raw_masks = outputs.get("out_binary_masks")
    raw_ids = _to_list(outputs.get("out_obj_ids"))
    if raw_masks is None:
        return []

    raw_masks_list = _to_list(raw_masks)
    out: list[tuple[int, np.ndarray]] = []
    for i, raw_mask in enumerate(raw_masks_list):
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
        if i < len(raw_ids):
            try:
                obj_id = int(raw_ids[i])
            except Exception:
                obj_id = i + 1
        else:
            obj_id = i + 1
        out.append((obj_id, mask))
    return out


def _object_metadata(outputs: dict[str, Any]) -> dict[int, dict[str, Any]]:
    out_obj_ids = _to_list(outputs.get("out_obj_ids"))
    out_probs = _to_list(outputs.get("out_probs"))
    out_boxes_xywh = _to_list(outputs.get("out_boxes_xywh"))
    meta: dict[int, dict[str, Any]] = {}
    for i, raw_id in enumerate(out_obj_ids):
        try:
            obj_id = int(raw_id)
        except Exception:
            continue
        confidence: float | None = None
        if i < len(out_probs):
            v = out_probs[i]
            if isinstance(v, list) and len(v) > 0:
                v = v[0]
            try:
                confidence = float(v)
            except Exception:
                confidence = None
        box = None
        if i < len(out_boxes_xywh):
            b = out_boxes_xywh[i]
            if isinstance(b, list) and len(b) >= 4:
                try:
                    box = [float(b[0]), float(b[1]), float(b[2]), float(b[3])]
                except Exception:
                    box = None
        meta[obj_id] = {"confidence": confidence, "box_xywh": box}
    return meta


def _overlay_masks_with_boxes(frame_bgr: np.ndarray, outputs: dict[str, Any]) -> np.ndarray:
    out = frame_bgr.copy()
    h, w = out.shape[:2]
    masks_with_ids = _iter_output_masks_with_ids(outputs, h, w)
    meta = _object_metadata(outputs)
    if not masks_with_ids:
        return out

    overlay = np.zeros_like(out)
    for obj_id, mask in masks_with_ids:
        color = (
            int((obj_id * 47) % 255),
            int((obj_id * 89 + 37) % 255),
            int((obj_id * 131 + 73) % 255),
        )
        overlay[mask] = color
        mask_u8 = (mask.astype(np.uint8) * 255)
        contours, _ = cv2.findContours(mask_u8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(out, contours, -1, color, 2)

        ys, xs = np.where(mask)
        if len(xs) == 0 or len(ys) == 0:
            continue
        x1, y1 = int(xs.min()), int(ys.min())
        x2, y2 = int(xs.max()), int(ys.max())
        cv2.rectangle(out, (x1, y1), (x2, y2), color, 2)
        conf = meta.get(obj_id, {}).get("confidence")
        label = f"id {obj_id}"
        if isinstance(conf, (float, int)):
            label += f" p {float(conf):.1f}"
        cv2.putText(
            out,
            label,
            (x1, max(14, y1 - 4)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )

    return cv2.addWeighted(out, 1.0, overlay, 0.35, 0.0)


def _binary_mask_iou(mask_a: np.ndarray, mask_b: np.ndarray) -> float:
    a = np.asarray(mask_a).astype(bool)
    b = np.asarray(mask_b).astype(bool)
    inter = float(np.logical_and(a, b).sum())
    if inter <= 0.0:
        return 0.0
    union = float(np.logical_or(a, b).sum())
    if union <= 0.0:
        return 0.0
    return inter / union


def _heuristic_overlap_pairs(
    masks_with_ids: list[tuple[int, np.ndarray]],
    iou_threshold: float,
) -> list[list[int]]:
    pairs: list[list[int]] = []
    for i in range(len(masks_with_ids)):
        obj_i, mask_i = masks_with_ids[i]
        for j in range(i + 1, len(masks_with_ids)):
            obj_j, mask_j = masks_with_ids[j]
            iou = _binary_mask_iou(mask_i, mask_j)
            if iou >= iou_threshold:
                pairs.append([int(obj_i), int(obj_j)])
    return pairs


def _extract_object_crops(
    frame_bgr: np.ndarray,
    outputs: dict[str, Any],
    *,
    max_object_crops: int,
    crop_context_ratio: float,
    overlay_alpha: float = 0.35,
) -> list[tuple[str, np.ndarray]]:
    h, w = frame_bgr.shape[:2]
    masks_with_ids = _iter_output_masks_with_ids(outputs, h, w)
    if not masks_with_ids:
        return []
    metadata = _object_metadata(outputs)

    rows: list[tuple[int, int, tuple[int, int, int, int], np.ndarray]] = []
    for obj_id, mask in masks_with_ids:
        ys, xs = np.where(mask)
        if len(xs) == 0 or len(ys) == 0:
            continue
        x1 = int(xs.min())
        y1 = int(ys.min())
        x2 = int(xs.max())
        y2 = int(ys.max())
        bw = max(1, x2 - x1 + 1)
        bh = max(1, y2 - y1 + 1)
        pad_x = int(round(crop_context_ratio * bw))
        pad_y = int(round(crop_context_ratio * bh))
        cx1 = max(0, x1 - pad_x)
        cy1 = max(0, y1 - pad_y)
        cx2 = min(w - 1, x2 + pad_x)
        cy2 = min(h - 1, y2 + pad_y)
        area = int(mask.sum())
        rows.append((area, int(obj_id), (cx1, cy1, cx2, cy2), mask))

    rows.sort(key=lambda x: x[0], reverse=True)
    if max_object_crops > 0:
        rows = rows[: int(max_object_crops)]

    tiles: list[tuple[str, np.ndarray]] = []
    for _area, obj_id, (cx1, cy1, cx2, cy2), full_mask in rows:
        crop = frame_bgr[cy1 : cy2 + 1, cx1 : cx2 + 1].copy()
        mask_crop = full_mask[cy1 : cy2 + 1, cx1 : cx2 + 1]
        color = (
            int((obj_id * 47) % 255),
            int((obj_id * 89 + 37) % 255),
            int((obj_id * 131 + 73) % 255),
        )
        overlay = np.zeros_like(crop)
        overlay[mask_crop] = color
        crop = cv2.addWeighted(crop, 1.0, overlay, overlay_alpha, 0.0)
        mask_u8 = (mask_crop.astype(np.uint8) * 255)
        contours, _ = cv2.findContours(mask_u8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(crop, contours, -1, color, 2)
        conf = metadata.get(obj_id, {}).get("confidence")
        label = f"obj {obj_id}"
        if isinstance(conf, (float, int)):
            label += f" p {float(conf):.1f}"
        tiles.append((label, crop))
    return tiles


def _sanitize_assessment(
    parsed: dict[str, Any] | None,
    *,
    frame_index: int,
    present_obj_ids: list[int],
) -> dict[str, Any]:
    allowed_ids = set(int(i) for i in present_obj_ids)
    out = {
        "frame_index": int(frame_index),
        "frame_validity": "valid",
        "reasons": [],
        "bad_object_ids": [],
        "overlap_pairs": [],
        "missing_creatures": False,
        "missing_creature_reason": "",
        "missing_creature_regions": [],
        "needs_rerun": False,
        "confidence": 0.0,
    }
    if not isinstance(parsed, dict):
        return out

    validity = str(parsed.get("frame_validity", "valid")).strip().lower()
    if validity in {"valid", "invalid"}:
        out["frame_validity"] = validity

    raw_reasons = parsed.get("reasons", [])
    if isinstance(raw_reasons, list):
        out["reasons"] = [str(x).strip() for x in raw_reasons if str(x).strip()]
    elif isinstance(raw_reasons, str) and raw_reasons.strip():
        out["reasons"] = [raw_reasons.strip()]

    raw_bad = parsed.get("bad_object_ids", [])
    if isinstance(raw_bad, list):
        filtered = []
        for x in raw_bad:
            try:
                xi = int(x)
            except Exception:
                continue
            if xi in allowed_ids:
                filtered.append(xi)
        out["bad_object_ids"] = sorted(set(filtered))

    raw_pairs = parsed.get("overlap_pairs", [])
    if isinstance(raw_pairs, list):
        safe_pairs: list[list[int]] = []
        for pair in raw_pairs:
            if not isinstance(pair, (list, tuple)) or len(pair) < 2:
                continue
            try:
                a = int(pair[0])
                b = int(pair[1])
            except Exception:
                continue
            if a in allowed_ids and b in allowed_ids and a != b:
                safe_pairs.append([min(a, b), max(a, b)])
        out["overlap_pairs"] = sorted([list(x) for x in {tuple(p) for p in safe_pairs}])

    mc = parsed.get("missing_creatures", False)
    if isinstance(mc, bool):
        out["missing_creatures"] = bool(mc)
    elif isinstance(mc, (int, float)):
        out["missing_creatures"] = bool(mc)

    mreason = str(parsed.get("missing_creature_reason", "")).strip()
    if mreason:
        out["missing_creature_reason"] = mreason

    raw_regions = parsed.get("missing_creature_regions", [])
    if isinstance(raw_regions, list):
        regions: list[dict[str, int]] = []
        for r in raw_regions:
            if not isinstance(r, dict):
                continue
            try:
                x = int(r.get("x", 0))
                y = int(r.get("y", 0))
                w = int(r.get("w", 0))
                h = int(r.get("h", 0))
            except Exception:
                continue
            if w > 0 and h > 0:
                regions.append({"x": x, "y": y, "w": w, "h": h})
        out["missing_creature_regions"] = regions

    nr = parsed.get("needs_rerun", False)
    if isinstance(nr, bool):
        out["needs_rerun"] = bool(nr)
    elif isinstance(nr, (int, float)):
        out["needs_rerun"] = bool(nr)

    try:
        conf = float(parsed.get("confidence", 0.0))
    except Exception:
        conf = 0.0
    out["confidence"] = float(max(0.0, min(1.0, conf)))
    return out


def discover_postprop_qa_with_mllm(
    *,
    video_path: str,
    send_generate_request_fn: Callable[[list[dict[str, Any]]], str | None],
    initial_text_prompt: str,
    results_by_frame: dict[int, dict[str, Any]],
    total_frames: int,
    output_dir: str,
    window_size: int = 10,
    window_stride: int = 1,
    max_object_crops: int = 8,
    crop_context_ratio: float = 0.25,
    collage_cols: int = 4,
    collage_tile_max_edge: int = 320,
    prompt_profile: str = "underwater",
    prompt_template_path: str | None = None,
    max_json_retries: int = 2,
    overlap_iou_threshold: float = 0.70,
    include_frames_without_outputs: bool = True,
) -> dict[str, Any]:
    os.makedirs(output_dir, exist_ok=True)
    qa_collage_dir = os.path.join(output_dir, "frame_collages")
    os.makedirs(qa_collage_dir, exist_ok=True)

    system_prompt = _resolve_system_prompt(
        prompt_profile=prompt_profile,
        prompt_template_path=prompt_template_path,
    )

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video for post-propagation QA: {video_path}")

    context_window: deque[tuple[int, np.ndarray]] = deque(maxlen=max(1, int(window_size)))
    per_frame_assessments: list[dict[str, Any]] = []
    bad_frame_indices: list[int] = []
    request_log: list[dict[str, Any]] = []

    frame_index = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if frame_index >= int(total_frames):
            break

        context_window.append((int(frame_index), frame.copy()))
        if window_stride > 1 and (frame_index % int(window_stride)) != 0:
            frame_index += 1
            continue

        outputs = results_by_frame.get(int(frame_index), {})
        if (not include_frames_without_outputs) and (not outputs):
            frame_index += 1
            continue

        frame_h, frame_w = frame.shape[:2]
        masks_with_ids = _iter_output_masks_with_ids(outputs, frame_h, frame_w)
        present_obj_ids = sorted(int(x[0]) for x in masks_with_ids)
        heuristic_overlap_pairs = _heuristic_overlap_pairs(
            masks_with_ids, iou_threshold=float(overlap_iou_threshold)
        )

        current_overlay = _overlay_masks_with_boxes(frame, outputs)
        crop_tiles = _extract_object_crops(
            frame,
            outputs,
            max_object_crops=max_object_crops,
            crop_context_ratio=float(crop_context_ratio),
        )

        tiles: list[tuple[str, np.ndarray]] = []
        for ctx_idx, ctx_frame in list(context_window):
            tiles.append((f"ctx f={ctx_idx}", ctx_frame))
        tiles.append((f"raw f={frame_index}", frame))
        tiles.append((f"overlay f={frame_index}", current_overlay))
        tiles.extend(crop_tiles)

        collage_path = os.path.join(qa_collage_dir, f"qa_frame_{frame_index:05d}.jpg")
        rows, cols = _build_labeled_collage(
            tiles,
            collage_path,
            cols=collage_cols,
            tile_max_edge=collage_tile_max_edge,
        )

        frame_list_text = ", ".join(str(x[0]) for x in list(context_window))
        instruction = (
            "You are reviewing segmentation QA quality.\n"
            f"Collage is row-major ({rows} rows x {cols} cols).\n"
            f"Current frame index: {frame_index}. Temporal context frames: [{frame_list_text}].\n"
            f"Present tracked object IDs in current frame: {present_obj_ids}.\n"
            f"Initial prompt context: '{initial_text_prompt}'.\n"
            "Decide if this frame is usable for high-quality training masks.\n"
            "Mark overlaps/merges, bad masks, and likely missed creatures.\n"
            "Return STRICT JSON ONLY with exact schema:\n"
            "{"
            '"frame_index":int,'
            '"frame_validity":"valid"|"invalid",'
            '"reasons":[str],'
            '"bad_object_ids":[int],'
            '"overlap_pairs":[[int,int]],'
            '"missing_creatures":bool,'
            '"missing_creature_reason":str,'
            '"missing_creature_regions":[{"x":int,"y":int,"w":int,"h":int}],'
            '"needs_rerun":bool,'
            '"confidence":float'
            "}\n"
            "Use only IDs listed above in bad_object_ids / overlap_pairs."
        )

        messages = [
            {"role": "system", "content": system_prompt},
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": collage_path},
                    {"type": "text", "text": instruction},
                ],
            },
        ]

        model_text: str | None = None
        parsed: dict[str, Any] | None = None
        max_attempts = max(1, int(max_json_retries) + 1)
        for attempt in range(max_attempts):
            model_text = send_generate_request_fn(messages)
            parsed = _extract_json_object(model_text or "")
            if isinstance(parsed, dict):
                break
            if attempt + 1 < max_attempts:
                messages.append(
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "text",
                                "text": (
                                    "Your last reply was not valid JSON. "
                                    "Reply again with STRICT JSON only and no extra text."
                                ),
                            }
                        ],
                    }
                )

        assessment = _sanitize_assessment(
            parsed,
            frame_index=int(frame_index),
            present_obj_ids=present_obj_ids,
        )

        # Fold in deterministic overlap evidence.
        if heuristic_overlap_pairs:
            existing = {tuple(p) for p in assessment.get("overlap_pairs", [])}
            for pair in heuristic_overlap_pairs:
                existing.add(tuple(pair))
            assessment["overlap_pairs"] = [list(x) for x in sorted(existing)]
            if not assessment["reasons"]:
                assessment["reasons"] = ["heuristic_overlap_detected"]

        # Heuristic fallback if JSON never parsed.
        if not isinstance(parsed, dict):
            q = analyze_frame_quality(frame)
            if bool(q.get("is_invalid", False)):
                assessment["frame_validity"] = "invalid"
                if not assessment["reasons"]:
                    assessment["reasons"] = list(q.get("invalid_reasons", []))
            assessment["confidence"] = 0.0
            assessment["source"] = "heuristic_fallback"
        else:
            assessment["source"] = "mllm"

        is_bad = (
            assessment["frame_validity"] == "invalid"
            or bool(assessment.get("bad_object_ids"))
            or bool(assessment.get("overlap_pairs"))
            or bool(assessment.get("missing_creatures"))
            or bool(assessment.get("needs_rerun"))
        )
        assessment["is_bad_frame"] = bool(is_bad)

        per_frame_assessments.append(assessment)
        if is_bad:
            bad_frame_indices.append(int(frame_index))

        request_log.append(
            {
                "frame_index": int(frame_index),
                "collage_path": collage_path,
                "present_obj_ids": present_obj_ids,
                "heuristic_overlap_pairs": heuristic_overlap_pairs,
                "raw_text": model_text,
                "parsed_json_ok": isinstance(parsed, dict),
            }
        )
        frame_index += 1

    cap.release()

    bad_frame_indices = sorted(set(int(i) for i in bad_frame_indices))
    bad_ratio = (
        float(len(bad_frame_indices)) / float(total_frames) if total_frames > 0 else 0.0
    )

    return {
        "mode": "postprop_qa_mllm",
        "total_frames": int(total_frames),
        "window_size": int(window_size),
        "window_stride": int(window_stride),
        "max_object_crops": int(max_object_crops),
        "crop_context_ratio": float(crop_context_ratio),
        "collage_cols": int(collage_cols),
        "collage_tile_max_edge": int(collage_tile_max_edge),
        "max_json_retries": int(max_json_retries),
        "overlap_iou_threshold": float(overlap_iou_threshold),
        "include_frames_without_outputs": bool(include_frames_without_outputs),
        "num_frames_assessed": len(per_frame_assessments),
        "bad_frame_indices": bad_frame_indices,
        "bad_frame_ratio": float(bad_ratio),
        "num_bad_frames": len(bad_frame_indices),
        "per_frame_assessments": per_frame_assessments,
        "requests": request_log,
    }
