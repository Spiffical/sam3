from __future__ import annotations

"""Deprecated experiment kept for reference; do not enable in production."""

import json
import math
import os
from typing import Any, Callable

import cv2
import numpy as np

from frame_output_utils import (
    encode_binary_mask_to_rle,
    frame_object_metadata,
    iter_output_masks_with_ids,
    merge_frame_outputs_by_obj_ids,
    read_video_frame,
)
from postprop_qa_mllm import (
    _build_labeled_collage,
    _draw_tile_label,
    _extract_json_object,
    _overlay_masks_with_boxes,
)


def _repo_root() -> str:
    return os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))


def _read_prompt_file(path: str) -> str:
    with open(path, "r", encoding="utf-8") as handle:
        return handle.read().strip()


def _default_detection_prompt_path(prompt_profile: str) -> str:
    base = os.path.join(_repo_root(), "sam3", "agent", "system_prompts")
    profile = str(prompt_profile or "").strip().lower()
    if profile == "underwater":
        return os.path.join(base, "system_prompt_missed_creature_discovery_underwater.txt")
    return os.path.join(base, "system_prompt_missed_creature_discovery_general.txt")


def _default_verify_prompt_path(prompt_profile: str) -> str:
    base = os.path.join(_repo_root(), "sam3", "agent", "system_prompts")
    profile = str(prompt_profile or "").strip().lower()
    if profile == "underwater":
        return os.path.join(base, "system_prompt_missed_creature_verify_underwater.txt")
    return os.path.join(base, "system_prompt_missed_creature_verify_general.txt")


def _resolve_prompt(
    *,
    prompt_path: str | None,
    default_path: str,
    fallback_text: str,
) -> str:
    if prompt_path:
        if not os.path.exists(prompt_path):
            raise FileNotFoundError(f"Prompt template not found: {prompt_path}")
        return _read_prompt_file(prompt_path)
    if os.path.exists(default_path):
        return _read_prompt_file(default_path)
    return fallback_text


def _write_json(path: str, payload: dict[str, Any]) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)


def _normalize_point_xy(
    point: Any,
    *,
    frame_w: int,
    frame_h: int,
) -> tuple[int, int] | None:
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


def _dedupe_points(
    points: list[tuple[int, int]],
    *,
    min_distance: float = 4.0,
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


def _sanitize_xyxy(
    box_xyxy: tuple[int, int, int, int] | list[int],
    *,
    frame_w: int,
    frame_h: int,
) -> tuple[int, int, int, int]:
    x1, y1, x2, y2 = [int(v) for v in box_xyxy]
    if x1 > x2:
        x1, x2 = x2, x1
    if y1 > y2:
        y1, y2 = y2, y1
    x1 = max(0, min(int(frame_w) - 1, x1))
    y1 = max(0, min(int(frame_h) - 1, y1))
    x2 = max(0, min(int(frame_w) - 1, x2))
    y2 = max(0, min(int(frame_h) - 1, y2))
    return (x1, y1, max(x1, x2), max(y1, y2))


def _expand_xyxy(
    box_xyxy: tuple[int, int, int, int],
    *,
    frame_w: int,
    frame_h: int,
    pad_ratio: float = 0.35,
    min_pad_px: int = 18,
) -> tuple[int, int, int, int]:
    x1, y1, x2, y2 = _sanitize_xyxy(box_xyxy, frame_w=frame_w, frame_h=frame_h)
    bw = max(1, x2 - x1 + 1)
    bh = max(1, y2 - y1 + 1)
    pad_x = max(int(min_pad_px), int(round(bw * float(pad_ratio))))
    pad_y = max(int(min_pad_px), int(round(bh * float(pad_ratio))))
    return _sanitize_xyxy(
        (x1 - pad_x, y1 - pad_y, x2 + pad_x, y2 + pad_y),
        frame_w=frame_w,
        frame_h=frame_h,
    )


def _mask_bbox_xyxy(mask: np.ndarray | None) -> tuple[int, int, int, int] | None:
    if mask is None:
        return None
    ys, xs = np.where(np.asarray(mask).astype(bool))
    if len(xs) == 0 or len(ys) == 0:
        return None
    return (int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max()))


def _mask_centroid(mask: np.ndarray | None) -> tuple[int, int] | None:
    if mask is None:
        return None
    ys, xs = np.where(np.asarray(mask).astype(bool))
    if len(xs) == 0 or len(ys) == 0:
        return None
    return (int(round(xs.mean())), int(round(ys.mean())))


def _mask_iou(mask_a: np.ndarray, mask_b: np.ndarray) -> float:
    a = np.asarray(mask_a).astype(bool)
    b = np.asarray(mask_b).astype(bool)
    inter = float(np.logical_and(a, b).sum())
    if inter <= 0.0:
        return 0.0
    union = float(np.logical_or(a, b).sum())
    if union <= 0.0:
        return 0.0
    return inter / union


def _blank_tile(shape_hw: tuple[int, int], label: str) -> np.ndarray:
    h, w = [max(1, int(v)) for v in shape_hw]
    tile = np.zeros((h, w, 3), dtype=np.uint8)
    cv2.putText(
        tile,
        "No candidate mask",
        (12, max(20, h // 2)),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )
    return _draw_tile_label(tile, label)


def _draw_mask_focus(
    frame_bgr: np.ndarray,
    *,
    focus_mask: np.ndarray | None,
    existing_masks: list[np.ndarray] | None = None,
    label: str | None = None,
    crop_xyxy: tuple[int, int, int, int] | None = None,
) -> np.ndarray:
    output = frame_bgr.copy()
    overlay = np.zeros_like(output)

    for existing_mask in existing_masks or []:
        mask_bool = np.asarray(existing_mask).astype(bool)
        overlay[mask_bool] = (70, 70, 70)

    if focus_mask is not None and np.asarray(focus_mask).any():
        focus_bool = np.asarray(focus_mask).astype(bool)
        overlay[focus_bool] = (0, 200, 255)
        contours, _ = cv2.findContours(
            (focus_bool.astype(np.uint8) * 255),
            cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_SIMPLE,
        )
        cv2.drawContours(output, contours, -1, (0, 200, 255), 2)

    output = cv2.addWeighted(output, 1.0, overlay, 0.35, 0.0)
    if crop_xyxy is not None:
        x1, y1, x2, y2 = crop_xyxy
        output = output[y1 : y2 + 1, x1 : x2 + 1].copy()
    if label:
        output = _draw_tile_label(output, label)
    return output


def _render_point_prompt_debug(
    frame_bgr: np.ndarray,
    *,
    positive_points: list[tuple[int, int]],
    current_outputs: dict[str, Any],
) -> np.ndarray:
    output = _overlay_masks_with_boxes(frame_bgr, current_outputs)
    for idx, (x, y) in enumerate(positive_points, start=1):
        cv2.circle(output, (int(x), int(y)), 7, (60, 220, 60), -1)
        cv2.circle(output, (int(x), int(y)), 10, (255, 255, 255), 1)
        cv2.putText(
            output,
            f"+{idx}",
            (int(x) + 8, int(y) - 6),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.48,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )
        cv2.putText(
            output,
            f"+{idx}",
            (int(x) + 8, int(y) - 6),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.48,
            (0, 0, 0),
            1,
            cv2.LINE_AA,
        )
    return output


def _comparison_tile(
    frame_bgr: np.ndarray,
    *,
    outputs: dict[str, Any],
    frame_index: int,
) -> np.ndarray:
    raw = _draw_tile_label(frame_bgr.copy(), f"raw f={int(frame_index)}")
    overlay = _draw_tile_label(
        _overlay_masks_with_boxes(frame_bgr, outputs),
        f"overlay f={int(frame_index)}",
    )
    h = max(raw.shape[0], overlay.shape[0])
    w = raw.shape[1] + overlay.shape[1] + 8
    tile = np.zeros((h, w, 3), dtype=np.uint8)
    tile[: raw.shape[0], : raw.shape[1]] = raw
    tile[: overlay.shape[0], raw.shape[1] + 8 : raw.shape[1] + 8 + overlay.shape[1]] = overlay
    return tile


def _candidate_items_from_outputs(
    outputs: dict[str, Any],
    *,
    frame_h: int,
    frame_w: int,
) -> list[dict[str, Any]]:
    metadata_by_obj = frame_object_metadata(outputs, frame_h, frame_w)
    items: list[dict[str, Any]] = []
    for obj_id, mask in iter_output_masks_with_ids(outputs, frame_h, frame_w):
        mask_bool = np.asarray(mask).astype(bool)
        bbox_xyxy = _mask_bbox_xyxy(mask_bool)
        centroid = _mask_centroid(mask_bool)
        meta = metadata_by_obj.get(int(obj_id), {})
        items.append(
            {
                "obj_id": int(obj_id),
                "mask": mask_bool,
                "mask_rle": encode_binary_mask_to_rle(mask_bool),
                "bbox_xyxy": bbox_xyxy,
                "centroid": centroid,
                "area": int(mask_bool.sum()),
                "score": float(meta.get("confidence") or 0.0),
            }
        )
    return items


def _select_candidate_item(
    outputs: dict[str, Any],
    *,
    requested_obj_id: int,
    frame_h: int,
    frame_w: int,
) -> dict[str, Any] | None:
    items = _candidate_items_from_outputs(outputs, frame_h=frame_h, frame_w=frame_w)
    if not items:
        return None
    for item in items:
        if int(item["obj_id"]) == int(requested_obj_id):
            return item
    items.sort(
        key=lambda item: (float(item.get("score") or 0.0), float(item.get("area") or 0.0)),
        reverse=True,
    )
    return items[0]


def _unwrap_backend_outputs(response: Any) -> dict[str, Any]:
    if not isinstance(response, dict):
        return {}
    nested = response.get("outputs")
    if isinstance(nested, dict):
        return nested
    return response


def _auto_max_images_per_request(configured_value: int) -> int:
    env_value = os.environ.get("SAM3_MAX_IMAGES_PER_REQUEST")
    if env_value:
        try:
            return max(1, min(int(configured_value), int(env_value)))
        except Exception:
            pass
    return max(1, int(configured_value))


def _window_frame_indices(
    *,
    start_index: int,
    total_frames: int,
    window_size: int,
    invalid_frame_indices: set[int],
) -> list[int]:
    end_index = min(int(total_frames), int(start_index) + max(1, int(window_size)))
    return [
        frame_index
        for frame_index in range(int(start_index), end_index)
        if int(frame_index) not in invalid_frame_indices
    ]


def _write_detection_images(
    *,
    video_path: str,
    results_by_frame: dict[int, dict[str, Any]],
    frame_indices: list[int],
    output_dir: str,
    max_images_per_request: int,
) -> tuple[list[str], list[list[int]], list[str]]:
    os.makedirs(output_dir, exist_ok=True)
    per_frame_tiles: list[tuple[int, np.ndarray, str]] = []
    debug_paths: list[str] = []

    for frame_index in frame_indices:
        frame_bgr = read_video_frame(video_path, frame_index)
        if frame_bgr is None:
            continue
        tile = _comparison_tile(
            frame_bgr,
            outputs=results_by_frame.get(int(frame_index), {}),
            frame_index=int(frame_index),
        )
        tile_path = os.path.join(output_dir, f"frame_compare_{int(frame_index):05d}.jpg")
        cv2.imwrite(tile_path, tile)
        per_frame_tiles.append((int(frame_index), tile, tile_path))
        debug_paths.append(tile_path)

    if not per_frame_tiles:
        return [], [], []

    image_budget = max(1, int(max_images_per_request))
    if len(per_frame_tiles) <= image_budget:
        return (
            [row[2] for row in per_frame_tiles],
            [[int(row[0])] for row in per_frame_tiles],
            debug_paths,
        )

    request_image_paths: list[str] = []
    request_frame_groups: list[list[int]] = []
    group_count = min(image_budget, len(per_frame_tiles))
    group_size = int(math.ceil(len(per_frame_tiles) / float(group_count)))
    for group_index in range(group_count):
        chunk = per_frame_tiles[group_index * group_size : (group_index + 1) * group_size]
        if not chunk:
            continue
        collage_path = os.path.join(output_dir, f"window_chunk_{group_index:02d}.jpg")
        _build_labeled_collage(
            [(f"f={int(frame_index)}", tile) for frame_index, tile, _ in chunk],
            collage_path,
            cols=min(2, max(1, len(chunk))),
            tile_max_edge=420,
        )
        request_image_paths.append(collage_path)
        request_frame_groups.append([int(frame_index) for frame_index, _, _ in chunk])
        debug_paths.append(collage_path)

    return request_image_paths, request_frame_groups, debug_paths


def _sanitize_detection_response(
    parsed: dict[str, Any] | None,
    *,
    allowed_frame_indices: set[int],
    frame_w: int,
    frame_h: int,
    max_issues: int,
) -> list[dict[str, Any]]:
    if not isinstance(parsed, dict):
        return []

    raw_issues = parsed.get("issues", [])
    if not isinstance(raw_issues, list):
        return []

    issues: list[dict[str, Any]] = []
    seen: set[tuple[int, tuple[tuple[int, int], ...]]] = set()
    for raw_issue in raw_issues:
        if not isinstance(raw_issue, dict):
            continue
        try:
            target_frame_index = int(raw_issue.get("target_frame_index"))
        except Exception:
            continue
        if target_frame_index not in allowed_frame_indices:
            continue

        raw_click_points = raw_issue.get("click_points")
        if raw_click_points is None:
            raw_click_points = raw_issue.get("clicks")
        if raw_click_points is None and isinstance(raw_issue.get("click_point"), (list, tuple)):
            raw_click_points = [raw_issue.get("click_point")]
        if raw_click_points is None and isinstance(raw_issue.get("point"), (list, tuple)):
            raw_click_points = [raw_issue.get("point")]
        if not isinstance(raw_click_points, list):
            continue

        click_points = _dedupe_points(
            [
                point
                for point in (
                    _normalize_point_xy(
                        raw_point,
                        frame_w=frame_w,
                        frame_h=frame_h,
                    )
                    for raw_point in raw_click_points
                )
                if point is not None
            ],
            min_distance=4.0,
        )
        if not click_points:
            continue

        try:
            confidence = float(raw_issue.get("confidence", 0.0))
        except Exception:
            confidence = 0.0
        confidence = max(0.0, min(1.0, confidence))
        description = str(raw_issue.get("description", "")).strip()

        issue_key = (
            int(target_frame_index),
            tuple(sorted((int(point[0]), int(point[1])) for point in click_points)),
        )
        if issue_key in seen:
            continue
        seen.add(issue_key)
        issues.append(
            {
                "target_frame_index": int(target_frame_index),
                "description": description,
                "confidence": confidence,
                "click_points": [[int(point[0]), int(point[1])] for point in click_points[:3]],
            }
        )
        if len(issues) >= max(1, int(max_issues)):
            break
    return issues


def _request_detection_issues(
    *,
    send_generate_request_fn: Callable[[list[dict[str, Any]]], str | None],
    system_prompt: str,
    image_paths: list[str],
    frame_groups: list[list[int]],
    window_frame_indices: list[int],
    max_issues: int,
    max_json_retries: int,
) -> tuple[dict[str, Any] | None, str | None]:
    image_descriptions = "\n".join(
        f"- image {index + 1}: frames {group}"
        for index, group in enumerate(frame_groups)
    )
    messages = [
        {"role": "system", "content": system_prompt},
        {
            "role": "user",
            "content": (
                [{"type": "image", "image": image_path} for image_path in image_paths]
                + [
                    {
                        "type": "text",
                        "text": (
                            f"Chronological window frames: {window_frame_indices}\n"
                            f"{image_descriptions}\n\n"
                            "Each image shows one or more labeled frame tiles. "
                            "Inside each tile, the left half is the raw frame and the right half is the current segmentation overlay. "
                            "A creature counts as missed only if it is clearly visible in the raw half and not already covered by a mask in the overlay half.\n\n"
                            f"Find at most {int(max_issues)} clear missed-creature cases in this window. "
                            "For each case, choose the single best frame for clicking and provide 1 to 3 positive click points [x,y] on the missed creature in original-frame pixel coordinates. "
                            "Do not report duplicates, already segmented animals, marine snow, debris, shadows, or ambiguous blobs. "
                            "Return strict JSON only."
                        ),
                    }
                ]
            ),
        },
    ]

    last_text: str | None = None
    for attempt in range(max(1, int(max_json_retries) + 1)):
        last_text = send_generate_request_fn(messages)
        parsed = _extract_json_object(last_text or "")
        if isinstance(parsed, dict):
            return parsed, last_text
        if attempt + 1 < max(1, int(max_json_retries) + 1):
            messages.append(
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": (
                                "Your previous reply did not contain valid JSON. "
                                "Reply again with strict JSON only using schema "
                                '{"issues":[{"target_frame_index":int,"description":str,"click_points":[[x,y]],"confidence":float}]}.'
                            ),
                        }
                    ],
                }
            )
    return None, last_text


def _build_candidate_crop_xyxy(
    *,
    candidate_item: dict[str, Any] | None,
    positive_points: list[tuple[int, int]],
    frame_w: int,
    frame_h: int,
) -> tuple[int, int, int, int]:
    bbox_xyxy = candidate_item.get("bbox_xyxy") if candidate_item else None
    if bbox_xyxy is None and positive_points:
        xs = [int(point[0]) for point in positive_points]
        ys = [int(point[1]) for point in positive_points]
        bbox_xyxy = (
            min(xs),
            min(ys),
            max(xs),
            max(ys),
        )
    if bbox_xyxy is None:
        return (0, 0, int(frame_w) - 1, int(frame_h) - 1)
    return _expand_xyxy(
        bbox_xyxy,
        frame_w=frame_w,
        frame_h=frame_h,
        pad_ratio=0.6,
        min_pad_px=20,
    )


def _build_attempt_history_text(attempts: list[dict[str, Any]]) -> str:
    if not attempts:
        return "No previous attempts."
    lines = []
    for attempt in attempts[-4:]:
        lines.append(
            f"attempt {int(attempt.get('attempt_index', 0))}: "
            f"points={attempt.get('positive_points', [])}, "
            f"decision={attempt.get('decision', 'unknown')}, "
            f"reason={attempt.get('reason', '')}"
        )
    return "\n".join(lines)


def _build_verification_collage(
    *,
    video_path: str,
    results_by_frame: dict[int, dict[str, Any]],
    frame_index: int,
    frame_bgr: np.ndarray,
    current_outputs: dict[str, Any],
    positive_points: list[tuple[int, int]],
    candidate_item: dict[str, Any] | None,
    output_path: str,
) -> tuple[str, tuple[int, int, int, int]]:
    frame_h, frame_w = frame_bgr.shape[:2]
    existing_items = _candidate_items_from_outputs(
        current_outputs,
        frame_h=frame_h,
        frame_w=frame_w,
    )
    existing_masks = [item["mask"] for item in existing_items]
    crop_xyxy = _build_candidate_crop_xyxy(
        candidate_item=candidate_item,
        positive_points=positive_points,
        frame_w=frame_w,
        frame_h=frame_h,
    )
    x1, y1, x2, y2 = crop_xyxy

    prompt_overlay = _render_point_prompt_debug(
        frame_bgr,
        positive_points=positive_points,
        current_outputs=current_outputs,
    )
    candidate_overlay = _draw_mask_focus(
        frame_bgr,
        focus_mask=(candidate_item.get("mask") if candidate_item else None),
        existing_masks=existing_masks,
        label=None,
    )
    raw_crop = frame_bgr[y1 : y2 + 1, x1 : x2 + 1].copy()
    current_crop = _overlay_masks_with_boxes(frame_bgr, current_outputs)[y1 : y2 + 1, x1 : x2 + 1].copy()
    prompt_crop = prompt_overlay[y1 : y2 + 1, x1 : x2 + 1].copy()
    candidate_crop = (
        candidate_overlay[y1 : y2 + 1, x1 : x2 + 1].copy()
        if candidate_item is not None
        else _blank_tile((raw_crop.shape[0], raw_crop.shape[1]), "candidate crop")
    )

    tiles: list[tuple[str, np.ndarray]] = [
        ("raw target", frame_bgr),
        ("current overlay", _overlay_masks_with_boxes(frame_bgr, current_outputs)),
        ("prompt overlay", prompt_overlay),
        ("raw crop", raw_crop),
        ("current crop", current_crop),
        ("candidate crop", candidate_crop),
        ("prompt crop", prompt_crop),
    ]

    for ctx_index in range(max(0, int(frame_index) - 2), int(frame_index) + 3):
        if ctx_index == int(frame_index):
            continue
        ctx_frame = read_video_frame(video_path, ctx_index)
        if ctx_frame is None:
            continue
        tiles.append(
            (
                f"context f={ctx_index}",
                _comparison_tile(
                    ctx_frame,
                    outputs=results_by_frame.get(int(ctx_index), {}),
                    frame_index=int(ctx_index),
                ),
            )
        )

    _build_labeled_collage(
        tiles,
        output_path,
        cols=3,
        tile_max_edge=360,
    )
    return output_path, crop_xyxy


def _sanitize_verification_response(
    parsed: dict[str, Any] | None,
    *,
    frame_w: int,
    frame_h: int,
) -> dict[str, Any]:
    decision = "reject"
    reason = ""
    confidence = 0.0
    additional_points: list[list[int]] = []

    if isinstance(parsed, dict):
        decision_raw = str(parsed.get("decision", "")).strip().lower()
        if decision_raw in {"accept", "retry", "reject"}:
            decision = decision_raw
        reason = str(parsed.get("reason", "")).strip()
        try:
            confidence = float(parsed.get("confidence", 0.0))
        except Exception:
            confidence = 0.0
        confidence = max(0.0, min(1.0, confidence))

        raw_points = parsed.get("additional_points")
        if raw_points is None and isinstance(parsed.get("suggested_point"), (list, tuple)):
            raw_points = [parsed.get("suggested_point")]
        if raw_points is None and isinstance(parsed.get("suggested_points"), list):
            raw_points = parsed.get("suggested_points")
        if isinstance(raw_points, list):
            additional_points = [
                [int(point[0]), int(point[1])]
                for point in _dedupe_points(
                    [
                        point
                        for point in (
                            _normalize_point_xy(
                                raw_point,
                                frame_w=frame_w,
                                frame_h=frame_h,
                            )
                            for raw_point in raw_points
                        )
                        if point is not None
                    ],
                    min_distance=4.0,
                )[:2]
            ]

    return {
        "decision": decision,
        "reason": reason,
        "confidence": confidence,
        "additional_points": additional_points,
    }


def _request_candidate_verdict(
    *,
    send_generate_request_fn: Callable[[list[dict[str, Any]]], str | None],
    system_prompt: str,
    collage_path: str,
    target_frame_index: int,
    issue_description: str,
    positive_points: list[tuple[int, int]],
    max_existing_iou: float,
    attempt_history_text: str,
    candidate_present: bool,
    max_json_retries: int,
    frame_w: int,
    frame_h: int,
) -> tuple[dict[str, Any], str | None]:
    messages = [
        {"role": "system", "content": system_prompt},
        {
            "role": "user",
            "content": [
                {"type": "image", "image": collage_path},
                {
                    "type": "text",
                    "text": (
                        f"Target frame: {int(target_frame_index)}\n"
                        f"Current positive clicks: {[list(point) for point in positive_points]}\n"
                        f"Issue description: {issue_description or '(none)'}\n"
                        f"Max IoU with any already-present mask on the target frame: {float(max_existing_iou):.3f}\n"
                        f"Candidate mask present: {bool(candidate_present)}\n\n"
                        "The collage includes the raw target frame, the current segmentation overlay, the click prompts, zoomed crops, and nearby temporal context. "
                        "Accept only if the candidate clearly adds a real missed creature that is not already segmented. "
                        "If the creature is visible but the current candidate is incomplete or wrong, return retry and provide 1 or 2 additional positive click points in original-frame pixel coordinates. "
                        "Keep all existing clicks; any points you return are extra clicks to add, not replacements.\n\n"
                        f"Previous attempts:\n{attempt_history_text}\n\n"
                        "Return strict JSON only."
                    ),
                },
            ],
        },
    ]

    last_text: str | None = None
    parsed: dict[str, Any] | None = None
    for attempt in range(max(1, int(max_json_retries) + 1)):
        last_text = send_generate_request_fn(messages)
        parsed = _extract_json_object(last_text or "")
        if isinstance(parsed, dict):
            break
        if attempt + 1 < max(1, int(max_json_retries) + 1):
            messages.append(
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": (
                                "Your previous reply did not contain valid JSON. "
                                "Reply again with strict JSON only using schema "
                                '{"decision":"accept|retry|reject","reason":str,"additional_points":[[x,y]],"confidence":float}.'
                            ),
                        }
                    ],
                }
            )

    return (
        _sanitize_verification_response(parsed, frame_w=frame_w, frame_h=frame_h),
        last_text,
    )


def _propagate_accepted_mask(
    *,
    backend: Any,
    session_id: str,
    results_by_frame: dict[int, dict[str, Any]],
    frame_index: int,
    obj_id: int,
    mask: np.ndarray,
) -> list[int]:
    backend.reset_session(session_id)
    backend.add_mask_prompt(
        session_id=session_id,
        frame_idx=int(frame_index),
        obj_id=int(obj_id),
        mask=np.asarray(mask).astype(bool),
    )
    updated_frames: dict[int, dict[str, Any]] = {}
    request = {
        "session_id": session_id,
        "type": "propagate_in_video",
        "start_frame_index": int(frame_index),
        "propagation_direction": "both",
    }
    for output in backend.propagate(request):
        updated_frames[int(output["frame_index"])] = output["outputs"]

    merged_frame_indices: list[int] = []
    for updated_frame_index, patched_outputs in updated_frames.items():
        base_outputs = results_by_frame.get(updated_frame_index, {})
        results_by_frame[updated_frame_index] = merge_frame_outputs_by_obj_ids(
            base_outputs,
            patched_outputs,
            replace_obj_ids={int(obj_id)},
        )
        merged_frame_indices.append(int(updated_frame_index))
    return sorted(merged_frame_indices)


def discover_postprop_missed_creatures_with_mllm(
    *,
    video_path: str,
    backend: Any,
    send_generate_request_fn: Callable[[list[dict[str, Any]]], str | None],
    initial_text_prompt: str,
    prompt_profile: str,
    results_by_frame: dict[int, dict[str, Any]],
    total_frames: int,
    hard_invalid_frame_indices: list[int],
    next_obj_id: int,
    frame_size_hw: tuple[int, int],
    output_dir: str,
    image_size: int,
    working_session_id: str | None,
    window_size: int = 20,
    window_stride: int = 10,
    max_issues_per_window: int = 4,
    max_rounds: int = 10,
    max_attempts_per_issue: int = 10,
    max_images_per_request: int = 20,
    detection_prompt_template_path: str | None = None,
    verify_prompt_template_path: str | None = None,
    max_json_retries: int = 2,
    duplicate_iou_threshold: float = 0.80,
) -> dict[str, Any]:
    os.makedirs(output_dir, exist_ok=True)
    detection_dir = os.path.join(output_dir, "detection")
    verification_dir = os.path.join(output_dir, "verification")
    os.makedirs(detection_dir, exist_ok=True)
    os.makedirs(verification_dir, exist_ok=True)

    detection_prompt = _resolve_prompt(
        prompt_path=detection_prompt_template_path,
        default_path=_default_detection_prompt_path(prompt_profile),
        fallback_text=(
            "Review chronological video frames and find clearly visible missed creatures. "
            "Each frame tile shows raw content beside the current segmentation overlay. "
            "Return strict JSON only with schema "
            '{"issues":[{"target_frame_index":int,"description":str,"click_points":[[x,y]],"confidence":float}]}.'
        ),
    )
    verify_prompt = _resolve_prompt(
        prompt_path=verify_prompt_template_path,
        default_path=_default_verify_prompt_path(prompt_profile),
        fallback_text=(
            "Verify whether a candidate mask adds a real missed creature. "
            "Return strict JSON only with schema "
            '{"decision":"accept|retry|reject","reason":str,"additional_points":[[x,y]],"confidence":float}.'
        ),
    )

    frame_h, frame_w = int(frame_size_hw[0]), int(frame_size_hw[1])
    invalid_frame_indices = set(int(x) for x in hard_invalid_frame_indices)
    image_budget = _auto_max_images_per_request(max_images_per_request)

    session_id = working_session_id
    owns_session = False
    if not session_id:
        session_id = backend.start_session(resource_path=video_path, image_size=image_size)
        owns_session = True

    detection_requests: list[dict[str, Any]] = []
    issues: list[dict[str, Any]] = []
    accepted_prompts: list[dict[str, Any]] = []
    next_obj_id_local = int(next_obj_id)

    try:
        backend.reset_session(session_id)
        for round_index in range(1, max(1, int(max_rounds)) + 1):
            accepted_this_round = 0
            round_rows: list[dict[str, Any]] = []

            for window_start in range(0, int(total_frames), max(1, int(window_stride))):
                window_frame_indices = _window_frame_indices(
                    start_index=window_start,
                    total_frames=total_frames,
                    window_size=window_size,
                    invalid_frame_indices=invalid_frame_indices,
                )
                if not window_frame_indices:
                    continue

                window_request_dir = os.path.join(
                    detection_dir,
                    f"round_{round_index:02d}",
                    f"window_{window_start:05d}",
                )
                image_paths, frame_groups, debug_paths = _write_detection_images(
                    video_path=video_path,
                    results_by_frame=results_by_frame,
                    frame_indices=window_frame_indices,
                    output_dir=window_request_dir,
                    max_images_per_request=image_budget,
                )
                if not image_paths:
                    continue

                parsed, raw_text = _request_detection_issues(
                    send_generate_request_fn=send_generate_request_fn,
                    system_prompt=detection_prompt,
                    image_paths=image_paths,
                    frame_groups=frame_groups,
                    window_frame_indices=window_frame_indices,
                    max_issues=max_issues_per_window,
                    max_json_retries=max_json_retries,
                )
                sanitized_issues = _sanitize_detection_response(
                    parsed,
                    allowed_frame_indices=set(int(x) for x in window_frame_indices),
                    frame_w=frame_w,
                    frame_h=frame_h,
                    max_issues=max_issues_per_window,
                )
                detection_entry = {
                    "round_index": int(round_index),
                    "window_start": int(window_start),
                    "window_frame_indices": [int(x) for x in window_frame_indices],
                    "request_image_paths": image_paths,
                    "request_frame_groups": frame_groups,
                    "debug_image_paths": debug_paths,
                    "raw_response": raw_text,
                    "issue_candidates": sanitized_issues,
                }
                detection_requests.append(detection_entry)

                for issue_index, issue in enumerate(sanitized_issues):
                    target_frame_index = int(issue["target_frame_index"])
                    if target_frame_index in invalid_frame_indices:
                        continue

                    target_frame_bgr = read_video_frame(video_path, target_frame_index)
                    if target_frame_bgr is None:
                        issues.append(
                            {
                                "round_index": int(round_index),
                                "window_start": int(window_start),
                                "target_frame_index": int(target_frame_index),
                                "status": "unresolved",
                                "failure_reason": "target_frame_unreadable",
                                "description": issue.get("description", ""),
                                "confidence": float(issue.get("confidence", 0.0)),
                                "attempts": [],
                            }
                        )
                        continue

                    issue_id = (
                        f"round{round_index:02d}_w{window_start:05d}_"
                        f"f{target_frame_index:05d}_issue{issue_index:02d}"
                    )
                    issue_dir = os.path.join(output_dir, "issues", issue_id)
                    os.makedirs(issue_dir, exist_ok=True)
                    positive_points = _dedupe_points(
                        [
                            (int(point[0]), int(point[1]))
                            for point in issue.get("click_points", [])
                        ],
                        min_distance=4.0,
                    )

                    issue_report: dict[str, Any] = {
                        "issue_id": issue_id,
                        "round_index": int(round_index),
                        "window_start": int(window_start),
                        "window_frame_indices": [int(x) for x in window_frame_indices],
                        "target_frame_index": int(target_frame_index),
                        "description": issue.get("description", ""),
                        "confidence": float(issue.get("confidence", 0.0)),
                        "initial_click_points": [[int(x), int(y)] for x, y in positive_points],
                        "status": "unresolved",
                        "attempts": [],
                    }

                    accepted = False
                    for attempt_index in range(1, max(1, int(max_attempts_per_issue)) + 1):
                        attempt_dir = os.path.join(issue_dir, f"attempt_{attempt_index:02d}")
                        os.makedirs(attempt_dir, exist_ok=True)

                        backend.reset_session(session_id)
                        response = backend.add_point_prompt(
                            session_id=session_id,
                            frame_idx=int(target_frame_index),
                            obj_id=1,
                            points=[(float(x), float(y), 1) for x, y in positive_points],
                            frame_size=(frame_w, frame_h),
                        )
                        attempt_outputs = _unwrap_backend_outputs(response)
                        candidate_item = _select_candidate_item(
                            attempt_outputs,
                            requested_obj_id=1,
                            frame_h=frame_h,
                            frame_w=frame_w,
                        )
                        current_outputs = results_by_frame.get(int(target_frame_index), {})
                        existing_items = _candidate_items_from_outputs(
                            current_outputs,
                            frame_h=frame_h,
                            frame_w=frame_w,
                        )
                        max_existing_iou = 0.0
                        duplicate_obj_id = None
                        if candidate_item is not None:
                            for existing_item in existing_items:
                                iou = _mask_iou(candidate_item["mask"], existing_item["mask"])
                                if iou > max_existing_iou:
                                    max_existing_iou = float(iou)
                                    duplicate_obj_id = int(existing_item["obj_id"])

                        verify_collage_path = os.path.join(attempt_dir, "verification_collage.jpg")
                        verify_collage_path, crop_xyxy = _build_verification_collage(
                            video_path=video_path,
                            results_by_frame=results_by_frame,
                            frame_index=int(target_frame_index),
                            frame_bgr=target_frame_bgr,
                            current_outputs=current_outputs,
                            positive_points=positive_points,
                            candidate_item=candidate_item,
                            output_path=verify_collage_path,
                        )
                        attempt_history_text = _build_attempt_history_text(issue_report["attempts"])
                        verdict, raw_verdict = _request_candidate_verdict(
                            send_generate_request_fn=send_generate_request_fn,
                            system_prompt=verify_prompt,
                            collage_path=verify_collage_path,
                            target_frame_index=int(target_frame_index),
                            issue_description=str(issue.get("description", "")),
                            positive_points=positive_points,
                            max_existing_iou=max_existing_iou,
                            attempt_history_text=attempt_history_text,
                            candidate_present=(candidate_item is not None),
                            max_json_retries=max_json_retries,
                            frame_w=frame_w,
                            frame_h=frame_h,
                        )

                        attempt_report = {
                            "attempt_index": int(attempt_index),
                            "positive_points": [[int(x), int(y)] for x, y in positive_points],
                            "candidate_present": bool(candidate_item is not None),
                            "candidate_bbox_xyxy": (
                                list(candidate_item["bbox_xyxy"])
                                if candidate_item and candidate_item.get("bbox_xyxy") is not None
                                else None
                            ),
                            "candidate_area": (
                                int(candidate_item.get("area") or 0) if candidate_item else 0
                            ),
                            "candidate_score": (
                                float(candidate_item.get("score") or 0.0)
                                if candidate_item
                                else 0.0
                            ),
                            "max_existing_iou": float(max_existing_iou),
                            "duplicate_obj_id": duplicate_obj_id,
                            "verification_collage_path": verify_collage_path,
                            "crop_xyxy": list(crop_xyxy),
                            "raw_verdict_response": raw_verdict,
                            **verdict,
                        }
                        issue_report["attempts"].append(attempt_report)

                        if candidate_item is None and verdict["decision"] == "accept":
                            verdict["decision"] = "retry"
                            attempt_report["decision"] = "retry"
                            attempt_report["reason"] = (
                                f"{attempt_report['reason']} "
                                if attempt_report["reason"]
                                else ""
                            ) + "No candidate mask was produced."

                        if (
                            candidate_item is not None
                            and max_existing_iou >= float(duplicate_iou_threshold)
                            and verdict["decision"] == "accept"
                        ):
                            verdict["decision"] = "reject"
                            attempt_report["decision"] = "reject"
                            attempt_report["reason"] = (
                                f"{attempt_report['reason']} "
                                if attempt_report["reason"]
                                else ""
                            ) + (
                                "Rejected automatically because the accepted candidate nearly duplicates "
                                f"existing object id {duplicate_obj_id}."
                            )

                        if verdict["decision"] == "accept" and candidate_item is not None:
                            accepted_obj_id = int(next_obj_id_local)
                            updated_frame_indices = _propagate_accepted_mask(
                                backend=backend,
                                session_id=session_id,
                                results_by_frame=results_by_frame,
                                frame_index=int(target_frame_index),
                                obj_id=accepted_obj_id,
                                mask=np.asarray(candidate_item["mask"]).astype(bool),
                            )
                            next_obj_id_local += 1
                            accepted_prompt = {
                                "frame_idx": int(target_frame_index),
                                "obj_id": int(accepted_obj_id),
                                "points": [
                                    [float(point[0]), float(point[1]), 1]
                                    for point in positive_points
                                ],
                                "label": 1,
                                "source": "postprop_missed_creature_clicks",
                                "description": issue.get("description", ""),
                                "round_index": int(round_index),
                                "attempt_index": int(attempt_index),
                            }
                            accepted_prompts.append(accepted_prompt)
                            issue_report["status"] = "accepted"
                            issue_report["accepted_obj_id"] = int(accepted_obj_id)
                            issue_report["accepted_attempt_index"] = int(attempt_index)
                            issue_report["accepted_click_points"] = [
                                [int(x), int(y)] for x, y in positive_points
                            ]
                            issue_report["updated_frame_indices"] = updated_frame_indices
                            accepted_this_round += 1
                            accepted = True
                            break

                        if verdict["decision"] == "retry":
                            new_points = _dedupe_points(
                                positive_points
                                + [
                                    (int(point[0]), int(point[1]))
                                    for point in verdict.get("additional_points", [])
                                ],
                                min_distance=4.0,
                            )
                            if len(new_points) == len(positive_points):
                                issue_report["failure_reason"] = "retry_without_new_points"
                                break
                            positive_points = new_points
                            continue

                        issue_report["failure_reason"] = (
                            "candidate_rejected"
                            if verdict["decision"] == "reject"
                            else "candidate_not_accepted"
                        )
                        break

                    if not accepted and "failure_reason" not in issue_report:
                        issue_report["failure_reason"] = "max_attempts_exhausted"
                    issues.append(issue_report)
                    round_rows.append(issue_report)

            if accepted_this_round <= 0:
                break

        report = {
            "mode": "postprop_missed_creatures_mllm",
            "total_frames": int(total_frames),
            "window_size": int(window_size),
            "window_stride": int(window_stride),
            "max_issues_per_window": int(max_issues_per_window),
            "max_rounds": int(max_rounds),
            "max_attempts_per_issue": int(max_attempts_per_issue),
            "max_images_per_request": int(max_images_per_request),
            "effective_max_images_per_request": int(image_budget),
            "max_json_retries": int(max_json_retries),
            "duplicate_iou_threshold": float(duplicate_iou_threshold),
            "detection_requests": detection_requests,
            "issues": issues,
            "accepted_issue_count": sum(1 for issue in issues if issue.get("status") == "accepted"),
            "unresolved_issue_count": sum(
                1 for issue in issues if issue.get("status") != "accepted"
            ),
            "accepted_prompts": accepted_prompts,
            "next_obj_id": int(next_obj_id_local),
        }
        _write_json(os.path.join(output_dir, "postprop_missed_creatures_report.json"), report)
        return report
    finally:
        try:
            backend.reset_session(session_id)
        except Exception:
            pass
        if owns_session:
            try:
                backend.close_session(session_id)
            except Exception:
                pass
