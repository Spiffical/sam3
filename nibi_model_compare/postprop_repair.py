from __future__ import annotations

import copy
import json
import os
import shutil
from collections import defaultdict
from typing import Any, Callable

import cv2
import numpy as np

from frame_output_utils import (
    decode_rle_to_mask,
    encode_binary_mask_to_rle,
    frame_object_metadata,
    iter_output_masks_with_ids,
    merge_frame_outputs_by_obj_ids,
    read_video_frame,
)
from postprop_qa_mllm import (
    _build_labeled_collage,
    _extract_json_object,
    _resolve_system_prompt,
    assess_postprop_frame_with_mllm,
)
from sam3.agent.client_sam3 import sam3_inference
from sam3.agent.helpers.mask_overlap_removal import remove_overlapping_masks


def _write_json(path: str, payload: dict[str, Any]) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)


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


def _mask_bbox_xyxy(mask: np.ndarray) -> tuple[int, int, int, int] | None:
    ys, xs = np.where(np.asarray(mask).astype(bool))
    if len(xs) == 0 or len(ys) == 0:
        return None
    return int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())


def _sanitize_xyxy(
    box_xyxy: tuple[int, int, int, int] | list[int],
    frame_w: int,
    frame_h: int,
) -> tuple[int, int, int, int]:
    if frame_w <= 0 or frame_h <= 0:
        return (0, 0, 0, 0)
    x1, y1, x2, y2 = [int(v) for v in box_xyxy]
    if x1 > x2:
        x1, x2 = x2, x1
    if y1 > y2:
        y1, y2 = y2, y1
    x1 = max(0, min(frame_w - 1, x1))
    y1 = max(0, min(frame_h - 1, y1))
    x2 = max(0, min(frame_w - 1, x2))
    y2 = max(0, min(frame_h - 1, y2))
    return (x1, y1, max(x1, x2), max(y1, y2))


def _expand_xyxy(
    box_xyxy: tuple[int, int, int, int],
    frame_w: int,
    frame_h: int,
    context_ratio: float = 0.30,
) -> tuple[int, int, int, int]:
    x1, y1, x2, y2 = _sanitize_xyxy(box_xyxy, frame_w, frame_h)
    bw = max(1, x2 - x1 + 1)
    bh = max(1, y2 - y1 + 1)
    pad_x = int(round(bw * float(context_ratio)))
    pad_y = int(round(bh * float(context_ratio)))
    return _sanitize_xyxy(
        (
        max(0, x1 - pad_x),
        max(0, y1 - pad_y),
        min(frame_w - 1, x2 + pad_x),
        min(frame_h - 1, y2 + pad_y),
        ),
        frame_w,
        frame_h,
    )


def _union_xyxy(
    boxes_xyxy: list[tuple[int, int, int, int]], frame_w: int, frame_h: int
) -> tuple[int, int, int, int]:
    x1 = min(b[0] for b in boxes_xyxy)
    y1 = min(b[1] for b in boxes_xyxy)
    x2 = max(b[2] for b in boxes_xyxy)
    y2 = max(b[3] for b in boxes_xyxy)
    return _sanitize_xyxy((x1, y1, x2, y2), frame_w, frame_h)


def _frame_mask_by_obj_id(
    results_by_frame: dict[int, dict[str, Any]],
    frame_index: int,
    obj_id: int,
    frame_h: int,
    frame_w: int,
) -> np.ndarray | None:
    outputs = results_by_frame.get(int(frame_index))
    if not outputs:
        return None
    for existing_obj_id, mask in iter_output_masks_with_ids(outputs, frame_h, frame_w):
        if int(existing_obj_id) == int(obj_id):
            return np.asarray(mask).astype(bool)
    return None


def _nearest_temporal_masks(
    results_by_frame: dict[int, dict[str, Any]],
    frame_index: int,
    obj_id: int,
    frame_h: int,
    frame_w: int,
    search_radius: int,
) -> list[tuple[int, np.ndarray]]:
    out: list[tuple[int, np.ndarray]] = []
    for direction in (-1, 1):
        for offset in range(1, max(1, int(search_radius)) + 1):
            idx = int(frame_index) + (direction * offset)
            if idx < 0:
                break
            mask = _frame_mask_by_obj_id(results_by_frame, idx, obj_id, frame_h, frame_w)
            if mask is not None and mask.any():
                out.append((idx, mask))
                break
    return out


def _render_mask_overlay(
    frame_bgr: np.ndarray,
    mask: np.ndarray | None,
    *,
    color: tuple[int, int, int] = (0, 255, 0),
    label: str | None = None,
    crop_xyxy: tuple[int, int, int, int] | None = None,
    extra_boxes_xyxy: list[tuple[int, int, int, int]] | None = None,
) -> np.ndarray:
    out = frame_bgr.copy()
    overlay = np.zeros_like(out)
    if mask is not None and np.asarray(mask).any():
        mask_bool = np.asarray(mask).astype(bool)
        overlay[mask_bool] = color
        mask_u8 = (mask_bool.astype(np.uint8) * 255)
        contours, _ = cv2.findContours(mask_u8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(out, contours, -1, color, 2)
    out = cv2.addWeighted(out, 1.0, overlay, 0.35, 0.0)
    if crop_xyxy is not None:
        x1, y1, x2, y2 = crop_xyxy
        out = out[y1 : y2 + 1, x1 : x2 + 1].copy()
    if extra_boxes_xyxy:
        for box in extra_boxes_xyxy:
            x1, y1, x2, y2 = box
            if crop_xyxy is not None:
                cx1, cy1, _, _ = crop_xyxy
                x1 -= cx1
                x2 -= cx1
                y1 -= cy1
                y2 -= cy1
            cv2.rectangle(out, (x1, y1), (x2, y2), (0, 200, 255), 2)
    if label:
        cv2.putText(
            out,
            label,
            (8, 18),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )
    return out


def _render_multi_mask_overlay(
    frame_bgr: np.ndarray,
    masks_by_obj_id: dict[int, np.ndarray],
    crop_xyxy: tuple[int, int, int, int],
) -> np.ndarray:
    out = frame_bgr.copy()
    overlay = np.zeros_like(out)
    palette = [
        (0, 255, 0),
        (0, 180, 255),
        (255, 200, 0),
        (255, 0, 180),
    ]
    for i, (obj_id, mask) in enumerate(sorted(masks_by_obj_id.items())):
        color = palette[i % len(palette)]
        mask_bool = np.asarray(mask).astype(bool)
        overlay[mask_bool] = color
        mask_u8 = (mask_bool.astype(np.uint8) * 255)
        contours, _ = cv2.findContours(mask_u8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(out, contours, -1, color, 2)
        bbox = _mask_bbox_xyxy(mask_bool)
        if bbox is not None:
            x1, y1, _, _ = bbox
            cv2.putText(
                out,
                f"id {obj_id}",
                (x1, max(14, y1 - 4)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                color,
                1,
                cv2.LINE_AA,
            )
    out = cv2.addWeighted(out, 1.0, overlay, 0.35, 0.0)
    x1, y1, x2, y2 = crop_xyxy
    return out[y1 : y2 + 1, x1 : x2 + 1].copy()


def _collect_masks_by_obj_ids(
    outputs: dict[str, Any],
    frame_h: int,
    frame_w: int,
    obj_ids: set[int] | None = None,
) -> dict[int, np.ndarray]:
    selected_obj_ids = {int(x) for x in (obj_ids or set())}
    masks: dict[int, np.ndarray] = {}
    for obj_id, mask in iter_output_masks_with_ids(outputs, frame_h, frame_w):
        if selected_obj_ids and int(obj_id) not in selected_obj_ids:
            continue
        masks[int(obj_id)] = np.asarray(mask).astype(bool)
    return masks


def _copy_file_if_exists(src_path: str | None, dst_path: str) -> bool:
    if not src_path or not os.path.isfile(src_path):
        return False
    os.makedirs(os.path.dirname(dst_path), exist_ok=True)
    shutil.copy2(src_path, dst_path)
    return True


def _deduplicate_candidate_masks(
    candidates: list[dict[str, Any]], iou_threshold: float = 0.95
) -> list[dict[str, Any]]:
    kept: list[dict[str, Any]] = []
    for candidate in candidates:
        candidate_mask = candidate["mask_full"]
        duplicate = False
        for kept_candidate in kept:
            if _mask_iou(candidate_mask, kept_candidate["mask_full"]) >= iou_threshold:
                duplicate = True
                break
        if not duplicate:
            kept.append(candidate)
    return kept


def _score_candidate(
    candidate_mask: np.ndarray,
    *,
    sam_score: float,
    temporal_masks: list[tuple[int, np.ndarray]],
    other_masks: list[np.ndarray],
    reference_area: float | None,
) -> dict[str, float]:
    temporal_iou = 0.0
    if temporal_masks:
        temporal_iou = float(
            np.mean([_mask_iou(candidate_mask, temporal_mask) for _, temporal_mask in temporal_masks])
        )

    overlap_other_max = 0.0
    if other_masks:
        overlap_other_max = float(max(_mask_iou(candidate_mask, other) for other in other_masks))

    candidate_area = float(np.asarray(candidate_mask).astype(bool).sum())
    area_score = 0.5
    if reference_area and reference_area > 0:
        area_score = max(
            0.0,
            1.0 - min(abs(candidate_area - reference_area) / float(reference_area), 1.0),
        )

    total_score = (
        (0.50 * float(sam_score))
        + (0.30 * temporal_iou)
        + (0.15 * area_score)
        - (0.30 * overlap_other_max)
    )
    return {
        "sam_score": float(sam_score),
        "temporal_iou": float(temporal_iou),
        "overlap_other_max": float(overlap_other_max),
        "area_score": float(area_score),
        "total_score": float(total_score),
    }


def _current_mask_score_prior(
    results_by_frame: dict[int, dict[str, Any]],
    frame_index: int,
    obj_id: int,
    frame_h: int,
    frame_w: int,
) -> float:
    outputs = results_by_frame.get(int(frame_index), {})
    metadata = frame_object_metadata(outputs, frame_h, frame_w)
    row = metadata.get(int(obj_id), {})
    tracker_confidence = row.get("tracker_confidence")
    if tracker_confidence is not None:
        return float(max(0.0, min(1.0, tracker_confidence)))
    confidence = row.get("confidence")
    if confidence is not None:
        return float(max(0.0, min(1.0, confidence)))
    return 0.35


def _assessment_bad_object_ids(assessment: dict[str, Any] | None) -> set[int]:
    if not isinstance(assessment, dict):
        return set()
    return {
        int(x)
        for x in assessment.get("bad_object_ids", [])
        if x is not None
    }


def _assessment_overlap_pairs(
    assessment: dict[str, Any] | None,
) -> set[tuple[int, int]]:
    if not isinstance(assessment, dict):
        return set()
    out: set[tuple[int, int]] = set()
    for pair in assessment.get("overlap_pairs", []):
        if not isinstance(pair, (list, tuple)) or len(pair) < 2:
            continue
        try:
            a, b = sorted((int(pair[0]), int(pair[1])))
        except Exception:
            continue
        out.add((a, b))
    return out


def _assessment_has_mask_issues(assessment: dict[str, Any] | None) -> bool:
    if not isinstance(assessment, dict):
        return False
    return bool(
        assessment.get("bad_object_ids")
        or assessment.get("overlap_pairs")
        or assessment.get("missing_creatures")
        or assessment.get("needs_rerun")
    )


def _assessment_is_raw_video_invalid(assessment: dict[str, Any] | None) -> bool:
    if not isinstance(assessment, dict):
        return False
    return (
        str(assessment.get("frame_validity", "")).strip().lower() == "invalid"
        and not _assessment_has_mask_issues(assessment)
    )


def _evaluate_issue_resolution(
    issue: dict[str, Any],
    assessment: dict[str, Any] | None,
) -> dict[str, Any]:
    issue_type = str(issue.get("issue_type", "")).strip().lower()
    target_obj_ids = [int(x) for x in issue.get("target_obj_ids", [])]
    bad_object_ids = _assessment_bad_object_ids(assessment)
    overlap_pairs = _assessment_overlap_pairs(assessment)

    unresolved_reasons: list[str] = []
    if issue_type == "poor_boundary":
        for obj_id in target_obj_ids:
            if obj_id in bad_object_ids:
                unresolved_reasons.append(f"obj_{obj_id}_still_bad")
            if any(obj_id in pair for pair in overlap_pairs):
                unresolved_reasons.append(f"obj_{obj_id}_still_overlaps")
    elif issue_type == "merged_objects":
        target_pairs = {
            tuple(sorted((target_obj_ids[i], target_obj_ids[j])))
            for i in range(len(target_obj_ids))
            for j in range(i + 1, len(target_obj_ids))
        }
        for pair in target_pairs:
            if pair in overlap_pairs:
                unresolved_reasons.append(
                    f"pair_{pair[0]}_{pair[1]}_still_overlaps"
                )
        for obj_id in target_obj_ids:
            if obj_id in bad_object_ids:
                unresolved_reasons.append(f"obj_{obj_id}_still_bad")
    elif issue_type == "missing_object":
        if bool((assessment or {}).get("missing_creatures")):
            unresolved_reasons.append("missing_creatures_still_true")
    else:
        if _assessment_has_mask_issues(assessment):
            unresolved_reasons.append("mask_issue_still_present")

    return {
        "issue_id": issue.get("issue_id"),
        "issue_type": issue_type,
        "target_obj_ids": target_obj_ids,
        "resolved": not unresolved_reasons,
        "unresolved_reasons": unresolved_reasons,
    }


def _issue_crop_box(
    *,
    issue: dict[str, Any],
    frame_bgr: np.ndarray,
    results_by_frame: dict[int, dict[str, Any]],
    frame_h: int,
    frame_w: int,
    search_radius: int,
) -> tuple[int, int, int, int]:
    boxes: list[tuple[int, int, int, int]] = []
    for obj_id in issue.get("target_obj_ids", []):
        current_mask = _frame_mask_by_obj_id(
            results_by_frame, issue["frame_index"], obj_id, frame_h, frame_w
        )
        if current_mask is not None:
            bbox = _mask_bbox_xyxy(current_mask)
            if bbox is not None:
                boxes.append(bbox)
        for _idx, temporal_mask in _nearest_temporal_masks(
            results_by_frame,
            issue["frame_index"],
            obj_id,
            frame_h,
            frame_w,
            search_radius=search_radius,
        ):
            bbox = _mask_bbox_xyxy(temporal_mask)
            if bbox is not None:
                boxes.append(bbox)

    for region in issue.get("missing_regions", []):
        try:
            x = int(region["x"])
            y = int(region["y"])
            w = int(region["w"])
            h = int(region["h"])
        except Exception:
            continue
        boxes.append((x, y, x + max(0, w - 1), y + max(0, h - 1)))

    if not boxes:
        return (0, 0, frame_w - 1, frame_h - 1)
    return _expand_xyxy(_union_xyxy(boxes, frame_w, frame_h), frame_w, frame_h)


def _prompt_bank(initial_text_prompt: str, prompt_profile: str) -> list[str]:
    prompts = [str(initial_text_prompt).strip()]
    profile = str(prompt_profile or "").strip().lower()
    if profile == "underwater":
        prompts.extend(["marine creature", "small creature", "creature"])
    else:
        prompts.extend(["animal", "creature"])

    unique: list[str] = []
    seen: set[str] = set()
    for prompt in prompts:
        prompt_norm = prompt.strip()
        if not prompt_norm:
            continue
        key = prompt_norm.lower()
        if key in seen:
            continue
        seen.add(key)
        unique.append(prompt_norm)
    return unique


def build_repair_issues(
    *,
    postprop_report: dict[str, Any],
    hard_invalid_frame_indices: set[int],
) -> tuple[list[dict[str, Any]], list[int]]:
    issues: list[dict[str, Any]] = []
    raw_video_invalid_frame_indices: set[int] = set()
    requests_by_frame = {
        int(row.get("frame_index")): row for row in postprop_report.get("requests", [])
    }

    for assessment in postprop_report.get("per_frame_assessments", []):
        frame_index = int(assessment.get("frame_index", -1))
        if frame_index < 0 or frame_index in hard_invalid_frame_indices:
            continue

        bad_object_ids = sorted(
            set(int(x) for x in assessment.get("bad_object_ids", []) if x is not None)
        )
        overlap_pairs = [
            [int(pair[0]), int(pair[1])]
            for pair in assessment.get("overlap_pairs", [])
            if isinstance(pair, (list, tuple)) and len(pair) >= 2
        ]
        missing_regions = [
            dict(region)
            for region in assessment.get("missing_creature_regions", [])
            if isinstance(region, dict)
        ]

        if bad_object_ids:
            for obj_id in bad_object_ids:
                issues.append(
                    {
                        "issue_id": f"f{frame_index:05d}_poor_boundary_obj{obj_id}",
                        "frame_index": frame_index,
                        "issue_type": "poor_boundary",
                        "target_obj_ids": [int(obj_id)],
                        "missing_regions": [],
                        "reasons": list(assessment.get("reasons", [])),
                        "confidence": float(assessment.get("confidence", 0.0)),
                    }
                )

        if overlap_pairs:
            for pair in overlap_pairs:
                a, b = sorted(pair[:2])
                issues.append(
                    {
                        "issue_id": f"f{frame_index:05d}_merged_objs_{a}_{b}",
                        "frame_index": frame_index,
                        "issue_type": "merged_objects",
                        "target_obj_ids": [a, b],
                        "missing_regions": [],
                        "reasons": list(assessment.get("reasons", [])),
                        "confidence": float(assessment.get("confidence", 0.0)),
                    }
                )

        if bool(assessment.get("missing_creatures")):
            if missing_regions:
                for region_idx, region in enumerate(missing_regions):
                    issues.append(
                        {
                            "issue_id": f"f{frame_index:05d}_missing_region_{region_idx}",
                            "frame_index": frame_index,
                            "issue_type": "missing_object",
                            "target_obj_ids": [],
                            "missing_regions": [region],
                            "reasons": list(assessment.get("reasons", [])),
                            "confidence": float(assessment.get("confidence", 0.0)),
                        }
                    )
            else:
                issues.append(
                    {
                        "issue_id": f"f{frame_index:05d}_missing_object",
                        "frame_index": frame_index,
                        "issue_type": "missing_object",
                        "target_obj_ids": [],
                        "missing_regions": [],
                        "reasons": list(assessment.get("reasons", [])),
                        "confidence": float(assessment.get("confidence", 0.0)),
                    }
                )

        has_repairable = bool(bad_object_ids or overlap_pairs or assessment.get("missing_creatures"))
        if (
            assessment.get("frame_validity") == "invalid"
            and not has_repairable
            and not bool(assessment.get("needs_rerun"))
        ):
            raw_video_invalid_frame_indices.add(frame_index)

        if bool(assessment.get("needs_rerun")) and not has_repairable:
            present_obj_ids = list(requests_by_frame.get(frame_index, {}).get("present_obj_ids", []))
            if present_obj_ids:
                for obj_id in present_obj_ids:
                    issues.append(
                        {
                            "issue_id": f"f{frame_index:05d}_rerun_obj{int(obj_id)}",
                            "frame_index": frame_index,
                            "issue_type": "poor_boundary",
                            "target_obj_ids": [int(obj_id)],
                            "missing_regions": [],
                            "reasons": list(assessment.get("reasons", [])) + ["needs_rerun"],
                            "confidence": float(assessment.get("confidence", 0.0)),
                        }
                    )
            else:
                raw_video_invalid_frame_indices.add(frame_index)

    return issues, sorted(raw_video_invalid_frame_indices)


def generate_mask_candidates_for_issue(
    *,
    video_path: str,
    frame_index: int,
    issue: dict[str, Any],
    results_by_frame: dict[int, dict[str, Any]],
    image_processor: Any,
    initial_text_prompt: str,
    prompt_profile: str,
    max_candidates: int,
    output_dir: str,
    search_radius: int,
) -> dict[str, Any]:
    frame_bgr = read_video_frame(video_path, frame_index)
    if frame_bgr is None:
        raise RuntimeError(f"Could not decode frame {frame_index} for repair.")

    frame_h, frame_w = frame_bgr.shape[:2]
    crop_xyxy = _issue_crop_box(
        issue=issue,
        frame_bgr=frame_bgr,
        results_by_frame=results_by_frame,
        frame_h=frame_h,
        frame_w=frame_w,
        search_radius=search_radius,
    )
    x1, y1, x2, y2 = crop_xyxy
    crop_bgr = frame_bgr[y1 : y2 + 1, x1 : x2 + 1].copy()
    if crop_bgr.size == 0:
        crop_xyxy = (0, 0, frame_w - 1, frame_h - 1)
        x1, y1, x2, y2 = crop_xyxy
        crop_bgr = frame_bgr.copy()

    issue_dir = os.path.join(output_dir, issue["issue_id"])
    os.makedirs(issue_dir, exist_ok=True)
    crop_path = os.path.join(issue_dir, "crop.jpg")
    if crop_bgr.size == 0 or not cv2.imwrite(crop_path, crop_bgr):
        raise RuntimeError(
            f"Could not write non-empty repair crop for issue {issue['issue_id']}."
        )

    target_candidates: list[dict[str, Any]] = []
    prompt_rows: list[dict[str, Any]] = []

    other_masks_by_obj_id = {
        obj_id: mask
        for obj_id, mask in iter_output_masks_with_ids(
            results_by_frame.get(frame_index, {}), frame_h, frame_w
        )
        if obj_id not in set(issue.get("target_obj_ids", []))
    }

    for prompt_text in _prompt_bank(initial_text_prompt, prompt_profile):
        outputs = sam3_inference(image_processor, crop_path, prompt_text)
        outputs = remove_overlapping_masks(outputs)
        scores = list(outputs.get("pred_scores", []))
        pred_masks = list(outputs.get("pred_masks", []))
        prompt_rows.append(
            {
                "prompt_text": prompt_text,
                "num_masks": len(pred_masks),
                "scores": [float(x) for x in scores],
            }
        )
        for mask_idx, mask_rle in enumerate(pred_masks):
            crop_mask = decode_rle_to_mask(mask_rle, crop_bgr.shape[0], crop_bgr.shape[1]).astype(bool)
            if not crop_mask.any():
                continue
            full_mask = np.zeros((frame_h, frame_w), dtype=bool)
            full_mask[y1 : y2 + 1, x1 : x2 + 1] = crop_mask
            target_candidates.append(
                {
                    "candidate_id": f"{prompt_text[:24].replace(' ', '_')}_{mask_idx}",
                    "prompt_text": prompt_text,
                    "mask_full": full_mask,
                    "mask_rle": encode_binary_mask_to_rle(full_mask),
                    "sam_score": float(scores[mask_idx]) if mask_idx < len(scores) else 0.0,
                    "crop_xyxy": crop_xyxy,
                }
            )

    target_candidates = _deduplicate_candidate_masks(target_candidates)

    candidate_sets: dict[int | str, list[dict[str, Any]]] = {}
    for target_obj_id in issue.get("target_obj_ids", []) or ["new_object"]:
        temporal_masks = []
        reference_area = None
        current_mask = None
        pinned_candidates: list[dict[str, Any]] = []
        if target_obj_id != "new_object":
            current_mask = _frame_mask_by_obj_id(
                results_by_frame, frame_index, int(target_obj_id), frame_h, frame_w
            )
            if current_mask is not None:
                reference_area = float(current_mask.sum())
            temporal_masks = _nearest_temporal_masks(
                results_by_frame,
                frame_index,
                int(target_obj_id),
                frame_h,
                frame_w,
                search_radius=search_radius,
            )

        scored_candidates: list[dict[str, Any]] = []
        if current_mask is not None and current_mask.any():
            current_scores = _score_candidate(
                current_mask,
                sam_score=_current_mask_score_prior(
                    results_by_frame,
                    frame_index,
                    int(target_obj_id),
                    frame_h,
                    frame_w,
                ),
                temporal_masks=temporal_masks,
                other_masks=list(other_masks_by_obj_id.values()),
                reference_area=reference_area,
            )
            pinned_candidates.append(
                {
                    "candidate_id": "current",
                    "prompt_text": "current_mask",
                    "mask_full": current_mask,
                    "mask_rle": encode_binary_mask_to_rle(current_mask),
                    **current_scores,
                }
            )
            remove_scores = _score_candidate(
                np.zeros_like(current_mask, dtype=bool),
                sam_score=0.0,
                temporal_masks=temporal_masks,
                other_masks=list(other_masks_by_obj_id.values()),
                reference_area=reference_area,
            )
            pinned_candidates.append(
                {
                    "candidate_id": "remove_mask",
                    "prompt_text": "remove_mask",
                    "mask_full": np.zeros_like(current_mask, dtype=bool),
                    "mask_rle": encode_binary_mask_to_rle(
                        np.zeros_like(current_mask, dtype=bool)
                    ),
                    **remove_scores,
                }
            )

        for candidate in target_candidates:
            feature_scores = _score_candidate(
                candidate["mask_full"],
                sam_score=candidate["sam_score"],
                temporal_masks=temporal_masks,
                other_masks=list(other_masks_by_obj_id.values()),
                reference_area=reference_area,
            )
            scored_candidates.append(
                {
                    "candidate_id": candidate["candidate_id"],
                    "prompt_text": candidate["prompt_text"],
                    "mask_full": candidate["mask_full"],
                    "mask_rle": candidate["mask_rle"],
                    **feature_scores,
                }
            )

        scored_candidates.sort(key=lambda row: row.get("total_score", 0.0), reverse=True)
        final_rows: list[dict[str, Any]] = []
        seen_candidate_ids: set[str] = set()
        for row in pinned_candidates:
            final_rows.append(row)
            seen_candidate_ids.add(str(row.get("candidate_id")))
        for row in scored_candidates:
            candidate_id = str(row.get("candidate_id"))
            if candidate_id in seen_candidate_ids:
                continue
            final_rows.append(row)
            seen_candidate_ids.add(candidate_id)
            if len(final_rows) >= max(1, int(max_candidates)):
                break
        candidate_sets[target_obj_id] = final_rows[: max(1, int(max_candidates))]

    candidate_summary = {
        key: [
            {
                k: v
                for k, v in row.items()
                if k not in {"mask_full"}
            }
            for row in rows
        ]
        for key, rows in candidate_sets.items()
    }
    _write_json(
        os.path.join(issue_dir, "candidate_summary.json"),
        {
            "issue": issue,
            "crop_xyxy": list(crop_xyxy),
            "prompt_runs": prompt_rows,
            "candidate_sets": candidate_summary,
        },
    )
    return {
        "issue": issue,
        "frame_index": int(frame_index),
        "crop_xyxy": crop_xyxy,
        "crop_path": crop_path,
        "candidate_sets": candidate_sets,
        "candidate_summary": candidate_summary,
        "issue_dir": issue_dir,
    }


def _resolve_chooser_system_prompt(
    prompt_profile: str,
    prompt_template_path: str | None,
) -> str:
    if prompt_template_path:
        with open(prompt_template_path, "r", encoding="utf-8") as handle:
            return handle.read().strip()

    profile = str(prompt_profile or "").strip().lower()
    domain = "underwater creature" if profile == "underwater" else "target object"
    return (
        "You are choosing the best SAM3 segmentation repair candidate.\n"
        "Use the temporal reference tiles and the candidate masks to decide which candidate best matches "
        f"the intended {domain}.\n"
        "Prefer boundary quality, temporal consistency, and avoiding overlap with other objects.\n"
        "Return STRICT JSON only with schema "
        '{"decision":"choose_candidate|keep_current|remove_mask|unrepairable","selected_candidate_id":str,'
        '"confidence":float,"reason":str}.'
    )


def choose_candidate_with_mllm(
    *,
    issue: dict[str, Any],
    target_obj_id: int | str,
    candidate_bundle: dict[str, Any],
    results_by_frame: dict[int, dict[str, Any]],
    video_path: str,
    send_generate_request_fn: Callable[[list[dict[str, Any]]], str | None],
    initial_text_prompt: str,
    prompt_profile: str,
    prompt_template_path: str | None,
    output_dir: str,
    search_radius: int,
    max_json_retries: int = 2,
) -> dict[str, Any]:
    frame_index = int(issue["frame_index"])
    frame_bgr = read_video_frame(video_path, frame_index)
    if frame_bgr is None:
        raise RuntimeError(f"Could not decode frame {frame_index} for candidate choice.")
    frame_h, frame_w = frame_bgr.shape[:2]
    crop_xyxy = tuple(int(x) for x in candidate_bundle["crop_xyxy"])
    x1, y1, x2, y2 = crop_xyxy

    tiles: list[tuple[str, np.ndarray]] = []
    if target_obj_id != "new_object":
        for temporal_frame_idx, temporal_mask in _nearest_temporal_masks(
            results_by_frame,
            frame_index,
            int(target_obj_id),
            frame_h,
            frame_w,
            search_radius=search_radius,
        ):
            temporal_frame = read_video_frame(video_path, temporal_frame_idx)
            if temporal_frame is None:
                continue
            tiles.append(
                (
                    f"ref f={temporal_frame_idx} obj={target_obj_id}",
                    _render_mask_overlay(
                        temporal_frame,
                        temporal_mask,
                        crop_xyxy=crop_xyxy,
                    ),
                )
            )

    current_masks_by_obj_id = {
        obj_id: mask
        for obj_id, mask in iter_output_masks_with_ids(
            results_by_frame.get(frame_index, {}), frame_h, frame_w
        )
        if obj_id in set(int(x) for x in issue.get("target_obj_ids", []))
    }
    tiles.append(
        (
            f"current f={frame_index}",
            _render_multi_mask_overlay(frame_bgr, current_masks_by_obj_id, crop_xyxy),
        )
    )
    tiles.append(("raw crop", frame_bgr[y1 : y2 + 1, x1 : x2 + 1].copy()))

    candidate_rows = candidate_bundle["candidate_sets"][target_obj_id]
    for candidate in candidate_rows:
        label = candidate["candidate_id"]
        tiles.append(
            (
                label,
                _render_mask_overlay(
                    frame_bgr,
                    candidate["mask_full"],
                    crop_xyxy=crop_xyxy,
                ),
            )
        )

    chooser_dir = os.path.join(output_dir, issue["issue_id"])
    os.makedirs(chooser_dir, exist_ok=True)
    collage_path = os.path.join(
        chooser_dir, f"choose_{str(target_obj_id).replace(' ', '_')}.jpg"
    )
    rows, cols = _build_labeled_collage(tiles, collage_path, cols=3, tile_max_edge=320)

    candidate_text = "\n".join(
        [
            f"- {row['candidate_id']}: prompt='{row['prompt_text']}', total_score={row['total_score']:.3f}, "
            f"sam_score={row['sam_score']:.3f}, temporal_iou={row['temporal_iou']:.3f}, "
            f"overlap_other_max={row['overlap_other_max']:.3f}"
            for row in candidate_rows
        ]
    )
    issue_desc = (
        f"{issue['issue_type']} for object id {target_obj_id}"
        if target_obj_id != "new_object"
        else f"{issue['issue_type']} with no current object id"
    )
    instruction = (
        f"Collage is row-major ({rows} rows x {cols} cols).\n"
        f"Initial prompt context: '{initial_text_prompt}'.\n"
        f"Repair issue: {issue_desc} on frame {frame_index}.\n"
        "Choose the best candidate mask. Candidate 'current' means keep the existing mask if present.\n"
        "Candidate 'remove_mask' means suppress this object mask in the repaired window.\n"
        "If none of the candidates are usable, return decision='unrepairable'.\n"
        "Candidates:\n"
        f"{candidate_text}\n"
        "Return STRICT JSON ONLY with schema:\n"
        "{"
        '"decision":"choose_candidate|keep_current|remove_mask|unrepairable",'
        '"selected_candidate_id":str,'
        '"confidence":float,'
        '"reason":str'
        "}"
    )
    messages = [
        {
            "role": "system",
            "content": _resolve_chooser_system_prompt(prompt_profile, prompt_template_path),
        },
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
                            "text": "Reply again with STRICT JSON only and no extra text.",
                        }
                    ],
                }
            )

    selected_candidate_id = "current"
    decision = "unrepairable"
    confidence = 0.0
    reason = "invalid_json"
    valid_ids = {row["candidate_id"] for row in candidate_rows}
    if isinstance(parsed, dict):
        decision_raw = str(parsed.get("decision", "")).strip().lower()
        if decision_raw in {
            "choose_candidate",
            "keep_current",
            "remove_mask",
            "unrepairable",
        }:
            decision = decision_raw
        selected_raw = str(parsed.get("selected_candidate_id", "current")).strip()
        if selected_raw in valid_ids:
            selected_candidate_id = selected_raw
        elif decision == "keep_current" and "current" in valid_ids:
            selected_candidate_id = "current"
        elif decision == "remove_mask" and "remove_mask" in valid_ids:
            selected_candidate_id = "remove_mask"
        try:
            confidence = float(parsed.get("confidence", 0.0))
        except Exception:
            confidence = 0.0
        reason = str(parsed.get("reason", "")).strip()

    choice = {
        "issue_id": issue["issue_id"],
        "frame_index": frame_index,
        "target_obj_id": target_obj_id,
        "decision": decision,
        "selected_candidate_id": selected_candidate_id,
        "confidence": float(max(0.0, min(1.0, confidence))),
        "reason": reason,
        "collage_path": collage_path,
        "raw_text": model_text,
        "parsed_json_ok": isinstance(parsed, dict),
    }
    _write_json(
        os.path.join(chooser_dir, f"choice_{str(target_obj_id).replace(' ', '_')}.json"),
        choice,
    )
    return choice


def _snapshot_window(
    results_by_frame: dict[int, dict[str, Any]], center_frame: int, radius: int
) -> dict[int, dict[str, Any]]:
    lo = max(0, int(center_frame) - int(radius))
    hi = int(center_frame) + int(radius)
    return {
        frame_index: copy.deepcopy(results_by_frame[frame_index])
        for frame_index in range(lo, hi + 1)
        if frame_index in results_by_frame
    }


def _restore_window(
    results_by_frame: dict[int, dict[str, Any]],
    snapshot: dict[int, dict[str, Any]],
    center_frame: int,
    radius: int,
) -> None:
    lo = max(0, int(center_frame) - int(radius))
    hi = int(center_frame) + int(radius)
    for frame_index in list(results_by_frame.keys()):
        if lo <= frame_index <= hi and frame_index not in snapshot:
            results_by_frame.pop(frame_index, None)
    for frame_index, outputs in snapshot.items():
        results_by_frame[frame_index] = copy.deepcopy(outputs)


def apply_repair_action(
    *,
    backend: Any,
    video_path: str,
    results_by_frame: dict[int, dict[str, Any]],
    frame_index: int,
    frame_size_wh: tuple[int, int],
    image_size: int,
    selected_masks_by_obj_id: dict[int, np.ndarray],
    remove_obj_ids: set[int] | None,
    repair_window: int,
) -> dict[str, Any]:
    updated_frames: dict[int, dict[str, Any]] = {}
    replace_obj_ids = set(int(x) for x in selected_masks_by_obj_id.keys())
    remove_obj_ids = {int(x) for x in (remove_obj_ids or set())}
    replace_obj_ids |= remove_obj_ids

    if selected_masks_by_obj_id:
        session_id = backend.start_session(resource_path=video_path, image_size=image_size)
        try:
            for obj_id, mask in selected_masks_by_obj_id.items():
                backend.add_mask_prompt(
                    session_id=session_id,
                    frame_idx=int(frame_index),
                    obj_id=int(obj_id),
                    mask=np.asarray(mask).astype(bool),
                )

            request = {
                "session_id": session_id,
                "type": "propagate_in_video",
                "start_frame_index": int(frame_index),
                "propagation_direction": "both",
                "max_frame_num_to_track": int(repair_window),
            }
            for output in backend.propagate(request):
                updated_frames[int(output["frame_index"])] = output["outputs"]
        finally:
            try:
                backend.close_session(session_id)
            except Exception:
                pass
    elif remove_obj_ids:
        if frame_index in results_by_frame:
            updated_frames[int(frame_index)] = {}

    merged_frame_indices: list[int] = []
    for updated_frame_index, patched_outputs in updated_frames.items():
        base_outputs = results_by_frame.get(updated_frame_index, {})
        results_by_frame[updated_frame_index] = merge_frame_outputs_by_obj_ids(
            base_outputs,
            patched_outputs,
            replace_obj_ids=replace_obj_ids,
        )
        merged_frame_indices.append(int(updated_frame_index))

    return {
        "frame_index": int(frame_index),
        "replace_obj_ids": sorted(replace_obj_ids),
        "remove_obj_ids": sorted(remove_obj_ids),
        "updated_frame_indices": sorted(merged_frame_indices),
    }


def _issues_crop_box(
    *,
    frame_issues: list[dict[str, Any]],
    frame_bgr: np.ndarray,
    results_by_frame: dict[int, dict[str, Any]],
    frame_h: int,
    frame_w: int,
    search_radius: int,
) -> tuple[int, int, int, int]:
    boxes: list[tuple[int, int, int, int]] = []
    for issue in frame_issues:
        boxes.append(
            _issue_crop_box(
                issue=issue,
                frame_bgr=frame_bgr,
                results_by_frame=results_by_frame,
                frame_h=frame_h,
                frame_w=frame_w,
                search_radius=search_radius,
            )
        )
    if not boxes:
        return (0, 0, frame_w - 1, frame_h - 1)
    return _union_xyxy(boxes, frame_w, frame_h)


def _write_visual_debug_sample(
    *,
    visual_debug_dir: str,
    sample_index: int,
    video_path: str,
    frame_index: int,
    frame_issues: list[dict[str, Any]],
    before_results_by_frame: dict[int, dict[str, Any]],
    after_results_by_frame: dict[int, dict[str, Any]],
    action: dict[str, Any],
    attempt_choices: list[dict[str, Any]],
    verification: dict[str, Any] | None,
    repair_window: int,
) -> dict[str, Any] | None:
    frame_bgr = read_video_frame(video_path, frame_index)
    if frame_bgr is None:
        return None

    frame_h, frame_w = frame_bgr.shape[:2]
    crop_xyxy = _issues_crop_box(
        frame_issues=frame_issues,
        frame_bgr=frame_bgr,
        results_by_frame=before_results_by_frame,
        frame_h=frame_h,
        frame_w=frame_w,
        search_radius=max(2, int(repair_window)),
    )
    replace_obj_ids = {int(x) for x in action.get("replace_obj_ids", [])}
    before_outputs = before_results_by_frame.get(frame_index, {})
    after_outputs = after_results_by_frame.get(frame_index, {})

    before_target_masks = _collect_masks_by_obj_ids(
        before_outputs, frame_h, frame_w, replace_obj_ids
    )
    after_target_masks = _collect_masks_by_obj_ids(
        after_outputs, frame_h, frame_w, replace_obj_ids
    )
    before_all_masks = _collect_masks_by_obj_ids(before_outputs, frame_h, frame_w)
    after_all_masks = _collect_masks_by_obj_ids(after_outputs, frame_h, frame_w)

    sample_name = (
        f"sample_{sample_index:03d}_f{frame_index:05d}_"
        f"{'pass' if verification and verification.get('verification_passed') else 'fail'}"
    )
    sample_dir = os.path.join(visual_debug_dir, sample_name)
    os.makedirs(sample_dir, exist_ok=True)

    before_after_path = os.path.join(sample_dir, "before_after.jpg")
    raw_crop = frame_bgr[
        crop_xyxy[1] : crop_xyxy[3] + 1,
        crop_xyxy[0] : crop_xyxy[2] + 1,
    ].copy()
    _build_labeled_collage(
        [
            ("raw crop", raw_crop),
            (
                "before targets",
                _render_multi_mask_overlay(frame_bgr, before_target_masks, crop_xyxy),
            ),
            (
                "after targets",
                _render_multi_mask_overlay(frame_bgr, after_target_masks, crop_xyxy),
            ),
            ("before all", _render_multi_mask_overlay(frame_bgr, before_all_masks, crop_xyxy)),
            ("after all", _render_multi_mask_overlay(frame_bgr, after_all_masks, crop_xyxy)),
        ],
        before_after_path,
        cols=2,
        tile_max_edge=420,
    )

    copied_choice_paths: list[str] = []
    for choice in attempt_choices:
        dst_name = (
            f"choice_{str(choice.get('target_obj_id')).replace(' ', '_')}_"
            f"{str(choice.get('selected_candidate_id', 'unknown')).replace(' ', '_')}.jpg"
        )
        dst_path = os.path.join(sample_dir, dst_name)
        if _copy_file_if_exists(choice.get("collage_path"), dst_path):
            copied_choice_paths.append(dst_path)

    sample_payload = {
        "frame_index": int(frame_index),
        "attempt_index": int(action.get("attempt_index", 0)),
        "issue_ids": [issue.get("issue_id") for issue in frame_issues],
        "replace_obj_ids": sorted(replace_obj_ids),
        "remove_obj_ids": sorted(int(x) for x in action.get("remove_obj_ids", [])),
        "updated_frame_indices": action.get("updated_frame_indices", []),
        "verification": verification,
        "choices": attempt_choices,
        "before_after_path": before_after_path,
        "choice_collages": copied_choice_paths,
    }
    _write_json(os.path.join(sample_dir, "sample.json"), sample_payload)
    return sample_payload


def _verify_frame_cluster(
    *,
    frame_index: int,
    frame_issues: list[dict[str, Any]],
    results_by_frame: dict[int, dict[str, Any]],
    video_path: str,
    send_generate_request_fn: Callable[[list[dict[str, Any]]], str | None],
    initial_text_prompt: str,
    prompt_profile: str,
    output_dir: str,
    overlap_iou_threshold: float,
) -> dict[str, Any]:
    verify_dir = os.path.join(output_dir, f"verify_{frame_index:05d}")
    os.makedirs(verify_dir, exist_ok=True)

    assessments: list[dict[str, Any]] = []
    requests: list[dict[str, Any]] = []
    bad_frame_indices: list[int] = []
    center_assessment: dict[str, Any] | None = None

    verify_indices = [idx for idx in [frame_index - 1, frame_index, frame_index + 1] if idx >= 0]
    for verify_index in verify_indices:
        if verify_index not in results_by_frame:
            continue
        frame_bgr = read_video_frame(video_path, verify_index)
        if frame_bgr is None:
            continue
        context_frames: list[tuple[int, np.ndarray]] = []
        for ctx_index in [verify_index - 1, verify_index, verify_index + 1]:
            if ctx_index < 0:
                continue
            ctx_frame = read_video_frame(video_path, ctx_index)
            if ctx_frame is not None:
                context_frames.append((ctx_index, ctx_frame))
        assessment, request_entry = assess_postprop_frame_with_mllm(
            frame_index=verify_index,
            frame_bgr=frame_bgr,
            context_frames=context_frames,
            outputs=results_by_frame.get(verify_index, {}),
            send_generate_request_fn=send_generate_request_fn,
            initial_text_prompt=initial_text_prompt,
            collage_output_dir=verify_dir,
            max_object_crops=6,
            crop_context_ratio=0.25,
            collage_cols=4,
            collage_tile_max_edge=280,
            overlap_iou_threshold=float(overlap_iou_threshold),
            system_prompt=_resolve_system_prompt(prompt_profile, None),
            max_json_retries=1,
        )
        assessments.append(assessment)
        requests.append(request_entry)
        if int(verify_index) == int(frame_index):
            center_assessment = assessment
        if assessment.get("is_bad_frame"):
            bad_frame_indices.append(int(verify_index))

    broad_center_frame_bad = bool(center_assessment and center_assessment.get("is_bad_frame"))
    center_assessment_missing = center_assessment is None
    raw_video_invalid = _assessment_is_raw_video_invalid(center_assessment)
    issue_results = [
        _evaluate_issue_resolution(issue, center_assessment) for issue in frame_issues
    ]
    unresolved_issue_ids = [
        row["issue_id"] for row in issue_results if not bool(row.get("resolved"))
    ]
    center_frame_bad = bool(center_assessment_missing or raw_video_invalid or unresolved_issue_ids)
    neighbor_bad_frame_indices = sorted(
        idx for idx in set(bad_frame_indices) if int(idx) != int(frame_index)
    )

    return {
        "frame_index": int(frame_index),
        "verify_indices": verify_indices,
        "bad_frame_indices": sorted(set(bad_frame_indices)),
        "broad_center_frame_bad": bool(broad_center_frame_bad),
        "center_frame_bad": bool(center_frame_bad),
        "center_assessment_missing": bool(center_assessment_missing),
        "raw_video_invalid": bool(raw_video_invalid),
        "neighbor_bad_frame_indices": neighbor_bad_frame_indices,
        "issue_results": issue_results,
        "unresolved_issue_ids": unresolved_issue_ids,
        "verification_passed": (
            not bool(center_assessment_missing)
            and not bool(raw_video_invalid)
            and not bool(unresolved_issue_ids)
        ),
        "assessments": assessments,
        "requests": requests,
    }


def repair_postprop_failures(
    *,
    video_path: str,
    backend: Any,
    image_processor: Any,
    send_generate_request_fn: Callable[[list[dict[str, Any]]], str | None],
    initial_text_prompt: str,
    prompt_profile: str,
    results_by_frame: dict[int, dict[str, Any]],
    postprop_report: dict[str, Any],
    hard_invalid_frame_indices: list[int],
    next_obj_id: int,
    total_frames: int,
    frame_size_hw: tuple[int, int],
    output_dir: str,
    image_size: int,
    repair_window: int = 8,
    max_attempts_per_frame: int = 2,
    max_candidates_per_issue: int = 5,
    chooser_prompt_template_path: str | None = None,
    verify_with_qa: bool = True,
    overlap_iou_threshold: float = 0.70,
) -> dict[str, Any]:
    os.makedirs(output_dir, exist_ok=True)
    issues, raw_video_invalid_frame_indices = build_repair_issues(
        postprop_report=postprop_report,
        hard_invalid_frame_indices=set(int(x) for x in hard_invalid_frame_indices),
    )
    issues_by_frame: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for issue in issues:
        issues_by_frame[int(issue["frame_index"])].append(issue)

    candidate_sets_log: list[dict[str, Any]] = []
    choices_log: list[dict[str, Any]] = []
    actions_log: list[dict[str, Any]] = []
    verification_log: list[dict[str, Any]] = []
    visual_debug_samples: list[dict[str, Any]] = []
    repaired_frame_indices: set[int] = set()
    unresolved_mask_issue_frame_indices: set[int] = set()
    next_obj_id_local = int(next_obj_id)
    visual_debug_dir = os.path.join(output_dir, "visual_debug")
    os.makedirs(visual_debug_dir, exist_ok=True)

    for frame_index in sorted(issues_by_frame.keys()):
        frame_issues = issues_by_frame[frame_index]
        original_snapshot = _snapshot_window(results_by_frame, frame_index, repair_window)
        frame_repaired = False
        frame_attempt_logs: list[dict[str, Any]] = []

        for attempt_index in range(max(1, int(max_attempts_per_frame))):
            _restore_window(results_by_frame, original_snapshot, frame_index, repair_window)
            selected_masks_by_obj_id: dict[int, np.ndarray] = {}
            remove_obj_ids: set[int] = set()
            tentative_new_obj_id = next_obj_id_local
            attempt_ok = True
            attempt_has_noncurrent_selection = False
            attempt_choices: list[dict[str, Any]] = []

            for issue in frame_issues:
                try:
                    candidate_bundle = generate_mask_candidates_for_issue(
                        video_path=video_path,
                        frame_index=frame_index,
                        issue=issue,
                        results_by_frame=results_by_frame,
                        image_processor=image_processor,
                        initial_text_prompt=initial_text_prompt,
                        prompt_profile=prompt_profile,
                        max_candidates=max_candidates_per_issue,
                        output_dir=os.path.join(output_dir, "candidates"),
                        search_radius=max(2, int(repair_window)),
                    )
                except Exception as exc:
                    candidate_sets_log.append(
                        {
                            "issue_id": issue["issue_id"],
                            "frame_index": int(frame_index),
                            "attempt_index": int(attempt_index),
                            "error": str(exc).strip() or repr(exc),
                            "stage": "candidate_generation",
                        }
                    )
                    attempt_ok = False
                    break
                candidate_sets_log.append(
                    {
                        "issue_id": issue["issue_id"],
                        "frame_index": int(frame_index),
                        "attempt_index": int(attempt_index),
                        "candidate_summary": candidate_bundle["candidate_summary"],
                    }
                )

                target_obj_ids = issue.get("target_obj_ids", []) or ["new_object"]
                for target_obj_id in target_obj_ids:
                    try:
                        choice = choose_candidate_with_mllm(
                            issue=issue,
                            target_obj_id=target_obj_id,
                            candidate_bundle=candidate_bundle,
                            results_by_frame=results_by_frame,
                            video_path=video_path,
                            send_generate_request_fn=send_generate_request_fn,
                            initial_text_prompt=initial_text_prompt,
                            prompt_profile=prompt_profile,
                            prompt_template_path=chooser_prompt_template_path,
                            output_dir=os.path.join(output_dir, "choices"),
                            search_radius=max(2, int(repair_window)),
                            max_json_retries=2,
                        )
                    except Exception as exc:
                        choices_log.append(
                            {
                                "issue_id": issue["issue_id"],
                                "frame_index": int(frame_index),
                                "attempt_index": int(attempt_index),
                                "target_obj_id": target_obj_id,
                                "error": str(exc).strip() or repr(exc),
                                "stage": "chooser",
                            }
                        )
                        attempt_ok = False
                        break
                    choices_log.append(dict(choice, attempt_index=int(attempt_index)))
                    attempt_choices.append(dict(choice, attempt_index=int(attempt_index)))

                    if choice["decision"] == "unrepairable":
                        attempt_ok = False
                        break

                    selected_candidate_id = choice["selected_candidate_id"]
                    rows = candidate_bundle["candidate_sets"][target_obj_id]
                    selected_row = next(
                        (row for row in rows if row["candidate_id"] == selected_candidate_id),
                        None,
                    )
                    if selected_row is None:
                        attempt_ok = False
                        break

                    if target_obj_id == "new_object":
                        if selected_candidate_id == "current":
                            attempt_ok = False
                            break
                        attempt_has_noncurrent_selection = True
                        selected_masks_by_obj_id[tentative_new_obj_id] = selected_row["mask_full"]
                        tentative_new_obj_id += 1
                    else:
                        if selected_candidate_id == "remove_mask" or choice["decision"] == "remove_mask":
                            attempt_has_noncurrent_selection = True
                            remove_obj_ids.add(int(target_obj_id))
                            selected_masks_by_obj_id.pop(int(target_obj_id), None)
                            continue
                        if selected_candidate_id == "current" and choice["decision"] == "keep_current":
                            current_mask = _frame_mask_by_obj_id(
                                results_by_frame,
                                frame_index,
                                int(target_obj_id),
                                frame_size_hw[0],
                                frame_size_hw[1],
                            )
                            if current_mask is not None:
                                selected_masks_by_obj_id[int(target_obj_id)] = current_mask
                        else:
                            attempt_has_noncurrent_selection = True
                            selected_masks_by_obj_id[int(target_obj_id)] = selected_row["mask_full"]

                if not attempt_ok:
                    break

            if attempt_ok and (selected_masks_by_obj_id or remove_obj_ids):
                if not attempt_has_noncurrent_selection:
                    actions_log.append(
                        {
                            "frame_index": int(frame_index),
                            "attempt_index": int(attempt_index),
                            "replace_obj_ids": sorted(
                                int(x) for x in selected_masks_by_obj_id.keys()
                            ),
                            "skipped": True,
                            "skip_reason": "all_selected_current",
                        }
                    )
                    break
                try:
                    action = apply_repair_action(
                        backend=backend,
                        video_path=video_path,
                        results_by_frame=results_by_frame,
                        frame_index=frame_index,
                        frame_size_wh=(frame_size_hw[1], frame_size_hw[0]),
                        image_size=image_size,
                        selected_masks_by_obj_id=selected_masks_by_obj_id,
                        remove_obj_ids=remove_obj_ids,
                        repair_window=repair_window,
                    )
                except Exception as exc:
                    actions_log.append(
                        {
                            "frame_index": int(frame_index),
                            "attempt_index": int(attempt_index),
                            "replace_obj_ids": sorted(
                                int(x) for x in selected_masks_by_obj_id.keys()
                            ),
                            "error": str(exc).strip() or repr(exc),
                            "stage": "apply_repair_action",
                        }
                    )
                    continue
                action["attempt_index"] = int(attempt_index)
                actions_log.append(action)
                frame_attempt_logs.append(action)

                if verify_with_qa:
                    try:
                        verification = _verify_frame_cluster(
                            frame_index=frame_index,
                            frame_issues=frame_issues,
                            results_by_frame=results_by_frame,
                            video_path=video_path,
                            send_generate_request_fn=send_generate_request_fn,
                            initial_text_prompt=initial_text_prompt,
                            prompt_profile=prompt_profile,
                            output_dir=os.path.join(output_dir, "verify"),
                            overlap_iou_threshold=overlap_iou_threshold,
                        )
                    except Exception as exc:
                        verification_log.append(
                            {
                                "frame_index": int(frame_index),
                                "attempt_index": int(attempt_index),
                                "error": str(exc).strip() or repr(exc),
                                "stage": "verify",
                            }
                        )
                        continue
                    verification["attempt_index"] = int(attempt_index)
                    verification_log.append(verification)
                    visual_sample = _write_visual_debug_sample(
                        visual_debug_dir=visual_debug_dir,
                        sample_index=len(visual_debug_samples),
                        video_path=video_path,
                        frame_index=frame_index,
                        frame_issues=frame_issues,
                        before_results_by_frame=original_snapshot,
                        after_results_by_frame=results_by_frame,
                        action=action,
                        attempt_choices=attempt_choices,
                        verification=verification,
                        repair_window=repair_window,
                    )
                    if visual_sample is not None:
                        visual_debug_samples.append(visual_sample)
                    if verification.get("verification_passed", False):
                        frame_repaired = True
                        repaired_frame_indices.update(action["updated_frame_indices"])
                        next_obj_id_local = tentative_new_obj_id
                        break
                else:
                    visual_sample = _write_visual_debug_sample(
                        visual_debug_dir=visual_debug_dir,
                        sample_index=len(visual_debug_samples),
                        video_path=video_path,
                        frame_index=frame_index,
                        frame_issues=frame_issues,
                        before_results_by_frame=original_snapshot,
                        after_results_by_frame=results_by_frame,
                        action=action,
                        attempt_choices=attempt_choices,
                        verification=None,
                        repair_window=repair_window,
                    )
                    if visual_sample is not None:
                        visual_debug_samples.append(visual_sample)
                    frame_repaired = True
                    repaired_frame_indices.update(action["updated_frame_indices"])
                    next_obj_id_local = tentative_new_obj_id
                    break

        if not frame_repaired:
            _restore_window(results_by_frame, original_snapshot, frame_index, repair_window)
            unresolved_mask_issue_frame_indices.add(int(frame_index))

    final_invalid_frame_indices = sorted(
        set(int(x) for x in hard_invalid_frame_indices)
        | set(int(x) for x in raw_video_invalid_frame_indices)
    )

    visual_manifest = {
        "sample_count": len(visual_debug_samples),
        "samples": visual_debug_samples,
    }
    _write_json(os.path.join(visual_debug_dir, "manifest.json"), visual_manifest)

    report = {
        "mode": "postprop_repair",
        "total_frames": int(total_frames),
        "repair_window": int(repair_window),
        "max_attempts_per_frame": int(max_attempts_per_frame),
        "max_candidates_per_issue": int(max_candidates_per_issue),
        "num_issues": len(issues),
        "issues": issues,
        "candidate_sets": candidate_sets_log,
        "choices": choices_log,
        "actions": actions_log,
        "verification": verification_log,
        "repaired_frame_indices": sorted(repaired_frame_indices),
        "unresolved_frame_indices": sorted(unresolved_mask_issue_frame_indices),
        "mask_issue_frame_indices": sorted(unresolved_mask_issue_frame_indices),
        "raw_video_invalid_frame_indices": sorted(raw_video_invalid_frame_indices),
        "final_invalid_frame_indices": final_invalid_frame_indices,
        "visual_debug_dir": visual_debug_dir,
        "visual_debug_manifest_path": os.path.join(visual_debug_dir, "manifest.json"),
        "next_obj_id": int(next_obj_id_local),
    }
    _write_json(os.path.join(output_dir, "postprop_repair_report.json"), report)
    return report
