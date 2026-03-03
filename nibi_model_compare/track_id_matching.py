from __future__ import annotations

from typing import Any

import numpy as np


def mask_iou(mask_a: np.ndarray, mask_b: np.ndarray) -> float:
    a = np.asarray(mask_a).astype(bool)
    b = np.asarray(mask_b).astype(bool)
    inter = float(np.logical_and(a, b).sum())
    if inter <= 0.0:
        return 0.0
    union = float(np.logical_or(a, b).sum())
    if union <= 0.0:
        return 0.0
    return inter / union


def assign_object_ids_by_iou(
    existing_masks_with_ids: list[tuple[int, np.ndarray]],
    new_masks: list[np.ndarray],
    *,
    next_obj_id: int,
    iou_match_threshold: float = 0.30,
) -> dict[str, Any]:
    """
    Assign object IDs to new masks by matching against existing masks on the same frame.
    Reuses IDs for strong IoU matches, otherwise allocates a fresh ID.
    """
    assigned_ids: list[int] = []
    assignments: list[dict[str, Any]] = []
    used_existing_ids: set[int] = set()

    for new_idx, new_mask in enumerate(new_masks):
        best_existing_id: int | None = None
        best_iou = 0.0

        for existing_id, existing_mask in existing_masks_with_ids:
            if existing_id in used_existing_ids:
                continue
            iou = mask_iou(existing_mask, new_mask)
            if iou > best_iou:
                best_iou = iou
                best_existing_id = int(existing_id)

        if best_existing_id is not None and best_iou >= iou_match_threshold:
            obj_id = best_existing_id
            used_existing_ids.add(obj_id)
            is_new_id = False
        else:
            obj_id = int(next_obj_id)
            next_obj_id += 1
            is_new_id = True

        assigned_ids.append(obj_id)
        assignments.append(
            {
                "new_mask_index": int(new_idx),
                "assigned_obj_id": int(obj_id),
                "matched_existing_obj_id": (
                    int(best_existing_id) if best_existing_id is not None else None
                ),
                "best_iou": float(best_iou),
                "is_new_obj_id": bool(is_new_id),
            }
        )

    return {
        "assigned_ids": assigned_ids,
        "assignments": assignments,
        "next_obj_id": int(next_obj_id),
    }
