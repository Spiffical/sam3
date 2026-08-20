"""Deterministic assembly and filtering for SAM3 text-prompt proposals."""

from __future__ import annotations

from typing import Any, Iterable

import numpy as np

from .helpers.mask_overlap_removal import (
    _decode_single_mask,
    mask_utils,
    remove_overlapping_masks,
)


def parse_normalized_region(value: str) -> tuple[float, float, float, float]:
    """Parse ``x1,y1,x2,y2`` normalized coordinates."""
    try:
        coords = tuple(float(item.strip()) for item in value.split(","))
    except ValueError as exc:
        raise ValueError(f"invalid normalized region {value!r}") from exc
    if len(coords) != 4:
        raise ValueError(
            f"normalized region must contain four comma-separated values: {value!r}"
        )
    x1, y1, x2, y2 = coords
    if not (0.0 <= x1 < x2 <= 1.0 and 0.0 <= y1 < y2 <= 1.0):
        raise ValueError(
            "normalized region must satisfy 0 <= x1 < x2 <= 1 and "
            f"0 <= y1 < y2 <= 1: {value!r}"
        )
    return x1, y1, x2, y2


def bbox_region_overlap_fraction(
    box_xywh: Iterable[float], region_xyxy: Iterable[float]
) -> float:
    """Return the fraction of a normalized proposal box inside a region."""
    x, y, width, height = (float(value) for value in box_xywh)
    rx1, ry1, rx2, ry2 = (float(value) for value in region_xyxy)
    area = max(0.0, width) * max(0.0, height)
    if area <= 0.0:
        return 0.0
    x2 = x + width
    y2 = y + height
    intersection_width = max(0.0, min(x2, rx2) - max(x, rx1))
    intersection_height = max(0.0, min(y2, ry2) - max(y, ry1))
    return (intersection_width * intersection_height) / area


def bbox_iom_xywh(box_a: Iterable[float], box_b: Iterable[float]) -> float:
    """Intersection divided by the smaller normalized bounding-box area."""
    ax1, ay1, aw, ah = (float(value) for value in box_a)
    bx1, by1, bw, bh = (float(value) for value in box_b)
    area_a = max(0.0, aw) * max(0.0, ah)
    area_b = max(0.0, bw) * max(0.0, bh)
    min_area = min(area_a, area_b)
    if min_area <= 0.0:
        return 0.0
    intersection_width = max(0.0, min(ax1 + aw, bx1 + bw) - max(ax1, bx1))
    intersection_height = max(0.0, min(ay1 + ah, by1 + bh) - max(ay1, by1))
    return (intersection_width * intersection_height) / min_area


def _select_indices(outputs: dict[str, Any], indices: list[int]) -> dict[str, Any]:
    selected = dict(outputs)
    for key in ("pred_boxes", "pred_scores", "pred_masks"):
        values = list(outputs.get(key, []))
        selected[key] = [values[index] for index in indices]
    return selected


def suppress_exclusion_regions(
    outputs: dict[str, Any],
    regions: list[tuple[float, float, float, float]],
    *,
    overlap_fraction: float = 0.80,
) -> tuple[dict[str, Any], list[int]]:
    """Drop proposals whose normalized box lies mostly inside a known overlay."""
    if not 0.0 <= overlap_fraction <= 1.0:
        raise ValueError("overlap_fraction must be between 0 and 1")
    boxes = list(outputs.get("pred_boxes", []))
    if not regions or not boxes:
        return _select_indices(outputs, list(range(len(boxes)))), []

    removed = [
        index
        for index, box in enumerate(boxes)
        if any(
            bbox_region_overlap_fraction(box, region) >= overlap_fraction
            for region in regions
        )
    ]
    removed_set = set(removed)
    kept = [index for index in range(len(boxes)) if index not in removed_set]
    return _select_indices(outputs, kept), removed


def merge_prompt_fragments(
    outputs: dict[str, Any],
    source_prompts: list[str],
    merge_prompts: list[str],
    *,
    bbox_iom_threshold: float = 0.15,
) -> tuple[dict[str, Any], list[str], list[list[int]]]:
    """Stitch bbox-overlapping fragments for explicitly configured prompts.

    SAM3 occasionally returns vertically adjacent, non-overlapping pieces of one
    organism. This heuristic is deliberately opt-in per evidence-backed prompt;
    applying it to every class could merge distinct crowded organisms.
    """
    if not 0.0 <= bbox_iom_threshold <= 1.0:
        raise ValueError("bbox_iom_threshold must be between 0 and 1")
    count = len(outputs.get("pred_masks", []))
    if len(source_prompts) != count:
        raise ValueError("source_prompts must align with pred_masks")
    enabled = {prompt.casefold() for prompt in merge_prompts}
    if count <= 1 or not enabled:
        return _select_indices(outputs, list(range(count))), list(source_prompts), []

    boxes = list(outputs.get("pred_boxes", []))
    parent = list(range(count))

    def find(index: int) -> int:
        while parent[index] != index:
            parent[index] = parent[parent[index]]
            index = parent[index]
        return index

    def union(left: int, right: int) -> None:
        left_root = find(left)
        right_root = find(right)
        if left_root != right_root:
            parent[right_root] = left_root

    for left in range(count):
        if source_prompts[left].casefold() not in enabled:
            continue
        for right in range(left + 1, count):
            if source_prompts[left].casefold() != source_prompts[right].casefold():
                continue
            if bbox_iom_xywh(boxes[left], boxes[right]) >= bbox_iom_threshold:
                union(left, right)

    components: dict[int, list[int]] = {}
    for index in range(count):
        components.setdefault(find(index), []).append(index)
    ordered_groups = sorted(components.values(), key=lambda group: min(group))
    merged_groups = [group for group in ordered_groups if len(group) > 1]
    if not merged_groups:
        return _select_indices(outputs, list(range(count))), list(source_prompts), []
    if mask_utils is None:
        raise ImportError("pycocotools is required to merge RLE mask fragments")

    height = int(outputs["orig_img_h"])
    width = int(outputs["orig_img_w"])
    merged = {
        **outputs,
        "pred_boxes": [],
        "pred_scores": [],
        "pred_masks": [],
    }
    merged_sources: list[str] = []
    scores = list(outputs.get("pred_scores", []))
    masks = list(outputs.get("pred_masks", []))
    for group in ordered_groups:
        if len(group) == 1:
            index = group[0]
            merged["pred_boxes"].append(boxes[index])
            merged["pred_scores"].append(scores[index])
            merged["pred_masks"].append(masks[index])
            merged_sources.append(source_prompts[index])
            continue

        binary = np.logical_or.reduce(
            [_decode_single_mask(masks[index], height, width) > 0 for index in group]
        )
        encoded = mask_utils.encode(np.asfortranarray(binary.astype(np.uint8)))
        counts = encoded["counts"]
        if isinstance(counts, bytes):
            counts = counts.decode("utf-8")
        ys, xs = np.nonzero(binary)
        x1 = int(xs.min())
        y1 = int(ys.min())
        x2 = int(xs.max()) + 1
        y2 = int(ys.max()) + 1
        merged["pred_boxes"].append(
            [x1 / width, y1 / height, (x2 - x1) / width, (y2 - y1) / height]
        )
        merged["pred_scores"].append(max(float(scores[index]) for index in group))
        merged["pred_masks"].append(counts)
        merged_sources.append(source_prompts[group[0]])
    return merged, merged_sources, merged_groups


def build_proposal_union(
    prompt_outputs: list[tuple[str, dict[str, Any]]],
    *,
    image_path: str,
    exclusion_regions: list[tuple[float, float, float, float]] | None = None,
    exclusion_overlap_fraction: float = 0.80,
    iom_threshold: float = 0.30,
    fragment_merge_prompts: list[str] | None = None,
    fragment_bbox_iom_threshold: float = 0.15,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Union prompt outputs, suppress overlays, and geometrically deduplicate."""
    if not prompt_outputs:
        raise ValueError("prompt_outputs must not be empty")
    first = prompt_outputs[0][1]
    height = int(first["orig_img_h"])
    width = int(first["orig_img_w"])
    merged: dict[str, Any] = {
        "original_image_path": image_path,
        "orig_img_h": height,
        "orig_img_w": width,
        "pred_boxes": [],
        "pred_scores": [],
        "pred_masks": [],
    }
    source_prompts: list[str] = []
    counts_by_prompt: dict[str, int] = {}
    for prompt, outputs in prompt_outputs:
        if int(outputs["orig_img_h"]) != height or int(outputs["orig_img_w"]) != width:
            raise ValueError("proposal outputs do not share one image size")
        count = len(outputs.get("pred_masks", []))
        counts_by_prompt[prompt] = count
        for key in ("pred_boxes", "pred_scores", "pred_masks"):
            merged[key].extend(list(outputs.get(key, [])))
        source_prompts.extend([prompt] * count)

    raw_count = len(merged["pred_masks"])
    filtered, overlay_removed = suppress_exclusion_regions(
        merged,
        list(exclusion_regions or []),
        overlap_fraction=exclusion_overlap_fraction,
    )
    overlay_removed_set = set(overlay_removed)
    filtered_sources = [
        prompt
        for index, prompt in enumerate(source_prompts)
        if index not in overlay_removed_set
    ]

    fragment_merged, fragment_sources, fragment_groups = merge_prompt_fragments(
        filtered,
        filtered_sources,
        list(fragment_merge_prompts or []),
        bbox_iom_threshold=fragment_bbox_iom_threshold,
    )
    deduplicated = remove_overlapping_masks(
        fragment_merged, iom_thresh=iom_threshold
    )
    kept_indices = list(
        deduplicated.get("kept_indices", range(len(deduplicated["pred_masks"])))
    )
    deduplicated_sources = [fragment_sources[index] for index in kept_indices]
    for key in ("kept_indices", "removed_indices", "iom_threshold"):
        deduplicated.pop(key, None)

    report = {
        "prompts": [prompt for prompt, _ in prompt_outputs],
        "counts_by_prompt": counts_by_prompt,
        "raw_proposal_count": raw_count,
        "overlay_removed_count": len(overlay_removed),
        "overlay_removed_indices_zero_based": overlay_removed,
        "post_overlay_count": len(filtered["pred_masks"]),
        "fragment_merge_prompts": list(fragment_merge_prompts or []),
        "fragment_bbox_iom_threshold": float(fragment_bbox_iom_threshold),
        "fragment_merge_groups_zero_based": fragment_groups,
        "post_fragment_merge_count": len(fragment_merged["pred_masks"]),
        "deduplicated_count": len(deduplicated["pred_masks"]),
        "deduplicated_source_prompts": deduplicated_sources,
        "exclusion_regions_xyxy_normalized": [
            list(region) for region in (exclusion_regions or [])
        ],
        "exclusion_overlap_fraction": float(exclusion_overlap_fraction),
        "iom_threshold": float(iom_threshold),
    }
    return deduplicated, report
