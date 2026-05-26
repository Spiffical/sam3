"""Set-of-Mark missed-creature discovery stage.

Given an existing per-frame text-agent run, on a small set of selected
target frames: produce dense SAM3 candidates, drop the ones already
covered, overlay numbered marks, ask the MLLM (with a few unmarked
reference frames as context) which marks are real biological subjects,
and write the accepted masks back into the per-frame outputs.

The MLLM never emits raw (x, y) coordinates -- it selects by mark id.
This sidesteps the failure mode that killed the prior point-proposal
loop (commit ae72a4c).

Spec: docs/superpowers/specs/2026-05-26-som-missed-creature-loop-design.md
"""

from __future__ import annotations

import json
import re

_ANSWER_RE = re.compile(r"<answer>\s*(.*?)\s*</answer>", re.DOTALL)


def parse_som_response(text: str, *, valid_ids: set[int] | None = None) -> list[int]:
    """Extract accepted mark ids from an MLLM response.

    Contract: the response is expected to contain a trailing
    ``<answer>{"accepted_marks": [<int>, ...]}</answer>`` block. Free-text
    reasoning may appear before the block but must not appear after.
    If multiple blocks appear we accept the last one (the MLLM's final
    answer). If ``valid_ids`` is provided, out-of-range ids are dropped.

    Returns an empty list if the response is missing the tag entirely or
    has a malformed payload -- callers should treat that as "no
    creatures accepted" rather than an error.

    The function is lenient on bad input: non-string ``text`` (including
    ``None``) returns an empty list rather than raising. Callers that have
    just gotten a model response that might be ``None`` due to upstream
    error get a safe default instead of an exception.
    """
    if not isinstance(text, str) or not text:
        return []
    matches = _ANSWER_RE.findall(text)
    if not matches:
        return []
    try:
        payload = json.loads(matches[-1])
    except json.JSONDecodeError:
        return []
    raw = payload.get("accepted_marks") if isinstance(payload, dict) else None
    if not isinstance(raw, list):
        return []
    out: list[int] = []
    for item in raw:
        if isinstance(item, bool):
            continue
        if not isinstance(item, int):
            continue
        if valid_ids is not None and item not in valid_ids:
            continue
        out.append(item)
    return out


def _uniform_pick(indices: list[int], k: int) -> list[int]:
    """Pick ``k`` indices uniformly spaced across ``indices``. Returns a
    sorted, deduplicated list. If ``k >= len(indices)`` returns all
    indices; ``k == 1`` returns the first.
    """
    if k >= len(indices):
        return sorted(indices)
    step = (len(indices) - 1) / (k - 1) if k > 1 else 0
    picked = [indices[round(i * step)] for i in range(k)]
    return sorted(set(picked))


def select_target_frames(
    frame_results: list[dict],
    *,
    strategy: str = "uniform",
    k: int = 8,
    explicit: list[int] | None = None,
    include_zero_mask_frames: bool = False,
    _motion_selector=None,
) -> list[int]:
    """Pick target frame indices for the SoM stage to run on.

    Args:
        frame_results: rows from ``frame_results.jsonl`` (per-frame agent results).
        strategy: ``"uniform"`` (default, K-evenly-spaced) or ``"motion"``
            (delegates to nibi_model_compare.keyframe_discovery).
        k: number of target frames to pick. When ``k == 1`` the FIRST
            valid frame is returned; callers that want a middle-pick
            should pass ``k=1`` only when frame ordering is irrelevant
            or should construct the index themselves.
        explicit: if given, return this list filtered to indices that appear
            in ``frame_results``. Overrides ``strategy`` and ``k``.
        include_zero_mask_frames: if True, frames where the text-agent
            produced 0 masks are still candidates. Default False.
        _motion_selector: test seam; if given, used in place of the motion
            keyframe-discovery helper.

    Returns: sorted list of selected frame indices.
    """
    all_indices = {int(row["frame_index"]) for row in frame_results}

    if explicit is not None:
        return sorted(idx for idx in explicit if idx in all_indices)

    if strategy not in ("uniform", "motion"):
        raise ValueError(
            f"Unknown strategy {strategy!r}; expected 'uniform' or 'motion'."
        )

    valid_rows = [
        row for row in frame_results
        if not row.get("error")
        and not row.get("skipped")
        and (include_zero_mask_frames or (row.get("num_masks") or 0) > 0)
    ]
    valid_indices = sorted(int(row["frame_index"]) for row in valid_rows)

    if not valid_indices:
        return []

    if strategy == "motion":
        selector = _motion_selector or _motion_keyframe_selector
        return sorted(selector(valid_rows, k))

    # uniform spacing
    # Python's built-in round() (banker's rounding) is used so that the
    # middle element of an odd-length list lands at the correct position
    # (e.g. round(1.5) == 2 keeps the bias toward the later, richer half).
    return _uniform_pick(valid_indices, k)


def _motion_keyframe_selector(valid_rows: list[dict], k: int) -> list[int]:
    """Delegate motion-based selection to the existing keyframe-discovery helper.

    NOTE: ``keyframe_discovery.discover_keyframes_from_motion`` requires a
    video file path and uses OpenCV optical-flow internally -- it cannot
    accept a pre-filtered list of frame indices.  Since the only public API
    in that module needs the raw video, we fall back to uniform spacing here
    so that callers using ``strategy="motion"`` without a custom
    ``_motion_selector`` still get a reasonable (deterministic) result.
    If a true motion-based selector is added to keyframe_discovery in the
    future, wire it up here.
    """
    indices = sorted(int(r["frame_index"]) for r in valid_rows)
    if not indices:
        return []
    return _uniform_pick(indices, k)


def _mask_iou(a, b) -> float:
    """IoU of two boolean numpy masks of the same shape."""
    import numpy as np

    a = np.asarray(a, dtype=bool)
    b = np.asarray(b, dtype=bool)
    inter = int(np.logical_and(a, b).sum())
    if inter == 0:
        return 0.0
    union = int(np.logical_or(a, b).sum())
    return inter / union


def _touches_edges(mask, edge_tol_px: int) -> int:
    """Number of frame edges (top/bottom/left/right) the mask touches
    within ``edge_tol_px`` pixels."""
    h, w = mask.shape
    t = edge_tol_px
    count = 0
    if mask[:t, :].any():
        count += 1
    if mask[h - t:, :].any():
        count += 1
    if mask[:, :t].any():
        count += 1
    if mask[:, w - t:].any():
        count += 1
    return count


def filter_candidates(
    candidates: list[dict],
    existing_masks: list[dict],
    *,
    iou_dedup: float,
    min_area_px: float,
    max_area_frac: float,
    edge_tol_px: int,
) -> list[dict]:
    """Filter dense SoM candidates.

    Drops:
      - candidates whose mask has IoU > iou_dedup against ANY existing mask
      - candidates with area < min_area_px
      - candidates with area > max_area_frac * (H * W)
      - candidates that touch the frame edge on more than one side

    Pass edge_tol_px=0 to disable edge filtering entirely.

    Returns the surviving candidates in the same order they were given.
    Callers that want the full pre-filter list with drop reasons should call
    ``filter_candidates_with_reasons`` instead (see below).
    """
    return [c for c, reason in filter_candidates_with_reasons(
        candidates, existing_masks,
        iou_dedup=iou_dedup, min_area_px=min_area_px,
        max_area_frac=max_area_frac, edge_tol_px=edge_tol_px,
    ) if reason is None]


def draw_numbered_marks(
    frame_bgr,
    candidates: list[dict],
    *,
    alpha: float = 0.35,
    palette_seed: int = 7,
):
    """Render numbered marks on top of a frame.

    Each candidate gets a translucent mask overlay (so the MLLM sees both
    the dot location and the implied shape) and a numeric label at the
    centroid. Marks are numbered 1..N in the order candidates are given
    (callers should pre-sort by descending area so larger objects get
    lower ids).

    Returns a new BGR (uint8) array of the same shape as the input frame.

    Color source: ColorPalette.default().by_idx(idx) -- wraps at palette
    length (~20 colours). ``palette_seed`` is ignored (kept for API
    stability) because ColorPalette.by_idx takes no seed argument; callers
    that need reproducible-but-different colours should wrap or subclass.
    """
    import cv2
    import numpy as np

    from sam3.agent.helpers.som_utils import ColorPalette

    out = frame_bgr.copy()
    if not candidates:
        return out

    # ColorPalette.by_idx(idx) -> Color; use .as_bgr() for cv2 compatibility.
    palette = ColorPalette.default()

    for idx, cand in enumerate(candidates, start=1):
        mask = np.asarray(cand["mask"], dtype=bool)
        color_obj = palette.by_idx(idx)          # Color dataclass (r, g, b)
        color_bgr = color_obj.as_bgr()           # (b, g, r) tuple of ints

        # Translucent mask overlay
        overlay = out.copy()
        overlay[mask] = (
            (1.0 - alpha) * out[mask].astype(np.float32)
            + alpha * np.asarray(color_bgr, dtype=np.float32)
        ).astype(np.uint8)
        out = overlay

        # Outline contour for crisp boundary
        contours, _ = cv2.findContours(
            mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        cv2.drawContours(out, contours, -1, [int(c) for c in color_bgr], 1)

        # Label at centroid, with shadow for readability
        ys, xs = np.where(mask)
        cy, cx = int(ys.mean()), int(xs.mean())
        label = str(idx)
        font = cv2.FONT_HERSHEY_SIMPLEX
        scale = 0.6
        thickness = 1
        (tw, th), _ = cv2.getTextSize(label, font, scale, thickness)
        org = (max(2, cx - tw // 2), max(th + 2, cy + th // 2))
        cv2.putText(out, label, org, font, scale, (0, 0, 0), thickness + 2, cv2.LINE_AA)
        cv2.putText(out, label, org, font, scale,
                    [int(c) for c in color_bgr], thickness, cv2.LINE_AA)

    return out


def filter_candidates_with_reasons(
    candidates: list[dict],
    existing_masks: list[dict],
    *,
    iou_dedup: float,
    min_area_px: float,
    max_area_frac: float,
    edge_tol_px: int,
) -> list[tuple[dict, str | None]]:
    """Like ``filter_candidates`` but returns ``(candidate, drop_reason)``
    for every input candidate so debug artefacts can log filter
    decisions. ``drop_reason`` is None for survivors.

    Pass edge_tol_px=0 to disable edge filtering entirely.
    """
    import numpy as np

    results: list[tuple[dict, str | None]] = []
    existing_arrays = [np.asarray(e["mask"], dtype=bool) for e in existing_masks]

    for cand in candidates:
        mask = np.asarray(cand["mask"], dtype=bool)
        h, w = mask.shape
        area = int(mask.sum())

        if area < min_area_px:
            results.append((cand, "too_small"))
            continue
        if area > max_area_frac * h * w:
            results.append((cand, "too_large"))
            continue
        if _touches_edges(mask, edge_tol_px) > 1:
            results.append((cand, "multi_edge_clipped"))
            continue

        dup = any(_mask_iou(mask, em) > iou_dedup for em in existing_arrays)
        if dup:
            results.append((cand, "duplicate_of_existing"))
            continue

        results.append((cand, None))

    return results
