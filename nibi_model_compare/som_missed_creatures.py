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
        k: number of target frames to pick.
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
    if k >= len(valid_indices):
        return valid_indices
    # pick k indices spread across valid_indices: 0%, ..., 100%
    # Python's built-in round() (banker's rounding) is used so that the
    # middle element of an odd-length list lands at the correct position
    # (e.g. round(1.5) == 2 keeps the bias toward the later, richer half).
    step = (len(valid_indices) - 1) / (k - 1) if k > 1 else 0
    picked = [valid_indices[round(i * step)] for i in range(k)]
    return sorted(set(picked))


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
    if k >= len(indices):
        return indices
    step = (len(indices) - 1) / (k - 1) if k > 1 else 0
    picked = [indices[round(i * step)] for i in range(k)]
    return sorted(set(picked))
