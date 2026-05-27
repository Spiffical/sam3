"""Set-of-Mark missed-creature discovery stage.

Given an existing per-frame text-agent run, on a small set of selected
target frames: use the MLLM to identify what was MISSED by the
text-prompted detector and propose click coordinates for SAM3 to
segment.

Architecture:
  1. Render existing text-agent masks as a translucent overlay.
  2. Ask MLLM (with overlay + temporal context): identify missed
     creatures and propose click (x, y) coordinates in normalized space.
  3. For each proposed click run SAM3 image point-mode → candidate mask.
  4. Filter candidates (size, dedup) via filter_candidates_with_reasons.
  5. Render numbered marks on survivors; ask MLLM judge to accept/reject.
  6. Accepted masks are merged back into the per-frame output.

Spec: docs/superpowers/specs/2026-05-26-som-missed-creature-loop-design.md
"""

from __future__ import annotations

import json
import os
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
):
    """Render numbered marks on top of a frame.

    Each candidate gets a translucent mask overlay (so the MLLM sees both
    the dot location and the implied shape) and a numeric label at the
    centroid. Marks are numbered 1..N in the order candidates are given
    (callers should pre-sort by descending area so larger objects get
    lower ids).

    Returns a new BGR (uint8) array of the same shape as the input frame.

    Color source: ColorPalette.default().by_idx(idx) -- wraps at palette
    length (~20 colours).
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
        if not mask.any():
            continue  # nothing to render for an empty mask
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


def build_som_prompt_messages(
    *,
    system_prompt: str,
    target_image_path: str,
    neighbour_image_paths: list[str],
    initial_text_prompt: str,
    num_marks: int,
) -> list[dict]:
    """Construct the multi-image SoM prompt for an MLLM call.

    The structure mirrors the agent's existing message shape so the same
    clients (sam3.agent.client_claude, sam3.agent.client_llm) can dispatch
    it without modification:

        [
          {"role": "system", "content": "<system prompt text>"},
          {"role": "user",   "content": [
              {"type": "image", "image": "<target>"},
              {"type": "text",  "text":  "<framing for target>"},
              {"type": "image", "image": "<neighbour 1>"},
              ...
              {"type": "text",  "text":  "<the strict answer-format instruction>"},
          ]},
        ]

    Args:
        system_prompt: the loaded system-prompt body (already includes
            underwater addendum if applicable).
        target_image_path: path to the marked target frame.
        neighbour_image_paths: 0..N unmarked reference frames, given in
            chronological order. Empty list is fine.
        initial_text_prompt: the user's original creature query (e.g.
            "small creatures"), echoed into the prompt for context.
        num_marks: how many marks were drawn on the target. The MLLM
            uses this to bound its accepted-marks list.

    Returns: list of two message dicts.
    """
    if num_marks < 1:
        raise ValueError(
            f"num_marks must be >= 1; got {num_marks}. SoM has nothing to ask about."
        )

    target_blurb = (
        f"The first image is the target frame, annotated with numbered "
        f"marks 1..{num_marks} on candidate masks that SAM3 produced "
        f"and that the text-prompted agent did not already cover."
    )
    if neighbour_image_paths:
        target_blurb += (
            f" The following {len(neighbour_image_paths)} images are "
            "unmarked reference frames from times near the target. Use "
            "them as additional perspectives -- some creatures move and "
            "some are stationary; do not require motion to accept a "
            "mark. A persistent biological subject should be visible "
            "in the reference frames (perhaps with slight lighting or "
            "viewpoint shifts), while transient artefacts (floating "
            "debris, glare, lighting flashes) usually are not."
        )

    answer_instruction = (
        f"The original creature query is: '{initial_text_prompt}'. "
        "Decide which of the numbered marks correspond to real biological "
        "subjects matching that query. Respond with your reasoning in free "
        "text, then end your response with EXACTLY ONE tag of this form "
        "and nothing else after it:\n"
        '<answer>{"accepted_marks": [<int>, ...]}</answer>\n'
        f"Accepted-mark ids must be in the range 1..{num_marks}. An empty "
        "list is valid if you do not see any biological subjects."
    )

    user_content: list[dict] = [
        {"type": "image", "image": target_image_path},
        {"type": "text", "text": target_blurb},
    ]
    for nb_path in neighbour_image_paths:
        user_content.append({"type": "image", "image": nb_path})
    user_content.append({"type": "text", "text": answer_instruction})

    return [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_content},
    ]


def merge_accepted_masks_into_row(
    existing_row: dict, accepted_candidates: list[dict],
) -> dict:
    """Return a new frame row that appends accepted SoM candidates to the
    existing per-frame outputs without mutating the input.

    Accepted candidates get fresh ``obj_id``s above ``max(existing) + 1``
    (or starting at 1 if there are no existing objects). The new row
    carries:
      - ``source: "som"`` at the top level
      - ``added_obj_ids: [ids that this stage added]``
      - ``source_per_obj_id: {"<id>": "text_agent"|"som"}`` for every id
    """
    import copy

    from nibi_model_compare.frame_output_utils import encode_binary_mask_to_rle

    row = copy.deepcopy(existing_row)
    # Normalise to plain int so downstream json.dumps never sees numpy.int64.
    row["out_obj_ids"] = [int(oid) for oid in row.get("out_obj_ids", [])]
    existing_ids = row["out_obj_ids"]
    next_id = (int(max(existing_ids)) + 1) if existing_ids else 1

    added_ids: list[int] = []
    for cand in accepted_candidates:
        row["out_obj_ids"].append(next_id)
        row["out_binary_masks_rle"].append(
            encode_binary_mask_to_rle(cand["mask"])
        )
        row["out_boxes_xywh"].append(list(cand["bbox_xywh"]))
        row["out_probs"].append(float(cand.get("score", 0.0)))
        row["out_tracker_probs"].append(float(cand.get("score", 0.0)))
        added_ids.append(next_id)
        next_id += 1

    row["source"] = "som"
    row["added_obj_ids"] = added_ids
    added_set = set(added_ids)
    row["source_per_obj_id"] = {
        str(oid): ("som" if oid in added_set else "text_agent")
        for oid in row["out_obj_ids"]
    }
    return row


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


DEFAULT_BROAD_PROMPTS = ("creature", "animal", "organism")


def generate_dense_candidates(
    *,
    image_path: str,
    broad_prompts=DEFAULT_BROAD_PROMPTS,
    output_folder: str,
    internal_iou_dedup: float = 0.5,
    _call_sam_service=None,
) -> list:
    """DEPRECATED: broad text prompts produce mostly duplicates of what the
    text-agent already found. Use generate_click_based_candidates instead."""
    """Produce dense SoM candidates for one frame.

    MVP strategy: run SAM3 text-prompted inference once per broad prompt
    (e.g. "creature", "animal", "organism"), parse each result file,
    union the masks, and dedupe intra-batch by IoU.

    Each returned dict has ``{"mask": np.ndarray (bool), "bbox_xywh":
    [x, y, w, h], "score": float, "source_prompt": str}``.

    The contract is the only thing other components depend on; if smoke
    tests show poor recall, a real grid-sampled AMG can replace this
    body without touching downstream code.

    Production caller note: the default ``_call_sam_service`` resolves
    to ``sam3.agent.client_sam3.call_sam_service``, which takes a
    ``sam3_processor`` as its leading positional argument in real
    deployments. The driver in scripts/run_som_missed_creatures.py is
    responsible for pre-currying that argument (see how agent_core does
    it) before invoking this function. The injectable seam exists
    primarily for tests.
    """
    import json
    import os

    import numpy as np

    from nibi_model_compare.frame_output_utils import decode_rle_to_mask

    call_sam = _call_sam_service
    if call_sam is None:
        from sam3.agent.client_sam3 import call_sam_service as _live_call
        call_sam = _live_call

    os.makedirs(output_folder, exist_ok=True)

    pooled = []
    for prompt in broad_prompts:
        result_path = call_sam(
            image_path=image_path,
            text_prompt=prompt,
            output_folder_path=output_folder,
        )
        with open(result_path, "r", encoding="utf-8") as f:
            payload = json.load(f)
        h = payload.get("orig_img_h")
        w = payload.get("orig_img_w")
        if not isinstance(h, int) or not isinstance(w, int) or h <= 0 or w <= 0:
            raise ValueError(
                f"SAM3 result for prompt {prompt!r} is missing valid orig_img_h/orig_img_w "
                f"(got h={h!r}, w={w!r}); cannot decode masks."
            )
        rles = payload.get("pred_masks") or []
        boxes = payload.get("pred_boxes") or []
        scores = payload.get("pred_scores") or [0.0] * len(rles)
        if boxes and len(boxes) != len(rles):
            print(
                f"⚠️ SoM dense generator: pred_boxes length {len(boxes)} "
                f"does not match pred_masks length {len(rles)} for prompt "
                f"{prompt!r}; skipping this prompt's results."
            )
            continue
        if scores and len(scores) != len(rles):
            print(
                f"⚠️ SoM dense generator: pred_scores length {len(scores)} "
                f"does not match pred_masks length {len(rles)} for prompt "
                f"{prompt!r}; substituting zeros."
            )
            scores = [0.0] * len(rles)
        for rle, bbox, score in zip(rles, boxes, scores):
            mask = decode_rle_to_mask(rle, h, w).astype(bool)
            pooled.append({
                "mask": mask,
                "bbox_xywh": list(bbox),
                "score": float(score),
                "source_prompt": prompt,
            })

    # Intra-batch dedup by IoU: walk in descending score order, keep cand
    # only if it has no >iou_dedup overlap with any already-kept mask.
    pooled.sort(key=lambda c: -c["score"])
    kept = []
    for cand in pooled:
        if any(_mask_iou(cand["mask"], k["mask"]) > internal_iou_dedup
               for k in kept):
            continue
        kept.append(cand)
    return kept


def load_system_prompt(profile: str) -> str:
    """Load the SoM system prompt for the requested profile.

    ``profile`` is one of: 'underwater', 'general'. Other values raise
    ValueError.
    """
    valid = {"underwater", "general"}
    if profile not in valid:
        raise ValueError(
            f"Unknown profile '{profile}'. Expected one of: {sorted(valid)}."
        )
    here = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    # __file__ is nibi_model_compare/som_missed_creatures.py -> repo root
    path = os.path.join(
        here,
        "sam3", "agent", "system_prompts",
        f"system_prompt_som_missed_creature_{profile}.txt",
    )
    with open(path, "r", encoding="utf-8") as f:
        return f.read()


def load_click_discovery_system_prompt(profile: str) -> str:
    """Load the click-discovery step system prompt for the requested profile.

    ``profile`` is one of: 'underwater', 'general'. Other values raise
    ValueError. The file is:
        sam3/agent/system_prompts/system_prompt_som_click_discovery_<profile>.txt
    """
    valid = {"underwater", "general"}
    if profile not in valid:
        raise ValueError(
            f"Unknown profile '{profile}'. Expected one of: {sorted(valid)}."
        )
    here = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    path = os.path.join(
        here,
        "sam3", "agent", "system_prompts",
        f"system_prompt_som_click_discovery_{profile}.txt",
    )
    with open(path, "r", encoding="utf-8") as f:
        return f.read()


def render_existing_masks_overlay(
    frame_bgr,
    existing_masks: list[dict],
    alpha: float = 0.4,
):
    """Render existing text-agent masks as translucent green overlays so the
    MLLM can see what's already covered when judging what's missed."""
    import cv2
    import numpy as np

    out = frame_bgr.copy()
    for m in existing_masks:
        mask = np.asarray(m["mask"], dtype=bool)
        if not mask.any():
            continue
        green = np.array([0, 255, 0], dtype=np.float32)  # BGR
        out_f = out.astype(np.float32)
        out_f[mask] = (1 - alpha) * out_f[mask] + alpha * green
        out = out_f.astype(np.uint8)
        # Outline for crispness
        contours, _ = cv2.findContours(
            mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        cv2.drawContours(out, contours, -1, (0, 200, 0), 1)
    return out


def parse_click_proposals(text: str) -> list[dict]:
    """Extract list of {x: float, y: float, description: str} from
    <answer>{"missed_creatures": [{"x": ..., "y": ..., "description": ...}, ...]}</answer>.

    Returns [] for any malformed or missing tag (lenient -- same policy as
    parse_som_response). Coordinates are accepted only if both x and y
    are floats in [0, 1]; out-of-range entries are dropped silently.

    LEGACY single-click-per-creature contract.  The new grouped contract is
    implemented by ``parse_creature_click_groups``.  This function is kept for
    backwards-compatibility and as documentation of the prior format.
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
    raw = payload.get("missed_creatures") if isinstance(payload, dict) else None
    if not isinstance(raw, list):
        return []
    out = []
    for item in raw:
        if not isinstance(item, dict):
            continue
        x = item.get("x")
        y = item.get("y")
        if isinstance(x, bool) or not isinstance(x, (int, float)):
            continue
        if isinstance(y, bool) or not isinstance(y, (int, float)):
            continue
        if not (0.0 <= float(x) <= 1.0 and 0.0 <= float(y) <= 1.0):
            continue
        desc = item.get("description")
        if not isinstance(desc, str):
            desc = ""
        out.append({"x": float(x), "y": float(y), "description": desc})
    return out


def parse_creature_click_groups(text: str) -> list[dict]:
    """Parse the NEW grouped-click MLLM contract.

    Expected answer tag::

        <answer>{"missed_creatures":[
          {"id":1,"description":"large tan crab","clicks":[
             {"x":0.12,"y":0.08,"label":1},
             {"x":0.15,"y":0.11,"label":1}
          ]},
          ...
        ]}</answer>

    Returns a list of dicts::

        {
          "id": int,
          "description": str,
          "clicks": [{"x": float, "y": float, "label": int}, ...],
        }

    Validation rules (lenient -- same policy as ``parse_som_response``):

    * Returns ``[]`` on missing / malformed ``<answer>`` tag.
    * Individual clicks with out-of-range coords (not in [0, 1]) are dropped.
    * Individual clicks with invalid labels (not 0 or 1) are dropped.
    * Creature entries whose ``clicks`` array is empty after filtering are
      dropped entirely.
    * If ``id`` values are absent or duplicated, sequential 1..N ids are
      re-assigned.
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
    raw = payload.get("missed_creatures") if isinstance(payload, dict) else None
    if not isinstance(raw, list):
        return []

    groups: list[dict] = []
    seen_ids: set = set()
    for item in raw:
        if not isinstance(item, dict):
            continue
        raw_clicks = item.get("clicks")
        if not isinstance(raw_clicks, list):
            continue

        good_clicks: list[dict] = []
        for c in raw_clicks:
            if not isinstance(c, dict):
                continue
            x = c.get("x")
            y = c.get("y")
            label = c.get("label")
            # Drop non-numeric coords (bool counts as int in Python, reject it)
            if isinstance(x, bool) or not isinstance(x, (int, float)):
                continue
            if isinstance(y, bool) or not isinstance(y, (int, float)):
                continue
            if not (0.0 <= float(x) <= 1.0 and 0.0 <= float(y) <= 1.0):
                continue
            if label not in (0, 1):
                continue
            good_clicks.append({"x": float(x), "y": float(y), "label": int(label)})

        if not good_clicks:
            continue  # creature has no usable clicks — drop it

        desc = item.get("description")
        if not isinstance(desc, str):
            desc = ""

        raw_id = item.get("id")
        # Accept id only if it's a plain positive int and not already seen
        if (
            not isinstance(raw_id, bool)
            and isinstance(raw_id, int)
            and raw_id > 0
            and raw_id not in seen_ids
        ):
            creature_id = raw_id
        else:
            creature_id = None  # will be re-assigned below

        seen_ids.add(creature_id)  # may add None; handled below
        groups.append({
            "id": creature_id,
            "description": desc,
            "clicks": good_clicks,
        })

    # Re-assign ids if any were missing / duplicated
    if any(g["id"] is None for g in groups):
        # Full re-assignment: keep ids that are unique ints and reassign the rest
        assigned: set[int] = {g["id"] for g in groups if g["id"] is not None}
        counter = 1
        for g in groups:
            if g["id"] is None:
                while counter in assigned:
                    counter += 1
                g["id"] = counter
                assigned.add(counter)
                counter += 1

    return groups


def render_proposed_clicks_overlay(
    frame_bgr,
    clicks: list[dict],
    existing_masks: list[dict] | None = None,
):
    """Render frame with existing-masks overlay (faint green) + proposed
    click points (numbered red dots with crosshair) for visualization.
    Used for debug artefact 03_proposed_clicks.png."""
    import cv2
    import numpy as np

    out = (
        frame_bgr.copy()
        if existing_masks is None
        else render_existing_masks_overlay(frame_bgr, existing_masks, alpha=0.2)
    )
    h, w = out.shape[:2]
    for idx, c in enumerate(clicks, start=1):
        cx = int(round(c["x"] * w))
        cy = int(round(c["y"] * h))
        cv2.circle(out, (cx, cy), 12, (0, 0, 255), 2)
        cv2.circle(out, (cx, cy), 2, (255, 255, 255), -1)
        cv2.drawMarker(out, (cx, cy), (0, 0, 255), cv2.MARKER_CROSS, 18, 1)
        label = str(idx)
        cv2.putText(
            out, label, (cx + 14, cy + 6),
            cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 3, cv2.LINE_AA,
        )
        cv2.putText(
            out, label, (cx + 14, cy + 6),
            cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 255), 1, cv2.LINE_AA,
        )
    return out


def render_proposed_click_groups_overlay(
    frame_bgr,
    groups: list[dict],
    existing_masks: list[dict] | None = None,
):
    """Render frame with existing-masks overlay (faint green) + proposed
    click GROUPS, color-coded by creature id.

    Each group entry is expected to be shaped as returned by
    ``parse_creature_click_groups``::

        {"id": int, "description": str,
         "clicks": [{"x": float, "y": float, "label": int}, ...]}

    Visual conventions:

    * Positive clicks (label=1) are drawn as a circle with a white centre
      dot -- the classic SAM positive-click marker.
    * Negative clicks (label=0) are drawn as an X with
      ``cv2.MARKER_TILTED_CROSS``.
    * Each click is labelled ``<creature_id>.<click_idx>`` (e.g. "1.1",
      "1.2", "2.1") to let reviewers quickly associate clicks with
      creatures.
    * Colors cycle through a small BGR palette keyed by ``id``.

    Keep ``render_proposed_clicks_overlay`` (flat list) for legacy use.
    """
    import cv2

    out = (
        frame_bgr.copy()
        if existing_masks is None
        else render_existing_masks_overlay(frame_bgr, existing_masks, alpha=0.2)
    )
    h, w = out.shape[:2]

    # BGR palette (red, cyan-ish, yellow, magenta, orange, purple)
    palette = [
        (0, 0, 255),    # red
        (255, 200, 0),  # cyan-ish
        (0, 255, 255),  # yellow
        (255, 0, 255),  # magenta
        (0, 200, 255),  # orange
        (200, 0, 255),  # purple
    ]

    for grp in groups:
        cid = int(grp.get("id", -1))
        color = palette[(cid - 1) % len(palette)] if cid > 0 else (200, 200, 200)
        for click_idx, c in enumerate(grp.get("clicks") or [], start=1):
            cx = int(round(c["x"] * w))
            cy = int(round(c["y"] * h))
            label_text = f"{cid}.{click_idx}"
            if int(c.get("label", 1)) == 1:
                cv2.circle(out, (cx, cy), 12, color, 2)
                cv2.circle(out, (cx, cy), 2, (255, 255, 255), -1)
            else:
                # negative -> tilted cross
                cv2.drawMarker(out, (cx, cy), color, cv2.MARKER_TILTED_CROSS, 20, 2)
            cv2.putText(out, label_text, (cx + 14, cy + 6), cv2.FONT_HERSHEY_SIMPLEX,
                        0.5, (0, 0, 0), 3, cv2.LINE_AA)
            cv2.putText(out, label_text, (cx + 14, cy + 6), cv2.FONT_HERSHEY_SIMPLEX,
                        0.5, color, 1, cv2.LINE_AA)
    return out


class Sam3PointService:
    """Run SAM3 single-image click mode on (x, y) prompts using the
    processor/model API documented in examples/sam3_for_sam1_task_example.ipynb.

    Construction takes the same Sam3Image model and Sam3Processor that
    the every-frame text-agent CLI builds. The processor must wrap a
    model built with enable_inst_interactivity=True.

    For each image, set_image is called once and the resulting
    inference_state is reused across all clicks on that image.
    """

    def __init__(self, model, processor):
        self.model = model
        self.processor = processor
        # Sanity: confirm the model supports predict_inst
        if not hasattr(model, "predict_inst"):
            raise RuntimeError(
                "Sam3Image model does not expose predict_inst; rebuild with "
                "enable_inst_interactivity=True."
            )

    def group_segment(
        self,
        image_path: str,
        groups: list[dict],
        output_folder: str | None = None,
    ) -> list[dict]:
        """For each creature group, run SAM3 click mode with ALL the group's
        positive + negative points jointly.

        ``groups`` is shaped as returned by ``parse_creature_click_groups``::

            [{"id": int, "description": str,
              "clicks": [{"x": float, "y": float, "label": int}, ...]}, ...]

        Returns one result per group::

            {"creature_id": int, "description": str,
             "mask": bool HxW, "score": float, "area_px": int,
             "select_reason": str,
             "spatial_match": "click_mode" | "click_mode_empty" | "click_mode_error",
             "clicks_used": list of {x, y, label}}

        Multiple foreground (label=1) clicks let SAM3 jointly disambiguate
        elongated objects; background (label=0) clicks exclude substrate or
        a neighbouring creature.

        ``output_folder`` is accepted for API compatibility but is unused in
        click mode (no per-call JSON artefacts to write).
        """
        import numpy as np
        from PIL import Image

        pil = Image.open(image_path).convert("RGB")
        w, h = pil.size
        inference_state = self.processor.set_image(pil)

        out: list[dict] = []
        for grp in groups:
            clicks = grp.get("clicks") or []
            # Defensive re-filter (parser should have already done this)
            good = [
                c for c in clicks
                if isinstance(c.get("x"), (int, float))
                and isinstance(c.get("y"), (int, float))
                and c.get("label") in (0, 1)
            ]
            if not good:
                out.append({
                    "creature_id": int(grp.get("id", -1)),
                    "description": str(grp.get("description", "")),
                    "mask": np.zeros((h, w), dtype=bool),
                    "score": 0.0,
                    "area_px": 0,
                    "select_reason": "no_clicks",
                    "spatial_match": "click_mode_empty",
                    "clicks_used": [],
                })
                continue

            point_coords = np.array(
                [[float(c["x"]) * w, float(c["y"]) * h] for c in good],
                dtype=np.float32,
            )
            point_labels = np.array([int(c["label"]) for c in good], dtype=np.int64)

            try:
                masks, scores, _logits = self.model.predict_inst(
                    inference_state,
                    point_coords=point_coords,
                    point_labels=point_labels,
                    multimask_output=True,
                )
            except Exception as exc:
                print(
                    f"[som] group_segment failed for creature {grp.get('id')!r} "
                    f"({grp.get('description')!r}): {type(exc).__name__}: {exc}"
                )
                out.append({
                    "creature_id": int(grp.get("id", -1)),
                    "description": str(grp.get("description", "")),
                    "mask": np.zeros((h, w), dtype=bool),
                    "score": 0.0,
                    "area_px": 0,
                    "select_reason": "click_mode_error",
                    "spatial_match": "click_mode_error",
                    "clicks_used": good,
                })
                continue

            # Normalise tensor/array shapes
            masks_np = masks.detach().cpu().numpy() if hasattr(masks, "detach") else np.asarray(masks)
            scores_np = scores.detach().cpu().numpy() if hasattr(scores, "detach") else np.asarray(scores)
            if masks_np.ndim == 4:  # (1, N, H, W) -> (N, H, W)
                masks_np = masks_np[0]
                scores_np = scores_np[0] if scores_np.ndim >= 1 else scores_np
            masks_np = masks_np.astype(bool)

            if masks_np.size == 0:
                out.append({
                    "creature_id": int(grp.get("id", -1)),
                    "description": str(grp.get("description", "")),
                    "mask": np.zeros((h, w), dtype=bool),
                    "score": 0.0,
                    "area_px": 0,
                    "select_reason": "click_mode_empty",
                    "spatial_match": "click_mode_empty",
                    "clicks_used": good,
                })
                continue

            # Smallest-in-band selection (same policy as the legacy point_segment)
            img_h, img_w = masks_np.shape[1], masks_np.shape[2]
            total_pixels = float(img_h * img_w)
            areas = masks_np.reshape(masks_np.shape[0], -1).sum(axis=1)
            min_area_for_selection = max(200, int(0.001 * total_pixels))
            max_area_for_selection = 0.5 * total_pixels

            eligible = sorted(
                [(int(a), i) for i, a in enumerate(areas)
                 if min_area_for_selection <= a <= max_area_for_selection]
            )
            if eligible:
                _area, best_idx = eligible[0]
                select_reason = "smallest_in_band"
            else:
                above_min = sorted(
                    [(int(a), i) for i, a in enumerate(areas) if a >= min_area_for_selection]
                )
                if above_min:
                    _area, best_idx = above_min[0]
                    select_reason = "smallest_above_min"
                else:
                    best_idx = int(np.argmax(scores_np))
                    select_reason = "fallback_highest_score"

            best_mask = masks_np[best_idx]
            best_score = float(scores_np[best_idx])
            area_px = int(best_mask.sum())

            print(
                f"[som] creature id={grp.get('id')} '{grp.get('description')}' "
                f"({len(good)} clicks) -> "
                f"mask {area_px}/{int(total_pixels)} "
                f"({area_px / total_pixels * 100:.1f}%) "
                f"reason={select_reason} score={best_score:.3f}"
            )

            out.append({
                "creature_id": int(grp.get("id", -1)),
                "description": str(grp.get("description", "")),
                "mask": best_mask,
                "score": best_score,
                "area_px": area_px,
                "select_reason": select_reason,
                "spatial_match": "click_mode",
                "clicks_used": good,
            })

        return out

    def refine_segment(
        self,
        image_path: str,
        group: dict,
        *,
        add_points: list[dict],
        output_folder: str | None = None,
    ) -> dict:
        """Re-run SAM3 click mode with the group's original clicks PLUS
        the new add_points (foreground or background by label). Returns
        the same shape group_segment returns, with select_reason and
        area_px reflecting the refinement run."""
        existing_clicks = list(group.get("clicks_used") or group.get("clicks") or [])
        combined_clicks = existing_clicks + list(add_points or [])
        refined_group = {
            "id": group.get("creature_id", group.get("id", -1)),
            "description": group.get("description", ""),
            "clicks": combined_clicks,
        }
        results = self.group_segment(image_path, [refined_group], output_folder=output_folder)
        return results[0]

    def point_segment(
        self, image_path: str, clicks: list[dict], output_folder: str | None = None,
    ) -> list[dict]:
        """Legacy single-click-per-creature wrapper around ``group_segment``.

        Each input click ``{x, y, description}`` becomes a 1-click foreground
        group fed to ``group_segment``. The result is re-shaped to the OLD
        consumer format ``{mask, score, sam_text_prompt: "", spatial_match,
        area_px, select_reason}`` so existing callers (tests, scripts) that
        were written against the prior API keep working without modification.

        LEGACY: kept for backwards compatibility with any callers.  New code
        should use ``group_segment`` directly with the grouped-click contract.
        """
        groups = [
            {
                "id": i + 1,
                "description": c.get("description", ""),
                "clicks": [{"x": c["x"], "y": c["y"], "label": 1}],
            }
            for i, c in enumerate(clicks)
        ]
        grp_results = self.group_segment(image_path, groups, output_folder=output_folder)
        out = []
        for r in grp_results:
            out.append({
                "mask": r["mask"],
                "score": r["score"],
                "sam_text_prompt": "",
                "spatial_match": r["spatial_match"],
                "area_px": r["area_px"],
                "select_reason": r["select_reason"],
            })
        return out


def generate_click_based_candidates(
    *,
    target_frame_bgr,
    target_image_path: str,
    existing_masks: list[dict],
    neighbour_image_paths: list[str],
    initial_text_prompt: str,
    discovery_system_prompt: str,
    output_folder: str,
    mllm_send,
    sam3_point_service,
) -> tuple[list[dict], list[dict], str | None]:
    """Returns (candidates, proposed_groups, mllm_response_text).

    - Renders an existing-masks overlay to disk (for the MLLM to see covered regions).
    - Asks the MLLM to propose click GROUPS for MISSED creatures (new grouped contract).
    - For each group, runs SAM3 image point-mode with ALL clicks in the group jointly.
    - Returns candidates in the {mask, bbox_xywh, score, source_prompt, creature_id,
      description, clicks_used, select_reason, spatial_match} shape that
      filter_candidates_with_reasons consumes, plus the raw group proposals
      (for visualization) and the raw MLLM response text (for debug).

    The return signature changed from ``(candidates, clicks, response_text)`` to
    ``(candidates, groups, response_text)`` -- ``groups`` is a list of dicts shaped
    as returned by ``parse_creature_click_groups``.
    """
    import cv2
    import numpy as np

    os.makedirs(output_folder, exist_ok=True)

    # 1. Render existing-masks overlay
    overlay = render_existing_masks_overlay(target_frame_bgr, existing_masks)
    overlay_path = os.path.join(output_folder, "02_existing_masks.png")
    cv2.imwrite(overlay_path, overlay)

    # 2. Build discovery messages (new grouped-click contract)
    user_text = (
        f"The first image is the TARGET FRAME with translucent green overlays "
        f"showing masks already produced by a text-prompted detector with query "
        f"'{initial_text_prompt}'. The subsequent images are REFERENCE FRAMES "
        f"from nearby times. Identify any creatures matching '{initial_text_prompt}' "
        f"that are visible in the TARGET FRAME but NOT covered by the green overlays. "
        f"For each missed creature, output a GROUP of click points in NORMALIZED "
        f"coordinates where (0,0) is the top-left and (1,1) is the bottom-right of "
        f"the TARGET image. Also include a 1-5 word description and a sequential id.\n\n"
        f"Each click has a label: 1=FOREGROUND (on the creature) or 0=BACKGROUND "
        f"(on substrate or a neighbouring creature to EXCLUDE from segmentation).\n\n"
        f"Output format -- free-text reasoning then EXACTLY ONE trailing tag:\n"
        f'<answer>{{"missed_creatures":[{{"id":1,"description":"<text>","clicks":[{{"x":<float>,"y":<float>,"label":1}},...]}},...]}}'
        f'</answer>\n\n'
        f"An empty list is valid if you see no missed creatures. Prefer fewer "
        f"high-confidence proposals over many uncertain ones."
    )
    messages = [
        {"role": "system", "content": discovery_system_prompt},
        {
            "role": "user",
            "content": [
                {"type": "image", "image": overlay_path},
                {"type": "text", "text": user_text},
                *[{"type": "image", "image": p} for p in neighbour_image_paths],
            ],
        },
    ]

    response_text = mllm_send(messages)
    groups = parse_creature_click_groups(response_text)

    if not groups:
        return [], [], response_text

    # 3. SAM3 click-mode per MLLM-proposed group (all clicks in a group jointly)
    sam_results = sam3_point_service.group_segment(
        target_image_path, groups, output_folder=output_folder,
    )

    candidates = []
    for grp, sam_out in zip(groups, sam_results):
        mask = sam_out["mask"]
        if not mask.any():
            candidates.append({
                "mask": mask,
                "bbox_xywh": [0, 0, 0, 0],
                "score": 0.0,
                "source_prompt": f"click_group[{grp.get('description', '')}]",
                "creature_id": grp.get("id"),
                "description": grp.get("description", ""),
                "clicks_used": sam_out.get("clicks_used", []),
                "sam_text_prompt": "",
                "spatial_match": sam_out["spatial_match"],
                "select_reason": sam_out.get("select_reason", ""),
            })
            continue
        ys, xs = np.where(mask)
        bbox = [
            int(xs.min()), int(ys.min()),
            int(xs.max() - xs.min() + 1),
            int(ys.max() - ys.min() + 1),
        ]
        candidates.append({
            "mask": mask,
            "bbox_xywh": bbox,
            "score": float(sam_out["score"]),
            "source_prompt": f"click_group[{grp.get('description', '')}]",
            "creature_id": grp.get("id"),
            "description": grp.get("description", ""),
            "clicks_used": sam_out.get("clicks_used", []),
            "sam_text_prompt": "",
            "spatial_match": sam_out["spatial_match"],
            "select_reason": sam_out.get("select_reason", ""),
            "area_px": sam_out.get("area_px", int(mask.sum())),
        })

    return candidates, groups, response_text


def parse_frame_validity(text: str) -> str | None:
    """Returns 'usable', 'corrupted', or None (malformed)."""
    import re
    m = re.findall(r"<validity>\s*(usable|corrupted)\s*</validity>", text or "", flags=re.IGNORECASE)
    if not m:
        return None
    return m[-1].lower()


def parse_refinement_response(text: str) -> dict | None:
    """Returns {"action": "accept"|"reject"|"refine",
                "add_points": list of {x,y,label}} or None (malformed)."""
    import json as _json_mod
    import re as _re_mod

    if not isinstance(text, str) or not text:
        return None
    matches = _re_mod.findall(r"<refine>\s*(.*?)\s*</refine>", text, flags=_re_mod.DOTALL)
    if not matches:
        return None
    raw = matches[-1].strip()
    try:
        payload = _json_mod.loads(raw)
    except _json_mod.JSONDecodeError:
        return None
    if not isinstance(payload, dict):
        return None
    action = payload.get("action")
    if action not in ("accept", "reject", "refine"):
        return None
    if action == "refine":
        raw_points = payload.get("add_points")
        if not isinstance(raw_points, list):
            return None
        good_points = []
        for pt in raw_points:
            if not isinstance(pt, dict):
                continue
            x = pt.get("x")
            y = pt.get("y")
            label = pt.get("label")
            if isinstance(x, bool) or not isinstance(x, (int, float)):
                continue
            if isinstance(y, bool) or not isinstance(y, (int, float)):
                continue
            if not (0.0 <= float(x) <= 1.0 and 0.0 <= float(y) <= 1.0):
                continue
            if label not in (0, 1):
                continue
            good_points.append({"x": float(x), "y": float(y), "label": int(label)})
        return {"action": "refine", "add_points": good_points}
    return {"action": action, "add_points": []}


def load_frame_quality_system_prompt(profile: str) -> str:
    """Load the SoM frame-quality screening system prompt for the requested profile.

    ``profile`` is one of: 'underwater', 'general'. Other values raise ValueError.
    """
    valid = {"underwater", "general"}
    if profile not in valid:
        raise ValueError(
            f"Unknown profile '{profile}'. Expected one of: {sorted(valid)}."
        )
    here = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    path = os.path.join(
        here,
        "sam3", "agent", "system_prompts",
        f"system_prompt_som_frame_quality_{profile}.txt",
    )
    with open(path, "r", encoding="utf-8") as f:
        return f.read()


def load_refinement_system_prompt(profile: str) -> str:
    """Load the SoM per-mark refinement system prompt for the requested profile.

    ``profile`` is one of: 'underwater', 'general'. Other values raise ValueError.
    """
    valid = {"underwater", "general"}
    if profile not in valid:
        raise ValueError(
            f"Unknown profile '{profile}'. Expected one of: {sorted(valid)}."
        )
    here = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    path = os.path.join(
        here,
        "sam3", "agent", "system_prompts",
        f"system_prompt_som_mark_refinement_{profile}.txt",
    )
    with open(path, "r", encoding="utf-8") as f:
        return f.read()


def screen_target_frames_for_quality(
    target_indices: list[int],
    *,
    video_path: str,
    load_frame,
    output_folder: str,
    mllm_send,
    valid_frame_pool: list[int],
    max_replacements_per_slot: int = 5,
) -> tuple[list[int], list[dict]]:
    """For each frame in target_indices, run an MLLM usability check.
    Replace any frame the MLLM flags as corrupted/unusable with the
    nearest frame in valid_frame_pool that hasn't been used and passes
    the same check, up to max_replacements_per_slot attempts.

    Saves each tested frame as a PNG under output_folder/quality_checks/
    so debug artefacts capture the screening.

    Returns (final_target_indices, screening_report) where
    screening_report is a list of {original_index, replaced_with,
    verdicts: [{frame_index, validity, reason}]}.
    """
    import cv2

    qc_dir = os.path.join(output_folder, "quality_checks")
    os.makedirs(qc_dir, exist_ok=True)

    try:
        system_prompt_text = load_frame_quality_system_prompt("underwater")
    except Exception:
        try:
            system_prompt_text = load_frame_quality_system_prompt("general")
        except Exception:
            system_prompt_text = (
                "Classify this frame. Output exactly one trailing tag: "
                "<validity>usable</validity> or <validity>corrupted</validity>."
            )

    pool_sorted = sorted(valid_frame_pool)
    used_indices: set[int] = set(target_indices)
    final_targets: list[int] = list(target_indices)
    screening_report: list[dict] = []

    for slot_idx, orig_idx in enumerate(target_indices):
        # Build candidate queue: original first, then nearest neighbours from pool
        # sweeping outward by distance.
        other_pool = [i for i in pool_sorted if i != orig_idx]
        other_pool.sort(key=lambda i: abs(i - orig_idx))
        candidates_to_try = [orig_idx] + other_pool

        verdicts: list[dict] = []
        replaced_with: int | None = None
        current = orig_idx

        for attempt_idx, candidate_idx in enumerate(candidates_to_try):
            if attempt_idx > 0 and attempt_idx > max_replacements_per_slot:
                # Exhausted replacement budget; keep original
                final_targets[slot_idx] = orig_idx
                break
            # Skip indices already claimed by other slots (except the original itself)
            if attempt_idx > 0 and candidate_idx in used_indices:
                continue

            frame_bgr = load_frame(video_path, candidate_idx)
            if frame_bgr is None:
                verdicts.append({
                    "frame_index": candidate_idx,
                    "validity": "corrupted",
                    "reason": "frame_unreadable",
                })
                continue

            # Save frame for debug
            frame_png = os.path.join(qc_dir, f"slot{slot_idx:02d}_f{candidate_idx:06d}.png")
            cv2.imwrite(frame_png, frame_bgr)

            user_content = [
                {"type": "image", "image": frame_png},
                {"type": "text", "text": (
                    "Is this frame visually usable for biological annotation? "
                    "Output exactly one tag at the end: "
                    "<validity>usable</validity> or <validity>corrupted</validity>. "
                    "Brief reasoning above is fine."
                )},
            ]
            messages = [
                {"role": "system", "content": system_prompt_text},
                {"role": "user", "content": user_content},
            ]
            try:
                response = mllm_send(messages)
            except Exception as exc:
                response = f"error: {exc}"

            validity = parse_frame_validity(response or "")
            if validity is None:
                validity = "usable"  # lenient default on malformed response

            verdicts.append({
                "frame_index": candidate_idx,
                "validity": validity,
                "reason": (response or "")[:200],
            })

            if validity == "usable":
                final_targets[slot_idx] = candidate_idx
                if candidate_idx != orig_idx:
                    replaced_with = candidate_idx
                    used_indices.add(candidate_idx)
                break
        else:
            # All candidates exhausted without finding a usable frame;
            # keep the original so we don't silently drop the slot.
            final_targets[slot_idx] = orig_idx

        screening_report.append({
            "original_index": orig_idx,
            "replaced_with": replaced_with,
            "verdicts": verdicts,
        })

    return final_targets, screening_report


def render_refinement_crop(
    frame_bgr,
    candidate: dict,
    pad_frac: float = 0.25,
) -> "np.ndarray":
    """Crop the frame around the candidate mask with padding, render
    the mask as a translucent green overlay, and overlay the original
    click points as small dots. Returns a cropped BGR array suitable
    for MLLM consumption."""
    import cv2
    import numpy as np

    h, w = frame_bgr.shape[:2]
    mask = np.asarray(candidate.get("mask"), dtype=bool)
    if not mask.any():
        # Fall back to full frame if no mask pixels
        return frame_bgr.copy()

    ys, xs = np.where(mask)
    x0, x1 = int(xs.min()), int(xs.max())
    y0, y1 = int(ys.min()), int(ys.max())

    bw = x1 - x0 + 1
    bh = y1 - y0 + 1
    pad_x = max(1, int(round(bw * pad_frac)))
    pad_y = max(1, int(round(bh * pad_frac)))

    cx0 = max(0, x0 - pad_x)
    cy0 = max(0, y0 - pad_y)
    cx1 = min(w - 1, x1 + pad_x)
    cy1 = min(h - 1, y1 + pad_y)

    crop = frame_bgr[cy0:cy1 + 1, cx0:cx1 + 1].copy()
    crop_mask = mask[cy0:cy1 + 1, cx0:cx1 + 1]

    # Green translucent overlay
    green = np.array([0, 255, 0], dtype=np.float32)
    crop_f = crop.astype(np.float32)
    crop_f[crop_mask] = 0.65 * crop_f[crop_mask] + 0.35 * green
    crop = crop_f.astype(np.uint8)

    # Overlay original click points
    clicks_used = candidate.get("clicks_used") or []
    for c in clicks_used:
        px = int(round(float(c["x"]) * w)) - cx0
        py = int(round(float(c["y"]) * h)) - cy0
        if 0 <= px < crop.shape[1] and 0 <= py < crop.shape[0]:
            color = (0, 0, 255) if int(c.get("label", 1)) == 1 else (255, 0, 0)
            cv2.circle(crop, (px, py), 6, color, 2)
            cv2.circle(crop, (px, py), 2, (255, 255, 255), -1)

    return crop


def build_refinement_messages(
    *,
    system_prompt: str,
    crop_image_path: str,
    original_clicks: list[dict],
    neighbour_paths: list[str],
    description: str | None,
) -> list[dict]:
    """Build the MLLM message list for a per-mark refinement call."""
    desc_text = description or "(unknown creature)"
    click_text = (
        ", ".join(
            f"({'fg' if int(c.get('label', 1)) == 1 else 'bg'} @ "
            f"{float(c['x']):.3f},{float(c['y']):.3f})"
            for c in (original_clicks or [])
        )
        or "(none)"
    )
    user_text = (
        f"Candidate creature: '{desc_text}'.\n"
        f"Original SAM3 clicks: {click_text}.\n"
        f"The first image shows the candidate mask (green overlay) in a zoomed crop.\n"
        f"The mask was REJECTED by the judge. Please review and output one of:\n"
        f"  <refine>{{\"action\":\"accept\"}}</refine>  — mask is actually fine\n"
        f"  <refine>{{\"action\":\"reject\"}}</refine>  — mask is unsalvageable\n"
        f"  <refine>{{\"action\":\"refine\",\"add_points\":[{{\"x\":<float>,\"y\":<float>,\"label\":1}},...]}}</refine>"
        f"  — propose new SAM3 clicks (coords in FULL-FRAME normalized [0,1])\n"
        f"Keep add_points to 1-2 entries. Always output exactly one tag."
    )
    user_content: list[dict] = [
        {"type": "image", "image": crop_image_path},
        {"type": "text", "text": user_text},
    ]
    for nb_path in (neighbour_paths or []):
        user_content.append({"type": "image", "image": nb_path})

    return [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_content},
    ]


from dataclasses import dataclass, field


@dataclass
class SomStageConfig:
    video_path: str
    frame_results_path: str
    frame_outputs_path: str
    output_dir: str
    prompt_profile: str
    initial_text_prompt: str
    num_target_frames: int
    frame_selection_strategy: str
    target_frames_explicit: list[int] | None
    num_neighbours: int
    neighbour_offset_frames: int
    iou_dedup: float
    min_area_px: float
    max_area_frac: float
    edge_tol_px: int
    internal_iou_dedup: float
    max_mllm_calls: int
    discovery_num_neighbours: int = 4
    screen_frame_quality: bool = True
    quality_check_max_replacements_per_slot: int = 5
    enable_refinement: bool = True
    max_refinement_iters: int = 2


def _load_video_frame_default(video_path: str, frame_index: int):
    """Default video frame loader via OpenCV. Tests pass in a fake."""
    import cv2

    cap = cv2.VideoCapture(video_path)
    try:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(frame_index))
        ok, frame = cap.read()
        if not ok:
            return None
        return frame
    finally:
        cap.release()


def _send_mllm_request_default(messages, **kwargs):
    """Default MLLM client. Tests pass in a fake."""
    from sam3.agent.client_claude import send_claude_request
    return send_claude_request(messages, **kwargs)


def _read_jsonl(path):
    rows = []
    if not os.path.exists(path):
        return rows
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return rows


def _existing_row_for_frame(frame_outputs: dict, frame_index: int) -> dict:
    """Find the existing-output row for a given frame index, or return an
    empty placeholder if the frame isn't in the outputs file."""
    for row in frame_outputs.get("frames", []):
        if int(row.get("frame_index", -1)) == int(frame_index):
            return row
    return {
        "frame_index": int(frame_index),
        "out_obj_ids": [],
        "out_binary_masks_rle": [],
        "out_boxes_xywh": [],
        "out_probs": [],
        "out_tracker_probs": [],
    }


def _save_redacted_messages(messages: list[dict], path: str) -> None:
    """Write a JSON-serialisable version of the MLLM message list (images
    replaced by path references rather than raw bytes)."""
    redacted = []
    for msg in messages:
        if isinstance(msg.get("content"), list):
            items = [
                ({"type": "image", "image": item["image"]}
                 if item.get("type") == "image"
                 else item)
                for item in msg["content"]
            ]
            redacted.append({"role": msg["role"], "content": items})
        else:
            redacted.append(msg)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(redacted, f, indent=2)


def _load_neighbours(
    load_frame,
    video_path: str,
    target_idx: int,
    num_neighbours: int,
    neighbour_offset_frames: int,
    nb_dir: str,
    tag: str,
) -> list[str]:
    """Load neighbour frames and write them to disk; return path list."""
    import cv2

    os.makedirs(nb_dir, exist_ok=True)
    paths: list[str] = []
    for offset_idx in range(1, num_neighbours + 1):
        for sign, label in ((-1, "neg"), (+1, "pos")):
            nb_idx = target_idx + sign * neighbour_offset_frames * offset_idx
            if nb_idx < 0:
                continue
            nb_frame = load_frame(video_path, nb_idx)
            if nb_frame is None:
                continue
            nb_path = os.path.join(nb_dir, f"{tag}_{label}{offset_idx}.png")
            cv2.imwrite(nb_path, nb_frame)
            paths.append(nb_path)
    return paths


def run_som_stage(
    cfg: SomStageConfig,
    *,
    _load_video_frame=None,
    _send_mllm_request=None,
    _sam3_point_service=None,
    _screen_frame_quality=None,
) -> dict:
    """Run the SoM missed-creature stage end-to-end.

    New architecture (click-discovery):
      0. (A) MLLM frame-quality screen: verify each target frame is usable,
             replace corrupted frames with nearest valid neighbours.
      1. Render existing masks as overlay → 02_existing_masks.png
      2. MLLM discovery call: proposes click (x,y) for missed creatures
      3. SAM3 image point-mode → candidate masks per click
      4. filter_candidates_with_reasons (size, dedup)
      5. draw_numbered_marks → 04_marked.png
      6. MLLM judge call: accepts/rejects numbered marks
      7. (C) Per-mark refinement sub-loop for rejected candidates
      8. merge accepted masks into augmented JSONL row

    Returns a summary dict with counts. All per-target artefacts and the
    augmented JSONL are written under ``cfg.output_dir``.

    Concurrency: this function appends to a single JSONL file with
    per-target flush. It is NOT safe to run two processes concurrently
    on the same output_dir -- there is no file lock around the append.
    Resume is supported within a single sequential run only.

    Test seams:
      _screen_frame_quality: if provided, replaces screen_target_frames_for_quality.
                             Signature: (target_indices, ...) -> (list[int], list[dict])
    """
    import cv2
    import numpy as np

    load_frame = _load_video_frame or _load_video_frame_default
    send_mllm = _send_mllm_request or _send_mllm_request_default
    point_service = _sam3_point_service  # may be None for judge-only tests

    os.makedirs(cfg.output_dir, exist_ok=True)

    frame_results = _read_jsonl(cfg.frame_results_path)
    with open(cfg.frame_outputs_path, "r", encoding="utf-8") as f:
        frame_outputs = json.load(f)

    targets = select_target_frames(
        frame_results,
        strategy=cfg.frame_selection_strategy,
        k=cfg.num_target_frames,
        explicit=cfg.target_frames_explicit,
    )

    # --- (A) MLLM frame-quality screening ---
    screening_report: list[dict] = []
    if cfg.screen_frame_quality and targets:
        # Build the full valid pool from frame_results (same filtering as select_target_frames)
        valid_pool = sorted(
            int(row["frame_index"]) for row in frame_results
            if not row.get("error") and not row.get("skipped")
            and (row.get("num_masks") or 0) > 0
        )
        screen_fn = _screen_frame_quality or screen_target_frames_for_quality
        targets, screening_report = screen_fn(
            targets,
            video_path=cfg.video_path,
            load_frame=load_frame,
            output_folder=cfg.output_dir,
            mllm_send=send_mllm,
            valid_frame_pool=valid_pool,
            max_replacements_per_slot=cfg.quality_check_max_replacements_per_slot,
        )
        # Save screening report
        with open(os.path.join(cfg.output_dir, "screening_report.json"), "w") as f:
            json.dump(screening_report, f, indent=2)

    # Resume support: read the augmented JSONL if it already exists
    augmented_path = os.path.join(cfg.output_dir, "augmented_frame_outputs.jsonl")
    already_done = {int(row["frame_index"])
                    for row in _read_jsonl(augmented_path)}

    judge_system_prompt = load_system_prompt(cfg.prompt_profile)
    discovery_system_prompt = load_click_discovery_system_prompt(cfg.prompt_profile)

    stats = {
        "targets_total": len(targets),
        "targets_processed": 0,
        "targets_skipped": 0,
        "targets_skipped_resume": 0,
        "mllm_calls": 0,
        "masks_accepted": 0,
        "targets_quality_replaced": sum(
            1 for r in screening_report if r.get("replaced_with") is not None
        ),
        "targets_quality_kept_corrupted": sum(
            1 for r in screening_report
            if r.get("replaced_with") is None
            and r.get("verdicts")
            and r["verdicts"][0].get("validity") == "corrupted"
        ),
    }

    # Open augmented JSONL in append mode (one row flushed per target)
    with open(augmented_path, "a", encoding="utf-8") as out_handle:
        for target_idx in targets:
            if target_idx in already_done:
                stats["targets_skipped_resume"] += 1
                continue
            # Budget cap counts discovery + judge calls together
            if stats["mllm_calls"] >= cfg.max_mllm_calls:
                print(f"[som] budget cap reached at {stats['mllm_calls']} mllm calls; stopping.")
                break

            target_dir = os.path.join(cfg.output_dir, f"target_{target_idx:06d}")
            os.makedirs(target_dir, exist_ok=True)

            target_frame = load_frame(cfg.video_path, target_idx)
            if target_frame is None:
                stats["targets_skipped"] += 1
                print(f"[som] frame {target_idx} unreadable; skipping.")
                continue

            target_img_path = os.path.join(target_dir, "01_raw.png")
            if not cv2.imwrite(target_img_path, target_frame):
                stats["targets_skipped"] += 1
                print(f"[som] frame {target_idx}: failed to write target raw PNG; skipping.")
                continue

            # Existing text-agent masks for this frame
            existing_row = _existing_row_for_frame(frame_outputs, target_idx)
            from nibi_model_compare.frame_output_utils import decode_rle_to_mask
            existing_masks = []
            for rle in existing_row.get("out_binary_masks_rle", []):
                try:
                    size = rle.get("size", []) if isinstance(rle, dict) else []
                    if len(size) >= 2:
                        h_rle, w_rle = int(size[0]), int(size[1])
                    else:
                        h_rle, w_rle = target_frame.shape[0], target_frame.shape[1]
                    mask = decode_rle_to_mask(rle, h_rle, w_rle).astype(bool)
                    existing_masks.append({"mask": mask})
                except Exception as exc:
                    print(
                        f"[som] frame {target_idx}: skipping malformed existing RLE "
                        f"({type(exc).__name__}: {exc})"
                    )
                    continue

            # --- DISCOVERY step ---

            # Discovery neighbour frames
            discovery_nb_paths = _load_neighbours(
                load_frame, cfg.video_path, target_idx,
                cfg.discovery_num_neighbours, cfg.neighbour_offset_frames,
                os.path.join(target_dir, "neighbours"), tag="disc",
            )

            if point_service is not None:
                candidates, groups, discovery_resp = generate_click_based_candidates(
                    target_frame_bgr=target_frame,
                    target_image_path=target_img_path,
                    existing_masks=existing_masks,
                    neighbour_image_paths=discovery_nb_paths,
                    initial_text_prompt=cfg.initial_text_prompt,
                    discovery_system_prompt=discovery_system_prompt,
                    output_folder=target_dir,
                    mllm_send=send_mllm,
                    sam3_point_service=point_service,
                )
                stats["mllm_calls"] += 1

                # Save discovery artefacts (groups = new grouped-click contract)
                with open(os.path.join(target_dir, "discovery_request.json"), "w") as f:
                    json.dump(groups, f, indent=2)
                with open(os.path.join(target_dir, "discovery_response.txt"), "w") as f:
                    f.write(discovery_resp or "")

                # Render proposed click groups → 03_proposed_clicks.png
                clicks_vis = render_proposed_click_groups_overlay(
                    target_frame, groups, existing_masks=existing_masks
                )
                cv2.imwrite(os.path.join(target_dir, "03_proposed_clicks.png"), clicks_vis)
            else:
                # No point service provided (test / fallback): skip discovery
                candidates = []
                groups = []

            if not candidates:
                stats["targets_skipped"] += 1
                print(f"[som] frame {target_idx} no_candidates_from_clicks; skipping.")
                continue

            # --- FILTER candidates ---
            survivors_with_reasons = filter_candidates_with_reasons(
                candidates, existing_masks,
                iou_dedup=cfg.iou_dedup,
                min_area_px=cfg.min_area_px,
                max_area_frac=cfg.max_area_frac,
                edge_tol_px=cfg.edge_tol_px,
            )
            survivors = [c for c, r in survivors_with_reasons if r is None]

            # Save candidate decisions for debug
            with open(os.path.join(target_dir, "candidates.json"), "w") as f:
                json.dump([
                    {
                        "bbox_xywh": c["bbox_xywh"],
                        "score": c["score"],
                        "source_prompt": c.get("source_prompt"),
                        "drop_reason": r,
                    }
                    for c, r in survivors_with_reasons
                ], f, indent=2)

            if not survivors:
                stats["targets_skipped"] += 1
                print(f"[som] frame {target_idx} nothing_after_dedup; skipping.")
                continue

            # Sort by descending area so larger marks get lower ids
            survivors.sort(key=lambda c: -int(np.asarray(c["mask"], dtype=bool).sum()))

            # Render marks → 04_marked.png
            marked = draw_numbered_marks(target_frame, survivors)
            marked_path = os.path.join(target_dir, "04_marked.png")
            if not cv2.imwrite(marked_path, marked):
                stats["targets_skipped"] += 1
                print(f"[som] frame {target_idx}: failed to write marked PNG; skipping.")
                continue

            # --- JUDGE step ---

            # Check budget before judge call
            if stats["mllm_calls"] >= cfg.max_mllm_calls:
                print(f"[som] budget cap reached at {stats['mllm_calls']} mllm calls; stopping.")
                break

            # Judge neighbour frames (may differ in count from discovery)
            judge_nb_paths = _load_neighbours(
                load_frame, cfg.video_path, target_idx,
                cfg.num_neighbours, cfg.neighbour_offset_frames,
                os.path.join(target_dir, "neighbours"), tag="judge",
            )

            messages = build_som_prompt_messages(
                system_prompt=judge_system_prompt,
                target_image_path=marked_path,
                neighbour_image_paths=judge_nb_paths,
                initial_text_prompt=cfg.initial_text_prompt,
                num_marks=len(survivors),
            )
            _save_redacted_messages(messages, os.path.join(target_dir, "05_judge_request.json"))

            response_text = send_mllm(messages)
            stats["mllm_calls"] += 1
            with open(os.path.join(target_dir, "05_judge_response.txt"), "w") as f:
                f.write(response_text or "")

            accepted_ids = parse_som_response(
                response_text,
                valid_ids=set(range(1, len(survivors) + 1)),
            )
            accepted_candidates = [survivors[i - 1] for i in accepted_ids]
            rejected_candidates = [
                survivors[i - 1]
                for i in range(1, len(survivors) + 1)
                if i not in set(accepted_ids)
            ]

            with open(os.path.join(target_dir, "06_accepted.json"), "w") as f:
                json.dump(accepted_ids, f)

            # --- (C) REFINEMENT sub-loop ---
            if (cfg.enable_refinement
                    and rejected_candidates
                    and point_service is not None
                    and stats["mllm_calls"] < cfg.max_mllm_calls):
                refinement_system_prompt = load_refinement_system_prompt(cfg.prompt_profile)
                for cand in rejected_candidates:
                    cand_id = cand.get("creature_id", "unknown")
                    ref_dir = os.path.join(target_dir, "refinement", str(cand_id))
                    os.makedirs(ref_dir, exist_ok=True)
                    for iter_idx in range(cfg.max_refinement_iters):
                        if stats["mllm_calls"] >= cfg.max_mllm_calls:
                            break
                        # Render crop with mask overlay and clicks
                        crop_bgr = render_refinement_crop(target_frame, cand)
                        crop_path = os.path.join(
                            ref_dir, f"iter_{iter_idx:02d}_crop.png"
                        )
                        cv2.imwrite(crop_path, crop_bgr)

                        ref_messages = build_refinement_messages(
                            system_prompt=refinement_system_prompt,
                            crop_image_path=crop_path,
                            original_clicks=cand.get("clicks_used", []),
                            neighbour_paths=judge_nb_paths,
                            description=cand.get("description"),
                        )
                        ref_resp = send_mllm(ref_messages)
                        stats["mllm_calls"] += 1

                        # Save artefacts
                        with open(
                            os.path.join(ref_dir, f"iter_{iter_idx:02d}_response.txt"), "w"
                        ) as f:
                            f.write(ref_resp or "")

                        parsed_ref = parse_refinement_response(ref_resp)
                        if parsed_ref is None or parsed_ref["action"] == "reject":
                            break
                        if parsed_ref["action"] == "accept":
                            accepted_candidates.append(cand)
                            break
                        # action == "refine": re-run SAM3 with new points
                        new_points = parsed_ref.get("add_points") or []
                        if not new_points:
                            # No actionable points; treat as reject
                            break
                        ref_result = point_service.refine_segment(
                            target_img_path, cand,
                            add_points=new_points,
                            output_folder=ref_dir,
                        )
                        # Update candidate in-place for next iteration
                        cand["mask"] = ref_result["mask"]
                        cand["area_px"] = ref_result.get("area_px", int(np.asarray(ref_result["mask"], dtype=bool).sum()))
                        cand["select_reason"] = ref_result.get("select_reason", "refined")
                        cand["clicks_used"] = (
                            list(cand.get("clicks_used") or []) + new_points
                        )
                        # Save refined mask vis
                        ref_mask_vis = draw_numbered_marks(target_frame, [cand])
                        cv2.imwrite(
                            os.path.join(ref_dir, f"iter_{iter_idx:02d}_mask.png"),
                            ref_mask_vis,
                        )
                        # Loop continues; next iteration checks the new mask

            # Render accepted masks → 07_accepted_masks.png
            accepted_vis = draw_numbered_marks(target_frame, accepted_candidates)
            cv2.imwrite(os.path.join(target_dir, "07_accepted_masks.png"), accepted_vis)

            new_row = merge_accepted_masks_into_row(existing_row, accepted_candidates)
            out_handle.write(json.dumps(new_row) + "\n")
            out_handle.flush()

            stats["targets_processed"] += 1
            stats["masks_accepted"] += len(accepted_candidates)

    # Write summary
    summary_path = os.path.join(cfg.output_dir, "summary.json")
    with open(summary_path, "w") as f:
        json.dump(stats, f, indent=2)

    return stats
