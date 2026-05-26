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
    import os

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
