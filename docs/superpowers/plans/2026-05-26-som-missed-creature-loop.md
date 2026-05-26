# SoM Missed-Creature Discovery Loop — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a standalone stage that takes a video and an existing `frame_outputs_rle.json` from the every-frame text-prompted SAM3 agent, automatically selects K target frames, and adds masks for creatures the text-agent missed on those frames via a Set-of-Mark loop (dense candidate proposals + MLLM picks the real biological subjects by numbered index).

**Architecture:** Standalone driver `scripts/run_som_missed_creatures.py`, all logic in `nibi_model_compare/som_missed_creatures.py`, system prompts under `sam3/agent/system_prompts/`. The MLLM never produces raw (x, y) coordinates — it selects from numbered mask candidates that SAM3 already produced. Heavy debug-artefact retention so Claude inspects intermediate images at every important step during development.

**Tech Stack:** Python 3.10 + the project's existing `.venv`, `unittest` for tests (matches repo convention), SAM3 text-prompt inference via `sam3.agent.client_sam3.call_sam_service`, Anthropic Claude via `sam3.agent.client_claude.send_claude_request`, mask helpers via `sam3.agent.helpers.mask_overlap_removal` and `nibi_model_compare.frame_output_utils`.

**Spec:** `docs/superpowers/specs/2026-05-26-som-missed-creature-loop-design.md`

---

## File structure

| Path | Responsibility |
|---|---|
| `nibi_model_compare/som_missed_creatures.py` | All logic. Pure functions for frame selection, candidate generation, filtering, mark rendering, prompt building, response parsing, mask writeback, and a top-level `run_som_stage(...)` orchestrator. |
| `scripts/run_som_missed_creatures.py` | argparse-only CLI entrypoint. Parses args, calls `run_som_stage`. |
| `scripts/merge_som_outputs.py` | Post-run utility that merges the augmented per-target JSONL back into a fresh `frame_outputs_rle.json` (input file is never mutated). |
| `sam3/agent/system_prompts/system_prompt_som_missed_creature_underwater.txt` | Underwater-profile system prompt for the SoM judgment call. |
| `sam3/agent/system_prompts/system_prompt_som_missed_creature_general.txt` | General-profile system prompt. |
| `tests/test_som_missed_creatures.py` | Unit + integration tests with monkey-patched SAM3 and MLLM. |

Files that already exist and are reused unchanged: `sam3/agent/helpers/mask_overlap_removal.py`, `nibi_model_compare/frame_output_utils.py`, `nibi_model_compare/frame_quality.py`, `nibi_model_compare/keyframe_discovery.py`, `sam3/agent/client_claude.py`, `sam3/agent/client_llm.py`, `sam3/agent/client_sam3.py`.

---

## Task 1: Project skeleton + response parser

**Files:**
- Create: `nibi_model_compare/som_missed_creatures.py`
- Create: `tests/test_som_missed_creatures.py`

- [ ] **Step 1: Create the new module with the response-parser stub.**

```python
# nibi_model_compare/som_missed_creatures.py
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

_ANSWER_RE = re.compile(r"<answer>\s*(\{.*?\})\s*</answer>", re.DOTALL)


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
```

- [ ] **Step 2: Write the failing tests for the parser.**

```python
# tests/test_som_missed_creatures.py
import unittest

from nibi_model_compare.som_missed_creatures import parse_som_response


class ParseSomResponseTests(unittest.TestCase):
    def test_happy_path(self):
        text = '<answer>{"accepted_marks": [1, 3, 5]}</answer>'
        self.assertEqual(parse_som_response(text), [1, 3, 5])

    def test_prose_before_tag(self):
        text = (
            "Looking at the marked image, marks 2 and 4 look like substrate.\n"
            "Marks 1 and 3 are clearly biological.\n"
            '<answer>{"accepted_marks": [1, 3]}</answer>'
        )
        self.assertEqual(parse_som_response(text), [1, 3])

    def test_multiple_answer_blocks_takes_last(self):
        text = (
            '<answer>{"accepted_marks": [1]}</answer>\n'
            "wait, on reflection...\n"
            '<answer>{"accepted_marks": [1, 2]}</answer>'
        )
        self.assertEqual(parse_som_response(text), [1, 2])

    def test_empty_list_is_valid(self):
        text = '<answer>{"accepted_marks": []}</answer>'
        self.assertEqual(parse_som_response(text), [])

    def test_missing_tag_returns_empty(self):
        self.assertEqual(parse_som_response("just text"), [])

    def test_malformed_json_returns_empty(self):
        self.assertEqual(parse_som_response("<answer>{not json}</answer>"), [])

    def test_out_of_range_ids_filtered_when_valid_ids_given(self):
        text = '<answer>{"accepted_marks": [1, 99, 3]}</answer>'
        self.assertEqual(
            parse_som_response(text, valid_ids={1, 2, 3}),
            [1, 3],
        )

    def test_non_int_ids_filtered(self):
        text = '<answer>{"accepted_marks": [1, "2", 3.5, true, 4]}</answer>'
        self.assertEqual(parse_som_response(text), [1, 4])


if __name__ == "__main__":
    unittest.main()
```

- [ ] **Step 3: Run tests.**

```bash
cd /home/sbialek/ONC/sam3 && .venv/bin/python -m unittest tests.test_som_missed_creatures -v
```

Expected: all 8 tests pass (the parser was written first because it's a self-contained pure function — TDD here is "write the function and its tests in lockstep"). If any fail, fix and re-run.

- [ ] **Step 4: Commit.**

```bash
cd /home/sbialek/ONC/sam3
git add nibi_model_compare/som_missed_creatures.py tests/test_som_missed_creatures.py
git commit -m "$(cat <<'EOF'
Add SoM missed-creature module skeleton + response parser

First task of the SoM discovery loop. Adds the module file with a strict
<answer>{"accepted_marks":[...]}</answer> parser that handles prose
preludes, multiple tags (last wins), malformed JSON (returns empty),
and optional valid-id filtering.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 2: Frame selection

**Files:**
- Modify: `nibi_model_compare/som_missed_creatures.py` (add `select_target_frames`)
- Modify: `tests/test_som_missed_creatures.py` (add `SelectTargetFramesTests`)

Frame-selection contract: produce K target frame indices given the existing per-frame results. Skip frames that errored, were marked invalid, or had `num_masks == 0` *iff* `--include-zero-mask-frames` is False (the default — frames where text-agent produced nothing are still candidates for SoM by request, but the *default* is to skip them since they're often invalid).

- [ ] **Step 1: Write the failing tests.**

Append to `tests/test_som_missed_creatures.py` (above the `if __name__` line):

```python
from nibi_model_compare.som_missed_creatures import select_target_frames


class SelectTargetFramesTests(unittest.TestCase):
    def _row(self, idx, **kw):
        base = {"frame_index": idx, "num_masks": 3, "error": None, "skipped": False}
        base.update(kw)
        return base

    def test_uniform_spacing_picks_K_valid_frames(self):
        frame_results = [self._row(i) for i in range(20)]
        out = select_target_frames(frame_results, strategy="uniform", k=5)
        self.assertEqual(len(out), 5)
        # uniformly spaced across 0..19
        self.assertEqual(out, [0, 4, 9, 14, 19])

    def test_uniform_skips_errored(self):
        frame_results = [
            self._row(0),
            self._row(1, error="IndexError"),
            self._row(2),
            self._row(3, error="ValueError"),
            self._row(4),
            self._row(5),
        ]
        out = select_target_frames(frame_results, strategy="uniform", k=3)
        # valid frames are [0, 2, 4, 5]; uniform spacing of 3 over 4 picks
        # indices 0, 2, 3 of that list -> [0, 4, 5]
        self.assertEqual(out, [0, 4, 5])

    def test_uniform_skips_skipped_and_zero_mask(self):
        frame_results = [
            self._row(0),
            self._row(1, skipped=True),
            self._row(2, num_masks=0),
            self._row(3),
            self._row(4),
        ]
        out = select_target_frames(frame_results, strategy="uniform", k=3)
        # valid: [0, 3, 4] -> all picked
        self.assertEqual(out, [0, 3, 4])

    def test_uniform_returns_fewer_when_k_exceeds_valid(self):
        frame_results = [self._row(i, error="x") for i in range(5)]
        frame_results.append(self._row(5))
        out = select_target_frames(frame_results, strategy="uniform", k=10)
        self.assertEqual(out, [5])

    def test_explicit_override_wins(self):
        frame_results = [self._row(i) for i in range(20)]
        out = select_target_frames(
            frame_results, strategy="uniform", k=5, explicit=[3, 7, 11]
        )
        self.assertEqual(out, [3, 7, 11])

    def test_explicit_override_filters_invalid_indices(self):
        frame_results = [self._row(i) for i in range(5)]
        out = select_target_frames(
            frame_results, strategy="uniform", k=5, explicit=[1, 99, 3]
        )
        self.assertEqual(out, [1, 3])

    def test_motion_strategy_delegates(self):
        called = {}

        def fake_motion(frame_results, k):
            called["ok"] = (len(frame_results), k)
            return [0, 2, 4]

        frame_results = [self._row(i) for i in range(5)]
        out = select_target_frames(
            frame_results, strategy="motion", k=3,
            _motion_selector=fake_motion,
        )
        self.assertEqual(out, [0, 2, 4])
        self.assertEqual(called["ok"], (5, 3))

    def test_include_zero_mask_frames_flag(self):
        frame_results = [
            self._row(0, num_masks=0),
            self._row(1, num_masks=0),
            self._row(2),
        ]
        out = select_target_frames(
            frame_results, strategy="uniform", k=3,
            include_zero_mask_frames=True,
        )
        self.assertEqual(out, [0, 1, 2])
```

- [ ] **Step 2: Run tests, verify they fail with ImportError.**

```bash
cd /home/sbialek/ONC/sam3 && .venv/bin/python -m unittest tests.test_som_missed_creatures.SelectTargetFramesTests -v 2>&1 | tail -10
```

Expected: ImportError for `select_target_frames`.

- [ ] **Step 3: Implement `select_target_frames` in `nibi_model_compare/som_missed_creatures.py`.**

Append to the module:

```python
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
    step = (len(valid_indices) - 1) / (k - 1) if k > 1 else 0
    picked = [valid_indices[round(i * step)] for i in range(k)]
    return sorted(set(picked))


def _motion_keyframe_selector(valid_rows: list[dict], k: int) -> list[int]:
    """Delegate motion-based selection to the existing keyframe-discovery
    helper. Imported lazily because keyframe_discovery has heavy deps
    (cv2 + optical flow) we don't want to pull in for unit tests.
    """
    from nibi_model_compare.keyframe_discovery import select_motion_keyframes

    return select_motion_keyframes(
        frame_indices=[int(r["frame_index"]) for r in valid_rows],
        k=k,
    )
```

Note: `keyframe_discovery.select_motion_keyframes` is referenced by name. If that function doesn't exist with this exact signature, the engineer should: (1) read `nibi_model_compare/keyframe_discovery.py` to find the actual API, (2) adapt the call inside `_motion_keyframe_selector`, (3) keep the public `select_target_frames` signature stable so the tests above pass. The motion test mocks the selector, so the public signature is what matters.

- [ ] **Step 4: Run tests, verify pass.**

```bash
cd /home/sbialek/ONC/sam3 && .venv/bin/python -m unittest tests.test_som_missed_creatures -v 2>&1 | tail -5
```

Expected: all tests in `ParseSomResponseTests` and `SelectTargetFramesTests` pass.

- [ ] **Step 5: Commit.**

```bash
cd /home/sbialek/ONC/sam3
git add nibi_model_compare/som_missed_creatures.py tests/test_som_missed_creatures.py
git commit -m "$(cat <<'EOF'
Add SoM frame selector (uniform default, motion delegation, explicit override)

Picks K target frames for the SoM stage from frame_results.jsonl,
skipping errored / skipped / zero-mask frames by default. Uniform
spacing is the baseline; motion mode delegates to the existing
keyframe-discovery helper. An explicit override bypasses both.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 3: Candidate filter (IoU dedup + area + edge)

**Files:**
- Modify: `nibi_model_compare/som_missed_creatures.py` (add `filter_candidates`)
- Modify: `tests/test_som_missed_creatures.py` (add `FilterCandidatesTests`)

The filter takes a list of *candidate* mask records (output of the dense generator in Task 7) and the list of *existing* masks already on the frame, and returns the candidates that survive: not too small, not too large, not edge-clipped, and not duplicates of anything already covered.

Each candidate is a dict `{"mask": np.ndarray (bool, HxW), "bbox_xywh": [x, y, w, h], "score": float}`. Existing masks have the same shape.

- [ ] **Step 1: Write the failing tests.**

Append to `tests/test_som_missed_creatures.py`:

```python
import numpy as np

from nibi_model_compare.som_missed_creatures import filter_candidates


def _disk_mask(h, w, cy, cx, r):
    yy, xx = np.ogrid[:h, :w]
    return ((yy - cy) ** 2 + (xx - cx) ** 2) <= r * r


def _bbox(mask):
    ys, xs = np.where(mask)
    return [int(xs.min()), int(ys.min()),
            int(xs.max() - xs.min() + 1), int(ys.max() - ys.min() + 1)]


def _cand(mask, score=0.9):
    return {"mask": mask, "bbox_xywh": _bbox(mask), "score": score}


class FilterCandidatesTests(unittest.TestCase):
    H, W = 64, 64

    def test_drops_high_iou_against_existing(self):
        cand_mask = _disk_mask(self.H, self.W, 20, 20, 6)
        existing_mask = _disk_mask(self.H, self.W, 20, 20, 6)  # identical
        candidates = [_cand(cand_mask)]
        existing = [{"mask": existing_mask}]
        out = filter_candidates(
            candidates, existing,
            iou_dedup=0.3, min_area_px=4, max_area_frac=0.5,
            edge_tol_px=2,
        )
        self.assertEqual(out, [])

    def test_keeps_low_iou_against_existing(self):
        cand_mask = _disk_mask(self.H, self.W, 20, 20, 6)
        existing_mask = _disk_mask(self.H, self.W, 50, 50, 6)
        out = filter_candidates(
            [_cand(cand_mask)], [{"mask": existing_mask}],
            iou_dedup=0.3, min_area_px=4, max_area_frac=0.5,
            edge_tol_px=2,
        )
        self.assertEqual(len(out), 1)

    def test_drops_too_small(self):
        tiny = _disk_mask(self.H, self.W, 20, 20, 1)  # ~5 px
        out = filter_candidates(
            [_cand(tiny)], [],
            iou_dedup=0.3, min_area_px=20, max_area_frac=0.5,
            edge_tol_px=2,
        )
        self.assertEqual(out, [])

    def test_drops_too_large(self):
        huge = np.ones((self.H, self.W), dtype=bool)  # full frame
        out = filter_candidates(
            [_cand(huge)], [],
            iou_dedup=0.3, min_area_px=4, max_area_frac=0.5,
            edge_tol_px=2,
        )
        self.assertEqual(out, [])

    def test_drops_multi_edge_clipped(self):
        # touches top + left edges
        m = np.zeros((self.H, self.W), dtype=bool)
        m[0:10, 0:10] = True
        out = filter_candidates(
            [_cand(m)], [],
            iou_dedup=0.3, min_area_px=4, max_area_frac=0.5,
            edge_tol_px=2,
        )
        self.assertEqual(out, [])

    def test_keeps_single_edge_clipped(self):
        # touches only top
        m = np.zeros((self.H, self.W), dtype=bool)
        m[0:10, 20:30] = True
        out = filter_candidates(
            [_cand(m)], [],
            iou_dedup=0.3, min_area_px=4, max_area_frac=0.5,
            edge_tol_px=2,
        )
        self.assertEqual(len(out), 1)

    def test_no_existing_masks_keeps_valid_candidates(self):
        m = _disk_mask(self.H, self.W, 30, 30, 6)
        out = filter_candidates(
            [_cand(m)], [],
            iou_dedup=0.3, min_area_px=4, max_area_frac=0.5,
            edge_tol_px=2,
        )
        self.assertEqual(len(out), 1)
```

- [ ] **Step 2: Run tests, verify they fail with ImportError.**

```bash
cd /home/sbialek/ONC/sam3 && .venv/bin/python -m unittest tests.test_som_missed_creatures.FilterCandidatesTests -v 2>&1 | tail -5
```

Expected: ImportError for `filter_candidates`.

- [ ] **Step 3: Implement `filter_candidates`.**

Append to `nibi_model_compare/som_missed_creatures.py`:

```python
def _mask_iou(a, b) -> float:
    """IoU of two boolean numpy masks of the same shape."""
    import numpy as np

    a = np.asarray(a, dtype=bool)
    b = np.asarray(b, dtype=bool)
    inter = int(np.logical_and(a, b).sum())
    if inter == 0:
        return 0.0
    union = int(np.logical_or(a, b).sum())
    return inter / max(1, union)


def _touches_edges(mask, edge_tol_px: int) -> int:
    """Number of frame edges (top/bottom/left/right) the mask touches
    within ``edge_tol_px`` pixels."""
    import numpy as np

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

    Returns the surviving candidates in the same order they were given,
    each augmented with a ``drop_reason: None`` field (or annotated and
    *not* returned if dropped). Callers that want the full pre-filter
    list with drop reasons should call ``filter_candidates_with_reasons``
    instead (see below).
    """
    return [c for c, reason in filter_candidates_with_reasons(
        candidates, existing_masks,
        iou_dedup=iou_dedup, min_area_px=min_area_px,
        max_area_frac=max_area_frac, edge_tol_px=edge_tol_px,
    ) if reason is None]


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
```

- [ ] **Step 4: Run tests, verify pass.**

```bash
cd /home/sbialek/ONC/sam3 && .venv/bin/python -m unittest tests.test_som_missed_creatures -v 2>&1 | tail -5
```

Expected: all parser + selector + filter tests pass.

- [ ] **Step 5: Commit.**

```bash
cd /home/sbialek/ONC/sam3
git add nibi_model_compare/som_missed_creatures.py tests/test_som_missed_creatures.py
git commit -m "$(cat <<'EOF'
Add SoM candidate filter (IoU dedup + area + edge policy)

Drops SAM3 candidates that are duplicates of text-agent masks (IoU > τ),
too small, too large, or clipped against multiple frame edges. Exposes
filter_candidates_with_reasons() so debug artefacts can record why each
candidate was kept or dropped.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 4: Mark renderer

**Files:**
- Modify: `nibi_model_compare/som_missed_creatures.py` (add `draw_numbered_marks`)
- Modify: `tests/test_som_missed_creatures.py` (add `DrawNumberedMarksTests`)

This task produces a visual artefact (`04_marked.png`). Per spec, Claude inspects it during smoke testing — but for unit testing we only verify it doesn't crash, returns the right shape, and changes pixels relative to input.

- [ ] **Step 1: Write the failing tests.**

Append to `tests/test_som_missed_creatures.py`:

```python
from nibi_model_compare.som_missed_creatures import draw_numbered_marks


class DrawNumberedMarksTests(unittest.TestCase):
    H, W = 64, 64

    def _frame(self):
        return np.zeros((self.H, self.W, 3), dtype=np.uint8) + 100

    def _cand_mask(self, cy, cx):
        yy, xx = np.ogrid[:self.H, :self.W]
        return ((yy - cy) ** 2 + (xx - cx) ** 2) <= 5 * 5

    def test_returns_same_shape(self):
        frame = self._frame()
        cands = [{"mask": self._cand_mask(20, 20), "bbox_xywh": [15, 15, 11, 11]}]
        out = draw_numbered_marks(frame, cands)
        self.assertEqual(out.shape, frame.shape)
        self.assertEqual(out.dtype, frame.dtype)

    def test_modifies_pixels(self):
        frame = self._frame()
        cands = [{"mask": self._cand_mask(20, 20), "bbox_xywh": [15, 15, 11, 11]}]
        out = draw_numbered_marks(frame, cands)
        self.assertFalse(np.array_equal(frame, out),
                         "Expected the marked frame to differ from input")

    def test_zero_candidates_returns_copy(self):
        frame = self._frame()
        out = draw_numbered_marks(frame, [])
        self.assertEqual(out.shape, frame.shape)
        # Should not raise, should not modify input
        np.testing.assert_array_equal(frame, np.zeros_like(frame) + 100)

    def test_handles_many_candidates(self):
        # Stress test: 20 marks in dense scene
        frame = self._frame()
        cands = []
        for i in range(20):
            cy = 8 + (i // 5) * 12
            cx = 8 + (i % 5) * 12
            cands.append({"mask": self._cand_mask(cy, cx),
                          "bbox_xywh": [cx - 5, cy - 5, 11, 11]})
        out = draw_numbered_marks(frame, cands)
        self.assertEqual(out.shape, frame.shape)
```

- [ ] **Step 2: Run tests, verify they fail with ImportError.**

```bash
cd /home/sbialek/ONC/sam3 && .venv/bin/python -m unittest tests.test_som_missed_creatures.DrawNumberedMarksTests -v 2>&1 | tail -5
```

Expected: ImportError for `draw_numbered_marks`.

- [ ] **Step 3: Implement `draw_numbered_marks`.**

Append to `nibi_model_compare/som_missed_creatures.py`:

```python
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

    Implementation notes for the engineer:
      - Use cv2 for label rendering (existing project style; see
        nibi_model_compare/postprocess_stage_missed_creatures.py for the
        ``cv2.putText`` font conventions used by the rest of the repo).
      - Use a ``ColorPalette`` (sam3/agent/helpers/som_utils.py exposes
        one) for per-mark color so adjacent marks are visually distinct.
    """
    import cv2
    import numpy as np

    from sam3.agent.helpers.som_utils import ColorPalette

    out = frame_bgr.copy()
    if not candidates:
        return out

    palette = ColorPalette.default()
    h, w = out.shape[:2]

    for idx, cand in enumerate(candidates, start=1):
        mask = np.asarray(cand["mask"], dtype=bool)
        color = palette.by_idx(idx, seed=palette_seed)  # (B, G, R) uint8 triple
        # Translucent mask overlay
        overlay = out.copy()
        overlay[mask] = (
            (1.0 - alpha) * out[mask].astype(np.float32)
            + alpha * np.asarray(color, dtype=np.float32)
        ).astype(np.uint8)
        out = overlay

        # Outline contour for crisp boundary
        contours, _ = cv2.findContours(
            mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        cv2.drawContours(out, contours, -1, [int(c) for c in color], 1)

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
                    [int(c) for c in color], thickness, cv2.LINE_AA)

    return out
```

Note on `ColorPalette.by_idx`: if `som_utils.ColorPalette` doesn't expose this exact method, read `sam3/agent/helpers/som_utils.py` to find the actual API and adapt. The function must return a BGR triple suitable for cv2. If no usable palette exists, fall back to:

```python
rng = np.random.default_rng(palette_seed + idx)
color = tuple(int(x) for x in rng.integers(80, 255, size=3))
```

- [ ] **Step 4: Run tests, verify pass.**

```bash
cd /home/sbialek/ONC/sam3 && .venv/bin/python -m unittest tests.test_som_missed_creatures -v 2>&1 | tail -5
```

Expected: all tests pass.

- [ ] **Step 5: Visual check — write the marked image to disk and inspect.**

Create a quick standalone snippet (don't commit this — it's a one-off check):

```bash
cd /home/sbialek/ONC/sam3 && .venv/bin/python - <<'PY'
import cv2
import numpy as np
from nibi_model_compare.som_missed_creatures import draw_numbered_marks

H, W = 256, 256
frame = (np.zeros((H, W, 3), dtype=np.uint8) + 80)
cands = []
for i in range(8):
    yy, xx = np.ogrid[:H, :W]
    cy = 32 + (i // 4) * 96
    cx = 32 + (i % 4) * 64
    m = ((yy - cy) ** 2 + (xx - cx) ** 2) <= 16 * 16
    cands.append({"mask": m, "bbox_xywh": [cx - 16, cy - 16, 33, 33]})

out = draw_numbered_marks(frame, cands)
cv2.imwrite("/tmp/som_marks_check.png", out)
print("Wrote /tmp/som_marks_check.png")
PY
```

Then use the Read tool on `/tmp/som_marks_check.png` and confirm:

- 8 numbered marks are visible, labels readable
- Numbers don't overlap each other
- Each mark has both a translucent fill and an outlined contour

If anything looks wrong, adjust the rendering in Step 3 and re-check.

- [ ] **Step 6: Commit.**

```bash
cd /home/sbialek/ONC/sam3
git add nibi_model_compare/som_missed_creatures.py tests/test_som_missed_creatures.py
git commit -m "$(cat <<'EOF'
Render numbered SoM marks on candidate masks

draw_numbered_marks() takes a BGR frame plus a list of {mask, bbox_xywh}
candidates and renders each as a translucent overlay + contour outline +
centered numeric label with a shadow for readability. Marks are numbered
1..N in input order; callers should pre-sort by descending area.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 5: Prompt builder

**Files:**
- Modify: `nibi_model_compare/som_missed_creatures.py` (add `build_som_prompt_messages`)
- Modify: `tests/test_som_missed_creatures.py` (add `BuildSomPromptMessagesTests`)

Builds the multi-image MLLM payload: one marked target frame + 2-4 unmarked reference frames + the user query. The exact message shape needs to match what `sam3.agent.client_llm.send_generate_request` / `sam3.agent.client_claude.send_claude_request` expect.

- [ ] **Step 1: Read the existing send-request signatures.**

```bash
cd /home/sbialek/ONC/sam3 && grep -n "def send_claude_request\|def send_generate_request" sam3/agent/client_claude.py sam3/agent/client_llm.py | head
```

The engineer should confirm both expect a `messages: list[dict]` argument where messages contain `{"role": "system"|"user"|"assistant", "content": [...]}` and content items are either `{"type": "text", "text": "..."}` or `{"type": "image", "image": "<path>"}`. This is the contract `agent_core.py` already uses; follow it.

- [ ] **Step 2: Write the failing tests.**

Append to `tests/test_som_missed_creatures.py`:

```python
import os
import tempfile

from nibi_model_compare.som_missed_creatures import build_som_prompt_messages


class BuildSomPromptMessagesTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.target_path = os.path.join(self.tmp.name, "target.png")
        self.neighbour_paths = [
            os.path.join(self.tmp.name, f"n{i}.png") for i in range(4)
        ]
        # Create dummy files so the builder doesn't reject them
        from PIL import Image
        for p in [self.target_path, *self.neighbour_paths]:
            Image.new("RGB", (32, 32)).save(p)

    def tearDown(self):
        self.tmp.cleanup()

    def test_messages_have_system_and_user_roles(self):
        msgs = build_som_prompt_messages(
            system_prompt="SYS",
            target_image_path=self.target_path,
            neighbour_image_paths=self.neighbour_paths,
            initial_text_prompt="small creatures",
            num_marks=5,
        )
        self.assertEqual(msgs[0]["role"], "system")
        self.assertEqual(msgs[0]["content"], "SYS")
        self.assertEqual(msgs[1]["role"], "user")

    def test_user_message_includes_target_then_neighbours(self):
        msgs = build_som_prompt_messages(
            system_prompt="SYS",
            target_image_path=self.target_path,
            neighbour_image_paths=self.neighbour_paths,
            initial_text_prompt="creature",
            num_marks=3,
        )
        content = msgs[1]["content"]
        image_items = [c for c in content if c.get("type") == "image"]
        self.assertEqual(len(image_items), 1 + len(self.neighbour_paths))
        self.assertEqual(image_items[0]["image"], self.target_path)
        for nb_item, nb_path in zip(image_items[1:], self.neighbour_paths):
            self.assertEqual(nb_item["image"], nb_path)

    def test_user_message_mentions_query_and_num_marks(self):
        msgs = build_som_prompt_messages(
            system_prompt="SYS",
            target_image_path=self.target_path,
            neighbour_image_paths=self.neighbour_paths,
            initial_text_prompt="small creatures",
            num_marks=7,
        )
        text_blobs = [
            c["text"] for c in msgs[1]["content"] if c.get("type") == "text"
        ]
        joined = " ".join(text_blobs)
        self.assertIn("small creatures", joined)
        self.assertIn("7", joined)
        self.assertIn("accepted_marks", joined)

    def test_neighbour_list_empty_is_allowed(self):
        msgs = build_som_prompt_messages(
            system_prompt="SYS",
            target_image_path=self.target_path,
            neighbour_image_paths=[],
            initial_text_prompt="x",
            num_marks=1,
        )
        content = msgs[1]["content"]
        image_items = [c for c in content if c.get("type") == "image"]
        self.assertEqual(len(image_items), 1)
```

- [ ] **Step 3: Run tests, verify they fail.**

```bash
cd /home/sbialek/ONC/sam3 && .venv/bin/python -m unittest tests.test_som_missed_creatures.BuildSomPromptMessagesTests -v 2>&1 | tail -5
```

- [ ] **Step 4: Implement `build_som_prompt_messages`.**

Append to `nibi_model_compare/som_missed_creatures.py`:

```python
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
```

- [ ] **Step 5: Run tests, verify pass.**

```bash
cd /home/sbialek/ONC/sam3 && .venv/bin/python -m unittest tests.test_som_missed_creatures -v 2>&1 | tail -5
```

- [ ] **Step 6: Commit.**

```bash
cd /home/sbialek/ONC/sam3
git add nibi_model_compare/som_missed_creatures.py tests/test_som_missed_creatures.py
git commit -m "$(cat <<'EOF'
Build multi-image SoM prompt message payload

build_som_prompt_messages() assembles the system + user payload that
goes to the MLLM client (Claude or OpenAI-compatible). Includes target
+ neighbour images, the user's original creature query, and a strict
<answer>{"accepted_marks":[...]}</answer> tail instruction. Honours the
'do not require motion' rule from the spec.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 6: Mask merger / writeback

**Files:**
- Modify: `nibi_model_compare/som_missed_creatures.py` (add `merge_accepted_masks_into_row`)
- Modify: `tests/test_som_missed_creatures.py` (add `MergeAcceptedMasksTests`)

Given an existing `frame_outputs_rle.json`-style row and the accepted candidates, produce the augmented row. New masks get `obj_id`s above `max(existing) + 1`, source-tagged `"som"`. RLE encoding uses the existing helper.

- [ ] **Step 1: Write the failing tests.**

Append to `tests/test_som_missed_creatures.py`:

```python
from nibi_model_compare.som_missed_creatures import merge_accepted_masks_into_row


class MergeAcceptedMasksTests(unittest.TestCase):
    def _disk_cand(self, cy, cx, r, h=64, w=64):
        yy, xx = np.ogrid[:h, :w]
        m = ((yy - cy) ** 2 + (xx - cx) ** 2) <= r * r
        ys, xs = np.where(m)
        bbox = [int(xs.min()), int(ys.min()),
                int(xs.max() - xs.min() + 1),
                int(ys.max() - ys.min() + 1)]
        return {"mask": m, "bbox_xywh": bbox, "score": 0.8}

    def test_appends_new_objs_with_unique_ids(self):
        existing_row = {
            "frame_index": 5,
            "out_obj_ids": [1, 2, 3],
            "out_binary_masks_rle": [{}, {}, {}],   # opaque payloads
            "out_boxes_xywh": [[0, 0, 1, 1], [0, 0, 1, 1], [0, 0, 1, 1]],
            "out_probs": [0.9, 0.9, 0.9],
            "out_tracker_probs": [0.9, 0.9, 0.9],
        }
        accepted = [self._disk_cand(20, 20, 6), self._disk_cand(40, 40, 6)]
        out = merge_accepted_masks_into_row(existing_row, accepted)
        self.assertEqual(out["frame_index"], 5)
        self.assertEqual(out["out_obj_ids"], [1, 2, 3, 4, 5])
        self.assertEqual(out["added_obj_ids"], [4, 5])
        self.assertEqual(out["source"], "som")
        self.assertEqual(out["source_per_obj_id"], {
            "1": "text_agent", "2": "text_agent", "3": "text_agent",
            "4": "som", "5": "som",
        })
        self.assertEqual(len(out["out_binary_masks_rle"]), 5)
        self.assertEqual(len(out["out_boxes_xywh"]), 5)

    def test_empty_accepted_returns_passthrough_row(self):
        existing_row = {
            "frame_index": 5,
            "out_obj_ids": [1, 2],
            "out_binary_masks_rle": [{}, {}],
            "out_boxes_xywh": [[0, 0, 1, 1], [0, 0, 1, 1]],
            "out_probs": [0.9, 0.9],
            "out_tracker_probs": [0.9, 0.9],
        }
        out = merge_accepted_masks_into_row(existing_row, [])
        self.assertEqual(out["out_obj_ids"], [1, 2])
        self.assertEqual(out["added_obj_ids"], [])
        self.assertEqual(out["source"], "som")
        self.assertEqual(out["source_per_obj_id"], {
            "1": "text_agent", "2": "text_agent",
        })

    def test_no_existing_objs_starts_from_id_1(self):
        existing_row = {
            "frame_index": 5,
            "out_obj_ids": [],
            "out_binary_masks_rle": [],
            "out_boxes_xywh": [],
            "out_probs": [],
            "out_tracker_probs": [],
        }
        accepted = [self._disk_cand(20, 20, 6)]
        out = merge_accepted_masks_into_row(existing_row, accepted)
        self.assertEqual(out["out_obj_ids"], [1])
        self.assertEqual(out["added_obj_ids"], [1])

    def test_input_row_not_mutated(self):
        existing_row = {
            "frame_index": 5,
            "out_obj_ids": [1],
            "out_binary_masks_rle": [{}],
            "out_boxes_xywh": [[0, 0, 1, 1]],
            "out_probs": [0.9],
            "out_tracker_probs": [0.9],
        }
        merge_accepted_masks_into_row(existing_row, [self._disk_cand(20, 20, 6)])
        self.assertEqual(existing_row["out_obj_ids"], [1])
```

- [ ] **Step 2: Run tests, verify they fail.**

```bash
cd /home/sbialek/ONC/sam3 && .venv/bin/python -m unittest tests.test_som_missed_creatures.MergeAcceptedMasksTests -v 2>&1 | tail -5
```

- [ ] **Step 3: Implement the merger.**

Append to `nibi_model_compare/som_missed_creatures.py`:

```python
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
    existing_ids = list(row.get("out_obj_ids", []))
    next_id = (max(existing_ids) + 1) if existing_ids else 1

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
    row["source_per_obj_id"] = {
        str(oid): ("som" if oid in added_ids else "text_agent")
        for oid in row["out_obj_ids"]
    }
    return row
```

- [ ] **Step 4: Run tests, verify pass.**

```bash
cd /home/sbialek/ONC/sam3 && .venv/bin/python -m unittest tests.test_som_missed_creatures -v 2>&1 | tail -5
```

- [ ] **Step 5: Commit.**

```bash
cd /home/sbialek/ONC/sam3
git add nibi_model_compare/som_missed_creatures.py tests/test_som_missed_creatures.py
git commit -m "$(cat <<'EOF'
Merge accepted SoM masks into per-frame output row

merge_accepted_masks_into_row() appends accepted candidates to an
existing frame_outputs_rle.json row with fresh obj_ids, a source tag,
and a source_per_obj_id map so downstream consumers can distinguish
text-agent masks from SoM-added ones. Input row is never mutated.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 7: Dense candidate generator

**Files:**
- Modify: `nibi_model_compare/som_missed_creatures.py` (add `generate_dense_candidates`)
- Modify: `tests/test_som_missed_creatures.py` (add `GenerateDenseCandidatesTests`)

This task replaces what the spec called "SAM3 automatic mask generation". The MVP uses SAM3's existing text-prompted API (`sam3.agent.client_sam3.call_sam_service`) with a *list of broad prompts* (e.g. `"creature"`, `"animal"`, `"organism"`) and unions the resulting masks. This avoids implementing a real grid-sampled AMG and reuses infrastructure that already works end-to-end. If smoke-test recall is poor, evolution path B (true AMG via point grid) replaces this function — the function's input/output contract stays the same.

- [ ] **Step 1: Confirm the SAM3 text-prompt API shape.**

```bash
cd /home/sbialek/ONC/sam3 && grep -n "def call_sam_service\|def sam3_inference" sam3/agent/client_sam3.py
```

Read the function signatures and verify: `call_sam_service(image_path, text_prompt, output_folder_path)` writes a JSON to a path and returns the path. The JSON has `pred_masks` (list of RLE), `pred_boxes`, `pred_scores`, `orig_img_h`, `orig_img_w`. This matches what `agent_core.py` does.

- [ ] **Step 2: Write the failing tests.**

Append to `tests/test_som_missed_creatures.py`:

```python
from nibi_model_compare.som_missed_creatures import generate_dense_candidates


class GenerateDenseCandidatesTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        from PIL import Image
        self.img_path = os.path.join(self.tmp.name, "frame.png")
        Image.new("RGB", (64, 64)).save(self.img_path)

    def tearDown(self):
        self.tmp.cleanup()

    def _write_sam_result(self, path, masks_xy_pairs):
        """Helper: write a JSON in the shape call_sam_service emits."""
        import json
        from nibi_model_compare.frame_output_utils import encode_binary_mask_to_rle
        rles = []
        boxes = []
        for cy, cx, r in masks_xy_pairs:
            yy, xx = np.ogrid[:64, :64]
            m = ((yy - cy) ** 2 + (xx - cx) ** 2) <= r * r
            rles.append(encode_binary_mask_to_rle(m))
            ys, xs = np.where(m)
            boxes.append([int(xs.min()), int(ys.min()),
                          int(xs.max() - xs.min() + 1),
                          int(ys.max() - ys.min() + 1)])
        payload = {
            "original_image_path": self.img_path,
            "orig_img_h": 64,
            "orig_img_w": 64,
            "pred_masks": rles,
            "pred_boxes": boxes,
            "pred_scores": [0.9] * len(rles),
        }
        with open(path, "w") as f:
            json.dump(payload, f)
        return path

    def test_calls_sam_per_prompt_and_unions(self):
        calls = []

        def fake_sam(image_path, text_prompt, output_folder_path):
            calls.append(text_prompt)
            n = len(calls)
            out_path = os.path.join(output_folder_path, f"out_{n}.json")
            if text_prompt == "creature":
                return self._write_sam_result(out_path, [(20, 20, 5)])
            if text_prompt == "animal":
                return self._write_sam_result(out_path, [(40, 40, 5)])
            return self._write_sam_result(out_path, [])

        out = generate_dense_candidates(
            image_path=self.img_path,
            broad_prompts=["creature", "animal"],
            output_folder=self.tmp.name,
            _call_sam_service=fake_sam,
        )
        self.assertEqual(calls, ["creature", "animal"])
        self.assertEqual(len(out), 2)
        for cand in out:
            self.assertIn("mask", cand)
            self.assertIn("bbox_xywh", cand)
            self.assertIn("score", cand)

    def test_zero_results_returns_empty(self):
        def fake_sam(image_path, text_prompt, output_folder_path):
            out_path = os.path.join(output_folder_path, f"out_{text_prompt}.json")
            return self._write_sam_result(out_path, [])

        out = generate_dense_candidates(
            image_path=self.img_path,
            broad_prompts=["creature"],
            output_folder=self.tmp.name,
            _call_sam_service=fake_sam,
        )
        self.assertEqual(out, [])

    def test_internal_dedup_by_iou(self):
        # Same prompt produces overlapping masks across calls -> intra-batch dedup
        def fake_sam(image_path, text_prompt, output_folder_path):
            n = len([f for f in os.listdir(output_folder_path)
                     if f.endswith(".json")])
            out_path = os.path.join(output_folder_path, f"out_{n}.json")
            return self._write_sam_result(out_path, [(20, 20, 5)])

        out = generate_dense_candidates(
            image_path=self.img_path,
            broad_prompts=["creature", "animal"],
            output_folder=self.tmp.name,
            _call_sam_service=fake_sam,
            internal_iou_dedup=0.5,
        )
        # Two identical masks -> dedup keeps one
        self.assertEqual(len(out), 1)
```

- [ ] **Step 3: Run tests, verify failure.**

```bash
cd /home/sbialek/ONC/sam3 && .venv/bin/python -m unittest tests.test_som_missed_creatures.GenerateDenseCandidatesTests -v 2>&1 | tail -10
```

- [ ] **Step 4: Implement `generate_dense_candidates`.**

Append to `nibi_model_compare/som_missed_creatures.py`:

```python
DEFAULT_BROAD_PROMPTS = ("creature", "animal", "organism")


def generate_dense_candidates(
    *,
    image_path: str,
    broad_prompts: list[str] | tuple[str, ...] = DEFAULT_BROAD_PROMPTS,
    output_folder: str,
    internal_iou_dedup: float = 0.5,
    _call_sam_service=None,
) -> list[dict]:
    """Produce dense SoM candidates for one frame.

    MVP strategy: run SAM3 text-prompted inference once per broad prompt
    (e.g. "creature", "animal", "organism"), parse each result file,
    union the masks, and dedupe intra-batch by IoU.

    Each returned dict has ``{"mask": np.ndarray (bool), "bbox_xywh":
    [x, y, w, h], "score": float, "source_prompt": str}``.

    The contract is the only thing other components depend on; if smoke
    tests show poor recall, a real grid-sampled AMG can replace this
    body without touching downstream code.
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

    pooled: list[dict] = []
    for prompt in broad_prompts:
        result_path = call_sam(
            image_path=image_path,
            text_prompt=prompt,
            output_folder_path=output_folder,
        )
        with open(result_path, "r", encoding="utf-8") as f:
            payload = json.load(f)
        rles = payload.get("pred_masks") or []
        boxes = payload.get("pred_boxes") or []
        scores = payload.get("pred_scores") or [0.0] * len(rles)
        for rle, bbox, score in zip(rles, boxes, scores):
            mask = decode_rle_to_mask(rle).astype(bool)
            pooled.append({
                "mask": mask,
                "bbox_xywh": list(bbox),
                "score": float(score),
                "source_prompt": prompt,
            })

    # Intra-batch dedup by IoU: walk in descending score order, keep cand
    # only if it has no >iou_dedup overlap with any already-kept mask.
    pooled.sort(key=lambda c: -c["score"])
    kept: list[dict] = []
    for cand in pooled:
        if any(_mask_iou(cand["mask"], k["mask"]) > internal_iou_dedup
               for k in kept):
            continue
        kept.append(cand)
    return kept
```

- [ ] **Step 5: Run tests, verify pass.**

```bash
cd /home/sbialek/ONC/sam3 && .venv/bin/python -m unittest tests.test_som_missed_creatures -v 2>&1 | tail -5
```

- [ ] **Step 6: Commit.**

```bash
cd /home/sbialek/ONC/sam3
git add nibi_model_compare/som_missed_creatures.py tests/test_som_missed_creatures.py
git commit -m "$(cat <<'EOF'
Generate dense SoM candidates via broad-prompt SAM3 union

generate_dense_candidates() runs SAM3 text-mode inference once per
broad prompt (default: creature/animal/organism), decodes the RLE
masks, and dedupes intra-batch by IoU. Output contract is stable so a
true grid-sampled AMG can replace this body later (evolution path B in
the spec).

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 8: System prompts

**Files:**
- Create: `sam3/agent/system_prompts/system_prompt_som_missed_creature_underwater.txt`
- Create: `sam3/agent/system_prompts/system_prompt_som_missed_creature_general.txt`

These are text files; no unit test. They are validated qualitatively in the smoke test (Task 12) by reading the MLLM's reasoning trace and accept/reject decisions.

- [ ] **Step 1: Write the underwater system prompt.**

Create `sam3/agent/system_prompts/system_prompt_som_missed_creature_underwater.txt`:

```
You are an expert in marine biology and an annotator for an underwater
video labelling pipeline. You will be shown one TARGET image annotated
with numbered marks on candidate masks, plus 2-4 unmarked REFERENCE
images from times near the target. Your job is to decide which marks on
the TARGET correspond to real biological subjects matching the user's
creature query.

Rules:

1. Each numbered mark covers a region that SAM3 thinks is an object.
   Some of these are creatures; many are substrate, debris, glare,
   silt clouds, or bits of scenery. You must filter.
2. Use the REFERENCE images as additional perspectives. Some creatures
   move and some are stationary - DO NOT REQUIRE MOTION to accept a
   mark. A real biological subject should be visible at roughly the
   same place across the references, possibly with small shifts due to
   camera drift or lighting changes. Floating debris and transient
   artefacts (glare, lighting flashes) often look different across
   frames or aren't present in every reference.
3. Tight, focused marks on a clearly biological subject (a crab body,
   a sea star arm, a snail shell, a worm tube) are ACCEPTABLE. Loose
   marks that cover mostly substrate or a large patch of sediment with
   only a fragment of creature visible are REJECTABLE.
4. If two marks cover the same creature, you may accept BOTH if they
   each show a useful biological subject (e.g. body vs. extended
   limb); when in doubt, accept the one that fits the creature more
   tightly.
5. If you cannot tell whether a mark is biological or substrate,
   prefer REJECT. False positives hurt this dataset more than false
   negatives because a downstream MLLM verification pass cannot easily
   undo bad accepts but missed accepts can be caught by a later pass.

Output format:

- Free text reasoning of any length explaining which marks you accept
  and why.
- Then, on its own line at the end of your response, EXACTLY ONE tag:
    <answer>{"accepted_marks": [<int>, ...]}</answer>
- The list may be empty if no marks are biological.
- Mark ids must be integers in the valid range you were told about.
- Do not write anything after the </answer> tag.
```

- [ ] **Step 2: Write the general (non-underwater) system prompt.**

Create `sam3/agent/system_prompts/system_prompt_som_missed_creature_general.txt`:

```
You will be shown one TARGET image annotated with numbered marks on
candidate masks, plus 2-4 unmarked REFERENCE images from times near
the target. Your job is to decide which marks on the TARGET correspond
to real subjects matching the user's query.

Rules:

1. Each numbered mark covers a region that SAM3 thinks is an object.
   Some of these match the query; many are background, clutter, or
   over-segmented surface texture. You must filter.
2. Use the REFERENCE images as additional perspectives. Some subjects
   move and some are stationary - DO NOT REQUIRE MOTION to accept a
   mark. A real subject should be visible at roughly the same place
   across the references, possibly with small shifts due to camera
   drift or lighting changes.
3. Tight, focused marks on a clearly matching subject are ACCEPTABLE.
   Loose marks that cover mostly background or only fragments of a
   subject are REJECTABLE.
4. If you cannot tell whether a mark matches the query, prefer REJECT.
   False positives hurt this dataset more than false negatives.

Output format:

- Free text reasoning of any length.
- End your response with EXACTLY ONE tag on its own line:
    <answer>{"accepted_marks": [<int>, ...]}</answer>
- Empty list is valid.
- Mark ids must be integers in the valid range you were told about.
- Do not write anything after the </answer> tag.
```

- [ ] **Step 3: Add a tiny sanity test that the prompts load.**

Append to `tests/test_som_missed_creatures.py`:

```python
from nibi_model_compare.som_missed_creatures import load_system_prompt


class LoadSystemPromptTests(unittest.TestCase):
    def test_loads_underwater(self):
        body = load_system_prompt("underwater")
        self.assertIn("marine biology", body.lower())
        self.assertIn("<answer>", body)

    def test_loads_general(self):
        body = load_system_prompt("general")
        self.assertIn("<answer>", body)
        self.assertNotIn("marine biology", body.lower())

    def test_unknown_profile_raises(self):
        with self.assertRaises(ValueError):
            load_system_prompt("nonsense")
```

- [ ] **Step 4: Implement the loader.**

Append to `nibi_model_compare/som_missed_creatures.py`:

```python
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
```

- [ ] **Step 5: Run tests, verify pass.**

```bash
cd /home/sbialek/ONC/sam3 && .venv/bin/python -m unittest tests.test_som_missed_creatures -v 2>&1 | tail -5
```

- [ ] **Step 6: Commit.**

```bash
cd /home/sbialek/ONC/sam3
git add sam3/agent/system_prompts/system_prompt_som_missed_creature_underwater.txt \
        sam3/agent/system_prompts/system_prompt_som_missed_creature_general.txt \
        nibi_model_compare/som_missed_creatures.py \
        tests/test_som_missed_creatures.py
git commit -m "$(cat <<'EOF'
Add SoM system prompts (underwater + general) and loader

Underwater profile coaches the MLLM with the spec's 'do not require
motion' rule, asks for free-text reasoning then a trailing
<answer>{"accepted_marks":[...]}</answer> tag, and prefers reject when
uncertain (false positives hurt this dataset more than false
negatives). load_system_prompt() resolves to the right file by
profile.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 9: Driver / orchestrator

**Files:**
- Modify: `nibi_model_compare/som_missed_creatures.py` (add `run_som_stage`)
- Modify: `tests/test_som_missed_creatures.py` (add `RunSomStageTests`)

This task wires everything together end-to-end with monkey-patched SAM3 and MLLM clients. The driver is the only function that reads/writes the video and the output JSONL.

- [ ] **Step 1: Write the failing integration tests.**

Append to `tests/test_som_missed_creatures.py`:

```python
import json as _json

from nibi_model_compare.som_missed_creatures import (
    run_som_stage,
    SomStageConfig,
)


class RunSomStageTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.workdir = self.tmp.name

        # Fake video: 60 black frames @ 30fps as PNG sequence (we mock
        # the frame loader, so we don't need an actual mp4 here)
        self.video_path = os.path.join(self.workdir, "fake.mp4")
        with open(self.video_path, "w") as f:
            f.write("(stub)")

        # Existing frame_results.jsonl
        self.frame_results_path = os.path.join(self.workdir, "frame_results.jsonl")
        with open(self.frame_results_path, "w") as f:
            for i in range(60):
                row = {
                    "frame_index": i,
                    "num_masks": 2,
                    "error": None,
                    "skipped": False,
                }
                f.write(_json.dumps(row) + "\n")

        # Existing frame_outputs_rle.json with two text-agent masks per frame
        self.frame_outputs_path = os.path.join(self.workdir, "frame_outputs_rle.json")
        from nibi_model_compare.frame_output_utils import encode_binary_mask_to_rle
        frames = []
        for i in range(60):
            yy, xx = np.ogrid[:64, :64]
            m1 = ((yy - 10) ** 2 + (xx - 10) ** 2) <= 4 * 4
            m2 = ((yy - 50) ** 2 + (xx - 50) ** 2) <= 4 * 4
            frames.append({
                "frame_index": i,
                "out_obj_ids": [1, 2],
                "out_binary_masks_rle": [
                    encode_binary_mask_to_rle(m1),
                    encode_binary_mask_to_rle(m2),
                ],
                "out_boxes_xywh": [[6, 6, 9, 9], [46, 46, 9, 9]],
                "out_probs": [0.9, 0.9],
                "out_tracker_probs": [0.9, 0.9],
            })
        with open(self.frame_outputs_path, "w") as f:
            _json.dump({"format_version": 2, "frames": frames}, f)

        # Fake video frame loader: every frame is a 64x64 grey image
        def fake_load_frame(video_path, frame_index):
            return np.full((64, 64, 3), 80, dtype=np.uint8)
        self.fake_load_frame = fake_load_frame

        # Fake SAM3 service: always returns one "new" mask in the centre
        # (not covered by the existing two corner masks)
        def fake_sam(image_path, text_prompt, output_folder_path):
            out_path = os.path.join(
                output_folder_path,
                f"sam_{text_prompt}_{os.path.basename(image_path)}.json",
            )
            from nibi_model_compare.frame_output_utils import encode_binary_mask_to_rle
            yy, xx = np.ogrid[:64, :64]
            m = ((yy - 32) ** 2 + (xx - 32) ** 2) <= 5 * 5
            ys, xs = np.where(m)
            payload = {
                "original_image_path": image_path,
                "orig_img_h": 64,
                "orig_img_w": 64,
                "pred_masks": [encode_binary_mask_to_rle(m)],
                "pred_boxes": [[int(xs.min()), int(ys.min()),
                                int(xs.max() - xs.min() + 1),
                                int(ys.max() - ys.min() + 1)]],
                "pred_scores": [0.85],
            }
            with open(out_path, "w") as f:
                _json.dump(payload, f)
            return out_path
        self.fake_sam = fake_sam

        # Fake MLLM: always accepts mark 1
        def fake_mllm(messages, **_kw):
            return '<answer>{"accepted_marks": [1]}</answer>'
        self.fake_mllm = fake_mllm

    def tearDown(self):
        self.tmp.cleanup()

    def _config(self, **overrides):
        cfg = dict(
            video_path=self.video_path,
            frame_results_path=self.frame_results_path,
            frame_outputs_path=self.frame_outputs_path,
            output_dir=os.path.join(self.workdir, "som_out"),
            prompt_profile="underwater",
            initial_text_prompt="small creatures",
            num_target_frames=3,
            frame_selection_strategy="uniform",
            target_frames_explicit=None,
            broad_prompts=["creature"],
            num_neighbours=2,
            neighbour_offset_frames=15,
            iou_dedup=0.3,
            min_area_px=4,
            max_area_frac=0.5,
            edge_tol_px=2,
            internal_iou_dedup=0.5,
            max_mllm_calls=100,
        )
        cfg.update(overrides)
        return SomStageConfig(**cfg)

    def test_end_to_end_appends_one_mask_per_target_frame(self):
        cfg = self._config()
        result = run_som_stage(
            cfg,
            _load_video_frame=self.fake_load_frame,
            _call_sam_service=self.fake_sam,
            _send_mllm_request=self.fake_mllm,
        )

        self.assertEqual(result["targets_processed"], 3)
        # Augmented JSONL exists with one row per target
        augmented = os.path.join(cfg.output_dir, "augmented_frame_outputs.jsonl")
        with open(augmented) as f:
            rows = [_json.loads(line) for line in f]
        self.assertEqual(len(rows), 3)
        for row in rows:
            # Each row has the 2 original + 1 new mask
            self.assertEqual(len(row["out_obj_ids"]), 3)
            self.assertEqual(row["added_obj_ids"], [3])
            self.assertEqual(row["source"], "som")

    def test_skips_frame_when_amg_returns_zero(self):
        def empty_sam(image_path, text_prompt, output_folder_path):
            out_path = os.path.join(output_folder_path, "empty.json")
            with open(out_path, "w") as f:
                _json.dump({
                    "original_image_path": image_path,
                    "orig_img_h": 64, "orig_img_w": 64,
                    "pred_masks": [], "pred_boxes": [], "pred_scores": [],
                }, f)
            return out_path

        cfg = self._config(num_target_frames=2)
        result = run_som_stage(
            cfg,
            _load_video_frame=self.fake_load_frame,
            _call_sam_service=empty_sam,
            _send_mllm_request=self.fake_mllm,
        )

        self.assertEqual(result["targets_processed"], 0)
        self.assertEqual(result["targets_skipped"], 2)
        # Augmented JSONL exists but is empty
        augmented = os.path.join(cfg.output_dir, "augmented_frame_outputs.jsonl")
        with open(augmented) as f:
            self.assertEqual(f.read().strip(), "")

    def test_mllm_returns_out_of_range_logs_and_continues(self):
        def oob_mllm(messages, **_kw):
            return '<answer>{"accepted_marks": [99]}</answer>'

        cfg = self._config(num_target_frames=2)
        result = run_som_stage(
            cfg,
            _load_video_frame=self.fake_load_frame,
            _call_sam_service=self.fake_sam,
            _send_mllm_request=oob_mllm,
        )

        self.assertEqual(result["targets_processed"], 2)
        # No new masks were appended because all proposed ids were OOB
        augmented = os.path.join(cfg.output_dir, "augmented_frame_outputs.jsonl")
        with open(augmented) as f:
            rows = [_json.loads(line) for line in f]
        for row in rows:
            self.assertEqual(row["added_obj_ids"], [])

    def test_budget_cap_stops_early(self):
        cfg = self._config(num_target_frames=10, max_mllm_calls=2)
        result = run_som_stage(
            cfg,
            _load_video_frame=self.fake_load_frame,
            _call_sam_service=self.fake_sam,
            _send_mllm_request=self.fake_mllm,
        )
        # Stopped after 2 MLLM calls (1 per target)
        self.assertLessEqual(result["mllm_calls"], 2)

    def test_resume_skips_already_processed_targets(self):
        cfg = self._config(num_target_frames=3)
        os.makedirs(cfg.output_dir, exist_ok=True)
        augmented = os.path.join(cfg.output_dir, "augmented_frame_outputs.jsonl")
        with open(augmented, "w") as f:
            # Pretend target 0 was already processed
            f.write(_json.dumps({
                "frame_index": 0,
                "source": "som",
                "added_obj_ids": [3],
                "out_obj_ids": [1, 2, 3],
                "out_binary_masks_rle": [],
                "out_boxes_xywh": [],
                "out_probs": [],
                "out_tracker_probs": [],
            }) + "\n")

        result = run_som_stage(
            cfg,
            _load_video_frame=self.fake_load_frame,
            _call_sam_service=self.fake_sam,
            _send_mllm_request=self.fake_mllm,
        )
        # Target 0 should have been resumed (not reprocessed)
        self.assertEqual(result["targets_skipped_resume"], 1)
        self.assertEqual(result["targets_processed"], 2)
```

- [ ] **Step 2: Run tests, verify failures.**

```bash
cd /home/sbialek/ONC/sam3 && .venv/bin/python -m unittest tests.test_som_missed_creatures.RunSomStageTests -v 2>&1 | tail -10
```

Expected: ImportError for `run_som_stage` and `SomStageConfig`.

- [ ] **Step 3: Implement the orchestrator.**

Append to `nibi_model_compare/som_missed_creatures.py`:

```python
from dataclasses import dataclass


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
    broad_prompts: list[str]
    num_neighbours: int
    neighbour_offset_frames: int
    iou_dedup: float
    min_area_px: float
    max_area_frac: float
    edge_tol_px: int
    internal_iou_dedup: float
    max_mllm_calls: int


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
    import json
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


def run_som_stage(
    cfg: SomStageConfig,
    *,
    _load_video_frame=None,
    _call_sam_service=None,
    _send_mllm_request=None,
) -> dict:
    """Run the SoM missed-creature stage end-to-end.

    Returns a summary dict with counts. All per-target artefacts and the
    augmented JSONL are written under ``cfg.output_dir``.
    """
    import json
    import os

    import cv2

    load_frame = _load_video_frame or _load_video_frame_default
    call_sam = _call_sam_service
    send_mllm = _send_mllm_request or _send_mllm_request_default

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

    # Resume support: read the augmented JSONL if it already exists
    augmented_path = os.path.join(cfg.output_dir, "augmented_frame_outputs.jsonl")
    already_done = {int(row["frame_index"])
                    for row in _read_jsonl(augmented_path)}

    system_prompt = load_system_prompt(cfg.prompt_profile)

    stats = {
        "targets_total": len(targets),
        "targets_processed": 0,
        "targets_skipped": 0,
        "targets_skipped_resume": 0,
        "mllm_calls": 0,
        "masks_accepted": 0,
    }

    # Open augmented JSONL in append mode (one row flushed per target)
    with open(augmented_path, "a", encoding="utf-8") as out_handle:
        for target_idx in targets:
            if target_idx in already_done:
                stats["targets_skipped_resume"] += 1
                continue
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
            cv2.imwrite(target_img_path, target_frame)

            # Dense candidates
            candidates = generate_dense_candidates(
                image_path=target_img_path,
                broad_prompts=list(cfg.broad_prompts),
                output_folder=os.path.join(target_dir, "sam_out"),
                internal_iou_dedup=cfg.internal_iou_dedup,
                _call_sam_service=call_sam,
            )

            if not candidates:
                stats["targets_skipped"] += 1
                print(f"[som] frame {target_idx} no_candidates; skipping.")
                continue

            # Existing text-agent masks for this frame
            existing_row = _existing_row_for_frame(frame_outputs, target_idx)
            from nibi_model_compare.frame_output_utils import decode_rle_to_mask
            existing_masks = [
                {"mask": decode_rle_to_mask(rle).astype(bool)}
                for rle in existing_row.get("out_binary_masks_rle", [])
            ]

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
            import numpy as np
            survivors.sort(key=lambda c: -int(np.asarray(c["mask"], dtype=bool).sum()))

            # Render marks
            marked = draw_numbered_marks(target_frame, survivors)
            marked_path = os.path.join(target_dir, "04_marked.png")
            cv2.imwrite(marked_path, marked)

            # Neighbour frames (unmarked)
            neighbour_paths: list[str] = []
            nb_dir = os.path.join(target_dir, "neighbours")
            os.makedirs(nb_dir, exist_ok=True)
            for offset_idx in range(1, cfg.num_neighbours + 1):
                for sign, label in ((-1, "neg"), (+1, "pos")):
                    nb_idx = target_idx + sign * cfg.neighbour_offset_frames * offset_idx
                    if nb_idx < 0:
                        continue
                    nb_frame = load_frame(cfg.video_path, nb_idx)
                    if nb_frame is None:
                        continue
                    nb_path = os.path.join(nb_dir, f"{label}{offset_idx}.png")
                    cv2.imwrite(nb_path, nb_frame)
                    neighbour_paths.append(nb_path)

            # MLLM call
            messages = build_som_prompt_messages(
                system_prompt=system_prompt,
                target_image_path=marked_path,
                neighbour_image_paths=neighbour_paths,
                initial_text_prompt=cfg.initial_text_prompt,
                num_marks=len(survivors),
            )
            with open(os.path.join(target_dir, "mllm_request.json"), "w") as f:
                # Don't serialise raw image bytes; just record the structure
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
                json.dump(redacted, f, indent=2)

            response_text = send_mllm(messages)
            stats["mllm_calls"] += 1
            with open(os.path.join(target_dir, "mllm_response.txt"), "w") as f:
                f.write(response_text or "")

            accepted_ids = parse_som_response(
                response_text,
                valid_ids=set(range(1, len(survivors) + 1)),
            )
            accepted_candidates = [
                survivors[i - 1] for i in accepted_ids
            ]
            with open(os.path.join(target_dir, "accepted.json"), "w") as f:
                json.dump(accepted_ids, f)

            new_row = merge_accepted_masks_into_row(
                existing_row, accepted_candidates,
            )
            out_handle.write(json.dumps(new_row) + "\n")
            out_handle.flush()

            stats["targets_processed"] += 1
            stats["masks_accepted"] += len(accepted_candidates)

    # Write summary
    summary_path = os.path.join(cfg.output_dir, "summary.json")
    with open(summary_path, "w") as f:
        json.dump(stats, f, indent=2)

    return stats
```

- [ ] **Step 4: Run tests, verify pass.**

```bash
cd /home/sbialek/ONC/sam3 && .venv/bin/python -m unittest tests.test_som_missed_creatures -v 2>&1 | tail -10
```

Expected: all tests in this file pass.

- [ ] **Step 5: Commit.**

```bash
cd /home/sbialek/ONC/sam3
git add nibi_model_compare/som_missed_creatures.py tests/test_som_missed_creatures.py
git commit -m "$(cat <<'EOF'
Implement SoM stage orchestrator with resume + budget cap

run_som_stage() wires frame selection -> dense candidate generation ->
filtering -> SoM annotation -> MLLM judgment -> writeback per target
frame, with append-only JSONL output that supports resume after a kill,
a max_mllm_calls budget cap, and full debug-artefact retention per
target (raw frame, marked frame, candidates.json with drop reasons,
mllm_request.json, mllm_response.txt, accepted.json).

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 10: CLI entrypoint

**Files:**
- Create: `scripts/run_som_missed_creatures.py`

- [ ] **Step 1: Write the script.**

Create `scripts/run_som_missed_creatures.py`:

```python
#!/usr/bin/env python3
"""Run the Set-of-Mark missed-creature discovery stage on a video.

Spec: docs/superpowers/specs/2026-05-26-som-missed-creature-loop-design.md
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from nibi_model_compare.som_missed_creatures import (  # noqa: E402
    SomStageConfig,
    run_som_stage,
)


def _csv_ints(text: str) -> list[int]:
    return [int(x) for x in text.split(",") if x.strip()]


def _csv_strs(text: str) -> list[str]:
    return [x.strip() for x in text.split(",") if x.strip()]


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("video_path")
    p.add_argument("--frame-results", required=True,
                   help="Path to frame_results.jsonl from the text-agent run.")
    p.add_argument("--frame-outputs", required=True,
                   help="Path to frame_outputs_rle.json from the text-agent run.")
    p.add_argument("--output-dir", required=True)
    p.add_argument("--prompt-profile", choices=["underwater", "general"],
                   default="underwater")
    p.add_argument("--prompt", default="small creatures",
                   help="Original creature query echoed into the MLLM prompt.")
    p.add_argument("--num-target-frames", type=int, default=8)
    p.add_argument("--frame-selection-strategy",
                   choices=["uniform", "motion"], default="uniform")
    p.add_argument("--target-frames", type=_csv_ints, default=None,
                   help="Explicit comma-separated frame indices (overrides "
                        "strategy + num-target-frames).")
    p.add_argument("--broad-prompts", type=_csv_strs,
                   default=["creature", "animal", "organism"])
    p.add_argument("--num-neighbours", type=int, default=2,
                   help="Reference frames per side. Total neighbours = 2 * this.")
    p.add_argument("--neighbour-offset-frames", type=int, default=30,
                   help="Frame gap between target and each neighbour ring.")
    p.add_argument("--iou-dedup", type=float, default=0.3)
    p.add_argument("--min-area-px", type=float, default=350.0)
    p.add_argument("--max-area-frac", type=float, default=0.5)
    p.add_argument("--edge-tol-px", type=int, default=2)
    p.add_argument("--internal-iou-dedup", type=float, default=0.5)
    p.add_argument("--max-mllm-calls", type=int, default=200)
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)

    # Stamp the output dir so reruns are isolated
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    output_dir = os.path.join(args.output_dir, timestamp)
    os.makedirs(output_dir, exist_ok=True)

    cfg = SomStageConfig(
        video_path=args.video_path,
        frame_results_path=args.frame_results,
        frame_outputs_path=args.frame_outputs,
        output_dir=output_dir,
        prompt_profile=args.prompt_profile,
        initial_text_prompt=args.prompt,
        num_target_frames=args.num_target_frames,
        frame_selection_strategy=args.frame_selection_strategy,
        target_frames_explicit=args.target_frames,
        broad_prompts=args.broad_prompts,
        num_neighbours=args.num_neighbours,
        neighbour_offset_frames=args.neighbour_offset_frames,
        iou_dedup=args.iou_dedup,
        min_area_px=args.min_area_px,
        max_area_frac=args.max_area_frac,
        edge_tol_px=args.edge_tol_px,
        internal_iou_dedup=args.internal_iou_dedup,
        max_mllm_calls=args.max_mllm_calls,
    )

    print(f"[som] writing artefacts to {output_dir}")
    stats = run_som_stage(cfg)
    print(f"[som] done: {stats}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
```

- [ ] **Step 2: Verify the script parses --help and imports cleanly.**

```bash
cd /home/sbialek/ONC/sam3 && .venv/bin/python scripts/run_som_missed_creatures.py --help 2>&1 | head -30
```

Expected: argparse help text listing all flags, exits 0.

- [ ] **Step 3: Commit.**

```bash
cd /home/sbialek/ONC/sam3
git add scripts/run_som_missed_creatures.py
git commit -m "$(cat <<'EOF'
Add CLI entrypoint for the SoM missed-creature stage

Thin argparse wrapper around run_som_stage(): all CLI defaults are the
ones the spec calls out (uniform selection, 8 target frames, ±2
neighbour rings at 30-frame offsets, iou_dedup=0.3, etc). Stamps the
output dir with a timestamp so reruns are isolated.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 11: Merge utility

**Files:**
- Create: `scripts/merge_som_outputs.py`
- Modify: `tests/test_som_missed_creatures.py` (add `MergeSomOutputsTests`)

Takes the augmented JSONL produced by the SoM stage and merges its rows into a fresh copy of `frame_outputs_rle.json`. The original is never mutated.

- [ ] **Step 1: Write the failing tests.**

Append to `tests/test_som_missed_creatures.py`:

```python
import subprocess


class MergeSomOutputsTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.workdir = self.tmp.name
        # Original frame_outputs with 2 frames, 1 mask each
        from nibi_model_compare.frame_output_utils import encode_binary_mask_to_rle
        yy, xx = np.ogrid[:32, :32]
        m1 = ((yy - 8) ** 2 + (xx - 8) ** 2) <= 3 * 3
        m2 = ((yy - 24) ** 2 + (xx - 24) ** 2) <= 3 * 3
        self.original_path = os.path.join(self.workdir, "fo.json")
        with open(self.original_path, "w") as f:
            _json.dump({
                "format_version": 2,
                "frames": [
                    {"frame_index": 0, "out_obj_ids": [1],
                     "out_binary_masks_rle": [encode_binary_mask_to_rle(m1)],
                     "out_boxes_xywh": [[5, 5, 7, 7]],
                     "out_probs": [0.9], "out_tracker_probs": [0.9]},
                    {"frame_index": 5, "out_obj_ids": [1],
                     "out_binary_masks_rle": [encode_binary_mask_to_rle(m2)],
                     "out_boxes_xywh": [[21, 21, 7, 7]],
                     "out_probs": [0.9], "out_tracker_probs": [0.9]},
                ],
            }, f)

        # Augmented JSONL: replaces frame 5 with +1 mask
        m3 = ((yy - 16) ** 2 + (xx - 16) ** 2) <= 3 * 3
        self.augmented_path = os.path.join(self.workdir, "aug.jsonl")
        with open(self.augmented_path, "w") as f:
            f.write(_json.dumps({
                "frame_index": 5,
                "source": "som",
                "added_obj_ids": [2],
                "out_obj_ids": [1, 2],
                "out_binary_masks_rle": [
                    encode_binary_mask_to_rle(m2),
                    encode_binary_mask_to_rle(m3),
                ],
                "out_boxes_xywh": [[21, 21, 7, 7], [13, 13, 7, 7]],
                "out_probs": [0.9, 0.85],
                "out_tracker_probs": [0.9, 0.85],
                "source_per_obj_id": {"1": "text_agent", "2": "som"},
            }) + "\n")
        self.out_path = os.path.join(self.workdir, "merged.json")

    def tearDown(self):
        self.tmp.cleanup()

    def test_merge_replaces_augmented_frames_only(self):
        cmd = [
            ".venv/bin/python", "scripts/merge_som_outputs.py",
            "--frame-outputs", self.original_path,
            "--augmented", self.augmented_path,
            "--output", self.out_path,
        ]
        result = subprocess.run(cmd, capture_output=True, text=True,
                                cwd=os.path.abspath(
                                    os.path.join(os.path.dirname(__file__),
                                                 "..")))
        self.assertEqual(result.returncode, 0,
                         f"merge failed: {result.stderr}")

        with open(self.out_path) as f:
            merged = _json.load(f)
        frames_by_idx = {f["frame_index"]: f for f in merged["frames"]}
        # Frame 0 unchanged
        self.assertEqual(frames_by_idx[0]["out_obj_ids"], [1])
        # Frame 5 augmented
        self.assertEqual(frames_by_idx[5]["out_obj_ids"], [1, 2])
        self.assertEqual(frames_by_idx[5]["source"], "som")

    def test_original_file_is_not_mutated(self):
        before = open(self.original_path).read()
        cmd = [
            ".venv/bin/python", "scripts/merge_som_outputs.py",
            "--frame-outputs", self.original_path,
            "--augmented", self.augmented_path,
            "--output", self.out_path,
        ]
        subprocess.run(cmd, capture_output=True, text=True,
                       cwd=os.path.abspath(
                           os.path.join(os.path.dirname(__file__), "..")))
        after = open(self.original_path).read()
        self.assertEqual(before, after)
```

- [ ] **Step 2: Run tests, verify they fail.**

```bash
cd /home/sbialek/ONC/sam3 && .venv/bin/python -m unittest tests.test_som_missed_creatures.MergeSomOutputsTests -v 2>&1 | tail -5
```

Expected: failure because `scripts/merge_som_outputs.py` doesn't exist.

- [ ] **Step 3: Write the merge script.**

Create `scripts/merge_som_outputs.py`:

```python
#!/usr/bin/env python3
"""Merge a SoM-augmented JSONL into a fresh copy of frame_outputs_rle.json.

The input frame_outputs file is never mutated; the merged result is
written to --output.
"""

from __future__ import annotations

import argparse
import copy
import json
import sys
from pathlib import Path


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--frame-outputs", required=True,
                   help="Original frame_outputs_rle.json (never mutated).")
    p.add_argument("--augmented", required=True,
                   help="Augmented JSONL emitted by run_som_stage.")
    p.add_argument("--output", required=True,
                   help="Where to write the merged JSON.")
    args = p.parse_args(argv)

    with open(args.frame_outputs, "r", encoding="utf-8") as f:
        original = json.load(f)
    augmented_rows: dict[int, dict] = {}
    with open(args.augmented, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            augmented_rows[int(row["frame_index"])] = row

    merged = copy.deepcopy(original)
    new_frames: list[dict] = []
    seen_indices: set[int] = set()
    for row in merged.get("frames", []):
        idx = int(row.get("frame_index", -1))
        seen_indices.add(idx)
        if idx in augmented_rows:
            new_frames.append(augmented_rows[idx])
        else:
            new_frames.append(row)
    # Augmented JSONL may contain frames that weren't in the original
    for idx, row in augmented_rows.items():
        if idx not in seen_indices:
            new_frames.append(row)
    new_frames.sort(key=lambda r: int(r["frame_index"]))
    merged["frames"] = new_frames

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(merged, f, indent=2)
        f.write("\n")
    print(f"[merge] wrote {args.output} ({len(new_frames)} frames; "
          f"{len(augmented_rows)} augmented)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
```

- [ ] **Step 4: Run tests, verify pass.**

```bash
cd /home/sbialek/ONC/sam3 && .venv/bin/python -m unittest tests.test_som_missed_creatures.MergeSomOutputsTests -v 2>&1 | tail -5
```

- [ ] **Step 5: Commit.**

```bash
cd /home/sbialek/ONC/sam3
git add scripts/merge_som_outputs.py tests/test_som_missed_creatures.py
git commit -m "$(cat <<'EOF'
Add post-run utility to merge SoM-augmented rows into frame_outputs_rle

scripts/merge_som_outputs.py takes the augmented JSONL produced by
run_som_stage() and a copy of the original frame_outputs_rle.json,
emits a merged JSON to --output, and never touches the input file.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 12: Smoke test on ONC video with visual inspection

**Files:** none new. This is verification work, not code.

The goal is to validate the full pipeline end-to-end on a real video, with Claude visually inspecting each intermediate artefact and reporting findings.

- [ ] **Step 1: Pick the test video and confirm the prior text-agent run exists.**

```bash
cd /home/sbialek/ONC/sam3 && ls runs/agent_every_frame/chinacreekclipped/ 2>/dev/null | head -3
```

Expected: at least one run directory (the hardened-run from earlier sessions). If absent, run the every-frame agent first via `scripts/run_sam3_agent_every_frame_video.py` against `assets/videos/onc/chinacreekclipped.mp4`.

- [ ] **Step 2: Run the SoM stage with conservative defaults on chinacreekclipped.**

```bash
cd /home/sbialek/ONC/sam3
RUN_DIR=$(ls -dt runs/agent_every_frame/chinacreekclipped/*/ | head -1)
.venv/bin/python scripts/run_som_missed_creatures.py \
  assets/videos/onc/chinacreekclipped.mp4 \
  --frame-results "$RUN_DIR/frame_results.jsonl" \
  --frame-outputs "$RUN_DIR/frame_outputs_rle.json" \
  --output-dir runs/som/chinacreekclipped \
  --prompt-profile underwater \
  --prompt "small creatures" \
  --num-target-frames 2 \
  --frame-selection-strategy uniform \
  --max-mllm-calls 4
```

Expected: stage prints `[som] writing artefacts to runs/som/.../...` followed by progress lines, then `[som] done: {...}` with non-zero `targets_processed`.

- [ ] **Step 3: Inspect the marked target frames.**

For each target directory under `runs/som/chinacreekclipped/<timestamp>/target_*/`:

```bash
cd /home/sbialek/ONC/sam3
ls runs/som/chinacreekclipped/*/target_*/04_marked.png
```

Use the Read tool to open each `04_marked.png` and judge:

- Are the marks reasonable? (Most should be on visually distinct objects, not on flat substrate.)
- Are the numeric labels readable? (No overlap, contrast OK.)
- How many marks per frame? (Roughly 5-25 is healthy; 1-3 suggests AMG is too sparse, 40+ suggests filters are too loose.)

Write a one-paragraph qualitative report on the per-frame mark quality. If marks look poor, the spec's escalation path is evolution **B** (true grid-sampled AMG).

- [ ] **Step 4: Inspect the MLLM accept/reject decisions.**

For each target directory, read `mllm_response.txt` and `accepted.json`. Judge:

- Did the MLLM accept marks that look biological in the marked image?
- Did it reject anything that's clearly substrate, debris, or glare?
- Does the reasoning trace cite specific visual evidence (shape, colour, persistence across reference frames)?

If the model accepts substrate-only marks, the spec's escalation path is evolution **C** (per-mark refinement) or a stronger system prompt.

- [ ] **Step 5: Merge and verify the augmented frame outputs.**

```bash
cd /home/sbialek/ONC/sam3
TIMESTAMP_DIR=$(ls -dt runs/som/chinacreekclipped/*/ | head -1)
.venv/bin/python scripts/merge_som_outputs.py \
  --frame-outputs "$RUN_DIR/frame_outputs_rle.json" \
  --augmented "$TIMESTAMP_DIR/augmented_frame_outputs.jsonl" \
  --output "$TIMESTAMP_DIR/merged_frame_outputs_rle.json"
```

Verify:
- Output prints `[merge] wrote ... (N frames; M augmented)` with M >= 0.
- The original `$RUN_DIR/frame_outputs_rle.json` is byte-identical to its pre-run state (`md5sum` it before and after to confirm).

- [ ] **Step 6: Repeat on the other 4 ONC videos.**

```bash
for V in rank01_taxa001_ann001_AXISCAMACCC8E891285_20210707T220008.000Z_t00000 \
         rank02_taxa001_ann001_AXISCAMACCC8E891285_20201121T000016.000Z_t00070 \
         rank03_taxa001_ann001_AXISCAMACCC8E891285_20201113T121535.000Z_t00090 \
         rank04_taxa001_ann001_AXISCAMACCC8E891285_20210713T100008.000Z_t00100; do
    echo "=== $V ==="
    # Caller must first ensure a text-agent run exists for this video.
    # Then re-run the steps above with the video-specific paths.
done
```

For each video, repeat Steps 3-4 (inspect marks + MLLM decisions). Aggregate the findings into a single report.

- [ ] **Step 7: Write up the smoke-test report.**

Create `docs/superpowers/specs/2026-05-26-som-missed-creature-loop-smoke-test.md` summarising:

- Per-video stats from each `summary.json` (targets processed, masks accepted).
- Qualitative observations from Claude's image inspections (with examples of good and bad accepts).
- Recommendation: ship as-is, escalate to evolution path B (denser candidates), or escalate to path C (per-mark refinement).

- [ ] **Step 8: Commit the smoke-test report.**

```bash
cd /home/sbialek/ONC/sam3
git add docs/superpowers/specs/2026-05-26-som-missed-creature-loop-smoke-test.md
git commit -m "$(cat <<'EOF'
Document smoke-test results for the SoM missed-creature stage

Five ONC videos, two target frames each. Captures per-video stats,
qualitative observations from per-target visual inspection, and a
recommendation on whether to ship the v1 contract as-is, escalate to
adaptive AMG density (evolution path B), or escalate to per-mark
point refinement (evolution path C).

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Self-review

**Spec coverage:** Each section of the spec maps to a task:
- "File layout" → Tasks 1, 8, 9, 10, 11 produce all files.
- "Frame selection" → Task 2.
- "Candidate generation" (broad-prompt strategy + filtering) → Tasks 3, 7.
- "SoM annotation" → Task 4.
- "MLLM contract" (prompts + parser + retry handled via empty fallback) → Tasks 1, 5, 8, 9.
- "Output schema" → Task 6 (single-row) + Task 11 (whole-file merge).
- "Error handling" (skip-with-reason for every failure) → Task 9 driver.
- "Developer-in-the-loop verification" → visual inspection step in Task 4 + Tasks 12.3/12.4.
- "Testing strategy" → unit + integration in Tasks 1-9; smoke in Task 12.
- "CLI surface" → Task 10.
- Evolution paths B, C → out of scope per spec; mentioned in Task 7 (contract preserved) and Task 12 (escalation criteria).

**Placeholder scan:** searched for "TBD", "TODO", "implement later", "similar to" — none present. Every code step shows the actual code; every test step shows the test bodies.

**Type consistency:** the candidate dict shape `{"mask", "bbox_xywh", "score", "source_prompt"}` is used identically across Tasks 3, 4, 6, 7, 9. `SomStageConfig` field names match the CLI in Task 10. `parse_som_response` returns `list[int]` everywhere it's called.

**Scope check:** single sub-project, all tasks build toward the same orchestrator and CLI. Evolution paths are documented but not scheduled.

---

## Execution handoff

Plan complete and saved to `docs/superpowers/plans/2026-05-26-som-missed-creature-loop.md`.

Two execution options:

1. **Subagent-Driven** (recommended) — fresh subagent per task, review between tasks, fast iteration on any task that surprises us.
2. **Inline Execution** — execute tasks in this session via the executing-plans skill, with checkpoints for review.

Which approach?
