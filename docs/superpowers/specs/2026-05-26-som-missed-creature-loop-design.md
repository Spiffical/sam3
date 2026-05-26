# Set-of-Mark Missed-Creature Discovery Loop

**Status**: design  ·  **Date**: 2026-05-26  ·  **Author**: Spencer + Claude

## Background

The end goal is a labelled dataset of bounding-boxed underwater creatures in
Ocean Networks Canada (ONC) videos. The repository already ships:

- A per-frame text-prompted SAM3 + MLLM agent
  (`scripts/run_sam3_agent_every_frame_video.py`, recently hardened) that
  emits `frame_outputs_rle.json` with RLE masks and `out_boxes_xywh`
  bounding boxes.
- A temporal post-processing pipeline (`nibi_model_compare/`) with frame
  quality scoring, motion- and MLLM-based keyframe discovery, ID
  reassignment, gap-fill, and outlier filtering.
- An attempt at an MLLM-driven missed-creature discovery loop
  (`nibi_model_compare/postprocess_stage_missed_creatures.py`,
  `postprop_missed_creatures.py`, and the
  `system_prompt_missed_creature_*` prompts), deprecated in commit
  `ae72a4c` because it "proved unreliable in practice". The dominant
  failure mode was bad point proposals — the MLLM hallucinated raw
  (x, y) coordinates, often placing points on substrate, rocks, or
  debris rather than creatures.

This spec replaces that failed approach with a Set-of-Mark (SoM) loop in
which the MLLM never produces raw coordinates. It selects already-existing
mask candidates by their numbered labels, sidestepping the failure that
killed the previous version.

## Goal

Build a standalone stage that, given a video and an existing
`frame_outputs_rle.json` from the text-agent run, automatically picks a
handful of target frames per video and adds masks (with bounding boxes)
for creatures the text-agent missed on those frames. Output augments the
input JSON without modifying it.

This is the first of three sub-projects the user identified
("MLLM→point→SAM3 loop", "best-frame selection", "multi-frame hidden-creature
audit"). The other two are out of scope here but the design leaves clean
seams for them to slot in later.

## Non-goals

- Frame selection by MLLM judgment ("which frames are best for labelling")
  — that is a separate sub-project. This spec uses a simple uniform-spacing
  default plus an optional motion-based selector that already exists in
  the repo.
- Per-mark refinement via point editing (the literal "examine the points
  and decide whether they need to be edited" loop the user described).
  Documented as evolution path **C** below.
- Adaptive SAM3 automatic-mask-generation density tuning. Documented as
  evolution path **B** below.
- Integration into `nibi_model_compare/postprocess_framewise_runner.py`
  as a stage. The first version ships as a standalone script; wiring
  into the runner is mechanical and follows once the loop is validated.
- ID consistency across frames. Each accepted SoM mask gets a fresh
  per-frame `obj_id`. Cross-frame tracking remains the job of the
  existing ID-reassignment stage (`postprocess_framewise_runner.py`).

## Design

### Pipeline diagram

```
                ┌──────────────────────────────────┐
                │ existing frame_outputs_rle.json  │
                │   (from text-prompted agent)     │
                └─────────────┬────────────────────┘
                              │
                              ▼
                ┌──────────────────────────────────┐
                │  select_target_frames(strategy)  │
                │   default: K uniform-spaced      │
                │   skips errored & invalid frames │
                └─────────────┬────────────────────┘
                              │  target indices
                              ▼
   ┌─────────────────  for each target frame  ──────────────────┐
   │                                                            │
   │  load target frame, neighbours (±1s, 2-4 frames)           │
   │                                                            │
   │  amg_masks ← SAM3 automatic_mask_generation(target)        │
   │  candidates ← drop masks IoU > τ_dedup vs existing_masks   │
   │  candidates ← filter area, edge policy                     │
   │                                                            │
   │  marked_target ← som_utils.draw_numbered_marks(            │
   │                       target, candidates)                  │
   │                                                            │
   │  response ← mllm.send(                                     │
   │       prompt   = system_prompt_som_missed_creature_*,      │
   │       images   = [marked_target] + neighbours,             │
   │       query    = initial_text_prompt,                      │
   │  )                                                         │
   │  accepted_ids ← parse <answer>{"accepted_marks":[...]}     │
   │                                                            │
   │  new_masks ← candidates filtered by accepted_ids           │
   │  emit augmented frame row (existing + new, source="som")   │
   │                                                            │
   └────────────────────────────────────────────────────────────┘
                              │
                              ▼
                ┌──────────────────────────────────┐
                │ augmented_frame_outputs_rle.json │
                │  (input + SoM-added masks)       │
                └──────────────────────────────────┘
```

### File layout

```
scripts/run_som_missed_creatures.py
    Standalone driver. argparse, top-level orchestration, debug saving.

scripts/merge_som_outputs.py
    Small post-run utility that merges the SoM-augmented JSONL back into
    a fresh frame_outputs_rle.json (input file is never mutated).

nibi_model_compare/som_missed_creatures.py
    All logic, importable and testable: frame selection, AMG wrapper,
    candidate filter, mark renderer wrapper, prompt builder, response
    parser, merger.

sam3/agent/system_prompts/
    system_prompt_som_missed_creature_underwater.txt
    system_prompt_som_missed_creature_general.txt

tests/test_som_missed_creatures.py
    Unit + integration tests with monkey-patched SAM3 and MLLM.
```

Reuses without modification:

- `sam3/agent/helpers/som_utils.py` — mark drawing
- `sam3/agent/helpers/mask_overlap_removal.py` — IoU helpers
- `nibi_model_compare/frame_output_utils.py` — RLE encode/decode, bbox
  derivation, schema
- `nibi_model_compare/frame_quality.py` — invalid-frame filter
- `nibi_model_compare/keyframe_discovery.py` — optional motion selector
- `sam3/agent/client_claude.py` / `client_llm.py` — MLLM clients

### Frame selection

Default: K uniformly-spaced indices drawn from the set of valid frames
(those where the text-agent produced a result, did not error, and were
not flagged invalid by `frame_quality.py`).

Flags:
- `--num-target-frames K` (default 8) — number of frames to label
- `--frame-selection-strategy {uniform,motion}` (default `uniform`)
- `--target-frames 12,47,93` — explicit override; overrides everything
  else above

The MLLM-judged "best frame" selector is a separate sub-project; when
that ships, it will produce a list of frame indices consumed via the
explicit-override path. No coupling required at this layer.

### Candidate generation

Phase 1 produces *dense* mask candidates by running SAM3's automatic
mask generator (AMG) on the target frame. AMG defaults are used at
first; if AMG is too sparse on ONC footage we tune
`points_per_side`, `pred_iou_thresh`, `min_mask_region_area` via flags.

Phase 2 *removes* candidates that the text-agent already covered:

```
discard candidate C if max_existing IoU(C, existing_mask) > τ_dedup
default τ_dedup = 0.3
```

Phase 3 applies area filters:

- Drop masks below `--min-area-px` (default 0.0005 × H × W ≈ 350 px on
  a 1024×768 frame).
- Drop masks above `--max-area-px` (default 0.5 × H × W) — these are
  almost always substrate/background.
- Drop masks whose bbox touches the frame edge by more than `--edge-tol-px`
  on more than one side — these are partial/clipped subjects that the
  MLLM cannot judge well from this frame alone.

What survives is the candidate set passed to SoM annotation.

### SoM annotation

Numbered marks are drawn on the target frame:

- Centroid dot with mark ID label
- Translucent fill of the mask outline so the MLLM can see both the
  point and the shape it implies

Mark IDs are assigned `1..N` ordered by descending mask area, so larger
candidates get lower numbers (easier to refer to in reasoning).

Reference frames (neighbours) are **not marked**. Their role is to give
the MLLM extra perspectives of the same scene so it can judge whether
each mark on the target corresponds to something biological that
persists in the world.

### MLLM contract

Request:
- One marked target frame
- 2–4 unmarked neighbour frames at offsets `[-2s, -1s, +1s, +2s]`
  clamped to the video bounds
- The original text query (e.g. `"small creatures"`)
- System prompt instructing the MLLM that:
  - Some creatures move and some are stationary; **do not require motion
    to accept a mark**. The reference frames provide additional
    perspectives — lighting drift, slight camera shifts, occasional
    motion — that help disambiguate persistent biology from transient
    artefacts (floating debris, glare).
  - Respond with exactly one `<answer>{"accepted_marks": [<int>, ...]}</answer>`
    block at the end of the response. Free-text reasoning may appear
    before the tag but must not appear after.

Response parsing:
- Extract the single trailing `<answer>...</answer>` block via regex
- `json.loads` the payload
- Keep only mark IDs in the valid `1..N` range; log and drop the rest
- Empty list is a valid outcome — frame had no missed creatures

### Output schema

The driver writes a new JSONL file alongside the input
`frame_outputs_rle.json` rather than mutating it. Each row corresponds
to one target frame:

```json
{
  "frame_index": 47,
  "source": "som",
  "input_existing_obj_ids": [1, 2, 3],
  "added_obj_ids": [4, 5],
  "out_obj_ids": [1, 2, 3, 4, 5],
  "out_binary_masks_rle": [...],
  "out_boxes_xywh": [...],
  "out_probs": [..., 0.0, 0.0],
  "out_tracker_probs": [...],
  "source_per_obj_id": {
    "1": "text_agent", "2": "text_agent", "3": "text_agent",
    "4": "som", "5": "som"
  }
}
```

A separate `merge_som_outputs.py` utility (also delivered with this
spec) merges the new SoM rows back into a fresh
`frame_outputs_rle.json` ready for the rest of the post-processing
pipeline. Keeping the merge as its own step preserves the invariant
that **a failure in the SoM stage never touches the existing JSON**.

### Error handling

| Failure | Handling |
|---|---|
| Video frame unreadable | Skip target, log `frame_unreadable`, continue |
| AMG returns 0 masks | Skip target, log `no_candidates`, continue |
| All candidates dedup'd away | Skip target, log `nothing_after_dedup`, continue |
| MLLM call raises (network, rate limit, …) | Retry with client's existing backoff; if still failing, skip target, log `mllm_error` |
| MLLM response missing `<answer>` block | Retry once with strict-format reminder (same pattern `agent_core.py` uses); if still missing, treat as `accepted_marks=[]` |
| `<answer>` JSON has out-of-range IDs | Filter to valid IDs, log the bad ones, continue |
| `<answer>` JSON empty list | Valid outcome, not an error |
| Frame quality flagged target as invalid upstream | Skip target, honour upstream |
| Process killed mid-run | Append-only JSONL flushed per-frame, driver checks output on restart and resumes |

Budget caps as backstops: `--max-frames N`, `--max-mllm-calls M`. Hitting
either prints a summary and exits 0.

### Developer-in-the-loop verification

Per user request, *every important step in the loop produces an
inspectable image artefact*, and during implementation and smoke
testing Claude opens those images via the Read tool and reports
qualitative observations back to the user before continuing. The
artefacts saved per target frame are:

```
runs/som/<video_stem>/<timestamp>/
  target_<idx>/
    01_raw.png                # target frame, no overlays
    02_amg_all.png            # all AMG candidates, no filter
    03_candidates.png         # post-dedup, post-area-filter candidates
    04_marked.png             # numbered marks on candidates
    05_accepted.png           # only marks the MLLM accepted, plus
                              #   existing text-agent masks for context
    neighbours/
      neg2.png  neg1.png  pos1.png  pos2.png
    mllm_request.json         # full payload sent
    mllm_response.txt         # raw response
    candidates.json           # per-candidate filter decisions
    accepted.json             # final accepted list
  summary.json
  augmented_frame_outputs.jsonl
```

Inspection checkpoints during smoke testing:

1. **After candidate filtering** (`03_candidates.png`): are the surviving
   candidates plausibly creatures? Are obvious creatures missing? (If
   the latter, this points at AMG params being too coarse — promote
   evolution path **B** to active.)
2. **After SoM annotation** (`04_marked.png`): are marks legible? Do
   numbers overlap so much the MLLM can't read them?
3. **After MLLM judgment** (`05_accepted.png` + `mllm_response.txt`):
   did the MLLM accept marks that look biological? Reject any obvious
   creatures? Place its confidence on visual evidence the reasoning
   trace can quote?

The user is not required to be in the loop during development — Claude
opens the images itself, judges them, and surfaces only the
observations that affect next steps.

### Testing strategy

Unit tests (`tests/test_som_missed_creatures.py`):

- `_filter_candidates` — synthetic mask grids, verify IoU dedup keeps
  non-overlapping and drops overlapping; verify area-filter respects
  min/max bounds; verify edge-touch policy.
- `_parse_som_response` — happy path, prose-before-tags, malformed
  JSON, out-of-range IDs, empty list, multiple `<answer>` blocks
  (we accept the last one only).
- `_select_target_frames` — uniform spacing skips invalid frames;
  K respected; explicit override wins; motion strategy delegates to
  the existing keyframe-discovery helper.
- `_merge_into_frame_outputs` — accepted masks get `obj_id`s above
  `max(existing) + 1`, `source: "som"` tag present, schema matches
  the existing `frame_outputs_rle.json` contract.

Integration tests in the same file, with monkey-patched SAM3 and
MLLM:

- 3 candidates returned by mock AMG, 1 overlaps existing text-agent
  mask → 2 candidates marked → mock MLLM accepts `[2]` → driver
  appends exactly one new mask with the correct `obj_id` and source
  tag.
- Mock MLLM returns `{"accepted_marks": [99]}` (out of range) →
  driver logs, accepts nothing, no crash.
- Mock AMG returns 0 masks → driver logs `no_candidates`, skips
  cleanly, continues to next target.

Smoke / qualitative tests (no asserts, manual + Claude inspection):

- For each video in `assets/videos/onc/` (currently 5 clips,
  ~300 frames @ 30fps each), run the driver with `--num-target-frames 2`.
  Total: ~10 target frames to eyeball.
- For each target, Claude reads `03_candidates.png`, `04_marked.png`,
  and `05_accepted.png` and writes a short qualitative note:
  - Were obvious creatures present in the AMG candidates?
  - Were marks readable?
  - Did the MLLM accept biologically plausible marks?
  - Did it accept anything that's clearly substrate, debris, or glare?
- Findings inform whether to ship A as-is, escalate to B (denser AMG),
  or escalate to C (per-mark refinement).

### CLI surface (first version)

```
python scripts/run_som_missed_creatures.py \
    <video_path> \
    --frame-results runs/agent_every_frame/.../frame_results.jsonl \
    --frame-outputs runs/agent_every_frame/.../frame_outputs_rle.json \
    --num-target-frames 8 \
    --frame-selection-strategy uniform \
    --neighbour-offset-frames 30 \
    --num-neighbours 4 \
    --amg-points-per-side 32 \
    --min-area-px 350 \
    --max-area-frac 0.5 \
    --edge-tol-px 2 \
    --iou-dedup 0.3 \
    --llm-provider claude \
    --claude-model claude-sonnet-4-6 \
    --output-dir runs/som/<video_stem>/<timestamp> \
    [--target-frames 12,47,93]           # optional override
    [--max-mllm-calls 100]               # budget cap
```

Defaults are chosen to be conservative for 1024×768 ONC footage at
30 fps. Tunable from CLI; not from environment variables.

## Evolution paths (out of scope)

These are documented so the spec is self-contained and so the first
version's seams stay clean.

**B — Adaptive AMG density.** When the MLLM reports any version of
"I see creatures that aren't marked" (forced by adding a second JSON
field `unmarked_creatures_visible: bool`), the driver re-runs AMG with
denser params (smaller `points_per_side`, lower
`min_mask_region_area`) and re-asks. Capped at K iterations per
frame.

**C — Per-mark point refinement.** Each accepted candidate enters a
refine sub-loop where the MLLM examines a zoomed crop of just that
mask and either accepts it as-is or proposes positive/negative point
edits in *crop-relative* coordinates (much easier for an MLLM to
ground than full-image coords). SAM3 point-mode then redraws the mask
with the edited point set; the MLLM judges again; up to N iterations
per mark. This is the literal "examine the points and decide whether
they need to be edited" loop the user described in the original
brainstorm.

Both paths can be added later without touching the contract this spec
defines: they add a layer between Phase 3 (MLLM judgment) and
Phase 4 (writeback).

## Risks

- **AMG too sparse on ONC footage.** Default `points_per_side` may
  miss tiny creatures (worms, juveniles). Mitigation: instrument
  `03_candidates.png` inspection; if obvious misses, escalate to **B**.
- **MLLM still picks substrate as creature.** SoM doesn't make the
  MLLM smarter — it only removes the coordinate-output failure mode.
  If the MLLM is poor at biology, accept rates will be poor too.
  Mitigation: smoke-test results inform whether the underwater
  system prompt needs strengthening.
- **Mark overlap unreadable for dense scenes.** Many small creatures
  in one frame produce overlapping numbered labels. Mitigation:
  `som_utils.draw_numbered_marks` already supports leader-line and
  staggered labels; if needed, fall back to a side-by-side legend
  image.
- **Resume logic.** Append-only JSONL with per-frame flush should be
  robust, but tested explicitly in integration tests.

## Open questions

None requiring resolution before implementation. Tunables (`τ_dedup`,
area filters, AMG params, neighbour offsets) are exposed as CLI flags
so the smoke-testing phase informs sensible defaults.

## Success criteria

After the smoke tests on the 5 ONC videos, we can answer:

- Does AMG produce useful candidates on ONC footage at all?
- Does the SoM MLLM call accept biologically plausible candidates?
- Are accepted candidates novel (not already covered by the text-agent)?
- What's the per-target-frame runtime and MLLM cost?

If accept rates are high and qualitative inspection looks reasonable,
the loop ships as the v1 missed-creature stage. If accept rates are
low because AMG misses creatures, escalate to **B**. If accept rates
are okay but masks are partial, escalate to **C**. Either escalation
path leaves the v1 contract intact.
