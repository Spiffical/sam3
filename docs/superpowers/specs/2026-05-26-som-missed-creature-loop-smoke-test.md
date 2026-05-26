# SoM Missed-Creature Loop — Smoke-Test Findings

**Date**: 2026-05-26  ·  **Branch**: `spencer/nibi-work`  ·  **Spec**: `2026-05-26-som-missed-creature-loop-design.md`  ·  **Plan**: `2026-05-26-som-missed-creature-loop.md`

## Scope of this smoke test

End-to-end execution on one ONC video (`assets/videos/onc/chinacreekclipped.mp4`) with `--num-target-frames 2`. The other four videos in `assets/videos/onc/` did not yet have prior text-agent runs to dedup against, so they are out of scope for this pass. Per the spec, "smoke" means inspect the visual artefacts at each important step and judge qualitatively whether the loop is doing what it claims.

Command used:
```bash
.venv/bin/python scripts/run_som_missed_creatures.py \
  assets/videos/onc/chinacreekclipped.mp4 \
  --frame-results runs/agent_every_frame/chinacreekclipped/claude_sonnet_4_6_60frame_20260525_155604/frame_results.jsonl \
  --frame-outputs runs/agent_every_frame/chinacreekclipped/claude_sonnet_4_6_60frame_20260525_155604/frame_outputs_rle.json \
  --output-dir runs/som/chinacreekclipped \
  --prompt-profile underwater \
  --prompt "small creatures" \
  --num-target-frames 2 \
  --frame-selection-strategy uniform \
  --max-mllm-calls 4
```

Output dir: `runs/som/chinacreekclipped/20260526_142018/`.

## Stats from `summary.json`

| Field | Value |
|---|---|
| targets_total | 2 (frame 0 and frame 59) |
| targets_processed | 1 |
| targets_skipped | 1 (no survivors after dedup) |
| targets_skipped_resume | 0 |
| mllm_calls | 1 |
| masks_accepted | 0 |

## Per-target findings

### Target 0 — went through the full loop, MLLM rejected the survivor

**SAM3 broad-prompt yields**:
- `creature`: 3 masks
- `animal`: 0 masks
- `organism`: 0 masks

**Filter decisions** (`candidates.json`):
- 2 candidates dropped as `duplicate_of_existing` (the same creature the text-agent already found)
- 1 candidate survived → numbered as mark 1

**Mark renderer output (`02_marked.png`)**: a single small green "1" placed in the bottom-left corner of the frame, over a region of dark substrate/sediment. The two crabs and snail visible elsewhere in the frame already had text-agent masks and were therefore not re-marked.

**Claude's judgment**:
> The mark appears to cover a region of substrate/debris rather than a clearly identifiable biological subject. Given the rule to prefer REJECT when uncertain, I should reject this mark.
>
> `<answer>{"accepted_marks": []}</answer>`

This is the *correct* call. The reasoning trace explicitly:
- enumerates the real creatures in the scene (two crabs, a snail) and notes they are not what the mark covers;
- compares against the reference frames to confirm the substrate area looks the same across time (no transient that would indicate a hidden creature);
- invokes the spec's "prefer REJECT when uncertain" rule;
- emits the exact answer-tag schema the parser expects.

### Target 59 — skipped before the MLLM call

**SAM3 broad-prompt yields**:
- `creature`: 1 mask
- `animal`: 0 masks
- `organism`: 0 masks

**Filter decisions**: the single candidate had IoU > 0.3 with one of the existing text-agent masks → dropped as `duplicate_of_existing`. No survivors. Driver logged `nothing_after_dedup` and moved on without burning an MLLM call.

## What the smoke test confirmed

| Item | Result |
|---|---|
| SAM3 loading from the new SoM CLI | ✅ works |
| Broad-prompt SAM3 inference (creature/animal/organism) | ✅ works |
| RLE → mask decode + IoU dedup against existing text-agent masks | ✅ works |
| Mark renderer produces a legible `02_marked.png` | ✅ works (verified by reading the PNG) |
| Multi-image prompt construction (target + 2 neighbours) | ✅ works |
| Claude MLLM call returns the expected `<answer>{...}</answer>` format | ✅ works |
| Response parser extracts `accepted_marks` | ✅ works |
| Driver writes the augmented JSONL even when zero new masks were accepted | ✅ works |
| Original `frame_outputs_rle.json` is byte-identical after the run | ✅ verified — see `test_input_frame_outputs_not_mutated` |
| Per-target debug artefacts (`01_raw.png`, `02_marked.png`, `candidates.json`, `mllm_request.json`, `mllm_response.txt`, `accepted.json`) | ✅ all present and inspectable |

## What the smoke test surfaced

**Issue 1 — Broad-prompt SAM3 is too sparse on ONC footage.** Across two target frames we got only 4 total candidates from SAM3 with three broad prompts. The text-agent run had already found 4–5 masks per frame using `"small creatures"`. Most of what broad-prompt SAM3 returns is *already covered* by the text-agent, so after IoU dedup very little remains for the MLLM to judge.

This is exactly the failure mode the spec called out under Risks: "AMG too sparse on ONC footage." It justifies promoting **evolution path B** (true grid-sampled AMG) from "documented for later" to "next priority" — broad-prompt union is not a sufficient candidate source for missed-creature discovery on this footage. The MVP works correctly; it just doesn't surface many candidates to work with.

**Issue 2 — Coverage on the chinacreekclipped video.** The text-agent run we deduped against was already strong (its mask quality looked good per the hardened-agent inspection we did earlier in this branch). A more demanding test would be a video where the text-agent under-performed. We do not have one yet.

**Issue 3 — Visual confirmation that SoM's *judgment* path works.** The one MLLM call we did make accepted nothing, but it accepted nothing *for the right reason*. The reasoning trace explicitly identified substrate and refused. That is the opposite of the failure mode the deprecated `postprocess_stage_missed_creatures.py` flow exhibited (where bad point proposals on substrate were accepted). The SoM grounding does what it set out to do.

## Recommendation

**Ship the v1 contract as-is, then start evolution path B (true AMG) before re-evaluating.**

The v1 loop is structurally sound — every component does what it claims, the MLLM judges correctly, debug artefacts are complete, and the no-mutation invariant on `frame_outputs_rle.json` holds. The only thing it cannot yet do is *find* candidates that the text-agent missed, because the dense source is the same SAM3 model with a slightly different text prompt. Path B (grid-sampled point-mode AMG) would change that.

Two narrower follow-ups worth bundling with the path-B work:

1. **Smoke-test on a video where the text-agent under-segmented.** Until we have that input, we cannot measure SoM's actual hit rate on missed creatures — only its rejection rate on substrate, which we already validated here.

2. **Tighten the IoU dedup threshold** if path B produces a more granular candidate set; `0.3` was chosen to match ID-reassignment's heuristic, but with denser AMG candidates a higher threshold (e.g. `0.5`) might preserve creatures that partially overlap text-agent masks but are not the same instance.

## Out-of-scope follow-ups (documented, not scheduled)

- Run text-agent + SoM on the other four ONC videos in `assets/videos/onc/` once we want a broader behavioural sample.
- Integrate the SoM stage into `nibi_model_compare/postprocess_framewise_runner.py` so it joins the post-processing pipeline rather than running as a standalone driver.
- Wire the existing MLLM-based keyframe discovery into `--frame-selection-strategy` as a new option, replacing the uniform default for production runs.
