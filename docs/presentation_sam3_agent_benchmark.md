# Presentation benchmark: persistent-mask SAM3 agent

_Exploratory run: 2026-08-16. Anthropic API calls were used._

## Run definition

This is the first SAM3-agent comparison on the five fixed frames in
`configs/presentation_benchmark_frames.json`. It used:

- Claude Sonnet 4.6 (`claude-sonnet-4-6`)
- the existing underwater prompt profile and initial query `small creatures`
- the existing persistent-mask behavior across agent iterations
- 20 maximum generations and 2,048 maximum completion tokens
- high-detail agent images and a SAM3 confidence threshold of 0.40
- three independent repeats per frame

No agent source was edited for this experiment. A 1,024-token plumbing run
completed, but truncated several tool calls; 2,048 tokens eliminated those
format failures and was used for every benchmark repeat.

The WSL-only run is:

`runs/presentation_benchmark/sam3_agent/20260816_sonnet46_persistent_vanilla_tok2048_repeats3/`

All 15 frame-runs completed with zero API or pipeline errors. The run is
marked exploratory because the inherited source tree is still uncommitted;
the run metadata records Git `6815d08` plus exact source-file hashes.

## Repeat-aware results

Counts and runtimes are mean plus or minus sample standard deviation over
three repeats.

| Frame | Visible targets | Masks | Runtime | Visual result |
| --- | ---: | ---: | ---: | --- |
| chinacreek | 5 | 5.00 ± 0.00 | 121.15 ± 53.46 s | All five targets, no visible false positives |
| rank01 | 6 | 6.33 ± 0.58 | 120.48 ± 73.36 s | Six correct fish every time; repeat 3 also selected a static fuzzy edge blob |
| rank02 | 1 | 1.00 ± 0.00 | 40.02 ± 1.15 s | One clean, tight fish mask in every repeat |
| rank03 | 1 | 1.67 ± 0.58 | 141.74 ± 77.90 s | Main animal found every time; repeats 1 and 3 also selected a drifting non-biological particle |
| rank04 | 1 | 0.00 ± 0.00 | 293.48 ± 8.45 s | Camouflaged fish missed in all repeats |

Across the corrected 14 visually reviewed target instances, per-repeat creature
recall was 13/14 in every repeat: **92.9% ± 0.0 percentage points**. Visual
precision was **93.2% ± 6.7 percentage points**. These are manual presentation
frame judgments, not a replacement for a larger annotated evaluation set.

China Creek is the one frame with an existing hand-reviewed mask reference.
All three repeats matched 5/5 masks at IoU at least 0.5, for **100% ± 0% mask
recall**. Per-repeat mean matched-mask IoU was **0.9949 ± 0.0044**.

The masks themselves were highly stable when a target was selected. The six
rank01 fish masks shared by every repeat were pixel-identical, and the rank03
main animal had pairwise IoU above 0.997. Most variance came from the agent
deciding whether to include non-biological edge/background structures.

## Visual QA and interpretation

The result is substantially stronger than the FathomNet baselines on the
multi-animal China Creek and rank01 scenes: SAM3 produces tight instance masks
and cleanly separates neighboring fish. Rank02 is also presentation-ready.

Two failures remain important:

- On rank01, the repeat-3 extra at approximately x=0.02, y=0.37 was initially
  described as a seventh partial fish. Denser review across the 10-second clip
  shows a fixed, fuzzy, featureless blob rather than an independently moving or
  structured animal. The later Sonnet 5 mask-guided whole-frame and enlarged
  temporal perimeter checks independently rejected it. The corrected target
  count is therefore six, and repeat 3 contains one false positive.
- On rank03, Sonnet's verification is stochastic. Review across the full
  300-frame video, with denser sampling around frames 60–150, shows the thin
  right-side candidate entering from off-screen, translating and rotating
  with the particle field while retaining a rigid, featureless silhouette.
  The central animal instead has bilateral fins or appendages whose posture
  changes across frames. The right-side mask is therefore classified as
  floating non-biological debris; only repeat 2 is clean.
- On rank04, Sonnet repeatedly describes the visible camouflaged fish, but
  SAM3 returns no mask for roughly twenty alternative phrases, including
  `fish`, `flatfish`, `benthic fish`, and `sculpin`. The failure is therefore
  the text-prompt-to-mask step, not failure of the agent to notice the animal.

Rank04 is the clearest motivation for the custom missed-creature click stage:
use the MLLM to point at a known missed target, then generate/refine its mask
without relying on SAM3 text retrieval.

## Presentation overlays

The Mac contains only the five finalized overlays, not WSL datasets, videos,
weights, caches, secrets, or general run outputs:

- `docs/presentation_sam3_agent_assets/chinacreek.png` — repeat 1
- `docs/presentation_sam3_agent_assets/rank01.png` — repeat 3, honestly showing
  the extra static edge-blob false positive
- `docs/presentation_sam3_agent_assets/rank02.png` — repeat 1
- `docs/presentation_sam3_agent_assets/rank03.png` — repeat 2, excluding the
  drifting particle
- `docs/presentation_sam3_agent_assets/rank04.png` — repeat 1, honest no-mask
  failure

The selected images are representative presentation examples; the table above
reports every repeat so the slide selection does not replace the evaluation.
