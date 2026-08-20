# Presentation benchmark: Sonnet 5 mask-guided custom flow

_Exploratory run: 2026-08-16. Anthropic API calls were used._

## Run definition

This is the custom-flow comparison on the five fixed frames in
`configs/presentation_benchmark_frames.json`. Every visual role used Claude
Sonnet 5 (`claude-sonnet-5`):

- persistent-mask SAM3 text-agent first pass
- missed-creature discovery using the existing underwater SoM prompt
- four nearby raw reference frames (±0.5 s and ±1.0 s)
- a high-resolution temporal perimeter fallback when the whole-frame missed
  pass returned no proposal
- SAM3 click mode with `refine_group_mm_hybrid`
- conservative post-mask verification

The first-pass masks are painted translucent green, so the clicker is asked to
find only unmasked animals. Whole-frame click review remains disabled. A mask
that does not match a first-pass mask must reach 0.70 post-mask creature
confidence before it is added; re-finds are never removed by this threshold.

The WSL-only scored run is:

`runs/presentation_benchmark/custom_flow/20260816_sonnet5_maskguided_repeats3/`

All 15 frame-runs completed with zero recorded API failures. One transient empty
API response retried successfully. The run is exploratory because the inherited
source tree is uncommitted; the run directory records the Git SHA and exact
source-file hashes.

## Repeat-aware results

Counts and click-stage runtimes are mean plus or minus sample standard deviation
over three independent repeats. Runtime excludes the already-completed Sonnet 5
first pass.

| Frame | Visible targets | First-pass masks | Final masks | Click-stage runtime | Visual result |
| --- | ---: | ---: | ---: | ---: | --- |
| chinacreek | 5 | 4.67 ± 0.58 | 5.00 ± 0.00 | 36.33 ± 18.48 s | All five; the clicker recovered the missing fifth animal in repeat 3 |
| rank01 | 6 | 6.00 ± 0.00 | 6.00 ± 0.00 | 37.88 ± 25.20 s | Six fish, with the static fuzzy edge blob excluded |
| rank02 | 1 | 1.00 ± 0.00 | 1.00 ± 0.00 | 29.80 ± 12.72 s | One clean fish; low-confidence redundant proposals were not added |
| rank03 | 1 | 1.00 ± 0.00 | 1.00 ± 0.00 | 8.84 ± 1.20 s | One real animal; marine snow/debris excluded in every repeat |
| rank04 | 1 | 0.00 ± 0.00 | 1.00 ± 0.00 | 16.83 ± 1.50 s | Camouflaged fish recovered by click mode in all three repeats |

Across 42 evaluated creature instances, first-pass visual recall was
**90.5% ± 4.1 percentage points**. The full custom flow reached **100% ± 0% visual
recall and 100% ± 0% visual precision** on these five presentation frames. The
five-frame click stage took **129.69 ± 57.01 seconds** per repeat.

These are manual judgments on a small, deliberately selected presentation set,
not a claim about performance on the full ONC distribution.

## Mask quality and failure analysis

The final masks are highly repeatable for four frames. Mean pairwise best-match
IoU across repeat pairs was:

| Frame | Mean pairwise IoU |
| --- | ---: |
| chinacreek | 0.9845 |
| rank01 | 1.0000 |
| rank02 | 1.0000 |
| rank03 | 0.9984 |
| rank04 | 0.7371 |

Rank04 is the important success and the main remaining weakness. Sonnet 5 found
it in all three repeats with creature confidence 0.85–0.90, but click placement
changed the left boundary/head coverage. Repeat 2 is the cleanest presentation
mask and is the selected asset.

The 0.70 new-mask threshold was added after an exploratory cold-finder run
accepted a static China Creek substrate shape at confidence 0.60. In the scored
mask-guided run it also prevented low-confidence or redundant rank01/rank02
proposals from changing otherwise-correct first-pass masks. This threshold is a
promising precision guard, but it was tuned on this small set and needs broader
validation.

## Corrected rank01 interpretation

The earlier Sonnet 4.6 report counted a seventh partial fish. Dense temporal
inspection shows that candidate is a fixed fuzzy blob at the far-left edge, and
both Sonnet 5's masked whole-frame pass and enlarged temporal perimeter pass
rejected it. The corrected visible target count is six. The earlier report and
asset annotation have been updated so the repeat-3 blob is treated as an honest
false positive, not a recovery.

## Presentation overlays

Only the five selected repeat-2 overlays were copied to the Mac:

- `docs/presentation_sonnet5_custom_assets/chinacreek.png`
- `docs/presentation_sonnet5_custom_assets/rank01.png`
- `docs/presentation_sonnet5_custom_assets/rank02.png`
- `docs/presentation_sonnet5_custom_assets/rank03.png`
- `docs/presentation_sonnet5_custom_assets/rank04.png`

No videos, datasets, weights, secrets, caches, or general run outputs were
copied from WSL.
