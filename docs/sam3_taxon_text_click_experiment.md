# WoRMS-taxonomy SAM3 proposals plus persistent Fable clicking

Date: 2026-08-17

## Recommended flow

1. Start from the fixed presentation frame and the Sonnet 4.6 first-pass masks.
2. Query SAM3 with the clip's WoRMS scientific/common names, followed by a
   compact generic bank: `small creatures`, `coral`, `branching coral`,
   `sponge`, and `anemone`.
3. Keep SAM3 proposals at score >= 0.45, remove overlaps with accepted masks,
   and batch-verify the remaining proposals with Fable 5 at confidence >= 0.70.
   A prompt is retrieval metadata, not a taxonomic assignment.
4. Persist first-pass and verified text masks as green context. Run two
   sequential full-frame Fable 5 discovery passes, focusing left then right.
   If these earlier stages leave the frame with zero masks, use three spatially
   focused passes instead.
5. Convert clicks with `refine_group_mm_hybrid`, verify recovered masks at
   confidence >= 0.80, and expose each accepted mask to the next pass.
6. Leave the high-resolution border fallback off by default. It is retained as
   an explicit diagnostic option and makes only one API attempt when enabled.

The full-frame context is important. A three-repeat A/B test that physically
cropped the left/right discovery inputs found 0.00 +/- 0.00 masks on the dense
coral frame, versus 3.67 +/- 2.08 with full-frame inputs and textual spatial
focus. The crop experiment was reverted.

## Three-repeat exploratory result

Configuration: Sonnet 4.6 first pass, compact WoRMS/generic SAM3 proposals at
0.45, Fable 5 batch proposal verification at 0.70, two persistent mask-guided
click passes, hybrid mask generation, recovery confidence 0.80. These runs used
an uncommitted source tree and are not scored experiments.

| Frame | First pass | Verified text additions | Click additions | Final masks | Runtime (s) |
| --- | ---: | ---: | ---: | ---: | ---: |
| Isolated crab | 1.00 +/- 0.00 | 0.00 +/- 0.00 | 0.00 +/- 0.00 | 1.00 +/- 0.00 | 50.96 +/- 6.32 |
| Sea stars on coral | 3.00 +/- 0.00 | 1.67 +/- 0.58 | 1.33 +/- 1.15 | 6.00 +/- 1.00 | 183.89 +/- 33.20 |
| Anemone/tubeworm field | 6.67 +/- 0.58 | 1.33 +/- 0.58 | 0.00 +/- 0.00 | 8.00 +/- 0.00 | 331.97 +/- 120.10 |
| Dense coral thicket | 0.00 +/- 0.00 | 0.00 +/- 0.00 | 3.67 +/- 2.08 | 3.67 +/- 2.08 | 196.61 +/- 49.18 |
| Wide coral garden | 0.00 +/- 0.00 | 4.00 +/- 0.00 | 4.67 +/- 0.58 | 8.67 +/- 0.58 | 168.09 +/- 22.32 |

Mask count is a coverage proxy, not recall. The isolated-crab control stayed at
exactly one mask in all three strict-threshold runs. WoRMS/generic SAM3 prompts
made the wide-garden result much more stable, but did not yield a safe proposal
on the dense thicket. Dense-thicket coverage still depends on stochastic Fable
click discovery.

The border fallback contributed zero proposals in all 15 frame-runs. On the
anemone/tubeworm frame it returned empty API content after retries in all three
runs, producing all three recorded API failures and much of that frame's
runtime variance. The production launcher now disables it.

## Evaluation caveat and next experiment

The in-app browser connected through the SSH tunnel but rendered the final WSL
PNGs as black, so the new final composites still require human visual sign-off.
Earlier visual inspection of a dense-frame plumbing overlay confirmed real
coral-colony masks but also visible misses.

The dense-frame three-pass test recovered 8, 8, and 7 masks: 7.67 +/- 0.58 at
288.13 +/- 8.70 seconds, with zero API failures. The two-pass comparison was
3.67 +/- 2.08 masks at 196.61 +/- 49.18 seconds. This supports the implemented
adaptive policy: two passes normally, with one extra pass only when Sonnet plus
verified text proposals found nothing.

The next evaluation should add mask-level human adjudication for the new
seven/eight-mask dense composites and compare the adaptive policy end to end on
all five frames. Continue reporting three independent repeats; do not treat raw
mask count as recall.
