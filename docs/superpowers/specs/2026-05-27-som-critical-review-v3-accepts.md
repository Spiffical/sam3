# Critical Review of v3 SoM Accepts

**Date**: 2026-05-27 · **Branch**: `spencer/nibi-work`
**Reviewer**: human-instructed visual audit
**Predecessor**: [`2026-05-26-som-missed-creature-loop-smoke-test-v3.md`](2026-05-26-som-missed-creature-loop-smoke-test-v3.md)

The v3 smoke-test report claimed **6 accepted masks across 3 ONC videos**. The user asked for an audit of each one: are the masks actually on creatures, or is the MLLM judge accepting incorrect masks? Spoiler — three of the six are not good.

## Method

For each accepted mask I built two views:

1. **Composite** — RAW | TEXT-AGENT MASKS (cyan) | TEXT-AGENT + SoM-ADDED (red). The user requested text-agent masks be visible alongside the SoM additions for context.
2. **Zoom crop** — RAW | RAW + MASK on a tight bbox around the added mask so a human (or me) can actually see what's under it.

Composites are in `2026-05-26-smoke-test-v3-images/critical/`; zoom crops in `2026-05-26-smoke-test-v3-images/critical/zooms/`.

## Verdict matrix

| Frame | Mask area (px) | Click description | Visual reality | Judge call | My audit |
|---|---|---|---|---|---|
| chinacreek f17 | 1,272 | "small creature" | Shrimp/small crustacean clearly visible in raw zoom; mask roughly covers body | ACCEPT (after refinement flipped) | ✅ **TRUE POSITIVE** |
| rank01 f113 | 3,603 | "faint fish upper right" | Dark scene, possible elongated fish-shape but indistinguishable from marine-snow streak | ACCEPT | ⚠️ **UNCERTAIN — too dark to confirm** |
| rank01 f177 | 2,158 | "small fish behind primary fish" | A fish is clearly visible in the zoom; mask roughly covers its body | ACCEPT | ✅ **TRUE POSITIVE** |
| rank01 f238 #1 | 11,574 | "small fish swimming center" | Multiple fish are visible in the area, but the mask is a BLOB covering several fish + interstitial water | ACCEPT | ❌ **BAD MASK** — covers multiple objects, not usable for instance-segmentation dataset |
| rank01 f238 #2 | 802 | "small fish upper left" | Sub-region is dark with marine-snow specks; nothing distinctly biological at the click location | ACCEPT | ❌ **FALSE POSITIVE** — looks like marine snow |
| rank02 f23 | 4,016 | "small benthic organism" | Zoom shows clean gravel substrate with no discernible biological subject | ACCEPT | ❌ **FALSE POSITIVE** — substrate |

**Summary: 2 true positives, 1 uncertain, 1 bad mask, 2 false positives. Roughly 33% acceptable rate** on the curated test set. **The judge is over-accepting.**

## Why the judge fails

Reading the judge's response text on the suspect cases is illuminating:

### rank01 f238 (bad mask covering multiple fish)

Judge's own reasoning:

> Mark 1 covers what appears to be a *cluster* of small creatures or a single creature — the irregular bright shape is consistent with a biological subject visible in the references.

The judge **knew** the mask was on a cluster and accepted anyway. The current prompt's bar is "is there biology under here" — but the correct bar for a labelling dataset is "is the mask cleanly outlining a single instance". The judge is silent on instance-level segmentation quality.

### rank02 f23 (substrate accepted)

> However, the mark is quite small and the image quality at that distance makes it difficult to be certain. […] Given the presence of a consistent object at roughly that location across frames and the organic shape of the mark, I'll accept mark 1 as corresponding to a small biological subject.

The judge admits it can't be sure but accepts because the *click was placed somewhere* and the shape is "organic-looking". The "prefer REJECT when uncertain" rule from the system prompt was not invoked here. Likely because the discovery's description ("small benthic organism") primed the judge to look for a small creature and find one — confirmation bias.

### rank01 f113 (uncertain — dark scene)

> The mark covers a bright, somewhat irregular shape that resembles a fish body rather than substrate or debris. […] I'll accept mark 1 as a biological subject matching the "small creatures" query.

This one is genuinely ambiguous. Marine snow flakes can look fish-shaped in still frames. Without checking temporal consistency more carefully (e.g. does the shape persist across multiple reference frames at the same pixel?) the judge can't tell.

## Rejects: were those right?

I also reviewed the 4 rejected cases (chinacreek f30/f45/f59, rank02 f166) by reading composites + their judge responses:

- **chinacreek f30, f45, f59**: SAM3 click mode returned large green substrate blobs at the proposed click locations. Judge correctly rejected as "covers mostly substrate." ✅ correct rejects.
- **rank02 f166**: text-agent already had a clean rockfish mask. SAM3 click-mode mask was slightly different shape but mostly on the same rockfish. Pre-judge filter let it through (IoU < 0.3 against existing, narrowly). Judge rejected as "duplicate-of-existing in spirit." ✅ correct reject.

So the rejects were all defensible. The problem is on the accept side.

## Root causes of the bad accepts

1. **Judge prompt doesn't require single-instance segmentation quality.** It checks "is there biology here," not "does this mask cleanly outline one creature."
2. **SAM3 click mode's smallest-in-band selection can still return a blob.** When a click lands between several small fish, the "smallest" of three multimask outputs may still be a big amorphous region.
3. **Discovery descriptions can be aspirational, not grounded.** "Small fish upper left" doesn't constrain the judge to actually verify a single fish is at the click location.
4. **Reference-frame corroboration is hand-wavy.** The judge often says "there's biology in this area of the references" without checking persistence at the specific click pixel. Marine snow and substrate look biological in single frames.
5. **No mask-shape sanity check.** A mask that is 11,574 px while the description says "small fish" should be auto-flagged. A mask with many disconnected components likely covers many creatures. Neither is checked today.

## Concrete remediation plan

### Cheap (prompt + filter changes)

1. **Add a per-component check** to `filter_candidates_with_reasons`: if the mask has > 1 connected component (with a sane min-component-area threshold), drop as `multi_component_blob`. This kills rank01 f238 #1 immediately because that mask is multiple disconnected fish-shapes.
2. **Tighten the judge system prompt**: explicit instructions to reject a mark if (a) the mask covers more than one distinct subject, (b) the mask boundary is mostly substrate, (c) the click description says "small" but the mask is > 2% of the frame. Make the prompt enumerate failure modes, not just success criteria.
3. **Require persistence checking**: have the judge call out "is the same shape at the same pixels in 3 of 4 reference frames?" rather than "is biology in this area?" This would have flagged rank01 f113 and rank02 f23 as uncertain.
4. **Pass the discovery description into the judge prompt** as a *contract*: "the discovery step claimed this is a *small fish*. Reject if the mask is larger than what a small fish would occupy at this image scale." Today the judge sees only the marked image, not the discovery's description.

### Medium (pipeline changes)

5. **Add a "shape sanity" stage between SAM3 and the judge.** For each candidate, compute (a) mask area as % of frame, (b) bbox aspect ratio, (c) number of connected components, (d) ratio of mask area to bbox area (solidity). Cheap to compute; cheap to filter on; would have caught rank01 f238 #1 and #2 and rank02 f23.
6. **Two-pass judge**: first call asks "is the *shape* of this mark a clean instance?" — accept/reject on segmentation quality alone. Second call (only on shape-accepted) asks "is this instance biological?" — accept/reject on biology. Today the two questions are conflated.

### Larger (architectural)

7. **Replace single-click discovery with bbox discovery for verification.** Have the MLLM propose a tight bbox around what it thinks is missed, run SAM3 in *box-prompt* mode (which is more constrained than point mode and less prone to blob masks), and compare. Bboxes from MLLMs are notoriously better than raw point coords, and SAM3 box mode is well-tested.
8. **Cross-check accepted masks against the same-frame text-agent mask population**: an accepted mask that's structurally different in *area distribution* from all the existing accepted masks on the same frame is suspicious. This is the same-frame consistency the judge is trying to use but with quantitative grounding.

## What this means for the dataset

If we were to ship the v3 accepts as-is, **3 of 6 new labels would be wrong or unusable**. Out of 20 target frames, the pipeline added 0 reliable labels on rank02/rank03/rank04, 1 reliable label on chinacreek (the shrimp), and 1–2 reliable labels on rank01 (the fish). **Effective production yield: ~3 good masks out of 20 frames analysed.**

The mechanism works. The judge needs a real upgrade before we shovel any of this into a training set.

## Recommended next iteration

Smallest viable set of changes for v4:
- (1) connected-components filter (1 hour, low risk, catches blob masks)
- (3) judge prompt rewrite for persistence + single-instance discipline (1 hour, low risk)
- (4) thread discovery description into judge prompt (~30 min, low risk)
- Then re-run on the same 5 ONC clips and audit again.

Larger changes (5, 7) are worth a follow-up brainstorm.

## Artefacts in this review

- `critical/{chinacreek_f17, rank01_f113, rank01_f177, rank01_f238, rank02_f23}.png` — composite views with text-agent masks (cyan) + SoM additions (red).
- `critical/{chinacreek_f30_rejected, chinacreek_f45_rejected, chinacreek_f59_rejected, rank02_f166_rejected}.png` — composite views for context.
- `critical/zooms/*.png` — per-accept zoomed-in raw vs. raw+mask crops.
- Per-target judge response text quoted above is from `runs/som/<video>/<timestamp>/target_NNNNNN/05_judge_response.txt`.
