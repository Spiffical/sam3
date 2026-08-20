# SeaTube five-frame presentation benchmark

_Exploratory run: 2026-08-16. Anthropic API calls were used for the agent and
annotation-matching stages._

## Fixed comparison set

All methods use the same decoded frame nearest 5.005 seconds in each of five
manually reviewed 10-second SeaTube clips. All whole-frame annotation
candidates are WoRMS records from one annotator, Meagan Putts (`userId=44113`).
The source definition is `configs/seatube_meagan_five_frames_v1.json`.

The five cases are an isolated crab, sea stars on coral, a mixed
anemone/tubeworm field, a dense coral thicket, and a wide coral garden. Video
context was inspected when selecting the frames and when judging ambiguous
biological versus non-biological targets.

This is a presentation set, not ground truth. Counts below are model outputs;
they are not recall or precision estimates. The inherited checkout was dirty at
Git `6815d0871171426b49851fa06d7d5b2934dfd5df`, so every run is explicitly
marked exploratory and records source hashes.

## Segmentation and box results

FathomNet values are deterministic box counts at confidence 0.10. Agent values
are mask counts reported as mean plus or minus sample standard deviation over
three independent repeats.

| Method | Crab | Stars/coral | Mixed field | Dense thicket | Wide garden |
| --- | ---: | ---: | ---: | ---: | ---: |
| MBARI-315k YOLOv8, 1280 px | 0 | 2 | 5 | 3 | 2 |
| Megalodon-2023 YOLOv8, 640 px | 1 | 1 | 6 | 7 | 17 |
| Benthic 2025, 640 px | 1 | 2 | 2 | 9 | 31 |
| Vanilla SAM3 agent, Sonnet 4.6 | 1.00 +/- 0.00 | 3.00 +/- 0.00 | 6.67 +/- 0.58 | 0.00 +/- 0.00 | 0.00 +/- 0.00 |
| Custom SAM3 agent, Sonnet 5 | 1.00 +/- 0.00 | 5.33 +/- 0.58 | 8.33 +/- 1.15 | 6.33 +/- 3.06 | 6.33 +/- 2.52 |

The vanilla run completed all 15 frame-runs. Its mean runtimes were 42.96,
328.98, 99.40, 368.41, and 298.24 seconds respectively. It cleanly segments
the crab, three sea stars, and most obvious animals in the mixed field. In both
dense scenes Sonnet describes plausible fauna but every SAM3 phrase lookup
returns no mask, across all three repeats.

The custom run also completed all 15 frame-runs without a persistent API
failure. It keeps first-pass masks, runs three region-diversified mask-guided
discovery passes with nearby frames at +/-0.5 and +/-1.0 seconds, uses SAM3
click mode plus `refine_group_mm_hybrid`, and keeps whole-frame review disabled.
The recovery prompt includes sessile and colonial animals such as coral,
gorgonians, and sponges. This expanded target definition is part of the custom
method and explains some of the count increase relative to the vanilla
`small creatures` prompt.

Visual review supports the qualitative comparison but also exposes limits:

- MBARI-315k under-detects these benthic scenes. Megalodon and Benthic 2025
  return more boxes, but the dense results include under-separation,
  fragmentation, and many overlapping boxes.
- The custom masks are tight and presentation-ready for the crab and sea-star
  cases, and recover major fauna in both scenes where the vanilla agent returns
  zero.
- Custom dense-scene discovery remains stochastic. The dense-thicket mask count
  is 7, 9, and 3; the wide-garden count is 6, 9, and 4. Repeat 2 is the cleanest
  representative, but interlaced coral colonies are sometimes merged and one
  small dark substrate candidate appeared in repeat 1.

WSL-only run roots:

- FathomNet: `runs/presentation_benchmark/seatube_meagan_v1/fathomnet/20260816_all4_imgsz640_1280`
- Vanilla: `runs/presentation_benchmark/seatube_meagan_v1/sam3_agent/20260816_sonnet46_repeats3`
- Custom: `runs/presentation_benchmark/seatube_meagan_v1/custom_flow/20260816_sonnet5_hybrid_passes3_gate050_fullfauna_repeats3`

## Sonnet 5 annotation-to-mask matching

The matcher receives only the candidate WoRMS annotations already attached to
the clip, the raw target frame, numbered masks, and temporal context. It cannot
invent a taxon, may assign one presence annotation to multiple objects, and
must explicitly leave unmatched objects and annotations. Consensus requires at
least two of three valid calls; confidence is reported as mean plus or minus
standard deviation.

| Frame | Masks | Candidate annotations | Consensus assignments | Unmatched masks | Unmatched annotations |
| --- | ---: | ---: | ---: | ---: | ---: |
| Isolated crab | 1 | 1 | 1 | 0 | 0 |
| Stars/coral | 5 | 2 | 4 | 1 | 0 |
| Mixed field | 9 | 5 | 6 | 3 | 3 |
| Dense thicket | 9 | 4 | 0 | 9 | 4 |
| Wide garden | 9 | 16 | 8 | 1 | 8 |

The crab match is strong (`Brachyura`, 0.97 +/- 0.00). Three sea-star masks
reach consensus as `Asteroidea`; the coral match is more uncertain. Mixed-field
consensus primarily assigns `Actiniaria`, with one `Lamellibrachia` match. The
wide-garden slide contains eight candidate-constrained labels, but several are
low-confidence and should be presented as model predictions, not truth.

Leaving every dense-thicket item unmatched is a useful honest failure case:
the segmentation masks are broad coral colonies, while the candidate records
name small epifauna that were not isolated as individual masks. The matcher did
not force plausible-sounding labels onto incompatible objects.

The three-repeat match run is:
`runs/presentation_benchmark/seatube_meagan_v1/annotation_match/20260816_sonnet5_custom_repeat2_repeats3`.

## Presentation sequence

The WSL-only presentation folder contains 35 validated images, seven per fixed
frame, with numeric filenames in display order:

`runs/presentation_benchmark/seatube_meagan_v1/presentation_sequence/20260816_v1`

The order is raw frame, MBARI-315k, Megalodon, Benthic 2025, vanilla Sonnet 4.6,
custom Sonnet 5, then Sonnet 5 annotation matches. The final matching images
show mask numbers, consensus labels, confidence, repeat support, and explicit
unmatched objects. After explicit user authorization, only these 35 curated
presentation images were copied to the Mac under
`docs/seatube_meagan_five_presentation_assets`. No video, dataset, weight,
secret, cache, or general run output was copied.

## 2026-08-18 presentation-asset refresh

The five `30_custom_sonnet5.png` and five
`40_sonnet5_annotation_matches.png` files were replaced after a qualitative,
frame-by-frame visual-QA pass. These are selected presentation examples, not a
new scored benchmark row; the scored comparison above remains the three-repeat
2026-08-16 experiment.

The refreshed segmentation examples contain 1, 6, 7, 44, and 17 masks. The
adaptive SAM3 phrase planner selected scene-specific retrieval phrases and did
not use `small creatures` for either dense coral scene. It selected `crabs` for
the isolated crab and `sea anemone` plus `Lamellibrachia` for the mixed field.
The dense-thicket diagnostic selected only `branching coral`. The wide garden
selected `coral`, `branching coral`, and `sea fan` before spatial residual
discovery. The sea-star/coral slide was subsequently corrected with the compact
taxon-plus-morphology bank described below.

Visual QA led to several implementation corrections:

- mask refinement and final verification now treat the discovery description as
  untrusted and require explicit `complete_identity` and `single_identity`
  decisions; this rejects both isolated branch fragments and masks that combine
  foreground/background colonies;
- SAM3 text candidates remain separate instances; there is no prompt-specific
  fragment stitch or expected object count. Text-proposal verification receives
  raw pixels, a thin contour-only candidate map, per-candidate temporal crops,
  exact mask overlap, and measured optical flow at +/-0.5 seconds. The resulting
  scene-agnostic depth/continuity check keeps the two near-disjoint background
  corals separate from each other and from the foreground coral;
- across three identical frame-02 repeats, both background candidates were kept
  every time (`6.00 +/- 0.00` final masks). The final mask JSON and rendered
  overlay hashes are identical across all three repeats;
- annotation matching now uses contour-only full-frame masks and raw per-object
  close-ups rather than a strongly color-filled identity map. Its general prompt
  compares structural morphology across scale, haze, focus, illumination, and
  depth while preserving separate segmentation identities;
- dense Sonnet 5 annotation matching now receives an 8,192-token minimum, with
  12,000 used for a truncated dense-frame repeat, instead of repeatedly
  exhausting the legacy 2,500-token budget.

The refreshed annotation slides use three valid responses per frame. Their
assignment / unmatched-object counts are 1/0, 6/0, 5/2, 0/44, and 7/10. The
matching prompt now treats annotation count as metadata rather than visual
evidence, not an assignment cap, while still preferring an explicit unmatched
object to an unsupported taxonomic guess. On frame 02 all three sea stars reach
Asteroidea consensus (0.82 +/- 0.06, support 3/3). The foreground coral and the
two distinct background corals each reach `Enallopsammia rostrata` consensus
(0.82 +/- 0.03, 3/3). Sharing the taxon does not merge their masks or biological
identities. This also removed
the earlier claim that all seven mixed-field masks were anemones; the two
ambiguous edge objects are shown as unmatched.

WSL-only selected segmentation view:
`runs/presentation_benchmark/seatube_meagan_v1/presentation_inputs/20260818_sonnet5_adaptive_visualqa_v1`.
The main three-repeat matching run is
`runs/presentation_benchmark/seatube_meagan_v1/annotation_match/20260818_sonnet5_adaptive_visualqa_repeats3_v1`;
the visually corrected mixed-field rerun is
`runs/presentation_benchmark/seatube_meagan_v1/annotation_match/20260818_sonnet5_frame03_visual_evidence_v2`.

The corrected WSL-only frame-02 segmentation repeats are
`runs/presentation_benchmark/seatube_meagan_v1/custom_flow/20260818_sonnet5_general_motion_identity_frame02_v16_r{1,2,3}`;
their shared presentation overlay comes from repeat 1. The corrected
three-repeat annotation match is
`runs/presentation_benchmark/seatube_meagan_v1/annotation_matching/20260818_sonnet5_general_morphology_match_frame02_v17_r1`.

### Frame-05 identity audit (exploratory)

Frame 05 exposed a general failure mode in which an independent keep/drop
review could preserve two fragments of one whip, while a union-only verifier
could accept a broad fan and a smooth whip that merely touched in projection.
The custom flow now reviews geometry-triggered components as same-identity
subgroups using shared target/earlier/later crops, then requires a focused
same-axis/same-branching-system confirmation before strict union verification.
Dense optical-flow scalars are deliberately excluded from this relationship
decision: stationary neighbors co-move, while one flexible identity can have
different local flow. Exact repeated identity/location attempts are also
suppressed across convergence passes without imposing a click or object cap.

The visually selected exploratory overlay is from WSL-only run
`runs/presentation_benchmark/seatube_meagan_v1/custom_flow/20260818_sonnet5_frame05_v37_collinear_fragment_gate_plumbing`.
It has 22 masks: the two collinear whip-fragment pairs are consolidated, while
the fan/whip contact and the white base organism/fan remain separate. This is a
plumbing and presentation result, not a scored comparison; a scored claim still
requires at least three repeats from committed source with mean +/- standard
deviation.

### Frame-04 convergence and frame-05 annotation refresh (exploratory)

The frame-05 relationship strategy was rerun on the selected 44-mask dense
thicket from WSL-only run
`runs/presentation_benchmark/seatube_meagan_v1/custom_flow/20260818_sonnet5_frame04_v39_temporal_relation_repeat_filter_plumbing`.
Five spatial passes evaluated 14 proposals. None survived complete-identity,
single-identity, background, and duplicate checks; the fifth pass repeated the
seven already evaluated locations and the cross-pass repeat filter ended the
loop without a click/pass cap. The final 44-mask overlay is byte-identical to
the previously selected presentation asset. The relationship audit proposed
two possible fragment pairs, but both focused checks reached only 0.62
confidence and therefore remained separate under the 0.75 merge threshold.

The refreshed frame-05 annotation slide uses the 22-mask v37 segmentation and
three valid Sonnet 5 responses from WSL-only run
`runs/presentation_benchmark/seatube_meagan_v1/annotation_match/20260818_sonnet5_frame05_v37_identity_geometry_repeats3_v1`.
Majority support alone proved too permissive: it retained several explicitly
tentative assignments with mean confidence below 0.50. The general consensus
gate now requires both support from at least 2/3 repeats and mean confidence at
least 0.55. The presentation result contains 10 assignments and 12 explicit
unmatched objects: five `Chrysogorgia` masks, one `Paragorgia`, one
`Farreidae`, two `Narella`, and one `Walteria`. Per-object confidence is shown
as mean +/- standard deviation on the slide. This remains exploratory because
the inherited source tree is intentionally uncommitted.
