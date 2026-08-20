# Sonnet 5 + SAM3 phrase/taxon experiment

_Exploratory diagnostic: 2026-08-17. This used Claude Sonnet 5 only. It did
not call Fable and did not run a click-recovery stage._

## Goal and protocol

Test whether a persistent-mask SAM3 agent loop can improve recall on three of
the crowded SeaTube presentation frames by varying only the initial phrase and
the available WoRMS taxa. All runs used:

- Anthropic model `claude-sonnet-5` for planning and mask inspection;
- SAM3 text grounding at confidence threshold 0.40;
- one fixed selected frame per clip;
- accumulated masks across agent turns; and
- rendered-overlay review rather than count-only evaluation.

These runs are exploratory rather than scored because the inherited source
checkout is intentionally dirty and has no committed experiment baseline.
Datasets, frames, model weights, secrets, and outputs remained on WSL.

## Prompt screen

| Frame | Prompt strategy | Accepted masks | Visual interpretation |
| --- | --- | ---: | --- |
| 03, anemone/tubeworm field | known taxa: `Actiniaria, Lamellibrachia, and Decapoda` | 6 | Broadest single screen; includes clear anemones and tube-associated objects, but one small top-left proposal is questionable. |
| 03 | force exact `Actiniaria` first | 4 | Cleaner anemone-only result, but narrower than the multi-taxon cue. |
| 04, dense coral thicket | `all visible marine life` | 0 | Failed after exhausting the generation cap. |
| 04 | sessile morphology list | 0 | Failed after generic coral, sea-fan, sponge, polyp, and related searches. |
| 04 | known taxa list | 0 | The agent paraphrased taxa, but no phrase grounded a mask. |
| 04 | force all four exact taxa separately | 0 | `Amphianthus`, `Comatulida`, `Ophiacanthidae`, and `Parazoanthidae` all returned zero; common-name and shape fallbacks also returned zero. |
| 05, wide coral garden | `all visible marine life` | 0 | Generic synonyms returned zero. |
| 05 | sessile morphology list | 0 | Generic coral/sponge/fan terms returned zero. |
| 05 | known taxa list | 0 | The agent paraphrased the list and never tried the useful exact taxon. |
| 05 | force exact `Walteria` first | 3 | Retrieved two plausible distinct upright structures, plus one overlapping partial duplicate. Most visible benthic life remained unmasked. |
| 05 | force exact `Chrysogorgia` first | 0 | No useful mask at threshold 0.40. |
| 05 | exact `Walteria`, then `Chrysogorgia` | 2 | `Chrysogorgia` added nothing; the extra review removed the overlapping Walteria duplicate and produced the cleanest frame-05 overlay. |

## What the images show

The successful frame-03 proposals follow discrete animal outlines rather than
simply painting the substrate. The broader taxa cue improves apparent recall
over exact `Actiniaria`, although the screen contains one questionable small
proposal near the on-screen graphic. Sonnet's own per-mask inspection can
remove that false positive, but selection varies across runs.

Exact `Walteria` is a real retrieval handle in frame 05, not a taxonomic
classification result. It masks only a couple of slender upright structures
and misses the many visible fans, corals, sponges, and other colonies. The
three-mask screen also contains an overlapping partial duplicate; counting it
as three creatures would be misleading.

The verifier also incorrectly described `Walteria` as a sea pen in one repeat.
The [World Porifera Database](https://marinespecies.org/porifera/porifera.php?p=image&pic=152401&tid=1550203)
places the genus in Porifera. This is direct evidence that the verifier's
free-text taxonomic explanation cannot be used as a label, even when its mask
accept/reject decision is visually useful.

Frame 04 is the clearest hard case. The raw image visibly contains a dense
field of branching colonies and attached bulb-like organisms, yet the direct
scientific names and a broad vocabulary of common/morphological alternatives
all return zero. Sonnet recognizes and describes the scene correctly; SAM3
does not ground those concepts in this frame.

## Three-repeat validation

| Condition | Selected masks by repeat | Mean +/- sample SD | Runtime, seconds (mean +/- sample SD) | Visual QA |
| --- | --- | ---: | ---: | --- |
| Frame 03, known taxa cue | 5, 7, 6 | 6.00 +/- 1.00 | 186 +/- 76 | SAM3 generated the same seven proposals. Sonnet variably retained a logo-adjacent false positive and small tube-worm-like objects. |
| Frame 03, exact `Actiniaria` first | 4, 4, 4 | 4.00 +/- 0.00 | 106 +/- 8 | The same four plausible anemones were retained in every repeat; two ambiguous/non-anemone proposals were consistently removed. |
| Frame 05, exact `Walteria` first | 3, 2, 3 | 2.67 +/- 0.58 | 107 +/- 41 | Repeats 1 and 3 retained a partial mask that overlaps the first mask. Visual distinct-instance count is 2, 2, 2 rather than 3, 2, 3. |

The multi-taxon cue is therefore the better recall-oriented prompt on frame 03,
while exact `Actiniaria` is the more stable narrow prompt. Exact `Walteria`
reliably retrieves two visible structures in frame 05, but the verifier needs a
deterministic overlap/containment deduplication pass; prompting it to keep
"distinct organisms" is insufficient. None of these text-only strategies
approaches all-life recall in the wide garden or dense thicket.

## Temporal QA and the refined frame-03 phrase union

Reviewing the full ten-second frame-03 clip changed the interpretation of two
ambiguous regions:

- the red sliver at the far-left edge of the selected frame is a real sessile
  organism/plume moving out of frame between 4.5 and 5.5 seconds; and
- a camouflaged fish-like animal is visible in the dark left crevice across
  adjacent frames, but it is not recovered by the `fish` prompt.

A focused, no-MLLM SAM3 screen tested ten visual/common-name variants for the
crevice animal. At threshold 0.40, `fish`, `fish head`, `animal in crevice`,
`eel`, `moray eel`, `blenny`, `goby`, and `rockfish` all returned zero.
`small fish` returned two masks, including the crevice animal, while
`small creature` returned one. This is a useful reminder that modifiers can
materially change SAM3 retrieval even when the underlying concept is the same.

The final Sonnet 5 condition forced both `sea anemone` and `small fish` before
returning. Against a video-audited seven-instance set:

| Repeat | Selected | TP | FP | FN | Failure mode |
| ---: | ---: | ---: | ---: | ---: | --- |
| 1 | 6 | 6 | 0 | 1 | The agent inspected after the first phrase and dropped the valid left-edge plume before adding `small fish`. |
| 2 | 8 | 7 | 1 | 0 | The agent accumulated both phrases but skipped inspection, retaining a logo-adjacent artifact. |
| 3 | 7 | 7 | 0 | 0 | The reviewed best repeat retained all seven video-confirmed organisms and removed the logo artifact. |

Across the three repeats, selected-mask count was **7.00 +/- 1.00** and runtime
was **123 +/- 96 seconds**. Video-audited precision was **0.958 +/- 0.072**,
recall **0.952 +/- 0.082**, and F1 **0.952 +/- 0.042** (sample standard
deviations). The best overlay is presentable, but the repeat variance is an
architecture warning: the same phrase bank can yield a false negative, a false
positive, or the correct set solely because Sonnet chooses a different review
order.

## Prompt policy supported by this experiment

1. Treat a WoRMS name as a retrieval handle, never as an assigned label.
2. Execute the evidence-backed phrase bank deterministically before handing
   proposals to the free-form verifier. For frame 03 this bank is
   `sea anemone` plus `small fish`; for frame 05 it includes exact `Walteria`.
   The agent must not be allowed to inspect halfway through the bank or skip
   inspection afterward.
3. Force each promising scientific name as an exact, separate
   `segment_phrase` call rather than allowing the agent to paraphrase it. The
   long frame-05 taxon list failed because Sonnet skipped exact `Walteria`.
4. Accumulate the union before inspection, then deterministically remove
   overlapping partial duplicates and known overlay/logo regions. Do not
   equate raw mask count with creature count.
5. Follow exact taxa with only a short evidence-backed common/morphology bank.
   Stop synonym search when SAM3 repeatedly returns zero; twenty variants did
   not recover the dense thicket.
6. Keep `small creatures` as a complementary proposal source for motile/small
   animals, but use the broader objective **all visible marine life** for
   deciding what must ultimately be covered.
7. Use adjacent frames to adjudicate ambiguous edge/camouflaged proposals;
   single-frame appearance caused Sonnet to reject a true plume in this test.
8. Escalate text-grounding failures to a later Sonnet 5 point/click recovery
   stage. Phrase engineering alone cannot reach high recall on frames 04 or 05.

The screen and validation runners are
`scripts/run_sonnet5_agent_prompt_screen.sh`,
`scripts/run_sonnet5_exact_taxon_refinement.sh`, and
`scripts/run_sonnet5_prompt_validation.sh`.

## Deterministic proposal-bank implementation and validation

The follow-up implementation removes phrase selection and review ordering from
the free-form agent loop:

1. execute every configured generic, morphology, and WoRMS retrieval phrase;
2. suppress masks whose boxes lie mostly inside a configured logo/overlay region;
3. union and deduplicate masks using intersection-over-minimum;
4. for evidence-backed prompts only, stitch bbox-overlapping fragments of one
   instance (`Walteria` on frame 05);
5. show the selected frame, a before/after temporal strip, and the numbered
   proposal union to Sonnet 5; and
6. make exactly one Sonnet request that returns the final mask subset. No Fable
   or click recovery was used in this experiment.

The implementation is in `sam3/agent/proposal_bank.py`, the proposal-only
verifier prompt is
`sam3/agent/system_prompts/system_prompt_proposal_verification.txt`, and the
fixed-frame banks are recorded directly in
`configs/seatube_meagan_five_frames_v1.json`.

Three-repeat WSL validation at SAM3 threshold 0.40 produced:

| Frame | Phrase-bank result | Final masks | Runtime, seconds | Visual QA |
| --- | --- | ---: | ---: | --- |
| 03, anemone/tubeworm field | 14 raw proposals; 2 logo proposals suppressed; 12 proposals geometrically reduced to 7 | 7.00 +/- 0.00 | 8.06 +/- 0.17 | All seven video-audited organisms retained in every repeat, including the left-edge plume and crevice fish; no logo false positive. Precision, recall, and F1 were each 1.000 +/- 0.000 against the seven-instance audit. |
| 04, dense coral thicket | Every one of 9 generic/morphology/taxon phrases returned zero | 0.00 +/- 0.00 | 5.93 +/- 0.10 | Clear failure: the frame visibly contains extensive branching life. Sonnet was not called because SAM3 produced no proposal to verify. |
| 05, wide coral garden | Only exact `Walteria` returned masks: 3 fragments, stitched into 2 instances | 2.00 +/- 0.00 | 7.41 +/- 0.31 | The two retained instances are plausible life, and the two pieces of the taller structure now share one instance ID. Many obvious fans, corals, and other colonies remain completely unmasked. |

The three final RLE mask sets were byte-identical within each condition. This
eliminates the frame-03 selection variance from the free-form phrase loop and
fixes the known logo and fragmented-instance errors. It does **not** solve the
dense-scene recall problem: when SAM3 text retrieval returns no mask, a
selection-only verifier has nothing to recover. The next recall milestone must
therefore add the Sonnet 5 point/click recovery stage after this deterministic
bank, especially for frames 04 and 05, while retaining this bank as the cheap
first proposal source.

WSL-only run roots:

- `runs/presentation_benchmark/seatube_meagan_v1/sonnet5_deterministic_proposal_v1/20260817_frame03`
- `runs/presentation_benchmark/seatube_meagan_v1/sonnet5_deterministic_proposal_v1/20260817_frames04_05`
- `runs/presentation_benchmark/seatube_meagan_v1/sonnet5_deterministic_proposal_v1/20260817_frame05_fragmentmerge_v2`
