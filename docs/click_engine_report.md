# Click-Engine Report — Finding Missed Creatures in ONC Seafloor Video

_Last updated: 2026-05-31_

## 1. Goal and where the click engine fits

We are building a labeled dataset of segmented underwater creatures in Ocean
Networks Canada (ONC) video. The production pipeline is:

1. **Text-prompt SAM3 agent** does a first pass over each frame.
2. **MLLM "click" verification** (the *click engine*, this report) finds creatures
   the first pass **MISSED**, by placing point clicks on them.
3. Each click is refined and handed to **SAM3 click-mode** to produce a mask.
4. A judge **accepts / rejects / refines** each mask.

This report covers stage 2 — the click engine — and its current quality. The goal
this round: **make the click engine identify as many real creatures as possible
while placing as few false ("stray") clicks as possible.**

## 2. How we measure (and a hard-won rule)

Two harnesses, both in `scripts/`:

- **End-to-end (`click_engine_probe.py --eval-e2e`)** — the MLLM places its own
  clicks (no ground truth), SAM3 masks them, and we score final mask IoU vs GT by
  greedy matching (threshold 0.5). Decomposes into **recall** (did we find the
  creature), **IoU** (mask quality), and **strays** (false masks). This is the
  whole subsystem exactly as production runs it.
- **Click-level bake-off (`tile_stray_bakeoff.py`)** — no SAM3. Just: *does each
  placed click land on a GT creature?* Recall = fraction of GT clicked; strays =
  clicks on substrate. Because there is no GPU stage, every strategy runs as its
  own parallel process. The click-level base strays (4.0) **exactly matched** the
  e2e baseline (4.0), so it is a faithful, cheap proxy for ranking strategies; the
  winner is then confirmed in the full e2e.

> **Rule, learned the hard way: always repeats ≥ 3, report mean ± std.**
> Per-frame recall std is ~0.09–0.19 — larger than many deltas we judge. Single-run
> screens have flipped conclusions in *both* directions (e.g. a repeats=1 screen
> once showed a frame going 0.40→1.00 that was really 0.87±0.09 → 0.93, a fluke).
> Repeats=1 is valid only as a crash/plumbing screen, never for per-frame deltas.

The benchmark set is 6 hard chinacreek frames (42, 45, 54, 59, 18, 24); "strays/pass"
below is the **sum over those 6 frames** per pass.

## 3. Recall lever #1 — motion difference (dead end)

Hypothesis: the creatures we miss are camouflaged movers a motion-difference
pre-pass could surface. **Falsified.** A pixel-level probe
(`scripts/motion_diff_probe.py`) over the chinacreek clip showed only ~1 creature
per scene actually moves (~95–99% diff coverage); the others are sessile (sea
stars, snails, anemones) and sit at 0% even at low thresholds. **The recall ceiling
is a static-camouflage detection problem, not a motion problem.** Motion-diff
abandoned.

## 4. Recall lever #2 — tiled high-resolution sweep (works)

Split the frame into a 2×2 overlapping grid, upscale each tile, and ask the MLLM to
find every creature in the high-res crop, then map clicks back. This surfaces
small/camouflaged static creatures lost in the downsampled whole-frame view.

Confirmed at repeats=3 vs the no-tiling baseline:

| Config | Recall | IoU(all) | IoU(detected) | Strays/pass |
|---|---|---|---|---|
| baseline iterative (no tiling) | 0.782 | 0.708 | 0.905 | 4.0 |
| tiled, untuned | 0.908 | 0.821 | 0.904 | 21.3 |
| tiled, tuned (strict gate + tighter prompt) | 0.920 | 0.835 | 0.908 | 12.7 |

Tiling buys **+0.13 recall** but the wide "find every camouflaged creature" prompt
over-triggers on high-res substrate texture, so it costs precision. Tuning cut
strays 21.3 → 12.7 while holding recall, but 12.7 (≈ 2 false masks/frame) still
reaches the downstream accept/reject judge.

## 5. Cutting strays — the bake-off (this round's main result)

We tested a 2×2 of **{wide, conservative} tile prompt × {still, temporal} context**
plus a no-tiling floor, at the click level, repeats=3. "Temporal" = the same crop
region ~0.5 s before/after, shown at the two *existence-deciding* stages (the tile
proposal and the creature gate). Design is **paired**: every strategy starts from an
identical cached base-pass click set per (frame, repeat), so the delta is the tile
mechanism alone.

| Strategy | Proposal prompt | Temporal | Recall | Strays/pass | Raw cand/frame |
|---|---|---|---|---|---|
| `base` (no tiling) | — | — | 0.851 | **4.0** | 0 |
| `S0_control` (≈ current production) | wide | no | 0.977 | 14.67 | ~12 |
| **`S1_cons`** | **conservative** | no | 0.920 | **5.0** | ~3.7 |
| `S2_temporal` | wide | yes | 1.000 | 17.33 | ~11 |
| **`S3_cons_temp`** | **conservative** | **yes** | **0.989** | **8.33** | ~7 |

_(Click-level recall runs a touch higher than e2e mask-recall because it doesn't
require IoU ≥ 0.5; rankings hold. Click strays track e2e strays closely — base 4.0
matched, S0 14.67 ≈ e2e 12.7.)_

**Two findings:**

1. **A conservative tile prompt is the dominant stray lever.** Reframing the
   proposal from "find every camouflaged creature, including small ones" to *"MOST
   crops contain NO animal; an empty answer is expected and correct; click only when
   confident"* collapsed raw candidates from ~12 to ~3.7 per frame and cut strays
   **66%** (14.67 → 5.0), for a small recall dip (0.977 → 0.92).

2. **Temporal context is a RECALL lever, not the precision lever we expected.** Added
   to a wide prompt it *raises* strays (seeing persistence makes the model commit
   more clicks) — but it is what cracks the hardest static-camouflage frame:
   **cc_f059 went base 0.73 → conservative-still 0.80 → temporal 1.0.** The
   conservative prompt alone cannot find those creatures; temporal can.

**`S3_cons_temp` Pareto-dominates current production on both axes** — recall
**0.989 vs 0.92** and strays **8.33 vs 12.7** — at the cost of extra neighbour-crop
MLLM calls. **`S1_cons`** is the cheap lean alternative: it matches today's shipped
recall (0.92) at roughly one-third the strays (5.0).

Scope caveat: temporal was applied to the proposal + gate; the click-*centering*
refine loop stayed still-frame this round (it governs precision, not whether to
click). Adding it there is a fast follow-up.

## 6. Cold-start discovery on rank01 @ 5 s

The rank01 clip is a genuinely hard, dark, low-contrast seafloor scene heavy with
marine snow. **The Claude text-agent first pass found ZERO creatures across all 15
analyzed frames** — so this is a true cold start, and there is no Claude GT and no
human annotation to score against. We therefore report what the engine *places* and
judge correctness visually.

Run: full engine (whole-frame iterative finder → `S3_cons_temp` tile sweep) on the
frame at 5.0 s (video frame 150, 1024×768).

**Result: 7 clicks identified — 4 from the whole-frame finder, 3 from the tile
sweep — where the first pass found 0.**

![rank01 @ 5s clicks](click_engine_assets/rank01_t5s_clicks_2x.png)

Yellow `B#` = whole-frame finder, cyan `T#` = tile sweep.

| Click | (x, y) | Source | Description | Visual judgment |
|---|---|---|---|---|
| B1 | (0.370, 0.407) | whole-frame | central flatfish/demersal fish | on the central bright elongated shape |
| B2 | (0.042, 0.495) | whole-frame | small creature at left edge | faint edge shape — plausible |
| B3 | (0.620, 0.134) | whole-frame | fish upper-center | **uncertain** — faint on dark background |
| B4 | (0.719, 0.239) | whole-frame | fish upper-right | clear elongated fish — solid |
| T1 | (0.345, 0.409) | tile | large fish on seafloor | **near-duplicate of B1** (~25 px, just over dedup) |
| T2 | (0.013, 0.331) | tile | fish at left edge | edge streak — plausible |
| T3 | (0.516, 0.269) | tile | fish upper-right | clear elongated fish — solid |

**Honest read:** ~6 distinct candidate creatures, of which B4 and T3 are clearly
correct, B1/B2/T2 plausible, B3 questionable, and **B1≈T1 is a redundant pair** the
25 px dedup just missed. The click descriptions explicitly cite temporal evidence
("consistently visible across frames", "showing movement between reference frames"),
confirming the neighbour-frame context is being used. Net: on a frame where the
production first pass returned nothing, the click engine surfaces a handful of real
fish — exactly its intended value-add.

**Two concrete improvements this surfaced:**
- Bump the dedup radius (or dedup tile clicks against whole-frame clicks more
  loosely) so B1/T1-type pairs collapse to one.
- B3-type faint single-frame detections argue for the temporal gate to apply on the
  whole-frame finder too, not just the tile sweep.

## 7. Center-of-mass targeting, the global review pass, and Sonnet vs Opus 4.7

Three changes were trialled together and measured with the new full-engine harness
(`tile_stray_bakeoff.py engine` — whole-frame finder → tile sweep → optional review,
scored vs GT at repeats=3 on the 6 frames). Note: this click-level path omits the
per-click zoomed verify gate, so absolute strays run higher than e2e; **recall is the
trustworthy signal** for these decisions.

**(a) Center-of-mass targeting (kept).** The proposal, the per-click zoomed refine,
and the tile prompts now instruct the model to click the creature's centre of mass
(thickest central body), not a fin/tail/edge. The refine loop is zoomed, so this is
reliable and carries no recall cost.

**(b) A whole-frame "review everything" pass (tried, REJECTED).** We added a final
stage that shows the MLLM all placed clicks on the full frame at once and asks it to
remove strays/duplicates and nudge clicks to centre of mass. It **hurt recall badly**:

| Engine (repeats=3) | Recall | Strays/pass | review removed/moved per pass |
|---|---|---|---|
| Sonnet 4.6, no review | **0.954** | 18.67 | — |
| Sonnet 4.6, + review | 0.828 | 19.67 | rm 2.0 / mv 6.0 |
| Opus 4.7, no review | 0.759 | 16.33 | — |
| Opus 4.7, + review | 0.632 | 14.67 | rm 0 / mv 6.3 |

The review costs ~**0.13 recall** for ≤2 fewer strays — a bad trade. Two failure
modes: at full-frame resolution it cannot resolve camouflaged creatures and removes
them as "substrate"; and its centre-of-mass *moves*, placed on a downsampled frame,
land off small-creature bodies and turn hits into misses. Restricting it to
dedup-only still lost recall (it called distinct nearby creatures "duplicates").
**Replaced** by: (i) the zoomed per-click refine for centre of mass — already
reliable — and (ii) a deterministic geometric dedup (merge clicks within ~30 px,
which catches the rank01 B1/T1 pair) with no recall risk. The MLLM review remains in
the code behind an opt-in `--review` flag, documented as recall-hurting.

**(c) Sonnet 4.6 vs Opus 4.7 (Sonnet wins decisively).** Same engine, no review:
**Sonnet recall 0.954 vs Opus 0.759**, consistent on every frame (Opus ≤ Sonnet on
all 6; e.g. cc_f018 1.00 vs 0.60, cc_f054 1.00 vs 0.73). Opus is the more
*conservative/precise* model — on rank01 it produced a cleaner-looking 4 clicks vs
Sonnet's 7 — but against GT that conservatism means it **misses real, camouflaged
creatures**. For a recall-first "find what the first pass missed" task, that is the
wrong trade: strays are cheap to filter downstream, missed creatures are
unrecoverable. **Keep Sonnet 4.6.** (Strays were similar, 18.7 vs 16.3.)

## 8. Full e2e + SAM3 confirmation (the numbers that matter for shipping)

Recommended engine run end-to-end **with SAM3**: Sonnet 4.6 + `S3_cons_temp`
(conservative + temporal tile) + centre-of-mass prompts + geometric dedup + the
per-click zoomed verify gate → `refine_mm` masks → greedy IoU match vs GT, repeats=3
on the 6 frames (`tile_stray_bakeoff.py e2e`). Placed in the full e2e progression:

| e2e config (6 frames, repeats=3) | Recall | IoU(all) | IoU(det) | Strays/pass |
|---|---|---|---|---|
| no-tile baseline | 0.782 | 0.708 | 0.905 | 4.0 |
| untuned tiled | 0.908 | 0.821 | 0.904 | 21.3 |
| tuned tiled (previous production) | 0.920 | 0.835 | 0.908 | 12.7 |
| **NEW: S3 + centre-of-mass + dedup** | **0.874** | 0.792 | 0.906 | **6.0** |

**Read:** the new engine **more than halves strays (12.7 → 6.0)** for a **−0.046
recall** and −0.043 IoU(all) cost; mask quality on detected creatures is unchanged
(IoU(det) 0.906). Versus the no-tile baseline it is an excellent ratio — **+0.09
recall for only +2 strays**, where the old tuned-tiled spent +8.7 strays for +0.14.
So it is a genuine precision/recall **trade**, not the clean Pareto win the click-level
proxy implied (the IoU≥0.5 threshold + verify gate attenuate the conservative
engine's recall). Per-frame, the temporal tile finally cracked the hardest frame
(**cc_f059 recall 0.93**, was 0.73 at baseline); the recall drag is **cc_f045 (0.67)**,
where the per-click verify gate drops ~3/pass and takes a couple of real creatures
with the strays.

## 9. Mask-generation investigation (diagnostics + zoom/hybrid generators)

Visual diagnostics (`tile_stray_bakeoff.py diag`, renders clicks/gate/masks vs GT)
on the recall-drag frames overturned the "soften the gate" hypothesis. On f045 and
f042 the per-click verify gate dropped **0 real creatures** (only substrate). The
recall losses were instead:
- **clicking misses** — a dark, low-contrast corner creature the finder + tile both
  skipped (f045 GT3); and
- **SAM3 mask failures on small/thin creatures** — `refine_mm` either `"abandon"`ed
  a thin brittle-star (empty mask, IoU 0; f045 GT1) or accepted a `"good"` mask that
  localized onto adjacent substrate (f042 GT2, the smallest at 1094 px). Compact
  creatures ≥~2000 px masked at IoU 0.87–0.96.

Two new generators were built and run at full e2e (repeats=3):
- **`refine_group_mm_zoom`** — segment on a zoomed crop around the (already
  creature-verified) click, never abandon. In single-frame diag it recovered both
  failures (brittle-star IoU 0→0.80, tiny creature 0→0.91) — but across the 6-frame
  e2e it **dropped IoU(det) ~0.07** (looser masks from the crop→frame resize) for no
  clear recall gain.
- **`refine_group_mm_hybrid`** — full-frame generator by default, zoom fallback only
  when full-frame abandons/returns an empty mask. This **kept IoU(det) at 0.902**
  (vs baseline 0.906) but recall (0.839) was **within noise of the mm baseline
  (0.874)** — the abandon-recovery fires too rarely to move the aggregate.

**Conclusion: mask-gen is not the recall lever.** The e2e recall ceiling lives in the
**clicking stage** (finding dark/camouflaged creatures), not masking. The hybrid is a
safe, harmless net (can only turn an empty mask into a real one) and is fine to keep,
but it is not a recall win. `cmd_e2e` is now crash-resumable (per-frame
`frames_cache.json` checkpoint) after repeated interruptions.

## 10. Switching the vision model: Claude Fable 5 (the biggest win)

§9 showed the recall ceiling is the clicking stage. The cleanest lever for that is the
vision model itself. Swapping every MLLM call (finder, tile proposal, gate, refine,
mask-judge) from Sonnet 4.6 to **Claude Fable 5** (`claude-fable-5`), same pipeline:

Click-level engine A/B (no SAM3, repeats=3): **Fable recall 1.00 on every frame**
(incl. the f059 ceiling, 0.73 at baseline) at **strays/pass 10.33**, vs Sonnet 0.943 /
17.67 — Fable strictly dominates (perfect clicking recall AND ~40% fewer strays).

Full e2e + SAM3 (repeats=3, hybrid masks):

| e2e config (6 frames, repeats=3) | Recall | IoU(all) | IoU(det) | Strays/pass |
|---|---|---|---|---|
| Sonnet 4.6, S3 + hybrid | 0.874 | 0.792 | 0.906 | 6.0 |
| old tuned-tiled production | 0.920 | 0.835 | 0.908 | 12.7 |
| **Fable 5, S3 + hybrid** | **0.977** | **0.862** | 0.883 | **5.0** |

Fable wins on recall (+0.10 over Sonnet, near-perfect), IoU(all), and strays (lowest).
The only regression is IoU(det) 0.883 vs 0.906 — expected: Fable *detects more* of the
hard camouflaged creatures, and those marginal ones mask a little looser, pulling the
detected-average down. **This single change beats every prompt/tiling/mask-gen lever
in this report combined.** Per-frame Fable e2e: f042 1.00, f045 1.00, f054 1.00,
f059 0.93, f018 0.93, f024 1.00. Fable click/mask renders per frame: see §12.

## 11. Recommendation and next steps

- **Ship the Fable engine: Claude Fable 5 + `S3_cons_temp` + centre-of-mass +
  geometric dedup + `refine_group_mm_hybrid`; NO whole-frame MLLM review.** This is the
  recommendation — recall 0.977, IoU(all) 0.862, strays 5.0, beating every Sonnet
  config on recall and strays simultaneously (§10). The pipeline structure (tiling,
  conservative prompt, dedup, hybrid masks) all still help; Fable multiplies them.
- (Sonnet fallback, if Fable cost/availability is an issue: same pipeline on Sonnet 4.6
  gives 0.874 / IoU(det) 0.906 / strays 6.0 — lower recall but slightly tighter masks.)
- **The recall ceiling is the CLICKING stage, not masking** (proven in §9: the gate
  drops 0 real creatures; mask-gen fixes were a wash on recall). Future recall work
  should target the finder/tile MISSING dark, low-contrast, camouflaged creatures —
  e.g. finer tiling (3×3) on dark regions, temporal context on the whole-frame
  finder, or a dark-region-specific pass. Don't spend more on mask generation for
  recall.
- Wire `S3_cons_temp` tile config + centre-of-mass prompts + hybrid generator into
  production `nibi_model_compare/som_missed_creatures.py`; validate on the broader
  16-frame set incl. rank03.

## 12. Fable clicks & masks — per-frame renders

Representative single runs of the full Fable engine (`tile_stray_bakeoff.py diag
--model claude-fable-5`) on each benchmark frame. **Left = clicks vs GT** (green
outline = GT creature; yellow ✗ = click on a creature; magenta ✗ = stray click).
**Right = SAM3 masks vs GT** (3 panels: RAW · GT green-fill · SAM3 masks, green =
matched a creature, red = unmatched). Numbers below each frame are the repeats=3
e2e recall and the single-run diag detection count.

### cc_f042 — e2e recall 1.00 · diag 4/5 detected
![clicks](click_engine_assets/fable/f042_clicks.png)
![masks](click_engine_assets/fable/f042_masks.png)

### cc_f045 — e2e recall 1.00 · diag 5/5 (recovered the brittle-star Sonnet abandoned)
![clicks](click_engine_assets/fable/f045_clicks.png)
![masks](click_engine_assets/fable/f045_masks.png)

### cc_f054 — e2e recall 1.00 · diag 5/5 detected
![clicks](click_engine_assets/fable/f054_clicks.png)
![masks](click_engine_assets/fable/f054_masks.png)

### cc_f059 — e2e recall 0.93 · diag 5/5 (the hardest frame; baseline was 0.73)
![clicks](click_engine_assets/fable/f059_clicks.png)
![masks](click_engine_assets/fable/f059_masks.png)

### cc_f018 — e2e recall 0.93 · diag 5/5 detected
![clicks](click_engine_assets/fable/f018_clicks.png)
![masks](click_engine_assets/fable/f018_masks.png)

### cc_f024 — e2e recall 1.00 · diag 4/4 detected
![clicks](click_engine_assets/fable/f024_clicks.png)
![masks](click_engine_assets/fable/f024_masks.png)

Across all six, Fable's whole-frame finder clicks every GT creature (5/5 or 4/4
pre-gate) with only 1–2 stray clicks, and SAM3 masks them at IoU ~0.79–0.95; the
lone per-frame mask miss is a thin/small creature whose mask falls below IoU 0.5.

## 12b. 16-frame validation (generalization, incl. cross-clip rank03)

The Fable engine (hybrid masks + conservative post-mask verify) run on the full
`EVAL_MANIFEST` — 12 chinacreek (easy + hard) + 4 rank03, repeats=3 (`e2e --manifest`):

**OVERALL (15 effective frames): recall 0.977 · IoU(all) 0.868 · IoU(det) 0.889 ·
cand_new/pass 12.67 · FP/pass 2.67 (≈0.18 true FP per frame).**

- Recall held at **0.977** — identical to the 6-hard-frame result — across easy and
  hard chinacreek frames alike (new frames 0/6/12/30/37/50: recall 0.92–1.00, FP 0–0.67).
- **rank03 is a different video, and Fable transfers cleanly:** r03_f000/004/005/009
  all **recall 1.00, IoU(det) 0.96–0.97, FP 0.00** (single larger creatures mask very
  tightly). (r03_f011 has no GT ≥ 0.5 and is skipped.)
- The bulk of "strays" (12.67/pass) are candidate-new real creatures the GT missed,
  kept in `candidates/`; genuine false positives are ~0.18/frame.

This is the broad confirmation that the model swap + pipeline generalize beyond the
6 tuning frames and to an unseen clip.

## 13. Strays are mostly real creatures the GT missed (post-mask verify)

Inspecting Fable's unmatched ("stray") masks (`diag` dumps each to `<frame>/strays/`)
showed they are **not** substrate false positives: visible legs (f059), the same
reddish squat-lobster recurring across adjacent frames (f054≈f059), pale crabs with
limbs — none clearly bare gravel. The Sonnet first-pass GT simply never found them.

A **conservative Fable post-mask verify** (`e2e --verify-masks`) shows Fable each
produced mask and drops ONLY confident substrate (keeps anything plausibly animal);
unmatched-but-kept masks are logged as **candidate-new labels** in `<run>/candidates/`.
Fable + hybrid + verify, repeats=3:

| metric | value |
|---|---|
| recall | 0.977 |
| IoU(det) | 0.883 |
| **cand_new / pass** (real, GT-missed, kept) | **4.0** |
| **FP / pass** (confident substrate, dropped) | **1.33** |

So the old "strays/pass 5.0" was ~¾ real creatures: the **true false-positive rate is
~1.33/pass (≈0.22/frame)**, and the verify keeps the ~4/pass net-new real creatures as
candidate labels rather than discarding them. Recall is unchanged (verify never touches
matched masks). Caveat: GT is incomplete, so cand_new vs FP is Fable judging itself — a
human spot-check of `candidates/` crops is the gold-standard confirmation.

## Appendix — key files

- `scripts/click_engine_probe.py` — e2e harness, iterative clicker, tiled sweep, SAM3 mask generators.
- `scripts/tile_stray_bakeoff.py` — click-level bake-off + full-engine comparison.
  Commands: `setup`, `run --strategy`, `discover --video --sec [--model --review]`,
  `engine --model [--repeats --review]` (full pipeline scored vs GT).
- `scripts/motion_diff_probe.py` — the (negative) motion-difference feasibility probe.
- Bake-off outputs: `runs/click_probe/tile_bakeoff/<strategy>/summary.json`.
- Model-comparison outputs: `runs/click_probe/tile_bakeoff/engine/S3_cons_temp_<model>[_noreview]/summary.json`.
- rank01 discovery: `runs/click_probe/tile_bakeoff/discover/rank01_..._t5s_f150_S3_cons_temp_<model>/` (`discover.json`, `clicks.png`, `clicks_prereview.png`).
- Report figures: `docs/click_engine_assets/` (rank01 renders, Sonnet vs Opus).
