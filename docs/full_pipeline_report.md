# Full-Pipeline Run Report — SAM3+Sonnet first pass + Fable 5 missed-creature stage

_Generated 2026-06-11. Validation video: **rank03** (10 s ONC clip). First fan-out
target before running the other 10 s clips (rank01/02/04)._

## The pipeline (per clip)

1. **3 frames spaced evenly through the 10 s clip** — frames 3 / 7 / 11 of the
   15-frame stream (≈2 s / 4.7 s / 7.3 s: near beginning, middle, near end), each with
   **±2 s of temporal context** available to the MLLM clicker.
2. **Sonnet frame-quality screen** — drops corrupted / black / blurred frames before
   any expensive work. (Here all 3 passed; rank03 is clean.)
3. **First-pass masks = SAM3 + Sonnet text-agent** (reused from the existing agent
   run). These are the baseline "already found" creatures.
4. **Fable 5 missed-creature engine** — whole-frame finder (**technique B**) + tile
   sweep (**technique T**), conservative + temporal, with ±2 s neighbour context →
   SAM3 hybrid masks → conservative Fable post-mask verify.
5. **Match each Fable mask vs the first-pass masks**: `re-found` (first pass already
   had it, IoU ≥ 0.5), `cand_new` (first pass **missed** it — a candidate new label),
   or `FP` (Fable verify dropped it as substrate).
6. **Fable keep-decision** over the 3 frames (keep all / some / none by mask quality
   & redundancy).

**Render legend** (each `composite_fNNN.png`): first-pass Sonnet masks = **green
outline**; Fable new (`cand_new`) masks = **cyan fill**; clicks tagged **B** (yellow,
whole-frame finder) / **T** (magenta, tile sweep), `*` = re-found a first-pass
creature; dropped FPs = dim red.

## Results (rank03)

| Frame | Quality | First-pass (Sonnet) | Fable re-found | Fable NEW (missed) | FP | Kept? |
|---|---|---|---|---|---|---|
| f3 (≈2 s) | usable | 1 | 1 (B, IoU 0.96) | 1 (T) — **spurious dup** | 0 | dropped |
| f7 (≈4.7 s) | usable | 1 | 1 (B, IoU 0.96) | 0 | 0 | **kept** |
| f11 (≈7.3 s) | usable | **0** | 0 | **2 (B)** — genuine | 0 | **kept** |

### f11 — the headline: Fable recovers what the first pass missed entirely
![f11](click_engine_assets/fullpipe_r03/f011_2x.png)

The SAM3+Sonnet first pass found **nothing** on this frame. Fable's whole-frame finder
caught a **clear central fish** (well-masked, elongated body, area ~4100 px) plus a
**small bottom-right creature** (~450 px). Both are real animals the first pass
dropped — exactly the value the missed-creature stage is meant to add.

### f7 — clean corroboration
![f7](click_engine_assets/fullpipe_r03/f007_2x.png)

One first-pass creature (the central fish); Fable re-found it precisely (IoU 0.96 vs
the first-pass mask), no spurious extras, no FP. The mask is tight on the body.

### f3 — same fish, but a near-duplicate inflated "new"
![f3](click_engine_assets/fullpipe_r03/f003_2x.png)

Only **one** creature is present (the central fish). Fable's whole-frame click (B)
re-found it (IoU 0.96), but the **tile sweep placed a second click 31 px away on the
SAME fish**, and because the greedy match had already assigned the first-pass creature
to the B mask, the T mask was mislabeled `cand_new`. It is **not** a new creature — it
is a duplicate detection.

## My conclusions (from examining the images)

1. **The core value is real and visible.** On f11, where the SAM3+Sonnet first pass
   produced zero masks, Fable found and segmented a clear fish (and a small second
   animal). The same central fish recurs in all three frames; Fable caught it in
   **all three** (re-found in f3/f7, "new" in f11), whereas the first pass missed it in
   f11 — so **Fable is more consistent across the clip than the first pass**.

2. **Mask quality is good where the creature is compact.** Re-found masks hit IoU 0.96
   vs the first-pass masks; the f11 central fish mask covers the elongated body
   cleanly. The only weak mask is the tiny ~450 px bottom-right creature on f11 — at
   that size SAM3 masks are looser (consistent with earlier findings).

3. **Whole-frame finder (B) is doing the heavy lifting; tile (T) added nothing real
   here** — its one contribution (f3) was a duplicate. On this clip the creatures are
   isolated and not small/camouflaged, so the tile sweep's recall edge doesn't apply
   and it only risks duplicates. Tile's value showed on the hard chinacreek frames;
   on clean clips it's near-redundant.

4. **A fixable defect: near-duplicate clicks inflate `cand_new`.** The f3 case is the
   recurring B-vs-T ~30 px near-duplicate (also seen on rank01). The 30 px geometric
   dedup just misses it, and the "greedy match claims the first-pass creature for one
   mask" logic then mislabels the twin as new. **Fix:** (a) raise the dedup radius to
   ~40 px (or scale it to creature size), and (b) after first-pass matching, drop any
   unmatched Fable mask whose IoU with an already-matched Fable mask is high
   (same-creature double-detection). This would have made f3 correctly show 0 new.

5. **No false positives survived** the conservative Fable verify on any frame (FP = 0).
   Combined with the f3 over-count, the practical message is: the current `cand_new`
   number slightly **over-states** new finds (duplicates), not under-states — and
   contains no substrate FPs.

6. **The keep-decision behaved sensibly:** it dropped f3 (redundant central fish,
   already cleanly captured in f7) and kept f7 (clean re-find) and f11 (genuine new
   finds). That is the right call for a dataset — keep the informative, non-redundant
   frames.

7. **Quality screen unexercised here** — rank03's 3 frames are all clean, so the screen
   passed everything; it wasn't stress-tested against a corrupted frame in this clip.

## Recommended next steps

- **Apply the dedup fix** (raise to ~40 px + post-match same-creature merge) and re-run;
  expect f3 to drop its spurious "new".
- **Fan out to rank01 / rank02 / rank04** (the other 10 s clips). rank01's first pass
  found 0 creatures, so it will be a strong showcase of Fable-only recovery (like f11
  here). Each is an independent resumable run.
- For a **dense-30 fps** version (exact 2 s / 5 s / 8 s frames with 60-frame context),
  the SAM3+Sonnet first pass must be re-run on per-clip mini-clips (the slow path,
  ~2–5 min/frame); the subsampled stream used here gives the MLLM equivalent ~2 s
  context for far less compute.

## Artifacts
- Per-frame composites + per-target debug: `runs/full_pipeline/r03/` (`composite_fNNN.png`,
  `f0NN/` subdirs with clicks/tiles/verify crops, `keep_decision.txt`, `summary.json`).
- Report figures: `docs/click_engine_assets/fullpipe_r03/`.
- Orchestrator: `scripts/full_pipeline_run.py` (`--video r03 [--frames a,b,c]`).
