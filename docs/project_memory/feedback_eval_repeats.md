---
name: feedback-eval-repeats
description: Always use repeats>=3 when A/B-testing MLLM-in-the-loop pipelines; single-run numbers are misleading in both directions
metadata: 
  node_type: memory
  type: feedback
  originSessionId: a369fe76-c337-43bd-9d51-52194af1ac35
---

When benchmarking MLLM-in-the-loop pipelines (e.g. the click-engine / SAM3 eval harness in `scripts/click_engine_probe.py`), run each frame **N>=3 times and report mean±std**, never single-run point estimates.

**Why:** During the click-system work, single-run end-to-end evals produced conclusions that were wrong in *both* directions. A single lucky verify_loop run scored recall 0.82, which made the multi-pass "iterative" clicker look like it added nothing; at repeats=3 verify_loop actually averaged 0.56 and iterative 0.81 (a real +0.25 gain). Separately, comparing a 3× run against a 1× run made the per-click verify pass look like it hurt recall when it was actually neutral. Per-frame recall std was ~0.09–0.19 — larger than many of the deltas being judged. The user explicitly asked for N-repeat averaging after we got burned by noise.

A second, even starker instance later: a repeats=1 screen reported the tiled clicker taking cc_f054 from recall **0.40 → 1.00** (+0.60). It looked like a blockbuster win. At repeats=3 the *baseline* on cc_f054 was actually **0.87±0.09** — the 0.40 was a single unlucky run — and tiled was 0.93, a real gain of only +0.06 (inside noise). The genuine tiling gain lived on entirely different frames (cc_f042 0.60→1.00, cc_f024 0.83→1.00). Trusting the screen would have credited the right idea for the wrong reason and badly mis-ranked which frames it helps.

**How to apply:** Any time you compare two configs of an MLLM pipeline, use matched repeat counts and look at the spread before believing a delta. `--eval-e2e --repeats N` in the click probe does this. Treat a single-run delta smaller than ~1 std as noise, not signal. Repeats=1 is fine ONLY as a crash/plumbing screen — never read per-frame deltas off it.
