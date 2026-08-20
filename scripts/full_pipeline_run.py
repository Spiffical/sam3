"""Full production-style pipeline run + report.

Per video: pick 3 evenly-spaced frames -> Sonnet frame-quality screen (drop
corrupted) -> for each surviving frame: SAM3+Sonnet FIRST-PASS masks (reused from
the existing agent frame_outputs) as the baseline, then the FABLE missed-creature
engine (whole-frame finder = technique 'B', tile sweep = technique 'T') seeded
against the first-pass masks -> SAM3 hybrid masks -> conservative Fable post-mask
verify -> match each Fable mask vs the first-pass masks (matched = first-pass
creature re-found; cand_new = a creature the first pass MISSED, kept; fp = dropped
substrate) -> keep-decision over the 3 frames. Emits per-frame composite renders +
a JSON the report assembler turns into markdown.

Models: first-pass = Sonnet (reused); frame-quality screen = Sonnet (claude-sonnet-4-6);
missed-creature engine + post-mask verify = Fable (claude-fable-5).

Usage:
  python scripts/full_pipeline_run.py --video cc            # run one registered video
  python scripts/full_pipeline_run.py --video cc --frames 15,30,45
"""
from __future__ import annotations

import argparse
import json
import os

import cv2
import numpy as np

import scripts.click_engine_probe as P
import scripts.tile_stray_bakeoff as B
from scripts.click_engine_probe import (
    load_target_and_gt, read_video_frame, send_claude_request, _extract_answer_json,
    refine_group_mm_hybrid, verify_clicks, verify_masks, _iou,
)

OUT = "runs/full_pipeline"
SONNET = "claude-sonnet-4-6"
FABLE = "claude-fable-5"
CFG = B.STRATEGIES["S3_cons_temp"]   # conservative + temporal tile sweep

# Video registry (reuses existing Sonnet first-pass frame_outputs as the baseline).
_AEF = "runs/agent_every_frame"
_R01 = (_AEF + "/rank01_taxa001_ann001_"
        "AXISCAMACCC8E891285_20210707T220008.000Z_t00000/claude_smoke_v4")
_R02 = (_AEF + "/rank02_taxa001_ann001_"
        "AXISCAMACCC8E891285_20201121T000016.000Z_t00070/claude_smoke_v4")
_R03 = (_AEF + "/rank03_taxa001_ann001_"
        "AXISCAMACCC8E891285_20201113T121535.000Z_t00090/claude_smoke_v4")
_R04 = (_AEF + "/rank04_taxa001_ann001_"
        "AXISCAMACCC8E891285_20210713T100008.000Z_t00100/claude_smoke")
# Each ONC clip is 10s. First pass subsamples it; offsets give the clicker ~2s of
# temporal context: 15-frame clips ~0.67s/frame (offset 3 ~= 2s); the 60-frame clip
# ~0.167s/frame (offset 12 ~= 2s). frames sit near begin/mid/end with 2s on each side.
def _entry(base):
    return {"frame_outputs": base + "/frame_outputs_rle.json", "video": None,
            "frames_dir": base + "/frame_inputs", "fps": None}
VIDEOS = {
    "r01": {**_entry(_R01), "n_total": 15, "frames": [3, 7, 11], "offsets": [3, 2, 1]},
    "r02": {**_entry(_R02), "n_total": 15, "frames": [3, 7, 11], "offsets": [3, 2, 1]},
    "r03": {**_entry(_R03), "n_total": 15, "frames": [3, 7, 11], "offsets": [3, 2, 1]},
    "r04": {**_entry(_R04), "n_total": 60, "frames": [12, 30, 48], "offsets": [12, 6, 3]},
}


def _set_src(v):
    P.SRC.update(video=v["video"], frames_dir=v["frames_dir"],
                 frame_outputs=v["frame_outputs"], fps=v["fps"])


def _api_failed(sdir):
    """True if any MLLM response in this frame's dir is '<none>' (the client wrote
    that after exhausting retries on a connection error) -- so we don't cache a
    frame whose results are corrupted by a dropped connection."""
    import glob
    for tp in glob.glob(os.path.join(sdir, "**", "*.txt"), recursive=True):
        try:
            if "<none>" in open(tp).read():
                return True
        except OSError:
            pass
    return False


def quality_screen(frame, sdir, tag):
    """Sonnet frame-quality screen: is this frame usable (not corrupted / black /
    severely blurred) to run creature detection on? Returns (ok: bool, reason)."""
    p = os.path.join(sdir, f"qc_{tag}.png")
    cv2.imwrite(p, frame)
    text = (
        "This is a single frame from an underwater seafloor monitoring video, to be "
        "used for creature detection. Judge ONLY technical usability: is the frame "
        "corrupted, all-black/all-grey, severely motion-blurred, or otherwise "
        "unusable? A normal (even murky or low-contrast) seafloor view IS usable.\n\n"
        'Output brief reasoning then EXACTLY ONE tag:\n'
        '<answer>{"usable": true}</answer>  -- frame is technically fine to analyse\n'
        '<answer>{"usable": false, "reason": "<short>"}</answer>  -- corrupted/unusable')
    r = send_claude_request([{"role": "user", "content": [
        {"type": "image", "image": p}, {"type": "text", "text": text}]}],
        model=SONNET, max_tokens=400)
    open(os.path.join(sdir, f"qc_{tag}.txt"), "w").write(r or "<none>")
    ans = _extract_answer_json(r) or {}
    ok = ans.get("usable") is not False        # default usable unless explicit false
    return ok, ans.get("reason", "")


def process_frame(v, tgt, sdir, service, offsets):
    """First-pass(Sonnet, reused) baseline + Fable missed-engine + SAM3 masks +
    verify + match-vs-first-pass. Returns a structured result dict."""
    _set_src(v)
    H, W, firstpass = load_target_and_gt(tgt, min_prob=0.5)   # Sonnet first-pass masks
    frame = read_video_frame(tgt)
    P.MODEL = FABLE
    # Fable engine: whole-frame finder (B) + tile sweep (T) + dedup + review.
    _bf, _ta, groups, meta = B._engine_on_frame(frame, tgt, W, H, CFG, sdir,
                                                v["fps"], review=False, offsets=offsets)
    tpath = os.path.join(sdir, "target.png")
    # per-click zoomed creature gate
    groups, n_gate = verify_clicks(groups, frame, W, H, sdir)
    # SAM3 hybrid masks, carrying the click + technique (source) provenance
    results = []
    for g in groups:
        res, _tr = refine_group_mm_hybrid(service, g, tpath, frame, W, H, sdir,
                                          max_clicks=5, max_attempts=3)
        res["source"] = g.get("source", "B")
        res["click"] = g["clicks"][0]
        results.append(res)
    # conservative Fable post-mask verify -> kept vs confident-substrate FP
    kept, dropped = verify_masks(results, frame, W, H, sdir, model=FABLE)
    # mask-level NMS: drop a Fable mask that overlaps an already-accepted Fable mask
    # (same-creature double-detection, e.g. a tile click 31px from a whole-frame
    # click on one fish) so it can't be mislabeled "new". Order = B (whole-frame)
    # before T (tile), so the whole-frame detection is preferred.
    nms_kept = []
    for r in kept:
        m = np.asarray(r["mask"]).astype(bool)
        if m.any() and any(_iou(m, np.asarray(k["mask"]).astype(bool)) >= 0.5
                           for k in nms_kept):
            continue
        nms_kept.append(r)
    kept = nms_kept
    # match kept Fable masks vs first-pass masks (greedy IoU >= 0.5)
    masks = [np.asarray(r["mask"]).astype(bool) for r in kept]
    pairs = []
    for ki, m in enumerate(masks):
        if not m.any():
            continue
        for gi, fp in enumerate(firstpass):
            pairs.append((_iou(m, fp["mask"]), ki, gi))
    taken_k, taken_g, match = set(), set(), {}
    for iou, ki, gi in sorted(pairs, reverse=True):
        if iou < 0.5 or ki in taken_k or gi in taken_g:
            continue
        taken_k.add(ki); taken_g.add(gi); match[ki] = (gi, iou)
    items = []
    for ki, r in enumerate(kept):
        m = masks[ki]
        if not m.any():
            continue
        st = "matched" if ki in match else "cand_new"
        items.append({"source": r["source"], "click": r["click"],
                      "area": int(m.sum()), "status": st,
                      "confidence": round(float(r.get("creature_confidence", 0.0)), 2),
                      "iou_firstpass": round(match[ki][1], 3) if ki in match else 0.0,
                      "mask": m})
    fp_items = [{"source": r.get("source", "B"), "click": r.get("click"),
                 "mask": np.asarray(r["mask"]).astype(bool)}
                for r in dropped if np.asarray(r["mask"]).any()]
    return {"tgt": tgt, "W": W, "H": H, "frame": frame, "firstpass": firstpass,
            "items": items, "fp": fp_items, "meta": meta, "n_gate": n_gate,
            "n_firstpass": len(firstpass),
            "n_matched": sum(1 for it in items if it["status"] == "matched"),
            "n_candnew": sum(1 for it in items if it["status"] == "cand_new"),
            "n_fp": len(fp_items)}


def _draw_click(img, click, W, H, color, label):
    p = (int(click["x"] * W), int(click["y"] * H))
    cv2.drawMarker(img, p, (0, 0, 0), cv2.MARKER_TILTED_CROSS, 24, 5)
    cv2.drawMarker(img, p, color, cv2.MARKER_TILTED_CROSS, 22, 2)
    cv2.putText(img, label, (p[0] + 11, p[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.55,
                (0, 0, 0), 4, cv2.LINE_AA)
    cv2.putText(img, label, (p[0] + 11, p[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.55,
                color, 1, cv2.LINE_AA)


def render_composite(res, path):
    """One render per frame: first-pass(Sonnet) masks green-outline; Fable cand_new
    masks cyan-fill + click tagged by technique (B yellow / T magenta); FP dim red."""
    out = res["frame"].copy()
    W, H = res["W"], res["H"]
    for fp in res["firstpass"]:                      # first-pass Sonnet masks
        cnts, _ = cv2.findContours(fp["mask"].astype(np.uint8), cv2.RETR_EXTERNAL,
                                   cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(out, cnts, -1, (0, 220, 0), 2)
    for it in res["items"]:                          # Fable masks
        m = it["mask"]
        if it["status"] == "cand_new":
            ov = out.copy(); ov[m] = (255, 255, 0)
            out = cv2.addWeighted(ov, 0.4, out, 0.6, 0)
            cnts, _ = cv2.findContours(m.astype(np.uint8), cv2.RETR_EXTERNAL,
                                       cv2.CHAIN_APPROX_SIMPLE)
            cv2.drawContours(out, cnts, -1, (255, 255, 0), 2)
        col = (0, 255, 255) if it["source"] == "B" else (255, 0, 255)
        tag = (it["source"] + ("*" if it["status"] == "matched" else "")
               + f" p={it.get('confidence', 0.0):.2f}")
        _draw_click(out, it["click"], W, H, col, tag)
    for fp in res["fp"]:                             # dropped FPs (dim red outline)
        cnts, _ = cv2.findContours(fp["mask"].astype(np.uint8), cv2.RETR_EXTERNAL,
                                   cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(out, cnts, -1, (0, 0, 200), 1)
    cv2.rectangle(out, (0, 0), (W, 46), (0, 0, 0), -1)
    cv2.putText(out, f"f{res['tgt']}  first-pass(green)={res['n_firstpass']}  "
                f"Fable new(cyan)={res['n_candnew']}  re-found(*)={res['n_matched']}  "
                f"FP(red)={res['n_fp']}", (6, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                (255, 255, 255), 1, cv2.LINE_AA)
    cv2.putText(out, "click tag: B=whole-frame  T=tile  *=re-found first-pass  "
                "p=Fable creature-confidence(0-1)", (6, 38), cv2.FONT_HERSHEY_SIMPLEX,
                0.45, (200, 200, 200), 1, cv2.LINE_AA)
    cv2.imwrite(path, out)


def keep_decision(res_list, sdir):
    """Fable judges, across the processed frames, which to KEEP for the dataset
    based on overall mask quality / creature content. Returns dict frame->keep."""
    # render a small montage of the composites for the judge
    imgs = [os.path.join(sdir, f"composite_f{r['tgt']:03d}.png") for r in res_list]
    listing = "; ".join(f"frame {r['tgt']}: {r['n_firstpass']} first-pass + "
                        f"{r['n_candnew']} new Fable creatures" for r in res_list)
    text = (
        f"These are {len(res_list)} analysed frames from one short seafloor clip, each "
        "showing segmentation masks of creatures (green = first-pass, cyan = newly "
        f"found). Summary: {listing}.\n"
        "For building a labelled dataset we keep the frames whose masks are good "
        "quality and informative; we may keep all, some, or none. For EACH frame "
        "decide keep or drop (drop only if masks are clearly poor / frame adds nothing "
        "beyond the others / redundant).\n\n"
        'Output reasoning then EXACTLY ONE tag:\n'
        '<answer>{"keep": [<frame indices to keep>]}</answer>')
    content = [{"type": "text", "text": text}]
    for p in imgs:
        content.append({"type": "image", "image": p})
    r = send_claude_request([{"role": "user", "content": content}], model=FABLE,
                            max_tokens=800)
    open(os.path.join(sdir, "keep_decision.txt"), "w").write(r or "<none>")
    ans = _extract_answer_json(r) or {}
    keep = set(ans.get("keep", [r["tgt"] for r in res_list]))
    return {r["tgt"]: (r["tgt"] in keep) for r in res_list}


def run_video(vkey, frames=None):
    v = VIDEOS[vkey]
    _set_src(v)
    frames = frames or v["frames"]
    # ~2s temporal context for the clicker (per-clip, see registry comment).
    offsets = v.get("offsets") or ([3, 2, 1] if v["frames_dir"] else [60, 30])
    sroot = os.path.join(OUT, vkey)
    os.makedirs(sroot, exist_ok=True)
    service = P.build_sam3_service()
    print(f"=== full pipeline video={vkey} frames={frames} ===", flush=True)

    # Stage 0: Sonnet frame-quality screen
    screened = []
    for tgt in frames:
        _set_src(v)
        frame = read_video_frame(tgt)
        ok, reason = quality_screen(frame, sroot, f"{tgt:03d}")
        print(f"  [qc] f{tgt}: {'USABLE' if ok else 'DROP ('+reason+')'}", flush=True)
        if ok:
            screened.append(tgt)
    if not screened:
        print("  all frames failed quality screen; nothing to do", flush=True)
        return

    # Stage 1-3: per surviving frame (resumable; a frame whose responses show API
    # failures is NOT cached, so a re-run after a dropped connection retries it).
    light_list = []
    for tgt in screened:
        sdir = os.path.join(sroot, f"f{tgt:03d}")
        rpath = os.path.join(sdir, "result_light.json")
        comp = os.path.join(sroot, f"composite_f{tgt:03d}.png")
        if os.path.exists(rpath) and os.path.exists(comp):
            light_list.append(json.load(open(rpath)))
            print(f"  [frame f{tgt}] CACHED (skip)", flush=True)
            continue
        os.makedirs(sdir, exist_ok=True)
        res = process_frame(v, tgt, sdir, service, offsets)
        render_composite(res, comp)
        light = {"tgt": res["tgt"], "n_firstpass": res["n_firstpass"],
                 "n_matched": res["n_matched"], "n_candnew": res["n_candnew"],
                 "n_fp": res["n_fp"], "composite": f"composite_f{tgt:03d}.png",
                 "items": [{"source": it["source"], "status": it["status"],
                            "area": it["area"], "confidence": it.get("confidence"),
                            "iou_firstpass": it["iou_firstpass"],
                            "x": round(it["click"]["x"], 4),
                            "y": round(it["click"]["y"], 4)} for it in res["items"]]}
        if _api_failed(sdir):
            print(f"  [frame f{tgt}] !! API failures (<none> responses) -- NOT caching;"
                  f" re-run to retry this frame", flush=True)
        else:
            json.dump(light, open(rpath, "w"), indent=2)
        confs = ", ".join(f"{i['status'][:4]}/{i['source']} p={i['confidence']:.2f}"
                          for i in res["items"])
        print(f"  [frame f{tgt}] first-pass {res['n_firstpass']}, Fable re-found "
              f"{res['n_matched']}, Fable NEW {res['n_candnew']}, FP dropped "
              f"{res['n_fp']}  | {confs}", flush=True)
        light_list.append(light)

    # Stage 4: keep-decision
    keep = keep_decision(light_list, sroot)
    print(f"  [keep] {keep}", flush=True)
    summary = {"video": vkey, "frames_requested": frames, "screened": screened,
               "keep": keep, "frames_detail": light_list}
    json.dump(summary, open(os.path.join(sroot, "summary.json"), "w"), indent=2)
    print(f"  wrote {sroot}/summary.json + composites", flush=True)
    return summary


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--video", required=True, choices=list(VIDEOS))
    ap.add_argument("--frames", default=None,
                    help="comma-separated frame indices (override registry default)")
    a = ap.parse_args()
    fr = [int(x) for x in a.frames.split(",")] if a.frames else None
    run_video(a.video, fr)
