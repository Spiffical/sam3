"""Parallel bake-off of tile-sweep prompt / temporal strategies to cut strays
while keeping recall. CLICK-LEVEL ONLY -- no SAM3, so every strategy is a pure
MLLM job that runs as its own OS process in parallel (no GPU contention).

Why click-level: in the e2e pipeline the extra strays that tiling introduces are
born in the tile PROPOSAL and survive the GATE -- both pure-MLLM stages. SAM3 only
turns surviving clicks into masks. So measuring "does each placed click land on a
GT creature?" isolates exactly the lever we are tuning. The absolute stray counts
here run a bit higher than e2e (SAM3 abandon + IoU matching remove a few more
downstream), but the RANKING across strategies is what we trust; the winner then
gets confirmed in the full e2e (with SAM3) at repeats=3.

Paired design: `setup` runs the iterative base pass (NO tiling) repeats x per frame
and caches the resulting click sets to base_found.json. Each strategy `run` starts
from the IDENTICAL base set for that (frame, repeat) and only adds its tile sweep,
so the strategy-to-strategy delta is the tile mechanism alone -- base-pass variance
is paired out.

Scope of the "temporal" knob: neighbour-frame context (the SAME crop region ~0.5s
before/after) is added to the two EXISTENCE-deciding stages -- the tile proposal and
the creature gate(s). The click-CENTERING refine loop (_verify_refine_one) is reused
unchanged (still-frame); it decides precision, not existence. A follow-up can add
temporal context there too if this round shows temporal helps.

Usage:
  python scripts/tile_stray_bakeoff.py setup            # once: cache base clicks
  python scripts/tile_stray_bakeoff.py run --strategy S0_control
  ... (run each strategy as a separate process, in parallel)
"""
from __future__ import annotations

import argparse
import json
import os

import cv2
import numpy as np

import scripts.click_engine_probe as P
from scripts.click_engine_probe import (
    ANSWER_FMT, MODEL, TASK_CORE, VERIFY_CLICK_FMT, _extract_answer_json,
    _mask_crop_geom, _parse_coarse, _render_mask_crop, _tile_boxes,
    _verify_refine_one, load_target_and_gt, read_video_frame, score_clicks,
    send_claude_request,
)

# All bake-off frames are chinacreek (video source, 30fps) -- the 6 hard frames
# the tuned-tiled e2e A/B already characterised, so numbers line up.
FRAMES = [42, 45, 54, 59, 18, 24]
OUT = "runs/click_probe/tile_bakeoff"
BASE_JSON = os.path.join(OUT, "base_found.json")
DEDUP_PX = 25
UPSCALE = 2          # tile proposal upscale (matches production _tile_sweep)
GATE_UPSCALE = 6     # gate crop upscale (matches production verify_clicks)

STRATEGIES = {
    "base":         dict(tile=False, prompt=None,   temporal=False),
    "S0_control":   dict(tile=True,  prompt="wide", temporal=False),
    "S1_cons":      dict(tile=True,  prompt="cons", temporal=False),
    "S2_temporal":  dict(tile=True,  prompt="wide", temporal=True),
    "S3_cons_temp": dict(tile=True,  prompt="cons", temporal=True),
}

_COM = (" Put each click on the creature's CENTER OF MASS -- the thickest, most "
        "central part of its body -- not a fin, tail, leg, antenna, or edge.")

WIDE_PROMPT = (
    " This image is a ZOOMED, high-resolution CROP of a larger seafloor frame -- "
    "it shows only part of the scene. Click each distinct LIVING ANIMAL visible "
    "in THIS crop, including small or well-camouflaged ones. Do NOT click bare "
    "gravel, loose shells or shell fragments, rocks, sticks, sediment texture, or "
    "shadows -- those are not animals. Give one click per animal." + _COM)

CONS_PROMPT = (
    " This image is a ZOOMED, high-resolution CROP of a larger seafloor frame -- "
    "it shows only PART of the scene. MOST such crops contain NO animal at all: "
    "just gravel, shell fragments, rock, sediment texture and shadows. Returning "
    "an EMPTY list is the correct, expected answer for the majority of crops. "
    "Click a point ONLY when you are genuinely confident a discrete living animal "
    "is present there; when in ANY doubt, do NOT click. Never click substrate, "
    "shells, rocks, sticks, sediment texture or shadows. At most one click per "
    "animal you are confident about." + _COM)

TEMPORAL_ADD = (
    " The extra image(s) show the SAME crop region about half a second before "
    "and/or after this moment. A real animal is present and consistent across "
    "these times (it may shift or move slightly); a shape that looks animal-like "
    "in only ONE frame, or that is just fixed background texture, is NOT an "
    "animal. Use this across-time check to avoid clicking substrate.")


def _set_cc_source():
    P.SRC["video"] = P.VIDEO
    P.SRC["frames_dir"] = None
    P.SRC["frame_outputs"] = P.FRAME_OUTPUTS
    P.SRC["fps"] = 30.0


def _get_nbs(tgt, n=2, offsets=(15, 30, 45)):
    """Return up to n (label, frame) temporal neighbours: one earlier + one later
    when available, falling back to extra earlier frames near the clip's end.
    ``offsets`` are frame deltas (video: 15/30/45 ~ ±0.5/1/1.5s; frames_dir: 1/2/3)."""
    offsets = tuple(offsets)
    out = []
    for off in offsets:                # nearest valid earlier
        if tgt - off >= 0:
            try:
                out.append((f"t-{off}", read_video_frame(tgt - off)))
                break
            except SystemExit:
                pass
    for off in offsets:                # nearest valid later
        try:
            f = read_video_frame(tgt + off)
            out.append((f"t+{off}", f))
            break
        except SystemExit:
            continue
    if len(out) < n:                   # top up with more earlier frames
        for off in offsets[1:]:
            if tgt - off >= 0:
                try:
                    out.append((f"t-{off}", read_video_frame(tgt - off)))
                except SystemExit:
                    pass
            if len(out) >= n:
                break
    return out[:n]


def _crop_up(img, box, upscale):
    l, t, r, b = box
    crop = img[t:b, l:r]
    return cv2.resize(crop, ((r - l) * upscale, (b - t) * upscale),
                      interpolation=cv2.INTER_CUBIC)


# ---------------------------------------------------------------------------
# setup: cache the paired base-pass click sets
# ---------------------------------------------------------------------------
def cmd_setup(repeats):
    _set_cc_source()
    os.makedirs(OUT, exist_ok=True)
    base = {}
    for tgt in FRAMES:
        H, W, gt = load_target_and_gt(tgt, min_prob=0.5)
        if not gt:
            print(f"  [setup] f{tgt}: no GT >=0.5, skipping", flush=True)
            continue
        frame = read_video_frame(tgt)
        per_rep = []
        for ri in range(repeats):
            sdir = os.path.join(OUT, "base", f"f{tgt:03d}_r{ri}")
            os.makedirs(sdir, exist_ok=True)
            tpath = os.path.join(sdir, "target.png")
            cv2.imwrite(tpath, frame)
            nbs_ctx = P.extract_neighbours(tgt, P.DEFAULT_OFFSETS, sdir, fps=30.0)
            score = P.run_iterative(tpath, frame, gt, nbs_ctx, W, H, sdir,
                                    refine="verify", label="base",
                                    tile_sweep=False)
            found = [{"x": g["clicks"][0]["x"], "y": g["clicks"][0]["y"],
                      "description": g.get("description", "")}
                     for g in score["groups"]]
            per_rep.append(found)
            print(f"  [setup] f{tgt} r{ri}: base found {len(found)} clicks "
                  f"(coverage {score['n_hit']}/{score['n_gt']})", flush=True)
        base[str(tgt)] = per_rep
    json.dump(base, open(BASE_JSON, "w"), indent=2)
    print(f"[setup] wrote {BASE_JSON}", flush=True)


# ---------------------------------------------------------------------------
# strategy stages
# ---------------------------------------------------------------------------
def _tile_proposal(frame, nbs, box, prompt_mode, temporal, tdir, ti):
    """One tile: ask the MLLM for creature clicks in the upscaled crop (optionally
    with same-region neighbour crops). Returns full-frame normalized candidates."""
    l, t, r, b = box
    tw, th = r - l, b - t
    W, H = frame.shape[1], frame.shape[0]
    zp = os.path.join(tdir, f"tile{ti}.png")
    cv2.imwrite(zp, _crop_up(frame, box, UPSCALE))
    imgs = [zp]
    if temporal:
        for lbl, nf in nbs:
            np_ = os.path.join(tdir, f"tile{ti}_{lbl}.png")
            cv2.imwrite(np_, _crop_up(nf, box, UPSCALE))
            imgs.append(np_)
    body = WIDE_PROMPT if prompt_mode == "wide" else CONS_PROMPT
    text = TASK_CORE + body + (TEMPORAL_ADD if temporal else "") + "\n\n" + ANSWER_FMT
    content = [{"type": "image", "image": imgs[0]}, {"type": "text", "text": text}]
    content += [{"type": "image", "image": p} for p in imgs[1:]]
    resp = send_claude_request([{"role": "user", "content": content}],
                               model=P.MODEL, max_tokens=1500)
    open(os.path.join(tdir, f"tile{ti}_resp.txt"), "w").write(resp or "<none>")
    out = []
    for cc in _parse_coarse(resp):
        out.append({"x": (l + cc["x"] * tw) / W, "y": (t + cc["y"] * th) / H,
                    "description": cc["description"]})
    return out


def _gate(groups, frame, nbs, W, H, sdir, strict, temporal, tag_prefix):
    """Creature gate. temporal=False delegates to the production still-frame
    verify_clicks; temporal=True shows the same click-centred crop across
    neighbour times and asks for an across-time creature judgement.

    strict=True keeps ONLY explicit creature:true (tile candidates);
    strict=False keeps unless explicit creature:false (high-prior union pass)."""
    if not temporal:
        return P.verify_clicks(groups, frame, W, H, sdir, strict=strict,
                               tag_prefix=tag_prefix)
    empty = np.zeros((H, W), bool)
    kept, dropped = [], 0
    for g in groups:
        tag = f"{tag_prefix}_id{g['id']}"
        geom = _mask_crop_geom(empty, g["clicks"], W, H, 0.16)
        cpath = os.path.join(sdir, f"{tag}.png")
        _render_mask_crop(frame, empty, g["clicks"], geom, cpath, GATE_UPSCALE)
        imgs = [cpath]
        for lbl, nf in nbs:
            npath = os.path.join(sdir, f"{tag}_{lbl}.png")
            _render_mask_crop(nf, empty, [], geom, npath, GATE_UPSCALE)  # no dot
            imgs.append(npath)
        desc = g.get("description", "the marked point")
        text = (
            "This is a zoomed crop of a seafloor video frame. A green dot marks one "
            f"point (proposed creature: '{desc}'). The other image(s) show the SAME "
            "region about half a second before/after (no dot, same pixel location). "
            "A real animal is present and consistent across these times (it may have "
            "shifted slightly off the dot); substrate, shell, gravel, sediment "
            "texture or shadow is not. Considering all the frames together, is the "
            "marked point on an actual creature?\n\n" + VERIFY_CLICK_FMT)
        content = [{"type": "image", "image": imgs[0]}, {"type": "text", "text": text}]
        content += [{"type": "image", "image": p} for p in imgs[1:]]
        resp = send_claude_request([{"role": "user", "content": content}],
                                   model=P.MODEL, max_tokens=600)
        open(os.path.join(sdir, f"{tag}.txt"), "w").write(resp or "<none>")
        ans = _extract_answer_json(resp) or {}
        drop = (ans.get("creature") is not True) if strict else (ans.get("creature") is False)
        if drop:
            dropped += 1
            continue
        kept.append(g)
    return kept, dropped


def _dup(x, y, ref, W, H):
    return any(abs(x - f["x"]) * W < DEDUP_PX and abs(y - f["y"]) * H < DEDUP_PX
               for f in ref)


def _geom_dedup(groups, W, H, px=40):
    """Merge clicks within ``px`` of each other (keep the first, drop the rest).
    Deterministic duplicate remover -- catches two markers on the same animal
    (e.g. rank01 B1/T1, ~26 px apart) without the recall risk of the MLLM review."""
    kept, removed = [], 0
    for g in groups:
        c = g["clicks"][0]
        if any(abs(c["x"] - k["clicks"][0]["x"]) * W < px and
               abs(c["y"] - k["clicks"][0]["y"]) * H < px for k in kept):
            removed += 1
            continue
        kept.append(g)
    for i, g in enumerate(kept, 1):
        g["id"] = i
    return kept, removed


def tile_sweep_cfg(cfg, frame, nbs, W, H, base_found, tdir):
    """Configurable tile sweep: proposal (wide/conservative, optional temporal) ->
    dedup vs base -> still-frame centering refine -> strict creature gate. Returns
    (added_clicks, n_raw_candidates, n_gate_dropped)."""
    os.makedirs(tdir, exist_ok=True)
    cand = []
    for ti, box in enumerate(_tile_boxes(W, H, 2, 0.15)):
        cand += _tile_proposal(frame, nbs, box, cfg["prompt"], cfg["temporal"],
                               tdir, ti)
    added = []
    for cc in cand:
        if _dup(cc["x"], cc["y"], base_found + added, W, H):
            continue
        pt = _verify_refine_one(frame, cc, W, H, tdir, f"r{len(added)}")
        if pt is None:
            continue
        fx, fy = pt
        if _dup(fx, fy, base_found + added, W, H):
            continue
        added.append({"x": fx, "y": fy, "description": cc["description"]})
    gated = 0
    if added:
        cg = [{"id": i + 1, "description": a["description"],
               "clicks": [{"x": a["x"], "y": a["y"], "label": 1}]}
              for i, a in enumerate(added)]
        kept, gated = _gate(cg, frame, nbs, W, H, tdir, strict=True,
                            temporal=cfg["temporal"], tag_prefix="tile_gate")
        keep_xy = {(g["clicks"][0]["x"], g["clicks"][0]["y"]) for g in kept}
        added = [a for a in added if (a["x"], a["y"]) in keep_xy]
    return added, len(cand), gated


def _run_one(cfg, frame, nbs, W, H, gt, base_found, sdir):
    """Run a single (frame, repeat). Returns metrics dict."""
    if cfg["tile"]:
        added, n_cand, gated = tile_sweep_cfg(cfg, frame, nbs, W, H, base_found,
                                              os.path.join(sdir, "tiles"))
    else:
        added, n_cand, gated = [], 0, 0

    # union of base + tile clicks, then the production non-strict verify pass
    placed = ([{"id": i + 1, "description": f["description"],
                "clicks": [{"x": f["x"], "y": f["y"], "label": 1}]}
               for i, f in enumerate(base_found + added)])
    placed2, ndrop = _gate(placed, frame, nbs, W, H, sdir, strict=False,
                           temporal=cfg["temporal"], tag_prefix="verify")
    score = score_clicks(placed2, gt, W, H)
    return {"n_gt": score["n_gt"], "n_hit": score["n_hit"],
            "n_stray": score["n_stray"], "n_added": len(added),
            "n_cand": n_cand, "n_tile_gated": gated, "n_verify_drop": ndrop}


REVIEW_FMT = (
    'Output: brief reasoning then EXACTLY ONE trailing tag:\n'
    '<answer>{"decisions":[{"id":1,"action":"keep"},'
    '{"id":2,"action":"remove_duplicate_of","other":1},'
    '{"id":3,"action":"move","x":<float>,"y":<float>}]}</answer>\n'
    'Include EVERY marker id exactly once. Use "remove_duplicate_of" ONLY when this '
    'marker points at the SAME individual animal as another marker (give that '
    'marker id in "other"). For "move", x,y are NORMALIZED [0,1] full-frame '
    'coordinates of the corrected centre of mass.')


def _render_numbered(frame, groups, path):
    """Full frame with each click drawn as its id number (for the review pass)."""
    out = frame.copy()
    H, W = out.shape[:2]
    for g in groups:
        c = g["clicks"][0]
        p = (int(c["x"] * W), int(c["y"] * H))
        cv2.drawMarker(out, p, (0, 0, 0), cv2.MARKER_TILTED_CROSS, 26, 5)
        cv2.drawMarker(out, p, (0, 255, 255), cv2.MARKER_TILTED_CROSS, 24, 2)
        cv2.circle(out, p, 15, (0, 255, 255), 2)
        cv2.putText(out, str(g["id"]), (p[0] + 13, p[1] - 11),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 5, cv2.LINE_AA)
        cv2.putText(out, str(g["id"]), (p[0] + 13, p[1] - 11),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2, cv2.LINE_AA)
    cv2.imwrite(path, out)


def global_review(groups, frame, nbs, W, H, sdir, temporal, tag="review"):
    """One full-frame pass over ALL placed clicks at once. The MLLM sees every
    numbered marker together and decides per marker: keep / remove (substrate or
    duplicate) / move (onto the creature's center of mass). Returns
    (new_groups, n_removed, n_moved). Seeing all clicks globally is what catches
    duplicate pairs and off-body clicks that per-click crops cannot."""
    if not groups:
        return groups, 0, 0
    for i, g in enumerate(groups, 1):
        g["id"] = i
    npath = os.path.join(sdir, f"{tag}_input.png")
    _render_numbered(frame, groups, npath)
    imgs = [npath]
    tnote = ""
    if temporal and nbs:
        for lbl, nf in nbs:
            p = os.path.join(sdir, f"{tag}_{lbl}.png")
            cv2.imwrite(p, nf)
            imgs.append(p)
        tnote = (" The extra image(s) show the SAME scene about half a second "
                 "before/after (no markers); a real animal is consistent across "
                 "these times, substrate is not.")
    listing = "; ".join(f"#{g['id']}={g.get('description', '')[:50]}" for g in groups)
    text = (
        "This seafloor frame has candidate creature clicks, each drawn as a "
        f"numbered yellow marker. The markers are: {listing}. Every marker has "
        "ALREADY been individually verified at high resolution as sitting on an "
        "animal, so do NOT remove a marker just because it looks faint or "
        "substrate-like at this zoom -- assume it is a real animal. Your ONLY two "
        "jobs, looking at all markers together:\n"
        "1. DUPLICATES: if two or more markers point at the SAME individual "
        "animal, keep the best-placed one and mark the others "
        "'remove_duplicate_of' (cite the marker kept). Markers on DIFFERENT "
        "nearby animals are NOT duplicates -- keep them all.\n"
        "2. CENTRE OF MASS: if a marker sits on a real animal but off-centre "
        "(on a fin/tail/leg/edge or beside the body), MOVE it onto that animal's "
        "centre of mass (thickest central body). Otherwise KEEP it as-is.\n"
        "When unsure, KEEP. Do not remove anything except a genuine duplicate."
        + tnote + "\n\n" + REVIEW_FMT)
    content = [{"type": "image", "image": imgs[0]}, {"type": "text", "text": text}]
    content += [{"type": "image", "image": p} for p in imgs[1:]]
    resp = send_claude_request([{"role": "user", "content": content}],
                               model=P.MODEL, max_tokens=1500)
    open(os.path.join(sdir, f"{tag}.txt"), "w").write(resp or "<none>")
    dec = _extract_answer_json(resp) or {}
    decisions = {d["id"]: d for d in dec.get("decisions", []) if isinstance(d, dict)
                 and "id" in d}
    new, removed, moved = [], 0, 0
    for g in groups:
        d = decisions.get(g["id"])
        act = (d or {}).get("action")
        if act in ("remove_duplicate_of", "remove"):  # tolerate either token
            removed += 1
            continue
        if act == "move" and "x" in d and "y" in d:
            x = min(1.0, max(0.0, float(d["x"])))
            y = min(1.0, max(0.0, float(d["y"])))
            g = {**g, "clicks": [{"x": x, "y": y, "label": 1}]}
            moved += 1
        new.append(g)            # KEEP, MOVE, or unmentioned (default keep)
    for i, g in enumerate(new, 1):
        g["id"] = i
    return new, removed, moved


def cmd_run(strategy, repeats):
    _set_cc_source()
    cfg = STRATEGIES[strategy]
    base = json.load(open(BASE_JSON))
    sroot = os.path.join(OUT, strategy)
    os.makedirs(sroot, exist_ok=True)
    rows = []
    tot_hit = tot_gt = 0
    tot_stray = 0.0
    print(f"=== bake-off strategy={strategy} cfg={cfg} repeats={repeats} ===",
          flush=True)
    for tgt in FRAMES:
        if str(tgt) not in base:
            continue
        H, W, gt = load_target_and_gt(tgt, min_prob=0.5)
        frame = read_video_frame(tgt)
        nbs = _get_nbs(tgt) if cfg["temporal"] else []
        per = []
        for ri in range(repeats):
            if ri >= len(base[str(tgt)]):
                break
            base_found = base[str(tgt)][ri]
            sdir = os.path.join(sroot, f"f{tgt:03d}_r{ri}")
            os.makedirs(sdir, exist_ok=True)
            per.append(_run_one(cfg, frame, nbs, W, H, gt, base_found, sdir))
        cov = [m["n_hit"] / max(1, m["n_gt"]) for m in per]
        stray = [m["n_stray"] for m in per]
        added = [m["n_added"] for m in per]
        cand = [m["n_cand"] for m in per]
        row = {"frame": f"cc_f{tgt:03d}", "n_gt": per[0]["n_gt"],
               "recall": round(float(np.mean(cov)), 3),
               "recall_std": round(float(np.std(cov)), 3),
               "stray": round(float(np.mean(stray)), 2),
               "added": round(float(np.mean(added)), 2),
               "cand": round(float(np.mean(cand)), 2)}
        rows.append(row)
        tot_hit += sum(m["n_hit"] for m in per)
        tot_gt += sum(m["n_gt"] for m in per)
        tot_stray += sum(stray)
        print(f"  [{strategy}] {row['frame']}: {row['n_gt']} GT, recall "
              f"{row['recall']:.2f}+-{row['recall_std']:.2f}, stray {row['stray']:.2f}, "
              f"added {row['added']:.2f}, cand {row['cand']:.2f}", flush=True)
    summary = {"strategy": strategy, "cfg": cfg, "repeats": repeats,
               "recall": round(tot_hit / max(1, tot_gt), 3),
               "strays_per_pass": round(tot_stray / max(1, repeats), 2),
               "frames": rows}
    json.dump(summary, open(os.path.join(sroot, "summary.json"), "w"), indent=2)
    print(f"OVERALL [{strategy}]: recall={summary['recall']} "
          f"strays/pass={summary['strays_per_pass']}", flush=True)


def _render_discover(frame, base, tile, path, title):
    """Full-frame render: whole-frame finder clicks (yellow, B#) and tile-sweep
    clicks (cyan, T#), each numbered, with a legend banner."""
    out = frame.copy()
    H, W = out.shape[:2]
    for kind, pts, col in (("B", base, (0, 255, 255)), ("T", tile, (255, 200, 0))):
        for i, c in enumerate(pts, 1):
            p = (int(c["x"] * W), int(c["y"] * H))
            cv2.drawMarker(out, p, (0, 0, 0), cv2.MARKER_TILTED_CROSS, 26, 5)
            cv2.drawMarker(out, p, col, cv2.MARKER_TILTED_CROSS, 24, 2)
            cv2.circle(out, p, 16, col, 2)
            lbl = f"{kind}{i}"
            cv2.putText(out, lbl, (p[0] + 12, p[1] - 12), cv2.FONT_HERSHEY_SIMPLEX,
                        0.6, (0, 0, 0), 4, cv2.LINE_AA)
            cv2.putText(out, lbl, (p[0] + 12, p[1] - 12), cv2.FONT_HERSHEY_SIMPLEX,
                        0.6, col, 1, cv2.LINE_AA)
    cv2.rectangle(out, (0, 0), (W, 30), (0, 0, 0), -1)
    banner = (f"{title}   whole-frame(B)={len(base)} [yellow]   "
              f"tile-sweep(T)={len(tile)} [cyan]   total={len(base) + len(tile)}")
    cv2.putText(out, banner, (6, 21), cv2.FONT_HERSHEY_SIMPLEX, 0.55,
                (255, 255, 255), 2, cv2.LINE_AA)
    cv2.imwrite(path, out)


def _engine_on_frame(frame, tgt, W, H, cfg, sdir, fps, review, offsets=(15, 30, 45)):
    """The full click engine on one frame (no GT needed): whole-frame iterative
    finder -> tile sweep -> optional global review. Returns
    (base_found, tile_added, reviewed_groups, meta) where reviewed_groups carry a
    'source' tag ('B' whole-frame / 'T' tile). ``offsets`` = temporal neighbour
    frame deltas (video clip: 15/30/45; frames_dir clip: 1/2/3)."""
    offsets = list(offsets)
    tpath = os.path.join(sdir, "target.png")
    cv2.imwrite(tpath, frame)
    # Stage 1: whole-frame iterative finder (primary recall engine, temporal ctx).
    nbs_ctx = P.extract_neighbours(tgt, offsets[:2], sdir, fps=fps)
    score = P.run_iterative(tpath, frame, [], nbs_ctx, W, H, sdir, refine="verify",
                            label="iter", tile_sweep=False)
    base_found = [{"x": g["clicks"][0]["x"], "y": g["clicks"][0]["y"],
                   "description": g.get("description", "")} for g in score["groups"]]
    # Stage 2: tile sweep (camouflage lever).
    nbs = _get_nbs(tgt, offsets=offsets) if cfg["temporal"] else []
    tile_added, n_cand, gated = ([], 0, 0)
    if cfg["tile"]:
        tile_added, n_cand, gated = tile_sweep_cfg(
            cfg, frame, nbs, W, H, base_found, os.path.join(sdir, "tiles"))
    groups = [{"id": i + 1, "source": "B", "description": c["description"],
               "clicks": [{"x": c["x"], "y": c["y"], "label": 1}]}
              for i, c in enumerate(base_found)]
    groups += [{"id": len(groups) + i + 1, "source": "T",
                "description": c["description"],
                "clicks": [{"x": c["x"], "y": c["y"], "label": 1}]}
               for i, c in enumerate(tile_added)]
    # Stage 3a: deterministic duplicate merge (catches near-coincident clicks like
    # the rank01 B1/T1 pair) -- safe, no recall risk vs the MLLM review pass.
    groups, dedup_removed = _geom_dedup(groups, W, H)
    # Stage 3b: OPTIONAL whole-frame MLLM review. OFF by default: at repeats=3 it
    # cost ~0.13 recall (it removes camouflaged creatures it can't resolve at full
    # frame res, and its centre-of-mass MOVES knock small-creature clicks off). The
    # centre-of-mass goal is already met by the per-click ZOOMED refine.
    removed = moved = 0
    if review and groups:
        groups, removed, moved = global_review(groups, frame, nbs, W, H, sdir,
                                                temporal=cfg["temporal"])
    meta = {"n_whole_frame": len(base_found), "n_tile": len(tile_added),
            "tile_raw_candidates": n_cand, "tile_gate_dropped": gated,
            "dedup_removed": dedup_removed,
            "review_removed": removed, "review_moved": moved}
    return base_found, tile_added, groups, meta


def cmd_discover(video, sec, strategy, fps, model, review):
    """Cold-start discovery on an arbitrary video frame (no GT). Runs the full
    engine and renders every placed click before and after the review pass."""
    P.SRC.update(video=video, frames_dir=None, frame_outputs=None, fps=fps)
    P.MODEL = model
    cfg = STRATEGIES[strategy]
    tgt = int(round(sec * fps))
    frame = read_video_frame(tgt)
    H, W = frame.shape[:2]
    mtag = model.split("-")[1] if "-" in model else model
    base = f"{os.path.splitext(os.path.basename(video))[0][:24]}_t{sec:g}s_f{tgt}"
    sdir = os.path.join(OUT, "discover", f"{base}_{strategy}_{mtag}")
    os.makedirs(sdir, exist_ok=True)
    print(f"=== discover video={os.path.basename(video)} t={sec}s (frame {tgt}) "
          f"strategy={strategy} model={model} review={review} {W}x{H} ===", flush=True)

    base_found, tile_added, reviewed, meta = _engine_on_frame(
        frame, tgt, W, H, cfg, sdir, fps, review)
    print(f"  [discover] whole-frame {meta['n_whole_frame']}, tile "
          f"{meta['n_tile']} (raw {meta['tile_raw_candidates']}, gated "
          f"{meta['tile_gate_dropped']}); review removed {meta['review_removed']}, "
          f"moved {meta['review_moved']}", flush=True)

    # pre-review render (provenance) and post-review render (final survivors)
    _render_discover(frame, base_found, tile_added,
                     os.path.join(sdir, "clicks_prereview.png"),
                     f"{base} {strategy}/{mtag} pre-review")
    fb = [{"x": g["clicks"][0]["x"], "y": g["clicks"][0]["y"]}
          for g in reviewed if g.get("source") == "B"]
    ft = [{"x": g["clicks"][0]["x"], "y": g["clicks"][0]["y"]}
          for g in reviewed if g.get("source") == "T"]
    render_path = os.path.join(sdir, "clicks.png")
    _render_discover(frame, fb, ft, render_path,
                     f"{base} {strategy}/{mtag} final")
    result = {"video": video, "sec": sec, "frame": tgt, "strategy": strategy,
              "model": model, "review": review, "cfg": cfg, "W": W, "H": H,
              "n_final": len(reviewed), **meta,
              "final_clicks": [{"x": round(g["clicks"][0]["x"], 4),
                                "y": round(g["clicks"][0]["y"], 4),
                                "source": g.get("source"),
                                "description": g.get("description", "")}
                               for g in reviewed],
              "render": render_path}
    json.dump(result, open(os.path.join(sdir, "discover.json"), "w"), indent=2)
    print(f"  [discover] FINAL creatures identified: {len(reviewed)} "
          f"(was {meta['n_whole_frame'] + meta['n_tile']} pre-review)", flush=True)
    print(f"  [discover] render -> {render_path}", flush=True)
    return result


def cmd_engine(model, repeats, review, strategy="S3_cons_temp"):
    """Quantitative model comparison: run the FULL engine (whole-frame + tile +
    review) end-to-end under ``model`` on the 6 hard chinacreek frames, score
    clicks vs GT. Unlike ``run`` this does NOT reuse the paired base cache -- it
    runs the whole pipeline live so the model is exercised at every stage."""
    _set_cc_source()
    P.MODEL = model
    cfg = STRATEGIES[strategy]
    mtag = model.split("-")[1] if "-" in model else model
    sroot = os.path.join(OUT, "engine", f"{strategy}_{mtag}"
                         + ("" if review else "_noreview"))
    os.makedirs(sroot, exist_ok=True)
    rows, tot_hit, tot_gt, tot_stray = [], 0, 0, 0.0
    tot_removed = tot_moved = 0.0
    print(f"=== engine model={model} strategy={strategy} review={review} "
          f"repeats={repeats} ===", flush=True)
    for tgt in FRAMES:
        H, W, gt = load_target_and_gt(tgt, min_prob=0.5)
        if not gt:
            continue
        frame = read_video_frame(tgt)
        per = []
        for ri in range(repeats):
            sdir = os.path.join(sroot, f"f{tgt:03d}_r{ri}")
            os.makedirs(sdir, exist_ok=True)
            _bf, _ta, reviewed, meta = _engine_on_frame(
                frame, tgt, W, H, cfg, sdir, 30.0, review)
            sc = score_clicks(reviewed, gt, W, H)
            per.append({**meta, "n_gt": sc["n_gt"], "n_hit": sc["n_hit"],
                        "n_stray": sc["n_stray"], "n_final": len(reviewed)})
        cov = [m["n_hit"] / max(1, m["n_gt"]) for m in per]
        stray = [m["n_stray"] for m in per]
        row = {"frame": f"cc_f{tgt:03d}", "n_gt": per[0]["n_gt"],
               "recall": round(float(np.mean(cov)), 3),
               "recall_std": round(float(np.std(cov)), 3),
               "stray": round(float(np.mean(stray)), 2),
               "final": round(float(np.mean([m["n_final"] for m in per])), 2),
               "removed": round(float(np.mean([m["review_removed"] for m in per])), 2),
               "moved": round(float(np.mean([m["review_moved"] for m in per])), 2)}
        rows.append(row)
        tot_hit += sum(m["n_hit"] for m in per)
        tot_gt += sum(m["n_gt"] for m in per)
        tot_stray += sum(stray)
        tot_removed += sum(m["review_removed"] for m in per)
        tot_moved += sum(m["review_moved"] for m in per)
        print(f"  [{mtag}] {row['frame']}: {row['n_gt']} GT, recall "
              f"{row['recall']:.2f}+-{row['recall_std']:.2f}, stray {row['stray']:.2f}, "
              f"final {row['final']:.2f}, removed {row['removed']:.2f}, moved "
              f"{row['moved']:.2f}", flush=True)
    summary = {"model": model, "strategy": strategy, "review": review,
               "repeats": repeats,
               "recall": round(tot_hit / max(1, tot_gt), 3),
               "strays_per_pass": round(tot_stray / max(1, repeats), 2),
               "removed_per_pass": round(tot_removed / max(1, repeats), 2),
               "moved_per_pass": round(tot_moved / max(1, repeats), 2),
               "frames": rows}
    json.dump(summary, open(os.path.join(sroot, "summary.json"), "w"), indent=2)
    print(f"OVERALL [{mtag} review={review}]: recall={summary['recall']} "
          f"strays/pass={summary['strays_per_pass']} "
          f"removed/pass={summary['removed_per_pass']} "
          f"moved/pass={summary['moved_per_pass']}", flush=True)
    return summary


def cmd_e2e(model, repeats, review, strategy="S3_cons_temp", do_verify=True,
            maskgen="mmzoom", verify_masks_on=False, manifest=False):
    """FULL e2e confirmation WITH SAM3: run the recommended engine (clicks) ->
    per-click verify gate -> SAM3 masks -> greedy IoU match vs GT (0.5). With
    ``verify_masks_on`` a conservative Fable post-mask check splits the old "stray"
    bucket into CAND_NEW (real creature, no GT match -> a candidate new label,
    kept) vs FP (confident substrate -> dropped). With ``manifest`` it runs the full
    EVAL_MANIFEST (12 chinacreek + 4 rank03, multi-source) instead of the 6 hard
    chinacreek frames. Single process (one SAM3 / GPU)."""
    P.MODEL = model
    cfg = STRATEGIES[strategy]
    mtag = model.split("-")[1] if "-" in model else model
    # Build the frame work-list (per-source, since rank03 uses frames_dir + 1/2/3
    # offsets vs chinacreek video + 15/30/45). fname keys the resumable cache, so a
    # 16-frame run reuses any matching cc_fNNN frames already cached.
    if manifest:
        srcs = P.EVAL_MANIFEST
    else:
        srcs = [{"name": "cc", "frame_outputs": P.FRAME_OUTPUTS, "video": P.VIDEO,
                 "frames_dir": None, "fps": 30.0, "targets": FRAMES}]
    work = []
    for src in srcs:
        offs = [1, 2, 3] if src["frames_dir"] else [15, 30, 45]
        for tgt in src["targets"]:
            work.append({"src": src, "name": src["name"], "tgt": tgt,
                         "fps": src["fps"], "offsets": offs})
    sroot = os.path.join(OUT, "e2e", f"{strategy}_{mtag}_{maskgen}"
                         + ("_review" if review else "") + ("" if do_verify else "_nogate")
                         + ("_vmask" if verify_masks_on else ""))
    os.makedirs(sroot, exist_ok=True)
    cand_dir = os.path.join(sroot, "candidates")
    os.makedirs(cand_dir, exist_ok=True)
    mask_fn = {"mmzoom": P.refine_group_mm_zoom, "hybrid": P.refine_group_mm_hybrid,
               "mm": P.refine_group_mm}[maskgen]
    service = P.build_sam3_service()
    # resumable cache: per-frame pooled contributions, so a crash/laptop-death
    # only loses the in-progress frame (relaunch skips completed ones).
    cache_path = os.path.join(sroot, "frames_cache.json")
    cache = json.load(open(cache_path)) if os.path.exists(cache_path) else {}
    print(f"=== e2e+SAM3 model={model} strategy={strategy} mask={maskgen} "
          f"review={review} verify={do_verify} repeats={repeats} "
          f"(resuming {len(cache)} cached frames) ===", flush=True)
    for w in work:
        tgt, fname = w["tgt"], f"{w['name']}_f{w['tgt']:03d}"
        P.SRC.update(video=w["src"]["video"], frames_dir=w["src"]["frames_dir"],
                     frame_outputs=w["src"]["frame_outputs"], fps=w["src"]["fps"])
        H, W, gt = load_target_and_gt(tgt, min_prob=0.5)
        if not gt:
            print(f"  [e2e/{mtag}] {fname}: no GT>=0.5, skipping", flush=True)
            continue
        if fname in cache:
            r = cache[fname]["row"]
            print(f"  [e2e/{mtag}] {fname}: CACHED recall {r['recall']:.2f} "
                  f"IoU(det) {r['iou_det']:.3f} stray {r['stray']:.2f}", flush=True)
            continue
        frame = read_video_frame(tgt)
        runs, f_ious, f_dets, f_strays, f_cand, f_fp = [], [], [], [], [], []
        for ri in range(repeats):
            sdir = os.path.join(sroot, f"{fname}_r{ri}")
            os.makedirs(sdir, exist_ok=True)
            _bf, _ta, groups, meta = _engine_on_frame(frame, tgt, W, H, cfg, sdir,
                                                       w["fps"], review,
                                                       offsets=w["offsets"])
            tpath = os.path.join(sdir, "target.png")
            # production Stage A.5: per-click zoomed content gate (drops strays)
            n_drop = 0
            if do_verify and groups:
                groups, n_drop = P.verify_clicks(groups, frame, W, H, sdir)
            # Stage B: each surviving click -> mask
            results = [mask_fn(service, g, tpath, frame, W, H, sdir,
                               max_clicks=5, max_attempts=3)[0] for g in groups]
            # Stage C (optional): conservative Fable post-mask verify -> kept vs FP
            if verify_masks_on and results:
                kept_res, dropped_res = P.verify_masks(results, frame, W, H, sdir,
                                                       model=model)
            else:
                kept_res, dropped_res = results, []
            n_fp = sum(1 for r in dropped_res if np.asarray(r["mask"]).any())
            masks = [np.asarray(r["mask"]).astype(bool) for r in kept_res]
            nonempty = [(i, m) for i, m in enumerate(masks) if m.any()]
            gt_best = {g["id"]: 0.0 for g in gt}
            taken, matched, pairs = set(), set(), []
            for pi, m in nonempty:
                for g in gt:
                    pairs.append((P._iou(m, g["mask"]), pi, g["id"]))
            for iou, pi, gid in sorted(pairs, reverse=True):
                if iou < 0.5 or pi in matched or gid in taken:
                    continue
                matched.add(pi); taken.add(gid); gt_best[gid] = iou
            # kept + non-empty + unmatched = candidate NEW creature (GT missed it)
            n_cand = len(nonempty) - len(matched)
            for pi, m in nonempty:
                if pi in matched:
                    continue
                geom = P._mask_crop_geom(m, [], W, H, 0.22)
                P._render_mask_crop(frame, m, [], geom,
                                    os.path.join(cand_dir, f"{fname}_r{ri}_p{pi}.png"), 6)
            ious_all = [gt_best[g["id"]] for g in gt]
            det = [v for v in ious_all if v >= 0.5]
            runs.append({"recall": len(det) / len(gt),
                         "iou_all": float(np.mean(ious_all)) if ious_all else 0.0,
                         "iou_det": float(np.mean(det)) if det else 0.0,
                         "n_cand": n_cand, "n_fp": n_fp, "n_drop": n_drop})
            for g in gt:
                f_ious.append(gt_best[g["id"]])
                f_dets.append(bool(gt_best[g["id"]] >= 0.5))
            f_strays.append(n_cand + n_fp); f_cand.append(n_cand); f_fp.append(n_fp)
        rm = lambda k: float(np.mean([r[k] for r in runs]))
        rs = lambda k: float(np.std([r[k] for r in runs]))
        row = {"frame": fname, "n_gt": len(gt),
               "recall": round(rm("recall"), 3), "recall_std": round(rs("recall"), 3),
               "iou_all": round(rm("iou_all"), 3), "iou_det": round(rm("iou_det"), 3),
               "cand_new": round(rm("n_cand"), 2), "fp": round(rm("n_fp"), 2),
               "stray": round(rm("n_cand") + rm("n_fp"), 2)}
        cache[fname] = {"row": row, "ious": f_ious, "dets": f_dets, "strays": f_strays,
                        "cand": f_cand, "fp": f_fp}
        json.dump(cache, open(cache_path, "w"), indent=2)   # checkpoint each frame
        print(f"  [e2e/{mtag}] {row['frame']}: {row['n_gt']} GT, recall "
              f"{row['recall']:.2f}+-{row['recall_std']:.2f}, IoU(det) "
              f"{row['iou_det']:.3f}, cand_new {row['cand_new']:.2f}, FP {row['fp']:.2f}",
              flush=True)
    # aggregate pooled metrics from the cache (in work order)
    pooled_iou, pooled_det, pooled_stray, rows = [], [], [], []
    pooled_cand, pooled_fp = [], []
    for w in work:
        fname = f"{w['name']}_f{w['tgt']:03d}"
        if fname in cache:
            c = cache[fname]
            pooled_iou += c["ious"]; pooled_det += c["dets"]; pooled_stray += c["strays"]
            pooled_cand += c.get("cand", []); pooled_fp += c.get("fp", [])
            rows.append(c["row"])
    summary = {"model": model, "strategy": strategy, "review": review,
               "do_verify": do_verify, "verify_masks": verify_masks_on, "repeats": repeats,
               "recall": round(float(np.mean(pooled_det)), 3) if pooled_det else 0.0,
               "mean_iou_all": round(float(np.mean(pooled_iou)), 3) if pooled_iou else 0.0,
               "mean_iou_detected": round(float(np.mean(
                   [i for i, d in zip(pooled_iou, pooled_det) if d])), 3)
               if any(pooled_det) else 0.0,
               "strays_per_pass": round(float(np.sum(pooled_stray)) / max(1, repeats), 2),
               "cand_new_per_pass": round(float(np.sum(pooled_cand)) / max(1, repeats), 2),
               "fp_per_pass": round(float(np.sum(pooled_fp)) / max(1, repeats), 2),
               "frames": rows}
    summary["maskgen"] = maskgen
    json.dump(summary, open(os.path.join(sroot, "summary.json"), "w"), indent=2)
    print(f"  cand_new/pass={summary['cand_new_per_pass']} (kept, GT-missed) "
          f"FP/pass={summary['fp_per_pass']} (dropped substrate)", flush=True)
    print(f"OVERALL [e2e {mtag} mask={maskgen} review={review}]: recall={summary['recall']} "
          f"IoU(all)={summary['mean_iou_all']} IoU(det)={summary['mean_iou_detected']} "
          f"strays/pass={summary['strays_per_pass']}", flush=True)
    return summary


def _render_gate(frame, gt, kept, dropped, path):
    """GT outlines (green); gate-KEPT clicks green dot, gate-DROPPED clicks red X.
    A red X sitting on a GT outline = the gate wrongly removed a real creature."""
    out = frame.copy()
    H, W = out.shape[:2]
    for g in gt:
        cnts, _ = cv2.findContours(g["mask"].astype(np.uint8), cv2.RETR_EXTERNAL,
                                   cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(out, cnts, -1, (0, 220, 0), 2)
        cx, cy = g["centroid"]
        cv2.putText(out, f"GT{g['id']}", (cx - 12, cy), cv2.FONT_HERSHEY_SIMPLEX,
                    0.5, (0, 220, 0), 1, cv2.LINE_AA)
    for tagname, grp, col, mk in (("kept", kept, (0, 255, 255), cv2.MARKER_TILTED_CROSS),
                                   ("drop", dropped, (0, 0, 255), cv2.MARKER_TILTED_CROSS)):
        for g in grp:
            c = g["clicks"][0]
            p = (int(c["x"] * W), int(c["y"] * H))
            cv2.drawMarker(out, p, (0, 0, 0), mk, 22, 5)
            cv2.drawMarker(out, p, col, mk, 20, 2)
    cv2.rectangle(out, (0, 0), (W, 26), (0, 0, 0), -1)
    cv2.putText(out, f"GT green | gate KEPT=yellow {len(kept)} | DROPPED=red "
                f"{len(dropped)}", (6, 19), cv2.FONT_HERSHEY_SIMPLEX, 0.55,
                (255, 255, 255), 2, cv2.LINE_AA)
    cv2.imwrite(path, out)


def cmd_diag(model, frame_idx, review, maskgen="mmzoom"):
    """Single-frame visual diagnostic: engine -> gate -> SAM3, rendered against GT.
    Saves diag_clicks (all placed clicks vs GT), diag_gate (gate keep/drop vs GT),
    diag_masks (SAM3 masks matched/unmatched). Prints per-GT and per-click detail."""
    _set_cc_source()
    P.MODEL = model
    cfg = STRATEGIES["S3_cons_temp"]
    H, W, gt = load_target_and_gt(frame_idx, min_prob=0.5)
    frame = read_video_frame(frame_idx)
    mtag = model.split("-")[1] if "-" in model else model
    sdir = os.path.join(OUT, "diag", f"f{frame_idx:03d}_{mtag}")
    os.makedirs(sdir, exist_ok=True)
    service = P.build_sam3_service()
    print(f"=== diag f{frame_idx} model={model} review={review} ({len(gt)} GT) ===",
          flush=True)
    _bf, _ta, groups, meta = _engine_on_frame(frame, frame_idx, W, H, cfg, sdir,
                                              30.0, review)
    tpath = os.path.join(sdir, "target.png")
    print(f"  engine: whole-frame {meta['n_whole_frame']}, tile {meta['n_tile']}, "
          f"dedup_removed {meta['dedup_removed']} -> {len(groups)} clicks", flush=True)

    sc = score_clicks(groups, gt, W, H)
    P.render_result(frame, gt, groups, sc, os.path.join(sdir, "diag_clicks.png"),
                    f"f{frame_idx} pre-gate clicks")
    print(f"  pre-gate: {sc['n_hit']}/{sc['n_gt']} GT clicked, {sc['n_stray']} stray clicks")

    pre_xy = [(g["clicks"][0]["x"], g["clicks"][0]["y"]) for g in groups]
    kept, ndrop = P.verify_clicks(groups, frame, W, H, sdir)
    kept_xy = {(g["clicks"][0]["x"], g["clicks"][0]["y"]) for g in kept}
    dropped = [g for g, xy in zip(groups, pre_xy) if xy not in kept_xy]
    _render_gate(frame, gt, kept, dropped, os.path.join(sdir, "diag_gate.png"))
    # did the gate drop a click that was ON a GT mask?
    bad_drops = 0
    for g in dropped:
        c = g["clicks"][0]
        px, py = int(c["x"] * (W - 1)), int(c["y"] * (H - 1))
        if any(gm["mask"][py, px] for gm in gt if 0 <= py < H and 0 <= px < W):
            bad_drops += 1
    print(f"  gate dropped {len(dropped)} ({bad_drops} were ON a GT creature = "
          f"recall loss); kept {len(kept)}")

    results, masks = [], []
    mask_fn = {"mmzoom": P.refine_group_mm_zoom, "hybrid": P.refine_group_mm_hybrid,
               "mm": P.refine_group_mm}[maskgen]
    for g in kept:
        res, _tr = mask_fn(service, g, tpath, frame, W, H, sdir,
                           max_clicks=5, max_attempts=3)
        results.append(res)
        masks.append(np.asarray(res["mask"]).astype(bool))
    gt_best = {g["id"]: 0.0 for g in gt}
    rows = []
    for r, m in zip(results, masks):
        best, mgid = 0.0, None
        for g in gt:
            iou = P._iou(m, g["mask"]) if m.any() else 0.0
            if iou > best:
                best, mgid = iou, g["id"]
        rows.append({"matched_gt": (mgid if best >= 0.5 else None),
                     "best_iou": round(best, 3)})
        if best >= 0.5:
            gt_best[mgid] = max(gt_best[mgid], best)
    P._render_sam3_report(frame, gt, results, rows, os.path.join(sdir, "diag_masks.png"),
                          f"f{frame_idx} masks")
    # dump a zoomed crop of EACH stray (unmatched, non-empty) mask so it can be
    # classified by eye: real creature the GT missed vs genuine false positive.
    sdump = os.path.join(sdir, "strays")
    os.makedirs(sdump, exist_ok=True)
    n_stray_dumped = 0
    for i, (r, m, row) in enumerate(zip(results, masks, rows)):
        if row["matched_gt"] is not None or not m.any():
            continue
        ys, xs = np.where(m)
        cx, cy = int(xs.mean()), int(ys.mean())
        half = max(60, int(1.4 * max(xs.max() - xs.min(), ys.max() - ys.min())))
        l = max(0, cx - half); t = max(0, cy - half)
        rr = min(W, cx + half); bb = min(H, cy + half)
        up = max(2, int(round(360 / max(1, max(rr - l, bb - t)))))
        crop = cv2.resize(frame[t:bb, l:rr].copy(),
                          ((rr - l) * up, (bb - t) * up), interpolation=cv2.INTER_CUBIC)
        mc = cv2.resize(m[t:bb, l:rr].astype(np.uint8), ((rr - l) * up, (bb - t) * up),
                        interpolation=cv2.INTER_NEAREST).astype(bool)
        ov = crop.copy(); ov[mc] = (255, 255, 0)
        crop = cv2.addWeighted(ov, 0.4, crop, 0.6, 0)
        cnts, _ = cv2.findContours(mc.astype(np.uint8), cv2.RETR_EXTERNAL,
                                   cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(crop, cnts, -1, (0, 255, 255), 2)
        cv2.rectangle(crop, (0, 0), (crop.shape[1], 22), (0, 0, 0), -1)
        cv2.putText(crop, f"f{frame_idx} stray '{r.get('description','')[:30]}' "
                    f"bestIoU{row['best_iou']} area{int(m.sum())}", (4, 16),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.42, (0, 255, 255), 1, cv2.LINE_AA)
        cv2.imwrite(os.path.join(sdump, f"stray_{n_stray_dumped}.png"), crop)
        n_stray_dumped += 1
    print(f"  dumped {n_stray_dumped} stray-mask crops -> {sdump}")
    hit = sum(1 for v in gt_best.values() if v >= 0.5)
    print(f"  SAM3: {hit}/{len(gt)} GT detected (IoU>=0.5). per-mask best IoU: "
          f"{[r['best_iou'] for r in rows]}")
    for g in gt:
        print(f"    GT{g['id']}: {'HIT' if gt_best[g['id']]>=0.5 else 'MISS'} "
              f"(best IoU {gt_best[g['id']]:.2f}, area {g['area']}px)")
    print(f"  renders -> {sdir}/diag_clicks.png diag_gate.png diag_masks.png", flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    sp = sub.add_parser("setup")
    sp.add_argument("--repeats", type=int, default=3)
    rp = sub.add_parser("run")
    rp.add_argument("--strategy", required=True, choices=list(STRATEGIES))
    rp.add_argument("--repeats", type=int, default=3)
    dp = sub.add_parser("discover")
    dp.add_argument("--video", required=True)
    dp.add_argument("--sec", type=float, required=True)
    dp.add_argument("--strategy", default="S3_cons_temp", choices=list(STRATEGIES))
    dp.add_argument("--fps", type=float, default=30.0)
    dp.add_argument("--model", default="claude-sonnet-4-6")
    dp.add_argument("--review", action="store_true",
                    help="enable the (recall-hurting) MLLM whole-frame review pass")
    ep = sub.add_parser("engine")
    ep.add_argument("--model", default="claude-sonnet-4-6")
    ep.add_argument("--strategy", default="S3_cons_temp", choices=list(STRATEGIES))
    ep.add_argument("--repeats", type=int, default=3)
    ep.add_argument("--review", action="store_true",
                    help="enable the (recall-hurting) MLLM whole-frame review pass")
    e2 = sub.add_parser("e2e")
    e2.add_argument("--model", default="claude-sonnet-4-6")
    e2.add_argument("--strategy", default="S3_cons_temp", choices=list(STRATEGIES))
    e2.add_argument("--repeats", type=int, default=3)
    e2.add_argument("--review", action="store_true")
    e2.add_argument("--no-gate", dest="do_verify", action="store_false")
    e2.add_argument("--maskgen", default="mmzoom", choices=["mmzoom", "hybrid", "mm"])
    e2.add_argument("--verify-masks", action="store_true",
                    help="conservative post-mask Fable check: split strays into "
                         "candidate-new (kept) vs false-positive (dropped)")
    e2.add_argument("--manifest", action="store_true",
                    help="run the full EVAL_MANIFEST (12 chinacreek + 4 rank03) "
                         "instead of the 6 hard chinacreek frames")
    dg = sub.add_parser("diag")
    dg.add_argument("--frame", type=int, required=True)
    dg.add_argument("--model", default="claude-sonnet-4-6")
    dg.add_argument("--review", action="store_true")
    dg.add_argument("--maskgen", default="mmzoom", choices=["mmzoom", "hybrid", "mm"])
    a = ap.parse_args()
    if a.cmd == "setup":
        cmd_setup(a.repeats)
    elif a.cmd == "run":
        cmd_run(a.strategy, a.repeats)
    elif a.cmd == "engine":
        cmd_engine(a.model, a.repeats, a.review, a.strategy)
    elif a.cmd == "e2e":
        cmd_e2e(a.model, a.repeats, a.review, a.strategy, a.do_verify, a.maskgen,
                a.verify_masks, a.manifest)
    elif a.cmd == "diag":
        cmd_diag(a.model, a.frame, a.review, a.maskgen)
    else:
        cmd_discover(a.video, a.sec, a.strategy, a.fps, a.model, a.review)
