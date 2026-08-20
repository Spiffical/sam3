#!/usr/bin/env python3
"""Controlled probe for the iterative click engine.

Goal: isolate the *click placement* ability of the discovery MLLM, decoupled
from SAM3. We take one frame where the text-agent already found N creatures,
treat those N masks as ground truth, hide them, give the MLLM the raw frame
plus temporal-context neighbours, and ask it to click every creature. A GT
creature is "hit" if a foreground click lands inside its mask.

This is deliberately standalone -- it does NOT import/mutate the SoM pipeline
beyond reusing the mask decoder and the click-group parser.

Usage:
    .venv/bin/python scripts/click_engine_probe.py --render-gt-only
    .venv/bin/python scripts/click_engine_probe.py --strategy baseline
    .venv/bin/python scripts/click_engine_probe.py --strategy all
"""
from __future__ import annotations

import argparse
import itertools
import json
import os
import re
from dataclasses import dataclass, field

import cv2
import numpy as np

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

from nibi_model_compare.frame_output_utils import decode_rle_to_mask
from nibi_model_compare.som_missed_creatures import parse_creature_click_groups
from sam3.agent.client_claude import send_claude_request

VIDEO = "assets/videos/onc/chinacreekclipped.mp4"
FRAME_OUTPUTS = (
    "runs/agent_every_frame/chinacreekclipped/"
    "claude_sonnet_4_6_60frame_20260525_155604/frame_outputs_rle.json"
)
DEFAULT_TARGET = 45
DEFAULT_OFFSETS = [15, 30, 45]  # +/- frames at 30fps => 0.5/1.0/1.5s
MODEL = "claude-sonnet-4-6"
OUT_ROOT = "runs/click_probe"


def _iteration_indices(max_clicks: int):
    """Yield refiner iterations; ``max_clicks <= 0`` means agent-controlled."""
    if max_clicks <= 0:
        return itertools.count()
    return range(max_clicks)


def _click_budget_reached(clicks, max_clicks: int) -> bool:
    """Return whether a positive, finite click ceiling has been reached."""
    return max_clicks > 0 and len(clicks) >= max_clicks


def _duplicate_click(clicks, candidate, tolerance: float = 1e-3) -> bool:
    """Detect a no-progress click without imposing a useful-click ceiling."""
    return any(
        int(click.get("label", -1)) == int(candidate.get("label", -2))
        and (
            (float(click.get("x", -10.0)) - float(candidate.get("x", 10.0))) ** 2
            + (float(click.get("y", -10.0)) - float(candidate.get("y", 10.0))) ** 2
        ) ** 0.5 <= tolerance
        for click in clicks
    )


def _merge_corrected_positive_click(clicks, corrected):
    """Replace the nearest positive seed while preserving all other clicks."""
    merged = [dict(click) for click in clicks]
    positive_indices = [
        index
        for index, click in enumerate(merged)
        if int(click.get("label", 1)) == 1
    ]
    if not positive_indices:
        return [dict(corrected)] + merged
    nearest = min(
        positive_indices,
        key=lambda index: (
            (float(merged[index]["x"]) - float(corrected["x"])) ** 2
            + (float(merged[index]["y"]) - float(corrected["y"])) ** 2
        ),
    )
    merged[nearest] = dict(corrected)
    return merged


def _bounded_corrected_positive_click(
    clicks,
    corrected,
    *,
    max_displacement: float = 0.10,
):
    """Apply a localization correction only when it remains near its seed.

    The visual localizer is allowed to move a slightly misplaced foreground
    point onto a solid part of the same target.  A large move is much more
    likely to switch to a nearby organism, especially in dense scenes.  In
    that case retain the discovery agent's original clicks rather than
    silently changing the biological identity being segmented.

    Returns ``(clicks, displacement, applied)``.  Displacement is measured in
    normalized full-frame coordinates from the nearest positive seed.
    """
    positive_clicks = [
        click for click in clicks if int(click.get("label", 1)) == 1
    ]
    if not positive_clicks:
        return [dict(corrected)] + [dict(click) for click in clicks], None, True
    displacement = min(
        (
            (float(click["x"]) - float(corrected["x"])) ** 2
            + (float(click["y"]) - float(corrected["y"])) ** 2
        ) ** 0.5
        for click in positive_clicks
    )
    if displacement > max_displacement:
        return [dict(click) for click in clicks], displacement, False
    return (
        _merge_corrected_positive_click(clicks, corrected),
        displacement,
        True,
    )


def response_token_budget(
    model: str,
    default: int,
    *,
    sonnet5_minimum: int = 4096,
    opus5_minimum: int = 4096,
) -> int:
    """Leave enough room for reasoning models before a structured answer.

    Sonnet 5 and Opus 5 can consume a small completion budget on internal
    reasoning and return either no text block or a JSON tag truncated
    mid-object. Older models keep their established, cheaper caps.
    """
    normalized_model = str(model).lower()
    if "sonnet-5" in normalized_model:
        return max(int(default), int(sonnet5_minimum))
    if "opus-5" in normalized_model:
        return max(int(default), int(opus5_minimum))
    return int(default)

# Labeled frames used by the click->mask eval harness (--eval). Each GT mask is
# treated as the answer; we seed one fg click at its centroid and measure how well
# a mask-generator variant reproduces it. rank03 GT is marine-snow contaminated, so
# the harness applies --gt-min-prob.
_R03DIR = ("runs/agent_every_frame/rank03_taxa001_ann001_"
           "AXISCAMACCC8E891285_20201113T121535.000Z_t00090/claude_smoke_v4")
EVAL_MANIFEST = [
    {"name": "cc", "frame_outputs": FRAME_OUTPUTS, "video": VIDEO,
     "frames_dir": None, "fps": 30.0,
     "targets": [0, 6, 12, 18, 24, 30, 37, 42, 45, 50, 54, 59]},
    {"name": "r03", "frame_outputs": _R03DIR + "/frame_outputs_rle.json",
     "video": None, "frames_dir": _R03DIR + "/frame_inputs", "fps": None,
     "targets": [0, 4, 5, 9, 11]},
]

# Frame source: either decode the raw mp4 (VIDEO), or -- when the SAM3 run was
# fed a pre-extracted clip -- read the saved frame_inputs/frame_NNNNNN.jpg whose
# index matches frame_index. Set via CLI; module-level so the loaders can see it.
SRC = {"video": VIDEO, "frames_dir": None, "frame_outputs": FRAME_OUTPUTS,
       "fps": 30.0}


# ----------------------------------------------------------------------------
# Data loading
# ----------------------------------------------------------------------------
def load_target_and_gt(target_idx: int, min_prob: float = 0.0):
    """Decode GT masks for the target frame. ``min_prob`` filters to SAM3
    high-confidence masks -- a coarse proxy for 'vetted real creature' that
    drops the low-confidence marine-snow masks the text agent sometimes emits."""
    d = json.load(open(SRC["frame_outputs"]))
    H, W = d["frame_size_hw"]
    rec = None
    for fr in d["frames"]:
        if int(fr["frame_index"]) == target_idx:
            rec = fr
            break
    if rec is None:
        raise SystemExit(f"frame {target_idx} not in frame_outputs")
    probs = rec.get("out_probs", [None] * len(rec["out_obj_ids"]))
    gt = []
    for oid, rle, box, prob in zip(
        rec["out_obj_ids"], rec["out_binary_masks_rle"], rec["out_boxes_xywh"], probs
    ):
        if prob is not None and prob < min_prob:
            continue
        m = decode_rle_to_mask(rle, H, W).astype(bool)
        ys, xs = np.where(m)
        cx, cy = int(xs.mean()), int(ys.mean())
        gt.append({"id": int(oid), "mask": m, "box": box, "centroid": (cx, cy),
                   "area": int(m.sum()), "prob": prob})
    return H, W, gt


def read_video_frame(idx: int):
    if SRC["frames_dir"]:
        p = os.path.join(SRC["frames_dir"], f"frame_{int(idx):06d}.jpg")
        frame = cv2.imread(p)
        if frame is None:
            raise SystemExit(f"could not read frame image {p}")
        return frame
    cap = cv2.VideoCapture(SRC["video"])
    cap.set(cv2.CAP_PROP_POS_FRAMES, int(idx))
    ok, frame = cap.read()
    cap.release()
    if not ok:
        raise SystemExit(f"could not read video frame {idx}")
    return frame


def extract_neighbours(target_idx: int, offsets: list[int], out_dir: str, fps: float = 30.0):
    """Return ordered list of (label, path) for temporal context frames."""
    items = []
    for off in sorted(offsets, reverse=True):  # earliest (most negative) first
        nb = target_idx - off
        if nb >= 0:
            items.append((-off, nb))
    for off in sorted(offsets):
        nb = target_idx + off
        items.append((+off, nb))
    out = []
    for signed_off, nb in items:
        try:
            frame = read_video_frame(nb)
        except SystemExit:
            continue  # neighbour frame missing (e.g. clip edge) -> skip it
        if fps:
            banner = f"t = {signed_off / fps:+.1f}s  (frame {nb})"
        else:
            banner = f"neighbour frame {nb} ({signed_off:+d})"
        labelled = frame.copy()
        cv2.rectangle(labelled, (0, 0), (labelled.shape[1], 26), (0, 0, 0), -1)
        cv2.putText(labelled, banner, (6, 19), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                    (0, 255, 255), 2, cv2.LINE_AA)
        p = os.path.join(out_dir, f"nb_{signed_off:+03d}.png")
        cv2.imwrite(p, labelled)
        out.append((banner, p))
    return out


# ----------------------------------------------------------------------------
# Scoring
# ----------------------------------------------------------------------------
def score_clicks(groups: list[dict], gt: list[dict], W: int, H: int,
                 tol_px: int = 8) -> dict:
    """A GT creature is hit if any label=1 click lands within ``tol_px`` of its
    mask (mask dilated by tol_px). Tolerance reflects that a click a few pixels
    off a thin creature body is still on-target for SAM3 point-mode."""
    fg_clicks = []
    for g in groups:
        for c in g["clicks"]:
            if c["label"] == 1:
                px, py = int(round(c["x"] * (W - 1))), int(round(c["y"] * (H - 1)))
                fg_clicks.append({"px": px, "py": py, "desc": g.get("description", ""),
                                  "gid": g.get("id")})
    if tol_px > 0:
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * tol_px + 1, 2 * tol_px + 1))
        gt_test = [cv2.dilate(g["mask"].astype(np.uint8), k).astype(bool) for g in gt]
    else:
        gt_test = [g["mask"] for g in gt]
    # nearest-click distance (px) from any fg click to each GT mask boundary,
    # to distinguish precision misses (click near body) from detection misses.
    nearest = {}
    for gtm in gt:
        inv = 1 - gtm["mask"].astype(np.uint8)
        dt = cv2.distanceTransform(inv, cv2.DIST_L2, 3)  # dist from each px to mask
        best = None
        for c in fg_clicks:
            if 0 <= c["py"] < H and 0 <= c["px"] < W:
                d = float(dt[c["py"], c["px"]])
                best = d if best is None else min(best, d)
        nearest[gtm["id"]] = best
    hits = {}
    used_clicks = set()
    for i, gtm in enumerate(gt):
        hit_click = None
        for j, c in enumerate(fg_clicks):
            if 0 <= c["py"] < H and 0 <= c["px"] < W and gt_test[i][c["py"], c["px"]]:
                hit_click = j
                used_clicks.add(j)
                break
        hits[gtm["id"]] = hit_click
    stray = [c for j, c in enumerate(fg_clicks) if j not in used_clicks]
    n_hit = sum(1 for v in hits.values() if v is not None)
    # coverage at multiple tolerances, derived from nearest distances
    cov_at = {t: sum(1 for d in nearest.values() if d is not None and d <= t)
              for t in (0, 8, 16, 24, 40)}
    return {
        "n_gt": len(gt),
        "n_hit": n_hit,
        "coverage": n_hit / max(1, len(gt)),
        "hits": hits,
        "nearest_px": nearest,
        "cov_at": cov_at,
        "n_fg_clicks": len(fg_clicks),
        "n_stray": len(stray),
        "fg_clicks": fg_clicks,
        "stray": stray,
    }


def render_result(frame, gt, groups, score, out_path, title=""):
    out = frame.copy()
    # GT masks: faint colored outline + id at centroid
    for gtm in gt:
        m = gtm["mask"].astype(np.uint8)
        contours, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        hit = score["hits"].get(gtm["id"]) is not None
        col = (0, 200, 0) if hit else (0, 0, 255)
        cv2.drawContours(out, contours, -1, col, 2)
        cx, cy = gtm["centroid"]
        cv2.putText(out, f"GT{gtm['id']}", (cx - 12, cy), cv2.FONT_HERSHEY_SIMPLEX,
                    0.5, (0, 0, 0), 3, cv2.LINE_AA)
        cv2.putText(out, f"GT{gtm['id']}", (cx - 12, cy), cv2.FONT_HERSHEY_SIMPLEX,
                    0.5, col, 1, cv2.LINE_AA)
    # proposed foreground clicks: yellow X (hit) / magenta X (stray)
    hit_pts = {(c["px"], c["py"]) for c in score["fg_clicks"]} - \
              {(c["px"], c["py"]) for c in score["stray"]}
    for c in score["fg_clicks"]:
        is_stray = c in score["stray"]
        col = (255, 0, 255) if is_stray else (0, 255, 255)
        p = (c["px"], c["py"])
        cv2.drawMarker(out, p, (0, 0, 0), cv2.MARKER_TILTED_CROSS, 22, 4)
        cv2.drawMarker(out, p, col, cv2.MARKER_TILTED_CROSS, 20, 2)
    banner = f"{title}  coverage {score['n_hit']}/{score['n_gt']}  stray {score['n_stray']}"
    cv2.rectangle(out, (0, 0), (out.shape[1], 28), (0, 0, 0), -1)
    cv2.putText(out, banner, (6, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255),
                2, cv2.LINE_AA)
    cv2.imwrite(out_path, out)


def _iou(a, b):
    inter = np.logical_and(a, b).sum()
    union = np.logical_or(a, b).sum()
    return float(inter) / float(union) if union else 0.0


_SAM3 = {"service": None}


def build_sam3_service(device="cuda"):
    """Lazily build the SAM3 click-mode service (model + processor). Reuses the
    exact construction path the SoM pipeline uses."""
    if _SAM3["service"] is not None:
        return _SAM3["service"]
    from scripts.run_sam3_agent_every_frame_video import ensure_runtime_deps, find_bpe_path
    import scripts.run_sam3_agent_every_frame_video as _efv
    from nibi_model_compare.som_missed_creatures import Sam3PointService
    ensure_runtime_deps()
    print("  [sam3] building image model (enable_inst_interactivity=True)...")
    model = _efv.build_sam3_image_model(
        bpe_path=find_bpe_path(), device=device, enable_inst_interactivity=True)
    processor = _efv.Sam3Processor(model, confidence_threshold=0.5)
    _SAM3["service"] = Sam3PointService(model, processor)
    return _SAM3["service"]


def _extract_answer_json(text):
    """Pull the JSON object from the last <answer>...</answer> tag."""
    if not text:
        return None
    m = re.findall(r"<answer>(.*?)</answer>", text, re.S)
    if not m:
        return None
    try:
        return json.loads(m[-1].strip())
    except Exception:
        return None


def _mask_crop_geom(mask, clicks, W, H, region_frac):
    """Crop window (l,t,cw,ch) centred on the mask (or clicks if mask empty),
    sized to contain the whole mask plus padding, at least region_frac of frame."""
    if mask.any():
        ys, xs = np.where(mask)
        bx0, bx1, by0, by1 = int(xs.min()), int(xs.max()), int(ys.min()), int(ys.max())
        cx, cy = (bx0 + bx1) // 2, (by0 + by1) // 2
        cw = int(max(W * region_frac, (bx1 - bx0) * 1.8))
        ch = int(max(H * region_frac, (by1 - by0) * 1.8))
    else:
        pts = [(c["x"] * W, c["y"] * H) for c in clicks] or [(W / 2, H / 2)]
        cx = int(sum(p[0] for p in pts) / len(pts))
        cy = int(sum(p[1] for p in pts) / len(pts))
        cw, ch = int(W * region_frac), int(H * region_frac)
    cw, ch = min(cw, W), min(ch, H)
    l = max(0, min(W - cw, cx - cw // 2))
    t = max(0, min(H - ch, cy - ch // 2))
    return l, t, cw, ch


def _render_mask_crop(frame, mask, clicks, geom, path, upscale):
    """Zoomed crop with the SAM3 mask outlined/filled and existing clicks drawn.
    fg (label 1) = green dot, bg (label 0) = red X."""
    l, t, cw, ch = geom
    crop = frame[t:t + ch, l:l + cw].copy()
    mc = mask[t:t + ch, l:l + cw].astype(np.uint8)
    zoom = cv2.resize(crop, (cw * upscale, ch * upscale), interpolation=cv2.INTER_CUBIC)
    mz = cv2.resize(mc, (cw * upscale, ch * upscale), interpolation=cv2.INTER_NEAREST).astype(bool)
    ov = zoom.copy()
    ov[mz] = (255, 255, 0)  # cyan fill (BGR)
    zoom = cv2.addWeighted(ov, 0.40, zoom, 0.60, 0)
    cnts, _ = cv2.findContours(mz.astype(np.uint8), cv2.RETR_EXTERNAL,
                               cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(zoom, cnts, -1, (255, 255, 0), 2)
    for c in clicks:
        px = int((c["x"] * frame.shape[1] - l) / cw * cw * upscale)
        py = int((c["y"] * frame.shape[0] - t) / ch * ch * upscale)
        if c["label"] == 1:
            cv2.circle(zoom, (px, py), 7, (0, 0, 0), -1)
            cv2.circle(zoom, (px, py), 5, (0, 255, 0), -1)
        else:
            cv2.drawMarker(zoom, (px, py), (0, 0, 255), cv2.MARKER_TILTED_CROSS, 18, 3)
    cv2.imwrite(path, zoom)


def _render_binary_mask_crop(mask, geom, path, upscale):
    """Render exact mask membership for quality review (white=in, black=out)."""
    l, t, cw, ch = geom
    crop = mask[t:t + ch, l:l + cw].astype(np.uint8)
    binary = cv2.resize(
        crop, (cw * upscale, ch * upscale), interpolation=cv2.INTER_NEAREST
    )
    binary = np.repeat((binary * 255)[:, :, None], 3, axis=2)
    cv2.rectangle(binary, (0, 0), (binary.shape[1], 28), (0, 0, 0), -1)
    cv2.putText(
        binary, "WHITE = MASK; BLACK = OUTSIDE", (6, 20),
        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA,
    )
    cv2.imwrite(path, binary)


MASK_JUDGE_FMT = (
    'Output: brief reasoning then EXACTLY ONE trailing tag, one of:\n'
    '<answer>{"verdict": "good"}</answer>'
    '  -- mask tightly+completely covers the body; ACCEPT it\n'
    '<answer>{"verdict": "add", "click": {"x": <float>, "y": <float>, '
    '"label": 0 or 1}}</answer>'
    '  -- add ONE click (label 1 = include a missed part, 0 = exclude bleed)\n'
    '<answer>{"verdict": "reject"}</answer>'
    '  -- mask is badly wrong (wrong object / hopeless); DISCARD clicks & restart\n'
    '<answer>{"verdict": "abandon"}</answer>'
    '  -- there is NO real creature here; produce no mask\n'
    'x,y are NORMALIZED [0,1] within THIS crop image (0,0=top-left, '
    '1,1=bottom-right).'
)


def _empty_result(grp, H, W, status):
    return {"creature_id": int(grp["id"]), "description": grp.get("description", ""),
            "mask": np.zeros((H, W), dtype=bool), "score": 0.0, "area_px": 0,
            "select_reason": status, "spatial_match": status,
            "clicks_used": [], "status": status}


def _reseed_click(frame, geom, description, sdir, tag, attempt, W, H):
    """On reject, show a clean zoomed crop with a fine numbered grid and ask for a
    fresh foreground seed click (or abandon). Returns ('add',(fx,fy)) |
    ('abandon',None) | None (parse fail)."""
    l, t, cw, ch = geom
    crop = frame[t:t + ch, l:l + cw]
    up = max(2, int(round(720 / max(cw, ch))))
    zoom = cv2.resize(crop, (cw * up, ch * up), interpolation=cv2.INTER_CUBIC)
    zoom = _numbered_grid_overlay(zoom, n=8)
    zp = os.path.join(sdir, f"reseed_{tag}_a{attempt}.png")
    cv2.imwrite(zp, zoom)
    text = (
        f"Fresh attempt. This is a ZOOMED crop that should contain: "
        f"'{description}'. The previous segmentation was rejected.\n"
        f"Place ONE foreground click on the CENTER of the creature's body, OR "
        f"abandon if there is no real creature here.\n\n"
        '<answer>{"verdict": "add", "click": {"x": <float>, "y": <float>, '
        '"label": 1}}</answer>\n'
        '<answer>{"verdict": "abandon"}</answer>\n'
        'x,y NORMALIZED [0,1] within THIS crop image.')
    r = send_claude_request(
        [{"role": "user", "content": [
            {"type": "image", "image": zp}, {"type": "text", "text": text}]}],
        model=MODEL, max_tokens=response_token_budget(MODEL, 600))
    open(os.path.join(sdir, f"reseed_{tag}_a{attempt}.txt"), "w").write(r or "<none>")
    ans = _extract_answer_json(r)
    if not ans:
        return None
    if ans.get("verdict") == "abandon":
        return "abandon", None
    c = ans.get("click") or {}
    if not (isinstance(c.get("x"), (int, float)) and isinstance(c.get("y"), (int, float))):
        return None
    return "add", ((l + float(c["x"]) * cw) / W, (t + float(c["y"]) * ch) / H)


def refine_group_with_sam3(service, group, target_path, frame, W, H, sdir,
                           max_clicks=5, max_attempts=3, region_frac=0.30):
    """Iteratively segment one creature with SAM3, showing the MLLM the mask each
    step. The MLLM returns one of: good (accept), add (one click), reject (discard
    clicks + restart this attempt with a fresh seed), abandon (no creature here).

    ``max_clicks <= 0`` removes the click ceiling so the MLLM alone chooses when
    to accept, reject, or abandon. ``max_attempts`` still bounds explicit restart
    attempts. On exhaustion the best-scoring mask seen is kept. Returns
    (result, trace); result carries ``status`` in
    {accepted, abandoned, exhausted} and ``clicks_used``."""
    seed = [dict(c) for c in group["clicks"]]
    desc = group.get("description", "")
    tag = f"id{group['id']}"
    trace = []
    best = None  # (score, result, clicks)

    for attempt in range(max_attempts):
        grp = {"id": group["id"], "description": desc, "clicks": [dict(c) for c in seed]}
        verdict = None
        duplicate_retries = 0
        duplicate_feedback = ""
        for it in _iteration_indices(max_clicks):
            result = service.group_segment(target_path, [grp])[0]
            result["status"] = "accepted"
            mask = np.asarray(result["mask"]).astype(bool)
            if best is None or float(result["score"]) > best[0]:
                best = (float(result["score"]), result, [dict(c) for c in grp["clicks"]])
            geom = _mask_crop_geom(mask, grp["clicks"], W, H, region_frac)
            up = max(2, int(round(720 / max(geom[2], geom[3]))))
            cpath = os.path.join(sdir, f"refine_{tag}_a{attempt}_it{it}.png")
            _render_mask_crop(frame, mask, grp["clicks"], geom, cpath, up)
            step = {"attempt": attempt, "it": it, "n_clicks": len(grp["clicks"]),
                    "area_px": int(result["area_px"]),
                    "score": round(float(result["score"]), 3)}
            trace.append(step)
            if _click_budget_reached(grp["clicks"], max_clicks):
                step["verdict"] = "click_budget"
                break
            text = (
                f"This is a ZOOMED crop of an underwater seafloor scene that should "
                f"contain: '{desc}'. The CYAN region is SAM3's CURRENT segmentation, "
                f"from the click(s) shown (green dot = foreground/on-creature, "
                f"red X = background/exclude).\n\n"
                f"Assess: does the cyan region tightly and COMPLETELY cover this "
                f"creature's body and ONLY its body (no substrate/background bleed, "
                f"no neighbouring object)?\n"
                f"{duplicate_feedback}\n" + MASK_JUDGE_FMT)
            r = send_claude_request(
                [{"role": "user", "content": [
                    {"type": "image", "image": cpath}, {"type": "text", "text": text}]}],
                model=MODEL, max_tokens=response_token_budget(MODEL, 800))
            open(os.path.join(sdir, f"refine_{tag}_a{attempt}_it{it}.txt"),
                 "w").write(r or "<none>")
            ans = _extract_answer_json(r)
            verdict = (ans or {}).get("verdict")
            if not ans or verdict == "good":
                step["verdict"] = "good" if ans else "parse_fail_accept"
                result["clicks_used"] = grp["clicks"]
                return result, trace
            if verdict == "abandon":
                step["verdict"] = "abandon"
                return _empty_result(grp, H, W, "abandoned"), trace
            if verdict == "reject":
                step["verdict"] = "reject"
                break
            click = ans.get("click") or {}
            if verdict == "add" and (
                    isinstance(click.get("x"), (int, float))
                    and isinstance(click.get("y"), (int, float))
                    and click.get("label") in (0, 1)):
                l, t, cw, ch = geom
                candidate = {
                    "x": (l + float(click["x"]) * cw) / W,
                    "y": (t + float(click["y"]) * ch) / H,
                    "label": int(click["label"]),
                }
                if _duplicate_click(grp["clicks"], candidate):
                    duplicate_retries += 1
                    step["verdict"] = "duplicate_click_retry"
                    duplicate_feedback = (
                        "Your previous requested click duplicated an existing "
                        "click and made no change. Choose a genuinely different "
                        "useful point, or answer good/reject/abandon."
                    )
                    if duplicate_retries >= 2:
                        step["verdict"] = "duplicate_click_no_progress"
                        break
                    continue
                duplicate_retries = 0
                duplicate_feedback = ""
                grp["clicks"].append(candidate)
                step["verdict"] = f"add_label{int(click['label'])}"
            else:
                step["verdict"] = "no_valid_click_accept"
                result["clicks_used"] = grp["clicks"]
                return result, trace

        # attempt ended without 'good' (reject or click budget). Re-seed if rejecting
        # and attempts remain.
        if verdict == "reject" and attempt + 1 < max_attempts:
            rs = _reseed_click(frame, geom, desc, sdir, tag, attempt + 1, W, H)
            if rs is None:
                continue
            if rs[0] == "abandon":
                trace.append({"attempt": attempt, "verdict": "reseed_abandon"})
                return _empty_result(group, H, W, "abandoned"), trace
            seed = [{"x": rs[1][0], "y": rs[1][1], "label": 1}]

    # attempts exhausted: keep best-scoring mask seen
    _score, result, clicks = best
    result["status"] = "exhausted"
    result["clicks_used"] = clicks
    trace.append({"verdict": "exhausted_keep_best",
                  "score": round(float(_score), 3)})
    return result, trace


def run_sam3_and_report(groups, target_path, frame, gt, W, H, sdir, label,
                        refine=False, max_clicks=5, max_attempts=3):
    """Feed the produced click groups into SAM3 point mode, match each output
    mask to the best GT by IoU, render a report panel, and dump metrics.

    When ``refine`` is set, each creature goes through the iterative
    segment->judge->{good,add,reject,abandon} loop (up to ``max_clicks`` clicks
    per attempt, ``max_attempts`` attempts) before the final mask is scored."""
    service = build_sam3_service()
    if refine:
        results, traces = [], {}
        for g in groups:
            res, tr = refine_group_with_sam3(service, g, target_path, frame, W, H,
                                             sdir, max_clicks=max_clicks,
                                             max_attempts=max_attempts)
            results.append(res)
            traces[str(g["id"])] = tr
            print(f"  [sam3-refine] id{g['id']} [{res.get('status','?')}]: "
                  f"{len(res['clicks_used'])} clicks, area {res['area_px']}px, "
                  f"trace {[t.get('verdict','-') for t in tr]}")
    else:
        results = service.group_segment(target_path, groups)
        traces = None
    # match each predicted mask to best GT (greedy by IoU)
    used = set()
    rows = []
    for r in results:
        m = np.asarray(r["mask"]).astype(bool)
        best_iou, best_gt = 0.0, None
        for g in gt:
            if g["id"] in used:
                continue
            i = _iou(m, g["mask"])
            if i > best_iou:
                best_iou, best_gt = i, g["id"]
        if best_gt is not None and best_iou >= 0.3:
            used.add(best_gt)
        rows.append({"creature_id": r["creature_id"],
                     "description": r["description"],
                     "area_px": int(r["area_px"]), "score": round(float(r["score"]), 3),
                     "select_reason": r["select_reason"],
                     "status": r.get("status"),
                     "best_iou": round(best_iou, 3),
                     "matched_gt": best_gt if best_iou >= 0.3 else None})

    _render_sam3_report(frame, gt, results, rows, os.path.join(sdir, "sam3_report.png"),
                        label)
    n_good = sum(1 for x in rows if x["matched_gt"] is not None)
    metrics = {"label": label, "n_pred": len(rows), "n_gt": len(gt),
               "n_matched_iou>=0.3": n_good,
               "mean_iou_matched": round(
                   float(np.mean([x["best_iou"] for x in rows if x["matched_gt"]]))
                   if n_good else 0.0, 3),
               "masks": rows}
    if traces is not None:
        metrics["refine"] = {"max_clicks": max_clicks, "max_attempts": max_attempts,
                             "traces": traces}
    json.dump(metrics, open(os.path.join(sdir, "sam3_metrics.json"), "w"), indent=2)
    print(f"  [sam3{'-refine' if refine else ''}] {label}: {len(rows)} masks, "
          f"{n_good}/{len(gt)} matched GT "
          f"(IoU>=0.3), mean IoU(matched)={metrics['mean_iou_matched']}")
    return metrics


# ----------------------------------------------------------------------------
# Click->mask eval harness: hold the input click FIXED (one fg click at each GT
# centroid) and measure how faithfully a mask-generator variant reproduces the GT.
# This decouples mask quality from detection recall so variants are comparable.
# ----------------------------------------------------------------------------
def _sam3_raw(service, image_path, point_coords=None, point_labels=None, box=None,
              multimask=True):
    """Direct SAM3 predict_inst call returning ALL candidate masks + scores.
    point_coords/box are in pixels. Returns (masks bool [N,H,W], scores [N])."""
    from PIL import Image
    pil = Image.open(image_path).convert("RGB")
    state = service.processor.set_image(pil)
    kw = {"multimask_output": multimask}
    if point_coords is not None:
        kw["point_coords"] = np.asarray(point_coords, dtype=np.float32)
        kw["point_labels"] = np.asarray(point_labels, dtype=np.int64)
    if box is not None:
        kw["box"] = np.asarray(box, dtype=np.float32)
    masks, scores, _ = service.model.predict_inst(state, **kw)
    masks_np = masks.detach().cpu().numpy() if hasattr(masks, "detach") else np.asarray(masks)
    scores_np = scores.detach().cpu().numpy() if hasattr(scores, "detach") else np.asarray(scores)
    if masks_np.ndim == 4:
        masks_np = masks_np[0]
        scores_np = scores_np[0] if scores_np.ndim >= 1 else scores_np
    return masks_np.astype(bool), np.atleast_1d(scores_np)


def _keep_positive_seed_components(mask, clicks):
    """Remove disconnected SAM3 spill while retaining every seeded component.

    Fine branch gaps remain untouched.  With multiple positive clicks, visible
    target pieces separated by occlusion can be retained deliberately by
    placing a positive click on each piece.  If rounding puts every positive
    seed just outside the mask, keep the largest component as a safe fallback.
    """
    mask = np.asarray(mask).astype(bool)
    if not mask.any():
        return mask
    positive = [c for c in clicks if int(c.get("label", 1)) == 1]
    if not positive:
        return mask
    n_labels, labels = cv2.connectedComponents(
        mask.astype(np.uint8), connectivity=8
    )
    if n_labels <= 2:
        return mask
    height, width = mask.shape
    seeded_labels = set()
    for click in positive:
        x = min(width - 1, max(0, int(round(float(click["x"]) * (width - 1)))))
        y = min(height - 1, max(0, int(round(float(click["y"]) * (height - 1)))))
        label = int(labels[y, x])
        if label:
            seeded_labels.add(label)
    if not seeded_labels:
        component_sizes = np.bincount(labels.ravel())
        component_sizes[0] = 0
        seeded_labels.add(int(np.argmax(component_sizes)))
    return np.isin(labels, list(seeded_labels))


def _clean_candidate_components(masks, clicks):
    return np.stack([
        _keep_positive_seed_components(mask, clicks) for mask in masks
    ])


def _select_in_band(masks_np, scores_np):
    """Smallest-in-band selection, mirroring Sam3PointService.group_segment."""
    h, w = masks_np.shape[1], masks_np.shape[2]
    total = float(h * w)
    areas = masks_np.reshape(masks_np.shape[0], -1).sum(axis=1)
    mn = max(200, int(0.001 * total))
    mx = 0.5 * total
    elig = sorted([(int(a), i) for i, a in enumerate(areas) if mn <= a <= mx])
    if elig:
        return elig[0][1], "smallest_in_band"
    above = sorted([(int(a), i) for i, a in enumerate(areas) if a >= mn])
    if above:
        return above[0][1], "smallest_above_min"
    return int(np.argmax(scores_np)), "fallback_highest_score"


def _smallest_valid(masks_np, valid):
    """Smallest-area candidate among those passing the area-plausibility gate.
    Caller must ensure ``valid`` has at least one True."""
    areas = masks_np.reshape(masks_np.shape[0], -1).sum(axis=1)
    elig = sorted((int(areas[i]), i) for i in range(len(areas)) if valid[i])
    return elig[0][1]


def _mk_result(grp, mask, score, reason, status, clicks):
    return {"creature_id": int(grp["id"]), "description": grp.get("description", ""),
            "mask": mask, "score": float(score), "area_px": int(mask.sum()),
            "select_reason": reason, "spatial_match": "click_mode",
            "clicks_used": clicks, "status": status}


def _segment_box(service, grp, image_path, frame, W, H, sdir, region_frac=0.30):
    """Show a zoomed crop with the seed click marked; ask the MLLM for a tight
    bbox around that creature; feed the box to SAM3."""
    tag = f"id{grp['id']}"
    clicks = grp["clicks"]
    geom = _mask_crop_geom(np.zeros((H, W), bool), clicks, W, H, region_frac)
    l, t, cw, ch = geom
    up = max(2, int(round(720 / max(cw, ch))))
    crop = frame[t:t + ch, l:l + cw]
    zoom = cv2.resize(crop, (cw * up, ch * up), interpolation=cv2.INTER_CUBIC)
    for c in clicks:
        px = int((c["x"] * W - l) * up)
        py = int((c["y"] * H - t) * up)
        cv2.drawMarker(zoom, (px, py), (0, 255, 0), cv2.MARKER_CROSS, 22, 2)
    zp = os.path.join(sdir, f"box_{tag}.png")
    cv2.imwrite(zp, zoom)
    text = (
        f"This is a ZOOMED crop of an underwater seafloor scene. A green cross "
        f"marks a creature ('{grp.get('description','')}').\n"
        f"Draw the TIGHTEST axis-aligned bounding box that contains the WHOLE "
        f"body of that one creature and nothing else.\n\n"
        '<answer>{"box": {"x0": <f>, "y0": <f>, "x1": <f>, "y1": <f>}}</answer>\n'
        '<answer>{"abandon": true}</answer>\n'
        'Coordinates NORMALIZED [0,1] within THIS crop (x0,y0=top-left corner, '
        'x1,y1=bottom-right corner).')
    r = send_claude_request(
        [{"role": "user", "content": [
            {"type": "image", "image": zp}, {"type": "text", "text": text}]}],
        model=MODEL, max_tokens=response_token_budget(MODEL, 600))
    open(os.path.join(sdir, f"box_{tag}.txt"), "w").write(r or "<none>")
    ans = _extract_answer_json(r) or {}
    if ans.get("abandon") is True or "box" not in ans:
        return _mk_result(grp, np.zeros((H, W), bool), 0.0, "abandoned", "abandoned", [])
    b = ans["box"]
    bx0 = l + float(b["x0"]) * cw
    by0 = t + float(b["y0"]) * ch
    bx1 = l + float(b["x1"]) * cw
    by1 = t + float(b["y1"]) * ch
    box = [min(bx0, bx1), min(by0, by1), max(bx0, bx1), max(by0, by1)]
    pc = np.array([[c["x"] * W, c["y"] * H] for c in clicks], dtype=np.float32)
    pl = np.array([c["label"] for c in clicks], dtype=np.int64)
    masks, scores = _sam3_raw(service, image_path, point_coords=pc, point_labels=pl,
                              box=box, multimask=True)
    idx, reason = _select_in_band(masks, scores)
    return _mk_result(grp, masks[idx], scores[idx], f"box+{reason}", "box",
                      clicks + [{"box": [round(v, 1) for v in box]}])


def _segment_multimask(service, grp, image_path, frame, W, H, sdir, region_frac=0.30):
    """Run SAM3 on the seed click, show the MLLM all 3 candidate masks, let it
    pick the best (or none)."""
    tag = f"id{grp['id']}"
    clicks = grp["clicks"]
    pc = np.array([[c["x"] * W, c["y"] * H] for c in clicks], dtype=np.float32)
    pl = np.array([c["label"] for c in clicks], dtype=np.int64)
    masks, scores = _sam3_raw(service, image_path, point_coords=pc, point_labels=pl,
                              multimask=True)
    geom = _mask_crop_geom(masks[int(np.argmax(scores))], clicks, W, H, region_frac)
    zp = os.path.join(sdir, f"multimask_{tag}.png")
    bp = os.path.join(sdir, f"multimask_{tag}_binary.png")
    rp = os.path.join(sdir, f"multimask_{tag}_review.png")
    _render_candidates(
        frame, masks, scores, clicks, geom, zp,
        binary_path=bp, review_path=rp,
    )
    text = (
        f"This review sheet has two aligned rows for candidate masks #0,#1,#2 "
        f"for '{grp.get('description','')}'. TOP = cyan overlay; BOTTOM = exact "
        f"binary truth where WHITE pixels are inside and BLACK pixels are outside. "
        f"Pick the index whose WHITE region most tightly and COMPLETELY covers the "
        f"creature's body and ONLY its body. If none is acceptable, choose -1.\n\n"
        '<answer>{"choice": 0}</answer> (or 1, 2, or -1)')
    r = send_claude_request(
        [{"role": "user", "content": [
            {"type": "image", "image": rp},
            {"type": "text", "text": text}]}],
        model=MODEL, max_tokens=response_token_budget(MODEL, 400))
    open(os.path.join(sdir, f"multimask_{tag}.txt"), "w").write(r or "<none>")
    ans = _extract_answer_json(r) or {}
    choice = ans.get("choice")
    if choice not in (0, 1, 2):
        if choice == -1:
            return _mk_result(grp, np.zeros((H, W), bool), 0.0, "abandoned", "abandoned", [])
        choice, _ = _select_in_band(masks, scores)  # parse fail -> heuristic
        return _mk_result(grp, masks[choice], scores[choice], "mllm_pick_fail_band",
                          "multimask", clicks)
    return _mk_result(grp, masks[choice], scores[choice], "mllm_pick", "multimask",
                      clicks)


def _render_candidates(
    frame, masks, scores, clicks, geom, path, up=None, binary_path=None,
    review_path=None,
):
    """Side-by-side panels of each SAM3 candidate mask (cyan) on a zoomed crop,
    with the current click(s) drawn (green=fg, red X=bg).

    When ``binary_path`` is supplied, also write an unambiguous companion where
    white is inside the mask and black is outside. The overlay alone can fool a
    vision model into treating visible branches beneath a translucent mask as
    the selected pixels.
    """
    l, t, cw, ch = geom
    if up is None:
        up = max(2, int(round(360 / max(cw, ch))))
    H, W = frame.shape[:2]
    panels = []
    binary_panels = []
    for i in range(masks.shape[0]):
        z = cv2.resize(frame[t:t + ch, l:l + cw].copy(), (cw * up, ch * up),
                       interpolation=cv2.INTER_CUBIC)
        mc = cv2.resize(masks[i][t:t + ch, l:l + cw].astype(np.uint8),
                        (cw * up, ch * up), interpolation=cv2.INTER_NEAREST).astype(bool)
        ov = z.copy()
        ov[mc] = (255, 255, 0)
        z = cv2.addWeighted(ov, 0.45, z, 0.55, 0)
        for c in clicks:
            px, py = int((c["x"] * W - l) * up), int((c["y"] * H - t) * up)
            if c.get("label", 1) == 1:
                cv2.circle(z, (px, py), 5, (0, 255, 0), -1)
            else:
                cv2.drawMarker(z, (px, py), (0, 0, 255), cv2.MARKER_TILTED_CROSS, 14, 2)
        cv2.rectangle(z, (0, 0), (z.shape[1], 24), (0, 0, 0), -1)
        cv2.putText(z, f"#{i} area{int(masks[i].sum())} s{scores[i]:.2f}", (4, 18),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
        panels.append(z)
        if binary_path is not None:
            binary = np.zeros_like(z)
            binary[mc] = (255, 255, 255)
            for c in clicks:
                px = int((c["x"] * W - l) * up)
                py = int((c["y"] * H - t) * up)
                if c.get("label", 1) == 1:
                    cv2.circle(binary, (px, py), 6, (0, 255, 0), -1)
                else:
                    cv2.drawMarker(
                        binary, (px, py), (0, 0, 255),
                        cv2.MARKER_TILTED_CROSS, 16, 2,
                    )
            cv2.rectangle(binary, (0, 0), (binary.shape[1], 24), (0, 0, 0), -1)
            cv2.putText(
                binary, f"#{i} WHITE = MASK", (4, 18),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1,
                cv2.LINE_AA,
            )
            binary_panels.append(binary)
    overlay_sheet = np.hstack(panels)
    cv2.imwrite(path, overlay_sheet)
    if binary_path is not None:
        binary_sheet = np.hstack(binary_panels)
        cv2.imwrite(binary_path, binary_sheet)
        if review_path is not None:
            cv2.imwrite(review_path, np.vstack([overlay_sheet, binary_sheet]))


MM_JUDGE_FMT = (
    'Output: brief reasoning then EXACTLY ONE trailing tag, one of:\n'
    '<answer>{"verdict": "good", "choice": <0|1|2>}</answer>'
    '  -- candidate #choice tightly+completely covers the body; ACCEPT it\n'
    '<answer>{"verdict": "add", "click": {"x": <f>, "y": <f>, "label": 0 or 1}}'
    '</answer>  -- none is complete: add a click (1=include missed part; '
    '0=exclude a specific cyan spill, touching neighbour, or background region)\n'
    '<answer>{"verdict": "reject"}</answer>'
    '  -- all candidates are hopeless/wrong object; DISCARD clicks & restart\n'
    '<answer>{"verdict": "abandon"}</answer>'
    '  -- there is NO real creature here; produce no mask\n'
    'x,y NORMALIZED [0,1] within ONE panel (the panels are identical crops). '
    'Use label 0 whenever cyan crosses the intended target boundary; put it '
    'inside the unwanted cyan region and never on another desired part of the target. '
    'GOOD IS A STRICT, NON-COMPARATIVE VERDICT: do not choose the least-bad '
    'candidate. Fine enclosed spaces between branches MAY be filled as part of '
    "one object's silhouette. What is forbidden is WHITE extending outside that "
    'object onto a visually separable neighbouring coral/plant, substrate lobe, '
    'or disconnected spill. Put label 0 on a WHITE pixel in that unwanted '
    'external/neighbor region. A negative click excludes that location from this '
    'target mask; it does not claim the neighbouring life is non-biological.'
)


def refine_group_mm(service, group, target_path, frame, W, H, sdir,
                    max_clicks=5, max_attempts=3, region_frac=0.30,
                    max_area_frac=0.60, strict_quality=False):
    """Combined best generator: each iteration runs SAM3 multimask, shows the MLLM
    all 3 candidates, and the MLLM either picks one (good), adds a click, rejects
    (restart), or abandons. Unifies the multimask-pick winner with the refine
    loop's safety mechanism. Returns (result, trace).

    ``max_area_frac`` only rejects DEGENERATE near-whole-frame candidates: a click
    on busy substrate can make SAM3 return a frame-spanning "region" mask. We keep
    this cap loose (default 0.60) on purpose -- a creature close to the camera can
    legitimately fill much of the frame, and creature size overlaps substrate-grab
    size, so area can't separate them. The content-based per-click verify pass
    (``verify_clicks``) is the real stray filter; this only kills the pathological
    whole-frame mask. If NO candidate is small enough, the click is abandoned."""
    seed = [dict(c) for c in group["clicks"]]
    desc = group.get("description", "")
    tag = f"id{group['id']}"
    trace = []
    best = None  # (score, result, clicks)
    area_cap = max_area_frac * float(H * W)

    for attempt in range(max_attempts):
        clicks = [dict(c) for c in seed]
        geom = None
        verdict = None
        duplicate_retries = 0
        duplicate_feedback = ""
        for it in _iteration_indices(max_clicks):
            pc = np.array([[c["x"] * W, c["y"] * H] for c in clicks], dtype=np.float32)
            pl = np.array([c["label"] for c in clicks], dtype=np.int64)
            masks, scores = _sam3_raw(service, target_path, point_coords=pc,
                                      point_labels=pl, multimask=True)
            masks = _clean_candidate_components(masks, clicks)
            areas = masks.reshape(masks.shape[0], -1).sum(axis=1)
            valid = areas <= area_cap
            if not valid.any():
                # every candidate spans implausibly much frame -> substrate click
                trace.append({"attempt": attempt, "it": it, "verdict": "no_valid_band",
                              "areas": [int(a) for a in areas]})
                return _empty_result(group, H, W, "abandoned"), trace
            bi = int(np.argmax(np.where(valid, scores, -np.inf)))
            if best is None or float(scores[bi]) > best[0]:
                best = (float(scores[bi]),
                        _mk_result(group, masks[bi], scores[bi], "mm_best_score",
                                   "exhausted", [dict(c) for c in clicks]))
            geom = _mask_crop_geom(masks[bi], clicks, W, H, region_frac)
            cpath = os.path.join(sdir, f"mm_{tag}_a{attempt}_it{it}.png")
            bpath = os.path.join(
                sdir, f"mm_{tag}_a{attempt}_it{it}_binary.png"
            )
            rpath = os.path.join(
                sdir, f"mm_{tag}_a{attempt}_it{it}_review.png"
            )
            _render_candidates(
                frame, masks, scores, clicks, geom, cpath,
                binary_path=bpath, review_path=rpath,
            )
            step = {"attempt": attempt, "it": it, "n_clicks": len(clicks)}
            trace.append(step)
            click_budget_reached = _click_budget_reached(clicks, max_clicks)
            if click_budget_reached and not strict_quality:
                ci = _smallest_valid(masks, valid)
                step["verdict"] = "click_budget"
                return _mk_result(group, masks[ci], scores[ci], "click_budget_band",
                                  "accepted", clicks), trace
            text = (
                f"The FIRST image is the raw full frame. The discovery description "
                f"'{desc}' is an UNTRUSTED HYPOTHESIS, not proof that the proposed "
                f"pixels form one complete object. The SECOND review sheet has two "
                f"aligned rows for three SAM3 masks "
                f"#0,#1,#2 for '{desc}'. TOP = cyan overlay on identical crops "
                f"(green dot = foreground/include, red X = background/exclude). "
                f"BOTTOM = exact binary truth: WHITE pixels are "
                f"inside the mask and BLACK pixels are outside. Judge the WHITE "
                f"pixels, not biological structure still visible underneath the "
                f"translucent cyan. Fine enclosed spaces between branches may be "
                f"filled; judge the complete outer silhouette of the individual "
                f"object. The "
                f"accepted mask must cover "
                f"the complete visible target while "
                f"excluding substrate and visually separable neighbouring life. "
                f"One accepted mask must be exactly ONE complete biological "
                f"identity. A small branch/patch of a larger continuous branching "
                f"system is incomplete: add label 1 on the missed same-object "
                f"extent. A union of separate foreground/background colonies or "
                f"different branching systems is merged: add label 0 inside the "
                f"unwanted identity. Do this even when both identities are living "
                f"or share the same taxon. "
                f"On-screen logos, lettering, timestamps, grid marks, and other "
                f"video/UI overlays are never part of a biological mask. Reject "
                f"them or add a label-0 click on an included overlay spill. "
                f"Do not accept a broad merged region merely because it contains "
                f"the target. Never accept the closest candidate if it includes "
                f"pieces of a visually separable neighbouring coral/plant or a "
                f"substrate spill. If every candidate "
                f"spills into one specific unwanted "
                f"region, add a label-0 click inside that spill; if the target itself "
                f"is incomplete, add label 1 on the missed part. "
                f"{duplicate_feedback}"
                + (
                    "NO CLICK BUDGET REMAINS: answer good only for a genuinely "
                    "acceptable candidate; otherwise answer reject. Do not ask "
                    "for another click.\n\n"
                    if click_budget_reached else "\n\n"
                ) +
                MM_JUDGE_FMT)
            r = send_claude_request(
                [{"role": "user", "content": [
                    {"type": "image", "image": target_path},
                    {"type": "image", "image": rpath},
                    {"type": "text", "text": text}]}],
                model=MODEL, max_tokens=response_token_budget(MODEL, 800))
            open(os.path.join(sdir, f"mm_{tag}_a{attempt}_it{it}.txt"),
                 "w").write(r or "<none>")
            ans = _extract_answer_json(r) or {}
            verdict = ans.get("verdict")
            if verdict == "good":
                ci = ans.get("choice")
                if ci not in (0, 1, 2) or not valid[ci]:
                    ci = _smallest_valid(masks, valid)  # snap to a plausible mask
                step["verdict"] = f"good#{ci}"
                return _mk_result(group, masks[ci], scores[ci], "mllm_pick",
                                  "accepted", clicks), trace
            if verdict == "abandon":
                step["verdict"] = "abandon"
                return _empty_result(group, H, W, "abandoned"), trace
            if verdict == "reject":
                step["verdict"] = "reject"
                break
            click = ans.get("click") or {}
            if verdict == "add" and not click_budget_reached and (
                    isinstance(click.get("x"), (int, float))
                    and isinstance(click.get("y"), (int, float))
                    and click.get("label") in (0, 1)):
                l, t, cw, ch = geom
                candidate = {
                    "x": (l + float(click["x"]) * cw) / W,
                    "y": (t + float(click["y"]) * ch) / H,
                    "label": int(click["label"]),
                }
                if _duplicate_click(clicks, candidate):
                    duplicate_retries += 1
                    step["verdict"] = "duplicate_click_retry"
                    duplicate_feedback = (
                        "Your previous requested click duplicated an existing "
                        "click and made no change. Choose a genuinely different "
                        "useful point, or answer good/reject/abandon. "
                    )
                    if duplicate_retries >= 2:
                        step["verdict"] = "duplicate_click_no_progress"
                        verdict = "reject"
                        break
                    continue
                duplicate_retries = 0
                duplicate_feedback = ""
                clicks.append(candidate)
                step["verdict"] = f"add_label{int(click['label'])}"
            else:
                if strict_quality:
                    step["verdict"] = (
                        "click_budget_reject" if click_budget_reached
                        else "parse_fail_reject"
                    )
                    verdict = "reject"
                    break
                ci = _smallest_valid(masks, valid)
                step["verdict"] = "parse_fail_band"
                return _mk_result(group, masks[ci], scores[ci], "parse_fail_band",
                                  "accepted", clicks), trace

        if verdict == "reject" and attempt + 1 < max_attempts and geom is not None:
            rs = _reseed_click(frame, geom, desc, sdir, tag, attempt + 1, W, H)
            if rs is None:
                continue
            if rs[0] == "abandon":
                trace.append({"attempt": attempt, "verdict": "reseed_abandon"})
                return _empty_result(group, H, W, "abandoned"), trace
            seed = [{"x": rs[1][0], "y": rs[1][1], "label": 1}]

    if strict_quality:
        trace.append({"verdict": "exhausted_no_verified_mask"})
        return _empty_result(group, H, W, "abandoned"), trace
    _score, result = best
    trace.append({"verdict": "exhausted_keep_best", "score": round(_score, 3)})
    return result, trace


def _sam3_raw_crop(service, frame, clicks, geom, up, sdir, tag, multimask=True):
    """Run SAM3 on a ZOOMED, upscaled crop (geom=l,t,cw,ch) so a small creature
    fills the field of view. ``clicks`` are full-frame normalized. Returns
    (masks_fullframe[N,H,W] bool, scores) -- candidate masks mapped back to
    full-frame coordinates."""
    H, W = frame.shape[:2]
    l, t, cw, ch = geom
    crop = frame[t:t + ch, l:l + cw]
    zoom = cv2.resize(crop, (cw * up, ch * up), interpolation=cv2.INTER_CUBIC)
    zp = os.path.join(sdir, f"segcrop_{tag}.png")
    cv2.imwrite(zp, zoom)
    pc = np.array([[(c["x"] * W - l) * up, (c["y"] * H - t) * up] for c in clicks],
                  dtype=np.float32)
    pl = np.array([c["label"] for c in clicks], dtype=np.int64)
    masks, scores = _sam3_raw(service, zp, point_coords=pc, point_labels=pl,
                              multimask=multimask)
    full = np.zeros((masks.shape[0], H, W), bool)
    for i in range(masks.shape[0]):
        # linear downscale + 0.5 threshold preserves edges better than NEAREST
        small = cv2.resize(masks[i].astype(np.float32), (cw, ch),
                           interpolation=cv2.INTER_AREA) >= 0.5
        full[i, t:t + ch, l:l + cw] = small
    return full, scores


ZOOM_JUDGE_FMT = (
    'A real creature IS present here -- it was already verified. You MUST end with '
    'a mask; NEVER claim there is no creature.\n'
    'Output: brief reasoning then EXACTLY ONE trailing tag:\n'
    '<answer>{"verdict": "good", "choice": <0|1|2>}</answer>'
    '  -- candidate #choice tightly+completely covers the body; ACCEPT it\n'
    '<answer>{"verdict": "add", "click": {"x": <f>, "y": <f>, "label": 0 or 1}}'
    '</answer>  -- none complete: add ONE click (1=include a missed part; '
    '0=exclude a specific cyan spill or touching neighbour) to improve the mask\n'
    '<answer>{"verdict": "reject"}</answer>  -- no candidate is acceptable and '
    'no further useful click can be made\n'
    'x,y NORMALIZED [0,1] within ONE panel (panels are identical crops). '
    'GOOD IS STRICT: small enclosed spaces between fine branches may be filled, '
    'but the mask must cover the complete individual object and no pieces of a '
    'visually separable neighbouring coral/plant. Add label 0 on a WHITE pixel '
    'belonging to the unwanted neighbour or external spill; never put it on the target.'
)


def refine_group_mm_zoom(service, group, target_path, frame, W, H, sdir,
                         max_clicks=4, max_attempts=1, region_frac=0.30,
                         seg_crop_frac=0.30, max_area_frac=0.85,
                         strict_quality=False):
    """Mask generator that segments on a ZOOMED CROP around the (already
    creature-verified) seed click and NEVER abandons -- it always returns the best
    body-covering mask. Fixes two e2e failure modes found in diagnostics:
    (a) ``refine_mm`` abandoning thin/small creatures (empty mask -> IoU 0), and
    (b) full-frame point prompts mislocalizing tiny (~1000px) creatures onto
    adjacent substrate. Same multimask + MLLM-pick + add-click loop, but on the
    crop and with the abandon/reject exits replaced by 'keep best in-band mask'.
    Signature mirrors ``refine_group_mm`` so it is a drop-in alternative."""
    seed = [dict(c) for c in group["clicks"]]
    desc = group.get("description", "")
    tag = f"id{group['id']}z"
    seg_geom = _mask_crop_geom(np.zeros((H, W), bool), seed, W, H, seg_crop_frac)
    l, t, cw, ch = seg_geom
    up = max(2, int(round(720 / max(cw, ch))))
    area_cap = max_area_frac * float(cw * ch)
    trace = []
    best = None
    clicks = [dict(c) for c in seed]
    duplicate_retries = 0
    duplicate_feedback = ""
    for it in _iteration_indices(max_clicks):
        masks, scores = _sam3_raw_crop(service, frame, clicks, seg_geom, up, sdir,
                                       f"{tag}_it{it}")
        masks = _clean_candidate_components(masks, clicks)
        areas = masks.reshape(masks.shape[0], -1).sum(axis=1)
        valid = areas <= area_cap
        if not valid.any():                      # nothing under cap -> allow all
            valid = np.ones(len(areas), bool)
        bi = int(np.argmax(np.where(valid, scores, -np.inf)))
        if best is None or float(scores[bi]) > best[0]:
            best = (float(scores[bi]), _mk_result(group, masks[bi], scores[bi],
                    "zoom_best", "accepted", [dict(c) for c in clicks]))
        cpath = os.path.join(sdir, f"mmz_{tag}_it{it}.png")
        bpath = os.path.join(sdir, f"mmz_{tag}_it{it}_binary.png")
        rpath = os.path.join(sdir, f"mmz_{tag}_it{it}_review.png")
        _render_candidates(
            frame, masks, scores, clicks, seg_geom, cpath,
            binary_path=bpath, review_path=rpath,
        )
        trace.append({"it": it, "n_clicks": len(clicks)})
        click_budget_reached = _click_budget_reached(clicks, max_clicks)
        if click_budget_reached and not strict_quality:
            ci = _smallest_valid(masks, valid)
            return _mk_result(group, masks[ci], scores[ci], "zoom_click_budget",
                              "accepted", clicks), trace
        text = (
            f"This review sheet has aligned rows for masks #0,#1,#2 for '{desc}'. "
            f"TOP = cyan overlay (green foreground, red-X background); BOTTOM = "
            f"exact binary truth where WHITE is inside and BLACK is outside. "
            f"Judge WHITE pixels. Fine enclosed spaces between branches may be "
            f"filled, but WHITE must not include a separable neighbouring plant/"
            f"coral or substrate outside the target's silhouette. "
            f"{duplicate_feedback}"
            + (
                "NO CLICK BUDGET REMAINS: answer good only for a genuinely "
                "acceptable candidate; otherwise do not request another click.\n\n"
                if click_budget_reached else "\n\n"
            )
            + ZOOM_JUDGE_FMT
        )
        r = send_claude_request(
            [{"role": "user", "content": [
                {"type": "image", "image": rpath},
                {"type": "text", "text": text}]}],
            model=MODEL, max_tokens=response_token_budget(MODEL, 700))
        open(os.path.join(sdir, f"mmz_{tag}_it{it}.txt"), "w").write(r or "<none>")
        ans = _extract_answer_json(r) or {}
        verdict = ans.get("verdict")
        trace[-1]["verdict"] = verdict
        if verdict == "good":
            ci = ans.get("choice")
            if ci not in (0, 1, 2) or not valid[ci]:
                ci = _smallest_valid(masks, valid)
            return _mk_result(group, masks[ci], scores[ci], "zoom_mllm_pick",
                              "accepted", clicks), trace
        click = ans.get("click") or {}
        if verdict == "add" and not click_budget_reached and \
                isinstance(click.get("x"), (int, float)) and \
                isinstance(click.get("y"), (int, float)) and click.get("label") in (0, 1):
            candidate = {
                "x": (l + float(click["x"]) * cw) / W,
                "y": (t + float(click["y"]) * ch) / H,
                "label": int(click["label"]),
            }
            if _duplicate_click(clicks, candidate):
                duplicate_retries += 1
                trace[-1]["verdict"] = "duplicate_click_retry"
                duplicate_feedback = (
                    "Your previous requested click duplicated an existing click "
                    "and made no change. Choose a genuinely different useful "
                    "point, or answer good/reject. "
                )
                if duplicate_retries >= 2:
                    trace[-1]["verdict"] = "duplicate_click_no_progress"
                    if strict_quality:
                        return _empty_result(group, H, W, "abandoned"), trace
                    _s, result = best
                    return result, trace
                continue
            duplicate_retries = 0
            duplicate_feedback = ""
            clicks.append(candidate)
        elif strict_quality:
            trace[-1]["verdict"] = (
                "click_budget_reject" if click_budget_reached
                else "strict_reject"
            )
            return _empty_result(group, H, W, "abandoned"), trace
        else:                                     # reject/abandon/parse-fail -> keep best
            ci = _smallest_valid(masks, valid)
            return _mk_result(group, masks[ci], scores[ci], "zoom_keepbest",
                              "accepted", clicks), trace
    if strict_quality:
        return _empty_result(group, H, W, "abandoned"), trace
    _s, result = best
    return result, trace


def refine_group_mm_hybrid(service, group, target_path, frame, W, H, sdir,
                           max_clicks=5, max_attempts=3, min_area_px=200,
                           strict_quality=False, zoom_crop_frac=0.30):
    """Best of both: run the full-frame ``refine_group_mm`` (which gives tight
    masks, IoU(det) ~0.91, on creatures it handles), and ONLY when it abandons or
    returns a degenerate/empty mask (< ``min_area_px``) fall back to the zoomed
    generator to recover that detection. This recovers the abandon-type losses
    (e.g. thin brittle-stars) without the ~0.07 IoU(det) hit that pure-zoom pays
    on every mask. Returns (result, trace)."""
    res, tr = refine_group_mm(service, group, target_path, frame, W, H, sdir,
                              max_clicks=max_clicks, max_attempts=max_attempts,
                              strict_quality=strict_quality)
    area = int(np.asarray(res["mask"]).sum())
    if res.get("status") == "abandoned" or area < min_area_px:
        zres, ztr = refine_group_mm_zoom(service, group, target_path, frame, W, H,
                                         sdir, max_clicks=max_clicks,
                                         strict_quality=strict_quality,
                                         seg_crop_frac=zoom_crop_frac)
        if int(np.asarray(zres["mask"]).sum()) >= min_area_px:
            zres["select_reason"] = "hybrid_zoom/" + zres.get("select_reason", "")
            return zres, tr + [{"hybrid": "zoom_recover",
                                "fullframe_area": area}] + ztr
    return res, tr


MASK_VERIFY_FMT = (
    'Output: brief reasoning then EXACTLY ONE trailing tag. ALWAYS include '
    '"confidence" -- your probability (0.0-1.0) that this is a HIGH-QUALITY, tight '
    'segmentation of the described real life form (1.0 = near-perfect organism '
    'mask, 0.0 = wrong/background-filled mask):\n'
    '<answer>{"keep": true, "confidence": <0.0-1.0>, '
    '"complete_identity": true, "single_identity": true}</answer>   -- WHITE '
    'follows the complete intended life form closely, excludes background/'
    'neighbours, and contains exactly one identity/depth layer\n'
    '<answer>{"keep": false, "confidence": <0.0-1.0>, '
    '"complete_identity": <true|false>, "single_identity": <true|false>, '
    '"failure": "fragment|merge|wrong|background", '
    '"repair_click": {"x": <float>, "y": <float>, "label": 0 or 1}}</answer>  '
    '-- WHITE is only a '
    'piece of a larger identity, merges identities/depth layers, targets the '
    'wrong object, or is mostly non-biological background. Include repair_click '
    'only when one unambiguous click can improve this SAME target: label 1 on a '
    'missed continuation for fragment; label 0 inside wrongly included pixels '
    'for merge/background. Omit repair_click for wrong/unsupported targets. '
    'Repair x,y are NORMALIZED [0,1] in the FULL FRAME.'
)


def _accept_mask_verdict(ans, *, strict_identity=False):
    """Interpret one mask verdict, optionally requiring explicit identity QA."""
    if strict_identity:
        return (
            ans.get("keep") is True
            and ans.get("complete_identity") is True
            and ans.get("single_identity") is True
        )
    return ans.get("keep") is not False


def _mask_quality_repair_click(ans):
    """Return a semantically valid full-frame repair click, if supplied."""
    failure = str(ans.get("failure", "")).strip().lower()
    click = ans.get("repair_click") or {}
    if failure not in {"fragment", "merge", "background"}:
        return None
    expected_label = 1 if failure == "fragment" else 0
    if (
        not isinstance(click.get("x"), (int, float))
        or not isinstance(click.get("y"), (int, float))
        or int(click.get("label", -1)) != expected_label
        or not 0.0 <= float(click["x"]) <= 1.0
        or not 0.0 <= float(click["y"]) <= 1.0
    ):
        return None
    return {
        "x": float(click["x"]),
        "y": float(click["y"]),
        "label": expected_label,
    }


def verify_masks(results, frame, W, H, sdir, model=None, region_frac=0.22,
                 upscale=6, tag_prefix="vm", allow_all_life=False,
                 strict_identity=False):
    """Post-mask life and boundary-quality check with overlay + binary truth."""
    model = model or MODEL
    kept, dropped = [], []
    for r in results:
        m = np.asarray(r["mask"]).astype(bool)
        if not m.any():
            r["creature_confidence"] = 0.0
            kept.append(r)                       # empty handled elsewhere
            continue
        review_clicks = [dict(c) for c in r.get("clicks_used", [])]
        geom = _mask_crop_geom(m, review_clicks, W, H, region_frac)
        cpath = os.path.join(sdir, f"{tag_prefix}_id{r.get('creature_id', 0)}.png")
        bpath = os.path.join(
            sdir, f"{tag_prefix}_id{r.get('creature_id', 0)}_binary.png"
        )
        review_path = os.path.join(
            sdir, f"{tag_prefix}_id{r.get('creature_id', 0)}_review.png"
        )
        raw_context_path = os.path.join(
            sdir, f"{tag_prefix}_id{r.get('creature_id', 0)}_raw.png"
        )
        context_path = os.path.join(
            sdir, f"{tag_prefix}_id{r.get('creature_id', 0)}_context.png"
        )
        _render_mask_crop(frame, m, review_clicks, geom, cpath, upscale)
        _render_binary_mask_crop(m, geom, bpath, upscale)
        overlay_review = cv2.imread(cpath)
        binary_review = cv2.imread(bpath)
        if overlay_review is None or binary_review is None:
            raise RuntimeError("could not render mask quality review sheet")
        cv2.imwrite(review_path, np.vstack([overlay_review, binary_review]))
        full_overlay = frame.copy()
        tinted = frame.copy()
        tinted[m] = (255, 255, 0)
        full_overlay = cv2.addWeighted(tinted, 0.42, full_overlay, 0.58, 0)
        contours, _ = cv2.findContours(
            m.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        cv2.drawContours(full_overlay, contours, -1, (255, 255, 0), 3, cv2.LINE_AA)
        cv2.imwrite(raw_context_path, frame)
        cv2.imwrite(context_path, full_overlay)
        desc = r.get("description", "the masked region")
        mask_y, mask_x = np.where(m)
        bbox_left = int(mask_x.min())
        bbox_right = int(mask_x.max())
        bbox_top = int(mask_y.min())
        bbox_bottom = int(mask_y.max())
        touched_edges = [
            edge
            for edge, touched in (
                ("left", bbox_left == 0),
                ("right", bbox_right == W - 1),
                ("top", bbox_top == 0),
                ("bottom", bbox_bottom == H - 1),
            )
            if touched
        ]
        edge_fact = ",".join(touched_edges) if touched_edges else "none"
        geometry_fact = (
            "DETERMINISTIC BINARY-MASK GEOMETRY: normalized bbox "
            f"x={bbox_left / max(1, W - 1):.3f}.."
            f"{bbox_right / max(1, W - 1):.3f}, "
            f"y={bbox_top / max(1, H - 1):.3f}.."
            f"{bbox_bottom / max(1, H - 1):.3f}; "
            f"touches frame edge(s)={edge_fact}. These facts are computed from "
            "WHITE pixels and override any visual guess about location or frame "
            "clipping. "
        )
        repair_history = r.get("postverify_repair_history") or []
        if repair_history:
            repair_summary = ", ".join(
                f"round {item.get('round')}: {item.get('failure', 'unknown')} "
                f"at ({float((item.get('click') or {}).get('x', 0.0)):.3f}, "
                f"{float((item.get('click') or {}).get('y', 0.0)):.3f}, "
                f"label={int((item.get('click') or {}).get('label', -1))})"
                for item in repair_history
            )
            repair_fact = (
                "REPAIR HISTORY FOR THIS SAME PROPOSAL: " + repair_summary + ". "
                "If the latest mask is still invalid and SAM3 is cycling between "
                "fragment and merge failures, or no genuinely new actionable "
                "correction remains, reject it WITHOUT repair_click so the system "
                "can abandon this proposal and continue searching. Do not repeat "
                "or slightly move an earlier click merely to keep trying. This is "
                "a no-progress decision, not permission to accept a bad mask. "
            )
        else:
            repair_fact = ""
        if allow_all_life:
            subject = (
                "real, distinct marine life: an animal, coherent animal colony, "
                "sponge, coral, macroalga, seagrass, or other living organism"
            )
        else:
            subject = "a real, distinct living animal"
        text = (
            "The FIRST image is the raw full frame and is the coordinate reference. "
            "The SECOND image is the same full frame with the candidate in cyan. "
            "The THIRD review sheet has two aligned rows for one zoomed candidate "
            f"mask for '{desc}'. TOP = cyan overlay. BOTTOM = exact binary truth: WHITE pixels "
            "are inside the mask and BLACK pixels are outside. "
            + geometry_fact
            + repair_fact
            + "Cyan/WHITE having a smooth, coherent silhouette is NOT evidence "
            "that an organism exists: SAM3 can create convincing blobs from "
            "water, shadow, haze, or substrate. The untouched FIRST image must "
            "show biological texture, branching, anatomy, or a credible natural "
            "boundary at the exact mask location. If the hypothesis says the "
            "target is frame-clipped but touches frame edge(s)=none, reject it "
            "as wrong/background. "
            f"Decide whether WHITE tightly segments {subject}. The discovery description is an "
            "UNTRUSTED HYPOTHESIS; infer the true biological identity and its full "
            "extent from the images. Judge mask membership, not life "
            "still visible beneath translucent cyan. For branching organisms, "
            "fine enclosed inter-branch spaces may be WHITE as part of the outer "
            "silhouette. Reject masks that are mostly background, extend well "
            "outside the target silhouette, merge visually separable neighbours "
            "or foreground/background depth layers, "
            "or select the wrong object, even if some real "
            "life is visible in the crop. Reject a small branch, appendage, or "
            "patch when the same organism/colony visibly continues outside WHITE; "
            "that is an incomplete fragment, not a valid separate target. Reject "
            "a union of two distinct branching systems even when both are life or "
            "the same taxon. Trace the WHITE boundary around the whole candidate: "
            "every non-frame-edge termination must coincide with a true visible "
            "outer boundary or a real occluder. If WHITE cuts through visually "
            "continuous branches of the same colony, complete_identity must be "
            "false. If a path through WHITE crosses an occlusion/depth transition "
            "or joins distinct branch systems, single_identity must be false. "
            "Set keep=true only when BOTH identity fields are true. On-screen "
            "logos, lettering, timestamps, "
            "grid marks, and other video/UI overlays are non-biological; reject "
            "a candidate that masks them instead of the described organism. Any "
            "green dot is an include click and "
            "must lie on the intended target; any red X is an exclude click and "
            "must lie outside it. "
            "\n\n" + MASK_VERIFY_FMT)
        resp = send_claude_request(
            [{"role": "user", "content": [
                {"type": "image", "image": raw_context_path},
                {"type": "image", "image": context_path},
                {"type": "image", "image": review_path},
                {"type": "text", "text": text}]}],
            model=model, max_tokens=response_token_budget(model, 500))
        open(os.path.join(sdir, f"{tag_prefix}_id{r.get('creature_id', 0)}.txt"),
             "w").write(resp or "<none>")
        ans = _extract_answer_json(resp) or {}
        conf = ans.get("confidence")
        try:
            conf = min(1.0, max(0.0, float(conf)))
        except (TypeError, ValueError):
            conf = None                          # model omitted it
        complete_identity = ans.get("complete_identity") is True
        single_identity = ans.get("single_identity") is True
        keep_flag = _accept_mask_verdict(
            ans, strict_identity=strict_identity
        )
        failure = str(ans.get("failure", "")).strip().lower()
        r["mask_complete_identity"] = complete_identity
        r["mask_single_identity"] = single_identity
        r["mask_quality_failure"] = (
            failure if failure in {"fragment", "merge", "wrong", "background"}
            else None
        )
        r["mask_quality_repair_click"] = _mask_quality_repair_click(ans)
        # fall back: if confidence missing, infer a coarse one from the keep flag
        r["creature_confidence"] = conf if conf is not None else (0.75 if keep_flag else 0.1)
        (kept if keep_flag else dropped).append(r)
    return kept, dropped


def _segment_variant(service, variant, target_path, frame, groups, gt, W, H, sdir,
                     max_clicks, max_attempts):
    """Run one mask-generator variant on GT-seeded groups; return results aligned
    1:1 with ``groups`` (and per-creature click counts + statuses)."""
    if variant == "single":
        results = service.group_segment(target_path, groups)
        for r in results:
            r.setdefault("status", "single")
        return results, [len(g["clicks"]) for g in groups]
    if variant in ("refine", "refine_mm"):
        loop = refine_group_mm if variant == "refine_mm" else refine_group_with_sam3
        results, nclicks = [], []
        for g in groups:
            res, _tr = loop(service, g, target_path, frame, W, H,
                            sdir, max_clicks=max_clicks,
                            max_attempts=max_attempts)
            results.append(res)
            nclicks.append(len(res.get("clicks_used", g["clicks"])))
        return results, nclicks
    if variant in ("box", "multimask"):
        fn = _segment_box if variant == "box" else _segment_multimask
        results = [fn(service, g, target_path, frame, W, H, sdir) for g in groups]
        return results, [len(g["clicks"]) for g in groups]
    raise SystemExit(f"unknown eval variant: {variant}")


def run_eval(variant, gt_min_prob=0.5, max_clicks=5, max_attempts=3,
             out_dir=None, manifest=None):
    """Sweep ``variant`` across the labeled manifest. For each GT creature, seed
    ONE fg click at its centroid, generate a mask, and IoU it against that GT.
    Writes a per-frame report + an aggregate JSON/table."""
    manifest = manifest or EVAL_MANIFEST
    out_dir = out_dir or os.path.join(OUT_ROOT, f"eval_{variant}")
    os.makedirs(out_dir, exist_ok=True)
    service = build_sam3_service()

    per_creature = []  # flat list of dicts across all frames
    frame_rows = []    # per-frame aggregate
    for src in manifest:
        SRC["frame_outputs"] = src["frame_outputs"]
        SRC["video"] = src["video"]
        SRC["frames_dir"] = src["frames_dir"]
        SRC["fps"] = src["fps"]
        for tgt in src["targets"]:
            H, W, gt = load_target_and_gt(tgt, min_prob=gt_min_prob)
            if not gt:
                continue
            frame = read_video_frame(tgt)
            fdir = os.path.join(out_dir, f"{src['name']}_f{tgt:03d}")
            os.makedirs(fdir, exist_ok=True)
            tpath = os.path.join(fdir, "target.png")
            cv2.imwrite(tpath, frame)
            groups = [{"id": g["id"],
                       "description": "the creature at the marked click location",
                       "clicks": [{"x": g["centroid"][0] / W,
                                   "y": g["centroid"][1] / H, "label": 1}]}
                      for g in gt]
            results, nclicks = _segment_variant(service, variant, tpath, frame,
                                                groups, gt, W, H, fdir,
                                                max_clicks, max_attempts)
            ious = []
            rows = []
            for g, r, nc in zip(gt, results, nclicks):
                m = np.asarray(r["mask"]).astype(bool)
                iou = _iou(m, g["mask"])
                ious.append(iou)
                rows.append({"creature_id": g["id"], "gt_area": int(g["area"]),
                             "mask_area": int(r["area_px"]), "iou": round(iou, 3),
                             "clicks": nc, "status": r.get("status")})
            # render: GT vs produced mask, with IoU labels (matched=iou>=0.5)
            view_rows = [{"matched_gt": (g["id"] if iou >= 0.5 else None),
                          "best_iou": round(iou, 3)}
                         for g, iou in zip(gt, ious)]
            _render_sam3_report(frame, gt, results, view_rows,
                                os.path.join(fdir, "eval_report.png"),
                                f"{variant} {src['name']}_f{tgt}")
            mean_iou = float(np.mean(ious)) if ious else 0.0
            n_abandon = sum(1 for r in rows if r["status"] == "abandoned")
            frame_rows.append({"frame": f"{src['name']}_f{tgt}", "n_gt": len(gt),
                               "mean_iou": round(mean_iou, 3),
                               "ge50": sum(1 for i in ious if i >= 0.5),
                               "ge70": sum(1 for i in ious if i >= 0.7),
                               "mean_clicks": round(float(np.mean(nclicks)), 2),
                               "false_abandon": n_abandon})
            for r in rows:
                r["frame"] = f"{src['name']}_f{tgt}"
            per_creature += rows
            print(f"  [eval:{variant}] {src['name']}_f{tgt}: {len(gt)} GT, "
                  f"mean IoU {mean_iou:.3f}, >=0.5 {frame_rows[-1]['ge50']}/{len(gt)}, "
                  f"abandon {n_abandon}")

    all_iou = [c["iou"] for c in per_creature]
    summary = {
        "variant": variant, "gt_min_prob": gt_min_prob,
        "n_creatures": len(per_creature),
        "mean_iou": round(float(np.mean(all_iou)), 3) if all_iou else 0.0,
        "median_iou": round(float(np.median(all_iou)), 3) if all_iou else 0.0,
        "pct_ge50": round(100 * np.mean([i >= 0.5 for i in all_iou]), 1) if all_iou else 0.0,
        "pct_ge70": round(100 * np.mean([i >= 0.7 for i in all_iou]), 1) if all_iou else 0.0,
        "mean_clicks": round(float(np.mean([c["clicks"] for c in per_creature])), 2)
        if per_creature else 0.0,
        "false_abandon": sum(1 for c in per_creature if c["status"] == "abandoned"),
        "frames": frame_rows, "per_creature": per_creature,
    }
    json.dump(summary, open(os.path.join(out_dir, "eval_summary.json"), "w"), indent=2)
    print(f"\n=== eval variant={variant} (gt_min_prob={gt_min_prob}) ===")
    print(f"{'frame':12s} {'nGT':>3s} {'meanIoU':>8s} {'>=.5':>5s} {'>=.7':>5s} "
          f"{'clk':>4s} {'aband':>5s}")
    for fr in frame_rows:
        print(f"{fr['frame']:12s} {fr['n_gt']:>3d} {fr['mean_iou']:>8.3f} "
              f"{fr['ge50']:>5d} {fr['ge70']:>5d} {fr['mean_clicks']:>4.1f} "
              f"{fr['false_abandon']:>5d}")
    print(f"OVERALL: n={summary['n_creatures']} mean IoU={summary['mean_iou']} "
          f"median={summary['median_iou']} %>=.5={summary['pct_ge50']} "
          f"%>=.7={summary['pct_ge70']} mean_clicks={summary['mean_clicks']} "
          f"false_abandon={summary['false_abandon']}")
    return summary


VERIFY_CLICK_FMT = (
    'Output: brief reasoning then EXACTLY ONE trailing tag:\n'
    '<answer>{"creature": true}</answer>   -- the marked point is ON a distinct '
    'animal body\n'
    '<answer>{"creature": false}</answer>  -- the marked point is on substrate / '
    'shell / gravel / open background, NOT an animal'
)

VERIFY_AND_CORRECT_CLICK_FMT = (
    'Output: brief reasoning then EXACTLY ONE trailing tag:\n'
    '<answer>{"creature": true, "click": {"x": <float>, "y": <float>, '
    '"label": 1}}</answer>  -- the described target is visible; place one corrected '
    'foreground click safely inside a solid part of that SAME target. Other '
    'positive clicks on different parts of this target will be preserved.\n'
    '<answer>{"creature": false}</answer>  -- the described target is not visible '
    'in this crop or the proposal is substrate/background\n'
    'x,y are NORMALIZED [0,1] within the FULL FRAME (IMAGE 1), not the crop.'
)


def verify_clicks(groups, frame, W, H, sdir, region_frac=0.16, upscale=6,
                  strict=False, tag_prefix="vc", allow_all_life=False,
                  correct_click=False, existing_mask_overlay=False,
                  allow_existing_mask_expansion=False,
                  max_correction_displacement=0.10):
    """Content-based stray filter: for each placed click, show the MLLM a zoomed
    crop centred on it and ask whether the marked point is actually on a creature.
    Drop the ones judged substrate. Returns (kept_groups, n_dropped).

    ``strict=False`` (default, for high-prior main-pass clicks): KEEP on parse
    failure or ambiguity -- only drop an explicit ``creature: false`` (favour
    recall; never silently drop a real find).
    ``strict=True`` (for lower-prior tile-sweep candidates): KEEP only an
    explicit ``creature: true``; drop false/ambiguous/parse-failure. The wide-net
    tile proposer over-clicks substrate, so survivors must clear a higher bar."""
    empty = np.zeros((H, W), bool)
    kept = []
    dropped = 0
    for g in groups:
        tag = f"{tag_prefix}_id{g['id']}"
        geom = _mask_crop_geom(empty, g["clicks"], W, H, region_frac)
        cpath = os.path.join(sdir, f"{tag}.png")
        _render_mask_crop(frame, empty, g["clicks"], geom, cpath, upscale)
        full_context_path = None
        if correct_click:
            full_context_path = os.path.join(sdir, f"{tag}_full.png")
            _render_mask_crop(
                frame,
                empty,
                g["clicks"],
                (0, 0, W, H),
                full_context_path,
                1,
            )
        desc = g.get("description", "the marked point")
        if allow_all_life:
            subject = (
                "marine life (animal, coral, sponge, coherent colony, macroalga, "
                "seagrass, or other living organism)"
            )
            answer_format = VERIFY_CLICK_FMT.replace(
                "distinct animal body", "distinct living organism or colony"
            ).replace("actual creature", "actual marine life")
        else:
            subject = "a discrete animal body"
            answer_format = VERIFY_CLICK_FMT
        if correct_click:
            coverage_note = ""
            if existing_mask_overlay:
                if allow_existing_mask_expansion:
                    coverage_note = (
                        " Translucent green-filled regions with green outlines are "
                        "existing accepted masks. The named priority target may "
                        "have an INCOMPLETE green mask. In that case, keep the "
                        "proposal and place the corrected foreground click safely "
                        "inside that same partially green target so SAM3 can propose "
                        "a more complete replacement. Do not switch to a different "
                        "green organism."
                    )
                else:
                    coverage_note = (
                        " Translucent green-filled regions with green outlines are "
                        "organisms already accepted by the pipeline. Do NOT place "
                        "the corrected click on any green-covered structure; keep "
                        "the same described target only if an uncovered part of it "
                        "is genuinely the missed target."
                    )
            text = (
                "IMAGE 1 is the full seafloor frame. IMAGE 2 is a zoomed crop "
                "around the proposal. Black-ringed green dots mark approximate "
                "positive proposals in both images for: "
                f"'{desc}'. A dot may be "
                f"slightly misplaced.{coverage_note} Find that SAME described "
                "target using the full scene for identity/depth context and the "
                "crop for fine detail, "
                f"and relocate the point safely inside a solid part of {subject}. "
                "Before correcting anything, verify that ALL positive green dots "
                "belong to exactly one continuous or credibly occlusion-separated "
                "biological identity at the same depth layer. If the group mixes "
                "separate foreground/background organisms, different branching "
                "systems, or a small fragment with another identity, return "
                "creature=false so the targets can be rediscovered separately. "
                "Do not switch to a different nearby organism. If the described "
                "target is absent or unsupported, reject it. Before submitting, "
                "double-check that the corrected point lands visibly on a thick "
                "branch, body, or solid base—not in water, between branches, or "
                "on any logo, lettering, timestamp, grid mark, or video/UI overlay.\n\n"
                + VERIFY_AND_CORRECT_CLICK_FMT
            )
        else:
            text = (
                "This is a zoomed crop of a seafloor video frame. A green dot marks one "
                f"point (proposed target: '{desc}'). Underwater life is often "
                f"camouflaged, so look carefully for {subject} AT the green dot. Is the "
                "marked point on actual marine life, or on substrate/background?\n\n"
                + answer_format)
        content = []
        if correct_click:
            content.extend([
                {"type": "text", "text": "IMAGE 1 — FULL FRAME CONTEXT:"},
                {"type": "image", "image": full_context_path},
                {"type": "text", "text": "IMAGE 2 — ZOOMED PROPOSAL CROP:"},
            ])
        content.extend([
            {"type": "image", "image": cpath},
            {"type": "text", "text": text},
        ])
        r = send_claude_request(
            [{"role": "user", "content": content}],
            model=MODEL, max_tokens=response_token_budget(MODEL, 600))
        open(os.path.join(sdir, f"{tag}.txt"), "w").write(r or "<none>")
        ans = _extract_answer_json(r) or {}
        if correct_click and ans.get("creature") is True:
            click = ans.get("click") or {}
            if (
                isinstance(click.get("x"), (int, float))
                and isinstance(click.get("y"), (int, float))
                and 0.0 <= float(click["x"]) <= 1.0
                and 0.0 <= float(click["y"]) <= 1.0
            ):
                corrected = dict(g)
                corrected["original_clicks"] = [dict(c) for c in g["clicks"]]
                corrected_positive = {
                    "x": float(click["x"]),
                    "y": float(click["y"]),
                    "label": 1,
                }
                (
                    corrected["clicks"],
                    correction_displacement,
                    correction_applied,
                ) = _bounded_corrected_positive_click(
                    g["clicks"],
                    corrected_positive,
                    max_displacement=max_correction_displacement,
                )
                corrected["proposed_corrected_click"] = corrected_positive
                corrected["click_localization_displacement"] = (
                    correction_displacement
                )
                corrected["click_localization"] = (
                    "full_frame_corrected"
                    if correction_applied
                    else "original_preserved_large_displacement"
                )
                kept.append(corrected)
                continue
        if correct_click:
            # A successful correction returned above. In correction mode, a
            # true-without-valid-coordinates response is not actionable.
            drop = True
        else:
            drop = (
                (ans.get("creature") is not True)
                if strict else (ans.get("creature") is False)
            )
        if drop:
            dropped += 1
            continue
        kept.append(g)
    return kept, dropped


def _e2e_one_run(service, loop, clicker, do_verify, frame, gt, W, H, fdir_run,
                 tgt, offsets, fps, max_clicks, max_attempts, save_report=True):
    """One full pass on a frame: place clicks -> (verify) -> mask -> score vs GT.
    Returns a metrics dict for this single (stochastic) run."""
    os.makedirs(fdir_run, exist_ok=True)
    tpath = os.path.join(fdir_run, "target.png")
    cv2.imwrite(tpath, frame)
    neighbours = extract_neighbours(tgt, offsets, fdir_run, fps=fps)

    # Stage A: MLLM places its own clicks (no GT knowledge).
    if clicker in ("iterative", "iterative_tiled"):
        score = run_iterative(tpath, frame, gt, neighbours, W, H, fdir_run,
                              refine="verify", label="iter_verify",
                              tile_sweep=(clicker == "iterative_tiled"))
    else:
        score = run_verify_loop(tpath, frame, gt, neighbours, W, H, fdir_run)
    placed = score["groups"]

    # Stage A.5: content-based per-click verify pass drops substrate strays.
    n_dropped = 0
    if do_verify and placed:
        placed, n_dropped = verify_clicks(placed, frame, W, H, fdir_run)

    # Stage B: each surviving click -> mask via the winning generator.
    results = []
    for g in placed:
        res, _tr = loop(service, g, tpath, frame, W, H, fdir_run,
                        max_clicks=max_clicks, max_attempts=max_attempts)
        results.append(res)

    # Greedy IoU match: produced masks -> GT (threshold 0.5).
    masks = [np.asarray(r["mask"]).astype(bool) for r in results]
    nonempty = [(i, m) for i, m in enumerate(masks) if m.any()]
    gt_best = {g["id"]: 0.0 for g in gt}
    gt_taken, pred_matched = set(), set()
    pairs = []
    for pi, m in nonempty:
        for g in gt:
            pairs.append((_iou(m, g["mask"]), pi, g["id"]))
    for iou, pi, gid in sorted(pairs, reverse=True):
        if iou < 0.5 or pi in pred_matched or gid in gt_taken:
            continue
        pred_matched.add(pi)
        gt_taken.add(gid)
        gt_best[gid] = iou
    n_abandon = sum(1 for i, r in enumerate(results)
                    if r.get("status") == "abandoned" or not masks[i].any())
    n_stray = len(nonempty) - len(pred_matched)

    ious_all = [gt_best[g["id"]] for g in gt]
    detected = [v for v in ious_all if v >= 0.5]
    if save_report:
        view_rows = []
        for r, m in zip(results, masks):
            best, mgid = 0.0, None
            for g in gt:
                iou = _iou(m, g["mask"]) if m.any() else 0.0
                if iou > best:
                    best, mgid = iou, g["id"]
            view_rows.append({"matched_gt": (mgid if best >= 0.5 else None),
                              "best_iou": round(best, 3)})
        _render_sam3_report(frame, gt, results, view_rows,
                            os.path.join(fdir_run, "e2e_report.png"),
                            f"e2e {os.path.basename(fdir_run)}")
    return {
        "n_gt": len(gt),
        "recall": len(detected) / len(gt),
        "mean_iou_all": float(np.mean(ious_all)) if ious_all else 0.0,
        "mean_iou_detected": float(np.mean(detected)) if detected else 0.0,
        "n_placed": len(placed), "n_stray": n_stray, "n_abandon": n_abandon,
        "n_dropped": n_dropped, "gt_ious": {g["id"]: gt_best[g["id"]] for g in gt},
    }


def run_eval_e2e(variant="refine_mm", gt_min_prob=0.5, max_clicks=5, max_attempts=3,
                 out_dir=None, manifest=None, clicker="verify_loop",
                 do_verify=True, repeats=1):
    """Production-realistic eval: the MLLM PLACES its own clicks (verify_loop) on
    the target frame using temporal neighbours, then those (imperfect) clicks feed
    the mask generator. Scores final mask IoU vs GT by greedy IoU match, and
    decomposes the result into clicking recall vs mask quality.

    Unlike ``run_eval`` (which seeds one perfect fg click at each GT centroid),
    this never sees GT during click placement -- it is the whole subsystem the
    way production runs it, and it exercises the abandon/reject safety net on the
    strays the MLLM inevitably produces."""
    manifest = manifest or EVAL_MANIFEST
    vtag = "_verify" if do_verify else ""
    out_dir = out_dir or os.path.join(OUT_ROOT, f"e2e_{clicker}_{variant}{vtag}")
    os.makedirs(out_dir, exist_ok=True)
    service = build_sam3_service()
    loop = refine_group_mm if variant == "refine_mm" else refine_group_with_sam3

    frame_rows = []
    pooled_iou = []        # flat IoU over every (frame, run, gt) triple
    pooled_det = []        # matching detected booleans
    pooled_stray, pooled_drop = [], []   # per-run totals (for averaging)
    for src in manifest:
        SRC["frame_outputs"] = src["frame_outputs"]
        SRC["video"] = src["video"]
        SRC["frames_dir"] = src["frames_dir"]
        SRC["fps"] = src["fps"]
        offsets = [1, 2, 3] if src["frames_dir"] else DEFAULT_OFFSETS
        for tgt in src["targets"]:
            H, W, gt = load_target_and_gt(tgt, min_prob=gt_min_prob)
            if not gt:
                continue
            frame = read_video_frame(tgt)
            fname = f"{src['name']}_f{tgt:03d}"
            runs = []
            for ri in range(repeats):
                fdir_run = os.path.join(out_dir, fname + (f"_r{ri}" if repeats > 1 else ""))
                m = _e2e_one_run(service, loop, clicker, do_verify, frame, gt, W, H,
                                 fdir_run, tgt, offsets, SRC["fps"],
                                 max_clicks, max_attempts)
                runs.append(m)
                for g in gt:
                    pooled_iou.append(m["gt_ious"][g["id"]])
                    pooled_det.append(m["gt_ious"][g["id"]] >= 0.5)
                pooled_stray.append(m["n_stray"])
                pooled_drop.append(m["n_dropped"])

            def _ms(key):  # mean, std across this frame's runs
                vals = [r[key] for r in runs]
                return float(np.mean(vals)), float(np.std(vals))
            rec_m, rec_s = _ms("recall")
            all_m, _ = _ms("mean_iou_all")
            det_m, _ = _ms("mean_iou_detected")
            plc_m, _ = _ms("n_placed")
            str_m, _ = _ms("n_stray")
            abn_m, _ = _ms("n_abandon")
            drp_m, _ = _ms("n_dropped")
            frame_rows.append({"frame": fname, "n_gt": len(gt),
                               "recall": round(rec_m, 3), "recall_std": round(rec_s, 3),
                               "mean_iou_all": round(all_m, 3),
                               "mean_iou_detected": round(det_m, 3),
                               "n_placed": round(plc_m, 1), "n_stray": round(str_m, 1),
                               "n_abandon": round(abn_m, 1), "n_dropped": round(drp_m, 1)})
            print(f"  [e2e:{variant}{vtag}] {fname}: {len(gt)} GT, "
                  f"recall {rec_m:.2f}+-{rec_s:.2f}, IoU(all) {all_m:.3f}, "
                  f"IoU(det) {det_m:.3f}, stray {str_m:.1f}, dropped {drp_m:.1f}")

    summary = {
        "variant": variant, "clicker": clicker, "do_verify": do_verify,
        "repeats": repeats, "gt_min_prob": gt_min_prob, "mode": "end_to_end",
        "n_gt_per_pass": sum(fr["n_gt"] for fr in frame_rows),
        "recall": round(float(np.mean(pooled_det)), 3) if pooled_det else 0.0,
        "mean_iou_all": round(float(np.mean(pooled_iou)), 3) if pooled_iou else 0.0,
        "mean_iou_detected": round(float(np.mean([i for i, d in zip(pooled_iou, pooled_det) if d])), 3)
        if any(pooled_det) else 0.0,
        "strays_per_pass": round(float(np.sum(pooled_stray)) / max(1, repeats), 1),
        "dropped_per_pass": round(float(np.sum(pooled_drop)) / max(1, repeats), 1),
        "frames": frame_rows,
    }
    json.dump(summary, open(os.path.join(out_dir, "eval_e2e_summary.json"), "w"), indent=2)
    print(f"\n=== END-TO-END variant={variant} clicker={clicker} verify={do_verify} "
          f"repeats={repeats} (gt_min_prob={gt_min_prob}) ===")
    print(f"{'frame':12s} {'nGT':>3s} {'recall':>11s} {'IoUall':>7s} {'IoUdet':>7s} "
          f"{'plc':>4s} {'stry':>4s} {'abnd':>4s} {'drop':>4s}")
    for fr in frame_rows:
        print(f"{fr['frame']:12s} {fr['n_gt']:>3d} "
              f"{fr['recall']:>5.2f}+-{fr['recall_std']:<4.2f} "
              f"{fr['mean_iou_all']:>7.3f} {fr['mean_iou_detected']:>7.3f} "
              f"{fr['n_placed']:>4.1f} {fr['n_stray']:>4.1f} {fr['n_abandon']:>4.1f} "
              f"{fr['n_dropped']:>4.1f}")
    print(f"OVERALL: recall={summary['recall']} meanIoU(all)={summary['mean_iou_all']} "
          f"meanIoU(detected)={summary['mean_iou_detected']} "
          f"strays/pass={summary['strays_per_pass']} dropped/pass={summary['dropped_per_pass']}")
    return summary


def _render_sam3_report(frame, gt, results, rows, out_path, label):
    """3-panel: RAW | GT (green fill) | SAM3 masks (green=matched, red=unmatched)."""
    H, W = frame.shape[:2]
    raw = frame.copy()
    gtv = frame.copy()
    ov = gtv.copy()
    for g in gt:
        ov[g["mask"]] = (0, 180, 0)
    gtv = cv2.addWeighted(ov, 0.45, gtv, 0.55, 0)
    pred = frame.copy()
    ovp = pred.copy()
    for r, row in zip(results, rows):
        m = np.asarray(r["mask"]).astype(bool)
        col = (0, 180, 0) if row["matched_gt"] is not None else (0, 0, 255)
        ovp[m] = col
    pred = cv2.addWeighted(ovp, 0.45, pred, 0.55, 0)
    for r, row in zip(results, rows):
        m = np.asarray(r["mask"]).astype(bool)
        if not m.any():
            continue
        ys, xs = np.where(m)
        cx, cy = int(xs.mean()), int(ys.mean())
        txt = f"IoU{row['best_iou']:.2f}"
        cv2.putText(pred, txt, (cx - 20, cy), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                    (0, 0, 0), 3, cv2.LINE_AA)
        cv2.putText(pred, txt, (cx - 20, cy), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                    (255, 255, 255), 1, cv2.LINE_AA)
        for c in r.get("clicks_used", []):
            if "box" in c:  # box prompt entry, not a point
                bx = c["box"]
                cv2.rectangle(pred, (int(bx[0]), int(bx[1])), (int(bx[2]), int(bx[3])),
                              (0, 255, 255), 1)
                continue
            if "x" not in c:
                continue
            p = (int(c["x"] * W), int(c["y"] * H))
            cv2.drawMarker(pred, p, (0, 255, 255), cv2.MARKER_TILTED_CROSS, 16, 2)
    labels = ["RAW", f"GT ({len(gt)})", f"{label}: SAM3 masks"]
    panels = [raw, gtv, pred]
    for img, t in zip(panels, labels):
        cv2.rectangle(img, (0, 0), (img.shape[1], 26), (0, 0, 0), -1)
        cv2.putText(img, t, (6, 19), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255),
                    2, cv2.LINE_AA)
    cv2.imwrite(out_path, np.hstack(panels))


def render_gt_fixture(frame, gt, out_path):
    out = frame.copy()
    overlay = out.copy()
    for gtm in gt:
        overlay[gtm["mask"]] = (0, 180, 0)
    out = cv2.addWeighted(overlay, 0.4, out, 0.6, 0)
    for gtm in gt:
        cx, cy = gtm["centroid"]
        cv2.drawMarker(out, (cx, cy), (0, 0, 255), cv2.MARKER_CROSS, 16, 2)
        cv2.putText(out, f"{gtm['id']} ({gtm['area']}px)", (cx + 8, cy),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 3, cv2.LINE_AA)
        cv2.putText(out, f"{gtm['id']} ({gtm['area']}px)", (cx + 8, cy),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
    cv2.imwrite(out_path, out)


# ----------------------------------------------------------------------------
# Strategies: each returns (messages, parse_fn)
# ----------------------------------------------------------------------------
ANSWER_FMT = (
    'Output format -- free-text reasoning then EXACTLY ONE trailing tag:\n'
    '<answer>{"missed_creatures":[{"id":1,"description":"<text>",'
    '"clicks":[{"x":<float>,"y":<float>,"label":1}]}]}</answer>\n'
    'x,y are NORMALIZED [0,1] with (0,0)=top-left, (1,1)=bottom-right of the FIRST image.'
)

TASK_CORE = (
    "You are auditing an underwater seafloor video for small creatures (fish, "
    "crabs, shrimp, snails, sea stars, etc.). Creatures are often well "
    "camouflaged against gravel/shell substrate. The FIRST image is the TARGET "
    "frame you must annotate. Place ONE foreground click (label 1) on the CENTER "
    "OF MASS of each distinct creature -- the thickest, most central part of its "
    "body (its main trunk), NOT a fin, tail, leg, antenna, or the body's edge. Be "
    "precise: the click must land ON the animal, not on adjacent substrate."
)

MOTION_HINT = (
    " The additional images are the SAME camera view at nearby times "
    "(t<0 before, t>0 after the target). Compare them: a shape that shifts, "
    "appears, or disappears between frames is almost certainly a creature, even "
    "if it blends into the substrate in the target frame. Use this motion cue to "
    "find camouflaged animals."
)


def _grid_overlay(frame, n=10):
    out = frame.copy()
    h, w = out.shape[:2]
    for i in range(1, n):
        x = int(w * i / n)
        y = int(h * i / n)
        cv2.line(out, (x, 0), (x, h), (255, 255, 255), 1, cv2.LINE_AA)
        cv2.line(out, (0, y), (w, y), (255, 255, 255), 1, cv2.LINE_AA)
    return out


def _numbered_grid_overlay(frame, n=10):
    """Grid lines plus a faint r,c label centered in each cell."""
    out = _grid_overlay(frame, n)
    h, w = out.shape[:2]
    fs = 0.32 if max(h, w) < 700 else 0.4
    for r in range(n):
        for c in range(n):
            cx = int(w * (c + 0.5) / n)
            cy = int(h * (r + 0.5) / n)
            lbl = f"{r},{c}"
            cv2.putText(out, lbl, (cx - 12, cy + 4), cv2.FONT_HERSHEY_SIMPLEX,
                        fs, (0, 0, 0), 2, cv2.LINE_AA)
            cv2.putText(out, lbl, (cx - 12, cy + 4), cv2.FONT_HERSHEY_SIMPLEX,
                        fs, (0, 255, 255), 1, cv2.LINE_AA)
    return out


def _draw_marker_norm(img, pt, path):
    """Draw a crosshair at normalized pt on a copy of img; save to path."""
    out = img.copy()
    h, w = out.shape[:2]
    p = (int(pt[0] * w), int(pt[1] * h))
    cv2.drawMarker(out, p, (0, 0, 0), cv2.MARKER_CROSS, 30, 5)
    cv2.drawMarker(out, p, (0, 0, 255), cv2.MARKER_CROSS, 28, 2)
    cv2.circle(out, p, 16, (0, 0, 255), 2)
    cv2.imwrite(path, out)


def build_messages(strategy: str, target_path: str, neighbours, out_dir: str):
    nb_paths = [p for _, p in neighbours]
    nb_labels = "; ".join(b for b, _ in neighbours)
    if strategy == "baseline_grid":
        # current-pipeline style: 10x10 grid, float coords + temporal refs
        gframe = _grid_overlay(cv2.imread(target_path))
        gp = os.path.join(out_dir, "target_grid.png")
        cv2.imwrite(gp, gframe)
        text = (TASK_CORE + MOTION_HINT +
                f"\n\nThe first image has a 10x10 white grid for reference. "
                f"Reference frames: {nb_labels}.\n\n" + ANSWER_FMT)
        imgs = [gp] + nb_paths
    elif strategy == "no_grid":
        text = (TASK_CORE + MOTION_HINT +
                f"\n\nReference frames: {nb_labels}.\n\n" + ANSWER_FMT)
        imgs = [target_path] + nb_paths
    elif strategy == "motion_first":
        text = (TASK_CORE +
                " IMPORTANT FIRST STEP: before deciding, explicitly walk through "
                "each reference frame and note any region that changes relative to "
                "the target frame. List those regions, THEN place clicks." +
                MOTION_HINT + f"\n\nReference frames: {nb_labels}.\n\n" + ANSWER_FMT)
        imgs = [target_path] + nb_paths
    elif strategy == "no_temporal":
        text = (TASK_CORE + "\n\n" + ANSWER_FMT)
        imgs = [target_path]
    else:
        raise SystemExit(f"unknown strategy {strategy}")
    content = [{"type": "image", "image": imgs[0]}, {"type": "text", "text": text}]
    content += [{"type": "image", "image": p} for p in imgs[1:]]
    return [{"role": "user", "content": content}]


SINGLE_CALL_STRATEGIES = ["no_temporal", "no_grid", "baseline_grid", "motion_first"]


# ----------------------------------------------------------------------------
# Two-stage zoom strategy
# ----------------------------------------------------------------------------
def _parse_coarse(text):
    """Reuse the grouped parser; each creature's first fg click is its coarse loc."""
    groups = parse_creature_click_groups(text or "")
    out = []
    for g in groups:
        fg = [c for c in g["clicks"] if c["label"] == 1]
        if not fg:
            continue
        out.append({"id": g["id"], "description": g.get("description", ""),
                    "x": fg[0]["x"], "y": fg[0]["y"]})
    return out


def _parse_single_point(text):
    """Parse a single {x,y} from a zoomed-crop refine call."""
    groups = parse_creature_click_groups(text or "")
    for g in groups:
        for c in g["clicks"]:
            if c["label"] == 1:
                return c["x"], c["y"]
    return None


def _zoom_refine_one(frame, cc, W, H, sdir, tag, crop_frac=0.30, upscale=4):
    """Crop a window around coarse point cc, upscale, ask for a precise click.
    Returns (fx, fy) normalized full-frame, or None if creature not in crop."""
    cw, ch = int(W * crop_frac), int(H * crop_frac)
    ccx, ccy = int(cc["x"] * W), int(cc["y"] * H)
    l = max(0, min(W - cw, ccx - cw // 2))
    t = max(0, min(H - ch, ccy - ch // 2))
    crop = frame[t:t + ch, l:l + cw]
    zoom = cv2.resize(crop, (cw * upscale, ch * upscale), interpolation=cv2.INTER_CUBIC)
    zoom = _grid_overlay(zoom, n=10)
    zp = os.path.join(sdir, f"crop_{tag}.png")
    cv2.imwrite(zp, zoom)
    s2_text = (
        f"This is a ZOOMED, high-resolution crop of an underwater scene. It "
        f"should contain: '{cc['description']}'. Place ONE click (label 1) "
        f"exactly on the CENTER of that creature's body. If the creature is "
        f"actually NOT in this crop, return an empty list.\n\n" + ANSWER_FMT)
    r2 = send_claude_request(
        [{"role": "user", "content": [
            {"type": "image", "image": zp}, {"type": "text", "text": s2_text}]}],
        model=MODEL, max_tokens=response_token_budget(MODEL, 800))
    open(os.path.join(sdir, f"refine_{tag}.txt"), "w").write(r2 or "<none>")
    pt = _parse_single_point(r2)
    if pt is None:
        return None
    return (l + pt[0] * cw) / W, (t + pt[1] * ch) / H


def run_zoom(target_path, frame, gt, neighbours, W, H, out_dir,
             crop_frac=0.30, upscale=4):
    """Stage 1: coarse-locate every creature. Stage 2: per creature, crop a
    window around the coarse point, upscale, ask for a precise click."""
    sdir = os.path.join(out_dir, "zoom")
    os.makedirs(sdir, exist_ok=True)
    nb_paths = [p for _, p in neighbours]
    nb_labels = "; ".join(b for b, _ in neighbours)

    s1_text = (TASK_CORE + MOTION_HINT +
               " For this first pass, give an APPROXIMATE click near each "
               "creature -- you will refine it later on a zoomed view." +
               f"\n\nReference frames: {nb_labels}.\n\n" + ANSWER_FMT)
    content = [{"type": "image", "image": target_path},
               {"type": "text", "text": s1_text}]
    content += [{"type": "image", "image": p} for p in nb_paths]
    r1 = send_claude_request([{"role": "user", "content": content}],
                             model=MODEL,
                             max_tokens=response_token_budget(MODEL, 2000))
    open(os.path.join(sdir, "stage1_response.txt"), "w").write(r1 or "<none>")
    coarse = _parse_coarse(r1)
    print(f"  [zoom] stage1 found {len(coarse)} creatures")

    refined_groups = []
    for cc in coarse:
        pt = _zoom_refine_one(frame, cc, W, H, sdir, f"id{cc['id']}", crop_frac, upscale)
        fx, fy = pt if pt is not None else (cc["x"], cc["y"])
        refined_groups.append({"id": cc["id"], "description": cc["description"],
                               "clicks": [{"x": fx, "y": fy, "label": 1}]})

    score = score_clicks(refined_groups, gt, W, H)
    score["groups"] = refined_groups
    render_result(frame, gt, refined_groups, score,
                  os.path.join(sdir, "result.png"), title="zoom")
    json.dump({"strategy": "zoom", "n_groups": len(refined_groups),
               "score": {k: v for k, v in score.items()
                         if k in ("n_gt", "n_hit", "coverage", "n_fg_clicks",
                                  "n_stray", "cov_at", "nearest_px")},
               "hits": {str(k): v for k, v in score["hits"].items()}},
              open(os.path.join(sdir, "score.json"), "w"), indent=2)
    return score


def _render_found(frame, found, path):
    """Draw already-accumulated clicks as numbered cyan dots for the next pass."""
    out = frame.copy()
    h, w = out.shape[:2]
    for i, f in enumerate(found, 1):
        p = (int(f["x"] * w), int(f["y"] * h))
        cv2.circle(out, p, 9, (0, 0, 0), -1)
        cv2.circle(out, p, 7, (0, 255, 255), -1)
        cv2.putText(out, str(i), (p[0] - 4, p[1] + 4), cv2.FONT_HERSHEY_SIMPLEX,
                    0.4, (0, 0, 0), 1, cv2.LINE_AA)
    cv2.imwrite(path, out)


def _tile_boxes(W, H, n, overlap):
    """Yield (l,t,r,b) pixel boxes for an n x n grid with fractional overlap."""
    bw, bh = W / n, H / n
    ox, oy = bw * overlap, bh * overlap
    for r in range(n):
        for c in range(n):
            l = max(0, int(c * bw - ox))
            t = max(0, int(r * bh - oy))
            rr = min(W, int((c + 1) * bw + ox))
            bb = min(H, int((r + 1) * bh + oy))
            yield l, t, rr, bb


def _tile_sweep(frame, W, H, sdir, found, tile_n=2, overlap=0.15, upscale=2,
                dedup_px=25):
    """High-resolution recall pass: split the frame into overlapping tiles, ask
    the MLLM to find ALL creatures in each upscaled tile, map clicks back to
    full-frame coords, dedup against each other and against ``found``, then
    refine each survivor with the self-correcting verify loop. Returns a list of
    new {"x","y","description"} clicks. Targets small/camouflaged STATIC
    creatures that get lost in the downsampled whole-frame view."""
    tdir = os.path.join(sdir, "tiles")
    os.makedirs(tdir, exist_ok=True)
    cand = []  # full-frame normalized coarse candidates
    for ti, (l, t, rr, bb) in enumerate(_tile_boxes(W, H, tile_n, overlap)):
        tw, th = rr - l, bb - t
        crop = frame[t:bb, l:rr]
        zoom = cv2.resize(crop, (tw * upscale, th * upscale),
                          interpolation=cv2.INTER_CUBIC)
        zp = os.path.join(tdir, f"tile{ti}.png")
        cv2.imwrite(zp, zoom)
        text = (TASK_CORE +
                " This image is a ZOOMED, high-resolution CROP of a larger "
                "seafloor frame -- it shows only part of the scene. Click each "
                "distinct LIVING ANIMAL visible in THIS crop, including small or "
                "well-camouflaged ones. Do NOT click bare gravel, loose shells or "
                "shell fragments, rocks, sticks, sediment texture, or shadows -- "
                "those are not animals. Give one click per animal.\n\n"
                + ANSWER_FMT)
        r = send_claude_request([{"role": "user", "content": [
            {"type": "image", "image": zp}, {"type": "text", "text": text}]}],
            model=MODEL, max_tokens=response_token_budget(MODEL, 1500))
        open(os.path.join(tdir, f"tile{ti}_resp.txt"), "w").write(r or "<none>")
        for cc in _parse_coarse(r):
            fx = (l + cc["x"] * tw) / W
            fy = (t + cc["y"] * th) / H
            cand.append({"x": fx, "y": fy, "description": cc["description"]})
    print(f"  [tile_sweep] {tile_n}x{tile_n}: {len(cand)} raw candidates")

    added = []
    for cc in cand:
        ref = found + added
        if any(abs(cc["x"] - f["x"]) * W < dedup_px and
               abs(cc["y"] - f["y"]) * H < dedup_px for f in ref):
            continue
        pt = _verify_refine_one(frame, cc, W, H, tdir, f"tile_{len(added)}")
        if pt is None:
            continue
        fx, fy = pt
        ref = found + added
        if any(abs(fx - f["x"]) * W < dedup_px and abs(fy - f["y"]) * H < dedup_px
               for f in ref):
            continue
        added.append({"x": fx, "y": fy, "description": cc["description"]})

    # Strict creature gate: tile candidates are lower-prior than the whole-frame
    # pass, so survivors must clear an explicit "yes, animal" bar (drop ambiguous).
    if added:
        cand_groups = [{"id": i + 1, "description": a["description"],
                        "clicks": [{"x": a["x"], "y": a["y"], "label": 1}]}
                       for i, a in enumerate(added)]
        kept, gated = verify_clicks(cand_groups, frame, W, H, tdir, strict=True,
                                    tag_prefix="tile_gate")
        kept_xy = {(g["clicks"][0]["x"], g["clicks"][0]["y"]) for g in kept}
        added = [a for a in added if (a["x"], a["y"]) in kept_xy]
        print(f"  [tile_sweep] strict gate dropped {gated}")
    print(f"  [tile_sweep] added {len(added)} new (had {len(found)})")
    return added


def run_iterative(target_path, frame, gt, neighbours, W, H, out_dir,
                  max_passes=4, crop_frac=0.30, upscale=4, refine="zoom",
                  label="iterative", tile_sweep=False, tile_n=2):
    """Accumulate creatures across passes. Each pass: show the target frame with
    already-found clicks marked, ask ONLY for creatures not yet marked, then
    refine each new proposal (``refine``: 'zoom' single-shot, or 'verify' the
    self-correcting loop). Stop when a pass adds nothing. If ``tile_sweep`` is
    set, finish with a high-res tiled recall pass for small/camouflaged static
    creatures the whole-frame view misses."""
    sdir = os.path.join(out_dir, label)
    os.makedirs(sdir, exist_ok=True)
    nb_paths = [p for _, p in neighbours]
    nb_labels = "; ".join(b for b, _ in neighbours)

    found = []  # accumulated refined clicks: {"x","y","description"}
    next_id = 1
    for pass_i in range(1, max_passes + 1):
        marked_path = os.path.join(sdir, f"pass{pass_i}_marked.png")
        _render_found(frame, found, marked_path)
        already = (f" {len(found)} creature(s) are ALREADY marked with numbered "
                   "cyan dots on the first image; do NOT click those again. Find "
                   "only ADDITIONAL creatures not yet marked." if found else
                   " No creatures are marked yet; find all of them.")
        text = (TASK_CORE + MOTION_HINT + already +
                " Give an APPROXIMATE click near each NEW creature; it will be "
                "refined on a zoomed view." +
                f"\n\nReference frames: {nb_labels}.\n\n" + ANSWER_FMT)
        content = [{"type": "image", "image": marked_path},
                   {"type": "text", "text": text}]
        content += [{"type": "image", "image": p} for p in nb_paths]
        r = send_claude_request([{"role": "user", "content": content}],
                                model=MODEL,
                                max_tokens=response_token_budget(MODEL, 2000))
        open(os.path.join(sdir, f"pass{pass_i}_response.txt"), "w").write(r or "<none>")
        coarse = _parse_coarse(r)
        print(f"  [iterative] pass {pass_i}: proposed {len(coarse)} new")

        added = 0
        for cc in coarse:
            # skip proposals that fall on an already-found click (model re-clicking)
            dup = any(abs(cc["x"] - f["x"]) * W < 25 and abs(cc["y"] - f["y"]) * H < 25
                      for f in found)
            if dup:
                continue
            tag = f"p{pass_i}_id{next_id}"
            if refine == "verify":
                pt = _verify_refine_one(frame, cc, W, H, sdir, tag)
            else:
                pt = _zoom_refine_one(frame, cc, W, H, sdir, tag, crop_frac, upscale)
            if pt is None:
                continue
            fx, fy = pt
            if any(abs(fx - f["x"]) * W < 25 and abs(fy - f["y"]) * H < 25 for f in found):
                continue
            found.append({"x": fx, "y": fy, "description": cc["description"]})
            next_id += 1
            added += 1
        print(f"  [iterative] pass {pass_i}: added {added} (total {len(found)})")
        if added == 0:
            break

    if tile_sweep:
        found.extend(_tile_sweep(frame, W, H, sdir, found, tile_n=tile_n))

    groups = [{"id": i + 1, "description": f["description"],
               "clicks": [{"x": f["x"], "y": f["y"], "label": 1}]}
              for i, f in enumerate(found)]
    score = score_clicks(groups, gt, W, H)
    score["groups"] = groups
    render_result(frame, gt, groups, score,
                  os.path.join(sdir, "result.png"), title=label)
    json.dump({"strategy": label, "n_groups": len(groups), "passes": pass_i,
               "score": {k: v for k, v in score.items()
                         if k in ("n_gt", "n_hit", "coverage", "n_fg_clicks",
                                  "n_stray", "cov_at", "nearest_px")},
               "hits": {str(k): v for k, v in score["hits"].items()}},
              open(os.path.join(sdir, "score.json"), "w"), indent=2)
    return score


def _verify_refine_one(frame, cc, W, H, sdir, tag, region_frac=0.22, fine_n=8,
                       upscale=6, max_refine=3):
    """Zoom into the region around cc, overlay a fine numbered grid, get a click,
    then iteratively show the model WHERE the click landed and let it correct
    itself. The conversation history gives the model memory of its prior coords.
    Returns (fx, fy) normalized full-frame, or None if creature not in crop."""
    cw, ch = int(W * region_frac), int(H * region_frac)
    ccx, ccy = int(cc["x"] * W), int(cc["y"] * H)
    l = max(0, min(W - cw, ccx - cw // 2))
    t = max(0, min(H - ch, ccy - ch // 2))
    crop = frame[t:t + ch, l:l + cw]
    zoom = cv2.resize(crop, (cw * upscale, ch * upscale), interpolation=cv2.INTER_CUBIC)
    zoom_grid = _numbered_grid_overlay(zoom, fine_n)
    grid_path = os.path.join(sdir, f"crop_{tag}.png")
    cv2.imwrite(grid_path, zoom_grid)

    desc = cc["description"]
    messages = [{"role": "user", "content": [
        {"type": "image", "image": grid_path},
        {"type": "text", "text":
            f"This is a ZOOMED, high-resolution crop of an underwater seafloor "
            f"scene. It should contain: '{desc}'. The yellow numbered grid (row,col) "
            f"is only a spatial reference. Place ONE click (label 1) exactly on the "
            f"CENTER OF MASS (the thickest, most central part) of that creature's "
            f"body -- not a fin, tail, leg, or edge. If the creature is genuinely "
            f"NOT in this crop, return an empty list.\n\n" + ANSWER_FMT}]}]

    pt = None
    for it in range(max_refine + 1):
        resp = send_claude_request(
            messages,
            model=MODEL,
            max_tokens=response_token_budget(MODEL, 700),
        )
        open(os.path.join(sdir, f"verify_{tag}_it{it}.txt"), "w").write(resp or "<none>")
        new_pt = _parse_single_point(resp)
        if new_pt is None:
            return None if pt is None else ((l + pt[0] * cw) / W, (t + pt[1] * ch) / H)
        prev = pt
        pt = new_pt
        mk_path = os.path.join(sdir, f"verify_{tag}_it{it}_mark.png")
        _draw_marker_norm(zoom, pt, mk_path)
        # converged: model repeated essentially the same coordinate
        if prev is not None and abs(pt[0] - prev[0]) * (cw * upscale) < 4 and \
           abs(pt[1] - prev[1]) * (ch * upscale) < 4:
            break
        if it == max_refine:
            break
        messages.append({"role": "assistant", "content": resp})
        messages.append({"role": "user", "content": [
            {"type": "image", "image": mk_path},
            {"type": "text", "text":
                "The red crosshair+circle shows EXACTLY where your click landed on "
                "the SAME crop. Judge honestly: is the crosshair on the creature's "
                "CENTER OF MASS -- its thickest, most central body region (not on "
                "substrate, not on a fin/tail/leg, not on the body's edge)? If it is "
                "well-placed, reply with the SAME coordinates to confirm. If it is "
                "off, reply with CORRECTED coordinates that move it onto the body's "
                "center of mass.\n\n" + ANSWER_FMT}]})
    return (l + pt[0] * cw) / W, (t + pt[1] * ch) / H


def run_verify_loop(target_path, frame, gt, neighbours, W, H, out_dir,
                    coarse_n=10, region_frac=0.22, fine_n=8, upscale=6, max_refine=3):
    """Stage 1: numbered-grid coarse cell detection (which cells hold creatures).
    Stage 2: per creature, zoom + fine grid + self-verify/correct loop."""
    sdir = os.path.join(out_dir, "verify_loop")
    os.makedirs(sdir, exist_ok=True)
    nb_paths = [p for _, p in neighbours]
    nb_labels = "; ".join(b for b, _ in neighbours)

    grid_img = _numbered_grid_overlay(frame, coarse_n)
    gp = os.path.join(sdir, "stage1_grid.png")
    cv2.imwrite(gp, grid_img)
    s1_text = (TASK_CORE + MOTION_HINT +
               f" The first image has a {coarse_n}x{coarse_n} numbered grid "
               "(row,col labels). For this first pass, name the grid cell holding "
               "each creature and give an APPROXIMATE click; it will be refined on "
               "a zoomed view." +
               f"\n\nReference frames: {nb_labels}.\n\n" + ANSWER_FMT)
    content = [{"type": "image", "image": gp}, {"type": "text", "text": s1_text}]
    content += [{"type": "image", "image": p} for p in nb_paths]
    r1 = send_claude_request([{"role": "user", "content": content}],
                             model=MODEL,
                             max_tokens=response_token_budget(MODEL, 2000))
    open(os.path.join(sdir, "stage1_response.txt"), "w").write(r1 or "<none>")
    coarse = _parse_coarse(r1)
    print(f"  [verify_loop] stage1 found {len(coarse)} creatures")

    refined_groups = []
    for cc in coarse:
        pt = _verify_refine_one(frame, cc, W, H, sdir, f"id{cc['id']}",
                                region_frac, fine_n, upscale, max_refine)
        if pt is None:
            continue
        refined_groups.append({"id": cc["id"], "description": cc["description"],
                               "clicks": [{"x": pt[0], "y": pt[1], "label": 1}]})

    score = score_clicks(refined_groups, gt, W, H)
    score["groups"] = refined_groups
    render_result(frame, gt, refined_groups, score,
                  os.path.join(sdir, "result.png"), title="verify_loop")
    json.dump({"strategy": "verify_loop", "n_groups": len(refined_groups),
               "score": {k: v for k, v in score.items()
                         if k in ("n_gt", "n_hit", "coverage", "n_fg_clicks",
                                  "n_stray", "cov_at", "nearest_px")},
               "hits": {str(k): v for k, v in score["hits"].items()}},
              open(os.path.join(sdir, "score.json"), "w"), indent=2)
    return score


def run_strategy(strategy, target_path, frame, gt, neighbours, W, H, out_dir):
    sdir = os.path.join(out_dir, strategy)
    os.makedirs(sdir, exist_ok=True)
    messages = build_messages(strategy, target_path, neighbours, sdir)
    resp = send_claude_request(
        messages,
        model=MODEL,
        max_tokens=response_token_budget(MODEL, 2000),
    )
    open(os.path.join(sdir, "response.txt"), "w").write(resp or "<none>")
    groups = parse_creature_click_groups(resp or "")
    score = score_clicks(groups, gt, W, H)
    score["groups"] = groups
    render_result(frame, gt, groups, score, os.path.join(sdir, "result.png"),
                  title=strategy)
    json.dump({"strategy": strategy, "n_groups": len(groups),
               "score": {k: v for k, v in score.items()
                         if k in ("n_gt", "n_hit", "coverage", "n_fg_clicks",
                                  "n_stray", "cov_at", "nearest_px")},
               "hits": {str(k): v for k, v in score["hits"].items()}},
              open(os.path.join(sdir, "score.json"), "w"), indent=2)
    return score


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--target", type=int, default=DEFAULT_TARGET)
    ap.add_argument("--strategy", default="baseline_grid")
    ap.add_argument("--render-gt-only", action="store_true")
    ap.add_argument("--frame-outputs", default=FRAME_OUTPUTS,
                    help="path to frame_outputs_rle.json (GT masks)")
    ap.add_argument("--video", default=VIDEO, help="raw mp4 (if no --frames-dir)")
    ap.add_argument("--frames-dir", default=None,
                    help="dir of frame_NNNNNN.jpg matching frame_index "
                         "(use when the SAM3 run was fed a pre-extracted clip)")
    ap.add_argument("--offsets", default=None,
                    help="comma-separated neighbour frame offsets, e.g. 1,2,3")
    ap.add_argument("--tag", default=None, help="output subdir name override")
    ap.add_argument("--gt-min-prob", type=float, default=0.0,
                    help="drop GT masks below this SAM3 prob (vet out marine snow)")
    ap.add_argument("--sam3", action="store_true",
                    help="feed produced clicks into SAM3 point mode + render mask report")
    ap.add_argument("--sam3-refine", action="store_true",
                    help="after SAM3, iteratively show the mask to the MLLM and add "
                         "one click at a time until it judges the mask good")
    ap.add_argument("--max-sam3-clicks", type=int, default=5,
                    help="max clicks per creature per attempt in the SAM3 refine loop")
    ap.add_argument("--max-sam3-attempts", type=int, default=3,
                    help="max restart attempts (reject -> re-seed) in the refine loop")
    ap.add_argument("--eval", action="store_true",
                    help="run the GT-seeded click->mask eval harness across the "
                         "labeled manifest (ignores --strategy)")
    ap.add_argument("--eval-e2e", action="store_true",
                    help="production-realistic eval: MLLM places its own clicks "
                         "(verify_loop), then generator makes masks; scores final "
                         "mask IoU vs GT with recall/stray decomposition")
    ap.add_argument("--clicker", default="verify_loop",
                    help="e2e click placer: verify_loop (single pass) | iterative "
                         "(multi-pass missed-creature re-sweep) | iterative_tiled "
                         "(iterative + high-res tiled recall pass)")
    ap.add_argument("--no-verify-clicks", action="store_true",
                    help="disable the content-based per-click verify pass in --eval-e2e")
    ap.add_argument("--repeats", type=int, default=1,
                    help="repeat each frame N times in --eval-e2e and report "
                         "mean+-std (tames MLLM stochasticity)")
    ap.add_argument("--variant", default="single",
                    help="eval mask-generator variant: single | refine | "
                         "box | multimask | refine_mm")
    args = ap.parse_args()

    if args.eval_e2e:
        run_eval_e2e(args.variant if args.variant != "single" else "refine_mm",
                     gt_min_prob=(args.gt_min_prob or 0.5),
                     max_clicks=args.max_sam3_clicks,
                     max_attempts=args.max_sam3_attempts,
                     clicker=args.clicker,
                     do_verify=not args.no_verify_clicks,
                     repeats=args.repeats)
        return

    if args.eval:
        run_eval(args.variant, gt_min_prob=(args.gt_min_prob or 0.5),
                 max_clicks=args.max_sam3_clicks,
                 max_attempts=args.max_sam3_attempts)
        return

    SRC["frame_outputs"] = args.frame_outputs
    SRC["video"] = args.video
    SRC["frames_dir"] = args.frames_dir
    if args.frames_dir:
        SRC["fps"] = None  # clip frames -> label neighbours by offset, not seconds
    offsets = ([int(x) for x in args.offsets.split(",")] if args.offsets
               else ([1, 2, 3] if args.frames_dir else DEFAULT_OFFSETS))

    tag = args.tag or f"f{args.target:03d}"
    out_dir = os.path.join(OUT_ROOT, tag)
    os.makedirs(out_dir, exist_ok=True)

    H, W, gt = load_target_and_gt(args.target, min_prob=args.gt_min_prob)
    frame = read_video_frame(args.target)
    target_path = os.path.join(out_dir, "target.png")
    cv2.imwrite(target_path, frame)
    render_gt_fixture(frame, gt, os.path.join(out_dir, "gt_fixture.png"))
    print(f"target frame {args.target}: {len(gt)} GT creatures, sizes "
          f"{[g['area'] for g in gt]}px, {W}x{H}")
    if args.render_gt_only:
        print(f"wrote {out_dir}/gt_fixture.png")
        return

    neighbours = extract_neighbours(args.target, offsets, out_dir, fps=SRC["fps"])
    print(f"neighbours: {[b for b,_ in neighbours]}")

    if args.strategy == "all":
        strategies = SINGLE_CALL_STRATEGIES + ["zoom", "iterative", "verify_loop"]
    else:
        strategies = [args.strategy]
    results = {}
    for s in strategies:
        if s == "zoom":
            sc = run_zoom(target_path, frame, gt, neighbours, W, H, out_dir)
        elif s == "iterative":
            sc = run_iterative(target_path, frame, gt, neighbours, W, H, out_dir)
        elif s == "iter_verify":
            sc = run_iterative(target_path, frame, gt, neighbours, W, H, out_dir,
                               refine="verify", label="iter_verify")
        elif s == "verify_loop":
            sc = run_verify_loop(target_path, frame, gt, neighbours, W, H, out_dir)
        else:
            sc = run_strategy(s, target_path, frame, gt, neighbours, W, H, out_dir)
        results[s] = sc
        print(f"  [{s}] coverage {sc['n_hit']}/{sc['n_gt']}  "
              f"fg_clicks {sc['n_fg_clicks']}  stray {sc['n_stray']}  "
              f"hits {sc['hits']}")
        if (args.sam3 or args.sam3_refine) and sc.get("groups"):
            sdir = os.path.join(out_dir, "iter_verify" if s == "iter_verify" else s)
            run_sam3_and_report(sc["groups"], target_path, frame, gt, W, H, sdir, s,
                                refine=args.sam3_refine,
                                max_clicks=args.max_sam3_clicks,
                                max_attempts=args.max_sam3_attempts)
    print("\n=== summary (coverage at tolerance px) ===")
    print(f"{'strategy':16s} {'t0':>4s} {'t8':>4s} {'t16':>4s} {'t24':>4s} {'t40':>4s}  stray")
    for s, sc in results.items():
        ca = sc["cov_at"]
        print(f"{s:16s} {ca[0]:>4d} {ca[8]:>4d} {ca[16]:>4d} {ca[24]:>4d} "
              f"{ca[40]:>4d}  {sc['n_stray']}")
    print("\nnearest click->GT distance (px), per strategy:")
    for s, sc in results.items():
        nd = {k: (round(v, 1) if v is not None else None)
              for k, v in sc["nearest_px"].items()}
        print(f"  {s:16s} {nd}")


if __name__ == "__main__":
    main()
