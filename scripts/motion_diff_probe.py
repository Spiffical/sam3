#!/usr/bin/env python3
"""Feasibility probe for a motion-difference pre-pass (click-engine recall).

Question this answers, cheaply and without SAM3/Claude: if we diff the target
frame against its temporal neighbours, do the resulting motion blobs actually
land ON the GT creatures (especially camouflaged ones), and is the signal
SPECIFIC (it does not just light up the whole frame)?

For each target frame we:
  1. load GT creature masks (from the same frame_outputs the eval uses),
  2. read the target + neighbour video frames,
  3. align neighbours to the target (ECC translation; camera is ~static),
  4. build a motion mask = thresholded median abs-diff across neighbours,
  5. report, per GT creature, the fraction of its mask covered by motion, and
     the global motion coverage of the whole frame (specificity).

If creature coverage >> frame coverage, motion-diff carries usable signal for
surfacing missed creatures.
"""
from __future__ import annotations

import argparse
import json
import os

import cv2
import numpy as np

from nibi_model_compare.frame_output_utils import decode_rle_to_mask

VIDEO = "assets/videos/onc/chinacreekclipped.mp4"
FRAME_OUTPUTS = (
    "runs/agent_every_frame/chinacreekclipped/"
    "claude_sonnet_4_6_60frame_20260525_155604/frame_outputs_rle.json"
)
OFFSETS = [15, 30, 45]
OUT_DIR = "runs/click_probe/motion_diff_probe"


def load_gt(frame_outputs, target_idx, min_prob=0.5):
    d = json.load(open(frame_outputs))
    H, W = d["frame_size_hw"]
    rec = next((fr for fr in d["frames"] if int(fr["frame_index"]) == target_idx), None)
    if rec is None:
        return None, None, []
    probs = rec.get("out_probs", [None] * len(rec["out_obj_ids"]))
    gt = []
    for oid, rle, prob in zip(rec["out_obj_ids"], rec["out_binary_masks_rle"], probs):
        if prob is not None and prob < min_prob:
            continue
        m = decode_rle_to_mask(rle, H, W).astype(bool)
        if m.any():
            gt.append({"id": int(oid), "mask": m, "area": int(m.sum())})
    return H, W, gt


def read_frame(video, idx, total):
    if idx < 0 or idx >= total:
        return None
    cap = cv2.VideoCapture(video)
    cap.set(cv2.CAP_PROP_POS_FRAMES, int(idx))
    ok, frame = cap.read()
    cap.release()
    return frame if ok else None


def align_to(ref_gray, mov_gray):
    """ECC translation-align mov->ref. Returns aligned gray or mov_gray on fail."""
    warp = np.eye(2, 3, dtype=np.float32)
    crit = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 50, 1e-4)
    try:
        cv2.findTransformECC(ref_gray, mov_gray, warp, cv2.MOTION_TRANSLATION, crit, None, 5)
        h, w = ref_gray.shape
        return cv2.warpAffine(mov_gray, warp, (w, h),
                              flags=cv2.INTER_LINEAR + cv2.WARP_INVERSE_MAP)
    except cv2.error:
        return mov_gray


def motion_mask(target, neighbours, blur=5, thresh=18, min_area=40):
    tg = cv2.GaussianBlur(cv2.cvtColor(target, cv2.COLOR_BGR2GRAY), (blur, blur), 0)
    diffs = []
    for nb in neighbours:
        ng = cv2.GaussianBlur(cv2.cvtColor(nb, cv2.COLOR_BGR2GRAY), (blur, blur), 0)
        ng = align_to(tg, ng)
        diffs.append(cv2.absdiff(tg, ng))
    if not diffs:
        return np.zeros(tg.shape, bool), []
    med = np.median(np.stack(diffs, 0), axis=0).astype(np.uint8)
    _, binm = cv2.threshold(med, thresh, 255, cv2.THRESH_BINARY)
    binm = cv2.morphologyEx(binm, cv2.MORPH_OPEN,
                            cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)))
    binm = cv2.morphologyEx(binm, cv2.MORPH_CLOSE,
                            cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7)))
    n, lab, stats, cent = cv2.connectedComponentsWithStats(binm, 8)
    blobs, keep = [], np.zeros_like(binm)
    for i in range(1, n):
        area = int(stats[i, cv2.CC_STAT_AREA])
        if area < min_area:
            continue
        keep[lab == i] = 255
        blobs.append({"area": area, "cx": float(cent[i][0]), "cy": float(cent[i][1])})
    return keep.astype(bool), blobs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--targets", type=int, nargs="*",
                    default=[42, 50, 59, 18, 24])  # hard + easy for contrast
    ap.add_argument("--thresh", type=int, default=18)
    ap.add_argument("--min-area", type=int, default=40)
    args = ap.parse_args()

    os.makedirs(OUT_DIR, exist_ok=True)
    d = json.load(open(FRAME_OUTPUTS))
    total = int(d.get("total_video_frames", 60))

    print(f"{'frame':9s} {'nGT':>3s} {'frmCov%':>7s}  per-creature motion coverage (cov% / area)")
    for tgt in args.targets:
        H, W, gt = load_gt(FRAME_OUTPUTS, tgt)
        if not gt:
            print(f"f{tgt:03d}: no GT")
            continue
        target = read_frame(VIDEO, tgt, total)
        nbs = []
        for off in OFFSETS:
            for nb_idx in (tgt - off, tgt + off):
                f = read_frame(VIDEO, nb_idx, total)
                if f is not None:
                    nbs.append(f)
        mm, blobs = motion_mask(target, nbs, thresh=args.thresh, min_area=args.min_area)
        frm_cov = 100.0 * mm.sum() / (H * W)
        per = []
        for g in gt:
            inter = np.logical_and(mm, g["mask"]).sum()
            cov = 100.0 * inter / max(1, g["area"])
            per.append(f"{cov:4.0f}%/{g['area']}")
        print(f"f{tgt:03d}     {len(gt):>3d} {frm_cov:>7.2f}  " + "  ".join(per))

        # save a visual: target with GT outlines (green) + motion blobs (magenta)
        vis = target.copy()
        ov = vis.copy()
        ov[mm] = (255, 0, 255)
        vis = cv2.addWeighted(ov, 0.4, vis, 0.6, 0)
        for g in gt:
            cs, _ = cv2.findContours(g["mask"].astype(np.uint8), cv2.RETR_EXTERNAL,
                                     cv2.CHAIN_APPROX_SIMPLE)
            cv2.drawContours(vis, cs, -1, (0, 255, 0), 2)
        for b in blobs:
            cv2.drawMarker(vis, (int(b["cx"]), int(b["cy"])), (0, 255, 255),
                           cv2.MARKER_TILTED_CROSS, 16, 2)
        cv2.imwrite(os.path.join(OUT_DIR, f"f{tgt:03d}_motion.png"), vis)


if __name__ == "__main__":
    main()
