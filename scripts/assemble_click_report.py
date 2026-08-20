#!/usr/bin/env python3
"""Assemble a final report across click_probe frames that ran with --sam3.

Scans runs/click_probe/*/<strategy>/sam3_metrics.json, stacks each frame's
3-panel sam3_report.png (RAW | GT | SAM3 masks) vertically with a frame-tag
banner, and writes:
  - runs/click_probe/REPORT.png      combined image
  - runs/click_probe/REPORT.md       per-frame + per-mask metrics tables
"""
import glob
import json
import os

import cv2
import numpy as np

ROOT = "runs/click_probe"


def main():
    metric_files = sorted(glob.glob(os.path.join(ROOT, "*", "*", "sam3_metrics.json")))
    if not metric_files:
        print("no sam3_metrics.json found under", ROOT)
        return

    panels = []
    md = ["# Click-engine + SAM3 mask report\n",
          "Pipeline: temporal-neighbour MLLM click placement (verify_loop) -> "
          "SAM3 point mode -> mask, matched to ground-truth by IoU.\n",
          "## Per-frame summary\n",
          "| frame | strategy | clicks/masks | GT | matched (IoU>=0.3) | mean IoU (matched) |",
          "|---|---|---|---|---|---|"]
    mask_rows = ["\n## Per-mask detail\n",
                 "| frame | creature | clicks | area px | SAM3 score | best IoU | matched GT |",
                 "|---|---|---|---|---|---|---|"]

    for mf in metric_files:
        strat_dir = os.path.dirname(mf)
        frame_tag = os.path.basename(os.path.dirname(strat_dir))
        m = json.load(open(mf))
        traces = (m.get("refine") or {}).get("traces", {})
        md.append(f"| {frame_tag} | {m['label']} | {m['n_pred']} | {m['n_gt']} | "
                  f"{m['n_matched_iou>=0.3']} | {m['mean_iou_matched']} |")
        for r in m["masks"]:
            tr = traces.get(str(r["creature_id"]))
            nclk = tr[-1]["n_clicks"] if tr else "-"
            mask_rows.append(f"| {frame_tag} | {r['description'][:44]} | {nclk} | "
                             f"{r['area_px']} | {r['score']} | {r['best_iou']} | "
                             f"{r['matched_gt']} |")

        rep = os.path.join(strat_dir, "sam3_report.png")
        if os.path.exists(rep):
            img = cv2.imread(rep)
            banner = np.zeros((30, img.shape[1], 3), np.uint8)
            cv2.putText(banner, frame_tag, (8, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                        (0, 255, 255), 2, cv2.LINE_AA)
            panels.append(np.vstack([banner, img]))

    if panels:
        wmax = max(p.shape[1] for p in panels)
        panels = [cv2.copyMakeBorder(p, 0, 0, 0, wmax - p.shape[1],
                                     cv2.BORDER_CONSTANT, value=(0, 0, 0))
                  if p.shape[1] < wmax else p for p in panels]
        combined = np.vstack(panels)
        out_png = os.path.join(ROOT, "REPORT.png")
        cv2.imwrite(out_png, combined)
        print("wrote", out_png, combined.shape)

    md += mask_rows
    out_md = os.path.join(ROOT, "REPORT.md")
    open(out_md, "w").write("\n".join(md) + "\n")
    print("wrote", out_md)
    print("\n".join(md))


if __name__ == "__main__":
    main()
