#!/usr/bin/env python3
"""Run the Set-of-Mark missed-creature discovery stage on a video.

Spec: docs/superpowers/specs/2026-05-26-som-missed-creature-loop-design.md
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from nibi_model_compare.som_missed_creatures import (  # noqa: E402
    SomStageConfig,
    run_som_stage,
)


def _csv_ints(text: str) -> list[int]:
    return [int(x) for x in text.split(",") if x.strip()]


def _csv_strs(text: str) -> list[str]:
    return [x.strip() for x in text.split(",") if x.strip()]


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("video_path")
    p.add_argument("--frame-results", required=True,
                   help="Path to frame_results.jsonl from the text-agent run.")
    p.add_argument("--frame-outputs", required=True,
                   help="Path to frame_outputs_rle.json from the text-agent run.")
    p.add_argument("--output-dir", required=True)
    p.add_argument("--prompt-profile", choices=["underwater", "general"],
                   default="underwater")
    p.add_argument("--prompt", default="small creatures",
                   help="Original creature query echoed into the MLLM prompt.")
    p.add_argument("--num-target-frames", type=int, default=8)
    p.add_argument("--frame-selection-strategy",
                   choices=["uniform", "motion"], default="uniform")
    p.add_argument("--target-frames", type=_csv_ints, default=None,
                   help="Explicit comma-separated frame indices (overrides "
                        "strategy + num-target-frames).")
    p.add_argument("--broad-prompts", type=_csv_strs,
                   default=["creature", "animal", "organism"])
    p.add_argument("--num-neighbours", type=int, default=2,
                   help="Reference frames per side. Total neighbours = 2 * this.")
    p.add_argument("--neighbour-offset-frames", type=int, default=30,
                   help="Frame gap between target and each neighbour ring.")
    p.add_argument("--iou-dedup", type=float, default=0.3)
    p.add_argument("--min-area-px", type=float, default=350.0)
    p.add_argument("--max-area-frac", type=float, default=0.5)
    p.add_argument("--edge-tol-px", type=int, default=2)
    p.add_argument("--internal-iou-dedup", type=float, default=0.5)
    p.add_argument("--max-mllm-calls", type=int, default=200)
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)

    # Stamp the output dir so reruns are isolated
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    output_dir = os.path.join(args.output_dir, timestamp)
    os.makedirs(output_dir, exist_ok=True)

    cfg = SomStageConfig(
        video_path=args.video_path,
        frame_results_path=args.frame_results,
        frame_outputs_path=args.frame_outputs,
        output_dir=output_dir,
        prompt_profile=args.prompt_profile,
        initial_text_prompt=args.prompt,
        num_target_frames=args.num_target_frames,
        frame_selection_strategy=args.frame_selection_strategy,
        target_frames_explicit=args.target_frames,
        broad_prompts=args.broad_prompts,
        num_neighbours=args.num_neighbours,
        neighbour_offset_frames=args.neighbour_offset_frames,
        iou_dedup=args.iou_dedup,
        min_area_px=args.min_area_px,
        max_area_frac=args.max_area_frac,
        edge_tol_px=args.edge_tol_px,
        internal_iou_dedup=args.internal_iou_dedup,
        max_mllm_calls=args.max_mllm_calls,
    )

    print(f"[som] writing artefacts to {output_dir}")
    stats = run_som_stage(cfg)
    print(f"[som] done: {stats}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
