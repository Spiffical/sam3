#!/usr/bin/env python3
"""Run the Set-of-Mark missed-creature discovery stage on a video.

Architecture: for each target frame the MLLM proposes click coordinates
for creatures it identifies as MISSED by the text-agent, then SAM3
image point-mode segments each click into a candidate mask, and a
separate MLLM judge step accepts/rejects the numbered candidates.

Spec: docs/superpowers/specs/2026-05-26-som-missed-creature-loop-design.md
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from functools import partial
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from nibi_model_compare.som_missed_creatures import (  # noqa: E402
    Sam3PointService,
    SomStageConfig,
    run_som_stage,
)


def _csv_ints(text: str) -> list[int]:
    return [int(x) for x in text.split(",") if x.strip()]


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

    # Deprecated: --broad-prompts is no longer used; kept as a no-op with a warning.
    p.add_argument("--broad-prompts", default=None,
                   help=argparse.SUPPRESS)  # hidden; no-op

    p.add_argument("--discovery-num-neighbours", type=int, default=4,
                   help="Reference frames per side for MLLM discovery call. "
                        "Total discovery neighbours = 2 * this.")
    p.add_argument("--num-neighbours", type=int, default=2,
                   help="Reference frames per side for MLLM judge call. "
                        "Total judge neighbours = 2 * this.")
    p.add_argument("--neighbour-offset-frames", type=int, default=30,
                   help="Frame gap between target and each neighbour ring.")
    p.add_argument("--iou-dedup", type=float, default=0.3)
    p.add_argument("--min-area-px", type=float, default=350.0)
    p.add_argument("--max-area-frac", type=float, default=0.5)
    p.add_argument("--edge-tol-px", type=int, default=2)
    p.add_argument("--internal-iou-dedup", type=float, default=0.5)
    p.add_argument("--max-mllm-calls", type=int, default=200)

    # SAM3 model loading
    p.add_argument("--device", default="cuda")
    p.add_argument("--checkpoint-path", default=None,
                   help="Path to SAM3 checkpoint. If unset, build_sam3_image_model "
                        "uses its own default.")
    p.add_argument("--confidence-threshold", type=float, default=0.5)
    p.add_argument("--compile", action="store_true")
    p.add_argument("--bpe-path", default=None,
                   help="Optional BPE path. If unset, uses find_bpe_path() from "
                        "the every-frame script.")

    # Anthropic / MLLM
    p.add_argument("--claude-model",
                   default=os.environ.get("ANTHROPIC_MODEL", "claude-sonnet-4-6"))
    p.add_argument("--max-completion-tokens", type=int, default=2048)
    p.add_argument("--image-detail", default="high", choices=["low", "high"])
    p.add_argument("--max-images-per-request", type=int, default=8)

    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)

    if args.broad_prompts is not None:
        print(
            "[som] WARNING: --broad-prompts is deprecated and has no effect. "
            "The discovery step now uses MLLM-proposed clicks + SAM3 text-mode."
        )

    # ---------------------------------------------------------------------------
    # Load SAM3 runtime dependencies (mirrors run_sam3_agent_every_frame_video.py)
    # ---------------------------------------------------------------------------
    from scripts.run_sam3_agent_every_frame_video import (  # noqa: E402
        ensure_runtime_deps,
        find_bpe_path,
    )
    import scripts.run_sam3_agent_every_frame_video as _efv  # noqa: E402

    ensure_runtime_deps()

    Sam3Processor = _efv.Sam3Processor
    build_sam3_image_model = _efv.build_sam3_image_model

    bpe_path = args.bpe_path or find_bpe_path()

    # Build the image model with enable_inst_interactivity=True so that
    # model.predict_inst is available for SAM3 click-mode segmentation.
    image_model = build_sam3_image_model(
        bpe_path=bpe_path,
        device=args.device,
        checkpoint_path=args.checkpoint_path,
        compile=args.compile,
        enable_inst_interactivity=True,
    )
    image_processor = Sam3Processor(
        image_model, confidence_threshold=args.confidence_threshold
    )

    # Sam3PointService uses the real SAM3 click API:
    #   processor.set_image(image) -> inference_state
    #   model.predict_inst(state, point_coords, point_labels, multimask_output=True)
    sam3_point_service = Sam3PointService(image_model, image_processor)

    # ---------------------------------------------------------------------------
    # Bind the Anthropic Claude callable
    # ---------------------------------------------------------------------------
    os.environ["SAM3_IMAGE_DETAIL"] = str(args.image_detail)
    os.environ["SAM3_MAX_IMAGES_PER_REQUEST"] = str(max(1, args.max_images_per_request))

    try:
        from dotenv import load_dotenv  # noqa: E402
        load_dotenv()
    except ImportError:
        pass  # python-dotenv is optional; fall back to os.environ only

    from sam3.agent.client_claude import send_claude_request  # noqa: E402

    anthropic_key = os.environ.get("ANTHROPIC_API_KEY")
    if not anthropic_key:
        raise SystemExit(
            "ANTHROPIC_API_KEY must be set (in .env or env) for SoM stage."
        )

    send_mllm = partial(
        send_claude_request,
        model=args.claude_model,
        api_key=anthropic_key,
        max_tokens=args.max_completion_tokens,
    )

    # ---------------------------------------------------------------------------
    # Build config and run
    # ---------------------------------------------------------------------------
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
        num_neighbours=args.num_neighbours,
        neighbour_offset_frames=args.neighbour_offset_frames,
        iou_dedup=args.iou_dedup,
        min_area_px=args.min_area_px,
        max_area_frac=args.max_area_frac,
        edge_tol_px=args.edge_tol_px,
        internal_iou_dedup=args.internal_iou_dedup,
        max_mllm_calls=args.max_mllm_calls,
        discovery_num_neighbours=args.discovery_num_neighbours,
    )

    print(f"[som] writing artefacts to {output_dir}")
    stats = run_som_stage(
        cfg,
        _send_mllm_request=send_mllm,
        _sam3_point_service=sam3_point_service,
    )
    print(f"[som] done: {stats}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
