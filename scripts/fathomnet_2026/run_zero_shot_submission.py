#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from sam3.competitions.fathomnet_2026.dataset import load_competition_dataset
from sam3.competitions.fathomnet_2026.zero_shot import (
    ZeroShotRunConfig,
    run_zero_shot_submission,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run the FathomNet 2026 zero-shot SAM3 + Qwen 3.5 submission pipeline "
            "over a COCO-format competition split."
        )
    )
    parser.add_argument("--dataset-json", required=True, help="Path to dataset_test.json.")
    parser.add_argument(
        "--images-dir",
        required=True,
        help="Directory containing the extracted image files for the split.",
    )
    parser.add_argument(
        "--output-dir",
        required=True,
        help="Directory where predictions, summaries, and submission.csv will be written.",
    )
    parser.add_argument(
        "--server-url",
        default="http://127.0.0.1:8006/v1",
        help="OpenAI-compatible server URL for the running Qwen 3.5 endpoint.",
    )
    parser.add_argument(
        "--model",
        default="Qwen/Qwen3.5-27B",
        help="Model name forwarded to the OpenAI-compatible endpoint.",
    )
    parser.add_argument("--api-key", default="", help="Optional API key override.")
    parser.add_argument("--device", default="cuda", help="SAM3 device, default: cuda")
    parser.add_argument(
        "--checkpoint-path",
        default="",
        help="Optional local SAM3 checkpoint path.",
    )
    parser.add_argument(
        "--prompt-profile",
        default="fathomnet_2026",
        help="SAM3 agent prompt profile, default: fathomnet_2026",
    )
    parser.add_argument(
        "--max-generations",
        type=int,
        default=10,
        help="Maximum Qwen tool-call generations per proposal pass.",
    )
    parser.add_argument(
        "--max-completion-tokens",
        type=int,
        default=1024,
        help="Maximum completion tokens for proposal and classification calls.",
    )
    parser.add_argument(
        "--image-detail",
        default="high",
        help="OpenAI image detail level for Qwen image inputs.",
    )
    parser.add_argument(
        "--max-images-per-request",
        type=int,
        default=3,
        help="Maximum images per multimodal request.",
    )
    parser.add_argument(
        "--agent-image-max-edge",
        type=int,
        default=768,
        help="Max image edge used for Qwen agent requests.",
    )
    parser.add_argument(
        "--agent-image-min-edge",
        type=int,
        default=384,
        help="Min image edge used for Qwen agent requests.",
    )
    parser.add_argument(
        "--confidence-threshold",
        type=float,
        default=0.0,
        help="SAM3 processor confidence threshold.",
    )
    parser.add_argument(
        "--max-images",
        type=int,
        default=0,
        help="Optional cap for quick smoke tests.",
    )
    parser.add_argument(
        "--proposal-iou-threshold",
        type=float,
        default=0.75,
        help="IoU threshold used to deduplicate proposal boxes across prompts.",
    )
    parser.add_argument(
        "--final-iou-threshold",
        type=float,
        default=0.55,
        help="Per-category IoU threshold used for the final NMS pass.",
    )
    parser.add_argument(
        "--crop-context-ratio",
        type=float,
        default=0.2,
        help="Relative crop padding used during Qwen category classification.",
    )
    parser.add_argument(
        "--classification-min-confidence",
        type=float,
        default=0.2,
        help="Minimum Qwen classification confidence required to keep a detection.",
    )
    parser.add_argument(
        "--compile-image-model",
        action="store_true",
        help="Compile the SAM3 image model before inference.",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Keep per-image proposal artifacts and histories.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    dataset = load_competition_dataset(args.dataset_json)
    config = ZeroShotRunConfig(
        server_url=args.server_url,
        model=args.model,
        api_key=args.api_key,
        device=args.device,
        checkpoint_path=args.checkpoint_path,
        prompt_profile=args.prompt_profile,
        max_generations=args.max_generations,
        max_completion_tokens=args.max_completion_tokens,
        image_detail=args.image_detail,
        max_images_per_request=args.max_images_per_request,
        agent_image_max_edge=args.agent_image_max_edge,
        agent_image_min_edge=args.agent_image_min_edge,
        confidence_threshold=args.confidence_threshold,
        compile_image_model=args.compile_image_model,
        max_images=args.max_images,
        proposal_iou_threshold=args.proposal_iou_threshold,
        final_iou_threshold=args.final_iou_threshold,
        crop_context_ratio=args.crop_context_ratio,
        classification_min_confidence=args.classification_min_confidence,
        keep_debug_artifacts=args.debug,
    )
    summary = run_zero_shot_submission(
        dataset=dataset,
        images_dir=args.images_dir,
        output_dir=args.output_dir,
        config=config,
    )
    print(
        "Completed FathomNet 2026 zero-shot run "
        f"for {summary['image_count']} image(s). "
        f"Submission CSV: {summary['submission_csv_path']}"
    )


if __name__ == "__main__":
    main()
