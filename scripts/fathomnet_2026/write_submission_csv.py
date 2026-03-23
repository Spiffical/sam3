#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from sam3.competitions.fathomnet_2026.dataset import load_competition_dataset
from sam3.competitions.fathomnet_2026.submission import (
    load_predictions_file,
    validate_prediction_ids,
    write_submission_csv,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Convert FathomNet 2026 JSON or JSONL predictions into a Kaggle "
            "submission CSV."
        )
    )
    parser.add_argument(
        "--predictions-path",
        required=True,
        help="Path to a JSON list or JSONL file of detection predictions.",
    )
    parser.add_argument(
        "--output-csv",
        required=True,
        help="Where to write the Kaggle submission CSV.",
    )
    parser.add_argument(
        "--dataset-json",
        default="",
        help=(
            "Optional COCO dataset json used to validate image_id and category_id "
            "values before writing the CSV."
        ),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    predictions = load_predictions_file(args.predictions_path)

    if args.dataset_json:
        dataset = load_competition_dataset(args.dataset_json)
        validate_prediction_ids(
            predictions=predictions,
            valid_image_ids={image.id for image in dataset.images},
            valid_category_ids={category.id for category in dataset.categories},
        )

    output_csv = write_submission_csv(predictions, args.output_csv)
    print(
        "Wrote FathomNet 2026 submission "
        f"with {len(predictions)} rows to {output_csv}"
    )


if __name__ == "__main__":
    main()
