#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from sam3.competitions.fathomnet_2026.dataset import (
    load_competition_dataset,
    write_image_manifest,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Validate a FathomNet 2026 COCO dataset json and write a manifest "
            "for the downloaded image directory."
        )
    )
    parser.add_argument(
        "--dataset-json",
        required=True,
        help="Path to dataset_test.json or dataset_train.json.",
    )
    parser.add_argument(
        "--images-dir",
        required=True,
        help="Directory containing downloaded competition images.",
    )
    parser.add_argument(
        "--output-path",
        required=True,
        help="Where to write the manifest JSON.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    dataset = load_competition_dataset(args.dataset_json)
    output_path = write_image_manifest(
        dataset=dataset,
        images_dir=args.images_dir,
        output_path=args.output_path,
    )
    print(
        "Wrote FathomNet 2026 manifest "
        f"for {len(dataset.images)} images to {output_path}"
    )


if __name__ == "__main__":
    main()
