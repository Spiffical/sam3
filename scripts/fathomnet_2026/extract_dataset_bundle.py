#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from sam3.competitions.fathomnet_2026.bundle import (
    extract_bundle,
    resolve_split_paths_from_metadata,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Extract a prepared FathomNet 2026 tar.zst bundle into a target "
            "directory and write a staging manifest for a chosen split."
        )
    )
    parser.add_argument("--bundle-path", required=True, help="Path to the tar.zst bundle.")
    parser.add_argument(
        "--metadata-path",
        required=True,
        help="Path to the metadata JSON created alongside the bundle.",
    )
    parser.add_argument(
        "--output-dir",
        required=True,
        help="Directory where the bundle should be extracted.",
    )
    parser.add_argument(
        "--split",
        default="test",
        choices=["train", "test"],
        help="Which split paths to resolve after extraction.",
    )
    parser.add_argument(
        "--staging-manifest",
        required=True,
        help="Where to write the staging manifest JSON.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    extracted_base = extract_bundle(
        bundle_path=args.bundle_path,
        output_dir=args.output_dir,
    )
    extracted_root, dataset_json_path, images_dir = resolve_split_paths_from_metadata(
        metadata_path=args.metadata_path,
        extracted_base_dir=extracted_base,
        split=args.split,
    )

    payload = {
        "bundle_path": str(Path(args.bundle_path).resolve()),
        "metadata_path": str(Path(args.metadata_path).resolve()),
        "extracted_base_dir": str(extracted_base),
        "extracted_root": str(extracted_root),
        "split": args.split,
        "dataset_json_path": str(dataset_json_path),
        "images_dir": str(images_dir),
    }
    staging_manifest = Path(args.staging_manifest).resolve()
    staging_manifest.parent.mkdir(parents=True, exist_ok=True)
    staging_manifest.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
