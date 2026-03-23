#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from sam3.competitions.fathomnet_2026.bundle import (
    DEFAULT_BUNDLE_FILENAME,
    DEFAULT_COMPETITION_NAME,
    DEFAULT_DATASET_SUBDIR,
    DEFAULT_METADATA_FILENAME,
    DEFAULT_PROJECT_DATA_ROOT,
    build_project_data_layout,
    create_tar_zstd_bundle,
    discover_dataset_layout,
    download_competition_with_kaggle,
    ensure_project_data_layout,
    extract_download_archives,
    write_bundle_metadata,
    write_manifests_for_discovered_layout,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Download the FathomNet 2026 Kaggle competition data, expand it under "
            "/project storage, and create a tar.zst bundle for Slurm staging."
        )
    )
    parser.add_argument(
        "--competition",
        default=DEFAULT_COMPETITION_NAME,
        help="Kaggle competition slug, default: fathomnet-2026",
    )
    parser.add_argument(
        "--project-data-root",
        default=DEFAULT_PROJECT_DATA_ROOT,
        help="Shared project data root on Nibi.",
    )
    parser.add_argument(
        "--dataset-subdir",
        default=DEFAULT_DATASET_SUBDIR,
        help="Subdirectory under the project data root for this competition.",
    )
    parser.add_argument(
        "--bundle-filename",
        default=DEFAULT_BUNDLE_FILENAME,
        help="Bundle filename written under bundles/.",
    )
    parser.add_argument(
        "--metadata-filename",
        default=DEFAULT_METADATA_FILENAME,
        help="Metadata filename written under bundles/.",
    )
    parser.add_argument(
        "--skip-download",
        action="store_true",
        help="Skip the Kaggle download step and reuse whatever is already in downloads/.",
    )
    parser.add_argument(
        "--skip-bundle",
        action="store_true",
        help="Skip tar.zst bundle creation after extraction/manifests.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Force redownload/reextract/rebundle when possible.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    layout = build_project_data_layout(
        project_data_root=args.project_data_root,
        dataset_subdir=args.dataset_subdir,
    )
    ensure_project_data_layout(layout)

    if not args.skip_download:
        zip_paths = download_competition_with_kaggle(
            layout=layout,
            competition_name=args.competition,
            force=args.force,
        )
        print(f"Downloaded {len(zip_paths)} archive(s) into {layout.downloads_dir}")
    else:
        print(f"Skipping Kaggle download; reusing archives in {layout.downloads_dir}")

    extracted_root = extract_download_archives(
        downloads_dir=layout.downloads_dir,
        extracted_dir=layout.extracted_dir,
        force=args.force,
    )
    discovered = discover_dataset_layout(extracted_root)
    manifests = write_manifests_for_discovered_layout(
        discovered=discovered,
        manifests_dir=layout.manifests_dir,
    )

    print(f"Expanded competition files under {extracted_root}")
    if discovered.train_dataset_json is not None:
        print(f"Train JSON: {discovered.train_dataset_json}")
        print(f"Train images: {discovered.train_images_dir}")
    if discovered.test_dataset_json is not None:
        print(f"Test JSON: {discovered.test_dataset_json}")
        print(f"Test images: {discovered.test_images_dir}")
    for split_name, manifest_path in sorted(manifests.items()):
        print(f"{split_name.title()} manifest: {manifest_path}")

    bundle_path = layout.bundles_dir / args.bundle_filename
    metadata_path = layout.bundles_dir / args.metadata_filename
    write_bundle_metadata(
        discovered=discovered,
        bundle_path=bundle_path,
        metadata_path=metadata_path,
    )

    if not args.skip_bundle:
        create_tar_zstd_bundle(
            extracted_root=extracted_root,
            bundle_path=bundle_path,
        )
        print(f"Bundle written to {bundle_path}")
    else:
        print(f"Skipping bundle creation; metadata written to {metadata_path}")

    print(f"Bundle metadata: {metadata_path}")


if __name__ == "__main__":
    main()
