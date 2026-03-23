from __future__ import annotations

import json
import importlib.util
import shutil
import subprocess
import sys
import tarfile
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from .dataset import load_competition_dataset, load_coco_payload, write_image_manifest


DEFAULT_COMPETITION_NAME = "fathomnet-2026"
DEFAULT_PROJECT_DATA_ROOT = "/project/rpp-kmoran/merileo/data"
DEFAULT_DATASET_SUBDIR = "fathomnet_2026_kaggle"
DEFAULT_BUNDLE_FILENAME = "fathomnet_2026_kaggle.tar.zst"
DEFAULT_METADATA_FILENAME = "fathomnet_2026_kaggle.metadata.json"


@dataclass(frozen=True)
class ProjectDataLayout:
    root_dir: Path
    downloads_dir: Path
    extracted_dir: Path
    manifests_dir: Path
    bundles_dir: Path


@dataclass(frozen=True)
class DiscoveredDatasetLayout:
    extracted_root: Path
    train_dataset_json: Optional[Path]
    test_dataset_json: Optional[Path]
    train_images_dir: Optional[Path]
    test_images_dir: Optional[Path]


def build_project_data_layout(
    *,
    project_data_root: str | Path = DEFAULT_PROJECT_DATA_ROOT,
    dataset_subdir: str = DEFAULT_DATASET_SUBDIR,
) -> ProjectDataLayout:
    root_dir = Path(project_data_root).expanduser().resolve() / dataset_subdir
    return ProjectDataLayout(
        root_dir=root_dir,
        downloads_dir=root_dir / "downloads",
        extracted_dir=root_dir / "expanded",
        manifests_dir=root_dir / "manifests",
        bundles_dir=root_dir / "bundles",
    )


def ensure_project_data_layout(layout: ProjectDataLayout) -> None:
    for path in (
        layout.root_dir,
        layout.downloads_dir,
        layout.extracted_dir,
        layout.manifests_dir,
        layout.bundles_dir,
    ):
        path.mkdir(parents=True, exist_ok=True)


def _run_command(command: list[str], *, cwd: Optional[Path] = None) -> None:
    subprocess.run(
        command,
        cwd=str(cwd) if cwd is not None else None,
        check=True,
    )


def _resolve_kaggle_command() -> list[str]:
    kaggle_exe = shutil.which("kaggle")
    if kaggle_exe:
        return [kaggle_exe]
    if importlib.util.find_spec("kaggle") is not None:
        return [sys.executable, "-m", "kaggle"]
    raise RuntimeError(
        "Could not find the Kaggle CLI. Install the `kaggle` package in the active "
        "environment or make sure the `kaggle` executable is on PATH."
    )


def download_competition_with_kaggle(
    *,
    layout: ProjectDataLayout,
    competition_name: str = DEFAULT_COMPETITION_NAME,
    force: bool = False,
) -> list[Path]:
    ensure_project_data_layout(layout)
    kaggle_cmd = _resolve_kaggle_command()
    command = kaggle_cmd + [
        "competitions",
        "download",
        "-c",
        competition_name,
        "-p",
        str(layout.downloads_dir),
    ]
    if force:
        command.append("--force")
    _run_command(command)
    return sorted(layout.downloads_dir.glob("*.zip"))


def _copy_non_zip_downloads(downloads_dir: Path, extracted_dir: Path) -> None:
    for source_path in downloads_dir.rglob("*"):
        if not source_path.is_file():
            continue
        if source_path.suffix.lower() == ".zip":
            continue
        relative_path = source_path.relative_to(downloads_dir)
        destination = extracted_dir / relative_path
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source_path, destination)


def extract_download_archives(
    *,
    downloads_dir: str | Path,
    extracted_dir: str | Path,
    force: bool = False,
) -> Path:
    downloads_root = Path(downloads_dir).resolve()
    extracted_root = Path(extracted_dir).resolve()
    if force and extracted_root.exists():
        shutil.rmtree(extracted_root)
    extracted_root.mkdir(parents=True, exist_ok=True)

    _copy_non_zip_downloads(downloads_root, extracted_root)

    processed_zips: set[Path] = set()
    while True:
        pending = sorted(
            {
                path.resolve()
                for search_root in (downloads_root, extracted_root)
                for path in search_root.rglob("*.zip")
                if path.resolve() not in processed_zips
            }
        )
        if not pending:
            break
        for zip_path in pending:
            target_dir = (
                extracted_root
                if downloads_root in zip_path.parents or zip_path.parent == downloads_root
                else zip_path.parent
            )
            with zipfile.ZipFile(zip_path, "r") as archive:
                archive.extractall(target_dir)
            processed_zips.add(zip_path.resolve())
    return extracted_root


def _find_dataset_json(
    extracted_root: Path,
    file_name: str,
) -> Optional[Path]:
    matches = sorted(extracted_root.rglob(file_name))
    return matches[0].resolve() if matches else None


def guess_images_dir(
    *,
    dataset_json_path: Path,
    search_root: Path,
    sample_size: int = 25,
) -> Optional[Path]:
    payload = load_coco_payload(dataset_json_path)
    raw_images = payload.get("images")
    if not isinstance(raw_images, list) or not raw_images:
        return None

    sample_file_names = [
        str(item.get("file_name"))
        for item in raw_images[: max(1, int(sample_size))]
        if isinstance(item, dict) and isinstance(item.get("file_name"), str)
    ]
    if not sample_file_names:
        return None

    direct_hits = sum((search_root / file_name).is_file() for file_name in sample_file_names)
    if direct_hits == len(sample_file_names):
        return search_root.resolve()

    basename_set = {Path(file_name).name for file_name in sample_file_names}
    candidate_hit_counts: dict[Path, int] = {}
    for file_path in search_root.rglob("*"):
        if not file_path.is_file():
            continue
        if file_path.name not in basename_set:
            continue
        parent = file_path.parent.resolve()
        candidate_hit_counts[parent] = candidate_hit_counts.get(parent, 0) + 1

    if not candidate_hit_counts:
        return None

    best_dir = max(
        candidate_hit_counts.items(),
        key=lambda item: (item[1], -len(str(item[0]))),
    )[0]
    return best_dir.resolve()


def discover_dataset_layout(extracted_root: str | Path) -> DiscoveredDatasetLayout:
    root = Path(extracted_root).resolve()
    train_dataset_json = _find_dataset_json(root, "dataset_train.json")
    test_dataset_json = _find_dataset_json(root, "dataset_test.json")
    train_images_dir = (
        guess_images_dir(dataset_json_path=train_dataset_json, search_root=root)
        if train_dataset_json is not None
        else None
    )
    test_images_dir = (
        guess_images_dir(dataset_json_path=test_dataset_json, search_root=root)
        if test_dataset_json is not None
        else None
    )
    return DiscoveredDatasetLayout(
        extracted_root=root,
        train_dataset_json=train_dataset_json,
        test_dataset_json=test_dataset_json,
        train_images_dir=train_images_dir,
        test_images_dir=test_images_dir,
    )


def write_manifests_for_discovered_layout(
    *,
    discovered: DiscoveredDatasetLayout,
    manifests_dir: str | Path,
) -> dict[str, Path]:
    manifests_root = Path(manifests_dir).resolve()
    manifests_root.mkdir(parents=True, exist_ok=True)
    outputs: dict[str, Path] = {}

    if discovered.train_dataset_json is not None and discovered.train_images_dir is not None:
        train_dataset = load_competition_dataset(discovered.train_dataset_json)
        outputs["train"] = write_image_manifest(
            dataset=train_dataset,
            images_dir=discovered.train_images_dir,
            output_path=manifests_root / "dataset_train_manifest.json",
        )

    if discovered.test_dataset_json is not None and discovered.test_images_dir is not None:
        test_dataset = load_competition_dataset(discovered.test_dataset_json)
        outputs["test"] = write_image_manifest(
            dataset=test_dataset,
            images_dir=discovered.test_images_dir,
            output_path=manifests_root / "dataset_test_manifest.json",
        )

    return outputs


def create_tar_zstd_bundle(
    *,
    extracted_root: str | Path,
    bundle_path: str | Path,
) -> Path:
    extracted = Path(extracted_root).resolve()
    bundle = Path(bundle_path).resolve()
    bundle.parent.mkdir(parents=True, exist_ok=True)
    if bundle.exists():
        bundle.unlink()

    if str(bundle).endswith(".tar.gz") or str(bundle).endswith(".tgz"):
        with tarfile.open(bundle, "w:gz") as archive:
            archive.add(extracted, arcname=extracted.name)
        return bundle

    if not str(bundle).endswith(".tar.zst"):
        raise ValueError(
            f"Unsupported bundle format for {bundle}. "
            "Use .tar.zst for Nibi or .tar.gz for local fallback."
        )

    if shutil.which("zstd") is None:
        raise RuntimeError(
            "zstd is required to create a .tar.zst bundle. "
            "Install zstd or use a --bundle-filename ending in .tar.gz."
        )

    _run_command(
        [
            "tar",
            "-I",
            "zstd -T0 -19",
            "-cf",
            str(bundle),
            "-C",
            str(extracted.parent),
            extracted.name,
        ]
    )
    return bundle


def write_bundle_metadata(
    *,
    discovered: DiscoveredDatasetLayout,
    bundle_path: str | Path,
    metadata_path: str | Path,
) -> Path:
    bundle = Path(bundle_path).resolve()
    metadata = Path(metadata_path).resolve()
    metadata.parent.mkdir(parents=True, exist_ok=True)

    def _relative_or_empty(path: Optional[Path]) -> str:
        if path is None:
            return ""
        return str(path.resolve().relative_to(discovered.extracted_root))

    payload = {
        "competition": DEFAULT_COMPETITION_NAME,
        "bundle_path": str(bundle),
        "extracted_root_name": discovered.extracted_root.name,
        "train": {
            "dataset_json_relpath": _relative_or_empty(discovered.train_dataset_json),
            "images_dir_relpath": _relative_or_empty(discovered.train_images_dir),
        },
        "test": {
            "dataset_json_relpath": _relative_or_empty(discovered.test_dataset_json),
            "images_dir_relpath": _relative_or_empty(discovered.test_images_dir),
        },
    }
    with metadata.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)
        handle.write("\n")
    return metadata


def extract_bundle(
    *,
    bundle_path: str | Path,
    output_dir: str | Path,
) -> Path:
    bundle = Path(bundle_path).resolve()
    output_root = Path(output_dir).resolve()
    output_root.mkdir(parents=True, exist_ok=True)

    if str(bundle).endswith(".tar.gz") or str(bundle).endswith(".tgz"):
        with tarfile.open(bundle, "r:gz") as archive:
            archive.extractall(output_root)
        return output_root

    if not str(bundle).endswith(".tar.zst"):
        raise ValueError(
            f"Unsupported bundle format for {bundle}. "
            "Use .tar.zst or .tar.gz."
        )

    if shutil.which("zstd") is None:
        raise RuntimeError(
            "zstd is required to extract a .tar.zst bundle. "
            "Install zstd or use a .tar.gz fallback bundle."
        )

    _run_command(
        [
            "tar",
            "-I",
            "zstd -T0 -19",
            "-xf",
            str(bundle),
            "-C",
            str(output_root),
        ]
    )
    return output_root


def resolve_split_paths_from_metadata(
    *,
    metadata_path: str | Path,
    extracted_base_dir: str | Path,
    split: str,
) -> tuple[Path, Path, Path]:
    metadata = json.loads(Path(metadata_path).read_text(encoding="utf-8"))
    split_key = str(split).strip().lower()
    if split_key not in {"train", "test"}:
        raise ValueError(f"Unsupported split: {split}")
    extracted_base = Path(extracted_base_dir).resolve()
    extracted_root = extracted_base / metadata["extracted_root_name"]
    split_payload = metadata.get(split_key, {})
    dataset_json_relpath = split_payload.get("dataset_json_relpath", "")
    images_dir_relpath = split_payload.get("images_dir_relpath", "")
    if not dataset_json_relpath or not images_dir_relpath:
        raise ValueError(
            f"Metadata for split={split_key!r} is incomplete in {metadata_path}"
        )
    dataset_json_path = extracted_root / dataset_json_relpath
    images_dir = extracted_root / images_dir_relpath
    return extracted_root, dataset_json_path.resolve(), images_dir.resolve()
