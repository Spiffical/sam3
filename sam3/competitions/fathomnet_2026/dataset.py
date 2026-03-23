from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Optional

from .categories import CompetitionCategory, load_categories_from_coco_payload


@dataclass(frozen=True)
class CompetitionImage:
    id: int
    file_name: str
    coco_url: Optional[str] = None
    width: Optional[int] = None
    height: Optional[int] = None


@dataclass(frozen=True)
class CompetitionDataset:
    json_path: Path
    images: tuple[CompetitionImage, ...]
    categories: tuple[CompetitionCategory, ...]


def load_coco_payload(json_path: str | Path) -> dict[str, Any]:
    path = Path(json_path)
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise ValueError(f"Expected a JSON object in {path}, got {type(payload)!r}")
    return payload


def load_competition_dataset(json_path: str | Path) -> CompetitionDataset:
    path = Path(json_path).resolve()
    payload = load_coco_payload(path)
    categories = tuple(load_categories_from_coco_payload(payload))

    raw_images = payload.get("images")
    if not isinstance(raw_images, list) or not raw_images:
        raise ValueError("COCO payload must include a non-empty 'images' list.")

    images: list[CompetitionImage] = []
    seen_image_ids: set[int] = set()
    for index, raw_image in enumerate(raw_images):
        if not isinstance(raw_image, Mapping):
            raise ValueError(f"Image at index {index} is not a mapping.")

        image_id = raw_image.get("id")
        file_name = raw_image.get("file_name")
        width = raw_image.get("width")
        height = raw_image.get("height")
        coco_url = raw_image.get("coco_url")

        if not isinstance(image_id, int):
            raise ValueError(f"Image at index {index} is missing an integer 'id'.")
        if not isinstance(file_name, str) or not file_name.strip():
            raise ValueError(
                f"Image at index {index} is missing a non-empty 'file_name'."
            )
        if image_id in seen_image_ids:
            raise ValueError(f"Duplicate image id found: {image_id}")
        if width is not None and not isinstance(width, int):
            raise ValueError(f"Image {image_id} has a non-integer 'width'.")
        if height is not None and not isinstance(height, int):
            raise ValueError(f"Image {image_id} has a non-integer 'height'.")
        if coco_url is not None and not isinstance(coco_url, str):
            raise ValueError(f"Image {image_id} has a non-string 'coco_url'.")

        seen_image_ids.add(image_id)
        images.append(
            CompetitionImage(
                id=image_id,
                file_name=file_name,
                coco_url=coco_url,
                width=width,
                height=height,
            )
        )

    images.sort(key=lambda image: image.id)
    return CompetitionDataset(json_path=path, images=tuple(images), categories=categories)


def build_image_manifest(
    dataset: CompetitionDataset,
    images_dir: str | Path,
) -> list[dict[str, Any]]:
    image_root = Path(images_dir).resolve()
    manifest: list[dict[str, Any]] = []
    for image in dataset.images:
        image_path = image_root / image.file_name
        manifest.append(
            {
                "image_id": image.id,
                "file_name": image.file_name,
                "image_path": str(image_path),
                "exists": image_path.exists(),
                "width": image.width,
                "height": image.height,
                "coco_url": image.coco_url,
            }
        )
    return manifest


def write_image_manifest(
    dataset: CompetitionDataset,
    images_dir: str | Path,
    output_path: str | Path,
) -> Path:
    output = Path(output_path).resolve()
    output.parent.mkdir(parents=True, exist_ok=True)

    payload = {
        "competition": "fathomnet_2026_kaggle",
        "dataset_json": str(dataset.json_path),
        "images_dir": str(Path(images_dir).resolve()),
        "image_count": len(dataset.images),
        "category_count": len(dataset.categories),
        "categories": [
            {"id": category.id, "name": category.name} for category in dataset.categories
        ],
        "images": build_image_manifest(dataset=dataset, images_dir=images_dir),
    }
    with output.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)
    return output
