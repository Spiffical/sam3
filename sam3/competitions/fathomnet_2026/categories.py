from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable, Mapping


EXPECTED_CATEGORY_NAMES = (
    "amphipod",
    "anemone",
    "barnacle",
    "benthic worm",
    "bivalve",
    "black coral",
    "bony fish",
    "brittle star",
    "calycophoran siphonophore",
    "chiton",
    "crab",
    "feather star",
    "hydroid",
    "isopod",
    "jelly",
    "larvacean",
    "octopus",
    "physonect siphonophore",
    "pyrosome",
    "sea cucumber",
    "sea fan",
    "sea pen",
    "sea slug",
    "sea snail",
    "sea squirt",
    "sea star",
    "shrimp",
    "soft coral",
    "sponge",
    "squat lobster",
    "stony coral",
    "urchin",
)


@dataclass(frozen=True)
class CompetitionCategory:
    id: int
    name: str


def normalize_category_name(name: str) -> str:
    if not isinstance(name, str):
        raise TypeError(f"Category names must be strings, got {type(name)!r}")
    return " ".join(name.strip().lower().split())


def _expected_category_name_set() -> set[str]:
    return {normalize_category_name(name) for name in EXPECTED_CATEGORY_NAMES}


def load_categories_from_coco_payload(
    payload: Mapping[str, Any],
) -> list[CompetitionCategory]:
    raw_categories = payload.get("categories")
    if not isinstance(raw_categories, list) or not raw_categories:
        raise ValueError("COCO payload must include a non-empty 'categories' list.")

    categories: list[CompetitionCategory] = []
    seen_ids: set[int] = set()
    seen_names: set[str] = set()

    for index, raw_category in enumerate(raw_categories):
        if not isinstance(raw_category, Mapping):
            raise ValueError(f"Category at index {index} is not a mapping.")

        category_id = raw_category.get("id")
        category_name = raw_category.get("name")
        if not isinstance(category_id, int):
            raise ValueError(f"Category at index {index} is missing an integer 'id'.")
        if not isinstance(category_name, str) or not category_name.strip():
            raise ValueError(f"Category at index {index} is missing a non-empty 'name'.")

        normalized_name = normalize_category_name(category_name)
        if category_id in seen_ids:
            raise ValueError(f"Duplicate category id found: {category_id}")
        if normalized_name in seen_names:
            raise ValueError(f"Duplicate category name found: {normalized_name!r}")

        seen_ids.add(category_id)
        seen_names.add(normalized_name)
        categories.append(CompetitionCategory(id=category_id, name=normalized_name))

    expected_names = _expected_category_name_set()
    actual_names = {category.name for category in categories}
    if actual_names != expected_names:
        missing = sorted(expected_names - actual_names)
        extra = sorted(actual_names - expected_names)
        parts = ["Competition category names did not match the FathomNet 2026 set."]
        if missing:
            parts.append(f"Missing: {missing}")
        if extra:
            parts.append(f"Unexpected: {extra}")
        raise ValueError(" ".join(parts))

    return sorted(categories, key=lambda category: category.id)


def build_category_name_to_id(
    categories: Iterable[CompetitionCategory],
) -> dict[str, int]:
    return {category.name: category.id for category in categories}


def build_category_id_to_name(
    categories: Iterable[CompetitionCategory],
) -> dict[int, str]:
    return {category.id: category.name for category in categories}
