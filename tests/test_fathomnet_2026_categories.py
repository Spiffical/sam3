from __future__ import annotations

import pytest

from sam3.competitions.fathomnet_2026.categories import (
    EXPECTED_CATEGORY_NAMES,
    build_category_name_to_id,
    load_categories_from_coco_payload,
)


def test_load_categories_from_coco_payload_validates_expected_names() -> None:
    payload = {
        "categories": [
            {"id": index + 1, "name": name}
            for index, name in enumerate(EXPECTED_CATEGORY_NAMES)
        ]
    }

    categories = load_categories_from_coco_payload(payload)

    assert len(categories) == len(EXPECTED_CATEGORY_NAMES)
    assert build_category_name_to_id(categories)["octopus"] > 0


def test_load_categories_from_coco_payload_rejects_unexpected_names() -> None:
    payload = {
        "categories": [
            {"id": index + 1, "name": name}
            for index, name in enumerate(EXPECTED_CATEGORY_NAMES[:-1])
        ]
        + [{"id": 999, "name": "kelp"}]
    }

    with pytest.raises(ValueError, match="did not match"):
        load_categories_from_coco_payload(payload)
