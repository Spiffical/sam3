from __future__ import annotations

from typing import Sequence

from .categories import EXPECTED_CATEGORY_NAMES


PRIMARY_DETECTION_PROMPT = (
    "segment all visible marine organisms in this image, including fish, "
    "gelatinous organisms, crustaceans, mollusks, echinoderms, corals, "
    "sponges, hydroids, anemones, and tunicates"
)

RESCUE_DETECTION_PROMPTS = (
    "segment visible fish and octopus",
    "segment visible jellys, siphonophores, pyrosomes, and larvaceans",
    (
        "segment visible crabs, shrimp, squat lobsters, amphipods, isopods, "
        "worms, mollusks, and echinoderms"
    ),
    (
        "segment visible anemones, corals, hydroids, sponges, sea pens, "
        "sea fans, and sea squirts"
    ),
)


def build_crop_classification_prompt(category_names: Sequence[str]) -> str:
    category_list = "\n".join(f"- {name}" for name in category_names)
    return (
        "You are labeling one marine organism detection for the FathomNet 2026 "
        "Kaggle competition.\n"
        "You will see a full image for context and a cropped detection region.\n"
        "Choose exactly one category from the allowed list when the detection is a "
        "real marine organism.\n"
        "If the box is background, debris, camera artifact, or too ambiguous to "
        "submit, set drop_detection to true.\n"
        "Return strict JSON with exactly these keys: "
        "category_name, confidence, drop_detection, reason.\n"
        "confidence must be a floating point number between 0 and 1.\n"
        "When drop_detection is true, set category_name to an empty string.\n"
        "Allowed category_name values:\n"
        f"{category_list}"
    )


DEFAULT_CROP_CLASSIFICATION_PROMPT = build_crop_classification_prompt(
    EXPECTED_CATEGORY_NAMES
)
