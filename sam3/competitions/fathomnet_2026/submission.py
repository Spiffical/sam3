from __future__ import annotations

import csv
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Optional, Sequence


@dataclass(frozen=True)
class DetectionPrediction:
    image_id: int
    category_id: int
    bbox_xywh: tuple[float, float, float, float]
    score: float
    annotation_id: Optional[str] = None


def _coerce_bbox_xywh(raw_bbox: Any) -> tuple[float, float, float, float]:
    if not isinstance(raw_bbox, Sequence) or isinstance(raw_bbox, (str, bytes)):
        raise ValueError("Prediction bbox must be a 4-element sequence.")
    if len(raw_bbox) != 4:
        raise ValueError("Prediction bbox must contain exactly 4 values.")

    bbox = tuple(float(value) for value in raw_bbox)
    x, y, width, height = bbox
    if width <= 0 or height <= 0:
        raise ValueError(f"Prediction bbox must have positive width/height, got {bbox}")
    if x < 0 or y < 0:
        raise ValueError(f"Prediction bbox must have non-negative x/y, got {bbox}")
    return bbox


def parse_prediction_record(
    raw_prediction: Mapping[str, Any], index: int
) -> DetectionPrediction:
    image_id = raw_prediction.get("image_id")
    category_id = raw_prediction.get("category_id")
    bbox_xywh = raw_prediction.get("bbox_xywh", raw_prediction.get("bbox"))
    score = raw_prediction.get("score")
    annotation_id = raw_prediction.get("annotation_id")

    if not isinstance(image_id, int):
        raise ValueError(f"Prediction {index} is missing an integer 'image_id'.")
    if not isinstance(category_id, int):
        raise ValueError(f"Prediction {index} is missing an integer 'category_id'.")
    if score is None:
        raise ValueError(f"Prediction {index} is missing 'score'.")

    score_value = float(score)
    if score_value < 0.0 or score_value > 1.0:
        raise ValueError(
            f"Prediction {index} has score {score_value}, expected [0.0, 1.0]."
        )

    annotation_id_value: Optional[str]
    if annotation_id is None:
        annotation_id_value = None
    else:
        annotation_id_value = str(annotation_id)

    return DetectionPrediction(
        image_id=image_id,
        category_id=category_id,
        bbox_xywh=_coerce_bbox_xywh(bbox_xywh),
        score=score_value,
        annotation_id=annotation_id_value,
    )


def load_predictions_file(predictions_path: str | Path) -> list[DetectionPrediction]:
    path = Path(predictions_path)
    raw_text = path.read_text(encoding="utf-8").strip()
    if not raw_text:
        return []

    raw_predictions: list[Mapping[str, Any]] = []
    if raw_text.startswith("["):
        parsed = json.loads(raw_text)
        if not isinstance(parsed, list):
            raise ValueError(f"Expected a JSON list in {path}")
        raw_predictions = parsed
    else:
        for line_number, line in enumerate(raw_text.splitlines(), start=1):
            line = line.strip()
            if not line:
                continue
            parsed_line = json.loads(line)
            if not isinstance(parsed_line, Mapping):
                raise ValueError(
                    f"Expected a JSON object on line {line_number} of {path}"
                )
            raw_predictions.append(parsed_line)

    predictions: list[DetectionPrediction] = []
    for index, raw_prediction in enumerate(raw_predictions, start=1):
        if not isinstance(raw_prediction, Mapping):
            raise ValueError(f"Prediction {index} in {path} is not a JSON object.")
        predictions.append(parse_prediction_record(raw_prediction, index))
    return predictions


def validate_prediction_ids(
    predictions: Iterable[DetectionPrediction],
    valid_image_ids: set[int],
    valid_category_ids: set[int],
) -> None:
    for index, prediction in enumerate(predictions, start=1):
        if prediction.image_id not in valid_image_ids:
            raise ValueError(
                f"Prediction {index} used image_id={prediction.image_id}, "
                "which is not present in the dataset manifest."
            )
        if prediction.category_id not in valid_category_ids:
            raise ValueError(
                f"Prediction {index} used category_id={prediction.category_id}, "
                "which is not present in the dataset categories."
            )


def _submission_rows(
    predictions: Sequence[DetectionPrediction],
) -> list[dict[str, str | int | float]]:
    rows: list[dict[str, str | int | float]] = []
    for index, prediction in enumerate(predictions, start=1):
        x, y, width, height = prediction.bbox_xywh
        annotation_id = prediction.annotation_id or str(index)
        rows.append(
            {
                "annotation_id": annotation_id,
                "image_id": prediction.image_id,
                "category_id": prediction.category_id,
                "bbox_x": round(x, 5),
                "bbox_y": round(y, 5),
                "bbox_width": round(width, 5),
                "bbox_height": round(height, 5),
                "score": round(prediction.score, 8),
            }
        )
    return rows


def write_submission_csv(
    predictions: Sequence[DetectionPrediction],
    output_csv_path: str | Path,
) -> Path:
    output = Path(output_csv_path).resolve()
    output.parent.mkdir(parents=True, exist_ok=True)

    fieldnames = [
        "annotation_id",
        "image_id",
        "category_id",
        "bbox_x",
        "bbox_y",
        "bbox_width",
        "bbox_height",
        "score",
    ]
    with output.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(_submission_rows(predictions))
    return output
