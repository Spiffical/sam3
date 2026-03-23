from __future__ import annotations

import csv

from sam3.competitions.fathomnet_2026.submission import (
    DetectionPrediction,
    load_predictions_file,
    write_submission_csv,
)


def test_write_submission_csv_writes_expected_columns(tmp_path) -> None:
    output_csv = tmp_path / "submission.csv"
    predictions = [
        DetectionPrediction(
            image_id=1,
            category_id=28,
            bbox_xywh=(476.0, 790.0, 77.0, 149.0),
            score=0.3,
            annotation_id="det-1",
        )
    ]

    write_submission_csv(predictions, output_csv)

    with output_csv.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))

    assert rows == [
        {
            "annotation_id": "det-1",
            "image_id": "1",
            "category_id": "28",
            "bbox_x": "476.0",
            "bbox_y": "790.0",
            "bbox_width": "77.0",
            "bbox_height": "149.0",
            "score": "0.3",
        }
    ]


def test_load_predictions_file_supports_jsonl(tmp_path) -> None:
    predictions_path = tmp_path / "predictions.jsonl"
    predictions_path.write_text(
        "\n".join(
            [
                '{"image_id": 1, "category_id": 2, "bbox_xywh": [1, 2, 3, 4], "score": 0.5}',
                '{"image_id": 3, "category_id": 4, "bbox": [5, 6, 7, 8], "score": 0.9}',
            ]
        ),
        encoding="utf-8",
    )

    predictions = load_predictions_file(predictions_path)

    assert len(predictions) == 2
    assert predictions[0].bbox_xywh == (1.0, 2.0, 3.0, 4.0)
    assert predictions[1].bbox_xywh == (5.0, 6.0, 7.0, 8.0)
