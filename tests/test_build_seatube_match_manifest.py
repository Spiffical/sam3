from datetime import timezone

import pytest

from scripts.build_seatube_match_manifest import (
    annotation_creator,
    creator_user_id,
    is_worms_taxon,
    parse_iso_utc,
    strict_row_relative_seconds,
    worms_taxa,
)


def test_strict_row_relative_seconds_accepts_contained_annotation() -> None:
    relative, duration = strict_row_relative_seconds(
        {
            "annotationId": 1,
            "startDate": "2026-01-01T00:01:17.000Z",
            "archiveClipStartDate": "2026-01-01T00:00:00.000Z",
            "clipDurationSeconds": 300,
        }
    )
    assert relative == 77.0
    assert duration == 300.0


@pytest.mark.parametrize(
    "timestamp",
    ["2025-12-31T23:59:59.999Z", "2026-01-01T00:05:00.000Z"],
)
def test_strict_row_relative_seconds_rejects_outside_mapping(timestamp: str) -> None:
    with pytest.raises(ValueError, match="lies outside mapped clip"):
        strict_row_relative_seconds(
            {
                "annotationId": 2,
                "startDate": timestamp,
                "archiveClipStartDate": "2026-01-01T00:00:00.000Z",
                "clipDurationSeconds": 300,
            }
        )


def test_worms_filter_uses_id_or_code() -> None:
    assert is_worms_taxon({"taxonomyId": 1})
    assert is_worms_taxon({"taxonomyCode": "WoRMS"})
    assert not is_worms_taxon({"taxonomyId": 2, "taxonomyCode": "CMECS"})
    row = {
        "taxonomy": [
            {"taxonomyId": 1, "taxonId": 10},
            {"taxonomyId": 2, "taxonId": 20},
        ]
    }
    assert worms_taxa(row) == [{"taxonomyId": 1, "taxonId": 10}]


def test_parse_iso_utc_normalizes_timezone() -> None:
    assert parse_iso_utc("2026-01-01T01:00:00+01:00").tzinfo == timezone.utc


def test_annotation_creator_omits_email_and_supports_export_shape() -> None:
    api_row = {
        "createdBy": {
            "userId": 145400,
            "firstName": "Calder",
            "lastName": "Guimond",
            "email": "private@example.test",
        }
    }
    assert annotation_creator(api_row) == {
        "user_id": 145400,
        "name": "Calder Guimond",
    }
    assert creator_user_id(api_row) == 145400

    export_row = {
        "creator_user_id": "140400",
        "creator_first_name": "Sofia",
        "creator_last_name": "Jimenez Gonzalez",
        "creator_email": "private@example.test",
    }
    assert annotation_creator(export_row) == {
        "user_id": 140400,
        "name": "Sofia Jimenez Gonzalez",
    }
