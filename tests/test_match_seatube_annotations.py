from scripts.match_seatube_annotations import (
    build_consensus,
    build_prompt,
    compress_id_ranges,
    effective_max_tokens,
    extract_answer_json,
    normalize_response,
)


ANNOTATIONS = [
    {"annotation_id": 10, "taxon_display_text": "Brachyura (crabs) | ID: 106673"},
    {"annotation_id": 20, "taxon_display_text": "Asteroidea | ID: 123080"},
]


def test_sonnet5_gets_minimum_reasoning_budget() -> None:
    assert effective_max_tokens("claude-sonnet-5", 2500) == 8192
    assert effective_max_tokens("claude-sonnet-5", 9000) == 9000
    assert effective_max_tokens("claude-sonnet-4-6", 2500) == 2500


def test_match_prompt_does_not_treat_annotation_count_as_visual_evidence() -> None:
    prompt = build_prompt("dense", ANNOTATIONS, 7)
    normalized_prompt = " ".join(prompt.split())
    assert "count is metadata, not visual evidence and NOT an assignment cap" in normalized_prompt
    assert "EACH numbered mask independently" in normalized_prompt
    assert "Prefer an explicit unmatched object" in normalized_prompt
    assert "three obvious sea stars should all receive" in normalized_prompt
    assert "without color fill" in normalized_prompt
    assert "palette colors and outlines" in normalized_prompt
    assert "Depth, haze, illumination, focus" in normalized_prompt
    assert "Separate colonies at different depths may share" in normalized_prompt


def test_matcher_source_renders_unfilled_object_closeups() -> None:
    source = open("scripts/match_seatube_annotations.py", encoding="utf-8").read()
    assert "def render_object_identity_sheet" in source
    assert "object_identity_sheet.png" in source
    assert "matching_prompt.txt" in source
    assert "render_object_identity_sheet(frame, masks" in source


def test_compress_id_ranges_for_dense_presentation_legend() -> None:
    assert compress_id_ranges([1, 2, 3, 5, 8, 9, 10]) == "1-3, 5, 8-10"
    assert compress_id_ranges([]) == "none"


def test_extract_and_normalize_constrained_matches() -> None:
    parsed = extract_answer_json(
        'reason <answer>{"matches":[{"object_ids":[1,99],'
        '"annotation_ids":[10,999],"confidence":1.4}],'
        '"unmatched_object_ids":[2],"unmatched_annotation_ids":[20]}</answer>'
    )
    normalized = normalize_response(
        parsed,
        object_count=2,
        annotations=ANNOTATIONS,
    )
    assert normalized is not None
    assert normalized["matches"] == [
        {
            "object_ids": [1],
            "annotation_ids": [10],
            "taxa": ["Brachyura (crabs)"],
            "confidence": 1.0,
            "reason": "",
        }
    ]
    assert normalized["unmatched_object_ids"] == [2]
    assert normalized["unmatched_annotation_ids"] == [20]


def test_consensus_requires_two_of_three_repeats() -> None:
    crab = {
        "matches": [
            {
                "object_ids": [1],
                "annotation_ids": [10],
                "taxa": ["Brachyura (crabs)"],
                "confidence": 0.9,
                "reason": "visible crab",
            }
        ]
    }
    star = {
        "matches": [
            {
                "object_ids": [1],
                "annotation_ids": [20],
                "taxa": ["Asteroidea"],
                "confidence": 0.7,
                "reason": "minority vote",
            }
        ]
    }
    consensus = build_consensus(
        [crab, crab, star],
        object_count=2,
        annotations=ANNOTATIONS,
        configured_repeats=3,
    )
    assert consensus["consensus_threshold"] == 2
    assert consensus["min_consensus_confidence"] == 0.55
    assert consensus["assignments"][0]["object_id"] == 1
    assert consensus["assignments"][0]["label"] == "Brachyura (crabs)"
    assert consensus["assignments"][0]["support"] == 2
    assert consensus["unmatched_object_ids"] == [2]
    assert consensus["unmatched_annotation_ids"] == [20]


def test_consensus_rejects_supported_but_low_confidence_guess() -> None:
    weak = {
        "matches": [
            {
                "object_ids": [1],
                "annotation_ids": [10],
                "taxa": ["Brachyura (crabs)"],
                "confidence": 0.45,
                "reason": "possible crab-like shape",
            }
        ]
    }
    consensus = build_consensus(
        [weak, weak, weak],
        object_count=1,
        annotations=ANNOTATIONS,
        configured_repeats=3,
    )
    assert consensus["assignments"] == []
    assert consensus["unmatched_object_ids"] == [1]
    assert consensus["unmatched_annotation_ids"] == [10, 20]
