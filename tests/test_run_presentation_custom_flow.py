import itertools
import json

import cv2
import numpy as np

from scripts.click_engine_probe import (
    MASK_VERIFY_FMT,
    _accept_mask_verdict,
    _click_budget_reached,
    _bounded_corrected_positive_click,
    _duplicate_click,
    _iteration_indices,
    _keep_positive_seed_components,
    _merge_corrected_positive_click,
    _mask_quality_repair_click,
    _render_candidates,
)
from scripts.run_presentation_custom_flow import (
    _adaptive_prompt_planner_prompt,
    _ambiguous_overlap_components,
    _click_counts,
    _consolidate_ambiguous_overlaps,
    _drop_conflicting_negative_clicks,
    _discovery_focus_region,
    _filter_prior_attempt_groups,
    _first_positive_click,
    _load_initial_masks,
    _mask_level_nms,
    _mask_guided_discovery_prompt,
    _parse_same_identity_groups,
    _replacement_for_contained_mask,
    _select_adaptive_prompt_specs,
    _should_run_border_scan,
    _write_json,
    _encode_mask,
    choose_mask_guided_pass_count,
)
from scripts.click_engine_probe import response_token_budget


def test_sparse_frame_gets_one_extra_mask_guided_pass() -> None:
    assert choose_mask_guided_pass_count(
        2,
        initial_known_count=0,
        sparse_extra_pass=True,
        finder_mode="mask-guided",
    ) == 3


def test_autonomous_mask_guided_passes_stay_autonomous() -> None:
    assert choose_mask_guided_pass_count(
        0,
        initial_known_count=0,
        sparse_extra_pass=True,
        finder_mode="mask-guided",
    ) == 0


def test_convergence_stops_on_agent_empty_not_failed_mask_generation() -> None:
    source = open("scripts/run_presentation_custom_flow.py", encoding="utf-8").read()
    assert '"agent_reported_no_missed_life"' in source
    assert '"agent_reported_only_previously_evaluated_life"' in source
    assert "and proposed_before_limit == 0" in source
    assert "and pass_index > 4" in source
    assert 'and not batch["recovered"]' not in source


def test_postverify_repairs_expose_history_and_stop_spatially_redundant_clicks() -> None:
    source = open("scripts/run_presentation_custom_flow.py", encoding="utf-8").read()
    verifier_source = open("scripts/click_engine_probe.py", encoding="utf-8").read()
    assert 'tolerance=0.025' in source
    assert '"postverify_repair_history"' in source
    assert "REPAIR HISTORY FOR THIS SAME PROPOSAL" in verifier_source
    assert "WITHOUT repair_click" in verifier_source


def test_border_scan_runs_once_during_unlimited_convergence() -> None:
    assert _should_run_border_scan(
        "every", convergence_mode=True, pass_index=1, requested_passes=0
    )
    assert not _should_run_border_scan(
        "every", convergence_mode=True, pass_index=2, requested_passes=0
    )
    assert _should_run_border_scan(
        "last", convergence_mode=True, pass_index=1, requested_passes=0
    )


def test_fixed_border_scan_modes_keep_literal_schedule() -> None:
    assert _should_run_border_scan(
        "every", convergence_mode=False, pass_index=2, requested_passes=4
    )
    assert not _should_run_border_scan(
        "last", convergence_mode=False, pass_index=3, requested_passes=4
    )
    assert _should_run_border_scan(
        "last", convergence_mode=False, pass_index=4, requested_passes=4
    )


def test_four_pass_discovery_is_a_true_two_by_two_sweep() -> None:
    assert _discovery_focus_region(1, 4)[:4] == (0.0, 0.5, 0.0, 0.5)
    assert _discovery_focus_region(2, 4)[:4] == (0.5, 1.0, 0.0, 0.5)
    assert _discovery_focus_region(3, 4)[:4] == (0.0, 0.5, 0.5, 1.0)
    assert _discovery_focus_region(4, 4)[:4] == (0.5, 1.0, 0.5, 1.0)


def test_unlimited_convergence_sweeps_quadrants_before_full_frame_stop() -> None:
    assert _discovery_focus_region(1, 0)[:4] == (0.0, 0.5, 0.0, 0.5)
    assert _discovery_focus_region(2, 0)[:4] == (0.5, 1.0, 0.0, 0.5)
    assert _discovery_focus_region(3, 0)[:4] == (0.0, 0.5, 0.5, 1.0)
    assert _discovery_focus_region(4, 0)[:4] == (0.5, 1.0, 0.5, 1.0)
    assert _discovery_focus_region(5, 0) == (
        0.0,
        1.0,
        0.0,
        1.0,
        "the entire residual frame",
    )


def test_unlimited_convergence_prompt_uses_required_sweep_before_stop() -> None:
    source = open("scripts/run_presentation_custom_flow.py", encoding="utf-8").read()
    assert "This is required convergence sweep pass" in source
    assert "An empty tile does not end convergence" in source
    assert "This is full-frame convergence discovery pass" in source


def test_discovery_shows_light_and_strong_coverage_plus_focus_crops() -> None:
    prompt = _mask_guided_discovery_prompt(
        pass_instruction="scan",
        has_focus_crops=True,
    )
    assert "THIRD image is a strong green coverage map" in prompt
    assert "FOURTH image is an outline-only map" in prompt
    assert "FIFTH image is an enlarged untouched crop" in prompt
    assert "normalized to the FULL FRAME, not the crop" in prompt
    source = open("scripts/run_presentation_custom_flow.py", encoding="utf-8").read()
    assert 'alpha=0.48' in source
    assert 'existing_masks_outline.png' in source
    assert 'guided_dir / "focus_raw.png"' in source
    assert 'guided_dir / "focus_strong_grid.png"' in source


def test_convergence_prompt_tracks_prior_attempts_without_forbidding_repair() -> None:
    source = open("scripts/run_presentation_custom_flow.py", encoding="utf-8").read()
    assert "These candidates were already evaluated" in source
    assert "same identity at the same location" in source
    assert "use a visibly different" in source
    assert "prior_mask_guided_attempts" in source
    assert '"n_prior_attempts_in_prompt"' in source


def test_cross_pass_repeat_filter_drops_same_identity_at_same_location() -> None:
    groups = [{
        "description": "small tan feathery coral bush on rocky slope",
        "clicks": [{"x": 0.298, "y": 0.500, "label": 1}],
    }]
    prior = [{
        "description": "small tan feathery coral bush on rocky slope, left-center",
        "click": {"x": 0.298, "y": 0.500},
    }]
    kept, repeated = _filter_prior_attempt_groups(groups, prior)
    assert kept == []
    assert repeated == groups


def test_cross_pass_repeat_filter_allows_new_identity_or_corrected_click() -> None:
    prior = [{
        "description": "white anemone at base of coral",
        "click": {"x": 0.50, "y": 0.50},
    }]
    groups = [
        {
            "description": "brown branching coral behind the anemone",
            "clicks": [{"x": 0.50, "y": 0.50, "label": 1}],
        },
        {
            "description": "white anemone at base of coral",
            "clicks": [{"x": 0.56, "y": 0.50, "label": 1}],
        },
    ]
    kept, repeated = _filter_prior_attempt_groups(groups, prior)
    assert kept == groups
    assert repeated == []


def test_zero_click_budget_is_unlimited() -> None:
    clicks = [{"x": 0.1, "y": 0.1, "label": 1}] * 100
    assert not _click_budget_reached(clicks, 0)
    assert list(itertools.islice(_iteration_indices(0), 7)) == list(range(7))


def test_positive_click_budget_remains_available() -> None:
    clicks = [{"x": 0.1, "y": 0.1, "label": 1}] * 3
    assert _click_budget_reached(clicks, 3)
    assert list(_iteration_indices(3)) == [0, 1, 2]


def test_duplicate_click_only_matches_same_label_and_location() -> None:
    clicks = [{"x": 0.25, "y": 0.5, "label": 1}]
    assert _duplicate_click(clicks, {"x": 0.2505, "y": 0.5, "label": 1})
    assert not _duplicate_click(clicks, {"x": 0.2505, "y": 0.5, "label": 0})


def test_click_localization_preserves_other_agent_selected_clicks() -> None:
    clicks = [
        {"x": 0.15, "y": 0.35, "label": 1},
        {"x": 0.03, "y": 0.55, "label": 1},
        {"x": 0.30, "y": 0.30, "label": 0},
    ]
    corrected = {"x": 0.11, "y": 0.19, "label": 1}
    assert _merge_corrected_positive_click(clicks, corrected) == [
        corrected,
        {"x": 0.03, "y": 0.55, "label": 1},
        {"x": 0.30, "y": 0.30, "label": 0},
    ]


def test_click_localization_applies_nearby_full_frame_correction() -> None:
    clicks = [
        {"x": 0.40, "y": 0.40, "label": 1},
        {"x": 0.50, "y": 0.50, "label": 0},
    ]
    corrected = {"x": 0.44, "y": 0.43, "label": 1}
    localized, displacement, applied = _bounded_corrected_positive_click(
        clicks, corrected, max_displacement=0.10
    )
    assert applied
    assert abs(displacement - 0.05) < 1e-12
    assert localized == [corrected, clicks[1]]


def test_click_localization_preserves_seed_instead_of_switching_target() -> None:
    clicks = [
        {"x": 0.71, "y": 0.34, "label": 1},
        {"x": 0.80, "y": 0.40, "label": 0},
    ]
    switched_target = {"x": 0.72, "y": 0.58, "label": 1}
    localized, displacement, applied = _bounded_corrected_positive_click(
        clicks, switched_target, max_displacement=0.10
    )
    assert not applied
    assert displacement > 0.10
    assert localized == clicks


def test_click_localizer_uses_full_frame_coordinates_and_context() -> None:
    source = open("scripts/click_engine_probe.py", encoding="utf-8").read()
    assert "IMAGE 1 is the full seafloor frame" in source
    assert "IMAGE 2 is a zoomed crop" in source
    assert "FULL FRAME (IMAGE 1), not the crop" in source
    assert 'f"{tag}_full.png"' in source


def test_mask_verifier_reports_identity_failure_class() -> None:
    assert "fragment|merge|wrong|background" in MASK_VERIFY_FMT
    assert "piece of a larger identity" in MASK_VERIFY_FMT
    assert "merges identities/depth layers" in MASK_VERIFY_FMT
    assert '"complete_identity": true' in MASK_VERIFY_FMT
    assert '"single_identity": true' in MASK_VERIFY_FMT
    assert '"repair_click"' in MASK_VERIFY_FMT
    assert "NORMALIZED [0,1] in the FULL FRAME" in MASK_VERIFY_FMT


def test_mask_verifier_separates_raw_full_frame_from_overlay() -> None:
    source = open("scripts/click_engine_probe.py", encoding="utf-8").read()
    assert "The FIRST image is the raw full frame" in source
    assert "The SECOND image is the same full frame" in source
    assert 'f"{tag_prefix}_id{r.get(\'creature_id\', 0)}_raw.png"' in source


def test_mask_verifier_reports_deterministic_bbox_and_frame_contact() -> None:
    source = open("scripts/click_engine_probe.py", encoding="utf-8").read()
    assert "DETERMINISTIC BINARY-MASK GEOMETRY" in source
    assert "touches frame edge(s)=" in source
    assert "override any visual guess about location or frame" in source
    assert "smooth, coherent silhouette is NOT evidence" in source
    assert "target is frame-clipped but touches frame edge(s)=none" in source


def test_mask_repair_click_enforces_failure_specific_label() -> None:
    assert _mask_quality_repair_click({
        "failure": "fragment",
        "repair_click": {"x": 0.6, "y": 0.7, "label": 1},
    }) == {"x": 0.6, "y": 0.7, "label": 1}
    assert _mask_quality_repair_click({
        "failure": "fragment",
        "repair_click": {"x": 0.6, "y": 0.7, "label": 0},
    }) is None


def test_ambiguous_overlap_components_join_transitive_candidates() -> None:
    first = np.zeros((10, 20), dtype=bool)
    second = np.zeros_like(first)
    third = np.zeros_like(first)
    isolated = np.zeros_like(first)
    first[2:6, 2:8] = True
    second[2:6, 5:11] = True
    third[2:6, 8:14] = True
    isolated[7:9, 16:19] = True
    components, pairs = _ambiguous_overlap_components(
        [first, second, third, isolated], containment_threshold=0.40
    )
    assert components == [[0, 1, 2]]
    assert [(p["left_index"], p["right_index"]) for p in pairs] == [
        (0, 1),
        (1, 2),
    ]


def test_ambiguous_overlap_components_include_small_near_contact_gap() -> None:
    first = np.zeros((100, 100), dtype=bool)
    second = np.zeros_like(first)
    first[20:40, 20:40] = True
    second[20:40, 41:61] = True
    components, pairs = _ambiguous_overlap_components(
        [first, second], containment_threshold=0.40, max_gap_fraction=0.02
    )
    assert components == [[0, 1]]
    assert pairs[0]["trigger"] == "near_contact"
    assert pairs[0]["smaller_mask_overlap"] == 0.0


def test_identity_relation_parser_allows_subgroup_of_larger_component() -> None:
    answer = {"same_identity_groups": [[2, 3], [5, 6]]}
    assert _parse_same_identity_groups(answer, [[1, 2, 3, 4], [5, 6]]) == [
        [2, 3], [5, 6]
    ]


def test_identity_relation_parser_rejects_cross_component_and_reused_ids() -> None:
    answer = {"same_identity_groups": [[1, 3], [1, 2], [3, 4], [4]]}
    assert _parse_same_identity_groups(answer, [[1, 2], [3, 4]]) == [
        [1, 2], [3, 4]
    ]


def test_overlap_audit_prompt_requests_relations_not_independent_keep_drop() -> None:
    source = open("scripts/run_presentation_custom_flow.py", encoding="utf-8").read()
    assert "smallest same-identity SUBGROUPS" in source
    assert '"same_identity_groups"' in source
    assert "A component can contain several unrelated organisms" in source
    assert "identity_component_temporal.png" in source
    assert "Co-motion alone never proves one identity" in source
    assert "Relationship mode therefore uses the raw" in source
    assert "identity_group_{confirmation_index}_confirm.png" in source
    assert "focused_relation_confirmed" in source
    assert "same_branching_system" in source
    assert "small or zero pixel overlap is NOT evidence against" in source
    assert "centerline and tangent through the boundary" in source


def test_overlap_identity_audit_is_noop_without_ambiguous_geometry(tmp_path) -> None:
    first = np.zeros((20, 20), dtype=bool)
    second = np.zeros_like(first)
    first[2:6, 2:6] = True
    second[12:16, 12:16] = True
    known = [{"mask": first}, {"mask": second}]
    frame_dir = tmp_path / "frame"
    frame_dir.mkdir()
    kept, metadata = _consolidate_ambiguous_overlaps(
        known,
        frame=np.zeros((20, 20, 3), dtype=np.uint8),
        width=20,
        height=20,
        frame_dir=frame_dir,
        model="unused",
        raw_index=0,
        temporal_offsets=[1],
        service=None,
        max_clicks=0,
        max_attempts=1,
        zoom_crop_frac=0.5,
    )
    assert kept == known
    assert metadata["components"] == []
    assert metadata["n_masks_removed"] == 0
    assert (frame_dir / "overlap_identity_audit" / "audit_summary.json").exists()
    assert _mask_quality_repair_click({
        "failure": "merge",
        "repair_click": {"x": 0.3, "y": 0.4, "label": 0},
    }) == {"x": 0.3, "y": 0.4, "label": 0}
    assert _mask_quality_repair_click({
        "failure": "wrong",
        "repair_click": {"x": 0.3, "y": 0.4, "label": 1},
    }) is None


def test_strict_mask_verdict_requires_explicit_complete_single_identity() -> None:
    assert _accept_mask_verdict({"keep": True})
    assert not _accept_mask_verdict({"keep": True}, strict_identity=True)
    assert not _accept_mask_verdict(
        {
            "keep": True,
            "complete_identity": False,
            "single_identity": True,
        },
        strict_identity=True,
    )
    assert _accept_mask_verdict(
        {
            "keep": True,
            "complete_identity": True,
            "single_identity": True,
        },
        strict_identity=True,
    )


def test_text_verifier_keeps_adjacent_same_taxon_instances_distinct() -> None:
    source = open("scripts/run_presentation_custom_flow.py", encoding="utf-8").read()
    assert "text_proposal_full_context.png" in source
    assert "call two masks duplicates" in source
    assert "share a taxon" in source
    assert "Candidates at different depth layers remain separate" in source
    assert "temporal parallax" in source
    assert "occlusion ordering" in source
    assert "Deterministic candidate-mask geometry" in source
    assert "Near-disjoint masks may touch" in source
    assert "VISIBLE extent" in source
    assert "text_proposal_temporal_identity.png" in source
    assert "trace an actual branch/body connection" in source
    assert "color-filled" in source
    assert "accepted masks are deliberately NOT drawn" in source
    assert "vs any prior accepted mask" in source
    assert "median optical flow" in source
    assert "separation/MAD" in source
    assert "text_proposal_batch_verify_prompt.txt" in source


def test_seastar_frame_has_no_frame_specific_fragment_merge_rule() -> None:
    manifest = json.load(
        open("configs/seatube_meagan_five_frames_v1.json", encoding="utf-8")
    )
    frame = next(
        row for row in manifest["frames"] if row["id"] == "02_seastars_on_coral"
    )
    assert "sam3_fragment_merge_prompts" not in frame
    assert "sam3_fragment_bbox_iom_threshold" not in frame


def test_benchmark_manifest_has_no_prompt_specific_fragment_merge_rules() -> None:
    manifest = json.load(
        open("configs/seatube_meagan_five_frames_v1.json", encoding="utf-8")
    )
    assert all(
        "sam3_fragment_merge_prompts" not in frame
        for frame in manifest["frames"]
    )


def test_non_sparse_frame_keeps_requested_pass_count() -> None:
    assert choose_mask_guided_pass_count(
        2,
        initial_known_count=1,
        sparse_extra_pass=True,
        finder_mode="mask-guided",
    ) == 2


def test_sparse_extra_pass_can_be_disabled() -> None:
    assert choose_mask_guided_pass_count(
        2,
        initial_known_count=0,
        sparse_extra_pass=False,
        finder_mode="mask-guided",
    ) == 2


def test_s3_mode_does_not_schedule_mask_guided_work() -> None:
    assert choose_mask_guided_pass_count(
        2,
        initial_known_count=0,
        sparse_extra_pass=True,
        finder_mode="s3",
    ) == 2


def test_negative_first_group_still_uses_foreground_seed() -> None:
    group = {
        "clicks": [
            {"x": 0.1, "y": 0.2, "label": 0},
            {"x": 0.3, "y": 0.4, "label": 1},
        ]
    }
    assert _first_positive_click(group) == {"x": 0.3, "y": 0.4, "label": 1}


def test_click_counts_separate_positive_and_negative() -> None:
    assert _click_counts([
        {"label": 1}, {"label": 0}, {"label": 0}, {"label": 2}
    ]) == {"positive": 1, "negative": 2}


def test_adaptive_prompt_selection_keeps_only_exact_allowed_phrases() -> None:
    selected = _select_adaptive_prompt_specs(
        ["small creatures", "branching coral", "Walteria"],
        {
            "phrases": [
                "branching coral",
                "invented sea plant",
                " Walteria ",
                "BRANCHING CORAL",
            ]
        },
    )
    assert [spec.text for spec in selected] == ["branching coral", "Walteria"]


def test_runner_separates_phrase_model_from_click_model() -> None:
    source = open(
        "scripts/run_presentation_custom_flow.py", encoding="utf-8"
    ).read()
    assert 'parser.add_argument(\n        "--phrase-model"' in source
    assert "model=phrase_model" in source
    assert '"phrase_model": phrase_model' in source


def test_adaptive_prompt_planner_warns_against_dense_scene_small_creatures() -> None:
    prompt = _adaptive_prompt_planner_prompt(
        "dense", "Dense coral thicket", ["small creatures", "coral"]
    )
    assert "use 'small creatures' only" in prompt
    assert "fine coral branches" in prompt
    assert "retrieval handles only" in prompt


def test_localization_drops_only_negatives_that_contradict_positive() -> None:
    groups, removed = _drop_conflicting_negative_clicks(
        [{"id": 1, "clicks": [
            {"x": 0.62, "y": 0.02, "label": 1},
            {"x": 0.6205, "y": 0.021, "label": 0},
            {"x": 0.70, "y": 0.10, "label": 0},
        ]}],
        1280,
        720,
    )
    assert removed == 1
    assert groups[0]["clicks"] == [
        {"x": 0.62, "y": 0.02, "label": 1},
        {"x": 0.70, "y": 0.10, "label": 0},
    ]


def test_discovery_prompt_explains_target_specific_negative_clicks() -> None:
    prompt = _mask_guided_discovery_prompt(pass_instruction="Scan left. ")
    assert "EXCLUDE THIS LOCATION FROM THIS TARGET'S MASK" in prompt
    assert "does not mean that the location contains no life" in prompt
    assert "never on another desired part" in prompt
    assert "Do not use a negative click solely" in prompt
    assert '"label":0' in prompt


def test_discovery_prompt_requires_positive_click_on_target() -> None:
    prompt = _mask_guided_discovery_prompt(pass_instruction="Scan left. ")
    assert "positive coordinate visibly lands ON" in prompt
    assert "not in nearby water or between branches" in prompt


def test_discovery_prompt_compares_raw_target_with_light_mask_overlay() -> None:
    prompt = _mask_guided_discovery_prompt(pass_instruction="Scan lower center. ")
    source = open("scripts/run_presentation_custom_flow.py", encoding="utf-8").read()
    assert "untouched raw TARGET FRAME" in prompt
    assert "Compare the raw and masked target pixel-for-pixel" in prompt
    assert "Do not assume a cluster is fully covered" in prompt
    assert "target_raw.png" in source
    assert "alpha=0.18" in source


def test_discovery_prompt_requires_one_complete_identity_per_group() -> None:
    prompt = _mask_guided_discovery_prompt(pass_instruction="Scan left. ")
    assert "EXACTLY ONE COMPLETE BIOLOGICAL IDENTITY" in prompt
    assert "subregion of a larger continuous organism" in prompt
    assert "foreground and background" in prompt
    assert "separate branching system requires separate groups" in prompt


def test_prior_final_masks_can_seed_residual_discovery(tmp_path) -> None:
    frame_id = "dense"
    frame_dir = tmp_path / frame_id
    frame_dir.mkdir()
    mask = np.zeros((12, 20), dtype=bool)
    mask[2:8, 4:10] = True
    _write_json(frame_dir / "final_masks_rle.json", {
        "frame_size_hw": [12, 20],
        "masks": [_encode_mask(mask)],
    })
    loaded = _load_initial_masks(tmp_path, frame_id, (12, 20))
    assert len(loaded) == 1
    assert loaded[0]["source"] == "initial"
    assert np.array_equal(loaded[0]["mask"], mask)


def test_interrupted_checkpoint_can_seed_residual_discovery(tmp_path) -> None:
    frame_id = "dense"
    frame_dir = tmp_path / frame_id
    frame_dir.mkdir()
    mask = np.zeros((12, 20), dtype=bool)
    mask[3:9, 5:11] = True
    _write_json(frame_dir / "checkpoint_masks_rle.json", {
        "frame_size_hw": [12, 20],
        "completed_mask_guided_passes": 7,
        "masks": [_encode_mask(mask)],
    })
    loaded = _load_initial_masks(tmp_path, frame_id, (12, 20))
    assert len(loaded) == 1
    assert loaded[0]["source"] == "initial"
    assert np.array_equal(loaded[0]["mask"], mask)


def test_verified_containment_expansion_can_replace_incomplete_mask() -> None:
    old = np.zeros((20, 20), dtype=bool)
    old[8:12, 8:12] = True
    new = np.zeros((20, 20), dtype=bool)
    new[5:15, 5:15] = True
    replacement = _replacement_for_contained_mask(
        {"mask": new, "creature_confidence": 0.85, "source": "M"},
        {
            "method": "mask_containment",
            "firstpass_index": 0,
            "overlap_coefficient": 1.0,
        },
        [{"mask": old, "source": "initial"}],
    )
    assert replacement is not None
    assert replacement["old_area_px"] == 16
    assert replacement["new_area_px"] == 100
    assert replacement["result"]["source"] == "initial"


def test_replacement_rejects_low_confidence_or_noncontainment_match() -> None:
    old = np.zeros((20, 20), dtype=bool)
    old[8:12, 8:12] = True
    new = np.zeros((20, 20), dtype=bool)
    new[5:15, 5:15] = True
    known = [{"mask": old, "source": "initial"}]
    base_match = {
        "method": "mask_containment",
        "firstpass_index": 0,
        "overlap_coefficient": 1.0,
    }
    assert _replacement_for_contained_mask(
        {"mask": new, "creature_confidence": 0.7}, base_match, known
    ) is None
    assert _replacement_for_contained_mask(
        {"mask": new, "creature_confidence": 0.9},
        {**base_match, "method": "seed_inside_firstpass"},
        known,
    ) is None


def test_reasoning_models_get_larger_structured_response_budget() -> None:
    assert response_token_budget("claude-sonnet-5", 800) == 4096
    assert response_token_budget(
        "claude-sonnet-5", 1600, sonnet5_minimum=8192
    ) == 8192
    assert response_token_budget("claude-opus-5", 600) == 4096
    assert response_token_budget(
        "claude-opus-5", 1600, opus5_minimum=8192
    ) == 8192
    assert response_token_budget("claude-sonnet-4-6", 800) == 800


def test_candidate_review_writes_binary_truth_companion(tmp_path) -> None:
    frame = np.full((20, 20, 3), 40, dtype=np.uint8)
    masks = np.zeros((3, 20, 20), dtype=bool)
    masks[0, 5:10, 6:12] = True
    masks[1, 2:18, 2:18] = True
    masks[2, 10:15, 10:15] = True
    overlay_path = tmp_path / "candidates.png"
    binary_path = tmp_path / "candidates_binary.png"
    review_path = tmp_path / "candidates_review.png"
    _render_candidates(
        frame,
        masks,
        np.array([0.9, 0.5, 0.4]),
        [{"x": 0.4, "y": 0.4, "label": 1}],
        (0, 0, 20, 20),
        str(overlay_path),
        up=2,
        binary_path=str(binary_path),
        review_path=str(review_path),
    )
    binary = cv2.imread(str(binary_path))
    assert overlay_path.exists()
    assert review_path.exists()
    assert binary is not None
    assert np.any(np.all(binary == 255, axis=2))
    assert np.any(np.all(binary == 0, axis=2))
    review = cv2.imread(str(review_path))
    overlay = cv2.imread(str(overlay_path))
    assert review.shape[0] == 2 * overlay.shape[0]


def test_mask_cleanup_keeps_seeded_component_and_filled_branch_gaps() -> None:
    mask = np.zeros((20, 20), dtype=bool)
    mask[2:12, 2:10] = True
    mask[5:8, 5:7] = False  # enclosed fine-topology gap stays unchanged
    mask[15:18, 15:18] = True  # disconnected neighbouring-organism spill
    cleaned = _keep_positive_seed_components(
        mask, [{"x": 4 / 19, "y": 4 / 19, "label": 1}]
    )
    assert cleaned[4, 4]
    assert not cleaned[6, 6]
    assert not cleaned[16, 16]


def test_mask_cleanup_can_keep_multiple_deliberately_seeded_pieces() -> None:
    mask = np.zeros((20, 20), dtype=bool)
    mask[2:6, 2:6] = True
    mask[12:17, 12:17] = True
    cleaned = _keep_positive_seed_components(mask, [
        {"x": 3 / 19, "y": 3 / 19, "label": 1},
        {"x": 14 / 19, "y": 14 / 19, "label": 1},
    ])
    assert np.array_equal(cleaned, mask)


def test_mask_level_nms_rejects_broad_container_of_clean_individual() -> None:
    clean = np.zeros((30, 30), dtype=bool)
    clean[5:15, 5:15] = True
    broad = np.zeros((30, 30), dtype=bool)
    broad[3:25, 3:25] = True
    kept, removed = _mask_level_nms([
        {
            "mask": broad,
            "name": "broad",
            "seed_click": {"x": 10 / 29, "y": 10 / 29},
        },
        {
            "mask": clean,
            "name": "clean",
            "seed_click": {"x": 9 / 29, "y": 9 / 29},
        },
    ])
    assert removed == 1
    assert [item["name"] for item in kept] == ["clean"]


def test_mask_level_nms_preserves_nested_silhouettes_with_distinct_seeds() -> None:
    clean = np.zeros((30, 30), dtype=bool)
    clean[5:15, 5:15] = True
    broad = np.zeros((30, 30), dtype=bool)
    broad[3:27, 3:27] = True
    kept, removed = _mask_level_nms([
        {
            "mask": broad,
            "name": "broad",
            "seed_click": {"x": 24 / 29, "y": 24 / 29},
        },
        {
            "mask": clean,
            "name": "clean",
            "seed_click": {"x": 9 / 29, "y": 9 / 29},
        },
    ])
    assert removed == 0
    assert {item["name"] for item in kept} == {"broad", "clean"}
