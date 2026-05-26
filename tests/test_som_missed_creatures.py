import unittest

from nibi_model_compare.som_missed_creatures import parse_som_response


class ParseSomResponseTests(unittest.TestCase):
    def test_happy_path(self):
        text = '<answer>{"accepted_marks": [1, 3, 5]}</answer>'
        self.assertEqual(parse_som_response(text), [1, 3, 5])

    def test_prose_before_tag(self):
        text = (
            "Looking at the marked image, marks 2 and 4 look like substrate.\n"
            "Marks 1 and 3 are clearly biological.\n"
            '<answer>{"accepted_marks": [1, 3]}</answer>'
        )
        self.assertEqual(parse_som_response(text), [1, 3])

    def test_multiple_answer_blocks_takes_last(self):
        text = (
            '<answer>{"accepted_marks": [1]}</answer>\n'
            "wait, on reflection...\n"
            '<answer>{"accepted_marks": [1, 2]}</answer>'
        )
        self.assertEqual(parse_som_response(text), [1, 2])

    def test_empty_list_is_valid(self):
        text = '<answer>{"accepted_marks": []}</answer>'
        self.assertEqual(parse_som_response(text), [])

    def test_missing_tag_returns_empty(self):
        self.assertEqual(parse_som_response("just text"), [])

    def test_malformed_json_returns_empty(self):
        self.assertEqual(parse_som_response("<answer>{not json}</answer>"), [])

    def test_out_of_range_ids_filtered_when_valid_ids_given(self):
        text = '<answer>{"accepted_marks": [1, 99, 3]}</answer>'
        self.assertEqual(
            parse_som_response(text, valid_ids={1, 2, 3}),
            [1, 3],
        )

    def test_non_int_ids_filtered(self):
        text = '<answer>{"accepted_marks": [1, "2", 3.5, true, 4]}</answer>'
        self.assertEqual(parse_som_response(text), [1, 4])

    def test_nested_json_in_answer_payload(self):
        text = '<answer>{"accepted_marks": [1, 3], "meta": {"note": "ok"}}</answer>'
        self.assertEqual(parse_som_response(text), [1, 3])

    def test_non_string_input_returns_empty(self):
        self.assertEqual(parse_som_response(None), [])
        self.assertEqual(parse_som_response(12345), [])
        self.assertEqual(parse_som_response(""), [])

    def test_empty_valid_ids_filters_everything(self):
        text = '<answer>{"accepted_marks": [1, 2, 3]}</answer>'
        self.assertEqual(parse_som_response(text, valid_ids=set()), [])


from nibi_model_compare.som_missed_creatures import parse_click_proposals


class ParseClickProposalsTests(unittest.TestCase):
    def test_happy_path(self):
        text = (
            'Reasoning here.\n'
            '<answer>{"missed_creatures":[{"x":0.5,"y":0.3,"description":"small crab"}]}</answer>'
        )
        result = parse_click_proposals(text)
        self.assertEqual(len(result), 1)
        self.assertAlmostEqual(result[0]["x"], 0.5)
        self.assertAlmostEqual(result[0]["y"], 0.3)
        self.assertEqual(result[0]["description"], "small crab")

    def test_multiple_proposals(self):
        text = (
            '<answer>{"missed_creatures":['
            '{"x":0.1,"y":0.2,"description":"crab"},'
            '{"x":0.9,"y":0.8,"description":"snail"}'
            ']}</answer>'
        )
        result = parse_click_proposals(text)
        self.assertEqual(len(result), 2)
        self.assertEqual(result[0]["description"], "crab")
        self.assertEqual(result[1]["description"], "snail")

    def test_empty_list_valid(self):
        text = '<answer>{"missed_creatures":[]}</answer>'
        self.assertEqual(parse_click_proposals(text), [])

    def test_out_of_range_coords_dropped(self):
        text = (
            '<answer>{"missed_creatures":['
            '{"x":1.5,"y":0.5,"description":"oob"},'
            '{"x":0.5,"y":-0.1,"description":"neg"},'
            '{"x":0.4,"y":0.6,"description":"valid"}'
            ']}</answer>'
        )
        result = parse_click_proposals(text)
        self.assertEqual(len(result), 1)
        self.assertEqual(result[0]["description"], "valid")

    def test_boundary_coords_accepted(self):
        text = (
            '<answer>{"missed_creatures":['
            '{"x":0.0,"y":0.0,"description":"top-left"},'
            '{"x":1.0,"y":1.0,"description":"bottom-right"}'
            ']}</answer>'
        )
        result = parse_click_proposals(text)
        self.assertEqual(len(result), 2)

    def test_non_numeric_coords_dropped(self):
        text = (
            '<answer>{"missed_creatures":['
            '{"x":"0.5","y":0.3,"description":"bad_x"},'
            '{"x":0.5,"y":true,"description":"bool_y"},'
            '{"x":0.5,"y":0.5,"description":"good"}'
            ']}</answer>'
        )
        result = parse_click_proposals(text)
        self.assertEqual(len(result), 1)
        self.assertEqual(result[0]["description"], "good")

    def test_missing_tag_returns_empty(self):
        self.assertEqual(parse_click_proposals("no answer tag here"), [])

    def test_malformed_json_returns_empty(self):
        self.assertEqual(parse_click_proposals("<answer>{bad json}</answer>"), [])

    def test_non_string_input_returns_empty(self):
        self.assertEqual(parse_click_proposals(None), [])
        self.assertEqual(parse_click_proposals(123), [])
        self.assertEqual(parse_click_proposals(""), [])

    def test_missing_description_defaults_to_empty_string(self):
        text = '<answer>{"missed_creatures":[{"x":0.5,"y":0.5}]}</answer>'
        result = parse_click_proposals(text)
        self.assertEqual(len(result), 1)
        self.assertEqual(result[0]["description"], "")

    def test_integer_coords_accepted(self):
        # x=0, y=1 are valid ints that should be treated as 0.0 and 1.0
        text = '<answer>{"missed_creatures":[{"x":0,"y":1,"description":"corner"}]}</answer>'
        result = parse_click_proposals(text)
        self.assertEqual(len(result), 1)
        self.assertAlmostEqual(result[0]["x"], 0.0)
        self.assertAlmostEqual(result[0]["y"], 1.0)

    def test_takes_last_answer_block(self):
        text = (
            '<answer>{"missed_creatures":[{"x":0.1,"y":0.1,"description":"first"}]}</answer>\n'
            'Actually...\n'
            '<answer>{"missed_creatures":[{"x":0.9,"y":0.9,"description":"final"}]}</answer>'
        )
        result = parse_click_proposals(text)
        self.assertEqual(len(result), 1)
        self.assertEqual(result[0]["description"], "final")


import numpy as np

from nibi_model_compare.som_missed_creatures import (
    render_existing_masks_overlay,
    render_proposed_clicks_overlay,
)


class RenderExistingMasksOverlayTests(unittest.TestCase):
    H, W = 64, 64

    def _frame(self):
        return np.zeros((self.H, self.W, 3), dtype=np.uint8) + 50

    def _disk_mask(self, cy, cx, r):
        yy, xx = np.ogrid[:self.H, :self.W]
        return ((yy - cy) ** 2 + (xx - cx) ** 2) <= r * r

    def test_empty_masks_returns_copy(self):
        frame = self._frame()
        out = render_existing_masks_overlay(frame, [])
        np.testing.assert_array_equal(out, frame)

    def test_modifies_pixels_under_mask(self):
        frame = self._frame()
        mask = self._disk_mask(32, 32, 10)
        out = render_existing_masks_overlay(frame, [{"mask": mask}], alpha=0.4)
        # Where the mask is True, pixels should differ from input
        self.assertFalse(np.array_equal(out[mask], frame[mask]))

    def test_input_not_mutated(self):
        frame = self._frame()
        original = frame.copy()
        mask = self._disk_mask(20, 20, 6)
        render_existing_masks_overlay(frame, [{"mask": mask}])
        np.testing.assert_array_equal(frame, original)

    def test_green_channel_increased(self):
        frame = np.zeros((self.H, self.W, 3), dtype=np.uint8)
        mask = self._disk_mask(32, 32, 8)
        out = render_existing_masks_overlay(frame, [{"mask": mask}], alpha=0.5)
        # BGR format: channel 1 is green; should be > 0 where mask is set
        self.assertTrue(out[mask, 1].mean() > 50)


class RenderProposedClicksOverlayTests(unittest.TestCase):
    H, W = 64, 64

    def _frame(self):
        return np.zeros((self.H, self.W, 3), dtype=np.uint8) + 80

    def test_no_clicks_returns_same_shape(self):
        frame = self._frame()
        out = render_proposed_clicks_overlay(frame, [])
        self.assertEqual(out.shape, frame.shape)

    def test_with_clicks_modifies_pixels(self):
        frame = self._frame()
        clicks = [{"x": 0.5, "y": 0.5, "description": "test"}]
        out = render_proposed_clicks_overlay(frame, clicks)
        self.assertFalse(np.array_equal(frame, out))

    def test_with_existing_masks_applies_overlay(self):
        frame = self._frame()
        yy, xx = np.ogrid[:self.H, :self.W]
        mask = ((yy - 10) ** 2 + (xx - 10) ** 2) <= 5 * 5
        clicks = [{"x": 0.7, "y": 0.7, "description": "crab"}]
        out = render_proposed_clicks_overlay(frame, clicks, existing_masks=[{"mask": mask}])
        self.assertFalse(np.array_equal(frame, out))
        self.assertEqual(out.shape, frame.shape)


from nibi_model_compare.som_missed_creatures import select_target_frames


class SelectTargetFramesTests(unittest.TestCase):
    def _row(self, idx, **kw):
        base = {"frame_index": idx, "num_masks": 3, "error": None, "skipped": False}
        base.update(kw)
        return base

    def test_uniform_spacing_picks_K_valid_frames(self):
        frame_results = [self._row(i) for i in range(20)]
        out = select_target_frames(frame_results, strategy="uniform", k=5)
        self.assertEqual(len(out), 5)
        # uniformly spaced across 0..19 using round(i * step) where step = 19/4 = 4.75
        # -> round(0)=0, round(4.75)=5, round(9.5)=10, round(14.25)=14, round(19)=19
        self.assertEqual(out, [0, 5, 10, 14, 19])

    def test_uniform_skips_errored(self):
        frame_results = [
            self._row(0),
            self._row(1, error="IndexError"),
            self._row(2),
            self._row(3, error="ValueError"),
            self._row(4),
            self._row(5),
        ]
        out = select_target_frames(frame_results, strategy="uniform", k=3)
        # valid frames are [0, 2, 4, 5]; uniform spacing of 3 over 4 picks
        # indices 0, 2, 3 of that list -> [0, 4, 5]
        self.assertEqual(out, [0, 4, 5])

    def test_uniform_skips_skipped_and_zero_mask(self):
        frame_results = [
            self._row(0),
            self._row(1, skipped=True),
            self._row(2, num_masks=0),
            self._row(3),
            self._row(4),
        ]
        out = select_target_frames(frame_results, strategy="uniform", k=3)
        # valid: [0, 3, 4] -> all picked
        self.assertEqual(out, [0, 3, 4])

    def test_uniform_returns_fewer_when_k_exceeds_valid(self):
        frame_results = [self._row(i, error="x") for i in range(5)]
        frame_results.append(self._row(5))
        out = select_target_frames(frame_results, strategy="uniform", k=10)
        self.assertEqual(out, [5])

    def test_explicit_override_wins(self):
        frame_results = [self._row(i) for i in range(20)]
        out = select_target_frames(
            frame_results, strategy="uniform", k=5, explicit=[3, 7, 11]
        )
        self.assertEqual(out, [3, 7, 11])

    def test_explicit_override_filters_invalid_indices(self):
        frame_results = [self._row(i) for i in range(5)]
        out = select_target_frames(
            frame_results, strategy="uniform", k=5, explicit=[1, 99, 3]
        )
        self.assertEqual(out, [1, 3])

    def test_motion_strategy_delegates(self):
        called = {}

        def fake_motion(frame_results, k):
            called["ok"] = (len(frame_results), k)
            return [0, 2, 4]

        frame_results = [self._row(i) for i in range(5)]
        out = select_target_frames(
            frame_results, strategy="motion", k=3,
            _motion_selector=fake_motion,
        )
        self.assertEqual(out, [0, 2, 4])
        self.assertEqual(called["ok"], (5, 3))

    def test_include_zero_mask_frames_flag(self):
        frame_results = [
            self._row(0, num_masks=0),
            self._row(1, num_masks=0),
            self._row(2),
        ]
        out = select_target_frames(
            frame_results, strategy="uniform", k=3,
            include_zero_mask_frames=True,
        )
        self.assertEqual(out, [0, 1, 2])

    def test_unknown_strategy_raises(self):
        frame_results = [self._row(i) for i in range(5)]
        with self.assertRaises(ValueError):
            select_target_frames(frame_results, strategy="unifrom", k=3)

    def test_k_equals_one_returns_first_valid(self):
        frame_results = [self._row(i) for i in range(10)]
        out = select_target_frames(frame_results, strategy="uniform", k=1)
        self.assertEqual(out, [0])


from nibi_model_compare.som_missed_creatures import filter_candidates


def _disk_mask(h, w, cy, cx, r):
    yy, xx = np.ogrid[:h, :w]
    return ((yy - cy) ** 2 + (xx - cx) ** 2) <= r * r


def _bbox(mask):
    ys, xs = np.where(mask)
    return [int(xs.min()), int(ys.min()),
            int(xs.max() - xs.min() + 1), int(ys.max() - ys.min() + 1)]


def _cand(mask, score=0.9):
    return {"mask": mask, "bbox_xywh": _bbox(mask), "score": score}


class FilterCandidatesTests(unittest.TestCase):
    H, W = 64, 64

    def test_drops_high_iou_against_existing(self):
        cand_mask = _disk_mask(self.H, self.W, 20, 20, 6)
        existing_mask = _disk_mask(self.H, self.W, 20, 20, 6)  # identical
        candidates = [_cand(cand_mask)]
        existing = [{"mask": existing_mask}]
        out = filter_candidates(
            candidates, existing,
            iou_dedup=0.3, min_area_px=4, max_area_frac=0.5,
            edge_tol_px=2,
        )
        self.assertEqual(out, [])

    def test_keeps_low_iou_against_existing(self):
        cand_mask = _disk_mask(self.H, self.W, 20, 20, 6)
        existing_mask = _disk_mask(self.H, self.W, 50, 50, 6)
        out = filter_candidates(
            [_cand(cand_mask)], [{"mask": existing_mask}],
            iou_dedup=0.3, min_area_px=4, max_area_frac=0.5,
            edge_tol_px=2,
        )
        self.assertEqual(len(out), 1)

    def test_drops_too_small(self):
        tiny = _disk_mask(self.H, self.W, 20, 20, 1)  # ~5 px
        out = filter_candidates(
            [_cand(tiny)], [],
            iou_dedup=0.3, min_area_px=20, max_area_frac=0.5,
            edge_tol_px=2,
        )
        self.assertEqual(out, [])

    def test_drops_too_large(self):
        huge = np.ones((self.H, self.W), dtype=bool)  # full frame
        out = filter_candidates(
            [_cand(huge)], [],
            iou_dedup=0.3, min_area_px=4, max_area_frac=0.5,
            edge_tol_px=2,
        )
        self.assertEqual(out, [])

    def test_drops_multi_edge_clipped(self):
        # touches top + left edges
        m = np.zeros((self.H, self.W), dtype=bool)
        m[0:10, 0:10] = True
        out = filter_candidates(
            [_cand(m)], [],
            iou_dedup=0.3, min_area_px=4, max_area_frac=0.5,
            edge_tol_px=2,
        )
        self.assertEqual(out, [])

    def test_keeps_single_edge_clipped(self):
        # touches only top
        m = np.zeros((self.H, self.W), dtype=bool)
        m[0:10, 20:30] = True
        out = filter_candidates(
            [_cand(m)], [],
            iou_dedup=0.3, min_area_px=4, max_area_frac=0.5,
            edge_tol_px=2,
        )
        self.assertEqual(len(out), 1)

    def test_no_existing_masks_keeps_valid_candidates(self):
        m = _disk_mask(self.H, self.W, 30, 30, 6)
        out = filter_candidates(
            [_cand(m)], [],
            iou_dedup=0.3, min_area_px=4, max_area_frac=0.5,
            edge_tol_px=2,
        )
        self.assertEqual(len(out), 1)

    def test_with_reasons_returns_all_with_correct_strings(self):
        tiny = _disk_mask(self.H, self.W, 20, 20, 1)
        huge = np.ones((self.H, self.W), dtype=bool)
        multi_edge = np.zeros((self.H, self.W), dtype=bool)
        multi_edge[0:5, 0:5] = True
        dup_mask = _disk_mask(self.H, self.W, 32, 32, 6)
        existing = [{"mask": _disk_mask(self.H, self.W, 32, 32, 6)}]
        good = _disk_mask(self.H, self.W, 12, 50, 5)

        from nibi_model_compare.som_missed_creatures import filter_candidates_with_reasons

        cands = [_cand(tiny), _cand(huge), _cand(multi_edge),
                 _cand(dup_mask), _cand(good)]
        out = filter_candidates_with_reasons(
            cands, existing,
            iou_dedup=0.3, min_area_px=20, max_area_frac=0.5,
            edge_tol_px=2,
        )
        reasons = [reason for _c, reason in out]
        self.assertEqual(reasons, [
            "too_small",
            "too_large",
            "multi_edge_clipped",
            "duplicate_of_existing",
            None,
        ])


from nibi_model_compare.som_missed_creatures import draw_numbered_marks


class DrawNumberedMarksTests(unittest.TestCase):
    H, W = 64, 64

    def _frame(self):
        return np.zeros((self.H, self.W, 3), dtype=np.uint8) + 100

    def _cand_mask(self, cy, cx):
        yy, xx = np.ogrid[:self.H, :self.W]
        return ((yy - cy) ** 2 + (xx - cx) ** 2) <= 5 * 5

    def test_returns_same_shape(self):
        frame = self._frame()
        cands = [{"mask": self._cand_mask(20, 20), "bbox_xywh": [15, 15, 11, 11]}]
        out = draw_numbered_marks(frame, cands)
        self.assertEqual(out.shape, frame.shape)
        self.assertEqual(out.dtype, frame.dtype)

    def test_modifies_pixels(self):
        frame = self._frame()
        cands = [{"mask": self._cand_mask(20, 20), "bbox_xywh": [15, 15, 11, 11]}]
        out = draw_numbered_marks(frame, cands)
        self.assertFalse(np.array_equal(frame, out),
                         "Expected the marked frame to differ from input")

    def test_zero_candidates_returns_copy(self):
        frame = self._frame()
        frame_before = frame.copy()
        out = draw_numbered_marks(frame, [])
        self.assertEqual(out.shape, frame.shape)
        np.testing.assert_array_equal(frame, frame_before)  # input not mutated
        np.testing.assert_array_equal(out, frame_before)    # output equals input

    def test_skips_empty_mask_without_crashing(self):
        frame = self._frame()
        empty = np.zeros((self.H, self.W), dtype=bool)
        out = draw_numbered_marks(frame, [{"mask": empty, "bbox_xywh": [0, 0, 0, 0]}])
        self.assertEqual(out.shape, frame.shape)
        # Should be unchanged because no candidate produced output
        np.testing.assert_array_equal(out, frame)

    def test_handles_many_candidates(self):
        # Stress test: 20 marks in dense scene
        frame = self._frame()
        cands = []
        for i in range(20):
            cy = 8 + (i // 5) * 12
            cx = 8 + (i % 5) * 12
            cands.append({"mask": self._cand_mask(cy, cx),
                          "bbox_xywh": [cx - 5, cy - 5, 11, 11]})
        out = draw_numbered_marks(frame, cands)
        self.assertEqual(out.shape, frame.shape)


import os
import tempfile

from nibi_model_compare.som_missed_creatures import build_som_prompt_messages


class BuildSomPromptMessagesTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.target_path = os.path.join(self.tmp.name, "target.png")
        self.neighbour_paths = [
            os.path.join(self.tmp.name, f"n{i}.png") for i in range(4)
        ]
        # Create dummy files so the builder doesn't reject them
        from PIL import Image
        for p in [self.target_path, *self.neighbour_paths]:
            Image.new("RGB", (32, 32)).save(p)

    def tearDown(self):
        self.tmp.cleanup()

    def test_messages_have_system_and_user_roles(self):
        msgs = build_som_prompt_messages(
            system_prompt="SYS",
            target_image_path=self.target_path,
            neighbour_image_paths=self.neighbour_paths,
            initial_text_prompt="small creatures",
            num_marks=5,
        )
        self.assertEqual(msgs[0]["role"], "system")
        self.assertEqual(msgs[0]["content"], "SYS")
        self.assertEqual(msgs[1]["role"], "user")

    def test_user_message_includes_target_then_neighbours(self):
        msgs = build_som_prompt_messages(
            system_prompt="SYS",
            target_image_path=self.target_path,
            neighbour_image_paths=self.neighbour_paths,
            initial_text_prompt="creature",
            num_marks=3,
        )
        content = msgs[1]["content"]
        image_items = [c for c in content if c.get("type") == "image"]
        self.assertEqual(len(image_items), 1 + len(self.neighbour_paths))
        self.assertEqual(image_items[0]["image"], self.target_path)
        for nb_item, nb_path in zip(image_items[1:], self.neighbour_paths):
            self.assertEqual(nb_item["image"], nb_path)

    def test_user_message_mentions_query_and_num_marks(self):
        msgs = build_som_prompt_messages(
            system_prompt="SYS",
            target_image_path=self.target_path,
            neighbour_image_paths=self.neighbour_paths,
            initial_text_prompt="small creatures",
            num_marks=7,
        )
        text_blobs = [
            c["text"] for c in msgs[1]["content"] if c.get("type") == "text"
        ]
        joined = " ".join(text_blobs)
        self.assertIn("small creatures", joined)
        self.assertIn("7", joined)
        self.assertIn("accepted_marks", joined)

    def test_neighbour_list_empty_is_allowed(self):
        msgs = build_som_prompt_messages(
            system_prompt="SYS",
            target_image_path=self.target_path,
            neighbour_image_paths=[],
            initial_text_prompt="x",
            num_marks=1,
        )
        content = msgs[1]["content"]
        image_items = [c for c in content if c.get("type") == "image"]
        self.assertEqual(len(image_items), 1)

    def test_zero_num_marks_raises(self):
        with self.assertRaises(ValueError):
            build_som_prompt_messages(
                system_prompt="SYS",
                target_image_path=self.target_path,
                neighbour_image_paths=[],
                initial_text_prompt="x",
                num_marks=0,
            )

    def test_answer_instruction_is_last_content_item(self):
        msgs = build_som_prompt_messages(
            system_prompt="SYS",
            target_image_path=self.target_path,
            neighbour_image_paths=self.neighbour_paths,
            initial_text_prompt="small creatures",
            num_marks=3,
        )
        last_item = msgs[1]["content"][-1]
        self.assertEqual(last_item.get("type"), "text")
        self.assertIn("accepted_marks", last_item["text"])


from nibi_model_compare.som_missed_creatures import merge_accepted_masks_into_row


class MergeAcceptedMasksTests(unittest.TestCase):
    def _disk_cand(self, cy, cx, r, h=64, w=64):
        yy, xx = np.ogrid[:h, :w]
        m = ((yy - cy) ** 2 + (xx - cx) ** 2) <= r * r
        ys, xs = np.where(m)
        bbox = [int(xs.min()), int(ys.min()),
                int(xs.max() - xs.min() + 1),
                int(ys.max() - ys.min() + 1)]
        return {"mask": m, "bbox_xywh": bbox, "score": 0.8}

    def test_appends_new_objs_with_unique_ids(self):
        existing_row = {
            "frame_index": 5,
            "out_obj_ids": [1, 2, 3],
            "out_binary_masks_rle": [{}, {}, {}],   # opaque payloads
            "out_boxes_xywh": [[0, 0, 1, 1], [0, 0, 1, 1], [0, 0, 1, 1]],
            "out_probs": [0.9, 0.9, 0.9],
            "out_tracker_probs": [0.9, 0.9, 0.9],
        }
        accepted = [self._disk_cand(20, 20, 6), self._disk_cand(40, 40, 6)]
        out = merge_accepted_masks_into_row(existing_row, accepted)
        self.assertEqual(out["frame_index"], 5)
        self.assertEqual(out["out_obj_ids"], [1, 2, 3, 4, 5])
        self.assertEqual(out["added_obj_ids"], [4, 5])
        self.assertEqual(out["source"], "som")
        self.assertEqual(out["source_per_obj_id"], {
            "1": "text_agent", "2": "text_agent", "3": "text_agent",
            "4": "som", "5": "som",
        })
        self.assertEqual(len(out["out_binary_masks_rle"]), 5)
        self.assertEqual(len(out["out_boxes_xywh"]), 5)

    def test_empty_accepted_returns_passthrough_row(self):
        existing_row = {
            "frame_index": 5,
            "out_obj_ids": [1, 2],
            "out_binary_masks_rle": [{}, {}],
            "out_boxes_xywh": [[0, 0, 1, 1], [0, 0, 1, 1]],
            "out_probs": [0.9, 0.9],
            "out_tracker_probs": [0.9, 0.9],
        }
        out = merge_accepted_masks_into_row(existing_row, [])
        self.assertEqual(out["out_obj_ids"], [1, 2])
        self.assertEqual(out["added_obj_ids"], [])
        self.assertEqual(out["source"], "som")
        self.assertEqual(out["source_per_obj_id"], {
            "1": "text_agent", "2": "text_agent",
        })

    def test_no_existing_objs_starts_from_id_1(self):
        existing_row = {
            "frame_index": 5,
            "out_obj_ids": [],
            "out_binary_masks_rle": [],
            "out_boxes_xywh": [],
            "out_probs": [],
            "out_tracker_probs": [],
        }
        accepted = [self._disk_cand(20, 20, 6)]
        out = merge_accepted_masks_into_row(existing_row, accepted)
        self.assertEqual(out["out_obj_ids"], [1])
        self.assertEqual(out["added_obj_ids"], [1])

    def test_input_row_not_mutated(self):
        existing_row = {
            "frame_index": 5,
            "out_obj_ids": [1],
            "out_binary_masks_rle": [{}],
            "out_boxes_xywh": [[0, 0, 1, 1]],
            "out_probs": [0.9],
            "out_tracker_probs": [0.9],
        }
        merge_accepted_masks_into_row(existing_row, [self._disk_cand(20, 20, 6)])
        self.assertEqual(existing_row["out_obj_ids"], [1])

    def test_numpy_int_existing_obj_ids_yields_python_int_outputs(self):
        import json
        existing_row = {
            "frame_index": 5,
            "out_obj_ids": [np.int64(1), np.int64(2)],
            "out_binary_masks_rle": [{}, {}],
            "out_boxes_xywh": [[0, 0, 1, 1], [0, 0, 1, 1]],
            "out_probs": [0.9, 0.9],
            "out_tracker_probs": [0.9, 0.9],
        }
        out = merge_accepted_masks_into_row(existing_row, [self._disk_cand(20, 20, 6)])
        # Roundtrips through JSON without raising
        serialized = json.dumps(out["out_obj_ids"])
        self.assertIn("3", serialized)
        self.assertIsInstance(out["added_obj_ids"][0], int)
        # Source map keys are stringified
        self.assertEqual(out["source_per_obj_id"]["3"], "som")

    def test_sparse_existing_obj_ids_continues_from_max(self):
        existing_row = {
            "frame_index": 5,
            "out_obj_ids": [1, 3, 7],
            "out_binary_masks_rle": [{}, {}, {}],
            "out_boxes_xywh": [[0, 0, 1, 1]] * 3,
            "out_probs": [0.9] * 3,
            "out_tracker_probs": [0.9] * 3,
        }
        out = merge_accepted_masks_into_row(existing_row, [self._disk_cand(20, 20, 6)])
        self.assertEqual(out["added_obj_ids"], [8])


from nibi_model_compare.som_missed_creatures import load_system_prompt
from nibi_model_compare.som_missed_creatures import load_click_discovery_system_prompt


class LoadSystemPromptTests(unittest.TestCase):
    def test_loads_underwater(self):
        body = load_system_prompt("underwater")
        self.assertIn("marine biology", body.lower())
        self.assertIn("<answer>", body)

    def test_loads_general(self):
        body = load_system_prompt("general")
        self.assertIn("<answer>", body)
        self.assertNotIn("marine biology", body.lower())

    def test_unknown_profile_raises(self):
        with self.assertRaises(ValueError):
            load_system_prompt("nonsense")


class LoadClickDiscoverySystemPromptTests(unittest.TestCase):
    def test_loads_underwater_discovery(self):
        body = load_click_discovery_system_prompt("underwater")
        self.assertIn("marine biology", body.lower())
        self.assertIn("<answer>", body)
        self.assertIn("missed_creatures", body)

    def test_loads_general_discovery(self):
        body = load_click_discovery_system_prompt("general")
        self.assertIn("<answer>", body)
        self.assertIn("missed_creatures", body)
        self.assertNotIn("marine biology", body.lower())

    def test_unknown_profile_raises(self):
        with self.assertRaises(ValueError):
            load_click_discovery_system_prompt("nonsense")


import json as _json

from nibi_model_compare.som_missed_creatures import (
    run_som_stage,
    SomStageConfig,
    Sam3PointService,
)


class FakeSam3PointService:
    """Deterministic fake that returns a center-disk mask for every click group.

    Exposes both ``group_segment`` (new primary API) and ``point_segment``
    (legacy wrapper) so existing tests that call either method still work.

    ``group_segment`` returns one result per group shaped as
    Sam3PointService.group_segment promises:
        {"creature_id": int, "description": str, "mask": bool HxW,
         "score": float, "area_px": int, "select_reason": str,
         "spatial_match": "click_mode", "clicks_used": list}

    ``point_segment`` returns the legacy shape as before.
    """

    def __init__(self, h=64, w=64, radius=5):
        self.h = h
        self.w = w
        self.radius = radius
        self.calls = []

    def group_segment(self, image_path, groups, output_folder=None):
        self.calls.append((image_path, [dict(g) for g in groups], output_folder))
        results = []
        for grp in groups:
            clicks = grp.get("clicks") or []
            # Use the first foreground click to place the mask center
            fg_clicks = [c for c in clicks if int(c.get("label", 1)) == 1]
            if fg_clicks:
                c = fg_clicks[0]
                cx = int(round(c["x"] * self.w))
                cy = int(round(c["y"] * self.h))
            else:
                cx, cy = self.w // 2, self.h // 2
            yy, xx = np.ogrid[:self.h, :self.w]
            mask = ((yy - cy) ** 2 + (xx - cx) ** 2) <= self.radius ** 2
            results.append({
                "creature_id": int(grp.get("id", -1)),
                "description": str(grp.get("description", "")),
                "mask": mask,
                "score": 0.85,
                "area_px": int(mask.sum()),
                "select_reason": "smallest_in_band",
                "spatial_match": "click_mode",
                "clicks_used": clicks,
            })
        return results

    def point_segment(self, image_path, clicks, output_folder=None):
        self.calls.append((image_path, [dict(c) for c in clicks], output_folder))
        results = []
        for click in clicks:
            x_n = click["x"]
            y_n = click["y"]
            cx = int(round(x_n * self.w))
            cy = int(round(y_n * self.h))
            yy, xx = np.ogrid[:self.h, :self.w]
            mask = ((yy - cy) ** 2 + (xx - cx) ** 2) <= self.radius ** 2
            results.append({
                "mask": mask,
                "score": 0.85,
                "sam_text_prompt": "",
                "spatial_match": "click_mode",
                "area_px": int(mask.sum()),
                "select_reason": "smallest_in_band",
            })
        return results


class RunSomStageTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.workdir = self.tmp.name

        # Fake video stub
        self.video_path = os.path.join(self.workdir, "fake.mp4")
        with open(self.video_path, "w") as f:
            f.write("(stub)")

        # Existing frame_results.jsonl
        self.frame_results_path = os.path.join(self.workdir, "frame_results.jsonl")
        with open(self.frame_results_path, "w") as f:
            for i in range(60):
                row = {
                    "frame_index": i,
                    "num_masks": 2,
                    "error": None,
                    "skipped": False,
                }
                f.write(_json.dumps(row) + "\n")

        # Existing frame_outputs_rle.json with two text-agent masks per frame
        self.frame_outputs_path = os.path.join(self.workdir, "frame_outputs_rle.json")
        from nibi_model_compare.frame_output_utils import encode_binary_mask_to_rle
        frames = []
        for i in range(60):
            yy, xx = np.ogrid[:64, :64]
            m1 = ((yy - 10) ** 2 + (xx - 10) ** 2) <= 4 * 4
            m2 = ((yy - 50) ** 2 + (xx - 50) ** 2) <= 4 * 4
            frames.append({
                "frame_index": i,
                "out_obj_ids": [1, 2],
                "out_binary_masks_rle": [
                    encode_binary_mask_to_rle(m1),
                    encode_binary_mask_to_rle(m2),
                ],
                "out_boxes_xywh": [[6, 6, 9, 9], [46, 46, 9, 9]],
                "out_probs": [0.9, 0.9],
                "out_tracker_probs": [0.9, 0.9],
            })
        with open(self.frame_outputs_path, "w") as f:
            _json.dump({"format_version": 2, "frames": frames}, f)

        # Fake video frame loader: every frame is a 64x64 grey image
        def fake_load_frame(video_path, frame_index):
            return np.full((64, 64, 3), 80, dtype=np.uint8)
        self.fake_load_frame = fake_load_frame

        # Fake SAM3 point service: creates a mask near the center (not covered
        # by the two corner existing masks)
        self.fake_point_service = FakeSam3PointService(h=64, w=64, radius=5)

        # Fake discovery MLLM: proposes one click at center (0.5, 0.5)
        # Fake judge MLLM: always accepts mark 1
        call_count = {"n": 0}

        def fake_mllm(messages, **_kw):
            call_count["n"] += 1
            # Determine if this is a discovery call (contains "missed_creatures"
            # in the user text) or a judge call
            user_content = messages[1]["content"] if len(messages) > 1 else []
            text_items = [
                item["text"] for item in user_content
                if isinstance(item, dict) and item.get("type") == "text"
            ]
            joined = " ".join(text_items)
            if "missed_creatures" in joined or "NORMALIZED" in joined:
                # Discovery response -- new grouped-click contract
                return (
                    'I see a creature at center.\n'
                    '<answer>{"missed_creatures":[{"id":1,"description":"center crab",'
                    '"clicks":[{"x":0.5,"y":0.5,"label":1}]}]}</answer>'
                )
            else:
                # Judge response
                return '<answer>{"accepted_marks": [1]}</answer>'

        self.fake_mllm = fake_mllm

    def tearDown(self):
        self.tmp.cleanup()

    def _config(self, **overrides):
        cfg = dict(
            video_path=self.video_path,
            frame_results_path=self.frame_results_path,
            frame_outputs_path=self.frame_outputs_path,
            output_dir=os.path.join(self.workdir, "som_out"),
            prompt_profile="underwater",
            initial_text_prompt="small creatures",
            num_target_frames=3,
            frame_selection_strategy="uniform",
            target_frames_explicit=None,
            num_neighbours=2,
            neighbour_offset_frames=15,
            iou_dedup=0.3,
            min_area_px=4,
            max_area_frac=0.5,
            edge_tol_px=2,
            internal_iou_dedup=0.5,
            max_mllm_calls=100,
            discovery_num_neighbours=4,
        )
        cfg.update(overrides)
        return SomStageConfig(**cfg)

    def test_end_to_end_appends_one_mask_per_target_frame(self):
        cfg = self._config()
        result = run_som_stage(
            cfg,
            _load_video_frame=self.fake_load_frame,
            _send_mllm_request=self.fake_mllm,
            _sam3_point_service=self.fake_point_service,
        )

        self.assertEqual(result["targets_processed"], 3)
        # Augmented JSONL exists with one row per target
        augmented = os.path.join(cfg.output_dir, "augmented_frame_outputs.jsonl")
        with open(augmented) as f:
            rows = [_json.loads(line) for line in f]
        self.assertEqual(len(rows), 3)
        for row in rows:
            # Each row has the 2 original + 1 new mask
            self.assertEqual(len(row["out_obj_ids"]), 3)
            self.assertEqual(row["added_obj_ids"], [3])
            self.assertEqual(row["source"], "som")

    def test_skips_frame_when_discovery_returns_no_clicks(self):
        def empty_discovery_mllm(messages, **_kw):
            # Always return empty discovery
            return '<answer>{"missed_creatures":[]}</answer>'

        cfg = self._config(num_target_frames=2)
        result = run_som_stage(
            cfg,
            _load_video_frame=self.fake_load_frame,
            _send_mllm_request=empty_discovery_mllm,
            _sam3_point_service=self.fake_point_service,
        )

        self.assertEqual(result["targets_processed"], 0)
        self.assertEqual(result["targets_skipped"], 2)
        # Augmented JSONL exists but is empty
        augmented = os.path.join(cfg.output_dir, "augmented_frame_outputs.jsonl")
        with open(augmented) as f:
            self.assertEqual(f.read().strip(), "")

    def test_mllm_judge_returns_out_of_range_logs_and_continues(self):
        call_count = {"n": 0}

        def oob_judge_mllm(messages, **_kw):
            call_count["n"] += 1
            user_content = messages[1]["content"] if len(messages) > 1 else []
            text_items = [
                item["text"] for item in user_content
                if isinstance(item, dict) and item.get("type") == "text"
            ]
            joined = " ".join(text_items)
            if "missed_creatures" in joined or "NORMALIZED" in joined:
                return (
                    '<answer>{"missed_creatures":[{"id":1,"description":"test",'
                    '"clicks":[{"x":0.5,"y":0.5,"label":1}]}]}</answer>'
                )
            # Judge: return out-of-range id
            return '<answer>{"accepted_marks": [99]}</answer>'

        cfg = self._config(num_target_frames=2)
        result = run_som_stage(
            cfg,
            _load_video_frame=self.fake_load_frame,
            _send_mllm_request=oob_judge_mllm,
            _sam3_point_service=self.fake_point_service,
        )

        self.assertEqual(result["targets_processed"], 2)
        # No new masks were appended because all proposed ids were OOB
        augmented = os.path.join(cfg.output_dir, "augmented_frame_outputs.jsonl")
        with open(augmented) as f:
            rows = [_json.loads(line) for line in f]
        for row in rows:
            self.assertEqual(row["added_obj_ids"], [])

    def test_budget_cap_stops_early(self):
        # With discovery+judge each costing 1 call, 2 targets = 4 calls total.
        # Cap at 2 means only 1 target fully processed.
        cfg = self._config(num_target_frames=10, max_mllm_calls=2)
        result = run_som_stage(
            cfg,
            _load_video_frame=self.fake_load_frame,
            _send_mllm_request=self.fake_mllm,
            _sam3_point_service=self.fake_point_service,
        )
        self.assertLessEqual(result["mllm_calls"], 2)

    def test_resume_skips_already_processed_targets(self):
        cfg = self._config(num_target_frames=3)
        os.makedirs(cfg.output_dir, exist_ok=True)
        augmented = os.path.join(cfg.output_dir, "augmented_frame_outputs.jsonl")
        with open(augmented, "w") as f:
            # Pretend target 0 was already processed
            f.write(_json.dumps({
                "frame_index": 0,
                "source": "som",
                "added_obj_ids": [3],
                "out_obj_ids": [1, 2, 3],
                "out_binary_masks_rle": [],
                "out_boxes_xywh": [],
                "out_probs": [],
                "out_tracker_probs": [],
            }) + "\n")

        result = run_som_stage(
            cfg,
            _load_video_frame=self.fake_load_frame,
            _send_mllm_request=self.fake_mllm,
            _sam3_point_service=self.fake_point_service,
        )
        # Target 0 should have been resumed (not reprocessed)
        self.assertEqual(result["targets_skipped_resume"], 1)
        self.assertEqual(result["targets_processed"], 2)

    def test_input_frame_outputs_not_mutated(self):
        import hashlib
        cfg = self._config(num_target_frames=2)
        digest_before = hashlib.sha256(
            open(self.frame_outputs_path, "rb").read()
        ).hexdigest()
        run_som_stage(
            cfg,
            _load_video_frame=self.fake_load_frame,
            _send_mllm_request=self.fake_mllm,
            _sam3_point_service=self.fake_point_service,
        )
        digest_after = hashlib.sha256(
            open(self.frame_outputs_path, "rb").read()
        ).hexdigest()
        self.assertEqual(digest_before, digest_after)

    def test_malformed_existing_rle_does_not_crash(self):
        # Corrupt the existing frame_outputs to have a bad RLE on one frame
        with open(self.frame_outputs_path) as f:
            fo = _json.load(f)
        fo["frames"][0]["out_binary_masks_rle"][0] = {"size": [64, 64], "counts": "INVALID"}
        with open(self.frame_outputs_path, "w") as f:
            _json.dump(fo, f)

        cfg = self._config(num_target_frames=3, target_frames_explicit=[0, 5, 10])
        # Should not raise; frame 0 just has its bad RLE silently dropped
        result = run_som_stage(
            cfg,
            _load_video_frame=self.fake_load_frame,
            _send_mllm_request=self.fake_mllm,
            _sam3_point_service=self.fake_point_service,
        )
        self.assertGreaterEqual(result["targets_processed"], 1)

    def test_artefact_filenames_are_correct(self):
        """Verify the new artefact naming scheme (02/03/04/05/06/07)."""
        cfg = self._config(num_target_frames=1, target_frames_explicit=[5])
        run_som_stage(
            cfg,
            _load_video_frame=self.fake_load_frame,
            _send_mllm_request=self.fake_mllm,
            _sam3_point_service=self.fake_point_service,
        )
        target_dir = os.path.join(cfg.output_dir, "target_000005")
        expected = [
            "01_raw.png",
            "02_existing_masks.png",
            "03_proposed_clicks.png",
            "04_marked.png",
            "05_judge_request.json",
            "05_judge_response.txt",
            "06_accepted.json",
        ]
        for fname in expected:
            self.assertTrue(
                os.path.exists(os.path.join(target_dir, fname)),
                f"Expected artefact missing: {fname}",
            )

    def test_discovery_request_json_saved(self):
        """discovery_request.json should be written with the grouped click proposals."""
        cfg = self._config(num_target_frames=1, target_frames_explicit=[5])
        run_som_stage(
            cfg,
            _load_video_frame=self.fake_load_frame,
            _send_mllm_request=self.fake_mllm,
            _sam3_point_service=self.fake_point_service,
        )
        target_dir = os.path.join(cfg.output_dir, "target_000005")
        req_path = os.path.join(target_dir, "discovery_request.json")
        self.assertTrue(os.path.exists(req_path))
        with open(req_path) as f:
            groups = _json.load(f)
        self.assertIsInstance(groups, list)
        self.assertEqual(len(groups), 1)
        # New grouped contract: each entry has an id, description, and clicks list
        self.assertIn("id", groups[0])
        self.assertIn("clicks", groups[0])
        self.assertAlmostEqual(groups[0]["clicks"][0]["x"], 0.5)


import subprocess


class MergeSomOutputsTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.workdir = self.tmp.name
        # Original frame_outputs with 2 frames, 1 mask each
        from nibi_model_compare.frame_output_utils import encode_binary_mask_to_rle
        yy, xx = np.ogrid[:32, :32]
        m1 = ((yy - 8) ** 2 + (xx - 8) ** 2) <= 3 * 3
        m2 = ((yy - 24) ** 2 + (xx - 24) ** 2) <= 3 * 3
        self.original_path = os.path.join(self.workdir, "fo.json")
        with open(self.original_path, "w") as f:
            _json.dump({
                "format_version": 2,
                "frames": [
                    {"frame_index": 0, "out_obj_ids": [1],
                     "out_binary_masks_rle": [encode_binary_mask_to_rle(m1)],
                     "out_boxes_xywh": [[5, 5, 7, 7]],
                     "out_probs": [0.9], "out_tracker_probs": [0.9]},
                    {"frame_index": 5, "out_obj_ids": [1],
                     "out_binary_masks_rle": [encode_binary_mask_to_rle(m2)],
                     "out_boxes_xywh": [[21, 21, 7, 7]],
                     "out_probs": [0.9], "out_tracker_probs": [0.9]},
                ],
            }, f)

        # Augmented JSONL: replaces frame 5 with +1 mask
        m3 = ((yy - 16) ** 2 + (xx - 16) ** 2) <= 3 * 3
        self.augmented_path = os.path.join(self.workdir, "aug.jsonl")
        with open(self.augmented_path, "w") as f:
            f.write(_json.dumps({
                "frame_index": 5,
                "source": "som",
                "added_obj_ids": [2],
                "out_obj_ids": [1, 2],
                "out_binary_masks_rle": [
                    encode_binary_mask_to_rle(m2),
                    encode_binary_mask_to_rle(m3),
                ],
                "out_boxes_xywh": [[21, 21, 7, 7], [13, 13, 7, 7]],
                "out_probs": [0.9, 0.85],
                "out_tracker_probs": [0.9, 0.85],
                "source_per_obj_id": {"1": "text_agent", "2": "som"},
            }) + "\n")
        self.out_path = os.path.join(self.workdir, "merged.json")

    def tearDown(self):
        self.tmp.cleanup()

    def test_merge_replaces_augmented_frames_only(self):
        cmd = [
            ".venv/bin/python", "scripts/merge_som_outputs.py",
            "--frame-outputs", self.original_path,
            "--augmented", self.augmented_path,
            "--output", self.out_path,
        ]
        result = subprocess.run(cmd, capture_output=True, text=True,
                                cwd=os.path.abspath(
                                    os.path.join(os.path.dirname(__file__),
                                                 "..")))
        self.assertEqual(result.returncode, 0,
                         f"merge failed: {result.stderr}")

        with open(self.out_path) as f:
            merged = _json.load(f)
        frames_by_idx = {f["frame_index"]: f for f in merged["frames"]}
        # Frame 0 unchanged
        self.assertEqual(frames_by_idx[0]["out_obj_ids"], [1])
        # Frame 5 augmented
        self.assertEqual(frames_by_idx[5]["out_obj_ids"], [1, 2])
        self.assertEqual(frames_by_idx[5]["source"], "som")

    def test_original_file_is_not_mutated(self):
        before = open(self.original_path).read()
        cmd = [
            ".venv/bin/python", "scripts/merge_som_outputs.py",
            "--frame-outputs", self.original_path,
            "--augmented", self.augmented_path,
            "--output", self.out_path,
        ]
        subprocess.run(cmd, capture_output=True, text=True,
                       cwd=os.path.abspath(
                           os.path.join(os.path.dirname(__file__), "..")))
        after = open(self.original_path).read()
        self.assertEqual(before, after)


class Sam3PointServiceSelectionTests(unittest.TestCase):
    def test_picks_smallest_in_band(self):
        import numpy as np
        from PIL import Image
        from nibi_model_compare.som_missed_creatures import Sam3PointService

        H, W = 200, 200
        # Three masks: tiny (50 px), small-creature (1600 px), huge (30000 px)
        m_tiny = np.zeros((H, W), dtype=bool); m_tiny[10:15, 10:20] = True
        m_small = np.zeros((H, W), dtype=bool); m_small[20:60, 20:60] = True
        m_huge = np.zeros((H, W), dtype=bool); m_huge[10:160, 10:210][:, :200] = True

        captured = {}

        class FakeProcessor:
            def set_image(self, image):
                captured["set_image"] = True
                return {"_state": "fake"}

        class FakeModel:
            def predict_inst(self, state, point_coords, point_labels,
                             multimask_output=True):
                captured["state"] = state
                # Return masks ordered tiny->small->huge with monotone scores
                masks = np.stack([m_tiny, m_small, m_huge], axis=0)
                scores = np.array([0.9, 0.7, 0.5], dtype=np.float32)
                logits = np.zeros((3, 256, 256), dtype=np.float32)
                return masks, scores, logits

        import tempfile, os
        with tempfile.TemporaryDirectory() as tmp:
            img_path = os.path.join(tmp, "f.png")
            Image.new("RGB", (W, H)).save(img_path)
            svc = Sam3PointService(FakeModel(), FakeProcessor())
            out = svc.point_segment(img_path, [{"x": 0.5, "y": 0.5, "description": "thing"}])
            self.assertEqual(len(out), 1)
            # Should pick m_small (1600 px), not m_tiny (50 px, below min) or m_huge
            picked_area = int(out[0]["mask"].sum())
            self.assertEqual(picked_area, int(m_small.sum()))
            self.assertEqual(out[0]["select_reason"], "smallest_in_band")

    def test_falls_back_when_all_below_min(self):
        import numpy as np
        from PIL import Image
        from nibi_model_compare.som_missed_creatures import Sam3PointService

        H, W = 200, 200
        # All three masks below the 200-pixel min (image is 200x200=40000 -> min=200)
        m1 = np.zeros((H, W), dtype=bool); m1[0:5, 0:5] = True       # 25 px
        m2 = np.zeros((H, W), dtype=bool); m2[10:18, 10:18] = True   # 64 px
        m3 = np.zeros((H, W), dtype=bool); m3[20:30, 20:30] = True   # 100 px

        class FakeProcessor:
            def set_image(self, image):
                return {}

        class FakeModel:
            def predict_inst(self, state, **kw):
                masks = np.stack([m1, m2, m3], axis=0)
                scores = np.array([0.1, 0.5, 0.9], dtype=np.float32)
                return masks, scores, np.zeros((3, 256, 256), dtype=np.float32)

        import tempfile, os
        with tempfile.TemporaryDirectory() as tmp:
            img_path = os.path.join(tmp, "f.png")
            Image.new("RGB", (W, H)).save(img_path)
            svc = Sam3PointService(FakeModel(), FakeProcessor())
            out = svc.point_segment(img_path, [{"x": 0.5, "y": 0.5, "description": "x"}])
            # Should fall back to highest score (0.9 -> m3)
            self.assertEqual(out[0]["select_reason"], "fallback_highest_score")


from nibi_model_compare.som_missed_creatures import parse_creature_click_groups


class ParseCreatureClickGroupsTests(unittest.TestCase):
    """Tests for the new grouped-click MLLM contract parser."""

    def test_happy_path_two_creatures_one_click_each(self):
        text = (
            '<answer>{"missed_creatures":['
            '{"id":1,"description":"large crab","clicks":[{"x":0.1,"y":0.2,"label":1}]},'
            '{"id":2,"description":"small snail","clicks":[{"x":0.8,"y":0.9,"label":1}]}'
            ']}</answer>'
        )
        result = parse_creature_click_groups(text)
        self.assertEqual(len(result), 2)
        self.assertEqual(result[0]["id"], 1)
        self.assertEqual(result[0]["description"], "large crab")
        self.assertEqual(len(result[0]["clicks"]), 1)
        self.assertAlmostEqual(result[0]["clicks"][0]["x"], 0.1)
        self.assertEqual(result[0]["clicks"][0]["label"], 1)
        self.assertEqual(result[1]["id"], 2)

    def test_mixed_positive_and_negative_labels_accepted(self):
        text = (
            '<answer>{"missed_creatures":['
            '{"id":1,"description":"crab","clicks":['
            '{"x":0.5,"y":0.5,"label":1},'
            '{"x":0.6,"y":0.6,"label":0}'
            ']}'
            ']}</answer>'
        )
        result = parse_creature_click_groups(text)
        self.assertEqual(len(result), 1)
        labels = [c["label"] for c in result[0]["clicks"]]
        self.assertEqual(labels, [1, 0])

    def test_out_of_range_coords_filtered(self):
        text = (
            '<answer>{"missed_creatures":['
            '{"id":1,"description":"oob crab","clicks":['
            '{"x":1.5,"y":0.5,"label":1},'
            '{"x":0.5,"y":-0.1,"label":1},'
            '{"x":0.4,"y":0.6,"label":1}'
            ']}'
            ']}</answer>'
        )
        result = parse_creature_click_groups(text)
        self.assertEqual(len(result), 1)
        self.assertEqual(len(result[0]["clicks"]), 1)
        self.assertAlmostEqual(result[0]["clicks"][0]["x"], 0.4)

    def test_empty_or_missing_clicks_drops_creature(self):
        text = (
            '<answer>{"missed_creatures":['
            '{"id":1,"description":"no clicks creature","clicks":[]},'
            '{"id":2,"description":"missing clicks key"},'
            '{"id":3,"description":"valid","clicks":[{"x":0.5,"y":0.5,"label":1}]}'
            ']}</answer>'
        )
        result = parse_creature_click_groups(text)
        self.assertEqual(len(result), 1)
        self.assertEqual(result[0]["id"], 3)

    def test_auto_assigns_ids_when_omitted(self):
        text = (
            '<answer>{"missed_creatures":['
            '{"description":"crab A","clicks":[{"x":0.1,"y":0.1,"label":1}]},'
            '{"description":"crab B","clicks":[{"x":0.9,"y":0.9,"label":1}]}'
            ']}</answer>'
        )
        result = parse_creature_click_groups(text)
        self.assertEqual(len(result), 2)
        ids = [g["id"] for g in result]
        self.assertEqual(sorted(ids), [1, 2])

    def test_returns_empty_on_missing_tag(self):
        self.assertEqual(parse_creature_click_groups("no answer here"), [])

    def test_returns_empty_on_malformed_json(self):
        self.assertEqual(parse_creature_click_groups("<answer>{bad json}</answer>"), [])

    def test_returns_empty_on_non_string_input(self):
        self.assertEqual(parse_creature_click_groups(None), [])
        self.assertEqual(parse_creature_click_groups(123), [])

    def test_duplicate_ids_reassigned_sequentially(self):
        text = (
            '<answer>{"missed_creatures":['
            '{"id":1,"description":"first","clicks":[{"x":0.1,"y":0.1,"label":1}]},'
            '{"id":1,"description":"duplicate id","clicks":[{"x":0.2,"y":0.2,"label":1}]}'
            ']}</answer>'
        )
        result = parse_creature_click_groups(text)
        self.assertEqual(len(result), 2)
        self.assertNotEqual(result[0]["id"], result[1]["id"])

    def test_invalid_label_filtered(self):
        text = (
            '<answer>{"missed_creatures":['
            '{"id":1,"description":"bad label","clicks":['
            '{"x":0.5,"y":0.5,"label":2},'
            '{"x":0.5,"y":0.5,"label":1}'
            ']}'
            ']}</answer>'
        )
        result = parse_creature_click_groups(text)
        self.assertEqual(len(result), 1)
        self.assertEqual(len(result[0]["clicks"]), 1)
        self.assertEqual(result[0]["clicks"][0]["label"], 1)

    def test_takes_last_answer_block(self):
        text = (
            '<answer>{"missed_creatures":[{"id":1,"description":"first",'
            '"clicks":[{"x":0.1,"y":0.1,"label":1}]}]}</answer>\n'
            'Actually...\n'
            '<answer>{"missed_creatures":[{"id":1,"description":"final",'
            '"clicks":[{"x":0.9,"y":0.9,"label":1}]}]}</answer>'
        )
        result = parse_creature_click_groups(text)
        self.assertEqual(len(result), 1)
        self.assertEqual(result[0]["description"], "final")


from nibi_model_compare.som_missed_creatures import Sam3PointService


class Sam3PointServiceGroupSegmentTests(unittest.TestCase):
    """Tests for Sam3PointService.group_segment."""

    def _make_service(self, h=200, w=200):
        """Return a Sam3PointService with a deterministic fake model."""
        captured = {}

        class FakeProcessor:
            def set_image(self, image):
                captured["set_image"] = True
                return {"_state": "fake"}

        class FakeModel:
            def predict_inst(self, state, point_coords, point_labels,
                             multimask_output=True):
                captured["point_coords"] = point_coords
                captured["point_labels"] = point_labels
                m_small = np.zeros((h, w), dtype=bool)
                m_small[20:60, 20:60] = True   # 1600 px (in band)
                m_huge = np.zeros((h, w), dtype=bool)
                m_huge[0:h, 0:w] = True         # full frame (above max)
                masks = np.stack([m_small, m_huge], axis=0)
                scores = np.array([0.9, 0.5], dtype=np.float32)
                return masks, scores, np.zeros((2, 256, 256), dtype=np.float32)

        self.captured = captured
        from PIL import Image
        import tempfile
        tmp = tempfile.mkdtemp()
        img_path = os.path.join(tmp, "frame.png")
        Image.new("RGB", (w, h)).save(img_path)
        return Sam3PointService(FakeModel(), FakeProcessor()), img_path

    def test_multi_click_group_passes_all_coords_to_predict_inst(self):
        svc, img_path = self._make_service()
        groups = [
            {
                "id": 1,
                "description": "elongated worm",
                "clicks": [
                    {"x": 0.1, "y": 0.2, "label": 1},
                    {"x": 0.3, "y": 0.4, "label": 1},
                ],
            }
        ]
        results = svc.group_segment(img_path, groups)
        # predict_inst should have received 2 rows
        self.assertEqual(self.captured["point_coords"].shape[0], 2)
        self.assertEqual(len(results), 1)

    def test_result_contains_required_keys(self):
        svc, img_path = self._make_service()
        groups = [
            {
                "id": 5,
                "description": "test crab",
                "clicks": [{"x": 0.5, "y": 0.5, "label": 1}],
            }
        ]
        results = svc.group_segment(img_path, groups)
        self.assertEqual(len(results), 1)
        r = results[0]
        for key in ("creature_id", "description", "mask", "score",
                    "area_px", "select_reason", "spatial_match", "clicks_used"):
            self.assertIn(key, r, f"Missing key: {key}")
        self.assertEqual(r["creature_id"], 5)
        self.assertEqual(r["description"], "test crab")
        self.assertEqual(r["spatial_match"], "click_mode")
        self.assertGreater(r["area_px"], 0)

    def test_negative_label_propagates_to_point_labels(self):
        svc, img_path = self._make_service()
        groups = [
            {
                "id": 2,
                "description": "crab near substrate",
                "clicks": [
                    {"x": 0.5, "y": 0.5, "label": 1},
                    {"x": 0.2, "y": 0.8, "label": 0},
                ],
            }
        ]
        svc.group_segment(img_path, groups)
        labels = list(self.captured["point_labels"])
        self.assertIn(1, labels)
        self.assertIn(0, labels)

    def test_empty_group_returns_click_mode_empty(self):
        svc, img_path = self._make_service()
        groups = [
            {"id": 3, "description": "no clicks", "clicks": []},
        ]
        results = svc.group_segment(img_path, groups)
        self.assertEqual(results[0]["spatial_match"], "click_mode_empty")
        self.assertEqual(results[0]["area_px"], 0)
        self.assertEqual(results[0]["clicks_used"], [])

    def test_multiple_groups_produce_one_result_each(self):
        svc, img_path = self._make_service()
        groups = [
            {"id": 1, "description": "A", "clicks": [{"x": 0.1, "y": 0.1, "label": 1}]},
            {"id": 2, "description": "B", "clicks": [{"x": 0.9, "y": 0.9, "label": 1}]},
        ]
        results = svc.group_segment(img_path, groups)
        self.assertEqual(len(results), 2)
        self.assertEqual(results[0]["creature_id"], 1)
        self.assertEqual(results[1]["creature_id"], 2)


from nibi_model_compare.som_missed_creatures import render_proposed_click_groups_overlay


class RenderProposedClickGroupsTests(unittest.TestCase):
    H, W = 64, 64

    def _frame(self):
        return np.zeros((self.H, self.W, 3), dtype=np.uint8) + 80

    def test_returns_same_shape(self):
        frame = self._frame()
        groups = [
            {"id": 1, "description": "crab", "clicks": [{"x": 0.5, "y": 0.5, "label": 1}]},
        ]
        out = render_proposed_click_groups_overlay(frame, groups)
        self.assertEqual(out.shape, frame.shape)
        self.assertEqual(out.dtype, frame.dtype)

    def test_empty_groups_returns_copy(self):
        frame = self._frame()
        out = render_proposed_click_groups_overlay(frame, [])
        self.assertEqual(out.shape, frame.shape)

    def test_modifies_pixels_for_positive_click(self):
        frame = self._frame()
        groups = [
            {"id": 1, "description": "crab", "clicks": [{"x": 0.5, "y": 0.5, "label": 1}]},
        ]
        out = render_proposed_click_groups_overlay(frame, groups)
        self.assertFalse(np.array_equal(frame, out))

    def test_different_groups_can_coexist(self):
        frame = self._frame()
        groups = [
            {"id": 1, "description": "crab A",
             "clicks": [{"x": 0.2, "y": 0.2, "label": 1}]},
            {"id": 2, "description": "crab B",
             "clicks": [{"x": 0.8, "y": 0.8, "label": 1},
                        {"x": 0.7, "y": 0.9, "label": 0}]},
        ]
        out = render_proposed_click_groups_overlay(frame, groups)
        self.assertEqual(out.shape, frame.shape)
        self.assertFalse(np.array_equal(frame, out))

    def test_with_existing_masks_applies_overlay(self):
        frame = self._frame()
        yy, xx = np.ogrid[:self.H, :self.W]
        mask = ((yy - 10) ** 2 + (xx - 10) ** 2) <= 5 * 5
        groups = [
            {"id": 1, "description": "crab",
             "clicks": [{"x": 0.7, "y": 0.7, "label": 1}]},
        ]
        out = render_proposed_click_groups_overlay(frame, groups, existing_masks=[{"mask": mask}])
        self.assertEqual(out.shape, frame.shape)
        self.assertFalse(np.array_equal(frame, out))


if __name__ == "__main__":
    unittest.main()
