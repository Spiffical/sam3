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


import numpy as np

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


if __name__ == "__main__":
    unittest.main()
