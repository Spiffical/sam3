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


from nibi_model_compare.som_missed_creatures import generate_dense_candidates


class GenerateDenseCandidatesTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        from PIL import Image
        self.img_path = os.path.join(self.tmp.name, "frame.png")
        Image.new("RGB", (64, 64)).save(self.img_path)

    def tearDown(self):
        self.tmp.cleanup()

    def _write_sam_result(self, path, masks_xy_pairs):
        """Helper: write a JSON in the shape call_sam_service emits."""
        import json
        from nibi_model_compare.frame_output_utils import encode_binary_mask_to_rle
        rles = []
        boxes = []
        for cy, cx, r in masks_xy_pairs:
            yy, xx = np.ogrid[:64, :64]
            m = ((yy - cy) ** 2 + (xx - cx) ** 2) <= r * r
            rles.append(encode_binary_mask_to_rle(m))
            ys, xs = np.where(m)
            boxes.append([int(xs.min()), int(ys.min()),
                          int(xs.max() - xs.min() + 1),
                          int(ys.max() - ys.min() + 1)])
        payload = {
            "original_image_path": self.img_path,
            "orig_img_h": 64,
            "orig_img_w": 64,
            "pred_masks": rles,
            "pred_boxes": boxes,
            "pred_scores": [0.9] * len(rles),
        }
        with open(path, "w") as f:
            json.dump(payload, f)
        return path

    def test_calls_sam_per_prompt_and_unions(self):
        calls = []

        def fake_sam(image_path, text_prompt, output_folder_path):
            calls.append(text_prompt)
            n = len(calls)
            out_path = os.path.join(output_folder_path, f"out_{n}.json")
            if text_prompt == "creature":
                return self._write_sam_result(out_path, [(20, 20, 5)])
            if text_prompt == "animal":
                return self._write_sam_result(out_path, [(40, 40, 5)])
            return self._write_sam_result(out_path, [])

        out = generate_dense_candidates(
            image_path=self.img_path,
            broad_prompts=["creature", "animal"],
            output_folder=self.tmp.name,
            _call_sam_service=fake_sam,
        )
        self.assertEqual(calls, ["creature", "animal"])
        self.assertEqual(len(out), 2)
        for cand in out:
            self.assertIn("mask", cand)
            self.assertIn("bbox_xywh", cand)
            self.assertIn("score", cand)

    def test_zero_results_returns_empty(self):
        def fake_sam(image_path, text_prompt, output_folder_path):
            out_path = os.path.join(output_folder_path, f"out_{text_prompt}.json")
            return self._write_sam_result(out_path, [])

        out = generate_dense_candidates(
            image_path=self.img_path,
            broad_prompts=["creature"],
            output_folder=self.tmp.name,
            _call_sam_service=fake_sam,
        )
        self.assertEqual(out, [])

    def test_internal_dedup_by_iou(self):
        # Same prompt produces overlapping masks across calls -> intra-batch dedup
        def fake_sam(image_path, text_prompt, output_folder_path):
            n = len([f for f in os.listdir(output_folder_path)
                     if f.endswith(".json")])
            out_path = os.path.join(output_folder_path, f"out_{n}.json")
            return self._write_sam_result(out_path, [(20, 20, 5)])

        out = generate_dense_candidates(
            image_path=self.img_path,
            broad_prompts=["creature", "animal"],
            output_folder=self.tmp.name,
            _call_sam_service=fake_sam,
            internal_iou_dedup=0.5,
        )
        # Two identical masks -> dedup keeps one
        self.assertEqual(len(out), 1)

    def test_pred_boxes_length_mismatch_drops_prompt_results(self):
        def fake_sam(image_path, text_prompt, output_folder_path):
            import json
            out_path = os.path.join(output_folder_path, f"out_{text_prompt}.json")
            # 2 masks, but only 1 box — corrupted-looking payload
            yy, xx = np.ogrid[:64, :64]
            m1 = ((yy - 20) ** 2 + (xx - 20) ** 2) <= 5 * 5
            m2 = ((yy - 40) ** 2 + (xx - 40) ** 2) <= 5 * 5
            from nibi_model_compare.frame_output_utils import encode_binary_mask_to_rle
            payload = {
                "original_image_path": image_path,
                "orig_img_h": 64,
                "orig_img_w": 64,
                "pred_masks": [encode_binary_mask_to_rle(m1),
                               encode_binary_mask_to_rle(m2)],
                "pred_boxes": [[15, 15, 11, 11]],  # short!
                "pred_scores": [0.9, 0.9],
            }
            with open(out_path, "w") as f:
                json.dump(payload, f)
            return out_path

        out = generate_dense_candidates(
            image_path=self.img_path,
            broad_prompts=["creature"],
            output_folder=self.tmp.name,
            _call_sam_service=fake_sam,
        )
        self.assertEqual(out, [])

    def test_missing_image_dimensions_raises(self):
        def fake_sam(image_path, text_prompt, output_folder_path):
            import json
            out_path = os.path.join(output_folder_path, "broken.json")
            with open(out_path, "w") as f:
                json.dump({
                    "original_image_path": image_path,
                    # NO orig_img_h / orig_img_w
                    "pred_masks": [{"size": [64, 64], "counts": "0"}],
                    "pred_boxes": [[0, 0, 1, 1]],
                    "pred_scores": [0.5],
                }, f)
            return out_path

        with self.assertRaises(ValueError):
            generate_dense_candidates(
                image_path=self.img_path,
                broad_prompts=["creature"],
                output_folder=self.tmp.name,
                _call_sam_service=fake_sam,
            )


from nibi_model_compare.som_missed_creatures import load_system_prompt


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


if __name__ == "__main__":
    unittest.main()
