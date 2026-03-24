import unittest

from scripts import reassign_frame_ids_mllm as mod


class OutlierCleanupHelpersTests(unittest.TestCase):
    def setUp(self) -> None:
        mod.ensure_runtime_deps()
        self.frame_h = 8
        self.frame_w = 8

    def _frame_row_from_masks(self, masks: list) -> dict:
        return {
            "frame_index": 0,
            "out_obj_ids": [idx + 1 for idx in range(len(masks))],
            "out_probs": [1.0 for _ in masks],
            "out_tracker_probs": [],
            "out_boxes_xywh": [None for _ in masks],
            "out_binary_masks_rle": [mod.encode_binary_mask_to_rle(mask) for mask in masks],
        }

    def test_remove_local_id_from_frame_row_removes_only_target_mask(self) -> None:
        import numpy as np

        mask_a = np.zeros((self.frame_h, self.frame_w), dtype=np.uint8)
        mask_b = np.zeros((self.frame_h, self.frame_w), dtype=np.uint8)
        mask_a[1:3, 1:3] = 1
        mask_b[4:6, 4:6] = 1
        frame_row = self._frame_row_from_masks([mask_a, mask_b])

        removed = mod.remove_local_id_from_frame_row(frame_row, local_id=2)

        self.assertTrue(removed)
        self.assertEqual(frame_row["out_obj_ids"], [1])
        self.assertEqual(len(frame_row["out_binary_masks_rle"]), 1)
        remaining_items = mod.decode_frame_row_masks(frame_row, self.frame_h, self.frame_w)
        self.assertEqual([item["local_id"] for item in remaining_items], [1])

    def test_mask_match_support_score_prefers_matching_reference(self) -> None:
        import numpy as np

        mask = np.zeros((self.frame_h, self.frame_w), dtype=np.uint8)
        shifted = np.zeros((self.frame_h, self.frame_w), dtype=np.uint8)
        far = np.zeros((self.frame_h, self.frame_w), dtype=np.uint8)
        mask[2:5, 2:5] = 1
        shifted[2:5, 3:6] = 1
        far[0:2, 0:2] = 1

        row = self._frame_row_from_masks([mask, shifted, far])
        items = mod.decode_frame_row_masks(row, self.frame_h, self.frame_w)

        close_score = mod.mask_match_support_score(items[0], items[1])
        far_score = mod.mask_match_support_score(items[0], items[2])

        self.assertGreater(close_score, far_score)

    def test_geometry_assessment_flags_partial_thin_mask(self) -> None:
        import numpy as np

        ref_mask = np.zeros((self.frame_h, self.frame_w), dtype=np.uint8)
        cand_mask = np.zeros((self.frame_h, self.frame_w), dtype=np.uint8)
        ref_mask[1:7, 3:5] = 1
        cand_mask[1:4, 3:5] = 1

        row = self._frame_row_from_masks([ref_mask, cand_mask])
        items = mod.decode_frame_row_masks(row, self.frame_h, self.frame_w)
        assessment = mod.assess_gap_fill_candidate_geometry(
            candidate_item=items[1],
            reference_item=items[0],
            hint_bbox_xyxy=items[0]["bbox_xyxy"],
        )

        self.assertEqual(assessment["status"], "partial")

    def test_build_point_prompt_points_keeps_multiple_positive_clicks(self) -> None:
        points = mod.build_point_prompt_points(
            positive_points=[(2, 2), (5, 5)],
            hint_bbox_xyxy=(1, 1, 6, 6),
            frame_w=self.frame_w,
            frame_h=self.frame_h,
        )

        positives = [(x, y) for x, y, label in points if int(label) == 1]
        self.assertIn((2, 2), positives)
        self.assertIn((5, 5), positives)

    def test_find_existing_issue_match_detects_already_present_mask(self) -> None:
        import numpy as np

        existing_mask = np.zeros((self.frame_h, self.frame_w), dtype=np.uint8)
        ref_mask = np.zeros((self.frame_h, self.frame_w), dtype=np.uint8)
        existing_mask[1:6, 3:5] = 1
        ref_mask[1:6, 3:5] = 1

        target_row = self._frame_row_from_masks([existing_mask])
        target_row["frame_index"] = 10
        ref_row = self._frame_row_from_masks([ref_mask])
        ref_row["frame_index"] = 9

        target_items = mod.decode_frame_row_masks(target_row, self.frame_h, self.frame_w)
        ref_items = mod.decode_frame_row_masks(ref_row, self.frame_h, self.frame_w)
        mask_items_by_frame = {
            10: target_items,
            9: ref_items,
        }

        match = mod.find_existing_issue_match(
            target_frame_index=10,
            reference_masks=[{"frame_index": 9, "local_id": 1}],
            mask_items_by_frame=mask_items_by_frame,
            hint_bbox_xyxy=target_items[0]["bbox_xyxy"],
            min_support_score=0.9,
        )

        self.assertIsNotNone(match)
        self.assertEqual(match["local_id"], 1)
        self.assertGreaterEqual(match["support_score"], 0.9)

    def test_find_existing_issue_match_uses_candidate_duplicate_overlap(self) -> None:
        import numpy as np

        existing_mask = np.zeros((self.frame_h, self.frame_w), dtype=np.uint8)
        ref_mask = np.zeros((self.frame_h, self.frame_w), dtype=np.uint8)
        existing_mask[1:6, 3:5] = 1
        ref_mask[1:6, 3:5] = 1

        target_row = self._frame_row_from_masks([existing_mask])
        target_row["frame_index"] = 10
        ref_row = self._frame_row_from_masks([ref_mask])
        ref_row["frame_index"] = 9

        target_items = mod.decode_frame_row_masks(target_row, self.frame_h, self.frame_w)
        ref_items = mod.decode_frame_row_masks(ref_row, self.frame_h, self.frame_w)
        candidate_item = dict(target_items[0])
        mask_items_by_frame = {
            10: target_items,
            9: ref_items,
        }

        match = mod.find_existing_issue_match(
            target_frame_index=10,
            reference_masks=[{"frame_index": 9, "local_id": 1}],
            mask_items_by_frame=mask_items_by_frame,
            hint_bbox_xyxy=target_items[0]["bbox_xyxy"],
            candidate_item=candidate_item,
            min_support_score=0.9,
        )

        self.assertIsNotNone(match)
        self.assertEqual(match["local_id"], 1)
        self.assertGreaterEqual(match["duplicate_iou"], 0.99)


if __name__ == "__main__":
    unittest.main()
