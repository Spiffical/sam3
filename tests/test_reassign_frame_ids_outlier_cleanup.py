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


if __name__ == "__main__":
    unittest.main()
