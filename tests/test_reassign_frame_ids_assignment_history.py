import unittest

from scripts import reassign_frame_ids_mllm as mod


class AssignmentHistoryReuseTests(unittest.TestCase):
    def setUp(self) -> None:
        mod.ensure_runtime_deps()
        self.frame_h = 8
        self.frame_w = 8

    def _frame_row_from_mask(self, *, frame_index: int, local_id: int, mask, box_xywh) -> dict:
        return {
            "frame_index": int(frame_index),
            "out_obj_ids": [int(local_id)],
            "out_probs": [1.0],
            "out_tracker_probs": [],
            "out_boxes_xywh": [list(box_xywh)],
            "out_binary_masks_rle": [mod.encode_binary_mask_to_rle(mask)],
        }

    def test_new_label_reuses_recent_matching_global_id(self) -> None:
        import numpy as np

        mask = np.zeros((self.frame_h, self.frame_w), dtype=np.uint8)
        mask[2:5, 2:5] = 1

        prior_row = self._frame_row_from_mask(
            frame_index=9,
            local_id=1,
            mask=mask,
            box_xywh=[2 / self.frame_w, 2 / self.frame_h, 3 / self.frame_w, 3 / self.frame_h],
        )
        current_row = self._frame_row_from_mask(
            frame_index=10,
            local_id=1,
            mask=mask,
            box_xywh=[2 / self.frame_w, 2 / self.frame_h, 3 / self.frame_w, 3 / self.frame_h],
        )

        mask_items_by_frame = {
            10: mod.decode_frame_row_masks(current_row, self.frame_h, self.frame_w),
        }
        working_rows = {
            9: prior_row,
            10: current_row,
        }
        existing_assignments = {
            9: {1: 7},
        }
        parsed_assignments = {
            10: {1: "new_a"},
        }

        resolved, dropped, ignored, next_gid = mod.apply_window_assignments(
            window_frame_indices=[10],
            mask_items_by_frame=mask_items_by_frame,
            working_frame_rows_by_index=working_rows,
            frame_h=self.frame_h,
            frame_w=self.frame_w,
            existing_frame_assignments=existing_assignments,
            parsed_assignments=parsed_assignments,
            next_global_id=8,
            allow_drop_assignments=False,
            assignment_history_frames=4,
            assignment_heuristic_min_score=0.85,
        )

        self.assertEqual(resolved[10][1], 7)
        self.assertEqual(dropped, {})
        self.assertEqual(ignored, {})
        self.assertEqual(next_gid, 8)


if __name__ == "__main__":
    unittest.main()
