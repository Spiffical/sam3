import unittest

from scripts.reassign_frame_ids_mllm import unwrap_backend_outputs


class UnwrapBackendOutputsTests(unittest.TestCase):
    def test_accepts_flat_backend_outputs(self) -> None:
        response = {
            "out_obj_ids": [1],
            "out_probs": [0.9],
            "out_boxes_xywh": [[10, 20, 30, 40]],
        }

        self.assertEqual(unwrap_backend_outputs(response), response)

    def test_accepts_nested_backend_outputs(self) -> None:
        response = {
            "frame_index": 12,
            "outputs": {
                "out_obj_ids": [1],
                "out_probs": [0.9],
            },
        }

        self.assertEqual(
            unwrap_backend_outputs(response),
            {
                "out_obj_ids": [1],
                "out_probs": [0.9],
            },
        )

    def test_non_dict_defaults_to_empty(self) -> None:
        self.assertEqual(unwrap_backend_outputs(None), {})


if __name__ == "__main__":
    unittest.main()
