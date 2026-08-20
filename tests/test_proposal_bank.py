import os
import tempfile
import unittest


class ProposalBankGeometryTests(unittest.TestCase):
    def test_parse_normalized_region(self):
        from sam3.agent.proposal_bank import parse_normalized_region

        self.assertEqual(
            parse_normalized_region("0, 0.1, 0.5, 1"),
            (0.0, 0.1, 0.5, 1.0),
        )
        with self.assertRaises(ValueError):
            parse_normalized_region("0,0,1")
        with self.assertRaises(ValueError):
            parse_normalized_region("0.5,0,0.4,1")

    def test_suppress_region_uses_fraction_of_proposal_box(self):
        from sam3.agent.proposal_bank import suppress_exclusion_regions

        outputs = {
            "pred_boxes": [[0.02, 0.01, 0.10, 0.05], [0.0, 0.08, 0.20, 0.20]],
            "pred_scores": [0.9, 0.8],
            "pred_masks": ["logo", "edge_organism"],
        }
        filtered, removed = suppress_exclusion_regions(
            outputs, [(0.0, 0.0, 0.36, 0.12)], overlap_fraction=0.80
        )
        self.assertEqual(removed, [0])
        self.assertEqual(filtered["pred_masks"], ["edge_organism"])

    def test_single_proposal_union_preserves_source_report(self):
        from sam3.agent.proposal_bank import build_proposal_union

        outputs = {
            "orig_img_h": 10,
            "orig_img_w": 20,
            "pred_boxes": [[0.2, 0.2, 0.2, 0.2]],
            "pred_scores": [0.9],
            "pred_masks": ["placeholder"],
        }
        union, report = build_proposal_union(
            [("sea anemone", outputs)], image_path="/tmp/frame.png"
        )
        self.assertEqual(union["pred_masks"], ["placeholder"])
        self.assertEqual(report["raw_proposal_count"], 1)
        self.assertEqual(report["deduplicated_source_prompts"], ["sea anemone"])

    def test_opt_in_fragment_merge_stitches_one_instance(self):
        import numpy as np
        from pycocotools import mask as mask_utils

        from sam3.agent.proposal_bank import merge_prompt_fragments

        encoded_masks = []
        for y1, y2 in ((2, 8), (7, 14)):
            mask = np.zeros((16, 16), dtype=np.uint8)
            mask[y1:y2, 5:8] = 1
            encoded = mask_utils.encode(np.asfortranarray(mask))
            counts = encoded["counts"]
            encoded_masks.append(
                counts.decode("utf-8") if isinstance(counts, bytes) else counts
            )
        outputs = {
            "orig_img_h": 16,
            "orig_img_w": 16,
            "pred_boxes": [[5 / 16, 2 / 16, 3 / 16, 6 / 16], [5 / 16, 7 / 16, 3 / 16, 7 / 16]],
            "pred_scores": [0.9, 0.8],
            "pred_masks": encoded_masks,
        }
        merged, sources, groups = merge_prompt_fragments(
            outputs,
            ["Walteria", "Walteria"],
            ["Walteria"],
            bbox_iom_threshold=0.10,
        )
        self.assertEqual(groups, [[0, 1]])
        self.assertEqual(sources, ["Walteria"])
        self.assertEqual(len(merged["pred_masks"]), 1)
        decoded = mask_utils.decode(
            {"size": [16, 16], "counts": merged["pred_masks"][0]}
        )
        self.assertEqual(int(decoded.sum()), 36)


class SeededProposalVerificationTests(unittest.TestCase):
    def test_one_request_selects_seeded_mask_without_sam_call(self):
        import numpy as np
        from PIL import Image
        from pycocotools import mask as mask_utils

        from sam3.agent import agent_core

        with tempfile.TemporaryDirectory() as workdir:
            image_path = os.path.join(workdir, "frame.png")
            context_path = os.path.join(workdir, "context.png")
            Image.new("RGB", (32, 32), color=(25, 35, 45)).save(image_path)
            Image.new("RGB", (32, 64), color=(15, 25, 35)).save(context_path)
            mask = np.zeros((32, 32), dtype=np.uint8)
            mask[8:24, 8:24] = 1
            encoded = mask_utils.encode(np.asfortranarray(mask))
            counts = encoded["counts"]
            if isinstance(counts, bytes):
                counts = counts.decode("utf-8")
            seeded = {
                "orig_img_h": 32,
                "orig_img_w": 32,
                "pred_boxes": [[0.25, 0.25, 0.5, 0.5]],
                "pred_scores": [0.95],
                "pred_masks": [counts],
            }
            calls = []

            def fake_send(messages):
                calls.append(messages)
                return '<tool>{"name":"select_masks_and_return","parameters":{"final_answer_masks":[1]}}</tool>'

            def forbidden_sam_call(**_kwargs):
                raise AssertionError("verification-only mode must not call SAM3")

            _, final_outputs, _ = agent_core.agent_inference(
                img_path=image_path,
                initial_text_prompt="all visible marine life",
                send_generate_request=fake_send,
                call_sam_service=forbidden_sam_call,
                output_dir=os.path.join(workdir, "out"),
                initial_outputs=seeded,
                initial_text_prompts=["sea anemone"],
                verification_only=True,
                verification_context_images=[context_path],
            )
            self.assertEqual(len(calls), 1)
            image_parts = [
                part
                for part in calls[0][1]["content"]
                if part.get("type") == "image"
            ]
            self.assertEqual(len(image_parts), 3)
            self.assertEqual(len(final_outputs["pred_masks"]), 1)

    def test_drop_decision_is_finalized_without_second_request(self):
        import numpy as np
        from PIL import Image
        from pycocotools import mask as mask_utils

        from sam3.agent import agent_core

        with tempfile.TemporaryDirectory() as workdir:
            image_path = os.path.join(workdir, "frame.png")
            Image.new("RGB", (16, 16), color=(0, 0, 0)).save(image_path)
            masks = []
            for start in (1, 9):
                mask = np.zeros((16, 16), dtype=np.uint8)
                mask[start : start + 4, start : start + 4] = 1
                encoded = mask_utils.encode(np.asfortranarray(mask))
                counts = encoded["counts"]
                masks.append(counts.decode("utf-8") if isinstance(counts, bytes) else counts)
            seeded = {
                "orig_img_h": 16,
                "orig_img_w": 16,
                "pred_boxes": [[0.06, 0.06, 0.25, 0.25], [0.56, 0.56, 0.25, 0.25]],
                "pred_scores": [0.9, 0.8],
                "pred_masks": masks,
            }
            calls = {"count": 0}

            def fake_send(_messages):
                calls["count"] += 1
                return '<tool>{"name":"drop_masks","parameters":{"mask_indices_to_drop":[2]}}</tool>'

            _, final_outputs, _ = agent_core.agent_inference(
                img_path=image_path,
                initial_text_prompt="all visible marine life",
                send_generate_request=fake_send,
                output_dir=os.path.join(workdir, "out"),
                initial_outputs=seeded,
                verification_only=True,
            )
            self.assertEqual(calls["count"], 1)
            self.assertEqual(len(final_outputs["pred_masks"]), 1)


if __name__ == "__main__":
    unittest.main()
