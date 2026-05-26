import unittest

from sam3.agent.agent_core import (
    _build_invalid_tool_state_redirect_message,
    _last_user_has_image_context,
)


class LastUserHasImageContextTests(unittest.TestCase):
    def test_true_when_image_at_index_0(self):
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": "/tmp/raw.png"},
                    {"type": "text", "text": "raw input"},
                ],
            },
        ]
        self.assertTrue(_last_user_has_image_context(messages))

    def test_true_when_image_at_index_1(self):
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "rendered masks below"},
                    {"type": "image", "image": "/tmp/masks.png"},
                ],
            },
        ]
        self.assertTrue(_last_user_has_image_context(messages))

    def test_false_when_text_only(self):
        messages = [
            {
                "role": "user",
                "content": [{"type": "text", "text": "no masks generated"}],
            },
        ]
        self.assertFalse(_last_user_has_image_context(messages))

    def test_false_when_last_is_assistant(self):
        messages = [
            {
                "role": "user",
                "content": [{"type": "image", "image": "/tmp/raw.png"}],
            },
            {"role": "assistant", "content": [{"type": "text", "text": "ok"}]},
        ]
        self.assertFalse(_last_user_has_image_context(messages))

    def test_false_when_content_not_list(self):
        messages = [{"role": "user", "content": "plain text"}]
        self.assertFalse(_last_user_has_image_context(messages))

    def test_false_when_messages_empty(self):
        self.assertFalse(_last_user_has_image_context([]))

    def test_false_when_content_missing(self):
        messages = [{"role": "user"}]
        self.assertFalse(_last_user_has_image_context(messages))


class InvalidToolStateRedirectMessageTests(unittest.TestCase):
    def test_mentions_tool_name_initial_prompt_and_valid_alternatives(self):
        msg = _build_invalid_tool_state_redirect_message(
            tool_name="examine_each_mask",
            initial_text_prompt="small creatures",
        )
        self.assertIn("examine_each_mask", msg)
        self.assertIn("small creatures", msg)
        self.assertIn("segment_phrase", msg)
        self.assertIn("report_no_mask", msg)


class AgentInferenceInvalidToolStateRecoveryTests(unittest.TestCase):
    """Verify the loop recovers when the model invokes a mask-state tool
    after a text-only user turn (e.g. when segment_phrase produced 0 masks).
    Pre-fix, this raised IndexError at agent_core.py:892 (examine_each_mask)
    or :818 (drop_masks).
    """

    def _run(self, invalid_tool_response):
        import json
        import os
        import tempfile

        from PIL import Image

        from sam3.agent import agent_core

        with tempfile.TemporaryDirectory() as workdir:
            img_path = os.path.join(workdir, "frame.png")
            Image.new("RGB", (64, 64), color=(0, 0, 0)).save(img_path)

            scripted_responses = iter(
                [
                    '<tool>{"name":"segment_phrase","parameters":{"text_prompt":"creature"}}</tool>',
                    invalid_tool_response,
                    '<tool>{"name":"report_no_mask","parameters":{}}</tool>',
                ]
            )

            def fake_send_generate_request(_messages):
                return next(scripted_responses)

            counter = {"n": 0}

            def fake_call_sam_service(image_path, text_prompt, output_folder_path):
                counter["n"] += 1
                fn = os.path.join(output_folder_path, f"sam_call_{counter['n']}.json")
                with open(fn, "w", encoding="utf-8") as f:
                    json.dump(
                        {
                            "original_image_path": image_path,
                            "orig_img_h": 64,
                            "orig_img_w": 64,
                            "pred_boxes": [],
                            "pred_scores": [],
                            "pred_masks": [],
                        },
                        f,
                    )
                return fn

            output_dir = os.path.join(workdir, "agent_out")
            os.makedirs(output_dir, exist_ok=True)

            _messages, final_outputs, _rendered = agent_core.agent_inference(
                img_path=img_path,
                initial_text_prompt="small creatures",
                debug=False,
                send_generate_request=fake_send_generate_request,
                call_sam_service=fake_call_sam_service,
                max_generations=10,
                output_dir=output_dir,
            )
            return final_outputs

    def test_recovers_from_examine_each_mask_after_zero_mask_segment(self):
        final_outputs = self._run(
            '<tool>{"name":"examine_each_mask","parameters":{}}</tool>'
        )
        self.assertEqual(final_outputs["pred_masks"], [])
        self.assertEqual(final_outputs["pred_boxes"], [])

    def test_recovers_from_drop_masks_after_zero_mask_segment(self):
        final_outputs = self._run(
            '<tool>{"name":"drop_masks","parameters":{"mask_indices_to_drop":[1]}}</tool>'
        )
        self.assertEqual(final_outputs["pred_masks"], [])
        self.assertEqual(final_outputs["pred_boxes"], [])


class DebugHistoryPersistsAfterSuccessTests(unittest.TestCase):
    """Successful frames should retain debug_history.json so downstream
    analyzers (scripts/analyze_agent_run.py) can count tool calls. Pre-fix
    the select_masks_and_return branch called cleanup_debug_files and
    erased the data.
    """

    def test_debug_history_persists_after_select_masks_and_return(self):
        import json
        import os
        import tempfile

        from PIL import Image

        from sam3.agent import agent_core

        with tempfile.TemporaryDirectory() as workdir:
            img_path = os.path.join(workdir, "frame.png")
            Image.new("RGB", (64, 64), color=(0, 0, 0)).save(img_path)

            scripted_responses = iter(
                [
                    '<tool>{"name":"segment_phrase","parameters":{"text_prompt":"creature"}}</tool>',
                    '<tool>{"name":"select_masks_and_return","parameters":{"final_answer_masks":[]}}</tool>',
                ]
            )

            def fake_send_generate_request(_messages):
                return next(scripted_responses)

            def fake_call_sam_service(image_path, text_prompt, output_folder_path):
                fn = os.path.join(output_folder_path, "sam_call.json")
                with open(fn, "w", encoding="utf-8") as f:
                    json.dump(
                        {
                            "original_image_path": image_path,
                            "orig_img_h": 64,
                            "orig_img_w": 64,
                            "pred_boxes": [],
                            "pred_scores": [],
                            "pred_masks": [],
                        },
                        f,
                    )
                return fn

            output_dir = os.path.join(workdir, "agent_out")
            os.makedirs(output_dir, exist_ok=True)

            agent_core.agent_inference(
                img_path=img_path,
                initial_text_prompt="small creatures",
                debug=True,
                send_generate_request=fake_send_generate_request,
                call_sam_service=fake_call_sam_service,
                max_generations=10,
                output_dir=output_dir,
            )

            debug_root = os.path.join(output_dir, "agent_debug_out")
            self.assertTrue(os.path.isdir(debug_root))
            # Find debug_history.json under the per-image debug folder.
            history_paths = []
            for sub in os.listdir(debug_root):
                cand = os.path.join(debug_root, sub, "debug_history.json")
                if os.path.exists(cand):
                    history_paths.append(cand)
            self.assertEqual(
                len(history_paths), 1,
                f"Expected exactly one debug_history.json under {debug_root}; "
                f"found {history_paths}",
            )


class GracefulCapTermsTests(unittest.TestCase):
    """When max_generations is exhausted, the agent should gracefully
    terminate via the configured fallback policy rather than always raising
    ValueError. For an "auto" policy (default for broad underwater queries),
    the cap should resolve to report_no_mask (no masks) or select_all (masks
    accumulated). For "strict_fail" the cap should still raise.
    """

    def _drive(self, *, prompt_profile, initial_prompt, max_generations, env=None):
        import json
        import os
        import tempfile

        from PIL import Image

        from sam3.agent import agent_core

        if env is not None:
            old_env = {k: os.environ.get(k) for k in env}
            for k, v in env.items():
                if v is None:
                    os.environ.pop(k, None)
                else:
                    os.environ[k] = v
        try:
            with tempfile.TemporaryDirectory() as workdir:
                img_path = os.path.join(workdir, "frame.png")
                Image.new("RGB", (64, 64), color=(0, 0, 0)).save(img_path)

                scripted_responses = iter(
                    [
                        '<tool>{"name":"segment_phrase","parameters":{"text_prompt":"c1"}}</tool>',
                        '<tool>{"name":"segment_phrase","parameters":{"text_prompt":"c2"}}</tool>',
                        '<tool>{"name":"segment_phrase","parameters":{"text_prompt":"c3"}}</tool>',
                    ]
                )

                def fake_send_generate_request(_messages):
                    return next(scripted_responses)

                counter = {"n": 0}

                def fake_call_sam_service(image_path, text_prompt, output_folder_path):
                    counter["n"] += 1
                    fn = os.path.join(output_folder_path, f"sam_{counter['n']}.json")
                    with open(fn, "w", encoding="utf-8") as f:
                        json.dump(
                            {
                                "original_image_path": image_path,
                                "orig_img_h": 64,
                                "orig_img_w": 64,
                                "pred_boxes": [],
                                "pred_scores": [],
                                "pred_masks": [],
                            },
                            f,
                        )
                    return fn

                # prompt_profile is loaded from SAM3_AGENT_PROMPT_PROFILE
                # inside _load_system_prompts; honour the same hook here.
                os.environ["SAM3_AGENT_PROMPT_PROFILE"] = prompt_profile

                output_dir = os.path.join(workdir, "agent_out")
                os.makedirs(output_dir, exist_ok=True)
                return agent_core.agent_inference(
                    img_path=img_path,
                    initial_text_prompt=initial_prompt,
                    debug=False,
                    send_generate_request=fake_send_generate_request,
                    call_sam_service=fake_call_sam_service,
                    max_generations=max_generations,
                    output_dir=output_dir,
                )
        finally:
            if env is not None:
                for k, v in old_env.items():
                    if v is None:
                        os.environ.pop(k, None)
                    else:
                        os.environ[k] = v

    def test_underwater_broad_prompt_terminates_at_cap_without_raising(self):
        _msgs, final_outputs, _rendered = self._drive(
            prompt_profile="underwater",
            initial_prompt="small creatures",
            max_generations=1,
        )
        self.assertEqual(final_outputs["pred_masks"], [])
        self.assertEqual(final_outputs["pred_boxes"], [])

    def test_strict_policy_still_raises_at_cap(self):
        with self.assertRaises(ValueError) as ctx:
            self._drive(
                prompt_profile="underwater",
                initial_prompt="small creatures",
                max_generations=1,
                env={"SAM3_TOOL_CALL_FALLBACK_POLICY": "strict_fail"},
            )
        self.assertIn("maximum number", str(ctx.exception).lower())


if __name__ == "__main__":
    unittest.main()
