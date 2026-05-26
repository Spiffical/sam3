import types
import unittest

from scripts.run_sam3_agent_every_frame_video import _summarize_llm_backend


class SummarizeLlmBackendTests(unittest.TestCase):
    def test_reports_claude_when_provider_is_claude(self):
        args = types.SimpleNamespace(
            llm_provider="claude",
            claude_model="claude-sonnet-4-6",
            model="Qwen/Qwen3.5-27B",
            server_url="http://127.0.0.1:8000/v1",
        )
        result = _summarize_llm_backend(args)
        self.assertEqual(result["llm_provider"], "claude")
        self.assertEqual(result["model"], "claude-sonnet-4-6")
        self.assertEqual(result["server_url"], "")

    def test_reports_openai_when_provider_is_openai(self):
        args = types.SimpleNamespace(
            llm_provider="openai",
            claude_model="claude-sonnet-4-6",
            model="Qwen/Qwen3.5-27B",
            server_url="http://127.0.0.1:8000/v1",
        )
        result = _summarize_llm_backend(args)
        self.assertEqual(result["llm_provider"], "openai")
        self.assertEqual(result["model"], "Qwen/Qwen3.5-27B")
        self.assertEqual(result["server_url"], "http://127.0.0.1:8000/v1")


if __name__ == "__main__":
    unittest.main()
