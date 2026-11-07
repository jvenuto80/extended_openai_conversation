"""Standalone tests for responses_api conversion helpers (no Home Assistant needed).

Run with: python3 tests/test_responses_conversion.py
"""

import importlib.util
from pathlib import Path
from types import SimpleNamespace
import unittest

_MODULE_PATH = (
    Path(__file__).resolve().parents[1]
    / "custom_components"
    / "extended_openai_conversation"
    / "responses_api.py"
)
_spec = importlib.util.spec_from_file_location("responses_api", _MODULE_PATH)
responses_api = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(responses_api)


class TokenParamTest(unittest.TestCase):
    def test_model_prefixes(self):
        for model in ("gpt-6.1", "GPT-6", "gpt-5-mini", "gpt-4o", "o3-mini", "azure-o4"):
            self.assertTrue(responses_api.use_max_completion_tokens(model), model)
        for model in ("gpt-4.1-mini", "gpt-3.5-turbo", "my-deployment", "", None):
            self.assertFalse(responses_api.use_max_completion_tokens(model), model)

    def test_force(self):
        self.assertEqual(
            responses_api.get_token_kwargs("my-deployment", 100, True),
            {"max_completion_tokens": 100},
        )
        self.assertEqual(
            responses_api.get_token_kwargs("gpt-4.1", 100), {"max_tokens": 100}
        )

    def test_reasoning_effort(self):
        for value in (None, "", "default"):
            self.assertIsNone(responses_api.normalize_reasoning_effort(value))
        self.assertEqual(responses_api.normalize_reasoning_effort("xhigh"), "xhigh")


class ToolConversionTest(unittest.TestCase):
    def test_chat_tool_and_bare_spec(self):
        params = {"type": "object", "properties": {"x": {"type": "string"}}}
        tools = [
            {
                "type": "function",
                "function": {"name": "a", "description": "A", "parameters": params},
            },
            {"name": "b", "description": "B", "strict": True, "parameters": params},
            {"name": "c"},
        ]
        self.assertEqual(
            responses_api.to_responses_tools(tools),
            [
                {
                    "type": "function",
                    "name": "a",
                    "description": "A",
                    "parameters": params,
                    "strict": False,
                },
                {
                    "type": "function",
                    "name": "b",
                    "description": "B",
                    "strict": True,
                    "parameters": params,
                },
                {
                    "type": "function",
                    "name": "c",
                    "parameters": {"type": "object", "properties": {}},
                    "strict": False,
                },
            ],
        )
        self.assertNotIn("type", tools[1])


class MessageConversionTest(unittest.TestCase):
    def test_full_history(self):
        messages = [
            {"role": "system", "content": "You are HA."},
            {"role": "user", "content": "Turn on light", "name": "user1"},
            {
                "role": "assistant",
                "tool_calls": [
                    {
                        "id": "call_1",
                        "type": "function",
                        "function": {"name": "execute_services", "arguments": "{}"},
                    }
                ],
            },
            {
                "tool_call_id": "call_1",
                "role": "tool",
                "name": "execute_services",
                "content": "[{'success': True}]",
            },
            {"role": "assistant", "content": "Done."},
            {"role": "user", "content": "Thanks"},
        ]
        instructions, items = responses_api.to_responses_input(messages)
        self.assertEqual(instructions, "You are HA.")
        self.assertEqual(
            items,
            [
                {"role": "user", "content": "Turn on light"},
                {
                    "type": "function_call",
                    "call_id": "call_1",
                    "name": "execute_services",
                    "arguments": "{}",
                },
                {
                    "type": "function_call_output",
                    "call_id": "call_1",
                    "output": "[{'success': True}]",
                },
                {"role": "assistant", "content": "Done."},
                {"role": "user", "content": "Thanks"},
            ],
        )

    def test_legacy_function_and_multipart(self):
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "What is this?"},
                    {"type": "image_url", "image_url": {"url": "https://x/y.png"}},
                ],
            },
            {
                "role": "assistant",
                "content": None,
                "function_call": {"name": "f", "arguments": '{"a": 1}'},
            },
            {"role": "function", "name": "f", "content": 42},
        ]
        instructions, items = responses_api.to_responses_input(messages)
        self.assertIsNone(instructions)
        self.assertEqual(
            items[0]["content"],
            [
                {"type": "input_text", "text": "What is this?"},
                {"type": "input_image", "image_url": "https://x/y.png"},
            ],
        )
        self.assertEqual(items[1]["type"], "function_call")
        self.assertEqual(items[2]["type"], "function_call_output")
        self.assertEqual(items[1]["call_id"], items[2]["call_id"])
        self.assertEqual(items[2]["output"], "42")


class ResponseParsingTest(unittest.TestCase):
    def test_function_calls_round_trip(self):
        response = SimpleNamespace(
            status="completed",
            incomplete_details=None,
            output_text="",
            output=[
                SimpleNamespace(type="reasoning", summary=[]),
                SimpleNamespace(
                    type="function_call",
                    call_id="call_9",
                    name="execute_services",
                    arguments='{"list": []}',
                ),
            ],
        )
        calls = responses_api.get_function_calls(response)
        self.assertEqual(
            calls,
            [{"call_id": "call_9", "name": "execute_services", "arguments": '{"list": []}'}],
        )
        message = responses_api.to_assistant_message("", calls)
        self.assertEqual(
            message,
            {
                "role": "assistant",
                "tool_calls": [
                    {
                        "id": "call_9",
                        "type": "function",
                        "function": {
                            "name": "execute_services",
                            "arguments": '{"list": []}',
                        },
                    }
                ],
            },
        )
        _, items = responses_api.to_responses_input([message])
        self.assertEqual(items[0]["call_id"], "call_9")
        self.assertFalse(responses_api.is_incomplete_max_tokens(response))

    def test_output_text_fallback_and_incomplete(self):
        response = {
            "status": "incomplete",
            "incomplete_details": {"reason": "max_output_tokens"},
            "output": [
                {"type": "message", "content": [{"type": "output_text", "text": "Hi"}]}
            ],
        }
        self.assertEqual(responses_api.get_output_text(response), "Hi")
        self.assertTrue(responses_api.is_incomplete_max_tokens(response))
        response["incomplete_details"] = {"reason": "content_filter"}
        self.assertFalse(responses_api.is_incomplete_max_tokens(response))


if __name__ == "__main__":
    unittest.main()
