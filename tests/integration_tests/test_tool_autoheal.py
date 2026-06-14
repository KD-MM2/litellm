"""Tests for tool_autoheal module — Unsloth parsers + Harmony support."""
import json
import pytest
from litellm.integrations.tool_autoheal import (
    parse_tool_calls_from_text,
    strip_harmony_tokens,
    strip_tool_markup,
)


class TestStripHarmonyTokens:
    def test_strip_channel_message(self):
        text = '<|channel|>commentary<|message|>Hello world'
        result = strip_harmony_tokens(text)
        assert result == 'Hello world'

    def test_strip_all_tags(self):
        # Harmony format: <|start|>role + <|channel|>channel<|message|>content<|end|>
        text = '<|start|>assistant<|channel|>analysis<|message|>{"answer": 42}<|end|>'
        result = strip_harmony_tokens(text)
        # <|start|> tag is gone but "assistant" role label stays (it's content, not a token)
        # <|channel|>analysis<|message|> gets stripped by HARMONY_CHANNEL_RE
        # <|end|> gets stripped
        assert '<|start|>' not in result
        assert '<|channel|>' not in result
        assert '<|message|>' not in result
        assert '<|end|>' not in result
        assert '{"answer": 42}' in result

    def test_preserve_normal_text(self):
        text = 'Normal text without tokens'
        result = strip_harmony_tokens(text)
        assert result == 'Normal text without tokens'

    def test_empty_string(self):
        assert strip_harmony_tokens('') == ''
        assert strip_harmony_tokens(None) is None


class TestStripToolMarkup:
    def test_strip_tool_call_closed(self):
        text = 'Some text {} more text'
        result = strip_tool_markup(text)
        assert 'tool_call' not in result
        assert 'Some text' in result
        assert 'more text' in result

    def test_strip_tool_call_unclosed_final_mode(self):
        text = 'Before  trailing'
        result = strip_tool_markup(text, final=True)
        assert 'tool_call' not in result
        assert 'Before' in result

    def test_strip_function_tag(self):
        text = '<function=web_search><parameter=q>hello</parameter></function>'
        result = strip_tool_markup(text)
        assert 'function=' not in result


class TestParseToolCalls:
    def test_parse_hermes_json_format(self):
        content = '{"name":"web_search","arguments":{"query":"bitcoin price"}}'
        result = parse_tool_calls_from_text(content)
        assert len(result) == 1
        assert result[0]["function"]["name"] == "web_search"
        assert json.loads(result[0]["function"]["arguments"]) == {"query": "bitcoin price"}

    def test_parse_functionary_xml_format(self):
        content = '<function=web_search><parameter=query>bitcoin price</parameter></function>'
        result = parse_tool_calls_from_text(content)
        assert len(result) == 1
        assert result[0]["function"]["name"] == "web_search"
        assert json.loads(result[0]["function"]["arguments"]) == {"query": "bitcoin price"}

    def test_parse_harmony_format(self):
        content = '<|channel|>commentary<|message|>{"name":"web_search","arguments":{"query":"test"}}'
        result = parse_tool_calls_from_text(content)
        assert len(result) == 1
        assert result[0]["function"]["name"] == "web_search"

    def test_parse_bare_json_fallback(self):
        content = '{"name":"get_weather","arguments":{"city":"Hanoi"}}'
        result = parse_tool_calls_from_text(content)
        assert len(result) == 1
        assert result[0]["function"]["name"] == "get_weather"

    def test_no_tool_call_in_normal_text(self):
        content = 'The capital of France is Paris.'
        result = parse_tool_calls_from_text(content)
        assert len(result) == 0

    def test_multiple_tool_calls(self):
        content = (
            '{"name":"search","arguments":{"q":"A"}}'
            '{"name":"search","arguments":{"q":"B"}}'
        )
        result = parse_tool_calls_from_text(content)
        assert len(result) == 2

    def test_partial_json_no_closing_tag(self):
        """Models often omit closing tags — parser should still work."""
        content = '{"name":"test","arguments":{}}'
        result = parse_tool_calls_from_text(content)
        assert len(result) == 1
        assert result[0]["function"]["name"] == "test"

    def test_markup_cleaning_preserves_json(self):
        """Markup stripping must not corrupt valid JSON inside tool calls."""
        content = 'before {"name":"test","arguments":{}} after'
        result = parse_tool_calls_from_text(content)
        assert len(result) == 1
        assert result[0]["function"]["name"] == "test"

    def test_deeply_nested_arguments_with_wrapper(self):
        """Deeply nested arguments work with tool_call wrapper (balanced-brace parser)."""
        content = '<tool_call>{"name":"complex_tool","arguments":{"nested":{"deep":{"key":"value"}}}}</tool_call>'
        result = parse_tool_calls_from_text(content)
        assert len(result) == 1
        args = json.loads(result[0]["function"]["arguments"])
        assert args["nested"]["deep"]["key"] == "value"

    def test_arguments_as_string_with_wrapper(self):
        """Arguments-as-string works with tool_call wrapper (balanced-brace parser)."""
        content = '<tool_call>{"name":"search","arguments":"{\\"query\\":\\"test\\"}"}</tool_call>'
        result = parse_tool_calls_from_text(content)
        assert len(result) == 1
        assert result[0]["function"]["name"] == "search"

    def test_multiple_parameters_functionary(self):
        content = '<function=search><parameter=q>hello</parameter><parameter=lang>en</parameter></function>'
        result = parse_tool_calls_from_text(content)
        assert len(result) == 1
        args = json.loads(result[0]["function"]["arguments"])
        assert args["q"] == "hello"
        assert args["lang"] == "en"


class TestToolAutoHealCustomLogger:
    @pytest.mark.asyncio
    async def test_hook_heals_harmony_response(self):
        from litellm.integrations.tool_autoheal import ToolAutoHeal
        from unittest.mock import MagicMock

        healer = ToolAutoHeal()

        choice = MagicMock()
        msg = MagicMock()
        msg.content = '<|channel|>commentary<|message|>{"name":"web_search","arguments":{"query":"test"}}'
        msg.tool_calls = None
        choice.message = msg

        response = MagicMock()
        response.choices = [choice]

        result = await healer.async_post_call_success_hook(
            data={}, user_api_key_dict=MagicMock(), response=response
        )

        assert len(result.choices[0].message.tool_calls) == 1
        assert result.choices[0].message.tool_calls[0]["function"]["name"] == "web_search"
        assert "<|channel|>" not in (result.choices[0].message.content or "")

    @pytest.mark.asyncio
    async def test_hook_passes_through_normal_response(self):
        from litellm.integrations.tool_autoheal import ToolAutoHeal
        from unittest.mock import MagicMock

        healer = ToolAutoHeal()

        choice = MagicMock()
        msg = MagicMock()
        msg.content = "Hello, I'm an AI assistant."
        msg.tool_calls = None
        choice.message = msg

        response = MagicMock()
        response.choices = [choice]

        result = await healer.async_post_call_success_hook(
            data={}, user_api_key_dict=MagicMock(), response=response
        )

        assert result.choices[0].message.content == "Hello, I'm an AI assistant."
        assert result.choices[0].message.tool_calls is None

    @pytest.mark.asyncio
    async def test_hook_handles_none_content(self):
        from litellm.integrations.tool_autoheal import ToolAutoHeal
        from unittest.mock import MagicMock

        healer = ToolAutoHeal()

        choice = MagicMock()
        msg = MagicMock()
        msg.content = None
        msg.tool_calls = None
        choice.message = msg

        response = MagicMock()
        response.choices = [choice]

        result = await healer.async_post_call_success_hook(
            data={}, user_api_key_dict=MagicMock(), response=response
        )
        # Should not raise, response passed through
        assert result is response

    @pytest.mark.asyncio
    async def test_hook_handles_empty_choices(self):
        from litellm.integrations.tool_autoheal import ToolAutoHeal
        from unittest.mock import MagicMock

        healer = ToolAutoHeal()

        response = MagicMock()
        response.choices = []

        result = await healer.async_post_call_success_hook(
            data={}, user_api_key_dict=MagicMock(), response=response
        )
        assert result is response

    @pytest.mark.asyncio
    async def test_pre_call_hook_forces_non_streaming(self):
        from litellm.integrations.tool_autoheal import ToolAutoHeal

        healer = ToolAutoHeal()

        # OpenWebUI sends stream=True
        data = {"stream": True, "model": "gpt-oss"}
        result = await healer.async_pre_call_hook(data, None, "completion")
        assert result is not None
        assert result["stream"] is False

    @pytest.mark.asyncio
    async def test_pre_call_hook_passes_non_streaming(self):
        from litellm.integrations.tool_autoheal import ToolAutoHeal

        healer = ToolAutoHeal()

        data = {"stream": False}
        result = await healer.async_pre_call_hook(data, None, "completion")
        assert result is not None
        assert result["stream"] is False
