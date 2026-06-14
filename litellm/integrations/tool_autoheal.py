# SPDX-License-Identifier: Apache-2.0
"""
ToolAutoHeal — Tool call auto-healing for LiteLLM.

Ports Unsloth's tool-call parsing logic (strip_tool_call_markup,
parse_tool_calls_from_text) from:
  studio/backend/core/inference/tool_call_parser.py
  studio/backend/core/tool_healing.py

And adds Harmony format support for GPT-OSS models
(<|channel|>, <|message|> tokens).

Zero external dependencies — only stdlib json + re.

Handled tool call formats (in parse order):
  1. <tool_call>{"name":"...","arguments":{...}}</tool_call>   (Hermes)
  2. <function=name><parameter=k>v</parameter></function>      (Functionary)
  3. <|message|>{"name":"...","arguments":{...}}                 (GPT-OSS Harmony)
  4. Bare {"name":"...","arguments":{...}}                       (Fallback)
"""

import json
import re

# ============================================================
# Pattern Group 1: Hermes/Functionary <tool_call> XML markup
# Ported verbatim from Unsloth
# ============================================================

_TOOL_CLOSED_PATS = [
    re.compile(r"<tool_call>.*?</tool_call>", re.DOTALL),
    re.compile(r"<function=[\w-]+>.*?</function>", re.DOTALL),
]
_TOOL_ALL_PATS = _TOOL_CLOSED_PATS + [
    re.compile(r"<tool_call>.*$", re.DOTALL),
    re.compile(r"<function=[\w-]+>.*$", re.DOTALL),
]

_TC_JSON_START_RE = re.compile(r"<tool_call>\s*\{")
_TC_FUNC_START_RE = re.compile(r"<function=([\w-]+)>\s*")
_TC_END_TAG_RE = re.compile(r"</tool_call>")
_TC_FUNC_CLOSE_RE = re.compile(r"\s*</function>\s*$")
_TC_PARAM_START_RE = re.compile(r"<parameter=([\w-]+)>\s*")
_TC_PARAM_CLOSE_RE = re.compile(r"\s*</parameter>\s*$")

# ============================================================
# Pattern Group 2: GPT-OSS Harmony format tokens
# ============================================================

HARMONY_CHANNEL_RE = re.compile(r"<\|channel\|>\w+<\|message\|>")
HARMONY_TAG_RE = re.compile(r"<\|(?:start|end|channel|message|return)\|(?:\w+)?>")
HARMONY_JSON_RE = re.compile(
    r"<\|message\|>\s*(\{[\s\S]*?\"name\"\s*:\s*\"[\s\S]*?\})"
)

# ============================================================
# Pattern Group 3: Bare JSON tool call fallback
# Matches { ... "name": "..." ... "arguments": ... }
# Handles up to 2 levels of nesting
# ============================================================

_NAMED_JSON_RE = re.compile(
    r'\{\s*"name"\s*:\s*"[^"]+"\s*,\s*"arguments"\s*:\s*'
    r'(?:\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}|"[^"]*")'
    r'(?:\s*,\s*"[^"]+"\s*:\s*(?:\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}|"[^"]*"|\d+))*'
    r'\s*\}',
    re.DOTALL,
)


def strip_tool_markup(text: str, *, final: bool = False) -> str:
    """Strip tool-call XML markup from text (Unsloth-compatible).

    When ``final`` is False, only fully closed tool-call blocks are removed
    (safe for streaming). When ``final`` is True, trailing incomplete
    blocks are removed too and the result is stripped.
    """
    if not text:
        return text
    patterns = _TOOL_ALL_PATS if final else _TOOL_CLOSED_PATS
    for pat in patterns:
        text = pat.sub("", text)
    return text.strip() if final else text


def strip_harmony_tokens(text: str | None) -> str | None:
    """Strip GPT-OSS Harmony format tokens from text.

    >>> strip_harmony_tokens('<|channel|>commentary<|message|>Hello')
    'Hello'
    >>> strip_harmony_tokens('<|start|>assistant<|end|>')
    ''
    """
    if text is None:
        return None
    if not text:
        return ""
    text = HARMONY_CHANNEL_RE.sub("", text)
    text = HARMONY_TAG_RE.sub("", text)
    return text.strip()


def _make_tool_call(json_str: str, idx: int) -> dict | None:
    """Parse a JSON string into an OpenAI-format tool_call dict.

    Returns None if JSON is invalid or missing required "name" field.
    """
    try:
        obj = json.loads(json_str)
        name = obj.get("name", "")
        args = obj.get("arguments", obj.get("parameters", {}))
        if not name:
            return None
        return {
            "id": f"call_{idx:04d}",
            "type": "function",
            "function": {
                "name": name,
                "arguments": json.dumps(args)
                if isinstance(args, dict)
                else str(args),
            },
        }
    except (json.JSONDecodeError, ValueError):
        return None


def parse_tool_calls_from_text(
    content: str, *, id_offset: int = 0
) -> list[dict]:
    """Extract OpenAI-format tool_calls from model output text.

    Checks 4 formats in order, stops at first match:

    1. ``<tool_call>{"name":"...","arguments":{...}}</tool_call>``
       Balanced-brace extraction, safe across JSON strings.
    2. ``<function=name><parameter=k>v</parameter></function>``
       XML-style with closing tags optional.
    3. ``<|message|>{"name":"...","arguments":{...}}``
       GPT-OSS Harmony format.
    4. Bare ``{"name":"...","arguments":{...}}``
       Generic JSON fallback.

    Returns:
        List of dicts in OpenAI format:
        ``{"id", "type": "function", "function": {"name", "arguments"}}``
        ``arguments`` is always a JSON string.
    """
    tool_calls: list[dict] = []

    # ── Pattern 1: JSON inside <tool_call> tags ──
    for m in _TC_JSON_START_RE.finditer(content):
        brace_start = m.end() - 1  # position of opening {
        depth, i = 0, brace_start
        in_string = False
        while i < len(content):
            ch = content[i]
            if in_string:
                if ch == "\\" and i + 1 < len(content):
                    i += 2
                    continue
                if ch == '"':
                    in_string = False
            elif ch == '"':
                in_string = True
            elif ch == "{":
                depth += 1
            elif ch == "}":
                depth -= 1
                if depth == 0:
                    break
            i += 1
        if depth == 0:
            json_str = content[brace_start : i + 1]
            tc = _make_tool_call(json_str, id_offset + len(tool_calls))
            if tc:
                tool_calls.append(tc)

    # ── Pattern 2: <function=name><parameter=k>v</parameter></function> ──
    if not tool_calls:
        func_starts = list(_TC_FUNC_START_RE.finditer(content))
        for idx, fm in enumerate(func_starts):
            func_name = fm.group(1)
            body_start = fm.end()

            next_func = (
                func_starts[idx + 1].start()
                if idx + 1 < len(func_starts)
                else len(content)
            )
            end_tag = _TC_END_TAG_RE.search(content[body_start:])
            body_end = body_start + end_tag.start() if end_tag else len(content)
            body_end = min(body_end, next_func)

            body = content[body_start:body_end]
            body = _TC_FUNC_CLOSE_RE.sub("", body)

            arguments: dict = {}
            param_starts = list(_TC_PARAM_START_RE.finditer(body))
            if len(param_starts) == 1:
                pm = param_starts[0]
                val = body[pm.end():]
                val = _TC_PARAM_CLOSE_RE.sub("", val)
                arguments[pm.group(1)] = val.strip()
            else:
                for pidx, pm in enumerate(param_starts):
                    val_start = pm.end()
                    next_param = (
                        param_starts[pidx + 1].start()
                        if pidx + 1 < len(param_starts)
                        else len(body)
                    )
                    val = body[val_start:next_param]
                    val = _TC_PARAM_CLOSE_RE.sub("", val)
                    arguments[pm.group(1)] = val.strip()

            tc = {
                "id": f"call_{id_offset + len(tool_calls):04d}",
                "type": "function",
                "function": {
                    "name": func_name,
                    "arguments": json.dumps(arguments),
                },
            }
            tool_calls.append(tc)

    # ── Pattern 3: GPT-OSS Harmony <|message|>{json} ──
    if not tool_calls:
        for m in HARMONY_JSON_RE.finditer(content):
            tc = _make_tool_call(m.group(1), id_offset + len(tool_calls))
            if tc:
                tool_calls.append(tc)

    # ── Pattern 4: Bare JSON with "name" key (generic fallback) ──
    if not tool_calls:
        cleaned = strip_harmony_tokens(content)
        for m in _NAMED_JSON_RE.finditer(cleaned):
            tc = _make_tool_call(m.group(), id_offset + len(tool_calls))
            if tc:
                tool_calls.append(tc)

    return tool_calls


# ============================================================
# LiteLLM CustomLogger Integration
# ============================================================

from typing import TYPE_CHECKING, Any

from litellm.integrations.custom_logger import CustomLogger

if TYPE_CHECKING:
    from litellm.proxy._types import UserAPIKeyAuth
    from litellm.types.utils import LLMResponseTypes
else:
    UserAPIKeyAuth = Any
    LLMResponseTypes = Any


class ToolAutoHeal(CustomLogger):
    """LiteLLM CustomLogger that auto-heals tool calls from model output.

    Handles:
    - Hermes/Functionary XML tool calls (``<tool_call>``, ``<function=>``)
    - GPT-OSS Harmony format (``<|channel|>``, ``<|message|>``)
    - Bare JSON tool calls (generic fallback)

    Usage in litellm config.yaml::

        litellm_settings:
          callbacks: ["tool_autoheal"]

    Or as Python import path::

        litellm_settings:
          callbacks: ["litellm.integrations.tool_autoheal.ToolAutoHeal"]
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    async def async_pre_call_hook(
        self,
        data: dict,
        user_api_key_dict: "UserAPIKeyAuth",
        call_type: str,
        **kwargs,
    ) -> dict | None:
        """Force non-streaming mode for tool-healing compatibility.

        OpenAI SDK's streaming parser chokes on raw XML/markup tool calls
        (``<tool_call>``, ``<function=>``, ``<|channel|>``).  By forcing
        ``stream=False`` we ensure the full response is captured so that
        ``async_post_call_success_hook`` can heal it before the client
        sees it.

        ``**kwargs`` absorbs additional parameters the LiteLLM framework
        may inject (e.g. ``cache``) without breaking.
        """
        if data.get("stream"):
            data["stream"] = False
        return data

    async def async_post_call_success_hook(
        self,
        data: dict,
        user_api_key_dict: "UserAPIKeyAuth",
        response: "LLMResponseTypes",
    ) -> Any:
        """Post-process response: heal tool calls + clean Harmony tokens."""
        try:
            choices = getattr(response, "choices", None)
            if not choices:
                return response

            for choice in choices:
                msg = getattr(choice, "message", None)
                if not msg:
                    continue

                content = getattr(msg, "content", None) or ""

                # Step 1: Try to extract structured tool calls
                tool_calls = parse_tool_calls_from_text(content)

                # Step 2: Clean content by stripping markup
                clean_content = strip_harmony_tokens(content)
                clean_content = strip_tool_markup(clean_content)
                clean_content = clean_content.strip() if clean_content else None

                if tool_calls:
                    msg.content = clean_content
                    existing = list(getattr(msg, "tool_calls", []) or [])
                    msg.tool_calls = existing + tool_calls
                elif clean_content:
                    msg.content = clean_content

            return response

        except Exception:
            return response
