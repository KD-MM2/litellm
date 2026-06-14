# ToolAutoHeal — Port Unsloth Tool Auto-Healing + Harmony Parser vào LiteLLM

> **For Hermes:** Use subagent-driven-development skill to implement this plan task-by-task.

**Goal:** Tạo module `ToolAutoHeal` trong LiteLLM, port toàn bộ logic tool-call auto-healing từ Unsloth (self-healing tool calls), thêm Harmony parser cho GPT-OSS, tích hợp thành CustomLogger.

**Architecture:** Module độc lập `litellm/integrations/tool_autoheal.py` chứa toàn bộ logic parse + heal. Đăng ký vào registry hoặc dùng `custom_import` trong config. Tích hợp qua `async_post_call_success_hook` để transform response trước khi trả về client.

**Phạm vi:** `tool_autoheal` = Unsloth auto-healing (Hermes, Functionary, bare JSON) + Harmony parser (GPT-OSS `<|channel|>` format)

**Tech Stack:** Python 3.11+, `re`, `json` (stdlib only — zero external dependencies), LiteLLM CustomLogger base class.

**Key Insight từ codebase analysis:**
- `proxy/utils.py:2414-2419`: `async_post_call_success_hook` **có thể thay thế response object** nếu return non-None
- CustomLogger base class ở `litellm/integrations/custom_logger.py:430`
- Callback registration qua `litellm_settings.callbacks` trong config YAML
- Không cần sửa code core litellm — chỉ thêm 1 file mới + config

---

### Task 1: Tạo module tool_autoheal.py — Core Parsers (Unsloth port)

**Objective:** Port toàn bộ logic tool-call parsing từ Unsloth `tool_healing.py` + `tool_call_parser.py` vào file mới

**Files:**
- Create: `C:\Users\KaoTD\Desktop\Workspace\litellm\litellm\integrations\tool_autoheal.py`

**Step 1: Tạo file với Unsloth tool parsers**

Copy và adapt từ `studio/backend/core/tool_healing.py` và `studio/backend/core/inference/tool_call_parser.py`:

```python
# SPDX-License-Identifier: Apache-2.0
"""
ToolAutoHeal — Tool call auto-healing for LiteLLM.

Ports Unsloth's tool-call parsing logic (strip_tool_call_markup,
parse_tool_calls_from_text) and adds Harmony format support for
GPT-OSS models (<|channel|>, <|message|> tokens).

Zero external dependencies — only stdlib json + re.
"""

import json
import re

# ============================================================
# Pattern 1: Hermes/Functionary <tool_call> and <function=> XML
# Ported from Unsloth studio/backend/core/inference/tool_call_parser.py
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
# Pattern 2: GPT-OSS Harmony format tokens
# ============================================================

HARMONY_CHANNEL_RE = re.compile(r"<\|channel\|>\w+<\|message\|>")
HARMONY_TAG_RE = re.compile(r"<\|(?:start|end|channel|message|return)\|(?:\w+)?>")
HARMONY_JSON_RE = re.compile(
    r"<\|message\|>\s*(\{[\s\S]*?\"name\"\s*:\s*\"[\s\S]*?\})"
)

# ============================================================
# Pattern 3: Raw JSON fallback (model outputs bare JSON with "name" key)
# ============================================================

RAW_TOOL_JSON_RE = re.compile(
    r'\{(?:[^{}]|\{[^{}]*\})*?"name"\s*:\s*"[^"]+"(?:[^{}]|\{[^{}]*\})*?"arguments"(?:[^{}]|\{[^{}]*\})*\}'
)


def strip_tool_markup(text: str, *, final: bool = False) -> str:
    """Strip tool-call XML markup from text (Unsloth-compatible)."""
    if not text:
        return text
    patterns = _TOOL_ALL_PATS if final else _TOOL_CLOSED_PATS
    for pat in patterns:
        text = pat.sub("", text)
    return text.strip() if final else text


def strip_harmony_tokens(text: str) -> str:
    """Strip GPT-OSS Harmony format tokens from text."""
    if not text:
        return text
    text = HARMONY_CHANNEL_RE.sub("", text)
    text = HARMONY_TAG_RE.sub("", text)
    return text.strip()


def _make_tool_call(json_str: str, idx: int) -> dict | None:
    """Parse a JSON string into an OpenAI-format tool_call dict."""
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


def parse_tool_calls_from_text(content: str, id_offset: int = 0) -> list[dict]:
    """
    Extract tool calls from model output text.
    
    Handles 4 formats (checked in order):
    1. <tool_call>{"name":"...","arguments":{...}}</tool_call>  (Hermes)
    2. <function=name><parameter=k>v</parameter></function>      (Functionary)
    3. <|message|>{"name":"...","arguments":{...}}                 (GPT-OSS Harmony)
    4. Bare {"name":"...","arguments":{...}}                       (Fallback)
    """
    tool_calls: list[dict] = []

    # Pattern 1: JSON inside <tool_call> tags (balanced-brace extraction)
    for m in _TC_JSON_START_RE.finditer(content):
        brace_start = m.end() - 1
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

    # Pattern 2: XML-style <function=name><parameter=k>v</parameter></function>
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

            arguments = {}
            param_starts = list(_TC_PARAM_START_RE.finditer(body))
            if len(param_starts) == 1:
                pm = param_starts[0]
                val = body[pm.end() :]
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

    # Pattern 3: GPT-OSS Harmony <|message|>{json}
    if not tool_calls:
        for m in HARMONY_JSON_RE.finditer(content):
            tc = _make_tool_call(m.group(1), id_offset + len(tool_calls))
            if tc:
                tool_calls.append(tc)

    # Pattern 4: Raw JSON with "name" key (generic fallback)  
    if not tool_calls:
        cleaned = strip_harmony_tokens(content)
        for m in RAW_TOOL_JSON_RE.finditer(cleaned):
            tc = _make_tool_call(m.group(), id_offset + len(tool_calls))
            if tc:
                tool_calls.append(tc)

    return tool_calls
```

**Step 2: Verify syntax**

```bash
cd C:\Users\KaoTD\Desktop\Workspace\litellm
python -c "from litellm.integrations.tool_autoheal import parse_tool_calls_from_text, strip_harmony_tokens; print('Import OK')"
```

**Step 3: Commit**

```bash
git add litellm/integrations/tool_autoheal.py
git commit -m "feat: add tool_autoheal module — Unsloth tool parsers + Harmony support"
```

---

### Task 2: Thêm CustomLogger class ToolAutoHeal

**Objective:** Wrap parsers thành CustomLogger subclass để dùng với litellm proxy

**Files:**
- Modify: `C:\Users\KaoTD\Desktop\Workspace\litellm\litellm\integrations\tool_autoheal.py` (append code)

**Step 1: Thêm class vào cuối file**

```python
# ============================================================
# LiteLLM CustomLogger Integration
# ============================================================

from typing import TYPE_CHECKING, Any, Optional

from litellm.integrations.custom_logger import CustomLogger

if TYPE_CHECKING:
    from litellm.proxy._types import UserAPIKeyAuth
    from litellm.types.utils import LLMResponseTypes, ModelResponse
else:
    UserAPIKeyAuth = Any
    LLMResponseTypes = Any
    ModelResponse = Any


class ToolAutoHeal(CustomLogger):
    """
    LiteLLM CustomLogger that auto-heals tool calls from model output.
    
    Handles:
    - Hermes/Functionary XML tool calls (<tool_call>, <function=>)
    - GPT-OSS Harmony format (<|channel|>, <|message|>)
    - Bare JSON tool calls
    
    Usage in litellm config.yaml:
        litellm_settings:
          callbacks: ["tool_autoheal"]
    
    Or as Python import path:
        litellm_settings:
          callbacks: ["litellm.integrations.tool_autoheal.ToolAutoHeal"]
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

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
                    # Model intended a tool call — use structured format
                    msg.content = clean_content
                    existing = list(getattr(msg, "tool_calls", []) or [])
                    msg.tool_calls = existing + tool_calls
                elif clean_content:
                    # Normal response with cleaned content
                    msg.content = clean_content

            return response

        except Exception:
            # Never break the pipeline on healing failure
            return response
```

**Step 2: Verify import**

```bash
python -c "from litellm.integrations.tool_autoheal import ToolAutoHeal; h = ToolAutoHeal(); print('CustomLogger OK')"
```

**Step 3: Commit**

```bash
git add litellm/integrations/tool_autoheal.py
git commit -m "feat: add ToolAutoHeal CustomLogger class"
```

---

### Task 3: Đăng ký callback vào litellm registry

**Objective:** Cho phép dùng tên ngắn `"tool_autoheal"` trong config YAML

**Files:**
- Modify: `C:\Users\KaoTD\Desktop\Workspace\litellm\litellm\litellm_core_utils\litellm_logging.py`

**Step 1: Thêm import và case trong `_init_custom_logger_compatible_class`**

Tìm hàm `_init_custom_logger_compatible_class` (line ~3857), thêm case mới trước `else` cuối cùng:

```python
        elif logging_integration == "tool_autoheal":
            from litellm.integrations.tool_autoheal import ToolAutoHeal

            for callback in _in_memory_loggers:
                if isinstance(callback, ToolAutoHeal):
                    return callback  # type: ignore

            _healer = ToolAutoHeal()
            _in_memory_loggers.append(_healer)
            return _healer  # type: ignore
```

**Step 2: Verify**

```bash
python -c "
from litellm.litellm_core_utils.litellm_logging import _init_custom_logger_compatible_class
cb = _init_custom_logger_compatible_class('tool_autoheal', None, None, {})
print(type(cb).__name__)
"
# Expected: ToolAutoHeal
```

**Step 3: Commit**

```bash
git add litellm/litellm_core_utils/litellm_logging.py
git commit -m "feat: register tool_autoheal in callback registry"
```

---

### Task 4: Tạo file config mẫu

**Objective:** File config để người dùng copy-paste chạy ngay

**Files:**
- Create: `C:\Users\KaoTD\Desktop\Workspace\litellm\config_harmony_example.yaml`

**Step 1: Viết config**

```yaml
model_list:
  - model_name: gpt-oss-20b
    litellm_params:
      model: openai/gpt-oss-20b
      api_base: http://127.0.0.1:8080/v1
      api_key: sk-dummy
      stream: false

general_settings:
  master_key: sk-litellm-master-key

litellm_settings:
  callbacks: ["tool_autoheal"]
  drop_params: true

# Optional: enable logging
# litellm_settings:
#   success_callback: ["prometheus"]
```

**Step 2: Commit**

```bash
git add config_harmony_example.yaml
git commit -m "docs: add config example for harmony auto-heal"
```

---

### Task 5: Viết unit tests cho tool_autoheal module

**Objective:** Test các parser với các format khác nhau

**Files:**
- Create: `C:\Users\KaoTD\Desktop\Workspace\litellm\tests\integration_tests\test_tool_autoheal.py`

**Step 1: Viết tests**

```python
"""Tests for tool_autoheal module."""
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
        text = '<|start|>assistant<|channel|>analysis<|message|>{"answer": 42}<|end|>'
        result = strip_harmony_tokens(text)
        assert result == '{"answer": 42}'

    def test_preserve_normal_text(self):
        text = 'Normal text without tokens'
        result = strip_harmony_tokens(text)
        assert result == 'Normal text without tokens'

    def test_empty_string(self):
        assert strip_harmony_tokens('') == ''
        assert strip_harmony_tokens(None) == None


class TestStripToolMarkup:
    def test_strip_tool_call_closed(self):
        text = 'Some text <tool_call>{"name":"test"}</tool_call> more text'
        result = strip_tool_markup(text)
        assert 'tool_call' not in result
        assert 'Some text' in result
        assert 'more text' in result

    def test_strip_function_tag(self):
        text = '<function=web_search><parameter=q>hello</parameter></function>'
        result = strip_tool_markup(text)
        assert 'function=' not in result


class TestParseToolCalls:
    def test_parse_hermes_json_format(self):
        content = '''<tool_call>{"name":"web_search","arguments":{"query":"bitcoin price"}}</tool_call>'''
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
            '<tool_call>{"name":"search","arguments":{"q":"A"}}</tool_call>'
            '<tool_call>{"name":"search","arguments":{"q":"B"}}</tool_call>'
        )
        result = parse_tool_calls_from_text(content)
        assert len(result) == 2
        assert result[0]["function"]["name"] == "search"
        assert result[1]["function"]["name"] == "search"

    def test_partial_json_recovery(self):
        """Models often omit closing tags — parser should still work."""
        content = '<tool_call>{"name":"test","arguments":{}}'
        result = parse_tool_calls_from_text(content)
        assert len(result) == 1
        assert result[0]["function"]["name"] == "test"
```

**Step 2: Chạy tests**

```bash
cd C:\Users\KaoTD\Desktop\Workspace\litellm
python -m pytest tests/integration_tests/test_tool_autoheal.py -v
```

**Expected**: All tests pass

**Step 3: Commit**

```bash
git add tests/integration_tests/test_tool_autoheal.py
git commit -m "test: add unit tests for tool_autoheal parsers"
```

---

### Task 6: Integration test — end-to-end với mock response

**Objective:** Test ToolAutoHeal CustomLogger với mock ModelResponse

**Files:**
- Modify: `C:\Users\KaoTD\Desktop\Workspace\litellm\tests\integration_tests\test_tool_autoheal.py` (append)

**Step 1: Thêm test CustomLogger**

```python
class TestToolAutoHealCustomLogger:
    @pytest.mark.asyncio
    async def test_hook_heals_harmony_response(self):
        from litellm.integrations.tool_autoheal import ToolAutoHeal
        from unittest.mock import MagicMock
        
        healer = ToolAutoHeal()
        
        # Build a mock response with Harmony-formatted content
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
        
        # Verify tool_call was extracted
        assert len(result.choices[0].message.tool_calls) == 1
        assert result.choices[0].message.tool_calls[0]["function"]["name"] == "web_search"
        # Content should be cleaned
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
```

**Step 2: Chạy integration tests**

```bash
pytest tests/integration_tests/test_tool_autoheal.py::TestToolAutoHealCustomLogger -v
```

**Step 3: Commit**

```bash
git add tests/integration_tests/test_tool_autoheal.py
git commit -m "test: add integration tests for ToolAutoHeal CustomLogger"
```

---

### Task 7: Kiểm tra end-to-end với llama-server thật (manual)

**Objective:** Xác nhận toàn bộ stack hoạt động

**Step 1: Chạy llama-server**

```bash
llama-server -m gpt-oss-20b-Q4_K_M.gguf -c 16384 -fa --jinja --port 8080 --host 0.0.0.0
```

**Step 2: Chạy litellm proxy với harmony config**

```bash
cd C:\Users\KaoTD\Desktop\Workspace\litellm
litellm --config config_harmony_example.yaml --port 4000 --host 0.0.0.0
```

**Step 3: Test bằng curl**

```bash
curl http://localhost:4000/v1/chat/completions \
  -H "Authorization: Bearer sk-litellm-master-key" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "gpt-oss-20b",
    "stream": false,
    "tools": [{"type":"function","function":{"name":"web_search","parameters":{"type":"object","properties":{"query":{"type":"string"}}}}}],
    "messages": [{"role":"user","content":"Search for Bitcoin price"}]
  }' | python -m json.tool
```

**Expected**: Response có `tool_calls` array với format chuẩn OpenAI, không còn `<|channel|>` tokens.

**Step 4: Commit nếu có fix**

---

### Tổng kết file changes

| File | Action | Purpose |
|---|---|---|
| `litellm/integrations/tool_autoheal.py` | **CREATE** | Core module: parsers + CustomLogger |
| `litellm/litellm_core_utils/litellm_logging.py` | MODIFY | Register `"tool_autoheal"` callback name |
| `config_harmony_example.yaml` | **CREATE** | Example config for users |
| `tests/integration_tests/test_tool_autoheal.py` | **CREATE** | Unit + integration tests |

### Risks & Notes

1. **Streaming mode**: `async_post_call_streaming_hook` có vấn đề (litellm bug #9639), nên config `stream: false` cho GPT-OSS
2. **Performance**: Regex processing trên text vài KB — overhead không đáng kể (~1ms)
3. **Compatibility**: Module chỉ dùng stdlib, tương thích Python 3.10+
4. **Registry approach**: Nếu không muốn sửa `litellm_logging.py`, có thể dùng `custom_import` path trong config thay vì tên ngắn
