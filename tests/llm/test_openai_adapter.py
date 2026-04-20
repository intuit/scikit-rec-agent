"""Unit tests for OpenAIAdapter — mainly schema/message translation."""

from __future__ import annotations

import json
from types import SimpleNamespace
from unittest.mock import MagicMock

from scikit_rec_agent.llm.openai import (
    OpenAIAdapter,
    _translate_finish_reason,
    _translate_messages,
    _translate_tools,
)


def test_translate_tools_wraps_in_function_envelope():
    anthropic_tools = [
        {
            "name": "echo",
            "description": "Echo a message.",
            "input_schema": {
                "type": "object",
                "properties": {"message": {"type": "string"}},
                "required": ["message"],
            },
        }
    ]
    translated = _translate_tools(anthropic_tools)
    assert translated[0]["type"] == "function"
    assert translated[0]["function"]["name"] == "echo"
    assert translated[0]["function"]["parameters"]["required"] == ["message"]


def test_translate_messages_splits_tool_result_blocks():
    messages = [
        {"role": "user", "content": "hi"},
        {
            "role": "assistant",
            "content": [
                {"type": "text", "text": "calling"},
                {"type": "tool_use", "id": "tu_1", "name": "echo", "input": {"message": "x"}},
            ],
        },
        {
            "role": "user",
            "content": [{"type": "tool_result", "tool_use_id": "tu_1", "content": '{"status":"ok"}'}],
        },
    ]
    out = _translate_messages(messages, system="sys")

    assert out[0] == {"role": "system", "content": "sys"}
    assert out[1] == {"role": "user", "content": "hi"}
    assistant = out[2]
    assert assistant["role"] == "assistant"
    assert assistant["content"] == "calling"
    assert assistant["tool_calls"][0]["id"] == "tu_1"
    assert json.loads(assistant["tool_calls"][0]["function"]["arguments"]) == {"message": "x"}
    tool = out[3]
    assert tool["role"] == "tool"
    assert tool["tool_call_id"] == "tu_1"


def test_translate_finish_reason():
    assert _translate_finish_reason("tool_calls") == "tool_use"
    assert _translate_finish_reason("stop") == "end_turn"
    assert _translate_finish_reason("length") == "max_tokens"
    assert _translate_finish_reason(None) == "end_turn"
    assert _translate_finish_reason("other") == "other"


def test_chat_returns_llm_response():
    # Build a fake OpenAI completion response
    function = SimpleNamespace(name="echo", arguments=json.dumps({"message": "hi"}))
    tc = SimpleNamespace(id="call_123", function=function)
    msg = SimpleNamespace(content="sure", tool_calls=[tc])
    choice = SimpleNamespace(message=msg, finish_reason="tool_calls")
    completion = SimpleNamespace(choices=[choice])

    client = MagicMock()
    client.chat.completions.create.return_value = completion

    adapter = OpenAIAdapter(client=client, model="gpt-test")
    resp = adapter.chat(messages=[{"role": "user", "content": "hi"}], tools=[], system="s")

    assert resp.content == "sure"
    assert resp.tool_calls[0].id == "call_123"
    assert resp.tool_calls[0].arguments == {"message": "hi"}
    assert resp.stop_reason == "tool_use"


def test_chat_stream_accumulates_tool_call_fragments():
    # Build streaming chunks emulating an OpenAI streaming response
    def make_chunk(content=None, tool_deltas=None, finish_reason=None):
        delta = SimpleNamespace(content=content, tool_calls=tool_deltas)
        choice = SimpleNamespace(delta=delta, finish_reason=finish_reason)
        return SimpleNamespace(choices=[choice])

    tool_delta_a = SimpleNamespace(
        index=0,
        id="call_abc",
        function=SimpleNamespace(name="echo", arguments='{"message"'),
    )
    tool_delta_b = SimpleNamespace(index=0, id=None, function=SimpleNamespace(name=None, arguments=':"hi"}'))
    chunks = [
        make_chunk(content="streaming "),
        make_chunk(content="text"),
        make_chunk(tool_deltas=[tool_delta_a]),
        make_chunk(tool_deltas=[tool_delta_b]),
        make_chunk(finish_reason="tool_calls"),
    ]
    client = MagicMock()
    client.chat.completions.create.return_value = iter(chunks)

    adapter = OpenAIAdapter(client=client)
    events = list(adapter.chat_stream(messages=[], tools=[], system=""))

    text_parts = [e.text for e in events if e.type == "text_delta"]
    assert "".join(text_parts) == "streaming text"
    tool_events = [e for e in events if e.type == "tool_call"]
    assert len(tool_events) == 1
    assert tool_events[0].tool_call.arguments == {"message": "hi"}
    done = [e for e in events if e.type == "done"]
    assert done[-1].stop_reason == "tool_use"
