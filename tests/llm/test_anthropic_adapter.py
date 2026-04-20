"""Unit tests for AnthropicAdapter."""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import MagicMock

from scikit_rec_agent.llm.anthropic import AnthropicAdapter


def _fake_message(text_parts=None, tool_uses=None, stop_reason="end_turn"):
    blocks = []
    for t in text_parts or []:
        blocks.append(SimpleNamespace(type="text", text=t))
    for tu in tool_uses or []:
        blocks.append(SimpleNamespace(type="tool_use", id=tu["id"], name=tu["name"], input=tu.get("input", {})))
    return SimpleNamespace(content=blocks, stop_reason=stop_reason)


def test_chat_parses_text_and_tool_blocks():
    client = MagicMock()
    client.messages.create.return_value = _fake_message(
        text_parts=["Hello, ", "world."],
        tool_uses=[{"id": "tu_1", "name": "echo", "input": {"message": "hi"}}],
        stop_reason="tool_use",
    )
    adapter = AnthropicAdapter(client=client, model="claude-test")

    resp = adapter.chat(messages=[{"role": "user", "content": "hi"}], tools=[], system="s")

    assert resp.content == "Hello, world."
    assert len(resp.tool_calls) == 1
    assert resp.tool_calls[0].name == "echo"
    assert resp.tool_calls[0].arguments == {"message": "hi"}
    assert resp.stop_reason == "tool_use"

    call_kwargs = client.messages.create.call_args.kwargs
    assert call_kwargs["model"] == "claude-test"
    assert call_kwargs["system"] == "s"


def test_chat_with_only_tool_use_has_none_content():
    client = MagicMock()
    client.messages.create.return_value = _fake_message(
        tool_uses=[{"id": "tu_x", "name": "noop", "input": {}}],
        stop_reason="tool_use",
    )
    adapter = AnthropicAdapter(client=client)

    resp = adapter.chat(messages=[], tools=[], system="")
    assert resp.content is None
    assert resp.tool_calls[0].id == "tu_x"


def test_chat_stream_yields_text_and_tool_events():
    # Build a synthetic event sequence that mimics Anthropic streaming.
    events = [
        SimpleNamespace(type="message_start"),
        SimpleNamespace(type="content_block_start", content_block=SimpleNamespace(type="text")),
        SimpleNamespace(type="content_block_delta", delta=SimpleNamespace(type="text_delta", text="Hello ")),
        SimpleNamespace(type="content_block_delta", delta=SimpleNamespace(type="text_delta", text="there")),
        SimpleNamespace(type="content_block_stop"),
        SimpleNamespace(
            type="content_block_start",
            content_block=SimpleNamespace(type="tool_use", id="tu_9", name="echo"),
        ),
        SimpleNamespace(
            type="content_block_delta",
            delta=SimpleNamespace(type="input_json_delta", partial_json='{"message":'),
        ),
        SimpleNamespace(
            type="content_block_delta",
            delta=SimpleNamespace(type="input_json_delta", partial_json='"hi"}'),
        ),
        SimpleNamespace(type="content_block_stop"),
        SimpleNamespace(type="message_delta", delta=SimpleNamespace(stop_reason="tool_use")),
        SimpleNamespace(type="message_stop"),
    ]
    client = MagicMock()
    client.messages.create.return_value = iter(events)
    adapter = AnthropicAdapter(client=client)

    out = list(adapter.chat_stream(messages=[], tools=[], system=""))

    text_deltas = [e.text for e in out if e.type == "text_delta"]
    assert text_deltas == ["Hello ", "there"]
    tool_events = [e for e in out if e.type == "tool_call"]
    assert len(tool_events) == 1
    assert tool_events[0].tool_call.arguments == {"message": "hi"}
    assert tool_events[0].tool_call.id == "tu_9"
    done = [e for e in out if e.type == "done"]
    assert done[-1].stop_reason == "tool_use"


def test_chat_stream_flushes_in_flight_tool_call_on_truncation():
    # Regression: stream ends mid tool_use (no content_block_stop, no
    # message_stop). The adapter must flush the partial tool_call and yield a
    # synthetic done so the agent doesn't silently lose the LLM's action.
    events = [
        SimpleNamespace(
            type="content_block_start",
            content_block=SimpleNamespace(type="tool_use", id="tu_trunc", name="echo"),
        ),
        SimpleNamespace(
            type="content_block_delta",
            delta=SimpleNamespace(type="input_json_delta", partial_json='{"message":"halfway"}'),
        ),
        # No content_block_stop, no message_stop — stream just ends.
    ]
    client = MagicMock()
    client.messages.create.return_value = iter(events)
    adapter = AnthropicAdapter(client=client)

    out = list(adapter.chat_stream(messages=[], tools=[], system=""))

    tool_events = [e for e in out if e.type == "tool_call"]
    assert len(tool_events) == 1
    assert tool_events[0].tool_call.id == "tu_trunc"
    assert tool_events[0].tool_call.arguments == {"message": "halfway"}
    done = [e for e in out if e.type == "done"]
    assert len(done) == 1
    assert done[0].stop_reason == "incomplete"
