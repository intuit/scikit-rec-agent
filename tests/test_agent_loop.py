"""Tests for the Agent loop: scripted LLM + fake tools + dispatch + multi-turn."""

from __future__ import annotations

import json

from scikit_rec_agent.agent import Agent
from scikit_rec_agent.tools import Tool, err, ok
from tests._helpers import ScriptedLLM


def _echo_tool() -> Tool:
    def echo(message: str, session) -> dict:
        return ok({"echoed": message})

    return Tool(
        name="echo",
        description="Echo a message.",
        input_schema={
            "type": "object",
            "properties": {"message": {"type": "string"}},
            "required": ["message"],
        },
        fn=echo,
    )


def _failing_tool() -> Tool:
    def boom(session) -> dict:
        raise RuntimeError("kaboom")

    return Tool(
        name="boom",
        description="Always raises.",
        input_schema={"type": "object", "properties": {}},
        fn=boom,
    )


def test_agent_emits_text_and_ends(session):
    llm = ScriptedLLM(responses=["Hello, human."])
    agent = Agent(llm=llm, tools=[], system_prompt="sys", session=session)

    events = list(agent.chat_turn("hi"))

    text_events = [e for e in events if e.type == "text_delta"]
    done_events = [e for e in events if e.type == "done"]
    assert "".join(e.text or "" for e in text_events) == "Hello, human."
    assert len(done_events) == 1
    assert session.messages[0] == {"role": "user", "content": "hi"}
    assert session.messages[-1]["role"] == "assistant"


def test_agent_dispatches_tool_and_loops_back(session):
    llm = ScriptedLLM(
        responses=[
            {
                "text": "calling",
                "tool_calls": [{"id": "tu_1", "name": "echo", "arguments": {"message": "ping"}}],
            },
            "final",
        ]
    )
    agent = Agent(llm=llm, tools=[_echo_tool()], system_prompt="sys", session=session)

    events = list(agent.chat_turn("echo please"))

    tool_results = [e for e in events if e.type == "tool_result"]
    assert len(tool_results) == 1
    assert tool_results[0].tool_result["status"] == "ok"
    assert tool_results[0].tool_result["data"]["echoed"] == "ping"

    # Message history: user → assistant(tool_use) → user(tool_result) → assistant(text)
    assert len(session.messages) == 4
    assert session.messages[1]["role"] == "assistant"
    assert any(b["type"] == "tool_use" for b in session.messages[1]["content"])
    assert session.messages[2]["role"] == "user"
    assert session.messages[2]["content"][0]["type"] == "tool_result"
    # tool_result content is JSON-encoded
    decoded = json.loads(session.messages[2]["content"][0]["content"])
    assert decoded["status"] == "ok"


def test_agent_unknown_tool_returns_error_envelope(session):
    llm = ScriptedLLM(
        responses=[
            {
                "text": "",
                "tool_calls": [{"id": "tu_x", "name": "nonexistent", "arguments": {}}],
            },
            "ok",
        ]
    )
    agent = Agent(llm=llm, tools=[_echo_tool()], system_prompt="sys", session=session)

    events = list(agent.chat_turn("bad call"))

    tool_results = [e for e in events if e.type == "tool_result"]
    assert tool_results[0].tool_result["status"] == "error"
    assert tool_results[0].tool_result["error_type"] == "UnknownTool"


def test_agent_catches_tool_exception(session):
    llm = ScriptedLLM(
        responses=[
            {
                "text": "",
                "tool_calls": [{"id": "tu_b", "name": "boom", "arguments": {}}],
            },
            "recovered",
        ]
    )
    agent = Agent(llm=llm, tools=[_failing_tool()], system_prompt="sys", session=session)

    events = list(agent.chat_turn("fail"))

    tool_results = [e for e in events if e.type == "tool_result"]
    assert tool_results[0].tool_result["status"] == "error"
    assert tool_results[0].tool_result["error_type"] == "RuntimeError"
    assert "kaboom" in tool_results[0].tool_result["message"]


def test_agent_stops_at_max_iterations(session):
    # LLM keeps calling the tool forever; loop should cap and return.
    repeating = {"text": "", "tool_calls": [{"id": "tu_loop", "name": "echo", "arguments": {"message": "x"}}]}
    llm = ScriptedLLM(responses=[repeating] * 25)
    agent = Agent(llm=llm, tools=[_echo_tool()], system_prompt="sys", session=session)

    events = list(agent.chat_turn("loop"))
    done = [e for e in events if e.type == "done"]
    assert done[-1].stop_reason == "max_iterations"


def test_tool_schema_shape():
    t = _echo_tool()
    schema = t.as_llm_schema()
    assert schema["name"] == "echo"
    assert "input_schema" in schema
    assert schema["input_schema"]["required"] == ["message"]


def test_error_envelope_shape():
    e = err("X", "boom", hint="retry")
    assert e == {"status": "error", "error_type": "X", "message": "boom", "hint": "retry"}
