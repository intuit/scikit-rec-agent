"""Test helpers shared across modules."""

from __future__ import annotations

from typing import Any, Iterator

from scikit_rec_agent.llm.base import LLMResponse, LLMStreamEvent, ToolCall


class ScriptedLLM:
    """Mock BaseLLM that replays a scripted sequence of responses.

    Each response is either a plain text string (emitted as a single text_delta
    + done), or a dict {"text": str, "tool_calls": [...]} where each tool call
    is {"id": str, "name": str, "arguments": dict}. Consumed in order.
    """

    def __init__(self, responses: list[Any]):
        self.responses = list(responses)
        self.calls: list[dict[str, Any]] = []

    def _event_sequence(self, response) -> list[LLMStreamEvent]:
        if isinstance(response, str):
            return [
                LLMStreamEvent(type="text_delta", text=response),
                LLMStreamEvent(type="done", stop_reason="end_turn"),
            ]
        events: list[LLMStreamEvent] = []
        if response.get("text"):
            events.append(LLMStreamEvent(type="text_delta", text=response["text"]))
        for tc in response.get("tool_calls", []):
            events.append(
                LLMStreamEvent(
                    type="tool_call",
                    tool_call=ToolCall(id=tc["id"], name=tc["name"], arguments=tc.get("arguments", {})),
                )
            )
        stop = "tool_use" if response.get("tool_calls") else "end_turn"
        events.append(LLMStreamEvent(type="done", stop_reason=stop))
        return events

    def chat(self, messages, tools, system) -> LLMResponse:
        self.calls.append({"messages": list(messages), "tools": list(tools), "system": system})
        resp = self.responses.pop(0)
        if isinstance(resp, str):
            return LLMResponse(content=resp, stop_reason="end_turn")
        tcs = [
            ToolCall(id=tc["id"], name=tc["name"], arguments=tc.get("arguments", {}))
            for tc in resp.get("tool_calls", [])
        ]
        return LLMResponse(
            content=resp.get("text"),
            tool_calls=tcs,
            stop_reason="tool_use" if tcs else "end_turn",
        )

    def chat_stream(self, messages, tools, system) -> Iterator[LLMStreamEvent]:
        self.calls.append({"messages": list(messages), "tools": list(tools), "system": system})
        resp = self.responses.pop(0)
        yield from self._event_sequence(resp)
