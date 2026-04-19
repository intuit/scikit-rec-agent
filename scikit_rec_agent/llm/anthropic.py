"""Anthropic adapter implementing the BaseLLM protocol.

Wraps `anthropic.Anthropic` clients. The agent's internal message and tool
formats are Anthropic-native, so this adapter is a thin pass-through.
"""

from __future__ import annotations

import json
from typing import TYPE_CHECKING, Any, Iterator

from scikit_rec_agent.llm.base import LLMResponse, LLMStreamEvent, ToolCall

if TYPE_CHECKING:
    import anthropic


DEFAULT_MODEL = "claude-sonnet-4-6"
DEFAULT_MAX_TOKENS = 4096


class AnthropicAdapter:
    def __init__(
        self,
        client: "anthropic.Anthropic",
        model: str = DEFAULT_MODEL,
        max_tokens: int = DEFAULT_MAX_TOKENS,
    ):
        self.client = client
        self.model = model
        self.max_tokens = max_tokens

    def chat(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]],
        system: str,
    ) -> LLMResponse:
        response = self.client.messages.create(
            model=self.model,
            max_tokens=self.max_tokens,
            system=system,
            messages=messages,
            tools=tools,
        )
        text_parts: list[str] = []
        tool_calls: list[ToolCall] = []
        for block in response.content:
            if block.type == "text":
                text_parts.append(block.text)
            elif block.type == "tool_use":
                tool_calls.append(ToolCall(id=block.id, name=block.name, arguments=dict(block.input)))
        return LLMResponse(
            content="".join(text_parts) if text_parts else None,
            tool_calls=tool_calls,
            stop_reason=response.stop_reason or "end_turn",
        )

    def chat_stream(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]],
        system: str,
    ) -> Iterator[LLMStreamEvent]:
        stream = self.client.messages.create(
            model=self.model,
            max_tokens=self.max_tokens,
            system=system,
            messages=messages,
            tools=tools,
            stream=True,
        )

        current_tool: dict[str, Any] | None = None
        tool_input_buffer = ""
        stop_reason: str | None = None
        saw_message_stop = False

        try:
            for event in stream:
                etype = getattr(event, "type", None)
                if etype == "content_block_start":
                    block = event.content_block
                    if getattr(block, "type", None) == "tool_use":
                        current_tool = {"id": block.id, "name": block.name}
                        tool_input_buffer = ""
                elif etype == "content_block_delta":
                    delta = event.delta
                    dtype = getattr(delta, "type", None)
                    if dtype == "text_delta":
                        yield LLMStreamEvent(type="text_delta", text=delta.text)
                    elif dtype == "input_json_delta":
                        tool_input_buffer += delta.partial_json
                elif etype == "content_block_stop":
                    if current_tool is not None:
                        try:
                            arguments = json.loads(tool_input_buffer) if tool_input_buffer else {}
                        except json.JSONDecodeError:
                            arguments = {}
                        yield LLMStreamEvent(
                            type="tool_call",
                            tool_call=ToolCall(
                                id=current_tool["id"],
                                name=current_tool["name"],
                                arguments=arguments,
                            ),
                        )
                        current_tool = None
                        tool_input_buffer = ""
                elif etype == "message_delta":
                    delta_stop = getattr(event.delta, "stop_reason", None)
                    if delta_stop:
                        stop_reason = delta_stop
                elif etype == "message_stop":
                    saw_message_stop = True
                    yield LLMStreamEvent(type="done", stop_reason=stop_reason or "end_turn")
        finally:
            # Stream may end without content_block_stop (network truncation,
            # API timeout, cancellation mid tool_use). Flush whatever's in
            # flight so the agent doesn't silently lose the tool call.
            if current_tool is not None:
                try:
                    arguments = json.loads(tool_input_buffer) if tool_input_buffer else {}
                except json.JSONDecodeError:
                    arguments = {}
                yield LLMStreamEvent(
                    type="tool_call",
                    tool_call=ToolCall(
                        id=current_tool["id"],
                        name=current_tool["name"],
                        arguments=arguments,
                    ),
                )
            if not saw_message_stop:
                # Never got message_stop — synthesize a done event so the
                # agent loop always terminates cleanly.
                yield LLMStreamEvent(type="done", stop_reason=stop_reason or "incomplete")
