"""OpenAI adapter implementing the BaseLLM protocol.

The agent speaks Anthropic-native message and tool formats internally. This
adapter translates both directions so OpenAI models can be driven through the
same Agent loop.
"""

from __future__ import annotations

import json
from typing import TYPE_CHECKING, Any, Iterator

from scikit_rec_agent.llm.base import LLMResponse, LLMStreamEvent, ToolCall

if TYPE_CHECKING:
    import openai


DEFAULT_MODEL = "gpt-4o"


class OpenAIAdapter:
    def __init__(self, client: "openai.OpenAI", model: str = DEFAULT_MODEL):
        self.client = client
        self.model = model

    def chat(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]],
        system: str,
    ) -> LLMResponse:
        completion = self.client.chat.completions.create(
            model=self.model,
            messages=_translate_messages(messages, system),
            tools=_translate_tools(tools) if tools else None,
        )
        choice = completion.choices[0]
        msg = choice.message
        tool_calls: list[ToolCall] = []
        for tc in msg.tool_calls or []:
            try:
                args = json.loads(tc.function.arguments)
            except json.JSONDecodeError:
                args = {}
            tool_calls.append(ToolCall(id=tc.id, name=tc.function.name, arguments=args))
        return LLMResponse(
            content=msg.content,
            tool_calls=tool_calls,
            stop_reason=_translate_finish_reason(choice.finish_reason),
        )

    def chat_stream(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]],
        system: str,
    ) -> Iterator[LLMStreamEvent]:
        stream = self.client.chat.completions.create(
            model=self.model,
            messages=_translate_messages(messages, system),
            tools=_translate_tools(tools) if tools else None,
            stream=True,
        )
        tool_buffers: dict[int, dict[str, Any]] = {}
        finish_reason: str | None = None

        for chunk in stream:
            if not chunk.choices:
                continue
            choice = chunk.choices[0]
            delta = choice.delta
            if getattr(delta, "content", None):
                yield LLMStreamEvent(type="text_delta", text=delta.content)
            for tc_delta in getattr(delta, "tool_calls", None) or []:
                idx = tc_delta.index
                buf = tool_buffers.setdefault(idx, {"id": None, "name": None, "args": ""})
                if tc_delta.id:
                    buf["id"] = tc_delta.id
                fn = getattr(tc_delta, "function", None)
                if fn is not None:
                    if getattr(fn, "name", None):
                        buf["name"] = fn.name
                    if getattr(fn, "arguments", None):
                        buf["args"] += fn.arguments
            if choice.finish_reason:
                finish_reason = choice.finish_reason

        for idx in sorted(tool_buffers):
            buf = tool_buffers[idx]
            try:
                arguments = json.loads(buf["args"]) if buf["args"] else {}
            except json.JSONDecodeError:
                arguments = {}
            yield LLMStreamEvent(
                type="tool_call",
                tool_call=ToolCall(id=buf["id"], name=buf["name"], arguments=arguments),
            )
        yield LLMStreamEvent(type="done", stop_reason=_translate_finish_reason(finish_reason))


def _translate_tools(tools: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Convert Anthropic-style tool schemas to OpenAI's nested function format."""
    translated = []
    for tool in tools:
        translated.append(
            {
                "type": "function",
                "function": {
                    "name": tool["name"],
                    "description": tool.get("description", ""),
                    "parameters": tool.get("input_schema", {"type": "object", "properties": {}}),
                },
            }
        )
    return translated


def _translate_messages(messages: list[dict[str, Any]], system: str) -> list[dict[str, Any]]:
    """Convert Anthropic-style messages to OpenAI chat message format.

    Anthropic stores tool_use and tool_result as content blocks within
    assistant/user messages. OpenAI uses separate tool_calls on the assistant
    message and separate role='tool' messages for results.
    """
    out: list[dict[str, Any]] = []
    if system:
        out.append({"role": "system", "content": system})

    for msg in messages:
        role = msg["role"]
        content = msg["content"]

        if isinstance(content, str):
            out.append({"role": role, "content": content})
            continue

        if role == "assistant":
            text_parts: list[str] = []
            tool_calls: list[dict[str, Any]] = []
            for block in content:
                btype = block.get("type")
                if btype == "text":
                    text_parts.append(block["text"])
                elif btype == "tool_use":
                    tool_calls.append(
                        {
                            "id": block["id"],
                            "type": "function",
                            "function": {
                                "name": block["name"],
                                "arguments": json.dumps(block.get("input", {})),
                            },
                        }
                    )
            entry: dict[str, Any] = {"role": "assistant", "content": "".join(text_parts) or None}
            if tool_calls:
                entry["tool_calls"] = tool_calls
            out.append(entry)
            continue

        # user role with content blocks → may contain tool_results and/or text
        text_parts = []
        for block in content:
            btype = block.get("type")
            if btype == "tool_result":
                tool_content = block.get("content", "")
                if isinstance(tool_content, list):
                    tool_content = "".join(b.get("text", "") for b in tool_content if b.get("type") == "text")
                out.append(
                    {
                        "role": "tool",
                        "tool_call_id": block["tool_use_id"],
                        "content": tool_content,
                    }
                )
            elif btype == "text":
                text_parts.append(block["text"])
        if text_parts:
            out.append({"role": "user", "content": "".join(text_parts)})

    return out


def _translate_finish_reason(reason: str | None) -> str:
    if reason == "tool_calls":
        return "tool_use"
    if reason == "stop":
        return "end_turn"
    if reason == "length":
        return "max_tokens"
    return reason or "end_turn"
