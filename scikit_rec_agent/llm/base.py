"""BaseLLM protocol and response dataclasses.

The agent depends only on this protocol, not on any specific LLM SDK. Adapters
for Anthropic and OpenAI are provided; any class that implements chat() and
chat_stream() satisfies the contract.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Iterator, Protocol, runtime_checkable


@dataclass
class ToolCall:
    id: str
    name: str
    arguments: dict[str, Any]


@dataclass
class LLMResponse:
    content: str | None
    tool_calls: list[ToolCall] = field(default_factory=list)
    stop_reason: str = "end_turn"


@dataclass
class LLMStreamEvent:
    type: str  # "text_delta" | "tool_call" | "done"
    text: str | None = None
    tool_call: ToolCall | None = None
    stop_reason: str | None = None


@runtime_checkable
class BaseLLM(Protocol):
    """Minimal LLM interface required by the Agent.

    Implementations receive messages in OpenAI-style format (role/content dicts
    with optional tool_calls / tool_call_id) and tools in Anthropic-style shape
    ({"name", "description", "input_schema"}). Adapters translate as needed.
    """

    def chat(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]],
        system: str,
    ) -> LLMResponse: ...

    def chat_stream(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]],
        system: str,
    ) -> Iterator[LLMStreamEvent]: ...
