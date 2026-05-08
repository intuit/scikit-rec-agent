"""Example: plug in a company-internal LLM via the BaseLLM protocol.

The agent never talks to a specific provider directly — it only knows about
`BaseLLM`. The built-in Anthropic and OpenAI adapters are two reference
implementations; anything satisfying the same protocol works. This file walks
through building an adapter for a fictional internal service ("Acme LLM")
so you can see exactly which translation points you need to handle.

Replace the FakeAcmeClient / response shapes below with your real internal
SDK. Everything from AcmeLLMAdapter downward is the template you keep.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Iterator

from scikit_rec_agent import Agent, LLMResponse, LLMStreamEvent, ToolCall

# ---------------------------------------------------------------------------
# 1. A stand-in for your company's LLM client.
#
# Replace this entire section with `from acme_llm import AcmeClient`
# (or whatever your real SDK is). The adapter below only cares about the
# SHAPE of this client: what kwargs it takes, what it returns, how it streams.
# ---------------------------------------------------------------------------


@dataclass
class _FakeToolCall:
    id: str
    name: str
    arguments: dict[str, Any]


@dataclass
class _FakeResponse:
    text: str | None
    tool_calls: list[_FakeToolCall] = field(default_factory=list)
    stop_reason: str = "done"


class FakeAcmeClient:
    """Mimics the surface your real Acme LLM client might expose.

    For the sake of the example the methods just return canned responses.
    The only interesting thing is that they accept provider-native shapes
    (whatever your internal API uses) — NOT Anthropic-native shapes. The
    adapter is what bridges the two.
    """

    def complete(
        self,
        *,
        model: str,
        system: str,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]],
    ) -> _FakeResponse:
        return _FakeResponse(text="(canned non-streaming reply)", stop_reason="done")

    def complete_stream(
        self,
        *,
        model: str,
        system: str,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]],
    ) -> Iterator[dict[str, Any]]:
        # Pretend the service emits chunks shaped like this. Replace with
        # whatever your real internal API streams.
        yield {"type": "text", "delta": "(streamed reply)"}
        yield {"type": "end", "stop_reason": "done"}


# ---------------------------------------------------------------------------
# 2. The adapter — the piece you keep.
# ---------------------------------------------------------------------------


class AcmeLLMAdapter:
    """BaseLLM implementation for the fictional Acme LLM service.

    The agent calls chat() or chat_stream() with Anthropic-native messages
    and tools. This adapter:
      - translates them to the internal service's wire format
      - invokes the client
      - translates the response back to LLMResponse / LLMStreamEvent
    """

    def __init__(self, client: FakeAcmeClient, model: str = "acme-default"):
        self.client = client
        self.model = model

    # ------ non-streaming ------

    def chat(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]],
        system: str,
    ) -> LLMResponse:
        response = self.client.complete(
            model=self.model,
            system=system,
            messages=_to_acme_messages(messages),
            tools=_to_acme_tools(tools),
        )
        return LLMResponse(
            content=response.text,
            tool_calls=[ToolCall(id=tc.id, name=tc.name, arguments=tc.arguments) for tc in response.tool_calls],
            stop_reason=_map_stop_reason(response.stop_reason),
        )

    # ------ streaming ------

    def chat_stream(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]],
        system: str,
    ) -> Iterator[LLMStreamEvent]:
        """Yield LLMStreamEvents as the service streams.

        The agent needs three event types:
          - text_delta: every incremental text chunk as it arrives
          - tool_call: ONE event per tool call, emitted only when the call
            is fully assembled (arguments JSON complete). Do NOT emit partial
            tool-call deltas — accumulate locally.
          - done: exactly one, at end of turn, with stop_reason.
        """
        stop_reason: str | None = None
        tool_buffers: dict[str, dict[str, Any]] = {}

        for chunk in self.client.complete_stream(
            model=self.model,
            system=system,
            messages=_to_acme_messages(messages),
            tools=_to_acme_tools(tools),
        ):
            ctype = chunk.get("type")
            if ctype == "text":
                yield LLMStreamEvent(type="text_delta", text=chunk["delta"])
            elif ctype == "tool_call_start":
                # Real services typically start a tool call with id + name,
                # then emit argument fragments. Buffer until complete.
                tool_buffers[chunk["id"]] = {
                    "name": chunk["name"],
                    "arguments": "",
                }
            elif ctype == "tool_call_delta":
                tool_buffers[chunk["id"]]["arguments"] += chunk["delta"]
            elif ctype == "tool_call_end":
                import json

                buf = tool_buffers.pop(chunk["id"])
                try:
                    args = json.loads(buf["arguments"]) if buf["arguments"] else {}
                except json.JSONDecodeError:
                    args = {}
                yield LLMStreamEvent(
                    type="tool_call",
                    tool_call=ToolCall(id=chunk["id"], name=buf["name"], arguments=args),
                )
            elif ctype == "end":
                stop_reason = chunk.get("stop_reason")

        yield LLMStreamEvent(type="done", stop_reason=_map_stop_reason(stop_reason))


# ---------------------------------------------------------------------------
# 3. Translation helpers — the only code that needs to know both formats.
#
# The agent's internal format is Anthropic-native (content blocks for
# assistant tool_use and user tool_result). Your internal service almost
# certainly uses something different. These helpers are where you translate.
# Look at llm/openai.py in this repo for a complete reference that goes from
# Anthropic-native to OpenAI's different shape.
# ---------------------------------------------------------------------------


def _to_acme_messages(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Translate Anthropic-native messages to the Acme service's format.

    Input messages look like:
      {"role": "user", "content": "hello"}
      {"role": "assistant", "content": [
         {"type": "text", "text": "..."},
         {"type": "tool_use", "id": "tu_1", "name": "f", "input": {...}}
      ]}
      {"role": "user", "content": [
         {"type": "tool_result", "tool_use_id": "tu_1", "content": "..."}
      ]}

    Shape them however your internal service expects. For this stub, we
    simply pass through — real code should convert tool_use / tool_result
    into whatever the service's schema calls them.
    """
    # TODO: translate to your service's shape.
    return list(messages)


def _to_acme_tools(tools: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Translate tool schemas.

    Input tools are Anthropic-native:
      {"name": "...", "description": "...", "input_schema": {...}}

    OpenAI, for example, wraps them:
      {"type": "function", "function": {"name": ..., "parameters": ...}}

    Your internal service may use yet another shape.
    """
    # TODO: translate to your service's shape.
    return list(tools)


def _map_stop_reason(raw: str | None) -> str:
    """Map your service's stop-reason strings to the agent's canonical values.

    The agent understands: "end_turn", "tool_use", "max_tokens", "max_iterations".
    Anything else passes through and is visible in the final AgentEvent — fine
    for diagnostics, but prefer to map to the canonical set when possible.
    """
    if raw in (None, "done", "stop", "complete"):
        return "end_turn"
    if raw in ("tool_call", "function_call", "tool_use"):
        return "tool_use"
    if raw in ("length", "max_tokens"):
        return "max_tokens"
    return raw or "end_turn"


# ---------------------------------------------------------------------------
# 4. Usage
# ---------------------------------------------------------------------------


if __name__ == "__main__":
    client = FakeAcmeClient()
    adapter = AcmeLLMAdapter(client, model="acme-sonnet")
    agent = Agent(llm=adapter)
    agent.chat()  # same CLI experience as the built-in adapters
