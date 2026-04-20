"""Agent loop — the orchestrator that ties an LLM to the tool belt.

Responsibilities:
  - Build system prompt and tool schemas from registered Tool objects
  - Stream LLM output to the caller as events (text_delta, tool_call,
    tool_result, warning, done)
  - Execute tool calls against the Session
  - Append conversation history to the Session
  - Run post-hoc hallucination safeguards on the model's final text

Internal message format is Anthropic-native (content blocks); the OpenAI
adapter translates on the way in and out. See llm/base.py for the BaseLLM
protocol. See safeguards.py for URL and foreign-reference detection.
"""

from __future__ import annotations

import json
import sys
from dataclasses import dataclass
from typing import Any, Iterator

from scikit_rec_agent.llm.base import BaseLLM, ToolCall
from scikit_rec_agent.safeguards import (
    EXTERNAL_REFERENCE_WARNING,
    URL_PATTERN,
    detect_foreign_references,
    detect_novel_urls,
)
from scikit_rec_agent.session import Session
from scikit_rec_agent.tools import Tool, err, get_default_tools

MAX_TOOL_ITERATIONS = 20


@dataclass
class AgentEvent:
    """Event yielded by chat_turn.

    Extends LLMStreamEvent with tool_result events emitted by the dispatch
    loop and warning events emitted by the end-of-turn safeguards.
    """

    type: str  # "text_delta" | "tool_call" | "tool_result" | "warning" | "done"
    text: str | None = None
    tool_call: ToolCall | None = None
    tool_result: dict[str, Any] | None = None
    tool_call_id: str | None = None
    stop_reason: str | None = None


class Agent:
    def __init__(
        self,
        llm: BaseLLM,
        tools: list[Tool] | None = None,
        system_prompt: str | None = None,
        session: Session | None = None,
        enable_safeguards: bool = True,
    ):
        """Construct an Agent.

        Args:
            enable_safeguards: when True (default), end-of-turn URL and
                foreign-code warnings are emitted as ``AgentEvent(type=
                "warning")`` — see :mod:`scikit_rec_agent.safeguards` for
                what is and is not detected. Set to False to suppress all
                safeguard warnings (useful for callers that display their
                own disclaimers or for testing).
        """
        from scikit_rec_agent.prompts import DEFAULT_SYSTEM_PROMPT

        # Catch adapters that forgot a method or misspelled one at
        # construction time rather than surfacing AttributeError mid-turn.
        missing = [m for m in ("chat", "chat_stream") if not callable(getattr(llm, m, None))]
        if missing:
            raise TypeError(
                f"llm argument does not implement BaseLLM: missing {missing}. "
                "See scikit_rec_agent.llm.base.BaseLLM for the required interface."
            )

        self.llm = llm
        self.tools = tools if tools is not None else get_default_tools()
        self.system_prompt = system_prompt if system_prompt is not None else DEFAULT_SYSTEM_PROMPT
        self.session = session if session is not None else Session()
        self.enable_safeguards = enable_safeguards
        self._tools_by_name = {t.name: t for t in self.tools}

    # ----- public API -----

    def chat_turn(self, user_message: str) -> Iterator[AgentEvent]:
        """Drive a single user turn to completion. Yields events as they occur.

        The loop continues until the LLM returns a response with no tool calls
        (end_turn). Each tool call is dispatched against self.session before
        the LLM is re-invoked.
        """
        self.session.messages.append({"role": "user", "content": user_message})
        self.session.user_supplied_urls.update(URL_PATTERN.findall(user_message))
        tool_schemas = [t.as_llm_schema() for t in self.tools]
        turn_text_parts: list[str] = []

        for _ in range(MAX_TOOL_ITERATIONS):
            text_parts: list[str] = []
            tool_calls: list[ToolCall] = []
            stop_reason: str | None = None

            for event in self.llm.chat_stream(
                messages=self.session.messages,
                tools=tool_schemas,
                system=self.system_prompt,
            ):
                if event.type == "text_delta" and event.text:
                    text_parts.append(event.text)
                    turn_text_parts.append(event.text)
                    yield AgentEvent(type="text_delta", text=event.text)
                elif event.type == "tool_call" and event.tool_call:
                    tool_calls.append(event.tool_call)
                    yield AgentEvent(type="tool_call", tool_call=event.tool_call)
                elif event.type == "done":
                    stop_reason = event.stop_reason

            text = "".join(text_parts)
            assistant_content = _build_assistant_content(text, tool_calls)
            if assistant_content:
                # Skip empty assistant turns. Anthropic's API rejects messages
                # with content=[] on the next invocation; happens in practice
                # with low max_tokens, mid-stream truncations, or refusal stops.
                self.session.messages.append({"role": "assistant", "content": assistant_content})

            if not tool_calls:
                reason = stop_reason or "end_turn"
                if not assistant_content:
                    reason = "empty_response"
                if self.enable_safeguards:
                    yield from _emit_safeguard_warnings(
                        "".join(turn_text_parts),
                        self.session.user_supplied_urls,
                    )
                yield AgentEvent(type="done", stop_reason=reason)
                return

            tool_result_blocks = []
            for call in tool_calls:
                result = self._dispatch_tool(call)
                tool_result_blocks.append(
                    {
                        "type": "tool_result",
                        "tool_use_id": call.id,
                        "content": json.dumps(result),
                    }
                )
                yield AgentEvent(
                    type="tool_result",
                    tool_call_id=call.id,
                    tool_result=result,
                )

            self.session.messages.append({"role": "user", "content": tool_result_blocks})

        yield AgentEvent(
            type="done",
            stop_reason="max_iterations",
        )

    def chat(self, stream=sys.stdout) -> None:
        """Interactive REPL convenience method. Library users typically drive
        chat_turn themselves; this is for quick manual use.
        """
        try:
            while True:
                user_input = input("you> ").strip()
                if not user_input:
                    continue
                if user_input.lower() in {"/exit", "/quit", "exit", "quit"}:
                    return
                for event in self.chat_turn(user_input):
                    if event.type == "text_delta" and event.text:
                        stream.write(event.text)
                        stream.flush()
                    elif event.type == "tool_call" and event.tool_call:
                        stream.write(f"\n[tool_call] {event.tool_call.name}({json.dumps(event.tool_call.arguments)})\n")
                        stream.flush()
                    elif event.type == "tool_result" and event.tool_result:
                        status = event.tool_result.get("status")
                        stream.write(f"[tool_result:{status}]\n")
                        stream.flush()
                    elif event.type == "warning" and event.text:
                        stream.write(f"\n[warning] {event.text}\n")
                        stream.flush()
                    elif event.type == "done":
                        stream.write("\n")
                        stream.flush()
        except (EOFError, KeyboardInterrupt):
            stream.write("\n")
            return

    # ----- internals -----

    def _dispatch_tool(self, call: ToolCall) -> dict[str, Any]:
        tool = self._tools_by_name.get(call.name)
        if tool is None:
            return err(
                "UnknownTool",
                f"No tool named '{call.name}'. Registered tools: {sorted(self._tools_by_name)}",
            )
        try:
            return tool.fn(**call.arguments, session=self.session)
        except TypeError as e:
            return err("ArgumentError", str(e), hint="Check the tool's input_schema for required fields.")
        except Exception as e:
            return err(type(e).__name__, str(e))


def _build_assistant_content(text: str, tool_calls: list[ToolCall]) -> list[dict[str, Any]]:
    blocks: list[dict[str, Any]] = []
    if text:
        blocks.append({"type": "text", "text": text})
    for call in tool_calls:
        blocks.append(
            {
                "type": "tool_use",
                "id": call.id,
                "name": call.name,
                "input": call.arguments,
            }
        )
    return blocks


def _emit_safeguard_warnings(turn_text: str, echoed_urls: set[str]) -> Iterator[AgentEvent]:
    """Yield warning events for hallucination risks found in the turn's text.

    Two distinct warnings so the user can act on each independently:
    - Novel URLs that the user didn't supply this session (possible URL
      fabrication)
    - Foreign package roots referenced in Python code blocks (unverified
      signatures from libraries the factory / tool loop can't validate)
    """
    if detect_novel_urls(turn_text, echoed_urls):
        yield AgentEvent(type="warning", text=EXTERNAL_REFERENCE_WARNING)
    foreign = detect_foreign_references(turn_text)
    if foreign:
        yield AgentEvent(
            type="warning",
            text=(
                "Python examples reference external libraries — signatures unverified: "
                f"{', '.join(sorted(foreign))}. Check against installed versions."
            ),
        )
