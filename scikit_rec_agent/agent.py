"""Agent loop — the orchestrator that ties an LLM to the tool belt.

Responsibilities:
  - Build system prompt and tool schemas from registered Tool objects
  - Stream LLM output to the caller as events (text_delta, tool_call, tool_result, done)
  - Execute tool calls against the Session
  - Append conversation history to the Session

Internal message format is Anthropic-native (content blocks); the OpenAI adapter
translates on the way in and out. See llm/base.py for the BaseLLM protocol.
"""

from __future__ import annotations

import ast
import json
import re
import sys
from dataclasses import dataclass
from typing import Any, Iterator

from scikit_rec_agent.llm.base import BaseLLM, ToolCall
from scikit_rec_agent.session import Session
from scikit_rec_agent.tools import Tool, err, get_default_tools

MAX_TOOL_ITERATIONS = 20

# Shipped adapters have no web retrieval, so any URL in model output is
# potentially fabricated. Flag deterministically rather than asking the LLM
# to self-flag — the metacognition that would catch the hallucination is the
# same thing that missed it.
_URL_PATTERN = re.compile(r"https?://\S+")
EXTERNAL_REFERENCE_WARNING = (
    "External references detected. URLs, dataset names, and citations may be "
    "hallucinated — verify before acting on them."
)

# The scikit-rec factory and agent tool loop validate configs at runtime, so
# library APIs have a backstop. External libraries (pandas/sklearn/torch/etc.)
# don't — the model may invent signatures that look plausible but don't match
# the installed version. Flag any foreign import in generated code blocks.
_GROUNDED_IMPORT_ROOTS = frozenset({"skrec", "scikit_rec", "scikit_rec_agent"})
_PYTHON_BLOCK_PATTERN = re.compile(r"```(?:python|py)\s*\n(.*?)```", re.DOTALL | re.IGNORECASE)


@dataclass
class AgentEvent:
    """Event yielded by chat_turn. Extends LLMStreamEvent with tool_result events
    the Agent loop emits after dispatching a tool call.
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
    ):
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
        self._tools_by_name = {t.name: t for t in self.tools}

    # ----- public API -----

    def chat_turn(self, user_message: str) -> Iterator[AgentEvent]:
        """Drive a single user turn to completion. Yields events as they occur.

        The loop continues until the LLM returns a response with no tool calls
        (end_turn). Each tool call is dispatched against self.session before
        the LLM is re-invoked.
        """
        self.session.messages.append({"role": "user", "content": user_message})
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
                turn_text = "".join(turn_text_parts)
                if _URL_PATTERN.search(turn_text):
                    yield AgentEvent(type="warning", text=EXTERNAL_REFERENCE_WARNING)
                foreign_imports = _foreign_imports_in_code_blocks(turn_text)
                if foreign_imports:
                    yield AgentEvent(
                        type="warning",
                        text=(
                            "Python examples reference external libraries — signatures unverified: "
                            f"{', '.join(sorted(foreign_imports))}. Check against installed versions."
                        ),
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


def _foreign_imports_in_code_blocks(text: str) -> set[str]:
    """Return the set of foreign import roots in any fenced ```python block.

    Library APIs we own (skrec / scikit_rec / scikit_rec_agent) are backstopped
    by the factory and tool loop at runtime. Everything else is ungrounded, so
    a foreign import in a code example is a plausibility — not a verified
    signature. Unparseable blocks are silently skipped to avoid false positives
    on pseudocode or truncated snippets.
    """
    foreign: set[str] = set()
    stdlib = getattr(sys, "stdlib_module_names", frozenset())
    for match in _PYTHON_BLOCK_PATTERN.finditer(text):
        try:
            tree = ast.parse(match.group(1))
        except SyntaxError:
            continue
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    root = alias.name.split(".", 1)[0]
                    if root not in _GROUNDED_IMPORT_ROOTS and root not in stdlib:
                        foreign.add(root)
            elif isinstance(node, ast.ImportFrom) and node.module and node.level == 0:
                root = node.module.split(".", 1)[0]
                if root not in _GROUNDED_IMPORT_ROOTS and root not in stdlib:
                    foreign.add(root)
    return foreign
