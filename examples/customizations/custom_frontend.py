"""Example: drive the Agent from something other than a terminal REPL.

The CLI is a thin wrapper over Agent.chat_turn() — any frontend (Jupyter,
Slack, web UI) can be built by calling chat_turn and consuming the event
iterator. Below is a minimal event-handler pattern suitable for a Jupyter
notebook or a Slack bot listener.
"""

from __future__ import annotations

from scikit_rec_agent import Agent


def run_one_turn(agent: Agent, user_message: str) -> None:
    """Drive a single turn and dispatch events to UI-specific handlers."""
    for event in agent.chat_turn(user_message):
        if event.type == "text_delta" and event.text:
            on_text_delta(event.text)
        elif event.type == "tool_call" and event.tool_call:
            on_tool_call(event.tool_call.name, event.tool_call.arguments)
        elif event.type == "tool_result" and event.tool_result:
            on_tool_result(event.tool_result.get("status"), event.tool_result)
        elif event.type == "done":
            on_done(event.stop_reason)


# ------ Plug these into your frontend ------


def on_text_delta(text: str) -> None:
    print(text, end="", flush=True)


def on_tool_call(name: str, arguments: dict) -> None:
    print(f"\n[→ {name}({arguments})]", flush=True)


def on_tool_result(status: str, result: dict) -> None:
    print(f"[← {status}]", flush=True)


def on_done(stop_reason: str | None) -> None:
    print(f"\n(turn complete: {stop_reason})", flush=True)


if __name__ == "__main__":
    import anthropic

    from scikit_rec_agent.llm.anthropic import AnthropicAdapter

    agent = Agent(llm=AnthropicAdapter(anthropic.Anthropic()))
    while True:
        msg = input("> ").strip()
        if not msg or msg.lower() in {"/exit", "/quit"}:
            break
        run_one_turn(agent, msg)
