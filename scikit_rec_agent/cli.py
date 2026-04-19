"""scikit-rec-agent CLI entry point.

Default behavior: auto-detect the LLM provider from env vars and launch an
interactive chat REPL. Users who want more control instantiate Agent directly
from Python — see examples/custom_frontend.py.
"""

from __future__ import annotations

import argparse
import os
import sys


def _build_adapter(provider: str, model: str | None):
    if provider == "anthropic":
        import anthropic  # lazy import

        from scikit_rec_agent.llm.anthropic import DEFAULT_MODEL, AnthropicAdapter

        return AnthropicAdapter(anthropic.Anthropic(), model=model or DEFAULT_MODEL)
    if provider == "openai":
        import openai  # lazy import

        from scikit_rec_agent.llm.openai import DEFAULT_MODEL, OpenAIAdapter

        return OpenAIAdapter(openai.OpenAI(), model=model or DEFAULT_MODEL)
    raise ValueError(f"Unknown provider '{provider}'. Valid: anthropic, openai.")


def _auto_detect_provider() -> str:
    has_anthropic = bool(os.environ.get("ANTHROPIC_API_KEY"))
    has_openai = bool(os.environ.get("OPENAI_API_KEY"))
    if has_anthropic and not has_openai:
        return "anthropic"
    if has_openai and not has_anthropic:
        return "openai"
    if has_anthropic and has_openai:
        raise SystemExit(
            "Both ANTHROPIC_API_KEY and OPENAI_API_KEY are set. Pass --provider {anthropic,openai} to disambiguate."
        )
    raise SystemExit(
        "No provider credentials found. Set ANTHROPIC_API_KEY or OPENAI_API_KEY "
        "(or pass --provider with the corresponding credentials)."
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="scikit-rec-agent")
    sub = parser.add_subparsers(dest="command", required=True)

    chat_parser = sub.add_parser("chat", help="Start an interactive chat session.")
    chat_parser.add_argument(
        "--provider",
        choices=["anthropic", "openai"],
        help="LLM provider. Auto-detected from env vars if omitted.",
    )
    chat_parser.add_argument("--model", help="Override the default model name for the chosen provider.")

    args = parser.parse_args(argv)
    if args.command != "chat":
        parser.print_help()
        return 1

    provider = args.provider or _auto_detect_provider()
    try:
        adapter = _build_adapter(provider, args.model)
    except Exception as e:
        sys.stderr.write(f"Failed to build {provider} adapter: {e}\n")
        return 2

    from scikit_rec_agent import Agent

    agent = Agent(llm=adapter)
    sys.stdout.write(f"scikit-rec-agent (provider={provider}). Type /exit to quit.\n")
    sys.stdout.flush()
    agent.chat()
    return 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
