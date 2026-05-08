"""Capture LLM outputs from the Intuit-gateway agent into halluc_corpus.jsonl.

Per-turn invocation: each call runs ONE chat_turn, persists Session state
to a per-theme pickle, and appends one row to the corpus JSONL. Multi-turn
elicitation within a theme is achieved by repeated invocations sharing the
same --theme.

The captured row contains the verbatim LLM output, the trusted-URL set the
detector would see, and live safeguard warnings for cross-reference.
Ground-truth labels (expected_url_set, expected_foreign_set) are filled in
afterward by a human applying Rule A; this driver leaves those as null.

Usage:
    python examples/build_corpus.py \\
        --theme dataset_hunt \\
        --prompt "where can I download MovieLens 25M?"
"""

from __future__ import annotations

import argparse
import datetime
import json
import pickle
import sys
from pathlib import Path

_HERE = Path(__file__).resolve()
_REPO_ROOT = _HERE.parent.parent

# Mirror private/chat.py's path setup so IntuitLLM and IntuitLLMAdapter import.
_INTUIT_AUTH_PARENT = Path("/Users/ssankararam/Shankar/FY2026/SBG-AI/basic_llm_call_genos")
if str(_INTUIT_AUTH_PARENT) not in sys.path:
    sys.path.insert(0, str(_INTUIT_AUTH_PARENT))
sys.path.insert(0, str(_REPO_ROOT / "private"))

from intuit_adapter import IntuitLLMAdapter  # noqa: E402
from intuit_auth.intuit_llm import IntuitLLM  # noqa: E402

from scikit_rec_agent import Agent  # noqa: E402
from scikit_rec_agent.prompts import DEFAULT_SYSTEM_PROMPT  # noqa: E402
from scikit_rec_agent.safeguards import SAFEGUARDS_VERSION  # noqa: E402
from scikit_rec_agent.session import Session  # noqa: E402

CORPUS_PATH = _REPO_ROOT / "examples" / "halluc_corpus.jsonl"
STATE_DIR = _REPO_ROOT / "examples" / "_corpus_state"
SYS_PROMPTS_PATH = _REPO_ROOT / "examples" / "system_prompts.json"
SYS_PROMPT_ID = "default_v1.0"


def _ensure_system_prompts_file() -> None:
    """Snapshot the agent's default system prompt under SYS_PROMPT_ID once."""
    if SYS_PROMPTS_PATH.exists():
        return
    SYS_PROMPTS_PATH.parent.mkdir(parents=True, exist_ok=True)
    SYS_PROMPTS_PATH.write_text(json.dumps({SYS_PROMPT_ID: DEFAULT_SYSTEM_PROMPT}, indent=2) + "\n")


def _load_session(theme: str) -> Session:
    path = STATE_DIR / f"{theme}.pkl"
    if path.exists():
        with path.open("rb") as f:
            return pickle.load(f)
    return Session()


def _save_session(theme: str, session: Session) -> None:
    STATE_DIR.mkdir(parents=True, exist_ok=True)
    path = STATE_DIR / f"{theme}.pkl"
    with path.open("wb") as f:
        pickle.dump(session, f)


def _next_id() -> str:
    n = 0
    if CORPUS_PATH.exists():
        with CORPUS_PATH.open() as f:
            n = sum(1 for _ in f)
    return f"halluc_{n:04d}"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--theme", required=True, help="elicitation theme; per-theme Session state is persisted")
    parser.add_argument("--prompt", required=True, help="user message to send")
    parser.add_argument(
        "--source",
        default="synthetic_elicitation",
        choices=["synthetic_elicitation", "real_session", "adversarial_handcrafted"],
    )
    parser.add_argument("--reset", action="store_true", help="drop any saved state for this theme before running")
    args = parser.parse_args()

    _ensure_system_prompts_file()

    if args.reset:
        state_path = STATE_DIR / f"{args.theme}.pkl"
        if state_path.exists():
            state_path.unlink()

    session = _load_session(args.theme)
    turn_index = sum(1 for m in session.messages if m["role"] == "user")

    adapter = IntuitLLMAdapter(IntuitLLM)
    agent = Agent(llm=adapter, session=session)

    text_parts: list[str] = []
    warnings: list[str] = []
    for event in agent.chat_turn(args.prompt):
        if event.type == "text_delta" and event.text:
            text_parts.append(event.text)
        elif event.type == "warning" and event.text:
            warnings.append(event.text)

    llm_output = "".join(text_parts)
    _save_session(args.theme, session)

    row = {
        "id": _next_id(),
        "source": args.source,
        "theme": args.theme,
        "turn_index": turn_index,
        "user_message": args.prompt,
        "system_prompt_id": SYS_PROMPT_ID,
        "trusted_urls": sorted(session.user_supplied_urls),
        "llm_output": llm_output,
        "model": adapter.model,
        "generation_date": datetime.date.today().isoformat(),
        "safeguards_version": SAFEGUARDS_VERSION,
        "live_warnings": warnings,
        # Filled in after the fact by a human applying Rule A. Null = unlabeled.
        "expected_url_flag": None,
        "expected_url_set": None,
        "expected_foreign_flag": None,
        "expected_foreign_set": None,
        "label_notes": None,
    }

    CORPUS_PATH.parent.mkdir(parents=True, exist_ok=True)
    with CORPUS_PATH.open("a") as f:
        f.write(json.dumps(row) + "\n")

    print(f"\n--- {row['id']} (theme={args.theme}, turn={turn_index}, model={adapter.model}) ---")
    print(llm_output)
    if warnings:
        print(f"\n[live detector emitted {len(warnings)} warning(s):]")
        for w in warnings:
            print(f"  - {w}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
