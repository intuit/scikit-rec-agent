# Contributing

Thanks for your interest. This project welcomes bug reports, questions, and pull requests.

## Development setup

The agent depends on [scikit-rec](https://github.com/intuit/scikit-rec), so install both editable from sibling checkouts:

```bash
git clone https://github.com/intuit/scikit-rec-agent.git
git clone https://github.com/intuit/scikit-rec.git

cd scikit-rec-agent
python3 -m venv .venv && source .venv/bin/activate
pip install -e ../scikit-rec
pip install -e ".[dev,anthropic,openai]"
```

## Running tests and lint

```bash
pytest                              # full suite (offline — mocked LLMs)
pytest tests/test_safeguards.py     # one file
pytest -k compare_models            # one test
ruff check . && ruff format --check .
```

Tests run offline — no `ANTHROPIC_API_KEY` / `OPENAI_API_KEY` needed. Those are only required when you actually `scikit-rec-agent chat`.

## Style

- `ruff` with the repo's `pyproject.toml` config (line length 120, isort, pyflakes) is the source of truth. Run it before committing.
- Keep comments sparse — explain *why*, not *what*. Identifier names should do the descriptive work.
- Match the existing module layout: one responsibility per file, no cross-imports between `tools/`, `llm/`, and `prompts/` beyond the shared protocols.

## Commit messages

Conventional-ish prefix + one-line summary:

```
feat: <what's new>
fix: <what broke>
refactor: <non-behavior-changing reorg>
docs: <readme / docstrings / examples>
test: <tests only>
```

Focus the message on the *why* rather than the *what*. The diff already shows what changed.

## Where contributions fit best

- **Bug fixes with regression tests** — always welcome.
- **New LLM adapters** implementing the `BaseLLM` protocol — welcome. See [`examples/custom_llm.py`](./examples/custom_llm.py) for the template.
- **New tools** — register via the `Tool` dataclass ([`examples/custom_tool.py`](./examples/custom_tool.py)). Library-level additions (vs. user-registered at construction) should motivate why every user benefits.
- **Prompt / heuristic refinements** — open an issue first to discuss. The system prompt is opinionated on purpose.
- **Dataset fetching, web tools, codegen helpers** — out of scope for v1. See [`agentic_design.md`](./agentic_design.md).

## Safeguards module scope

[`scikit_rec_agent/safeguards.py`](./scikit_rec_agent/safeguards.py) declares a versioned contract (`SAFEGUARDS_VERSION`) describing exactly what the hallucination detectors catch and — equally important — what they don't. PRs that expand the detector surface should first open an issue: scope creep is the main risk.

## Filing issues

Include:
- Python version, `scikit-rec` version, `scikit-rec-agent` version
- LLM provider + model if relevant
- Minimal reproducer (a CLI transcript or a `chat_turn` snippet)
- Expected vs. actual behavior

For security-relevant reports, please contact the maintainers privately before opening a public issue.

## License

By contributing you agree your contributions are licensed under [Apache-2.0](./LICENSE).
