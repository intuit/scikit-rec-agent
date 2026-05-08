# scikit-rec-agent

Conversational AI agent that uses [scikit-rec](https://github.com/intuit/scikit-rec) as its tool belt. The agent reasons about the user's data and goals, then calls scikit-rec APIs via structured tool use to build, evaluate, and compare recommendation systems.

## Install

```bash
pip install scikit-rec-agent[anthropic]     # with Claude
pip install scikit-rec-agent[openai]        # with GPT-4
pip install scikit-rec-agent                # bring your own LLM
pip install scikit-rec-agent[anthropic,torch]  # + deep-learning models
```

## CLI

```bash
export ANTHROPIC_API_KEY=...
scikit-rec-agent chat
```

Auto-detects the provider from env vars. Pass `--provider {anthropic,openai}` if both are set.

## Library

```python
import anthropic
from scikit_rec_agent import Agent
from scikit_rec_agent.llm.anthropic import AnthropicAdapter

agent = Agent(llm=AnthropicAdapter(anthropic.Anthropic()))
for event in agent.chat_turn("I have click data at /data/interactions.csv — help me build a ranker"):
    ...
```

See [`examples/`](./examples/) for:
- `custom_tool.py` — register a user-defined tool
- `custom_prompt.py` — extend or replace the system prompt
- `custom_llm.py` — plug in your company's internal LLM via the `BaseLLM` protocol
- `custom_frontend.py` — drive the agent from Jupyter / Slack / web
- `movielens_session.md` — annotated end-to-end transcript

## What it does

Fourteen tools cover the full scikit-rec workflow — from raw data to a saved, tuned model:

| Tool | What it does |
|---|---|
| `profile_data` | Loads a CSV/parquet and reports shape, dtypes, sparsity, target type, and temporal range. Heuristic role detection for USER_ID / ITEM_ID / OUTCOME / TIMESTAMP. |
| `validate_data` | Checks a file against scikit-rec's required schema. Suggests column-rename mappings when names are close. |
| `transform_data` | Reshapes a raw file into one of nine scikit-rec contracts (long, long-with-timestamp, long-multi-reward, wide multi-output, multiclass, prebuilt sequences, sessions, users features, items features). Auto-detects source shape; applies pivot, melt, aggregate, dedupe, and cast as needed. |
| `create_datasets` | Builds scikit-rec Dataset handles from file paths. Auto-generates schemas from dtypes; auto-dispatches to `InteractionsDataset` / `InteractionMultiOutputDataset` / `InteractionMultiClassDataset`. |
| `split_data` | Splits a bundle into train/valid/test using temporal, leave-last-n-per-user, random-split-per-user, leave-n-users-out, or random-split. Errors loudly on degenerate splits (e.g. per-user split on one-row-per-user data). |
| `train_model` | Trains a recommender from a `RecommenderConfig` dict via scikit-rec's factory. Failure envelopes carry a `category` from the diagnose registry plus a one-line `hint`. |
| `sweep_methods` | Trains and evaluates multiple methods on the same bundle and returns a ranked leaderboard. Modes: `list` (menu only), `auto` (data-aware filter + hyperparameter resize), `all` (every entry), `broad` (every capability-compatible triple), or explicit method dicts / short_names. Idempotent across re-runs. |
| `diagnose_training_failure` | Pattern-matches a failed `train_model` envelope against a 14-pattern registry and returns ranked candidate fixes with structured actions. Auto-retries the top safe fix; bounded by `max_retries` to prevent loops. |
| `evaluate_model` | Runs offline evaluation on a trained model with any of 7 evaluator types × 9 metrics at multiple k values. Auto-builds `eval_kwargs` from the bundle's validation interactions for the `simple` evaluator. |
| `compare_models` | Renders a markdown leaderboard across all (or a chosen subset of) trained models in the session, sorted by a primary metric. |
| `run_hpo` | Optuna-driven hyperparameter search over a user-specified `search_space`. Persists the best config and writes the tuned model into the session. |
| `save_model` | Persists a trained model to the local file-based registry with optional tags. |
| `list_models` | Lists saved models in the registry with their metadata and tags. |
| `load_model` | Restores a saved model into the current session for further use. |

The system prompt is built at import time from scikit-rec's live enum maps, so new recommender / scorer / estimator types get picked up automatically.

## Hallucination safeguards

The agent runs two deterministic detectors on every turn's output:

- **URL echo check** — flags `https://...` links the model introduces that the user did not supply this session. Shipped adapters have no web retrieval, so model-introduced URLs are common fabrications.
- **Foreign-reference check** — scans fenced Python blocks for imports and bare-alias usage outside `{skrec, scikit_rec, scikit_rec_agent, stdlib}`. Library APIs we own have a runtime backstop via the scikit-rec factory; external libraries don't.

Warnings are emitted as `AgentEvent(type="warning")` and never enter conversation history. Opt out with `Agent(..., enable_safeguards=False)`.

### Scope and limitations

The detectors are deliberately narrow. **They catch the common confident-plausible-looking fabrication case with near-zero false positives, not every possible hallucination.** What they do *not* catch:

- Semantic errors inside trusted APIs (wrong `RecommenderConfig` shape, poor metric choice). The scikit-rec factory catches bad configs at `train_model`; the rest is on the user.
- Invented keyword arguments for external libraries. We flag `pandas` as unverified, not the specific `make_up_kwarg=True`.
- Fabricated dataset names, paper citations, or prose claims. We only inspect URLs and Python code blocks.
- Adversarial evasion (aliased `importlib`, f-string import args, triple-backticks inside docstrings, `ast.parse`-rejecting blocks).

See [`scikit_rec_agent/safeguards.py`](./scikit_rec_agent/safeguards.py) for the full contract.

## Architecture

See [`agentic_design.md`](./agentic_design.md) for the authoritative spec.

## Contributing

Contributions welcome — see [CONTRIBUTING.md](./CONTRIBUTING.md) for dev setup, test commands, and where new work fits best.

## License

Apache-2.0
