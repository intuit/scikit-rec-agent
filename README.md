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

See [`examples/customizations/`](./examples/customizations/) for:
- `custom_tool.py` — register a user-defined tool
- `custom_prompt.py` — extend or replace the system prompt
- `custom_llm.py` — plug in your company's internal LLM via the `BaseLLM` protocol
- `custom_frontend.py` — drive the agent from Jupyter / Slack / web

See [`examples/transcripts/`](./examples/transcripts/) for full captured chat sessions:
- `movielens_session.md` — the **sweep flow**: compare 7 methods on MovieLens-1M
- `movielens_hierarchical_session.md` — the **one-model design flow**: walk through the picker step by step on the same data

## What it does

Fifteen tools cover the full scikit-rec workflow — from raw data to a saved, tuned model:

| Tool | What it does |
|---|---|
| `profile_data` | Loads a CSV/parquet and reports shape, dtypes, sparsity, target type, and temporal range. Heuristic role detection for USER_ID / ITEM_ID / OUTCOME / TIMESTAMP. |
| `validate_data` | Checks a file against scikit-rec's required schema. Suggests column-rename mappings when names are close. |
| `transform_data` | Reshapes a raw file into one of nine scikit-rec contracts (long, long-with-timestamp, long-multi-reward, wide multi-output, multiclass, prebuilt sequences, sessions, users features, items features). Auto-detects source shape; applies pivot, melt, aggregate, dedupe, and cast as needed. |
| `create_datasets` | Builds scikit-rec Dataset handles from file paths. Auto-generates schemas from dtypes; auto-dispatches to `InteractionsDataset` / `InteractionMultiOutputDataset` / `InteractionMultiClassDataset`. |
| `split_data` | Splits a bundle into train/valid/test using temporal, leave-last-n-per-user, random-split-per-user, leave-n-users-out, or random-split. Errors loudly on degenerate splits (e.g. per-user split on one-row-per-user data). |
| `list_compatible_options` | Drives the **hierarchical model-design flow**: walks the user through recommender_type → scorer_type → estimator_type → model_type → hyperparameters one step at a time. Each option carries a `what_it_is / when_to_pick / tradeoff_vs_alternatives` triple. The terminal step returns an `assembled_config` that plugs straight into `train_model`. |
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

## How to talk to it

The agent expects natural language. There's no DSL, no required prompt structure — just describe your data and goal. Two main paths cover most workflows; the agent picks based on what you ask for.

### Path A — Compare-everything sweep

For "I want results — show me which method works best on my data."

```
"I have click data at /data/interactions.csv with users in users.csv
 and items in items.csv. Compare a few methods and tell me which works
 best."
```

What happens:
1. `profile_data` + `validate_data` on each file
2. `transform_data` if the shape doesn't match the target contract
3. `create_datasets` + `split_data`
4. `sweep_methods(methods="list")` — agent surfaces the menu (XGBoost, MF, NCF, Two-Tower, DCN, NFM, SASRec — whichever fit your data) with brief descriptions, asks you to pick or say "all"
5. `sweep_methods(methods=[...])` — trains + evaluates the picked methods, returns a ranked leaderboard
6. Agent reports the winner; offers to save / run HPO

The auto-sweep table is **data-aware** by default (`methods="auto"`): MF only runs in the high-sparsity regime, embedding methods only when n_rows ≥ 5K, sequential only with timestamps. Hyperparameters are tier-sized to your data scale. Pass `methods="all"` to override the filter and run every entry as-is.

See [`examples/transcripts/movielens_session.md`](examples/transcripts/movielens_session.md) — 7 methods on MovieLens-1M, SASRec wins with NDCG@10 ≈ 0.021.

### Path B — Design one good model, with help

For "I want to *understand* the choice, not just see a leaderboard."

```
"Walk me through how to choose a recommender for this data. I want
 to understand the design space."
```

What happens — the agent walks the **hierarchical flow** via `list_compatible_options`:

1. **`recommender_type`** — Ranking? Sequential? Uplift? Bandits? Each option carries a `what_it_is / when_to_pick / tradeoff_vs_alternatives` triple. Options that don't fit your data (e.g. sequential when there's no TIMESTAMP) are filtered out automatically.
2. **`scorer_type`** — given your previous pick, what scoring strategy applies. Universal / independent / multioutput / multiclass / sequential / hierarchical, again with explanations.
3. **`estimator_type`** — tabular (XGBoost) / embedding (MF, NCF, Two-Tower, DCN, NFM) / sequential (SASRec, HRNN). Filtered by data size (embedding needs ≥5K rows, etc).
4. **`model_type`** — pick the specific family.
5. **Terminal step** — agent shows the data-tier-sized default hyperparameters with `what_it_is` and `why_this_default`. Three actions:
   - `train_with_defaults` — accept the sized defaults, train one model
   - `train_with_overrides` — change specific hyperparameters before training
   - `run_hpo` — search the pre-suggested ranges via Optuna

**Uplift gets one extra step.** Picking `recommender_type=uplift` adds `required_recommender_params` to the terminal payload — `control_item_id` (which ITEM_ID is the control / no-recommendation case?) and `mode` (T-Learner / S-Learner / X-Learner, each with its own triple). Both are user-supplied; the agent won't silently default them. `train_with_defaults` is blocked for uplift; you go through `train_with_overrides`.

See [`examples/transcripts/movielens_hierarchical_session.md`](examples/transcripts/movielens_hierarchical_session.md) for a real walk-through on MovieLens-1M.

### Sweep vs design — which to pick

| Goal | Path |
|---|---|
| "What works best on my data?" | A — sweep |
| "Should I use sequential or ranking? Help me choose." | B — design |
| "Compare 3 specific methods I picked." | A — sweep with explicit `methods=[...]` |
| "I want uplift. Help me set it up." | B — design |
| "Bulk compare everything and run HPO on the winner." | A — sweep, then `run_hpo` on the winner |
| "Train one specific model I already have a config for." | Skip both — use `train_model` directly |

### Recover from a training failure

```
"train_model errored — here's the envelope: {error_type: ValueError,
 message: 'Input contains NaN', ...}. Help."
```

The agent calls `diagnose_training_failure`, pattern-matches the error against a 14-pattern registry, returns ranked candidate fixes with structured actions. Bounded retries (max 2 per `model_name`) prevent loops; if the category is `unknown`, it surfaces the raw error to you instead of guessing.

### What the agent will ask back

The agent asks targeted clarifying questions when the data or goal is genuinely ambiguous:

- "Which column is your timestamp?" — when `profile_data`'s heuristic role detection is uncertain
- "Ranking or sequential? Your data has timestamps so both are valid" — when a design choice has real tradeoffs
- "Your `gender` column is `M`/`F` strings. Drop it, label-encode (0/1), or one-hot?" — when `train_model` would otherwise fail on object-dtype features
- "What's your control item ID for uplift?" — when the design path lands on uplift and there's no sensible default

It does **not** ask you to write any code. Tool calls happen behind the scenes; you only see them if you watch `chat_turn`'s event stream.

### Reading the output

Every tool returns a JSON envelope. The shape is:

```json
{"status": "ok", "data": {...}}                                    // success
{"status": "error", "error_type": "...", "message": "...",         // failure
 "hint": "actionable next step", "category": "diagnostics-bucket"}
```

`category` and `hint` are populated by the diagnose registry — when present they tell you exactly what failed and how to react. Sweep leaderboards sort by your primary metric; rows with `status: "error"` are kept in the leaderboard with their per-method failure category for later inspection.

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
