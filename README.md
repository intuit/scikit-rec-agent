# scikit-rec-agent

[![CI](https://github.com/intuit/scikit-rec-agent/actions/workflows/ci.yml/badge.svg)](https://github.com/intuit/scikit-rec-agent/actions/workflows/ci.yml)
[![PyPI](https://img.shields.io/pypi/v/scikit-rec-agent)](https://pypi.org/project/scikit-rec-agent/)
[![License](https://img.shields.io/badge/license-Apache--2.0-blue)](LICENSE)
[![Python](https://img.shields.io/pypi/pyversions/scikit-rec-agent)](https://pypi.org/project/scikit-rec-agent/)

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
| `transform_data` | Reshapes a raw file into one of nine scikit-rec contracts (long, long-with-timestamp, long-multi-reward, wide multi-output, multiclass, prebuilt sequences, sessions, users features, items features). Auto-detects source shape; applies pivot, melt, aggregate, dedupe, and cast as needed. Preserves user features across wide↔long reshapes. Surfaces a `dropped_targets` manifest when single-class ITEM_* columns are auto-dropped. |
| `create_datasets` | Builds scikit-rec Dataset handles from file paths. Auto-generates schemas from dtypes; auto-dispatches to `InteractionsDataset` / `InteractionMultiOutputDataset` / `InteractionMultiClassDataset`. For wide multi-output / multi-class bundles, auto-merges user features into the interactions frame (the wide scorers reject a separate users frame). Refuses bad joins upfront via a USER_ID overlap check. |
| `split_data` | Splits a bundle into train/valid/test using temporal, leave-last-n-per-user, random-split-per-user, leave-n-users-out, or random-split. Errors loudly on degenerate splits (e.g. per-user split on one-row-per-user data). |
| `list_compatible_options` | Drives the **hierarchical model-design flow**: walks the user through recommender_type → scorer_type → estimator_type → model_type → hyperparameters one step at a time. Each option carries a `what_it_is / when_to_pick / tradeoff_vs_alternatives` triple. The terminal step returns an `assembled_config` that plugs straight into `train_model`. |
| `train_model` | Trains a recommender from a `RecommenderConfig` dict via scikit-rec's factory. Accepts a `scorer_config` block (e.g. `{'on_degenerate_target': 'constant'}` to enable MultioutputScorer's constant-predictor fallback for single-class targets). Auto-picks a curated default config when none is supplied. Failure envelopes carry a `category` from the diagnose registry plus a one-line `hint`. |
| `sweep_methods` | Trains and evaluates multiple methods on the same bundle and returns a ranked leaderboard. Modes: `list` (menu only — adds a `reshape_recommendation` field when wide_multioutput could be widened by melting to long), `auto` (data-aware filter + hyperparameter resize), `all` (every entry — requires `confirmed_all=True`), `broad` (every capability-compatible triple), or explicit method dicts / short_names. Required to set `drop_non_winners` explicitly on >100K-row bundles. Idempotent across re-runs. |
| `diagnose_training_failure` | Pattern-matches a failed `train_model` envelope against a 26-pattern registry and returns ranked candidate fixes with structured actions. Auto-retries the top safe fix; bounded by `max_retries` to prevent loops. Multioutput-specific patterns (binary-only targets, retriever incompatibility, item_subset rejection, users-frame rejection, single-class targets) walk before generic sklearn fallbacks so the more-specific diagnosis fires first. |
| `evaluate_model` | Runs offline evaluation on a trained model with any of 7 evaluator types × 9 metrics at multiple k values. Auto-builds `eval_kwargs` from the bundle's validation interactions for the `simple` evaluator (including the wide multi-output shape — `(n_users, n_targets)` logged_rewards from ITEM_* columns). Returns per-target metrics with `per_label=True` on MultioutputScorer (classification + regression) or long-format UniversalScorer (roc_auc / pr_auc). Non-@k metrics (rmse, mae, roc_auc, pr_auc) compute once regardless of k_values. |
| `compare_models` | Renders a markdown leaderboard across all (or a chosen subset of) trained models in the session, sorted by a primary metric. |
| `run_hpo` | Optuna-driven hyperparameter search over a user-specified `search_space`. Persists the best config and writes the tuned model into the session. |
| `save_model` | Persists a trained model to the local file-based registry with optional tags. |
| `list_models` | Lists saved models in the registry with their metadata and tags. |
| `load_model` | Restores a saved model into the current session for further use. |

The system prompt is built at import time from scikit-rec's live enum maps, so new recommender / scorer / estimator types get picked up automatically.

### Multi-output / multi-target workflows

The wide_multioutput contract — one row per user, several `ITEM_*` columns as joint prediction targets — is fully supported end-to-end:

- **Classifier and regressor modes**: binary `ITEM_*` columns route to MultioutputScorer (classifier); continuous `ITEM_*` route to regressor mode. The auto-sweep ships both `xgb_multioutput` and `xgb_multioutput_regression`; the data profile picks the right one based on the column dtype.
- **Per-target metrics**: pass `per_label=True` to `evaluate_model` or `sweep_methods` to get `Dict[str, float]` keyed by ITEM_* name. The macro-averaged scalar is the default; per-target is the deliberate "show me each label" path.
- **Degenerate single-class targets**: `transform_data` auto-drops them by default and lists them in `dropped_targets`. To keep them with a constant-predictor fallback, pass `scorer_config={'on_degenerate_target': 'constant'}` to `train_model` instead — `degenerate_targets` then surfaces in the train envelope.
- **Long-format equivalent**: melt the wide contract into long_interactions with `transform_data` to broaden the comparison to the universal-scorer methods (XGBoost, MF, NCF, Two-Tower, DCN, NFM). Side features are preserved across the reshape.

### "Ask before deciding" — programmatic backstop

Five user-decision points are enforced as `MissingDecision` error envelopes at the tool layer (not just prompt guidance) so an LLM that ignores the system prompt can't silently default through them:

- **Primary metric** — required schema field on `sweep_methods` and `compare_models`.
- **`per_label`** — required on multioutput bundles with ≥2 ITEM_* targets (default None → MissingDecision).
- **`drop_non_winners`** — required on bundles >100K rows (default None → MissingDecision). MF + NCF + Two-Tower together can hold 1–3 GB of user embeddings.
- **`methods='all'`** — requires `confirmed_all=True` so the menu-pick flow isn't bypassed.
- **Reshape vs stay** — `sweep_methods(methods='list')` on wide_multioutput surfaces a `reshape_recommendation` field for the agent to relay.

When you receive a `MissingDecision` envelope, read its `message` for the question and re-call the tool with the user's answer in the named parameter.

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

The agent calls `diagnose_training_failure`, pattern-matches the error against a 26-pattern registry, returns ranked candidate fixes with structured actions. Bounded retries (max 2 per `model_name`) prevent loops; if the category is `unknown`, it surfaces the raw error to you instead of guessing.

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
