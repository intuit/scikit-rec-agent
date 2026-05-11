# scikit-rec-agent: Design Document

Conversational AI agent that uses scikit-rec as its tool belt. The agent reasons about the user's data and goals, then calls scikit-rec APIs via structured tool use to build, evaluate, and compare recommendation systems.

This document is the authoritative spec for the implementation. It reflects the decisions locked in during design review and the factory contract provided by scikit-rec PR landed on 2026-04-17 (commits `74a773c` + `137d278` + `5bdc7d0`).

---

## Design Decisions

| Decision | Choice | Rationale |
|---|---|---|
| Distribution | **Single pip package** (`scikit-rec-agent`) | Installable library with CLI entry point. Examples live in `examples/`, no separate cookbook repo. |
| LLM provider | **Bring-your-own** via `BaseLLM` protocol | Users pass any LLM that implements `chat()` + `chat_stream()`. Ship Anthropic + OpenAI adapters at launch. |
| System prompt | **Swappable at `Agent()` construction** | Default prompt exported; users pass `system_prompt=...` to override or extend. |
| Tool registry | **Pluggable at `Agent()` construction** | 11 default tools ship with the library; users extend or replace via `tools=...`. |
| Interface | **CLI** for v1 | `scikit-rec-agent chat`. Jupyter/web layered on top of `Agent` later. |
| Model registry | **Local filesystem** | `~/.scikit-rec/registry/` — JSON metadata + pickle. |
| Tool scope | **11 tools, everything in v1** | No v1.1 tier. If it's worth shipping, it ships now. |
| Recommender scope | **Full scikit-rec capability matrix** | All 6 recommenders × 6 scorers × 3 estimator planes. Driven end-to-end via `create_recommender_pipeline`. |
| `suggest_pipelines` | **In-prompt reasoning, not a tool** | The LLM emits candidate `RecommenderConfig` dicts as text; `train_model` validates via the factory. |
| Config validation | **Delegated to scikit-rec factory** | Agent does not re-implement enum checks. Bad configs fail at `train_model` with the factory's error message surfaced to the LLM. |
| Streaming | **Yes** | Stream LLM text deltas to terminal for responsive UX during long tool executions. |

---

## Architecture

```
User (CLI)
  |
  v
Agent Loop (BaseLLM protocol + tool dispatch + streaming)
  |
  |--- BaseLLM protocol ----+---- AnthropicAdapter (Claude)
  |                          +---- OpenAIAdapter (GPT-4)
  |                          +---- UserCustomAdapter (anything)
  |
  v
Tools Layer (10 structured tool-use functions)
  |
  v
scikit-rec
  |
  |-- skrec.orchestrator.create_recommender_pipeline(config)
  |     Recommender -> Scorer -> Estimator
  |-- skrec.orchestrator.HyperparameterOptimizer (used by run_hpo)
  |-- skrec.dataset.{Interactions,Users,Items}Dataset
  |
  v
Model Registry (~/.scikit-rec/registry/)
```

The agent is stateful across turns. `Session` holds loaded datasets, trained pipeline handles, and evaluation results. Tools mutate this session; model objects themselves never enter the LLM context — only `model_id` handles and metadata.

---

## Prerequisite: scikit-rec Factory Contract

The agent depends on a single entry point:

```python
from skrec.orchestrator import create_recommender_pipeline, RecommenderConfig
```

`create_recommender_pipeline(config: RecommenderConfig) -> BaseRecommender` builds the full Estimator → Scorer → Recommender chain from a dict. It covers the entire scikit-rec capability matrix (post PR `74a773c`):

### Recommender types
`ranking`, `bandits`, `sequential`, `hierarchical_sequential`, `uplift`, `gcsl`

### Scorer types
`universal`, `independent`, `multiclass`, `multioutput`, `sequential`, `hierarchical`

### Estimator planes (`estimator_type` discriminator)
- `tabular` — XGBoost classifier/regressor, MultiOutputClassifier (LightGBM, sklearn wrappers available directly but not via factory enum today — acceptable for v1)
- `embedding` (`model_type` ∈ {`matrix_factorization`, `ncf`, `two_tower`, `deep_cross_network`, `neural_factorization`})
- `sequential` (`model_type` ∈ {`sasrec_classifier`, `sasrec_regressor`, `hrnn_classifier`, `hrnn_regressor`})

### Required fields
- `recommender_type` — **required**, raises `ValueError` if missing or `None`.
- `scorer_type` — **required**, raises `ValueError` if missing.
- `estimator_config` — required; `estimator_type` defaults to `"tabular"`.
- `recommender_params` — required only for recommenders that need them (e.g. `uplift` requires `control_item_id`). Keys irrelevant to the chosen recommender are silently ignored.

### Cross-cutting validators the factory already enforces
The agent relies on these and does **not** re-implement them:

- `sequential` / `hierarchical_sequential` recommenders require `estimator_type="sequential"`
- `sequential` recommender requires `scorer_type="sequential"`
- `hierarchical_sequential` requires `scorer_type="hierarchical"`
- `sequential` / `hierarchical` scorers require `estimator_type="sequential"`
- `embedding` estimators are rejected by `multioutput` / `multiclass` / `independent` scorers
- `uplift` recommender requires `scorer_type ∈ {"independent", "universal"}`

When a bad config reaches `train_model`, the factory raises `ValueError` / `TypeError` / `NotImplementedError`. The tool captures the message verbatim and returns it as a tool error — the LLM reads the error and corrects the config without the agent needing a parallel validator.

### Canonical config shapes (copy these into the system prompt)

```python
# 1. Tabular ranking
{
  "recommender_type": "ranking",
  "scorer_type": "universal",
  "estimator_config": {
    "ml_task": "classification",
    "xgboost": {"n_estimators": 100, "max_depth": 5, "learning_rate": 0.1},
  },
}

# 2. Embedding ranking (Two-Tower / NCF / MF / DCN / NFM)
{
  "recommender_type": "ranking",
  "scorer_type": "universal",
  "estimator_config": {
    "estimator_type": "embedding",
    "embedding": {"model_type": "two_tower", "params": {"embedding_dim": 32}},
  },
}

# 3. Sequential (SASRec / HRNN)
{
  "recommender_type": "sequential",
  "scorer_type": "sequential",
  "estimator_config": {
    "estimator_type": "sequential",
    "sequential": {"model_type": "sasrec_classifier", "params": {"hidden_units": 64, "max_len": 50}},
  },
  "recommender_params": {"max_len": 50},
}

# 4. Uplift (T-Learner / S-Learner / X-Learner)
{
  "recommender_type": "uplift",
  "scorer_type": "independent",
  "estimator_config": {"ml_task": "classification", "xgboost": {"n_estimators": 100}},
  "recommender_params": {"control_item_id": "control", "mode": "t_learner"},
}

# 5. GCSL (multi-objective)
{
  "recommender_type": "gcsl",
  "scorer_type": "universal",
  "estimator_config": {"ml_task": "classification", "xgboost": {"n_estimators": 100}},
  "recommender_params": {
    "inference_method": {
      "type": "predefined_value",
      "params": {"goal_values": {"OUTCOME_revenue": 1.0}},
    },
  },
}

# 6. Contextual bandits
{
  "recommender_type": "bandits",
  "scorer_type": "universal",
  "estimator_config": {"ml_task": "classification", "xgboost": {"n_estimators": 100}},
}
```

### XGBoost hyperparameter hints
`XGBConfig` is a `TypedDict(total=False)` with these typed keys (full passthrough — any other XGBoost param is accepted too):
`n_estimators`, `max_depth`, `learning_rate`, `subsample`, `colsample_bytree`, `colsample_bynode`, `objective`, `eval_metric`, `n_jobs`, `random_state`.

---

## LLM Provider Abstraction

Users bring their own LLM. The agent depends on a `BaseLLM` protocol, not a specific SDK.

### Protocol

```python
from typing import Protocol, Iterator, Any

class BaseLLM(Protocol):
    def chat(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]],
        system: str,
    ) -> LLMResponse:
        """Non-streaming chat completion with tool definitions.

        Args:
            messages: Conversation history in OpenAI-style format.
            tools: List of tool schemas (JSON Schema format, provider-agnostic).
            system: System prompt string.

        Returns:
            LLMResponse with text content and/or tool calls.
        """
        ...

    def chat_stream(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]],
        system: str,
    ) -> Iterator[LLMStreamEvent]:
        """Streaming variant. Yields text deltas and tool calls.

        Tool calls are atomic events — they're only emitted once fully received.
        Only text deltas stream token-by-token.
        """
        ...
```

### Response types

```python
@dataclass
class ToolCall:
    id: str              # Unique ID for this tool call
    name: str            # Tool name (e.g. "profile_data")
    arguments: dict      # Parsed JSON arguments

@dataclass
class LLMResponse:
    content: str | None          # Text response (may be None if only tool calls)
    tool_calls: list[ToolCall]   # Zero or more tool calls
    stop_reason: str             # "end_turn", "tool_use", "max_tokens", etc.

@dataclass
class LLMStreamEvent:
    type: str                    # "text_delta" | "tool_call" | "done"
    text: str | None = None
    tool_call: ToolCall | None = None
    stop_reason: str | None = None
```

### Built-in adapters

Both ship in v1:

```python
class AnthropicAdapter(BaseLLM):
    """Wraps anthropic.Anthropic client to BaseLLM protocol."""
    def __init__(self, client: "anthropic.Anthropic", model: str = "claude-sonnet-4-6"):
        ...

class OpenAIAdapter(BaseLLM):
    """Wraps openai.OpenAI client to BaseLLM protocol."""
    def __init__(self, client: "openai.OpenAI", model: str = "gpt-4o"):
        ...
```

### Usage

```python
from scikit_rec_agent import Agent
from scikit_rec_agent.llm import AnthropicAdapter
import anthropic

llm = AnthropicAdapter(anthropic.Anthropic(), model="claude-sonnet-4-6")
agent = Agent(llm=llm)
agent.chat()  # interactive CLI session
```

```python
# Or bring your own
from scikit_rec_agent import Agent, BaseLLM

class MyLLM(BaseLLM):
    def chat(self, messages, tools, system): ...
    def chat_stream(self, messages, tools, system): ...

agent = Agent(llm=MyLLM())
```

---

## Session State

The agent is stateful across turns. The `Session` dataclass holds:

```python
@dataclass
class Session:
    loaded_datasets: dict[str, dict]       # path -> {profile, dataset_objects}
    trained_models: dict[str, ModelHandle] # model_id -> handle
    messages: list[dict]                   # conversation history

@dataclass
class ModelHandle:
    model_id: str                # e.g. "twotower_1712345678"
    name: str                    # human-readable name
    config: RecommenderConfig    # the config passed to train_model
    recommender: BaseRecommender # the actual trained pipeline
    training_time_seconds: float
    datasets_used: dict          # paths/schema info
    metrics: dict[str, float]    # metric_name@k -> value (accumulates across evaluate_model calls)
    tags: list[str]
    created_at: str              # ISO timestamp
```

### What enters the LLM context
Only `model_id`, `name`, config, metrics, training time, and status messages. The `recommender` object stays in Python memory and is referenced by `model_id` from tool calls.

### `model_id` generation
`{model_type}_{unix_timestamp}` — e.g. `twotower_1712345678`. Deterministic enough to be collision-free in a session, readable enough for the LLM to reference unambiguously.

---

## Extension Points

### 1. System prompt

```python
from scikit_rec_agent import Agent
from scikit_rec_agent.prompts import DEFAULT_SYSTEM_PROMPT

custom_prompt = DEFAULT_SYSTEM_PROMPT + "\n\nOur team uses NDCG@10 as the primary metric."
agent = Agent(llm=llm, system_prompt=custom_prompt)
```

### 2. Tool registry

```python
from scikit_rec_agent import Agent
from scikit_rec_agent.tools import Tool
from scikit_rec_agent import get_default_tools

def fetch_from_snowflake(query: str, session: Session) -> dict:
    ...

custom_tool = Tool(
    name="fetch_from_snowflake",
    schema={...},  # JSON schema
    fn=fetch_from_snowflake,
)

agent = Agent(llm=llm, tools=[*get_default_tools(), custom_tool])
```

Tool functions receive the `Session` as a keyword arg so user-defined tools can read and mutate the same state.

### 3. CLI / frontend

`scikit-rec-agent chat` is thin glue: constructs an `Agent`, reads stdin, prints streamed output. Users who want Jupyter, Slack, or a web UI instantiate `Agent` directly and drive it with their own I/O loop. `Agent.chat_turn(user_message)` returns an event iterator — the CLI has no privileged access.

---

## Agent Tools (15 tools)

| Tool | Purpose | Wraps |
|---|---|---|
| `profile_data` | Load CSV/parquet; report shape, dtypes, cardinality, sparsity, temporal range, target type | pandas + heuristics |
| `validate_data` | Schema-compliance check against scikit-rec required schemas. Returns violations + auto-fix suggestions. | Compare against `InteractionsDataset.REQUIRED_SCHEMA_PATH_TRAINING` etc. |
| `transform_data` | Reshape a raw file into one of nine scikit-rec contracts via composable ops (rename, pivot, melt, aggregate, dedupe, cast, parse_timestamp). Auto-detects source shape; preserves user features across wide↔long reshapes; surfaces a `dropped_targets` manifest when single-class ITEM_* columns are auto-dropped. | Internal op registry |
| `create_datasets` | Build `InteractionsDataset` / `UsersDataset` / `ItemsDataset` handles. Auto-generate YAML schema to tmp dir if not provided. Supports `column_mapping` to rename user columns → scikit-rec names. Auto-merges user features into the interactions frame for wide multi-output / multi-class bundles and refuses bad joins via a USER_ID overlap check. | `DatasetSchema.create` + dataset constructors |
| `split_data` | Split a bundle's interactions into train/valid/test using a recsys-appropriate strategy (temporal, leave_last_n_per_user, random_split_per_user, leave_n_users_out, random_split). Updates the bundle in place. | `skrec.split` |
| `list_compatible_options` | Drives the hierarchical model-design flow: recommender_type → scorer_type → estimator_type → model_type → hyperparameters, one step at a time. Each option carries a `what_it_is / when_to_pick / tradeoff_vs_alternatives` triple. Terminal step returns an `assembled_config` for `train_model`. | Live capability-matrix introspection |
| `train_model` | Train a recommender pipeline from a `RecommenderConfig`. Accepts a `scorer_config` block (e.g. `{'on_degenerate_target': 'constant'}`) for scorer-level knobs. Creates datasets internally if called with paths; uses pre-built datasets if called with dataset handles. Auto-picks a curated default config when none is supplied. | `create_recommender_pipeline` + `.train()` |
| `sweep_methods` | Train + evaluate multiple methods on the same bundle and return a ranked leaderboard. Modes: `list` / `auto` / `all` (requires `confirmed_all=True`) / `broad` / explicit method dicts or short_names. `MissingDecision` on >100K-row bundles without explicit `drop_non_winners`. | Iterates `train_model` + `evaluate_model` |
| `diagnose_training_failure` | Pattern-match a failed `train_model` envelope against the 26-pattern registry and return ranked candidate fixes. Multioutput-specific patterns walk before generic sklearn fallbacks. Auto-retries the top safe fix; bounded by `max_retries`. | Internal `_REGISTRY` walk |
| `evaluate_model` | Run evaluation: evaluator type + metrics + multiple k values. Supports all 7 evaluator types × 9 metrics. Auto-builds `eval_kwargs` from validation interactions (including the wide multi-output `(n_users, n_targets)` shape). `per_label=True` returns Dict[label, value] for MultioutputScorer (all metrics) or long-format UniversalScorer (roc_auc / pr_auc). | `BaseRecommender.evaluate()` |
| `compare_models` | Tabulate metrics across trained models. Markdown table sorted by primary metric. | Session state lookup |
| `run_hpo` | Optuna-based hyperparameter optimization. Requires a bundle with validation interactions (use `split_data` first). Returns best config + trial results, optionally retrains the best config and registers it. | `HyperparameterOptimizer.run_optimization()` |
| `save_model` | Persist model + config + metrics to local registry | pickle + JSON metadata |
| `list_models` | List models in the local registry (not just session) with metadata. | Filesystem scan of `~/.scikit-rec/registry/` |
| `load_model` | Load a registered model into the current session. | pickle + session state mutation |

**`suggest_pipelines` is deliberately NOT a tool.** Model selection happens either through the design flow (`list_compatible_options` walks the user through each axis) or through the sweep flow (`sweep_methods` runs the menu + leaderboard). The factory validates configs on entry — there's no need for a separate Python-side validator.

The v1 spec below documents the schemas for the original 11 tools. The four added since (`transform_data`, `list_compatible_options`, `sweep_methods`, `diagnose_training_failure`) follow the same input/output envelope contract; see their `Tool` dataclass declarations in [`scikit_rec_agent/tools/`](./scikit_rec_agent/tools/) for the live JSON schemas.

### Tool error contract

Every tool's return value is a JSON-serializable dict with a consistent envelope:

```python
# Success
{"status": "ok", "data": {...}}

# Error (factory raised, file missing, evaluation failed, etc.)
{"status": "error", "error_type": "ValueError", "message": "...", "hint": "optional suggestion"}
```

Both shapes are passed back as the tool result. The LLM reads the `message` field and self-corrects on error. `hint` is used for high-confidence fixes we can synthesize locally (e.g. `"Your column 'user' was detected as the user ID — pass column_mapping={'user': 'USER_ID'}"`).

---

## Tool Schemas (v1)

### profile_data

```json
{
  "name": "profile_data",
  "description": "Load and profile a data file. Reports shape, dtypes, cardinality of ID columns, sparsity, value distributions, temporal range, and whether the target looks implicit (binary) or explicit (ratings).",
  "input_schema": {
    "type": "object",
    "properties": {
      "file_path": {"type": "string", "description": "Path to CSV or parquet file"},
      "file_type": {"type": "string", "enum": ["interactions", "users", "items"]}
    },
    "required": ["file_path", "file_type"]
  }
}
```
**Returns:** `shape`, `columns` (name, dtype, null_count, n_unique, sample_values), `id_columns_detected`, `target_column_detected`, `target_type` (binary/rating/continuous), `temporal_range` (if timestamp found), `duplicate_pairs_count`, `sparsity`.

### validate_data

```json
{
  "name": "validate_data",
  "description": "Validate a data file against scikit-rec required schemas. Reports missing required columns, wrong dtypes, and suggests column renames if near-matches are detected.",
  "input_schema": {
    "type": "object",
    "properties": {
      "file_path": {"type": "string"},
      "file_type": {"type": "string", "enum": ["interactions", "users", "items"]},
      "is_training": {"type": "boolean", "default": true}
    },
    "required": ["file_path", "file_type"]
  }
}
```
**Returns:** `valid` (bool), `missing_columns`, `wrong_dtypes`, `suggested_column_mapping` (fuzzy-matched renames), `extra_columns` (passed through as features).

### create_datasets

```json
{
  "name": "create_datasets",
  "description": "Build scikit-rec Dataset handles. Auto-generates YAML schemas from the data types if client_schema_path is not provided. Applies column_mapping to rename columns to USER_ID/ITEM_ID/OUTCOME as needed. Registers the handles in the session under a dataset_bundle_id.",
  "input_schema": {
    "type": "object",
    "properties": {
      "bundle_id": {"type": "string"},
      "interactions_path": {"type": "string"},
      "users_path": {"type": "string"},
      "items_path": {"type": "string"},
      "column_mapping": {
        "type": "object",
        "description": "Map user's column names to scikit-rec names, e.g. {\"userid\": \"USER_ID\", \"clicked\": \"OUTCOME\"}"
      },
      "schemas": {
        "type": "object",
        "description": "Optional pre-written YAML schema paths keyed by file_type"
      }
    },
    "required": ["bundle_id", "interactions_path"]
  }
}
```
**Returns:** `bundle_id`, paths to generated schema files (so user can inspect / version them), summary of the three datasets. Also supports optional `valid_interactions_path` and `test_interactions_path` for users who pre-split their data.

### split_data

```json
{
  "name": "split_data",
  "description": "Split a dataset bundle's interactions into train/validation/test using a recommender-systems-appropriate strategy. Updates the bundle in place: the bundle's interactions becomes the training split, and valid_interactions / test_interactions are populated. Strategies: temporal (chronological, production-realistic default); leave_last_n_per_user (sequential-model standard); random_split_per_user (per-user random holdout, preserves all users in train); leave_n_users_out (full user holdout — only honest cold-start eval); random_split (rarely appropriate — sanity checks only).",
  "input_schema": {
    "type": "object",
    "properties": {
      "bundle_id": {"type": "string"},
      "strategy": {"type": "string", "enum": ["temporal", "leave_last_n_per_user", "random_split_per_user", "leave_n_users_out", "random_split"]},
      "valid_fraction": {"type": "number"},
      "test_fraction": {"type": "number", "default": 0.0},
      "n_valid": {"type": "integer"},
      "n_test": {"type": "integer", "default": 0},
      "n_valid_users": {"type": "integer"},
      "n_test_users": {"type": "integer", "default": 0},
      "user_col": {"type": "string", "default": "USER_ID"},
      "timestamp_col": {"type": "string", "default": "TIMESTAMP"},
      "random_state": {"type": "integer"}
    },
    "required": ["bundle_id", "strategy"]
  }
}
```
**Returns:** `bundle_id`, `strategy`, `train_rows`, `valid_rows`, `test_rows`, `paths` (to the temp CSVs), and `info` (strategy-specific diagnostics: date ranges for temporal, dropped users for leave_last_n_per_user, etc.).

### train_model

```json
{
  "name": "train_model",
  "description": "Train a recommender pipeline from a RecommenderConfig. Supply either a dataset bundle_id from create_datasets, OR raw file paths with optional column_mapping (train_model will call create_datasets internally). If the bundle has validation interactions attached (from split_data), they are used automatically. Config is validated by scikit-rec's factory — bad configs raise with a specific error that you can use to correct and retry.",
  "input_schema": {
    "type": "object",
    "properties": {
      "model_name": {"type": "string"},
      "config": {
        "type": "object",
        "description": "RecommenderConfig dict: recommender_type, scorer_type, estimator_config, optional recommender_params. See system prompt for canonical shapes."
      },
      "bundle_id": {"type": "string", "description": "From create_datasets. If provided, paths/column_mapping are ignored."},
      "interactions_path": {"type": "string"},
      "users_path": {"type": "string"},
      "items_path": {"type": "string"},
      "column_mapping": {"type": "object"}
    },
    "required": ["model_name", "config"]
  }
}
```
**Returns:** `model_id`, `model_name`, `status`, `training_time_seconds`, `estimator_type`, `scorer_type`, `recommender_type`.

### evaluate_model

```json
{
  "name": "evaluate_model",
  "description": "Evaluate a trained model using offline evaluation. Supports all 7 evaluator types and all 9 metrics at multiple k values. Results cached on the recommender's evaluation_session — subsequent calls with the same eval_kwargs are free.",
  "input_schema": {
    "type": "object",
    "properties": {
      "model_id": {"type": "string"},
      "evaluator_type": {
        "type": "string",
        "enum": ["simple", "replay_match", "IPS", "DR", "direct_method", "SNIPS", "policy_weighted"]
      },
      "metrics": {
        "type": "array",
        "items": {
          "type": "string",
          "enum": ["NDCG_at_k", "MAP_at_k", "MRR_at_k", "precision_at_k", "recall_at_k", "average_reward_at_k", "roc_auc", "pr_auc", "expected_reward"]
        }
      },
      "k_values": {"type": "array", "items": {"type": "integer"}},
      "eval_kwargs": {"type": "object", "description": "logged_items, logged_rewards, logging_proba, expected_rewards — as required by the evaluator type"}
    },
    "required": ["model_id", "evaluator_type", "metrics", "k_values"]
  }
}
```
**Returns:** `model_id`, `results` (list of `{metric, k, value}`).

### compare_models

```json
{
  "name": "compare_models",
  "description": "Compare trained models in the current session. Returns a markdown leaderboard sorted by primary metric.",
  "input_schema": {
    "type": "object",
    "properties": {
      "model_ids": {"type": "array", "items": {"type": "string"}, "description": "If empty, compares all trained models in the session."},
      "primary_metric": {"type": "string"},
      "k": {"type": "integer"}
    },
    "required": ["primary_metric", "k"]
  }
}
```
**Returns:** markdown table (models × metrics) + JSON version.

### run_hpo

```json
{
  "name": "run_hpo",
  "description": "Run Optuna hyperparameter optimization on a base RecommenderConfig. Supports TPE, GP, CMA-ES, random, grid, QMC samplers. Results persisted to a parquet file keyed by study_name.",
  "input_schema": {
    "type": "object",
    "properties": {
      "study_name": {"type": "string"},
      "base_config": {"type": "object", "description": "RecommenderConfig with fixed values"},
      "search_space": {
        "type": "object",
        "description": "Dot-notation param paths → dimension specs. Each spec is {type: int|float|categorical, low, high, step?, log?, choices?}. Example: {'estimator_config.xgboost.n_estimators': {type: 'int', low: 50, high: 500, step: 50}}"
      },
      "metric_definitions": {
        "type": "array",
        "items": {"type": "string"},
        "description": "Metric names like 'NDCG@10' or 'MAP@5'."
      },
      "objective_metric": {"type": "string"},
      "bundle_id": {"type": "string", "description": "Dataset bundle from create_datasets — must include validation datasets."},
      "n_trials": {"type": "integer"},
      "sampler": {"type": "string", "enum": ["tpe", "gp", "cmaes", "random", "grid", "qmc"], "default": "tpe"},
      "direction": {"type": "string", "enum": ["maximize", "minimize"], "default": "maximize"}
    },
    "required": ["study_name", "base_config", "search_space", "metric_definitions", "objective_metric", "bundle_id", "n_trials"]
  }
}
```
**Returns:** `best_params`, `best_value`, `n_complete_trials`, `results_parquet_path`, and a `model_id` if the best config is automatically re-trained at the end (configurable via `retrain_best`, default `true`).

### save_model

```json
{
  "name": "save_model",
  "description": "Persist a trained model, its config, and evaluation metrics to the local registry (~/.scikit-rec/registry/<model_name>/).",
  "input_schema": {
    "type": "object",
    "properties": {
      "model_id": {"type": "string"},
      "tags": {"type": "array", "items": {"type": "string"}}
    },
    "required": ["model_id"]
  }
}
```
**Returns:** `registry_path`, `model_name`, `saved_at`.

### list_models

```json
{
  "name": "list_models",
  "description": "List all models in the local registry (persistent — not just current session). Returns metadata and metrics so the user can choose one to load.",
  "input_schema": {
    "type": "object",
    "properties": {
      "tag_filter": {"type": "array", "items": {"type": "string"}},
      "recommender_type_filter": {"type": "string"}
    }
  }
}
```
**Returns:** list of `{model_name, model_id, recommender_type, tags, saved_at, metrics}`.

### load_model

```json
{
  "name": "load_model",
  "description": "Load a registered model into the current session. Subsequent evaluate_model / compare_models / save_model calls can reference it by model_id.",
  "input_schema": {
    "type": "object",
    "properties": {
      "model_name": {"type": "string"}
    },
    "required": ["model_name"]
  }
}
```
**Returns:** `model_id`, `name`, `config`, `metrics` from registry.

---

## Conversation Flow

### Typical session

```
Turn 1 — User describes data
  User: "I have click-stream data with user features and an item catalog"
  Agent: Asks for file paths, clarifies goal (ranking? bandit? sequential?)

Turn 2 — Profile + validate
  Agent calls: profile_data (x3) + validate_data (x3)
  Agent: "1.2M interactions across 50K users / 10K items. Target is binary (clicks).
          Timestamps span 6 months. User features: age, gender, segment.
          Item features: category, price. Sparsity 99.76%. 342 duplicate pairs.
          Your columns 'userid' / 'itemid' / 'clicked' need to be mapped to
          USER_ID / ITEM_ID / OUTCOME — I'll pass column_mapping automatically."

Turn 3 — Agent suggests pipelines IN-PROMPT
  Agent (text, no tool call): "Given binary implicit feedback with timestamps
     and side features, three candidates:

     1. XGBoost + UniversalScorer + Ranking — fast baseline, handles features well.
        {recommender_type: ranking, scorer_type: universal,
         estimator_config: {ml_task: classification,
                            xgboost: {n_estimators: 200, max_depth: 6}}}

     2. Two-Tower + UniversalScorer + Ranking — learns user/item embeddings,
        robust to cold-start.
        {recommender_type: ranking, scorer_type: universal,
         estimator_config: {estimator_type: embedding,
                            embedding: {model_type: two_tower,
                                        params: {embedding_dim: 64}}}}

     3. SASRec + SequentialScorer + Sequential — exploits timestamps.
        {recommender_type: sequential, scorer_type: sequential,
         estimator_config: {estimator_type: sequential,
                            sequential: {model_type: sasrec_classifier,
                                         params: {hidden_units: 64, max_len: 50}}},
         recommender_params: {max_len: 50}}

     Train all three?"

Turn 4 — User confirms
  User: "Yes, train all three."
  Agent calls: create_datasets (auto-schema + column_mapping)
  Agent calls: train_model x3
  Agent: "XGBoost: 45s. Two-Tower: 3m 12s. SASRec: 8m 41s."

Turn 5 — Evaluate
  Agent calls: evaluate_model x3 (Simple, NDCG/Precision/MAP @ 10)
  Agent calls: compare_models
  Agent: "Two-Tower wins. NDCG@10: XGB 0.312, TT 0.347, SASRec 0.338."

Turn 6 — HPO
  User: "Optimize Two-Tower."
  Agent calls: run_hpo (TPE, 50 trials, embedding_dim / lr search space)
  Agent: "Best NDCG@10: 0.381 (embedding_dim=128, lr=0.003).
          Re-trained the best config as twotower_hpo_1712400000."
  Agent calls: save_model
```

### Edge cases the agent handles (via the in-prompt heuristics)

- **Missing columns**: `validate_data` detects near-matches, returns `suggested_column_mapping`; agent passes it to `create_datasets`.
- **Rating scale (1–5) vs binary**: `profile_data` reports `target_type`; agent picks `regression` vs `classification` accordingly.
- **Too sparse for embeddings**: agent warns when < 100K interactions and recommends XGBoost over Two-Tower/NCF.
- **No timestamps**: agent skips sequential candidates.
- **Causal evaluation**: agent asks for `logging_proba` / `expected_rewards` and sets `evaluator_type` to IPS/DR/DM.
- **Multi-outcome rewards (revenue + clicks)**: agent suggests GCSL with `predefined_value` or `mean_scalarization` inference methods.

---

## System Prompt

The default system prompt (lives in `src/scikit_rec_agent/prompts/system.py`) encodes:

1. **Role and tone** — domain expert, concise, never trains what the data can't support.
2. **scikit-rec architecture recap** — the 3-layer model, when to use each recommender type.
3. **Capability matrix** — authoritative enums for `recommender_type`, `scorer_type`, `estimator_type`, `model_type` (embedding), `model_type` (sequential), `inference_method.type`, `retriever.type`, `sampler`. **These enums should be imported from `skrec.orchestrator.factory` at prompt build time** so the prompt can't drift from the factory — e.g. read `_EMBEDDING_ESTIMATOR_MAP.keys()` directly.
4. **Canonical config shapes** — the 6 shapes from the Prereq section above, copied verbatim.
5. **Decision heuristics**:
   - Data size thresholds (when embeddings outperform XGBoost)
   - Feature availability (dense features → DeepFM, sparse → MF)
   - Sparsity bounds (embedding models need ≥ ~100K interactions)
   - Target type → `ml_task` mapping
   - Timestamps present → sequential is an option
6. **Evaluator selection**:
   - Held-out split + randomized logging → `simple`
   - Logged from production with known propensities → `IPS` / `SNIPS` / `DR`
   - Reward model available → `direct_method`
   - Exploration deployment → `replay_match` / `policy_weighted`
7. **Metric selection by use case** — implicit feedback → NDCG/MAP/Precision; revenue → expected_reward; CTR → roc_auc / pr_auc.
8. **Guardrails**:
   - Always call `validate_data` before `train_model`.
   - Don't suggest embedding models on < 100K interactions.
   - Warn about overfitting with small validation sets.
   - Flag premature HPO (run baselines first).
9. **Tool-calling discipline**:
   - `suggest_pipelines` is IN-PROMPT — emit configs in text, don't invent a tool call for it.
   - Always set both `recommender_type` AND `scorer_type` explicitly.
   - On factory errors, read the error message and self-correct — don't re-raise to the user.

---

## Repo Structure

```
scikit-rec-agent/
├── pyproject.toml
├── README.md
├── scikit_rec_agent/               # flat layout (matches scikit-rec)
│   ├── __init__.py                 # Exports: Agent, BaseLLM, Tool, Session, get_default_tools, DEFAULT_SYSTEM_PROMPT
│   ├── agent.py                    # Agent loop: BaseLLM + tool dispatch + streaming
│   ├── session.py                  # Session + ModelHandle dataclasses
│   ├── llm/
│   │   ├── __init__.py             # Exports BaseLLM, LLMResponse, LLMStreamEvent, ToolCall
│   │   ├── base.py                 # Protocol + dataclasses
│   │   ├── anthropic.py            # AnthropicAdapter
│   │   └── openai.py               # OpenAIAdapter
│   ├── tools/
│   │   ├── __init__.py             # get_default_tools(); Tool dataclass
│   │   ├── profiling.py            # profile_data, validate_data
│   │   ├── datasets.py             # create_datasets (incl. auto-schema generation)
│   │   ├── training.py             # train_model
│   │   ├── evaluation.py           # evaluate_model, compare_models
│   │   ├── hpo.py                  # run_hpo
│   │   └── registry.py             # save_model, list_models, load_model
│   ├── prompts/
│   │   ├── __init__.py
│   │   ├── system.py               # DEFAULT_SYSTEM_PROMPT (built at import from factory enums)
│   │   └── _capability.py          # Runtime-derived capability matrix → string
│   └── cli.py                      # Entry point: scikit-rec-agent chat
├── tests/
│   ├── fixtures/                   # Tiny CSVs + mocked LLM transcripts
│   ├── test_profiling.py
│   ├── test_datasets.py
│   ├── test_training.py
│   ├── test_evaluation.py
│   ├── test_hpo.py
│   ├── test_registry.py
│   ├── test_llm_adapters.py        # Anthropic + OpenAI with mocked API
│   └── test_agent_integration.py   # End-to-end with scripted LLM
└── examples/
    ├── customizations/
    │   ├── custom_tool.py              # Adding a user-defined tool
    │   ├── custom_prompt.py            # Overriding the default system prompt
    │   ├── custom_llm.py               # Plug in your company's internal LLM via BaseLLM
    │   └── custom_frontend.py          # Driving Agent from Jupyter / Slack / web
    └── transcripts/
        ├── movielens_session.md                # Captured sweep-flow session
        └── movielens_hierarchical_session.md   # Captured hierarchical-flow session
```

---

## Dependencies

```toml
[project]
name = "scikit-rec-agent"
requires-python = ">=3.10"
dependencies = [
    "scikit-rec>=0.3.0,<1.0.0",
]

[project.optional-dependencies]
anthropic = ["anthropic>=0.40.0"]
openai    = ["openai>=1.0.0"]
torch     = ["scikit-rec[torch]"]   # passthrough for sequential / embedding models
aws       = ["scikit-rec[aws]"]     # passthrough for S3 dataset loading
dev       = ["pytest>=7.0", "pytest-cov>=4.0", "ruff>=0.4", "mypy>=1.0"]
```

Core has zero LLM SDK dependencies. Users install the adapter they need:

```bash
pip install scikit-rec-agent[anthropic]        # Claude
pip install scikit-rec-agent[openai]           # GPT-4
pip install scikit-rec-agent[anthropic,torch]  # Claude + deep-learning models
pip install scikit-rec-agent                   # bring your own LLM
```

All ML dependencies come transitively through `scikit-rec`.

---

## Build Plan

1. **Day 1 — Skeleton**: `pyproject.toml`, `llm/{base,anthropic,openai}.py`, `session.py`, `agent.py` loop, mocked-LLM smoke test (one scripted `train_model` call end-to-end).
2. **Days 2–4 — Tools**: all 11 tools against `create_recommender_pipeline`, `skrec.split`, and `HyperparameterOptimizer`. Use `skrec.examples.datasets.sample_*` for fixtures.
3. **Days 5–6 — System prompt + CLI**: build the capability matrix from factory enums at import time (derive, don't hardcode); CLI entry with streaming; single end-to-end transcript example.
4. **Day 7 — Tests + polish**: per-tool tests, adapter tests with mocked API, end-to-end scripted-LLM integration test, README.

Out of scope for v1: Jupyter widget, web UI, MLflow registry backend, non-XGBoost tabular estimators (LightGBM / logreg / sklearn wrappers) via factory — they work today if manually constructed but aren't in the factory's enum yet, which is a scikit-rec follow-up, not an agent concern.
