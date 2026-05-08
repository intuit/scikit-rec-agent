"""Session and ModelHandle — in-memory state that persists across turns.

The Agent loop mutates a Session via tool calls. Only metadata ever enters the
LLM context: dataset summaries, model_ids, configs, metrics. The actual
InteractionsDataset and BaseRecommender objects live here and are referenced by
handle.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any


@dataclass
class DatasetBundle:
    """A set of scikit-rec Dataset objects registered under a single bundle_id.

    `interactions` is required; the rest are optional. `valid_interactions` and
    `test_interactions` exist when the user (or the split_data tool) has split
    the source data. Users and items are reference catalogs and are NOT split.
    """

    bundle_id: str
    interactions: Any  # skrec.dataset.InteractionsDataset (or a multi-output / multi-class subclass)
    users: Any = None
    items: Any = None
    valid_interactions: Any = None
    test_interactions: Any = None
    schema_paths: dict[str, str] = field(default_factory=dict)
    source_paths: dict[str, str] = field(default_factory=dict)
    # "interactions" (long format), "interaction_multioutput" (wide multi-output),
    # or "interaction_multiclass". split_data and downstream tools that need to
    # rebuild Dataset objects from CSVs read this to pick the right Dataset class.
    dataset_type: str = "interactions"


@dataclass
class ModelHandle:
    model_id: str
    name: str
    config: dict[str, Any]  # RecommenderConfig
    recommender: Any = None  # skrec.recommender.BaseRecommender
    training_time_seconds: float = 0.0
    datasets_used: dict[str, Any] = field(default_factory=dict)
    metrics: dict[str, float] = field(default_factory=dict)  # "NDCG_at_k@10" → 0.347
    tags: list[str] = field(default_factory=list)
    created_at: str = ""
    # Set to True after the first successful evaluate_model populated
    # recommendation scores on the recommender's evaluation_session. Used to
    # skip redundant score_items_kwargs passes across subsequent calls so
    # scikit-rec's internal cache is honored.
    score_cache_populated: bool = False


@dataclass
class FailureRecord:
    """One failed train_model attempt — registered automatically by the
    train_model tool's except blocks. Diagnoses key off this list to know
    what was already tried for a given model_name."""

    model_name: str
    config: dict[str, Any]
    bundle_args: dict[str, Any]
    error_envelope: dict[str, Any]
    diagnosis_category: str
    timestamp: str
    fix_applied: dict[str, Any] | None = None
    retry_outcome: dict[str, Any] | None = None


@dataclass
class Session:
    loaded_datasets: dict[str, DatasetBundle] = field(default_factory=dict)
    trained_models: dict[str, ModelHandle] = field(default_factory=dict)
    failure_history: list[FailureRecord] = field(default_factory=list)
    # Sweep-side deterministic-id → train_model auto-id alias. Lets
    # sweep_methods cache "have I trained this (bundle, config) before?"
    # without mutating the keys of trained_models, so the model_id that
    # train_model returns to its caller stays stable.
    sweep_cache: dict[str, str] = field(default_factory=dict)
    messages: list[dict[str, Any]] = field(default_factory=list)
    # URLs the URL-echo check treats as trusted: anything the user typed or
    # pasted this session, plus URLs present in the agent's system prompt
    # (seeded once in Agent.__init__). The detector subtracts this set from
    # URLs in model output so we don't warn on "you said X; here's X back"
    # nor on the model quoting a doc link from its own instructions.
    user_supplied_urls: set[str] = field(default_factory=set)


def new_model_id(recommender_type: str) -> str:
    """Generate a deterministic, collision-resistant model_id."""
    return f"{recommender_type}_{int(time.time() * 1000)}"
