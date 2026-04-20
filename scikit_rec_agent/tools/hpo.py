"""run_hpo tool.

Wraps skrec.orchestrator.hpo.HyperparameterOptimizer. Resolves a bundle's
train/validation datasets, runs Optuna optimization, persists results as
parquet, and (by default) re-trains the best config as a fresh model
registered in the session.
"""

from __future__ import annotations

import copy
import os
import tempfile
import time
from datetime import datetime, timezone
from typing import Any

from scikit_rec_agent.session import ModelHandle, new_model_id
from scikit_rec_agent.tools import Tool, err, ok


def _deep_update_dot(source: dict, overrides: dict) -> dict:
    """Apply dot-notation overrides into nested dict. Same semantics as
    skrec.orchestrator.hpo.deep_update, copied here to avoid depending on a
    private helper."""
    import collections.abc

    for key, value in overrides.items():
        keys = key.split(".")
        d = source
        for k in keys[:-1]:
            if k not in d or not isinstance(d[k], collections.abc.MutableMapping):
                d[k] = {}
            d = d[k]
        final = keys[-1]
        if isinstance(value, collections.abc.Mapping) and value:
            if final in d and isinstance(d[final], collections.abc.MutableMapping):
                _deep_update_dot(d[final], value)
            else:
                d[final] = value
        else:
            d[final] = value
    return source


def _run_hpo(
    study_name: str,
    base_config: dict[str, Any],
    search_space: dict[str, Any],
    metric_definitions: list[str],
    objective_metric: str,
    bundle_id: str,
    n_trials: int,
    session,
    sampler: str = "tpe",
    direction: str = "maximize",
    evaluator_type: str = "simple",
    persistence_path: str | None = None,
    retrain_best: bool = True,
    retrain_model_name: str | None = None,
) -> dict[str, Any]:
    from skrec.orchestrator import create_recommender_pipeline
    from skrec.orchestrator.hpo import HyperparameterOptimizer

    bundle = session.loaded_datasets.get(bundle_id)
    if bundle is None:
        return err("BundleNotFound", f"No bundle '{bundle_id}' in session.")
    if bundle.valid_interactions is None:
        return err(
            "MissingValidation",
            f"Bundle '{bundle_id}' has no validation interactions. Call split_data first.",
            hint="Run split_data with strategy='temporal' or 'leave_last_n_per_user' before run_hpo.",
        )

    base_config = dict(base_config)
    base_config.setdefault("recommender_params", {})

    if persistence_path is None:
        tmp = tempfile.mkdtemp(prefix=f"skragent_hpo_{study_name}_")
        persistence_path = os.path.join(tmp, f"{study_name}.parquet")

    try:
        optimizer = HyperparameterOptimizer(
            base_config=base_config,
            search_space=search_space,
            metric_definitions=metric_definitions,
            training_interactions_ds=bundle.interactions,
            validation_interactions_ds=bundle.valid_interactions,
            training_users_ds=bundle.users,
            training_items_ds=bundle.items,
            validation_users_ds=bundle.users,
            evaluator_type=evaluator_type,
            persistence_path=persistence_path,
        )
    except Exception as e:
        return err(type(e).__name__, f"HPO init failed: {e}")

    try:
        results_df, study = optimizer.run_optimization(
            n_trials=n_trials,
            objective_metric=objective_metric,
            sampler=sampler,
            direction=direction,
            study_name=study_name,
        )
    except Exception as e:
        return err(type(e).__name__, f"HPO run failed: {e}")

    best_params = dict(study.best_params)
    best_value = float(study.best_value)
    n_complete = sum(1 for t in study.trials if t.state.name == "COMPLETE")

    retrained_model_id: str | None = None
    if retrain_best and best_params:
        final_config = copy.deepcopy(base_config)
        _deep_update_dot(final_config, best_params)
        try:
            recommender = create_recommender_pipeline(final_config)
        except Exception as e:
            return err(type(e).__name__, f"retrain pipeline build failed: {e}")

        train_kwargs: dict[str, Any] = {"interactions_ds": bundle.interactions}
        if bundle.users is not None:
            train_kwargs["users_ds"] = bundle.users
        if bundle.items is not None:
            train_kwargs["items_ds"] = bundle.items
        if bundle.valid_interactions is not None:
            train_kwargs["valid_interactions_ds"] = bundle.valid_interactions
            if bundle.users is not None:
                train_kwargs["valid_users_ds"] = bundle.users

        started = time.time()
        try:
            recommender.train(**train_kwargs)
        except Exception as e:
            return err(type(e).__name__, f"retrain failed: {e}")
        elapsed = time.time() - started

        recommender_type = final_config.get("recommender_type", "unknown")
        retrained_model_id = new_model_id(f"{recommender_type}_hpo")
        handle = ModelHandle(
            model_id=retrained_model_id,
            name=retrain_model_name or f"{study_name}_best",
            config=final_config,
            recommender=recommender,
            training_time_seconds=elapsed,
            datasets_used={
                "bundle_id": bundle.bundle_id,
                "source_paths": dict(bundle.source_paths),
            },
            tags=["hpo_best", study_name],
            created_at=datetime.now(timezone.utc).isoformat(),
        )
        session.trained_models[retrained_model_id] = handle

    return ok(
        {
            "study_name": study_name,
            "best_params": best_params,
            "best_value": best_value,
            "n_complete_trials": n_complete,
            "results_parquet_path": persistence_path,
            "retrained_model_id": retrained_model_id,
            "n_trials_in_table": int(len(results_df)),
        }
    )


TOOL_RUN_HPO = Tool(
    name="run_hpo",
    description=(
        "Run Optuna hyperparameter optimization on a base RecommenderConfig. Requires a "
        "bundle with both training and validation interactions (run split_data first). "
        "By default, the best config is re-trained on the same data after the search and "
        "registered as a new model_id in the session. Supports TPE, GP, CMA-ES, random, "
        "grid, and QMC samplers."
    ),
    input_schema={
        "type": "object",
        "properties": {
            "study_name": {"type": "string"},
            "base_config": {"type": "object", "description": "RecommenderConfig with fixed values."},
            "search_space": {
                "type": "object",
                "description": (
                    "Dot-notation param paths → dimension specs. Each spec is "
                    "{type: int|float|categorical, low, high, step?, log?, choices?}. "
                    "Example: {'estimator_config.xgboost.n_estimators': {type: 'int', low: 50, high: 500, step: 50}}."
                ),
            },
            "metric_definitions": {
                "type": "array",
                "items": {"type": "string"},
                "description": "Metric names like 'NDCG@10' or 'MAP@5'.",
            },
            "objective_metric": {"type": "string"},
            "bundle_id": {"type": "string"},
            "n_trials": {"type": "integer"},
            "sampler": {
                "type": "string",
                "enum": ["tpe", "gp", "cmaes", "random", "grid", "qmc"],
                "default": "tpe",
            },
            "direction": {"type": "string", "enum": ["maximize", "minimize"], "default": "maximize"},
            "evaluator_type": {
                "type": "string",
                "enum": ["simple", "replay_match", "IPS", "DR", "direct_method", "SNIPS", "policy_weighted"],
                "default": "simple",
            },
            "persistence_path": {"type": "string"},
            "retrain_best": {"type": "boolean", "default": True},
            "retrain_model_name": {"type": "string"},
        },
        "required": [
            "study_name",
            "base_config",
            "search_space",
            "metric_definitions",
            "objective_metric",
            "bundle_id",
            "n_trials",
        ],
    },
    fn=_run_hpo,
)
