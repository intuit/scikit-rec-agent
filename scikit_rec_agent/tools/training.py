"""train_model tool.

Wraps `skrec.orchestrator.create_recommender_pipeline` plus the recommender's
`.train()` call. The factory validates configs on entry; bad configs raise
ValueError/TypeError/NotImplementedError which we capture as error envelopes
so the LLM can read the message and self-correct.
"""

from __future__ import annotations

import time
from datetime import datetime, timezone
from typing import Any

from scikit_rec_agent.session import DatasetBundle, ModelHandle, new_model_id
from scikit_rec_agent.tools import Tool, err, ok
from scikit_rec_agent.tools.datasets import _create_datasets as _build_bundle


def _resolve_bundle(
    bundle_id: str | None,
    interactions_path: str | None,
    users_path: str | None,
    items_path: str | None,
    column_mapping: dict[str, str] | None,
    session,
) -> tuple[DatasetBundle | None, dict[str, Any] | None]:
    """Return (bundle, error_envelope). Exactly one is non-None."""
    if bundle_id:
        bundle = session.loaded_datasets.get(bundle_id)
        if bundle is None:
            return None, err(
                "BundleNotFound",
                f"No bundle '{bundle_id}'. Call create_datasets first or pass raw paths.",
            )
        return bundle, None
    if not interactions_path:
        return None, err(
            "MissingArgument",
            "train_model requires either bundle_id or interactions_path.",
        )
    implicit_name = f"implicit_bundle_{int(time.time() * 1000)}"
    result = _build_bundle(
        bundle_id=implicit_name,
        interactions_path=interactions_path,
        session=session,
        users_path=users_path,
        items_path=items_path,
        column_mapping=column_mapping,
    )
    if result["status"] != "ok":
        return None, result
    return session.loaded_datasets[implicit_name], None


def _train_model(
    model_name: str,
    config: dict[str, Any],
    session,
    bundle_id: str | None = None,
    interactions_path: str | None = None,
    users_path: str | None = None,
    items_path: str | None = None,
    column_mapping: dict[str, str] | None = None,
) -> dict[str, Any]:
    from skrec.orchestrator import create_recommender_pipeline

    from scikit_rec_agent.tools.diagnose import _quick_diagnose, record_failure

    if not isinstance(config, dict):
        return err("ArgumentError", "config must be a dict.")

    config = dict(config)
    config.setdefault("recommender_params", {})

    bundle_args = {
        "bundle_id": bundle_id,
        "interactions_path": interactions_path,
        "users_path": users_path,
        "items_path": items_path,
        "column_mapping": column_mapping,
    }
    bundle_args = {k: v for k, v in bundle_args.items() if v is not None}

    bundle, err_env = _resolve_bundle(bundle_id, interactions_path, users_path, items_path, column_mapping, session)
    if err_env is not None:
        return err_env
    assert bundle is not None

    try:
        recommender = create_recommender_pipeline(config)
    except (ValueError, TypeError, NotImplementedError) as e:
        diagnosis = _quick_diagnose(e)
        envelope = err(
            type(e).__name__,
            str(e),
            hint=diagnosis.first_fix_description or "Check recommender_type, scorer_type, and estimator_config.",
            category=diagnosis.category,
        )
        record_failure(session, model_name, config, bundle_args, envelope, diagnosis)
        return envelope

    import inspect

    accepted_train_kwargs = set(inspect.signature(recommender.train).parameters)

    train_kwargs: dict[str, Any] = {"interactions_ds": bundle.interactions}
    if bundle.users is not None:
        train_kwargs["users_ds"] = bundle.users
    if bundle.items is not None:
        train_kwargs["items_ds"] = bundle.items
    # Per-recommender-class kwarg differences: SequentialRecommender derives
    # its validation split internally via `use_validation: bool` and rejects
    # `valid_interactions_ds`. Filter to what the actual train() accepts so
    # the recommender_type doesn't have to be hardcoded here.
    if bundle.valid_interactions is not None and "valid_interactions_ds" in accepted_train_kwargs:
        train_kwargs["valid_interactions_ds"] = bundle.valid_interactions
    if bundle.users is not None and bundle.valid_interactions is not None and "valid_users_ds" in accepted_train_kwargs:
        train_kwargs["valid_users_ds"] = bundle.users
    if "use_validation" in accepted_train_kwargs and bundle.valid_interactions is not None:
        # Sequential recommenders take a bool flag instead of a separate
        # validation dataset — they derive the split from interactions_ds.
        train_kwargs["use_validation"] = True

    started = time.time()
    try:
        recommender.train(**train_kwargs)
    except Exception as e:
        diagnosis = _quick_diagnose(e)
        envelope = err(
            type(e).__name__,
            f"Training failed: {e}",
            hint=diagnosis.first_fix_description,
            category=diagnosis.category,
        )
        record_failure(session, model_name, config, bundle_args, envelope, diagnosis)
        return envelope
    elapsed = time.time() - started

    recommender_type = config.get("recommender_type", "unknown")
    model_id = new_model_id(str(recommender_type))
    handle = ModelHandle(
        model_id=model_id,
        name=model_name,
        config=config,
        recommender=recommender,
        training_time_seconds=elapsed,
        datasets_used={
            "bundle_id": bundle.bundle_id,
            "source_paths": dict(bundle.source_paths),
        },
        created_at=datetime.now(timezone.utc).isoformat(),
    )
    session.trained_models[model_id] = handle

    return ok(
        {
            "model_id": model_id,
            "model_name": model_name,
            "status": "trained",
            "training_time_seconds": elapsed,
            "recommender_type": config.get("recommender_type"),
            "scorer_type": config.get("scorer_type"),
            "estimator_type": config.get("estimator_config", {}).get("estimator_type", "tabular"),
        }
    )


TOOL_TRAIN_MODEL = Tool(
    name="train_model",
    description=(
        "Train a recommender pipeline from a RecommenderConfig. Supply either a dataset "
        "`bundle_id` from create_datasets, OR raw file paths (train_model will call "
        "create_datasets internally). Config is validated by scikit-rec's factory — bad "
        "configs return an error envelope you can use to correct the config and retry. "
        "If the bundle has validation interactions (from split_data), they are used "
        "automatically."
    ),
    input_schema={
        "type": "object",
        "properties": {
            "model_name": {"type": "string"},
            "config": {
                "type": "object",
                "description": (
                    "RecommenderConfig dict: recommender_type, scorer_type, estimator_config, "
                    "optional recommender_params. See system prompt for canonical shapes."
                ),
            },
            "bundle_id": {"type": "string"},
            "interactions_path": {"type": "string"},
            "users_path": {"type": "string"},
            "items_path": {"type": "string"},
            "column_mapping": {"type": "object"},
        },
        "required": ["model_name", "config"],
    },
    fn=_train_model,
)
