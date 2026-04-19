"""Tests for train_model tool — uses a tabular ranking config against the
sample binary-reward fixture."""

from __future__ import annotations

from scikit_rec_agent.tools.datasets import TOOL_CREATE_DATASETS
from scikit_rec_agent.tools.training import TOOL_TRAIN_MODEL

_TABULAR_CONFIG = {
    "recommender_type": "ranking",
    "scorer_type": "universal",
    "estimator_config": {
        "ml_task": "classification",
        "xgboost": {"n_estimators": 20, "max_depth": 3},
    },
}


def test_train_model_from_bundle_id(binary_reward_paths, session):
    TOOL_CREATE_DATASETS.fn(
        bundle_name="b",
        interactions_path=binary_reward_paths["interactions"],
        users_path=binary_reward_paths["users"],
        items_path=binary_reward_paths["items"],
        session=session,
    )
    result = TOOL_TRAIN_MODEL.fn(
        model_name="tab_small",
        config=_TABULAR_CONFIG,
        bundle_id="b",
        session=session,
    )
    assert result["status"] == "ok"
    model_id = result["data"]["model_id"]
    assert model_id in session.trained_models
    handle = session.trained_models[model_id]
    assert handle.config["recommender_type"] == "ranking"
    assert handle.training_time_seconds >= 0


def test_train_model_factory_error_returns_envelope(binary_reward_paths, session):
    TOOL_CREATE_DATASETS.fn(
        bundle_name="b",
        interactions_path=binary_reward_paths["interactions"],
        users_path=binary_reward_paths["users"],
        items_path=binary_reward_paths["items"],
        session=session,
    )
    bad_config = {
        "recommender_type": "sequential",
        "scorer_type": "universal",  # mismatch — must be "sequential" for seq recommender
        "estimator_config": {"ml_task": "classification"},
    }
    result = TOOL_TRAIN_MODEL.fn(model_name="bad", config=bad_config, bundle_id="b", session=session)
    assert result["status"] == "error"
    assert "scorer_type" in result["message"] or "sequential" in result["message"]


def test_train_model_missing_bundle_returns_error(session):
    result = TOOL_TRAIN_MODEL.fn(model_name="x", config=_TABULAR_CONFIG, bundle_id="missing", session=session)
    assert result["status"] == "error"
    assert result["error_type"] == "BundleNotFound"
