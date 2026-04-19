"""Tests for evaluate_model and compare_models tools."""

from __future__ import annotations

import pytest

from scikit_rec_agent.tools.datasets import TOOL_CREATE_DATASETS
from scikit_rec_agent.tools.evaluation import TOOL_COMPARE_MODELS, TOOL_EVALUATE_MODEL
from scikit_rec_agent.tools.splitting import TOOL_SPLIT_DATA
from scikit_rec_agent.tools.training import TOOL_TRAIN_MODEL

_TABULAR_CONFIG = {
    "recommender_type": "ranking",
    "scorer_type": "universal",
    "estimator_config": {
        "ml_task": "classification",
        "xgboost": {"n_estimators": 20, "max_depth": 3},
    },
}


@pytest.fixture
def trained_model(binary_reward_paths, session):
    TOOL_CREATE_DATASETS.fn(
        bundle_name="b",
        interactions_path=binary_reward_paths["interactions"],
        users_path=binary_reward_paths["users"],
        items_path=binary_reward_paths["items"],
        session=session,
    )
    TOOL_SPLIT_DATA.fn(
        bundle_id="b",
        strategy="random_split",
        valid_fraction=0.2,
        session=session,
        random_state=1,
    )
    result = TOOL_TRAIN_MODEL.fn(model_name="m", config=_TABULAR_CONFIG, bundle_id="b", session=session)
    assert result["status"] == "ok"
    return result["data"]["model_id"]


def test_evaluate_records_metrics(trained_model, session):
    result = TOOL_EVALUATE_MODEL.fn(
        model_id=trained_model,
        evaluator_type="simple",
        metrics=["NDCG_at_k", "precision_at_k"],
        k_values=[5, 10],
        session=session,
    )
    assert result["status"] == "ok"
    data = result["data"]
    assert len(data["results"]) == 4  # 2 metrics × 2 k
    handle = session.trained_models[trained_model]
    assert "NDCG_at_k@5" in handle.metrics
    assert "NDCG_at_k@10" in handle.metrics
    assert "precision_at_k@5" in handle.metrics


def test_evaluate_unknown_metric_returns_error(trained_model, session):
    result = TOOL_EVALUATE_MODEL.fn(
        model_id=trained_model,
        evaluator_type="simple",
        metrics=["bogus_metric"],
        k_values=[10],
        session=session,
    )
    assert result["status"] == "error"
    assert result["error_type"] == "InvalidMetric"


def test_compare_models_renders_markdown(trained_model, session):
    # Evaluate at least once so metrics exist
    TOOL_EVALUATE_MODEL.fn(
        model_id=trained_model,
        evaluator_type="simple",
        metrics=["NDCG_at_k"],
        k_values=[10],
        session=session,
    )
    result = TOOL_COMPARE_MODELS.fn(primary_metric="NDCG_at_k", k=10, session=session)
    assert result["status"] == "ok"
    md = result["data"]["markdown"]
    assert "model_name" in md
    assert trained_model in md or "m" in md
    assert len(result["data"]["rows"]) == 1


def test_compare_with_no_models_errors(session):
    result = TOOL_COMPARE_MODELS.fn(primary_metric="NDCG_at_k", k=10, session=session)
    assert result["status"] == "error"
    assert result["error_type"] == "NoModels"
