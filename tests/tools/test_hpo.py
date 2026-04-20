"""Smoke test for run_hpo — tiny random-sampler search."""

from __future__ import annotations

from scikit_rec_agent.tools.datasets import TOOL_CREATE_DATASETS
from scikit_rec_agent.tools.hpo import TOOL_RUN_HPO
from scikit_rec_agent.tools.splitting import TOOL_SPLIT_DATA

_BASE_CONFIG = {
    "recommender_type": "ranking",
    "scorer_type": "universal",
    "estimator_config": {
        "ml_task": "classification",
        "xgboost": {"n_estimators": 10, "max_depth": 3},
    },
}


def test_run_hpo_tiny_random_search(binary_reward_paths, session):
    TOOL_CREATE_DATASETS.fn(
        bundle_id="b",
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
        random_state=7,
    )
    result = TOOL_RUN_HPO.fn(
        study_name="tiny_study",
        base_config=_BASE_CONFIG,
        search_space={"estimator_config.xgboost.max_depth": {"type": "int", "low": 2, "high": 4}},
        metric_definitions=["NDCG@10"],
        objective_metric="NDCG@10",
        bundle_id="b",
        n_trials=2,
        sampler="random",
        session=session,
        retrain_best=False,
    )
    assert result["status"] == "ok"
    assert result["data"]["n_complete_trials"] >= 1
    assert result["data"]["results_parquet_path"].endswith(".parquet")


def test_run_hpo_requires_validation(binary_reward_paths, session):
    TOOL_CREATE_DATASETS.fn(
        bundle_id="novalid",
        interactions_path=binary_reward_paths["interactions"],
        users_path=binary_reward_paths["users"],
        items_path=binary_reward_paths["items"],
        session=session,
    )
    result = TOOL_RUN_HPO.fn(
        study_name="s",
        base_config=_BASE_CONFIG,
        search_space={"estimator_config.xgboost.max_depth": {"type": "int", "low": 2, "high": 4}},
        metric_definitions=["NDCG@10"],
        objective_metric="NDCG@10",
        bundle_id="novalid",
        n_trials=1,
        session=session,
    )
    assert result["status"] == "error"
    assert result["error_type"] == "MissingValidation"
