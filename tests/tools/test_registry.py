"""Tests for save_model / list_models / load_model."""

from __future__ import annotations

from scikit_rec_agent.tools.datasets import TOOL_CREATE_DATASETS
from scikit_rec_agent.tools.registry import (
    TOOL_LIST_MODELS,
    TOOL_LOAD_MODEL,
    TOOL_SAVE_MODEL,
)
from scikit_rec_agent.tools.training import TOOL_TRAIN_MODEL

_CONFIG = {
    "recommender_type": "ranking",
    "scorer_type": "universal",
    "estimator_config": {
        "ml_task": "classification",
        "xgboost": {"n_estimators": 10, "max_depth": 2},
    },
}


def _train(binary_reward_paths, session, name="reg_test"):
    TOOL_CREATE_DATASETS.fn(
        bundle_id="b",
        interactions_path=binary_reward_paths["interactions"],
        users_path=binary_reward_paths["users"],
        items_path=binary_reward_paths["items"],
        session=session,
    )
    result = TOOL_TRAIN_MODEL.fn(model_name=name, config=_CONFIG, bundle_id="b", session=session)
    return result["data"]["model_id"]


def test_save_list_load_roundtrip(binary_reward_paths, session, tmp_registry):
    model_id = _train(binary_reward_paths, session, name="roundtrip_model")
    save = TOOL_SAVE_MODEL.fn(model_id=model_id, tags=["baseline"], session=session)
    assert save["status"] == "ok"

    listing = TOOL_LIST_MODELS.fn(session=session)
    names = {m["model_name"] for m in listing["data"]["models"]}
    assert "roundtrip_model" in names

    # Fresh session → load restores the handle
    from scikit_rec_agent.session import Session

    new_session = Session()
    load = TOOL_LOAD_MODEL.fn(model_name="roundtrip_model", session=new_session)
    assert load["status"] == "ok"
    assert load["data"]["tags"] == ["baseline"]
    assert load["data"]["model_id"] in new_session.trained_models


def test_save_collision_returns_error(binary_reward_paths, session, tmp_registry):
    model_id = _train(binary_reward_paths, session, name="dup_name")
    first = TOOL_SAVE_MODEL.fn(model_id=model_id, session=session)
    assert first["status"] == "ok"
    second = TOOL_SAVE_MODEL.fn(model_id=model_id, session=session)
    assert second["status"] == "error"
    assert second["error_type"] == "NameCollision"
    assert "hint" in second


def test_list_empty_registry(session, tmp_registry):
    result = TOOL_LIST_MODELS.fn(session=session)
    assert result["status"] == "ok"
    assert result["data"]["models"] == []


def test_load_nonexistent_model(session, tmp_registry):
    result = TOOL_LOAD_MODEL.fn(model_name="never_saved", session=session)
    assert result["status"] == "error"
    assert result["error_type"] == "ModelNotFound"
