"""Tests for split_data tool."""

from __future__ import annotations

from scikit_rec_agent.tools.datasets import TOOL_CREATE_DATASETS
from scikit_rec_agent.tools.splitting import TOOL_SPLIT_DATA


def _make_bundle(binary_reward_paths, session, name="b"):
    TOOL_CREATE_DATASETS.fn(
        bundle_name=name,
        interactions_path=binary_reward_paths["interactions"],
        users_path=binary_reward_paths["users"],
        items_path=binary_reward_paths["items"],
        session=session,
    )


def test_split_data_random_populates_bundle(binary_reward_paths, session):
    # Sample binary fixture has 5000 users × 1 row each; per-user splits degenerate,
    # so the plain random_split is the right coverage here.
    _make_bundle(binary_reward_paths, session)
    result = TOOL_SPLIT_DATA.fn(
        bundle_id="b",
        strategy="random_split",
        valid_fraction=0.2,
        test_fraction=0.1,
        session=session,
        random_state=42,
    )
    assert result["status"] == "ok"
    bundle = session.loaded_datasets["b"]
    assert bundle.valid_interactions is not None
    assert bundle.test_interactions is not None
    assert result["data"]["train_rows"] > 0
    assert result["data"]["valid_rows"] > 0
    assert result["data"]["test_rows"] > 0


def test_split_data_rejects_unknown_bundle(session):
    result = TOOL_SPLIT_DATA.fn(
        bundle_id="missing",
        strategy="random_split",
        valid_fraction=0.2,
        session=session,
    )
    assert result["status"] == "error"
    assert result["error_type"] == "BundleNotFound"


def test_split_data_rejects_unknown_strategy(binary_reward_paths, session):
    _make_bundle(binary_reward_paths, session)
    result = TOOL_SPLIT_DATA.fn(
        bundle_id="b",
        strategy="nonsense",
        session=session,
    )
    assert result["status"] == "error"
    assert result["error_type"] == "InvalidStrategy"
