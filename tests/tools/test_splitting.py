"""Tests for split_data tool."""

from __future__ import annotations

from scikit_rec_agent.tools.datasets import TOOL_CREATE_DATASETS
from scikit_rec_agent.tools.splitting import TOOL_SPLIT_DATA


def _make_bundle(binary_reward_paths, session, name="b"):
    TOOL_CREATE_DATASETS.fn(
        bundle_id=name,
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


def test_split_data_degenerate_per_user_on_wide_data(tmp_path, session):
    """One row per user + per-user split → 0 valid rows. The tool must
    detect this and return a structured error explaining the cause and
    recommending a different strategy, rather than silently writing an
    empty validation file."""
    import pandas as pd

    df = pd.DataFrame(
        {
            "USER_ID": [f"u{i}" for i in range(20)],
            "ITEM_payroll": [1, 0] * 10,
            "ITEM_invoice": [0, 1] * 10,
            "feat1": list(range(20)),
        }
    )
    p = tmp_path / "wide.csv"
    df.to_csv(p, index=False)

    TOOL_CREATE_DATASETS.fn(
        bundle_id="wide",
        interactions_path=str(p),
        session=session,
    )
    result = TOOL_SPLIT_DATA.fn(
        bundle_id="wide",
        strategy="random_split_per_user",
        valid_fraction=0.2,
        session=session,
        random_state=1,
    )
    assert result["status"] == "error"
    assert result["error_type"] == "DegenerateSplit"
    assert result["category"] == "degenerate_split"
    # The message must point at the actual fix
    assert "leave_n_users_out" in result["message"] or "random_split" in result["message"]


def test_split_data_clears_stale_test_on_resplit(binary_reward_paths, session):
    # Regression: first split with test_fraction>0 populates test_interactions.
    # A second split with test_fraction=0 must clear that stale handle — else
    # subsequent evaluations could use test data from the previous split.
    _make_bundle(binary_reward_paths, session)
    first = TOOL_SPLIT_DATA.fn(
        bundle_id="b",
        strategy="random_split",
        valid_fraction=0.2,
        test_fraction=0.1,
        session=session,
        random_state=1,
    )
    assert first["status"] == "ok"
    assert session.loaded_datasets["b"].test_interactions is not None

    second = TOOL_SPLIT_DATA.fn(
        bundle_id="b",
        strategy="random_split",
        valid_fraction=0.2,
        session=session,
        random_state=2,
    )
    assert second["status"] == "ok"
    bundle = session.loaded_datasets["b"]
    assert bundle.test_interactions is None
    assert "test_interactions" not in bundle.source_paths
