"""Tests for create_datasets tool."""

from __future__ import annotations

import pandas as pd

from scikit_rec_agent.tools.datasets import TOOL_CREATE_DATASETS


def test_create_datasets_happy_path(binary_reward_paths, session):
    result = TOOL_CREATE_DATASETS.fn(
        bundle_name="b1",
        interactions_path=binary_reward_paths["interactions"],
        users_path=binary_reward_paths["users"],
        items_path=binary_reward_paths["items"],
        session=session,
    )
    assert result["status"] == "ok"
    bundle = session.loaded_datasets["b1"]
    assert bundle.interactions is not None
    assert bundle.users is not None
    assert bundle.items is not None
    assert bundle.valid_interactions is None


def test_create_datasets_with_column_mapping(tmp_path, session):
    # Write a CSV with non-canonical column names.
    df = pd.DataFrame(
        {
            "userid": ["u0", "u1", "u2"],
            "itemid": ["i0", "i1", "i2"],
            "clicked": [1.0, 0.0, 1.0],
        }
    )
    p = tmp_path / "bad_interactions.csv"
    df.to_csv(p, index=False)

    result = TOOL_CREATE_DATASETS.fn(
        bundle_name="mapped",
        interactions_path=str(p),
        session=session,
        column_mapping={"userid": "USER_ID", "itemid": "ITEM_ID", "clicked": "OUTCOME"},
    )
    assert result["status"] == "ok"
    assert set(result["data"]["columns"]) == {"USER_ID", "ITEM_ID", "OUTCOME"}


def test_create_datasets_file_not_found(session):
    result = TOOL_CREATE_DATASETS.fn(
        bundle_name="bad",
        interactions_path="/nonexistent.csv",
        session=session,
    )
    assert result["status"] == "error"
    assert result["error_type"] == "FileNotFoundError"
