"""Tests for create_datasets tool."""

from __future__ import annotations

import pandas as pd

from scikit_rec_agent.tools.datasets import TOOL_CREATE_DATASETS


def test_create_datasets_happy_path(binary_reward_paths, session):
    result = TOOL_CREATE_DATASETS.fn(
        bundle_id="b1",
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
        bundle_id="mapped",
        interactions_path=str(p),
        session=session,
        column_mapping={"userid": "USER_ID", "itemid": "ITEM_ID", "clicked": "OUTCOME"},
    )
    assert result["status"] == "ok"
    assert set(result["data"]["columns"]) == {"USER_ID", "ITEM_ID", "OUTCOME"}


def test_create_datasets_file_not_found(session):
    result = TOOL_CREATE_DATASETS.fn(
        bundle_id="bad",
        interactions_path="/nonexistent.csv",
        session=session,
    )
    assert result["status"] == "error"
    assert result["error_type"] == "FileNotFoundError"


def test_create_datasets_coerces_integer_outcome_to_float(tmp_path, session):
    # Regression: binary click logs arrive as OUTCOME in {0, 1}, which pandas
    # infers as int64. scikit-rec's required interactions schema declares
    # OUTCOME: float and rejects schemas where OUTCOME is any other type. The
    # auto-schema generator must force the required type regardless of dtype.
    import pandas as pd

    df = pd.DataFrame(
        {
            "USER_ID": ["u0", "u1", "u2"],
            "ITEM_ID": ["i0", "i1", "i2"],
            "OUTCOME": [0, 1, 1],  # int64 dtype
        }
    )
    p = tmp_path / "integer_outcome.csv"
    df.to_csv(p, index=False)

    result = TOOL_CREATE_DATASETS.fn(bundle_id="int_outcome", interactions_path=str(p), session=session)
    assert result["status"] == "ok", result
    assert "int_outcome" in session.loaded_datasets
