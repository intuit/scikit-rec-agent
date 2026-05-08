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


def test_create_datasets_auto_detects_multioutput(tmp_path, session):
    """Wide multi-output: USER_ID + ≥2 ITEM_* columns, no ITEM_ID/OUTCOME →
    create_datasets must construct an InteractionMultiOutputDataset, not the
    default InteractionsDataset (which would reject the schema)."""
    df = pd.DataFrame(
        {
            "USER_ID": ["u0", "u1", "u2"],
            "feat1": [1, 2, 3],
            "ITEM_payroll": [1.0, 0.0, 1.0],
            "ITEM_invoice": [0.0, 1.0, 1.0],
            "ITEM_budget": [1.0, 1.0, 0.0],
        }
    )
    p = tmp_path / "wide.csv"
    df.to_csv(p, index=False)

    result = TOOL_CREATE_DATASETS.fn(
        bundle_id="wide",
        interactions_path=str(p),
        session=session,
    )
    assert result["status"] == "ok", result
    assert result["data"]["dataset_type"] == "interaction_multioutput"
    bundle = session.loaded_datasets["wide"]
    assert bundle.dataset_type == "interaction_multioutput"
    # The bundle's interactions handle is the right scikit-rec subclass.
    from skrec.dataset.interactions_dataset import InteractionMultiOutputDataset
    assert isinstance(bundle.interactions, InteractionMultiOutputDataset)


def test_create_datasets_auto_detects_multiclass(tmp_path, session):
    """Wide multi-class: USER_ID + ITEM_ID, no OUTCOME → InteractionMultiClassDataset."""
    df = pd.DataFrame(
        {
            "USER_ID": ["u0", "u1", "u2"],
            "ITEM_ID": ["A", "B", "A"],
            "feat1": [1, 2, 3],
        }
    )
    p = tmp_path / "mc.csv"
    df.to_csv(p, index=False)

    result = TOOL_CREATE_DATASETS.fn(
        bundle_id="mc",
        interactions_path=str(p),
        session=session,
    )
    assert result["status"] == "ok", result
    assert result["data"]["dataset_type"] == "interaction_multiclass"
    from skrec.dataset.interactions_dataset import InteractionMultiClassDataset
    assert isinstance(session.loaded_datasets["mc"].interactions, InteractionMultiClassDataset)


def test_create_datasets_explicit_dataset_type_override(tmp_path, session):
    """Caller can force the dataset_type rather than relying on auto-detection."""
    df = pd.DataFrame(
        {
            "USER_ID": ["u0", "u1"],
            "ITEM_payroll": [1.0, 0.0],
            "ITEM_invoice": [0.0, 1.0],
        }
    )
    p = tmp_path / "wide.csv"
    df.to_csv(p, index=False)

    result = TOOL_CREATE_DATASETS.fn(
        bundle_id="wide",
        interactions_path=str(p),
        session=session,
        dataset_type="interaction_multioutput",
    )
    assert result["status"] == "ok"
    assert result["data"]["dataset_type"] == "interaction_multioutput"


def test_create_datasets_invalid_dataset_type_returns_error(tmp_path, session, binary_reward_paths):
    result = TOOL_CREATE_DATASETS.fn(
        bundle_id="x",
        interactions_path=binary_reward_paths["interactions"],
        session=session,
        dataset_type="bogus",
    )
    assert result["status"] == "error"
    assert result["error_type"] == "InvalidDatasetType"


def test_create_datasets_long_format_still_defaults_to_interactions(binary_reward_paths, session):
    """Auto-detection on a vanilla long-format file picks `interactions`,
    keeping the original behaviour."""
    result = TOOL_CREATE_DATASETS.fn(
        bundle_id="b",
        interactions_path=binary_reward_paths["interactions"],
        session=session,
    )
    assert result["status"] == "ok"
    assert result["data"]["dataset_type"] == "interactions"


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
