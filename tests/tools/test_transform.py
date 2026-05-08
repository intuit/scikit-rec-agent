"""Tests for the transform_data tool.

One end-to-end test per target_contract plus per-op behavioural checks. All
tests use synthetic in-memory frames written to tmp_path; no scikit-rec
training is invoked here.
"""

from __future__ import annotations

import pandas as pd
import pytest

from scikit_rec_agent.tools.datasets import TOOL_CREATE_DATASETS
from scikit_rec_agent.tools.transform import (
    TOOL_TRANSFORM_DATA,
    _detect_source_shape,
    _op_aggregate_to_sequences,
    _op_bulk_rename_targets,
    _op_dedupe_user_id,
    _op_drop_null_columns,
    _op_melt_to_long,
    _op_parse_timestamp,
    _op_pivot_to_wide,
    _validate_against_contract,
)

# ---------------------------------------------------------------------------
# Per-op unit tests
# ---------------------------------------------------------------------------


def test_bulk_rename_targets_label_to_item():
    df = pd.DataFrame({"user": [1], "label_clicks": [1], "label_revenue": [0]})
    out = _op_bulk_rename_targets(df, r"label_(.*)")
    assert "ITEM_clicks" in out.columns
    assert "ITEM_revenue" in out.columns
    assert "label_clicks" not in out.columns


def test_pivot_to_wide_basic():
    df = pd.DataFrame(
        {
            "u": ["a", "a", "b"],
            "i": ["x", "y", "x"],
            "o": [1.0, 0.0, 1.0],
        }
    )
    out = _op_pivot_to_wide(df, "u", "i", "o")
    assert "USER_ID" in out.columns
    assert "ITEM_x" in out.columns
    assert "ITEM_y" in out.columns
    assert len(out) == 2  # 2 unique users


def test_melt_to_long_basic():
    df = pd.DataFrame({"u": ["a", "b"], "ITEM_x": [1, 0], "ITEM_y": [0, 1]})
    out = _op_melt_to_long(df, "u", ["ITEM_x", "ITEM_y"])
    assert "USER_ID" in out.columns
    assert "ITEM_ID" in out.columns
    assert "OUTCOME" in out.columns
    assert len(out) == 4


def test_aggregate_to_sequences_requires_timestamp():
    df = pd.DataFrame({"u": ["a", "a"], "i": ["x", "y"]})
    with pytest.raises(ValueError, match="timestamp_col"):
        _op_aggregate_to_sequences(df, "u", "i", None, None)


def test_aggregate_to_sequences_orders_by_timestamp():
    df = pd.DataFrame(
        {
            "u": ["a", "a"],
            "i": ["second", "first"],
            "ts": [200, 100],
        }
    )
    out = _op_aggregate_to_sequences(df, "u", "i", None, "ts")
    seq = out.iloc[0]["ITEM_SEQUENCE"]
    assert seq == ["first", "second"]


def test_parse_timestamp_to_int_seconds():
    df = pd.DataFrame({"date": ["2024-01-01", "2024-01-02"]})
    out = _op_parse_timestamp(df, "date")
    assert "TIMESTAMP" in out.columns
    assert "date" not in out.columns


def test_dedupe_user_id_keeps_first():
    df = pd.DataFrame({"USER_ID": ["a", "a", "b"], "v": [1, 2, 3]})
    out = _op_dedupe_user_id(df)
    assert len(out) == 2
    assert list(out["v"]) == [1, 3]


def test_drop_null_columns_removes_fully_nan():
    df = pd.DataFrame({"keep": [1, 2], "drop_me": [None, None]})
    out = _op_drop_null_columns(df)
    assert "keep" in out.columns
    assert "drop_me" not in out.columns


# ---------------------------------------------------------------------------
# Source shape detection
# ---------------------------------------------------------------------------


def test_detect_long_shape():
    df = pd.DataFrame({"USER_ID": ["a", "a", "b"], "ITEM_ID": ["x", "y", "x"]})
    assert _detect_source_shape(df, None, None) == "long"


def test_detect_wide_already_shape():
    df = pd.DataFrame({"USER_ID": ["a", "b"], "ITEM_x": [1, 0], "ITEM_y": [0, 1]})
    assert _detect_source_shape(df, None, None) == "wide_already"


def test_detect_wide_with_label_prefix():
    df = pd.DataFrame({"USER_ID": ["a", "b"], "label_x": [1, 0], "label_y": [0, 1]})
    assert _detect_source_shape(df, None, None) == "wide_with_other_prefix"


# ---------------------------------------------------------------------------
# End-to-end per-contract tests
# ---------------------------------------------------------------------------


def _interactions_long(tmp_path) -> str:
    p = tmp_path / "long.csv"
    pd.DataFrame(
        {
            "uid": ["u1", "u1", "u2", "u3"],
            "iid": ["i_a", "i_b", "i_a", "i_c"],
            "rew": [1, 0, 1, 1],
        }
    ).to_csv(p, index=False)
    return str(p)


def _interactions_long_with_ts(tmp_path) -> str:
    p = tmp_path / "long_ts.csv"
    pd.DataFrame(
        {
            "uid": ["u1", "u1", "u2"],
            "iid": ["i_a", "i_b", "i_a"],
            "rew": [1, 0, 1],
            "event_time": ["2024-01-01", "2024-01-02", "2024-01-01"],
        }
    ).to_csv(p, index=False)
    return str(p)


def _wide_label_prefix(tmp_path) -> str:
    p = tmp_path / "wide.csv"
    pd.DataFrame(
        {
            "company_id": ["u1", "u2", "u3"],
            "feat1": [1, 2, 3],
            "label_payroll": [1, 0, 1],
            "label_invoice": [0, 1, 1],
            "label_budget": [1, 1, 0],
        }
    ).to_csv(p, index=False)
    return str(p)


def test_long_interactions_contract(tmp_path, session):
    src = _interactions_long(tmp_path)
    out = str(tmp_path / "out.csv")
    result = TOOL_TRANSFORM_DATA.fn(
        file_path=src,
        output_path=out,
        target_contract="long_interactions",
        user_id_column="uid",
        item_id_column="iid",
        outcome_column="rew",
        session=session,
    )
    assert result["status"] == "ok", result
    df = pd.read_csv(out)
    assert {"USER_ID", "ITEM_ID", "OUTCOME"}.issubset(df.columns)
    assert df["OUTCOME"].dtype == float


def test_long_interactions_via_melt_from_wide(tmp_path, session):
    """Wide source (one row per user, label_* columns) → long_interactions
    via the melt path. Each user contributes one row per target column."""
    src = tmp_path / "wide.csv"
    pd.DataFrame(
        {
            "company_id": ["u1", "u2"],
            "feat1": [10, 20],
            "label_a": [1, 0],
            "label_b": [0, 1],
            "label_c": [1, 1],
        }
    ).to_csv(src, index=False)
    out = str(tmp_path / "out.csv")
    result = TOOL_TRANSFORM_DATA.fn(
        file_path=str(src),
        output_path=out,
        target_contract="long_interactions",
        user_id_column="company_id",
        target_rename_pattern=r"label_(.*)",
        session=session,
    )
    assert result["status"] == "ok", result
    assert "melt_to_long" in result["data"]["ops_applied"]
    df = pd.read_csv(out)
    assert {"USER_ID", "ITEM_ID", "OUTCOME"}.issubset(df.columns)
    # 2 users × 3 labels = 6 long rows
    assert len(df) == 6
    assert set(df["ITEM_ID"]) == {"label_a", "label_b", "label_c"}


def test_long_interactions_melt_explicit_target_columns(tmp_path, session):
    src = tmp_path / "wide.csv"
    pd.DataFrame(
        {
            "uid": ["u1", "u2"],
            "ignored": ["x", "y"],
            "click": [1, 0],
            "purchase": [0, 1],
        }
    ).to_csv(src, index=False)
    out = str(tmp_path / "out.csv")
    result = TOOL_TRANSFORM_DATA.fn(
        file_path=str(src),
        output_path=out,
        target_contract="long_interactions",
        user_id_column="uid",
        target_columns=["click", "purchase"],
        session=session,
    )
    assert result["status"] == "ok", result
    df = pd.read_csv(out)
    assert len(df) == 4
    assert set(df["ITEM_ID"]) == {"click", "purchase"}


def test_long_with_timestamp_contract(tmp_path, session):
    src = _interactions_long_with_ts(tmp_path)
    out = str(tmp_path / "out.csv")
    result = TOOL_TRANSFORM_DATA.fn(
        file_path=src,
        output_path=out,
        target_contract="long_with_timestamp",
        user_id_column="uid",
        item_id_column="iid",
        outcome_column="rew",
        timestamp_column="event_time",
        session=session,
    )
    assert result["status"] == "ok", result
    df = pd.read_csv(out)
    assert "TIMESTAMP" in df.columns
    assert "event_time" not in df.columns


def test_wide_multioutput_via_rename_pattern(tmp_path, session):
    src = _wide_label_prefix(tmp_path)
    out = str(tmp_path / "out.csv")
    result = TOOL_TRANSFORM_DATA.fn(
        file_path=src,
        output_path=out,
        target_contract="wide_multioutput",
        user_id_column="company_id",
        target_rename_pattern=r"label_(.*)",
        feature_columns=["feat1"],
        session=session,
    )
    assert result["status"] == "ok", result
    df = pd.read_csv(out)
    assert "USER_ID" in df.columns
    item_cols = [c for c in df.columns if c.startswith("ITEM_")]
    assert len(item_cols) >= 2
    assert "feat1" in df.columns
    # 1 row per USER_ID
    assert df["USER_ID"].duplicated().sum() == 0


def test_wide_multioutput_pivot_from_long(tmp_path, session):
    src = _interactions_long(tmp_path)
    out = str(tmp_path / "out.csv")
    result = TOOL_TRANSFORM_DATA.fn(
        file_path=src,
        output_path=out,
        target_contract="wide_multioutput",
        user_id_column="uid",
        item_id_column="iid",
        outcome_column="rew",
        session=session,
    )
    assert result["status"] == "ok", result
    df = pd.read_csv(out)
    item_cols = [c for c in df.columns if c.startswith("ITEM_")]
    assert len(item_cols) >= 2


def test_multiclass_contract(tmp_path, session):
    src = tmp_path / "mc.csv"
    pd.DataFrame({"uid": ["u1", "u2"], "iid": ["A", "B"]}).to_csv(src, index=False)
    out = str(tmp_path / "out.csv")
    result = TOOL_TRANSFORM_DATA.fn(
        file_path=str(src),
        output_path=out,
        target_contract="multiclass",
        user_id_column="uid",
        item_id_column="iid",
        session=session,
    )
    assert result["status"] == "ok", result
    df = pd.read_csv(out)
    assert {"USER_ID", "ITEM_ID"}.issubset(df.columns)
    assert "OUTCOME" not in df.columns


def test_prebuilt_sequences_contract(tmp_path, session):
    src = tmp_path / "seq.csv"
    pd.DataFrame(
        {
            "u": ["a", "a", "a", "b"],
            "i": ["x", "y", "z", "x"],
            "ts": [100, 200, 300, 100],
        }
    ).to_csv(src, index=False)
    out = str(tmp_path / "out.csv")
    result = TOOL_TRANSFORM_DATA.fn(
        file_path=str(src),
        output_path=out,
        target_contract="prebuilt_sequences",
        user_id_column="u",
        item_id_column="i",
        timestamp_column="ts",
        session=session,
    )
    assert result["status"] == "ok", result
    # CSV roundtrip turns lists into stringified lists; just check column exists.
    df = pd.read_csv(out)
    assert "ITEM_SEQUENCE" in df.columns


def test_users_features_contract(tmp_path, session):
    src = tmp_path / "u.csv"
    pd.DataFrame({"uid": ["a", "a", "b"], "age": [30, 30, 25]}).to_csv(src, index=False)
    out = str(tmp_path / "out.csv")
    result = TOOL_TRANSFORM_DATA.fn(
        file_path=str(src),
        output_path=out,
        target_contract="users_features",
        user_id_column="uid",
        feature_columns=["age"],
        session=session,
    )
    assert result["status"] == "ok", result
    df = pd.read_csv(out)
    assert {"USER_ID", "age"}.issubset(df.columns)
    assert df["USER_ID"].duplicated().sum() == 0


def test_items_features_contract(tmp_path, session):
    src = tmp_path / "i.csv"
    pd.DataFrame({"iid": ["a", "b"], "category": ["x", "y"]}).to_csv(src, index=False)
    out = str(tmp_path / "out.csv")
    result = TOOL_TRANSFORM_DATA.fn(
        file_path=str(src),
        output_path=out,
        target_contract="items_features",
        item_id_column="iid",
        feature_columns=["category"],
        session=session,
    )
    assert result["status"] == "ok", result
    df = pd.read_csv(out)
    assert {"ITEM_ID", "category"}.issubset(df.columns)


def test_long_multi_reward_contract(tmp_path, session):
    src = tmp_path / "lmr.csv"
    pd.DataFrame(
        {
            "uid": ["u1", "u2"],
            "iid": ["A", "B"],
            "rew": [1, 0],
            "revenue": [10.0, 0.0],
        }
    ).to_csv(src, index=False)
    out = str(tmp_path / "out.csv")
    result = TOOL_TRANSFORM_DATA.fn(
        file_path=str(src),
        output_path=out,
        target_contract="long_multi_reward",
        user_id_column="uid",
        item_id_column="iid",
        outcome_column="rew",
        auxiliary_outcome_columns=["revenue"],
        session=session,
    )
    assert result["status"] == "ok", result
    df = pd.read_csv(out)
    assert "OUTCOME_revenue" in df.columns


def test_sessions_contract_requires_session_sequences_column(tmp_path, session):
    src = tmp_path / "s.csv"
    pd.DataFrame({"USER_ID": ["a"], "x": [1]}).to_csv(src, index=False)
    out = str(tmp_path / "out.csv")
    result = TOOL_TRANSFORM_DATA.fn(
        file_path=str(src),
        output_path=out,
        target_contract="sessions",
        session=session,
    )
    assert result["status"] == "error"
    assert "SESSION_SEQUENCES" in result["message"]


# ---------------------------------------------------------------------------
# Validation + safety guards
# ---------------------------------------------------------------------------


def test_invalid_contract_returns_error(tmp_path, session):
    src = _interactions_long(tmp_path)
    out = str(tmp_path / "out.csv")
    result = TOOL_TRANSFORM_DATA.fn(
        file_path=src,
        output_path=out,
        target_contract="bogus",
        session=session,
    )
    assert result["status"] == "error"
    assert result["error_type"] == "InvalidContract"


def test_in_place_write_forbidden(tmp_path, session):
    src = _interactions_long(tmp_path)
    result = TOOL_TRANSFORM_DATA.fn(
        file_path=src,
        output_path=src,
        target_contract="long_interactions",
        user_id_column="uid",
        item_id_column="iid",
        outcome_column="rew",
        session=session,
    )
    assert result["status"] == "error"
    assert result["error_type"] == "InPlaceWriteForbidden"


def test_validate_against_contract_catches_dupes():
    df = pd.DataFrame({"USER_ID": ["a", "a"], "ITEM_x": [1, 0], "ITEM_y": [0, 1]})
    res = _validate_against_contract(df, "wide_multioutput")
    assert not res["valid"]
    assert any("duplicates" in i.lower() for i in res["issues"])


def test_validate_against_contract_catches_missing_targets():
    df = pd.DataFrame({"USER_ID": ["a"], "ITEM_x": [1]})
    res = _validate_against_contract(df, "wide_multioutput")
    assert not res["valid"]


# ---------------------------------------------------------------------------
# Integration with create_datasets — ensures transform output is consumable
# ---------------------------------------------------------------------------


def test_long_transform_output_feeds_create_datasets(tmp_path, session):
    src = _interactions_long(tmp_path)
    out = str(tmp_path / "transformed.csv")
    result = TOOL_TRANSFORM_DATA.fn(
        file_path=src,
        output_path=out,
        target_contract="long_interactions",
        user_id_column="uid",
        item_id_column="iid",
        outcome_column="rew",
        session=session,
    )
    assert result["status"] == "ok"
    cd = TOOL_CREATE_DATASETS.fn(
        bundle_id="b",
        interactions_path=out,
        session=session,
    )
    assert cd["status"] == "ok", cd
    assert "b" in session.loaded_datasets
