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


def test_parse_timestamp_rejects_unparseable_values():
    """Regression: pd.to_datetime(errors='coerce') silently produces NaT for
    bad values, then astype('int64') turns them into INT64_MIN. Must raise
    naming the offending column + count instead."""
    df = pd.DataFrame({"date": ["2024-01-01", "not a date", "also bad"]})
    with pytest.raises(ValueError, match="unparseable"):
        _op_parse_timestamp(df, "date")


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


def test_wide_multioutput_non_binary_string_targets_rejected_with_friendly_message(tmp_path, session):
    """classification of non-numeric ITEM_* targets runs BEFORE the
    float cast. Without the reorder, the cast at _op_cast_dtypes raises
    pandas' opaque ``could not convert string to float: 'yes'`` and the
    friendly migration message never fires. The reorder makes the user-
    facing error explain the issue + the migration paths.
    """
    df = pd.DataFrame(
        {
            "uid": ["u1", "u2", "u3"],
            "label_a": ["yes", "no", "yes"],
            "label_b": ["yes", "yes", "no"],
        }
    )
    src = tmp_path / "wide_strings.csv"
    df.to_csv(src, index=False)
    out = str(tmp_path / "transformed.csv")
    result = TOOL_TRANSFORM_DATA.fn(
        file_path=str(src),
        output_path=out,
        target_contract="wide_multioutput",
        user_id_column="uid",
        target_rename_pattern=r"label_(.*)",
        session=session,
    )
    assert result["status"] == "error"
    msg = result["message"].lower()
    # Must mention binary-numeric requirement AND surface the bad values
    # AND name at least one of the affected columns. The previous error
    # path produced pandas' generic "could not convert string to float".
    assert "binary numeric" in msg
    assert "yes" in msg
    assert any(c in result["message"] for c in ("ITEM_a", "ITEM_b"))


def test_classify_wide_multioutput_targets_coerces_string_numerics():
    """object-dtype ITEM_* columns with values like ['1', '0', '1']
    cast cleanly to binary numeric — the helper coerces them via
    pd.to_numeric before deciding 'non-binary'. Without this, a clean
    CSV round-trip of {0, 1} that pandas read as object dtype would
    surface a misleading 'non-binary' error.
    """
    import pandas as pd

    from scikit_rec_agent.tools.transform import _classify_wide_multioutput_targets

    df = pd.DataFrame(
        {
            "USER_ID": ["u1", "u2", "u3", "u4"],
            "ITEM_string_binary": ["1", "0", "1", "0"],
            "ITEM_string_genuine_text": ["yes", "no", "yes", "no"],
        }
    )
    bad_non_binary, degenerate, ok_cols = _classify_wide_multioutput_targets(df)

    # The string-but-numeric column coerces and lands in ok
    assert "ITEM_string_binary" in ok_cols
    # The genuine-text column still flagged as non-binary
    assert any(c == "ITEM_string_genuine_text" for c, _ in bad_non_binary)


def test_classify_wide_multioutput_targets_three_bins():
    """Unit cover for _classify_wide_multioutput_targets — the helper at
    the heart of the wide_multioutput rejection / auto-drop logic. Sorts
    columns into three bins:
      - bad_non_binary: non-numeric or non-{0,1} values
      - degenerate: single-class
      - ok: binary numeric with both classes present
    """
    import pandas as pd

    from scikit_rec_agent.tools.transform import _classify_wide_multioutput_targets

    df = pd.DataFrame(
        {
            "USER_ID": ["u1", "u2", "u3", "u4"],
            "ITEM_ok_a": [0, 1, 1, 0],  # both classes — ok
            "ITEM_ok_b": [1.0, 0.0, 0.0, 1.0],
            "ITEM_dead": [0, 0, 0, 0],  # single-class — degenerate
            "ITEM_strings": ["yes", "no", "yes", "no"],  # non-numeric
            "ITEM_multi": [0, 1, 2, 0],  # multi-class
        }
    )
    bad_non_binary, degenerate, ok_cols = _classify_wide_multioutput_targets(df)

    bad_names = {c for c, _ in bad_non_binary}
    deg_names = {c for c, _ in degenerate}
    assert bad_names == {"ITEM_strings", "ITEM_multi"}
    assert deg_names == {"ITEM_dead"}
    assert set(ok_cols) == {"ITEM_ok_a", "ITEM_ok_b"}


def test_compute_long_per_label_metric_handles_degenerate_per_item():
    """Unit cover for _compute_long_per_label_metric — items where the
    masked y_true has fewer than 2 classes get NaN (sklearn would raise);
    items with valid signal get the metric value.
    """
    import math

    import numpy as np

    from scikit_rec_agent.tools.evaluation import _compute_long_per_label_metric

    # Two items: 'good' has both classes, 'dead' has only class 1 (degenerate)
    predictions = {
        "items": ["good", "dead"],
        "y_true": np.array(
            [
                [1.0, 1.0],
                [0.0, 1.0],
                [1.0, 1.0],
                [0.0, 1.0],
            ]
        ),
        "y_score": np.array(
            [
                [0.8, 0.5],
                [0.2, 0.4],
                [0.9, 0.7],
                [0.1, 0.3],
            ]
        ),
    }
    result = _compute_long_per_label_metric("roc_auc", predictions)
    assert result["good"] == 1.0  # perfect separation
    assert math.isnan(result["dead"])


def test_wide_multioutput_all_degenerate_surfaces_surviving_columns(tmp_path, session):
    """when every (or all-but-one) ITEM_* target is single-class, the
    error must surface the surviving non-degenerate column(s) so the user
    can decide to keep it manually. Prior message listed only the
    degenerate columns, hiding the survivor.
    """
    df = pd.DataFrame(
        {
            "uid": ["u1", "u2", "u3", "u4"],
            "label_dead_a": [1, 1, 1, 1],
            "label_dead_b": [0, 0, 0, 0],
            "label_alive": [0, 1, 1, 0],
        }
    )
    src = tmp_path / "wide_deg.csv"
    df.to_csv(src, index=False)
    out = str(tmp_path / "transformed.csv")
    result = TOOL_TRANSFORM_DATA.fn(
        file_path=str(src),
        output_path=out,
        target_contract="wide_multioutput",
        user_id_column="uid",
        target_rename_pattern=r"label_(.*)",
        session=session,
    )
    assert result["status"] == "error"
    msg = result["message"]
    # Both degenerate names AND the surviving one must appear, so the user
    # can decide to keep it as a single-target problem.
    assert "ITEM_dead_a" in msg
    assert "ITEM_dead_b" in msg
    assert "ITEM_alive" in msg
    assert "surviving" in msg.lower()


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
