"""Tests for profile_data and validate_data tools."""

from __future__ import annotations

import pandas as pd

from scikit_rec_agent.tools.profiling import TOOL_PROFILE_DATA, TOOL_VALIDATE_DATA


def test_profile_interactions_reports_shape_and_target(binary_reward_paths, session):
    result = TOOL_PROFILE_DATA.fn(
        file_path=binary_reward_paths["interactions"],
        file_type="interactions",
        session=session,
    )
    assert result["status"] == "ok"
    data = result["data"]
    assert data["file_type"] == "interactions"
    assert data["shape"]["n_rows"] > 0
    assert data["target_type"] in ("binary", "continuous", "rating")
    assert data["id_columns_detected"]["user_id"] == "USER_ID"
    assert data["id_columns_detected"]["item_id"] == "ITEM_ID"
    assert data["target_column_detected"] == "OUTCOME"


def test_profile_file_not_found(session):
    result = TOOL_PROFILE_DATA.fn(file_path="/nonexistent.csv", file_type="interactions", session=session)
    assert result["status"] == "error"
    assert result["error_type"] == "FileNotFoundError"


def test_validate_good_file_returns_valid_true(binary_reward_paths, session):
    result = TOOL_VALIDATE_DATA.fn(
        file_path=binary_reward_paths["interactions"],
        file_type="interactions",
        session=session,
        is_training=True,
    )
    assert result["status"] == "ok"
    assert result["data"]["valid"] is True
    assert result["data"]["missing_columns"] == []


def test_profile_classifies_constant_target(tmp_path, session):
    # Degenerate target (all 1s) should not be reported as 'binary' — that
    # misleads model-selection heuristics ("binary → classification").
    df = pd.DataFrame({"USER_ID": ["u0"] * 5, "ITEM_ID": [f"i{i}" for i in range(5)], "OUTCOME": [1.0] * 5})
    p = tmp_path / "constant.csv"
    df.to_csv(p, index=False)
    result = TOOL_PROFILE_DATA.fn(file_path=str(p), file_type="interactions", session=session)
    assert result["status"] == "ok"
    assert result["data"]["target_type"] == "constant"


def test_validate_missing_columns_suggests_mapping(tmp_path, session):
    # Build a file with wrong-named columns to force a suggestion.
    bad = pd.DataFrame({"userid": ["u1"], "itemid": ["i1"], "clicked": [1.0]})
    p = tmp_path / "bad.csv"
    bad.to_csv(p, index=False)

    result = TOOL_VALIDATE_DATA.fn(file_path=str(p), file_type="interactions", session=session)
    assert result["status"] == "ok"
    data = result["data"]
    assert data["valid"] is False
    assert set(data["missing_columns"]) == {"USER_ID", "ITEM_ID", "OUTCOME"}
    # At least USER_ID / ITEM_ID should have fuzzy suggestions
    assert "userid" in data["suggested_column_mapping"]
    assert data["suggested_column_mapping"]["userid"] == "USER_ID"
