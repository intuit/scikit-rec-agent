"""transform_data tool.

Reshape a raw CSV/Parquet into a scikit-rec contract format. The LLM-facing
contract is the ``target_contract`` enum; small composable internal ops do
the work. Source shape is auto-detected from the data.

Output is a canonical CSV (or Parquet) at ``output_path`` plus a final
self-validation against the target contract. Schemas are NOT written here —
``create_datasets`` regenerates them from dtypes on the next call.
"""

from __future__ import annotations

import os
import re
from typing import Any

import pandas as pd

from scikit_rec_agent.tools import Tool, err, ok

# ---------------------------------------------------------------------------
# Target contract definitions
# ---------------------------------------------------------------------------

_TARGET_CONTRACTS = (
    "long_interactions",
    "long_with_timestamp",
    "long_multi_reward",
    "wide_multioutput",
    "multiclass",
    "prebuilt_sequences",
    "sessions",
    "users_features",
    "items_features",
)

# What the validator demands of the OUTPUT for each contract.
_CONTRACT_REQUIRED_COLS: dict[str, tuple[str, ...]] = {
    "long_interactions": ("USER_ID", "ITEM_ID", "OUTCOME"),
    "long_with_timestamp": ("USER_ID", "ITEM_ID", "OUTCOME", "TIMESTAMP"),
    "long_multi_reward": ("USER_ID", "ITEM_ID", "OUTCOME"),
    "wide_multioutput": ("USER_ID",),
    "multiclass": ("USER_ID", "ITEM_ID"),
    "prebuilt_sequences": ("USER_ID", "ITEM_SEQUENCE"),
    "sessions": ("USER_ID", "SESSION_SEQUENCES"),
    "users_features": ("USER_ID",),
    "items_features": ("ITEM_ID",),
}

_CONTRACTS_REQUIRING_DEDUPE = (
    "wide_multioutput",
    "multiclass",
    "prebuilt_sequences",
    "sessions",
    "users_features",
    "items_features",
)


# ---------------------------------------------------------------------------
# Operation primitives
# ---------------------------------------------------------------------------


def _op_rename_columns(df: pd.DataFrame, mapping: dict[str, str]) -> pd.DataFrame:
    return df.rename(columns=mapping)


def _op_bulk_rename_targets(
    df: pd.DataFrame,
    pattern: str,
    replacement: str = r"ITEM_\1",
) -> pd.DataFrame:
    """Apply a regex-based bulk rename. Default replacement turns
    ``label_foo`` into ``ITEM_foo`` for the canonical wide-multioutput
    pattern."""
    rx = re.compile(pattern)
    new_cols = {}
    for c in df.columns:
        m = rx.fullmatch(c)
        if m:
            new_cols[c] = rx.sub(replacement, c)
    return df.rename(columns=new_cols)


def _op_pivot_to_wide(
    df: pd.DataFrame,
    user_col: str,
    item_col: str,
    outcome_col: str,
) -> pd.DataFrame:
    pivoted = df.pivot_table(
        index=user_col,
        columns=item_col,
        values=outcome_col,
        aggfunc="max",
        fill_value=0,
    )
    pivoted.columns = [f"ITEM_{c}" for c in pivoted.columns]
    return pivoted.reset_index().rename(columns={user_col: "USER_ID"})


def _op_melt_to_long(
    df: pd.DataFrame,
    user_col: str,
    item_columns: list[str],
) -> pd.DataFrame:
    melted = df.melt(
        id_vars=[user_col],
        value_vars=item_columns,
        var_name="ITEM_ID",
        value_name="OUTCOME",
    )
    return melted.rename(columns={user_col: "USER_ID"})


def _op_aggregate_to_sequences(
    df: pd.DataFrame,
    user_col: str,
    item_col: str,
    outcome_col: str | None,
    timestamp_col: str | None,
) -> pd.DataFrame:
    if timestamp_col is None or timestamp_col not in df.columns:
        raise ValueError(
            "aggregate_to_sequences requires a timestamp_col present in the data; sequence ordering must not be silent."
        )
    sorted_df = df.sort_values([user_col, timestamp_col])
    agg: dict[str, Any] = {item_col: list}
    if outcome_col and outcome_col in sorted_df.columns:
        agg[outcome_col] = list
    grouped = sorted_df.groupby(user_col).agg(agg).reset_index()
    rename = {user_col: "USER_ID", item_col: "ITEM_SEQUENCE"}
    if outcome_col and outcome_col in grouped.columns:
        rename[outcome_col] = "OUTCOME_SEQUENCE"
    return grouped.rename(columns=rename)


def _op_parse_timestamp(df: pd.DataFrame, timestamp_col: str) -> pd.DataFrame:
    out = df.copy()
    out["TIMESTAMP"] = pd.to_datetime(out[timestamp_col], errors="coerce")
    out["TIMESTAMP"] = out["TIMESTAMP"].astype("int64") // 10**9
    out["TIMESTAMP"] = out["TIMESTAMP"].astype(str)
    if timestamp_col != "TIMESTAMP" and timestamp_col in out.columns:
        out = out.drop(columns=[timestamp_col])
    return out


def _op_cast_dtypes(df: pd.DataFrame, casts: dict[str, str]) -> pd.DataFrame:
    out = df.copy()
    for col, target in casts.items():
        if col not in out.columns:
            continue
        if target == "str":
            out[col] = out[col].astype(str)
        elif target == "float":
            out[col] = out[col].astype(float)
        elif target == "int":
            out[col] = out[col].astype("int64")
    return out


def _op_dedupe_user_id(df: pd.DataFrame) -> pd.DataFrame:
    if "USER_ID" not in df.columns:
        return df
    return df.drop_duplicates(subset=["USER_ID"], keep="first").reset_index(drop=True)


def _op_drop_null_columns(df: pd.DataFrame) -> pd.DataFrame:
    return df.dropna(axis=1, how="all")


def _op_select_features(
    df: pd.DataFrame,
    keep_cols: list[str],
    feature_columns: list[str] | None,
    feature_pattern: str | None,
) -> pd.DataFrame:
    keep = list(keep_cols)
    if feature_columns:
        keep += [c for c in feature_columns if c in df.columns and c not in keep]
    if feature_pattern:
        rx = re.compile(feature_pattern)
        keep += [c for c in df.columns if rx.fullmatch(c) and c not in keep]
    keep = [c for c in keep if c in df.columns]
    return df[keep]


# ---------------------------------------------------------------------------
# Source shape detection
# ---------------------------------------------------------------------------


def _detect_source_shape(
    df: pd.DataFrame,
    user_id_column: str | None,
    item_id_column: str | None,
) -> str:
    """Cheap signals only — never block on detection failure, just return a
    best guess. The contract validator at the end is the actual safety net."""
    user_col = user_id_column or ("USER_ID" if "USER_ID" in df.columns else None)
    if user_col is None or user_col not in df.columns:
        return "unknown"

    list_cols = [c for c in df.columns if df[c].apply(lambda v: isinstance(v, (list, tuple))).any()]
    if list_cols:
        return "pre_aggregated_sequences"

    rows_per_user = df.groupby(user_col).size().max() if len(df) else 0
    if rows_per_user > 1:
        return "long"

    item_star = [c for c in df.columns if c.startswith("ITEM_") and c != "ITEM_ID"]
    if len(item_star) >= 2:
        return "wide_already"

    label_cols = [c for c in df.columns if c.lower().startswith("label_")]
    if len(label_cols) >= 2:
        return "wide_with_other_prefix"

    return "single_row_per_user"


# ---------------------------------------------------------------------------
# Per-contract transform plans
# ---------------------------------------------------------------------------


def _to_long_interactions(
    df, user_id_column, item_id_column, outcome_column, target_columns=None, target_rename_pattern=None, **_
):
    """Long-format reshape. Two source paths:

    - already long (item_id_column + outcome_column supplied): rename + cast.
    - wide (target_columns or target_rename_pattern supplied, no item_id):
      melt the indicated columns into (USER_ID, ITEM_ID, OUTCOME) rows.
    """
    user_col = user_id_column or "USER_ID"

    if not item_id_column and (target_columns or target_rename_pattern):
        if target_rename_pattern and not target_columns:
            rx = re.compile(target_rename_pattern)
            target_columns = [c for c in df.columns if rx.fullmatch(c)]
            if not target_columns:
                raise ValueError(f"target_rename_pattern '{target_rename_pattern}' matched no columns.")
        df = _op_melt_to_long(df, user_col, target_columns)
        df = _op_cast_dtypes(df, {"USER_ID": "str", "ITEM_ID": "str", "OUTCOME": "float"})
        return df, ["melt_to_long", "cast_dtypes"]

    rename: dict[str, str] = {}
    if user_id_column and user_id_column != "USER_ID":
        rename[user_id_column] = "USER_ID"
    if item_id_column and item_id_column != "ITEM_ID":
        rename[item_id_column] = "ITEM_ID"
    if outcome_column and outcome_column != "OUTCOME":
        rename[outcome_column] = "OUTCOME"
    if rename:
        df = _op_rename_columns(df, rename)
    df = _op_cast_dtypes(df, {"USER_ID": "str", "ITEM_ID": "str", "OUTCOME": "float"})
    return df, ["rename_columns", "cast_dtypes"]


def _to_long_with_timestamp(df, user_id_column, item_id_column, outcome_column, timestamp_column, **_):
    if timestamp_column is None:
        raise ValueError("long_with_timestamp requires timestamp_column.")
    df, ops = _to_long_interactions(
        df,
        user_id_column,
        item_id_column,
        outcome_column,
    )
    df = _op_parse_timestamp(df, timestamp_column)
    ops.append("parse_timestamp")
    return df, ops


def _to_long_multi_reward(df, user_id_column, item_id_column, outcome_column, auxiliary_outcome_columns, **_):
    df, ops = _to_long_interactions(df, user_id_column, item_id_column, outcome_column)
    if auxiliary_outcome_columns:
        rename = {
            c: f"OUTCOME_{c}" if not c.startswith("OUTCOME_") else c
            for c in auxiliary_outcome_columns
            if c in df.columns
        }
        df = _op_rename_columns(df, rename)
        for c in df.columns:
            if c.startswith("OUTCOME_"):
                df[c] = df[c].astype(float)
        ops.append("rename_aux_outcomes")
    return df, ops


def _to_wide_multioutput(
    df, user_id_column, item_id_column, outcome_column, target_columns, target_rename_pattern, dedupe_user_id, **_
):
    ops: list[str] = []
    user_col = user_id_column or "USER_ID"

    item_star_present = [c for c in df.columns if c.startswith("ITEM_") and c != "ITEM_ID"]

    if target_rename_pattern and not item_star_present:
        df = _op_bulk_rename_targets(df, target_rename_pattern)
        ops.append("bulk_rename_targets")
    elif target_columns:
        rename = {c: f"ITEM_{c}" if not c.startswith("ITEM_") else c for c in target_columns}
        df = _op_rename_columns(df, rename)
        ops.append("rename_target_columns")
    elif item_id_column and outcome_column:
        # Long → wide pivot path
        df = _op_pivot_to_wide(df, user_col, item_id_column, outcome_column)
        ops.append("pivot_to_wide")
        user_col = "USER_ID"

    if user_col != "USER_ID" and user_col in df.columns:
        df = _op_rename_columns(df, {user_col: "USER_ID"})
        ops.append("rename_user_id")

    item_targets = [c for c in df.columns if c.startswith("ITEM_") and c != "ITEM_ID"]
    if len(item_targets) < 2:
        raise ValueError(
            f"wide_multioutput requires ≥2 ITEM_* target columns after transform; "
            f"got {len(item_targets)}: {item_targets}"
        )

    df = _op_cast_dtypes(df, {"USER_ID": "str", **{c: "float" for c in item_targets}})
    ops.append("cast_dtypes")

    if dedupe_user_id:
        df = _op_dedupe_user_id(df)
        ops.append("dedupe_user_id")
    return df, ops


def _to_multiclass(df, user_id_column, item_id_column, dedupe_user_id, **_):
    rename: dict[str, str] = {}
    if user_id_column and user_id_column != "USER_ID":
        rename[user_id_column] = "USER_ID"
    if item_id_column and item_id_column != "ITEM_ID":
        rename[item_id_column] = "ITEM_ID"
    if rename:
        df = _op_rename_columns(df, rename)
    df = _op_cast_dtypes(df, {"USER_ID": "str", "ITEM_ID": "str"})
    if "OUTCOME" in df.columns:
        df = df.drop(columns=["OUTCOME"])
    if dedupe_user_id:
        df = _op_dedupe_user_id(df)
    return df, ["rename_columns", "cast_dtypes", "drop_outcome", "dedupe_user_id"]


def _to_prebuilt_sequences(df, user_id_column, item_id_column, outcome_column, timestamp_column, **_):
    user_col = user_id_column or "USER_ID"
    item_col = item_id_column or "ITEM_ID"
    if user_col not in df.columns or item_col not in df.columns:
        raise ValueError(f"prebuilt_sequences requires {user_col} and {item_col} columns in the source.")
    df = _op_aggregate_to_sequences(
        df,
        user_col,
        item_col,
        outcome_column,
        timestamp_column,
    )
    return df, ["aggregate_to_sequences"]


def _to_sessions(df, user_id_column, **_):
    """Sessions contract: USER_ID + SESSION_SEQUENCES (list-of-lists).

    Source data is expected to already carry a SESSION_SEQUENCES column;
    constructing valid session boundaries from raw triples requires
    user-supplied session-id semantics this tool can't guess. Surface a
    clear error otherwise.
    """
    user_col = user_id_column or "USER_ID"
    if "SESSION_SEQUENCES" not in df.columns:
        raise ValueError(
            "sessions contract requires a SESSION_SEQUENCES column in the source. "
            "If your data is flat (USER_ID, ITEM_ID, SESSION_ID), aggregate it to "
            "lists-of-lists yourself before calling transform_data — the right "
            "session boundaries are user-domain knowledge."
        )
    if user_col != "USER_ID":
        df = _op_rename_columns(df, {user_col: "USER_ID"})
    df = _op_dedupe_user_id(df)
    return df, ["rename_user_id", "dedupe_user_id"]


def _to_users_features(df, user_id_column, dedupe_user_id, **_):
    user_col = user_id_column or "USER_ID"
    if user_col != "USER_ID":
        df = _op_rename_columns(df, {user_col: "USER_ID"})
    df = _op_cast_dtypes(df, {"USER_ID": "str"})
    if dedupe_user_id:
        df = _op_dedupe_user_id(df)
    return df, ["rename_user_id", "cast_dtypes", "dedupe_user_id"]


def _to_items_features(df, item_id_column, **_):
    item_col = item_id_column or "ITEM_ID"
    if item_col != "ITEM_ID":
        df = _op_rename_columns(df, {item_col: "ITEM_ID"})
    df = _op_cast_dtypes(df, {"ITEM_ID": "str"})
    if "ITEM_ID" in df.columns:
        df = df.drop_duplicates(subset=["ITEM_ID"]).reset_index(drop=True)
    return df, ["rename_item_id", "cast_dtypes", "dedupe_item_id"]


_PLAN_BY_CONTRACT = {
    "long_interactions": _to_long_interactions,
    "long_with_timestamp": _to_long_with_timestamp,
    "long_multi_reward": _to_long_multi_reward,
    "wide_multioutput": _to_wide_multioutput,
    "multiclass": _to_multiclass,
    "prebuilt_sequences": _to_prebuilt_sequences,
    "sessions": _to_sessions,
    "users_features": _to_users_features,
    "items_features": _to_items_features,
}


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------


def _validate_against_contract(df: pd.DataFrame, contract: str) -> dict[str, Any]:
    required = _CONTRACT_REQUIRED_COLS.get(contract, ())
    issues: list[str] = []
    for col in required:
        if col not in df.columns:
            issues.append(f"missing required column '{col}'")

    if contract == "wide_multioutput":
        item_targets = [c for c in df.columns if c.startswith("ITEM_") and c != "ITEM_ID"]
        if len(item_targets) < 2:
            issues.append(f"wide_multioutput requires ≥2 ITEM_* target columns, found {len(item_targets)}")

    if contract in _CONTRACTS_REQUIRING_DEDUPE and "USER_ID" in df.columns:
        if df["USER_ID"].duplicated().any():
            issues.append(f"{contract} requires 1 row per USER_ID; duplicates found")

    if contract == "prebuilt_sequences" and "ITEM_SEQUENCE" in df.columns:
        if not df["ITEM_SEQUENCE"].apply(lambda v: isinstance(v, (list, tuple))).all():
            issues.append("ITEM_SEQUENCE column must contain list/tuple values")

    return {"valid": not issues, "issues": issues}


# ---------------------------------------------------------------------------
# IO helpers
# ---------------------------------------------------------------------------


def _read(file_path: str) -> pd.DataFrame:
    if file_path.endswith(".parquet"):
        return pd.read_parquet(file_path)
    return pd.read_csv(file_path)


def _write(df: pd.DataFrame, output_path: str) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(output_path)) or ".", exist_ok=True)
    if output_path.endswith(".parquet"):
        df.to_parquet(output_path, index=False)
    else:
        df.to_csv(output_path, index=False)


# ---------------------------------------------------------------------------
# Tool entrypoint
# ---------------------------------------------------------------------------


def _transform_data(
    file_path: str,
    output_path: str,
    target_contract: str,
    session,
    user_id_column: str | None = None,
    item_id_column: str | None = None,
    outcome_column: str | None = None,
    timestamp_column: str | None = None,
    target_columns: list[str] | None = None,
    target_rename_pattern: str | None = None,
    auxiliary_outcome_columns: list[str] | None = None,
    feature_columns: list[str] | None = None,
    feature_pattern: str | None = None,
    drop_null_columns: bool = True,
    dedupe_user_id: bool = False,
) -> dict[str, Any]:
    if target_contract not in _TARGET_CONTRACTS:
        return err(
            "InvalidContract",
            f"Unknown target_contract '{target_contract}'. Valid: {list(_TARGET_CONTRACTS)}",
        )
    if not os.path.exists(file_path):
        return err("FileNotFoundError", f"file_path not found: {file_path}")
    if os.path.abspath(file_path) == os.path.abspath(output_path):
        return err(
            "InPlaceWriteForbidden",
            "output_path must differ from file_path; transform_data never writes in-place.",
        )

    try:
        df = _read(file_path)
    except Exception as e:
        return err(type(e).__name__, f"Failed to read {file_path}: {e}")

    source_shape = _detect_source_shape(df, user_id_column, item_id_column)
    if drop_null_columns:
        df = _op_drop_null_columns(df)

    plan = _PLAN_BY_CONTRACT[target_contract]
    try:
        df, ops_applied = plan(
            df,
            user_id_column=user_id_column,
            item_id_column=item_id_column,
            outcome_column=outcome_column,
            timestamp_column=timestamp_column,
            target_columns=target_columns,
            target_rename_pattern=target_rename_pattern,
            auxiliary_outcome_columns=auxiliary_outcome_columns,
            dedupe_user_id=dedupe_user_id or target_contract in _CONTRACTS_REQUIRING_DEDUPE,
        )
    except Exception as e:
        return err(type(e).__name__, str(e))

    keep_anchor: list[str] = list(_CONTRACT_REQUIRED_COLS.get(target_contract, ()))
    if target_contract == "wide_multioutput":
        keep_anchor += [c for c in df.columns if c.startswith("ITEM_") and c != "ITEM_ID"]
    if target_contract == "long_multi_reward":
        keep_anchor += [c for c in df.columns if c.startswith("OUTCOME_")]
    if target_contract == "prebuilt_sequences" and "OUTCOME_SEQUENCE" in df.columns:
        keep_anchor.append("OUTCOME_SEQUENCE")

    if feature_columns or feature_pattern:
        df = _op_select_features(df, keep_anchor, feature_columns, feature_pattern)
        ops_applied.append("select_features")

    validation = _validate_against_contract(df, target_contract)
    if not validation["valid"]:
        return err(
            "ContractViolation",
            f"Transform output does not satisfy {target_contract}: {validation['issues']}",
            hint="Re-run profile_data on the source to confirm role hints; some contracts "
            "require explicit user_id_column / item_id_column / target_columns.",
        )

    try:
        _write(df, output_path)
    except Exception as e:
        return err(type(e).__name__, f"Failed to write {output_path}: {e}")

    return ok(
        {
            "output_path": output_path,
            "target_contract": target_contract,
            "source_shape_detected": source_shape,
            "ops_applied": ops_applied,
            "columns": list(df.columns),
            "n_rows": int(len(df)),
            "validation": validation,
        }
    )


TOOL_TRANSFORM_DATA = Tool(
    name="transform_data",
    description=(
        "Reshape a raw CSV/Parquet into a scikit-rec contract format. Use after "
        "`profile_data` + `validate_data` when the input shape doesn't match the "
        "contract the chosen recommender requires. Auto-detects source shape and "
        "applies the right reshape (rename, pivot, melt, aggregate, dedupe, cast). "
        "Writes a canonical file at `output_path`; call `create_datasets` on that "
        "path next."
    ),
    input_schema={
        "type": "object",
        "properties": {
            "file_path": {"type": "string"},
            "output_path": {"type": "string"},
            "target_contract": {
                "type": "string",
                "enum": list(_TARGET_CONTRACTS),
            },
            "user_id_column": {"type": "string"},
            "item_id_column": {"type": "string"},
            "outcome_column": {"type": "string"},
            "timestamp_column": {"type": "string"},
            "target_columns": {
                "type": "array",
                "items": {"type": "string"},
                "description": "Columns to become ITEM_* targets (wide_multioutput).",
            },
            "target_rename_pattern": {
                "type": "string",
                "description": "Regex matching wide target columns, e.g. 'label_(.*)'. Renamed to ITEM_\\1.",
            },
            "auxiliary_outcome_columns": {
                "type": "array",
                "items": {"type": "string"},
                "description": "Long_multi_reward auxiliary OUTCOME_* sources.",
            },
            "feature_columns": {
                "type": "array",
                "items": {"type": "string"},
                "description": "Explicit feature columns to keep alongside the contract anchor columns.",
            },
            "feature_pattern": {
                "type": "string",
                "description": "Regex matching feature columns to keep.",
            },
            "drop_null_columns": {"type": "boolean", "default": True},
            "dedupe_user_id": {"type": "boolean", "default": False},
        },
        "required": ["file_path", "output_path", "target_contract"],
    },
    fn=_transform_data,
)
