"""profile_data and validate_data tools.

profile_data produces a data summary the LLM uses to pick an appropriate
recommender architecture. validate_data checks the file against scikit-rec's
required schemas and suggests column-name fixes when possible.
"""

from __future__ import annotations

import difflib
import os
from typing import Any

import pandas as pd
import yaml

from scikit_rec_agent.tools import Tool, err, ok

# ---------------------------------------------------------------------------
# profile_data
# ---------------------------------------------------------------------------

_ID_COL_HINTS = ("id", "uid", "uuid")
_USER_ID_HINTS = ("user", "customer", "member", "account")
_ITEM_ID_HINTS = ("item", "product", "content", "article", "movie", "sku")
_TARGET_HINTS = ("outcome", "label", "target", "y", "click", "rating", "reward", "conversion")
_TIMESTAMP_HINTS = ("time", "timestamp", "date", "ts", "event_time")


def _read_file(file_path: str) -> pd.DataFrame:
    if file_path.endswith(".parquet"):
        return pd.read_parquet(file_path)
    return pd.read_csv(file_path)


def _detect_id_columns(df: pd.DataFrame) -> dict[str, str | None]:
    cols = {c.lower(): c for c in df.columns}
    detected = {"user_id": None, "item_id": None}
    for key, hints in (("user_id", _USER_ID_HINTS), ("item_id", _ITEM_ID_HINTS)):
        for col_lower, col in cols.items():
            if any(h in col_lower for h in hints) and any(h in col_lower for h in _ID_COL_HINTS + ("_",)):
                detected[key] = col
                break
        if detected[key] is None:
            # Loose fallback: any column whose lowercased name contains the hint word
            for col_lower, col in cols.items():
                if any(col_lower == h or col_lower.startswith(h + "_") for h in hints):
                    detected[key] = col
                    break
    return detected


def _detect_target_column(df: pd.DataFrame) -> str | None:
    for col in df.columns:
        if any(h in col.lower() for h in _TARGET_HINTS):
            return col
    return None


def _detect_timestamp_column(df: pd.DataFrame) -> str | None:
    for col in df.columns:
        if any(h in col.lower() for h in _TIMESTAMP_HINTS):
            return col
    return None


def _classify_target(series: pd.Series) -> str:
    non_null = series.dropna()
    n_unique = non_null.nunique()
    # A single-value target is degenerate — model selection heuristics
    # shouldn't try to fit a classifier to it.
    if n_unique <= 1:
        return "constant"
    if series.dtype == bool or set(non_null.unique()).issubset({0, 1, 0.0, 1.0}):
        return "binary"
    if pd.api.types.is_integer_dtype(series) or pd.api.types.is_float_dtype(series):
        uniques = non_null.unique()
        if len(uniques) <= 10 and set(uniques).issubset(set(range(1, 11))):
            return "rating"
        return "continuous"
    return "categorical"


def _column_summary(df: pd.DataFrame) -> list[dict[str, Any]]:
    out = []
    for col in df.columns:
        series = df[col]
        sample_values = series.dropna().head(5).tolist()
        out.append(
            {
                "name": col,
                "dtype": str(series.dtype),
                "null_count": int(series.isna().sum()),
                "n_unique": int(series.nunique(dropna=True)),
                "sample_values": [_json_safe(v) for v in sample_values],
            }
        )
    return out


def _json_safe(v: Any) -> Any:
    if isinstance(v, (int, float, str, bool)) or v is None:
        return v
    if hasattr(v, "isoformat"):
        return v.isoformat()
    return str(v)


def _profile_data(file_path: str, file_type: str, session) -> dict[str, Any]:
    if not os.path.exists(file_path):
        return err("FileNotFoundError", f"File not found: {file_path}")
    try:
        df = _read_file(file_path)
    except Exception as e:
        return err(type(e).__name__, f"Failed to read {file_path}: {e}")

    summary: dict[str, Any] = {
        "file_path": file_path,
        "file_type": file_type,
        "shape": {"n_rows": int(df.shape[0]), "n_cols": int(df.shape[1])},
        "columns": _column_summary(df),
    }

    if file_type == "interactions":
        ids = _detect_id_columns(df)
        target = _detect_target_column(df)
        timestamp = _detect_timestamp_column(df)

        summary["id_columns_detected"] = ids
        summary["target_column_detected"] = target
        if target is not None:
            summary["target_type"] = _classify_target(df[target])
        else:
            summary["target_type"] = None

        if timestamp is not None:
            try:
                ts = pd.to_datetime(df[timestamp], errors="coerce")
                summary["temporal_range"] = {
                    "column": timestamp,
                    "min": ts.min().isoformat() if pd.notna(ts.min()) else None,
                    "max": ts.max().isoformat() if pd.notna(ts.max()) else None,
                }
            except Exception:
                summary["temporal_range"] = {"column": timestamp, "min": None, "max": None}

        user_col = ids.get("user_id")
        item_col = ids.get("item_id")
        if user_col and item_col:
            n_users = df[user_col].nunique()
            n_items = df[item_col].nunique()
            summary["sparsity"] = 1.0 - len(df) / (n_users * n_items) if n_users > 0 and n_items > 0 else None
            summary["duplicate_pairs_count"] = int(df.duplicated(subset=[user_col, item_col]).sum())
    elif file_type == "users":
        ids = _detect_id_columns(df)
        summary["id_columns_detected"] = {"user_id": ids.get("user_id")}
    elif file_type == "items":
        ids = _detect_id_columns(df)
        summary["id_columns_detected"] = {"item_id": ids.get("item_id")}

    return ok(summary)


TOOL_PROFILE_DATA = Tool(
    name="profile_data",
    description=(
        "Load and profile a data file. Reports shape, dtypes, cardinality of ID columns, "
        "sparsity, value distributions, temporal range, and whether the target looks implicit "
        "(binary) or explicit (ratings)."
    ),
    input_schema={
        "type": "object",
        "properties": {
            "file_path": {"type": "string", "description": "Path to CSV or parquet file"},
            "file_type": {"type": "string", "enum": ["interactions", "users", "items"]},
        },
        "required": ["file_path", "file_type"],
    },
    fn=_profile_data,
)


# ---------------------------------------------------------------------------
# validate_data
# ---------------------------------------------------------------------------


def _load_required_schema(file_type: str, is_training: bool) -> list[dict[str, str]]:
    from skrec.dataset.interactions_dataset import InteractionsDataset
    from skrec.dataset.items_dataset import ItemsDataset
    from skrec.dataset.users_dataset import UsersDataset

    if file_type == "interactions":
        path = (
            InteractionsDataset.REQUIRED_SCHEMA_PATH_TRAINING
            if is_training
            else InteractionsDataset.REQUIRED_SCHEMA_PATH_INFERENCE
        )
    elif file_type == "users":
        path = UsersDataset.REQUIRED_SCHEMA_PATH
    elif file_type == "items":
        path = ItemsDataset.REQUIRED_SCHEMA_PATH
    else:
        raise ValueError(f"Unknown file_type: {file_type}")
    with open(path) as f:
        return yaml.safe_load(f)["columns"]


def _dtype_matches(required_type: str, pandas_dtype: Any) -> bool:
    if required_type == "str":
        return pd.api.types.is_object_dtype(pandas_dtype) or pd.api.types.is_string_dtype(pandas_dtype)
    if required_type in ("float", "double"):
        return pd.api.types.is_float_dtype(pandas_dtype) or pd.api.types.is_integer_dtype(pandas_dtype)
    if required_type == "int":
        return pd.api.types.is_integer_dtype(pandas_dtype)
    if required_type == "bool":
        return pd.api.types.is_bool_dtype(pandas_dtype)
    return True


def _validate_data(file_path: str, file_type: str, session, is_training: bool = True) -> dict[str, Any]:
    if not os.path.exists(file_path):
        return err("FileNotFoundError", f"File not found: {file_path}")
    try:
        df = _read_file(file_path)
    except Exception as e:
        return err(type(e).__name__, f"Failed to read {file_path}: {e}")

    try:
        required = _load_required_schema(file_type, is_training)
    except Exception as e:
        return err(type(e).__name__, f"Could not load required schema: {e}")

    user_cols = list(df.columns)
    user_cols_lower = {c.lower(): c for c in user_cols}

    missing_columns: list[str] = []
    wrong_dtypes: list[dict[str, str]] = []
    suggested_column_mapping: dict[str, str] = {}

    for req in required:
        req_name = req["name"]
        req_type = req["type"]
        if req_name in user_cols:
            if not _dtype_matches(req_type, df[req_name].dtype):
                wrong_dtypes.append(
                    {
                        "column": req_name,
                        "expected": req_type,
                        "actual": str(df[req_name].dtype),
                    }
                )
            continue

        missing_columns.append(req_name)
        matches = difflib.get_close_matches(req_name.lower(), list(user_cols_lower), n=1, cutoff=0.5)
        if matches:
            suggested_column_mapping[user_cols_lower[matches[0]]] = req_name

    required_names = {r["name"] for r in required}
    extra_columns = [c for c in user_cols if c not in required_names]

    return ok(
        {
            "valid": not missing_columns and not wrong_dtypes,
            "file_type": file_type,
            "is_training": is_training,
            "missing_columns": missing_columns,
            "wrong_dtypes": wrong_dtypes,
            "suggested_column_mapping": suggested_column_mapping,
            "extra_columns": extra_columns,
        }
    )


TOOL_VALIDATE_DATA = Tool(
    name="validate_data",
    description=(
        "Validate a data file against scikit-rec required schemas. Reports missing required "
        "columns, wrong dtypes, and suggests column renames if near-matches are detected."
    ),
    input_schema={
        "type": "object",
        "properties": {
            "file_path": {"type": "string"},
            "file_type": {"type": "string", "enum": ["interactions", "users", "items"]},
            "is_training": {"type": "boolean", "default": True},
        },
        "required": ["file_path", "file_type"],
    },
    fn=_validate_data,
)
