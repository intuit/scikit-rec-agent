"""create_datasets tool.

Builds scikit-rec Dataset handles from file paths. If column_mapping is
provided (e.g. {"userid": "USER_ID"}), the data file is renamed-and-copied to
a temp CSV so the Dataset sees the expected column names. Schemas are
auto-generated from dtypes unless the caller provides explicit YAML paths.

Registers the resulting bundle on the Session under bundle_id == bundle_name.
"""

from __future__ import annotations

import os
import tempfile
from typing import Any

import pandas as pd
import yaml

from scikit_rec_agent.session import DatasetBundle
from scikit_rec_agent.tools import Tool, err, ok

_SUPPORTED_TYPES = {"int", "float", "str"}


def _pandas_to_schema_type(dtype) -> str:
    if pd.api.types.is_integer_dtype(dtype):
        return "int"
    if pd.api.types.is_float_dtype(dtype):
        return "float"
    return "str"


def _generate_schema(df: pd.DataFrame) -> dict[str, Any]:
    columns = [{"name": c, "type": _pandas_to_schema_type(df[c].dtype)} for c in df.columns]
    return {"columns": columns}


def _rename_and_write(src_path: str, column_mapping: dict[str, str], tmp_dir: str) -> str:
    df = pd.read_parquet(src_path) if src_path.endswith(".parquet") else pd.read_csv(src_path)
    df = df.rename(columns=column_mapping)
    out_path = os.path.join(tmp_dir, os.path.basename(src_path).rsplit(".", 1)[0] + ".csv")
    df.to_csv(out_path, index=False)
    return out_path


def _prepare_source(path: str, column_mapping: dict[str, str] | None, tmp_dir: str) -> tuple[str, pd.DataFrame]:
    """Return (file_path, dataframe) with column_mapping applied if provided."""
    if column_mapping:
        final_path = _rename_and_write(path, column_mapping, tmp_dir)
    else:
        final_path = path
    df = pd.read_parquet(final_path) if final_path.endswith(".parquet") else pd.read_csv(final_path)
    return final_path, df


def _write_schema(df: pd.DataFrame, name: str, tmp_dir: str, explicit: str | None) -> str:
    if explicit:
        return explicit
    schema = _generate_schema(df)
    out_path = os.path.join(tmp_dir, f"{name}_schema.yaml")
    with open(out_path, "w") as f:
        yaml.safe_dump(schema, f, sort_keys=False)
    return out_path


def _create_datasets(
    bundle_name: str,
    interactions_path: str,
    session,
    users_path: str | None = None,
    items_path: str | None = None,
    valid_interactions_path: str | None = None,
    test_interactions_path: str | None = None,
    column_mapping: dict[str, str] | None = None,
    schemas: dict[str, str] | None = None,
) -> dict[str, Any]:
    from skrec.dataset.interactions_dataset import InteractionsDataset
    from skrec.dataset.items_dataset import ItemsDataset
    from skrec.dataset.users_dataset import UsersDataset

    if not os.path.exists(interactions_path):
        return err("FileNotFoundError", f"interactions_path not found: {interactions_path}")

    schemas = schemas or {}
    tmp_dir = tempfile.mkdtemp(prefix=f"skragent_{bundle_name}_")
    schema_paths: dict[str, str] = {}
    source_paths: dict[str, str] = {"interactions": interactions_path}

    try:
        inter_path, inter_df = _prepare_source(interactions_path, column_mapping, tmp_dir)
        inter_schema = _write_schema(inter_df, "interactions", tmp_dir, schemas.get("interactions"))
        schema_paths["interactions"] = inter_schema
        interactions_ds = InteractionsDataset(data_location=inter_path, client_schema_path=inter_schema)

        users_ds = None
        if users_path:
            if not os.path.exists(users_path):
                return err("FileNotFoundError", f"users_path not found: {users_path}")
            u_path, u_df = _prepare_source(users_path, column_mapping, tmp_dir)
            u_schema = _write_schema(u_df, "users", tmp_dir, schemas.get("users"))
            schema_paths["users"] = u_schema
            source_paths["users"] = users_path
            users_ds = UsersDataset(data_location=u_path, client_schema_path=u_schema)

        items_ds = None
        if items_path:
            if not os.path.exists(items_path):
                return err("FileNotFoundError", f"items_path not found: {items_path}")
            i_path, i_df = _prepare_source(items_path, column_mapping, tmp_dir)
            i_schema = _write_schema(i_df, "items", tmp_dir, schemas.get("items"))
            schema_paths["items"] = i_schema
            source_paths["items"] = items_path
            items_ds = ItemsDataset(data_location=i_path, client_schema_path=i_schema)

        valid_ds = None
        if valid_interactions_path:
            if not os.path.exists(valid_interactions_path):
                return err("FileNotFoundError", f"valid_interactions_path not found: {valid_interactions_path}")
            v_path, _ = _prepare_source(valid_interactions_path, column_mapping, tmp_dir)
            source_paths["valid_interactions"] = valid_interactions_path
            valid_ds = InteractionsDataset(data_location=v_path, client_schema_path=inter_schema)

        test_ds = None
        if test_interactions_path:
            if not os.path.exists(test_interactions_path):
                return err("FileNotFoundError", f"test_interactions_path not found: {test_interactions_path}")
            t_path, _ = _prepare_source(test_interactions_path, column_mapping, tmp_dir)
            source_paths["test_interactions"] = test_interactions_path
            test_ds = InteractionsDataset(data_location=t_path, client_schema_path=inter_schema)
    except Exception as e:
        return err(type(e).__name__, str(e))

    bundle = DatasetBundle(
        bundle_id=bundle_name,
        interactions=interactions_ds,
        users=users_ds,
        items=items_ds,
        valid_interactions=valid_ds,
        test_interactions=test_ds,
        schema_paths=schema_paths,
        source_paths=source_paths,
    )
    session.loaded_datasets[bundle_name] = bundle

    return ok(
        {
            "bundle_id": bundle_name,
            "schema_paths": schema_paths,
            "columns": list(inter_df.columns),
            "n_interactions": int(len(inter_df)),
            "has_users": users_ds is not None,
            "has_items": items_ds is not None,
            "has_valid": valid_ds is not None,
            "has_test": test_ds is not None,
        }
    )


TOOL_CREATE_DATASETS = Tool(
    name="create_datasets",
    description=(
        "Build scikit-rec Dataset handles. Auto-generates YAML schemas from the data types "
        "if schemas are not provided. Applies column_mapping to rename columns to USER_ID/"
        "ITEM_ID/OUTCOME as needed. Registers the handles in the session under bundle_id = "
        "bundle_name. Optionally registers validation and test interaction files under the "
        "same bundle."
    ),
    input_schema={
        "type": "object",
        "properties": {
            "bundle_name": {"type": "string"},
            "interactions_path": {"type": "string"},
            "users_path": {"type": "string"},
            "items_path": {"type": "string"},
            "valid_interactions_path": {"type": "string"},
            "test_interactions_path": {"type": "string"},
            "column_mapping": {
                "type": "object",
                "description": 'Map user\'s column names to scikit-rec names, e.g. {"userid": "USER_ID"}.',
            },
            "schemas": {
                "type": "object",
                "description": (
                    "Optional pre-written YAML schema paths keyed by file_type ('interactions', 'users', 'items')."
                ),
            },
        },
        "required": ["bundle_name", "interactions_path"],
    },
    fn=_create_datasets,
)
