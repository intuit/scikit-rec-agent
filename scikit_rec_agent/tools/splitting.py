"""split_data tool.

Thin wrapper over skrec.split. Reads the interactions DataFrame from a bundle,
splits via the requested strategy, writes train/valid/test CSVs to a temp
directory, and updates the bundle in place so downstream tools can consume
bundle.interactions (train), bundle.valid_interactions, bundle.test_interactions.
"""

from __future__ import annotations

import os
import tempfile
from typing import Any

import pandas as pd

from scikit_rec_agent.tools import Tool, err, ok

_STRATEGIES = (
    "temporal",
    "leave_last_n_per_user",
    "random_split_per_user",
    "leave_n_users_out",
    "random_split",
)


def _load_interactions(bundle) -> pd.DataFrame:
    path = bundle.source_paths.get("interactions")
    if path is None or not os.path.exists(path):
        raise FileNotFoundError(f"bundle interactions source not found: {path}")
    return pd.read_parquet(path) if path.endswith(".parquet") else pd.read_csv(path)


def _split_data(
    bundle_id: str,
    strategy: str,
    session,
    valid_fraction: float | None = None,
    test_fraction: float = 0.0,
    n_valid: int | None = None,
    n_test: int = 0,
    n_valid_users: int | None = None,
    n_test_users: int = 0,
    user_col: str | None = None,
    timestamp_col: str | None = None,
    random_state: int | None = None,
) -> dict[str, Any]:
    from skrec.constants import TIMESTAMP_COL, USER_ID_NAME
    from skrec.dataset.interactions_dataset import InteractionsDataset
    from skrec.split import (
        leave_last_n_per_user,
        leave_n_users_out,
        random_split,
        random_split_per_user,
        temporal_split,
    )

    bundle = session.loaded_datasets.get(bundle_id)
    if bundle is None:
        return err("BundleNotFound", f"No bundle '{bundle_id}' in session. Call create_datasets first.")

    if strategy not in _STRATEGIES:
        return err(
            "InvalidStrategy",
            f"Unknown strategy '{strategy}'. Valid: {list(_STRATEGIES)}",
        )

    try:
        df = _load_interactions(bundle)
    except Exception as e:
        return err(type(e).__name__, str(e))

    user_col = user_col or USER_ID_NAME
    timestamp_col = timestamp_col or TIMESTAMP_COL

    try:
        if strategy == "temporal":
            if valid_fraction is None:
                return err("MissingArgument", "temporal requires valid_fraction.")
            result = temporal_split(
                df,
                valid_fraction=valid_fraction,
                test_fraction=test_fraction,
                timestamp_col=timestamp_col,
            )
        elif strategy == "leave_last_n_per_user":
            if n_valid is None:
                return err("MissingArgument", "leave_last_n_per_user requires n_valid.")
            result = leave_last_n_per_user(
                df,
                n_valid=n_valid,
                n_test=n_test,
                user_col=user_col,
                timestamp_col=timestamp_col,
            )
        elif strategy == "random_split_per_user":
            if valid_fraction is None:
                return err("MissingArgument", "random_split_per_user requires valid_fraction.")
            result = random_split_per_user(
                df,
                valid_fraction=valid_fraction,
                test_fraction=test_fraction,
                user_col=user_col,
                random_state=random_state,
            )
        elif strategy == "leave_n_users_out":
            if n_valid_users is None:
                return err("MissingArgument", "leave_n_users_out requires n_valid_users.")
            result = leave_n_users_out(
                df,
                n_valid_users=n_valid_users,
                n_test_users=n_test_users,
                user_col=user_col,
                random_state=random_state,
            )
        else:  # random_split
            if valid_fraction is None:
                return err("MissingArgument", "random_split requires valid_fraction.")
            result = random_split(
                df,
                valid_fraction=valid_fraction,
                test_fraction=test_fraction,
                random_state=random_state,
            )
    except Exception as e:
        return err(type(e).__name__, str(e))

    tmp_dir = tempfile.mkdtemp(prefix=f"skragent_split_{bundle_id}_")
    inter_schema = bundle.schema_paths.get("interactions")

    train_path = os.path.join(tmp_dir, "train.csv")
    result.train.to_csv(train_path, index=False)
    bundle.interactions = InteractionsDataset(data_location=train_path, client_schema_path=inter_schema)
    bundle.source_paths["interactions"] = train_path

    valid_path = os.path.join(tmp_dir, "valid.csv")
    result.valid.to_csv(valid_path, index=False)
    bundle.valid_interactions = InteractionsDataset(data_location=valid_path, client_schema_path=inter_schema)
    bundle.source_paths["valid_interactions"] = valid_path

    test_path = None
    if result.test is not None:
        test_path = os.path.join(tmp_dir, "test.csv")
        result.test.to_csv(test_path, index=False)
        bundle.test_interactions = InteractionsDataset(data_location=test_path, client_schema_path=inter_schema)
        bundle.source_paths["test_interactions"] = test_path

    return ok(
        {
            "bundle_id": bundle_id,
            "strategy": strategy,
            "train_rows": int(len(result.train)),
            "valid_rows": int(len(result.valid)),
            "test_rows": int(len(result.test)) if result.test is not None else 0,
            "paths": {
                "train": train_path,
                "valid": valid_path,
                "test": test_path,
            },
            "info": _jsonify_info(result.info),
        }
    )


def _jsonify_info(info: dict[str, Any]) -> dict[str, Any]:
    out = {}
    for k, v in info.items():
        if isinstance(v, tuple):
            out[k] = [item.isoformat() if hasattr(item, "isoformat") else item for item in v]
        elif hasattr(v, "isoformat"):
            out[k] = v.isoformat()
        elif isinstance(v, list):
            out[k] = [item.isoformat() if hasattr(item, "isoformat") else item for item in v]
        else:
            out[k] = v
    return out


TOOL_SPLIT_DATA = Tool(
    name="split_data",
    description=(
        "Split a dataset bundle's interactions into train/validation/test using a strategy "
        "appropriate for recommendation systems. Updates the bundle in place: the bundle's "
        "`interactions` becomes the training split, and `valid_interactions` / "
        "`test_interactions` are populated.\n\n"
        "Strategies:\n"
        "- temporal: chronological split using `valid_fraction` (+ optional `test_fraction`). "
        "Production-realistic default.\n"
        "- leave_last_n_per_user: per user, hold out last `n_valid` (+ optional `n_test`) rows "
        "by timestamp. Standard for sequential models.\n"
        "- random_split_per_user: per user, random hold-out of `valid_fraction` (+ optional "
        "`test_fraction`). Preserves all users in train.\n"
        "- leave_n_users_out: hold out `n_valid_users` (+ optional `n_test_users`) entire users. "
        "For honest cold-start evaluation.\n"
        "- random_split: pure random row split. Rarely appropriate for recsys — use only as a "
        "sanity check."
    ),
    input_schema={
        "type": "object",
        "properties": {
            "bundle_id": {"type": "string", "description": "Existing bundle from create_datasets."},
            "strategy": {"type": "string", "enum": list(_STRATEGIES)},
            "valid_fraction": {"type": "number"},
            "test_fraction": {"type": "number", "default": 0.0},
            "n_valid": {"type": "integer"},
            "n_test": {"type": "integer", "default": 0},
            "n_valid_users": {"type": "integer"},
            "n_test_users": {"type": "integer", "default": 0},
            "user_col": {"type": "string", "default": "USER_ID"},
            "timestamp_col": {"type": "string", "default": "TIMESTAMP"},
            "random_state": {"type": "integer"},
        },
        "required": ["bundle_id", "strategy"],
    },
    fn=_split_data,
)
