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
    # Route through ``_resolve_interactions_path`` so that wide_multioutput /
    # multiclass bundles (where users were auto-merged into the interactions
    # frame) split on the MERGED content. Splitting the user-provided
    # un-merged file would strip features from each post-split slice.
    from scikit_rec_agent.tools.datasets import _resolve_interactions_path

    path = _resolve_interactions_path(bundle.source_paths)
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
    from skrec.dataset.interactions_dataset import (
        InteractionMultiClassDataset,
        InteractionMultiOutputDataset,
        InteractionsDataset,
    )
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

    requested_valid = (
        (valid_fraction is not None and valid_fraction > 0)
        or (n_valid is not None and n_valid > 0)
        or (n_valid_users is not None and n_valid_users > 0)
    )
    if requested_valid and len(result.valid) == 0:
        per_user_strategies = ("random_split_per_user", "leave_last_n_per_user")
        rows_per_user = int(df.groupby(user_col).size().max()) if user_col in df.columns else None
        if strategy in per_user_strategies and rows_per_user == 1:
            return err(
                "DegenerateSplit",
                (
                    f"strategy='{strategy}' produced 0 validation rows. The data has "
                    f"1 row per {user_col} (likely a wide multi-output / multi-class / "
                    f"feature-table shape), so a per-user split has nothing to hold "
                    f"out. Use strategy='leave_n_users_out' (with `n_valid_users`) or "
                    f"strategy='random_split' (with `valid_fraction`) on this shape."
                ),
                hint=(
                    "Wide / one-row-per-user data needs a user-level or row-level split, "
                    "not a per-user split. Re-call split_data with leave_n_users_out."
                ),
                category="degenerate_split",
            )
        return err(
            "DegenerateSplit",
            (
                f"strategy='{strategy}' produced 0 validation rows despite a non-zero "
                f"valid request. Total rows: {len(df)}. This usually means the requested "
                f"hold-out fraction is too small for the data size, or the strategy is a "
                f"poor fit for the data shape."
            ),
            hint="Try a larger valid_fraction or a different split strategy.",
            category="degenerate_split",
        )

    tmp_dir = tempfile.mkdtemp(prefix=f"skragent_split_{bundle_id}_")
    inter_schema = bundle.schema_paths.get("interactions")

    _DATASET_CLASS = {
        "interactions": InteractionsDataset,
        "interaction_multioutput": InteractionMultiOutputDataset,
        "interaction_multiclass": InteractionMultiClassDataset,
    }
    ds_cls = _DATASET_CLASS.get(bundle.dataset_type, InteractionsDataset)

    train_path = os.path.join(tmp_dir, "train.csv")
    result.train.to_csv(train_path, index=False)
    bundle.interactions = ds_cls(data_location=train_path, client_schema_path=inter_schema)
    bundle.source_paths["interactions"] = train_path
    # The train slice was already extracted from the (possibly merged)
    # frame; the auto-merge artifact is now stale. Drop it so subsequent
    # _resolve_interactions_path calls read the post-split train file
    # rather than the pre-split merged one.
    bundle.source_paths.pop("merged_interactions", None)

    valid_path = os.path.join(tmp_dir, "valid.csv")
    result.valid.to_csv(valid_path, index=False)
    bundle.valid_interactions = ds_cls(data_location=valid_path, client_schema_path=inter_schema)
    bundle.source_paths["valid_interactions"] = valid_path

    test_path: str | None = None
    if result.test is not None:
        test_path = os.path.join(tmp_dir, "test.csv")
        result.test.to_csv(test_path, index=False)
        bundle.test_interactions = ds_cls(data_location=test_path, client_schema_path=inter_schema)
        bundle.source_paths["test_interactions"] = test_path
    else:
        # Overwrite rather than leave stale handles from a prior split/create.
        bundle.test_interactions = None
        bundle.source_paths.pop("test_interactions", None)

    # Validation data changed — any model trained/evaluated against this
    # bundle now has a stale score cache. Reset so the next evaluate_model
    # re-scores automatically without the user needing to pass refresh_scores.
    for handle in session.trained_models.values():
        if (handle.datasets_used or {}).get("bundle_id") == bundle_id:
            handle.score_cache_populated = False

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
