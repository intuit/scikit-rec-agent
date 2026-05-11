"""sweep_methods tool — deterministic, capability-filtered, idempotent sweeps.

Trains multiple methods on one bundle, evaluates them on the same metric set,
and returns a ranked leaderboard. Capability-incompatible combos are dropped
before training. The sweep is idempotent: methods whose
``(bundle_id, config)`` already exist in ``session.trained_models`` reuse the
cached handle. Per-method failures don't stop the sweep — they're recorded
with ``status='error'`` and the others continue.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
from typing import Any

import pandas as pd

from scikit_rec_agent.tools import Tool, err, ok

# ---------------------------------------------------------------------------
# Compatibility filter (mirrors scikit-rec's factory guard rails)
# ---------------------------------------------------------------------------

# scorer_type → set of estimator-family tags it accepts. We tag estimator
# families as 'tabular', 'embedding', 'sequential' to match what scikit-rec's
# factory branches on. The exact mapping reproduces the guard rails at
# skrec/orchestrator/factory.py (_TABULAR_SCORER_TYPES /
# _EMBEDDING_INCOMPATIBLE_SCORERS / sequential / hierarchical branches).
_SCORER_ESTIMATOR_COMPAT: dict[str, set[str]] = {
    "universal": {"tabular", "embedding"},
    "independent": {"tabular"},
    "multiclass": {"tabular"},
    "multioutput": {"tabular"},
    "sequential": {"sequential"},
    "hierarchical": {"sequential"},
}

_RECOMMENDER_SCORER_COMPAT: dict[str, set[str]] = {
    "ranking": {"universal", "multiclass", "multioutput", "independent"},
    "bandits": {"universal", "independent"},
    "uplift": {"universal", "independent"},
    "gcsl": {"universal", "independent"},
    "sequential": {"sequential"},
    "hierarchical_sequential": {"hierarchical"},
}


# ---------------------------------------------------------------------------
# Auto-sweep table — opinionated default per contract shape
# ---------------------------------------------------------------------------

_TABULAR_XGB = {
    "ml_task": "classification",
    "xgboost": {"n_estimators": 50, "max_depth": 4, "learning_rate": 0.1},
}

_TABULAR_REGRESSION = {
    "ml_task": "regression",
    "xgboost": {"n_estimators": 50, "max_depth": 4, "learning_rate": 0.1},
}


# Embedding-family configs.  Each maps to one entry in scikit-rec's
# orchestrator.factory._EMBEDDING_ESTIMATOR_MAP. The factory wires the params
# straight through to the underlying estimator's __init__, so the parameter
# names below MUST match each class's signature (n_factors for MF,
# {gmf,mlp}_embedding_dim for NCF, {user,item,final}_embedding_dim for
# Two-Tower, embedding_dim for DCN/NFM). Epoch counts are deliberately low
# so a default sweep finishes in tens of seconds on small samples — users
# can still pass an explicit `methods` list with bigger configs when they
# want more thorough training.
def _embedding_method(short_name: str, model_type: str, params: dict) -> dict:
    return {
        "short_name": short_name,
        "recommender_type": "ranking",
        "scorer_type": "universal",
        "estimator_type": "embedding",
        "estimator_config": {
            "estimator_type": "embedding",
            "embedding": {"model_type": model_type, "params": params},
        },
    }


# Auto-sweep embedding families.
#
# All five families are included. NCF / Two-Tower / DCN / NFM previously
# segfaulted inside torch's BCE when invoked after a numpy-based MF in the
# same process; the agent's `__init__.py` pins OMP_NUM_THREADS /
# MKL_NUM_THREADS / VECLIB_MAXIMUM_THREADS to 1 to avoid the macOS
# Accelerate state pollution that caused it. Param names below match each
# estimator's signature (n_factors for MF, {gmf,mlp}_embedding_dim for NCF,
# {user,item,final}_embedding_dim for Two-Tower, embedding_dim for DCN /
# NFM); list-typed architectural params (mlp_layers, hidden_dim1) take the
# values used by scikit-rec's own integration tests so they are known good.
# Epoch counts are deliberately low so a default sweep finishes in tens of
# seconds — users wanting more thorough training can pass an explicit
# `methods` list with bigger configs.
_EMBEDDING_FAMILIES = [
    _embedding_method("mf_universal", "matrix_factorization", {"n_factors": 16, "epochs": 5, "random_state": 42}),
    _embedding_method(
        "ncf_universal",
        "ncf",
        {
            "ncf_type": "neumf",
            "gmf_embedding_dim": 8,
            "mlp_embedding_dim": 8,
            "mlp_layers": [16, 8],
            "dropout": 0.1,
            "learning_rate": 0.01,
            "epochs": 3,
            "batch_size": 32,
            "random_state": 42,
        },
    ),
    _embedding_method(
        "two_tower_universal",
        "two_tower",
        {
            "user_embedding_dim": 16,
            "item_embedding_dim": 16,
            "final_embedding_dim": 8,
            "user_tower_hidden_dim1": 32,
            "item_tower_hidden_dim1": 32,
            "epochs": 3,
            "batch_size": 32,
            "random_state": 42,
        },
    ),
    _embedding_method(
        "dcn_universal",
        "deep_cross_network",
        {
            "embedding_dim": 16,
            "num_cross_layers": 2,
            "deep_hidden_dim1": 32,
            "epochs": 3,
            "batch_size": 32,
            "random_state": 42,
        },
    ),
    _embedding_method(
        "nfm_universal",
        "neural_factorization",
        {"embedding_dim": 16, "mlp_hidden_dim1": 32, "epochs": 3, "batch_size": 32, "random_state": 42},
    ),
]

_AUTO_SWEEPS: dict[str, list[dict[str, Any]]] = {
    # The whole point of scikit-rec is that one bundle (interactions + users +
    # items) feeds every estimator family — see
    # scikit-rec/examples/generic/factory_pipeline_demo.ipynb. The auto-sweep
    # mirrors that: tabular XGBoost, then the five embedding families
    # (MF, NCF, Two-Tower, DCN, NFM), all on the same data.
    "long_interactions": [
        {
            "short_name": "xgb_universal",
            "recommender_type": "ranking",
            "scorer_type": "universal",
            "estimator_type": "tabular",
            "estimator_config": _TABULAR_XGB,
        },
        *_EMBEDDING_FAMILIES,
    ],
    "long_with_timestamp": [
        {
            "short_name": "xgb_universal",
            "recommender_type": "ranking",
            "scorer_type": "universal",
            "estimator_type": "tabular",
            "estimator_config": _TABULAR_XGB,
        },
        *_EMBEDDING_FAMILIES,
        {
            # SequentialRecommender._build_sequences runs internally on the raw
            # (USER_ID, ITEM_ID, TIMESTAMP, OUTCOME) frame — no separate
            # prebuilt_sequences contract needed. See
            # scikit-rec/skrec/recommender/sequential/sequential_recommender.py:_build_sequences.
            # epochs=2 + small max_len keeps the sweep tractable; users with
            # bigger data can pass an explicit `methods` list with longer training.
            "short_name": "sasrec_sequential",
            "recommender_type": "sequential",
            "scorer_type": "sequential",
            "estimator_type": "sequential",
            "estimator_config": {
                "estimator_type": "sequential",
                "sequential": {
                    "model_type": "sasrec_classifier",
                    "params": {"hidden_units": 16, "max_len": 10, "epochs": 2},
                },
            },
            "recommender_params": {"max_len": 10},
        },
    ],
    "wide_multioutput": [
        # The classifier entry handles binary ITEM_* targets (the common case
        # we ship for); the regression entry handles continuous ITEM_*
        # targets. ``_filter_by_profile`` picks one based on the profiled
        # target_type derived from the ITEM_* columns. Listing both makes the
        # sweep self-selecting — callers pass methods='all' without having
        # to know which mode their data needs.
        {
            "short_name": "xgb_multioutput",
            "recommender_type": "ranking",
            "scorer_type": "multioutput",
            "estimator_type": "tabular",
            "estimator_config": _TABULAR_XGB,
        },
        {
            "short_name": "xgb_multioutput_regression",
            "recommender_type": "ranking",
            "scorer_type": "multioutput",
            "estimator_type": "tabular",
            "estimator_config": _TABULAR_REGRESSION,
        },
    ],
    "multiclass": [
        {
            "short_name": "xgb_multiclass",
            "recommender_type": "ranking",
            "scorer_type": "multiclass",
            "estimator_type": "tabular",
            "estimator_config": _TABULAR_XGB,
        },
    ],
    "prebuilt_sequences": [
        {
            "short_name": "sasrec_sequential",
            "recommender_type": "sequential",
            "scorer_type": "sequential",
            "estimator_type": "sequential",
            "estimator_config": {
                "estimator_type": "sequential",
                "sequential": {
                    "model_type": "sasrec_classifier",
                    "params": {"hidden_units": 32, "max_len": 20},
                },
            },
            "recommender_params": {"max_len": 20},
        },
    ],
    "users_features": [],
    "items_features": [],
}


# ---------------------------------------------------------------------------
# Bundle contract detection
# ---------------------------------------------------------------------------


def _read(path: str | None) -> pd.DataFrame:
    if not path or not os.path.exists(path):
        return pd.DataFrame()
    return pd.read_parquet(path) if path.endswith(".parquet") else pd.read_csv(path)


# ---------------------------------------------------------------------------
# Data-aware sweep selection
#
# The static `_AUTO_SWEEPS` table answers "what methods exist for this
# contract"; the helpers below answer "which of those make sense for THIS
# specific dataset, sized appropriately." Two stages:
#
#   1. _filter_by_profile(method, profile) -> bool
#      drops methods that are wrong for the data shape (e.g. embedding
#      models on <5K rows where they can't beat XGBoost; CF / MF on dense
#      data where collaborative signal doesn't help).
#
#   2. _resize_for_data_scale(method, profile) -> method
#      scales hyperparameters to the data tier — small data gets fewer
#      epochs / smaller embeddings, large data gets more.
#
# Both stages read a profile dict produced by `_profile_bundle`, which is
# computed cheaply from the bundle's source CSVs at sweep time. No
# create_datasets / split_data API change required.
#
# Threshold values are heuristic defaults chosen to be sensible across
# common recsys workloads — they should be tuned on real workload data
# over time. See the SCALE_TIERS table below; comments explain each rule.
# ---------------------------------------------------------------------------


def _profile_bundle(bundle) -> dict[str, Any]:
    """Cheap profile from the bundle's interactions / users / items source
    files. Returns a dict the filter+resize pipeline can read.

    Computed live (not cached on the bundle) because:
    - it's fast (one CSV read + a few groupbys on data the user already
      handed us),
    - keeping it stateless avoids cache-invalidation bugs when split_data
      mutates source_paths in place.
    """
    from scikit_rec_agent.tools.datasets import _resolve_interactions_path

    df = _read(_resolve_interactions_path(bundle.source_paths))
    if df.empty:
        return {
            "n_rows": 0,
            "n_users": 0,
            "sparsity": None,
            "target_type": None,
            "has_timestamps": False,
            "has_user_features": False,
            "has_item_features": False,
        }

    n_rows = int(len(df))

    sparsity: float | None = None
    n_users = 0
    if "USER_ID" in df.columns and "ITEM_ID" in df.columns:
        n_users = int(df["USER_ID"].nunique())
        n_items = int(df["ITEM_ID"].nunique())
        if n_users > 0 and n_items > 0:
            sparsity = 1.0 - (n_rows / (n_users * n_items))
    elif "USER_ID" in df.columns:
        n_users = int(df["USER_ID"].nunique())

    target_type: str | None = None
    if "OUTCOME" in df.columns:
        non_null = df["OUTCOME"].dropna()
        n_unique = int(non_null.nunique())
        if n_unique <= 1:
            target_type = "constant"
        elif set(non_null.unique()).issubset({0, 1, 0.0, 1.0}):
            target_type = "binary"
        else:
            target_type = "continuous"
    else:
        # Wide_multioutput / multiclass shape: no OUTCOME, targets are the
        # ITEM_* columns. Sample their unique values to classify across all
        # targets — if every target is binary, the auto-sweep stays on the
        # classifier branch; if any target is continuous, switch to the
        # regression branch via the resize step. Single-class targets
        # surface as 'constant' too (transform_data's auto-drop should
        # have removed them, but it's defensive to flag them here as well).
        item_cols = [c for c in df.columns if c.startswith("ITEM_") and c != "ITEM_ID"]
        if item_cols:
            seen: set[float] = set()
            any_non_numeric = False
            for c in item_cols:
                if not pd.api.types.is_numeric_dtype(df[c]):
                    any_non_numeric = True
                    break
                seen.update(float(v) for v in df[c].dropna().unique())
            if any_non_numeric:
                # Non-numeric ITEM_* columns shouldn't reach the profiler in
                # practice (transform_data rejects them upfront), but if
                # they do, the safest tag is None — let the upstream
                # classifier-vs-regressor path handle it.
                target_type = None
            elif len(seen) <= 1:
                target_type = "constant"
            elif seen.issubset({0.0, 1.0}):
                target_type = "binary"
            else:
                target_type = "continuous"

    has_timestamps = "TIMESTAMP" in df.columns

    # Side-feature signals: "has features beyond the canonical ID column".
    # bundle.users having any column other than USER_ID counts as user
    # features; same for items. We count via the underlying DataFrames
    # since that's what the scorers actually consume.
    has_user_features = False
    if bundle.users is not None:
        users_path = bundle.source_paths.get("users")
        users_df = _read(users_path) if users_path else pd.DataFrame()
        has_user_features = bool(set(users_df.columns) - {"USER_ID"})

    has_item_features = False
    if bundle.items is not None:
        items_path = bundle.source_paths.get("items")
        items_df = _read(items_path) if items_path else pd.DataFrame()
        has_item_features = bool(set(items_df.columns) - {"ITEM_ID"})

    return {
        "n_rows": n_rows,
        "n_users": n_users,
        "sparsity": sparsity,
        "target_type": target_type,
        "has_timestamps": has_timestamps,
        "has_user_features": has_user_features,
        "has_item_features": has_item_features,
    }


# Data-scale tiers — one knob (n_rows) drives multiple hyperparameters.
# Embeddings need data to train; XGBoost saturates faster but benefits from
# more trees on large data. Cutoffs are heuristic; tune on real workloads.
_SCALE_TIERS = [
    {
        "max_rows": 5_000,
        "name": "tiny",
        "epochs": 2,
        "embedding_dim": 8,
        "n_factors": 8,
        "xgb_n_estimators": 30,
        "xgb_max_depth": 3,
        "lgbm_n_estimators": 50,
        "lgbm_num_leaves": 15,
    },
    {
        "max_rows": 100_000,
        "name": "small",
        "epochs": 5,
        "embedding_dim": 16,
        "n_factors": 16,
        "xgb_n_estimators": 100,
        "xgb_max_depth": 5,
        "lgbm_n_estimators": 100,
        "lgbm_num_leaves": 31,
    },
    {
        "max_rows": 1_000_000,
        "name": "medium",
        "epochs": 10,
        "embedding_dim": 32,
        "n_factors": 32,
        "xgb_n_estimators": 200,
        "xgb_max_depth": 6,
        "lgbm_n_estimators": 300,
        "lgbm_num_leaves": 63,
    },
    {
        "max_rows": float("inf"),
        "name": "large",
        "epochs": 20,
        "embedding_dim": 64,
        "n_factors": 64,
        "xgb_n_estimators": 300,
        "xgb_max_depth": 8,
        "lgbm_n_estimators": 500,
        "lgbm_num_leaves": 127,
    },
]


def _scale_tier(n_rows: int) -> dict[str, Any]:
    for tier in _SCALE_TIERS:
        if n_rows < tier["max_rows"]:
            return tier
    return _SCALE_TIERS[-1]


def _filter_by_profile(method: dict[str, Any], profile: dict[str, Any]) -> tuple[bool, str | None]:
    """Decide whether to keep a method given the data profile.
    Returns (keep, reason_if_dropped).

    Rules per family:
    - **xgb_*** : always keep — safe baseline at every scale.
    - **mf_universal** : keep if sparsity >= 0.90 and n_rows >= 1000. MF
      shines in the classic CF regime; on dense data the collaborative
      signal collapses, on tiny data ALS can't factorize stably.
    - **ncf / two_tower / dcn / nfm** : keep if n_rows >= 5000. Deep
      models with thousands of parameters need enough rows to train
      meaningfully; XGBoost dominates below that.
    - **two_tower** additionally requires has_user_features OR
      has_item_features — its whole point is feature towers.
    - **dcn / nfm** also benefit from features but can run without; we
      keep the n_rows floor only.
    - **sasrec_sequential** : keep if has_timestamps AND n_rows >= 1000.
      Sequential models are data-efficient but need the timestamp signal.
    - **continuous target** : drop classification configs across the board
      (the resize step swaps regression in for the kept XGBoost entries).
    """
    short = method.get("short_name", "")
    n_rows = profile.get("n_rows", 0) or 0
    target_type = profile.get("target_type")

    # Wide_multioutput classifier vs regression auto-pick. The auto-sweep
    # lists both entries; the profile's target_type (derived from ITEM_*
    # uniques) decides which to keep. binary → classifier; continuous →
    # regression; constant → drop both (transform_data should have caught
    # it but defensive); None → keep both (let upstream surface the issue).
    if "multioutput" in short:
        is_regression_entry = "regression" in short
        if target_type == "binary" and is_regression_entry:
            return False, (
                f"{short} dropped: data has binary ITEM_* targets, the "
                "classifier-mode multioutput entry is the right pick."
            )
        if target_type == "continuous" and not is_regression_entry:
            return False, (
                f"{short} dropped: data has continuous ITEM_* targets, the "
                "regression-mode multioutput entry is the right pick."
            )
        if target_type == "constant":
            return False, f"{short} dropped: every ITEM_* target is single-class."

    if target_type == "continuous":
        # XGBoost: resize swaps ml_task='classification' → 'regression' below.
        # SASRec/HRNN: resize swaps `*_classifier` model_type → `*_regressor`.
        # Embedding families (MF/NCF/Two-Tower/DCN/NFM): the auto-sweep configs
        # are classifier-shaped (BCE loss, etc.) and we don't have a clean
        # auto-swap for them today, so they're filtered out here. Users who
        # want to compare regression variants of embedding families can
        # supply explicit method dicts.
        if "xgb" in short:
            pass  # resize swaps ml_task
        elif any(family in short for family in ("sasrec", "hrnn")):
            pass  # resize swaps model_type below
        else:
            return False, (
                f"target is continuous; {short} has no auto-swap to a regression "
                "variant in this version. Pass an explicit method dict if you want it."
            )

    if "mf" in short and "_universal" in short:
        sparsity = profile.get("sparsity")
        # 0.90 floor (was 0.95): standard CF benchmarks live in 0.80–0.95;
        # MovieLens-1M sits at ~0.949, retail/catalogue datasets around 0.90.
        # The original 0.95 threshold filtered most realistic CF data out.
        if sparsity is None or sparsity < 0.90:
            return False, f"MF needs sparsity ≥ 0.90 (CF regime); got {sparsity}"
        if n_rows < 1_000:
            return False, f"MF needs n_rows ≥ 1000 to factorize stably; got {n_rows}"
        # Upper bound: scikit-rec's MatrixFactorizationEstimator runs ALS
        # in a pure-numpy Python loop, doing one (n_factors × n_factors)
        # ridge solve per user per iteration. At ~500K users / 5M rows
        # the loop wedges the laptop without producing a single log line
        # (observed at ~1M users / 13.7M rows for ~9 hours, no progress).
        # See skrec/estimator/embedding/matrix_factorization_estimator.py
        # :130-159. Until the upstream ships a vectorized ALS, drop MF
        # at this scale and surface the reason in the leaderboard so
        # callers know it isn't a silent omission.
        n_users = profile.get("n_users") or 0
        if n_rows > 5_000_000 or n_users > 500_000:
            return False, (
                f"MF dropped at this scale: scikit-rec's ALS implementation "
                f"runs pure-numpy per-user ridge solves in a Python loop and "
                f"wedges on n_users>500K or n_rows>5M (observed: n_users="
                f"{n_users}, n_rows={n_rows}). Vectorized ALS isn't upstream "
                f"yet; until then MF is out at this scale. Other CF baselines "
                f"(NCF / DCN / NFM / Two-Tower) handle large data via "
                f"mini-batch SGD and stay in the sweep. To override (e.g. "
                f"overnight run or after upstream ships vectorized ALS), "
                f"call `train_model` directly with the mf_universal config "
                f"instead of going through `sweep_methods` — the gate only "
                f"fires for the auto-sweep path."
            )
        return True, None

    if any(family in short for family in ("ncf", "two_tower", "dcn", "nfm")):
        if n_rows < 5_000:
            return False, f"{short} needs n_rows ≥ 5000 to beat XGBoost; got {n_rows}"
        if "two_tower" in short:
            if not (profile.get("has_user_features") or profile.get("has_item_features")):
                return False, "Two-Tower needs user OR item features"
        return True, None

    if "sasrec" in short:
        if not profile.get("has_timestamps"):
            return False, "SASRec needs TIMESTAMP column"
        if n_rows < 1_000:
            return False, f"SASRec needs n_rows ≥ 1000; got {n_rows}"
        return True, None

    return True, None  # XGBoost variants and unknown methods default to keep


def _resize_for_data_scale(method: dict[str, Any], profile: dict[str, Any]) -> dict[str, Any]:
    """Return a copy of `method` with hyperparameters scaled to the data
    tier picked by `_scale_tier(n_rows)`. The original method is not mutated.

    Rules per estimator family:
    - **tabular xgboost** : tier sets n_estimators, max_depth, ml_task
      (classification → regression for continuous targets).
    - **tabular lightgbm** : tier sets n_estimators, num_leaves, ml_task.
    - **embedding (matrix_factorization)** : tier sets n_factors, epochs.
    - **embedding (ncf / dcn / nfm)** : tier sets embedding_dim, epochs.
    - **embedding (two_tower)** : tier sets user/item/final_embedding_dim,
      epochs.
    - **sequential (sasrec / hrnn)** : tier sets hidden_units, max_len,
      epochs.
    """
    import copy

    new = copy.deepcopy(method)
    n_rows = profile.get("n_rows", 0) or 0
    tier = _scale_tier(n_rows)
    target_type = profile.get("target_type")
    short = new.get("short_name", "")
    estimator_config = new.get("estimator_config", {})

    if "xgboost" in estimator_config:
        if target_type == "continuous":
            estimator_config["ml_task"] = "regression"
        estimator_config["xgboost"]["n_estimators"] = tier["xgb_n_estimators"]
        estimator_config["xgboost"]["max_depth"] = tier["xgb_max_depth"]

    if "lightgbm" in estimator_config:
        if target_type == "continuous":
            estimator_config["ml_task"] = "regression"
        estimator_config["lightgbm"]["n_estimators"] = tier["lgbm_n_estimators"]
        estimator_config["lightgbm"]["num_leaves"] = tier["lgbm_num_leaves"]

    embedding = estimator_config.get("embedding", {})
    if embedding:
        params = embedding.setdefault("params", {})
        model_type = embedding.get("model_type")
        if model_type == "matrix_factorization":
            params["n_factors"] = tier["n_factors"]
            params["epochs"] = tier["epochs"]
        elif model_type == "ncf":
            params["gmf_embedding_dim"] = tier["embedding_dim"] // 2 or 4
            params["mlp_embedding_dim"] = tier["embedding_dim"] // 2 or 4
            params["epochs"] = tier["epochs"]
        elif model_type == "two_tower":
            params["user_embedding_dim"] = tier["embedding_dim"]
            params["item_embedding_dim"] = tier["embedding_dim"]
            params["final_embedding_dim"] = max(tier["embedding_dim"] // 2, 4)
            params["epochs"] = tier["epochs"]
        elif model_type in ("deep_cross_network", "neural_factorization"):
            params["embedding_dim"] = tier["embedding_dim"]
            params["epochs"] = tier["epochs"]

    sequential = estimator_config.get("sequential", {})
    if sequential:
        # Swap *_classifier → *_regressor when the target is continuous.
        # SASRec and HRNN both expose paired classifier / regressor variants
        # via the orchestrator factory, so we can do this without changing
        # any other config piece. The filter step (_filter_by_profile) above
        # relies on this swap actually happening — keep them in sync.
        seq_model_type = sequential.get("model_type", "")
        if target_type == "continuous" and seq_model_type.endswith("_classifier"):
            sequential["model_type"] = seq_model_type.replace("_classifier", "_regressor")
        params = sequential.setdefault("params", {})
        # Sequential keeps a tighter lid on hidden_units to control torch
        # memory; max_len and epochs scale with the tier.
        params["hidden_units"] = max(tier["embedding_dim"] // 2, 8)
        params["max_len"] = max(min(tier["embedding_dim"] // 2, 50), 10)
        params["epochs"] = max(tier["epochs"], 2)

    new["short_name"] = f"{short}_{tier['name']}" if not short.endswith(tier["name"]) else short
    return new


def _data_aware_methods(
    method_list: list[dict[str, Any]], profile: dict[str, Any]
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Apply filter then resize. Returns (kept_methods, dropped_with_reason)."""
    kept: list[dict[str, Any]] = []
    dropped: list[dict[str, Any]] = []
    for method in method_list:
        keep, reason = _filter_by_profile(method, profile)
        if not keep:
            dropped.append({"method": dict(method), "reason": reason})
            continue
        kept.append(_resize_for_data_scale(method, profile))
    return kept, dropped


def contract_from_dataframe(df: pd.DataFrame) -> str:
    """Single source of truth for "what data shape does this look like".

    Vocabulary: ``long_interactions`` / ``long_with_timestamp`` /
    ``long_multi_reward`` / ``wide_multioutput`` / ``multiclass`` /
    ``prebuilt_sequences`` / ``sessions`` / ``unknown``.

    Used by both ``_detect_bundle_contract`` (sweep flow's data-aware
    selection) and ``_detect_dataset_type`` (datasets.py's choice of
    scikit-rec Dataset class). Putting both on top of a shared helper
    means the sweep's ``contract`` and a bundle's ``dataset_type`` can
    never disagree about the same data — they read from the same rules.
    """
    if df.empty:
        return "unknown"
    cols = set(df.columns)
    item_star = [c for c in df.columns if c.startswith("ITEM_") and c != "ITEM_ID"]

    if "ITEM_SEQUENCE" in cols and "USER_ID" in cols:
        return "prebuilt_sequences"
    if "SESSION_SEQUENCES" in cols:
        return "sessions"
    if {"USER_ID", "ITEM_ID", "OUTCOME"}.issubset(cols):
        if "TIMESTAMP" in cols:
            return "long_with_timestamp"
        outcome_aux = [c for c in cols if c.startswith("OUTCOME_")]
        if outcome_aux:
            return "long_multi_reward"
        return "long_interactions"
    if "USER_ID" in cols and "ITEM_ID" in cols and "OUTCOME" not in cols:
        return "multiclass"
    if "USER_ID" in cols and len(item_star) >= 2:
        return "wide_multioutput"
    return "unknown"


def _detect_bundle_contract(bundle) -> str:
    from scikit_rec_agent.tools.datasets import _resolve_interactions_path

    return contract_from_dataframe(_read(_resolve_interactions_path(bundle.source_paths)))


# ---------------------------------------------------------------------------
# Method resolution
# ---------------------------------------------------------------------------


def _normalise_method(m: dict[str, Any]) -> dict[str, Any]:
    out = dict(m)
    out.setdefault("recommender_type", "ranking")
    out.setdefault("scorer_type", "universal")
    out.setdefault("estimator_type", "tabular")
    out.setdefault("estimator_config", _TABULAR_XGB)
    out.setdefault("short_name", _short_name(out))
    return out


def _short_name(method: dict[str, Any]) -> str:
    parts = [
        method.get("estimator_type", "tabular"),
        method.get("scorer_type", "universal"),
        method.get("recommender_type", "ranking"),
    ]
    return "_".join(parts)


def _broad_sweep_for_contract(contract: str) -> list[dict[str, Any]]:
    """Every {recommender × scorer × tabular-estimator} triple compatible with
    the contract. Sequential scorers only included if the contract is
    sequence-shaped."""
    methods: list[dict[str, Any]] = []
    contract_compatible = _contract_compatible_scorers(contract)
    for rec, scorers in _RECOMMENDER_SCORER_COMPAT.items():
        for scorer in scorers & contract_compatible:
            for estimator_family in _SCORER_ESTIMATOR_COMPAT[scorer]:
                if estimator_family != "tabular":
                    continue
                methods.append(
                    {
                        "short_name": f"{rec}_{scorer}_tabular",
                        "recommender_type": rec,
                        "scorer_type": scorer,
                        "estimator_type": "tabular",
                        "estimator_config": _TABULAR_XGB,
                        **({"recommender_params": {"control_item_id": "control"}} if rec == "uplift" else {}),
                    }
                )
    return methods


def _contract_compatible_scorers(contract: str) -> set[str]:
    if contract in ("long_interactions", "long_with_timestamp", "long_multi_reward"):
        return {"universal", "independent"}
    if contract == "wide_multioutput":
        return {"multioutput", "universal"}
    if contract == "multiclass":
        return {"multiclass"}
    if contract == "prebuilt_sequences":
        return {"sequential"}
    if contract == "sessions":
        return {"hierarchical"}
    return set()


# ---------------------------------------------------------------------------
# Capability filter
# ---------------------------------------------------------------------------


def _is_compatible(method: dict[str, Any]) -> tuple[bool, str | None]:
    rec = method.get("recommender_type")
    scorer = method.get("scorer_type")
    est = method.get("estimator_type", "tabular")

    rec_scorers = _RECOMMENDER_SCORER_COMPAT.get(rec)
    if rec_scorers is None:
        return False, f"recommender_type '{rec}' not in capability matrix."
    if scorer not in rec_scorers:
        return False, f"recommender '{rec}' does not accept scorer '{scorer}'."

    est_set = _SCORER_ESTIMATOR_COMPAT.get(scorer)
    if est_set is None:
        return False, f"scorer_type '{scorer}' not in capability matrix."
    if est not in est_set:
        return False, (f"scorer '{scorer}' does not accept estimator family '{est}' (allowed: {sorted(est_set)}).")
    return True, None


def _filter_by_capability(methods: list[dict[str, Any]]):
    runnable, dropped = [], []
    for m in methods:
        compatible, reason = _is_compatible(m)
        if compatible:
            runnable.append(m)
        else:
            dropped.append({"method": dict(m), "reason": reason})
    return runnable, dropped


# ---------------------------------------------------------------------------
# Idempotency hash
# ---------------------------------------------------------------------------


def _config_canonical(method: dict[str, Any]) -> dict[str, Any]:
    return {k: method[k] for k in sorted(method) if k != "short_name"}


def _config_hash(bundle_id: str, method: dict[str, Any]) -> str:
    canonical = json.dumps(
        {"bundle_id": bundle_id, "method": _config_canonical(method)},
        sort_keys=True,
        default=str,
    )
    return hashlib.sha256(canonical.encode()).hexdigest()


# ---------------------------------------------------------------------------
# Train + evaluate per method
# ---------------------------------------------------------------------------


_METRIC_NAME_RX = re.compile(r"^(?P<name>[A-Za-z_]+?)(?:[_@](?P<k>\d+))?$")


def _split_metric_spec(metric_spec: str, default_k: int) -> tuple[str, int]:
    """Accept 'NDCG_at_k', 'NDCG_at_10', 'NDCG@10', 'roc_auc'.

    scikit-rec's canonical metric names all end in literal '_at_k' (not
    '_at_<number>'). When an LLM passes 'NDCG_at_10' the regex's optional
    `_<digits>` group eats the trailing `_10`, leaving 'NDCG_at' as the
    name — which scikit-rec's metric registry rejects with 'Unknown
    metric'. Re-append '_k' when we detect that pattern so the canonical
    name survives extraction.
    """
    m = _METRIC_NAME_RX.match(metric_spec)
    if not m:
        return metric_spec, default_k
    name = m.group("name")
    k = m.group("k")
    if name.endswith("_at"):
        name = name + "_k"
    return name, int(k) if k else default_k


def _build_train_args(method: dict[str, Any]) -> dict[str, Any]:
    config = {
        "recommender_type": method["recommender_type"],
        "scorer_type": method["scorer_type"],
        "estimator_config": method["estimator_config"],
    }
    if "recommender_params" in method:
        config["recommender_params"] = method["recommender_params"]
    return config


def _train_and_evaluate(
    method: dict[str, Any],
    bundle_id: str,
    sweep_cache_id: str,
    metrics: list[str],
    eval_top_k: int,
    evaluator_type: str,
    session,
    per_label: bool = False,
) -> dict[str, Any]:
    """Train + evaluate one method. Returns a row dict for the leaderboard.

    `sweep_cache_id` is the sweep's deterministic id (used for caching across
    re-runs); the actual model_id we return to the caller is whatever
    train_model assigned (an auto-generated `<recommender>_<timestamp>` id).
    We record the alias in `session.sweep_cache` so a later sweep call can
    look up "have I trained this exact (bundle, config) before?" without
    mutating the keys of `session.trained_models` — that contract stays
    stable for any other tool / agent code that holds onto train_model's
    returned id.
    """
    from scikit_rec_agent.tools.evaluation import TOOL_EVALUATE_MODEL
    from scikit_rec_agent.tools.training import TOOL_TRAIN_MODEL

    config = _build_train_args(method)
    train_result = TOOL_TRAIN_MODEL.fn(
        model_name=sweep_cache_id,
        config=config,
        bundle_id=bundle_id,
        session=session,
    )
    if train_result["status"] != "ok":
        return {
            "status": "error",
            "error": train_result.get("message"),
            "category": train_result.get("category", "unknown"),
            "metrics": {},
            "model_id": None,
        }

    trained_model_id = train_result["data"]["model_id"]
    session.sweep_cache[sweep_cache_id] = trained_model_id
    handle = session.trained_models[trained_model_id]

    metric_pairs = [_split_metric_spec(m, eval_top_k) for m in metrics]
    metric_names = sorted({n for n, _ in metric_pairs})
    k_values = sorted({k for _, k in metric_pairs})

    eval_result = TOOL_EVALUATE_MODEL.fn(
        model_id=trained_model_id,
        evaluator_type=evaluator_type,
        metrics=metric_names,
        k_values=k_values,
        session=session,
        per_label=per_label,
    )
    if eval_result["status"] != "ok":
        return {
            "status": "error",
            "error": eval_result.get("message"),
            "category": eval_result.get("category", "unknown"),
            "metrics": dict(handle.metrics),
            "model_id": trained_model_id,
        }

    metric_map = {f"{name}@{k}": handle.metrics.get(f"{name}@{k}") for name, k in metric_pairs}
    # When per_label fired, surface the raw per-target dicts on each leaderboard
    # row too so the agent can render per-label tables without re-querying
    # handle.metrics. The macro-averaged scalar still appears under
    # metric_map for sort / ranking purposes.
    per_label_map: dict[str, dict[str, float]] = {}
    if per_label:
        for entry in eval_result["data"].get("results", []):
            if "per_label" in entry:
                per_label_map[f"{entry['metric']}@{entry['k']}"] = entry["per_label"]
    payload: dict[str, Any] = {
        "status": "ok",
        "metrics": metric_map,
        "model_id": trained_model_id,
    }
    if per_label_map:
        payload["per_label"] = per_label_map
    return payload


def _evaluate_existing(
    model_id: str,
    metrics: list[str],
    eval_top_k: int,
    evaluator_type: str,
    session,
) -> dict[str, Any]:
    """Run evaluate_model on an already-trained handle and return the
    sweep-row payload. Used when the sweep finds a cached handle whose
    metrics dict is empty (typical after a prior sweep where training
    succeeded but evaluation failed for an unrelated reason like a bad
    metric name)."""
    from scikit_rec_agent.tools.evaluation import TOOL_EVALUATE_MODEL

    handle = session.trained_models.get(model_id)
    if handle is None:
        return {
            "status": "error",
            "error": f"No handle for model_id '{model_id}'",
            "category": "unknown",
        }

    metric_pairs = [_split_metric_spec(m, eval_top_k) for m in metrics]
    metric_names = sorted({n for n, _ in metric_pairs})
    k_values = sorted({k for _, k in metric_pairs})

    eval_result = TOOL_EVALUATE_MODEL.fn(
        model_id=model_id,
        evaluator_type=evaluator_type,
        metrics=metric_names,
        k_values=k_values,
        session=session,
    )
    if eval_result["status"] != "ok":
        return {
            "status": "error",
            "error": eval_result.get("message"),
            "category": eval_result.get("category", "unknown"),
        }
    metric_map = {f"{name}@{k}": handle.metrics.get(f"{name}@{k}") for name, k in metric_pairs}
    return {"status": "ok", "metrics": metric_map}


def _resolve_primary_metric_key(primary_metric: str, eval_top_k: int) -> str:
    name, k = _split_metric_spec(primary_metric, eval_top_k)
    return f"{name}@{k}"


# Set of scikit-rec metric names where LOWER values are better. Anything not
# in this set is sorted higher-is-better. Today scikit-rec ships only
# higher-is-better metrics (NDCG / MAP / MRR / precision / recall /
# average_reward / ROC-AUC / PR-AUC / expected_reward); this set is the
# extension point for when loss-shaped metrics land.
_LOWER_IS_BETTER_METRICS: set[str] = {
    "logloss",
    "rmse",
    "mse",
    "mae",
    "log_loss",
    "cross_entropy",
}


def _metric_higher_is_better(primary_metric: str) -> bool:
    """Return True if higher metric values are better, False for loss-shaped."""
    name, _ = _split_metric_spec(primary_metric, default_k=10)
    return name.lower() not in _LOWER_IS_BETTER_METRICS


# ---------------------------------------------------------------------------
# Tool entrypoint
# ---------------------------------------------------------------------------


def _sweep_methods(
    bundle_id: str,
    metrics: list[str],
    primary_metric: str,
    session,
    methods: Any = "auto",
    eval_top_k: int = 10,
    evaluator_type: str = "simple",
    skip_existing: bool = True,
    auto_save_top_k: int = 0,
    name_prefix: str = "sweep",
    drop_non_winners: bool | None = None,
    per_label: bool | None = None,
    confirmed_all: bool = False,
) -> dict[str, Any]:
    bundle = session.loaded_datasets.get(bundle_id)
    if bundle is None:
        return err("BundleNotFound", f"No bundle '{bundle_id}' in session.")

    contract = _detect_bundle_contract(bundle)

    # Programmatic MUST-ASK guardrail: drop_non_winners must be explicit on
    # large bundles. Default None means "agent did not pass a value" — on
    # bundles with >100K rows, a sweep that retains every trained
    # recommender easily holds 1–3 GB of user embeddings (MF + NCF +
    # Two-Tower together). Refuse upfront so the agent asks the user
    # rather than silently consuming laptop RAM. Pass True or False
    # explicitly to override.
    #
    # Skipped when methods="list" — that mode returns the menu without
    # training, so drop_non_winners is irrelevant. Without this skip, the
    # menu path itself would block on a question the user can't answer
    # until they've SEEN the menu (chicken-and-egg).
    is_list_mode = isinstance(methods, str) and methods == "list"
    if drop_non_winners is None and not is_list_mode:
        profile_quick = _profile_bundle(bundle)
        n_rows = profile_quick.get("n_rows") or 0
        if n_rows > 100_000:
            return err(
                "MissingDecision",
                (
                    f"Bundle has {n_rows:,} rows. A sweep that retains every trained "
                    "recommender can hold 1–3 GB of user embeddings (MF + NCF + "
                    "Two-Tower together). Pass `drop_non_winners=True` to release "
                    "non-winning recommenders after evaluation, or "
                    "`drop_non_winners=False` to keep them all. The decision must be "
                    "explicit on bundles >100K rows to avoid silent memory pressure."
                ),
                hint=(
                    "Ask the user: 'Drop non-winners after evaluation to save RAM? "
                    "(true / false)'. Pass their answer as drop_non_winners=..."
                ),
                category="missing_decision",
            )
        drop_non_winners = False
    if drop_non_winners is None:
        drop_non_winners = False

    # Programmatic MUST-ASK guardrail: per_label must be explicit on
    # multi-target multioutput bundles. Mirror the evaluate_model gate so
    # the sweep path doesn't bypass it — without this, a sweep on a
    # 13-target QBO bundle would silently produce macro-only metrics
    # while individual evaluate_model calls would have asked.
    # Skipped for methods="list" (no training happens). Two trigger
    # conditions, matching the evaluate_model gate (and the system prompt
    # MUST-ASK rule §2):
    #   (a) wide_multioutput bundle with ≥2 ITEM_* targets
    #   (b) long-format interactions bundle paired with a classification
    #       metric (roc_auc / pr_auc)
    _CLASSIFICATION_METRIC_NAMES = {"roc_auc", "pr_auc"}
    if per_label is None and not is_list_mode:
        classification_requested = any(m in _CLASSIFICATION_METRIC_NAMES for m in metrics)
        if bundle.dataset_type == "interaction_multioutput":
            from scikit_rec_agent.tools.datasets import _resolve_interactions_path

            inter_path = _resolve_interactions_path(bundle.source_paths)
            n_targets = 0
            if inter_path:
                df_for_gate = _read(inter_path)
                n_targets = sum(1 for c in df_for_gate.columns if c.startswith("ITEM_") and c != "ITEM_ID")
            if n_targets >= 2:
                return err(
                    "MissingDecision",
                    (
                        f"Bundle has {n_targets} ITEM_* targets. The MUST-ASK policy "
                        "requires per_label to be explicit on multi-target multioutput "
                        "sweeps — defaulting silently to macro-averaged hides exactly the "
                        "per-target detail users typically want. Pass per_label=True to "
                        "produce Dict[str, float] per metric (one entry per ITEM_*), or "
                        "per_label=False for a single macro-averaged scalar per method."
                    ),
                    hint=(
                        "Ask the user: 'Per-target metrics (one number per ITEM_*) or a "
                        "single macro-averaged scalar?'. Pass their answer as per_label=..."
                    ),
                    category="missing_decision",
                )
        elif bundle.dataset_type == "interactions" and classification_requested:
            requested = sorted(m for m in metrics if m in _CLASSIFICATION_METRIC_NAMES)
            return err(
                "MissingDecision",
                (
                    f"Long-format interactions bundle paired with classification metric(s) "
                    f"{requested}. The MUST-ASK policy requires per_label to be explicit — "
                    "per_label=True groups validation predictions by ITEM_ID and runs the "
                    "metric per group (UniversalScorer only); per_label=False returns the "
                    "single macro-averaged scalar per method."
                ),
                hint=(
                    "Ask the user: 'Per-item classification metrics (one number per "
                    "ITEM_ID) or a single macro-averaged scalar?'. Pass their answer "
                    "as per_label=..."
                ),
                category="missing_decision",
            )
    if per_label is None:
        per_label = False

    # Programmatic MUST-ASK guardrail: methods="all" without prior menu
    # listing requires explicit confirmation. Listing the menu via
    # methods="list" and asking the user is the safer flow. Allow "all"
    # only when the caller affirms confirmed_all=True (or has just done
    # methods="list" and is following up — but that requires session
    # state we don't track here, so the flag is the canonical opt-in).
    if isinstance(methods, str) and methods == "all" and not confirmed_all:
        return err(
            "MissingDecision",
            (
                "methods='all' would run every compatible recommender. The MUST-ASK "
                "policy expects the user to pick from the menu first. Either call "
                "sweep_methods(methods='list') to surface the menu and let the user "
                "choose, OR pass confirmed_all=True if the user already said 'all' / "
                "'every option' verbatim."
            ),
            hint=(
                "Default flow: call methods='list', surface the numbered menu, ask "
                "the user which to run, then re-call with methods=[short_names]. "
                "If the user upfront said 'try all', pass confirmed_all=True."
            ),
            category="missing_decision",
        )

    if isinstance(methods, str) and methods == "list":
        # Menu-only mode: return the auto-sweep candidates for this contract,
        # capability-filtered and numbered, without training anything. The
        # agent surfaces this list to the user, the user picks, and the
        # agent then re-calls sweep_methods with an explicit list of
        # short_names. This makes model selection a deliberate user choice
        # instead of an opaque agent decision.
        method_list = list(_AUTO_SWEEPS.get(contract, []))
        runnable, dropped = _filter_by_capability(method_list)
        menu = []
        for i, m in enumerate(runnable, start=1):
            menu.append(
                {
                    "option": i,
                    "short_name": m["short_name"],
                    "recommender_type": m["recommender_type"],
                    "scorer_type": m["scorer_type"],
                    "estimator_type": m["estimator_type"],
                    "estimator_config": m["estimator_config"],
                }
            )
        # Programmatic MUST-ASK guardrail: reshape-vs-stay. wide_multioutput
        # ships only the multioutput entries today; long_interactions ships
        # the full XGBoost + 5 embedding families. If the user wants to
        # broaden the comparison on wide data, melting to long_interactions
        # opens up the universal/independent scorers. Surface the option
        # so the agent ASKS the user rather than defaulting to the narrow
        # menu silently.
        reshape_recommendation: str | None = None
        if contract == "wide_multioutput":
            # Internal agent guidance, not user-verbatim text — kept short
            # so the agent can rephrase for context. The agent should ask
            # before reshaping. Surfaced regardless of menu size: even when
            # _AUTO_SWEEPS["wide_multioutput"] grows beyond today's
            # classifier+regressor pair, the wide → long melt always opens
            # the broader universal-scorer family. Removing the count gate
            # avoids a silent regression if upstream ships a third wide
            # method.
            reshape_recommendation = (
                f"Wide_multioutput ships {len(menu)} method(s); melting to "
                "long_interactions (or long_with_timestamp) opens the "
                "universal-scorer family (~6 methods). Ask the user before "
                "reshaping."
            )

        return ok(
            {
                "bundle_id": bundle_id,
                "contract": contract,
                "available_methods": menu,
                "n_available": len(menu),
                "n_dropped_incompatible": len(dropped),
                "dropped_methods": dropped,
                "reshape_recommendation": reshape_recommendation,
                "instructions": (
                    "Surface this numbered list to the user with a brief "
                    "description of each method (tabular vs embedding family, "
                    "key hyperparameters), and ask which option(s) to run. "
                    "Then call sweep_methods again with `methods=[...]` "
                    "containing the chosen `short_name` strings. To run every "
                    "option, pass methods='all' AND confirmed_all=True only "
                    "when the user has explicitly said 'all' / 'every'."
                ),
            }
        )

    data_aware_dropped: list[dict[str, Any]] = []
    if isinstance(methods, str) and methods == "auto":
        # Data-aware: filter the static auto-sweep table by the bundle's
        # actual data profile, then resize hyperparameters to the data tier.
        # Falls back to the unfiltered/unsized list if the profile comes
        # back empty (e.g. a malformed bundle); we'd rather try too many
        # methods than zero.
        raw_methods = list(_AUTO_SWEEPS.get(contract, []))
        if not raw_methods:
            return err(
                "AutoSweepUnavailable",
                f"No auto sweep is defined for contract shape '{contract}'. "
                f'Pass an explicit `methods` list or change `methods="broad"`.',
            )
        profile = _profile_bundle(bundle)
        if profile.get("n_rows", 0) > 0:
            method_list, data_aware_dropped = _data_aware_methods(raw_methods, profile)
            if not method_list:
                return err(
                    "AutoSweepEmptyAfterFilter",
                    (
                        f"All {len(raw_methods)} auto-sweep candidates were filtered out "
                        f"by the data profile (n_rows={profile['n_rows']}, "
                        f"sparsity={profile['sparsity']}, target={profile['target_type']}). "
                        f"Use methods='all' to override the filter, or pass an explicit list."
                    ),
                    hint="Profile signals don't match any auto-sweep entry's keep rule.",
                    category="auto_sweep_empty_after_filter",
                )
        else:
            method_list = raw_methods
    elif isinstance(methods, str) and methods == "all":
        # Convenience: run every entry from the contract's auto-sweep.
        # Different from "broad" — auto is curated, broad is exhaustive
        # capability-matrix.
        method_list = list(_AUTO_SWEEPS.get(contract, []))
        if not method_list:
            return err(
                "AutoSweepUnavailable",
                f"No methods defined for contract '{contract}'.",
            )
    elif isinstance(methods, str) and methods == "broad":
        method_list = _broad_sweep_for_contract(contract)
        if not method_list:
            return err(
                "BroadSweepUnavailable",
                f"No broad sweep candidates available for contract '{contract}'.",
            )
    elif isinstance(methods, str):
        # Single string that wasn't one of the special tokens above —
        # treat it as a one-element short_name list so models can ask for
        # `methods="xgb_universal"` without first wrapping it in `[...]`.
        # Validate against the contract's auto-sweep so a typo still fails
        # loudly instead of silently turning into an empty sweep.
        available = {m["short_name"]: m for m in _AUTO_SWEEPS.get(contract, [])}
        if methods not in available:
            return err(
                "UnknownMethodShortName",
                (
                    f"methods='{methods}' is not a recognised mode "
                    f"('auto'/'all'/'broad'/'list') and not a short_name in "
                    f"the auto-sweep for contract '{contract}'. "
                    f"Available short_names: {sorted(available)}. "
                    f"To run a single method, pass it as a list: methods=['{methods}']."
                ),
            )
        method_list = [available[methods]]
    elif isinstance(methods, list):
        # Accept either a list of dicts (full method config) or a list of
        # short_names referring to auto-sweep entries. The latter is what the
        # user picks from the menu in `methods="list"` mode.
        if methods and all(isinstance(m, str) for m in methods):
            available = {m["short_name"]: m for m in _AUTO_SWEEPS.get(contract, [])}
            unknown = [m for m in methods if m not in available]
            if unknown:
                return err(
                    "UnknownMethodShortName",
                    f"short_names not in auto-sweep for contract '{contract}': {unknown}. "
                    f"Available: {sorted(available)}",
                )
            method_list = [available[m] for m in methods]
        else:
            method_list = [_normalise_method(m) for m in methods]
    else:
        return err(
            "ArgumentError",
            "methods must be 'auto', 'all', 'broad', 'list', a single short_name string, "
            "or an explicit list of method dicts / short_names.",
        )

    runnable, dropped = _filter_by_capability(method_list)

    primary_key = _resolve_primary_metric_key(primary_metric, eval_top_k)
    leaderboard: list[dict[str, Any]] = []

    for method in runnable:
        config_hash = _config_hash(bundle_id, method)
        short = method.get("short_name") or _short_name(method)
        sweep_cache_id = f"{name_prefix}_{short}_{config_hash[:8]}"

        # Cache lookup: was this exact (bundle, config) trained in a prior
        # sweep call? `session.sweep_cache` aliases the sweep's deterministic
        # id to whatever model_id train_model produced, so we look up the
        # alias and then fetch the handle from session.trained_models. This
        # keeps train_model's returned model_id stable across re-runs (no
        # silent re-keying behind the caller's back).
        cached_train_id = session.sweep_cache.get(sweep_cache_id) if skip_existing else None
        if cached_train_id and cached_train_id in session.trained_models:
            handle = session.trained_models[cached_train_id]
            # If a previous sweep trained this model but its evaluation failed
            # (e.g. bad metric name on the prior call), the cached handle has
            # an empty metrics dict. Re-evaluate it instead of returning a
            # leaderboard row with no numbers — the user expects "cached"
            # to mean "trained AND scored", not "trained but unscored".
            if handle.metrics:
                leaderboard.append(
                    {
                        "method": method,
                        "model_id": cached_train_id,
                        "sweep_cache_id": sweep_cache_id,
                        "status": "cached",
                        "metrics": dict(handle.metrics),
                    }
                )
                continue
            eval_only = _evaluate_existing(
                model_id=cached_train_id,
                metrics=metrics,
                eval_top_k=eval_top_k,
                evaluator_type=evaluator_type,
                session=session,
            )
            if eval_only["status"] == "ok":
                leaderboard.append(
                    {
                        "method": method,
                        "model_id": cached_train_id,
                        "sweep_cache_id": sweep_cache_id,
                        "status": "cached",
                        "metrics": eval_only["metrics"],
                    }
                )
            else:
                leaderboard.append(
                    {
                        "method": method,
                        "model_id": cached_train_id,
                        "sweep_cache_id": sweep_cache_id,
                        "status": "error",
                        "error": eval_only.get("error"),
                        "category": eval_only.get("category"),
                        "metrics": {},
                    }
                )
            continue

        result = _train_and_evaluate(
            method=method,
            bundle_id=bundle_id,
            sweep_cache_id=sweep_cache_id,
            metrics=metrics,
            eval_top_k=eval_top_k,
            evaluator_type=evaluator_type,
            session=session,
            per_label=per_label,
        )
        if result["status"] == "ok":
            row = {
                "method": method,
                "model_id": result["model_id"],
                "sweep_cache_id": sweep_cache_id,
                "status": "ok",
                "metrics": result["metrics"],
            }
            if "per_label" in result:
                row["per_label"] = result["per_label"]
            leaderboard.append(row)
        else:
            leaderboard.append(
                {
                    "method": method,
                    "model_id": result.get("model_id"),
                    "sweep_cache_id": sweep_cache_id,
                    "status": "error",
                    "error": result.get("error"),
                    "category": result.get("category"),
                    "metrics": result.get("metrics", {}),
                }
            )

    # Sort direction: most ranking metrics are higher-is-better (NDCG, MAP,
    # MRR, precision, recall, average_reward, ROC-AUC, PR-AUC, expected_reward),
    # but a future metric registry could add loss-shaped or error-shaped
    # entries (logloss, RMSE) where lower-is-better. Detect that from the
    # metric NAME so we don't silently rank a loss-metric leaderboard
    # backwards. The set covers everything in scikit-rec's RecommenderMetricType
    # today; unknown metrics default to higher-is-better with a guard rail.
    higher_is_better = _metric_higher_is_better(primary_metric)

    def sort_key(row):
        v = row.get("metrics", {}).get(primary_key)
        if v is None:
            return float("inf")  # missing metrics sort last regardless of direction
        v = float(v)
        return -v if higher_is_better else v

    leaderboard.sort(key=sort_key)

    if auto_save_top_k > 0:
        from scikit_rec_agent.tools.registry import TOOL_SAVE_MODEL

        for row in leaderboard[:auto_save_top_k]:
            if row["status"] in ("ok", "cached"):
                save_id = row.get("model_id")
                handle = session.trained_models.get(save_id)
                if handle is not None and handle.recommender is not None:
                    TOOL_SAVE_MODEL.fn(model_id=handle.model_id, session=session)

    if drop_non_winners and auto_save_top_k > 0:
        for row in leaderboard[auto_save_top_k:]:
            handle = session.trained_models.get(row.get("model_id"))
            if handle is not None:
                handle.recommender = None

    payload = {
        "bundle_id": bundle_id,
        "contract": contract,
        "primary_metric": primary_metric,
        "primary_metric_key": primary_key,
        "n_methods_requested": len(method_list),
        "n_runnable": len(runnable),
        "n_dropped_incompatible": len(dropped),
        "dropped_methods": dropped,
        "n_dropped_by_data_profile": len(data_aware_dropped),
        "dropped_by_data_profile": data_aware_dropped,
        "leaderboard": leaderboard,
        "winner": leaderboard[0] if leaderboard and leaderboard[0]["status"] in ("ok", "cached") else None,
    }

    # If every runnable method failed, surface the sweep as a top-level error
    # rather than 'ok with winner=null'. Otherwise the LLM has no structured
    # signal to break out of a retry loop, since per-method errors are buried
    # inside the leaderboard. Pick the most common per-method category so the
    # error envelope's `category` reflects what's actually wrong.
    if leaderboard and all(row["status"] == "error" for row in leaderboard):
        from collections import Counter

        cats = Counter(row.get("category", "unknown") for row in leaderboard)
        dominant_cat = cats.most_common(1)[0][0]
        sample_msg = next(
            (row.get("error") for row in leaderboard if row.get("error")),
            "all sweep methods failed",
        )
        envelope = err(
            "SweepAllMethodsFailed",
            (
                f"All {len(leaderboard)} runnable method(s) failed. "
                f"Dominant failure category: {dominant_cat}. "
                f"Sample error: {sample_msg}"
            ),
            hint=(
                "Check the leaderboard for per-method error messages and "
                "categories. Common fixes: drop or encode object-dtype side "
                "features before create_datasets; pick a different evaluator_type; "
                "or run diagnose_training_failure on the most informative row."
            ),
            category=dominant_cat,
        )
        # Preserve the leaderboard alongside the error so the LLM can still
        # inspect per-method failures without re-running the sweep.
        envelope["data"] = payload
        return envelope

    return ok(payload)


TOOL_SWEEP_METHODS = Tool(
    name="sweep_methods",
    description=(
        "Deterministically train and evaluate multiple modeling methods on the same "
        "dataset bundle, then return a ranked leaderboard. Use this instead of looping "
        "`train_model` + `evaluate_model` by hand when the user wants to compare methods. "
        "Skips capability-incompatible combos before training. Idempotent across sessions: "
        "models with the same (bundle, config) hash are reused. On error in one method, "
        "the others continue."
    ),
    input_schema={
        "type": "object",
        "properties": {
            "bundle_id": {"type": "string"},
            "methods": {
                "description": (
                    "Choose how to drive the sweep. Modes:\n"
                    "- 'list': return the menu of available methods for this bundle "
                    "without training. Use this FIRST so the user can pick.\n"
                    "- 'auto' / 'all': run every curated method for the contract.\n"
                    "- 'broad': every capability-compatible triple from the matrix.\n"
                    "- list of strings: short_names from the menu (e.g. "
                    "['xgb_universal', 'mf_universal']).\n"
                    "- list of dicts: full method configs for advanced users."
                ),
            },
            "metrics": {
                "type": "array",
                "items": {"type": "string"},
                "description": "Metric specs e.g. ['NDCG_at_k', 'roc_auc'] or ['NDCG_at_k@10'].",
            },
            "primary_metric": {
                "type": "string",
                "description": "Metric used to rank the leaderboard, e.g. 'NDCG_at_k' or 'NDCG_at_k@10'.",
            },
            "eval_top_k": {"type": "integer", "default": 10},
            "evaluator_type": {"type": "string", "default": "simple"},
            "skip_existing": {"type": "boolean", "default": True},
            "auto_save_top_k": {"type": "integer", "default": 0},
            "name_prefix": {"type": "string", "default": "sweep"},
            "drop_non_winners": {
                "type": ["boolean", "null"],
                "default": None,
                "description": (
                    "Whether to drop non-winning recommenders from the session "
                    "after evaluation. Required (true or false) on bundles with "
                    ">100K rows — the agent's MUST-ASK policy makes this an "
                    "explicit user choice, since the retained recommenders can "
                    "hold 1–3 GB of user embeddings on the wrong configuration. "
                    "Defaults to False on bundles ≤100K rows."
                ),
            },
            "confirmed_all": {
                "type": "boolean",
                "default": False,
                "description": (
                    "Set True only when the user has explicitly said 'all' / "
                    "'every method' / 'try every option' verbatim. Required when "
                    "passing methods='all' so the agent doesn't bypass the menu "
                    "elicitation step. Default flow: methods='list' → present "
                    "menu → user picks → re-call with methods=[short_names]."
                ),
            },
            "per_label": {
                "type": ["boolean", "null"],
                "default": None,
                "description": (
                    "Forward to each method's evaluate_model call. The MUST-ASK policy "
                    "requires this to be explicit (true or false) on multi-target "
                    "multioutput sweeps — defaulting silently to macro hides exactly "
                    "the per-target detail users typically want. Defaults to False "
                    "elsewhere. When True, classification metrics (roc_auc / pr_auc) "
                    "return per-target dicts: for MultioutputScorer they come natively "
                    "from scikit-rec; for long-format UniversalScorer the agent groups "
                    "validation predictions by ITEM_ID and runs sklearn per group. Each "
                    "leaderboard row gets a `per_label: {{metric@k: {{label: value}}}}` "
                    "field in addition to the macro `metrics`. Rejected for ranking "
                    "metrics (NDCG/MAP/recall@k aggregate across targets per user)."
                ),
            },
        },
        "required": ["bundle_id", "metrics", "primary_metric"],
    },
    fn=_sweep_methods,
)
