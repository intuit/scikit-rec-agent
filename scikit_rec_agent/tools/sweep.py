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
        {
            "short_name": "xgb_multioutput",
            "recommender_type": "ranking",
            "scorer_type": "multioutput",
            "estimator_type": "tabular",
            "estimator_config": _TABULAR_XGB,
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
    df = _read(bundle.source_paths.get("interactions"))
    if df.empty:
        return {
            "n_rows": 0,
            "sparsity": None,
            "target_type": None,
            "has_timestamps": False,
            "has_user_features": False,
            "has_item_features": False,
        }

    n_rows = int(len(df))

    sparsity: float | None = None
    if "USER_ID" in df.columns and "ITEM_ID" in df.columns:
        n_users = int(df["USER_ID"].nunique())
        n_items = int(df["ITEM_ID"].nunique())
        if n_users > 0 and n_items > 0:
            sparsity = 1.0 - (n_rows / (n_users * n_items))

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
    },
    {
        "max_rows": 100_000,
        "name": "small",
        "epochs": 5,
        "embedding_dim": 16,
        "n_factors": 16,
        "xgb_n_estimators": 100,
        "xgb_max_depth": 5,
    },
    {
        "max_rows": 1_000_000,
        "name": "medium",
        "epochs": 10,
        "embedding_dim": 32,
        "n_factors": 32,
        "xgb_n_estimators": 200,
        "xgb_max_depth": 6,
    },
    {
        "max_rows": float("inf"),
        "name": "large",
        "epochs": 20,
        "embedding_dim": 64,
        "n_factors": 64,
        "xgb_n_estimators": 300,
        "xgb_max_depth": 8,
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
    - **mf_universal** : keep if sparsity > 0.95 and n_rows >= 1000. MF
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

    if target_type == "continuous":
        # For now only XGBoost has a clean regression swap; embedding /
        # sequential families would need outcome_type adjustments per class.
        if "xgb" not in short:
            return False, f"target is continuous; {short} doesn't auto-swap to regression"

    if "mf" in short and "_universal" in short:
        sparsity = profile.get("sparsity")
        if sparsity is None or sparsity < 0.95:
            return False, f"MF needs sparsity ≥ 0.95 (CF regime); got {sparsity}"
        if n_rows < 1_000:
            return False, f"MF needs n_rows ≥ 1000 to factorize stably; got {n_rows}"
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


def _detect_bundle_contract(bundle) -> str:
    df = _read(bundle.source_paths.get("interactions"))
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
    model_id: str,
    metrics: list[str],
    eval_top_k: int,
    evaluator_type: str,
    session,
) -> dict[str, Any]:
    from scikit_rec_agent.tools.evaluation import TOOL_EVALUATE_MODEL
    from scikit_rec_agent.tools.training import TOOL_TRAIN_MODEL

    config = _build_train_args(method)
    train_result = TOOL_TRAIN_MODEL.fn(
        model_name=model_id,
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

    # Re-key the handle under the sweep's deterministic model_id so that a
    # later sweep with skip_existing=True can find it. train_model itself
    # registers under an auto-generated `<recommender>_<timestamp>` id;
    # without this re-key the cache hit branch never fires.
    trained_model_id = train_result["data"]["model_id"]
    handle = session.trained_models.pop(trained_model_id)
    handle.model_id = model_id
    handle.name = model_id
    session.trained_models[model_id] = handle

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
            "metrics": dict(handle.metrics),
            "model_id": model_id,
        }

    metric_map = {f"{name}@{k}": handle.metrics.get(f"{name}@{k}") for name, k in metric_pairs}
    return {
        "status": "ok",
        "metrics": metric_map,
        "model_id": model_id,
    }


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
    drop_non_winners: bool = False,
) -> dict[str, Any]:
    bundle = session.loaded_datasets.get(bundle_id)
    if bundle is None:
        return err("BundleNotFound", f"No bundle '{bundle_id}' in session.")

    contract = _detect_bundle_contract(bundle)

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
        return ok(
            {
                "bundle_id": bundle_id,
                "contract": contract,
                "available_methods": menu,
                "n_available": len(menu),
                "n_dropped_incompatible": len(dropped),
                "dropped_methods": dropped,
                "instructions": (
                    "Surface this numbered list to the user with a brief "
                    "description of each method (tabular vs embedding family, "
                    "key hyperparameters), and ask which option(s) to run. "
                    "Then call sweep_methods again with `methods=[...]` "
                    "containing the chosen `short_name` strings, or "
                    "`methods='all'` to run every option."
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
            "methods must be 'auto', 'all', 'broad', 'list', or an explicit list of method dicts / short_names.",
        )

    runnable, dropped = _filter_by_capability(method_list)

    primary_key = _resolve_primary_metric_key(primary_metric, eval_top_k)
    leaderboard: list[dict[str, Any]] = []

    for method in runnable:
        config_hash = _config_hash(bundle_id, method)
        short = method.get("short_name") or _short_name(method)
        model_id = f"{name_prefix}_{short}_{config_hash[:8]}"

        if skip_existing and model_id in session.trained_models:
            handle = session.trained_models[model_id]
            # If a previous sweep trained this model but its evaluation failed
            # (e.g. bad metric name on the prior call), the cached handle has
            # an empty metrics dict. Re-evaluate it instead of returning a
            # leaderboard row with no numbers — the user expects "cached"
            # to mean "trained AND scored", not "trained but unscored".
            if handle.metrics:
                leaderboard.append(
                    {
                        "method": method,
                        "model_id": model_id,
                        "status": "cached",
                        "metrics": dict(handle.metrics),
                    }
                )
                continue
            eval_only = _evaluate_existing(
                model_id=model_id,
                metrics=metrics,
                eval_top_k=eval_top_k,
                evaluator_type=evaluator_type,
                session=session,
            )
            if eval_only["status"] == "ok":
                leaderboard.append(
                    {
                        "method": method,
                        "model_id": model_id,
                        "status": "cached",
                        "metrics": eval_only["metrics"],
                    }
                )
            else:
                leaderboard.append(
                    {
                        "method": method,
                        "model_id": model_id,
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
            model_id=model_id,
            metrics=metrics,
            eval_top_k=eval_top_k,
            evaluator_type=evaluator_type,
            session=session,
        )
        if result["status"] == "ok":
            leaderboard.append(
                {
                    "method": method,
                    "model_id": model_id,
                    "status": "ok",
                    "metrics": result["metrics"],
                }
            )
        else:
            leaderboard.append(
                {
                    "method": method,
                    "model_id": model_id,
                    "status": "error",
                    "error": result.get("error"),
                    "category": result.get("category"),
                    "metrics": result.get("metrics", {}),
                }
            )

    def sort_key(row):
        v = row.get("metrics", {}).get(primary_key)
        return -float(v) if v is not None else float("inf")

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
            "drop_non_winners": {"type": "boolean", "default": False},
        },
        "required": ["bundle_id", "metrics", "primary_metric"],
    },
    fn=_sweep_methods,
)
