"""list_compatible_options tool — drives the hierarchical model-design flow.

Walks the user through a top-down picker: recommender_type → scorer_type →
estimator_type → model_type → hyperparameters. At each step the tool
returns only the options compatible with the user's prior picks AND the
data, with a written explanation triple per option.

The tool is a single endpoint (progressive disclosure): pass `bundle_id`
and whatever's been picked so far in `current_choices`, get the next
dimension's menu back. When all four discrete dimensions are picked the
tool returns the terminal hyperparameter payload, with three next-action
options (train_with_defaults / train_with_overrides / run_hpo) and an
`assembled_config` dict ready to feed `train_model`.

Reuses the capability matrix and data-aware infrastructure already in
`tools/sweep.py` (`_RECOMMENDER_SCORER_COMPAT`, `_SCORER_ESTIMATOR_COMPAT`,
`_profile_bundle`, `_resize_for_data_scale`).
"""

from __future__ import annotations

import copy
from typing import Any

from scikit_rec_agent.prompts._explanations import (
    ESTIMATOR_TYPE_EXPLANATIONS,
    RECOMMENDER_EXPLANATIONS,
    SCORER_EXPLANATIONS,
    UPLIFT_PARAM_EXPLANATIONS,
    model_explanations_for,
)
from scikit_rec_agent.tools import Tool, err, ok

# The four discrete dimensions, in walk order. The fifth step
# (hyperparameters) is the terminal payload.
_DIMENSIONS_IN_ORDER = ("recommender_type", "scorer_type", "estimator_type", "model_type")


# ---------------------------------------------------------------------------
# Per-step option builders
# ---------------------------------------------------------------------------


def _options_for_recommender_type(profile: dict[str, Any]) -> tuple[list[str], list[str]]:
    """Drop recommender_types whose data prerequisites aren't met. Returns
    (kept_values, dropped_values_with_reason_pairs_unflat).

    `gcsl` is deliberately dropped at this stage even though scikit-rec
    supports it. The hierarchical flow can't yet collect gcsl's required
    `inference_method` config (mean_scalarization / percentile_value /
    predefined_value), so picking it would lead to a factory error at
    train time. Leaving it out of the menu keeps the surface honest.
    """
    from scikit_rec_agent.tools.sweep import _RECOMMENDER_SCORER_COMPAT

    all_values = list(_RECOMMENDER_SCORER_COMPAT.keys())
    kept, dropped_pairs = [], []
    for v in all_values:
        if v == "gcsl":
            dropped_pairs.append(
                (v, "the agent doesn't yet capture gcsl's inference_method config — train-time would fail")
            )
            continue
        if v in ("sequential", "hierarchical_sequential") and not profile.get("has_timestamps"):
            dropped_pairs.append((v, "no TIMESTAMP column in interactions"))
            continue
        if v == "hierarchical_sequential" and not profile.get("has_session_boundaries"):
            dropped_pairs.append((v, "no SESSION_SEQUENCES column"))
            continue
        kept.append(v)
    return kept, [f"{v}: {reason}" for v, reason in dropped_pairs]


def _options_for_scorer_type(recommender_type: str, profile: dict[str, Any]) -> tuple[list[str], list[str]]:
    from scikit_rec_agent.tools.sweep import _RECOMMENDER_SCORER_COMPAT

    if recommender_type not in _RECOMMENDER_SCORER_COMPAT:
        return [], [f"unknown recommender_type '{recommender_type}'"]

    candidates = sorted(_RECOMMENDER_SCORER_COMPAT[recommender_type])
    kept, dropped_pairs = [], []
    for scorer in candidates:
        # multioutput needs ≥2 ITEM_* targets (wide multi-output shape).
        if scorer == "multioutput" and not profile.get("has_wide_targets"):
            dropped_pairs.append((scorer, "data has no ≥2 ITEM_* target columns"))
            continue
        # multiclass needs ITEM_ID without OUTCOME (multiclass shape).
        if scorer == "multiclass" and profile.get("target_type") not in (None, "categorical"):
            dropped_pairs.append((scorer, "data has an OUTCOME column; multiclass forbids OUTCOME"))
            continue
        kept.append(scorer)
    return kept, [f"{v}: {reason}" for v, reason in dropped_pairs]


def _options_for_estimator_type(scorer_type: str, profile: dict[str, Any]) -> tuple[list[str], list[str]]:
    from scikit_rec_agent.tools.sweep import _SCORER_ESTIMATOR_COMPAT

    if scorer_type not in _SCORER_ESTIMATOR_COMPAT:
        return [], [f"unknown scorer_type '{scorer_type}'"]

    candidates = sorted(_SCORER_ESTIMATOR_COMPAT[scorer_type])
    kept, dropped_pairs = [], []
    n_rows = profile.get("n_rows", 0) or 0
    for est in candidates:
        if est == "embedding" and n_rows < 5_000:
            dropped_pairs.append((est, f"embedding needs n_rows ≥ 5000; got {n_rows}"))
            continue
        kept.append(est)
    return kept, [f"{v}: {reason}" for v, reason in dropped_pairs]


def _options_for_model_type(estimator_type: str, profile: dict[str, Any]) -> tuple[list[str], list[str]]:
    if estimator_type == "tabular":
        from skrec.orchestrator.factory import capability_matrix as _cm
        cm = _cm()
        all_tabular = list(cm.get("tabular_model_types", ("xgboost",)))
        kept, dropped_pairs = [], []
        for m in all_tabular:
            if m == "deepfm":
                if profile.get("target_type") == "continuous":
                    dropped_pairs.append((m, "deepfm only supports classification; data has continuous targets"))
                    continue
                try:
                    import torch  # noqa: F401
                except ImportError:
                    dropped_pairs.append((m, "deepfm requires scikit-rec[torch] — torch not installed"))
                    continue
            kept.append(m)
        return kept, [f"{v}: {reason}" for v, reason in dropped_pairs]
    if estimator_type == "embedding":
        # All five embedding model_types are reachable via the factory; data-size
        # gate already applied at the estimator_type step. Two-Tower needs features.
        candidates = list(model_explanations_for("embedding").keys())
        kept, dropped_pairs = [], []
        for m in candidates:
            if m == "two_tower" and not (profile.get("has_user_features") or profile.get("has_item_features")):
                dropped_pairs.append((m, "two_tower needs user OR item features"))
                continue
            kept.append(m)
        return kept, [f"{v}: {reason}" for v, reason in dropped_pairs]
    if estimator_type == "sequential":
        # Pick classifier vs regressor based on target_type when known.
        candidates = list(model_explanations_for("sequential").keys())
        target_type = profile.get("target_type")
        kept = []
        for m in candidates:
            if target_type == "binary" and "regressor" in m:
                continue
            if target_type == "continuous" and "classifier" in m:
                continue
            kept.append(m)
        return kept, []
    return [], [f"unknown estimator_type '{estimator_type}'"]


# ---------------------------------------------------------------------------
# Assembly: from current_choices → a complete RecommenderConfig dict
# ---------------------------------------------------------------------------


def _assemble_method(choices: dict[str, str], profile: dict[str, Any]) -> dict[str, Any]:
    """Build a sweep-style method dict from the four discrete picks; the
    sweep helpers (`_resize_for_data_scale`) then size the hyperparameters
    to the data tier."""
    estimator_type = choices["estimator_type"]
    model_type = choices["model_type"]

    method: dict[str, Any] = {
        "short_name": f"{model_type}_{choices['scorer_type']}",
        "recommender_type": choices["recommender_type"],
        "scorer_type": choices["scorer_type"],
        "estimator_type": estimator_type,
    }

    if estimator_type == "tabular":
        ml_task = "regression" if profile.get("target_type") == "continuous" else "classification"
        if model_type == "lightgbm":
            model_key = "lightgbm"
            model_defaults: dict[str, Any] = {"n_estimators": 50, "max_depth": -1, "learning_rate": 0.1, "num_leaves": 31}
        elif model_type == "deepfm":
            model_key = "deepfm"
            model_defaults = {"embedding_dim": 16, "hidden_dim1": 64, "hidden_dim2": 32, "epochs": 5, "batch_size": 256, "lr": 0.001}
        else:
            model_key = "xgboost"
            model_defaults = {"n_estimators": 50, "max_depth": 4, "learning_rate": 0.1}
        method["estimator_config"] = {
            "ml_task": ml_task,
            model_key: model_defaults,
        }
    elif estimator_type == "embedding":
        method["estimator_config"] = {
            "estimator_type": "embedding",
            "embedding": {"model_type": model_type, "params": _default_embedding_params(model_type)},
        }
    elif estimator_type == "sequential":
        method["estimator_config"] = {
            "estimator_type": "sequential",
            "sequential": {
                "model_type": model_type,
                "params": {"hidden_units": 16, "max_len": 10, "epochs": 2},
            },
        }
        method["recommender_params"] = {"max_len": 10}

    # Uplift recommenders need recommender_params filled in (control_item_id +
    # mode). We pre-seed both as None so apply_overrides finds them in the
    # right bucket when the user supplies values; the terminal payload's
    # `required_recommender_params` field tells the agent what to ask the
    # user for. Leaving them as None means train_model will refuse upfront
    # rather than crash deep inside scikit-rec's factory.
    if choices["recommender_type"] == "uplift":
        method.setdefault("recommender_params", {})
        method["recommender_params"].setdefault("control_item_id", None)
        method["recommender_params"].setdefault("mode", None)

    return method


def _default_embedding_params(model_type: str) -> dict[str, Any]:
    """Minimum-viable defaults per embedding family; sized later by
    _resize_for_data_scale for the actual data tier."""
    base = {"epochs": 3, "batch_size": 32, "random_state": 42}
    if model_type == "matrix_factorization":
        return {"n_factors": 16, "epochs": 5, "random_state": 42}
    if model_type == "ncf":
        return {
            **base,
            "ncf_type": "neumf",
            "gmf_embedding_dim": 8,
            "mlp_embedding_dim": 8,
            "mlp_layers": [16, 8],
            "dropout": 0.1,
            "learning_rate": 0.01,
        }
    if model_type == "two_tower":
        return {
            **base,
            "user_embedding_dim": 16,
            "item_embedding_dim": 16,
            "final_embedding_dim": 8,
            "user_tower_hidden_dim1": 32,
            "item_tower_hidden_dim1": 32,
        }
    if model_type == "deep_cross_network":
        return {**base, "embedding_dim": 16, "num_cross_layers": 2, "deep_hidden_dim1": 32}
    if model_type == "neural_factorization":
        return {**base, "embedding_dim": 16, "mlp_hidden_dim1": 32}
    return {}


def _strip_to_recommender_config(method: dict[str, Any]) -> dict[str, Any]:
    """Drop the sweep-internal fields (short_name, estimator_type top-level)
    so what remains is a pure RecommenderConfig that train_model accepts."""
    config = copy.deepcopy(method)
    config.pop("short_name", None)
    config.pop("estimator_type", None)
    return config


def suggest_search_space(method: dict[str, Any], profile: dict[str, Any]) -> dict[str, Any]:
    """Return a hyperparameter search space suitable for ``run_hpo`` given
    the picked method and the bundle's data profile.

    Each entry is keyed by the bare hyperparameter name and shaped like:

    ``{"type": "int" | "float" | "categorical", "low": ..., "high": ...,
       "log_scale": bool, "rationale": str}``

    Ranges are heuristic defaults — sensible across typical recsys workloads,
    not tuned for any specific dataset. Centered around the values produced
    by ``_resize_for_data_scale`` so a search around them is the natural
    refinement of "train with defaults". Pass the result straight into
    ``run_hpo``'s ``search_space`` argument.

    Per-family coverage: tabular (XGBoost, LightGBM), embedding (MF / NCF /
    Two-Tower / DCN / NFM), sequential (SASRec / HRNN). DeepFM and estimator
    families NOT yet routed through scikit-rec's factory return ``{}`` (caller
    should treat that as "no automatic search space — write your own").
    """
    estimator_config = method.get("estimator_config", {})
    space: dict[str, Any] = {}

    if "xgboost" in estimator_config:
        space.update(
            {
                "n_estimators": {
                    "type": "int",
                    "low": 30,
                    "high": 500,
                    "rationale": "more trees help on larger data; diminishing returns past ~300",
                },
                "max_depth": {
                    "type": "int",
                    "low": 3,
                    "high": 10,
                    "rationale": "depth 3-6 is typical; >10 overfits without regularisation",
                },
                "learning_rate": {
                    "type": "float",
                    "low": 0.01,
                    "high": 0.3,
                    "log_scale": True,
                    "rationale": "log-uniform LR; lower LR pairs with more trees",
                },
            }
        )
        return space

    if "lightgbm" in estimator_config:
        space.update(
            {
                "n_estimators": {
                    "type": "int",
                    "low": 50,
                    "high": 1000,
                    "rationale": "LightGBM trains shallower trees quickly; more trees are affordable without proportional slowdown",
                },
                "num_leaves": {
                    "type": "int",
                    "low": 15,
                    "high": 255,
                    "rationale": "controls model complexity; keep < 2^max_depth to avoid overfitting",
                },
                "learning_rate": {
                    "type": "float",
                    "low": 0.01,
                    "high": 0.3,
                    "log_scale": True,
                    "rationale": "log-uniform LR; lower LR pairs with more trees",
                },
                "min_child_samples": {
                    "type": "int",
                    "low": 5,
                    "high": 100,
                    "rationale": "minimum leaf population; higher values regularise on small data",
                },
            }
        )
        return space

    embedding = estimator_config.get("embedding", {})
    model_type = embedding.get("model_type")
    if model_type == "matrix_factorization":
        space.update(
            {
                "n_factors": {
                    "type": "int",
                    "low": 8,
                    "high": 128,
                    "rationale": "powers of 2 typical; 8-32 for small data, up to 128 for very sparse large catalogues",
                },
                "regularization": {
                    "type": "float",
                    "low": 0.001,
                    "high": 0.1,
                    "log_scale": True,
                    "rationale": "log-uniform regulariser; higher for noisier data",
                },
                "epochs": {
                    "type": "int",
                    "low": 5,
                    "high": 50,
                    "rationale": "ALS converges fast; 5-20 is typical",
                },
            }
        )
        return space
    if model_type == "ncf":
        space.update(
            {
                "gmf_embedding_dim": {
                    "type": "int",
                    "low": 8,
                    "high": 64,
                    "rationale": "GMF branch embedding size",
                },
                "mlp_embedding_dim": {
                    "type": "int",
                    "low": 8,
                    "high": 64,
                    "rationale": "MLP branch embedding size",
                },
                "dropout": {
                    "type": "float",
                    "low": 0.0,
                    "high": 0.5,
                    "rationale": "dropout between MLP layers; tune to fight overfitting",
                },
                "learning_rate": {
                    "type": "float",
                    "low": 1e-4,
                    "high": 1e-2,
                    "log_scale": True,
                    "rationale": "Adam LR for torch models",
                },
                "epochs": {
                    "type": "int",
                    "low": 3,
                    "high": 30,
                    "rationale": "early stopping handles upper bound",
                },
            }
        )
        return space
    if model_type == "two_tower":
        space.update(
            {
                "user_embedding_dim": {
                    "type": "int",
                    "low": 8,
                    "high": 128,
                    "rationale": "user-tower output dimensionality",
                },
                "item_embedding_dim": {
                    "type": "int",
                    "low": 8,
                    "high": 128,
                    "rationale": "item-tower output dimensionality",
                },
                "final_embedding_dim": {
                    "type": "int",
                    "low": 4,
                    "high": 64,
                    "rationale": "combined dot-product space; usually smaller than tower outputs",
                },
                "learning_rate": {
                    "type": "float",
                    "low": 1e-4,
                    "high": 1e-2,
                    "log_scale": True,
                    "rationale": "Adam LR for torch models",
                },
                "epochs": {
                    "type": "int",
                    "low": 3,
                    "high": 30,
                    "rationale": "early stopping handles upper bound",
                },
            }
        )
        return space
    if model_type in ("deep_cross_network", "neural_factorization"):
        space.update(
            {
                "embedding_dim": {
                    "type": "int",
                    "low": 8,
                    "high": 128,
                    "rationale": "feature embedding size; scale with feature count",
                },
                "learning_rate": {
                    "type": "float",
                    "low": 1e-4,
                    "high": 1e-2,
                    "log_scale": True,
                    "rationale": "Adam LR for torch models",
                },
                "epochs": {
                    "type": "int",
                    "low": 3,
                    "high": 30,
                    "rationale": "early stopping handles upper bound",
                },
            }
        )
        return space

    sequential = estimator_config.get("sequential", {})
    seq_model_type = sequential.get("model_type")
    if seq_model_type and seq_model_type.startswith(("sasrec", "hrnn")):
        space.update(
            {
                "hidden_units": {
                    "type": "int",
                    "low": 16,
                    "high": 256,
                    "rationale": "transformer/RNN hidden size; powers of 2 typical",
                },
                "max_len": {
                    "type": "int",
                    "low": 10,
                    "high": 200,
                    "rationale": "max history length per user; bound by per-user history p95",
                },
                "learning_rate": {
                    "type": "float",
                    "low": 1e-4,
                    "high": 1e-2,
                    "log_scale": True,
                    "rationale": "Adam LR for torch models",
                },
                "epochs": {
                    "type": "int",
                    "low": 2,
                    "high": 20,
                    "rationale": "sequential models converge in tens of epochs",
                },
            }
        )
        return space

    return {}


def apply_overrides(assembled_config: dict[str, Any], overrides: dict[str, Any]) -> dict[str, Any]:
    """Merge user-supplied hyperparameter overrides onto the terminal step's
    `assembled_config` dict. Returns a deep copy — original is not mutated.

    Each override key is the bare hyperparameter name (e.g. ``n_estimators``,
    ``hidden_units``, ``learning_rate``). The function walks the known
    locations a hyperparameter can live in inside a RecommenderConfig:

    - ``estimator_config.{xgboost|lightgbm|deepfm}.<name>``   (tabular family)
    - ``estimator_config.embedding.params.<name>``             (embedding family)
    - ``estimator_config.sequential.params.<name>``            (sequential family)
    - ``recommender_params.<name>``                            (e.g. max_len)

    The first location that already contains the key wins; if the key isn't
    found anywhere, it lands in the most likely params bucket for the
    estimator type so the user's intent isn't dropped silently. Used by the
    `train_with_overrides` action of the hierarchical-flow terminal step.
    """
    new_config = copy.deepcopy(assembled_config)
    estimator_config = new_config.setdefault("estimator_config", {})
    recommender_params = new_config.setdefault("recommender_params", {})

    tabular_params = next(
        (estimator_config[k] for k in ("xgboost", "lightgbm", "deepfm") if k in estimator_config),
        None,
    )
    embedding_params = estimator_config.get("embedding", {}).get("params")
    sequential_params = estimator_config.get("sequential", {}).get("params")

    fallback_bucket: dict[str, Any] | None
    if tabular_params is not None:
        fallback_bucket = tabular_params
    elif embedding_params is not None:
        fallback_bucket = embedding_params
    elif sequential_params is not None:
        fallback_bucket = sequential_params
    else:
        fallback_bucket = recommender_params

    # Some hyperparameters are mirrored across buckets and must stay in sync.
    # `max_len` for sequential recommenders is the canonical case: it lives in
    # both `estimator_config.sequential.params.max_len` (controls how the
    # estimator pads / trims sequences) AND `recommender_params.max_len` (the
    # SequentialRecommender uses it during data preparation to truncate user
    # histories). If only one bucket gets updated the train/recommend
    # pipeline silently uses two different lengths. The mirror_keys set
    # below enumerates the param names that must update every bucket they
    # appear in, not just the first.
    mirror_keys = {"max_len"}

    for key, value in overrides.items():
        if key in mirror_keys:
            updated_any = False
            for bucket in (tabular_params, embedding_params, sequential_params, recommender_params):
                if bucket is not None and key in bucket:
                    bucket[key] = value
                    updated_any = True
            if not updated_any:
                fallback_bucket[key] = value
            continue

        for bucket in (tabular_params, embedding_params, sequential_params, recommender_params):
            if bucket is not None and key in bucket:
                bucket[key] = value
                break
        else:
            # Never seen — drop into the fallback bucket for the estimator type.
            fallback_bucket[key] = value

    return new_config


# ---------------------------------------------------------------------------
# Tool entrypoint
# ---------------------------------------------------------------------------


def _list_compatible_options(
    bundle_id: str,
    session,
    current_choices: dict[str, str] | None = None,
) -> dict[str, Any]:
    from scikit_rec_agent.tools.sweep import _profile_bundle, _resize_for_data_scale

    bundle = session.loaded_datasets.get(bundle_id)
    if bundle is None:
        return err("BundleNotFound", f"No bundle '{bundle_id}' in session.")

    choices = dict(current_choices or {})

    profile = _profile_bundle(bundle)
    # Augment the profile with the wide-targets signal that
    # _options_for_scorer_type needs (sweep's profile doesn't track it
    # directly; we infer from contract detection).
    from scikit_rec_agent.tools.sweep import _detect_bundle_contract

    contract = _detect_bundle_contract(bundle)
    profile["has_wide_targets"] = contract == "wide_multioutput"
    profile["has_session_boundaries"] = contract == "sessions"

    # Walk dimensions in order; first one with no choice picked is the next
    # dimension to surface.
    next_dim = next((d for d in _DIMENSIONS_IN_ORDER if d not in choices), None)

    if next_dim is None:
        # All four discrete dimensions picked — return the terminal payload.
        return _terminal_payload(choices, profile, _resize_for_data_scale)

    # Build the option set for this dimension.
    if next_dim == "recommender_type":
        values, dropped = _options_for_recommender_type(profile)
        explanations_map = RECOMMENDER_EXPLANATIONS
    elif next_dim == "scorer_type":
        values, dropped = _options_for_scorer_type(choices["recommender_type"], profile)
        explanations_map = SCORER_EXPLANATIONS
    elif next_dim == "estimator_type":
        values, dropped = _options_for_estimator_type(choices["scorer_type"], profile)
        explanations_map = ESTIMATOR_TYPE_EXPLANATIONS
    elif next_dim == "model_type":
        values, dropped = _options_for_model_type(choices["estimator_type"], profile)
        explanations_map = model_explanations_for(choices["estimator_type"])
    else:  # unreachable
        return err("UnknownDimension", f"Unexpected dimension '{next_dim}'.")

    options = []
    for v in values:
        triple = explanations_map.get(v, {})
        options.append(
            {
                "value": v,
                "what_it_is": triple.get("what_it_is", ""),
                "when_to_pick": triple.get("when_to_pick", ""),
                "tradeoff_vs_alternatives": triple.get("tradeoff_vs_alternatives", ""),
            }
        )

    payload: dict[str, Any] = {
        "next_dimension": next_dim,
        "options": options,
        "data_signals_used": _public_signals(profile),
        "current_choices": dict(choices),
        "is_terminal": False,
    }

    if not options:
        # Dead end. Tell the agent why and suggest backing up one step.
        prev_dim = (
            _DIMENSIONS_IN_ORDER[_DIMENSIONS_IN_ORDER.index(next_dim) - 1] if next_dim != "recommender_type" else None
        )
        payload["why_no_options"] = (
            f"No compatible {next_dim} options for the current picks "
            f"({choices}) on this data. Dropped: {dropped}. "
            + (f"Back up to {prev_dim} and pick differently." if prev_dim else "")
        )
        if prev_dim:
            payload["back_to_step"] = prev_dim

    return ok(payload)


def _terminal_payload(choices: dict[str, str], profile: dict[str, Any], resize_fn) -> dict[str, Any]:
    """Build the hyperparameter step payload + assembled_config + next_action_options."""
    method = _assemble_method(choices, profile)
    sized_method = resize_fn(method, profile)
    config = _strip_to_recommender_config(sized_method)
    config.pop("short_name", None)

    estimator_type = choices["estimator_type"]
    if estimator_type == "tabular":
        ec = config["estimator_config"]
        tabular_key = next((k for k in ("xgboost", "lightgbm", "deepfm") if k in ec), None)
        params = ec[tabular_key] if tabular_key else {}
    elif estimator_type == "embedding":
        params = config["estimator_config"]["embedding"]["params"]
    elif estimator_type == "sequential":
        params = config["estimator_config"]["sequential"]["params"]
    else:
        params = {}

    tier_name = _tier_name_for(profile.get("n_rows", 0) or 0)
    default_params = {
        name: {
            "value": value,
            "what_it_is": _annotate_param(estimator_type, choices.get("model_type"), name),
            "why_this_default": f"data tier '{tier_name}'",
        }
        for name, value in params.items()
    }

    suggested_search_space = suggest_search_space(sized_method, profile)

    # Recommender-type-specific extras: uplift needs control_item_id and mode
    # before train_model will accept the config. Surface them with the same
    # explanation shape as a normal dimension menu so the agent presents
    # them uniformly. Other recommender_types (ranking / sequential / bandits)
    # don't need user-supplied recommender_params today.
    required_recommender_params: dict[str, Any] = {}
    if choices["recommender_type"] == "uplift":
        required_recommender_params = copy.deepcopy(UPLIFT_PARAM_EXPLANATIONS)

    train_cost = _estimated_cost("train", choices, profile)
    hpo_cost = _estimated_cost("hpo", choices, profile)

    return ok(
        {
            "next_dimension": "hyperparameters",
            "is_terminal": True,
            "current_choices": dict(choices),
            "data_signals_used": _public_signals(profile),
            "default_params": default_params,
            "suggested_search_space": suggested_search_space,
            "required_recommender_params": required_recommender_params,
            "next_action_options": [
                {
                    "action": "train_with_defaults",
                    "description": (
                        "Train once with the default hyperparameters above. Fastest path "
                        "to a baseline. Calls train_model with the assembled_config below."
                    ),
                    # Uplift recommender_params don't have safe defaults — the user MUST
                    # supply control_item_id (and pick a mode). Force them through
                    # train_with_overrides instead.
                    "available": not required_recommender_params,
                    "estimated_cost": train_cost,
                    **(
                        {
                            "blocked_reason": (
                                "Uplift requires user-supplied recommender_params "
                                f"({list(required_recommender_params.keys())}); "
                                "use train_with_overrides instead."
                            )
                        }
                        if required_recommender_params
                        else {}
                    ),
                },
                {
                    "action": "train_with_overrides",
                    "description": (
                        "Override specific hyperparameter values, then train once. The "
                        "agent should call apply_overrides(assembled_config, "
                        "{param_name: new_value, ...}) to merge user picks into the right "
                        "place in the config, then call train_model on the result. For "
                        "uplift, this is the only train path — supply control_item_id "
                        "and mode in the overrides dict."
                    ),
                    "available": True,
                    "estimated_cost": train_cost,
                    "helper": "scikit_rec_agent.tools.design.apply_overrides",
                },
                {
                    "action": "run_hpo",
                    "description": (
                        "Search the hyperparameter ranges in suggested_search_space via "
                        "Optuna and pick the best. Pass suggested_search_space as run_hpo's "
                        "search_space argument; the bundle, base config, and metric are "
                        "all already known. For uplift, supply control_item_id and mode "
                        "in the base config's recommender_params before running HPO."
                    ),
                    "available": bool(suggested_search_space),
                    "estimated_cost": hpo_cost,
                    # search_space_hint shows the agent the search space ready to feed
                    # run_hpo — same shape, different name to avoid confusion with the
                    # top-level `suggested_search_space` field that the agent surfaces
                    # to the user. Both fields point at the same dict by reference.
                    "search_space_hint": suggested_search_space,
                },
            ],
            "assembled_config": config,
        }
    )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _estimated_cost(action: str, choices: dict[str, str], profile: dict[str, Any]) -> str:
    """Rough wall-clock estimate for the given terminal action.

    Heuristic, not measured — based on data tier, estimator family, and
    how many models the action trains. Surfaced as `estimated_cost` on
    each `next_action_options` entry so the agent can present the cost
    tradeoff to the user. Strings rather than numbers because the
    granularity is "minutes vs hours", not seconds.
    """
    n_rows = profile.get("n_rows", 0) or 0
    estimator_type = choices.get("estimator_type", "tabular")

    # Baseline single-train cost by tier × estimator family.
    if estimator_type == "tabular":
        if n_rows < 5_000:
            train = "<10s"
        elif n_rows < 100_000:
            train = "10-60s"
        elif n_rows < 1_000_000:
            train = "1-5 min"
        else:
            train = "5-30 min"
    elif estimator_type == "embedding":
        if n_rows < 5_000:
            train = "10-30s"
        elif n_rows < 100_000:
            train = "30s-3 min"
        elif n_rows < 1_000_000:
            train = "3-15 min"
        else:
            train = "15-60 min"
    elif estimator_type == "sequential":
        if n_rows < 5_000:
            train = "30s-2 min"
        elif n_rows < 100_000:
            train = "2-10 min"
        elif n_rows < 1_000_000:
            train = "10-60 min"
        else:
            train = "1-4 hr"
    else:
        train = "unknown"

    if action == "train":
        return train
    if action == "hpo":
        # Optuna default ~20-50 trials. Cost scales linearly with trials and
        # with the per-train cost; stating a 20× multiplier is sensible.
        return f"~20× the train cost ({train} per trial × ~20 trials)"
    return "unknown"


def _public_signals(profile: dict[str, Any]) -> dict[str, Any]:
    """Strip the few internal keys from the profile dict before exposing
    it to the LLM."""
    return {
        "n_rows": profile.get("n_rows"),
        "sparsity": profile.get("sparsity"),
        "target_type": profile.get("target_type"),
        "has_timestamps": profile.get("has_timestamps"),
        "has_user_features": profile.get("has_user_features"),
        "has_item_features": profile.get("has_item_features"),
        "has_wide_targets": profile.get("has_wide_targets", False),
        "has_session_boundaries": profile.get("has_session_boundaries", False),
    }


def _tier_name_for(n_rows: int) -> str:
    if n_rows < 5_000:
        return "tiny (<5K rows)"
    if n_rows < 100_000:
        return "small (5K–100K rows)"
    if n_rows < 1_000_000:
        return "medium (100K–1M rows)"
    return "large (≥1M rows)"


def _annotate_param(estimator_type: str, model_type: str | None, param_name: str) -> str:
    """Short descriptions for the most common hyperparameters. Empty string
    for unknown params is fine — the LLM still has the value."""
    common = {
        "epochs": "Training passes over the data",
        "batch_size": "Rows per gradient step",
        "learning_rate": "Optimiser step size",
        "random_state": "RNG seed for reproducibility",
        "n_estimators": "Number of boosted trees",
        "max_depth": "Maximum tree depth",
        "n_factors": "Latent factor dimensionality",
        "embedding_dim": "Embedding vector size",
        "gmf_embedding_dim": "GMF branch embedding size",
        "mlp_embedding_dim": "MLP branch embedding size",
        "user_embedding_dim": "User tower output size",
        "item_embedding_dim": "Item tower output size",
        "final_embedding_dim": "Combined embedding size",
        "user_tower_hidden_dim1": "User tower hidden layer width",
        "item_tower_hidden_dim1": "Item tower hidden layer width",
        "deep_hidden_dim1": "Deep branch hidden layer width",
        "mlp_hidden_dim1": "MLP hidden layer width",
        "num_cross_layers": "Number of explicit cross layers",
        "mlp_layers": "MLP layer widths (list)",
        "dropout": "Dropout probability",
        "ncf_type": "NCF variant: gmf / mlp / neumf",
        "hidden_units": "Transformer hidden size",
        "max_len": "Max sequence length per user",
    }
    return common.get(param_name, "")


TOOL_LIST_COMPATIBLE_OPTIONS = Tool(
    name="list_compatible_options",
    description=(
        "Drives the hierarchical model-design flow for a single recommender. Walks the user "
        "through the design space dimension by dimension (recommender_type → scorer_type → "
        "estimator_type → model_type → hyperparameters), filtering options by the data and "
        "the user's prior picks. Each option carries a written explanation triple "
        "(what_it_is / when_to_pick / tradeoff_vs_alternatives) — surface those to the user "
        "so they can pick informed. The terminal step returns an assembled_config dict that "
        "plugs straight into train_model. Use this when the user wants ONE good model, not "
        "a sweep comparison."
    ),
    input_schema={
        "type": "object",
        "properties": {
            "bundle_id": {"type": "string"},
            "current_choices": {
                "type": "object",
                "description": (
                    "What the user has already picked. Keys are dimension names "
                    "(recommender_type / scorer_type / estimator_type / model_type), values "
                    "are the chosen string. Pass {} on the first call to get the recommender_type menu."
                ),
            },
        },
        "required": ["bundle_id"],
    },
    fn=_list_compatible_options,
)
