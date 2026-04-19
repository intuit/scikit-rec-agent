"""evaluate_model and compare_models tools."""

from __future__ import annotations

import os
from typing import Any

import numpy as np
import pandas as pd

from scikit_rec_agent.tools import Tool, err, ok


def _read(path: str) -> pd.DataFrame:
    if not path or not os.path.exists(path):
        return pd.DataFrame()
    return pd.read_parquet(path) if path.endswith(".parquet") else pd.read_csv(path)


def _build_score_items_kwargs(session, handle) -> dict[str, pd.DataFrame]:
    """Load the evaluation-time interactions + users DataFrames.

    For an offline evaluation, the scorer needs to produce ranked scores for
    the same users the logged data refers to. We prefer validation
    interactions; if the bundle has none, we fall back to training
    interactions (which is less honest but keeps the tool functional).
    """
    bundle_id = (handle.datasets_used or {}).get("bundle_id")
    bundle = session.loaded_datasets.get(bundle_id) if bundle_id else None
    if bundle is None:
        return {}
    kwargs: dict[str, pd.DataFrame] = {}
    inter_path = bundle.source_paths.get("valid_interactions") or bundle.source_paths.get("interactions")
    users_path = bundle.source_paths.get("users")
    inter_df = _read(inter_path) if inter_path else pd.DataFrame()
    users_df = _read(users_path) if users_path else pd.DataFrame()
    if not inter_df.empty:
        kwargs["interactions"] = inter_df
    if not users_df.empty:
        kwargs["users"] = users_df
    return kwargs


def _build_eval_kwargs_from_validation(
    session, handle, user_col: str = "USER_ID", item_col: str = "ITEM_ID", outcome_col: str = "OUTCOME"
) -> dict[str, Any]:
    """For the 'simple' evaluator on implicit-feedback data, derive logged_items
    / logged_rewards from the bundle's validation interactions. Returns {} if
    validation data isn't available — the evaluator will raise with a clear
    message in that case.
    """
    bundle_id = (handle.datasets_used or {}).get("bundle_id")
    bundle = session.loaded_datasets.get(bundle_id) if bundle_id else None
    if bundle is None:
        return {}
    valid_path = bundle.source_paths.get("valid_interactions")
    if not valid_path:
        return {}
    df = _read(valid_path)
    if df.empty or user_col not in df.columns:
        return {}
    grouped = df.groupby(user_col).agg({item_col: list, outcome_col: list}).reset_index()
    if grouped.empty:
        return {}
    max_len = int(grouped[item_col].apply(len).max())
    items = np.array(
        [row + [""] * (max_len - len(row)) for row in grouped[item_col]],
        dtype=object,
    )
    rewards = np.array(
        [row + [0.0] * (max_len - len(row)) for row in grouped[outcome_col]],
        dtype=float,
    )
    return {"logged_items": items, "logged_rewards": rewards}


def _evaluate_model(
    model_id: str,
    evaluator_type: str,
    metrics: list[str],
    k_values: list[int],
    session,
    eval_kwargs: dict[str, Any] | None = None,
) -> dict[str, Any]:
    from skrec.evaluator.datatypes import RecommenderEvaluatorType
    from skrec.metrics.datatypes import RecommenderMetricType

    handle = session.trained_models.get(model_id)
    if handle is None:
        return err("ModelNotFound", f"No trained model '{model_id}' in session.")

    try:
        eval_type = RecommenderEvaluatorType(evaluator_type)
    except ValueError:
        valid = [e.value for e in RecommenderEvaluatorType]
        return err("InvalidEvaluator", f"Unknown evaluator '{evaluator_type}'. Valid: {valid}")

    resolved_metrics = []
    for m in metrics:
        try:
            resolved_metrics.append((m, RecommenderMetricType(m)))
        except ValueError:
            valid = [x.value for x in RecommenderMetricType]
            return err("InvalidMetric", f"Unknown metric '{m}'. Valid: {valid}")

    # On the first call (nothing cached), we must provide score_items_kwargs.
    # Subsequent calls with changed metrics/k reuse cached scores.
    score_items_kwargs = _build_score_items_kwargs(session, handle) or None

    effective_eval_kwargs: dict[str, Any] = dict(eval_kwargs or {})
    if not effective_eval_kwargs and evaluator_type == "simple":
        effective_eval_kwargs = _build_eval_kwargs_from_validation(session, handle)

    results = []
    first_call = True
    for metric_name, metric_enum in resolved_metrics:
        for k in k_values:
            try:
                value = handle.recommender.evaluate(
                    eval_type=eval_type,
                    metric_type=metric_enum,
                    eval_top_k=int(k),
                    score_items_kwargs=score_items_kwargs if first_call else None,
                    eval_kwargs=effective_eval_kwargs if first_call else None,
                )
                first_call = False
            except Exception as e:
                return err(
                    type(e).__name__,
                    f"evaluate failed for {metric_name}@{k}: {e}",
                )
            key = f"{metric_name}@{k}"
            handle.metrics[key] = float(value)
            results.append({"metric": metric_name, "k": int(k), "value": float(value)})

    return ok({"model_id": model_id, "evaluator_type": evaluator_type, "results": results})


TOOL_EVALUATE_MODEL = Tool(
    name="evaluate_model",
    description=(
        "Evaluate a trained model using offline evaluation. Supports all 7 evaluator types "
        "and all 9 metrics at multiple k values. For the 'simple' evaluator, "
        "logged_items / logged_rewards are auto-derived from the bundle's validation "
        "interactions if eval_kwargs is not provided. Results cached on the recommender's "
        "evaluation_session and also accumulated onto the model's handle for later "
        "comparison."
    ),
    input_schema={
        "type": "object",
        "properties": {
            "model_id": {"type": "string"},
            "evaluator_type": {
                "type": "string",
                "enum": ["simple", "replay_match", "IPS", "DR", "direct_method", "SNIPS", "policy_weighted"],
            },
            "metrics": {
                "type": "array",
                "items": {
                    "type": "string",
                    "enum": [
                        "NDCG_at_k",
                        "MAP_at_k",
                        "MRR_at_k",
                        "precision_at_k",
                        "recall_at_k",
                        "average_reward_at_k",
                        "roc_auc",
                        "pr_auc",
                        "expected_reward",
                    ],
                },
            },
            "k_values": {"type": "array", "items": {"type": "integer"}},
            "eval_kwargs": {
                "type": "object",
                "description": (
                    "Evaluator-specific kwargs: logged_items, logged_rewards, logging_proba, "
                    "expected_rewards. Auto-derived from validation interactions when omitted "
                    "for 'simple'."
                ),
            },
        },
        "required": ["model_id", "evaluator_type", "metrics", "k_values"],
    },
    fn=_evaluate_model,
)


# ---------------------------------------------------------------------------
# compare_models
# ---------------------------------------------------------------------------


def _compare_models(
    primary_metric: str,
    k: int,
    session,
    model_ids: list[str] | None = None,
) -> dict[str, Any]:
    if model_ids:
        handles = [session.trained_models.get(mid) for mid in model_ids]
        if any(h is None for h in handles):
            missing = [mid for mid, h in zip(model_ids, handles) if h is None]
            return err("ModelNotFound", f"model_ids not in session: {missing}")
    else:
        handles = list(session.trained_models.values())

    if not handles:
        return err("NoModels", "Session has no trained models to compare.")

    primary_key = f"{primary_metric}@{k}"

    def sort_key(h):
        return -float(h.metrics.get(primary_key, float("-inf")))

    sorted_handles = sorted(handles, key=sort_key)

    all_metric_keys: set[str] = set()
    for h in sorted_handles:
        all_metric_keys.update(h.metrics.keys())
    metric_cols = sorted(all_metric_keys)

    header = ["model_name", "model_id", "recommender_type", "training_time_s", *metric_cols]
    rows = []
    for h in sorted_handles:
        row = [
            h.name,
            h.model_id,
            str(h.config.get("recommender_type", "")),
            f"{h.training_time_seconds:.2f}",
        ]
        for mk in metric_cols:
            v = h.metrics.get(mk)
            row.append(f"{v:.4f}" if v is not None else "—")
        rows.append(row)

    md_lines = ["| " + " | ".join(header) + " |"]
    md_lines.append("|" + "|".join(["---"] * len(header)) + "|")
    for r in rows:
        md_lines.append("| " + " | ".join(r) + " |")
    markdown = "\n".join(md_lines)

    json_rows = []
    for h in sorted_handles:
        json_rows.append(
            {
                "model_name": h.name,
                "model_id": h.model_id,
                "recommender_type": h.config.get("recommender_type"),
                "training_time_seconds": h.training_time_seconds,
                "metrics": dict(h.metrics),
            }
        )

    return ok(
        {
            "primary_metric": primary_metric,
            "k": k,
            "markdown": markdown,
            "rows": json_rows,
        }
    )


TOOL_COMPARE_MODELS = Tool(
    name="compare_models",
    description=(
        "Compare trained models in the current session. Returns a markdown leaderboard "
        "sorted by the primary metric (highest first)."
    ),
    input_schema={
        "type": "object",
        "properties": {
            "model_ids": {
                "type": "array",
                "items": {"type": "string"},
                "description": "If empty or omitted, compares all trained models in the session.",
            },
            "primary_metric": {"type": "string"},
            "k": {"type": "integer"},
        },
        "required": ["primary_metric", "k"],
    },
    fn=_compare_models,
)
