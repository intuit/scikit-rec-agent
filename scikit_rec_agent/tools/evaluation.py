"""evaluate_model and compare_models tools."""

from __future__ import annotations

import os
from typing import Any

import numpy as np
import pandas as pd

from scikit_rec_agent.tools import Tool, err, ok


def _read(path: str) -> pd.DataFrame:
    """Read an interactions/users/items CSV or parquet, then re-cast the
    canonical ID columns to str.

    scikit-rec treats USER_ID and ITEM_ID as categorical strings throughout
    its schemas. transform_data and create_datasets cast them in-memory,
    but a CSV roundtrip silently re-infers numeric-looking IDs (e.g. '1',
    '661') as int64 when read back here. That breaks comparison-based
    operations later in the eval path with messages like '<' not supported
    between instances of 'str' and 'int'. Forcing the dtype on read keeps
    every dataframe the agent loads in lockstep with the schema.
    """
    if not path or not os.path.exists(path):
        return pd.DataFrame()
    df = pd.read_parquet(path) if path.endswith(".parquet") else pd.read_csv(path)
    for col in ("USER_ID", "ITEM_ID"):
        if col in df.columns:
            df[col] = df[col].astype(str)
    return df


def _build_score_items_kwargs(session, handle) -> dict[str, pd.DataFrame]:
    """Load the evaluation-time interactions + users DataFrames.

    For an offline evaluation, the scorer needs to produce ranked scores for
    the same users the logged data refers to. We prefer validation
    interactions; if the bundle has none, we fall back to training
    interactions (which is less honest but keeps the tool functional).

    Embedding estimators (matrix_factorization, two_tower, NCF, DCN, NFM)
    require a pre-computed `EMBEDDING` column on the users DataFrame to
    score. The trained estimator already holds those embeddings internally
    from the fit step — its predict() docstring explicitly supports a
    "Batch Prediction Mode" where `users=None` means "use my internal
    embeddings". So when the bundle's recommender wraps an embedding
    estimator, we omit `users` entirely and let the estimator self-supply.
    Otherwise the universal scorer errors with `users DataFrame must
    contain 'EMBEDDING' column for embedding estimators.`
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
    if not users_df.empty and not _has_embedding_estimator(handle):
        kwargs["users"] = users_df
    return kwargs


def _has_embedding_estimator(handle) -> bool:
    """True if the recommender on this handle wraps a BaseEmbeddingEstimator."""
    try:
        from skrec.estimator.embedding.base_embedding_estimator import BaseEmbeddingEstimator

        estimator = getattr(getattr(handle.recommender, "scorer", None), "estimator", None)
        return isinstance(estimator, BaseEmbeddingEstimator)
    except Exception:
        return False


def _is_sequential_recommender(handle) -> bool:
    """True if the recommender's score_items returns shape (n_users, n_items),
    not (n_rows, n_items). Sequential / hierarchical recommenders aggregate
    raw interactions into per-user sequences before scoring, so logged arrays
    must match that per-user N — not the per-row N a tabular / embedding
    scorer produces."""
    try:
        from skrec.recommender.sequential.sequential_recommender import SequentialRecommender

        return isinstance(handle.recommender, SequentialRecommender)
    except Exception:
        return False


def _build_eval_kwargs_from_validation(session, handle) -> dict[str, Any]:
    """Build logged_items / logged_rewards from the bundle's validation
    interactions, shaped to match the recommender's score_items output.

    Two shape regimes:

    - **Per-row** ``(n_rows, 1)`` — for tabular and non-sequential embedding
      scorers, which score each validation row as its own instance. Each
      row's "logged interaction" is exactly the one (item, reward) pair on
      that row.
    - **Per-user** ``(n_users, L_max)`` padded — for sequential and
      hierarchical recommenders, whose ``score_items`` aggregates raw
      interactions into per-user sequences and returns one score row per
      user. Aggregation order mirrors scikit-rec's
      ``SequentialRecommender._build_sequences`` (sort by USER_ID +
      TIMESTAMP, then ``groupby(sort=False)``) so the per-user rows in
      logged_items align positionally with the rows in target_proba.

    Returns ``{}`` when the bundle has no validation data, or when the data
    isn't long-format (wide_multioutput / multiclass) — the caller surfaces
    that as a structured error.
    """
    from skrec.constants import ITEM_ID_NAME, LABEL_NAME, TIMESTAMP_COL, USER_ID_NAME

    bundle_id = (handle.datasets_used or {}).get("bundle_id")
    bundle = session.loaded_datasets.get(bundle_id) if bundle_id else None
    if bundle is None:
        return {}
    valid_path = bundle.source_paths.get("valid_interactions")
    if not valid_path:
        return {}
    df = _read(valid_path)
    if df.empty or USER_ID_NAME not in df.columns:
        return {}
    if ITEM_ID_NAME not in df.columns or LABEL_NAME not in df.columns:
        return {}

    if _is_sequential_recommender(handle):
        sort_cols = [USER_ID_NAME]
        if TIMESTAMP_COL in df.columns:
            sort_cols.append(TIMESTAMP_COL)
        df_sorted = df.sort_values(sort_cols)
        grouped = df_sorted.groupby(USER_ID_NAME, sort=False).agg({ITEM_ID_NAME: list, LABEL_NAME: list}).reset_index()
        if grouped.empty:
            return {}
        max_len = int(grouped[ITEM_ID_NAME].apply(len).max())
        items = np.array(
            [[str(x) for x in row] + [""] * (max_len - len(row)) for row in grouped[ITEM_ID_NAME]],
            dtype=object,
        )
        rewards = np.array(
            [list(row) + [0.0] * (max_len - len(row)) for row in grouped[LABEL_NAME]],
            dtype=float,
        )
        return {"logged_items": items, "logged_rewards": rewards}

    items = df[ITEM_ID_NAME].astype(str).to_numpy(dtype=object).reshape(-1, 1)
    rewards = df[LABEL_NAME].astype(float).to_numpy().reshape(-1, 1)
    return {"logged_items": items, "logged_rewards": rewards}


def _evaluate_model(
    model_id: str,
    evaluator_type: str,
    metrics: list[str],
    k_values: list[int],
    session,
    eval_kwargs: dict[str, Any] | None = None,
    refresh_scores: bool = False,
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

    # Only build + pass score_items_kwargs on the FIRST evaluate_model call
    # for this model (or when the caller explicitly asks to refresh). The
    # recommender's evaluation_session caches recommendation scores; passing
    # score_items_kwargs every time re-scores from scratch and defeats the
    # cache. split_data automatically resets the flag for affected handles
    # so the next call re-scores against fresh validation data. Pass
    # refresh_scores=True to force a re-score for other reasons.
    should_rescore = refresh_scores or not handle.score_cache_populated
    score_items_kwargs = _build_score_items_kwargs(session, handle) if should_rescore else None
    # _build_score_items_kwargs can return {} when the source bundle was
    # evicted (e.g., after load_model). Normalize to None so scikit-rec
    # uses its cache rather than re-scoring with empty inputs.
    if not score_items_kwargs:
        score_items_kwargs = None

    effective_eval_kwargs: dict[str, Any] = dict(eval_kwargs or {})
    if not effective_eval_kwargs and evaluator_type == "simple":
        effective_eval_kwargs = _build_eval_kwargs_from_validation(session, handle)
        # Wide multi-output / multi-class bundles can't be auto-built here.
        # Surface the limitation as a labelled error rather than letting the
        # downstream evaluator KeyError on missing ITEM_ID / OUTCOME columns.
        if not effective_eval_kwargs:
            bundle_id = (handle.datasets_used or {}).get("bundle_id")
            bundle = session.loaded_datasets.get(bundle_id) if bundle_id else None
            if bundle is not None and bundle.dataset_type in (
                "interaction_multioutput",
                "interaction_multiclass",
            ):
                return err(
                    "WideBundleEvalUnsupported",
                    (
                        f"evaluator_type='simple' on a {bundle.dataset_type} bundle "
                        f"can't auto-build eval_kwargs (logged_items / logged_rewards "
                        f"are derived from long-format ITEM_ID / OUTCOME columns, which "
                        f"don't exist in this shape). Pass `eval_kwargs` explicitly with "
                        f"the per-target arrays you want scored, or use a different "
                        f"evaluator_type."
                    ),
                    hint=(
                        "For wide multi-output evaluation, build logged_items / "
                        "logged_rewards arrays from the ITEM_* columns yourself and "
                        "pass them as eval_kwargs."
                    ),
                    category="wide_bundle_eval_unsupported",
                )

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
                from scikit_rec_agent.tools.diagnose import _quick_diagnose

                diagnosis = _quick_diagnose(e)
                return err(
                    type(e).__name__,
                    f"evaluate failed for {metric_name}@{k}: {e}",
                    hint=diagnosis.first_fix_description,
                    category=diagnosis.category,
                )
            key = f"{metric_name}@{k}"
            handle.metrics[key] = float(value)
            results.append({"metric": metric_name, "k": int(k), "value": float(value)})
    if score_items_kwargs is not None:
        handle.score_cache_populated = True

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
            "refresh_scores": {
                "type": "boolean",
                "default": False,
                "description": (
                    "Force re-scoring against the bundle's validation data. Pass True after "
                    "you've re-split the bundle or re-trained the model so cached scores are "
                    "invalidated. Default False keeps scikit-rec's score cache warm across calls."
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
