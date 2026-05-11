"""evaluate_model and compare_models tools."""

from __future__ import annotations

import os
from typing import Any

import numpy as np
import pandas as pd

from scikit_rec_agent.tools import Tool, err, ok

# Metric name partitions for the eval loop. These MUST stay in lockstep —
# the full ``RecommenderMetricType`` enum is partitioned into ranking
# (depend on k) and non-ranking (don't depend on k). When upstream adds
# a new metric, add it to whichever side it belongs and verify the union
# still covers every enum member. Pulled out of the eval body so both
# sites (per_label-vs-ranking pre-filter + non-K-metric dedup loop) share
# one source of truth.
_RANKING_METRIC_NAMES = frozenset(
    {
        "NDCG_at_k",
        "MAP_at_k",
        "MRR_at_k",
        "precision_at_k",
        "recall_at_k",
        "average_reward_at_k",
    }
)
_NON_K_METRIC_NAMES = frozenset({"roc_auc", "pr_auc", "rmse", "mae", "expected_reward"})
# Evaluator/metric compatibility sets. Two cases the agent rejects upfront
# so the LLM gets a localized envelope rather than the deep upstream
# traceback. Module-level for consistency with the metric partitions above
# (and to avoid re-allocating sets on every _evaluate_model call).
_SIMPLE_ONLY_METRICS = frozenset({"rmse", "mae"})
_NON_COUNTERFACTUAL_METRICS = frozenset({"roc_auc", "pr_auc"})  # simple OR replay_match
_COUNTERFACTUAL_EVALUATORS = frozenset({"IPS", "DR", "SNIPS", "direct_method", "policy_weighted"})


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

    The ``users`` frame is stripped in two cases:

    - **Embedding estimators** (matrix_factorization, two_tower, NCF, DCN,
      NFM) require a pre-computed ``EMBEDDING`` column on the users
      DataFrame to score. The trained estimator already holds those
      embeddings internally from fit — its predict() docstring supports
      "Batch Prediction Mode" where ``users=None`` means "use my internal
      embeddings". Pass ``users`` and the universal scorer errors with
      ``users DataFrame must contain 'EMBEDDING' column for embedding
      estimators.``
    - **MultioutputScorer** (classifier or regressor mode) consumes user
      features as plain columns inside the interactions frame alongside
      USER_ID + ITEM_*. A separate users DataFrame is rejected upstream
      with ``Multioutput Scorer cannot accept Users Dataframe, set it to
      None!`` (skrec/scorer/multioutput.py:582).
    """
    bundle_id = (handle.datasets_used or {}).get("bundle_id")
    bundle = session.loaded_datasets.get(bundle_id) if bundle_id else None
    if bundle is None:
        return {}
    kwargs: dict[str, pd.DataFrame] = {}
    # valid_interactions (post-split) takes precedence; for unsplit bundles
    # route via _resolve_interactions_path so auto-merged wide bundles read
    # the merged content (with features) rather than the user-provided file.
    from scikit_rec_agent.tools.datasets import _resolve_interactions_path

    inter_path = bundle.source_paths.get("valid_interactions") or _resolve_interactions_path(bundle.source_paths)
    users_path = bundle.source_paths.get("users")
    inter_df = _read(inter_path) if inter_path else pd.DataFrame()
    users_df = _read(users_path) if users_path else pd.DataFrame()
    if not inter_df.empty:
        kwargs["interactions"] = inter_df
    if not users_df.empty and not _scorer_rejects_users_frame(handle):
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


def _scorer_rejects_users_frame(handle) -> bool:
    """True if the scorer on this handle won't accept a separate users
    DataFrame at score_items time. Covers both embedding estimators (which
    expect users=None so the internal fit-time embeddings are used) and
    MultioutputScorer (which consumes user features as in-frame columns).
    """
    if _has_embedding_estimator(handle):
        return True
    try:
        from skrec.scorer.multioutput import MultioutputScorer

        return isinstance(getattr(handle.recommender, "scorer", None), MultioutputScorer)
    except Exception:
        return False


def _scorer_is_classifier_multioutput(handle) -> bool:
    """True if the handle's recommender wraps a classifier-mode
    ``MultioutputScorer``. Regressor-mode scorers and non-multioutput
    scorers return False.

    Used by the logged_rewards binary pre-check: only classifier mode
    enforces the {0, 1} contract — regressors accept continuous rewards.

    The truth is derived from the wrapped estimator's sklearn-style
    ``_estimator_type`` attribute when ``is_classifier`` isn't present
    on the scorer. The previous default-True for missing ``is_classifier``
    false-positived regressor-mode scorers on older wheels into the binary
    check, surfacing a misleading NonBinaryLoggedRewards on continuous
    targets. Defaulting via estimator type keeps the read aligned with
    the actual fit mode without depending on a single attribute that
    upstream may rename.
    """
    try:
        from skrec.scorer.multioutput import MultioutputScorer

        scorer = getattr(handle.recommender, "scorer", None)
        if not isinstance(scorer, MultioutputScorer):
            return False
        if hasattr(scorer, "is_classifier"):
            return bool(scorer.is_classifier)
        estimator = getattr(scorer, "estimator", None)
        return getattr(estimator, "_estimator_type", None) == "classifier"
    except Exception:
        return False


def _build_long_per_label_predictions(handle, bundle, session) -> dict[str, Any]:
    """Score the bundle's validation interactions and pivot into per-label
    aligned arrays for downstream classification metrics.

    Returns a dict with keys:
      ``items``: list of item names in canonical order (== bundle's scorer.item_names)
      ``y_true``: ``(n_users, n_items)`` ground-truth matrix (NaN = not observed in valid)
      ``y_score``: ``(n_users, n_items)`` predicted score matrix

    Or, on failure: ``{"__error__": <err envelope>}``.

    The universal scorer's ``score_items`` returns ``(n_users, n_items)``
    score DataFrame keyed by item names. We pivot the validation
    interactions to the same shape (users × items, value = OUTCOME)
    so the two matrices line up element-wise. Per-item classification
    metrics are then a column-wise sklearn call.
    """
    valid_path = bundle.source_paths.get("valid_interactions")
    if not valid_path or not os.path.exists(valid_path):
        return {
            "__error__": err(
                "MissingValidationInteractions",
                "per_label classification on a long-format universal scorer requires "
                "the bundle to have validation interactions registered. Re-run "
                "`split_data` so the bundle picks up a valid_interactions source.",
                category="missing_validation_interactions",
            )
        }
    valid_df = _read(valid_path)
    if valid_df.empty or not {"USER_ID", "ITEM_ID", "OUTCOME"}.issubset(valid_df.columns):
        return {
            "__error__": err(
                "InvalidValidationInteractions",
                "Validation interactions must carry USER_ID + ITEM_ID + OUTCOME for per-label "
                f"long-format scoring; got columns {sorted(valid_df.columns)}.",
                category="invalid_validation_interactions",
            )
        }

    # Score every (user × item) pair in the validation slice via the
    # scorer's score_items. UniversalScorer returns a DataFrame whose
    # columns are item names (one column per item in scorer.item_names).
    score_kwargs = _build_score_items_kwargs(session, handle) if session is not None else {}
    if not score_kwargs:
        # _build_score_items_kwargs needs the session to find the bundle;
        # the long-per-label path is invoked with session in hand, so this
        # is rare. Fall back to re-reading manually for safety.
        score_kwargs = {"interactions": valid_df.copy()}

    scores_df = handle.recommender.scorer.score_items(**score_kwargs)
    if "USER_ID" in scores_df.columns:
        scores_df = scores_df.drop(columns=["USER_ID"])
    # Ground truth: pivot valid (USER_ID, ITEM_ID, OUTCOME) into the same
    # (n_users, n_items) shape, indexed by USER_ID.
    y_true_df = valid_df.pivot_table(
        index="USER_ID",
        columns="ITEM_ID",
        values="OUTCOME",
        aggfunc="max",  # one row per (user, item) in long format; max is a no-op
    )
    # Align column order with scorer.item_names so y_true and y_score
    # line up positionally.
    scorer = handle.recommender.scorer
    item_names = [str(x) for x in getattr(scorer, "item_names", [])]
    if not item_names:
        item_names = list(scores_df.columns.astype(str))
    common_items = [c for c in item_names if c in y_true_df.columns and c in scores_df.columns]
    if not common_items:
        return {
            "__error__": err(
                "ItemAxisMisaligned",
                "Couldn't align validation ITEM_ID values with the scorer's item catalogue. "
                f"Sample valid ITEM_IDs: {sorted(y_true_df.columns)[:3]} ; "
                f"sample scorer items: {item_names[:3]}.",
                category="item_axis_misaligned",
            )
        }
    # Align by USER_ID across both frames. Some users may appear in the
    # scorer output but not the pivoted truth (or vice versa) if the
    # split / scoring diverge; intersection is the safe path.
    common_users = sorted(set(y_true_df.index) & set(scores_df.index))
    if not common_users:
        return {
            "__error__": err(
                "UserAxisMisaligned",
                "score_items produced no users that overlap with the validation interactions' "
                "USER_IDs. Likely a USER_ID dtype mismatch between scoring inputs and the "
                "validation file.",
                category="user_axis_misaligned",
            )
        }
    y_true = y_true_df.loc[common_users, common_items].to_numpy(dtype=float)
    y_score = scores_df.loc[common_users, common_items].to_numpy(dtype=float)
    return {
        "items": common_items,
        "y_true": y_true,
        "y_score": y_score,
    }


def _compute_long_per_label_metric(metric_name: str, predictions: dict[str, Any]) -> dict[str, float]:
    """Compute a classification metric per item column on aligned
    (y_true, y_score) matrices. Returns ``Dict[item_name, value]`` with
    NaN for columns where the metric is undefined (single-class y_true).
    """
    from sklearn.metrics import average_precision_score, roc_auc_score

    items = predictions["items"]
    y_true = predictions["y_true"]
    y_score = predictions["y_score"]
    out: dict[str, float] = {}
    for i, name in enumerate(items):
        yt = y_true[:, i]
        ys = y_score[:, i]
        mask = ~np.isnan(yt)
        yt_clean = yt[mask]
        ys_clean = ys[mask]
        if yt_clean.size == 0 or len(np.unique(yt_clean)) < 2:
            out[name] = float("nan")
            continue
        if metric_name == "roc_auc":
            out[name] = float(roc_auc_score(yt_clean, ys_clean))
        elif metric_name == "pr_auc":
            out[name] = float(average_precision_score(yt_clean, ys_clean))
        else:
            out[name] = float("nan")
    return out


def _scorer_is_regressor_multioutput(handle) -> bool:
    """True if the handle's recommender wraps a regressor-mode
    ``MultioutputScorer``. Counterpart to ``_scorer_is_classifier_multioutput``.
    Used to gate the ``score_items_kwargs`` always-pass policy in the metric
    loop — both modes route through ``_evaluate_multioutput`` which has no
    score cache.

    Derived from ``is_classifier`` when present, else the wrapped
    estimator's sklearn ``_estimator_type``. Returning False on ambiguity
    (rather than True) avoids triggering downstream regressor-only paths
    against a misclassified scorer.
    """
    try:
        from skrec.scorer.multioutput import MultioutputScorer

        scorer = getattr(handle.recommender, "scorer", None)
        if not isinstance(scorer, MultioutputScorer):
            return False
        if hasattr(scorer, "is_classifier"):
            return not bool(scorer.is_classifier)
        estimator = getattr(scorer, "estimator", None)
        return getattr(estimator, "_estimator_type", None) == "regressor"
    except Exception:
        return False


def _check_logged_rewards_binary(rewards) -> dict[str, Any] | None:
    """Return ``None`` if ``rewards`` is binary-valued (NaN allowed),
    otherwise a descriptor ``{n_bad, sample}`` with up to five offending
    values for surfacing in the error envelope.
    """
    arr = np.asarray(rewards, dtype=float).ravel()
    valid = arr[~np.isnan(arr)]
    if valid.size == 0:
        return None
    bad_mask = (valid != 0.0) & (valid != 1.0)
    if not bad_mask.any():
        return None
    bad_values = valid[bad_mask]
    sample = sorted({float(v) for v in bad_values.tolist()})[:5]
    return {"n_bad": int(bad_mask.sum()), "sample": sample}


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

    Three shape regimes:

    - **Per-row** ``(n_rows, 1)`` — long-format tabular and non-sequential
      embedding scorers, which score each validation row as its own
      instance. Each row's "logged interaction" is the one (item, reward)
      pair on that row.
    - **Per-user wide** ``(n_users, n_targets)`` — wide_multioutput /
      multiclass bundles where each user has one row carrying ``ITEM_*``
      target columns. logged_items repeats the target names across users;
      logged_rewards is the binary target matrix. ``MultioutputScorer``
      returns scores in the same shape.
    - **Per-user sequence** ``(n_users, L_max)`` padded — sequential and
      hierarchical recommenders, whose ``score_items`` aggregates raw
      interactions into per-user sequences and returns one score row per
      user. Aggregation order mirrors scikit-rec's
      ``SequentialRecommender._build_sequences`` (sort by USER_ID +
      TIMESTAMP, then ``groupby(sort=False)``) so the per-user rows in
      logged_items align positionally with the rows in target_proba.

    Returns ``{}`` when the bundle has no validation data or when the
    bundle's columns can't satisfy any regime above — the caller surfaces
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

    # Wide multi-output: the validation file has one row per user with
    # ITEM_<name> binary target columns (and possibly user-feature
    # columns merged in by create_datasets). MultioutputScorer scores
    # shape (n_users, n_items); align logged_items / logged_rewards to
    # that. ITEM_ID / OUTCOME aren't present here — the wide layout is
    # the contract.
    if bundle.dataset_type == "interaction_multioutput":
        item_cols = [c for c in df.columns if c.startswith("ITEM_") and c != "ITEM_ID"]
        if len(item_cols) >= 2:
            n_users = len(df)
            item_names = np.array(item_cols, dtype=object)
            items = np.tile(item_names, (n_users, 1))
            rewards = df[item_cols].to_numpy(dtype=float)
            return {"logged_items": items, "logged_rewards": rewards}

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
    per_label: bool | None = None,
) -> dict[str, Any]:
    from skrec.evaluator.datatypes import RecommenderEvaluatorType
    from skrec.metrics.datatypes import RecommenderMetricType

    handle = session.trained_models.get(model_id)
    if handle is None:
        return err("ModelNotFound", f"No trained model '{model_id}' in session.")

    # Validate evaluator_type and metrics BEFORE the MissingDecision gates
    # below. Catching an obviously-bad input here (bogus evaluator name,
    # unknown metric name) saves a round-trip: without this hoist, the
    # user answers the elicitation question only to discover their
    # evaluator was misspelled. Schema-level enum validation already
    # rejects most of these at the tool boundary, but we re-check here
    # because programmatic callers (sweep_methods, tests) bypass the
    # JSON-schema layer.
    try:
        eval_type_check = RecommenderEvaluatorType(evaluator_type)
    except ValueError:
        valid = [e.value for e in RecommenderEvaluatorType]
        return err("InvalidEvaluator", f"Unknown evaluator '{evaluator_type}'. Valid: {valid}")
    for m in metrics:
        try:
            RecommenderMetricType(m)
        except ValueError:
            valid = [x.value for x in RecommenderMetricType]
            return err("InvalidMetric", f"Unknown metric '{m}'. Valid: {valid}")
    del eval_type_check  # re-resolved below into the canonical local

    # Programmatic MUST-ASK guardrail: per_label must be explicit on
    # multioutput bundles with multiple targets, OR long-format
    # interactions bundles paired with a classification metric. Both
    # cases have a meaningful per-target axis (ITEM_* columns for
    # multioutput, distinct ITEM_ID values for long-format universal);
    # silently defaulting to macro hides exactly the detail the user
    # likely wants. The system prompt MUST-ASK rule §2 lists both —
    # this backstop covers both.
    if per_label is None:
        bundle_id_for_gate = (handle.datasets_used or {}).get("bundle_id")
        bundle_for_gate = session.loaded_datasets.get(bundle_id_for_gate) if bundle_id_for_gate else None
        _classification_metric_requested = any(m in _NON_COUNTERFACTUAL_METRICS for m in metrics)
        n_targets = 0
        gate_context: str | None = None  # 'multioutput' | 'long_classification'
        if bundle_for_gate is not None and bundle_for_gate.dataset_type == "interaction_multioutput":
            from scikit_rec_agent.tools.datasets import _resolve_interactions_path

            inter_path = _resolve_interactions_path(bundle_for_gate.source_paths)
            if inter_path:
                df_for_gate = _read(inter_path)
                n_targets = sum(1 for c in df_for_gate.columns if c.startswith("ITEM_") and c != "ITEM_ID")
            if n_targets >= 2:
                gate_context = "multioutput"
        elif (
            bundle_for_gate is not None
            and bundle_for_gate.dataset_type == "interactions"
            and _classification_metric_requested
        ):
            # Long-format with classification metric path: per_label runs
            # the long-format user-space pivot (UniversalScorer only —
            # the narrowing happens further below in the per_label-not-
            # multioutput branch). Don't read the file to count ITEM_IDs;
            # any >1-item bundle is gate-worthy and we'd rather err on
            # the side of asking. Surface a generic "multi-item" framing
            # rather than the exact count.
            gate_context = "long_classification"
        if gate_context == "multioutput":
            return err(
                "MissingDecision",
                (
                    f"Bundle has {n_targets} ITEM_* targets. The MUST-ASK policy "
                    "requires per_label to be explicit on multi-target bundles — "
                    "defaulting silently to macro-averaged hides exactly the "
                    "per-target detail users typically want. Pass per_label=True "
                    "to return Dict[str, float] keyed by ITEM_*, or per_label=False "
                    "for the single macro-averaged scalar."
                ),
                hint=(
                    "Ask the user: 'Do you want per-target metrics (one number "
                    "per ITEM_*) or a single macro-averaged scalar?'. Pass their "
                    "answer as per_label=..."
                ),
                category="missing_decision",
            )
        if gate_context == "long_classification":
            return err(
                "MissingDecision",
                (
                    "Long-format interactions bundle paired with classification metric(s) "
                    f"{sorted(m for m in metrics if m in _NON_COUNTERFACTUAL_METRICS)}. "
                    "The MUST-ASK policy requires per_label to be explicit here — "
                    "per_label=True groups validation predictions by ITEM_ID and runs the "
                    "metric per group (UniversalScorer only); per_label=False returns the "
                    "single macro-averaged scalar. Pass True or False explicitly."
                ),
                hint=(
                    "Ask the user: 'Per-item classification metrics (one number per "
                    "ITEM_ID) or a single macro-averaged scalar?'. Pass their answer "
                    "as per_label=..."
                ),
                category="missing_decision",
            )
        per_label = False

    # Evaluator/metric compatibility — two cases the agent rejects upfront
    # so the LLM gets a localized error rather than chasing a deep upstream
    # exception:
    #
    # 1. Regression metrics (rmse, mae) on any non-simple evaluator. The
    #    off-policy evaluators (IPS / DR / SNIPS / direct_method /
    #    policy_weighted) reweight rewards by treatment/logging propensities
    #    and don't have a standard composition with L2/L1 distance.
    # 2. Classification metrics (roc_auc, pr_auc) on the counterfactual
    #    evaluators (IPS / DR / SNIPS). Upstream's ROCAUC/PRAUC.calculate
    #    raises verbatim: "Counterfactual evaluators (IPS, DR, SNIPS) are
    #    not compatible with classification metrics. Use SimpleEvaluator
    #    or ReplayMatchEvaluator." Pre-gate so the user sees the agent's
    #    structured envelope, not the upstream traceback.
    simple_only_requested = {m for m in metrics if m in _SIMPLE_ONLY_METRICS}
    if simple_only_requested and evaluator_type != "simple":
        return err(
            "EvaluatorMetricMismatch",
            f"Metric(s) {sorted(simple_only_requested)} only compose with "
            "evaluator_type='simple'. The off-policy evaluators (IPS / DR / SNIPS / "
            "direct_method / policy_weighted) reweight rewards by treatment/logging "
            "propensities and don't have a standard composition with L2/L1 "
            "distance. Use evaluator_type='simple' for rmse/mae, or pick a "
            "reward-based metric (average_reward_at_k, expected_reward) for the "
            "off-policy evaluators.",
            hint="Switch evaluator_type to 'simple' or change the metric.",
            category="evaluator_metric_mismatch",
        )

    classification_requested = {m for m in metrics if m in _NON_COUNTERFACTUAL_METRICS}
    if classification_requested and evaluator_type in _COUNTERFACTUAL_EVALUATORS:
        return err(
            "EvaluatorMetricMismatch",
            f"Classification metric(s) {sorted(classification_requested)} are not "
            f"compatible with counterfactual evaluator '{evaluator_type}'. Upstream "
            "ROCAUC / PRAUC raise: 'Counterfactual evaluators (IPS, DR, SNIPS) are "
            "not compatible with classification metrics. Use SimpleEvaluator or "
            "ReplayMatchEvaluator.' Pick evaluator_type='simple' or 'replay_match' "
            "for roc_auc / pr_auc.",
            hint="Switch evaluator_type to 'simple' or 'replay_match'.",
            category="evaluator_metric_mismatch",
        )

    # per_label flag. Two supported paths:
    #
    # 1) MultioutputScorer (classifier or regressor mode) — forwarded
    #    natively to scikit-rec's ``recommender.evaluate(per_label=True)``,
    #    which returns ``Dict[str, float]``. Ranking metrics are rejected
    #    upstream (ranking aggregates across all targets per user, not
    #    per target) — we pre-filter that combination here so the error
    #    is localized to evaluate_model rather than surfacing from deep
    #    inside ``_evaluate_multioutput``.
    #
    # 2) Long-format universal scorer with a classification metric
    #    (roc_auc / pr_auc) — handled here in user-space: score the
    #    validation interactions, pivot to (USER_ID × ITEM_ID), and
    #    compute the metric per ITEM_ID. This is what your training
    #    code's per-target loop does manually for the multioutput case,
    #    adapted to the long contract.
    #
    # For other combinations (ranking metric, non-multioutput regression,
    # non-classification non-multioutput) per_label has no meaning and
    # we reject it explicitly.
    # Compute scorer mode once and reuse downstream — the per_label gate
    # and the caching policy block below both need the same answer.
    is_classifier_multioutput = _scorer_is_classifier_multioutput(handle)
    is_regressor_multioutput = _scorer_is_regressor_multioutput(handle)
    multioutput = is_classifier_multioutput or is_regressor_multioutput
    if per_label and multioutput:
        ranking_metrics_requested = [m for m in metrics if m in _RANKING_METRIC_NAMES]
        if ranking_metrics_requested:
            return err(
                "PerLabelUnsupported",
                f"per_label=True is incompatible with ranking metric(s) "
                f"{ranking_metrics_requested} on MultioutputScorer — ranking "
                "aggregates across all targets per user, not per target. "
                "Use classification (roc_auc, pr_auc) or regression (rmse, mae) "
                "metrics with per_label=True, or drop per_label for ranking.",
                hint="Pair per_label=True only with non-ranking metrics, or set per_label=False.",
                category="per_label_unsupported_for_ranking",
            )
    bundle_for_per_label = None
    if per_label and not multioutput:
        bundle_id = (handle.datasets_used or {}).get("bundle_id")
        bundle_for_per_label = session.loaded_datasets.get(bundle_id) if bundle_id else None
        # The long-format per_label path is implemented around
        # UniversalScorer's (n_users, n_items) score_items shape. Narrow
        # the gate to UniversalScorer explicitly — IndependentScorer also
        # consumes the long contract but fits per-item independent models;
        # its score_items return shape isn't guaranteed to match the
        # column-aligned pivot the per-item metric loop assumes. If/when
        # we extend support there, widen the gate; for now reject upfront
        # rather than letting a deeper shape mismatch surface.
        try:
            from skrec.scorer.universal import UniversalScorer

            scorer_is_universal = isinstance(getattr(handle.recommender, "scorer", None), UniversalScorer)
        except Exception:
            scorer_is_universal = False
        long_classification_eligible = (
            bundle_for_per_label is not None
            and bundle_for_per_label.dataset_type == "interactions"
            and scorer_is_universal
            and all(m in {"roc_auc", "pr_auc"} for m in metrics)
        )
        if not long_classification_eligible:
            return err(
                "PerLabelUnsupported",
                "per_label=True is only supported for (a) MultioutputScorer recommenders "
                "with classification or regression metrics, or (b) long-format "
                "UniversalScorer recommenders (interactions bundle, USER_ID + ITEM_ID + "
                "OUTCOME) paired with a classification metric (roc_auc / pr_auc). For "
                "other combinations the metric is already a single scalar, per-target "
                "isn't defined, or the scorer's score_items shape isn't aligned with "
                "the per-item metric loop.",
                hint=(
                    "Either set per_label=False, or pair per_label=True with "
                    "classification metrics on a long-format UniversalScorer recommender."
                ),
            )

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

    # Pre-validate logged_rewards binary contract for classifier-mode
    # MultioutputScorer. Skip silently for regressor mode (continuous targets
    # are valid) and for non-multioutput bundles. Without this, dirty floats
    # like 0.5 in the validation slice would slip past the agent's auto-build
    # cast and surface as scikit-rec's mid-evaluate error
    # ("logged_rewards to be binary numeric") halfway through the metric loop.
    rewards = effective_eval_kwargs.get("logged_rewards")
    if rewards is not None and is_classifier_multioutput:
        bad = _check_logged_rewards_binary(rewards)
        if bad is not None:
            return err(
                "NonBinaryLoggedRewards",
                (
                    "Classifier-mode MultioutputScorer evaluation requires logged_rewards "
                    "to be binary numeric — values strictly in {0, 1} (or {0.0, 1.0}, NaN "
                    f"allowed for ignore-mask). Saw {bad['n_bad']} non-binary value(s) "
                    f"(sample: {bad['sample']}). Pre-encode the validation slice before "
                    "evaluate_model — e.g. df_eval[col] = (df_eval[col] == 'yes').astype(float)."
                ),
                hint=(
                    "Fix the validation file's ITEM_* columns to be strictly {0, 1} (NaN "
                    "OK), then re-run split_data so the bundle picks up the cleaned slice."
                ),
                category="non_binary_logged_rewards",
            )

    # MultioutputScorer's `_evaluate_multioutput` path is non-cached — it
    # demands `score_items_kwargs={'interactions': df}` on every call (see
    # ``ranking_recommender.py:657-661``). For other scorers we keep the
    # one-shot caching optimisation that lets metric/k sweeps reuse the
    # underlying recommendation_scores from the first call. Detect once
    # and pick the policy.
    multioutput_eval = multioutput  # alias for readability below; same value

    # Long-format universal-scorer + per_label + classification metric path:
    # compute per-ITEM_ID metrics in user-space (scikit-rec's evaluate has
    # no native per_label support for universal scorers). We do this once
    # per call and share the cached predictions across all metric+k pairs.
    long_per_label = per_label and not multioutput_eval and bundle_for_per_label is not None
    long_predictions: dict[str, Any] | None = None
    if long_per_label:
        long_predictions = _build_long_per_label_predictions(handle, bundle_for_per_label, session)
        if isinstance(long_predictions, dict) and long_predictions.get("__error__"):
            return long_predictions["__error__"]

    # Non-K metrics use the module-level ``_NON_K_METRIC_NAMES`` partition
    # (paired with ``_RANKING_METRIC_NAMES`` at module top). Iterating
    # k_values for these would compute the same value N times and pollute
    # handle.metrics with duplicate keys like ``rmse@5`` and ``rmse@10``.
    # We collapse k_values to a single sentinel loop iteration and key
    # the metric without the ``@k`` suffix.

    results = []
    first_call = True
    for metric_name, metric_enum in resolved_metrics:
        non_k_metric = metric_name in _NON_K_METRIC_NAMES
        ks_for_metric: list[int | None] = [None] if non_k_metric else list(k_values)
        for k in ks_for_metric:
            try:
                if long_per_label:
                    value = _compute_long_per_label_metric(metric_name, long_predictions)
                else:
                    pass_score_kwargs = score_items_kwargs if (first_call or multioutput_eval) else None
                    pass_eval_kwargs = effective_eval_kwargs if (first_call or multioutput_eval) else None
                    # per_label is only forwarded when the scorer supports it
                    # (MultioutputScorer). For other scorers we don't pass the
                    # kwarg at all to avoid TypeError on overload-narrower
                    # recommender subclasses. For non-@k metrics, pass a
                    # benign sentinel (eval_top_k=1) that the non-ranking
                    # metrics (RMSE.calculate / MAE.calculate /
                    # ROCAUC.calculate / PRAUC.calculate / expected_reward)
                    # ignore at the metric layer. ``1`` instead of ``0`` to
                    # survive any future upstream guard like
                    # ``if eval_top_k <= 0: raise``.
                    eval_call_kwargs = {
                        "eval_type": eval_type,
                        "metric_type": metric_enum,
                        "eval_top_k": int(k) if k is not None else 1,
                        "score_items_kwargs": pass_score_kwargs,
                        "eval_kwargs": pass_eval_kwargs,
                    }
                    if multioutput_eval and per_label:
                        eval_call_kwargs["per_label"] = True
                    value = handle.recommender.evaluate(**eval_call_kwargs)
                    first_call = False
            except Exception as e:
                from scikit_rec_agent.tools.diagnose import _quick_diagnose

                diagnosis = _quick_diagnose(e)
                where = metric_name if non_k_metric else f"{metric_name}@{k}"
                return err(
                    type(e).__name__,
                    f"evaluate failed for {where}: {e}",
                    hint=diagnosis.first_fix_description,
                    category=diagnosis.category,
                )
            key = metric_name if non_k_metric else f"{metric_name}@{k}"
            if isinstance(value, dict):
                # Per-label dict result. Flatten into handle.metrics with
                # keys `{metric}[@k]_per_label_{label}` so compare_models stays
                # mechanical (no nested-dict rendering required), and surface
                # the raw dict on the result entry too so callers can render
                # per-target tables without re-querying the handle.
                per_label_dict = {str(label): float(v) for label, v in value.items()}
                for label, v in per_label_dict.items():
                    handle.metrics[f"{key}_per_label_{label}"] = v
                entry: dict[str, Any] = {"metric": metric_name, "per_label": per_label_dict}
                if not non_k_metric:
                    entry["k"] = int(k)
                results.append(entry)
            else:
                handle.metrics[key] = float(value)
                entry = {"metric": metric_name, "value": float(value)}
                if not non_k_metric:
                    entry["k"] = int(k)
                results.append(entry)
    if score_items_kwargs is not None and not long_per_label:
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
                        "rmse",
                        "mae",
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
            "per_label": {
                "type": ["boolean", "null"],
                "default": None,
                "description": (
                    "Whether to return per-target metric values (Dict[str, float] keyed by "
                    "ITEM_*) instead of the macro-averaged scalar. The MUST-ASK policy "
                    "requires this to be explicit (true or false) on multioutput bundles "
                    "with ≥2 targets — defaulting silently to macro hides exactly the "
                    "per-target detail users typically want. Defaults to False elsewhere. "
                    "Supported paths: (a) MultioutputScorer with classification (roc_auc, "
                    "pr_auc) or regression (rmse, mae) metrics — forwarded natively to "
                    "scikit-rec; (b) long-format UniversalScorer with roc_auc / pr_auc — "
                    "agent groups validation predictions by ITEM_ID and runs sklearn "
                    "per group. Rejected for ranking metrics (scikit-rec aggregates "
                    "ranking metrics across targets per user, so per-target NDCG/MAP "
                    "has no meaning) and for IndependentScorer (different score_items "
                    "shape contract). Result entries carry a `per_label: {label: value}` "
                    "dict; handle.metrics keys are flattened to `{metric}[@k]_per_label_"
                    "{label}` for compare_models."
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
