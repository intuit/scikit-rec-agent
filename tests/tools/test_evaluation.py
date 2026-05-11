"""Tests for evaluate_model and compare_models tools."""

from __future__ import annotations

from typing import Any

import pytest

from scikit_rec_agent.tools.datasets import TOOL_CREATE_DATASETS
from scikit_rec_agent.tools.evaluation import TOOL_COMPARE_MODELS, TOOL_EVALUATE_MODEL
from scikit_rec_agent.tools.splitting import TOOL_SPLIT_DATA
from scikit_rec_agent.tools.training import TOOL_TRAIN_MODEL

try:
    import torch  # type: ignore[import-unresolved]  # noqa: F401

    _torch_available = True
except ImportError:
    _torch_available = False

requires_torch = pytest.mark.skipif(not _torch_available, reason="PyTorch not installed")

_TABULAR_CONFIG = {
    "recommender_type": "ranking",
    "scorer_type": "universal",
    "estimator_config": {
        "ml_task": "classification",
        "xgboost": {"n_estimators": 20, "max_depth": 3},
    },
}


@pytest.fixture
def trained_model(binary_reward_paths, session):
    TOOL_CREATE_DATASETS.fn(
        bundle_id="b",
        interactions_path=binary_reward_paths["interactions"],
        users_path=binary_reward_paths["users"],
        items_path=binary_reward_paths["items"],
        session=session,
    )
    TOOL_SPLIT_DATA.fn(
        bundle_id="b",
        strategy="random_split",
        valid_fraction=0.2,
        session=session,
        random_state=1,
    )
    result = TOOL_TRAIN_MODEL.fn(model_name="m", config=_TABULAR_CONFIG, bundle_id="b", session=session)
    assert result["status"] == "ok"
    return result["data"]["model_id"]


def test_evaluate_records_metrics(trained_model, session):
    result = TOOL_EVALUATE_MODEL.fn(
        model_id=trained_model,
        evaluator_type="simple",
        metrics=["NDCG_at_k", "precision_at_k"],
        k_values=[5, 10],
        session=session,
    )
    assert result["status"] == "ok"
    data = result["data"]
    assert len(data["results"]) == 4  # 2 metrics × 2 k
    handle = session.trained_models[trained_model]
    assert "NDCG_at_k@5" in handle.metrics
    assert "NDCG_at_k@10" in handle.metrics
    assert "precision_at_k@5" in handle.metrics


def test_evaluate_unknown_metric_returns_error(trained_model, session):
    result = TOOL_EVALUATE_MODEL.fn(
        model_id=trained_model,
        evaluator_type="simple",
        metrics=["bogus_metric"],
        k_values=[10],
        session=session,
    )
    assert result["status"] == "error"
    assert result["error_type"] == "InvalidMetric"


def test_compare_models_renders_markdown(trained_model, session):
    # Evaluate at least once so metrics exist
    TOOL_EVALUATE_MODEL.fn(
        model_id=trained_model,
        evaluator_type="simple",
        metrics=["NDCG_at_k"],
        k_values=[10],
        session=session,
    )
    result = TOOL_COMPARE_MODELS.fn(primary_metric="NDCG_at_k", k=10, session=session)
    assert result["status"] == "ok"
    md = result["data"]["markdown"]
    assert "model_name" in md
    # Must actually render the trained model_id into the leaderboard, not
    # just any substring that happens to appear in the table header.
    assert trained_model in md
    assert len(result["data"]["rows"]) == 1
    assert result["data"]["rows"][0]["model_name"] == "m"
    assert result["data"]["rows"][0]["model_id"] == trained_model


def test_compare_with_no_models_errors(session):
    result = TOOL_COMPARE_MODELS.fn(primary_metric="NDCG_at_k", k=10, session=session)
    assert result["status"] == "error"
    assert result["error_type"] == "NoModels"


def test_eval_kwargs_per_user_shape_for_sequential_recommender(tmp_path, session):
    """Sequential recommenders aggregate per-user before scoring, so
    logged_items must be shape (n_users, L_max) — not (n_rows, 1) — to
    match target_proba's first dim. Regression: when SASRec was added to
    the auto-sweep, evaluation crashed with logged_items_shape_mismatch
    because the helper produced per-row arrays. Verified here with a fake
    sequential handle so the test runs in milliseconds without torch."""
    import pandas as pd

    from scikit_rec_agent.session import ModelHandle
    from scikit_rec_agent.tools.datasets import TOOL_CREATE_DATASETS
    from scikit_rec_agent.tools.evaluation import _build_eval_kwargs_from_validation
    from scikit_rec_agent.tools.splitting import TOOL_SPLIT_DATA

    # Build a small long-with-timestamp bundle so _build_eval_kwargs_from_validation
    # has real validation data to aggregate over.
    df = pd.DataFrame(
        {
            "USER_ID": ["u1", "u1", "u1", "u2", "u2"],
            "ITEM_ID": ["i_a", "i_b", "i_c", "i_a", "i_d"],
            "OUTCOME": [1.0, 1.0, 0.0, 1.0, 1.0],
            "TIMESTAMP": ["1", "2", "3", "1", "2"],
        }
    )
    p = tmp_path / "long_ts.csv"
    df.to_csv(p, index=False)
    TOOL_CREATE_DATASETS.fn(bundle_id="b", interactions_path=str(p), session=session)
    TOOL_SPLIT_DATA.fn(
        bundle_id="b",
        strategy="random_split",
        valid_fraction=0.4,
        session=session,
        random_state=1,
    )

    # Stand up a fake recommender that subclasses SequentialRecommender so
    # _is_sequential_recommender returns True without needing torch.
    from skrec.recommender.sequential.sequential_recommender import SequentialRecommender

    class _FakeSeqRec(SequentialRecommender):
        def __init__(self):
            pass  # skip the real __init__ (would need a scorer)

    fake_handle = ModelHandle(
        model_id="seq",
        name="seq",
        config={},
        recommender=_FakeSeqRec(),
        datasets_used={"bundle_id": "b"},
    )
    kwargs = _build_eval_kwargs_from_validation(session, fake_handle)
    assert "logged_items" in kwargs
    items = kwargs["logged_items"]
    rewards = kwargs["logged_rewards"]

    # Per-user shape: rows = unique users in validation, cols = max items per user
    assert items.ndim == 2
    assert items.shape[0] <= df["USER_ID"].nunique()  # ≤ 2 unique users in validation
    # And NOT the per-row shape (which would be 2 if both validation rows survive,
    # but with 1-col width — distinct from per-user which has > 1 col when a user
    # has multiple validation rows).
    # Stronger: shape rows count matches validation user count
    valid_path = session.loaded_datasets["b"].source_paths.get("valid_interactions")
    valid_df = pd.read_csv(valid_path)
    assert items.shape[0] == valid_df["USER_ID"].nunique()
    assert items.shape == rewards.shape


@requires_torch
def test_evaluate_real_sequential_recommender_end_to_end(tmp_path, session):
    """Companion to the FakeSeqRec shape test above — this one trains a real
    SASRec via the agent's train_model + evaluate_model path on a small
    long_with_timestamp bundle. Verifies the per-user logged_items shape
    actually aligns with what scikit-rec's SequentialScorer expects, not
    just what our helper guesses. Gated on torch (skipif) since SASRec is
    a torch estimator; runs in the test-torch CI job alongside the other
    torch-using tests.
    """
    import pandas as pd

    # Tiny long-with-timestamp bundle: enough rows for SASRec to train
    # without crashing but small enough to finish in seconds.
    n_users = 20
    rows_per_user = 8
    df_rows = []
    for u in range(n_users):
        for r in range(rows_per_user):
            df_rows.append(
                {
                    "USER_ID": f"u{u}",
                    "ITEM_ID": f"i{(u + r) % 10}",
                    "OUTCOME": float((u + r) % 2),
                    "TIMESTAMP": str(u * 100 + r),
                }
            )
    p = tmp_path / "long_ts.csv"
    pd.DataFrame(df_rows).to_csv(p, index=False)

    TOOL_CREATE_DATASETS.fn(bundle_id="seq_b", interactions_path=str(p), session=session)
    TOOL_SPLIT_DATA.fn(
        bundle_id="seq_b",
        strategy="random_split",
        valid_fraction=0.25,
        session=session,
        random_state=1,
    )

    sasrec_config = {
        "recommender_type": "sequential",
        "scorer_type": "sequential",
        "estimator_config": {
            "estimator_type": "sequential",
            "sequential": {
                "model_type": "sasrec_classifier",
                "params": {"hidden_units": 8, "max_len": 5, "epochs": 1, "random_state": 42},
            },
        },
        "recommender_params": {"max_len": 5},
    }
    train_result = TOOL_TRAIN_MODEL.fn(
        model_name="real_sasrec",
        config=sasrec_config,
        bundle_id="seq_b",
        session=session,
    )
    assert train_result["status"] == "ok", train_result
    model_id = train_result["data"]["model_id"]

    # The load-bearing assertion: evaluate_model with the simple evaluator
    # must NOT raise the "Mismatch in N dimension" shape error. The auto-
    # built per-user logged_items / logged_rewards shape must align with
    # SequentialScorer's per-user target_proba.
    eval_result = TOOL_EVALUATE_MODEL.fn(
        model_id=model_id,
        evaluator_type="simple",
        metrics=["NDCG_at_k"],
        k_values=[5],
        session=session,
    )
    assert eval_result["status"] == "ok", eval_result
    handle = session.trained_models[model_id]
    assert "NDCG_at_k@5" in handle.metrics, (
        f"Expected NDCG_at_k@5 to land on the handle after a successful eval; got metrics={dict(handle.metrics)}"
    )


def test_evaluate_model_enriches_errors_with_category(trained_model, session):
    """Mirror of the train_model enrichment: when evaluate_model raises, the
    envelope must carry a `category` from the diagnose registry so the LLM
    has a structured handle. Forces a dtype_mismatch by replacing the
    recommender's evaluate with one that raises the canonical str/int
    comparison error."""
    handle = session.trained_models[trained_model]

    def _raise_dtype_mismatch(**kwargs):
        raise TypeError("'<' not supported between instances of 'str' and 'int'")

    handle.recommender.evaluate = _raise_dtype_mismatch
    result = TOOL_EVALUATE_MODEL.fn(
        model_id=trained_model,
        evaluator_type="simple",
        metrics=["NDCG_at_k"],
        k_values=[5],
        session=session,
    )
    assert result["status"] == "error"
    assert result["category"] == "dtype_mismatch_across_files"
    assert result.get("hint")  # diagnose registry's first-fix description


def test_score_items_kwargs_omits_users_for_embedding_estimator(binary_reward_paths, session):
    """Embedding estimators self-supply user embeddings during prediction
    (Batch Mode in BaseEmbeddingEstimator.predict_proba). Our eval helper
    must NOT pass `users` for them — the universal scorer otherwise rejects
    the call with `users DataFrame must contain 'EMBEDDING' column`."""
    from scikit_rec_agent.session import ModelHandle
    from scikit_rec_agent.tools.evaluation import _build_score_items_kwargs

    TOOL_CREATE_DATASETS.fn(
        bundle_id="b",
        interactions_path=binary_reward_paths["interactions"],
        users_path=binary_reward_paths["users"],
        items_path=binary_reward_paths["items"],
        session=session,
    )

    # Fake a handle whose recommender wraps a BaseEmbeddingEstimator instance.
    from skrec.estimator.embedding.matrix_factorization_estimator import MatrixFactorizationEstimator

    class _FakeScorer:
        estimator = MatrixFactorizationEstimator(n_factors=4)

    class _FakeRecommender:
        scorer = _FakeScorer()

    handle = ModelHandle(
        model_id="emb",
        name="emb",
        config={},
        recommender=_FakeRecommender(),
        datasets_used={"bundle_id": "b"},
    )
    session.trained_models["emb"] = handle

    kwargs = _build_score_items_kwargs(session, handle)
    assert "interactions" in kwargs
    assert "users" not in kwargs, (
        "users DataFrame must NOT be passed for embedding estimators — they self-supply embeddings in batch mode."
    )


def test_score_items_kwargs_includes_users_for_tabular_estimator(trained_model, session):
    """Sanity: tabular estimators DO need the users DataFrame, so the helper
    must include it for them. Regression guard against the embedding-estimator
    branch over-firing."""
    from scikit_rec_agent.tools.evaluation import _build_score_items_kwargs

    handle = session.trained_models[trained_model]
    kwargs = _build_score_items_kwargs(session, handle)
    # The trained_model fixture uses XGBoost (tabular) with users + items
    assert "users" in kwargs


def test_evaluate_wide_multiclass_bundle_returns_structured_error(tmp_path, session):
    """Wide multi-class bundle + simple evaluator + no explicit eval_kwargs
    must return the structured WideBundleEvalUnsupported error.

    NB: ``interaction_multioutput`` no longer hits this branch — the agent's
    ``_build_eval_kwargs_from_validation`` now auto-builds logged_items /
    logged_rewards for the wide_multioutput contract (see the matching
    branch in evaluation.py). Multi-class bundles (USER_ID + ITEM_ID, no
    OUTCOME, no ITEM_*) still fall through to this gate.
    """
    import pandas as pd

    # Wide multi-class shape: USER_ID + ITEM_ID, no OUTCOME, no ITEM_* cols.
    df = pd.DataFrame(
        {
            "USER_ID": [f"u{i}" for i in range(20)],
            "ITEM_ID": [f"class_{i % 3}" for i in range(20)],
            "feat1": list(range(20)),
        }
    )
    p = tmp_path / "multiclass.csv"
    df.to_csv(p, index=False)
    TOOL_CREATE_DATASETS.fn(bundle_id="mc", interactions_path=str(p), session=session)
    TOOL_SPLIT_DATA.fn(
        bundle_id="mc",
        strategy="leave_n_users_out",
        n_valid_users=4,
        session=session,
        random_state=1,
    )

    # Stick a fake handle on the session so evaluate_model has something to look up.
    from scikit_rec_agent.session import ModelHandle

    handle = ModelHandle(
        model_id="m_mc",
        name="m_mc",
        config={"recommender_type": "ranking", "scorer_type": "multiclass"},
        recommender=None,
        datasets_used={"bundle_id": "mc"},
    )
    session.trained_models["m_mc"] = handle

    result = TOOL_EVALUATE_MODEL.fn(
        model_id="m_mc",
        evaluator_type="simple",
        metrics=["NDCG_at_k"],
        k_values=[5],
        session=session,
    )
    assert result["status"] == "error"
    assert result["error_type"] == "WideBundleEvalUnsupported"
    assert result["category"] == "wide_bundle_eval_unsupported"
    assert "interaction_multiclass" in result["message"]


def test_evaluate_wide_multioutput_now_auto_builds_eval_kwargs(tmp_path, session):
    """Regression: ``_build_eval_kwargs_from_validation`` now produces
    logged_items / logged_rewards from the wide-multioutput validation slice,
    so evaluate_model gets past the unsupported gate. With a stub recommender
    it crashes inside the evaluator instead — the failure mode we care about
    here is "auto-build did fire", not "evaluation succeeded with no model"."""
    import pandas as pd

    df = pd.DataFrame(
        {
            "USER_ID": [f"u{i}" for i in range(20)],
            "ITEM_a": [1.0, 0.0] * 10,
            "ITEM_b": [0.0, 1.0] * 10,
            "feat1": list(range(20)),
        }
    )
    p = tmp_path / "wide.csv"
    df.to_csv(p, index=False)
    TOOL_CREATE_DATASETS.fn(bundle_id="wide", interactions_path=str(p), session=session)
    TOOL_SPLIT_DATA.fn(
        bundle_id="wide",
        strategy="leave_n_users_out",
        n_valid_users=4,
        session=session,
        random_state=1,
    )

    from scikit_rec_agent.session import ModelHandle
    from scikit_rec_agent.tools.evaluation import _build_eval_kwargs_from_validation

    handle = ModelHandle(
        model_id="m_wide",
        name="m_wide",
        config={"recommender_type": "ranking", "scorer_type": "multioutput"},
        recommender=None,
        datasets_used={"bundle_id": "wide"},
    )
    session.trained_models["m_wide"] = handle

    kwargs = _build_eval_kwargs_from_validation(session, handle)
    assert "logged_items" in kwargs
    assert "logged_rewards" in kwargs
    # Two ITEM_* columns × 4 valid users.
    assert kwargs["logged_rewards"].shape == (4, 2)
    assert kwargs["logged_items"].shape == (4, 2)


def test_check_logged_rewards_binary_accepts_binary_and_nan():
    """the binary pre-check tolerates NaN (treated as ignore-mask)
    but flags any non-{0, 1} non-NaN value with a sample. Direct unit cover
    so the helper's contract doesn't drift if callers change."""
    import numpy as np

    from scikit_rec_agent.tools.evaluation import _check_logged_rewards_binary

    assert _check_logged_rewards_binary(np.array([0.0, 1.0, 0.0, 1.0])) is None
    assert _check_logged_rewards_binary(np.array([0.0, np.nan, 1.0])) is None
    assert _check_logged_rewards_binary(np.array([np.nan, np.nan])) is None
    bad = _check_logged_rewards_binary(np.array([0.0, 0.5, 1.0, 0.3]))
    assert bad is not None
    assert bad["n_bad"] == 2
    assert set(bad["sample"]).issubset({0.3, 0.5})


def test_evaluate_per_label_missing_decision_on_multi_target_bundle(tmp_path, session):
    """per_label=None on a wide_multioutput bundle with ≥2
    ITEM_* targets returns MissingDecision so the agent asks the user
    whether to surface per-target or macro. Build a minimal wide bundle
    and stub the recommender so the gate fires before training.
    """
    import pandas as pd

    from scikit_rec_agent.session import ModelHandle
    from scikit_rec_agent.tools.datasets import TOOL_CREATE_DATASETS

    df = pd.DataFrame(
        {
            "USER_ID": [f"u{i}" for i in range(10)],
            "ITEM_a": [i % 2 for i in range(10)],
            "ITEM_b": [(i + 1) % 2 for i in range(10)],
            "ITEM_c": [(i + 2) % 2 for i in range(10)],
        }
    )
    p = tmp_path / "wide.csv"
    df.to_csv(p, index=False)
    TOOL_CREATE_DATASETS.fn(bundle_id="w", interactions_path=str(p), session=session)

    handle = ModelHandle(
        model_id="m_w",
        name="m_w",
        config={"recommender_type": "ranking", "scorer_type": "multioutput"},
        recommender=None,  # gate fires before recommender is touched
        datasets_used={"bundle_id": "w"},
    )
    session.trained_models["m_w"] = handle

    result = TOOL_EVALUATE_MODEL.fn(
        model_id="m_w",
        evaluator_type="simple",
        metrics=["roc_auc"],
        k_values=[10],
        session=session,
        # per_label deliberately omitted → defaults to None → must MissingDecision
    )
    assert result["status"] == "error"
    assert result["error_type"] == "MissingDecision"
    assert result["category"] == "missing_decision"
    assert "per_label" in result["message"]


def test_evaluate_per_label_missing_decision_on_long_format_classification(trained_model, session):
    """the per_label MissingDecision gate fires on long-format
    interactions bundles paired with a classification metric (roc_auc /
    pr_auc) — not just on wide_multioutput. The system prompt MUST-ASK
    rule §2 lists both cases; this test pins the backstop's symmetry
    with the rule. Without this gate, a sweep / evaluate on a long bundle
    with roc_auc and per_label=None would silently default to macro.
    """
    result = TOOL_EVALUATE_MODEL.fn(
        model_id=trained_model,  # long-format universal bundle (XGBoost ranking)
        evaluator_type="simple",
        metrics=["roc_auc"],
        k_values=[10],
        # per_label deliberately omitted → defaults to None → must MissingDecision
        session=session,
    )
    assert result["status"] == "error"
    assert result["error_type"] == "MissingDecision"
    assert result["category"] == "missing_decision"
    assert "per_label" in result["message"]
    assert "Long-format" in result["message"]


def test_evaluate_per_label_default_safe_on_non_multioutput(trained_model, session):
    """the guardrail must NOT fire on non-multioutput bundles. Default
    None → treated as False; the macro-averaged scalar is the right
    default for a long-format universal scorer with NDCG. Verifies the
    gate is data-conditional, not blanket.
    """
    result = TOOL_EVALUATE_MODEL.fn(
        model_id=trained_model,
        evaluator_type="simple",
        metrics=["NDCG_at_k"],
        k_values=[5],
        session=session,
        # per_label omitted → None → falls back to False, no MissingDecision
    )
    assert result["status"] == "ok"


def test_evaluate_per_label_ranking_metric_on_multioutput_rejected(tmp_path, session):
    """ranking metric + per_label=True + MultioutputScorer must return
    a localized PerLabelUnsupported envelope, not bubble up upstream's deep
    error. Stub a multioutput scorer so the gate fires without running a
    real fit.
    """
    from sklearn.linear_model import LogisticRegression
    from sklearn.multioutput import MultiOutputClassifier
    from skrec.scorer.multioutput import MultioutputScorer

    from scikit_rec_agent.session import ModelHandle

    class _StubMultioutputScorer(MultioutputScorer):
        def __init__(self):
            # Skip the real __init__ — we only need isinstance() to be True
            # and ``is_classifier`` to be set.
            self.is_classifier = True
            self.estimator = MultiOutputClassifier(LogisticRegression())

    class _StubRecommender:
        scorer = _StubMultioutputScorer()

        def evaluate(self, **kwargs):
            raise AssertionError("evaluate should not have been called — pre-filter must reject first")

    handle = ModelHandle(
        model_id="m_mout",
        name="m_mout",
        config={"recommender_type": "ranking", "scorer_type": "multioutput"},
        recommender=_StubRecommender(),
        datasets_used={"bundle_id": None},
    )
    session.trained_models["m_mout"] = handle

    result = TOOL_EVALUATE_MODEL.fn(
        model_id="m_mout",
        evaluator_type="simple",
        metrics=["NDCG_at_k"],
        k_values=[10],
        per_label=True,
        session=session,
    )
    assert result["status"] == "error"
    assert result["error_type"] == "PerLabelUnsupported"
    assert result["category"] == "per_label_unsupported_for_ranking"


def test_evaluate_non_k_metric_passes_eval_top_k_sentinel_to_real_recommender(trained_model, session):
    """eval_top_k=1 sentinel for non-K metrics (rmse, mae, roc_auc,
    pr_auc, expected_reward) must survive a real
    ``handle.recommender.evaluate`` call — not just a stub. Without this
    integration cover, a future upstream guard like ``if top_k <= 0:
    raise`` (or rejecting 1 as too small for ranking metrics) would
    silently break every non-K metric path with no test signal.

    Uses the existing trained_model fixture (long-format universal scorer +
    XGBoost) which exercises the real upstream evaluator codepath. ROC-AUC
    on a tiny fixture may not be meaningful, but the test contract is
    'evaluate_model returns ok' — not 'metric value is informative'.
    """
    result = TOOL_EVALUATE_MODEL.fn(
        model_id=trained_model,
        evaluator_type="simple",
        metrics=["roc_auc"],
        k_values=[5, 10],  # multiple k_values — should still record one entry
        per_label=False,
        session=session,
    )
    assert result["status"] == "ok", result
    # Exactly one entry for roc_auc despite two k_values
    roc_entries = [e for e in result["data"]["results"] if e["metric"] == "roc_auc"]
    assert len(roc_entries) == 1
    assert "k" not in roc_entries[0]
    handle = session.trained_models[trained_model]
    assert "roc_auc" in handle.metrics
    assert "roc_auc@5" not in handle.metrics
    assert "roc_auc@10" not in handle.metrics


def test_evaluate_non_k_metric_recorded_without_k_suffix(trained_model, session):
    """rmse / mae / roc_auc / pr_auc don't depend on k.
    handle.metrics must key them without ``@k`` and not duplicate across
    k_values. Wrap recommender.evaluate to return a fixed scalar so this
    runs without a regressor estimator."""
    handle = session.trained_models[trained_model]
    handle.recommender.evaluate = lambda **kw: 0.42

    result = TOOL_EVALUATE_MODEL.fn(
        model_id=trained_model,
        evaluator_type="simple",
        metrics=["roc_auc"],
        k_values=[5, 10, 20],
        # Explicit per_label=False — the long-format-classification gate
        # would otherwise MissingDecision on this combination.
        per_label=False,
        session=session,
    )
    assert result["status"] == "ok"
    # Exactly one entry returned despite three k values.
    assert len(result["data"]["results"]) == 1
    assert result["data"]["results"][0]["metric"] == "roc_auc"
    assert "k" not in result["data"]["results"][0]
    # Handle key has no @k suffix.
    assert "roc_auc" in handle.metrics
    assert "roc_auc@5" not in handle.metrics
    assert "roc_auc@10" not in handle.metrics


def test_evaluate_classification_metric_on_counterfactual_evaluator_rejected(trained_model, session):
    """roc_auc / pr_auc on IPS / DR / SNIPS — upstream ROCAUC.calculate
    raises 'Counterfactual evaluators (IPS, DR, SNIPS) are not compatible
    with classification metrics. Use SimpleEvaluator or ReplayMatchEvaluator.'
    Pre-gate so the user gets the agent's structured envelope rather than
    chasing the upstream traceback.
    """
    for evaluator in ("IPS", "DR", "SNIPS"):
        result = TOOL_EVALUATE_MODEL.fn(
            model_id=trained_model,
            evaluator_type=evaluator,
            metrics=["roc_auc"],
            k_values=[10],
            per_label=False,
            session=session,
        )
        assert result["status"] == "error", evaluator
        assert result["error_type"] == "EvaluatorMetricMismatch", evaluator
        assert result["category"] == "evaluator_metric_mismatch", evaluator


def test_evaluate_classification_metric_accepts_replay_match(trained_model, session):
    """upstream says 'Use SimpleEvaluator or ReplayMatchEvaluator'
    — roc_auc on replay_match must NOT trigger the agent gate. The exact
    metric calculation may still succeed or fail at the evaluator layer
    (depends on logged_items shape), but the agent shouldn't pre-reject
    a valid evaluator+metric pair.
    """
    result = TOOL_EVALUATE_MODEL.fn(
        model_id=trained_model,
        evaluator_type="replay_match",
        metrics=["roc_auc"],
        k_values=[10],
        per_label=False,
        session=session,
    )
    # The gate must not fire — either it succeeds, or fails with a
    # downstream error. The forbidden outcome is EvaluatorMetricMismatch.
    assert result.get("error_type") != "EvaluatorMetricMismatch"


def test_evaluate_invalid_evaluator_fires_before_missing_decision(tmp_path, session):
    """input validation must happen BEFORE the MissingDecision gates.
    A bogus evaluator name + per_label=None on a multi-target bundle
    should surface InvalidEvaluator immediately — not require the user
    to first answer the per_label elicitation, only to discover their
    evaluator was misspelled on round 2.
    """
    import pandas as pd

    from scikit_rec_agent.session import ModelHandle
    from scikit_rec_agent.tools.datasets import TOOL_CREATE_DATASETS

    df = pd.DataFrame(
        {
            "USER_ID": [f"u{i}" for i in range(10)],
            "ITEM_a": [i % 2 for i in range(10)],
            "ITEM_b": [(i + 1) % 2 for i in range(10)],
        }
    )
    p = tmp_path / "wide.csv"
    df.to_csv(p, index=False)
    TOOL_CREATE_DATASETS.fn(bundle_id="wb_r5", interactions_path=str(p), session=session)

    handle = ModelHandle(
        model_id="m_r5",
        name="m_r5",
        config={"recommender_type": "ranking", "scorer_type": "multioutput"},
        recommender=None,
        datasets_used={"bundle_id": "wb_r5"},
    )
    session.trained_models["m_r5"] = handle

    result = TOOL_EVALUATE_MODEL.fn(
        model_id="m_r5",
        evaluator_type="bogus_eval",
        metrics=["roc_auc"],
        k_values=[10],
        session=session,
        # per_label deliberately None — would normally MissingDecision
    )
    # InvalidEvaluator wins because we hoisted validation above the gate
    assert result["status"] == "error"
    assert result["error_type"] == "InvalidEvaluator"


def test_evaluate_rmse_on_non_simple_evaluator_rejected(trained_model, session):
    """rmse / mae only compose with evaluator_type='simple'. Off-policy
    evaluators reject upfront with a clear category so the LLM can pivot
    rather than chasing a less-localized upstream error."""
    result = TOOL_EVALUATE_MODEL.fn(
        model_id=trained_model,
        evaluator_type="IPS",
        metrics=["rmse"],
        k_values=[10],
        session=session,
    )
    assert result["status"] == "error"
    assert result["error_type"] == "EvaluatorMetricMismatch"
    assert result["category"] == "evaluator_metric_mismatch"


def test_score_items_kwargs_omits_users_for_multioutput_scorer(binary_reward_paths, session):
    """MultioutputScorer rejects a separate users frame at score_items
    time with ``Multioutput Scorer cannot accept Users Dataframe`` — the
    helper must strip users for it just like for embedding estimators."""
    from skrec.scorer.multioutput import MultioutputScorer

    from scikit_rec_agent.session import ModelHandle
    from scikit_rec_agent.tools.evaluation import _build_score_items_kwargs

    TOOL_CREATE_DATASETS.fn(
        bundle_id="b",
        interactions_path=binary_reward_paths["interactions"],
        users_path=binary_reward_paths["users"],
        items_path=binary_reward_paths["items"],
        session=session,
    )

    class _StubMultioutputScorer(MultioutputScorer):
        def __init__(self):
            self.is_classifier = True

    class _StubRecommender:
        scorer = _StubMultioutputScorer()

    handle = ModelHandle(
        model_id="m_mout",
        name="m_mout",
        config={},
        recommender=_StubRecommender(),
        datasets_used={"bundle_id": "b"},
    )
    session.trained_models["m_mout"] = handle

    kwargs = _build_score_items_kwargs(session, handle)
    assert "interactions" in kwargs
    assert "users" not in kwargs


def test_scorer_is_classifier_multioutput_uses_estimator_type(session):
    """when ``is_classifier`` isn't on the scorer (older or stripped
    wheel), the read derives from ``estimator._estimator_type`` instead of
    defaulting True. A regressor-mode scorer is correctly identified as
    NOT classifier — pinning so a future refactor doesn't reintroduce the
    default-True false positive on continuous targets."""
    from skrec.scorer.multioutput import MultioutputScorer

    from scikit_rec_agent.session import ModelHandle
    from scikit_rec_agent.tools.evaluation import (
        _scorer_is_classifier_multioutput,
        _scorer_is_regressor_multioutput,
    )

    class _RegressorEstimator:
        _estimator_type = "regressor"

    class _ClassifierEstimator:
        _estimator_type = "classifier"

    class _StubScorerRegressor(MultioutputScorer):
        def __init__(self):
            # Deliberately omit is_classifier — exercises the fallback path.
            self.estimator = _RegressorEstimator()

    class _StubScorerClassifier(MultioutputScorer):
        def __init__(self):
            self.estimator = _ClassifierEstimator()

    class _R:
        def __init__(self, scorer):
            self.scorer = scorer

    h_reg = ModelHandle(
        model_id="hr",
        name="hr",
        config={},
        recommender=_R(_StubScorerRegressor()),
        datasets_used={},
    )
    h_clf = ModelHandle(
        model_id="hc",
        name="hc",
        config={},
        recommender=_R(_StubScorerClassifier()),
        datasets_used={},
    )

    assert _scorer_is_classifier_multioutput(h_reg) is False
    assert _scorer_is_regressor_multioutput(h_reg) is True
    assert _scorer_is_classifier_multioutput(h_clf) is True
    assert _scorer_is_regressor_multioutput(h_clf) is False


def test_evaluate_reuses_score_cache_across_calls(trained_model, session):
    # Regression: the first evaluate_model pass populates the recommender's
    # score cache; the second call must SKIP passing score_items_kwargs so
    # scikit-rec's cache is honored instead of re-scoring. Verified by
    # spying on recommender.evaluate.
    handle = session.trained_models[trained_model]
    assert handle.score_cache_populated is False

    real_evaluate = handle.recommender.evaluate
    seen_kwargs: list[dict[str, Any] | None] = []

    def _spy(**kw):
        seen_kwargs.append(kw.get("score_items_kwargs"))
        return real_evaluate(**kw)

    handle.recommender.evaluate = _spy

    first = TOOL_EVALUATE_MODEL.fn(
        model_id=trained_model,
        evaluator_type="simple",
        metrics=["NDCG_at_k"],
        k_values=[5],
        session=session,
    )
    assert first["status"] == "ok"
    assert handle.score_cache_populated is True
    # First evaluate(): score_items_kwargs present (non-None dict).
    assert seen_kwargs[0] is not None

    second = TOOL_EVALUATE_MODEL.fn(
        model_id=trained_model,
        evaluator_type="simple",
        metrics=["precision_at_k"],
        k_values=[5],
        session=session,
    )
    assert second["status"] == "ok"
    # Second evaluate(): score_items_kwargs is None — cache reused.
    assert seen_kwargs[1] is None
