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


def test_evaluate_wide_bundle_returns_structured_error(tmp_path, session):
    """Wide multi-output bundle + simple evaluator + no explicit eval_kwargs
    must return a structured WideBundleEvalUnsupported error rather than
    KeyError'ing on the missing ITEM_ID / OUTCOME columns."""
    import pandas as pd

    # Build a wide-multioutput bundle without going through a real train.
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

    # Stick a fake handle on the session so evaluate_model has something to look up.
    from scikit_rec_agent.session import ModelHandle

    handle = ModelHandle(
        model_id="m_wide",
        name="m_wide",
        config={"recommender_type": "ranking", "scorer_type": "multioutput"},
        recommender=None,
        datasets_used={"bundle_id": "wide"},
    )
    session.trained_models["m_wide"] = handle

    result = TOOL_EVALUATE_MODEL.fn(
        model_id="m_wide",
        evaluator_type="simple",
        metrics=["NDCG_at_k"],
        k_values=[5],
        session=session,
    )
    assert result["status"] == "error"
    assert result["error_type"] == "WideBundleEvalUnsupported"
    assert result["category"] == "wide_bundle_eval_unsupported"
    assert "interaction_multioutput" in result["message"]


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
