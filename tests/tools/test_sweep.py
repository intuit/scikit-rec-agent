"""Tests for sweep_methods tool."""

from __future__ import annotations

import pandas as pd
import pytest

from scikit_rec_agent.tools.datasets import TOOL_CREATE_DATASETS
from scikit_rec_agent.tools.splitting import TOOL_SPLIT_DATA
from scikit_rec_agent.tools.sweep import (
    TOOL_SWEEP_METHODS,
    _AUTO_SWEEPS,
    _EMBEDDING_FAMILIES,
    _broad_sweep_for_contract,
    _config_hash,
    _detect_bundle_contract,
    _filter_by_capability,
    _is_compatible,
    _normalise_method,
    _split_metric_spec,
)


# ---------------------------------------------------------------------------
# Capability filter — pure logic
# ---------------------------------------------------------------------------


def test_capability_filter_drops_multioutput_embedding():
    method = {
        "recommender_type": "ranking",
        "scorer_type": "multioutput",
        "estimator_type": "embedding",
        "estimator_config": {"estimator_type": "embedding"},
    }
    ok, reason = _is_compatible(method)
    assert not ok
    assert "embedding" in reason.lower()


def test_capability_filter_keeps_universal_embedding():
    method = {
        "recommender_type": "ranking",
        "scorer_type": "universal",
        "estimator_type": "embedding",
    }
    ok, _ = _is_compatible(method)
    assert ok


def test_capability_filter_drops_unknown_recommender():
    method = {
        "recommender_type": "made_up",
        "scorer_type": "universal",
        "estimator_type": "tabular",
    }
    ok, reason = _is_compatible(method)
    assert not ok
    assert "made_up" in reason


def test_filter_by_capability_returns_runnable_and_dropped():
    methods = [
        {"recommender_type": "ranking", "scorer_type": "universal", "estimator_type": "tabular"},
        {"recommender_type": "ranking", "scorer_type": "multioutput", "estimator_type": "embedding"},
    ]
    runnable, dropped = _filter_by_capability(methods)
    assert len(runnable) == 1
    assert len(dropped) == 1
    assert dropped[0]["reason"]


# ---------------------------------------------------------------------------
# Idempotency hash
# ---------------------------------------------------------------------------


def test_config_hash_stable_under_field_order():
    a = {"recommender_type": "ranking", "scorer_type": "universal", "estimator_type": "tabular"}
    b = {"scorer_type": "universal", "recommender_type": "ranking", "estimator_type": "tabular"}
    assert _config_hash("b1", a) == _config_hash("b1", b)


def test_config_hash_changes_with_bundle():
    method = {"recommender_type": "ranking", "scorer_type": "universal", "estimator_type": "tabular"}
    assert _config_hash("b1", method) != _config_hash("b2", method)


def test_config_hash_ignores_short_name():
    a = {"recommender_type": "ranking", "scorer_type": "universal", "estimator_type": "tabular", "short_name": "x"}
    b = {"recommender_type": "ranking", "scorer_type": "universal", "estimator_type": "tabular", "short_name": "y"}
    assert _config_hash("b1", a) == _config_hash("b1", b)


# ---------------------------------------------------------------------------
# Metric spec parsing
# ---------------------------------------------------------------------------


def test_split_metric_at_form():
    name, k = _split_metric_spec("NDCG_at_k", default_k=10)
    assert (name, k) == ("NDCG_at_k", 10)


def test_split_metric_with_explicit_k():
    name, k = _split_metric_spec("NDCG_at_k@20", default_k=10)
    assert (name, k) == ("NDCG_at_k", 20)


def test_split_metric_at_number_form_canonicalises():
    """Regression: when the LLM passes 'NDCG_at_10' (numeric instead of
    literal 'k'), the regex previously yielded name='NDCG_at' which
    scikit-rec rejects with 'Unknown metric'. The parser now restores
    the trailing '_k' so the canonical name survives extraction."""
    name, k = _split_metric_spec("NDCG_at_10", default_k=99)
    assert (name, k) == ("NDCG_at_k", 10)
    name, k = _split_metric_spec("precision_at_5", default_k=99)
    assert (name, k) == ("precision_at_k", 5)


# ---------------------------------------------------------------------------
# Auto-sweep + broad-sweep tables
# ---------------------------------------------------------------------------


def test_auto_sweep_has_long_interactions_methods():
    methods = _AUTO_SWEEPS["long_interactions"]
    assert len(methods) >= 1
    for m in methods:
        ok, _ = _is_compatible(m)
        assert ok, f"auto-sweep entry not compatible: {m}"


def test_auto_sweep_wide_multioutput_uses_multioutput_scorer():
    methods = _AUTO_SWEEPS["wide_multioutput"]
    assert any(m["scorer_type"] == "multioutput" for m in methods)


def test_broad_sweep_long_interactions_excludes_sequential():
    methods = _broad_sweep_for_contract("long_interactions")
    scorers = {m["scorer_type"] for m in methods}
    assert "sequential" not in scorers
    assert "hierarchical" not in scorers


# ---------------------------------------------------------------------------
# Contract detection
# ---------------------------------------------------------------------------


def test_detect_bundle_contract_long_interactions(binary_reward_paths, session):
    TOOL_CREATE_DATASETS.fn(
        bundle_id="b",
        interactions_path=binary_reward_paths["interactions"],
        users_path=binary_reward_paths["users"],
        items_path=binary_reward_paths["items"],
        session=session,
    )
    contract = _detect_bundle_contract(session.loaded_datasets["b"])
    assert contract == "long_interactions"


# ---------------------------------------------------------------------------
# End-to-end: sweep on the binary reward fixture
# ---------------------------------------------------------------------------


@pytest.fixture
def split_bundle(binary_reward_paths, session):
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
    return session


def test_sweep_explicit_methods_runs_and_ranks(split_bundle):
    session = split_bundle
    result = TOOL_SWEEP_METHODS.fn(
        bundle_id="b",
        methods=[
            {
                "short_name": "xgb_small",
                "recommender_type": "ranking",
                "scorer_type": "universal",
                "estimator_type": "tabular",
                "estimator_config": {
                    "ml_task": "classification",
                    "xgboost": {"n_estimators": 10, "max_depth": 3},
                },
            }
        ],
        metrics=["NDCG_at_k"],
        primary_metric="NDCG_at_k",
        eval_top_k=5,
        session=session,
    )
    assert result["status"] == "ok", result
    data = result["data"]
    assert data["n_runnable"] == 1
    assert len(data["leaderboard"]) == 1
    row = data["leaderboard"][0]
    assert row["status"] == "ok"
    assert "NDCG_at_k@5" in row["metrics"]


def test_sweep_filters_incompatible_explicit_methods(split_bundle):
    session = split_bundle
    result = TOOL_SWEEP_METHODS.fn(
        bundle_id="b",
        methods=[
            {
                "short_name": "good",
                "recommender_type": "ranking",
                "scorer_type": "universal",
                "estimator_type": "tabular",
                "estimator_config": {
                    "ml_task": "classification",
                    "xgboost": {"n_estimators": 10},
                },
            },
            {
                "short_name": "incompatible",
                "recommender_type": "ranking",
                "scorer_type": "multioutput",
                "estimator_type": "embedding",
                "estimator_config": {"estimator_type": "embedding"},
            },
        ],
        metrics=["NDCG_at_k"],
        primary_metric="NDCG_at_k",
        eval_top_k=5,
        session=session,
    )
    assert result["status"] == "ok"
    data = result["data"]
    assert data["n_runnable"] == 1
    assert data["n_dropped_incompatible"] == 1
    assert data["dropped_methods"][0]["method"]["short_name"] == "incompatible"


def test_sweep_cached_with_empty_metrics_re_evaluates(split_bundle):
    """If a previous sweep call trained the model but its evaluation failed
    (so handle.metrics is empty), the next sweep call must NOT just return
    a 'cached' row with no metrics — it must re-evaluate so the leaderboard
    gets real numbers. Regression for a real-data flow where the agent
    passed a bad metric name on attempt 1, then the correct name on
    attempt 2 and we ended up with all-empty cached rows."""
    session = split_bundle
    method = {
        "short_name": "xgb_small",
        "recommender_type": "ranking",
        "scorer_type": "universal",
        "estimator_type": "tabular",
        "estimator_config": {
            "ml_task": "classification",
            "xgboost": {"n_estimators": 10, "max_depth": 3},
        },
    }
    # First run: trains + evaluates normally.
    first = TOOL_SWEEP_METHODS.fn(
        bundle_id="b",
        methods=[method],
        metrics=["NDCG_at_k"],
        primary_metric="NDCG_at_k",
        eval_top_k=5,
        session=session,
    )
    assert first["status"] == "ok"
    model_id = first["data"]["leaderboard"][0]["model_id"]
    handle = session.trained_models[model_id]
    # Simulate the broken state: training succeeded but eval failed last time
    # so metrics never landed on the handle.
    handle.metrics.clear()
    handle.score_cache_populated = False

    # Second run with skip_existing should hit cached branch but re-evaluate.
    second = TOOL_SWEEP_METHODS.fn(
        bundle_id="b",
        methods=[method],
        metrics=["NDCG_at_k"],
        primary_metric="NDCG_at_k",
        eval_top_k=5,
        skip_existing=True,
        session=session,
    )
    assert second["status"] == "ok"
    row = second["data"]["leaderboard"][0]
    assert row["status"] == "cached"
    # Critically: metrics are populated, not empty
    assert row["metrics"], f"cached path returned empty metrics: {row}"
    assert "NDCG_at_k@5" in row["metrics"]


def test_sweep_idempotency_skip_existing(split_bundle):
    session = split_bundle
    method = {
        "short_name": "xgb_small",
        "recommender_type": "ranking",
        "scorer_type": "universal",
        "estimator_type": "tabular",
        "estimator_config": {
            "ml_task": "classification",
            "xgboost": {"n_estimators": 10, "max_depth": 3},
        },
    }
    first = TOOL_SWEEP_METHODS.fn(
        bundle_id="b",
        methods=[method],
        metrics=["NDCG_at_k"],
        primary_metric="NDCG_at_k",
        eval_top_k=5,
        session=session,
    )
    assert first["status"] == "ok"
    n_after_first = len(session.trained_models)

    second = TOOL_SWEEP_METHODS.fn(
        bundle_id="b",
        methods=[method],
        metrics=["NDCG_at_k"],
        primary_metric="NDCG_at_k",
        eval_top_k=5,
        skip_existing=True,
        session=session,
    )
    assert second["status"] == "ok"
    # Same bundle + same method = no new model trained
    assert len(session.trained_models) == n_after_first
    assert second["data"]["leaderboard"][0]["status"] == "cached"


def test_sweep_failure_isolation(split_bundle):
    """A failing method recorded as status='error', other methods still run."""
    session = split_bundle
    result = TOOL_SWEEP_METHODS.fn(
        bundle_id="b",
        methods=[
            {
                "short_name": "good",
                "recommender_type": "ranking",
                "scorer_type": "universal",
                "estimator_type": "tabular",
                "estimator_config": {
                    "ml_task": "classification",
                    "xgboost": {"n_estimators": 10},
                },
            },
            {
                "short_name": "bad",
                "recommender_type": "ranking",
                "scorer_type": "universal",
                "estimator_type": "tabular",
                # Invalid ml_task triggers a factory error this method
                # surfaces as status='error' without poisoning the others.
                "estimator_config": {"ml_task": "bogus_task"},
            },
        ],
        metrics=["NDCG_at_k"],
        primary_metric="NDCG_at_k",
        eval_top_k=5,
        session=session,
    )
    assert result["status"] == "ok"
    data = result["data"]
    statuses = [row["status"] for row in data["leaderboard"]]
    assert "ok" in statuses
    assert "error" in statuses


def test_sweep_returns_error_when_all_methods_fail(split_bundle):
    """When every runnable method errors, the sweep must return a top-level
    error envelope (not status=ok with winner=null) so the LLM has a
    structured signal to break out of a retry loop. The leaderboard is
    preserved on the envelope's `data` field for inspection."""
    session = split_bundle
    result = TOOL_SWEEP_METHODS.fn(
        bundle_id="b",
        methods=[
            {
                "short_name": "bad1",
                "recommender_type": "ranking",
                "scorer_type": "universal",
                "estimator_type": "tabular",
                "estimator_config": {"ml_task": "bogus_task"},
            },
            {
                "short_name": "bad2",
                "recommender_type": "ranking",
                "scorer_type": "universal",
                "estimator_type": "tabular",
                "estimator_config": {"ml_task": "another_bogus"},
            },
        ],
        metrics=["NDCG_at_k"],
        primary_metric="NDCG_at_k",
        eval_top_k=5,
        session=session,
    )
    assert result["status"] == "error"
    assert result["error_type"] == "SweepAllMethodsFailed"
    # Category is propagated from the per-method failures
    assert "category" in result
    # Leaderboard preserved alongside the error for the LLM to inspect
    assert "data" in result
    assert len(result["data"]["leaderboard"]) == 2
    assert all(r["status"] == "error" for r in result["data"]["leaderboard"])


def test_sweep_auto_picks_methods_for_long_interactions(split_bundle):
    """`methods='auto'` for a long_interactions bundle should expose the
    expected method families (tabular + embedding). Use methods='list' to
    get the menu without paying the cost of actually training every entry —
    auto and list draw from the same _AUTO_SWEEPS table, so listing is a
    sound proxy for what auto would run."""
    session = split_bundle
    result = TOOL_SWEEP_METHODS.fn(
        bundle_id="b",
        methods="list",
        metrics=["NDCG_at_k"],
        primary_metric="NDCG_at_k",
        eval_top_k=5,
        session=session,
    )
    assert result["status"] == "ok", result
    data = result["data"]
    assert data["contract"] == "long_interactions"
    assert data["n_available"] >= 1
    short_names = {m["short_name"] for m in data["available_methods"]}
    # Expect at least the tabular baseline + at least one embedding family
    assert "xgb_universal" in short_names
    assert any("mf" in s or "ncf" in s or "two_tower" in s or "dcn" in s or "nfm" in s for s in short_names)


def test_sweep_no_bundle_returns_error(session):
    result = TOOL_SWEEP_METHODS.fn(
        bundle_id="missing",
        methods="auto",
        metrics=["NDCG_at_k"],
        primary_metric="NDCG_at_k",
        session=session,
    )
    assert result["status"] == "error"
    assert result["error_type"] == "BundleNotFound"


def test_sweep_list_mode_returns_menu(split_bundle):
    """methods='list' returns the candidate menu without training."""
    session = split_bundle
    n_models_before = len(session.trained_models)
    result = TOOL_SWEEP_METHODS.fn(
        bundle_id="b",
        methods="list",
        metrics=["NDCG_at_k"],
        primary_metric="NDCG_at_k",
        session=session,
    )
    assert result["status"] == "ok"
    data = result["data"]
    assert "available_methods" in data
    assert data["n_available"] >= 1
    # Each menu entry has a numbered option + short_name
    first = data["available_methods"][0]
    assert first["option"] == 1
    assert "short_name" in first
    # Critically: nothing was trained
    assert len(session.trained_models) == n_models_before


def test_sweep_methods_short_names_picks_from_menu(split_bundle):
    """User picks options by short_name → sweep_methods looks them up in the
    contract's auto-sweep table and runs only those."""
    session = split_bundle
    result = TOOL_SWEEP_METHODS.fn(
        bundle_id="b",
        methods=["xgb_universal"],
        metrics=["NDCG_at_k"],
        primary_metric="NDCG_at_k",
        eval_top_k=5,
        session=session,
    )
    assert result["status"] == "ok"
    assert result["data"]["n_runnable"] == 1
    assert result["data"]["leaderboard"][0]["method"]["short_name"] == "xgb_universal"


def test_sweep_methods_unknown_short_name_returns_error(split_bundle):
    session = split_bundle
    result = TOOL_SWEEP_METHODS.fn(
        bundle_id="b",
        methods=["totally_made_up"],
        metrics=["NDCG_at_k"],
        primary_metric="NDCG_at_k",
        session=session,
    )
    assert result["status"] == "error"
    assert result["error_type"] == "UnknownMethodShortName"
    assert "totally_made_up" in result["message"]


def test_all_embedding_families_train_on_same_bundle(binary_reward_paths, tmp_path, session):
    """scikit-rec's design promise: one bundle (interactions + users + items)
    feeds every estimator family. This test iterates over `_EMBEDDING_FAMILIES`
    and trains each on the same DatasetBundle — proving the data-input
    uniformity whether the list has one entry or five.

    A regression here would mean an auto-sweep entry has wrong parameter
    names (n_factors vs embedding_dim is the canonical mistake) or the
    train path requires family-specific data prep. The macOS BLAS thread
    pollution that previously made NCF / Two-Tower / DCN / NFM segfault
    after MF in the same process is mitigated at package import in
    scikit_rec_agent/__init__.py — without that fix this test would
    SIGSEGV the whole pytest worker, not just fail.
    """
    from scikit_rec_agent.tools.sweep import _build_train_args

    TOOL_CREATE_DATASETS.fn(
        bundle_id="b",
        interactions_path=binary_reward_paths["interactions"],
        users_path=binary_reward_paths["users"],
        items_path=binary_reward_paths["items"],
        session=session,
    )
    bundle = session.loaded_datasets["b"]

    from skrec.orchestrator import create_recommender_pipeline

    failures = {}
    for method in _EMBEDDING_FAMILIES:
        config = _build_train_args(method)
        try:
            recommender = create_recommender_pipeline(config)
            train_kwargs = {
                "interactions_ds": bundle.interactions,
                "users_ds": bundle.users,
                "items_ds": bundle.items,
            }
            recommender.train(**train_kwargs)
        except Exception as e:
            failures[method["short_name"]] = f"{type(e).__name__}: {e}"

    assert not failures, (
        "All five embedding families must train on the same bundle. "
        f"Failures: {failures}"
    )


def test_normalise_method_fills_defaults():
    out = _normalise_method({"recommender_type": "ranking", "scorer_type": "universal"})
    assert out["estimator_type"] == "tabular"
    assert "estimator_config" in out
    assert "short_name" in out
