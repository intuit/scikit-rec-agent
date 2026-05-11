"""Tests for sweep_methods tool."""

from __future__ import annotations

import pytest

from scikit_rec_agent.tools.datasets import TOOL_CREATE_DATASETS
from scikit_rec_agent.tools.splitting import TOOL_SPLIT_DATA
from scikit_rec_agent.tools.sweep import (
    _AUTO_SWEEPS,
    _EMBEDDING_FAMILIES,
    TOOL_SWEEP_METHODS,
    _broad_sweep_for_contract,
    _config_hash,
    _data_aware_methods,
    _detect_bundle_contract,
    _filter_by_capability,
    _filter_by_profile,
    _is_compatible,
    _normalise_method,
    _profile_bundle,
    _resize_for_data_scale,
    _scale_tier,
    _split_metric_spec,
)

try:
    import torch  # type: ignore[import-unresolved]  # noqa: F401

    _torch_available = True
except ImportError:
    _torch_available = False

requires_torch = pytest.mark.skipif(not _torch_available, reason="PyTorch not installed")

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


def test_metric_higher_is_better_default_and_loss_metrics():
    """Regression: a fixed-direction sort_key would order loss-shaped metrics
    backwards. Sweep now consults a small registry of lower-is-better names
    so future loss metrics sort correctly without flipping every existing
    higher-is-better metric."""
    from scikit_rec_agent.tools.sweep import _metric_higher_is_better

    # All current scikit-rec metrics are higher-is-better.
    for m in ("NDCG_at_k", "MAP_at_k", "MRR_at_k", "precision_at_k", "recall_at_k", "roc_auc", "pr_auc"):
        assert _metric_higher_is_better(m), f"{m} should be higher-is-better"
    # Loss-shaped names are lower-is-better.
    for m in ("logloss", "log_loss", "rmse", "mse", "mae", "cross_entropy"):
        assert not _metric_higher_is_better(m), f"{m} should be lower-is-better"


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


def test_auto_sweep_wide_multioutput_has_classifier_and_regressor_entries():
    """both classifier and regression entries must exist so a wide
    bundle with continuous ITEM_* values can reach MultiOutputRegressor
    via methods='all'. _filter_by_profile picks one based on profiled
    target_type.
    """
    methods = _AUTO_SWEEPS["wide_multioutput"]
    short_names = {m["short_name"] for m in methods}
    assert "xgb_multioutput" in short_names
    assert "xgb_multioutput_regression" in short_names


def test_filter_by_profile_binary_wide_keeps_classifier_drops_regression():
    """when ITEM_* are binary, the classifier entry survives and the
    regression entry is filtered out with a clear reason. Inverse case
    pinned in the regression-targets test below.
    """
    profile = {"n_rows": 1000, "n_users": 100, "target_type": "binary"}
    keep_clf, _ = _filter_by_profile({"short_name": "xgb_multioutput"}, profile)
    keep_reg, reason = _filter_by_profile({"short_name": "xgb_multioutput_regression"}, profile)
    assert keep_clf is True
    assert keep_reg is False
    assert "binary" in (reason or "").lower()


def test_filter_by_profile_continuous_wide_keeps_regression_drops_classifier():
    profile = {"n_rows": 1000, "n_users": 100, "target_type": "continuous"}
    keep_clf, reason = _filter_by_profile({"short_name": "xgb_multioutput"}, profile)
    keep_reg, _ = _filter_by_profile({"short_name": "xgb_multioutput_regression"}, profile)
    assert keep_clf is False
    assert "continuous" in (reason or "").lower()
    assert keep_reg is True


def test_sweep_methods_per_label_missing_decision_on_multi_target_wide(tmp_path, session):
    """sweep_methods on a multi-target wide_multioutput bundle without
    explicit per_label returns MissingDecision so the sweep doesn't bypass
    the evaluate_model gate. Without this mirror, a 13-target QBO sweep
    would silently produce macro-only metrics while individual
    evaluate_model calls would have asked.
    """
    import pandas as pd

    from scikit_rec_agent.tools.datasets import TOOL_CREATE_DATASETS
    from scikit_rec_agent.tools.sweep import TOOL_SWEEP_METHODS

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
    TOOL_CREATE_DATASETS.fn(bundle_id="g1w", interactions_path=str(p), session=session)

    result = TOOL_SWEEP_METHODS.fn(
        bundle_id="g1w",
        metrics=["roc_auc"],
        primary_metric="roc_auc",
        methods=["xgb_multioutput"],
        session=session,
        drop_non_winners=False,  # explicit, so that gate doesn't fire
        # per_label deliberately omitted → defaults to None → must MissingDecision
    )
    assert result["status"] == "error"
    assert result["error_type"] == "MissingDecision"
    assert result["category"] == "missing_decision"
    assert "per_label" in result["message"]


def test_sweep_methods_per_label_missing_decision_on_long_format_classification(binary_reward_paths, session):
    """sweep_methods gate now mirrors evaluate_model — fires on
    long-format interactions bundles paired with a classification metric
    too, not just on wide_multioutput. Pin the symmetry with the system
    prompt MUST-ASK rule §2 which lists both cases.
    """
    from scikit_rec_agent.tools.datasets import TOOL_CREATE_DATASETS
    from scikit_rec_agent.tools.sweep import TOOL_SWEEP_METHODS

    TOOL_CREATE_DATASETS.fn(
        bundle_id="long_clf",
        interactions_path=binary_reward_paths["interactions"],
        users_path=binary_reward_paths["users"],
        items_path=binary_reward_paths["items"],
        session=session,
    )
    result = TOOL_SWEEP_METHODS.fn(
        bundle_id="long_clf",
        metrics=["roc_auc"],
        primary_metric="roc_auc",
        methods=["xgb_universal"],
        session=session,
        drop_non_winners=False,
        # per_label deliberately omitted on a long bundle + roc_auc → MissingDecision
    )
    assert result["status"] == "error"
    assert result["error_type"] == "MissingDecision"
    assert result["category"] == "missing_decision"
    assert "per_label" in result["message"]
    assert "Long-format" in result["message"] or "long" in result["message"].lower()


def test_sweep_methods_per_label_default_safe_on_long_bundle(binary_reward_paths, session):
    """long-format universal bundle should NOT trigger the
    per_label gate. The gate is data-conditional (multi-target multioutput
    only), not blanket. Without the carve-out, every sweep call would
    require explicit per_label which is annoying for the common ranking
    case.
    """
    from scikit_rec_agent.tools.datasets import TOOL_CREATE_DATASETS
    from scikit_rec_agent.tools.splitting import TOOL_SPLIT_DATA
    from scikit_rec_agent.tools.sweep import TOOL_SWEEP_METHODS

    TOOL_CREATE_DATASETS.fn(
        bundle_id="g1l",
        interactions_path=binary_reward_paths["interactions"],
        users_path=binary_reward_paths["users"],
        items_path=binary_reward_paths["items"],
        session=session,
    )
    TOOL_SPLIT_DATA.fn(
        bundle_id="g1l",
        strategy="random_split",
        valid_fraction=0.2,
        session=session,
        random_state=1,
    )
    result = TOOL_SWEEP_METHODS.fn(
        bundle_id="g1l",
        metrics=["NDCG_at_k"],
        primary_metric="NDCG_at_k",
        methods=["xgb_universal"],
        session=session,
        drop_non_winners=False,
        # per_label omitted → None → falls through to False, no MissingDecision
    )
    assert result["status"] == "ok", result


def test_sweep_methods_drop_non_winners_missing_decision_on_large_bundle(tmp_path, session):
    """a bundle with >100K rows must have drop_non_winners
    set explicitly. Default None triggers MissingDecision so the agent
    asks the user rather than silently consuming laptop RAM with retained
    recommenders. Build a 100,001-row bundle and verify the gate fires.
    """
    import pandas as pd

    from scikit_rec_agent.tools.datasets import TOOL_CREATE_DATASETS
    from scikit_rec_agent.tools.sweep import TOOL_SWEEP_METHODS

    n = 100_001
    df = pd.DataFrame(
        {
            "USER_ID": [f"u{i % 1000}" for i in range(n)],
            "ITEM_ID": [f"i{i % 500}" for i in range(n)],
            "OUTCOME": [float(i % 2) for i in range(n)],
        }
    )
    p = tmp_path / "big.csv"
    df.to_csv(p, index=False)
    TOOL_CREATE_DATASETS.fn(bundle_id="big", interactions_path=str(p), session=session)

    result = TOOL_SWEEP_METHODS.fn(
        bundle_id="big",
        metrics=["NDCG_at_k"],
        primary_metric="NDCG_at_k",
        methods=["xgb_universal"],
        session=session,
        # drop_non_winners deliberately omitted → defaults to None → must MissingDecision
    )
    assert result["status"] == "error"
    assert result["error_type"] == "MissingDecision"
    assert result["category"] == "missing_decision"
    assert "drop_non_winners" in result["message"]


def test_sweep_methods_list_does_not_trigger_drop_non_winners_gate(tmp_path, session):
    """methods='list' returns the menu without training anything, so
    drop_non_winners is irrelevant. Without this skip, calling
    sweep_methods(methods='list') on a >100K-row bundle would block on a
    question the user can't answer until they've SEEN the menu (chicken-
    and-egg). Pin the carve-out.
    """
    import pandas as pd

    from scikit_rec_agent.tools.datasets import TOOL_CREATE_DATASETS
    from scikit_rec_agent.tools.sweep import TOOL_SWEEP_METHODS

    n = 100_001
    df = pd.DataFrame(
        {
            "USER_ID": [f"u{i % 1000}" for i in range(n)],
            "ITEM_ID": [f"i{i % 500}" for i in range(n)],
            "OUTCOME": [float(i % 2) for i in range(n)],
        }
    )
    p = tmp_path / "big.csv"
    df.to_csv(p, index=False)
    TOOL_CREATE_DATASETS.fn(bundle_id="bigl", interactions_path=str(p), session=session)

    result = TOOL_SWEEP_METHODS.fn(
        bundle_id="bigl",
        metrics=["NDCG_at_k"],
        primary_metric="NDCG_at_k",
        methods="list",  # menu mode — gate must NOT fire
        session=session,
    )
    assert result["status"] == "ok", result
    assert "available_methods" in result["data"]


def test_sweep_methods_drop_non_winners_default_safe_on_small_bundle(binary_reward_paths, session):
    """the guardrail must NOT fire on small bundles. The default-None
    behavior is "ask explicitly only when memory pressure is real" —
    blocking every call would be obnoxious. Verify the small fixture
    proceeds without an explicit drop_non_winners.
    """
    from scikit_rec_agent.tools.datasets import TOOL_CREATE_DATASETS
    from scikit_rec_agent.tools.splitting import TOOL_SPLIT_DATA
    from scikit_rec_agent.tools.sweep import TOOL_SWEEP_METHODS

    TOOL_CREATE_DATASETS.fn(
        bundle_id="small",
        interactions_path=binary_reward_paths["interactions"],
        users_path=binary_reward_paths["users"],
        items_path=binary_reward_paths["items"],
        session=session,
    )
    TOOL_SPLIT_DATA.fn(
        bundle_id="small",
        strategy="random_split",
        valid_fraction=0.2,
        session=session,
        random_state=1,
    )
    result = TOOL_SWEEP_METHODS.fn(
        bundle_id="small",
        metrics=["NDCG_at_k"],
        primary_metric="NDCG_at_k",
        methods=["xgb_universal"],
        session=session,
    )
    assert result["status"] == "ok", result


def test_sweep_methods_all_requires_confirmed_all_flag(binary_reward_paths, session):
    """methods='all' without confirmed_all=True returns
    MissingDecision so the agent runs through the list → ask → re-call
    flow instead of silently dispatching every method.
    """
    from scikit_rec_agent.tools.datasets import TOOL_CREATE_DATASETS
    from scikit_rec_agent.tools.sweep import TOOL_SWEEP_METHODS

    TOOL_CREATE_DATASETS.fn(
        bundle_id="b",
        interactions_path=binary_reward_paths["interactions"],
        session=session,
    )
    result = TOOL_SWEEP_METHODS.fn(
        bundle_id="b",
        metrics=["NDCG_at_k"],
        primary_metric="NDCG_at_k",
        methods="all",  # without confirmed_all
        session=session,
        drop_non_winners=False,
    )
    assert result["status"] == "error"
    assert result["error_type"] == "MissingDecision"
    assert "confirmed_all" in result["message"]


def test_sweep_methods_list_emits_reshape_recommendation_for_wide_multioutput(tmp_path, session):
    """methods='list' on a wide_multioutput bundle surfaces
    a reshape_recommendation field so the agent asks the user whether to
    stay on the narrow wide menu or melt to long_interactions for a
    broader comparison.
    """
    import pandas as pd

    from scikit_rec_agent.tools.datasets import TOOL_CREATE_DATASETS
    from scikit_rec_agent.tools.sweep import TOOL_SWEEP_METHODS

    df = pd.DataFrame(
        {
            "USER_ID": [f"u{i}" for i in range(10)],
            "ITEM_a": [i % 2 for i in range(10)],
            "ITEM_b": [(i + 1) % 2 for i in range(10)],
        }
    )
    p = tmp_path / "wide.csv"
    df.to_csv(p, index=False)
    TOOL_CREATE_DATASETS.fn(bundle_id="w", interactions_path=str(p), session=session)

    result = TOOL_SWEEP_METHODS.fn(
        bundle_id="w",
        metrics=["roc_auc"],
        primary_metric="roc_auc",
        methods="list",
        session=session,
    )
    assert result["status"] == "ok"
    assert result["data"]["contract"] == "wide_multioutput"
    rec = result["data"].get("reshape_recommendation")
    assert rec is not None
    assert "long_interactions" in rec or "long_with_timestamp" in rec


def test_profile_bundle_wide_multioutput_classifies_item_dtype(tmp_path, session):
    """_profile_bundle inspects ITEM_* dtype/uniques for wide bundles
    (where OUTCOME is absent). Binary ITEM_* → target_type='binary';
    continuous ITEM_* → target_type='continuous'. Without this, the
    classifier-vs-regressor filter above would never fire for wide bundles
    even when both entries are listed in _AUTO_SWEEPS.
    """
    import pandas as pd

    from scikit_rec_agent.tools.datasets import TOOL_CREATE_DATASETS

    df_binary = pd.DataFrame(
        {
            "USER_ID": [f"u{i}" for i in range(20)],
            "ITEM_a": [i % 2 for i in range(20)],
            "ITEM_b": [(i + 1) % 2 for i in range(20)],
        }
    )
    p_bin = tmp_path / "wide_bin.csv"
    df_binary.to_csv(p_bin, index=False)
    TOOL_CREATE_DATASETS.fn(bundle_id="bw_bin", interactions_path=str(p_bin), session=session)
    profile = _profile_bundle(session.loaded_datasets["bw_bin"])
    assert profile["target_type"] == "binary"

    df_cont = pd.DataFrame(
        {
            "USER_ID": [f"u{i}" for i in range(20)],
            "ITEM_a": [float(i) * 1.7 for i in range(20)],
            "ITEM_b": [float(i) * 0.3 + 5 for i in range(20)],
        }
    )
    p_cont = tmp_path / "wide_cont.csv"
    df_cont.to_csv(p_cont, index=False)
    TOOL_CREATE_DATASETS.fn(bundle_id="bw_cont", interactions_path=str(p_cont), session=session)
    profile = _profile_bundle(session.loaded_datasets["bw_cont"])
    assert profile["target_type"] == "continuous"


def test_broad_sweep_long_interactions_excludes_sequential():
    methods = _broad_sweep_for_contract("long_interactions")
    scorers = {m["scorer_type"] for m in methods}
    assert "sequential" not in scorers
    assert "hierarchical" not in scorers


# ---------------------------------------------------------------------------
# Contract detection
# ---------------------------------------------------------------------------


def test_contract_detectors_agree_on_overlapping_shapes(tmp_path):
    """Regression: there used to be two separate detection functions
    (sweep._detect_bundle_contract vs datasets._detect_dataset_type) with
    different vocabularies AND slightly different rules. They could disagree
    on edge cases like 'USER_ID + ITEM_ID + ≥2 ITEM_* + no OUTCOME'.
    Both now go through sweep.contract_from_dataframe, so they must agree
    via the _CONTRACT_TO_DATASET_TYPE mapping for every shape we recognise."""
    import pandas as pd

    from scikit_rec_agent.tools.datasets import _CONTRACT_TO_DATASET_TYPE, _detect_dataset_type
    from scikit_rec_agent.tools.sweep import contract_from_dataframe

    fixtures: list[tuple[str, pd.DataFrame]] = [
        # long_interactions
        (
            "long_interactions",
            pd.DataFrame({"USER_ID": ["u1"], "ITEM_ID": ["i1"], "OUTCOME": [1.0]}),
        ),
        # long_with_timestamp
        (
            "long_with_timestamp",
            pd.DataFrame({"USER_ID": ["u1"], "ITEM_ID": ["i1"], "OUTCOME": [1.0], "TIMESTAMP": ["1"]}),
        ),
        # long_multi_reward
        (
            "long_multi_reward",
            pd.DataFrame({"USER_ID": ["u1"], "ITEM_ID": ["i1"], "OUTCOME": [1.0], "OUTCOME_revenue": [10.0]}),
        ),
        # wide_multioutput
        (
            "wide_multioutput",
            pd.DataFrame({"USER_ID": ["u1"], "ITEM_a": [1], "ITEM_b": [0]}),
        ),
        # multiclass
        (
            "multiclass",
            pd.DataFrame({"USER_ID": ["u1"], "ITEM_ID": ["A"]}),
        ),
        # prebuilt_sequences
        (
            "prebuilt_sequences",
            pd.DataFrame({"USER_ID": ["u1"], "ITEM_SEQUENCE": [["a", "b"]]}),
        ),
        # sessions
        (
            "sessions",
            pd.DataFrame({"USER_ID": ["u1"], "SESSION_SEQUENCES": [[["a", "b"], ["c"]]]}),
        ),
    ]
    for expected_contract, df in fixtures:
        contract = contract_from_dataframe(df)
        assert contract == expected_contract, (
            f"contract_from_dataframe returned {contract!r} for shape "
            f"{list(df.columns)}; expected {expected_contract!r}"
        )
        # _detect_dataset_type must match the translation map.
        ds_type = _detect_dataset_type(df)
        assert ds_type == _CONTRACT_TO_DATASET_TYPE[expected_contract], (
            f"detect_dataset_type returned {ds_type!r} for {expected_contract}; "
            f"expected {_CONTRACT_TO_DATASET_TYPE[expected_contract]!r}"
        )


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


def test_sweep_does_not_rekey_session_trained_models(split_bundle):
    """Regression: sweep used to pop the trained model out of
    session.trained_models and re-insert under its own deterministic
    sweep id. That broke train_model's contract — the model_id it
    returned was no longer findable in session.trained_models.

    Now the deterministic sweep id is a separate alias in
    session.sweep_cache, and the model stays at train_model's auto-id.
    """
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
    result = TOOL_SWEEP_METHODS.fn(
        bundle_id="b",
        methods=[method],
        metrics=["NDCG_at_k"],
        primary_metric="NDCG_at_k",
        eval_top_k=5,
        session=session,
    )
    row = result["data"]["leaderboard"][0]
    train_model_id = row["model_id"]
    sweep_cache_id = row["sweep_cache_id"]

    # The model_id returned by the sweep == train_model's auto-generated id,
    # which IS findable in session.trained_models.
    assert train_model_id in session.trained_models
    # The deterministic sweep id is recorded in session.sweep_cache as an
    # alias to that model_id.
    assert session.sweep_cache.get(sweep_cache_id) == train_model_id
    # The deterministic sweep id is NOT itself a key in trained_models.
    assert sweep_cache_id not in session.trained_models or sweep_cache_id == train_model_id


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

    torch is an OPTIONAL dependency of scikit-rec-agent (only pulled in via
    ``pip install scikit-rec-agent[torch]``); the four torch-based families
    (NCF / Two-Tower / DCN / NFM) skip when it isn't installed. MF is pure
    numpy and is always asserted, so the test still has teeth on a
    torch-free CI environment.
    """
    from scikit_rec_agent.tools.sweep import _build_train_args

    try:
        import torch  # noqa: F401

        torch_available = True
    except ImportError:
        torch_available = False

    TOOL_CREATE_DATASETS.fn(
        bundle_id="b",
        interactions_path=binary_reward_paths["interactions"],
        users_path=binary_reward_paths["users"],
        items_path=binary_reward_paths["items"],
        session=session,
    )
    bundle = session.loaded_datasets["b"]

    from skrec.orchestrator import create_recommender_pipeline

    failures: dict[str, str] = {}
    skipped: list[str] = []
    trained: list[str] = []
    for method in _EMBEDDING_FAMILIES:
        model_type = method["estimator_config"]["embedding"]["model_type"]
        is_torch_model = model_type != "matrix_factorization"
        if is_torch_model and not torch_available:
            skipped.append(method["short_name"])
            continue

        config = _build_train_args(method)
        try:
            recommender = create_recommender_pipeline(config)
            train_kwargs = {
                "interactions_ds": bundle.interactions,
                "users_ds": bundle.users,
                "items_ds": bundle.items,
            }
            recommender.train(**train_kwargs)
            trained.append(method["short_name"])
        except Exception as e:
            failures[method["short_name"]] = f"{type(e).__name__}: {e}"

    assert not failures, (
        "Embedding families that ran must train on the same bundle. "
        f"Failures: {failures}. Skipped (no torch): {skipped}."
    )
    # MF (numpy, no torch) must always train regardless of environment.
    assert "mf_universal" in trained, (
        f"MF should always be trainable (it's pure numpy). trained={trained}, skipped={skipped}"
    )


def test_scale_tier_picks_correct_band():
    assert _scale_tier(100)["name"] == "tiny"
    assert _scale_tier(50_000)["name"] == "small"
    assert _scale_tier(500_000)["name"] == "medium"
    assert _scale_tier(5_000_000)["name"] == "large"


def test_filter_drops_embeddings_on_tiny_data():
    profile = {
        "n_rows": 500,
        "sparsity": 0.99,
        "target_type": "binary",
        "has_timestamps": True,
        "has_user_features": True,
        "has_item_features": True,
    }
    for method in _EMBEDDING_FAMILIES:
        keep, reason = _filter_by_profile(method, profile)
        if "xgb" in method["short_name"]:
            assert keep
        else:
            # MF needs n_rows>=1000; NCF/Two-Tower/DCN/NFM need n_rows>=5000
            assert not keep, f"expected to drop {method['short_name']} on n_rows=500"
            assert reason


def test_filter_keeps_mf_in_sparse_cf_regime():
    profile = {
        "n_rows": 50_000,
        "sparsity": 0.99,
        "target_type": "binary",
        "has_timestamps": False,
        "has_user_features": False,
        "has_item_features": False,
    }
    mf = next(m for m in _EMBEDDING_FAMILIES if "mf" in m["short_name"])
    keep, reason = _filter_by_profile(mf, profile)
    assert keep, f"MF should survive sparse CF regime; reason={reason}"


def test_filter_keeps_mf_at_realistic_cf_sparsity():
    """Regression: previous floor was 0.95 which filtered out MovieLens-1M
    (sparsity ≈ 0.949) and most retail / catalogue datasets (typically
    0.80–0.95). 0.90 floor keeps the realistic CF regime."""
    mf = next(m for m in _EMBEDDING_FAMILIES if "mf" in m["short_name"])
    profile = {"n_rows": 50_000, "sparsity": 0.90, "target_type": "binary"}
    keep, _ = _filter_by_profile(mf, profile)
    assert keep, "MF must survive at sparsity = 0.90 (the new floor)"


def test_filter_drops_mf_on_dense_data():
    profile = {
        "n_rows": 50_000,
        "sparsity": 0.5,
        "target_type": "binary",
        "has_timestamps": False,
        "has_user_features": True,
        "has_item_features": True,
    }
    mf = next(m for m in _EMBEDDING_FAMILIES if "mf" in m["short_name"])
    keep, reason = _filter_by_profile(mf, profile)
    assert not keep
    assert "sparsity" in reason.lower()


def test_filter_drops_two_tower_without_features():
    profile = {
        "n_rows": 50_000,
        "sparsity": 0.95,
        "target_type": "binary",
        "has_timestamps": False,
        "has_user_features": False,
        "has_item_features": False,
    }
    tt = next(m for m in _EMBEDDING_FAMILIES if "two_tower" in m["short_name"])
    keep, reason = _filter_by_profile(tt, profile)
    assert not keep
    assert "feature" in reason.lower()


def test_filter_drops_non_xgb_on_continuous_target():
    profile = {
        "n_rows": 100_000,
        "sparsity": 0.99,
        "target_type": "continuous",
        "has_timestamps": True,
        "has_user_features": True,
        "has_item_features": True,
    }
    for method in _EMBEDDING_FAMILIES:
        keep, reason = _filter_by_profile(method, profile)
        if "xgb" in method["short_name"]:
            assert keep
        else:
            assert not keep, f"continuous target should drop {method['short_name']}"
            assert "continuous" in reason.lower()


def test_resize_sequential_for_continuous_target_swaps_to_regressor():
    """The filter step keeps SASRec/HRNN under continuous targets on the
    promise the resize step swaps `*_classifier` → `*_regressor`. Verify
    that swap actually happens — otherwise the filter is silently lying
    and we'd train a classifier on continuous data."""
    method = {
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
    }
    profile = {"n_rows": 50_000, "target_type": "continuous"}
    resized = _resize_for_data_scale(method, profile)
    assert resized["estimator_config"]["sequential"]["model_type"] == "sasrec_regressor", (
        f"continuous target should swap classifier→regressor; got "
        f"{resized['estimator_config']['sequential']['model_type']!r}"
    )


def test_resize_xgb_for_continuous_target_swaps_ml_task():
    method = {
        "short_name": "xgb_universal",
        "recommender_type": "ranking",
        "scorer_type": "universal",
        "estimator_type": "tabular",
        "estimator_config": {"ml_task": "classification", "xgboost": {"n_estimators": 50, "max_depth": 4}},
    }
    profile = {"n_rows": 50_000, "target_type": "continuous"}
    resized = _resize_for_data_scale(method, profile)
    assert resized["estimator_config"]["ml_task"] == "regression"
    # original method untouched
    assert method["estimator_config"]["ml_task"] == "classification"


def test_resize_scales_embedding_dim_with_data_size():
    mf = next(m for m in _EMBEDDING_FAMILIES if "mf" in m["short_name"])

    tiny = _resize_for_data_scale(mf, {"n_rows": 500, "target_type": "binary"})
    medium = _resize_for_data_scale(mf, {"n_rows": 500_000, "target_type": "binary"})
    large = _resize_for_data_scale(mf, {"n_rows": 5_000_000, "target_type": "binary"})

    tiny_factors = tiny["estimator_config"]["embedding"]["params"]["n_factors"]
    medium_factors = medium["estimator_config"]["embedding"]["params"]["n_factors"]
    large_factors = large["estimator_config"]["embedding"]["params"]["n_factors"]

    assert tiny_factors < medium_factors < large_factors


def test_data_aware_methods_drops_then_resizes(binary_reward_paths, session):
    """End-to-end on the binary_reward fixture: 5000 users × 3 items × 1 row
    each. Sparsity ≈ 0.667 (5000/(5000*3) = 0.333 density), n_rows = 5000.
    Expect MF to be dropped (dense), embeddings dropped (n_rows boundary),
    XGBoost kept and sized to the 'tiny'/'small' tier."""
    from scikit_rec_agent.tools.datasets import TOOL_CREATE_DATASETS

    TOOL_CREATE_DATASETS.fn(
        bundle_id="b",
        interactions_path=binary_reward_paths["interactions"],
        users_path=binary_reward_paths["users"],
        items_path=binary_reward_paths["items"],
        session=session,
    )
    bundle = session.loaded_datasets["b"]
    profile = _profile_bundle(bundle)

    methods = list(_AUTO_SWEEPS["long_interactions"])
    kept, dropped = _data_aware_methods(methods, profile)
    kept_names = {m["short_name"] for m in kept}
    dropped_names = {d["method"]["short_name"] for d in dropped}

    # XGBoost survives every profile
    assert any("xgb" in n for n in kept_names)
    # MF dropped because dense (~33% density, way under 0.95 sparsity floor)
    assert any("mf" in n for n in dropped_names)


def test_sweep_auto_uses_data_aware_path(split_bundle):
    """methods='auto' on a long_interactions bundle MUST report data-aware
    filtering in the response payload — not just status==ok with no
    evidence of filtering. The split_bundle fixture is binary_reward
    (5000 users × 3 items × 1 row each → tiny tier, sparsity ≈ 0.33),
    which is a known-poor fit for embedding methods. We therefore expect:

      - status == ok (the filter doesn't error on this shape — XGBoost
        survives even at tiny scale)
      - n_dropped_by_data_profile >= 1 (at least one embedding family
        gets filtered out for being below the n_rows or sparsity floor)
      - dropped_by_data_profile rows include a `reason` string

    Without these stronger assertions the test could pass even if the
    data-aware path was bypassed entirely.
    """
    session = split_bundle
    result = TOOL_SWEEP_METHODS.fn(
        bundle_id="b",
        methods="auto",
        metrics=["NDCG_at_k"],
        primary_metric="NDCG_at_k",
        eval_top_k=5,
        session=session,
    )
    assert result["status"] == "ok", result
    data = result["data"]
    assert "n_dropped_by_data_profile" in data
    assert data["n_dropped_by_data_profile"] >= 1, (
        f"Expected the data-aware filter to drop at least one method on the "
        f"binary_reward fixture; got n_dropped_by_data_profile="
        f"{data['n_dropped_by_data_profile']}. Either the filter isn't running "
        f"or its rules no longer match this fixture's shape."
    )
    # Each dropped entry carries a reason explaining which rule fired.
    for drop in data["dropped_by_data_profile"]:
        assert drop.get("reason"), f"data-profile drop missing reason: {drop}"


def test_normalise_method_fills_defaults():
    out = _normalise_method({"recommender_type": "ranking", "scorer_type": "universal"})
    assert out["estimator_type"] == "tabular"
    assert "estimator_config" in out
    assert "short_name" in out
