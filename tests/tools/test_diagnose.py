"""Tests for diagnose_training_failure + enriched train_model envelopes."""

from __future__ import annotations

import pandas as pd

from scikit_rec_agent.session import FailureRecord, Session
from scikit_rec_agent.tools.datasets import TOOL_CREATE_DATASETS
from scikit_rec_agent.tools.diagnose import (
    TOOL_DIAGNOSE_TRAINING_FAILURE,
    Diagnosis,
    Fix,
    _apply_fix,
    _match,
    _quick_diagnose,
    record_failure,
)
from scikit_rec_agent.tools.training import TOOL_TRAIN_MODEL


# ---------------------------------------------------------------------------
# Pattern matcher: one test per registered failure class
# ---------------------------------------------------------------------------


def test_match_single_class_target():
    env = {"error_type": "ValueError", "message": "y contains only one class"}
    d = _match(env)
    assert d.category == "single_class_target"


def test_match_nan_in_features():
    env = {"error_type": "ValueError", "message": "Input contains NaN."}
    d = _match(env)
    assert d.category == "nan_in_features"


def test_match_schema_mismatch_user_id():
    env = {"error_type": "KeyError", "message": "column not found: USER_ID"}
    d = _match(env)
    assert d.category == "schema_mismatch"


def test_match_wide_multioutput_underspecified():
    env = {"error_type": "ValueError", "message": "MultioutputScorer requires at least 2 ITEM_* target columns"}
    d = _match(env)
    assert d.category == "wide_multioutput_underspecified"


def test_match_capability_mismatch_embedding():
    env = {"error_type": "TypeError", "message": "Scorer type 'multioutput' does not support BaseEmbeddingEstimator"}
    d = _match(env)
    assert d.category == "capability_mismatch"


def test_match_oom():
    env = {"error_type": "MemoryError", "message": "Unable to allocate 8.0 GiB for an array"}
    d = _match(env)
    assert d.category == "oom"


def test_match_numerical_instability():
    env = {"error_type": "RuntimeError", "message": "SVD did not converge"}
    d = _match(env)
    assert d.category == "numerical_instability"


def test_match_missing_dependency():
    env = {"error_type": "ModuleNotFoundError", "message": "No module named 'torch'"}
    d = _match(env)
    assert d.category == "missing_dependency"


def test_match_object_dtype_features():
    env = {
        "error_type": "ValueError",
        "message": (
            "DataFrame.dtypes for data must be int, float, bool or category. "
            "Invalid columns:wholesale: object, acct_attached: object"
        ),
    }
    d = _match(env)
    assert d.category == "object_dtype_features"
    # Fix is non-auto-retryable — user judgment needed
    assert d.fixes
    assert all(not f.auto_retryable for f in d.fixes)


def test_match_wide_bundle_eval_unsupported():
    env = {
        "error_type": "WideBundleEvalUnsupported",
        "message": "can't auto-build eval_kwargs on interaction_multioutput bundle",
    }
    d = _match(env)
    assert d.category == "wide_bundle_eval_unsupported"
    assert all(not f.auto_retryable for f in d.fixes)


def test_match_wide_bundle_eval_keyerror_message():
    """Same category should match the bare KeyError shape too — for cases
    where the structured envelope wasn't produced (older evaluator paths
    or third-party wrappers)."""
    env = {
        "error_type": "KeyError",
        "message": "Column(s) ['ITEM_ID', 'OUTCOME'] do not exist",
    }
    d = _match(env)
    assert d.category == "wide_bundle_eval_unsupported"


def test_match_dtype_mismatch_across_files():
    env = {
        "error_type": "TypeError",
        "message": "evaluate failed for NDCG_at_k@10: '<' not supported between instances of 'str' and 'int'",
    }
    d = _match(env)
    assert d.category == "dtype_mismatch_across_files"
    assert d.fixes
    # User judgment needed (recast across files); not auto-retryable
    assert all(not f.auto_retryable for f in d.fixes)


def test_match_evaluator_needs_eval_kwargs():
    env = {
        "error_type": "ValueError",
        "message": "evaluate failed for NDCG_at_k@10: eval_kwargs is required to compute modified rewards. Provide logged_items, logged_rewards, and any other required arguments.",
    }
    d = _match(env)
    assert d.category == "evaluator_needs_eval_kwargs"
    # First fix is auto-retryable (swap to 'simple' evaluator)
    assert d.fixes
    assert d.fixes[0].auto_retryable is True


def test_match_logged_items_shape_mismatch():
    env = {
        "error_type": "ValueError",
        "message": "evaluate failed for NDCG_at_k@10: Mismatch in N dimension: target_proba (9209) vs logged_items (88)",
    }
    d = _match(env)
    assert d.category == "logged_items_shape_mismatch"


def test_match_data_shape_mismatch():
    env = {"error_type": "ValueError", "message": "Found input variables with inconsistent numbers of samples: [10, 8]"}
    d = _match(env)
    assert d.category == "data_shape_mismatch"


def test_match_degenerate_target():
    env = {"error_type": "ValueError", "message": "no positive samples in training set"}
    d = _match(env)
    assert d.category == "degenerate_target_or_features"


def test_match_timeout():
    env = {"error_type": "TimeoutError", "message": "Training exceeded max_train_seconds=10"}
    d = _match(env)
    assert d.category == "timeout"


def test_match_unknown_falls_through():
    env = {"error_type": "RuntimeError", "message": "an entirely novel error mode"}
    d = _match(env)
    assert d.category == "unknown"
    assert d.fixes == []


# ---------------------------------------------------------------------------
# _quick_diagnose smoke
# ---------------------------------------------------------------------------


def test_quick_diagnose_from_exception():
    try:
        raise ValueError("Input contains NaN.")
    except ValueError as e:
        d = _quick_diagnose(e)
    assert d.category == "nan_in_features"
    assert d.first_fix_description is not None


# ---------------------------------------------------------------------------
# Tool: no-failure case
# ---------------------------------------------------------------------------


def test_tool_returns_error_when_no_failure_history(session):
    res = TOOL_DIAGNOSE_TRAINING_FAILURE.fn(model_name="never_failed", session=session)
    assert res["status"] == "error"
    assert res["error_type"] == "NoFailureFound"


# ---------------------------------------------------------------------------
# Tool: unknown category surfaces raw message, no auto-retry
# ---------------------------------------------------------------------------


def test_unknown_category_surfaces_raw_and_no_auto_retry(session):
    env = {"error_type": "RuntimeError", "message": "weird unmatched error"}
    diagnosis = _match(env)
    record_failure(
        session,
        model_name="m",
        config={"recommender_type": "ranking"},
        bundle_args={"bundle_id": "b"},
        envelope=env,
        diagnosis=diagnosis,
    )
    res = TOOL_DIAGNOSE_TRAINING_FAILURE.fn(
        model_name="m", auto_retry=True, session=session
    )
    assert res["status"] == "ok"
    assert res["data"]["category"] == "unknown"
    assert res["data"]["auto_retried"] is False
    assert "weird unmatched error" in " ".join(res["data"]["causes"])


# ---------------------------------------------------------------------------
# Tool: retry cap
# ---------------------------------------------------------------------------


def test_retry_cap_blocks_after_max_retries(session):
    env = {"error_type": "ValueError", "message": "Input contains NaN"}
    config = {
        "recommender_type": "ranking",
        "scorer_type": "universal",
        "estimator_config": {"ml_task": "classification", "xgboost": {"n_estimators": 1}},
    }
    # Simulate two retries already applied for this model_name.
    for _ in range(2):
        rec = record_failure(
            session, "m", config, {"bundle_id": "b"}, env, _match(env)
        )
        rec.fix_applied = {"description": "tried something", "action": {"type": "noop"}}
    # Now record a fresh failure with no fix_applied:
    record_failure(session, "m", config, {"bundle_id": "b"}, env, _match(env))
    res = TOOL_DIAGNOSE_TRAINING_FAILURE.fn(
        model_name="m", auto_retry=True, max_retries=2, session=session
    )
    assert res["status"] == "ok"
    assert res["data"]["auto_retried"] is False
    assert "max_retries" in res["data"].get("retry_blocked_reason", "")


# ---------------------------------------------------------------------------
# Tool: missing_dependency has no auto-retryable fix → blocks
# ---------------------------------------------------------------------------


def test_missing_dependency_no_auto_retryable(session):
    env = {"error_type": "ModuleNotFoundError", "message": "No module named 'torch'"}
    config = {"recommender_type": "ranking", "scorer_type": "universal"}
    record_failure(session, "m", config, {"bundle_id": "b"}, env, _match(env))
    res = TOOL_DIAGNOSE_TRAINING_FAILURE.fn(
        model_name="m", auto_retry=True, session=session
    )
    assert res["status"] == "ok"
    assert res["data"]["auto_retried"] is False
    blocked = res["data"].get("retry_blocked_reason", "")
    assert "auto-retryable" in blocked or "user approval" in blocked
    # User-facing fix is present in candidate_fixes
    descs = [f["description"] for f in res["data"]["candidate_fixes"]]
    assert any("install" in d.lower() for d in descs)


# ---------------------------------------------------------------------------
# _apply_fix: nested and flat path setting
# ---------------------------------------------------------------------------


def test_apply_fix_nested_path():
    fix = Fix(
        description="lower lr",
        action={"type": "modify_config", "set": {"estimator_config.xgboost.learning_rate": 0.01}},
        auto_retryable=True,
    )
    config = {"estimator_config": {"xgboost": {"learning_rate": 0.1}}}
    new = _apply_fix(fix, config)
    assert new["estimator_config"]["xgboost"]["learning_rate"] == 0.01
    # Original is not mutated
    assert config["estimator_config"]["xgboost"]["learning_rate"] == 0.1


def test_apply_fix_flat_path():
    fix = Fix(
        description="swap scorer",
        action={"type": "modify_config", "set": {"scorer_type": "universal"}},
        auto_retryable=True,
    )
    config = {"scorer_type": "multioutput"}
    new = _apply_fix(fix, config)
    assert new["scorer_type"] == "universal"


# ---------------------------------------------------------------------------
# train_model integration: failure records + enriched envelope
# ---------------------------------------------------------------------------


def test_train_model_records_failure_with_category(binary_reward_paths, session):
    TOOL_CREATE_DATASETS.fn(
        bundle_id="b",
        interactions_path=binary_reward_paths["interactions"],
        users_path=binary_reward_paths["users"],
        items_path=binary_reward_paths["items"],
        session=session,
    )
    bad_config = {
        "recommender_type": "sequential",
        "scorer_type": "universal",
        "estimator_config": {"ml_task": "classification"},
    }
    result = TOOL_TRAIN_MODEL.fn(
        model_name="bad", config=bad_config, bundle_id="b", session=session
    )
    assert result["status"] == "error"
    # Envelope carries the new fields
    assert "category" in result
    # Failure history populated
    assert len(session.failure_history) == 1
    rec = session.failure_history[0]
    assert rec.model_name == "bad"
    assert rec.diagnosis_category == result["category"]


# ---------------------------------------------------------------------------
# previously_attempted_fixes filters out repeats
# ---------------------------------------------------------------------------


def test_previously_attempted_fix_is_filtered(session):
    env = {"error_type": "ValueError", "message": "Input contains NaN"}
    config = {
        "recommender_type": "ranking",
        "scorer_type": "universal",
        "estimator_config": {"ml_task": "classification"},
    }
    # First failure: pretend a fix was applied
    rec1 = record_failure(session, "m", config, {"bundle_id": "b"}, env, _match(env))
    nan_fix = _match(env).fixes[0]
    rec1.fix_applied = {"description": nan_fix.description, "action": nan_fix.action}

    # Second failure with same pattern
    record_failure(session, "m", config, {"bundle_id": "b"}, env, _match(env))

    res = TOOL_DIAGNOSE_TRAINING_FAILURE.fn(
        model_name="m", auto_retry=False, session=session
    )
    assert res["status"] == "ok"
    desc_set = {f["description"] for f in res["data"]["candidate_fixes"]}
    assert nan_fix.description not in desc_set
