"""Tests for train_model tool — uses a tabular ranking config against the
sample binary-reward fixture."""

from __future__ import annotations

from scikit_rec_agent.tools.datasets import TOOL_CREATE_DATASETS
from scikit_rec_agent.tools.training import TOOL_TRAIN_MODEL

_TABULAR_CONFIG = {
    "recommender_type": "ranking",
    "scorer_type": "universal",
    "estimator_config": {
        "ml_task": "classification",
        "xgboost": {"n_estimators": 20, "max_depth": 3},
    },
}


def test_train_model_from_bundle_id(binary_reward_paths, session):
    TOOL_CREATE_DATASETS.fn(
        bundle_id="b",
        interactions_path=binary_reward_paths["interactions"],
        users_path=binary_reward_paths["users"],
        items_path=binary_reward_paths["items"],
        session=session,
    )
    result = TOOL_TRAIN_MODEL.fn(
        model_name="tab_small",
        config=_TABULAR_CONFIG,
        bundle_id="b",
        session=session,
    )
    assert result["status"] == "ok"
    model_id = result["data"]["model_id"]
    assert model_id in session.trained_models
    handle = session.trained_models[model_id]
    assert handle.config["recommender_type"] == "ranking"
    assert handle.training_time_seconds >= 0


def test_train_model_factory_error_returns_envelope(binary_reward_paths, session):
    TOOL_CREATE_DATASETS.fn(
        bundle_id="b",
        interactions_path=binary_reward_paths["interactions"],
        users_path=binary_reward_paths["users"],
        items_path=binary_reward_paths["items"],
        session=session,
    )
    bad_config = {
        "recommender_type": "sequential",
        "scorer_type": "universal",  # mismatch — must be "sequential" for seq recommender
        "estimator_config": {"ml_task": "classification"},
    }
    result = TOOL_TRAIN_MODEL.fn(model_name="bad", config=bad_config, bundle_id="b", session=session)
    assert result["status"] == "error"
    assert "scorer_type" in result["message"] or "sequential" in result["message"]


def test_train_model_missing_bundle_returns_error(session):
    result = TOOL_TRAIN_MODEL.fn(model_name="x", config=_TABULAR_CONFIG, bundle_id="missing", session=session)
    assert result["status"] == "error"
    assert result["error_type"] == "BundleNotFound"


def test_train_model_scorer_config_defers_when_capability_key_missing(binary_reward_paths, session, monkeypatch):
    """when capability_matrix() exists but predates the
    scorer_config_keys entry (older skrec wheel), the agent must defer to
    the factory rather than reject every key against an empty whitelist.
    Without this guard, a valid on_degenerate_target='constant' on
    multioutput would false-reject under InvalidScorerConfigKey.

    Stubs capability_matrix to return a dict without 'scorer_config_keys'
    and verifies the agent proceeds to training (where the factory itself
    validates the keys).
    """
    from skrec.orchestrator import factory as _factory

    # Fake old skrec: capability_matrix exists but no scorer_config_keys.
    def _stub_cm():
        return {
            "recommender_types": ("ranking",),
            "scorer_types": ("universal", "multioutput"),
        }

    monkeypatch.setattr(_factory, "capability_matrix", _stub_cm)

    TOOL_CREATE_DATASETS.fn(
        bundle_id="b_b7",
        interactions_path=binary_reward_paths["interactions"],
        users_path=binary_reward_paths["users"],
        items_path=binary_reward_paths["items"],
        session=session,
    )
    # Pass a legitimately-named key. With the old guard logic this would
    # false-reject (empty allowlist + non-empty merged = unknown keys).
    # With the new guard, validation is skipped and the factory accepts it.
    result = TOOL_TRAIN_MODEL.fn(
        model_name="b7_defer",
        config=_TABULAR_CONFIG,
        bundle_id="b_b7",
        scorer_config={"on_degenerate_target": "raise"},
        session=session,
    )
    # Either training succeeds (factory passed the kwarg through) or it
    # surfaces a non-InvalidScorerConfigKey error. The defining property
    # is that the agent didn't reject upfront on an empty whitelist.
    assert result["status"] == "ok" or result.get("error_type") != "InvalidScorerConfigKey", result


def test_train_model_rejects_unsupported_scorer_config_key(binary_reward_paths, session):
    """train_model validates scorer_config keys against the upstream
    capability_matrix BEFORE the factory call so the LLM gets a clear
    InvalidScorerConfigKey envelope. Universal scorer accepts no
    scorer_config keys today; pass one and we must reject upfront.
    """
    TOOL_CREATE_DATASETS.fn(
        bundle_id="b",
        interactions_path=binary_reward_paths["interactions"],
        users_path=binary_reward_paths["users"],
        items_path=binary_reward_paths["items"],
        session=session,
    )
    result = TOOL_TRAIN_MODEL.fn(
        model_name="bad_sc",
        config=_TABULAR_CONFIG,  # universal scorer
        bundle_id="b",
        scorer_config={"on_degenerate_target": "constant"},
        session=session,
    )
    assert result["status"] == "error"
    assert result["error_type"] == "InvalidScorerConfigKey"
    assert result["category"] == "invalid_scorer_config_key"
    assert "on_degenerate_target" in result["message"]


def test_train_model_cleans_up_implicit_bundle_on_failure(binary_reward_paths, session):
    """when config=None + raw paths register an
    ``implicit_bundle_<ms>`` and a later validation fails (factory raise,
    InvalidScorerConfigKey, etc.), the implicit bundle must be popped.
    Without cleanup, repeated failing calls leak one DataFrame-laden
    bundle per call into session.loaded_datasets.

    Force a factory raise via a bad scorer_type/estimator combination
    that survives the early validation gates but fails at
    create_recommender_pipeline. Verify (a) the call returns an error
    envelope and (b) no implicit_bundle_<ms> entries remain in
    session.loaded_datasets.
    """
    before_keys = set(session.loaded_datasets.keys())
    bad_config = {
        "recommender_type": "sequential",
        "scorer_type": "universal",  # mismatch — sequential needs scorer_type='sequential'
        "estimator_config": {"ml_task": "classification"},
    }
    result = TOOL_TRAIN_MODEL.fn(
        model_name="g2_fail",
        config=bad_config,
        # No bundle_id; raw paths → triggers implicit bundle creation
        interactions_path=binary_reward_paths["interactions"],
        users_path=binary_reward_paths["users"],
        items_path=binary_reward_paths["items"],
        session=session,
    )
    assert result["status"] == "error"

    # The implicit_bundle_<ms> created during this call must have been
    # popped on the failure path.
    after_keys = set(session.loaded_datasets.keys())
    new_keys = after_keys - before_keys
    leaked = [k for k in new_keys if k.startswith("implicit_bundle_")]
    assert not leaked, f"implicit bundle(s) leaked into session: {leaked}"


def test_train_model_preserves_caller_provided_bundle_on_failure(binary_reward_paths, session):
    """when the caller passes an explicit bundle_id and
    training fails, the bundle must NOT be popped — the caller owns its
    lifecycle. Without this safety, the cleanup would have over-corrected
    and removed bundles the caller still expects to be there.
    """
    TOOL_CREATE_DATASETS.fn(
        bundle_id="caller_owned",
        interactions_path=binary_reward_paths["interactions"],
        users_path=binary_reward_paths["users"],
        items_path=binary_reward_paths["items"],
        session=session,
    )
    assert "caller_owned" in session.loaded_datasets

    bad_config = {
        "recommender_type": "sequential",
        "scorer_type": "universal",
        "estimator_config": {"ml_task": "classification"},
    }
    result = TOOL_TRAIN_MODEL.fn(
        model_name="g2_fail2",
        config=bad_config,
        bundle_id="caller_owned",
        session=session,
    )
    assert result["status"] == "error"
    # The caller-provided bundle must still be there
    assert "caller_owned" in session.loaded_datasets


def test_train_model_default_config_works_with_raw_paths(binary_reward_paths, session):
    """when config=None AND only raw paths are supplied (no bundle_id),
    train_model must build the bundle internally first, then read the
    contract from it. Previously errored upfront with 'register the bundle
    first via create_datasets'.
    """
    result = TOOL_TRAIN_MODEL.fn(
        model_name="rawpath_default",
        bundle_id=None,
        interactions_path=binary_reward_paths["interactions"],
        users_path=binary_reward_paths["users"],
        items_path=binary_reward_paths["items"],
        session=session,
        # config deliberately omitted — exercises the default-config branch
    )
    assert result["status"] == "ok", result
    data = result["data"]
    # default_method_applied surfaces the chosen short_name so the auditor
    # can see which curated default the auto-pick chose.
    assert "default_method_applied" in data


def test_train_model_threads_scorer_config_through_factory(tmp_path, session):
    """scorer_config={'on_degenerate_target': 'constant'} on a
    multioutput config reaches the underlying scorer and the
    degenerate_targets manifest surfaces in the train_model envelope.
    Built on a tiny wide-multioutput frame where one target is single-class
    in the training slice.
    """
    import pandas as pd

    # Two ITEM_* columns: ITEM_a has both classes; ITEM_b is all 0s
    # (degenerate). CONSTANT policy should let the fit succeed and emit
    # a per-target constant for ITEM_b.
    df = pd.DataFrame(
        {
            "USER_ID": [f"u{i}" for i in range(40)],
            "ITEM_a": [i % 2 for i in range(40)],
            "ITEM_b": [0] * 40,
            "feat1": list(range(40)),
        }
    )
    p = tmp_path / "wide.csv"
    df.to_csv(p, index=False)

    cd = TOOL_CREATE_DATASETS.fn(bundle_id="wb", interactions_path=str(p), session=session)
    assert cd["status"] == "ok", cd

    config = {
        "recommender_type": "ranking",
        "scorer_type": "multioutput",
        "estimator_config": {
            "ml_task": "classification",
            "xgboost": {"n_estimators": 10, "max_depth": 2},
        },
    }
    result = TOOL_TRAIN_MODEL.fn(
        model_name="mout_const",
        config=config,
        bundle_id="wb",
        scorer_config={"on_degenerate_target": "constant"},
        session=session,
    )
    assert result["status"] == "ok", result
    data = result["data"]
    # scorer_config_applied surfaced so the auditor can see the choice
    assert data.get("scorer_config_applied") == {"on_degenerate_target": "constant"}
    # And ITEM_b is recorded in the degenerate_targets manifest (the
    # whole point of CONSTANT mode is letting the fit succeed AND surfacing
    # which targets fell back to a constant predictor).
    assert "degenerate_targets" in data
    assert "ITEM_b" in data["degenerate_targets"]
