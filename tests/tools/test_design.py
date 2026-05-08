"""Tests for the hierarchical-flow tool list_compatible_options.

Each test creates a real bundle (long-with-timestamp or wide-multioutput),
walks the four-step picker, and asserts the menu shape + constraint
propagation.
"""

from __future__ import annotations

import pandas as pd
import pytest

from scikit_rec_agent.prompts._explanations import (
    EMBEDDING_MODEL_EXPLANATIONS,
    ESTIMATOR_TYPE_EXPLANATIONS,
    RECOMMENDER_EXPLANATIONS,
    SCORER_EXPLANATIONS,
    SEQUENTIAL_MODEL_EXPLANATIONS,
)
from scikit_rec_agent.tools.datasets import TOOL_CREATE_DATASETS
from scikit_rec_agent.tools.design import (
    TOOL_LIST_COMPATIBLE_OPTIONS,
    apply_overrides,
    suggest_search_space,
)

# ---------------------------------------------------------------------------
# Bundle fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def long_with_timestamp_bundle(tmp_path, session):
    """Long-format binary feedback with TIMESTAMP — the typical recsys shape."""
    df = pd.DataFrame(
        {
            "USER_ID": [f"u{i}" for i in range(20) for _ in range(3)],
            "ITEM_ID": [f"i{(j * 7 + i) % 10}" for i in range(20) for j in range(3)],
            "OUTCOME": [(i + j) % 2 for i in range(20) for j in range(3)],
            "TIMESTAMP": [str(i * 10 + j) for i in range(20) for j in range(3)],
        }
    )
    p = tmp_path / "long_ts.csv"
    df.to_csv(p, index=False)

    users = pd.DataFrame({"USER_ID": [f"u{i}" for i in range(20)], "age": list(range(20))})
    items = pd.DataFrame({"ITEM_ID": [f"i{i}" for i in range(10)], "category": [i % 3 for i in range(10)]})
    up = tmp_path / "users.csv"
    ip = tmp_path / "items.csv"
    users.to_csv(up, index=False)
    items.to_csv(ip, index=False)

    TOOL_CREATE_DATASETS.fn(
        bundle_id="b",
        interactions_path=str(p),
        users_path=str(up),
        items_path=str(ip),
        session=session,
    )
    return session


@pytest.fixture
def long_no_timestamp_bundle(tmp_path, session):
    """Long format WITHOUT TIMESTAMP — sequential recommenders should be filtered out."""
    df = pd.DataFrame(
        {
            "USER_ID": [f"u{i}" for i in range(20) for _ in range(3)],
            "ITEM_ID": [f"i{(j * 7 + i) % 10}" for i in range(20) for j in range(3)],
            "OUTCOME": [(i + j) % 2 for i in range(20) for j in range(3)],
        }
    )
    p = tmp_path / "long.csv"
    df.to_csv(p, index=False)
    TOOL_CREATE_DATASETS.fn(bundle_id="b", interactions_path=str(p), session=session)
    return session


# ---------------------------------------------------------------------------
# Step 1: recommender_type
# ---------------------------------------------------------------------------


def test_step1_returns_recommender_options_with_explanations(long_with_timestamp_bundle):
    session = long_with_timestamp_bundle
    result = TOOL_LIST_COMPATIBLE_OPTIONS.fn(bundle_id="b", current_choices={}, session=session)
    assert result["status"] == "ok"
    data = result["data"]
    assert data["next_dimension"] == "recommender_type"
    assert data["is_terminal"] is False
    assert len(data["options"]) >= 1

    # Each option carries the full explanation triple.
    for opt in data["options"]:
        assert opt["what_it_is"]
        assert opt["when_to_pick"]
        assert opt["tradeoff_vs_alternatives"]

    # Sequential should be in the menu (data has timestamps).
    values = {opt["value"] for opt in data["options"]}
    assert "sequential" in values
    assert "ranking" in values


def test_step1_drops_sequential_when_no_timestamps(long_no_timestamp_bundle):
    session = long_no_timestamp_bundle
    result = TOOL_LIST_COMPATIBLE_OPTIONS.fn(bundle_id="b", current_choices={}, session=session)
    assert result["status"] == "ok"
    values = {opt["value"] for opt in result["data"]["options"]}
    assert "sequential" not in values
    assert "hierarchical_sequential" not in values
    # Ranking should still be there.
    assert "ranking" in values


# ---------------------------------------------------------------------------
# Step 2: scorer_type
# ---------------------------------------------------------------------------


def test_step2_scorer_options_filtered_by_recommender(long_with_timestamp_bundle):
    session = long_with_timestamp_bundle
    result = TOOL_LIST_COMPATIBLE_OPTIONS.fn(
        bundle_id="b",
        current_choices={"recommender_type": "sequential"},
        session=session,
    )
    assert result["status"] == "ok"
    values = {opt["value"] for opt in result["data"]["options"]}
    # sequential recommender requires sequential scorer
    assert values == {"sequential"}, f"unexpected scorers for sequential: {values}"


def test_step2_ranking_offers_all_compatible_scorers(long_with_timestamp_bundle):
    session = long_with_timestamp_bundle
    result = TOOL_LIST_COMPATIBLE_OPTIONS.fn(
        bundle_id="b",
        current_choices={"recommender_type": "ranking"},
        session=session,
    )
    values = {opt["value"] for opt in result["data"]["options"]}
    # universal and independent always available; multioutput dropped (not wide); multiclass dropped (has OUTCOME)
    assert "universal" in values
    assert "independent" in values
    assert "multioutput" not in values  # data isn't wide_multioutput
    assert "multiclass" not in values  # data has OUTCOME


# ---------------------------------------------------------------------------
# Step 3: estimator_type
# ---------------------------------------------------------------------------


def test_step3_drops_embedding_below_5k_rows(long_with_timestamp_bundle):
    session = long_with_timestamp_bundle  # 60 rows total → tiny tier
    result = TOOL_LIST_COMPATIBLE_OPTIONS.fn(
        bundle_id="b",
        current_choices={"recommender_type": "ranking", "scorer_type": "universal"},
        session=session,
    )
    values = {opt["value"] for opt in result["data"]["options"]}
    # Tabular always survives, embedding gets the n_rows≥5K floor
    assert "tabular" in values
    assert "embedding" not in values


# ---------------------------------------------------------------------------
# Step 4: model_type
# ---------------------------------------------------------------------------


def test_step4_tabular_returns_xgboost_only(long_with_timestamp_bundle):
    session = long_with_timestamp_bundle
    result = TOOL_LIST_COMPATIBLE_OPTIONS.fn(
        bundle_id="b",
        current_choices={
            "recommender_type": "ranking",
            "scorer_type": "universal",
            "estimator_type": "tabular",
        },
        session=session,
    )
    values = [opt["value"] for opt in result["data"]["options"]]
    assert values == ["xgboost"]


def test_step4_sequential_filters_by_target_type(long_with_timestamp_bundle):
    session = long_with_timestamp_bundle  # binary OUTCOME
    result = TOOL_LIST_COMPATIBLE_OPTIONS.fn(
        bundle_id="b",
        current_choices={
            "recommender_type": "sequential",
            "scorer_type": "sequential",
            "estimator_type": "sequential",
        },
        session=session,
    )
    values = {opt["value"] for opt in result["data"]["options"]}
    # binary target → only classifier variants
    assert "sasrec_classifier" in values
    assert "hrnn_classifier" in values
    assert "sasrec_regressor" not in values
    assert "hrnn_regressor" not in values


# ---------------------------------------------------------------------------
# Step 5: terminal payload
# ---------------------------------------------------------------------------


def test_step5_terminal_payload_shape(long_with_timestamp_bundle):
    session = long_with_timestamp_bundle
    result = TOOL_LIST_COMPATIBLE_OPTIONS.fn(
        bundle_id="b",
        current_choices={
            "recommender_type": "ranking",
            "scorer_type": "universal",
            "estimator_type": "tabular",
            "model_type": "xgboost",
        },
        session=session,
    )
    assert result["status"] == "ok"
    data = result["data"]
    assert data["is_terminal"] is True
    assert data["next_dimension"] == "hyperparameters"
    assert data["default_params"]
    # default_params entries are {value, what_it_is, why_this_default}
    for name, entry in data["default_params"].items():
        assert "value" in entry
        assert "why_this_default" in entry

    # next_action_options exposes train_with_defaults
    actions = {opt["action"] for opt in data["next_action_options"]}
    assert "train_with_defaults" in actions
    assert "train_with_overrides" in actions
    assert "run_hpo" in actions

    # assembled_config is a real RecommenderConfig dict, not the sweep wrapper
    cfg = data["assembled_config"]
    assert "recommender_type" in cfg
    assert "scorer_type" in cfg
    assert "estimator_config" in cfg
    assert "short_name" not in cfg


def test_step5_assembled_config_round_trips_through_factory(long_with_timestamp_bundle):
    """assembled_config from the terminal step must be accepted by
    create_recommender_pipeline without modification."""
    from skrec.orchestrator import create_recommender_pipeline

    session = long_with_timestamp_bundle
    result = TOOL_LIST_COMPATIBLE_OPTIONS.fn(
        bundle_id="b",
        current_choices={
            "recommender_type": "ranking",
            "scorer_type": "universal",
            "estimator_type": "tabular",
            "model_type": "xgboost",
        },
        session=session,
    )
    cfg = result["data"]["assembled_config"]
    # Should not raise.
    recommender = create_recommender_pipeline(cfg)
    assert recommender is not None


# ---------------------------------------------------------------------------
# apply_overrides
# ---------------------------------------------------------------------------


def test_apply_overrides_merges_into_xgboost_bucket():
    cfg = {
        "recommender_type": "ranking",
        "scorer_type": "universal",
        "estimator_config": {
            "ml_task": "classification",
            "xgboost": {"n_estimators": 50, "max_depth": 4, "learning_rate": 0.1},
        },
        "recommender_params": {},
    }
    out = apply_overrides(cfg, {"n_estimators": 200, "max_depth": 8})
    assert out["estimator_config"]["xgboost"]["n_estimators"] == 200
    assert out["estimator_config"]["xgboost"]["max_depth"] == 8
    assert out["estimator_config"]["xgboost"]["learning_rate"] == 0.1  # unchanged
    # Original is untouched.
    assert cfg["estimator_config"]["xgboost"]["n_estimators"] == 50


def test_apply_overrides_merges_into_embedding_params_bucket():
    cfg = {
        "recommender_type": "ranking",
        "scorer_type": "universal",
        "estimator_config": {
            "estimator_type": "embedding",
            "embedding": {
                "model_type": "matrix_factorization",
                "params": {"n_factors": 16, "epochs": 5},
            },
        },
        "recommender_params": {},
    }
    out = apply_overrides(cfg, {"n_factors": 64, "epochs": 20})
    assert out["estimator_config"]["embedding"]["params"]["n_factors"] == 64
    assert out["estimator_config"]["embedding"]["params"]["epochs"] == 20


def test_apply_overrides_unknown_key_lands_in_fallback_bucket():
    """If a user supplies an override for a key not currently in the config,
    it should land in the most likely params bucket so their intent isn't
    silently dropped."""
    cfg = {
        "recommender_type": "ranking",
        "scorer_type": "universal",
        "estimator_config": {
            "ml_task": "classification",
            "xgboost": {"n_estimators": 50},
        },
        "recommender_params": {},
    }
    out = apply_overrides(cfg, {"reg_alpha": 0.1})
    # Lands in xgboost bucket (the fallback for tabular)
    assert out["estimator_config"]["xgboost"]["reg_alpha"] == 0.1


def test_apply_overrides_mirrors_max_len_across_buckets():
    """Regression: SequentialRecommender uses ``max_len`` for data preparation
    (recommender_params.max_len) AND the underlying estimator uses it for
    padding/trimming (estimator_config.sequential.params.max_len). If only
    one bucket gets updated, the train-time and recommend-time lengths
    disagree silently. apply_overrides must mirror max_len into every
    bucket where it appears."""
    cfg = {
        "recommender_type": "sequential",
        "scorer_type": "sequential",
        "estimator_config": {
            "estimator_type": "sequential",
            "sequential": {
                "model_type": "sasrec_classifier",
                "params": {"max_len": 10, "hidden_units": 16},
            },
        },
        "recommender_params": {"max_len": 10},
    }
    out = apply_overrides(cfg, {"max_len": 50})
    assert out["estimator_config"]["sequential"]["params"]["max_len"] == 50
    assert out["recommender_params"]["max_len"] == 50, (
        "max_len override must propagate to recommender_params; otherwise the "
        "recommender prepares 10-len histories while the estimator expects 50."
    )


def test_apply_overrides_non_mirror_key_uses_first_match_wins():
    """Non-mirror keys (everything except max_len) keep the first-match-wins
    behaviour. hidden_units only lives in estimator_config.sequential.params,
    so an override for it lands there and nowhere else."""
    cfg = {
        "recommender_type": "sequential",
        "scorer_type": "sequential",
        "estimator_config": {
            "estimator_type": "sequential",
            "sequential": {
                "model_type": "sasrec_classifier",
                "params": {"max_len": 10, "hidden_units": 16},
            },
        },
        "recommender_params": {"max_len": 10},
    }
    out = apply_overrides(cfg, {"hidden_units": 64})
    assert out["estimator_config"]["sequential"]["params"]["hidden_units"] == 64


# ---------------------------------------------------------------------------
# suggest_search_space
# ---------------------------------------------------------------------------


def test_suggest_search_space_for_xgboost():
    method = {
        "estimator_config": {
            "ml_task": "classification",
            "xgboost": {"n_estimators": 50, "max_depth": 4},
        },
    }
    space = suggest_search_space(method, profile={"n_rows": 50_000})
    assert "n_estimators" in space
    assert "max_depth" in space
    assert "learning_rate" in space
    assert space["learning_rate"]["log_scale"] is True
    for entry in space.values():
        assert entry["rationale"]


def test_suggest_search_space_for_matrix_factorization():
    method = {
        "estimator_config": {
            "embedding": {"model_type": "matrix_factorization", "params": {"n_factors": 16}},
        },
    }
    space = suggest_search_space(method, profile={"n_rows": 50_000})
    assert "n_factors" in space
    assert "regularization" in space
    assert "epochs" in space


def test_suggest_search_space_for_ncf_includes_both_branches():
    method = {
        "estimator_config": {
            "embedding": {"model_type": "ncf", "params": {"gmf_embedding_dim": 8, "mlp_embedding_dim": 8}},
        },
    }
    space = suggest_search_space(method, profile={"n_rows": 50_000})
    assert "gmf_embedding_dim" in space
    assert "mlp_embedding_dim" in space
    assert "dropout" in space


def test_suggest_search_space_for_sasrec():
    method = {
        "estimator_config": {
            "sequential": {"model_type": "sasrec_classifier", "params": {"hidden_units": 16}},
        },
    }
    space = suggest_search_space(method, profile={"n_rows": 50_000})
    assert "hidden_units" in space
    assert "max_len" in space
    assert "learning_rate" in space


def test_suggest_search_space_returns_empty_for_unknown_estimator():
    method = {"estimator_config": {"some_other_thing": {}}}
    space = suggest_search_space(method, profile={"n_rows": 50_000})
    assert space == {}


# ---------------------------------------------------------------------------
# Terminal payload exposes search space and run_hpo availability
# ---------------------------------------------------------------------------


def test_terminal_payload_exposes_search_space_and_marks_run_hpo_available(long_with_timestamp_bundle):
    session = long_with_timestamp_bundle
    result = TOOL_LIST_COMPATIBLE_OPTIONS.fn(
        bundle_id="b",
        current_choices={
            "recommender_type": "ranking",
            "scorer_type": "universal",
            "estimator_type": "tabular",
            "model_type": "xgboost",
        },
        session=session,
    )
    assert result["status"] == "ok"
    data = result["data"]
    # The search space is present and non-empty for XGBoost.
    space = data["suggested_search_space"]
    assert space
    assert "n_estimators" in space

    # run_hpo action is now available=True.
    actions = {opt["action"]: opt for opt in data["next_action_options"]}
    assert actions["run_hpo"]["available"] is True
    assert actions["train_with_defaults"]["available"] is True
    assert actions["train_with_overrides"]["available"] is True


# ---------------------------------------------------------------------------
# gcsl filtered out + uplift recommender_params handling
# ---------------------------------------------------------------------------


def test_gcsl_dropped_from_recommender_menu(long_with_timestamp_bundle):
    """gcsl is intentionally hidden until the agent can capture its
    inference_method config — picking it would otherwise lead to a
    factory error at train time."""
    session = long_with_timestamp_bundle
    result = TOOL_LIST_COMPATIBLE_OPTIONS.fn(bundle_id="b", current_choices={}, session=session)
    values = {opt["value"] for opt in result["data"]["options"]}
    assert "gcsl" not in values


def test_uplift_appears_in_recommender_menu(long_with_timestamp_bundle):
    session = long_with_timestamp_bundle
    result = TOOL_LIST_COMPATIBLE_OPTIONS.fn(bundle_id="b", current_choices={}, session=session)
    values = {opt["value"] for opt in result["data"]["options"]}
    assert "uplift" in values


def test_uplift_terminal_payload_exposes_required_recommender_params(long_with_timestamp_bundle):
    """Walking to the terminal step under uplift must surface
    control_item_id and mode with the same explanation shape used by
    other dimensions."""
    session = long_with_timestamp_bundle
    result = TOOL_LIST_COMPATIBLE_OPTIONS.fn(
        bundle_id="b",
        current_choices={
            "recommender_type": "uplift",
            "scorer_type": "independent",
            "estimator_type": "tabular",
            "model_type": "xgboost",
        },
        session=session,
    )
    assert result["status"] == "ok"
    data = result["data"]
    rec_params = data["required_recommender_params"]
    assert "control_item_id" in rec_params
    assert "mode" in rec_params

    # control_item_id is a single required value the user supplies.
    cid = rec_params["control_item_id"]
    assert cid["user_must_supply"] is True
    assert cid["what_it_is"]
    assert cid["why_required"]
    assert cid["hint_to_user"]

    # mode has a sub-menu with the three learner variants, each with a triple.
    mode = rec_params["mode"]
    assert mode["user_must_supply"] is True
    assert len(mode["options"]) == 3
    learner_values = {o["value"] for o in mode["options"]}
    assert learner_values == {"t_learner", "s_learner", "x_learner"}
    for opt in mode["options"]:
        assert opt["what_it_is"]
        assert opt["when_to_pick"]
        assert opt["tradeoff_vs_alternatives"]


def test_uplift_assembled_config_pre_seeds_recommender_params(long_with_timestamp_bundle):
    session = long_with_timestamp_bundle
    result = TOOL_LIST_COMPATIBLE_OPTIONS.fn(
        bundle_id="b",
        current_choices={
            "recommender_type": "uplift",
            "scorer_type": "independent",
            "estimator_type": "tabular",
            "model_type": "xgboost",
        },
        session=session,
    )
    cfg = result["data"]["assembled_config"]
    assert cfg["recommender_params"]["control_item_id"] is None
    assert cfg["recommender_params"]["mode"] is None


def test_uplift_terminal_blocks_train_with_defaults(long_with_timestamp_bundle):
    """train_with_defaults should be unavailable when required_recommender_params
    can't be silently filled in."""
    session = long_with_timestamp_bundle
    result = TOOL_LIST_COMPATIBLE_OPTIONS.fn(
        bundle_id="b",
        current_choices={
            "recommender_type": "uplift",
            "scorer_type": "independent",
            "estimator_type": "tabular",
            "model_type": "xgboost",
        },
        session=session,
    )
    actions = {opt["action"]: opt for opt in result["data"]["next_action_options"]}
    assert actions["train_with_defaults"]["available"] is False
    assert "blocked_reason" in actions["train_with_defaults"]
    # train_with_overrides is the user's path
    assert actions["train_with_overrides"]["available"] is True


def test_apply_overrides_fills_uplift_recommender_params(long_with_timestamp_bundle):
    """End-to-end: terminal payload → apply_overrides → factory accepts it."""
    from skrec.orchestrator import create_recommender_pipeline

    from scikit_rec_agent.tools.design import apply_overrides

    session = long_with_timestamp_bundle
    result = TOOL_LIST_COMPATIBLE_OPTIONS.fn(
        bundle_id="b",
        current_choices={
            "recommender_type": "uplift",
            "scorer_type": "independent",
            "estimator_type": "tabular",
            "model_type": "xgboost",
        },
        session=session,
    )
    cfg = result["data"]["assembled_config"]
    completed = apply_overrides(cfg, {"control_item_id": "control", "mode": "t_learner"})
    assert completed["recommender_params"]["control_item_id"] == "control"
    assert completed["recommender_params"]["mode"] == "t_learner"
    # And the factory accepts it without raising.
    recommender = create_recommender_pipeline(completed)
    assert recommender is not None


# ---------------------------------------------------------------------------
# Bundle resolution + bad-input handling
# ---------------------------------------------------------------------------


def test_bundle_not_found(session):
    result = TOOL_LIST_COMPATIBLE_OPTIONS.fn(bundle_id="missing", session=session)
    assert result["status"] == "error"
    assert result["error_type"] == "BundleNotFound"


# ---------------------------------------------------------------------------
# Explanations registry completeness
# ---------------------------------------------------------------------------


def test_recommender_explanations_cover_all_scikit_rec_types():
    """If scikit-rec adds a new recommender_type, this test fails until the
    explanations registry gains a triple for it."""
    from skrec.orchestrator import RECOMMENDER_TYPES

    missing = set(RECOMMENDER_TYPES) - set(RECOMMENDER_EXPLANATIONS)
    assert not missing, f"recommender_type explanations missing: {missing}"


def test_scorer_explanations_cover_all_scikit_rec_types():
    from skrec.orchestrator import SCORER_TYPES

    missing = set(SCORER_TYPES) - set(SCORER_EXPLANATIONS)
    assert not missing, f"scorer_type explanations missing: {missing}"


def test_estimator_explanations_cover_all_scikit_rec_types():
    from skrec.orchestrator import ESTIMATOR_TYPES

    missing = set(ESTIMATOR_TYPES) - set(ESTIMATOR_TYPE_EXPLANATIONS)
    assert not missing, f"estimator_type explanations missing: {missing}"


def test_embedding_model_explanations_cover_all_scikit_rec_types():
    from skrec.orchestrator import capability_matrix

    available = capability_matrix()["embedding_model_types"]
    missing = set(available) - set(EMBEDDING_MODEL_EXPLANATIONS)
    assert not missing, f"embedding model_type explanations missing: {missing}"


def test_sequential_model_explanations_cover_all_scikit_rec_types():
    from skrec.orchestrator import capability_matrix

    available = capability_matrix()["sequential_model_types"]
    missing = set(available) - set(SEQUENTIAL_MODEL_EXPLANATIONS)
    assert not missing, f"sequential model_type explanations missing: {missing}"
