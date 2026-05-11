"""Hand-curated explanations for every value the hierarchical model-design
flow surfaces to the user.

Each dimension (recommender_type / scorer_type / estimator_type / model_type)
maps a value name to a triple:

  - what_it_is              : one-sentence definition.
  - when_to_pick            : data / goal signals that make this the right pick.
  - tradeoff_vs_alternatives: cost or limitation vs. other options at this step.

These are static reference text — they should NOT be invented per session by
the LLM. The hierarchical-flow tool (`list_compatible_options`) reads from
this registry and includes the triples in its tool result so the agent surfaces
the same explanation every run, regardless of which model is driving.

Adding a new value to scikit-rec without a matching entry here is a test
failure (see tests/tools/test_design.py).
"""

from __future__ import annotations

from typing import Any

# ---------------------------------------------------------------------------
# recommender_type
# ---------------------------------------------------------------------------

RECOMMENDER_EXPLANATIONS: dict[str, dict[str, str]] = {
    "ranking": {
        "what_it_is": (
            "Standard ranking recommender — score every item per user, return the top-K. "
            "The default starting point for most recsys problems."
        ),
        "when_to_pick": (
            "General implicit feedback (clicks, views, purchases) or explicit ratings. "
            "Pick this when there's a single objective and no causal / treatment structure."
        ),
        "tradeoff_vs_alternatives": (
            "No temporal awareness (sequential is better when order matters); single objective "
            "(gcsl is better for multi-target); no causal lift (uplift is better for treatment effects)."
        ),
    },
    "sequential": {
        "what_it_is": (
            "Predicts the next item given the user's recent interaction sequence — "
            "a transformer / RNN over per-user history."
        ),
        "when_to_pick": (
            "Data has timestamps and order matters: news feeds, music streams, browsing sessions. "
            "Especially strong with long per-user histories."
        ),
        "tradeoff_vs_alternatives": (
            "Requires TIMESTAMP and benefits from rich histories; new users with no prior "
            "interactions get cold-start treatment. Heavier to train than tabular ranking."
        ),
    },
    "hierarchical_sequential": {
        "what_it_is": (
            "Two-level sequential model: per-session sequences inside a per-user sequence-of-sessions. "
            "Used by HRNN family."
        ),
        "when_to_pick": (
            "Data has explicit session boundaries (e.g. news visits, music listening sessions) and "
            "intra-session vs cross-session patterns differ."
        ),
        "tradeoff_vs_alternatives": (
            "Requires sessionised data (USER_ID + SESSION_SEQUENCES); collapses to plain sequential "
            "if sessions are 1-deep. More expressive but more data-hungry."
        ),
    },
    "uplift": {
        "what_it_is": (
            "Causal recommender — estimates the *incremental* effect of recommending item A vs. a "
            "baseline / control. T-Learner, S-Learner, or X-Learner variants."
        ),
        "when_to_pick": (
            "You have logged treatment / control assignments and care about lift, not raw engagement. "
            "Common in marketing campaigns, promotions, A/B-tested rollouts."
        ),
        "tradeoff_vs_alternatives": (
            "Needs an explicit `control_item_id`; doesn't optimise for raw clicks. Strictly more "
            "complex than ranking — pick only when causal interpretation matters."
        ),
    },
    "gcsl": {
        "what_it_is": (
            "Goal-Conditioned Supervised Learning — multi-objective recommender that combines "
            "several reward signals (revenue, clicks, dwell-time, …) via configurable scalarisation."
        ),
        "when_to_pick": (
            "You have multiple correlated outcome columns (OUTCOME, OUTCOME_revenue, OUTCOME_clicks) "
            "and want one model to balance them."
        ),
        "tradeoff_vs_alternatives": (
            "Requires `inference_method` config (mean_scalarization / percentile_value / "
            "predefined_value). Less interpretable than separate per-objective rankers."
        ),
    },
    "bandits": {
        "what_it_is": (
            "Contextual bandit — balances exploration of new items with exploitation of known winners. "
            "On-policy by design."
        ),
        "when_to_pick": (
            "Active deployment with continual learning; cold-start items need exposure even if they "
            "look weak under the current policy."
        ),
        "tradeoff_vs_alternatives": (
            "Tuning the explore/exploit balance is non-trivial; offline evaluation under bandit "
            "logging needs IPS / DR (not 'simple') to be honest."
        ),
    },
}


# ---------------------------------------------------------------------------
# scorer_type
# ---------------------------------------------------------------------------

# Cross-cutting convertibility note shared across the long-format scorers
# (universal / independent) and called out from multioutput too. Pulled out
# so the wording stays consistent — drift between scorer descriptions is a
# real source of confusion when the agent walks `list_compatible_options`.
_LONG_FROM_WIDE_CONVERTIBILITY = (
    "Convertibility note — wide ↔ long: this scorer consumes the long-format "
    "(USER_ID, ITEM_ID, OUTCOME) contract. If your source is wide multi-output "
    "(one row per user, multiple ITEM_* binary columns), call transform_data "
    "with target_contract='long_interactions' first; non-target columns ride "
    "along as features through the melt, so user features are preserved. The "
    "reverse direction (long → wide for multioutput) is also supported via "
    "target_contract='wide_multioutput', but the pivot fills missing "
    "(user, item) pairs with zeros — only safe when every user has every "
    "label observed (or when implicit zero is a valid 'didn't take it' "
    "signal, which is the typical multi-label classification setup)."
)

_WIDE_FROM_LONG_CONVERTIBILITY = (
    "Convertibility note — wide ↔ long: multioutput consumes the wide format "
    "(one row per user, multiple ITEM_* targets). If your source is long "
    "(USER_ID, ITEM_ID, OUTCOME), call transform_data with "
    "target_contract='wide_multioutput' to pivot — but this is only safe "
    "when every label is observed for every user (or when implicit zeros "
    "are meaningful 'didn't take it' values, which is the typical "
    "multi-label classification setup). The reverse direction (wide → long) "
    "is the easy one and feature-preserving; use it when you want to compare "
    "multioutput against universal / independent on the same data."
)


SCORER_EXPLANATIONS: dict[str, dict[str, str]] = {
    "universal": {
        "what_it_is": (
            "One model that takes (user, item) as input and returns a score. Handles tabular "
            "estimators (XGBoost) and embedding estimators (MF / NCF / Two-Tower / DCN / NFM)."
        ),
        "when_to_pick": (
            "Default for ranking. Works whenever you have a single OUTCOME column and want one "
            "model trained jointly over user × item interactions."
        ),
        "tradeoff_vs_alternatives": (
            "Trains one big model on all data — less interpretable than independent per-item models, "
            "but typically beats them on raw accuracy. " + _LONG_FROM_WIDE_CONVERTIBILITY
        ),
    },
    "independent": {
        "what_it_is": ("One separate model per item — items get scored without a shared user × item joint model."),
        "when_to_pick": (
            "Uplift modelling (separate models for treatment vs control items), GCSL multi-target "
            "(separate model per reward), or bandits where adding new items shouldn't trigger a full retrain."
        ),
        "tradeoff_vs_alternatives": (
            "Doesn't share user × item structure across items; tabular estimators only "
            "(rejects embedding estimators per scikit-rec's factory rules). " + _LONG_FROM_WIDE_CONVERTIBILITY
        ),
    },
    "multiclass": {
        "what_it_is": ("Single classifier where each item is a class and the user is the input. ITEM_ID is the label."),
        "when_to_pick": (
            "Wide multi-class data: one row per user, ITEM_ID names which class they fall into "
            "(category, segment, persona). Small finite item space (≤ ~100s)."
        ),
        "tradeoff_vs_alternatives": (
            "Doesn't scale to large item catalogues; the OUTCOME column is forbidden in this contract."
        ),
    },
    "multioutput": {
        "what_it_is": (
            "Single model that emits one score per target — vector-valued output where each output head is one ITEM_*. "
            "Two modes: classifier (binary ITEM_* targets — values strictly in {0, 1}) and regressor (continuous "
            "ITEM_* targets, e.g. dollar amounts or counts). Mode is picked by ``estimator_config.ml_task`` "
            "('classification' or 'regression') and the matching xgb estimator is wired in. The auto-sweep "
            "lists both — the contract+profile picks the right one based on the ITEM_* dtypes / uniques. "
            "When you call ``recommender.evaluate(per_label=True)`` it returns Dict[str, float] keyed by "
            "ITEM_* name (one entry per fit-time target, NaN for any target the metric is undefined on)."
        ),
        "when_to_pick": (
            "Wide multi-output / multi-label data: one row per user with several action columns "
            "(label_X / ITEM_X) you want to predict jointly. Use classifier mode for binary actions, "
            "regressor mode for continuous quantities. Don't pair ranking metrics with per_label — "
            "ranking aggregates across all targets per user, not per target."
        ),
        "tradeoff_vs_alternatives": (
            "Tabular estimators only (rejects embedding estimators); requires ≥2 ITEM_* target columns. "
            "User features are supported but must live as additional columns inside the interactions "
            "DataFrame alongside USER_ID + ITEM_*; a separate users_path is not consumed (create_datasets "
            "auto-merges on USER_ID for you). Item-level features have no place in the wide layout — "
            "items are encoded as columns, not rows — so melt to long_interactions if you need them. "
            "Retriever is rejected (there's no row-level ITEM_ID to retrieve over). Evaluation rejects "
            "an active item_subset (catalogue narrowing on a per-target scorer is ill-defined). "
            "Single-class targets in the train slice raise under the default RAISE policy — pass "
            "``scorer_config={'on_degenerate_target': 'constant'}`` to fall back to a constant "
            "predictor, or let transform_data auto-drop them. " + _WIDE_FROM_LONG_CONVERTIBILITY
        ),
    },
    "sequential": {
        "what_it_is": (
            "Sequence-aware scorer — single forward pass over all candidate items per user, "
            "conditioned on the user's encoded sequence."
        ),
        "when_to_pick": (
            "Used with `recommender_type='sequential'` (SASRec, HRNN). Default for sequence-aware models."
        ),
        "tradeoff_vs_alternatives": (
            "Only paired with sequential estimators; rejects tabular and plain embedding estimators."
        ),
    },
    "hierarchical": {
        "what_it_is": ("Two-level scorer: session-level then item-level. Used by HRNN."),
        "when_to_pick": ("Used with `recommender_type='hierarchical_sequential'` on session-shaped data."),
        "tradeoff_vs_alternatives": (
            "Requires sessionised input (SESSION_SEQUENCES column); only paired with sequential estimators."
        ),
    },
}


# ---------------------------------------------------------------------------
# estimator_type
# ---------------------------------------------------------------------------

ESTIMATOR_TYPE_EXPLANATIONS: dict[str, dict[str, str]] = {
    "tabular": {
        "what_it_is": (
            "Tree-based estimator (XGBoost). Trains on flat (user_features × item_features × outcome) rows."
        ),
        "when_to_pick": (
            "Default for almost any data size. Strong baseline that almost always beats embedding-based "
            "models on <100K interactions."
        ),
        "tradeoff_vs_alternatives": (
            "No item embeddings — won't generalise to unseen items as well as embedding methods. "
            "Capable but not the SOTA family for very large catalogues."
        ),
    },
    "embedding": {
        "what_it_is": (
            "Neural / matrix-factorisation models that learn dense user and item embeddings. "
            "Includes MF, NCF, Two-Tower, DCN, NFM."
        ),
        "when_to_pick": (
            "≥5K interactions. Sparse implicit feedback where collaborative signal helps. Cold-start-ish "
            "scenarios where item / user embeddings can be transferred."
        ),
        "tradeoff_vs_alternatives": (
            "Slower to train than tabular; needs more data to beat XGBoost. Requires pre-computed user "
            "embeddings at score time (the agent handles this transparently)."
        ),
    },
    "sequential": {
        "what_it_is": (
            "Sequence models — transformers (SASRec) and RNNs (HRNN). Each user's history is a sequence; "
            "the model predicts the next item."
        ),
        "when_to_pick": ("Data has TIMESTAMP and per-user histories of meaningful length (>5 events typical)."),
        "tradeoff_vs_alternatives": (
            "Requires TIMESTAMP; cold-start users have no history to encode. Heavier compute than "
            "tabular or plain embedding."
        ),
    },
}


# ---------------------------------------------------------------------------
# model_type — split by estimator_type because the values come from
# different scikit-rec maps.
# ---------------------------------------------------------------------------

TABULAR_MODEL_EXPLANATIONS: dict[str, dict[str, str]] = {
    "xgboost": {
        "what_it_is": (
            "XGBoost gradient-boosted trees. ml_task is auto-derived from your target "
            "(classification for binary / categorical, regression for continuous)."
        ),
        "when_to_pick": "Default tabular pick — always available, fast to train, strong baseline.",
        "tradeoff_vs_alternatives": (
            "Generally faster to train than LightGBM on smaller datasets; slightly heavier memory "
            "footprint. Strong default when LightGBM behaviour is untested on the data."
        ),
    },
    "lightgbm": {
        "what_it_is": (
            "LightGBM gradient-boosted trees. Leaf-wise growth with histogram binning. "
            "ml_task is auto-derived the same way as XGBoost."
        ),
        "when_to_pick": (
            "Large tabular datasets (>100K rows) or high-cardinality categoricals where LightGBM's "
            "leaf-wise splits and native categorical handling give a speed/accuracy edge over XGBoost."
        ),
        "tradeoff_vs_alternatives": (
            "Often faster than XGBoost on large data; more prone to overfitting on small datasets "
            "without careful num_leaves / min_child_samples tuning. Requires lightgbm>=4.6.0 "
            "(already a scikit-rec core dependency)."
        ),
    },
    "deepfm": {
        "what_it_is": (
            "DeepFM classifier — factorisation machine + DNN over flat tabular features. "
            "Tabular input shape, PyTorch training. Classification only."
        ),
        "when_to_pick": (
            "Large datasets with many sparse or interaction-heavy features where explicit "
            "second-order feature interactions help (e.g. rich side-feature catalogs with "
            "user × item cross terms)."
        ),
        "tradeoff_vs_alternatives": (
            "Requires scikit-rec[torch]. Only supports ml_task='classification'. "
            "Slower to train than tree methods; gains come from automatic feature-interaction "
            "learning without manual feature engineering. Needs more data than XGBoost to pay off."
        ),
    },
}

EMBEDDING_MODEL_EXPLANATIONS: dict[str, dict[str, str]] = {
    "matrix_factorization": {
        "what_it_is": (
            "Classic matrix factorisation via ALS (closed-form alternating ridge solves) or "
            "per-sample SGD. Pure numpy implementation — no torch dependency."
        ),
        "when_to_pick": (
            "Sparse implicit-feedback baseline. Robust on >95% sparsity with ≥1K interactions. "
            "Fastest of the embedding family to train."
        ),
        "tradeoff_vs_alternatives": (
            "Doesn't use side features (only USER_ID × ITEM_ID); other embedding methods exploit features. "
            "Does NOT take a `batch_size` parameter — ALS has no notion of mini-batches by "
            "construction (each step solves a per-user / per-item ridge in closed form over all "
            "observed entries), and the SGD variant is per-sample. Don't pattern-match a "
            "`batch_size` onto it from neural-net configs."
        ),
    },
    "ncf": {
        "what_it_is": (
            "Neural Collaborative Filtering — learns GMF + MLP branches with neural interaction layers. Torch-based."
        ),
        "when_to_pick": ("≥5K interactions; want non-linear interaction modelling beyond plain MF dot-product."),
        "tradeoff_vs_alternatives": (
            "More parameters to tune than MF (gmf_embedding_dim, mlp_embedding_dim, mlp_layers); "
            "needs GPU for large catalogues."
        ),
    },
    "two_tower": {
        "what_it_is": (
            "Two-Tower with separate user and item towers, dot-product (or trilinear) composition. Torch-based."
        ),
        "when_to_pick": (
            "You have user features AND item features; want fast retrieval (item embeddings precomputed). "
            "Production-friendly."
        ),
        "tradeoff_vs_alternatives": (
            "Requires both user and item features. Adds context_mode tuning. Heavier than NCF for similar accuracy."
        ),
    },
    "deep_cross_network": {
        "what_it_is": (
            "Deep & Cross Network — explicit feature crosses via cross layers + a deep MLP branch. Torch-based."
        ),
        "when_to_pick": ("Rich side features where explicit interactions matter (CTR prediction, ad ranking)."),
        "tradeoff_vs_alternatives": (
            "Heavier than NCF / MF; benefits from many features. Without features, falls back to similar "
            "behaviour as NCF."
        ),
    },
    "neural_factorization": {
        "what_it_is": (
            "Neural Factorization Machine — bi-interaction pooling over feature embeddings + an MLP. Torch-based."
        ),
        "when_to_pick": (
            "Pairwise feature interactions matter and you want them learned end-to-end. Common in CTR-like settings."
        ),
        "tradeoff_vs_alternatives": (
            "Similar regime to DCN but uses bi-interaction instead of explicit cross layers; tradeoff is empirical."
        ),
    },
}

SEQUENTIAL_MODEL_EXPLANATIONS: dict[str, dict[str, str]] = {
    "sasrec_classifier": {
        "what_it_is": (
            "Self-Attentive Sequential Recommendation — a transformer over the user's interaction sequence. "
            "Classifier head for binary / categorical OUTCOME."
        ),
        "when_to_pick": (
            "Default sequential pick for clicks / purchases / next-item prediction. Strong on long sequences."
        ),
        "tradeoff_vs_alternatives": (
            "Heavier than HRNN; quadratic attention cost in sequence length. max_len caps the cost."
        ),
    },
    "sasrec_regressor": {
        "what_it_is": "SASRec with a regression head — predicts a continuous OUTCOME (rating, dwell-time).",
        "when_to_pick": ("Continuous-target sequential modelling: ratings, time-on-item, monetary outcomes."),
        "tradeoff_vs_alternatives": ("Same architectural cost as the classifier variant; pick based on target type."),
    },
    "hrnn_classifier": {
        "what_it_is": ("Hierarchical RNN — session-level RNN inside a user-level RNN. Classifier head."),
        "when_to_pick": (
            "Sessionised data where intra-session and cross-session patterns differ. "
            "Cheaper than transformer for moderate sequence lengths."
        ),
        "tradeoff_vs_alternatives": ("Requires sessionised input. Less expressive than SASRec on long flat sequences."),
    },
    "hrnn_regressor": {
        "what_it_is": "HRNN with a regression head — sessionised continuous-target prediction.",
        "when_to_pick": ("Sessionised data with continuous OUTCOME (rating-per-session, dwell-time)."),
        "tradeoff_vs_alternatives": ("Same as hrnn_classifier; pick based on target type."),
    },
}


# ---------------------------------------------------------------------------
# Recommender-specific recommender_params explanations
#
# When the user picks `recommender_type='uplift'` the hierarchical flow's
# terminal step exposes these in a `required_recommender_params` field so
# the agent surfaces them with the same explanatory shape as a normal
# dimension menu. The user picks values, the agent uses `apply_overrides`
# to merge them into the assembled_config's `recommender_params` bucket.
# ---------------------------------------------------------------------------

UPLIFT_PARAM_EXPLANATIONS: dict[str, dict[str, Any]] = {
    "control_item_id": {
        "what_it_is": ("The ITEM_ID value that represents the control / no-recommendation case in your data."),
        "why_required": (
            "Uplift estimates the *causal lift* of recommending an item vs. a baseline. The control "
            "item is that baseline — the model trains on rows assigned to it as the no-treatment arm "
            "and learns counterfactual scores for treated rows."
        ),
        "user_must_supply": True,
        "hint_to_user": (
            "Look at your ITEM_ID values. The control is often a sentinel like 'control', 'none', "
            "or 'baseline'. If you ran A/B-tested rollouts, the control bucket is whatever item you "
            "logged for users in the control arm."
        ),
    },
    "mode": {
        "what_it_is": "Which uplift learner variant to fit.",
        "why_required": (
            "Uplift modelling has multiple statistical estimators with materially different "
            "bias / variance tradeoffs. There's no single best default — the right pick depends "
            "on your data balance and how heterogeneous you expect treatment effects to be."
        ),
        "user_must_supply": True,
        "options": [
            {
                "value": "t_learner",
                "what_it_is": (
                    "Two separate models — one fit on treated rows, one on control rows. The "
                    "uplift for an instance is the difference of their predictions."
                ),
                "when_to_pick": (
                    "Most common default. Stable when both arms have plenty of data and the "
                    "treatment effect is roughly homogeneous across the feature space."
                ),
                "tradeoff_vs_alternatives": (
                    "Doubles the model count vs S-Learner; small-sample arms can fit poorly because "
                    "each model only sees one arm's worth of data."
                ),
            },
            {
                "value": "s_learner",
                "what_it_is": (
                    "A single shared model with the treatment indicator passed as an input feature. "
                    "Uplift is the difference of predictions with treatment=1 vs treatment=0."
                ),
                "when_to_pick": (
                    "When one arm is much smaller than the other — sharing parameters across arms "
                    "lets the small-arm signal benefit from the large-arm data."
                ),
                "tradeoff_vs_alternatives": (
                    "The treatment effect can wash out if the model under-weights the treatment "
                    "feature relative to other inputs."
                ),
            },
            {
                "value": "x_learner",
                "what_it_is": (
                    "Cross-fit two-stage estimator: train T-Learner first, derive imputed treatment "
                    "effects on the held-out arm, then fit a second-stage model that predicts the "
                    "uplift directly. Lower variance via cross-fitting."
                ),
                "when_to_pick": (
                    "Heterogeneous treatment effects across users, or when you want the lowest-variance "
                    "uplift estimate available."
                ),
                "tradeoff_vs_alternatives": (
                    "Most compute (trains 4-6 sub-models); harder to debug when something goes wrong; "
                    "needs both arms reasonably populated."
                ),
            },
        ],
    },
}


def model_explanations_for(estimator_type: str) -> dict[str, dict[str, str]]:
    """Return the model_type → triple map for the given estimator_type."""
    if estimator_type == "tabular":
        return TABULAR_MODEL_EXPLANATIONS
    if estimator_type == "embedding":
        return EMBEDDING_MODEL_EXPLANATIONS
    if estimator_type == "sequential":
        return SEQUENTIAL_MODEL_EXPLANATIONS
    return {}


__all__ = [
    "RECOMMENDER_EXPLANATIONS",
    "SCORER_EXPLANATIONS",
    "ESTIMATOR_TYPE_EXPLANATIONS",
    "TABULAR_MODEL_EXPLANATIONS",
    "EMBEDDING_MODEL_EXPLANATIONS",
    "SEQUENTIAL_MODEL_EXPLANATIONS",
    "UPLIFT_PARAM_EXPLANATIONS",
    "model_explanations_for",
]
