"""Runtime-derived capability matrix for the system prompt.

Reaches into scikit-rec factory private maps. If scikit-rec exposes a public
capability-matrix accessor later (tracked as a follow-up), swap this over.
"""

from __future__ import annotations

# Top-level enums are hardcoded (the factory enforces them via if/elif chains,
# not enum maps). Tests in tests/test_prompts.py assert the prompt mentions
# every value so drift gets caught in CI.
RECOMMENDER_TYPES = (
    "ranking",
    "bandits",
    "sequential",
    "hierarchical_sequential",
    "uplift",
    "gcsl",
)

SCORER_TYPES = (
    "universal",
    "independent",
    "multiclass",
    "multioutput",
    "sequential",
    "hierarchical",
)

ESTIMATOR_TYPES = ("tabular", "embedding", "sequential")

EVALUATOR_TYPES = (
    "simple",
    "replay_match",
    "IPS",
    "DR",
    "direct_method",
    "SNIPS",
    "policy_weighted",
)

METRIC_TYPES = (
    "NDCG_at_k",
    "MAP_at_k",
    "MRR_at_k",
    "precision_at_k",
    "recall_at_k",
    "average_reward_at_k",
    "roc_auc",
    "pr_auc",
    "expected_reward",
)


def _get_from_factory(name: str) -> list[str]:
    """Read a private map from skrec.orchestrator.factory. Returns empty list
    if the import fails, so the capability matrix degrades gracefully if
    scikit-rec restructures.
    """
    try:
        from skrec.orchestrator import factory

        val = getattr(factory, name, None)
        if val is None:
            return []
        return list(val.keys())
    except Exception:
        return []


def embedding_model_types() -> list[str]:
    return _get_from_factory("_EMBEDDING_ESTIMATOR_MAP")


def sequential_model_types() -> list[str]:
    return _get_from_factory("_SEQUENTIAL_ESTIMATOR_MAP")


def inference_method_types() -> list[str]:
    return _get_from_factory("_INFERENCE_METHOD_MAP")


def retriever_types() -> list[str]:
    return _get_from_factory("_RETRIEVER_MAP")


def capability_matrix() -> str:
    lines = [
        f"- recommender_type ∈ {{{', '.join(RECOMMENDER_TYPES)}}}",
        f"- scorer_type ∈ {{{', '.join(SCORER_TYPES)}}}",
        f"- estimator_type ∈ {{{', '.join(ESTIMATOR_TYPES)}}}",
        f"- embedding model_type ∈ {{{', '.join(embedding_model_types())}}}",
        f"- sequential model_type ∈ {{{', '.join(sequential_model_types())}}}",
        f"- inference_method.type ∈ {{{', '.join(inference_method_types())}}}",
        f"- retriever.type ∈ {{{', '.join(retriever_types())}}}",
        f"- evaluator_type ∈ {{{', '.join(EVALUATOR_TYPES)}}}",
        f"- metric ∈ {{{', '.join(METRIC_TYPES)}}}",
    ]
    return "\n".join(lines)
