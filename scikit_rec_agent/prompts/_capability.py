"""Runtime-derived capability matrix for the system prompt.

Consumes scikit-rec's public `capability_matrix()` accessor and public enum
tuples (added in scikit-rec 0.3.0). When scikit-rec adds a new recommender /
scorer / estimator / model_type, it flows into the prompt automatically on
the next agent run — no manual sync required.
"""

from __future__ import annotations

from skrec.orchestrator import (
    ESTIMATOR_TYPES,
    RECOMMENDER_TYPES,
    SCORER_TYPES,
)
from skrec.orchestrator import (
    capability_matrix as _factory_capability_matrix,
)


def _init_eval_metric_types() -> tuple[tuple[str, ...], tuple[str, ...]]:
    cm = _factory_capability_matrix()
    if "evaluator_types" in cm and "metric_types" in cm:
        return cm["evaluator_types"], cm["metric_types"]
    from skrec.evaluator.datatypes import RecommenderEvaluatorType
    from skrec.metrics.datatypes import RecommenderMetricType

    return (
        tuple(e.value for e in RecommenderEvaluatorType),
        tuple(m.value for m in RecommenderMetricType),
    )


# Back-compat re-exports. Tests in tests/test_prompts.py read these names.
EVALUATOR_TYPES, METRIC_TYPES = _init_eval_metric_types()


def embedding_model_types() -> tuple[str, ...]:
    return _factory_capability_matrix()["embedding_model_types"]


def sequential_model_types() -> tuple[str, ...]:
    return _factory_capability_matrix()["sequential_model_types"]


def inference_method_types() -> tuple[str, ...]:
    return _factory_capability_matrix()["inference_method_types"]


def retriever_types() -> tuple[str, ...]:
    return _factory_capability_matrix()["retriever_types"]


def capability_matrix() -> str:
    cm = _factory_capability_matrix()
    tabular_model_types = cm.get("tabular_model_types", ("xgboost",))
    lines = [
        f"- recommender_type ∈ {{{', '.join(RECOMMENDER_TYPES)}}}",
        f"- scorer_type ∈ {{{', '.join(SCORER_TYPES)}}}",
        f"- estimator_type ∈ {{{', '.join(ESTIMATOR_TYPES)}}}",
        f"- tabular model_type ∈ {{{', '.join(tabular_model_types)}}}",
        f"- embedding model_type ∈ {{{', '.join(cm['embedding_model_types'])}}}",
        f"- sequential model_type ∈ {{{', '.join(cm['sequential_model_types'])}}}",
        f"- inference_method.type ∈ {{{', '.join(cm['inference_method_types'])}}}",
        f"- retriever.type ∈ {{{', '.join(cm['retriever_types'])}}}",
        f"- evaluator_type ∈ {{{', '.join(EVALUATOR_TYPES)}}}",
        f"- metric ∈ {{{', '.join(METRIC_TYPES)}}}",
    ]
    return "\n".join(lines)
