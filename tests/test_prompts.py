"""Tests that the default system prompt reflects the live capability matrix."""

from __future__ import annotations

from scikit_rec_agent.prompts import DEFAULT_SYSTEM_PROMPT
from scikit_rec_agent.prompts._capability import (
    EVALUATOR_TYPES,
    METRIC_TYPES,
    RECOMMENDER_TYPES,
    SCORER_TYPES,
    embedding_model_types,
    sequential_model_types,
)


def test_prompt_mentions_all_recommender_types():
    for rt in RECOMMENDER_TYPES:
        assert rt in DEFAULT_SYSTEM_PROMPT


def test_prompt_mentions_all_scorer_types():
    for st in SCORER_TYPES:
        assert st in DEFAULT_SYSTEM_PROMPT


def test_prompt_mentions_all_evaluator_types():
    for et in EVALUATOR_TYPES:
        assert et in DEFAULT_SYSTEM_PROMPT


def test_prompt_mentions_all_metrics():
    for m in METRIC_TYPES:
        assert m in DEFAULT_SYSTEM_PROMPT


def test_prompt_includes_embedding_model_types():
    # Enforce that if scikit-rec adds a new embedding model, our prompt picks it up.
    for m in embedding_model_types():
        assert m in DEFAULT_SYSTEM_PROMPT, f"Missing embedding model_type: {m}"


def test_prompt_includes_sequential_model_types():
    for m in sequential_model_types():
        assert m in DEFAULT_SYSTEM_PROMPT, f"Missing sequential model_type: {m}"


def test_prompt_has_canonical_configs():
    assert "recommender_type" in DEFAULT_SYSTEM_PROMPT
    assert "two_tower" in DEFAULT_SYSTEM_PROMPT
    assert "sasrec_classifier" in DEFAULT_SYSTEM_PROMPT
