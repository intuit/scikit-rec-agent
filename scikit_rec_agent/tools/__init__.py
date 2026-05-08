"""Tool registry.

A Tool wraps a Python function with a JSON schema the LLM consumes. Tool
functions accept their schema-defined kwargs plus a `session: Session` final
kwarg that the Agent loop injects. They MUST return a JSON-serializable dict
matching the envelope:

    success: {"status": "ok", "data": {...}}
    error:   {"status": "error", "error_type": "ValueError", "message": "...",
              "hint": "optional high-confidence fix suggestion"}

Errors should be returned as envelopes, not raised, unless the error is a bug
in the tool itself.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable


@dataclass
class Tool:
    name: str
    description: str
    input_schema: dict[str, Any]
    fn: Callable[..., dict[str, Any]]

    def as_llm_schema(self) -> dict[str, Any]:
        """Anthropic-native tool schema shape."""
        return {
            "name": self.name,
            "description": self.description,
            "input_schema": self.input_schema,
        }


def ok(data: dict[str, Any] | None = None) -> dict[str, Any]:
    return {"status": "ok", "data": data or {}}


def err(
    error_type: str,
    message: str,
    hint: str | None = None,
    category: str | None = None,
) -> dict[str, Any]:
    envelope: dict[str, Any] = {
        "status": "error",
        "error_type": error_type,
        "message": message,
    }
    if hint:
        envelope["hint"] = hint
    if category:
        envelope["category"] = category
    return envelope


def _collect_default_tools() -> list[Tool]:
    """Lazy import so optional extras don't eagerly require scikit-rec at import."""
    from scikit_rec_agent.tools.datasets import TOOL_CREATE_DATASETS
    from scikit_rec_agent.tools.evaluation import TOOL_COMPARE_MODELS, TOOL_EVALUATE_MODEL
    from scikit_rec_agent.tools.hpo import TOOL_RUN_HPO
    from scikit_rec_agent.tools.profiling import TOOL_PROFILE_DATA, TOOL_VALIDATE_DATA
    from scikit_rec_agent.tools.registry import (
        TOOL_LIST_MODELS,
        TOOL_LOAD_MODEL,
        TOOL_SAVE_MODEL,
    )
    from scikit_rec_agent.tools.diagnose import TOOL_DIAGNOSE_TRAINING_FAILURE
    from scikit_rec_agent.tools.splitting import TOOL_SPLIT_DATA
    from scikit_rec_agent.tools.sweep import TOOL_SWEEP_METHODS
    from scikit_rec_agent.tools.training import TOOL_TRAIN_MODEL
    from scikit_rec_agent.tools.transform import TOOL_TRANSFORM_DATA

    return [
        TOOL_PROFILE_DATA,
        TOOL_VALIDATE_DATA,
        TOOL_TRANSFORM_DATA,
        TOOL_CREATE_DATASETS,
        TOOL_SPLIT_DATA,
        TOOL_TRAIN_MODEL,
        TOOL_DIAGNOSE_TRAINING_FAILURE,
        TOOL_SWEEP_METHODS,
        TOOL_EVALUATE_MODEL,
        TOOL_COMPARE_MODELS,
        TOOL_RUN_HPO,
        TOOL_SAVE_MODEL,
        TOOL_LIST_MODELS,
        TOOL_LOAD_MODEL,
    ]


def get_default_tools() -> list[Tool]:
    """Return the default tools. Importing this triggers tool module imports
    which require scikit-rec — fine in practice since scikit-rec is a main dep.
    """
    return _collect_default_tools()


__all__ = ["Tool", "ok", "err", "get_default_tools"]
