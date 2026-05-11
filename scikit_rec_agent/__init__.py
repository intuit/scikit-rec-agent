"""scikit-rec-agent: conversational AI agent for scikit-rec."""

# macOS-only BLAS thread-state workaround.
#
# When numpy (used by MatrixFactorization and other ALS-style estimators)
# trains in the same process as a torch model (NCF, Two-Tower, DCN, NFM),
# the macOS Accelerate / OpenBLAS runtime is left in a state that segfaults
# inside torch's BCE forward pass. Reproducible: ``MF.train()`` then
# ``NCF.train()`` in one Python process exits with SIGSEGV; either alone in
# a fresh process is fine. Pinning OMP_NUM_THREADS / MKL_NUM_THREADS /
# VECLIB_MAXIMUM_THREADS to 1 before numpy / torch import time avoids it.
#
# We pin only on darwin (Linux + Windows don't reproduce), use ``setdefault``
# (so an existing OMP_NUM_THREADS=8 in the user's shell wins), and honour
# ``SCIKIT_REC_AGENT_DISABLE_BLAS_PIN=1`` for users who want to skip it
# entirely. The pin is opinionated; the opt-out makes it not silent.
import os as _os
import sys as _sys

if _sys.platform == "darwin" and _os.environ.get("SCIKIT_REC_AGENT_DISABLE_BLAS_PIN") != "1":
    for _k in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
        _os.environ.setdefault(_k, "1")
    del _k
del _os, _sys

# Imports MUST happen after the env-var pin above so torch / numpy read
# the threading limits at import time. ruff E402 is intentionally suppressed.
from scikit_rec_agent.agent import Agent, AgentEvent  # noqa: E402
from scikit_rec_agent.llm.base import BaseLLM, LLMResponse, LLMStreamEvent, ToolCall  # noqa: E402
from scikit_rec_agent.prompts import DEFAULT_SYSTEM_PROMPT  # noqa: E402
from scikit_rec_agent.session import DatasetBundle, ModelHandle, Session  # noqa: E402
from scikit_rec_agent.tools import Tool, get_default_tools  # noqa: E402

__all__ = [
    "Agent",
    "AgentEvent",
    "BaseLLM",
    "LLMResponse",
    "LLMStreamEvent",
    "ToolCall",
    "DEFAULT_SYSTEM_PROMPT",
    "DatasetBundle",
    "ModelHandle",
    "Session",
    "Tool",
    "get_default_tools",
]
