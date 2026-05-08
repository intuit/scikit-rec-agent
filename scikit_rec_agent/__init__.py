"""scikit-rec-agent: conversational AI agent for scikit-rec."""

# macOS BLAS thread-state workaround. When numpy (used by MatrixFactorization
# and other ALS-style estimators) trains in the same process as a torch model
# (NCF, Two-Tower, DCN, NFM), the macOS Accelerate / OpenBLAS runtime is left
# in a state that segfaults inside torch's BCE forward pass. Reproducible:
# `MF.train()` then `NCF.train()` in one Python process exits with SIGSEGV,
# while either alone in a fresh process is fine. Setting these three env vars
# before numpy / torch are imported pins thread counts to 1 and avoids the
# pollution. Cost is mild — ALS / matrix multiplies don't parallelise — but
# the alternative is "torch-based estimators silently kill the process", which
# is much worse for an agent that runs sweeps. Set HERE (package import root)
# rather than in the CLI so library users get the fix automatically.
import os as _os

for _k in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    _os.environ.setdefault(_k, "1")
del _os, _k

from scikit_rec_agent.agent import Agent, AgentEvent
from scikit_rec_agent.llm.base import BaseLLM, LLMResponse, LLMStreamEvent, ToolCall
from scikit_rec_agent.prompts import DEFAULT_SYSTEM_PROMPT
from scikit_rec_agent.session import DatasetBundle, ModelHandle, Session
from scikit_rec_agent.tools import Tool, err, get_default_tools, ok

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
    "ok",
    "err",
]
