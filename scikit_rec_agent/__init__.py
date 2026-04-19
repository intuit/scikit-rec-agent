"""scikit-rec-agent: conversational AI agent for scikit-rec."""

from scikit_rec_agent.agent import Agent, AgentEvent
from scikit_rec_agent.llm.base import BaseLLM, LLMResponse, LLMStreamEvent, ToolCall
from scikit_rec_agent.prompts import DEFAULT_SYSTEM_PROMPT
from scikit_rec_agent.session import DatasetBundle, ModelHandle, Session
from scikit_rec_agent.tools import Tool, err, get_default_tools, ok

DEFAULT_TOOLS = get_default_tools  # callable; users invoke to materialize the list

__all__ = [
    "Agent",
    "AgentEvent",
    "BaseLLM",
    "LLMResponse",
    "LLMStreamEvent",
    "ToolCall",
    "DEFAULT_SYSTEM_PROMPT",
    "DEFAULT_TOOLS",
    "DatasetBundle",
    "ModelHandle",
    "Session",
    "Tool",
    "get_default_tools",
    "ok",
    "err",
]
