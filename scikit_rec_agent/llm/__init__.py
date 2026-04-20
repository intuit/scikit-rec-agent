"""LLM provider abstraction layer.

Users bring their own LLM via the BaseLLM protocol. Anthropic and OpenAI adapters
ship in the box; any compliant implementation of the protocol works.
"""

from scikit_rec_agent.llm.base import (
    BaseLLM,
    LLMResponse,
    LLMStreamEvent,
    ToolCall,
)

__all__ = ["BaseLLM", "LLMResponse", "LLMStreamEvent", "ToolCall"]
