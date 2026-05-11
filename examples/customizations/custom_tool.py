"""Example: register a user-defined tool alongside the 15 defaults.

Tools receive their schema-defined kwargs plus a `session: Session` final arg
injected by the Agent loop. They MUST return a JSON-serializable dict matching
the {"status": "ok", "data": {...}} or {"status": "error", ...} envelope.
"""

from __future__ import annotations

from scikit_rec_agent import Agent, Tool, get_default_tools
from scikit_rec_agent.llm import BaseLLM
from scikit_rec_agent.tools import ok


def fetch_from_warehouse(query: str, session) -> dict:
    """Pretend we're calling Snowflake or BigQuery with `query`."""
    # Real implementation would execute `query` and write a CSV to disk,
    # then return its path so the LLM can pass it to create_datasets.
    return ok({"rows_returned": 0, "output_path": "/tmp/warehouse_export.csv", "query": query})


def build_agent(llm: BaseLLM) -> Agent:
    warehouse_tool = Tool(
        name="fetch_from_warehouse",
        description="Run a SQL query against the data warehouse and write results to a CSV.",
        input_schema={
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "SQL query to execute."},
            },
            "required": ["query"],
        },
        fn=fetch_from_warehouse,
    )
    return Agent(llm=llm, tools=[*get_default_tools(), warehouse_tool])


if __name__ == "__main__":
    import anthropic  # noqa: F401  # requires [anthropic] extra

    from scikit_rec_agent.llm.anthropic import AnthropicAdapter

    adapter = AnthropicAdapter(anthropic.Anthropic())
    agent = build_agent(adapter)
    agent.chat()
