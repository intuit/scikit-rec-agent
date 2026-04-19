"""Example: extend or replace the default system prompt.

The default prompt encodes the full capability matrix + heuristics. Teams with
strong conventions (specific metrics, forbidden architectures, house style)
can either append to it or replace it entirely.
"""

from __future__ import annotations

from scikit_rec_agent import DEFAULT_SYSTEM_PROMPT, Agent

TEAM_ADDENDUM = """

# Team conventions (added by our team)

- Always report NDCG@10 as the primary metric; other metrics are secondary.
- Prefer tabular ranking for fast experiments unless data is >1M interactions.
- Any model intended for production must be saved via save_model with a
  'production-candidate' tag and include the git SHA of the training script.
"""


def build_agent(llm) -> Agent:
    custom = DEFAULT_SYSTEM_PROMPT + TEAM_ADDENDUM
    return Agent(llm=llm, system_prompt=custom)


if __name__ == "__main__":
    import anthropic

    from scikit_rec_agent.llm.anthropic import AnthropicAdapter

    agent = build_agent(AnthropicAdapter(anthropic.Anthropic()))
    agent.chat()
