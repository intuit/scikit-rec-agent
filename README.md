# scikit-rec-agent

Conversational AI agent that uses [scikit-rec](https://github.com/intuit/scikit-rec) as its tool belt. The agent reasons about the user's data and goals, then calls scikit-rec APIs via structured tool use to build, evaluate, and compare recommendation systems.

## Install

```bash
pip install scikit-rec-agent[anthropic]     # with Claude
pip install scikit-rec-agent[openai]        # with GPT-4
pip install scikit-rec-agent                # bring your own LLM
pip install scikit-rec-agent[anthropic,torch]  # + deep-learning models
```

## CLI

```bash
export ANTHROPIC_API_KEY=...
scikit-rec-agent chat
```

Auto-detects the provider from env vars. Pass `--provider {anthropic,openai}` if both are set.

## Library

```python
import anthropic
from scikit_rec_agent import Agent
from scikit_rec_agent.llm.anthropic import AnthropicAdapter

agent = Agent(llm=AnthropicAdapter(anthropic.Anthropic()))
for event in agent.chat_turn("I have click data at /data/interactions.csv — help me build a ranker"):
    ...
```

See [`examples/`](./examples/) for:
- `custom_tool.py` — register a user-defined tool
- `custom_prompt.py` — extend or replace the system prompt
- `custom_llm.py` — plug in your company's internal LLM via the `BaseLLM` protocol
- `custom_frontend.py` — drive the agent from Jupyter / Slack / web
- `movielens_session.md` — annotated end-to-end transcript

## What it does

Eleven tools cover the full scikit-rec workflow: profile data, validate schemas, build datasets, split (temporal / per-user / cold-start), train (6 recommender types × 6 scorers × 3 estimator planes), evaluate (7 evaluator types × 9 metrics), compare, run HPO (Optuna), and persist to a local model registry.

The system prompt is built at import time from scikit-rec's live enum maps, so new recommender/scorer/estimator types get picked up automatically.

## Architecture

See [`agentic_design.md`](./agentic_design.md) for the authoritative spec.

## License

Apache-2.0
