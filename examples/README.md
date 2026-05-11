# Examples

## `customizations/`

Short, runnable scripts showing each extension point:

| Script | What it shows |
|---|---|
| `custom_tool.py` | Register a user-defined tool (e.g. a warehouse query) alongside the defaults |
| `custom_prompt.py` | Extend or replace the system prompt |
| `custom_llm.py` | Plug in your company's internal LLM via the `BaseLLM` protocol |
| `custom_frontend.py` | Drive the agent from Jupyter, Slack, or a web UI |

Each script runs end-to-end with `python <script>.py` after installing the relevant extras:

```bash
pip install scikit-rec-agent[anthropic]
```

## `transcripts/`

Full captured chat sessions showing the two main workflows:

| Transcript | What it shows |
|---|---|
| `movielens_session.md` | **Sweep flow** — compare 7 methods on MovieLens-1M, pick the winner |
| `movielens_hierarchical_session.md` | **Design flow** — walk through the model picker step by step on the same data |
