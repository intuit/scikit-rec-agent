"""Realistic safeguard scenarios seeded from actual LLM outputs.

The AST/regex tests in test_safeguards.py verify *implementation correctness*.
This file verifies *practical value* — the detector firing (or correctly
not firing) on chunks that look like real model output.

Each scenario is labeled with the failure mode it represents so that when a
future reviewer asks "does this detector catch real hallucinations?", the
answer is a test, not a claim.
"""

from __future__ import annotations

from scikit_rec_agent.safeguards import (
    detect_foreign_references,
    detect_novel_urls,
)

# ---------- URL fabrications ----------


def test_fabricated_kaggle_link_for_amazon_books():
    # GPT-4o-mini output observed during development: a confident Kaggle URL
    # with a user slug that does not exist. Pattern-matched "recommender
    # dataset → Kaggle → here's a URL" from training data.
    model_text = (
        "You can download the Amazon Books recommendation dataset from "
        "the following link: [Amazon Books Dataset]"
        "(https://www.kaggle.com/datasets/shelve/amazon-books)"
    )
    assert detect_novel_urls(model_text, echoed=set()) != set()


def test_echoed_user_supplied_url_does_not_warn():
    # User pasted a URL; model quotes it back. Should not warn.
    model_text = (
        "Using the interactions at https://storage.example.com/clicks.csv, I can profile it with the profile_data tool."
    )
    echoed = {"https://storage.example.com/clicks.csv"}
    assert detect_novel_urls(model_text, echoed=echoed) == set()


def test_model_adds_new_url_alongside_echoed_one():
    # Mixed case: user supplied one URL, model helpfully adds another that
    # didn't come from the user. Warn on the novel one only.
    model_text = (
        "Your data at https://storage.example.com/clicks.csv is fine. "
        "You might also compare against https://arxiv.org/abs/2101.99999."
    )
    echoed = {"https://storage.example.com/clicks.csv"}
    novel = detect_novel_urls(model_text, echoed=echoed)
    assert novel == {"https://arxiv.org/abs/2101.99999"}


def test_markdown_link_punctuation_does_not_prevent_echo_match():
    # URLs inside markdown link syntax — `\S+` captures the closing paren,
    # _normalize_url strips it, so an echo still matches.
    model_text = "see ([the dataset](https://kaggle.com/foo))."
    assert detect_novel_urls(model_text, echoed={"https://kaggle.com/foo"}) == set()


# ---------- Unimported pandas / sklearn / torch ----------


def test_model_shows_pandas_snippet_without_import():
    # Common "here's how to load it" output — the import gets dropped,
    # leaving a bare alias usage. Without the warning, users paste this
    # and hit NameError at runtime.
    model_text = """Here's how to load your interactions:

```python
df = pd.read_csv('/data/interactions.csv', parse_dates=['timestamp'])
df.head()
```
"""
    assert "pandas" in detect_foreign_references(model_text)


def test_model_shows_sklearn_pipeline_without_import():
    model_text = """```python
clf = sklearn.ensemble.RandomForestClassifier(n_estimators=100)
clf.fit(X_train, y_train)
```"""
    assert "sklearn" in detect_foreign_references(model_text)


def test_model_shows_torch_model_without_import():
    # "Here's a two-tower model" — torch is almost never aliased, and the
    # import is commonly elided in teaching-style output.
    model_text = """```python
model = torch.nn.Sequential(
    torch.nn.Linear(64, 32),
    torch.nn.ReLU(),
    torch.nn.Linear(32, 1),
)
```"""
    assert "torch" in detect_foreign_references(model_text)


# ---------- Grounded library usage stays clean ----------


def test_scikit_rec_config_dict_does_not_warn():
    # Canonical config dict the system prompt teaches the model to emit.
    # RecommenderConfig keys are JSON; the accompanying python block uses
    # only skrec.
    model_text = """I'll train a tabular ranker:

```python
from skrec.orchestrator import create_recommender_pipeline
config = {
    "recommender_type": "ranking",
    "scorer_type": "universal",
    "estimator_config": {
        "ml_task": "classification",
        "xgboost": {"n_estimators": 200, "max_depth": 6},
    },
}
model = create_recommender_pipeline(config)
```
"""
    assert detect_foreign_references(model_text) == set()


def test_scikit_rec_agent_code_does_not_warn():
    model_text = """```python
from scikit_rec_agent import Agent
from scikit_rec_agent.llm.anthropic import AnthropicAdapter
import anthropic

agent = Agent(llm=AnthropicAdapter(anthropic.Anthropic()))
for event in agent.chat_turn("hi"):
    pass
```"""
    # `anthropic` IS flagged — it's a real external lib, unverified signatures.
    # This is the correct behavior: grounding only extends to the three
    # library namespaces the agent actually owns.
    assert detect_foreign_references(model_text) == {"anthropic"}


# ---------- Mixed snippets ----------


def test_mixed_grounded_and_foreign_only_flags_foreign():
    # Typical "load via pandas then train via skrec" tutorial.
    model_text = """```python
import pandas as pd
from skrec.orchestrator import create_recommender_pipeline

df = pd.read_csv('/data/x.csv')
model = create_recommender_pipeline({"recommender_type": "ranking", "scorer_type": "universal"})
```"""
    # pandas flagged, skrec not.
    assert detect_foreign_references(model_text) == {"pandas"}


def test_multiple_foreign_libraries_all_flagged():
    model_text = """```python
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from skrec.orchestrator import RecommenderConfig

X, y = pd.read_csv("x").values, np.zeros(100)
X_tr, X_te, y_tr, y_te = train_test_split(X, y)
```"""
    assert detect_foreign_references(model_text) == {"pandas", "numpy", "sklearn"}


# ---------- False-positive guards ----------


def test_prose_mentioning_libraries_without_code_blocks_does_not_warn():
    # A model explaining options in English shouldn't trigger the code
    # detector even if it names pandas, sklearn, torch, etc.
    model_text = (
        "You have two options: use pandas to preprocess, or feed raw CSVs "
        "to create_datasets. The sklearn-style split is also fine. "
        "Consider torch if you want embeddings."
    )
    assert detect_foreign_references(model_text) == set()


def test_bash_install_block_not_flagged_as_python():
    # Models often show `pip install` in a bash fence. Not Python; no warning.
    model_text = """First install dependencies:
```bash
pip install pandas scikit-learn
```"""
    assert detect_foreign_references(model_text) == set()


def test_model_echoing_user_pasted_snippet_does_not_double_warn():
    # If the user pasted a snippet containing `pd.read_csv`, and the model
    # quotes it back verbatim, the warning still fires — the detector
    # intentionally does not track code-block echoes (only URLs). This
    # documents that behavior; users who want echo suppression for code
    # should disable safeguards or strip quoted blocks before display.
    model_text = "Just like the snippet you showed:\n```python\ndf = pd.read_csv('x')\n```"
    assert detect_foreign_references(model_text) == {"pandas"}
