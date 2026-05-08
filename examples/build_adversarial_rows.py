# ruff: noqa: E501
"""Append hand-authored adversarial rows to halluc_corpus.jsonl.

These rows do not come from an LLM. Each one targets a specific edge case
of the URL detector or the foreign-reference AST scanner — the cases the
plan listed under "Known failure modes to probe." Labels are pre-populated
per Rule A applied semantically (the rule, not the detector's
implementation), so the resulting confusion matrix tells us where the
implementation diverges from the rule.

Idempotent: skips append if rows with source=adversarial_handcrafted
already exist (so re-running this does not duplicate the corpus).
"""

from __future__ import annotations

import datetime
import json
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO_ROOT))

from scikit_rec_agent.safeguards import SAFEGUARDS_VERSION  # noqa: E402

CORPUS_PATH = _REPO_ROOT / "examples" / "halluc_corpus.jsonl"
SYS_PROMPT_ID = "default_v1.0"
THEME = "ast_edge_cases"
TODAY = datetime.date.today().isoformat()


def _row(
    *,
    user_message: str,
    llm_output: str,
    expected_url_set: list[str],
    expected_foreign_set: list[str],
    label_notes: str,
    trusted_urls: list[str] | None = None,
) -> dict:
    """Build one adversarial row with labels pre-populated per Rule A."""
    return {
        "id": None,  # filled in at write time
        "source": "adversarial_handcrafted",
        "theme": THEME,
        "turn_index": 0,
        "user_message": user_message,
        "system_prompt_id": SYS_PROMPT_ID,
        "trusted_urls": trusted_urls or [],
        "llm_output": llm_output,
        "model": "n/a (hand-authored)",
        "generation_date": TODAY,
        "safeguards_version": SAFEGUARDS_VERSION,
        "live_warnings": [],
        "expected_url_flag": bool(expected_url_set),
        "expected_url_set": sorted(expected_url_set),
        "expected_foreign_flag": bool(expected_foreign_set),
        "expected_foreign_set": sorted(expected_foreign_set),
        "label_notes": label_notes,
    }


# -----------------------------------------------------------------------------
# Foreign-ref detector — POSITIVES (detector should fire)
# -----------------------------------------------------------------------------

ROWS: list[dict] = [
    _row(
        user_message="Show me how to load a CSV in Python.",
        llm_output="```python\nimport pandas as pd\ndf = pd.read_csv('data.csv')\n```",
        expected_url_set=[],
        expected_foreign_set=["pandas"],
        label_notes="baseline: explicit foreign import + use",
    ),
    _row(
        user_message="What's a quick classifier?",
        llm_output="```python\nfrom sklearn.ensemble import RandomForestClassifier\nclf = RandomForestClassifier()\n```",
        expected_url_set=[],
        expected_foreign_set=["sklearn"],
        label_notes="from-X-import-Y form",
    ),
    _row(
        user_message="Make the import optional.",
        llm_output="```python\ntry:\n    import torch\nexcept ImportError:\n    torch = None\n\nif torch is not None:\n    x = torch.zeros(3)\n```",
        expected_url_set=[],
        expected_foreign_set=["torch"],
        label_notes="import inside try/except ImportError — should still flag torch",
    ),
    _row(
        user_message="Wrap that loader in a function.",
        llm_output="```python\ndef get_data(path):\n    import pandas as pd\n    return pd.read_csv(path)\n```",
        expected_url_set=[],
        expected_foreign_set=["pandas"],
        label_notes="import inside a function body — scope-aware visitor should still classify",
    ),
    _row(
        user_message="How do I dynamically import a backend?",
        llm_output="```python\nimport importlib\nbackend = importlib.import_module('torch')\nbackend.set_default_dtype(backend.float32)\n```",
        expected_url_set=[],
        expected_foreign_set=["torch"],
        label_notes="dynamic import via importlib.import_module(literal)",
    ),
    _row(
        user_message="What about __import__?",
        llm_output="```python\nxgb = __import__('xgboost')\nmodel = xgb.XGBRegressor()\n```",
        expected_url_set=[],
        expected_foreign_set=["xgboost"],
        label_notes="dynamic import via __import__(literal)",
    ),
    _row(
        user_message="Quick numpy snippet.",
        llm_output="```python\narr = np.array([1, 2, 3])\nprint(arr.mean())\n```",
        expected_url_set=[],
        expected_foreign_set=["numpy"],
        label_notes="bare alias usage with no matching import — common LLM laziness",
    ),
    _row(
        user_message="Show inline plotting setup.",
        llm_output="```python\n%matplotlib inline\n!pip install torch\nimport torch\nx = torch.zeros(3)\n```",
        expected_url_set=[],
        expected_foreign_set=["torch"],
        label_notes="notebook magics + shell magics interleaved with real Python; magics must be stripped before parse",
    ),
    # -------------------------------------------------------------------------
    # Foreign-ref detector — NEGATIVES (detector should NOT fire)
    # -------------------------------------------------------------------------
    _row(
        user_message="Hello world.",
        llm_output="```python\nimport skrec\nfrom scikit_rec_agent.tools import get_default_tools\nprint('hi')\n```",
        expected_url_set=[],
        expected_foreign_set=[],
        label_notes="trusted-package imports only — pure negative",
    ),
    _row(
        user_message="Pure stdlib snippet.",
        llm_output="```python\nimport os\nimport json\nimport math\n\ndata = json.loads('{}')\nprint(math.pi)\n```",
        expected_url_set=[],
        expected_foreign_set=[],
        label_notes="stdlib-only — should not flag",
    ),
    _row(
        user_message="Take a function as input.",
        llm_output="```python\ndef apply_to_pd(pd, value):\n    return pd.normalize(value)\n```",
        expected_url_set=[],
        expected_foreign_set=[],
        label_notes="`pd` is a function parameter, not pandas — must not flag (scope-aware visitor)",
    ),
    _row(
        user_message="Use a local shim.",
        llm_output="```python\nimport skrec\npd = skrec.pandas_compat\ndf = pd.DataFrame({'x': [1, 2]})\n```",
        expected_url_set=[],
        expected_foreign_set=[],
        label_notes="alias rebound to a trusted package — `pd` here is not pandas; visitor must respect rebind",
    ),
    _row(
        user_message="Iterate the registry.",
        llm_output="```python\nimport skrec\nresult = [pd for pd in skrec.items() if pd.is_active()]\n```",
        expected_url_set=[],
        expected_foreign_set=[],
        label_notes="`pd` bound by a comprehension — comprehension scope should mask the alias",
    ),
    _row(
        user_message="Show me a shell command.",
        llm_output="```bash\nimport torch  # not actually python — bash cell\npip install pandas\n```",
        expected_url_set=[],
        expected_foreign_set=[],
        label_notes="bash-tagged code block — body looks Python-ish but should be skipped (non-python cell magic class)",
    ),
    # -------------------------------------------------------------------------
    # Documented detector limitations — FN cases (rule-A says positive, detector misses)
    # -------------------------------------------------------------------------
    _row(
        user_message="Reach into a backend.",
        llm_output="```python\nimport importlib as il\nbackend = il.import_module('torch')\n```",
        expected_url_set=[],
        expected_foreign_set=["torch"],
        label_notes="aliased importlib — known FN: visitor only matches literal `importlib.import_module`",
    ),
    _row(
        user_message="Build the import target dynamically.",
        llm_output="```python\nimport importlib\nname = 'tor' + 'ch'\nbackend = importlib.import_module(name)\n```",
        expected_url_set=[],
        expected_foreign_set=["torch"],
        label_notes="non-literal first arg to import_module — known FN per safeguards docstring",
    ),
    # -------------------------------------------------------------------------
    # URL detector — POSITIVES (verify URL_PATTERN scope and normalization)
    # -------------------------------------------------------------------------
    _row(
        user_message="Drop a code example with a docs link.",
        llm_output="```python\n# See https://made-up.example/docs for details\nimport skrec\n```",
        expected_url_set=["https://made-up.example/docs"],
        expected_foreign_set=[],
        label_notes="URL inside fenced code block — URL_PATTERN runs on full text, so this should still flag",
    ),
    _row(
        user_message="Cite something parenthetically.",
        llm_output="The dataset (see https://made-up.example/data.csv) has 10M rows.",
        expected_url_set=["https://made-up.example/data.csv"],
        expected_foreign_set=[],
        label_notes="URL with trailing punctuation eaten by `\\S+` — normalize step should strip the closing paren",
    ),
    # -------------------------------------------------------------------------
    # Documented limitations — out-of-scope by design
    # -------------------------------------------------------------------------
    _row(
        user_message="Where does the agent live?",
        llm_output="The project is hosted at arxiv.org/abs/2024.12345 and intuit.com/recsys.",
        expected_url_set=[],
        expected_foreign_set=[],
        label_notes="scheme-less URLs — out of scope by design (URL_PATTERN requires http/https). Cite as scope choice in §7.3",
    ),
    _row(
        user_message="Train a model.",
        llm_output="```python\nimport skrec\nresult = skrec.nonexistent_function(weird_kwarg=True)\n```",
        expected_url_set=[],
        expected_foreign_set=[],
        label_notes="trusted-package function fabrication — rule-A says no foreign ref (skrec is trusted), so detector correctly does not flag. This IS a hallucination but of a different kind; cite in §7.3 as out-of-scope",
    ),
    _row(
        user_message="Quote the dataset paper.",
        llm_output='According to the canonical paper, "the model achieves NDCG@10 = 0.87 on MovieLens-25M" — see the original publication.',
        expected_url_set=[],
        expected_foreign_set=[],
        label_notes="prose-level numerical fabrication, no URL or code — out-of-scope for both detectors. Section 7.3 example",
    ),
]


def _next_id_start(corpus_path: Path) -> int:
    if not corpus_path.exists():
        return 0
    with corpus_path.open() as f:
        return sum(1 for _ in f)


def _existing_adversarial_count(corpus_path: Path) -> int:
    if not corpus_path.exists():
        return 0
    n = 0
    with corpus_path.open() as f:
        for line in f:
            if json.loads(line)["source"] == "adversarial_handcrafted":
                n += 1
    return n


def main() -> int:
    if _existing_adversarial_count(CORPUS_PATH) > 0:
        print(f"refusing to append: {CORPUS_PATH} already contains adversarial rows")
        print("delete or filter them first if you want to regenerate")
        return 1

    start = _next_id_start(CORPUS_PATH)
    CORPUS_PATH.parent.mkdir(parents=True, exist_ok=True)
    with CORPUS_PATH.open("a") as f:
        for i, row in enumerate(ROWS):
            row["id"] = f"halluc_{start + i:04d}"
            f.write(json.dumps(row) + "\n")
    print(f"appended {len(ROWS)} adversarial rows ({start:04d}-{start + len(ROWS) - 1:04d})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
