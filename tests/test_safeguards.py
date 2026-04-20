"""Tests for the hallucination safeguards in scikit_rec_agent.safeguards."""

from __future__ import annotations

import pytest

from scikit_rec_agent.safeguards import (
    EXTERNAL_REFERENCE_WARNING,
    URL_PATTERN,
    detect_foreign_references,
    detect_novel_urls,
)

# ---------- URL detection ----------


def test_url_pattern_matches_http_and_https():
    assert URL_PATTERN.search("see http://a.b/c")
    assert URL_PATTERN.search("see https://a.b/c")


def test_url_pattern_ignores_local_paths_and_prose():
    assert not URL_PATTERN.search("/data/interactions.csv")
    assert not URL_PATTERN.search("plain prose, no link")


def test_detect_novel_urls_echoes_are_suppressed():
    text = "already seen: https://kaggle.com/x"
    assert detect_novel_urls(text, {"https://kaggle.com/x"}) == set()


def test_detect_novel_urls_returns_only_new_ones():
    text = "old: https://kaggle.com/x new: https://arxiv.org/y"
    echoed = {"https://kaggle.com/x"}
    assert detect_novel_urls(text, echoed) == {"https://arxiv.org/y"}


def test_detect_novel_urls_handles_trailing_punctuation():
    # URL_PATTERN's \S+ swallows trailing punctuation; normalization strips it
    # so "https://x.com/foo." matches "https://x.com/foo" for echo purposes.
    assert detect_novel_urls("see (https://kaggle.com/x),", {"https://kaggle.com/x"}) == set()


def test_external_reference_warning_mentions_urls_only():
    # Earlier drafts overpromised "dataset names, citations" that detection
    # never actually matched — keep the warning honest about URLs only.
    assert "URL" in EXTERNAL_REFERENCE_WARNING
    assert "dataset name" not in EXTERNAL_REFERENCE_WARNING.lower()


# ---------- Static imports ----------


def test_library_only_imports_are_clean():
    code = "```python\nfrom skrec.orchestrator import create_recommender_pipeline\nimport json\n```"
    assert detect_foreign_references(code) == set()


def test_static_foreign_imports_are_flagged():
    code = "```python\nimport pandas as pd\nfrom sklearn.tree import DecisionTreeClassifier\n```"
    assert detect_foreign_references(code) == {"pandas", "sklearn"}


def test_grounded_roots_not_flagged():
    # All three grounded roots must stay unflagged even when used in realistic
    # combinations. This is the contract the system prompt also advertises.
    code = (
        "```python\n"
        "from skrec.orchestrator import RecommenderConfig\n"
        "from scikit_rec import x\n"
        "from scikit_rec_agent import Agent\n"
        "```"
    )
    assert detect_foreign_references(code) == set()


# ---------- Dynamic imports ----------


def test_importlib_import_module_is_flagged():
    code = '```python\nimport importlib\nmod = importlib.import_module("torch")\n```'
    assert "torch" in detect_foreign_references(code)


def test_builtin_dunder_import_is_flagged():
    code = '```python\nmod = __import__("requests")\n```'
    assert "requests" in detect_foreign_references(code)


def test_dynamic_import_ignores_fstring_args():
    # Best-effort: non-constant args are out of scope, not a bug.
    code = '```python\nimportlib.import_module(f"tor{ch}")\n```'
    assert detect_foreign_references(code) == set()


# ---------- Bare alias usage ----------


def test_bare_alias_without_import_is_flagged():
    code = '```python\ndf = pd.read_csv("x.csv")\n```'
    assert "pandas" in detect_foreign_references(code)


def test_aliased_import_counts_once():
    code = '```python\nimport pandas as pd\ndf = pd.read_csv("x.csv")\n```'
    assert detect_foreign_references(code) == {"pandas"}


def test_locally_assigned_alias_does_not_trigger():
    # `pd = 42; pd.bit_length()` — pd is an int here, not pandas.
    code = "```python\npd = 42\nprint(pd.bit_length())\n```"
    assert "pandas" not in detect_foreign_references(code)


# ---------- Scope awareness (round-2 review fix) ----------


def test_function_param_does_not_suppress_module_level_usage():
    # The bug the round-2 review caught: def demo(pd) made pd "locally defined"
    # globally, hiding bare pd.read_csv at module level. Scope-aware visitor
    # must bind pd only inside demo's scope.
    code = "```python\ndef demo(pd):\n    return pd.head()\n\ndf = pd.read_csv('x.csv')\n```"
    assert "pandas" in detect_foreign_references(code)


def test_class_body_binding_is_invisible_to_methods():
    # Python semantics: methods can't see class-body names without `self.`
    # or `ClassName.`. Detector must flag bare pd inside the method even if
    # the class body declares `pd = something`.
    code = "```python\nclass Foo:\n    pd = something\n    def bar(self):\n        return pd.read_csv('x.csv')\n```"
    assert "pandas" in detect_foreign_references(code)


def test_module_level_import_visible_in_nested_function():
    code = "```python\nimport pandas as pd\ndef demo():\n    return pd.read_csv('x.csv')\n```"
    assert detect_foreign_references(code) == {"pandas"}


def test_walrus_binds_alias():
    code = "```python\nif (pd := some_getter()) is not None:\n    print(pd.head())\n```"
    assert "pandas" not in detect_foreign_references(code)


@pytest.mark.parametrize(
    "snippet",
    [
        "rows = [pd for pd in data]",
        "gen = (pd for pd in data)",
        "try:\n    x = 1\nexcept Exception as pd:\n    print(pd.args)",
        "with open('f') as pd:\n    print(pd.read())",
        "for pd in items:\n    print(pd)",
    ],
)
def test_other_binding_forms_suppress_alias(snippet):
    assert "pandas" not in detect_foreign_references(f"```python\n{snippet}\n```")


def test_comprehension_binding_does_not_leak_outward():
    # Comprehension scope is local; after the comp, pd is unbound again.
    code = "```python\nrows = [pd for pd in data]\ndf = pd.read_csv('x.csv')\n```"
    assert "pandas" in detect_foreign_references(code)


# ---------- Fence variants & untagged blocks ----------


@pytest.mark.parametrize("tag", ["python", "py", "py3", "ipython", "jupyter", "PYTHON", ""])
def test_fence_variants_are_scanned(tag):
    # Empty tag → untagged fence. Models often drop the language hint.
    fence = f"```{tag}\n" if tag else "```\n"
    code = f"{fence}import pandas\n```"
    assert detect_foreign_references(code) == {"pandas"}


def test_non_python_fences_are_ignored():
    assert detect_foreign_references("```bash\npip install pandas\n```") == set()
    assert detect_foreign_references("```shell\nimport torch\n```") == set()


# ---------- Notebook magics ----------


def test_line_magics_stripped_before_parse():
    code = "```python\n%matplotlib inline\nimport pandas\n```"
    assert detect_foreign_references(code) == {"pandas"}


def test_shell_lines_stripped_before_parse():
    code = "```python\n!pip install pandas\nimport pandas\n```"
    assert detect_foreign_references(code) == {"pandas"}


def test_cell_magic_with_python_body_still_scanned():
    # %%timeit body is Python — we must not skip it.
    code = "```python\n%%timeit\nimport pandas\n```"
    assert detect_foreign_references(code) == {"pandas"}


@pytest.mark.parametrize("magic", ["html", "bash", "shell", "sh", "javascript", "js", "sql", "markdown", "md"])
def test_non_python_cell_magics_skip_entire_block(magic):
    # %%html / %%bash / etc. — body is not Python. Scanning would either
    # ast.parse-fail (silent skip of a block that might still contain
    # Python-looking noise) or falsely "pass" — skip the whole block cleanly.
    code = f"```python\n%%{magic}\nimport pandas\n```"
    assert detect_foreign_references(code) == set()


# ---------- Robustness ----------


def test_syntax_errors_silently_skipped():
    # Pseudocode / truncated snippets must not produce false positives.
    code = "```python\nthis is not(( python\n```"
    assert detect_foreign_references(code) == set()


def test_empty_and_whitespace_only_blocks():
    assert detect_foreign_references("```python\n\n```") == set()
    assert detect_foreign_references("```python\n   \n```") == set()


def test_multiple_blocks_are_all_scanned():
    code = "First:\n```python\nimport pandas\n```\nSecond:\n```python\nimport torch\n```"
    assert detect_foreign_references(code) == {"pandas", "torch"}


def test_relative_imports_not_flagged():
    # `from . import foo` has level > 0 and no reachable module name; skip.
    code = "```python\nfrom . import foo\nfrom ..bar import baz\n```"
    assert detect_foreign_references(code) == set()


# ---------- RHS-before-LHS visit order (round-3 review fix) ----------


def test_self_assignment_from_alias_flags_rhs():
    # The bug the round-3 review caught: visit_Assign bound the target
    # before descending into value, so `pd = pd.read_csv(...)` read `pd`
    # on the RHS while `pd` was already "bound" to itself. Must flag.
    code = "```python\npd = pd.read_csv('x.csv')\n```"
    assert "pandas" in detect_foreign_references(code)


def test_walrus_self_assignment_flags_rhs():
    code = "```python\nif (pd := pd.get()):\n    print(pd)\n```"
    assert "pandas" in detect_foreign_references(code)


def test_augmented_assignment_reads_alias_on_rhs():
    code = "```python\ntotal += pd.Series([1, 2]).sum()\n```"
    assert "pandas" in detect_foreign_references(code)


def test_annotated_assignment_reads_alias_in_annotation():
    code = "```python\nx: pd.DataFrame = load()\n```"
    assert "pandas" in detect_foreign_references(code)


def test_for_iterable_reads_alias_before_target_binds():
    code = "```python\nfor pd in pd.items():\n    print(pd)\n```"
    assert "pandas" in detect_foreign_references(code)


def test_with_context_expr_reads_alias_before_as_binds():
    code = "```python\nwith pd.HDFStore('x') as pd:\n    pass\n```"
    assert "pandas" in detect_foreign_references(code)


def test_comprehension_first_iterable_reads_in_enclosing_scope():
    # `[pd for pd in pd.items()]` — the iterable `pd.items()` evaluates in
    # the enclosing scope where pd is unbound.
    code = "```python\nrows = [pd for pd in pd.items()]\n```"
    assert "pandas" in detect_foreign_references(code)


# ---------- Decorators and annotations ----------


def test_decorator_with_foreign_alias_is_flagged():
    code = "```python\n@pd.wrap\ndef f():\n    return 1\n```"
    assert "pandas" in detect_foreign_references(code)


def test_function_annotations_with_foreign_alias_are_flagged():
    code = "```python\ndef f(x: pd.DataFrame) -> pd.Series:\n    return x.iloc[0]\n```"
    assert "pandas" in detect_foreign_references(code)


# ---------- New alias coverage ----------


def test_bare_torch_flagged():
    code = "```python\nmodel = torch.nn.Linear(10, 5)\n```"
    assert "torch" in detect_foreign_references(code)


def test_bare_sklearn_flagged():
    code = "```python\nclf = sklearn.ensemble.RandomForestClassifier()\n```"
    assert "sklearn" in detect_foreign_references(code)


def test_bare_jnp_flagged_as_jax():
    code = "```python\nx = jnp.array([1, 2])\n```"
    assert "jax" in detect_foreign_references(code)


# ---------- Nested scope shadowing ----------


def test_nested_function_in_method_does_not_inherit_class_body():
    # Double-nested scope: class Foo { def bar { def inner() { pd.x } } }
    # `pd` declared in class body must not leak into `inner`.
    code = (
        "```python\n"
        "class Foo:\n"
        "    pd = something\n"
        "    def bar(self):\n"
        "        def inner():\n"
        "            return pd.read_csv('x')\n"
        "        return inner()\n"
        "```"
    )
    assert "pandas" in detect_foreign_references(code)


# ---------- URL shape coverage ----------


def test_markdown_link_url_is_echo_matched():
    # `[label](https://x.com/foo)` — \S+ captures the trailing `)`;
    # _normalize_url strips it so echo matching stays stable.
    text = "see [the site](https://x.com/foo)"
    assert detect_novel_urls(text, {"https://x.com/foo"}) == set()
