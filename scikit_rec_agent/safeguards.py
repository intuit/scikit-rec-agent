"""Post-hoc hallucination safeguards for agent output.

Two deterministic detectors run at the end of each user turn. Warnings are
emitted as ``AgentEvent(type="warning")`` by the Agent loop and never enter
conversation history, so the LLM cannot learn to evade them.

What this catches
-----------------
- URLs in model output that the user did not supply this session. Shipped
  adapters have no web retrieval, so model-introduced URLs are potential
  fabrications (a common failure mode for "where do I download dataset X?"
  prompts).
- References in fenced Python blocks to packages outside the trusted set
  (``skrec``, ``scikit_rec``, ``scikit_rec_agent``, Python stdlib). This
  includes ``import pandas``, ``importlib.import_module("torch")``, and
  bare usage like ``pd.read_csv(...)`` without the matching import.

What this DOES NOT catch
------------------------
- Semantic errors inside trusted APIs (wrong ``RecommenderConfig`` shape,
  nonsensical metric choice, off-by-one indexing). The scikit-rec factory
  catches bad configs at ``train_model``; everything else is on the user.
- Invented keyword arguments for real external libraries — if the model
  writes ``pd.read_csv(make_up_kwarg=True)``, we flag pandas as unverified
  but not the specific kwarg. Checking kwargs would require resolving
  ``inspect.signature`` against the user's installed versions.
- Fabricated dataset names, paper citations, arXiv IDs, or prose claims.
  We only inspect URLs and Python code blocks; free-text claims pass through.
- Adversarial evasion. ``importlib`` under an alias, f-string import args,
  triple-backticks inside docstrings inside code fences, and malformed
  blocks that ``ast.parse`` rejects all escape detection silently. The
  detector targets the natural "confident plausible-looking fabrication"
  failure mode, not motivated evasion.

Design rationale
----------------
Deterministic post-hoc detection was chosen over asking the LLM to
self-flag, because the same metacognition that would catch the
hallucination is what failed to catch it the first time. We deliberately
stay narrow: catching 80% of the common cases with near-zero false
positives beats a "comprehensive" detector that engineers learn to ignore.

Opt out per-instance via ``Agent(..., enable_safeguards=False)``.
"""

from __future__ import annotations

import ast
import re
import sys

#: Semantic version of the detector contract. Bump the major when the set
#: of things-we-catch changes in a way users would notice.
SAFEGUARDS_VERSION = "1.0"

URL_PATTERN = re.compile(r"https?://\S+")

EXTERNAL_REFERENCE_WARNING = (
    "URLs detected in this response. The shipped adapters have no web "
    "retrieval, so URLs may be fabricated — verify before using."
)

# Library roots that are grounded by prompt context, factory validation, or
# tool schemas. Foreign roots outside this set + stdlib get flagged.
_GROUNDED_IMPORT_ROOTS = frozenset({"skrec", "scikit_rec", "scikit_rec_agent"})

# Fenced code blocks. Language tag is optional — many models drop it, and
# untagged blocks that parse as Python are worth scanning too.
_PYTHON_BLOCK_PATTERN = re.compile(
    r"```(?:python|py|py3|ipython|jupyter)?\s*\n(.*?)```",
    re.DOTALL | re.IGNORECASE,
)

# Strip line-magic / shell-magic lines (`%matplotlib`, `!pip install`, `?help`)
# before ast.parse. Without this, any such line kills the whole block.
# The trailing-`?` IPython form (`help?`) is rare enough to skip.
_NOTEBOOK_MAGIC_LINE = re.compile(r"^\s*[!%?].*$", re.MULTILINE)

# Cell magics whose bodies are NOT Python. If a block starts with one of
# these, skip scanning entirely — trying to ast.parse an HTML/SQL/shell body
# will either explode or, worse, parse partially and miss real imports.
_NON_PYTHON_CELL_MAGICS = frozenset({
    "html", "bash", "shell", "sh", "javascript", "js", "sql",
    "markdown", "md", "writefile", "perl", "ruby", "latex", "svg",
})

# Conservative bare-alias → package map. Models often emit `pd.read_csv(...)`
# or `sklearn.ensemble.RandomForestClassifier(...)` without the corresponding
# import, assuming the reader has it in scope. The map includes both classic
# aliases (`pd`, `np`) and full package names that are almost never used as
# local variable names (`torch`, `sklearn`). Kept narrow — single-letter
# aliases (`F`, `T`, `nn`) collide with common local variable names and
# would produce false positives.
_COMMON_ALIASES = {
    "pd": "pandas",
    "np": "numpy",
    "plt": "matplotlib",
    "sns": "seaborn",
    "tf": "tensorflow",
    "xgb": "xgboost",
    "lgb": "lightgbm",
    "jnp": "jax",
    "torch": "torch",
    "sklearn": "sklearn",
}


def detect_novel_urls(text: str, echoed: set[str]) -> set[str]:
    """URLs in `text` minus `echoed` (URLs the user typed this session)."""
    detected = {_normalize_url(u) for u in URL_PATTERN.findall(text)}
    echoed_norm = {_normalize_url(u) for u in echoed}
    return detected - echoed_norm


def detect_foreign_references(text: str) -> set[str]:
    """Foreign package roots referenced in any fenced Python block.

    Catches four leak modes:
      1. Static imports (`import torch`, `from sklearn import X`)
      2. Dynamic imports (`importlib.import_module("torch")`, `__import__`)
      3. Bare alias usage (`pd.read_csv(...)`) with `pd` not bound in any
         enclosing scope at the usage site
      4. Cell magics and line magics that would otherwise kill ast.parse

    Scope-aware: `def demo(pd): ...` binds `pd` only inside `demo`, not at
    module level, so bare `pd.read_csv(...)` outside the function still
    flags. Class-body scope follows Python's visibility rules loosely —
    methods do not inherit class-body bindings.
    """
    foreign: set[str] = set()
    stdlib = getattr(sys, "stdlib_module_names", frozenset())

    for match in _PYTHON_BLOCK_PATTERN.finditer(text):
        block = match.group(1)
        if _starts_with_non_python_cell_magic(block):
            continue
        cleaned = _NOTEBOOK_MAGIC_LINE.sub("", block)
        try:
            tree = ast.parse(cleaned)
        except SyntaxError:
            continue
        visitor = _AliasUsageVisitor(stdlib=stdlib)
        visitor.visit(tree)
        foreign |= visitor.foreign
    return foreign


# ---------- internals ----------


def _normalize_url(url: str) -> str:
    """Strip surrounding punctuation that `\\S+` greedily swallowed."""
    return url.rstrip(".,;:!?)>]}\"'")


def _starts_with_non_python_cell_magic(block: str) -> bool:
    """True if the first non-empty line is `%%X` with X a non-Python magic."""
    for line in block.splitlines():
        if not line.strip():
            continue
        match = re.match(r"^\s*%%(\w+)", line)
        if match:
            return match.group(1).lower() in _NON_PYTHON_CELL_MAGICS
        return False
    return False


def _attribute_root_name(node: ast.Attribute) -> str | None:
    """Walk to the base of a dotted chain; return the root Name id, if any."""
    current: ast.AST = node
    while isinstance(current, ast.Attribute):
        current = current.value
    return current.id if isinstance(current, ast.Name) else None


def _dynamic_import_target(call: ast.Call) -> str | None:
    """Extract module arg from `importlib.import_module(...)` / `__import__(...)`.

    Best-effort: misses aliased `importlib` imports, f-string args, and
    getattr-based reflection. Those are acceptable holes for a detector
    aimed at the common "lazy fabrication" case.
    """
    func = call.func
    is_import_module = (
        isinstance(func, ast.Attribute)
        and isinstance(func.value, ast.Name)
        and func.value.id == "importlib"
        and func.attr == "import_module"
    )
    is_builtin_import = isinstance(func, ast.Name) and func.id == "__import__"
    if not (is_import_module or is_builtin_import) or not call.args:
        return None
    first = call.args[0]
    if isinstance(first, ast.Constant) and isinstance(first.value, str):
        return first.value
    return None


class _AliasUsageVisitor(ast.NodeVisitor):
    """Single-pass, scope-aware walker over one parsed code block.

    Maintains a stack of scopes. Names are bound into the innermost scope
    by imports, assignments, function/class defs, loop variables, exception
    handlers, walrus expressions, `global`/`nonlocal`, and function/lambda
    parameters. Bare-alias attribute access is flagged only when the root
    name is unbound in every enclosing scope.
    """

    def __init__(self, stdlib: frozenset[str]):
        self._stdlib = stdlib
        # Scope frames are (kind, names). Kind is "module", "function", or
        # "class". Python hides class-body bindings from nested function
        # scopes: `class Foo: pd = X; def bar(self): pd` raises NameError.
        # The kind marker lets _is_bound skip class scopes when resolving
        # from inside a nested scope.
        self._scopes: list[tuple[str, set[str]]] = [("module", set())]
        self.foreign: set[str] = set()

    # ---- classification + binding helpers ----

    def _classify(self, module_path: str | None) -> None:
        if not module_path:
            return
        root = module_path.split(".", 1)[0]
        if root and root not in _GROUNDED_IMPORT_ROOTS and root not in self._stdlib:
            self.foreign.add(root)

    def _bind(self, name: str) -> None:
        self._scopes[-1][1].add(name)

    def _is_bound(self, name: str) -> bool:
        innermost_kind = self._scopes[-1][0]
        for i, (kind, names) in enumerate(self._scopes):
            is_innermost = i == len(self._scopes) - 1
            # Python class-body names are invisible to nested functions, but
            # visible to statements directly inside the class body itself.
            if kind == "class" and not is_innermost and innermost_kind != "class":
                continue
            if name in names:
                return True
        return False

    def _bind_target(self, target: ast.AST, scope: set[str] | None = None) -> None:
        target_scope = scope if scope is not None else self._scopes[-1][1]
        if isinstance(target, ast.Name):
            target_scope.add(target.id)
        elif isinstance(target, (ast.Tuple, ast.List)):
            for elt in target.elts:
                self._bind_target(elt, target_scope)
        elif isinstance(target, ast.Starred):
            self._bind_target(target.value, target_scope)

    def _visit_target(self, target: ast.AST) -> None:
        """Bind a target if it's a pure-write form (Name/Tuple/List/Starred).

        For Attribute/Subscript targets (`a.b = x`, `a[b] = x`), the target
        expression itself contains reads that must be checked — visit it
        without binding, since those forms don't introduce new names.
        """
        if isinstance(target, (ast.Name, ast.Tuple, ast.List, ast.Starred)):
            self._bind_target(target)
        else:
            self.visit(target)

    def _visit_callable(self, node: ast.AST) -> None:
        """Push a function scope seeded with function/lambda parameters.

        Parameter defaults (e.g. `def f(x=pd.default)`) evaluate in the
        enclosing scope, but generic_visit descends into them AFTER the
        function scope is pushed. This is semantically imprecise but
        harmless for alias detection: the pushed function scope only
        contains the params, not `pd`, so lookups fall through to the
        outer scope correctly.
        """
        scope: set[str] = set()
        args = node.args
        # Python 3.10+ — posonlyargs is always present.
        for params in (args.args, args.kwonlyargs, args.posonlyargs):
            for arg in params:
                scope.add(arg.arg)
        if args.vararg:
            scope.add(args.vararg.arg)
        if args.kwarg:
            scope.add(args.kwarg.arg)
        self._scopes.append(("function", scope))
        self.generic_visit(node)
        self._scopes.pop()

    # ---- scope creators ----

    def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
        self._bind(node.name)
        self._visit_callable(node)

    def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> None:
        self._bind(node.name)
        self._visit_callable(node)

    def visit_Lambda(self, node: ast.Lambda) -> None:
        self._visit_callable(node)

    def visit_ClassDef(self, node: ast.ClassDef) -> None:
        # Class name binds in the enclosing scope. The class body gets its
        # own scope, but _is_bound skips class scopes when resolving from
        # nested function scopes — matching Python's class-body visibility
        # rules (method bodies do NOT see class-level bindings).
        self._bind(node.name)
        self._scopes.append(("class", set()))
        self.generic_visit(node)
        self._scopes.pop()

    def visit_ListComp(self, node: ast.ListComp) -> None:
        self._visit_comp(node)

    def visit_SetComp(self, node: ast.SetComp) -> None:
        self._visit_comp(node)

    def visit_DictComp(self, node: ast.DictComp) -> None:
        self._visit_comp(node)

    def visit_GeneratorExp(self, node: ast.GeneratorExp) -> None:
        self._visit_comp(node)

    def _visit_comp(self, node: ast.AST) -> None:
        # Python comprehension scoping: the FIRST generator's iterable is
        # evaluated in the enclosing scope; everything else (subsequent
        # iterables, all `if` conditions, the element expression) is
        # evaluated inside the comprehension's own function-like scope.
        # So `[pd for pd in pd.items()]` must flag the outer `pd` in the
        # iterable but not the bound loop var in the element.
        if node.generators:
            self.visit(node.generators[0].iter)

        scope: set[str] = set()
        for generator in node.generators:
            self._bind_target(generator.target, scope)
        self._scopes.append(("function", scope))

        for generator in node.generators[1:]:
            self.visit(generator.iter)
        for generator in node.generators:
            for condition in generator.ifs:
                self.visit(condition)

        if isinstance(node, (ast.ListComp, ast.SetComp, ast.GeneratorExp)):
            self.visit(node.elt)
        elif isinstance(node, ast.DictComp):
            self.visit(node.key)
            self.visit(node.value)

        self._scopes.pop()

    # ---- binding statements ----

    def visit_Import(self, node: ast.Import) -> None:
        for alias in node.names:
            self._bind(alias.asname or alias.name.split(".", 1)[0])
            self._classify(alias.name)
        self.generic_visit(node)

    def visit_ImportFrom(self, node: ast.ImportFrom) -> None:
        for alias in node.names:
            self._bind(alias.asname or alias.name)
        if node.module and node.level == 0:
            self._classify(node.module)
        self.generic_visit(node)

    def visit_Assign(self, node: ast.Assign) -> None:
        # Visit RHS first so `pd = pd.read_csv(x)` catches the bare-alias read
        # before the LHS binding hides it. generic_visit would bind before
        # descending into the value because `targets` comes before `value` in
        # ast.Assign._fields — the exact shape of the original round-3 bug.
        self.visit(node.value)
        for target in node.targets:
            self._visit_target(target)

    def visit_AugAssign(self, node: ast.AugAssign) -> None:
        # `pd += pd.foo()` reads pd implicitly AND on the RHS before the
        # rebind. Visit value and target (target may be Attribute/Subscript).
        self.visit(node.value)
        self.visit(node.target)
        self._bind_target(node.target)

    def visit_AnnAssign(self, node: ast.AnnAssign) -> None:
        # Annotation and value are both evaluated before the bind.
        if node.value is not None:
            self.visit(node.value)
        self.visit(node.annotation)
        self._visit_target(node.target)

    def visit_NamedExpr(self, node: ast.NamedExpr) -> None:
        # Walrus: `(pd := pd.get_df())` — RHS evaluated before target binds.
        self.visit(node.value)
        self._bind_target(node.target)

    def visit_For(self, node: ast.For) -> None:
        # Iterable is evaluated in the enclosing scope BEFORE the loop var
        # is bound. `for pd in pd.items():` must flag the unbound pd on iter.
        self.visit(node.iter)
        self._visit_target(node.target)
        for stmt in node.body:
            self.visit(stmt)
        for stmt in node.orelse:
            self.visit(stmt)

    def visit_AsyncFor(self, node: ast.AsyncFor) -> None:
        self.visit(node.iter)
        self._visit_target(node.target)
        for stmt in node.body:
            self.visit(stmt)
        for stmt in node.orelse:
            self.visit(stmt)

    def visit_With(self, node: ast.With) -> None:
        # Context expressions evaluate before `as` target binds.
        for item in node.items:
            self.visit(item.context_expr)
            if item.optional_vars is not None:
                self._visit_target(item.optional_vars)
        for stmt in node.body:
            self.visit(stmt)

    def visit_AsyncWith(self, node: ast.AsyncWith) -> None:
        for item in node.items:
            self.visit(item.context_expr)
            if item.optional_vars is not None:
                self._visit_target(item.optional_vars)
        for stmt in node.body:
            self.visit(stmt)

    def visit_ExceptHandler(self, node: ast.ExceptHandler) -> None:
        if node.name:
            self._bind(node.name)
        self.generic_visit(node)

    def visit_Global(self, node: ast.Global) -> None:
        for name in node.names:
            self._bind(name)
        self.generic_visit(node)

    def visit_Nonlocal(self, node: ast.Nonlocal) -> None:
        for name in node.names:
            self._bind(name)
        self.generic_visit(node)

    # ---- detection ----

    def visit_Call(self, node: ast.Call) -> None:
        self._classify(_dynamic_import_target(node))
        self.generic_visit(node)

    def visit_Attribute(self, node: ast.Attribute) -> None:
        root = _attribute_root_name(node)
        if root and root in _COMMON_ALIASES and not self._is_bound(root):
            self._classify(_COMMON_ALIASES[root])
        self.generic_visit(node)
