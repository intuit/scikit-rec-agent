"""diagnose_training_failure tool + the failure registry.

The registry is an ordered list of ``FailurePattern`` records. ``_match`` walks
them in order and returns the first hit. Each pattern carries a category
enum, a short list of likely causes, and a ranked list of ``Fix`` records.

Two consumers:

- ``_quick_diagnose`` — synchronous, no-side-effects matcher used by
  ``train_model``'s except blocks to enrich the failure envelope with a
  ``hint`` and machine-readable ``category`` the moment the failure happens.
- ``TOOL_DIAGNOSE_TRAINING_FAILURE`` — LLM-facing tool that returns the full
  diagnosis (causes + ranked candidate fixes) and optionally auto-applies the
  top auto-retryable fix to retrain.
"""

from __future__ import annotations

import copy
import json
import re
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any

from scikit_rec_agent.session import FailureRecord
from scikit_rec_agent.tools import Tool, err, ok

# ---------------------------------------------------------------------------
# Registry primitives
# ---------------------------------------------------------------------------


@dataclass
class Fix:
    description: str
    action: dict[str, Any]
    auto_retryable: bool = False


@dataclass
class FailurePattern:
    name: str
    category: str
    pattern: re.Pattern
    causes: list[str]
    fixes: list[Fix]
    error_types: tuple[str, ...] = field(default_factory=tuple)


@dataclass
class Diagnosis:
    name: str
    category: str
    causes: list[str]
    fixes: list[Fix]

    @property
    def first_fix_description(self) -> str | None:
        return self.fixes[0].description if self.fixes else None


# ---------------------------------------------------------------------------
# The registry
# ---------------------------------------------------------------------------


_REGISTRY: list[FailurePattern] = [
    FailurePattern(
        name="single_class_target",
        category="single_class_target",
        pattern=re.compile(r"(y contains|contains only) (?:1 class|only one class)", re.IGNORECASE),
        causes=["Target column has only one unique value across the training fold."],
        fixes=[
            Fix(
                description="Drop the degenerate target from the candidate set.",
                action={"type": "drop_targets"},
                auto_retryable=False,
            ),
            Fix(
                description="Use a stratified split or oversample to ensure both classes are present.",
                action={"type": "advise_user", "text": "Re-run split_data with a stratified strategy."},
                auto_retryable=False,
            ),
        ],
    ),
    FailurePattern(
        name="nan_in_features",
        category="nan_in_features",
        pattern=re.compile(
            r"(input contains nan|cannot convert float nan to integer|contains nan|missing values)",
            re.IGNORECASE,
        ),
        causes=[
            "A feature column contains NaN values that the estimator cannot consume. "
            "XGBoost on its own handles NaN natively, but several upstream paths in scikit-rec "
            "(MultiOutputClassifier wrapper, sklearn check_array calls, embedding estimators) "
            "reject NaN before XGBoost ever sees it."
        ],
        fixes=[
            Fix(
                # NOTE: NaN handling is fundamentally a data-side problem — there's no
                # safe auto-retry that can fix it without changing the user's data.
                # Both fixes below are non-auto-retryable; the agent surfaces them to
                # the user and asks them to re-run create_datasets on the cleaned file.
                description="Drop fully-NaN columns and impute partial-NaN columns in the source data.",
                action={
                    "type": "advise_user",
                    "text": (
                        "transform_data has `drop_null_columns=True` by default — that handles "
                        "100% NaN columns. For partially-NaN columns, impute before "
                        "create_datasets: median for numeric features, mode for categorical, "
                        "or use pandas' fillna(0) if 0 is a sensible 'missing' value for your "
                        "domain. Identify the offending column from the error message itself."
                    ),
                },
                auto_retryable=False,
            ),
            Fix(
                description="Drop the offending feature column entirely if it's not load-bearing.",
                action={
                    "type": "advise_user",
                    "text": (
                        "If the NaN column isn't a useful feature (high-cardinality identifiers, "
                        "audit fields, or columns with >50% missing), it's often easier to drop it "
                        "than to impute. Pass an explicit `feature_columns` list to transform_data "
                        "to keep only the features you want."
                    ),
                },
                auto_retryable=False,
            ),
        ],
    ),
    FailurePattern(
        name="schema_mismatch",
        category="schema_mismatch",
        pattern=re.compile(
            r"(column not found.*(USER_ID|ITEM_ID|OUTCOME)|"
            r"required column.*(USER_ID|ITEM_ID|OUTCOME).*not present|"
            r"missing required column)",
            re.IGNORECASE,
        ),
        causes=["The data file is missing one of the contract's required columns."],
        fixes=[
            Fix(
                description="Run transform_data with the appropriate target_contract for this scorer.",
                action={"type": "call_tool", "tool": "transform_data"},
                auto_retryable=False,
            ),
        ],
    ),
    FailurePattern(
        name="wide_multioutput_underspecified",
        category="wide_multioutput_underspecified",
        pattern=re.compile(r"(at least 2 ITEM_\*|requires .*2 ITEM_\* target|<2 valid targets)", re.IGNORECASE),
        causes=["MultioutputScorer requires ≥2 ITEM_* target columns; the bundle has fewer."],
        fixes=[
            Fix(
                description="Verify all label columns made it through transform.",
                action={"type": "advise_user", "text": "Re-run profile_data and confirm target_columns."},
                auto_retryable=False,
            ),
            Fix(
                description="Switch to a single-output scorer (universal) with one target.",
                action={"type": "modify_config", "set": {"scorer_type": "universal"}},
                auto_retryable=True,
            ),
        ],
    ),
    FailurePattern(
        name="capability_mismatch",
        category="capability_mismatch",
        pattern=re.compile(
            r"(does not support BaseEmbeddingEstimator|"
            r"scorer .* does not support .* estimator|"
            r"incompatible scorer.*estimator)",
            re.IGNORECASE,
        ),
        causes=["The chosen scorer does not accept the chosen estimator family."],
        fixes=[
            Fix(
                description="Swap to a tabular estimator (XGBoost) compatible with most scorers.",
                action={
                    "type": "modify_config",
                    "set": {
                        "estimator_config": {
                            "ml_task": "classification",
                            "xgboost": {"n_estimators": 100, "max_depth": 5, "learning_rate": 0.1},
                        }
                    },
                },
                auto_retryable=True,
            ),
            Fix(
                description="Swap to UniversalScorer which accepts the broadest estimator set.",
                action={"type": "modify_config", "set": {"scorer_type": "universal"}},
                auto_retryable=True,
            ),
        ],
    ),
    FailurePattern(
        name="oom",
        category="oom",
        pattern=re.compile(r"(MemoryError|Unable to allocate.* (?:GiB|MiB)|out of memory)", re.IGNORECASE),
        error_types=("MemoryError",),
        causes=["Data or model exceeded available memory."],
        fixes=[
            Fix(
                description="Reduce n_estimators / batch_size to lower memory footprint.",
                action={
                    "type": "modify_config",
                    "set": {"estimator_config.xgboost.n_estimators": 50},
                },
                auto_retryable=True,
            ),
            Fix(
                description="Sample the dataset before training.",
                action={"type": "advise_user", "text": "Sample the source data and re-run create_datasets."},
                auto_retryable=False,
            ),
        ],
    ),
    FailurePattern(
        name="numerical_instability",
        category="numerical_instability",
        pattern=re.compile(
            r"(SVD did not converge|LinAlgError|singular matrix|"
            r"overflow encountered in (?:exp|log)|invalid value encountered)",
            re.IGNORECASE,
        ),
        causes=["Optimizer hit a numerically unstable configuration: high LR, collinearity, or outliers."],
        fixes=[
            Fix(
                description="Lower the learning_rate to stabilise optimisation.",
                action={"type": "modify_config", "set": {"estimator_config.xgboost.learning_rate": 0.01}},
                auto_retryable=True,
            ),
            Fix(
                description="Standardise features and drop fully-collinear columns.",
                action={"type": "advise_user", "text": "Pre-process features outside the agent."},
                auto_retryable=False,
            ),
        ],
    ),
    FailurePattern(
        name="missing_dependency",
        category="missing_dependency",
        pattern=re.compile(r"No module named '?(?:torch|lightgbm|xgboost)'?", re.IGNORECASE),
        error_types=("ModuleNotFoundError", "ImportError"),
        causes=["A backend (torch / lightgbm / xgboost) is required but not installed."],
        fixes=[
            Fix(
                description="Install the missing backend (e.g. `pip install scikit-rec[torch]`).",
                action={"type": "advise_user", "text": "pip install the missing backend then retry."},
                auto_retryable=False,
            ),
            Fix(
                description="Swap to a tabular estimator that uses an already-installed backend.",
                action={
                    "type": "modify_config",
                    "set": {
                        "estimator_config": {
                            "ml_task": "classification",
                            "xgboost": {"n_estimators": 100, "max_depth": 5, "learning_rate": 0.1},
                        }
                    },
                },
                auto_retryable=False,
            ),
        ],
    ),
    FailurePattern(
        name="object_dtype_features",
        category="object_dtype_features",
        pattern=re.compile(
            r"(DataFrame\.dtypes for data must be (?:int|float|bool|category)|"
            r"could not convert string to float|"
            r"Invalid columns?:\s*\w+:\s*object|"
            r"DataFrame\.dtypes must be int, float)",
            re.IGNORECASE,
        ),
        causes=[
            "One or more feature columns are object/string dtype, but the "
            "estimator (XGBoost / LightGBM) only accepts numeric features.",
        ],
        fixes=[
            Fix(
                description=(
                    "Decide which object-dtype columns matter. Drop the ones "
                    "you don't need; one-hot or label-encode the ones you do. "
                    "Re-run create_datasets on the cleaned file."
                ),
                action={
                    "type": "advise_user",
                    "text": (
                        "Look at the error message — it names the offending columns "
                        "(e.g. 'Invalid columns: wholesale: object, acct_attached: object'). "
                        "For each: drop it if it's not a useful feature, or convert to "
                        "numeric (one-hot encode strings, label-encode ordered categories) "
                        "before calling create_datasets."
                    ),
                },
                auto_retryable=False,
            ),
        ],
    ),
    FailurePattern(
        name="wide_bundle_eval_unsupported",
        category="wide_bundle_eval_unsupported",
        pattern=re.compile(
            r"(Column\(s\) \['ITEM_ID', 'OUTCOME'\]|"
            r"WideBundleEvalUnsupported|"
            r"can't auto-build eval_kwargs)",
            re.IGNORECASE,
        ),
        causes=[
            "The 'simple' evaluator's auto-built eval_kwargs derive "
            "logged_items / logged_rewards from long-format ITEM_ID / OUTCOME "
            "columns, which wide multi-output / multi-class bundles don't carry.",
        ],
        fixes=[
            Fix(
                description=(
                    "Pass `eval_kwargs` explicitly to evaluate_model with "
                    "per-target arrays, or evaluate per-ITEM_* column outside "
                    "the agent and report the metrics yourself."
                ),
                action={
                    "type": "advise_user",
                    "text": (
                        "evaluate_model on wide bundles needs explicit eval_kwargs. "
                        "Build logged_items and logged_rewards from the bundle's "
                        "ITEM_* columns and pass them in."
                    ),
                },
                auto_retryable=False,
            ),
        ],
    ),
    FailurePattern(
        name="evaluator_needs_eval_kwargs",
        category="evaluator_needs_eval_kwargs",
        pattern=re.compile(
            r"eval_kwargs is required to compute modified rewards",
            re.IGNORECASE,
        ),
        causes=[
            "The chosen evaluator (IPS / SNIPS / DR / direct_method / "
            "replay_match / policy_weighted) requires explicit "
            "logged_items / logged_rewards / logging_proba / "
            "expected_rewards. Only the 'simple' evaluator can auto-build "
            "these from validation interactions.",
        ],
        fixes=[
            Fix(
                description=(
                    "Switch evaluator_type to 'simple' (auto-builds logged "
                    "arrays from the bundle's validation interactions), or "
                    "pass eval_kwargs explicitly with the per-target arrays."
                ),
                action={"type": "modify_config", "set": {"evaluator_type": "simple"}},
                auto_retryable=True,
            ),
            Fix(
                description=(
                    "If you need IPS / SNIPS / DR specifically, build "
                    "logged_items, logged_rewards, and logging_proba arrays "
                    "from your logged data and pass them as eval_kwargs."
                ),
                action={
                    "type": "advise_user",
                    "text": (
                        "These evaluators need policy logging info (logging_proba) "
                        "you have to provide — they can't be derived from raw "
                        "interactions alone."
                    ),
                },
                auto_retryable=False,
            ),
        ],
    ),
    FailurePattern(
        name="logged_items_shape_mismatch",
        category="logged_items_shape_mismatch",
        pattern=re.compile(
            r"Mismatch in N dimension: target_proba \(\d+\) vs logged_items \(\d+\)",
            re.IGNORECASE,
        ),
        causes=[
            "logged_items's first dimension must match the recommender's "
            "scored instance count (target_proba.shape[0]). The agent's "
            "auto-build path may have aggregated by user when the scorer "
            "produced per-row scores, or vice versa.",
        ],
        fixes=[
            Fix(
                description=(
                    "This indicates an internal shape bug in the agent's "
                    "_build_eval_kwargs_from_validation helper, not a user "
                    "error. File against the agent repo if you see this."
                ),
                action={
                    "type": "advise_user",
                    "text": (
                        "Internal shape bug — please report. Workaround: pass "
                        "eval_kwargs explicitly to evaluate_model with per-row "
                        "logged_items of shape (N, 1) where N equals the row count "
                        "in the bundle's validation interactions."
                    ),
                },
                auto_retryable=False,
            ),
        ],
    ),
    FailurePattern(
        name="dtype_mismatch_across_files",
        category="dtype_mismatch_across_files",
        pattern=re.compile(
            r"('<' not supported between instances of 'str' and 'int'|"
            r"'>' not supported between instances of 'str' and 'int'|"
            r"unorderable types: str\(\) [<>] int\(\)|"
            r"comparing different types)",
            re.IGNORECASE,
        ),
        causes=[
            "Two files in the bundle (interactions / users / items) carry the "
            "same ID column with different dtypes (e.g. ITEM_ID is int in the "
            "items file but str in the interactions file). scikit-rec joins on "
            "those IDs at evaluation time and the mixed types break ordered "
            "comparisons.",
        ],
        fixes=[
            Fix(
                description=(
                    "Cast the offending ID column to a consistent dtype across "
                    "ALL files (str is the safest choice, since user / item IDs "
                    "are categorical even when they look numeric). Re-write the "
                    "files with the cast applied, then re-run create_datasets."
                ),
                action={
                    "type": "advise_user",
                    "text": (
                        "Pick str for USER_ID and ITEM_ID across interactions, users, "
                        "and items files. Pandas defaults to int when the IDs are all "
                        "digits, which silently mismatches the items / users tables."
                    ),
                },
                auto_retryable=False,
            ),
        ],
    ),
    FailurePattern(
        name="data_shape_mismatch",
        category="data_shape_mismatch",
        pattern=re.compile(
            r"(Found input variables with inconsistent numbers of samples|shapes .* not aligned)", re.IGNORECASE
        ),
        causes=["X / y row counts disagree — usually a stale split or a join that dropped rows."],
        fixes=[
            Fix(
                description="Re-run transform_data so X and y are produced from the same source frame.",
                action={"type": "call_tool", "tool": "transform_data"},
                auto_retryable=False,
            ),
        ],
    ),
    FailurePattern(
        name="degenerate_target_or_features",
        category="degenerate_target_or_features",
        pattern=re.compile(r"(no positive samples|all features.*constant|empty training set)", re.IGNORECASE),
        causes=["After splitting, the training fold has no positives or only constant features."],
        fixes=[
            Fix(
                description="Use a stratified split to retain positives in train.",
                action={"type": "advise_user", "text": "Re-run split_data with random_split_per_user."},
                auto_retryable=False,
            ),
        ],
    ),
    FailurePattern(
        name="timeout",
        category="timeout",
        pattern=re.compile(r"(timeout|exceeded.*max_train_seconds)", re.IGNORECASE),
        error_types=("TimeoutError",),
        causes=["Training did not finish within the per-method timeout."],
        fixes=[
            Fix(
                description="Reduce n_estimators / epochs.",
                action={"type": "modify_config", "set": {"estimator_config.xgboost.n_estimators": 30}},
                auto_retryable=True,
            ),
        ],
    ),
]


_UNKNOWN = FailurePattern(
    name="unknown",
    category="unknown",
    pattern=re.compile(r".*"),
    causes=["No registered pattern matched the error message."],
    fixes=[],
)


# ---------------------------------------------------------------------------
# Matcher
# ---------------------------------------------------------------------------


def _match(envelope: dict[str, Any]) -> Diagnosis:
    msg = str(envelope.get("message", "") or "")
    error_type = str(envelope.get("error_type", "") or "")
    for pat in _REGISTRY:
        if pat.error_types and error_type and error_type not in pat.error_types:
            # Type filter rejects this pattern, but only when error_type is known.
            continue
        if pat.pattern.search(msg):
            return Diagnosis(
                name=pat.name,
                category=pat.category,
                causes=list(pat.causes),
                fixes=list(pat.fixes),
            )
    return Diagnosis(
        name=_UNKNOWN.name,
        category=_UNKNOWN.category,
        causes=[*_UNKNOWN.causes, msg] if msg else list(_UNKNOWN.causes),
        fixes=[],
    )


def _quick_diagnose(exc: BaseException) -> Diagnosis:
    """Synchronous helper used by train_model's except blocks."""
    envelope = {"error_type": type(exc).__name__, "message": str(exc)}
    return _match(envelope)


def quick_diagnose_envelope(envelope: dict[str, Any]) -> Diagnosis:
    """Public alias for callers that already hold an error envelope."""
    return _match(envelope)


# ---------------------------------------------------------------------------
# Failure history bookkeeping
# ---------------------------------------------------------------------------


def record_failure(
    session,
    model_name: str,
    config: dict[str, Any],
    bundle_args: dict[str, Any],
    envelope: dict[str, Any],
    diagnosis: Diagnosis,
) -> FailureRecord:
    rec = FailureRecord(
        model_name=model_name,
        config=copy.deepcopy(config),
        bundle_args=dict(bundle_args),
        error_envelope=dict(envelope),
        diagnosis_category=diagnosis.category,
        timestamp=datetime.now(timezone.utc).isoformat(),
    )
    session.failure_history.append(rec)
    return rec


def _last_failure_for_model(session, model_name: str) -> FailureRecord | None:
    for rec in reversed(session.failure_history):
        if rec.model_name == model_name:
            return rec
    return None


def _retries_for_model(session, model_name: str) -> int:
    return sum(1 for r in session.failure_history if r.model_name == model_name and r.fix_applied is not None)


def _previously_attempted_fix_signatures(session, model_name: str) -> set[str]:
    sigs: set[str] = set()
    for rec in session.failure_history:
        if rec.model_name == model_name and rec.fix_applied is not None:
            sigs.add(_fix_signature_dict(rec.fix_applied))
    return sigs


# ---------------------------------------------------------------------------
# Fix application
# ---------------------------------------------------------------------------


def _set_nested(d: dict[str, Any], path: str, value: Any) -> None:
    parts = path.split(".")
    cursor = d
    for p in parts[:-1]:
        nxt = cursor.get(p)
        if not isinstance(nxt, dict):
            nxt = {}
            cursor[p] = nxt
        cursor = nxt
    cursor[parts[-1]] = value


def _apply_fix(fix: Fix, config: dict[str, Any]) -> dict[str, Any]:
    new_config = copy.deepcopy(config)
    action = fix.action
    if action.get("type") == "modify_config":
        for path, value in action.get("set", {}).items():
            if "." in path:
                _set_nested(new_config, path, value)
            else:
                new_config[path] = value
    return new_config


def _fix_to_dict(fix: Fix) -> dict[str, Any]:
    return {
        "description": fix.description,
        "action": fix.action,
        "auto_retryable": fix.auto_retryable,
    }


def _fix_signature(fix: Fix) -> str:
    return _fix_signature_dict(_fix_to_dict(fix))


def _fix_signature_dict(fix_dict: dict[str, Any]) -> str:
    return json.dumps(
        {"description": fix_dict.get("description"), "action": fix_dict.get("action")},
        sort_keys=True,
        default=str,
    )


# ---------------------------------------------------------------------------
# Tool entrypoint
# ---------------------------------------------------------------------------


def _diagnose_training_failure(
    model_name: str,
    session,
    error_envelope: dict[str, Any] | None = None,
    auto_retry: bool = False,
    max_retries: int = 2,
) -> dict[str, Any]:
    if error_envelope is None:
        rec = _last_failure_for_model(session, model_name)
        if rec is None:
            return err(
                "NoFailureFound",
                f"No recent failed train_model for '{model_name}' in session.failure_history.",
            )
        error_envelope = rec.error_envelope
        last_record = rec
    else:
        last_record = _last_failure_for_model(session, model_name)

    diagnosis = _match(error_envelope)
    tried = _previously_attempted_fix_signatures(session, model_name)
    candidate_fixes = [f for f in diagnosis.fixes if _fix_signature(f) not in tried]

    payload: dict[str, Any] = {
        "model_name": model_name,
        "category": diagnosis.category,
        "name": diagnosis.name,
        "causes": diagnosis.causes,
        "candidate_fixes": [_fix_to_dict(f) for f in candidate_fixes],
        "previously_attempted_fixes": [json.loads(s) for s in sorted(tried)],
        "auto_retried": False,
    }

    if not auto_retry:
        return ok(payload)

    if not candidate_fixes:
        payload["retry_blocked_reason"] = (
            "No candidate fixes available — diagnosis is unknown or all fixes already tried."
        )
        return ok(payload)

    retry_count = _retries_for_model(session, model_name)
    if retry_count >= max_retries:
        payload["retry_blocked_reason"] = f"max_retries={max_retries} reached for '{model_name}'."
        return ok(payload)

    top_fix = next(
        (f for f in candidate_fixes if f.auto_retryable and f.action.get("type") == "modify_config"),
        None,
    )
    if top_fix is None:
        # If a candidate is auto_retryable but its action type isn't a config
        # mutation we know how to apply (modify_config), surface it as a
        # blocker instead of silently invoking _apply_fix as a no-op and
        # looping until max_retries with the same error.
        non_modify = [f for f in candidate_fixes if f.auto_retryable and f.action.get("type") != "modify_config"]
        if non_modify:
            payload["retry_blocked_reason"] = (
                "Top auto-retryable fix's action type is not 'modify_config' "
                f"({non_modify[0].action.get('type')!r}); _apply_fix doesn't know "
                "how to apply non-config-mutating actions. Mark such fixes "
                "auto_retryable=False or extend _apply_fix to handle the action type."
            )
        else:
            payload["retry_blocked_reason"] = "No auto-retryable fixes available; user approval needed."
        return ok(payload)

    if last_record is None:
        payload["retry_blocked_reason"] = (
            "Cannot auto-retry: no previous train_model attempt recorded for this model_name."
        )
        return ok(payload)

    new_config = _apply_fix(top_fix, last_record.config)
    from scikit_rec_agent.tools.training import TOOL_TRAIN_MODEL  # late import to avoid cycle

    retry_result = TOOL_TRAIN_MODEL.fn(
        model_name=model_name,
        config=new_config,
        session=session,
        **last_record.bundle_args,
    )

    last_record.fix_applied = _fix_to_dict(top_fix)
    last_record.retry_outcome = retry_result

    payload["auto_retried"] = True
    payload["applied_fix"] = _fix_to_dict(top_fix)
    payload["retry_result"] = retry_result
    return ok(payload)


TOOL_DIAGNOSE_TRAINING_FAILURE = Tool(
    name="diagnose_training_failure",
    description=(
        "Inspect a failed train_model error envelope and return a structured diagnosis "
        "with ranked candidate fixes. Optionally auto-applies the top fix and re-trains. "
        "Use this whenever train_model returns status='error' instead of guessing a "
        "config change manually. Bounded retries prevent loops."
    ),
    input_schema={
        "type": "object",
        "properties": {
            "model_name": {"type": "string"},
            "error_envelope": {
                "type": "object",
                "description": (
                    "The full error envelope returned by train_model. Pass it verbatim. "
                    "If omitted, the tool looks up the most recent failed train_model "
                    "call in session.failure_history."
                ),
            },
            "auto_retry": {
                "type": "boolean",
                "default": False,
                "description": "If True, apply the top auto-retryable fix and re-call train_model.",
            },
            "max_retries": {
                "type": "integer",
                "default": 2,
                "description": "Hard cap on retries per model_name in this session.",
            },
        },
        "required": ["model_name"],
    },
    fn=_diagnose_training_failure,
)
