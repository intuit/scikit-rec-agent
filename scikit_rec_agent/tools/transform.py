"""transform_data tool.

Reshape a raw CSV/Parquet into a scikit-rec contract format. The LLM-facing
contract is the ``target_contract`` enum; small composable internal ops do
the work. Source shape is auto-detected from the data.

Output is a canonical CSV (or Parquet) at ``output_path`` plus a final
self-validation against the target contract. Schemas are NOT written here —
``create_datasets`` regenerates them from dtypes on the next call.
"""

from __future__ import annotations

import os
import re
from typing import Any

import pandas as pd

from scikit_rec_agent.tools import Tool, err, ok

# ---------------------------------------------------------------------------
# Target contract definitions
# ---------------------------------------------------------------------------

_TARGET_CONTRACTS = (
    "long_interactions",
    "long_with_timestamp",
    "long_multi_reward",
    "wide_multioutput",
    "multiclass",
    "prebuilt_sequences",
    "sessions",
    "users_features",
    "items_features",
)

# What the validator demands of the OUTPUT for each contract.
_CONTRACT_REQUIRED_COLS: dict[str, tuple[str, ...]] = {
    "long_interactions": ("USER_ID", "ITEM_ID", "OUTCOME"),
    "long_with_timestamp": ("USER_ID", "ITEM_ID", "OUTCOME", "TIMESTAMP"),
    "long_multi_reward": ("USER_ID", "ITEM_ID", "OUTCOME"),
    "wide_multioutput": ("USER_ID",),
    "multiclass": ("USER_ID", "ITEM_ID"),
    "prebuilt_sequences": ("USER_ID", "ITEM_SEQUENCE"),
    "sessions": ("USER_ID", "SESSION_SEQUENCES"),
    "users_features": ("USER_ID",),
    "items_features": ("ITEM_ID",),
}

_CONTRACTS_REQUIRING_DEDUPE = (
    "wide_multioutput",
    "multiclass",
    "prebuilt_sequences",
    "sessions",
    "users_features",
    "items_features",
)


# ---------------------------------------------------------------------------
# Operation primitives
# ---------------------------------------------------------------------------


def _op_rename_columns(df: pd.DataFrame, mapping: dict[str, str]) -> pd.DataFrame:
    return df.rename(columns=mapping)


def _op_bulk_rename_targets(
    df: pd.DataFrame,
    pattern: str,
    replacement: str = r"ITEM_\1",
) -> pd.DataFrame:
    """Apply a regex-based bulk rename. Default replacement turns
    ``label_foo`` into ``ITEM_foo`` for the canonical wide-multioutput
    pattern."""
    rx = re.compile(pattern)
    new_cols = {}
    for c in df.columns:
        m = rx.fullmatch(c)
        if m:
            new_cols[c] = rx.sub(replacement, c)
    return df.rename(columns=new_cols)


def _op_pivot_to_wide(
    df: pd.DataFrame,
    user_col: str,
    item_col: str,
    outcome_col: str,
) -> pd.DataFrame:
    """Long → wide pivot. Non-{user, item, outcome} columns are preserved as
    user-level features by joining the per-user first observation back onto
    the pivoted frame. Without this preservation a downstream multioutput
    or multiclass scorer can't access user features that were carried in
    the long file (it would have to re-merge from a separate users file
    every time, which the wide contract doesn't support — see
    create_datasets's auto-merge guard).
    """
    pivoted = df.pivot_table(
        index=user_col,
        columns=item_col,
        values=outcome_col,
        aggfunc="max",
        fill_value=0,
    )
    pivoted.columns = [f"ITEM_{c}" for c in pivoted.columns]
    pivoted = pivoted.reset_index().rename(columns={user_col: "USER_ID"})

    feature_cols = [c for c in df.columns if c not in {user_col, item_col, outcome_col}]
    if feature_cols:
        # Take the first observed value per user. Wide format is 1-row-per-user;
        # if the long source had per-user constant features (the common case),
        # `first` is exact. If features vary per row inside a user, `first` is
        # an explicit, predictable choice — callers who need a different
        # reduction should aggregate before transform_data.
        per_user_features = (
            df[[user_col, *feature_cols]]
            .drop_duplicates(subset=[user_col], keep="first")
            .rename(columns={user_col: "USER_ID"})
        )
        pivoted = pivoted.merge(per_user_features, on="USER_ID", how="left")
    return pivoted


def _op_melt_to_long(
    df: pd.DataFrame,
    user_col: str,
    item_columns: list[str],
) -> pd.DataFrame:
    """Wide → long melt. Non-target columns ride along as id_vars so that
    user-level features (and any other side columns) are preserved on every
    (user, item) row in the long output. Without this, a wide-multioutput
    file with embedded features melts into a stripped USER_ID/ITEM_ID/OUTCOME
    triple and the universal / independent scorers downstream lose the
    feature signal.

    Validates that ``item_columns`` actually exist in the source frame and
    that ``ITEM_ID`` / ``OUTCOME`` aren't already-existing columns (which
    would conflict with the melt's ``var_name`` / ``value_name`` and
    surface as opaque pandas errors). The most common cause is calling
    ``transform_data`` on a file that's already in long format — catch
    that explicitly with an actionable message.
    """
    missing_targets = [c for c in item_columns if c not in df.columns]
    if missing_targets:
        existing_targets = [c for c in df.columns if c not in {user_col, "USER_ID", "ITEM_ID", "OUTCOME"}]
        raise ValueError(
            f"melt_to_long: target columns {missing_targets!r} are not in the source DataFrame. "
            f"Available non-id columns: {existing_targets!r}. The most common cause is "
            "calling transform_data with target_contract='long_interactions' on a file "
            "that's already long-format (carries USER_ID / ITEM_ID / OUTCOME). If the "
            "source is already long, skip the transform; if it's wide, double-check "
            "target_columns names match the actual ITEM_* / label_* column names."
        )
    if "ITEM_ID" in df.columns or "OUTCOME" in df.columns:
        already = sorted({c for c in ("ITEM_ID", "OUTCOME") if c in df.columns})
        raise ValueError(
            f"melt_to_long would emit columns {already!r}, but the source DataFrame already "
            f"has them — most likely the file is already in long format. Skip the "
            "long_interactions transform on already-long input, or rename / drop the "
            "conflicting source columns first."
        )

    feature_cols = [c for c in df.columns if c != user_col and c not in item_columns]
    melted = df.melt(
        id_vars=[user_col, *feature_cols],
        value_vars=item_columns,
        var_name="ITEM_ID",
        value_name="OUTCOME",
    )
    return melted.rename(columns={user_col: "USER_ID"})


def _op_aggregate_to_sequences(
    df: pd.DataFrame,
    user_col: str,
    item_col: str,
    outcome_col: str | None,
    timestamp_col: str | None,
) -> pd.DataFrame:
    if timestamp_col is None or timestamp_col not in df.columns:
        raise ValueError(
            "aggregate_to_sequences requires a timestamp_col present in the data; sequence ordering must not be silent."
        )
    sorted_df = df.sort_values([user_col, timestamp_col])
    agg: dict[str, Any] = {item_col: list}
    if outcome_col and outcome_col in sorted_df.columns:
        agg[outcome_col] = list
    grouped = sorted_df.groupby(user_col).agg(agg).reset_index()
    rename = {user_col: "USER_ID", item_col: "ITEM_SEQUENCE"}
    if outcome_col and outcome_col in grouped.columns:
        rename[outcome_col] = "OUTCOME_SEQUENCE"
    return grouped.rename(columns=rename)


def _op_parse_timestamp(df: pd.DataFrame, timestamp_col: str) -> pd.DataFrame:
    """Parse a timestamp column to canonical Unix-second TIMESTAMP.

    pandas coerces unparseable timestamp values to ``NaT``, which then
    silently becomes ``INT64_MIN`` (-9223372036854775808) when cast to
    int64. We catch that explicitly: any rows with NaT after coercion
    raise a ValueError naming the column and the count of bad rows so
    the caller (transform_data tool) can surface it to the user.
    """
    out = df.copy()
    parsed = pd.to_datetime(out[timestamp_col], errors="coerce")
    n_invalid = int(parsed.isna().sum())
    if n_invalid > 0:
        # Inspect a few of the bad source values to make the error actionable.
        bad_mask = parsed.isna()
        sample = out.loc[bad_mask, timestamp_col].head(3).tolist()
        raise ValueError(
            f"timestamp_column '{timestamp_col}' has {n_invalid} unparseable "
            f"value(s) (e.g. {sample!r}). pandas coerces these to NaT, which "
            f"becomes INT64_MIN when cast to int64 — fix or drop the bad "
            f"rows before transform_data."
        )
    out["TIMESTAMP"] = (parsed.astype("int64") // 10**9).astype(str)
    if timestamp_col != "TIMESTAMP" and timestamp_col in out.columns:
        out = out.drop(columns=[timestamp_col])
    return out


def _op_cast_dtypes(df: pd.DataFrame, casts: dict[str, str]) -> pd.DataFrame:
    out = df.copy()
    for col, target in casts.items():
        if col not in out.columns:
            continue
        if target == "str":
            out[col] = out[col].astype(str)
        elif target == "float":
            out[col] = out[col].astype(float)
        elif target == "int":
            out[col] = out[col].astype("int64")
    return out


def _op_dedupe_user_id(df: pd.DataFrame) -> pd.DataFrame:
    if "USER_ID" not in df.columns:
        return df
    return df.drop_duplicates(subset=["USER_ID"], keep="first").reset_index(drop=True)


def _op_drop_null_columns(df: pd.DataFrame) -> pd.DataFrame:
    return df.dropna(axis=1, how="all")


_BINARY_VALUES: set[float] = {0.0, 1.0}


def _classify_wide_multioutput_targets(
    df: pd.DataFrame,
) -> tuple[
    list[tuple[str, list]],
    list[tuple[str, float]],
    list[str],
]:
    """Inspect the post-pivot ITEM_* targets and classify them.

    Mirrors scikit-rec's ``MultioutputScorer._validate_targets`` so the
    agent can fail (or auto-drop) at the transform stage rather than
    letting users sail into a confusing fit-time error halfway through a
    sweep. Return three lists:

    - ``bad_non_binary``: ``[(column, sample_unique_values)]`` — columns
      with values outside ``{0, 1}`` (also catches multi-class targets
      since {0, 1, 2, ...} fails the binary filter). Migration guidance
      comes from upstream's exact error wording.
    - ``degenerate``: ``[(column, only_value)]`` — columns with a single
      unique value across the post-pivot frame. Caller decides whether
      to drop or block.
    - ``ok``: column names that pass both checks.
    """
    bad_non_binary: list[tuple[str, list]] = []
    degenerate: list[tuple[str, float]] = []
    ok_cols: list[str] = []
    item_cols = [c for c in df.columns if c.startswith("ITEM_") and c != "ITEM_ID"]
    for col in item_cols:
        # Coerce string-typed numeric columns (e.g. CSV round-trips of
        # already-binary {0, 1} that pandas read as object dtype) before
        # rejecting as non-binary. Without this, a column with values
        # ``["1", "0", "1", "0"]`` would surface as a misleading
        # "non-binary" error when the real fix is just a dtype cast.
        # ``errors='coerce'`` turns genuinely non-numeric strings into
        # NaN — those still fall through to the bad_non_binary branch
        # because the post-coerce uniques include {nan} only when EVERY
        # value was non-numeric.
        if not pd.api.types.is_numeric_dtype(df[col]):
            coerced = pd.to_numeric(df[col], errors="coerce")
            non_null_count = int(coerced.notna().sum())
            if non_null_count > 0 and non_null_count == int(df[col].notna().sum()):
                # All originally-non-null values coerced cleanly. Treat
                # as numeric for the rest of the classification.
                values_series = coerced
            else:
                # At least one non-coercible value (e.g. 'yes'/'no'
                # strings) — surface the original samples in the error,
                # not the post-coerce NaNs.
                sample = df[col].dropna().unique()[:5].tolist()
                bad_non_binary.append((col, sample))
                continue
        else:
            values_series = df[col]
        unique = set(values_series.dropna().unique().tolist())
        if not unique:
            # All-NaN — treat as degenerate with NaN as the seen value.
            degenerate.append((col, float("nan")))
            continue
        # Cast keys to float so {0, 1} and {0.0, 1.0} compare equal.
        unique_float = {float(v) for v in unique}
        extra = unique_float - _BINARY_VALUES
        if extra:
            sample = sorted(extra)[:5]
            bad_non_binary.append((col, sample))
            continue
        if len(unique_float) < 2:
            only = next(iter(unique_float))
            degenerate.append((col, only))
            continue
        ok_cols.append(col)
    return bad_non_binary, degenerate, ok_cols


def _op_select_features(
    df: pd.DataFrame,
    keep_cols: list[str],
    feature_columns: list[str] | None,
    feature_pattern: str | None,
) -> pd.DataFrame:
    keep = list(keep_cols)
    if feature_columns:
        keep += [c for c in feature_columns if c in df.columns and c not in keep]
    if feature_pattern:
        rx = re.compile(feature_pattern)
        keep += [c for c in df.columns if rx.fullmatch(c) and c not in keep]
    keep = [c for c in keep if c in df.columns]
    return df[keep]


# ---------------------------------------------------------------------------
# Source shape detection
# ---------------------------------------------------------------------------


def _detect_source_shape(
    df: pd.DataFrame,
    user_id_column: str | None,
    item_id_column: str | None,
) -> str:
    """Cheap signals only — never block on detection failure, just return a
    best guess. The contract validator at the end is the actual safety net."""
    user_col = user_id_column or ("USER_ID" if "USER_ID" in df.columns else None)
    if user_col is None or user_col not in df.columns:
        return "unknown"

    list_cols = [c for c in df.columns if df[c].apply(lambda v: isinstance(v, (list, tuple))).any()]
    if list_cols:
        return "pre_aggregated_sequences"

    rows_per_user = df.groupby(user_col).size().max() if len(df) else 0
    if rows_per_user > 1:
        return "long"

    item_star = [c for c in df.columns if c.startswith("ITEM_") and c != "ITEM_ID"]
    if len(item_star) >= 2:
        return "wide_already"

    label_cols = [c for c in df.columns if c.lower().startswith("label_")]
    if len(label_cols) >= 2:
        return "wide_with_other_prefix"

    return "single_row_per_user"


# ---------------------------------------------------------------------------
# Per-contract transform plans
# ---------------------------------------------------------------------------


def _to_long_interactions(
    df, user_id_column, item_id_column, outcome_column, target_columns=None, target_rename_pattern=None, **_
):
    """Long-format reshape. Two source paths:

    - already long (item_id_column + outcome_column supplied): rename + cast.
    - wide (target_columns or target_rename_pattern supplied, no item_id):
      melt the indicated columns into (USER_ID, ITEM_ID, OUTCOME) rows. Side
      feature columns are preserved as id_vars by ``_op_melt_to_long``.

    Reject the ambiguous combination of ``item_id_column`` AND
    ``target_columns`` upfront — both signal different intents (rename
    long input vs melt wide input) and downstream the rename path
    silently wins, producing a USER_ID-less frame and a confusing
    ContractViolation. Catch the contradiction with a clear hint so
    LLM callers stop looping on the wrong arg combination.

    Return shape (BREAKING vs pre-0.3.x agent versions): all per-contract
    transform plans now return ``(df, ops, extras)`` — a 3-tuple, not the
    2-tuple ``(df, ops)`` used before the multioutput rework. The
    ``extras`` dict carries side-channel information (e.g. ``dropped_targets``
    on the wide path). All internal callers were updated; external code
    importing this underscore-private helper directly needs to unpack
    three values.
    """
    user_col = user_id_column or "USER_ID"

    if item_id_column and (target_columns or target_rename_pattern):
        raise ValueError(
            "long_interactions: pass either `item_id_column`+`outcome_column` (the source "
            "is already long, just needs renaming) OR `target_columns`/`target_rename_pattern` "
            "(the source is wide and needs melting) — not both. With both supplied the "
            "tool can't tell whether to rename or to melt. For a wide → long reshape, "
            "drop item_id_column and outcome_column from the call."
        )

    if not item_id_column and (target_columns or target_rename_pattern):
        if target_rename_pattern and not target_columns:
            rx = re.compile(target_rename_pattern)
            target_columns = [c for c in df.columns if rx.fullmatch(c)]
            if not target_columns:
                raise ValueError(f"target_rename_pattern '{target_rename_pattern}' matched no columns.")
        df = _op_melt_to_long(df, user_col, target_columns)
        df = _op_cast_dtypes(df, {"USER_ID": "str", "ITEM_ID": "str", "OUTCOME": "float"})
        return df, ["melt_to_long", "cast_dtypes"], {}

    rename: dict[str, str] = {}
    if user_id_column and user_id_column != "USER_ID":
        rename[user_id_column] = "USER_ID"
    if item_id_column and item_id_column != "ITEM_ID":
        rename[item_id_column] = "ITEM_ID"
    if outcome_column and outcome_column != "OUTCOME":
        rename[outcome_column] = "OUTCOME"
    if rename:
        df = _op_rename_columns(df, rename)
    df = _op_cast_dtypes(df, {"USER_ID": "str", "ITEM_ID": "str", "OUTCOME": "float"})
    return df, ["rename_columns", "cast_dtypes"], {}


def _to_long_with_timestamp(df, user_id_column, item_id_column, outcome_column, timestamp_column, **_):
    if timestamp_column is None:
        raise ValueError("long_with_timestamp requires timestamp_column.")
    df, ops, extras = _to_long_interactions(
        df,
        user_id_column,
        item_id_column,
        outcome_column,
    )
    df = _op_parse_timestamp(df, timestamp_column)
    ops.append("parse_timestamp")
    return df, ops, extras


def _to_long_multi_reward(df, user_id_column, item_id_column, outcome_column, auxiliary_outcome_columns, **_):
    df, ops, extras = _to_long_interactions(df, user_id_column, item_id_column, outcome_column)
    if auxiliary_outcome_columns:
        rename = {
            c: f"OUTCOME_{c}" if not c.startswith("OUTCOME_") else c
            for c in auxiliary_outcome_columns
            if c in df.columns
        }
        df = _op_rename_columns(df, rename)
        for c in df.columns:
            if c.startswith("OUTCOME_"):
                df[c] = df[c].astype(float)
        ops.append("rename_aux_outcomes")
    return df, ops, extras


def _to_wide_multioutput(
    df,
    user_id_column,
    item_id_column,
    outcome_column,
    target_columns,
    target_rename_pattern,
    dedupe_user_id,
    drop_degenerate_targets=True,
    **_,
):
    ops: list[str] = []
    user_col = user_id_column or "USER_ID"

    item_star_present = [c for c in df.columns if c.startswith("ITEM_") and c != "ITEM_ID"]

    if target_rename_pattern and not item_star_present:
        df = _op_bulk_rename_targets(df, target_rename_pattern)
        ops.append("bulk_rename_targets")
    elif target_columns:
        rename = {c: f"ITEM_{c}" if not c.startswith("ITEM_") else c for c in target_columns}
        df = _op_rename_columns(df, rename)
        ops.append("rename_target_columns")
    elif item_id_column and outcome_column:
        # Long → wide pivot path
        df = _op_pivot_to_wide(df, user_col, item_id_column, outcome_column)
        ops.append("pivot_to_wide")
        user_col = "USER_ID"

    if user_col != "USER_ID" and user_col in df.columns:
        df = _op_rename_columns(df, {user_col: "USER_ID"})
        ops.append("rename_user_id")

    item_targets = [c for c in df.columns if c.startswith("ITEM_") and c != "ITEM_ID"]
    if len(item_targets) < 2:
        raise ValueError(
            f"wide_multioutput requires ≥2 ITEM_* target columns after transform; "
            f"got {len(item_targets)}: {item_targets}"
        )

    # Classify BEFORE the float cast. The cast on a column carrying strings
    # like 'yes' / 'no' would raise pandas' opaque "could not convert string
    # to float: 'yes'" — short-circuiting the friendly migration message
    # below. _classify_wide_multioutput_targets handles non-numeric dtypes
    # natively (samples the first few uniques as the offending values).
    bad_non_binary, degenerate, ok_cols = _classify_wide_multioutput_targets(df)
    if bad_non_binary:
        details = "; ".join(f"{col!r} (saw values: {vals})" for col, vals in bad_non_binary)
        raise ValueError(
            "wide_multioutput targets must be binary numeric — values strictly in {0, 1} "
            "(or {0.0, 1.0}). The following ITEM_* column(s) contain non-binary values: "
            f"{details}. Pre-encode at the caller, e.g.: "
            "df['ITEM_x'] = (df['ITEM_x'] == 'yes').astype(float). For multi-class targets "
            "specifically, see migration paths: (1) MulticlassScorer in long format for a "
            "single multi-class target; (2) one-hot encode multi-class targets into binary "
            "columns."
        )

    # Cast happens AFTER the validate-and-reject gate above. Surviving
    # columns are now guaranteed numeric+binary (or all-NaN), so the float
    # cast can't raise on dirty values. Auto-drop of single-class targets
    # still runs below — scikit-rec's factory threads
    # ``scorer_config={"on_degenerate_target": "constant"}`` through to
    # MultioutputScorer as of 0.3.1.dev6, so callers can opt into CONSTANT
    # mode by passing that through train_model rather than relying on the
    # drop here.
    df = _op_cast_dtypes(df, {"USER_ID": "str", **{c: "float" for c in item_targets}})
    ops.append("cast_dtypes")

    dropped_targets: list[dict[str, Any]] = []
    if degenerate and drop_degenerate_targets:
        cols_to_drop = [col for col, _ in degenerate]
        if not ok_cols or len(ok_cols) < 2:
            # Refuse to silently strip the wide layout below the ≥2-target floor —
            # the MultioutputScorer can't be fit on <2 surviving columns and the
            # downstream contract validator would reject anyway. Surface the
            # surviving column too (when there is exactly one) so the caller
            # can decide whether to keep it manually as a single-target
            # univariate problem.
            details = ", ".join(f"{col!r} (only value: {val!r})" for col, val in degenerate)
            surviving = f" Surviving non-degenerate column(s): {ok_cols!r}." if ok_cols else ""
            raise ValueError(
                "Every (or all-but-one) wide_multioutput target column is degenerate "
                "(single-class) — there's nothing for MultioutputScorer to fit on after "
                "auto-drop. Affected columns: "
                f"{details}.{surviving} Either fix the source data so each target has "
                "both classes represented, or use a stratified split that retains both "
                "classes per target."
            )
        df = df.drop(columns=cols_to_drop)
        ops.append("drop_degenerate_targets")
        for col, val in degenerate:
            dropped_targets.append(
                {
                    "column": col,
                    "reason": "single_class",
                    "value_seen": val,
                    "hint": (
                        "MultioutputScorer fit-time validation rejects single-class targets "
                        "under the default RAISE policy. Auto-dropped here for safety. To "
                        "keep this target with a constant-predictor fallback, pass "
                        "drop_degenerate_targets=False to transform_data AND pass "
                        "scorer_config={'on_degenerate_target': 'constant'} on train_model "
                        "(scikit-rec ≥ 0.3.1.dev6 threads it through the factory)."
                    ),
                }
            )
    elif degenerate:
        # User opted out of auto-drop — surface the manifest so the caller
        # picks one of the three real workflows below. ``False`` means
        # "stop, let me decide"; it is NOT a "keep them" path (the scorer
        # would still raise at fit time under the default RAISE policy).
        details = ", ".join(f"{col!r} (only value: {val!r})" for col, val in degenerate)
        raise ValueError(
            "wide_multioutput targets with only one class in the training slice cannot be "
            f"fit under MultioutputScorer's default RAISE policy: {details}.\n"
            "Three workflows from here:\n"
            "  (1) Re-run transform_data with drop_degenerate_targets=True to auto-drop "
            "the listed columns and proceed; their names will appear in the result "
            "envelope under `dropped_targets`.\n"
            "  (2) Drop the columns at the source (fix the data), then re-run.\n"
            "  (3) Keep the columns with a constant-predictor fallback: leave "
            "drop_degenerate_targets=True here AND pass "
            "scorer_config={'on_degenerate_target': 'constant'} to train_model — "
            "the scorer-side policy is the right surface for keep-with-fallback."
        )

    if dedupe_user_id:
        df = _op_dedupe_user_id(df)
        ops.append("dedupe_user_id")
    return df, ops, {"dropped_targets": dropped_targets}


def _to_multiclass(df, user_id_column, item_id_column, dedupe_user_id, **_):
    rename: dict[str, str] = {}
    if user_id_column and user_id_column != "USER_ID":
        rename[user_id_column] = "USER_ID"
    if item_id_column and item_id_column != "ITEM_ID":
        rename[item_id_column] = "ITEM_ID"
    if rename:
        df = _op_rename_columns(df, rename)
    df = _op_cast_dtypes(df, {"USER_ID": "str", "ITEM_ID": "str"})
    if "OUTCOME" in df.columns:
        df = df.drop(columns=["OUTCOME"])
    if dedupe_user_id:
        df = _op_dedupe_user_id(df)
    return df, ["rename_columns", "cast_dtypes", "drop_outcome", "dedupe_user_id"], {}


def _to_prebuilt_sequences(df, user_id_column, item_id_column, outcome_column, timestamp_column, **_):
    user_col = user_id_column or "USER_ID"
    item_col = item_id_column or "ITEM_ID"
    if user_col not in df.columns or item_col not in df.columns:
        raise ValueError(f"prebuilt_sequences requires {user_col} and {item_col} columns in the source.")
    df = _op_aggregate_to_sequences(
        df,
        user_col,
        item_col,
        outcome_column,
        timestamp_column,
    )
    return df, ["aggregate_to_sequences"], {}


def _to_sessions(df, user_id_column, **_):
    """Sessions contract: USER_ID + SESSION_SEQUENCES (list-of-lists).

    Source data is expected to already carry a SESSION_SEQUENCES column;
    constructing valid session boundaries from raw triples requires
    user-supplied session-id semantics this tool can't guess. Surface a
    clear error otherwise.
    """
    user_col = user_id_column or "USER_ID"
    if "SESSION_SEQUENCES" not in df.columns:
        raise ValueError(
            "sessions contract requires a SESSION_SEQUENCES column in the source. "
            "If your data is flat (USER_ID, ITEM_ID, SESSION_ID), aggregate it to "
            "lists-of-lists yourself before calling transform_data — the right "
            "session boundaries are user-domain knowledge."
        )
    if user_col != "USER_ID":
        df = _op_rename_columns(df, {user_col: "USER_ID"})
    df = _op_dedupe_user_id(df)
    return df, ["rename_user_id", "dedupe_user_id"], {}


def _to_users_features(df, user_id_column, dedupe_user_id, **_):
    user_col = user_id_column or "USER_ID"
    if user_col != "USER_ID":
        df = _op_rename_columns(df, {user_col: "USER_ID"})
    df = _op_cast_dtypes(df, {"USER_ID": "str"})
    if dedupe_user_id:
        df = _op_dedupe_user_id(df)
    return df, ["rename_user_id", "cast_dtypes", "dedupe_user_id"], {}


def _to_items_features(df, item_id_column, **_):
    item_col = item_id_column or "ITEM_ID"
    if item_col != "ITEM_ID":
        df = _op_rename_columns(df, {item_col: "ITEM_ID"})
    df = _op_cast_dtypes(df, {"ITEM_ID": "str"})
    if "ITEM_ID" in df.columns:
        df = df.drop_duplicates(subset=["ITEM_ID"]).reset_index(drop=True)
    return df, ["rename_item_id", "cast_dtypes", "dedupe_item_id"], {}


_PLAN_BY_CONTRACT = {
    "long_interactions": _to_long_interactions,
    "long_with_timestamp": _to_long_with_timestamp,
    "long_multi_reward": _to_long_multi_reward,
    "wide_multioutput": _to_wide_multioutput,
    "multiclass": _to_multiclass,
    "prebuilt_sequences": _to_prebuilt_sequences,
    "sessions": _to_sessions,
    "users_features": _to_users_features,
    "items_features": _to_items_features,
}


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------


def _validate_against_contract(df: pd.DataFrame, contract: str) -> dict[str, Any]:
    required = _CONTRACT_REQUIRED_COLS.get(contract, ())
    issues: list[str] = []
    for col in required:
        if col not in df.columns:
            issues.append(f"missing required column '{col}'")

    if contract == "wide_multioutput":
        item_targets = [c for c in df.columns if c.startswith("ITEM_") and c != "ITEM_ID"]
        if len(item_targets) < 2:
            issues.append(f"wide_multioutput requires ≥2 ITEM_* target columns, found {len(item_targets)}")

    if contract in _CONTRACTS_REQUIRING_DEDUPE and "USER_ID" in df.columns:
        if df["USER_ID"].duplicated().any():
            issues.append(f"{contract} requires 1 row per USER_ID; duplicates found")

    if contract == "prebuilt_sequences" and "ITEM_SEQUENCE" in df.columns:
        if not df["ITEM_SEQUENCE"].apply(lambda v: isinstance(v, (list, tuple))).all():
            issues.append("ITEM_SEQUENCE column must contain list/tuple values")

    return {"valid": not issues, "issues": issues}


# ---------------------------------------------------------------------------
# IO helpers
# ---------------------------------------------------------------------------


def _read(file_path: str) -> pd.DataFrame:
    if file_path.endswith(".parquet"):
        return pd.read_parquet(file_path)
    return pd.read_csv(file_path)


def _write(df: pd.DataFrame, output_path: str) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(output_path)) or ".", exist_ok=True)
    if output_path.endswith(".parquet"):
        df.to_parquet(output_path, index=False)
    else:
        df.to_csv(output_path, index=False)


# ---------------------------------------------------------------------------
# Tool entrypoint
# ---------------------------------------------------------------------------


def _transform_data(
    file_path: str,
    output_path: str,
    target_contract: str,
    session,
    user_id_column: str | None = None,
    item_id_column: str | None = None,
    outcome_column: str | None = None,
    timestamp_column: str | None = None,
    target_columns: list[str] | None = None,
    target_rename_pattern: str | None = None,
    auxiliary_outcome_columns: list[str] | None = None,
    feature_columns: list[str] | None = None,
    feature_pattern: str | None = None,
    drop_null_columns: bool = True,
    dedupe_user_id: bool = False,
    drop_degenerate_targets: bool = True,
) -> dict[str, Any]:
    if target_contract not in _TARGET_CONTRACTS:
        return err(
            "InvalidContract",
            f"Unknown target_contract '{target_contract}'. Valid: {list(_TARGET_CONTRACTS)}",
        )
    if not os.path.exists(file_path):
        return err("FileNotFoundError", f"file_path not found: {file_path}")
    if os.path.abspath(file_path) == os.path.abspath(output_path):
        return err(
            "InPlaceWriteForbidden",
            "output_path must differ from file_path; transform_data never writes in-place.",
        )

    try:
        df = _read(file_path)
    except Exception as e:
        return err(type(e).__name__, f"Failed to read {file_path}: {e}")

    source_shape = _detect_source_shape(df, user_id_column, item_id_column)
    if drop_null_columns:
        df = _op_drop_null_columns(df)

    plan = _PLAN_BY_CONTRACT[target_contract]
    try:
        df, ops_applied, plan_extras = plan(
            df,
            user_id_column=user_id_column,
            item_id_column=item_id_column,
            outcome_column=outcome_column,
            timestamp_column=timestamp_column,
            target_columns=target_columns,
            target_rename_pattern=target_rename_pattern,
            auxiliary_outcome_columns=auxiliary_outcome_columns,
            dedupe_user_id=dedupe_user_id or target_contract in _CONTRACTS_REQUIRING_DEDUPE,
            drop_degenerate_targets=drop_degenerate_targets,
        )
    except Exception as e:
        return err(type(e).__name__, str(e))

    keep_anchor: list[str] = list(_CONTRACT_REQUIRED_COLS.get(target_contract, ()))
    if target_contract == "wide_multioutput":
        keep_anchor += [c for c in df.columns if c.startswith("ITEM_") and c != "ITEM_ID"]
    if target_contract == "long_multi_reward":
        keep_anchor += [c for c in df.columns if c.startswith("OUTCOME_")]
    if target_contract == "prebuilt_sequences" and "OUTCOME_SEQUENCE" in df.columns:
        keep_anchor.append("OUTCOME_SEQUENCE")

    if feature_columns or feature_pattern:
        df = _op_select_features(df, keep_anchor, feature_columns, feature_pattern)
        ops_applied.append("select_features")

    validation = _validate_against_contract(df, target_contract)
    if not validation["valid"]:
        return err(
            "ContractViolation",
            f"Transform output does not satisfy {target_contract}: {validation['issues']}",
            hint="Re-run profile_data on the source to confirm role hints; some contracts "
            "require explicit user_id_column / item_id_column / target_columns.",
        )

    try:
        _write(df, output_path)
    except Exception as e:
        return err(type(e).__name__, f"Failed to write {output_path}: {e}")

    payload = {
        "output_path": output_path,
        "target_contract": target_contract,
        "source_shape_detected": source_shape,
        "ops_applied": ops_applied,
        "columns": list(df.columns),
        "n_rows": int(len(df)),
        "validation": validation,
    }
    # Surface plan-emitted extras (e.g. wide_multioutput's dropped_targets
    # manifest) so the agent and the human-facing transcript both see them.
    for k, v in plan_extras.items():
        if v:
            payload[k] = v
    return ok(payload)


TOOL_TRANSFORM_DATA = Tool(
    name="transform_data",
    description=(
        "Reshape a raw CSV/Parquet into a scikit-rec contract format. Use after "
        "`profile_data` + `validate_data` when the input shape doesn't match the "
        "contract the chosen recommender requires. Auto-detects source shape and "
        "applies the right reshape (rename, pivot, melt, aggregate, dedupe, cast). "
        "Writes a canonical file at `output_path`; call `create_datasets` on that "
        "path next."
    ),
    input_schema={
        "type": "object",
        "properties": {
            "file_path": {"type": "string"},
            "output_path": {"type": "string"},
            "target_contract": {
                "type": "string",
                "enum": list(_TARGET_CONTRACTS),
            },
            "user_id_column": {"type": "string"},
            "item_id_column": {"type": "string"},
            "outcome_column": {"type": "string"},
            "timestamp_column": {"type": "string"},
            "target_columns": {
                "type": "array",
                "items": {"type": "string"},
                "description": "Columns to become ITEM_* targets (wide_multioutput).",
            },
            "target_rename_pattern": {
                "type": "string",
                "description": "Regex matching wide target columns, e.g. 'label_(.*)'. Renamed to ITEM_\\1.",
            },
            "auxiliary_outcome_columns": {
                "type": "array",
                "items": {"type": "string"},
                "description": "Long_multi_reward auxiliary OUTCOME_* sources.",
            },
            "feature_columns": {
                "type": "array",
                "items": {"type": "string"},
                "description": "Explicit feature columns to keep alongside the contract anchor columns.",
            },
            "feature_pattern": {
                "type": "string",
                "description": "Regex matching feature columns to keep.",
            },
            "drop_null_columns": {"type": "boolean", "default": True},
            "dedupe_user_id": {"type": "boolean", "default": False},
            "drop_degenerate_targets": {
                "type": "boolean",
                "default": True,
                "description": (
                    "wide_multioutput-only. Controls transform_data's behavior when one "
                    "or more ITEM_* targets have a single class across the post-pivot "
                    "frame (MultioutputScorer's default RAISE policy refuses these at "
                    "fit time). "
                    "TRUE (default): auto-drop the degenerate ITEM_* columns and surface "
                    "their names in the result envelope under `dropped_targets`. The "
                    "sweep proceeds cleanly. "
                    "FALSE: do NOT drop — instead raise a ValueError listing the "
                    "offending columns so the caller decides explicitly. This is the "
                    "'stop and let me handle it' path, not a 'keep them' path. To "
                    "actually keep degenerate columns with a constant-predictor "
                    "fallback at fit time, leave this TRUE (or drop the columns at the "
                    "source) AND additionally pass "
                    "scorer_config={'on_degenerate_target': 'constant'} to train_model "
                    "— the scorer-side policy is the right surface for that semantic."
                ),
            },
        },
        "required": ["file_path", "output_path", "target_contract"],
    },
    fn=_transform_data,
)
