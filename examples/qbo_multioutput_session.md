# Example Transcript: QBO-style Wide Multi-Output Data

A recorded interaction showing the three new tools in context:
**`transform_data`**, **`sweep_methods`**, and **`diagnose_training_failure`**.

The user has wide-format company-action data — one row per company with many
`label_*` action columns. This is the contract `MultioutputScorer` consumes,
but the agent has to reshape it (`label_*` → `ITEM_*` and dedupe per company)
before training. Then the user asks for a method comparison rather than
hand-picking; one of the methods fails and the agent diagnoses the failure
and retries automatically.

The agent's tool calls are shown in `[tool_call]` blocks; results are
summarized.

---

**User:** I have data at `/data/companies.csv` — one row per QBO company with
13 `label_*` columns marking which actions they took, plus some company-level
features. Help me predict what actions the next companies will take.

**Agent:**

```
[tool_call] profile_data(file_path="/data/companies.csv", file_type="interactions")
```

68K rows, 1 row per `qbo_company_id`. 13 `label_*` columns, all binary. Side
features: `industry`, `company_age_months`, `n_employees`, `state`. No
timestamps.

This is a classic **wide multi-output** setup — predict multiple correlated
binary actions per user. The right contract is `wide_multioutput`, scored by
`MultioutputScorer`. Two things to fix before training:

1. `qbo_company_id` → `USER_ID`
2. `label_*` → `ITEM_*`

```
[tool_call] validate_data(file_path="/data/companies.csv", file_type="interactions")
```

Confirms: missing `USER_ID`. Suggested rename `{qbo_company_id: USER_ID}`.
Column rename alone isn't enough — `MultioutputScorer` needs `ITEM_*` target
columns, and your targets are `label_*`. I'll use `transform_data`.

```
[tool_call] transform_data(
    file_path="/data/companies.csv",
    output_path="/tmp/companies_wide.csv",
    target_contract="wide_multioutput",
    user_id_column="qbo_company_id",
    target_rename_pattern="label_(.*)",
    feature_columns=["industry", "company_age_months", "n_employees", "state"])
```

```
[tool_result:ok]
ops_applied: [bulk_rename_targets, rename_user_id, cast_dtypes,
              dedupe_user_id, select_features]
columns: USER_ID, industry, company_age_months, n_employees, state,
         ITEM_invoice, ITEM_payroll, ITEM_budget, ... (13 ITEM_*)
n_rows: 68000
validation: {valid: true, issues: []}
```

```
[tool_call] validate_data(file_path="/tmp/companies_wide.csv", file_type="interactions")
[tool_call] create_datasets(bundle_id="qbo", interactions_path="/tmp/companies_wide.csv")
[tool_call] split_data(bundle_id="qbo", strategy="random_split_per_user",
    valid_fraction=0.2, random_state=1)
```

Bundle registered. 54.4K train / 13.6K validation rows.

**Agent (continued):** Rather than hand-pick a method, I'll sweep the default
set for the wide multi-output contract — that filters incompatible combos
before training, runs a deterministic comparison, and ranks by NDCG@10.

```
[tool_call] sweep_methods(
    bundle_id="qbo",
    methods="auto",
    metrics=["NDCG_at_k", "roc_auc"],
    primary_metric="NDCG_at_k",
    eval_top_k=10,
    evaluator_type="simple")
```

```
[tool_result:ok]
contract: wide_multioutput
n_methods_requested: 4
n_runnable: 3
n_dropped_incompatible: 1
dropped_methods:
  - {short_name: "deepfm_multioutput", reason:
     "scorer 'multioutput' does not accept estimator family 'embedding'"}
leaderboard:
  1. xgb_multioutput     status=ok    NDCG@10=0.412 roc_auc=0.781
  2. lightgbm_multioutput status=ok    NDCG@10=0.405 roc_auc=0.776
  3. logreg_multioutput  status=error  category=numerical_instability
winner: xgb_multioutput
```

The DeepFM entry was filtered before training — `MultioutputScorer` doesn't
accept embedding estimators (matches scikit-rec's factory rule). The LogReg
entry blew up with a numerical instability. Let me diagnose it.

```
[tool_call] diagnose_training_failure(
    model_name="sweep_logreg_multioutput_3a91c2f1",
    auto_retry=true,
    max_retries=2)
```

```
[tool_result:ok]
category: numerical_instability
causes: ["Optimizer hit a numerically unstable configuration: high LR,
          collinearity, or outliers."]
candidate_fixes:
  1. {description: "Lower the learning_rate to stabilise optimisation.",
      action: {type: "modify_config",
               set: {"estimator_config.xgboost.learning_rate": 0.01}},
      auto_retryable: true}
  2. {description: "Standardise features and drop fully-collinear columns.",
      auto_retryable: false}
auto_retried: true
applied_fix: <#1>
retry_result: {status: "ok", model_id: "sweep_logreg_..."}
```

The retry succeeded with the lower learning rate. Updated leaderboard:

```
[tool_call] compare_models(primary_metric="NDCG_at_k", k=10)
```

| model_name | recommender | training_time_s | NDCG@10 | roc_auc |
|---|---|---|---|---|
| xgb_multioutput | ranking | 38.2 | 0.412 | 0.781 |
| lightgbm_multioutput | ranking | 42.7 | 0.405 | 0.776 |
| logreg_multioutput (retried) | ranking | 22.1 | 0.378 | 0.749 |

XGBoost wins. Want me to run HPO on it next, or save and stop here?

**User:** Save it and stop.

**Agent:**

```
[tool_call] save_model(model_id="sweep_xgb_multioutput_a7f31..." ,
    tags=["qbo","wide_multioutput","sweep_winner"])
```

Saved. The sweep is idempotent — if you re-run `sweep_methods` on the same
bundle with the same methods, the trained models are reused (status will be
`cached`) so you can extend with new methods without re-paying for the old
ones.

---

## What's new in this transcript vs. `movielens_session.md`

- **`transform_data`** replaces a manual reshape. The agent picks
  `target_contract="wide_multioutput"`, supplies the regex
  `label_(.*)`, and the tool produces a contract-conformant artifact in one
  call (rename, cast, dedupe, feature selection).
- **`sweep_methods`** replaces a hand-rolled loop of `train_model` +
  `evaluate_model`. It pre-filters incompatible combos (here: DeepFM under
  `MultioutputScorer`), trains the rest in one shot, and is idempotent on
  re-run via a `(bundle_id, config)` hash.
- **`diagnose_training_failure`** replaces the LLM guessing what to fix.
  When LogReg failed with a numerical-instability error, the diagnosis
  pattern-matched the message, returned ranked candidate fixes, and the
  auto-retry applied the lowest-LR fix to retrain successfully — bounded by
  `max_retries=2`.

The training-time error envelope itself now carries a machine-readable
`category` field (e.g. `"numerical_instability"`) and a `hint`, so even
without calling `diagnose_training_failure` the LLM has structured guidance
on what failed and how to react.
