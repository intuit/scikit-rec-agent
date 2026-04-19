# Example Transcript: MovieLens-style Click Data

This is a recorded interaction between a user and the agent. It illustrates
the full workflow: profile → validate → create datasets → split → train
baselines → evaluate → compare → HPO → save.

The agent's tool calls are shown in `[tool_call]` blocks; results are
summarized.

---

**User:** I have MovieLens-style click-stream data at `/data/interactions.csv`
with user features at `/data/users.csv` and item features at `/data/items.csv`.
Help me find the best ranking model.

**Agent:**

```
[tool_call] profile_data(file_path="/data/interactions.csv", file_type="interactions")
[tool_call] profile_data(file_path="/data/users.csv", file_type="users")
[tool_call] profile_data(file_path="/data/items.csv", file_type="items")
```

1.2M interactions across 50K users and 10K items. Target is binary (clicks).
Timestamps span six months. User features: age, gender, segment. Item features:
category, price. Sparsity 99.76%. 342 duplicate pairs. Your columns
`userid` / `itemid` / `clicked` need to be mapped to `USER_ID` / `ITEM_ID` /
`OUTCOME` — I'll handle that via `column_mapping`.

```
[tool_call] validate_data(file_path="/data/interactions.csv", file_type="interactions")
```

Confirmed: missing USER_ID, ITEM_ID, OUTCOME. Suggestion `{userid: USER_ID,
itemid: ITEM_ID, clicked: OUTCOME}` looks right.

**Agent (continued):** Given binary implicit feedback with timestamps and
side features, three candidates:

1. **XGBoost + UniversalScorer + Ranking** — fast baseline, handles features well.
   ```python
   {"recommender_type": "ranking", "scorer_type": "universal",
    "estimator_config": {"ml_task": "classification",
                         "xgboost": {"n_estimators": 200, "max_depth": 6}}}
   ```
2. **Two-Tower + UniversalScorer + Ranking** — learns user/item embeddings, robust to cold-start.
   ```python
   {"recommender_type": "ranking", "scorer_type": "universal",
    "estimator_config": {"estimator_type": "embedding",
                         "embedding": {"model_type": "two_tower",
                                       "params": {"embedding_dim": 64}}}}
   ```
3. **SASRec + SequentialScorer + Sequential** — exploits timestamps.
   ```python
   {"recommender_type": "sequential", "scorer_type": "sequential",
    "estimator_config": {"estimator_type": "sequential",
                         "sequential": {"model_type": "sasrec_classifier",
                                        "params": {"hidden_units": 64, "max_len": 50}}},
    "recommender_params": {"max_len": 50}}
   ```

Train all three?

**User:** Yes, go.

**Agent:**

```
[tool_call] create_datasets(bundle_name="movielens", interactions_path="...",
    users_path="...", items_path="...",
    column_mapping={"userid": "USER_ID", "itemid": "ITEM_ID", "clicked": "OUTCOME"})
[tool_call] split_data(bundle_id="movielens", strategy="temporal",
    valid_fraction=0.1, test_fraction=0.1)
[tool_call] train_model(model_name="xgb_baseline", config=..., bundle_id="movielens")
[tool_call] train_model(model_name="two_tower", config=..., bundle_id="movielens")
[tool_call] train_model(model_name="sasrec", config=..., bundle_id="movielens")
```

XGBoost: 45s. Two-Tower: 3m 12s. SASRec: 8m 41s.

```
[tool_call] evaluate_model(model_id="...", evaluator_type="simple",
    metrics=["NDCG_at_k", "precision_at_k", "MAP_at_k"], k_values=[5, 10])
  (x3, once per model)
[tool_call] compare_models(primary_metric="NDCG_at_k", k=10)
```

Two-Tower wins. NDCG@10: XGB 0.312, Two-Tower 0.347, SASRec 0.338. Sequential
didn't beat the simpler embedding model here — likely your sessions aren't
long enough.

**User:** Tune Two-Tower.

**Agent:**

```
[tool_call] run_hpo(study_name="tt_search", base_config=..., 
    search_space={"estimator_config.embedding.params.embedding_dim": {"type":"int","low":32,"high":256,"step":32},
                  "estimator_config.embedding.params.learning_rate": {"type":"float","low":0.0001,"high":0.01,"log":true}},
    metric_definitions=["NDCG@10"], objective_metric="NDCG@10",
    bundle_id="movielens", n_trials=50, sampler="tpe")
```

Best NDCG@10: 0.381 at `embedding_dim=128, learning_rate=0.003`. Re-trained as
`ranking_hpo_17764…`.

```
[tool_call] save_model(model_id="ranking_hpo_17764...", tags=["two_tower","hpo_winner"])
```

Registered at `~/.scikit-rec/registry/tt_search_best/`.
