"""train_model tool.

Wraps `skrec.orchestrator.create_recommender_pipeline` plus the recommender's
`.train()` call. The factory validates configs on entry; bad configs raise
ValueError/TypeError/NotImplementedError which we capture as error envelopes
so the LLM can read the message and self-correct.
"""

from __future__ import annotations

import time
from datetime import datetime, timezone
from typing import Any

from scikit_rec_agent.session import DatasetBundle, ModelHandle, new_model_id
from scikit_rec_agent.tools import Tool, err, ok
from scikit_rec_agent.tools.datasets import _create_datasets as _build_bundle


def _resolve_bundle(
    bundle_id: str | None,
    interactions_path: str | None,
    users_path: str | None,
    items_path: str | None,
    column_mapping: dict[str, str] | None,
    session,
) -> tuple[DatasetBundle | None, dict[str, Any] | None]:
    """Return (bundle, error_envelope). Exactly one is non-None."""
    if bundle_id:
        bundle = session.loaded_datasets.get(bundle_id)
        if bundle is None:
            return None, err(
                "BundleNotFound",
                f"No bundle '{bundle_id}'. Call create_datasets first or pass raw paths.",
            )
        return bundle, None
    if not interactions_path:
        return None, err(
            "MissingArgument",
            "train_model requires either bundle_id or interactions_path.",
        )
    implicit_name = f"implicit_bundle_{int(time.time() * 1000)}"
    result = _build_bundle(
        bundle_id=implicit_name,
        interactions_path=interactions_path,
        session=session,
        users_path=users_path,
        items_path=items_path,
        column_mapping=column_mapping,
    )
    if result["status"] != "ok":
        return None, result
    return session.loaded_datasets[implicit_name], None


def _train_model(
    model_name: str,
    session,
    config: dict[str, Any] | None = None,
    bundle_id: str | None = None,
    interactions_path: str | None = None,
    users_path: str | None = None,
    items_path: str | None = None,
    column_mapping: dict[str, str] | None = None,
    scorer_config: dict[str, Any] | None = None,
) -> dict[str, Any]:
    from skrec.orchestrator import create_recommender_pipeline

    from scikit_rec_agent.tools.diagnose import _quick_diagnose, record_failure
    from scikit_rec_agent.tools.sweep import _AUTO_SWEEPS, _detect_bundle_contract

    # ``scorer_config`` can also be threaded through ``config['scorer_config']``
    # to mirror the upstream RecommenderConfig shape. The explicit kwarg is a
    # convenience for callers (and matches the agent tool signature surfaced
    # below); both paths converge into the same dict on the config below.

    # Implicit-bundle cleanup state. When the caller didn't pass bundle_id
    # (raw-paths mode), _resolve_bundle registers an
    # ``implicit_bundle_<ms>`` entry in session.loaded_datasets so the
    # default-config branch and the training body share one source of
    # truth. If anything fails AFTER that registration (factory raise,
    # InvalidScorerConfigKey, train() exception), the implicit bundle
    # must be popped — otherwise repeated failing calls leak one
    # DataFrame-laden bundle per call.
    #
    # The body below runs inside a try/finally so that every existing
    # error return (and any future ones) trigger the cleanup
    # automatically. ``_training_succeeded`` flips to True only on the
    # single success path right before ``return ok(payload)``; the
    # finally inspects it after the return value is computed.
    _caller_provided_bundle_id = bundle_id is not None
    _implicit_bundle_id: str | None = None
    _training_succeeded = False

    try:
        # Default-config branch needs a registered bundle to read the contract
        # from. If the caller passed raw paths instead of bundle_id, resolve
        # them via _resolve_bundle FIRST so the auto-pick has something to
        # inspect. Without this hoist, ``config=None`` + raw paths errors
        # with a misleading "register the bundle first" message even though
        # the function supports the raw-paths path otherwise.
        if config is None:
            bundle_for_default, err_env = _resolve_bundle(
                bundle_id, interactions_path, users_path, items_path, column_mapping, session
            )
            if err_env is not None:
                return err_env
            assert bundle_for_default is not None
            # Stash the resolved bundle_id so we don't re-resolve below.
            bundle_id = bundle_for_default.bundle_id
            if not _caller_provided_bundle_id:
                _implicit_bundle_id = bundle_id

        # Default-config path: when the caller doesn't supply a `config`, look up
        # the bundle's contract and use the first curated auto-sweep entry.
        # Friendlier than rejecting with ArgumentError — gpt-4o-mini-class
        # models commonly omit the config dict on simple "train one model"
        # prompts, then loop forever in the design walkthrough trying to build
        # one from scratch. Falling back to a sensible default lets that path
        # succeed while preserving the explicit-config behaviour for callers
        # who do pass one.
        _default_method_short_name: str | None = None
        if config is None:
            # ``bundle_for_default`` resolved above (handles both bundle_id and
            # raw-paths inputs uniformly). Detect the contract and pick the
            # first curated default for it.
            contract = _detect_bundle_contract(bundle_for_default)
            candidates = _AUTO_SWEEPS.get(contract, [])
            if not candidates:
                return err(
                    "ArgumentError",
                    f"No curated default exists for contract '{contract}' — pass an explicit "
                    f"config={{recommender_type, scorer_type, estimator_config}}.",
                )
            method = candidates[0]
            config = {
                "recommender_type": method["recommender_type"],
                "scorer_type": method["scorer_type"],
                "estimator_config": method["estimator_config"],
            }
            if "recommender_params" in method:
                config["recommender_params"] = method["recommender_params"]
            # Stash the chosen short_name for the result envelope so the caller
            # can see which curated default was applied.
            _default_method_short_name = method.get("short_name")

        if not isinstance(config, dict):
            return err("ArgumentError", "config must be a dict.")

        config = dict(config)
        config.setdefault("recommender_params", {})

        # Merge the explicit kwarg into config['scorer_config'] so a single
        # dict makes it to create_recommender_pipeline. Explicit kwarg wins on
        # key conflict (callers passing scorer_config kwarg explicitly intend
        # to override). Validate accepted keys against the upstream
        # capability_matrix so a bad key surfaces before the factory call.
        if scorer_config is not None or "scorer_config" in config:
            merged = dict(config.get("scorer_config") or {})
            if scorer_config:
                merged.update(scorer_config)
            if merged:
                scorer_type = config.get("scorer_type")
                try:
                    from skrec.orchestrator.factory import capability_matrix

                    cm = capability_matrix()
                    # Two flavors of "old skrec":
                    #   (a) capability_matrix doesn't exist at all → ImportError below
                    #   (b) capability_matrix exists but predates the scorer_config_keys
                    #       entry (it was added alongside the factory's scorer_config
                    #       plumbing). In (b) cm.get("scorer_config_keys") returns
                    #       None — defer to the factory rather than reject every key
                    #       against an empty whitelist (which would false-reject the
                    #       valid on_degenerate_target for multioutput).
                    if "scorer_config_keys" in cm:
                        allowed = set(cm["scorer_config_keys"].get(scorer_type or "", ()))
                        unknown = set(merged) - allowed
                        if unknown:
                            return err(
                                "InvalidScorerConfigKey",
                                f"scorer_type={scorer_type!r} does not accept scorer_config keys: "
                                f"{sorted(unknown)}. Accepted: {sorted(allowed) or '(none)'}.",
                                hint=(
                                    "Drop the unsupported key(s), or change scorer_type. Check "
                                    "skrec.orchestrator.factory.capability_matrix()['scorer_config_keys'] "
                                    "for the live mapping."
                                ),
                                category="invalid_scorer_config_key",
                            )
                except ImportError:
                    # Older skrec without capability_matrix at all — defer to the
                    # factory, which will surface any bad keys at construction time.
                    pass
                config["scorer_config"] = merged

        bundle_args = {
            "bundle_id": bundle_id,
            "interactions_path": interactions_path,
            "users_path": users_path,
            "items_path": items_path,
            "column_mapping": column_mapping,
        }
        bundle_args = {k: v for k, v in bundle_args.items() if v is not None}

        bundle, err_env = _resolve_bundle(bundle_id, interactions_path, users_path, items_path, column_mapping, session)
        if err_env is not None:
            return err_env
        assert bundle is not None
        # If this is the call that registered the bundle (caller passed raw
        # paths, didn't pass bundle_id, and the config!=None branch didn't
        # already capture an _implicit_bundle_id above), record it now so the
        # finally-block cleanup can pop it on a downstream failure.
        if not _caller_provided_bundle_id and _implicit_bundle_id is None:
            _implicit_bundle_id = bundle.bundle_id

        try:
            recommender = create_recommender_pipeline(config)
        except (ValueError, TypeError, NotImplementedError) as e:
            diagnosis = _quick_diagnose(e)
            envelope = err(
                type(e).__name__,
                str(e),
                hint=diagnosis.first_fix_description or "Check recommender_type, scorer_type, and estimator_config.",
                category=diagnosis.category,
            )
            record_failure(session, model_name, config, bundle_args, envelope, diagnosis)
            return envelope

        import inspect

        accepted_train_kwargs = set(inspect.signature(recommender.train).parameters)

        train_kwargs: dict[str, Any] = {"interactions_ds": bundle.interactions}
        if bundle.users is not None:
            train_kwargs["users_ds"] = bundle.users
        if bundle.items is not None:
            train_kwargs["items_ds"] = bundle.items
        # Per-recommender-class kwarg differences: SequentialRecommender derives
        # its validation split internally via `use_validation: bool` and rejects
        # `valid_interactions_ds`. Filter to what the actual train() accepts so
        # the recommender_type doesn't have to be hardcoded here.
        if bundle.valid_interactions is not None and "valid_interactions_ds" in accepted_train_kwargs:
            train_kwargs["valid_interactions_ds"] = bundle.valid_interactions
        if (
            bundle.users is not None
            and bundle.valid_interactions is not None
            and "valid_users_ds" in accepted_train_kwargs
        ):
            train_kwargs["valid_users_ds"] = bundle.users
        if "use_validation" in accepted_train_kwargs and bundle.valid_interactions is not None:
            # Sequential recommenders take a bool flag instead of a separate
            # validation dataset — they derive the split from interactions_ds.
            train_kwargs["use_validation"] = True

        started = time.time()
        try:
            recommender.train(**train_kwargs)
        except Exception as e:
            diagnosis = _quick_diagnose(e)
            envelope = err(
                type(e).__name__,
                f"Training failed: {e}",
                hint=diagnosis.first_fix_description,
                category=diagnosis.category,
            )
            record_failure(session, model_name, config, bundle_args, envelope, diagnosis)
            return envelope
        elapsed = time.time() - started

        recommender_type = config.get("recommender_type", "unknown")
        model_id = new_model_id(str(recommender_type))
        handle = ModelHandle(
            model_id=model_id,
            name=model_name,
            config=config,
            recommender=recommender,
            training_time_seconds=elapsed,
            datasets_used={
                "bundle_id": bundle.bundle_id,
                "source_paths": dict(bundle.source_paths),
            },
            created_at=datetime.now(timezone.utc).isoformat(),
        )
        session.trained_models[model_id] = handle

        payload = {
            "model_id": model_id,
            "model_name": model_name,
            "status": "trained",
            "training_time_seconds": elapsed,
            "recommender_type": config.get("recommender_type"),
            "scorer_type": config.get("scorer_type"),
            "estimator_type": config.get("estimator_config", {}).get("estimator_type", "tabular"),
        }
        if _default_method_short_name is not None:
            payload["default_method_applied"] = _default_method_short_name
        if config.get("scorer_config"):
            payload["scorer_config_applied"] = dict(config["scorer_config"])
        # Surface MultioutputScorer's degenerate_targets manifest. Under the
        # CONSTANT policy, the scorer records which targets fell back to a
        # constant predictor (always empty under RAISE). evaluate(per_label=True)
        # emits NaN for those targets' classification / regression metrics;
        # without this manifest the user has no way to learn which columns
        # were affected. Active once a caller passes
        # scorer_config={'on_degenerate_target': 'constant'} through; harmless
        # no-op under the default RAISE policy.
        degenerate = getattr(getattr(recommender, "scorer", None), "degenerate_targets", None)
        if degenerate:
            payload["degenerate_targets"] = {str(k): float(v) for k, v in degenerate.items()}
        _training_succeeded = True
        return ok(payload)
    finally:
        if not _training_succeeded and _implicit_bundle_id is not None:
            session.loaded_datasets.pop(_implicit_bundle_id, None)


TOOL_TRAIN_MODEL = Tool(
    name="train_model",
    description=(
        "Train a recommender pipeline from a RecommenderConfig. Supply either a dataset "
        "`bundle_id` from create_datasets, OR raw file paths (train_model will call "
        "create_datasets internally). If `config` is omitted, train_model picks the "
        "first curated method for the bundle's contract from the auto-sweep table — "
        "the same default that `sweep_methods(methods='all')` would run, surfaced as "
        "`default_method_applied` in the result envelope. Config is validated by "
        "scikit-rec's factory — bad configs return an error envelope you can use to "
        "correct the config and retry. If the bundle has validation interactions "
        "(from split_data), they are used automatically."
    ),
    input_schema={
        "type": "object",
        "properties": {
            "model_name": {"type": "string"},
            "config": {
                "type": "object",
                "description": (
                    "RecommenderConfig dict: recommender_type, scorer_type, estimator_config, "
                    "optional recommender_params. See system prompt for canonical shapes. "
                    "Omit to use the first curated default for the bundle's contract."
                ),
            },
            "bundle_id": {"type": "string"},
            "interactions_path": {"type": "string"},
            "users_path": {"type": "string"},
            "items_path": {"type": "string"},
            "column_mapping": {"type": "object"},
            "scorer_config": {
                "type": "object",
                "description": (
                    "Per-scorer constructor kwargs threaded into "
                    "skrec.orchestrator.factory.create_scorer. Per-scorer "
                    "accepted-keys live on capability_matrix()['scorer_config_keys']. "
                    "Today the only non-empty entry is "
                    "scorer_type='multioutput' which accepts "
                    "{'on_degenerate_target': 'raise' | 'constant'}. "
                    "Pass 'constant' to let MultioutputScorer fall back to a "
                    "constant predictor for single-class ITEM_* targets instead "
                    "of raising — the affected columns surface under "
                    "degenerate_targets in the train_model envelope."
                ),
            },
        },
        "required": ["model_name"],
    },
    fn=_train_model,
)
