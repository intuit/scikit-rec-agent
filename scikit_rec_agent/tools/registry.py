"""save_model, list_models, load_model — local filesystem model registry.

Registry layout: ~/.scikit-rec/registry/<model_name>/
  model.pkl    — pickled BaseRecommender
  meta.json    — dict of model metadata (config, metrics, tags, etc.)

Collision policy: save_model refuses to overwrite an existing <model_name>
directory and returns an error envelope suggesting a tag-suffixed retry.
"""

from __future__ import annotations

import json
import pickle
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from scikit_rec_agent.session import ModelHandle
from scikit_rec_agent.tools import Tool, err, ok

REGISTRY_ROOT = Path.home() / ".scikit-rec" / "registry"


def _model_dir(model_name: str) -> Path:
    return REGISTRY_ROOT / model_name


def _save_model(model_id: str, session, tags: list[str] | None = None) -> dict[str, Any]:
    handle = session.trained_models.get(model_id)
    if handle is None:
        return err("ModelNotFound", f"No trained model '{model_id}' in session.")

    mdir = _model_dir(handle.name)
    if mdir.exists():
        return err(
            "NameCollision",
            f"Registry already contains '{handle.name}' at {mdir}.",
            hint=f"Rename the model (e.g. '{handle.name}_v2') or manually delete the existing directory.",
        )

    mdir.mkdir(parents=True, exist_ok=False)
    with open(mdir / "model.pkl", "wb") as f:
        pickle.dump(handle.recommender, f)

    merged_tags = list(dict.fromkeys([*handle.tags, *(tags or [])]))
    meta = {
        "model_id": handle.model_id,
        "name": handle.name,
        "config": handle.config,
        "metrics": dict(handle.metrics),
        "tags": merged_tags,
        "training_time_seconds": handle.training_time_seconds,
        "datasets_used": handle.datasets_used,
        "created_at": handle.created_at,
        "saved_at": datetime.now(timezone.utc).isoformat(),
    }
    with open(mdir / "meta.json", "w") as f:
        json.dump(meta, f, indent=2, default=str)

    handle.tags = merged_tags

    return ok(
        {
            "model_name": handle.name,
            "model_id": handle.model_id,
            "registry_path": str(mdir),
            "saved_at": meta["saved_at"],
        }
    )


TOOL_SAVE_MODEL = Tool(
    name="save_model",
    description=(
        "Persist a trained model, its config, and evaluation metrics to the local "
        "registry at ~/.scikit-rec/registry/<model_name>/. Refuses to overwrite an "
        "existing model with the same name."
    ),
    input_schema={
        "type": "object",
        "properties": {
            "model_id": {"type": "string"},
            "tags": {"type": "array", "items": {"type": "string"}},
        },
        "required": ["model_id"],
    },
    fn=_save_model,
)


# ---------------------------------------------------------------------------
# list_models
# ---------------------------------------------------------------------------


def _list_models(
    session,
    tag_filter: list[str] | None = None,
    recommender_type_filter: str | None = None,
) -> dict[str, Any]:
    if not REGISTRY_ROOT.exists():
        return ok({"models": []})

    models = []
    for mdir in sorted(REGISTRY_ROOT.iterdir()):
        meta_path = mdir / "meta.json"
        if not meta_path.exists():
            continue
        try:
            with open(meta_path) as f:
                meta = json.load(f)
        except Exception:
            continue

        rec_type = (meta.get("config") or {}).get("recommender_type")
        if recommender_type_filter and rec_type != recommender_type_filter:
            continue
        if tag_filter:
            tags = set(meta.get("tags") or [])
            if not set(tag_filter).issubset(tags):
                continue

        models.append(
            {
                "model_name": meta.get("name"),
                "model_id": meta.get("model_id"),
                "recommender_type": rec_type,
                "tags": meta.get("tags") or [],
                "saved_at": meta.get("saved_at"),
                "metrics": meta.get("metrics") or {},
            }
        )

    return ok({"models": models})


TOOL_LIST_MODELS = Tool(
    name="list_models",
    description=(
        "List all models in the local registry (persistent, not just current session). "
        "Returns metadata and metrics so the user can pick one to load."
    ),
    input_schema={
        "type": "object",
        "properties": {
            "tag_filter": {"type": "array", "items": {"type": "string"}},
            "recommender_type_filter": {"type": "string"},
        },
    },
    fn=_list_models,
)


# ---------------------------------------------------------------------------
# load_model
# ---------------------------------------------------------------------------


def _load_model(model_name: str, session) -> dict[str, Any]:
    mdir = _model_dir(model_name)
    if not mdir.exists():
        return err("ModelNotFound", f"No model named '{model_name}' in registry at {REGISTRY_ROOT}.")
    if not (mdir / "meta.json").exists() or not (mdir / "model.pkl").exists():
        return err("CorruptRegistry", f"Registry entry '{model_name}' missing meta.json or model.pkl.")

    try:
        with open(mdir / "meta.json") as f:
            meta = json.load(f)
        with open(mdir / "model.pkl", "rb") as f:
            recommender = pickle.load(f)
    except Exception as e:
        return err(type(e).__name__, f"Failed to load '{model_name}': {e}")

    model_id = meta.get("model_id") or model_name
    handle = ModelHandle(
        model_id=model_id,
        name=meta.get("name", model_name),
        config=meta.get("config") or {},
        recommender=recommender,
        training_time_seconds=meta.get("training_time_seconds") or 0.0,
        datasets_used=meta.get("datasets_used") or {},
        metrics=dict(meta.get("metrics") or {}),
        tags=list(meta.get("tags") or []),
        created_at=meta.get("created_at") or "",
    )
    session.trained_models[model_id] = handle

    return ok(
        {
            "model_id": model_id,
            "model_name": handle.name,
            "config": handle.config,
            "metrics": handle.metrics,
            "tags": handle.tags,
        }
    )


TOOL_LOAD_MODEL = Tool(
    name="load_model",
    description=(
        "Load a registered model into the current session. Subsequent evaluate_model / "
        "compare_models / save_model calls can reference it by the returned model_id."
    ),
    input_schema={
        "type": "object",
        "properties": {
            "model_name": {"type": "string"},
        },
        "required": ["model_name"],
    },
    fn=_load_model,
)
