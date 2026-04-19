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
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from scikit_rec_agent.session import ModelHandle
from scikit_rec_agent.tools import Tool, err, ok

REGISTRY_ROOT = Path.home() / ".scikit-rec" / "registry"

# Reject anything that isn't a simple identifier — keeps the LLM (or a
# confused user) from steering save/load toward arbitrary filesystem paths
# via `../` traversal or absolute paths. Both cases would otherwise bypass
# REGISTRY_ROOT entirely, and load_model's pickle.load would become an RCE
# vector against any .pkl the agent can be convinced to point at.
_VALID_MODEL_NAME = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]*$")


def _validate_model_name(model_name: str) -> str | None:
    """Return an error message if model_name is unsafe, else None."""
    if not isinstance(model_name, str) or not model_name:
        return "model_name must be a non-empty string."
    if model_name in {".", ".."}:
        return "model_name must not be '.' or '..'."
    if not _VALID_MODEL_NAME.match(model_name):
        return "model_name must match [A-Za-z0-9][A-Za-z0-9_.-]* (no slashes, no leading dots)."
    return None


def _safe_model_dir(model_name: str) -> Path:
    """Resolve a registry path and confirm it's inside REGISTRY_ROOT.

    Belt-and-suspenders on top of _validate_model_name: even if that
    validation is softened in future, `.resolve().is_relative_to()` blocks
    traversal.
    """
    root = REGISTRY_ROOT.resolve()
    mdir = (REGISTRY_ROOT / model_name).resolve()
    if not mdir.is_relative_to(root):
        raise ValueError(f"model_name '{model_name}' escapes registry root.")
    return mdir


def _save_model(model_id: str, session, tags: list[str] | None = None) -> dict[str, Any]:
    handle = session.trained_models.get(model_id)
    if handle is None:
        return err("ModelNotFound", f"No trained model '{model_id}' in session.")

    name_err = _validate_model_name(handle.name)
    if name_err:
        return err("InvalidModelName", name_err)
    try:
        mdir = _safe_model_dir(handle.name)
    except ValueError as e:
        return err("InvalidModelName", str(e))

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
    name_err = _validate_model_name(model_name)
    if name_err:
        return err("InvalidModelName", name_err)
    try:
        mdir = _safe_model_dir(model_name)
    except ValueError as e:
        return err("InvalidModelName", str(e))
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
        "compare_models / save_model calls can reference it by the returned model_id. "
        "TRUST NOTE: load_model unpickles arbitrary Python objects from the registry "
        "path. Only load models you or a colleague saved — don't point it at files you "
        "didn't produce yourself."
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
