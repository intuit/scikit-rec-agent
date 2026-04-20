"""Shared test fixtures."""

from __future__ import annotations

import os
from typing import Any

import pytest

from scikit_rec_agent.session import Session


@pytest.fixture
def session() -> Session:
    return Session()


@pytest.fixture
def sample_data_root() -> str:
    """Absolute path to skrec.examples.datasets/ for raw-path access in tests."""
    import skrec.examples.datasets as sd

    return os.path.dirname(sd.__file__)


@pytest.fixture
def binary_reward_paths(sample_data_root) -> dict[str, str]:
    base = os.path.join(sample_data_root, "sample_binary_reward")
    return {
        "interactions": os.path.join(base, "interactions.csv"),
        "users": os.path.join(base, "users.csv"),
        "items": os.path.join(base, "items.csv"),
    }


@pytest.fixture
def continuous_reward_paths(sample_data_root) -> dict[str, str]:
    base = os.path.join(sample_data_root, "sample_continuous_reward")
    return {
        "interactions": os.path.join(base, "interactions.csv"),
        "users": os.path.join(base, "users.csv"),
        "items": os.path.join(base, "items.csv"),
    }


@pytest.fixture
def tmp_registry(tmp_path, monkeypatch) -> Any:
    """Point the registry root at a tmp_path so save_model doesn't pollute $HOME."""
    from scikit_rec_agent.tools import registry

    target = tmp_path / "registry"
    monkeypatch.setattr(registry, "REGISTRY_ROOT", target)
    return target
