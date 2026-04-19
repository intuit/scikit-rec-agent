"""CLI smoke test — verify argument parsing + provider auto-detection."""

from __future__ import annotations

import pytest

from scikit_rec_agent import cli


def test_auto_detect_anthropic(monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "x")
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    assert cli._auto_detect_provider() == "anthropic"


def test_auto_detect_openai(monkeypatch):
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    monkeypatch.setenv("OPENAI_API_KEY", "y")
    assert cli._auto_detect_provider() == "openai"


def test_auto_detect_both_raises(monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "x")
    monkeypatch.setenv("OPENAI_API_KEY", "y")
    with pytest.raises(SystemExit):
        cli._auto_detect_provider()


def test_auto_detect_none_raises(monkeypatch):
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    with pytest.raises(SystemExit):
        cli._auto_detect_provider()


def test_build_adapter_rejects_unknown_provider():
    with pytest.raises(ValueError):
        cli._build_adapter("bogus", None)
