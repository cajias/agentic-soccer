"""Smoke tests for check_milestones with heavy/optional imports mocked.

The real engine, MCP server, requests, and pygame are not assumed to be present.
These tests confirm the checker (a) stays importable, (b) never raises out of a
check, and (c) reports the right pass/fail given mocked dependencies.
"""

from __future__ import annotations

import sys
import types
from typing import TYPE_CHECKING

import pytest

import check_milestones as cm


if TYPE_CHECKING:
    from pathlib import Path


def _fake_module(name: str, **attrs: object) -> types.ModuleType:
    """Build a throwaway module object exposing the given attributes."""
    mod = types.ModuleType(name)
    for key, value in attrs.items():
        setattr(mod, key, value)
    return mod


def test_module_imports_without_heavy_deps():
    """Importing the checker must not require requests, pygame, or the engine."""
    assert hasattr(cm, "main")
    assert callable(cm.check_m1)


@pytest.mark.parametrize(
    "check_name",
    ["check_m1", "check_m2", "check_m3", "check_m4", "check_m5"],
)
def test_checks_return_bool_without_raising(check_name: str):
    """Every check degrades to a bool even with no server or deps present."""
    result = getattr(cm, check_name)()
    assert isinstance(result, bool)


def test_main_returns_count_in_range():
    """main() returns an integer milestone count between 0 and 5."""
    count = cm.main()
    assert isinstance(count, int)
    assert 0 <= count <= cm._TOTAL_MILESTONES


def test_check_m3_detects_coach_cycle(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    """A replay file containing a coach_cycle entry passes M3."""
    replay = tmp_path / "replay.jsonl"
    replay.write_text(
        '{"tick": 1}\n{"coach_cycle": {"team": "home"}}\n',
        encoding="utf-8",
    )
    monkeypatch.setattr(cm, "_REPLAY_PATH", str(replay))
    assert cm.check_m3() is True


def test_check_m3_without_coach_cycle(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    """A replay file with no coach_cycle entry fails M3."""
    replay = tmp_path / "replay.jsonl"
    replay.write_text('{"tick": 1}\n{"tick": 2}\n', encoding="utf-8")
    monkeypatch.setattr(cm, "_REPLAY_PATH", str(replay))
    assert cm.check_m3() is False


def test_check_m3_missing_file(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    """A missing replay file fails M3 without raising."""
    monkeypatch.setattr(cm, "_REPLAY_PATH", str(tmp_path / "nope.jsonl"))
    assert cm.check_m3() is False


def test_check_m1_success_with_mocked_engine(monkeypatch: pytest.MonkeyPatch):
    """M1 passes when a mocked SoccerEngine reports a 2-element score."""

    class _FakeEngine:
        """Stand-in for the real engine that returns a finished match's stats."""

        def __init__(self, **_: object) -> None: ...

        def run(self) -> dict:
            """Return minimal end-of-match stats with a score."""
            return {"score": [1, 0]}

    monkeypatch.setitem(
        sys.modules,
        "simulator.engine",
        _fake_module("simulator.engine", SoccerEngine=_FakeEngine),
    )
    assert cm.check_m1() is True


def test_check_m2_success_with_mocked_requests(monkeypatch: pytest.MonkeyPatch):
    """M2 passes when a mocked requests returns an ok response with text."""

    class _FakeResp:
        ok = True
        text = "MATCH STATE (0:00) | Score: Home 0 - Away 0"

    def _get(*_args: object, **_kwargs: object) -> _FakeResp:
        return _FakeResp()

    monkeypatch.setitem(sys.modules, "requests", _fake_module("requests", get=_get))
    assert cm.check_m2() is True


def test_check_m5_success_with_mocked_visualizer(monkeypatch: pytest.MonkeyPatch):
    """M5 passes when pygame imports and load_replay yields frames."""

    def _load_replay(_path: str) -> list[dict]:
        return [{"tick": 0}]

    monkeypatch.setitem(sys.modules, "pygame", _fake_module("pygame"))
    monkeypatch.setitem(sys.modules, "replay", _fake_module("replay"))
    monkeypatch.setitem(
        sys.modules,
        "replay.visualizer",
        _fake_module("replay.visualizer", load_replay=_load_replay),
    )
    assert cm.check_m5() is True
