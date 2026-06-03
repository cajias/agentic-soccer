"""Tests for the wiring entry points (check_milestones, team_loop).

These don't require gfootball, a running MCP server, or pygame: they verify the
*defensive* behaviour of the milestone checker (every check fails gracefully when
its dependency is absent) and input validation of the coach entry point.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

import agents.coach_loop as coach_loop_module
from agents import coach_session
from entrypoints import check_milestones


if TYPE_CHECKING:
    from pathlib import Path


def test_run_coach_rejects_unknown_team() -> None:
    """run_coach raises ValueError for a team name that is not home/away."""
    with pytest.raises(ValueError, match="unknown team"):
        coach_session.run_coach("midfield")


def test_run_coach_valid_team_invokes_coach_loop(monkeypatch: pytest.MonkeyPatch) -> None:
    """run_coach('home') drives agents.coach_loop.coach_loop with team='home'.

    The valid path lazy-imports ``coach_loop``; we replace it with a recording
    fake so no HTTP client, LLM, or gfootball is touched. ``max_cycles=0`` is
    forwarded so the real loop would do zero work even if reached.
    """
    calls: list[dict[str, object]] = []

    def fake_coach_loop(team: str, **kwargs: object) -> None:
        calls.append({"team": team, **kwargs})

    monkeypatch.setattr(coach_loop_module, "coach_loop", fake_coach_loop)

    coach_session.run_coach("home", max_cycles=0)

    assert len(calls) == 1
    assert calls[0]["team"] == "home"
    assert calls[0]["max_cycles"] == 0
    # A tick_source default is supplied by run_coach when none is passed.
    assert callable(calls[0]["tick_source"])


def test_team_loop_token_map_matches_known_teams() -> None:
    """The auth token map is keyed by exactly the known teams."""
    assert set(coach_session.TOKENS) == {"home", "away"}


def test_check_milestones_all_defensive_returns_zero(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """In a clean env (no gfootball/server/replay/pygame) main() returns 0, no crash."""
    # Run where there is no match/replay.jsonl so M3/M5 also see nothing.
    monkeypatch.chdir(tmp_path)
    count = check_milestones.main()
    assert count == 0


def test_check_milestones_individual_checks_are_falsey(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Each milestone check returns False (not raises) when its dependency is absent."""
    monkeypatch.chdir(tmp_path)
    # Each returns a bool and does not raise when its dependency is unavailable.
    assert check_milestones.check_m1() is False  # gfootball absent
    assert check_milestones.check_m3() is False  # no replay file
    assert check_milestones.check_m5() is False  # pygame absent
