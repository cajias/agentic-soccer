"""Tests for the wiring entry points (check_milestones, team_loop).

These don't require gfootball, a running MCP server, or pygame: they verify the
*defensive* behaviour of the milestone checker (every check fails gracefully when
its dependency is absent) and input validation of the coach entry point.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

import check_milestones
import team_loop


if TYPE_CHECKING:
    from pathlib import Path


def test_run_coach_rejects_unknown_team() -> None:
    """run_coach raises ValueError for a team name that is not home/away."""
    with pytest.raises(ValueError, match="unknown team"):
        team_loop.run_coach("midfield")


def test_team_loop_token_map_matches_known_teams() -> None:
    """The auth token map is keyed by exactly the known teams."""
    assert set(team_loop.TOKENS) == {"home", "away"}


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
