"""Tests for the template-based state narrator."""

from __future__ import annotations

from typing import Any

import pytest

from agentic_soccer.simulator.narrator import GAME_MODES, Narrator, narrate


# Eleven default positions per team in the gfootball attacking-toward-+x layout
# for the home side (away mirrors these). Roughly: GK deep, defenders, mids,
# forwards. Values are illustrative, not from a real engine dump.
_HOME = [
    [-0.95, 0.00],  # 0 GK
    [-0.70, -0.10],  # 1 CB left
    [-0.70, 0.10],  # 2 CB right
    [-0.60, -0.30],  # 3 LB
    [-0.60, 0.30],  # 4 RB
    [-0.30, -0.15],  # 5 CM left
    [-0.30, 0.00],  # 6 CM center
    [-0.30, 0.15],  # 7 CM right
    [0.10, -0.30],  # 8 LW
    [0.10, 0.30],  # 9 RW
    [0.20, 0.00],  # 10 ST
]

_AWAY = [
    [0.95, 0.00],
    [0.70, 0.10],
    [0.70, -0.10],
    [0.60, 0.30],
    [0.60, -0.30],
    [0.30, 0.15],
    [0.30, 0.00],
    [0.30, -0.15],
    [-0.10, 0.30],
    [-0.10, -0.30],
    [-0.20, 0.00],
]

_STILL = [[0.0, 0.0] for _ in range(11)]


def _base_obs(**overrides: object) -> dict[str, Any]:
    """A normal-play observation; override any field via kwargs."""
    obs = {
        "left_team": [list(p) for p in _HOME],
        "right_team": [list(p) for p in _AWAY],
        "left_team_direction": [list(p) for p in _STILL],
        "right_team_direction": [list(p) for p in _STILL],
        "ball": [0.0, 0.0, 0.0],
        "ball_owned_team": -1,
        "ball_owned_player": -1,
        "score": [0, 0],
        "steps_remaining": 3000,
        "game_mode": 0,
    }
    obs.update(overrides)
    return obs


def test_kickoff_contains_expected_phrases() -> None:
    """Kickoff renders the clock, score, mode label and every role."""
    obs = _base_obs(game_mode=1, steps_remaining=3000)
    report = narrate(obs, team="home")

    assert "MATCH STATE (0:00)" in report
    assert "Score: Home 0 – Away 0" in report  # noqa: RUF001 - en dash matches narrator output
    assert "KickOff" in report
    assert "YOUR TEAM: home" in report
    # All eleven roles are described.
    for role in ("GK", "CB (left)", "ST", "LW", "RW"):
        assert role in report


def test_normal_play_with_home_possession() -> None:
    """Home possession names the carrier, zone and on-the-ball phrase."""
    # Home ST carries the ball in the attacking third.
    obs = _base_obs(
        ball=[0.20, 0.00, 0.0],
        ball_owned_team=0,
        ball_owned_player=10,
        score=[1, 0],
        steps_remaining=1500,  # 45:00 elapsed
    )
    report = narrate(obs, team="home")

    assert "MATCH STATE (45:00)" in report
    assert "Score: Home 1 – Away 0" in report  # noqa: RUF001 - en dash matches narrator output
    assert "POSSESSION: Home" in report
    assert "ST has the ball" in report
    assert "attacking third" in report
    # The carrier is described as on the ball.
    assert "on the ball" in report


def test_contested_loose_ball() -> None:
    """An unowned ball reads as contested with a loose-ball note."""
    obs = _base_obs(ball=[0.0, 0.0, 0.0], ball_owned_team=-1)
    report = narrate(obs, team="home")
    assert "POSSESSION: Contested" in report
    assert "loose ball" in report


def test_corner_kick_phrases() -> None:
    """A corner game mode surfaces the set-piece line."""
    # Ball deep in the opposition corner; away has a corner against home? No --
    # narrate from home view, ball near opposition goal line corner.
    obs = _base_obs(
        game_mode=4,
        ball=[0.99, 0.40, 0.0],
        ball_owned_team=0,
        ball_owned_player=9,
    )
    report = narrate(obs, team="home")
    assert "Corner" in report
    assert "Set piece in play: Corner." in report
    assert "POSSESSION: Home" in report


def test_goal_kick_phrases() -> None:
    """A goal kick puts the GK on the ball in their own box."""
    obs = _base_obs(
        game_mode=2,
        ball=[-0.95, 0.0, 0.0],
        ball_owned_team=0,
        ball_owned_player=0,
    )
    report = narrate(obs, team="home")
    assert "GoalKick" in report
    assert "Set piece in play: GoalKick." in report
    # GK has the ball in their own penalty area.
    assert "GK has the ball" in report
    assert "own penalty area" in report


def test_away_perspective_mirrors_zones() -> None:
    """The away view mirrors zones relative to the home view."""
    # Ball at x=+0.9 is the home attacking end == the away defensive end.
    obs = _base_obs(
        ball=[0.90, 0.0, 0.0],
        ball_owned_team=1,
        ball_owned_player=0,
    )
    home_report = narrate(obs, team="home")
    away_report = narrate(obs, team="away")

    assert "YOUR TEAM: away" in away_report
    assert "POSSESSION: Away" in away_report
    # Same ball, mirrored: it sits in home's opposition box, which is away's
    # own box. The perspective flip must produce opposite zone phrasing.
    assert "opposition penalty area" in home_report
    assert "own penalty area" in away_report


def test_pressure_description_changes_with_opponents() -> None:
    """Many opponents deep in our third trigger the heavy-pressure read."""
    # Push five away players deep into the home defensive third.
    away = [list(p) for p in _AWAY]
    for i in range(5):
        away[i] = [-0.80, 0.0]
    obs = _base_obs(right_team=away)
    report = narrate(obs, team="home")
    assert "Heavy pressure" in report


def test_narrator_class_binds_team() -> None:
    """The Narrator class produces the same output as the bound function call."""
    obs = _base_obs()
    assert narrate(obs, "away") == Narrator("away").narrate(obs)


def test_invalid_team_raises() -> None:
    """An unknown team perspective raises ValueError."""
    with pytest.raises(ValueError, match="team must be one of"):
        narrate(_base_obs(), team="sideline")


def test_all_game_modes_render_without_error() -> None:
    """Every supported game mode renders a report without raising."""
    for mode in GAME_MODES:
        report = narrate(_base_obs(game_mode=mode), team="home")
        assert "MATCH STATE" in report
