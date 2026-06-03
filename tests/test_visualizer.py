"""Tests for replay.visualizer data loading (no rendering)."""

from __future__ import annotations

import json
from typing import TYPE_CHECKING, Any

import pytest

from agentic_soccer.replay.visualizer import (
    Frame,
    load_replay,
    parse_frame,
)


if TYPE_CHECKING:
    from pathlib import Path


def _tick(**overrides: object) -> dict[str, Any]:
    """Return a minimal valid replay record, with optional field overrides."""
    record = {
        "tick": 1,
        "t": 42.3,
        "players": [
            {"id": 0, "team": "home", "role": "GK", "x": -0.9, "y": 0.0},
            {"id": 10, "team": "away", "role": "ST", "x": 0.3, "y": -0.1},
        ],
        "ball": {"x": 0.3, "y": -0.1},
        "score": [1, 0],
    }
    record.update(overrides)
    return record


def _write_jsonl(path: Path, records: list[dict[str, Any]]) -> Path:
    """Write records to ``path`` as JSONL and return the path."""
    path.write_text("\n".join(json.dumps(r) for r in records) + "\n", encoding="utf-8")
    return path


def test_load_replay_parses_all_frames(tmp_path: Path) -> None:
    """All non-blank lines load as Frame objects in file order."""
    path = _write_jsonl(tmp_path / "r.jsonl", [_tick(tick=1), _tick(tick=2)])
    frames = load_replay(path)
    assert len(frames) == 2
    assert all(isinstance(f, Frame) for f in frames)
    assert [f.tick for f in frames] == [1, 2]


def test_player_fields_parsed(tmp_path: Path) -> None:
    """Each player's id, team, role, and position are parsed faithfully."""
    frames = load_replay(_write_jsonl(tmp_path / "r.jsonl", [_tick()]))
    gk = frames[0].players[0]
    assert gk.id == 0
    assert gk.team == "home"
    assert gk.role == "GK"
    assert gk.x == pytest.approx(-0.9)
    assert gk.y == pytest.approx(0.0)


def test_ball_and_score_parsed(tmp_path: Path) -> None:
    """Ball coordinates, score tuple, and match time are parsed."""
    frames = load_replay(_write_jsonl(tmp_path / "r.jsonl", [_tick()]))
    frame = frames[0]
    assert frame.ball_x == pytest.approx(0.3)
    assert frame.ball_y == pytest.approx(-0.1)
    assert frame.score == (1, 0)
    assert frame.t == pytest.approx(42.3)


def test_blank_lines_are_skipped(tmp_path: Path) -> None:
    """Blank and whitespace-only lines between records are ignored."""
    path = tmp_path / "r.jsonl"
    path.write_text(
        json.dumps(_tick(tick=1)) + "\n\n   \n" + json.dumps(_tick(tick=2)) + "\n",
        encoding="utf-8",
    )
    frames = load_replay(path)
    assert [f.tick for f in frames] == [1, 2]


def test_no_coach_cycle_gives_empty_list(tmp_path: Path) -> None:
    """A frame without a coach_cycle yields an empty coach_cycles list."""
    frames = load_replay(_write_jsonl(tmp_path / "r.jsonl", [_tick()]))
    assert frames[0].coach_cycles == []


def test_coach_cycle_list_form(tmp_path: Path) -> None:
    """The logger writes coach_cycle as a list of blocks."""
    cycle = [
        {
            "tick": 1,
            "team": "home",
            "alerts": [
                {
                    "player_id": "sterling_lw",
                    "coach_reasoning": "press high",
                    "player_decision": "stepping up",
                    "override_written": True,
                },
            ],
        },
    ]
    frames = load_replay(_write_jsonl(tmp_path / "r.jsonl", [_tick(coach_cycle=cycle)]))
    cycles = frames[0].coach_cycles
    assert len(cycles) == 1
    assert cycles[0].team == "home"
    assert len(cycles[0].alerts) == 1
    alert = cycles[0].alerts[0]
    assert alert.player_id == "sterling_lw"
    assert alert.override_written is True
    assert alert.coach_reasoning == "press high"


def test_coach_cycle_dict_form_is_normalized(tmp_path: Path) -> None:
    """A bare dict (single-block form) is normalized to a one-element list."""
    cycle = {"team": "away", "alerts": [{"player_id": "9", "override_written": False}]}
    frames = load_replay(_write_jsonl(tmp_path / "r.jsonl", [_tick(coach_cycle=cycle)]))
    cycles = frames[0].coach_cycles
    assert len(cycles) == 1
    assert cycles[0].team == "away"
    assert cycles[0].alerts[0].player_id == "9"
    assert cycles[0].alerts[0].override_written is False


def test_parse_frame_accepts_list_ball() -> None:
    """Ball may be an [x, y] list as well as a {x, y} dict."""
    frame = parse_frame({"tick": 5, "t": 1.0, "players": [], "ball": [0.1, -0.2], "score": [0, 0]})
    assert frame.ball_x == pytest.approx(0.1)
    assert frame.ball_y == pytest.approx(-0.2)


def test_missing_file_raises(tmp_path: Path) -> None:
    """Loading a non-existent replay raises FileNotFoundError."""
    with pytest.raises(FileNotFoundError):
        load_replay(tmp_path / "does_not_exist.jsonl")
