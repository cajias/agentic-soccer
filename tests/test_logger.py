"""Tests for the replay logger."""

from __future__ import annotations

import json
from typing import TYPE_CHECKING, Any

from agentic_soccer.replay.logger import ReplayLogger


if TYPE_CHECKING:
    from pathlib import Path


def _obs() -> dict[str, Any]:
    """Minimal observation: 11 home + 11 away positions and a ball."""
    left = [[i * 0.01, i * 0.02] for i in range(11)]
    right = [[-i * 0.01, -i * 0.02] for i in range(11)]
    return {"left_team": left, "right_team": right, "ball": [0.5, -0.25]}


def test_log_tick_writes_valid_json_line(tmp_path: Path) -> None:
    """log_tick writes one valid JSON line with 22 players, ball, and score."""
    path = tmp_path / "match" / "replay.jsonl"
    logger = ReplayLogger(str(path))
    logger.log_tick(0, 0.0, _obs(), [0, 0])
    logger.close()

    lines = path.read_text().splitlines()
    assert len(lines) == 1
    record = json.loads(lines[0])
    assert record["tick"] == 0
    assert record["t"] == 0.0
    assert record["score"] == [0, 0]
    assert record["ball"] == {"x": 0.5, "y": -0.25}
    assert len(record["players"]) == 22
    # First home player is the goalkeeper; first away player too.
    assert record["players"][0] == {"id": 0, "team": "home", "role": "GK", "x": 0.0, "y": 0.0}
    assert record["players"][11]["team"] == "away"
    assert record["players"][11]["role"] == "GK"
    assert record["players"][10]["role"] == "ST"


def test_file_and_dir_created_if_missing(tmp_path: Path) -> None:
    """The log file and any missing parent directories are created on init."""
    path = tmp_path / "match" / "nested" / "replay.jsonl"
    assert not path.parent.exists()
    logger = ReplayLogger(str(path))
    logger.log_tick(0, 0.0, _obs(), [0, 0])
    logger.close()
    assert path.exists()


def test_multiple_ticks_accumulate(tmp_path: Path) -> None:
    """Successive log_tick calls accumulate as ordered JSONL lines."""
    path = tmp_path / "match" / "replay.jsonl"
    logger = ReplayLogger(str(path))
    for tick in range(5):
        logger.log_tick(tick, tick * 0.1, _obs(), [tick, 0])
    logger.close()

    lines = path.read_text().splitlines()
    assert len(lines) == 5
    for tick, line in enumerate(lines):
        record = json.loads(line)
        assert record["tick"] == tick
        assert record["score"] == [tick, 0]


def test_log_coach_cycle_embeds_into_last_tick(tmp_path: Path) -> None:
    """log_coach_cycle embeds a coach_cycle block into the most recent tick."""
    path = tmp_path / "match" / "replay.jsonl"
    logger = ReplayLogger(str(path))
    logger.log_tick(0, 0.0, _obs(), [0, 0])
    logger.log_tick(1, 0.1, _obs(), [0, 0])

    alerts = [
        {
            "player_id": 10,
            "coach_reasoning": "press high",
            "player_decision": "advance",
            "override_written": True,
            "target_position": [0.6, 0.0],
            "duration_ticks": 30,
            "response_time_s": 1.2,
        },
    ]
    logger.log_coach_cycle(1, "home", alerts)
    logger.close()

    lines = path.read_text().splitlines()
    assert len(lines) == 2
    # First tick is untouched.
    assert "coach_cycle" not in json.loads(lines[0])
    last = json.loads(lines[1])
    assert last["tick"] == 1
    assert last["coach_cycle"][0]["team"] == "home"
    assert last["coach_cycle"][0]["alerts"] == alerts


def test_log_tick_after_coach_cycle_appends_correctly(tmp_path: Path) -> None:
    """Appending continues correctly after an atomic coach-cycle rewrite."""
    path = tmp_path / "match" / "replay.jsonl"
    logger = ReplayLogger(str(path))
    logger.log_tick(0, 0.0, _obs(), [0, 0])
    logger.log_coach_cycle(0, "away", [{"player_id": 5}])
    logger.log_tick(1, 0.1, _obs(), [1, 0])
    logger.close()

    lines = path.read_text().splitlines()
    assert len(lines) == 2
    assert json.loads(lines[0])["coach_cycle"][0]["team"] == "away"
    assert json.loads(lines[1])["tick"] == 1


def test_log_coach_cycle_targets_matching_tick_not_last_line(tmp_path: Path) -> None:
    """A late cycle for an older tick lands on that tick's line, not the newest.

    Mirrors the real race: the simulator writes newer ticks while the coach is
    mid-decision, so the cycle must attach to its own ``tick``.
    """
    path = tmp_path / "match" / "replay.jsonl"
    logger = ReplayLogger(str(path))
    logger.log_tick(0, 0.0, _obs(), [0, 0])
    logger.log_tick(1, 0.1, _obs(), [0, 0])
    logger.log_tick(2, 0.2, _obs(), [0, 0])
    # Coach started at tick 0 but only finished after ticks 1 and 2 were written.
    logger.log_coach_cycle(0, "home", [{"player_id": 9}])
    logger.close()

    records = [json.loads(line) for line in path.read_text().splitlines()]
    assert "coach_cycle" in records[0]
    assert records[0]["coach_cycle"][0]["tick"] == 0
    assert "coach_cycle" not in records[1]
    assert "coach_cycle" not in records[2]
