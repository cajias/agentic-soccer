"""End-to-end test: a recorded game can be loaded and rendered correctly.

Two halves:

1. Data path -- a synthetic replay.jsonl (with a coach_cycle tick) loads via the
   visualizer's ``load_replay`` and every frame's coordinates sit inside the
   pitch bounds (x in [-1, 1], y in [-0.42, 0.42]).
2. Graphics path -- headless pygame (SDL dummy driver) renders each frame using
   the real draw helpers, including the coach overlay, without raising. This
   proves the rendering pipeline works on real replay data.
"""

from __future__ import annotations

import json
import os
from typing import TYPE_CHECKING, Any

import pytest

from replay.visualizer import (
    HEIGHT,
    WIDTH,
    _draw_ball,
    _draw_boxes,
    _draw_coach_overlay,
    _draw_hud,
    _draw_pitch,
    _draw_players,
    _load_font,
    load_replay,
)


if TYPE_CHECKING:
    from collections.abc import Iterator
    from pathlib import Path
    from types import ModuleType

X_MIN, X_MAX = -1.0, 1.0
Y_MIN, Y_MAX = -0.42, 0.42


def _player(pid: int, team: str, role: str, x: float, y: float) -> dict[str, Any]:
    return {"id": pid, "team": team, "role": role, "x": x, "y": y}


def _make_players() -> list[dict[str, Any]]:
    """11 home + 11 away players with in-bounds coordinates."""
    roles = ["GK", "LB", "CB", "CB", "RB", "LM", "CM", "CM", "RM", "ST", "ST"]
    players: list[dict[str, Any]] = []
    for i in range(11):
        x = -0.9 + 0.15 * i  # spread across [-0.9, 0.6]
        y = -0.4 + 0.08 * i  # spread within [-0.4, 0.4]
        players.append(_player(i, "home", roles[i], round(x, 3), round(y, 3)))
    for i in range(11):
        x = 0.9 - 0.15 * i
        y = 0.4 - 0.08 * i
        players.append(_player(11 + i, "away", roles[i], round(x, 3), round(y, 3)))
    return players


def _tick(tick: int, *, coach: bool = False) -> dict[str, Any]:
    record: dict[str, Any] = {
        "tick": tick,
        "t": float(tick) * 0.1,
        "players": _make_players(),
        "ball": {"x": 0.1 * tick - 0.2, "y": 0.05 * tick - 0.1},
        "score": [tick // 3, 0],
    }
    if coach:
        record["coach_cycle"] = [
            {
                "tick": tick,
                "team": "home",
                "alerts": [
                    {
                        "player_id": "9",
                        "coach_reasoning": "push the striker high",
                        "player_decision": "stepping up",
                        "override_written": True,
                    },
                ],
            },
        ]
    return record


def _write_replay(path: Path) -> Path:
    records = [
        _tick(0),
        _tick(1),
        _tick(2, coach=True),  # one tick carries a coach cycle
        _tick(3),
        _tick(4),
    ]
    path.write_text("\n".join(json.dumps(r) for r in records) + "\n", encoding="utf-8")
    return path


# ---------------------------------------------------------------------------
# Data path
# ---------------------------------------------------------------------------
def test_replay_loads_with_valid_coords(tmp_path: Path) -> None:
    """The synthetic replay loads as 5 frames, all coordinates in bounds."""
    frames = load_replay(_write_replay(tmp_path / "replay.jsonl"))
    assert len(frames) == 5

    for frame in frames:
        assert len(frame.players) == 22
        for p in frame.players:
            assert X_MIN <= p.x <= X_MAX, f"x out of range: {p.x}"
            assert Y_MIN <= p.y <= Y_MAX, f"y out of range: {p.y}"
        assert X_MIN <= frame.ball_x <= X_MAX
        assert Y_MIN <= frame.ball_y <= Y_MAX

    # The coach cycle on tick 2 parsed into a CoachCycle block.
    coach_frame = frames[2]
    assert len(coach_frame.coach_cycles) == 1
    assert coach_frame.coach_cycles[0].team == "home"


# ---------------------------------------------------------------------------
# Graphics path (headless)
# ---------------------------------------------------------------------------
@pytest.fixture
def headless_pygame() -> Iterator[ModuleType]:
    """Initialise pygame against the SDL dummy video/audio drivers."""
    os.environ["SDL_VIDEODRIVER"] = "dummy"
    os.environ["SDL_AUDIODRIVER"] = "dummy"
    pygame = pytest.importorskip("pygame")
    if pygame.init()[1] > 0:  # (passed, failed) module counts
        pytest.skip("pygame could not initialise headlessly")
    try:
        yield pygame
    finally:
        pygame.quit()


def test_replay_renders_headless(tmp_path: Path, headless_pygame: ModuleType) -> None:
    """Every draw helper runs on real frames -- including the coach overlay."""
    pygame = headless_pygame

    frames = load_replay(_write_replay(tmp_path / "replay.jsonl"))
    surface = pygame.Surface((WIDTH, HEIGHT))
    font = _load_font(12)

    for frame in frames:
        _draw_pitch(surface)
        _draw_boxes(surface)
        _draw_players(surface, frame, font, set())
        _draw_ball(surface, frame)
        _draw_hud(surface, frame, font, 1.0)
        # Render the coach overlay on whichever frame carries a cycle.
        if frame.coach_cycles:
            _draw_coach_overlay(surface, frame, frame.coach_cycles, font)

    # Sanity: the coach frame exercised the overlay path at least once.
    assert any(f.coach_cycles for f in frames)
