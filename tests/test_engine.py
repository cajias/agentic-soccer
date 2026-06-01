"""Tests for the soccer match engine.

gfootball is mocked via ``sys.modules`` and the shared MCP game state is replaced
with a fake, so these run without the compiled C++ engine or the MCP server. The
fake env supplies (22, 115) observations plus a raw observation (score + the keys
the narrator reads) and captures the actions the engine applies.
"""

from __future__ import annotations

import json
import sys
import types
from typing import TYPE_CHECKING, Any

import numpy as np
import pytest

from simulator.engine import (
    RIGHT,
    SoccerEngine,
    _direction_action,
    default_action,
)


if TYPE_CHECKING:
    from pathlib import Path


# ---------------------------------------------------------------------------
# Fakes
# ---------------------------------------------------------------------------
def _raw_obs(score: list[int]) -> dict[str, Any]:
    """A minimal gfootball-style raw observation the narrator can consume."""
    return {
        "left_team": np.zeros((11, 2), dtype=np.float32),
        "right_team": np.zeros((11, 2), dtype=np.float32),
        "left_team_direction": np.zeros((11, 2), dtype=np.float32),
        "right_team_direction": np.zeros((11, 2), dtype=np.float32),
        "ball": np.zeros(3, dtype=np.float32),
        "ball_owned_team": -1,
        "game_mode": 0,
        "score": list(score),
    }


class _FakeUnwrapped:
    def __init__(self, env: _FakeEnv) -> None:
        self._env = env

    def observation(self) -> list[dict[str, Any]]:
        # 22 controlled players; only index 0 (absolute home frame) is read.
        return [_raw_obs(self._env.score) for _ in range(22)]


class _FakeEnv:
    """Minimal gfootball env: zero observations, a scripted home goal at t=5."""

    def __init__(self, steps_until_done: int = 10_000) -> None:
        self._t = 0
        self._steps_until_done = steps_until_done
        self.score = [0, 0]
        self.unwrapped = _FakeUnwrapped(self)
        self.closed = False
        self.last_actions: list[int] | None = None

    def reset(self) -> np.ndarray:
        self._t = 0
        return np.zeros((22, 115), dtype=np.float32)

    def step(
        self, actions: list[int],
    ) -> tuple[np.ndarray, np.ndarray, bool, dict[str, float]]:
        assert len(actions) == 22, "engine must supply 22 actions"
        assert all(0 <= int(a) <= 18 for a in actions), "actions must be valid ints"
        self.last_actions = list(actions)
        self._t += 1
        if self._t == 5:  # exercise score-extraction plumbing
            self.score[0] += 1
        obs = np.zeros((22, 115), dtype=np.float32)
        reward = np.zeros(22, dtype=np.float32)
        done = self._t >= self._steps_until_done
        return obs, reward, done, {"score_reward": 0.0}

    def close(self) -> None:
        self.closed = True


class _FakeGameState:
    """Records tick/narrator writes and serves seeded per-team overrides."""

    def __init__(self, overrides: dict[str, dict[str, Any]] | None = None) -> None:
        self._overrides = overrides or {"home": {}, "away": {}}
        self.tick = 0
        self.narrator: dict[str, str] = {"home": "", "away": ""}
        self.tick_history: list[int] = []

    def set_tick(self, tick: int) -> None:
        self.tick = tick
        self.tick_history.append(tick)

    def get_active_overrides(self, team: str) -> dict[str, dict[str, Any]]:
        return self._overrides.get(team, {})

    def set_narrator(self, team: str, text: str) -> None:
        self.narrator[team] = text


def _install_fake_gfootball(monkeypatch: pytest.MonkeyPatch, env: _FakeEnv) -> None:
    """Inject a fake ``gfootball.env`` module returning ``env``."""
    fake_env_mod = types.ModuleType("gfootball.env")
    fake_env_mod.create_environment = lambda **_kwargs: env  # type: ignore[attr-defined]
    fake_pkg = types.ModuleType("gfootball")
    fake_pkg.env = fake_env_mod  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "gfootball", fake_pkg)
    monkeypatch.setitem(sys.modules, "gfootball.env", fake_env_mod)


def _override_record(target: list[float], duration: int = 0) -> dict[str, Any]:
    """Shape a GAME_STATE override record as the MCP server stores it."""
    return {
        "player_id": "0",
        "override": {"target_position": target, "duration_ticks": duration, "reasoning": "x"},
        "created_tick": 0,
        "expires_at_tick": duration,
    }


# ---------------------------------------------------------------------------
# Pure helpers (no env required)
# ---------------------------------------------------------------------------
@pytest.mark.parametrize(
    ("dx", "dy", "expected"),
    [
        (1, 0, 5),  # RIGHT
        (0, 1, 7),  # BOTTOM (+y is down)
        (-1, 0, 1),  # LEFT
        (0, -1, 3),  # TOP
        (1, 1, 6),  # BOTTOM_RIGHT
        (-1, -1, 2),  # TOP_LEFT
        (1, -1, 4),  # TOP_RIGHT
        (-1, 1, 8),  # BOTTOM_LEFT
        (0, 0, 0),  # IDLE (dead zone)
    ],
)
def test_direction_action(dx: int, dy: int, expected: int) -> None:
    """Cardinal/diagonal deltas map to the expected gfootball action ids."""
    assert _direction_action(dx, dy) == expected


def test_default_action_goalkeeper_holds_line() -> None:
    """The goalkeeper heuristic never sends the keeper further upfield."""
    # GK far from its goal line (own goal at x=-1) should not run further upfield.
    assert default_action(0, np.array([0.0, 0.0]), np.array([0.0, 0.0])) != RIGHT


def test_default_action_returns_valid_action() -> None:
    """Every role's heuristic returns a valid gfootball action id (0-18)."""
    for idx in range(11):
        action = default_action(idx, np.array([0.1, 0.0]), np.array([0.3, 0.2]))
        assert 0 <= action <= 18


# ---------------------------------------------------------------------------
# Full run
# ---------------------------------------------------------------------------
def test_engine_runs_short_match(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """A short match advances ticks, scores, narrates, and writes a replay."""
    env = _FakeEnv()
    _install_fake_gfootball(monkeypatch, env)
    gs = _FakeGameState()

    replay = tmp_path / "replay.jsonl"
    engine = SoccerEngine(replay_path=str(replay), match_steps=50, game_state=gs)
    stats = engine.run()

    assert stats["score"] == [1, 0]
    assert stats["ticks"] == 50
    assert stats["home_goals"] == 1
    assert stats["away_goals"] == 0
    assert env.closed

    # Tick was published to the shared state, and narrator text for both teams.
    assert gs.tick == 50
    assert gs.tick_history[:3] == [0, 1, 2]
    assert gs.narrator["home"]
    assert gs.narrator["away"]

    lines = replay.read_text(encoding="utf-8").strip().splitlines()
    assert len(lines) == 50
    first = json.loads(lines[0])
    assert first["tick"] == 0
    assert first["score"] == [0, 0]
    assert len(first["players"]) == 22
    assert {p["team"] for p in first["players"]} == {"home", "away"}
    assert json.loads(lines[-1])["score"] == [1, 0]


def test_override_steers_player(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """An active target_position override steers the player toward the target."""
    env = _FakeEnv()
    _install_fake_gfootball(monkeypatch, env)
    # Home player 0 sits at (0, 0); override target (1, 0) -> move RIGHT.
    gs = _FakeGameState({"home": {"0": _override_record([1.0, 0.0])}, "away": {}})

    engine = SoccerEngine(replay_path=str(tmp_path / "r.jsonl"), match_steps=1, game_state=gs)
    engine.run()

    assert env.last_actions is not None
    assert env.last_actions[0] == RIGHT


def test_override_expiry_handled_by_game_state(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """An empty active-override set falls back to heuristics (no crash)."""
    env = _FakeEnv()
    _install_fake_gfootball(monkeypatch, env)
    gs = _FakeGameState({"home": {}, "away": {}})

    engine = SoccerEngine(replay_path=str(tmp_path / "r.jsonl"), match_steps=3, game_state=gs)
    stats = engine.run()

    assert stats["ticks"] == 3
    assert env.last_actions is not None
    assert len(env.last_actions) == 22
