"""End-to-end test: a full short match can be played.

gfootball is not installed on the host (only in the Docker image), so we inject
a fake ``gfootball.env`` module via ``sys.modules`` before importing the engine.
The fake env, raw-observation shape, and the gfootball installer are reused from
``test_engine.py``. Unlike the unit tests, this exercise drives the *real*
:class:`mcp_server.game_state.GameState` so the full
``set_override`` -> ``get_active_overrides`` -> engine-applies-it contract runs
end to end, and asserts a real replay.jsonl is produced on disk.
"""

from __future__ import annotations

import json
import sys
import types
from typing import TYPE_CHECKING

import numpy as np

from mcp_server.game_state import GameState
from simulator.engine import RIGHT, SoccerEngine


if TYPE_CHECKING:
    from pathlib import Path

    from _pytest.monkeypatch import MonkeyPatch


# ---------------------------------------------------------------------------
# Fakes (mirroring tests/test_engine.py so this runs without the C++ engine)
# ---------------------------------------------------------------------------
def _raw_obs(score: list[int]) -> dict:
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

    def observation(self) -> list[dict]:
        return [_raw_obs(self._env.score) for _ in range(22)]


class _FakeEnv:
    """Minimal gfootball env: zero observations with a scripted home goal at t=5."""

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
        if self._t == 5:
            self.score[0] += 1
        obs = np.zeros((22, 115), dtype=np.float32)
        reward = np.zeros(22, dtype=np.float32)
        done = self._t >= self._steps_until_done
        return obs, reward, done, {"score_reward": 0.0}

    def close(self) -> None:
        self.closed = True


def _install_fake_gfootball(monkeypatch, env: _FakeEnv) -> None:
    """Inject a fake ``gfootball.env`` module returning ``env``."""
    fake_env_mod = types.ModuleType("gfootball.env")
    fake_env_mod.create_environment = lambda **_kwargs: env  # type: ignore[attr-defined]
    fake_pkg = types.ModuleType("gfootball")
    fake_pkg.env = fake_env_mod  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "gfootball", fake_pkg)
    monkeypatch.setitem(sys.modules, "gfootball.env", fake_env_mod)


# ---------------------------------------------------------------------------
# E2E: a game can be played
# ---------------------------------------------------------------------------
def test_e2e_match_plays_and_writes_replay(tmp_path: Path, monkeypatch) -> None:
    """A 50-step match runs end to end and produces a well-formed replay file."""
    env = _FakeEnv()
    _install_fake_gfootball(monkeypatch, env)

    gs = GameState()  # fresh instance for isolation; not the module singleton

    replay = tmp_path / "replay.jsonl"
    engine = SoccerEngine(replay_path=str(replay), match_steps=50, game_state=gs)
    stats = engine.run()

    # --- return contract ---------------------------------------------------
    assert isinstance(stats["score"], list)
    assert len(stats["score"]) == 2
    assert stats["ticks"] > 0
    assert stats["ticks"] == 50
    assert stats["home_goals"] == stats["score"][0]
    assert stats["away_goals"] == stats["score"][1]
    assert env.closed

    # --- replay file: one JSON object per tick -----------------------------
    assert replay.exists()
    lines = replay.read_text(encoding="utf-8").strip().splitlines()
    assert len(lines) == 50

    for raw in lines:  # every line is valid JSON with the expected shape
        record = json.loads(raw)
        assert "tick" in record
        assert isinstance(record["score"], list)
        assert len(record["score"]) == 2

    first = json.loads(lines[0])
    assert first["tick"] == 0
    # 22 players = 11 home + 11 away, each with id/team/role/x/y.
    assert len(first["players"]) == 22
    teams = [p["team"] for p in first["players"]]
    assert teams.count("home") == 11
    assert teams.count("away") == 11
    for player in first["players"]:
        for field in ("id", "team", "role", "x", "y"):
            assert field in player, f"player missing {field}"
    # ball present with coordinates.
    ball = first["ball"]
    if isinstance(ball, dict):
        assert "x" in ball
        assert "y" in ball
    else:
        assert len(ball) >= 2


def test_e2e_coach_override_is_applied(tmp_path: Path, monkeypatch) -> None:
    """An override set via the real GameState steers the player and survives the run.

    Home player 9 sits at the origin; target (0.7, 0.0) must produce a RIGHT
    action. The override (duration 100) is still active after the 50-step match,
    proving the engine read it through ``get_active_overrides`` each tick.
    """
    env = _FakeEnv()
    _install_fake_gfootball(monkeypatch, env)

    gs = GameState()
    gs.set_override(
        "home",
        "9",
        {"target_position": [0.7, 0.0], "duration_ticks": 100, "reasoning": "test"},
    )

    replay = tmp_path / "replay.jsonl"
    engine = SoccerEngine(replay_path=str(replay), match_steps=50, game_state=gs)
    engine.run()

    # The engine applied the override: player 9's last action moved RIGHT toward target.
    assert env.last_actions is not None
    assert env.last_actions[9] == RIGHT

    # And the override (duration 100) is still active after 50 ticks.
    active = gs.get_active_overrides("home")
    assert "9" in active
    assert active["9"]["override"]["target_position"] == [0.7, 0.0]
