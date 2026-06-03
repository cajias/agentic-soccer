"""Soccer match engine wrapping Google Research Football (gfootball).

Runs an 11v11 match with all 22 players under agent control. The engine runs
inside the gfootball Docker image (Linux, where gfootball builds cleanly) and
shares an in-process :class:`mcp_server.game_state.GameState` singleton with the
MCP server: the engine advances the tick, publishes narrator text, and reads the
per-player behaviour overrides that coach/player agents post through the MCP
tools.

Each tick:

1. ``GAME_STATE.set_tick`` advances the shared tick (drives override expiry).
2. ``GAME_STATE.get_active_overrides(team)`` returns the live overrides; a player
   with an active ``target_position`` is steered toward it, otherwise it falls
   back to a role-based heuristic (:func:`default_action`).
3. The 22 actions are applied via ``env.step``.
4. The score is read from the raw observation and ``narrate`` output is published
   to ``GAME_STATE`` for each team.
5. The tick is appended to a JSONL replay (``match/replay.jsonl``; a mounted
   volume in the container).

Coordinate frames
-----------------
gfootball presents every agent-controlled player its *own* observation in a
left-to-right attacking frame (own goal at ``x = -1``) and flips right-team
actions back to absolute internally. So **actions** are computed from each
player's own observation (``obs_list[m]``) with no manual mirroring, while
**logging** reads absolute positions from the left-team frame (``obs_list[0]``).
Override ``target_position`` values are interpreted in the target player's own
attacking frame (the same frame the narrator presents to that team's agents).
"""

from __future__ import annotations

import os
from typing import TYPE_CHECKING, Any, Protocol

import numpy as np

from replay.logger import ReplayLogger
from simulator.narrator import narrate
from simulator.roles import ROLES


if TYPE_CHECKING:
    from collections.abc import Mapping


class _GameStateLike(Protocol):
    """The slice of ``mcp_server.game_state.GameState`` the engine depends on."""

    def set_tick(self, tick: int) -> None: ...

    def set_narrator(self, team: str, text: str) -> None: ...

    def get_active_overrides(self, team: str) -> Mapping[str, Any]: ...

# gfootball renders via SDL; default to the headless dummy driver but honour an
# existing value (the Docker image sets SDL_VIDEODRIVER=dummy itself). Set before
# the gfootball env is imported lazily in ``SoccerEngine.__init__``.
os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
os.environ.setdefault("SDL_AUDIODRIVER", "dummy")


# ---------------------------------------------------------------------------
# gfootball "default" action set: 19 discrete actions addressed by int index.
# Only the movement subset is used by the heuristics; the rest (pass/shot/...)
# are listed for reference.
# ---------------------------------------------------------------------------
IDLE = 0
LEFT = 1
TOP_LEFT = 2
TOP = 3
TOP_RIGHT = 4
RIGHT = 5
BOTTOM_RIGHT = 6
BOTTOM = 7
BOTTOM_LEFT = 8
# 9 long_pass, 10 high_pass, 11 short_pass, 12 shot, 13 sprint,
# 14 release_direction, 15 release_sprint, 16 sliding, 17 dribble,
# 18 release_dribble.

# 8 directional actions ordered clockwise from "right" to match arctan2(dy, dx)
# octants. gfootball's y axis increases *downward*, so positive dy => "bottom".
_DIRECTIONS = [RIGHT, BOTTOM_RIGHT, BOTTOM, BOTTOM_LEFT, LEFT, TOP_LEFT, TOP, TOP_RIGHT]

# Match timing.
MATCH_DURATION_MIN = 90.0
FULL_MATCH_STEPS = 3000

# Squad sizes: 11 players per team, all 22 agent-controlled.
TEAM_SIZE = 11
NUM_CONTROLLED = 22

# Heuristic thresholds (own attacking frame, ball x in [-1, 1]).
_DEFENDER_CHASE_X = -0.3  # defenders chase only when the ball is this deep
_ATTACKER_PUSH_X = -0.2  # attackers push forward once the ball clears this

# simple115v2 observation slice offsets (115-float vector, absolute frame for
# a left-team player): 22 left pos, 22 left dir, 22 right pos, 22 right dir,
# 3 ball xyz, 3 ball dir, 3 ball-owned one-hot, 11 active one-hot, 7 game mode.
_LEFT_POS = slice(0, 22)
_RIGHT_POS = slice(44, 66)
_BALL_XY = slice(88, 90)

_DEADZONE = 1e-3
_POSITION_DIMS = 2  # an (x, y) target position


def _direction_action(dx: float, dy: float) -> int:
    """Map a movement vector to one of the 8 gfootball directional actions.

    Returns :data:`IDLE` when the vector is within the dead zone.
    """
    if abs(dx) < _DEADZONE and abs(dy) < _DEADZONE:
        return IDLE
    sector = round(float(np.arctan2(dy, dx)) / (np.pi / 4)) % 8
    return _DIRECTIONS[sector]


def default_action(team_index: int, pos: np.ndarray, ball: np.ndarray) -> int:
    """Heuristic action for a player without an active override.

    All inputs are in the player's *own* attacking frame (own goal at
    ``x = -1``). Behaviour by role keeps movement coherent without modelling a
    real AI:

    * ``GK``  hugs its own goal line, tracking the ball laterally.
    * ``CB``/``LB``/``RB`` hold a defensive line, chasing only when the ball is
      deep in their half.
    * ``CM`` follow the ball.
    * ``LW``/``RW``/``ST`` push into the attacking half, else support midfield.
    """
    role = ROLES[team_index] if team_index < len(ROLES) else "SUB"
    ball_x, ball_y = float(ball[0]), float(ball[1])

    if role == "GK":
        target = (-0.95, max(-0.1, min(0.1, ball_y)))
    elif role in ("CB", "LB", "RB"):
        target = (ball_x, ball_y) if ball_x < _DEFENDER_CHASE_X else (-0.4, ball_y * 0.7)
    elif role == "CM":
        target = (ball_x, ball_y)
    else:  # LW, RW, ST and any sub
        target = (0.6, ball_y) if ball_x > _ATTACKER_PUSH_X else (0.0, ball_y)

    return _direction_action(target[0] - float(pos[0]), target[1] - float(pos[1]))


class SoccerEngine:
    """Runs an 11v11 gfootball match driven by the shared MCP game state."""

    def __init__(
        self,
        replay_path: str = "match/replay.jsonl",
        match_steps: int = FULL_MATCH_STEPS,
        game_state: _GameStateLike | None = None,
    ) -> None:
        """Create the gfootball env, replay logger, and bind the shared state.

        ``game_state`` defaults to the in-process ``mcp_server.server.GAME_STATE``
        singleton (imported lazily so tests can inject a fake without importing
        the MCP server).
        """
        # Imported lazily so this module loads without a compiled gfootball.
        import gfootball.env as football_env  # noqa: PLC0415

        self.env = football_env.create_environment(
            env_name="11_vs_11_stochastic",
            representation="simple115v2",
            render=False,
            number_of_left_players_agent_controls=11,
            number_of_right_players_agent_controls=11,
            write_goal_dumps=False,
            write_full_episode_dumps=False,
            logdir="",
        )
        self.logger = ReplayLogger(replay_path)
        self.match_steps = match_steps
        self.tick = 0

        if game_state is None:
            from mcp_server.server import GAME_STATE  # noqa: PLC0415

            game_state = GAME_STATE
        self.game_state = game_state

    @staticmethod
    def _override_target(overrides: Mapping[str, Any], team_index: int) -> list[float] | None:
        """Extract a player's ``target_position`` from active overrides, if any.

        ``overrides`` is ``GAME_STATE.get_active_overrides(team)`` output: it maps
        ``str(squad_index)`` to a record whose ``"override"`` payload holds the
        agent-written ``target_position``.
        """
        record = overrides.get(str(team_index))
        if not isinstance(record, dict):
            return None
        payload = record.get("override")
        if not isinstance(payload, dict):
            return None
        target = payload.get("target_position")
        if isinstance(target, (list, tuple)) and len(target) >= _POSITION_DIMS:
            return [float(target[0]), float(target[1])]
        return None

    def _build_actions(
        self,
        obs_list: np.ndarray,
        home_overrides: Mapping[str, Any],
        away_overrides: Mapping[str, Any],
    ) -> list[int]:
        """Compute the 22 actions for one tick from per-player observations."""
        actions: list[int] = []
        for m in range(NUM_CONTROLLED):
            team_index = m % TEAM_SIZE
            own_obs = np.asarray(obs_list[m])
            pos = own_obs[2 * team_index : 2 * team_index + 2]
            ball = own_obs[_BALL_XY]
            overrides = home_overrides if m < TEAM_SIZE else away_overrides
            target = self._override_target(overrides, team_index)

            if target is not None:
                actions.append(_direction_action(target[0] - float(pos[0]), target[1] - float(pos[1])))
            else:
                actions.append(default_action(team_index, pos, ball))
        return actions

    def _raw_observation(self) -> list[dict[str, Any]] | None:
        """Return gfootball's raw per-agent observation list, or None on error.

        ``raw[0]`` is the absolute (home/left) frame dict carrying ``score`` and
        everything the narrator needs.
        """
        try:
            obs: list[dict[str, Any]] = self.env.unwrapped.observation()
        except (AttributeError, TypeError):
            return None
        return obs

    @staticmethod
    def _score_from_raw(raw: list[dict[str, Any]] | None) -> list[int]:
        """Read the ``[home, away]`` goal tally from a raw observation list.

        ``info`` carries only ``score_reward`` (a scalar delta), so the actual
        per-team score lives on the raw observation dict.
        """
        if not raw:
            return [0, 0]
        try:
            return [int(raw[0]["score"][0]), int(raw[0]["score"][1])]
        except (KeyError, IndexError, TypeError):
            return [0, 0]

    def _publish_narration(
        self, raw: list[dict[str, Any]] | None, score: list[int],
    ) -> None:
        """Publish per-team narrator text to the shared game state.

        Best-effort: narration must never interrupt the match, so any failure is
        swallowed. The narrator mirrors the absolute (home) observation itself for
        the away perspective, so the same dict drives both teams.
        """
        if not raw:
            return
        try:
            obs = dict(raw[0])
            obs["score"] = score
            obs["steps_remaining"] = max(0, self.match_steps - self.tick)
            self.game_state.set_narrator("home", narrate(obs, "home"))
            self.game_state.set_narrator("away", narrate(obs, "away"))
        except Exception:  # noqa: BLE001 - narration is best-effort, never fatal
            return

    def _log_tick(self, obs_list: np.ndarray, score: list[int]) -> None:
        """Append the current tick to the replay using absolute positions."""
        absolute = np.asarray(obs_list[0])
        left = absolute[_LEFT_POS].reshape(11, 2)
        right = absolute[_RIGHT_POS].reshape(11, 2)
        ball = absolute[_BALL_XY]
        minutes = self.tick / FULL_MATCH_STEPS * MATCH_DURATION_MIN
        obs = {"left_team": left.tolist(), "right_team": right.tolist(), "ball": ball.tolist()}
        self.logger.log_tick(self.tick, minutes, obs, score)

    @staticmethod
    def _is_done(done: bool | np.ndarray) -> bool:
        """Normalise gfootball's ``done`` (bool or per-agent array) to a bool."""
        if isinstance(done, bool):
            return done
        return bool(np.any(done))

    def run(self) -> dict[str, Any]:
        """Run a full match, logging every tick. Returns final stats."""
        gs = self.game_state
        try:
            obs_list = self.env.reset()
            done = False

            while not done and self.tick < self.match_steps:
                gs.set_tick(self.tick)
                home_overrides = gs.get_active_overrides("home")
                away_overrides = gs.get_active_overrides("away")

                actions = self._build_actions(obs_list, home_overrides, away_overrides)
                obs_list, _reward, done_flag, _info = self.env.step(actions)
                done = self._is_done(done_flag)

                raw = self._raw_observation()
                score = self._score_from_raw(raw)
                self._publish_narration(raw, score)
                self._log_tick(obs_list, score)
                self.tick += 1

            gs.set_tick(self.tick)
            final_score = self._score_from_raw(self._raw_observation())
        finally:
            # Always release the gfootball env and flush/close the replay file,
            # even if the match loop raises mid-game.
            self.logger.close()
            self.env.close()

        print(f"Match complete: Home {final_score[0]} - {final_score[1]} Away ({self.tick} ticks)")
        return {
            "score": final_score,
            "ticks": self.tick,
            "home_goals": final_score[0],
            "away_goals": final_score[1],
        }
