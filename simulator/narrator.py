"""State narrator: turns a gfootball observation into a coach-readable report.

Template-based natural language generation. No LLM calls. Given a gfootball
(or compatible) observation dict and a team perspective ("home" or "away"),
:func:`narrate` returns a multi-line string a human coach can read at a glance.

Coordinate conventions (gfootball):
    x in [-1, 1]  -- the home team attacks toward +x, the away team toward -x.
    y in [-0.42, 0.42] -- negative is one touchline, positive the other.
    ball z is height; ignored for zone reasoning.

All zone reasoning is written from an "attacking toward +x" viewpoint. For the
away perspective we mirror every coordinate (x, y) -> (-x, -y) up front, so the
same logic describes "your" half, channels, and the opposition box correctly.
"""

from __future__ import annotations

import math
from typing import Any


# Human-readable role labels by squad index, for narration only. Intentionally
# distinct from :data:`simulator.roles.ROLES` (which carries terse role codes);
# do not merge the two.
ROLE_LABELS: tuple[str, ...] = (
    "GK",
    "CB (left)",
    "CB (right)",
    "LB",
    "RB",
    "CM (left)",
    "CM (center)",
    "CM (right)",
    "LW",
    "RW",
    "ST",
)

GAME_MODES: dict[int, str] = {
    0: "Normal",
    1: "KickOff",
    2: "GoalKick",
    3: "FreeKick",
    4: "Corner",
    5: "ThrowIn",
    6: "Penalty",
}

# Full match in gfootball is 3000 steps == 90 minutes -> 1.8 s per step.
_STEPS_PER_MATCH = 3000
_SECONDS_PER_MATCH = 90 * 60

# Field geometry thresholds (attacking-toward-+x frame).
_THIRD = 1.0 / 3.0
_PENALTY_X = 0.7  # |x| beyond this and inside the box width is the penalty area
_PENALTY_Y = 0.27
_CHANNEL_Y = 0.14  # |y| beyond this leaves the central corridor
_MOVING_SPEED = 0.005  # per-step velocity magnitude above which a player is "moving"
_NEAR_BALL = 0.06  # distance at which a player is engaging the ball
_CHASE_RANGE = 0.3  # distance within which moving toward the ball reads as "closing"

# Opponent counts inside our defensive third that define pressure bands.
_HEAVY_PRESSURE = 4
_MODERATE_PRESSURE = 2

_VALID_TEAMS = ("home", "away")


def _mirror(point: list[float] | tuple[float, ...]) -> tuple[float, float]:
    """Mirror an (x, y[, z]) point to the attacking-toward-+x frame for away."""
    return (-point[0], -point[1])


def _clock(steps_remaining: int) -> str:
    """Render elapsed match time as MM:SS from steps remaining."""
    steps_remaining = max(0, min(_STEPS_PER_MATCH, int(steps_remaining)))
    elapsed_steps = _STEPS_PER_MATCH - steps_remaining
    elapsed_seconds = elapsed_steps * (_SECONDS_PER_MATCH / _STEPS_PER_MATCH)
    minutes = int(elapsed_seconds // 60)
    seconds = int(elapsed_seconds % 60)
    return f"{minutes}:{seconds:02d}"


def _zone(x: float, y: float) -> str:
    """Describe the pitch zone of a point in the attacking-toward-+x frame."""
    # Penalty areas first -- most specific.
    if x <= -_PENALTY_X and abs(y) <= _PENALTY_Y:
        return "own penalty area"
    if x >= _PENALTY_X and abs(y) <= _PENALTY_Y:
        return "opposition penalty area"

    # Longitudinal third.
    if x < -_THIRD:
        third = "defensive third"
    elif x > _THIRD:
        third = "attacking third"
    else:
        third = "midfield"

    # Lateral channel.
    if y < -_CHANNEL_Y:
        channel = "left channel"
    elif y > _CHANNEL_Y:
        channel = "right channel"
    else:
        channel = "center"

    return f"{third}, {channel}"


def _player_action(
    pos: tuple[float, float],
    vel: tuple[float, float],
    ball: tuple[float, float],
    has_ball: bool,
) -> str:
    """Describe what a single player is doing from position, velocity and ball."""
    zone = _zone(*pos)
    if has_ball:
        return f"on the ball in {zone}"

    dist_to_ball = math.hypot(ball[0] - pos[0], ball[1] - pos[1])
    speed = math.hypot(*vel)

    if dist_to_ball <= _NEAR_BALL:
        return f"challenging for the ball in {zone}"

    if speed < _MOVING_SPEED:
        return f"holding position in {zone}"

    # Moving: pick the dominant axis for a readable direction.
    if abs(vel[0]) >= abs(vel[1]):
        heading = "pushing forward" if vel[0] > 0 else "tracking back"
    else:
        heading = "shifting right" if vel[1] > 0 else "shifting left"

    # If they are moving generally toward the ball, say so instead.
    toward_ball = (ball[0] - pos[0]) * vel[0] + (ball[1] - pos[1]) * vel[1]
    if toward_ball > 0 and dist_to_ball < _CHASE_RANGE:
        return f"closing on the ball in {zone}"

    return f"{heading} in {zone}"


def _possession_line(
    obs: dict[str, Any],
    my_team: int,
    owner_positions: list[tuple[float, float]],
) -> str:
    """Build the POSSESSION line (absolute Home/Away/Contested + ball-carrier)."""
    owned_team = obs.get("ball_owned_team", -1)
    ball = _ball_xy(obs, my_team)

    if owned_team not in (0, 1):
        return f"POSSESSION: Contested — loose ball in {_zone(*ball)}"

    side = "Home" if owned_team == 0 else "Away"
    player_idx = obs.get("ball_owned_player", -1)
    role = ROLE_LABELS[player_idx] if 0 <= player_idx < len(ROLE_LABELS) else "a player"

    # Locate the carrier in our perspective frame for the zone phrase.
    carrier_in_frame = 0 <= player_idx < len(owner_positions)
    carrier_zone = _zone(*owner_positions[player_idx]) if carrier_in_frame else _zone(*ball)

    return f"POSSESSION: {side} — {role} has the ball in {carrier_zone}"


def _ball_xy(obs: dict[str, Any], my_team: int) -> tuple[float, float]:
    """Return the ball (x, y) in our attacking-toward-+x frame."""
    ball = obs["ball"]
    xy = (ball[0], ball[1])
    return _mirror(xy) if my_team == 1 else xy


def _team_points(
    obs: dict[str, Any],
    key: str,
    my_team: int,
) -> list[tuple[float, float]]:
    """Return a team's points (x, y) in our attacking-toward-+x frame."""
    raw = obs[key]
    if my_team == 1:
        return [_mirror(p) for p in raw]
    return [(p[0], p[1]) for p in raw]


def _game_situation(
    obs: dict[str, Any],
    my_team: int,
    my_positions: list[tuple[float, float]],
    opp_positions: list[tuple[float, float]],
) -> str:
    """Compose a 2-3 sentence tactical read of the current state."""
    ball = _ball_xy(obs, my_team)
    owned_team = obs.get("ball_owned_team", -1)
    sentences: list[str] = []

    # Set-piece / game mode note.
    mode = GAME_MODES.get(int(obs.get("game_mode", 0)), "Normal")
    if mode != "Normal":
        sentences.append(f"Set piece in play: {mode}.")

    # Pressure: opponents inside our defensive third.
    opp_in_def_third = sum(1 for x, _ in opp_positions if x < -_THIRD)
    if opp_in_def_third >= _HEAVY_PRESSURE:
        sentences.append(
            f"Heavy pressure — {opp_in_def_third} opponents are camped in your "
            "defensive third.",
        )
    elif opp_in_def_third >= _MODERATE_PRESSURE:
        sentences.append(
            f"Moderate pressure with {opp_in_def_third} opponents pressing into "
            "your defensive third.",
        )
    else:
        sentences.append("Little defensive pressure right now; the back line is calm.")

    # Threat level / where the ball is.
    if ball[0] >= _PENALTY_X and abs(ball[1]) <= _PENALTY_Y:
        sentences.append("The ball is in the opposition box — a real scoring threat.")
    elif ball[0] <= -_PENALTY_X and abs(ball[1]) <= _PENALTY_Y:
        sentences.append("Danger: the ball is sitting in your own penalty area.")
    elif ball[0] > _THIRD:
        sentences.append("Play is in the attacking third; look to commit runners.")
    elif ball[0] < -_THIRD:
        sentences.append("Play is pinned in your defensive third; stay compact.")
    else:
        sentences.append("The ball is in midfield, where the game is being contested.")

    # Open zone hint based on possession.
    if owned_team == my_team:
        attackers_high = sum(1 for x, _ in my_positions if x > _THIRD)
        sentences.append(
            f"You have {attackers_high} players advanced into the attacking third.",
        )

    return " ".join(sentences)


def narrate(obs: dict[str, Any], team: str = "home") -> str:
    """Return a coach-readable report for an observation from a team's view.

    Args:
        obs: A gfootball-style observation dict (see module docstring).
        team: Whose perspective to narrate for, "home" or "away".

    Returns:
        A multi-line report string.
    """
    team = team.lower()
    if team not in _VALID_TEAMS:
        msg = f"team must be one of {_VALID_TEAMS}, got {team!r}"
        raise ValueError(msg)
    my_team = 0 if team == "home" else 1

    score = obs.get("score", [0, 0])
    clock = _clock(obs.get("steps_remaining", _STEPS_PER_MATCH))

    # All positions resolved into our attacking-toward-+x frame.
    my_key = "left_team" if my_team == 0 else "right_team"
    opp_key = "right_team" if my_team == 0 else "left_team"
    my_dir_key = "left_team_direction" if my_team == 0 else "right_team_direction"

    my_positions = _team_points(obs, my_key, my_team)
    opp_positions = _team_points(obs, opp_key, my_team)
    my_velocities = _team_points(obs, my_dir_key, my_team)
    ball = _ball_xy(obs, my_team)

    owned_team = obs.get("ball_owned_team", -1)
    owned_player = obs.get("ball_owned_player", -1)

    # Owner positions for the possession line are always in our frame; the
    # carrier may be on either team, so pick the right source.
    owner_positions = my_positions if owned_team == my_team else opp_positions

    lines: list[str] = []
    lines.append(
        f"MATCH STATE ({clock}) | Score: Home {score[0]} – Away {score[1]}",  # noqa: RUF001 - en dash is intentional scoreline styling
    )
    lines.append("")
    lines.append(_possession_line(obs, my_team, owner_positions))
    lines.append("")
    lines.append("WHAT YOUR PLAYERS ARE DOING:")

    for idx, role in enumerate(ROLE_LABELS):
        pos = my_positions[idx]
        vel = my_velocities[idx] if idx < len(my_velocities) else (0.0, 0.0)
        has_ball = owned_team == my_team and owned_player == idx
        lines.append(f"- {role}: {_player_action(pos, vel, ball, has_ball)}")

    lines.append("")
    lines.append("GAME SITUATION:")
    lines.append(_game_situation(obs, my_team, my_positions, opp_positions))
    lines.append("")
    lines.append(f"YOUR TEAM: {team}")

    return "\n".join(lines)


class Narrator:
    """Convenience wrapper bound to a single team perspective."""

    def __init__(self, team: str = "home") -> None:
        """Bind the narrator to a team perspective ("home" or "away")."""
        self.team = team

    def narrate(self, obs: dict[str, Any]) -> str:
        """Return a coach-readable report for the bound team."""
        return narrate(obs, self.team)
