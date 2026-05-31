"""Coach agent loop: periodically inspects the match and alerts out-of-position players.

Every cycle (driven externally at ~15s intervals) the coach:

1. reads the current match status (narrator text) for its team via the MCP client;
2. parses each player's reported behavior and compares the pitch zone they are in
   against the zone their personality says they should occupy (a pure heuristic —
   no LLM at the coach level);
3. spawns a :class:`~agents.player_agent.PlayerAgent` for up to three of the most
   out-of-position players, which each decide hold-vs-override via an LLM; and
4. logs the whole cycle to the replay file as a ``coach_cycle`` block.

``run_cycle(tick)`` performs exactly one pass and is pure with respect to time —
the caller supplies the current sim tick (the simulator owns the replay file and
knows the tick). :meth:`CoachLoop.drive` is the thin loop that sleeps between
cycles; tests call ``run_cycle`` directly and never sleep.
"""

from __future__ import annotations

import time
from typing import TYPE_CHECKING, Any

from agents.personalities import (
    ZONE_ATTACKING,
    ZONE_DEFENSIVE,
    ZONE_MIDFIELD,
    squad_for,
)
from agents.player_agent import DEFAULT_MODEL, LLMClient, MatchClient, PlayerAgent


if TYPE_CHECKING:
    from collections.abc import Callable, Mapping, Sequence

    from replay.logger import ReplayLogger


# How often the coach acts, in wall-clock seconds.
COACH_INTERVAL_S = 15.0

# Never alert more than this many players in one cycle.
MAX_ALERTS_PER_CYCLE = 3

# Minimum zone-band distance (0=defensive, 1=midfield, 2=attacking) between where
# a player is and where they should be before the coach intervenes. A distance of
# 2 means clearly out of position (e.g. a forward in the defensive third).
OUT_OF_POSITION_THRESHOLD = 2

# Behavior-text zone phrases -> band. Order matters: check the most specific
# phrases first so "opposition penalty area" wins over a bare "third".
_ZONE_BANDS: tuple[tuple[str, int], ...] = (
    ("opposition penalty area", 2),
    ("own penalty area", 0),
    ("attacking third", 2),
    ("defensive third", 0),
    ("midfield", 1),
)

_EXPECTED_BANDS: dict[str, int] = {
    ZONE_DEFENSIVE: 0,
    ZONE_MIDFIELD: 1,
    ZONE_ATTACKING: 2,
}

_BAND_DESCRIPTIONS: dict[int, str] = {
    0: "deep in your defensive third",
    1: "in midfield",
    2: "high in the attacking third",
}

_PLAYERS_HEADER = "WHAT YOUR PLAYERS ARE DOING:"


def _behavior_band(behavior: str) -> int | None:
    """Return the pitch-zone band implied by a narrator behavior line, or None."""
    lowered = behavior.lower()
    for phrase, band in _ZONE_BANDS:
        if phrase in lowered:
            return band
    return None


def parse_player_behaviors(status: str) -> list[str]:
    """Extract the per-player behavior strings from narrator ``status`` text.

    Returns the behavior text for each ``- ROLE: behavior`` line under the
    "WHAT YOUR PLAYERS ARE DOING" header, in squad-index order (0-10). Returns
    an empty list if the section is absent.
    """
    lines = status.splitlines()
    try:
        start = next(i for i, line in enumerate(lines) if line.strip() == _PLAYERS_HEADER)
    except StopIteration:
        return []

    behaviors: list[str] = []
    for line in lines[start + 1 :]:
        stripped = line.strip()
        if not stripped.startswith("- "):
            break  # the player block ends at the first non-bullet line
        _, _, behavior = stripped[2:].partition(": ")
        behaviors.append(behavior)
    return behaviors


def _alert_text(personality: Mapping[str, Any], current_band: int, expected_band: int) -> str:
    """Compose the coach's reasoning for flagging a player."""
    name = personality.get("name", "Player")
    role = personality.get("role", "?")
    where = _BAND_DESCRIPTIONS[current_band]
    should = _BAND_DESCRIPTIONS[expected_band]
    fix = "push higher up the pitch" if current_band < expected_band else "track back into position"
    return f"{name} ({role}) is {where} but should be {should} — {fix}."


def select_alerts(
    squad: Mapping[int, Mapping[str, Any]],
    behaviors: Sequence[str],
) -> list[tuple[int, str]]:
    """Pick the most out-of-position players to alert (pure heuristic).

    Compares each player's current zone band against their personality's
    expected band; players whose distance meets :data:`OUT_OF_POSITION_THRESHOLD`
    are candidates, ranked by distance (then by squad index) and capped at
    :data:`MAX_ALERTS_PER_CYCLE`.
    """
    scored: list[tuple[int, int, str]] = []  # (distance, player_id, reason)
    for pid, behavior in enumerate(behaviors):
        personality = squad.get(pid)
        if personality is None:
            continue
        current = _behavior_band(behavior)
        if current is None:
            continue
        expected = _EXPECTED_BANDS[personality["expected_zone"]]
        distance = abs(current - expected)
        if distance >= OUT_OF_POSITION_THRESHOLD:
            scored.append((distance, pid, _alert_text(personality, current, expected)))

    # Most out-of-position first; ties broken by squad index for determinism.
    scored.sort(key=lambda item: (-item[0], item[1]))
    return [(pid, reason) for _, pid, reason in scored[:MAX_ALERTS_PER_CYCLE]]


class CoachLoop:
    """Periodic, heuristic coach that delegates per-player decisions to sub-agents."""

    def __init__(
        self,
        team: str,
        client: MatchClient,
        replay_logger: ReplayLogger,
        *,
        anthropic_client: LLMClient | None = None,
        model: str = DEFAULT_MODEL,
    ) -> None:
        """Bind the coach to a ``team``, an MCP ``client`` and a ``replay_logger``."""
        self.team = team
        self.client = client
        self.replay_logger = replay_logger
        self.squad = squad_for(team)
        self._anthropic = anthropic_client
        self._model = model

    def run_cycle(self, tick: int) -> list[dict]:
        """Run one coach pass at ``tick``; returns the per-player alert records.

        Up to :data:`MAX_ALERTS_PER_CYCLE` records are returned and, when any
        player was alerted, logged to the replay file as a ``coach_cycle`` block
        embedded in ``tick``'s line. Cycles that produce no alerts are not logged
        (they carry no decisions and would just accumulate empty blocks).
        """
        status = self.client.get_match_status(self.team)
        candidates = select_alerts(self.squad, parse_player_behaviors(status))

        alerts: list[dict] = []
        for player_id, reason in candidates:
            agent = PlayerAgent(
                self.team,
                self.client,
                anthropic_client=self._anthropic,
                model=self._model,
            )
            alerts.append(agent.decide(player_id, self.squad[player_id], reason, status))

        if alerts:
            self.replay_logger.log_coach_cycle(tick, self.team, alerts)
        return alerts

    def drive(
        self,
        tick_source: Callable[[], int],
        *,
        interval_s: float = COACH_INTERVAL_S,
        max_cycles: int | None = None,
        sleep: Callable[[float], None] = time.sleep,
    ) -> None:
        """Run cycles forever (or ``max_cycles`` times), sleeping ``interval_s`` between.

        ``tick_source`` returns the current sim tick at cycle time. ``sleep`` is
        injectable so tests can drive the loop without real delays.
        """
        cycles = 0
        while max_cycles is None or cycles < max_cycles:
            self.run_cycle(tick_source())
            cycles += 1
            if max_cycles is None or cycles < max_cycles:
                sleep(interval_s)


class ServerMatchClient:
    """In-process :class:`MatchClient` that calls the MCP server tool functions.

    Useful for a single-process launch; the two-session design instead points
    each coach at the MCP server over HTTP with its own team token.
    """

    def __init__(self, token: str) -> None:
        """Store the team token used to authorize every tool call."""
        self._token = token

    def get_match_status(self, team: str) -> str:
        """Return narrator text for ``team`` via the server tool."""
        # Lazy import: avoid pulling in FastMCP unless this adapter is used.
        from mcp_server.server import get_match_status  # noqa: PLC0415

        return get_match_status(team, token=self._token)

    def update_player_override(self, team: str, player_id: str, override: dict) -> dict:
        """Store an override for ``player_id`` on ``team`` via the server tool."""
        from mcp_server.server import update_player_override  # noqa: PLC0415

        return update_player_override(team, player_id, override, token=self._token)
