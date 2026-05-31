"""Thread-safe shared state between the simulator and the MCP server.

The simulator writes narrator text and advances ``tick``; agent clients read
narrator text and post per-player behavior overrides through the MCP tools in
:mod:`mcp_server.server`. All access is guarded by a single lock so the
simulator thread and the MCP request handlers never race.
"""

from __future__ import annotations

import threading
from typing import Any


# Recognized teams. Used to validate inputs everywhere state is keyed by team.
TEAMS: tuple[str, ...] = ("home", "away")


class GameState:
    """Mutable match state shared across threads.

    Attributes:
        narrator_text: Per-team narrator text, set by the simulator.
        player_overrides: ``team -> player_id -> override`` mapping. Each stored
            override carries the ``override`` payload plus a ``created_tick``
            stamp used to compute expiry.
        tick: Current simulation tick, advanced by the simulator.
    """

    def __init__(self) -> None:
        """Initialize empty per-team narrator text and overrides at tick 0."""
        self._lock = threading.Lock()
        self.narrator_text: dict[str, str] = dict.fromkeys(TEAMS, "")
        self.player_overrides: dict[str, dict[str, dict]] = {team: {} for team in TEAMS}
        self.tick: int = 0
        # [home_goals, away_goals], set by the simulator; exposed by the REST
        # /get_match_status route. No reader depends on it yet.
        self.score: list[int] = [0, 0]

    # -- tick ---------------------------------------------------------------

    def set_tick(self, tick: int) -> None:
        """Set the absolute simulation tick (called by the simulator)."""
        with self._lock:
            self.tick = tick

    # -- score --------------------------------------------------------------

    def set_score(self, score: list[int]) -> None:
        """Set the [home, away] score (called by the simulator)."""
        with self._lock:
            self.score = [int(score[0]), int(score[1])]

    def get_score(self) -> list[int]:
        """Return a copy of the current [home, away] score."""
        with self._lock:
            return list(self.score)

    def advance_tick(self, by: int = 1) -> int:
        """Advance the tick and return the new value."""
        with self._lock:
            self.tick += by
            return self.tick

    # -- narrator -----------------------------------------------------------

    def set_narrator(self, team: str, text: str) -> None:
        """Set narrator text for ``team`` (called by the simulator)."""
        _check_team(team)
        with self._lock:
            self.narrator_text[team] = text

    def get_narrator(self, team: str) -> str:
        """Return the current narrator text for ``team`` (empty if unset)."""
        _check_team(team)
        with self._lock:
            return self.narrator_text.get(team, "")

    # -- overrides ----------------------------------------------------------

    def set_override(self, team: str, player_id: str, override: dict) -> dict:
        """Store a behavior override for a player, stamped with the current tick.

        Returns the stored record (payload + ``created_tick`` + ``expires_at_tick``).
        """
        _check_team(team)
        record = {
            "player_id": player_id,
            "override": dict(override),
            "created_tick": self.tick,
            "expires_at_tick": self.tick + int(override.get("duration_ticks", 0)),
        }
        with self._lock:
            self.player_overrides[team][player_id] = record
            return dict(record)

    def get_active_overrides(self, team: str) -> dict[str, dict]:
        """Return active overrides for ``team``, deleting any that have expired.

        An override is expired once ``tick - created_tick >= duration_ticks``.
        Expired entries are removed from state (auto-removal), not just filtered
        from the result.
        """
        _check_team(team)
        with self._lock:
            overrides = self.player_overrides[team]
            expired = [
                player_id
                for player_id, record in overrides.items()
                if self._is_expired(record)
            ]
            for player_id in expired:
                del overrides[player_id]
            # Return deep-ish copies so callers can't mutate stored state.
            return {pid: dict(record) for pid, record in overrides.items()}

    def clear_override(self, team: str, player_id: str) -> bool:
        """Remove an override for a player. Returns True if one was present."""
        _check_team(team)
        with self._lock:
            return self.player_overrides[team].pop(player_id, None) is not None

    def reset(self) -> None:
        """Reset all state to initial values (primarily for tests)."""
        with self._lock:
            self.narrator_text = dict.fromkeys(TEAMS, "")
            self.player_overrides = {team: {} for team in TEAMS}
            self.tick = 0
            self.score = [0, 0]

    # -- internals ----------------------------------------------------------

    def _is_expired(self, record: dict[str, Any]) -> bool:
        duration = int(record["override"].get("duration_ticks", 0))
        # A non-positive duration never expires on its own; the simulator must
        # clear it explicitly. Positive durations expire once elapsed.
        if duration <= 0:
            return False
        return (self.tick - record["created_tick"]) >= duration


def _check_team(team: str) -> None:
    if team not in TEAMS:
        msg = f"unknown team {team!r}; expected one of {TEAMS}"
        raise ValueError(msg)
