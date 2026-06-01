"""Replay logger: records match ticks to a JSONL file for later replay.

Each tick is one JSON line. Coach-cycle annotations are embedded back into the
most recent tick's line via an atomic rewrite (write `.tmp` then ``Path.replace``).
"""

from __future__ import annotations

import json
import os
import threading
from pathlib import Path
from typing import TYPE_CHECKING, Any

from simulator.roles import ROLES


if TYPE_CHECKING:
    from collections.abc import Sequence


__all__ = ["ROLES", "ReplayLogger"]


def _role(index: int) -> str:
    """Return the role label for a player index, defaulting to ``"SUB"``."""
    return ROLES[index] if 0 <= index < len(ROLES) else "SUB"


def _players(team: str, positions: Sequence[Sequence[float]]) -> list[dict[str, Any]]:
    """Build player records for one team from an array of (x, y) positions."""
    players: list[dict[str, Any]] = []
    for i, pos in enumerate(positions):
        players.append(
            {
                "id": i,
                "team": team,
                "role": _role(i),
                "x": float(pos[0]),
                "y": float(pos[1]),
            },
        )
    return players


class ReplayLogger:
    """Append-only JSONL replay logger with atomic coach-cycle annotation."""

    def __init__(self, path: str = "match/replay.jsonl") -> None:
        """Open ``path`` for appending, creating parent directories as needed."""
        self.path = Path(path)
        # Always ensure the parent dir exists before opening — the container's
        # working dir does not guarantee match/ exists (mkdir of "." is a no-op).
        self.path.parent.mkdir(parents=True, exist_ok=True)
        # Open in append mode so reopening an existing replay continues it.
        self._fh = self.path.open("a", encoding="utf-8")
        # Guards every write: in the single-process launch the simulator thread
        # appends ticks while the two coach threads annotate their tick line.
        self._lock = threading.Lock()

    def log_tick(self, tick: int, t: float, obs: dict[str, Any], score: list[int]) -> None:
        """Append one tick to the replay log.

        ``obs`` has ``left_team`` (home positions), ``right_team`` (away
        positions), and ``ball``. Writes a line of the form::

            {tick, t, players:[{id,team,role,x,y}], ball:{x,y}, score:[h,a]}
        """
        players = _players("home", obs["left_team"]) + _players("away", obs["right_team"])
        ball = obs["ball"]
        record = {
            "tick": tick,
            "t": t,
            "players": players,
            "ball": {"x": float(ball[0]), "y": float(ball[1])},
            "score": list(score),
        }
        with self._lock:
            self._fh.write(json.dumps(record) + "\n")
            self._fh.flush()

    def log_coach_cycle(self, tick: int, team: str, alerts: list[dict[str, Any]]) -> None:
        """Embed a ``coach_cycle`` block into the line for ``tick``.

        Targets the line whose ``tick`` matches the argument (the most recent
        such line if several share it), falling back to the last line if no
        match is found — this keeps the cycle attached to its own tick even when
        the simulator has written newer ticks while the coach was deciding. The
        file is rewritten atomically (``.tmp`` then ``Path.replace``). ``alerts``
        entries look like::

            {player_id, coach_reasoning, player_decision, override_written,
             target_position, duration_ticks, response_time_s}
        """
        with self._lock:
            # Ensure buffered writes are on disk before we read the file back.
            self._fh.flush()
            with self.path.open(encoding="utf-8") as fh:
                lines = fh.readlines()

            idx, record = self._locate_tick(lines, tick)
            if idx is None:
                return  # no tick lines written yet — nothing to annotate

            block = {"tick": tick, "team": team, "alerts": alerts}
            record.setdefault("coach_cycle", []).append(block)
            lines[idx] = json.dumps(record) + "\n"

            tmp = self.path.with_suffix(self.path.suffix + ".tmp")
            with tmp.open("w", encoding="utf-8") as fh:
                fh.writelines(lines)
                fh.flush()
                # fsync the temp file before the rename so a crash mid-write
                # cannot leave a truncated replay behind the atomic swap.
                os.fsync(fh.fileno())
            tmp.replace(self.path)

            # The appended fd now points at the replaced inode; reopen for appends.
            self._fh.close()
            self._fh = self.path.open("a", encoding="utf-8")

    @staticmethod
    def _locate_tick(lines: list[str], tick: int) -> tuple[int | None, dict[str, Any]]:
        """Find the line to annotate for ``tick``.

        Returns ``(index, parsed_record)`` for the most recent line whose
        ``tick`` matches, else the last valid line, else ``(None, {})``.
        """
        last_idx: int | None = None
        last_record: dict[str, Any] = {}
        for i in range(len(lines) - 1, -1, -1):
            try:
                record = json.loads(lines[i])
            except json.JSONDecodeError:
                continue
            if last_idx is None:
                last_idx, last_record = i, record  # newest valid line (fallback)
            if record.get("tick") == tick:
                return i, record
        return last_idx, last_record

    def close(self) -> None:
        """Flush and close the underlying file handle."""
        with self._lock:
            if not self._fh.closed:
                self._fh.flush()
                self._fh.close()
