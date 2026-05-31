"""Replay logger: records match ticks to a JSONL file for later replay.

Each tick is one JSON line. Coach-cycle annotations are embedded back into the
most recent tick's line via an atomic rewrite (write `.tmp` then ``os.replace``).
"""

from __future__ import annotations

import json
import os


# Player roles by index within a team (gfootball 11v11 layout).
ROLES = ["GK", "CB", "CB", "LB", "RB", "CM", "CM", "CM", "LW", "RW", "ST"]


def _role(index: int) -> str:
    """Return the role label for a player index, defaulting to ``"SUB"``."""
    return ROLES[index] if 0 <= index < len(ROLES) else "SUB"


def _players(team: str, positions) -> list[dict]:
    """Build player records for one team from an array of (x, y) positions."""
    players = []
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
        self.path = path
        parent = os.path.dirname(path)
        if parent:
            os.makedirs(parent, exist_ok=True)
        # Open in append mode so reopening an existing replay continues it.
        self._fh = open(path, "a", encoding="utf-8")

    def log_tick(self, tick: int, t: float, obs: dict, score: list[int]) -> None:
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
        self._fh.write(json.dumps(record) + "\n")
        self._fh.flush()

    def log_coach_cycle(self, tick: int, team: str, alerts: list[dict]) -> None:
        """Embed a ``coach_cycle`` block into the line for ``tick``.

        Targets the line whose ``tick`` matches the argument (the most recent
        such line if several share it), falling back to the last line if no match
        is found. This keeps the cycle attached to its own tick even when the
        simulator has written newer ticks while the coach was deciding. The file
        is rewritten atomically (``.tmp`` then ``os.replace``). ``alerts`` entries
        look like::

            {player_id, coach_reasoning, player_decision, override_written,
             target_position, duration_ticks, response_time_s}
        """
        # Ensure buffered writes are on disk before we read the file back.
        self._fh.flush()
        with open(self.path, encoding="utf-8") as fh:
            lines = fh.readlines()
        if not lines:
            return

        # Find the most recent line whose tick matches; else fall back to the
        # last line. The simulator may have appended newer ticks while the coach
        # was mid-cycle, so the block must attach to its own tick, not the tail.
        idx = len(lines) - 1
        for i in range(len(lines) - 1, -1, -1):
            try:
                if json.loads(lines[i]).get("tick") == tick:
                    idx = i
                    break
            except json.JSONDecodeError:
                continue

        record = json.loads(lines[idx])
        block = {"tick": tick, "team": team, "alerts": alerts}
        record.setdefault("coach_cycle", []).append(block)
        lines[idx] = json.dumps(record) + "\n"

        tmp = self.path + ".tmp"
        with open(tmp, "w", encoding="utf-8") as fh:
            fh.writelines(lines)
            fh.flush()
            os.fsync(fh.fileno())
        os.replace(tmp, self.path)

        # The appended fd now points at the replaced inode; reopen for appends.
        self._fh.close()
        self._fh = open(self.path, "a", encoding="utf-8")

    def close(self) -> None:
        """Flush and close the underlying file handle."""
        if not self._fh.closed:
            self._fh.flush()
            self._fh.close()
