"""Coach session entry point: run one team's heuristic coach loop.

A coach reads its team's narrator text, flags out-of-position players, and posts
behaviour overrides through the MCP layer. :func:`run_coach` is reusable:
``simulator.py`` launches both coaches in-process (sharing the engine's replay
logger and the live tick), and it can also be run directly for a single team::

    python team_loop.py home

The in-process :class:`~agents.coach_loop.ServerMatchClient` shares ``GAME_STATE``
only with a server running in the SAME process. A true separate-session
(two-process / two Claude sessions) design needs an HTTP-backed ``MatchClient``,
which is not implemented yet — see the note in ``ServerMatchClient``.
"""

from __future__ import annotations

import sys
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Callable

    from replay.logger import ReplayLogger


# Team -> token. Mirrors ``mcp_server.server.TOKENS``; a coach holds exactly one.
TOKENS: dict[str, str] = {"home": "home-secret-abc", "away": "away-secret-xyz"}

DEFAULT_REPLAY = "match/replay.jsonl"


def run_coach(
    team: str,
    *,
    replay_logger: ReplayLogger | None = None,
    replay_path: str = DEFAULT_REPLAY,
    tick_source: Callable[[], int] | None = None,
    max_cycles: int | None = None,
) -> None:
    """Drive one team's coach loop (blocks until ``max_cycles``, else forever).

    Args:
        team: ``"home"`` or ``"away"``.
        replay_logger: Shared logger to annotate with ``coach_cycle`` blocks; a
            fresh one on ``replay_path`` is created when omitted. Pass the
            simulator's logger so coach cycles land in the same replay file.
        replay_path: Replay path used only when ``replay_logger`` is omitted.
        tick_source: Callable returning the current sim tick; defaults to reading
            the shared ``GAME_STATE.tick``.
        max_cycles: Stop after this many cycles (``None`` = run forever).
    """
    if team not in TOKENS:
        msg = f"unknown team {team!r}; expected one of {tuple(TOKENS)}"
        raise ValueError(msg)

    from agents.coach_loop import CoachLoop, ServerMatchClient  # noqa: PLC0415
    from mcp_server.server import GAME_STATE  # noqa: PLC0415
    from replay.logger import ReplayLogger  # noqa: PLC0415

    logger = replay_logger if replay_logger is not None else ReplayLogger(replay_path)

    if tick_source is None:

        def tick_source() -> int:
            return GAME_STATE.tick

    client = ServerMatchClient(token=TOKENS[team])
    coach = CoachLoop(team=team, client=client, replay_logger=logger)
    coach.drive(tick_source=tick_source, max_cycles=max_cycles)


def main() -> None:
    """Run a coach for the team named on the command line (default ``home``)."""
    team = sys.argv[1] if len(sys.argv) > 1 else "home"
    run_coach(team)


if __name__ == "__main__":
    main()
