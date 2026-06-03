r"""Orchestrated milestone run: start server + match (+ scripted override), then check.

Everything runs in ONE process so the MCP server, the match, and the replay
logger share state:

1. The FastMCP HTTP server runs in a daemon thread (so M2/M4 have a live
   endpoint).
2. A short match runs in the main thread, sharing ``GAME_STATE`` and its
   ``ReplayLogger`` with the server (so an override write logs a coach_cycle).
3. A scripted ``update_player_override`` fires mid-match — no LLM/SDK — which
   makes the server record a ``coach_cycle`` block into ``match/replay.jsonl``
   (satisfies M3).
4. ``entrypoints.check_milestones`` runs as a subprocess (a fresh process for its own M1
   gfootball env; M2/M4 reach the still-running server over HTTP; M3/M5 read the
   replay file). Its final ``0-5`` line is the result.

Run in-container::

    docker run --rm -e SDL_VIDEODRIVER=dummy -e SDL_AUDIODRIVER=dummy \\
        -e MCP_HOST=0.0.0.0 -v "$PWD/match:/app/match" \\
        agentic-soccer:latest python -m entrypoints.run_milestones
"""

from __future__ import annotations

import os
import subprocess
import sys
import threading
import time


os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
os.environ.setdefault("SDL_AUDIODRIVER", "dummy")
# Default MCP_HOST to all-interfaces: this runner only executes inside the
# gfootball Docker image, where the host must reach the in-container server.
# The bind is env-gated (MCP_HOST) and the MCP tools are token-authed.
os.environ.setdefault(
    "MCP_HOST",
    "0.0.0.0",  # noqa: S104 - in-container runner; host must reach the server, token-authed
)

_MATCH_STEPS = 400
_OVERRIDE_AT_TICK = 40
_SERVER_BOOT_S = 2.0


def _serve() -> None:
    """Run the FastMCP HTTP server (blocking) — used in a daemon thread."""
    from agentic_soccer.mcp_server.server import main as serve  # noqa: PLC0415

    serve()


def _scripted_override(game_state: object) -> None:
    """Fire one override mid-match to prove the coach_cycle path (no LLM)."""
    from agentic_soccer.mcp_server.server import update_player_override  # noqa: PLC0415

    while getattr(game_state, "tick", 0) < _OVERRIDE_AT_TICK:
        time.sleep(0.05)
    update_player_override(
        "home",
        "9",
        {
            "target_position": [0.7, 0.0],
            "duration_ticks": 200,
            "reasoning": "scripted: push RW high",
        },
        token="home-secret-abc",  # noqa: S106 - non-secret local team token
    )
    print("scripted override posted -> coach_cycle should be in replay", flush=True)


def main() -> int:
    """Start server + match + scripted override, then run check_milestones."""
    from agentic_soccer.mcp_server.server import GAME_STATE, set_replay_logger  # noqa: PLC0415
    from agentic_soccer.simulator.engine import SoccerEngine  # noqa: PLC0415

    threading.Thread(target=_serve, daemon=True, name="mcp-server").start()
    time.sleep(_SERVER_BOOT_S)
    print("MCP server up on :8765", flush=True)

    engine = SoccerEngine(
        match_steps=_MATCH_STEPS,
        replay_path="match/replay.jsonl",
        game_state=GAME_STATE,
    )
    set_replay_logger(engine.logger)
    threading.Thread(
        target=_scripted_override, args=(GAME_STATE,), daemon=True, name="override",
    ).start()

    stats = engine.run()
    print(f"match done: {stats}", flush=True)

    # check_milestones in a fresh process: its M1 gets a clean gfootball env, while
    # M2/M4 reach the still-running server (daemon thread in THIS process) over HTTP.
    result = subprocess.run(
        [sys.executable, "-m", "agentic_soccer.entrypoints.check_milestones"],
        capture_output=True,
        text=True,
        check=False,
    )
    sys.stdout.write(result.stdout)
    if result.stderr:
        sys.stderr.write(result.stderr)
    lines = [ln for ln in result.stdout.strip().splitlines() if ln.strip()]
    try:
        return int(lines[-1])
    except (ValueError, IndexError):
        return 0


if __name__ == "__main__":
    sys.exit(main())
