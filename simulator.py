"""Single-process launcher: MCP server + simulation + in-process coaches.

This is the "wire everything together" entry point. It:

1. starts the FastMCP HTTP server in a daemon thread (so coach/player agents can
   connect over HTTP with their team tokens);
2. launches both heuristic coaches in daemon threads; and
3. runs the gfootball match in the main thread.

The server, coaches, and simulator all share the ``mcp_server.server.GAME_STATE``
singleton, so overrides posted via MCP take effect live and narrator text is
published back each tick. All three write to a single thread-safe
:class:`~replay.logger.ReplayLogger` (the engine's), so coach-cycle annotations
land in the same ``match/replay.jsonl``.

Run inside the gfootball Docker image::

    SDL_VIDEODRIVER=dummy python simulator.py

NOTE: this file shares a name with the ``simulator/`` package; it is an entry
script (run as ``python simulator.py``), never imported as a module.
"""

from __future__ import annotations

import os
import threading
import time
from typing import Any


os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
os.environ.setdefault("SDL_AUDIODRIVER", "dummy")

# Run the two heuristic coaches in-process. Disable with RUN_COACHES=0.
_RUN_COACHES = os.environ.get("RUN_COACHES", "1") != "0"
# Give the HTTP server a moment to bind before the match starts.
_SERVER_BOOT_S = 1.0


def _serve_mcp() -> None:
    """Run the FastMCP HTTP server (blocking); used in a daemon thread."""
    from mcp_server.server import main as serve  # noqa: PLC0415

    serve()


def main() -> dict[str, Any]:
    """Launch server + coaches + match in one process; return final match stats."""
    from mcp_server.server import GAME_STATE  # noqa: PLC0415
    from simulator.engine import SoccerEngine  # noqa: PLC0415

    # 1. MCP HTTP server in the background so external agents can connect.
    threading.Thread(target=_serve_mcp, daemon=True, name="mcp-server").start()
    time.sleep(_SERVER_BOOT_S)

    # 2. Engine owns the shared GAME_STATE and the (thread-safe) replay logger.
    engine = SoccerEngine(game_state=GAME_STATE)

    # 3. Coaches share the engine's logger + the live tick (in-process).
    if _RUN_COACHES:
        from team_loop import run_coach  # noqa: PLC0415

        def tick_source() -> int:
            return GAME_STATE.tick

        for team in ("home", "away"):
            threading.Thread(
                target=run_coach,
                args=(team,),
                kwargs={"replay_logger": engine.logger, "tick_source": tick_source},
                daemon=True,
                name=f"coach-{team}",
            ).start()

    # 4. Run the match (blocks until done); daemon coaches/server stop with it.
    stats = engine.run()
    print(f"Final: Home {stats['home_goals']} - {stats['away_goals']} Away")
    return stats


if __name__ == "__main__":
    main()
