"""Container entrypoint: run the MCP server and the simulation loop together.

Both must live in ONE process because they share the in-process
``mcp_server.server.GAME_STATE`` (per the MCP-server contract). The FastMCP
HTTP server runs in a daemon thread on port 8765; the match loop runs in the
main thread and mutates GAME_STATE, which the MCP tools read.
"""

from __future__ import annotations

import os
import threading
import time


def _start_mcp_server() -> None:
    """Start the FastMCP HTTP server (blocking) — run inside a daemon thread.

    NOTE: ``mcp_server.server.main()`` binds ``host="127.0.0.1"``, which is
    unreachable from outside the container even with ``-p 8765:8765``. For the
    containerized deployment we must bind ``0.0.0.0``, so we drive ``mcp.run``
    directly with the server's transport/port rather than calling ``main()``.
    """
    from mcp_server.server import PORT, mcp  # noqa: PLC0415

    # Bind all interfaces: this is the in-container server entrypoint, and
    # 127.0.0.1 would be unreachable from the host even with -p 8765:8765.
    # Exposure is limited to the Docker port mapping and the MCP tools are
    # token-authed; the host bind is the deliberate purpose of this entrypoint.
    mcp.run(
        transport="http",
        host="0.0.0.0",  # noqa: S104 - container entrypoint; host must reach the server
        port=PORT,
    )


def _run_match() -> int:
    """Run the headless match loop in this process. Returns process exit code.

    The engine reads coach overrides via the SAME in-process
    ``mcp_server.server.GAME_STATE`` singleton the FastMCP server mutates — that
    shared object is the whole reason server + match run in one process.
    """
    from mcp_server.server import set_replay_logger  # noqa: PLC0415
    from simulator.engine import SoccerEngine  # noqa: PLC0415

    engine = SoccerEngine()
    # Share the engine's replay logger with the MCP server so every override
    # write records a coach_cycle into the same replay.jsonl.
    set_replay_logger(engine.logger)
    engine.run()
    return 0


def main() -> int:
    """Launch the MCP server (daemon thread) then run the match in the main thread."""
    os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
    os.environ.setdefault("SDL_AUDIODRIVER", "dummy")

    server_thread = threading.Thread(
        target=_start_mcp_server,
        name="mcp-server",
        daemon=True,
    )
    server_thread.start()
    time.sleep(2)  # let the server bind :8765 before the match starts
    print("docker_entry: MCP server thread started on :8765", flush=True)

    return _run_match()


if __name__ == "__main__":
    raise SystemExit(main())
