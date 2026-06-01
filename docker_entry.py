"""Container entrypoint: run the MCP server and the simulation loop together.

Both must live in ONE process because they share the in-process
``mcp_server.server.GAME_STATE`` (per the MCP-server contract). The FastMCP
HTTP server runs in a daemon thread on port 8765; the match loop runs in the
main thread and mutates GAME_STATE, which the MCP tools read.

This wiring depends on the final APIs of the MCP server (task #3) and the
simulator (task #5). It is written defensively: each integration point is
resolved at runtime so the container still builds and the failure mode is a
clear message rather than an import-time crash. Adjust the marked call sites
once those modules land.
"""

from __future__ import annotations

import os
import sys
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
    shared object is the whole reason server + match run in one process. The
    contract (see check_milestones.py and team-lead) is
    ``from simulator.engine import SoccerEngine; SoccerEngine().run()``; we honor
    that first, then fall back to module-level entries, then degrade gracefully.
    """
    from simulator import engine  # noqa: PLC0415

    # Preferred: the SoccerEngine class (the documented integration contract).
    soccer_engine = getattr(engine, "SoccerEngine", None)
    if soccer_engine is not None:
        instance = soccer_engine()
        # Share the engine's replay logger with the MCP server so every override
        # write records a coach_cycle into the same replay.jsonl.
        logger = getattr(instance, "logger", None)
        if logger is not None:
            from mcp_server.server import set_replay_logger  # noqa: PLC0415

            set_replay_logger(logger)
        for run_name in ("run", "run_match", "play", "main"):
            run = getattr(instance, run_name, None)
            if callable(run):
                run()
                return 0

    # Fallback: a module-level runnable entry.
    for entry in ("run_match", "main", "run"):
        fn = getattr(engine, entry, None)
        if callable(fn):
            fn()
            return 0

    sys.stderr.write(
        "docker_entry: simulator.engine exposes no SoccerEngine().run() or "
        "module run_match/main/run yet (task #5 stub). MCP server is up on "
        ":8765; holding so the container stays useful.\n",
    )
    sys.stderr.flush()
    # Keep the server alive so the container stays useful while the sim lands.
    while True:
        time.sleep(3600)


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
