r"""Simulator entry point: run a single full match and print the result.

Designed to run inside the gfootball Docker image (Linux), where gfootball
builds cleanly. The engine shares the in-process ``mcp_server.server.GAME_STATE``
singleton with the MCP server, so player overrides posted by the coach/player
agents take effect live and narrator text is published back each tick.

Usage (inside the container)::

    SDL_VIDEODRIVER=dummy python simulator_main.py

``match/`` is a mounted volume, so the replay is written to ``match/replay.jsonl``
relative to the working directory.
"""

from __future__ import annotations

import os
from typing import Any

# Force gfootball's SDL backend to run headless before the engine is imported.
os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
os.environ.setdefault("SDL_AUDIODRIVER", "dummy")


def main() -> dict[str, Any]:
    """Run a full match and return its final stats."""
    from simulator.engine import SoccerEngine  # noqa: PLC0415

    engine = SoccerEngine()
    stats = engine.run()
    print(f"Final: Home {stats['home_goals']} - {stats['away_goals']} Away")
    return stats


if __name__ == "__main__":
    main()
