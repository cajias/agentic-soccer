"""Match engine stub.

Wraps the underlying soccer simulation (e.g. gfootball) and exposes a tick-based
API that the coach/player agents and the MCP server drive.
"""

from __future__ import annotations


class Engine:
    """Stub match engine. Replace with the real simulation integration."""

    def __init__(self) -> None:
        self.tick = 0

    def step(self) -> None:
        """Advance the simulation by one tick."""
        self.tick += 1


def main() -> None:
    raise NotImplementedError("engine stub")


if __name__ == "__main__":
    main()
