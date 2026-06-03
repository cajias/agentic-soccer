"""Headless GIF frame dumper for agentic-soccer replays.

Renders every Nth frame of a ``replay.jsonl`` to PNGs via the visualizer's
``render_frame`` seam under SDL's dummy driver (no display). Run inside the
Docker image (pygame + assets present); ffmpeg assembles the GIF on the host.

The COACH banner is held for ``BANNER_HOLD_TICKS`` ticks after each coaching
cycle so sparse overrides stay visible across the sub-sampled frames.

Usage (in container):
  python scripts/capture_gif.py match/replay.jsonl match/frames 8
"""

from __future__ import annotations

import dataclasses
import os
import sys
from pathlib import Path


os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
os.environ.setdefault("SDL_AUDIODRIVER", "dummy")

import pygame  # SDL env defaults above must be set before importing pygame

from agentic_soccer.replay.visualizer import (
    HEIGHT,
    WIDTH,
    load_assets,
    load_replay,
    render_frame,
)


def _hold_banners(frames: list, hold_ticks: int) -> list:
    """Carry each coach cycle forward for ``hold_ticks`` so its banner persists.

    ``render_frame`` only draws the banner on the exact cycle tick; with sparse
    cycles and sub-sampling the banner would flash for a single frame. ``Frame``
    is frozen, so we rebuild held copies with :func:`dataclasses.replace`.
    """
    last_cycles: list = []
    last_tick = -(10**9)
    held: list = []
    for idx, fr in enumerate(frames):
        if fr.coach_cycles:
            last_cycles = list(fr.coach_cycles)
            last_tick = idx
            held.append(fr)
        elif idx - last_tick <= hold_ticks and last_cycles:
            held.append(dataclasses.replace(fr, coach_cycles=last_cycles))
        else:
            held.append(fr)
    return held


def main() -> int:
    """Render sub-sampled replay frames to PNGs and return a process exit code."""
    replay_path = sys.argv[1] if len(sys.argv) > 1 else "match/replay.jsonl"
    out_dir = Path(sys.argv[2] if len(sys.argv) > 2 else "match/frames")
    step = int(sys.argv[3]) if len(sys.argv) > 3 else 8
    hold_ticks = int(os.environ.get("BANNER_HOLD_TICKS", str(step * 6)))

    out_dir.mkdir(parents=True, exist_ok=True)
    pygame.init()  # inits font/display under the dummy driver

    frames = _hold_banners(load_replay(replay_path), hold_ticks)
    assets = load_assets()
    surface = pygame.Surface((WIDTH, HEIGHT))

    n = 0
    for i in range(0, len(frames), step):
        render_frame(surface, frames, i, assets)
        pygame.image.save(surface, str(out_dir / f"frame_{n:05d}.png"))
        n += 1

    print(f"WROTE {n} frames from {len(frames)} ticks (step={step}) -> {out_dir}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
