"""Before/after demo: baseline (no coach) vs coached (one override applied).

Runs two short matches with the SAME engine and compares the replays to show
that a coach instruction measurably changes a player's behaviour. Designed to
run inside the Docker image (real gfootball). Writes two replay files and prints
a quantitative diff.

    docker run --rm -e SDL_VIDEODRIVER=dummy -e SDL_AUDIODRIVER=dummy \
        -v "$PWD/match:/app/match" agentic-soccer:latest python demo_before_after.py
"""

from __future__ import annotations

import json
from pathlib import Path

from mcp_server.server import GAME_STATE
from simulator.engine import SoccerEngine


STEPS = 400
TARGET_PLAYER = 9  # home RW (Salah)
# Minimum avg-x shift (in gfootball pitch units) we treat as a real behaviour change.
MIN_MEANINGFUL_SHIFT = 0.02
OVERRIDE = {"target_position": [0.7, 0.0], "duration_ticks": STEPS,
            "reasoning": "demo: push RW high and central"}


def _avg_x(replay_path: str, team: str, player_id: int) -> float:
    """Mean x-position of one player across all ticks in a replay."""
    xs = [
        p["x"]
        for line in Path(replay_path).read_text().splitlines()
        for p in json.loads(line)["players"]
        if p["team"] == team and p["id"] == player_id
    ]
    return sum(xs) / len(xs) if xs else 0.0


def _coach_cycles(replay_path: str) -> int:
    return sum(
        1 for line in Path(replay_path).read_text().splitlines()
        if "coach_cycle" in json.loads(line)
    )


def main() -> None:
    """Run the baseline and coached matches and print a quantitative diff."""
    # BEFORE: no overrides, pure built-in AI.
    GAME_STATE.clear_all() if hasattr(GAME_STATE, "clear_all") else None
    before = "match/replay_before.jsonl"
    SoccerEngine(match_steps=STEPS, replay_path=before).run()

    # AFTER: apply one coach override before the run (as a coach session would
    # via update_player_override), then run with the same engine.
    after = "match/replay_after.jsonl"
    eng = SoccerEngine(match_steps=STEPS, replay_path=after)
    GAME_STATE.set_override("home", str(TARGET_PLAYER), OVERRIDE)
    eng.run()

    b_x = _avg_x(before, "home", TARGET_PLAYER)
    a_x = _avg_x(after, "home", TARGET_PLAYER)
    print("\n===== BEFORE / AFTER =====")
    print(f"BEFORE  coach_cycles={_coach_cycles(before)}  RW avg_x={b_x:+.3f}")
    print(f"AFTER   coach_cycles={_coach_cycles(after)}  RW avg_x={a_x:+.3f}")
    print(f"DELTA   RW avg_x shifted {a_x - b_x:+.3f} toward goal "
          f"({'measurable change' if abs(a_x - b_x) > MIN_MEANINGFUL_SHIFT else 'NO change'})")


if __name__ == "__main__":
    main()
