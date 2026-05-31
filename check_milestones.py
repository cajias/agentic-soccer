#!/usr/bin/env python
"""Check which milestones are complete. Outputs 0-5 as the final stdout line.

Each check is defensive: any failure (missing module, no running server, no
replay file) is caught and counts as "not met" rather than crashing the runner.
Heavy/optional dependencies (requests, pygame, the engine) are imported lazily
inside each check so this script stays importable even before those parts of the
system exist — and before the gfootball Docker image is built.

M1 runs the engine in-process; M2/M4 talk to a *running* ``simulator.py`` over
HTTP (cross-process), so start the simulator first for those to pass.
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

# Headless: the engine and pygame must not open a real window or audio device.
os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
os.environ.setdefault("SDL_AUDIODRIVER", "dummy")

# Endpoints and thresholds kept as named constants (out of inline comparisons).
_MCP_URL = "http://localhost:8765/get_match_status"
_HTTP_TIMEOUT = 2.0
_MIN_STATUS_LEN = 10
_SCORE_FIELDS = 2
_EXPECTED_TEAMS = 2
_M1_STEPS = 50
_M1_REPLAY_PATH = "match/test_replay.jsonl"
_REPLAY_PATH = "match/replay.jsonl"
_TOTAL_MILESTONES = 5

# Team tokens (mirror mcp_server.server.TOKENS); every tool call needs one.
_TOKENS = {"home": "home-secret-abc", "away": "away-secret-xyz"}


def check_m1() -> bool:
    """M1: Headless match runs to completion with a score."""
    try:
        from simulator.engine import SoccerEngine  # noqa: PLC0415 - optional, lazy

        engine = SoccerEngine(match_steps=_M1_STEPS, replay_path=_M1_REPLAY_PATH)
        stats = engine.run()
    except Exception as e:  # noqa: BLE001 - any failure means the milestone is unmet
        print(f"M1 fail: {e}", file=sys.stderr)
        return False
    score = stats.get("score")
    return isinstance(score, list) and len(score) == _SCORE_FIELDS


def check_m2() -> bool:
    """M2: MCP server starts and get_match_status returns narrator text."""
    try:
        import requests  # noqa: PLC0415 - optional, lazy

        resp = requests.get(
            _MCP_URL,
            params={"team": "home"},
            headers={"Authorization": f"Bearer {_TOKENS['home']}"},
            timeout=_HTTP_TIMEOUT,
        )
    except Exception as e:  # noqa: BLE001 - any failure means the milestone is unmet
        print(f"M2 fail: {e}", file=sys.stderr)
        return False
    return bool(resp.ok) and len(resp.text) > _MIN_STATUS_LEN


def check_m3() -> bool:
    """M3: replay.jsonl exists and has at least one coach_cycle entry."""
    try:
        replay = Path(_REPLAY_PATH)
        if not replay.exists():
            return False
        with replay.open(encoding="utf-8") as f:
            return any("coach_cycle" in json.loads(line) for line in f if line.strip())
    except Exception as e:  # noqa: BLE001 - any failure means the milestone is unmet
        print(f"M3 fail: {e}", file=sys.stderr)
        return False


def check_m4() -> bool:
    """M4: Two token-scoped MCP connections work simultaneously."""
    try:
        import threading  # noqa: PLC0415 - optional, lazy

        import requests  # noqa: PLC0415 - optional, lazy

        results: list[bool] = []

        def try_team(team: str, token: str) -> None:
            """Hit the status endpoint as one team and record success.

            Defensive: a connection failure records False rather than letting an
            exception escape the worker thread.
            """
            try:
                resp = requests.get(
                    _MCP_URL,
                    params={"team": team},
                    headers={"Authorization": f"Bearer {token}"},
                    timeout=_HTTP_TIMEOUT,
                )
                results.append(bool(resp.ok))
            except Exception as e:  # noqa: BLE001 - a down server means the milestone is unmet
                print(f"M4 {team} fail: {e}", file=sys.stderr)
                results.append(False)

        threads = [threading.Thread(target=try_team, args=(team, token)) for team, token in _TOKENS.items()]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
    except Exception as e:  # noqa: BLE001 - any failure means the milestone is unmet
        print(f"M4 fail: {e}", file=sys.stderr)
        return False
    return len(results) == _EXPECTED_TEAMS and all(results)


def check_m5() -> bool:
    """M5: pygame replayer launches and loads replay.jsonl without crashing."""
    try:
        import pygame  # noqa: F401, PLC0415 - imported only to confirm it installs

        from replay.visualizer import load_replay  # noqa: PLC0415 - optional, lazy

        frames = load_replay(_REPLAY_PATH)
    except Exception as e:  # noqa: BLE001 - any failure means the milestone is unmet
        print(f"M5 fail: {e}", file=sys.stderr)
        return False
    return len(frames) > 0


def main() -> int:
    """Run all milestone checks and return the count passed (0-5)."""
    checks = [check_m1, check_m2, check_m3, check_m4, check_m5]
    count = 0
    for i, check in enumerate(checks, 1):
        passed = check()
        status = "PASS" if passed else "----"
        print(f"M{i}: {status} {(check.__doc__ or '').strip()}")
        if passed:
            count += 1
    print(f"\nMilestones: {count}/{_TOTAL_MILESTONES}")
    return count


if __name__ == "__main__":
    milestones_passed = main()
    sys.stdout.write(f"{milestones_passed}\n")  # final line is the machine-readable number
