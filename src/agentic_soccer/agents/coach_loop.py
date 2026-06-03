"""Coach loop run by each Claude Code coach session (one per team).

Every cycle (~15s) the coach:

1. reads the match status (narrator text) for its team via the MCP HTTP client;
2. asks Claude — loaded with the ``.claude/agents/coach-*.md`` system prompt — to
   return structured JSON ``{"alerts": [{"player_id", "situation", "reasoning"}]}``
   naming up to three players whose default AI behavior is wrong for the moment
   (falling back to a cheap zone-based heuristic if the LLM call/parse fails);
3. spawns a :func:`~agents.player_agent.player_agent` sub-agent for each alert,
   which decides hold-vs-override and writes any override over HTTP; and
4. logs the cycle to the replay file as a ``coach_cycle`` block.

``run_coach_cycle`` performs one pass and is pure with respect to time — the
caller supplies the sim ``tick`` (the simulator owns the replay file). The
``coach_loop`` driver is the thin loop that sleeps between cycles; tests call
``run_coach_cycle``/``decide_alerts`` directly and never sleep.

Architecture: the simulator + MCP server run in a Docker container with port
8765 mapped to the host; coach sessions run on the host and reach the server at
``http://localhost:8765``.
"""

from __future__ import annotations

import json
import time
from typing import TYPE_CHECKING, Any, cast

from agentic_soccer.agents.personalities import (
    COACH_PERSONALITIES,
    ZONE_ATTACKING,
    ZONE_DEFENSIVE,
    ZONE_MIDFIELD,
    load_system_prompt,
    player_by_index,
    players_for_team,
)
from agentic_soccer.agents.player_agent import MCPHttpClient, player_agent


if TYPE_CHECKING:
    from collections.abc import Callable

    from agentic_soccer.agents.player_agent import LLMClient, MatchClient
    from agentic_soccer.replay.logger import ReplayLogger


COACH_INTERVAL_S = 15.0
MAX_ALERTS_PER_CYCLE = 3
COACH_MODEL = "claude-sonnet-4-6"
_MAX_TOKENS = 1024

# Heuristic: minimum zone-band distance (0=def,1=mid,2=att) before intervening.
OUT_OF_POSITION_THRESHOLD = 2

_PLAYERS_HEADER = "WHAT YOUR PLAYERS ARE DOING:"

# Behavior-text zone phrases -> band (most specific first).
_ZONE_BANDS: tuple[tuple[str, int], ...] = (
    ("opposition penalty area", 2),
    ("own penalty area", 0),
    ("attacking third", 2),
    ("defensive third", 0),
    ("midfield", 1),
)
_EXPECTED_BANDS: dict[str, int] = {ZONE_DEFENSIVE: 0, ZONE_MIDFIELD: 1, ZONE_ATTACKING: 2}
_BAND_DESCRIPTIONS: dict[int, str] = {
    0: "deep in the defensive third",
    1: "in midfield",
    2: "high in the attacking third",
}

_DEFAULT_COACH_PROMPT = (
    "You are a football coach. Read the match report and identify up to 3 players "
    "whose current behavior is wrong for the situation."
)

_COACH_INSTRUCTION = (
    'Respond with ONLY JSON: {"alerts": [{"player_id": "<id>", "situation": "<what is wrong and '
    'what to do>", "reasoning": "<why>"}]}. List at most 3 alerts; use the exact player_id values '
    "from your roster. If everyone is well-positioned, return an empty alerts list."
)


# -- narrator parsing -------------------------------------------------------


def parse_player_behaviors(status: str) -> list[str]:
    """Extract per-player behavior strings (squad-index order) from narrator text."""
    lines = status.splitlines()
    try:
        start = next(i for i, line in enumerate(lines) if line.strip() == _PLAYERS_HEADER)
    except StopIteration:
        return []
    behaviors: list[str] = []
    for line in lines[start + 1 :]:
        stripped = line.strip()
        if not stripped.startswith("- "):
            break
        _, _, behavior = stripped[2:].partition(": ")
        behaviors.append(behavior)
    return behaviors


def _behavior_band(behavior: str) -> int | None:
    lowered = behavior.lower()
    for phrase, band in _ZONE_BANDS:
        if phrase in lowered:
            return band
    return None


def heuristic_alerts(team: str, status: str) -> list[dict[str, Any]]:
    """Cheap fallback: flag the most out-of-position players from narrator text.

    Returns ``[{player_id, situation, reasoning}]`` (≤ :data:`MAX_ALERTS_PER_CYCLE`),
    used when the coach LLM call or its JSON parse fails.
    """
    scored: list[tuple[int, int, dict[str, Any]]] = []  # (distance, index, alert)
    for index, behavior in enumerate(parse_player_behaviors(status)):
        profile = player_by_index(team, index)
        if profile is None:
            continue
        current = _behavior_band(behavior)
        if current is None:
            continue
        expected = _EXPECTED_BANDS[profile["expected_zone"]]
        distance = abs(current - expected)
        if distance < OUT_OF_POSITION_THRESHOLD:
            continue
        where = _BAND_DESCRIPTIONS[current]
        should = _BAND_DESCRIPTIONS[expected]
        fix = "push higher up the pitch" if current < expected else "recover into position"
        alert = {
            "player_id": profile["player_id"],
            "situation": f"You are {where} but should be {should} — {fix}.",
            "reasoning": f"{profile['role']} out of position (zone distance {distance}).",
        }
        scored.append((distance, index, alert))
    scored.sort(key=lambda item: (-item[0], item[1]))
    return [alert for _, _, alert in scored[:MAX_ALERTS_PER_CYCLE]]


# -- coach LLM decision -----------------------------------------------------


def _parse_alerts_json(text: str, valid_ids: set[str]) -> list[dict[str, Any]]:
    """Parse the coach's JSON reply into validated alert dicts.

    Tolerates fenced code blocks and surrounding prose by extracting the first
    ``{...}`` object. Drops alerts whose ``player_id`` is not on the roster.
    """
    start, end = text.find("{"), text.rfind("}")
    if start == -1 or end == -1 or end < start:
        msg = "no JSON object found in coach response"
        raise ValueError(msg)
    payload = json.loads(text[start : end + 1])
    alerts: list[dict[str, Any]] = []
    for raw in payload.get("alerts", []):
        pid = raw.get("player_id")
        if pid not in valid_ids:
            continue  # drop unknown ids, then cap to MAX valid alerts
        alerts.append(
            {
                "player_id": pid,
                "situation": raw.get("situation", ""),
                "reasoning": raw.get("reasoning", ""),
            },
        )
        if len(alerts) >= MAX_ALERTS_PER_CYCLE:
            break
    return alerts


def _coach_text(response: object) -> str:
    """Concatenate the text blocks of an Anthropic response."""
    parts = [getattr(b, "text", "") for b in getattr(response, "content", []) or [] if getattr(b, "type", None) == "text"]
    return "".join(parts)


def decide_alerts(
    team: str,
    status: str,
    *,
    anthropic_client: LLMClient | None = None,
    model: str = COACH_MODEL,
) -> list[dict[str, Any]]:
    """Ask the coach LLM which players to alert; fall back to the heuristic.

    Returns ``[{player_id, situation, reasoning}]`` (≤ :data:`MAX_ALERTS_PER_CYCLE`).
    Any error (no client, API failure, unparseable reply) falls back to
    :func:`heuristic_alerts` so the coach never stalls.
    """
    roster = players_for_team(team)
    try:
        client = anthropic_client if anthropic_client is not None else _default_anthropic()
        system_prompt = load_system_prompt(COACH_PERSONALITIES[team]["agent_file"]) or _DEFAULT_COACH_PROMPT
        user = f"{status}\n\n{_COACH_INSTRUCTION}"
        response = client.messages.create(
            model=model,
            max_tokens=_MAX_TOKENS,
            system=system_prompt,
            messages=[{"role": "user", "content": user}],
        )
        return _parse_alerts_json(_coach_text(response), set(roster))
    except Exception:  # noqa: BLE001 - any LLM/parse failure degrades to the heuristic
        return heuristic_alerts(team, status)


def _default_anthropic() -> LLMClient:
    import anthropic  # noqa: PLC0415 - lazy: only construct a real client when no fake is injected

    # anthropic.Anthropic satisfies LLMClient structurally at runtime; cast
    # because its generated stubs don't match our minimal Protocol.
    return cast("LLMClient", anthropic.Anthropic())


# -- cycle + driver ---------------------------------------------------------


def run_coach_cycle(  # noqa: PLR0913 - orchestration entry; the extra params are injectable DI seams
    team: str,
    tick: int,
    *,
    client: MatchClient,
    replay_logger: ReplayLogger | None = None,
    anthropic_client: LLMClient | None = None,
    spawn: Callable[..., dict[str, Any]] = player_agent,
    mcp_base_url: str = "http://localhost:8765",
    session: object | None = None,
    coach_model: str = COACH_MODEL,
) -> list[dict[str, Any]]:
    """Run one coach pass at ``tick``; returns the per-player alert records.

    Reads status via ``client``, decides alerts (LLM + heuristic fallback),
    spawns a player agent per alert, and — when any player was alerted — logs a
    ``coach_cycle`` block to ``replay_logger`` at ``tick``.
    """
    status = client.get_match_status(team)
    decisions = decide_alerts(team, status, anthropic_client=anthropic_client, model=coach_model)

    alerts = [
        spawn(
            decision["player_id"],
            decision["situation"],
            team,
            mcp_base_url,
            anthropic_client=anthropic_client,
            session=session,
        )
        for decision in decisions
    ]

    if alerts and replay_logger is not None:
        replay_logger.log_coach_cycle(tick, team, alerts)
    return alerts


def coach_loop(  # noqa: PLR0913 - public driver; the extra params are injectable DI seams for tests
    team: str,
    mcp_base_url: str = "http://localhost:8765",
    *,
    replay_logger: ReplayLogger | None = None,
    tick_source: Callable[[], int] | None = None,
    interval_s: float = COACH_INTERVAL_S,
    max_cycles: int | None = None,
    sleep: Callable[[float], None] = time.sleep,
) -> None:
    """Main coach loop: run cycles forever (or ``max_cycles``), sleeping between.

    Builds an :class:`MCPHttpClient` for ``team`` and drives
    :func:`run_coach_cycle`. ``tick_source`` returns the current sim tick
    (defaults to a monotonic counter when the simulator tick is not wired in).
    """
    token = COACH_PERSONALITIES[team]["token"]
    client = MCPHttpClient(mcp_base_url, token)
    counter = 0
    cycles = 0
    while max_cycles is None or cycles < max_cycles:
        tick = tick_source() if tick_source is not None else counter
        run_coach_cycle(
            team,
            tick,
            client=client,
            replay_logger=replay_logger,
            mcp_base_url=mcp_base_url,
        )
        counter += 1
        cycles += 1
        if max_cycles is None or cycles < max_cycles:
            sleep(interval_s)


def main() -> None:
    """CLI entry point: ``python -m agents.coach_loop <home|away>``."""
    import sys  # noqa: PLC0415 - only needed for the CLI path

    team = sys.argv[1] if len(sys.argv) > 1 else "home"
    coach_loop(team)


if __name__ == "__main__":
    main()
