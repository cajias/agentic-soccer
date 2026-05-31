"""Player sub-agent: decides whether to override one player's in-engine AI.

The coach spawns a player agent for each alert. The agent loads the player's
``.claude/agents/player-*.md`` body as its LLM system prompt, asks Claude — via a
single forced tool call — to either ``hold`` (trust the engine) or ``override``
with a target position and duration, and on an override writes the directive back
through the MCP server over HTTP (Docker host -> container :8765).

The Anthropic client and the HTTP ``session`` are injected so the agent is fully
testable with fakes; nothing here touches the network on its own unless a real
client/session is constructed lazily.
"""

from __future__ import annotations

import time
from typing import TYPE_CHECKING, Any, Protocol

import anthropic
import requests

from agents.personalities import (
    COACH_PERSONALITIES,
    PLAYER_PERSONALITIES,
    load_system_prompt,
)


if TYPE_CHECKING:
    from collections.abc import Mapping


# Players decide frequently; Haiku is fast and cheap (matches player-*.md).
DEFAULT_MODEL = "claude-haiku-4-5-20251001"

# Override window (sim ticks) when the model omits a duration.
DEFAULT_OVERRIDE_TICKS = 150

_MAX_TOKENS = 400
_HTTP_TIMEOUT = 10.0

_DEFAULT_PLAYER_PROMPT = (
    "You are a football player. Given your coach's alert and the match report, "
    "decide whether to override your in-engine AI with a target position, or hold."
)

# Forced tool: structured, parse-free output.
_DIRECTIVE_TOOL: dict[str, Any] = {
    "name": "set_player_directive",
    "description": (
        "Decide whether to override the player's in-engine AI. 'override' redirects "
        "the player to a target position for a number of ticks; 'hold' trusts the engine."
    ),
    "input_schema": {
        "type": "object",
        "properties": {
            "action": {"type": "string", "enum": ["override", "hold"]},
            "target_position": {
                "type": "array",
                "items": {"type": "number"},
                "description": "Desired [x, y] in [-1,1]x[-0.42,0.42], attacking toward +x. Required for override.",
            },
            "duration_ticks": {"type": "integer", "description": "How long the override lasts, in sim ticks."},
            "reasoning": {"type": "string", "description": "One sentence justifying the decision."},
        },
        "required": ["action", "reasoning"],
    },
}


class MCPHttpClient:
    """Talks to the MCP server over HTTP using ``requests``.

    The session is injectable so tests can supply a fake. Auth uses the team
    token via both ``X-Team-Token`` and ``Authorization: Bearer`` headers, which
    ``mcp_server.server`` accepts.
    """

    def __init__(self, base_url: str, token: str, *, session: object | None = None, timeout: float = _HTTP_TIMEOUT) -> None:
        """Bind to ``base_url`` with a team ``token``; ``session`` defaults to a new requests Session."""
        self._base = base_url.rstrip("/")
        self._token = token
        self._session = session if session is not None else requests.Session()
        self._timeout = timeout

    def _headers(self) -> dict[str, str]:
        return {"X-Team-Token": self._token, "Authorization": f"Bearer {self._token}"}

    def get_match_status(self, team: str) -> str:
        """Return the narrator text for ``team`` (HTTP GET ``/get_match_status``)."""
        resp = self._session.get(
            f"{self._base}/get_match_status",
            params={"team": team},
            headers=self._headers(),
            timeout=self._timeout,
        )
        resp.raise_for_status()
        data = resp.json()
        return data.get("narrator", "") if isinstance(data, dict) else str(data)

    def update_player_override(self, team: str, player_id: str, override: dict) -> dict:
        """Store a behavior override (HTTP POST ``/update_player_override``)."""
        resp = self._session.post(
            f"{self._base}/update_player_override",
            json={"team": team, "player_id": player_id, "override": override},
            headers=self._headers(),
            timeout=self._timeout,
        )
        resp.raise_for_status()
        return resp.json()


class MatchClient(Protocol):
    """Structural type for the MCP client the agents depend on."""

    def get_match_status(self, team: str) -> str:
        """Return the current narrator text for ``team``."""
        ...

    def update_player_override(self, team: str, player_id: str, override: dict) -> dict:
        """Store a behavior override for one player on ``team``."""
        ...


class _LLMMessages(Protocol):
    def create(self, **kwargs: object) -> object: ...


class LLMClient(Protocol):
    """Minimal structural view of the Anthropic client (its ``messages`` API)."""

    messages: _LLMMessages


def _extract_directive(response: object) -> dict | None:
    """Pull the first ``set_player_directive`` tool input from a response."""
    for block in getattr(response, "content", []) or []:
        if getattr(block, "type", None) == "tool_use" and getattr(block, "name", None) == _DIRECTIVE_TOOL["name"]:
            return dict(block.input)
    return None


class PlayerAgent:
    """LLM-driven decision agent for a single player on one team."""

    def __init__(
        self,
        team: str,
        client: MatchClient,
        *,
        anthropic_client: LLMClient | None = None,
        model: str = DEFAULT_MODEL,
    ) -> None:
        """Bind the agent to a ``team`` and the MCP ``client`` (Anthropic client injected for tests)."""
        self.team = team
        self.client = client
        self._anthropic = anthropic_client
        self._model = model

    def _llm(self) -> LLMClient:
        """Return the Anthropic client, constructing a default one if needed."""
        if self._anthropic is None:
            self._anthropic = anthropic.Anthropic()
        return self._anthropic

    def decide(self, player_id: str, personality: Mapping[str, Any], situation: str) -> dict:
        """Decide hold-vs-override for ``player_id`` and write any override.

        Returns the replay ``coach_cycle`` alert record::

            {player_id, coach_reasoning, player_decision, override_written,
             target_position, duration_ticks, response_time_s}

        ``coach_reasoning`` carries the coach's ``situation``; the player's own
        rationale goes into the engine override's ``reasoning`` field.
        """
        system_prompt = load_system_prompt(personality.get("agent_file", "")) or _DEFAULT_PLAYER_PROMPT
        user = f"Coach alert: {situation}\n\nCall set_player_directive exactly once with your decision."

        start = time.perf_counter()
        response = self._llm().messages.create(
            model=self._model,
            max_tokens=_MAX_TOKENS,
            system=system_prompt,
            tools=[_DIRECTIVE_TOOL],
            tool_choice={"type": "tool", "name": _DIRECTIVE_TOOL["name"]},
            messages=[{"role": "user", "content": user}],
        )
        response_time_s = round(time.perf_counter() - start, 4)

        directive = _extract_directive(response) or {"action": "hold"}
        target_position = directive.get("target_position")
        # An 'override' without a usable target is treated as a safe hold.
        is_override = directive.get("action") == "override" and target_position is not None
        duration_ticks = int(directive.get("duration_ticks") or DEFAULT_OVERRIDE_TICKS) if is_override else 0

        override_written = False
        if is_override:
            override = {
                "target_position": [float(target_position[0]), float(target_position[1])],
                "duration_ticks": duration_ticks,
                "reasoning": directive.get("reasoning", ""),
            }
            self.client.update_player_override(self.team, player_id, override)
            override_written = True

        return {
            "player_id": player_id,
            "coach_reasoning": situation,
            "player_decision": "override" if is_override else "hold",
            "override_written": override_written,
            "target_position": list(target_position) if is_override else None,
            "duration_ticks": duration_ticks,
            "response_time_s": response_time_s,
        }


def player_agent(  # noqa: PLR0913 - spec signature plus injectable DI seams (anthropic/session/model)
    player_id: str,
    situation: str,
    team: str,
    mcp_base_url: str = "http://localhost:8765",
    *,
    anthropic_client: LLMClient | None = None,
    session: object | None = None,
    model: str = DEFAULT_MODEL,
) -> dict:
    """Spawn a player agent to decide on an override and return its alert record.

    Looks up the personality, builds an :class:`MCPHttpClient` with the team
    token, and delegates to :meth:`PlayerAgent.decide`.
    """
    personality = PLAYER_PERSONALITIES[player_id]
    token = COACH_PERSONALITIES[team]["token"]
    client = MCPHttpClient(mcp_base_url, token, session=session)
    agent = PlayerAgent(team, client, anthropic_client=anthropic_client, model=model)
    return agent.decide(player_id, personality, situation)
