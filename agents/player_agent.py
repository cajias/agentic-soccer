"""Player sub-agent: decides whether to override a single player's AI behavior.

A player agent is spawned by the coach for one flagged player per cycle. It is
handed the player's role, personality, the coach's alert (why it was flagged),
and the current match status (the narrator text). It asks an LLM — via the
Anthropic SDK, using a single structured tool call — to either ``hold`` (trust
the in-engine AI) or ``override`` with a target position and duration. On an
override decision it writes the directive back through the MCP match client.

Both the Anthropic client and the match client are injected so the agent is
fully testable with fakes; nothing here touches the network on its own unless an
Anthropic client is constructed lazily from the environment.
"""

from __future__ import annotations

import time
from typing import TYPE_CHECKING, Any, Protocol

import anthropic


if TYPE_CHECKING:
    from collections.abc import Mapping


# Default model for player decisions. Haiku is fast and cheap for the high
# frequency of player sub-agent calls.
DEFAULT_MODEL = "claude-haiku-4-5-20251001"

# Fallback override window (sim ticks) when the model omits a duration.
DEFAULT_OVERRIDE_TICKS = 50

_MAX_TOKENS = 400

# The single tool the model must call. Forcing tool_choice gives us structured,
# parse-free output.
_DIRECTIVE_TOOL: dict[str, Any] = {
    "name": "set_player_directive",
    "description": (
        "Decide whether to override the player's in-engine AI. Choose 'hold' to "
        "trust the engine, or 'override' to redirect the player to a target "
        "position on the pitch for a number of ticks."
    ),
    "input_schema": {
        "type": "object",
        "properties": {
            "decision": {
                "type": "string",
                "enum": ["override", "hold"],
                "description": "Whether to override the engine AI or hold.",
            },
            "target_position": {
                "type": "array",
                "items": {"type": "number"},
                "description": "Desired [x, y] in [-1,1]x[-0.42,0.42], attacking toward +x. Required for override.",
            },
            "duration_ticks": {
                "type": "integer",
                "description": "How many sim ticks the override should last.",
            },
            "reasoning": {
                "type": "string",
                "description": "One sentence justifying the decision.",
            },
        },
        "required": ["decision", "reasoning"],
    },
}


class MatchClient(Protocol):
    """The subset of the MCP match interface the agents depend on."""

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


def _build_prompt(personality: Mapping[str, Any], alert: str, match_status: str) -> str:
    """Compose the user prompt for the player decision."""
    traits = ", ".join(personality.get("traits", []))
    return (
        f"You are {personality.get('name', 'a player')} ({personality.get('role', '?')}), "
        f"player #{personality.get('id')}. Traits: {traits}. "
        f"Your usual game: {personality.get('tendency', '')}\n\n"
        f"Your coach flagged you: {alert}\n\n"
        f"Current match report:\n{match_status}\n\n"
        "Decide whether to override your in-engine AI to fix this, or hold if the "
        "engine has it right. If overriding, give a target [x, y] and a duration "
        "in ticks. Call set_player_directive exactly once."
    )


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
        """Bind the agent to a ``team`` and the MCP ``client``.

        ``anthropic_client`` is injected for tests; if omitted it is created
        lazily from the environment on first use.
        """
        self.team = team
        self.client = client
        self._anthropic = anthropic_client
        self._model = model

    def _llm(self) -> LLMClient:
        """Return the Anthropic client, constructing a default one if needed."""
        if self._anthropic is None:
            # Constructed lazily so tests (which inject a fake) never need an API key.
            self._anthropic = anthropic.Anthropic()
        return self._anthropic

    def decide(self, player_id: int, personality: Mapping[str, Any], alert: str, match_status: str) -> dict:
        """Decide hold-vs-override for one player and write any override.

        Returns a result dict shaped for the replay ``coach_cycle`` alert::

            {player_id, coach_reasoning, player_decision, override_written,
             target_position, duration_ticks, response_time_s}

        ``coach_reasoning`` carries the coach's ``alert`` text; the player's own
        rationale is folded into the engine override's ``reasoning`` field.
        """
        prompt = _build_prompt(personality, alert, match_status)

        start = time.perf_counter()
        response = self._llm().messages.create(
            model=self._model,
            max_tokens=_MAX_TOKENS,
            tools=[_DIRECTIVE_TOOL],
            tool_choice={"type": "tool", "name": _DIRECTIVE_TOOL["name"]},
            messages=[{"role": "user", "content": prompt}],
        )
        response_time_s = round(time.perf_counter() - start, 4)

        directive = _extract_directive(response) or {"decision": "hold"}
        target_position = directive.get("target_position")
        # Treat an "override" with no usable target as a hold — never write a
        # malformed directive to the engine.
        is_override = directive.get("decision") == "override" and target_position is not None
        duration_ticks = int(directive.get("duration_ticks") or DEFAULT_OVERRIDE_TICKS) if is_override else 0

        override_written = False
        if is_override:
            override = {
                "target_position": [float(target_position[0]), float(target_position[1])],
                "duration_ticks": duration_ticks,
                "reasoning": directive.get("reasoning", ""),
            }
            self.client.update_player_override(self.team, str(player_id), override)
            override_written = True

        return {
            "player_id": player_id,
            "coach_reasoning": alert,
            "player_decision": "override" if is_override else "hold",
            "override_written": override_written,
            "target_position": list(target_position) if is_override else None,
            "duration_ticks": duration_ticks,
            "response_time_s": response_time_s,
        }
