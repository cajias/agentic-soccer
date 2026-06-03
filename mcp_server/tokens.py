"""Single source of truth for team auth tokens.

Every MCP tool call carries a bearer token that maps to exactly one team. The
simulator/launcher hands each coach its team's token. These constants are the
ONLY place the secrets are defined; ``mcp_server.server``,
``agents.personalities`` and ``check_milestones`` all import from here.
"""

from __future__ import annotations


# Token -> team.
TOKENS: dict[str, str] = {
    "home-secret-abc": "home",
    "away-secret-xyz": "away",
}

# Team -> token (reverse map, for callers that start from a team name).
TEAM_TOKENS: dict[str, str] = {team: token for token, team in TOKENS.items()}
