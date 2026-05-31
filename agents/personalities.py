"""Roster and coach configuration for the agent layer.

Player IDs use the ``{name}_{role}`` form the ``.claude/agents/player-*.md`` files
declare (e.g. ``salah_rw``). Each entry records the squad ``player_index`` (0-10,
matching ``simulator.narrator`` / ``replay.logger``), the ``agent_file`` whose body
is used as the LLM system prompt, the ``team``, and an ``expected_zone`` band used
by the coach's heuristic fallback when the LLM is unavailable.

System prompts are loaded from the agent ``.md`` files at runtime via
:func:`load_system_prompt` (YAML frontmatter stripped). If a file is missing the
caller falls back to a built-in default, so the agents never hard-fail when the
``.claude/agents`` directory is not present.
"""

from __future__ import annotations

from pathlib import Path


# Pitch bands a role should occupy (team's attacking-toward-+x frame).
ZONE_DEFENSIVE = "defensive"
ZONE_MIDFIELD = "midfield"
ZONE_ATTACKING = "attacking"

# A '---'-fenced frontmatter splits a doc into ['', frontmatter, body].
_FRONTMATTER_PARTS = 3

# Role -> expected band, for the heuristic fallback.
_ZONE_BY_ROLE: dict[str, str] = {
    "GK": ZONE_DEFENSIVE,
    "CB": ZONE_DEFENSIVE,
    "LB": ZONE_DEFENSIVE,
    "RB": ZONE_DEFENSIVE,
    "CM": ZONE_MIDFIELD,
    "LW": ZONE_ATTACKING,
    "RW": ZONE_ATTACKING,
    "ST": ZONE_ATTACKING,
}


def _player(player_id: str, role: str, team: str, player_index: int, agent_slug: str) -> dict:
    """Build one roster entry; ``expected_zone`` is derived from ``role``."""
    return {
        "player_id": player_id,
        "role": role,
        "team": team,
        "player_index": player_index,
        "agent_file": f".claude/agents/{agent_slug}.md",
        "expected_zone": _ZONE_BY_ROLE[role],
    }


# Home XI — mirrors .claude/agents/coach-home.md and the player-*.md files.
_HOME: list[dict] = [
    _player("alisson_gk", "GK", "home", 0, "player-alisson"),
    _player("gomez_cb", "CB", "home", 1, "player-gomez"),
    _player("matip_cb", "CB", "home", 2, "player-matip"),
    _player("robertson_lb", "LB", "home", 3, "player-robertson"),
    _player("alexander-arnold_rb", "RB", "home", 4, "player-alexander-arnold"),
    _player("fabinho_cm", "CM", "home", 5, "player-fabinho"),
    _player("henderson_cm", "CM", "home", 6, "player-henderson"),
    _player("wijnaldum_cm", "CM", "home", 7, "player-wijnaldum"),
    _player("sterling_lw", "LW", "home", 8, "player-sterling"),
    _player("salah_rw", "RW", "home", 9, "player-salah"),
    _player("firmino_st", "ST", "home", 10, "player-firmino"),
]

# Away XI — generic opposition; mirrors coach-away.md's role layout.
_AWAY: list[dict] = [
    _player("away_gk", "GK", "away", 0, "away-gk"),
    _player("away_cb_left", "CB", "away", 1, "away-cb-left"),
    _player("away_cb_right", "CB", "away", 2, "away-cb-right"),
    _player("away_lb", "LB", "away", 3, "away-lb"),
    _player("away_rb", "RB", "away", 4, "away-rb"),
    _player("away_cm_def", "CM", "away", 5, "away-cm-defensive"),
    _player("away_cm_b2b", "CM", "away", 6, "away-cm-box-to-box"),
    _player("away_cm_att", "CM", "away", 7, "away-cm-attacking"),
    _player("away_lw", "LW", "away", 8, "away-lw"),
    _player("away_rw", "RW", "away", 9, "away-rw"),
    _player("away_st", "ST", "away", 10, "away-st"),
]

# Flat, name-keyed roster across both teams. Each entry carries its ``team``.
PLAYER_PERSONALITIES: dict[str, dict] = {p["player_id"]: p for p in (*_HOME, *_AWAY)}

# Coach configuration: system-prompt source and the team token used to
# authenticate MCP writes. Tokens match mcp_server.server.TOKENS.
COACH_PERSONALITIES: dict[str, dict] = {
    "home": {"agent_file": ".claude/agents/coach-home.md", "token": "home-secret-abc"},
    "away": {"agent_file": ".claude/agents/coach-away.md", "token": "away-secret-xyz"},
}


def players_for_team(team: str) -> dict[str, dict]:
    """Return the ``player_id -> profile`` mapping for one team."""
    return {pid: p for pid, p in PLAYER_PERSONALITIES.items() if p["team"] == team}


def player_by_index(team: str, player_index: int) -> dict | None:
    """Return the roster entry for a squad index on ``team`` (or None)."""
    for profile in PLAYER_PERSONALITIES.values():
        if profile["team"] == team and profile["player_index"] == player_index:
            return profile
    return None


def load_system_prompt(agent_file: str) -> str | None:
    """Return the body of an agent ``.md`` file (YAML frontmatter stripped).

    Returns None if the file does not exist, so callers can fall back to a
    built-in default prompt.
    """
    path = Path(agent_file)
    if not path.is_file():
        return None
    text = path.read_text(encoding="utf-8")
    if text.startswith("---"):
        # Frontmatter is the first '---' fenced block: split into
        # ['', frontmatter, body]; the prompt is the body.
        parts = text.split("---", 2)
        if len(parts) == _FRONTMATTER_PARTS:
            return parts[2].strip()
    return text.strip()
