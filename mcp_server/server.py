"""FastMCP server exposing the soccer simulator to agent coach clients.

The same four operations are reachable two ways, both backed by the single
:data:`GAME_STATE` singleton (which the in-process engine also reads/writes):

* **MCP tools** (JSON-RPC at ``/mcp``) — for MCP-native clients.
* **REST routes** (plain HTTP, registered via ``@mcp.custom_route``) — for
  simple ``requests``-based clients (the coach HTTP client, milestone checks).

Both surfaces enforce identical token auth: a team token supplied via the
``Authorization: Bearer`` or ``X-Team-Token`` header (or, for the MCP tools, an
explicit ``token`` argument; for REST, a ``token`` query param). A token may
only act on its own team's state — mismatched and missing tokens are rejected
everywhere, which is the security boundary when the server binds ``0.0.0.0``.

Run with::

    uv run --no-sync python -m mcp_server.server
"""

from __future__ import annotations

import os
from typing import TYPE_CHECKING

from fastmcp import FastMCP
from fastmcp.server.dependencies import get_http_headers
from starlette.responses import JSONResponse, Response

from mcp_server.game_state import GameState


if TYPE_CHECKING:
    from starlette.requests import Request


# Token -> team. The simulator/launcher hands each coach exactly one token.
TOKENS: dict[str, str] = {
    "home-secret-abc": "home",
    "away-secret-xyz": "away",
}

PORT = 8765

# Shared state. The simulator writes here; tool handlers read/write here.
GAME_STATE = GameState()

mcp: FastMCP = FastMCP("agentic-soccer")

# Optional replay logger shared with the in-process engine. When the launcher
# (e.g. docker_entry) calls set_replay_logger, every override write also records
# a coach_cycle block into the replay — this is how a coach (a Claude CLI
# session, or a scripted call) satisfies the "coach_cycle in replay" milestone
# with no LLM/SDK involved.
_REPLAY_LOGGER: object | None = None


def set_replay_logger(logger: object) -> None:
    """Share the engine's ReplayLogger so override writes log coach cycles."""
    global _REPLAY_LOGGER  # noqa: PLW0603 - module-level singleton wiring
    _REPLAY_LOGGER = logger


def _record_coach_cycle(team: str, player_id: str, override: dict) -> None:
    """Record a coach_cycle block in the replay for an override write.

    Best-effort: a logging failure (or no logger wired) must never break the
    override itself.
    """
    if _REPLAY_LOGGER is None:
        return
    alert = {
        "player_id": player_id,
        "coach_reasoning": override.get("reasoning", ""),
        "player_decision": "override",
        "override_written": True,
        "target_position": override.get("target_position"),
        "duration_ticks": override.get("duration_ticks", 0),
        "response_time_s": 0.0,
    }
    try:
        _REPLAY_LOGGER.log_coach_cycle(GAME_STATE.tick, team, [alert])  # type: ignore[attr-defined]
    except Exception:  # noqa: BLE001 - logging must never break an override write
        return


class AuthError(PermissionError):
    """Raised when a request's token is missing, invalid, or not for the team.

    Carries the HTTP ``status`` the REST routes should return (401 for a
    missing/invalid token, 403 for a valid token used against the wrong team).
    """

    def __init__(self, message: str, status: int = 401) -> None:
        """Store ``message`` and the HTTP ``status`` to surface over REST."""
        super().__init__(message)
        self.status = status


def _parse_bearer(authorization: str | None, x_team_token: str | None) -> str | None:
    """Extract a token from an ``Authorization``/``X-Team-Token`` header pair.

    Accepts ``Authorization: Bearer <token>`` (or a bare token) and falls back
    to the ``X-Team-Token`` header.
    """
    if authorization:
        if authorization.lower().startswith("bearer "):
            return authorization[7:].strip()
        return authorization.strip()
    return x_team_token


def _token_from_headers() -> str | None:
    """Pull a team token from the MCP request context, if one exists.

    Returns None outside an HTTP request (e.g. tests or in-memory clients),
    where the explicit ``token`` argument is used instead.
    """
    try:
        headers = get_http_headers()
    except RuntimeError:
        return None
    return _parse_bearer(
        headers.get("authorization") or headers.get("Authorization"),
        headers.get("x-team-token") or headers.get("X-Team-Token"),
    )


def _token_from_request(request: Request) -> str | None:
    """Pull a team token directly off a Starlette request (REST routes).

    Reads it from headers (``Authorization``/``X-Team-Token``) or a ``token``
    query parameter. Read straight off ``request`` rather than via the MCP
    request context, which is not populated inside ``@mcp.custom_route``.
    """
    return _parse_bearer(
        request.headers.get("authorization"),
        request.headers.get("x-team-token"),
    ) or request.query_params.get("token")


def _require_team(team: str, token: str | None) -> str:
    """Validate that ``token`` authorizes acting on ``team``.

    Returns the resolved team on success; raises :class:`AuthError` otherwise.
    """
    token = token or _token_from_headers()
    if not token:
        msg = "missing team token"
        raise AuthError(msg, status=401)
    resolved = TOKENS.get(token)
    if resolved is None:
        msg = "invalid team token"
        raise AuthError(msg, status=401)
    if resolved != team:
        msg = f"token authorizes team {resolved!r}, not {team!r}"
        raise AuthError(msg, status=403)
    return resolved


def get_match_status(team: str, token: str | None = None) -> str:
    """Return the current narrator text for ``team``.

    Requires a token matching ``team``.
    """
    _require_team(team, token)
    return GAME_STATE.get_narrator(team)


def update_player_override(
    team: str,
    player_id: str,
    override: dict,
    token: str | None = None,
) -> dict:
    """Store a behavior override for one player on ``team``.

    ``override`` schema::

        {"target_position": [x, y], "duration_ticks": int, "reasoning": str}

    Requires a token matching ``team`` — mismatched writes are rejected.
    Returns the stored record including ``created_tick`` and ``expires_at_tick``.
    """
    _require_team(team, token)
    record = GAME_STATE.set_override(team, player_id, override)
    _record_coach_cycle(team, player_id, override)
    return {"status": "ok", "team": team, "player_id": player_id, "record": record}


def get_player_overrides(team: str, token: str | None = None) -> dict:
    """Return all active (non-expired) overrides for ``team``.

    Expired overrides are auto-removed from state as a side effect. Requires a
    token matching ``team``.
    """
    _require_team(team, token)
    active = GAME_STATE.get_active_overrides(team)
    return {"team": team, "tick": GAME_STATE.tick, "overrides": active}


def clear_player_override(team: str, player_id: str, token: str | None = None) -> dict:
    """Remove the active override for one player on ``team``.

    Requires a token matching ``team``. Returns confirmation including whether
    an override was actually present.
    """
    _require_team(team, token)
    cleared = GAME_STATE.clear_override(team, player_id)
    return {
        "status": "ok",
        "team": team,
        "player_id": player_id,
        "cleared": cleared,
    }


# Register the plain functions as MCP tools. Keeping the functions defined
# separately (rather than decorating in place) lets tests import and call them
# directly while still exposing them over the MCP layer.
mcp.tool(get_match_status)
mcp.tool(update_player_override)
mcp.tool(get_player_overrides)
mcp.tool(clear_player_override)


# --------------------------------------------------------------------------
# REST routes (plain HTTP, for ``requests``-based clients) served alongside the
# JSON-RPC ``/mcp`` endpoint. They share GAME_STATE and the SAME strict token
# auth as the MCP tools above — a missing or mismatched token is rejected.
# --------------------------------------------------------------------------


def _error_response(exc: Exception) -> Response:
    """Turn an auth/validation error into a JSON error response.

    :class:`AuthError` carries its own HTTP ``status``; other errors (e.g. an
    unknown team) map to 400.
    """
    return JSONResponse({"error": str(exc)}, status_code=getattr(exc, "status", 400))


async def _parse_body(request: Request) -> dict:
    """Parse a JSON object request body, raising ValueError if it is not one."""
    body = await request.json()
    if not isinstance(body, dict):
        # Deliberately ValueError (not TypeError) so callers catch malformed
        # JSON and wrong-shape bodies through one branch -> a single 400.
        msg = "request body must be a JSON object"
        raise ValueError(msg)  # noqa: TRY004
    return body


async def _route_match_status(request: Request) -> Response:
    """GET ``/get_match_status`` (alias ``/match_status``) -> narrator + score."""
    team = request.query_params.get("team", "")
    try:
        _require_team(team, _token_from_request(request))
    except (AuthError, ValueError) as exc:
        return _error_response(exc)
    return JSONResponse(
        {
            "narrator": GAME_STATE.get_narrator(team),
            "tick": GAME_STATE.tick,
            "score": GAME_STATE.get_score(),
            "team": team,
        },
    )


async def _route_get_overrides(request: Request) -> Response:
    """GET ``/get_player_overrides`` (alias ``/player_overrides``) -> active overrides."""
    team = request.query_params.get("team", "")
    try:
        _require_team(team, _token_from_request(request))
    except (AuthError, ValueError) as exc:
        return _error_response(exc)
    return JSONResponse(
        {
            "team": team,
            "tick": GAME_STATE.tick,
            "overrides": GAME_STATE.get_active_overrides(team),
        },
    )


async def _route_update_override(request: Request) -> Response:
    """POST ``/update_player_override`` (alias ``/player_override``) -> store override."""
    try:
        body = await _parse_body(request)
    except ValueError:
        return JSONResponse({"error": "invalid JSON body"}, status_code=400)
    team = body.get("team", "")
    player_id = str(body.get("player_id", ""))
    override = body.get("override") or {}
    try:
        _require_team(team, _token_from_request(request))
    except (AuthError, ValueError) as exc:
        return _error_response(exc)
    record = GAME_STATE.set_override(team, player_id, override)
    _record_coach_cycle(team, player_id, override)
    return JSONResponse(
        {"status": "ok", "team": team, "player_id": player_id, "record": record},
    )


async def _route_clear_override(request: Request) -> Response:
    """POST ``/clear_player_override`` -> remove a player's override."""
    try:
        body = await _parse_body(request)
    except ValueError:
        return JSONResponse({"error": "invalid JSON body"}, status_code=400)
    team = body.get("team", "")
    player_id = str(body.get("player_id", ""))
    try:
        _require_team(team, _token_from_request(request))
    except (AuthError, ValueError) as exc:
        return _error_response(exc)
    cleared = GAME_STATE.clear_override(team, player_id)
    return JSONResponse(
        {"status": "ok", "team": team, "player_id": player_id, "cleared": cleared},
    )


# Both the lead's long names and the coach HTTP client's short names are served
# so neither client needs to change. (Flagged for reconciliation.)
mcp.custom_route("/get_match_status", methods=["GET"])(_route_match_status)
mcp.custom_route("/match_status", methods=["GET"])(_route_match_status)
mcp.custom_route("/get_player_overrides", methods=["GET"])(_route_get_overrides)
mcp.custom_route("/player_overrides", methods=["GET"])(_route_get_overrides)
mcp.custom_route("/update_player_override", methods=["POST"])(_route_update_override)
mcp.custom_route("/player_override", methods=["POST"])(_route_update_override)
mcp.custom_route("/clear_player_override", methods=["POST"])(_route_clear_override)


def main() -> None:
    """Start the MCP server over HTTP.

    Binds to ``127.0.0.1`` on :data:`PORT` by default (safe for local dev).
    Override with ``MCP_HOST`` (e.g. ``0.0.0.0`` to bind externally — needed in
    a container where the port must be reachable from the host) and ``MCP_PORT``.
    Token auth applies regardless of bind address and is the security boundary
    when bound to ``0.0.0.0``.
    """
    host = os.environ.get("MCP_HOST", "127.0.0.1")
    port = int(os.environ.get("MCP_PORT", str(PORT)))
    mcp.run(transport="http", host=host, port=port)


if __name__ == "__main__":
    main()
