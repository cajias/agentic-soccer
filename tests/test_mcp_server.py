"""Tests for the agentic-soccer MCP server and its shared game state.

Covers the four required behaviors via direct calls (state + auth logic) plus
an in-memory FastMCP client test (tool registration + serialization) and a
real port-bind smoke test for the HTTP transport on :data:`PORT`.

Run with::

    uv run --no-sync pytest tests/test_mcp_server.py -v
"""

from __future__ import annotations

import asyncio
import http.client
import json
import socket
import threading
import time
from typing import TYPE_CHECKING, Any

import pytest
from fastmcp import Client
from fastmcp.exceptions import ToolError
from starlette.testclient import TestClient


if TYPE_CHECKING:
    from collections.abc import Iterator

from mcp_server import server
from mcp_server.server import (
    GAME_STATE,
    PORT,
    AuthError,
    clear_player_override,
    get_match_status,
    get_player_overrides,
    update_player_override,
)


HOME_TOKEN = "home-secret-abc"  # noqa: S105
AWAY_TOKEN = "away-secret-xyz"  # noqa: S105

SAMPLE_OVERRIDE: dict[str, Any] = {
    "target_position": [0.5, 0.0],
    "duration_ticks": 10,
    "reasoning": "push up the right wing",
}


@pytest.fixture(autouse=True)
def reset_state() -> Iterator[None]:
    """Reset the shared singleton before and after every test."""
    GAME_STATE.reset()
    yield
    GAME_STATE.reset()


# -- required behaviors ----------------------------------------------------


def test_get_match_status_returns_narrator_text() -> None:
    """get_match_status returns the narrator text set for the team."""
    GAME_STATE.set_narrator("home", "Home presses high after kickoff.")
    assert get_match_status("home", token=HOME_TOKEN) == "Home presses high after kickoff."


def test_update_player_override_rejects_wrong_team_token() -> None:
    """A write with another team's token is rejected and leaves state untouched."""
    with pytest.raises(AuthError):
        update_player_override("home", "7", SAMPLE_OVERRIDE, token=AWAY_TOKEN)
    # The rejected write must not have touched state.
    assert GAME_STATE.player_overrides["home"] == {}


def test_get_player_overrides_excludes_and_removes_expired() -> None:
    """Expired overrides are excluded from the result and deleted from state."""
    update_player_override("home", "7", SAMPLE_OVERRIDE, token=HOME_TOKEN)
    # Still active at tick 0.
    active = get_player_overrides("home", token=HOME_TOKEN)["overrides"]
    assert "7" in active

    # Advance the clock past duration_ticks -> override expires.
    GAME_STATE.set_tick(int(SAMPLE_OVERRIDE["duration_ticks"]))
    active = get_player_overrides("home", token=HOME_TOKEN)["overrides"]
    assert "7" not in active
    # Auto-removed from state, not merely filtered from the return value.
    assert "7" not in GAME_STATE.player_overrides["home"]


def test_clear_player_override_removes_override() -> None:
    """clear_player_override deletes the override and confirms the removal."""
    update_player_override("home", "9", SAMPLE_OVERRIDE, token=HOME_TOKEN)
    assert "9" in GAME_STATE.player_overrides["home"]

    result = clear_player_override("home", "9", token=HOME_TOKEN)
    assert result["cleared"] is True
    assert "9" not in GAME_STATE.player_overrides["home"]

    # Clearing an absent override is a no-op confirmation.
    again = clear_player_override("home", "9", token=HOME_TOKEN)
    assert again["cleared"] is False


# -- auth edge cases -------------------------------------------------------


def test_missing_and_invalid_tokens_rejected() -> None:
    """Missing or unrecognized tokens are rejected."""
    with pytest.raises(AuthError):
        get_match_status("home", token=None)
    with pytest.raises(AuthError):
        get_match_status("home", token="not-a-real-token")  # noqa: S106


def test_get_match_status_validates_team_token() -> None:
    """A valid token for one team cannot read another team's status."""
    with pytest.raises(AuthError):
        get_match_status("home", token=AWAY_TOKEN)


# -- through the MCP layer (registration + serialization) ------------------


def test_tools_callable_through_in_memory_client() -> None:
    """Tools are registered and callable through the MCP layer end to end."""
    GAME_STATE.set_narrator("away", "Away sits deep and counters.")

    async def go() -> None:
        async with Client(server.mcp) as client:
            names = {t.name for t in await client.list_tools()}
            assert {
                "get_match_status",
                "update_player_override",
                "get_player_overrides",
                "clear_player_override",
            } <= names

            status = await client.call_tool(
                "get_match_status", {"team": "away", "token": AWAY_TOKEN},
            )
            assert status.data == "Away sits deep and counters."

            stored = await client.call_tool(
                "update_player_override",
                {
                    "team": "away",
                    "player_id": "4",
                    "override": SAMPLE_OVERRIDE,
                    "token": AWAY_TOKEN,
                },
            )
            assert stored.data["status"] == "ok"

            overrides = await client.call_tool(
                "get_player_overrides", {"team": "away", "token": AWAY_TOKEN},
            )
            assert "4" in overrides.data["overrides"]

            # A mismatched token is rejected at the MCP layer too.
            with pytest.raises(ToolError):
                await client.call_tool(
                    "update_player_override",
                    {
                        "team": "away",
                        "player_id": "5",
                        "override": SAMPLE_OVERRIDE,
                        "token": HOME_TOKEN,
                    },
                )

    asyncio.run(go())


def _free_port() -> int:
    """Return an OS-assigned free TCP port on the loopback interface."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def test_default_port_is_8765() -> None:
    """The default bind port stays 8765 (the contract the Docker host maps)."""
    assert PORT == 8765


def test_http_transport_honors_host_and_port_env(monkeypatch: pytest.MonkeyPatch) -> None:
    """``main()`` binds the host/port from MCP_HOST/MCP_PORT env overrides.

    Uses 127.0.0.1 + a free ephemeral port so the test never collides with a
    fixed port or triggers a firewall prompt; this exercises the same env-driven
    bind path the Docker entrypoint uses with MCP_HOST=0.0.0.0.
    """
    port = _free_port()
    monkeypatch.setenv("MCP_HOST", "127.0.0.1")
    monkeypatch.setenv("MCP_PORT", str(port))

    thread = threading.Thread(target=server.main, daemon=True)
    thread.start()

    deadline = time.time() + 10
    connected = False
    while time.time() < deadline:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
            sock.settimeout(0.5)
            if sock.connect_ex(("127.0.0.1", port)) == 0:
                connected = True
                break
        time.sleep(0.2)

    assert connected, f"server did not bind to 127.0.0.1:{port} from env"

    # Hit a REST route over the REAL socket served by mcp.run (not just the
    # http_app ASGI shortcut) — this is the exact path check_milestones and the
    # coach HTTP client use, so confirm routing + auth work end to end.
    GAME_STATE.set_narrator("home", "Live over the real socket.")
    conn = http.client.HTTPConnection("127.0.0.1", port, timeout=2)
    conn.request(
        "GET",
        "/get_match_status?team=home",
        headers={"Authorization": f"Bearer {HOME_TOKEN}"},
    )
    resp = conn.getresponse()
    payload = json.loads(resp.read())
    conn.close()
    assert resp.status == 200
    assert payload["narrator"] == "Live over the real socket."


# -- REST routes (plain HTTP for requests-based clients) -------------------


def _bearer(token: str) -> dict[str, str]:
    """Build an Authorization: Bearer header dict for ``token``."""
    return {"Authorization": f"Bearer {token}"}


@pytest.fixture
def rest_client() -> Iterator[TestClient]:
    """A Starlette TestClient over the server's real ASGI app (runs lifespan)."""
    with TestClient(server.mcp.http_app()) as client:
        yield client


def test_rest_status_enforces_token_and_team(rest_client: TestClient) -> None:
    """GET status: tokenless and mismatched-token reads are rejected; match works."""
    GAME_STATE.set_narrator("home", "Home builds patiently from the back.")
    GAME_STATE.set_score([2, 1])

    # No token -> rejected (REST is no weaker than the MCP tools).
    assert rest_client.get("/get_match_status", params={"team": "home"}).status_code == 401

    # Wrong team's token -> rejected.
    bad = rest_client.get("/get_match_status", params={"team": "home"}, headers=_bearer(AWAY_TOKEN))
    assert bad.status_code == 403

    # Matching token -> 200 with narrator + score.
    ok = rest_client.get("/get_match_status", params={"team": "home"}, headers=_bearer(HOME_TOKEN))
    assert ok.status_code == 200
    assert ok.json()["narrator"] == "Home builds patiently from the back."
    assert ok.json()["score"] == [2, 1]

    # Alias path + X-Team-Token header both work.
    alias = rest_client.get("/match_status", params={"team": "home"}, headers={"X-Team-Token": HOME_TOKEN})
    assert alias.status_code == 200
    assert alias.json()["narrator"] == "Home builds patiently from the back."


def test_rest_update_override_writes_and_scopes(rest_client: TestClient) -> None:
    """POST override: matching token persists to GAME_STATE; mismatch is rejected and leaves state untouched."""
    ok = rest_client.post(
        "/update_player_override",
        json={"team": "home", "player_id": "7", "override": SAMPLE_OVERRIDE},
        headers=_bearer(HOME_TOKEN),
    )
    assert ok.status_code == 200
    assert ok.json()["status"] == "ok"
    assert "7" in GAME_STATE.player_overrides["home"]

    # Alias path also writes.
    alias = rest_client.post(
        "/player_override",
        json={"team": "home", "player_id": "8", "override": SAMPLE_OVERRIDE},
        headers={"X-Team-Token": HOME_TOKEN},
    )
    assert alias.status_code == 200
    assert "8" in GAME_STATE.player_overrides["home"]

    # Mismatched token -> rejected, state untouched.
    bad = rest_client.post(
        "/update_player_override",
        json={"team": "home", "player_id": "9", "override": SAMPLE_OVERRIDE},
        headers=_bearer(AWAY_TOKEN),
    )
    assert bad.status_code == 403
    assert "9" not in GAME_STATE.player_overrides["home"]


def test_rest_get_and_clear_overrides(rest_client: TestClient) -> None:
    """GET overrides returns active entries; POST clear removes one — both token-scoped."""
    rest_client.post(
        "/update_player_override",
        json={"team": "home", "player_id": "7", "override": SAMPLE_OVERRIDE},
        headers=_bearer(HOME_TOKEN),
    )

    listed = rest_client.get("/get_player_overrides", params={"team": "home"}, headers=_bearer(HOME_TOKEN))
    assert listed.status_code == 200
    assert "7" in listed.json()["overrides"]

    # Tokenless list is rejected.
    assert rest_client.get("/get_player_overrides", params={"team": "home"}).status_code == 401

    cleared = rest_client.post(
        "/clear_player_override",
        json={"team": "home", "player_id": "7"},
        headers=_bearer(HOME_TOKEN),
    )
    assert cleared.status_code == 200
    assert cleared.json()["cleared"] is True
    assert "7" not in GAME_STATE.player_overrides["home"]
