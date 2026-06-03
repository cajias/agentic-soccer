"""Tests for the coach loop and player agent (mocked Anthropic + mocked HTTP)."""

from __future__ import annotations

import json
from types import SimpleNamespace
from typing import TYPE_CHECKING, Any

from agentic_soccer.agents.coach_loop import (
    MAX_ALERTS_PER_CYCLE,
    decide_alerts,
    heuristic_alerts,
    parse_player_behaviors,
    run_coach_cycle,
)
from agentic_soccer.agents.personalities import PLAYER_PERSONALITIES, players_for_team
from agentic_soccer.agents.player_agent import MCPHttpClient, player_agent
from agentic_soccer.replay.logger import ReplayLogger


if TYPE_CHECKING:
    from pathlib import Path


_ALERT_KEYS = {
    "player_id",
    "coach_reasoning",
    "player_decision",
    "override_written",
    "target_position",
    "duration_ticks",
    "response_time_s",
}


# --- fakes ----------------------------------------------------------------


class _FakeResponse:
    def __init__(self, payload: dict[str, Any]) -> None:
        self._payload = payload

    def raise_for_status(self) -> None:
        return None

    def json(self) -> dict[str, Any]:
        return self._payload


class _FakeSession:
    """Stand-in for requests.Session: serves fixed payloads, records calls."""

    def __init__(
        self,
        get_payload: dict[str, Any] | None = None,
        post_payload: dict[str, Any] | None = None,
    ) -> None:
        self._get_payload = get_payload or {"narrator": "report"}
        self._post_payload = post_payload or {"status": "ok"}
        self.get_calls: list[dict[str, Any]] = []
        self.post_calls: list[dict[str, Any]] = []

    # headers/timeout mirror requests.Session but are unused by the fake.
    def get(
        self,
        url: str,
        params: dict[str, Any] | None = None,
        headers: dict[str, str] | None = None,  # noqa: ARG002
        timeout: float | None = None,  # noqa: ARG002
    ) -> _FakeResponse:
        self.get_calls.append({"url": url, "params": params})
        return _FakeResponse(self._get_payload)

    def post(
        self,
        url: str,
        json: dict[str, Any] | None = None,
        headers: dict[str, str] | None = None,  # noqa: ARG002
        timeout: float | None = None,  # noqa: ARG002
    ) -> _FakeResponse:
        self.post_calls.append({"url": url, "json": json})
        return _FakeResponse(self._post_payload)


class _FakeAnthropic:
    """Anthropic stub: messages.create returns a response with fixed blocks."""

    def __init__(self, blocks: list[SimpleNamespace]) -> None:
        self._blocks = blocks
        self.calls: list[dict[str, Any]] = []
        self.messages = self

    def create(self, **kwargs: object) -> SimpleNamespace:
        self.calls.append(kwargs)
        return SimpleNamespace(content=self._blocks)


def _text(s: str) -> SimpleNamespace:
    return SimpleNamespace(type="text", text=s)


def _tool(directive: dict[str, Any]) -> SimpleNamespace:
    return SimpleNamespace(type="tool_use", name="set_player_directive", input=directive)


def _status_with(behaviors: list[str]) -> str:
    roles = ["GK", "CB (left)", "CB (right)", "LB", "RB", "CM (left)", "CM (center)", "CM (right)", "LW", "RW", "ST"]
    lines = ["MATCH STATE (10:00) | Score: Home 0 - Away 0", "", "WHAT YOUR PLAYERS ARE DOING:"]
    lines += [f"- {role}: {behavior}" for role, behavior in zip(roles, behaviors, strict=True)]
    lines += ["", "GAME SITUATION:", "Midfield battle.", "", "YOUR TEAM: home"]
    return "\n".join(lines)


# Salah (RW, idx 9) stranded in the defensive third; everyone else in band.
_SALAH_DEEP = _status_with(
    ["holding position in own penalty area"] * 5
    + ["holding position in midfield, center"] * 3
    + ["holding position in attacking third, left channel",  # LW ok
       "holding position in own penalty area",  # RW (salah) FLAG
       "holding position in attacking third, center"],  # ST ok
)


# --- coach LLM parsing ----------------------------------------------------


def test_decide_alerts_parses_structured_json() -> None:
    """The coach's JSON reply is parsed into validated alert dicts."""
    reply = '{"alerts": [{"player_id": "salah_rw", "situation": "track back", "reasoning": "caught high"}]}'
    coach = _FakeAnthropic([_text(reply)])
    alerts = decide_alerts("home", "report", anthropic_client=coach)
    assert alerts == [{"player_id": "salah_rw", "situation": "track back", "reasoning": "caught high"}]


def test_decide_alerts_drops_unknown_ids_and_caps() -> None:
    """Unknown player_ids are dropped and the list is capped at 3."""
    entries = [
        {"player_id": "salah_rw", "situation": "a", "reasoning": "r"},
        {"player_id": "sterling_lw", "situation": "b", "reasoning": "r"},
        {"player_id": "not_a_player", "situation": "c", "reasoning": "r"},
        {"player_id": "firmino_st", "situation": "d", "reasoning": "r"},
        {"player_id": "fabinho_cm", "situation": "e", "reasoning": "r"},
    ]
    coach = _FakeAnthropic([_text(json.dumps({"alerts": entries}))])
    alerts = decide_alerts("home", "report", anthropic_client=coach)
    assert len(alerts) == MAX_ALERTS_PER_CYCLE
    assert "not_a_player" not in {a["player_id"] for a in alerts}


def test_decide_alerts_falls_back_to_heuristic_on_bad_json() -> None:
    """An unparseable coach reply degrades to the zone-based heuristic."""
    coach = _FakeAnthropic([_text("sorry, no json here")])
    alerts = decide_alerts("home", _SALAH_DEEP, anthropic_client=coach)
    assert any(a["player_id"] == "salah_rw" for a in alerts)


def test_heuristic_alerts_flags_out_of_position() -> None:
    """The heuristic flags a forward stranded in the defensive third."""
    alerts = heuristic_alerts("home", _SALAH_DEEP)
    assert [a["player_id"] for a in alerts] == ["salah_rw"]


def test_parse_player_behaviors_orders_eleven() -> None:
    """The narrator parser returns 11 behaviors in squad order."""
    behaviors = parse_player_behaviors(_SALAH_DEEP)
    assert len(behaviors) == 11
    assert "own penalty area" in behaviors[9]


# --- player agent (mocked Anthropic + mocked HTTP) ------------------------


def test_player_agent_override_posts_to_mcp() -> None:
    """action='override' POSTs the directive and reports it written."""
    session = _FakeSession()
    directive = {"action": "override", "target_position": [0.6, -0.1], "duration_ticks": 120, "reasoning": "stay wide"}
    result = player_agent(
        "salah_rw", "hold your width", "home", "http://sim:8765",
        anthropic_client=_FakeAnthropic([_tool(directive)]), session=session,
    )
    assert result["player_decision"] == "override"
    assert result["override_written"] is True
    assert result["target_position"] == [0.6, -0.1]
    assert result["duration_ticks"] == 120
    assert set(result) == _ALERT_KEYS
    assert len(session.post_calls) == 1
    posted = session.post_calls[0]["json"]
    assert posted["player_id"] == "salah_rw"
    assert posted["override"]["target_position"] == [0.6, -0.1]
    assert posted["override"]["duration_ticks"] == 120


def test_player_agent_hold_does_not_post() -> None:
    """action='hold' writes no override (no HTTP POST)."""
    session = _FakeSession()
    result = player_agent(
        "fabinho_cm", "stay deep", "home",
        anthropic_client=_FakeAnthropic([_tool({"action": "hold", "reasoning": "screening fine"})]), session=session,
    )
    assert result["player_decision"] == "hold"
    assert result["override_written"] is False
    assert result["target_position"] is None
    assert session.post_calls == []


def test_player_agent_override_without_target_downgrades_to_hold() -> None:
    """An override lacking a target position is treated as a safe hold."""
    session = _FakeSession()
    result = player_agent(
        "sterling_lw", "push up", "home",
        anthropic_client=_FakeAnthropic([_tool({"action": "override", "reasoning": "?"})]), session=session,
    )
    assert result["player_decision"] == "hold"
    assert session.post_calls == []


# --- coach cycle (end-to-end completion check) ----------------------------


class _FakeMatchClient:
    def __init__(self, status: str) -> None:
        self.status = status

    def get_match_status(self, team: str) -> str:  # noqa: ARG002
        return self.status

    # Structural MatchClient stub: args unused by the fixed-status fake.
    def update_player_override(
        self,
        team: str,  # noqa: ARG002
        player_id: str,  # noqa: ARG002
        override: dict[str, Any],  # noqa: ARG002
    ) -> dict[str, Any]:
        return {"status": "ok"}


# Matches the player_agent spawn signature; several args are unused here.
def _fake_spawn(
    player_id: str,
    situation: str,
    team: str,  # noqa: ARG001
    mcp_base_url: str,  # noqa: ARG001
    *,
    anthropic_client: object = None,  # noqa: ARG001
    session: object = None,  # noqa: ARG001
) -> dict[str, Any]:
    return {
        "player_id": player_id,
        "coach_reasoning": situation,
        "player_decision": "override",
        "override_written": True,
        "target_position": [0.0, 0.0],
        "duration_ticks": 150,
        "response_time_s": 0.01,
    }


def test_run_coach_cycle_caps_and_logs_coach_cycle(tmp_path: Path) -> None:
    """One cycle alerts <=3, spawns players, and logs all 7 alert keys at the tick."""
    path = tmp_path / "match" / "replay.jsonl"
    logger = ReplayLogger(str(path))
    obs = {"left_team": [[0.0, 0.0]] * 11, "right_team": [[0.0, 0.0]] * 11, "ball": [0.0, 0.0]}
    logger.log_tick(7, 12.6, obs, [0, 0])

    entries = [
        {"player_id": pid, "situation": "fix it", "reasoning": "r"}
        for pid in ("salah_rw", "sterling_lw", "firmino_st", "fabinho_cm")
    ]
    coach = _FakeAnthropic([_text(json.dumps({"alerts": entries}))])

    alerts = run_coach_cycle(
        "home", 7,
        client=_FakeMatchClient("report"),
        replay_logger=logger,
        anthropic_client=coach,
        spawn=_fake_spawn,
    )
    logger.close()

    assert len(alerts) == MAX_ALERTS_PER_CYCLE
    record = json.loads(path.read_text().splitlines()[-1])
    assert record["tick"] == 7
    cycle = record["coach_cycle"][0]
    assert cycle["team"] == "home"
    assert len(cycle["alerts"]) == 3
    for alert in cycle["alerts"]:
        assert set(alert) == _ALERT_KEYS


def test_run_coach_cycle_no_alerts_logs_nothing(tmp_path: Path) -> None:
    """A cycle that produces no alerts writes no coach_cycle block."""
    path = tmp_path / "match" / "replay.jsonl"
    logger = ReplayLogger(str(path))
    obs = {"left_team": [[0.0, 0.0]] * 11, "right_team": [[0.0, 0.0]] * 11, "ball": [0.0, 0.0]}
    logger.log_tick(0, 0.0, obs, [0, 0])

    coach = _FakeAnthropic([_text('{"alerts": []}')])
    alerts = run_coach_cycle("home", 0, client=_FakeMatchClient("report"), replay_logger=logger, anthropic_client=coach)
    logger.close()

    assert alerts == []
    assert "coach_cycle" not in json.loads(path.read_text().splitlines()[0])


def test_roster_covers_both_teams() -> None:
    """Each team has 11 players keyed by the {name}_{role} convention."""
    assert len(players_for_team("home")) == 11
    assert len(players_for_team("away")) == 11
    assert PLAYER_PERSONALITIES["salah_rw"]["player_index"] == 9


def test_mcp_http_client_uses_agreed_routes() -> None:
    """The HTTP client hits the REST routes agreed with mcp-agent and unwraps narrator."""
    session = _FakeSession(get_payload={"narrator": "match report"})
    client = MCPHttpClient("http://sim:8765", "home-secret-abc", session=session)

    assert client.get_match_status("home") == "match report"
    assert session.get_calls[0]["url"] == "http://sim:8765/get_match_status"
    assert session.get_calls[0]["params"] == {"team": "home"}

    client.update_player_override("home", "salah_rw", {"target_position": [0.0, 0.0], "duration_ticks": 150})
    assert session.post_calls[0]["url"] == "http://sim:8765/update_player_override"
    assert session.post_calls[0]["json"] == {
        "team": "home",
        "player_id": "salah_rw",
        "override": {"target_position": [0.0, 0.0], "duration_ticks": 150},
    }
