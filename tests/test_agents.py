"""Tests for the coach loop, player agent, and personalities."""

from __future__ import annotations

import json
from types import SimpleNamespace

from agents.coach_loop import (
    MAX_ALERTS_PER_CYCLE,
    CoachLoop,
    parse_player_behaviors,
    select_alerts,
)
from agents.personalities import squad_for
from agents.player_agent import PlayerAgent
from replay.logger import ReplayLogger


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


class _FakeClient:
    """In-memory MatchClient: serves a fixed status, records override writes."""

    def __init__(self, status: str) -> None:
        self.status = status
        self.overrides: list[tuple[str, str, dict]] = []

    def get_match_status(self, team: str) -> str:  # noqa: ARG002
        return self.status

    def update_player_override(self, team: str, player_id: str, override: dict) -> dict:
        self.overrides.append((team, player_id, override))
        return {"status": "ok"}


class _FakeAnthropic:
    """Anthropic stub whose messages.create returns a canned tool directive."""

    def __init__(self, directive: dict) -> None:
        self._directive = directive
        self.calls: list[dict] = []
        self.messages = self

    def create(self, **kwargs: object) -> SimpleNamespace:
        self.calls.append(kwargs)
        block = SimpleNamespace(type="tool_use", name="set_player_directive", input=dict(self._directive))
        return SimpleNamespace(content=[block])


def _status_with(behaviors: list[str]) -> str:
    """Build narrator-format status text from 11 ordered behavior strings."""
    roles = ["GK", "CB (left)", "CB (right)", "LB", "RB", "CM (left)", "CM (center)", "CM (right)", "LW", "RW", "ST"]
    lines = ["MATCH STATE (10:00) | Score: Home 0 - Away 0", "", "WHAT YOUR PLAYERS ARE DOING:"]
    lines += [f"- {role}: {behavior}" for role, behavior in zip(roles, behaviors, strict=True)]
    lines += ["", "GAME SITUATION:", "Midfield battle.", "", "YOUR TEAM: home"]
    return "\n".join(lines)


# Four defenders pushed into the attacking third (ids 1-4 out of position),
# everyone else in their proper band -> only 3 should be alerted (the cap).
_FOUR_DEFENDERS_UPFIELD = _status_with(
    [
        "holding position in own penalty area",  # GK ok
        "on the ball in opposition penalty area",  # CB(l) FLAG
        "pushing forward in attacking third, center",  # CB(r) FLAG
        "shifting right in attacking third, right channel",  # LB FLAG
        "closing on the ball in attacking third, center",  # RB FLAG
        "holding position in midfield, center",  # CM(l) ok
        "challenging for the ball in midfield, center",  # CM(c) ok
        "holding position in midfield, center",  # CM(r) ok
        "pushing forward in attacking third, left channel",  # LW ok
        "on the ball in attacking third, right channel",  # RW ok
        "holding position in attacking third, center",  # ST ok
    ],
)


# --- personalities --------------------------------------------------------


def test_squad_maps_all_eleven_indices():
    """Each team's squad covers indices 0-10 with matching ids."""
    for team in ("home", "away"):
        squad = squad_for(team)
        assert set(squad) == set(range(11))
        assert all(squad[i]["id"] == i for i in range(11))
    # The task's named Liverpool players are present on the home side.
    names = {p["name"] for p in squad_for("home").values()}
    assert {"Sterling", "Firmino", "Salah", "Fabinho", "Robertson"} <= names


# --- parser ---------------------------------------------------------------


def test_parse_player_behaviors_returns_ordered_lines():
    """The parser extracts all 11 behavior strings in squad-index order."""
    behaviors = parse_player_behaviors(_FOUR_DEFENDERS_UPFIELD)
    assert len(behaviors) == 11
    assert behaviors[0] == "holding position in own penalty area"
    assert behaviors[10] == "holding position in attacking third, center"


def test_parse_player_behaviors_missing_section():
    """Absent player section yields an empty list rather than raising."""
    assert parse_player_behaviors("no players section here") == []


# --- selection heuristic --------------------------------------------------


def test_select_alerts_flags_out_of_position_and_caps():
    """Out-of-position players are flagged, capped at MAX_ALERTS_PER_CYCLE."""
    behaviors = parse_player_behaviors(_FOUR_DEFENDERS_UPFIELD)
    alerts = select_alerts(squad_for("home"), behaviors)
    assert len(alerts) == MAX_ALERTS_PER_CYCLE
    # Ties (all distance 2) broken by squad index -> lowest ids first.
    assert [pid for pid, _ in alerts] == [1, 2, 3]
    assert all(isinstance(reason, str) and reason for _, reason in alerts)


def test_select_alerts_none_when_in_position():
    """A status where everyone is in their band produces no alerts."""
    in_position = _status_with(
        ["holding position in own penalty area"] * 5
        + ["holding position in midfield, center"] * 3
        + ["holding position in attacking third, center"] * 3,
    )
    assert select_alerts(squad_for("home"), parse_player_behaviors(in_position)) == []


# --- player agent ---------------------------------------------------------


def test_player_agent_override_writes_directive():
    """An 'override' directive is written to the client and reported correctly."""
    client = _FakeClient("")
    anthropic = _FakeAnthropic(
        {"decision": "override", "target_position": [0.6, -0.1], "duration_ticks": 40, "reasoning": "push up"},
    )
    agent = PlayerAgent("home", client, anthropic_client=anthropic)
    result = agent.decide(9, squad_for("home")[9], "get higher", "status text")

    assert result["player_decision"] == "override"
    assert result["override_written"] is True
    assert result["target_position"] == [0.6, -0.1]
    assert result["duration_ticks"] == 40
    assert result["coach_reasoning"] == "get higher"
    assert isinstance(result["response_time_s"], float)
    assert set(result) == _ALERT_KEYS
    # The directive reached the client with the engine override schema.
    assert len(client.overrides) == 1
    team, pid, override = client.overrides[0]
    assert (team, pid) == ("home", "9")
    assert override["target_position"] == [0.6, -0.1]
    assert override["duration_ticks"] == 40
    assert "reasoning" in override


def test_player_agent_hold_writes_nothing():
    """A 'hold' directive writes no override and reports a clean hold."""
    client = _FakeClient("")
    agent = PlayerAgent("home", client, anthropic_client=_FakeAnthropic({"decision": "hold", "reasoning": "fine"}))
    result = agent.decide(5, squad_for("home")[5], "stay deep", "status text")

    assert result["player_decision"] == "hold"
    assert result["override_written"] is False
    assert result["target_position"] is None
    assert result["duration_ticks"] == 0
    assert client.overrides == []


def test_player_agent_override_without_target_downgrades_to_hold():
    """An 'override' lacking a target position is treated as a safe hold."""
    client = _FakeClient("")
    agent = PlayerAgent("home", client, anthropic_client=_FakeAnthropic({"decision": "override", "reasoning": "?"}))
    result = agent.decide(8, squad_for("home")[8], "push up", "status text")

    assert result["player_decision"] == "hold"
    assert result["override_written"] is False
    assert client.overrides == []


# --- coach loop (end-to-end completion check) -----------------------------


def test_run_cycle_alerts_caps_and_logs_coach_cycle(tmp_path):
    """One cycle alerts <=3 players, writes overrides, and logs all 7 alert keys."""
    path = tmp_path / "match" / "replay.jsonl"
    logger = ReplayLogger(str(path))
    # A tick line must exist for log_coach_cycle to embed into.
    obs = {"left_team": [[0.0, 0.0]] * 11, "right_team": [[0.0, 0.0]] * 11, "ball": [0.0, 0.0]}
    logger.log_tick(7, 12.6, obs, [0, 0])

    client = _FakeClient(_FOUR_DEFENDERS_UPFIELD)
    anthropic = _FakeAnthropic(
        {"decision": "override", "target_position": [-0.6, 0.0], "duration_ticks": 30, "reasoning": "recover"},
    )
    coach = CoachLoop("home", client, logger, anthropic_client=anthropic)

    alerts = coach.run_cycle(7)
    logger.close()

    # Capped at 3, lowest ids first, all overrides written.
    assert len(alerts) == MAX_ALERTS_PER_CYCLE
    assert [a["player_id"] for a in alerts] == [1, 2, 3]
    assert all(a["override_written"] for a in alerts)
    assert len(client.overrides) == 3

    # The coach_cycle block was embedded into tick 7's line with all 7 keys.
    record = json.loads(path.read_text().splitlines()[-1])
    assert record["tick"] == 7
    cycle = record["coach_cycle"][0]
    assert cycle["team"] == "home"
    assert len(cycle["alerts"]) == 3
    for alert in cycle["alerts"]:
        assert set(alert) == _ALERT_KEYS


def test_drive_runs_fixed_cycles_without_sleeping(tmp_path):
    """drive() runs exactly max_cycles passes using an injected no-op sleep."""
    path = tmp_path / "match" / "replay.jsonl"
    logger = ReplayLogger(str(path))
    obs = {"left_team": [[0.0, 0.0]] * 11, "right_team": [[0.0, 0.0]] * 11, "ball": [0.0, 0.0]}
    for tick in range(3):
        logger.log_tick(tick, float(tick), obs, [0, 0])

    client = _FakeClient(_status_with(["holding position in midfield, center"] * 11))
    coach = CoachLoop("home", client, logger, anthropic_client=_FakeAnthropic({"decision": "hold", "reasoning": "ok"}))

    sleeps: list[float] = []
    coach.drive(lambda: 2, max_cycles=3, sleep=sleeps.append)
    logger.close()

    # 3 cycles, sleeping only between them (not after the last).
    assert len(sleeps) == 2
