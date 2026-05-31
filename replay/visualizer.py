"""16-bit style pygame replayer for agentic-soccer matches.

Reads a JSONL replay produced by :mod:`replay.logger` and plays it back as a
pixel-art football match: a green pitch with white markings, coloured player
squares (home red, away blue), a white ball, and a score/clock HUD. When a tick
carries a ``coach_cycle`` block the replayer flashes a banner, draws each
alerted player's decision near their square, and outlines overridden players in
yellow.

Data loading (:func:`load_replay` and the ``parse_*`` helpers) is deliberately
independent of pygame so it can be unit-tested without a display.

Replay line schema (see :mod:`replay.logger`)::

    {"tick": int, "t": float,
     "players": [{"id": int, "team": "home"|"away", "role": str,
                  "x": float, "y": float}, ...],
     "ball": {"x": float, "y": float},
     "score": [home, away],
     "coach_cycle": [{"tick": int, "team": "home"|"away",
                      "alerts": [{"player_id": ..., "coach_reasoning": str,
                                  "player_decision": str,
                                  "override_written": bool}, ...]}, ...]}

``coach_cycle`` is optional. The logger appends it as a *list* of blocks; a bare
dict (older single-block form) is also accepted.

Run::

    uv run python -m replay.visualizer [match/replay.jsonl]
"""

from __future__ import annotations

import json
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pygame


# --- Pitch coordinate system -------------------------------------------------
# gfootball world coordinates: x in [-1, 1] (home goal at -1, away goal at +1),
# y in [-Y_EXTENT, Y_EXTENT].
X_EXTENT = 1.0
Y_EXTENT = 0.42

# --- Window / pitch geometry (pixels) ---------------------------------------
WIDTH = 900
HEIGHT = 600
MARGIN_X = 40
MARGIN_TOP = 64  # leaves room for the score/clock HUD
MARGIN_BOTTOM = 32
PITCH_LEFT = MARGIN_X
PITCH_TOP = MARGIN_TOP
PITCH_W = WIDTH - 2 * MARGIN_X
PITCH_H = HEIGHT - MARGIN_TOP - MARGIN_BOTTOM

# --- Colours (16-bit palette) -----------------------------------------------
PITCH_GREEN = (45, 90, 27)  # #2d5a1b
PITCH_STRIPE = (52, 102, 31)
LINE_WHITE = (235, 235, 235)
HOME_RED = (204, 34, 34)  # #cc2222
AWAY_BLUE = (34, 68, 204)  # #2244cc
BALL_WHITE = (245, 245, 245)
OVERRIDE_YELLOW = (245, 224, 66)
TEXT_WHITE = (240, 240, 240)
TEXT_DARK = (16, 16, 16)
BUBBLE_BG = (250, 250, 235)

# --- Player / ball sizes (pixels) -------------------------------------------
PLAYER_SIZE = 14
BALL_RADIUS = 6

# --- Playback ----------------------------------------------------------------
FPS = 30
SPEEDS = (0.5, 1.0, 2.0, 4.0)
DEFAULT_SPEED_IDX = 1
SCRUB_STEP = 5
BANNER_FRAMES = 90  # 3 seconds at 30 fps

# Number of elements expected in an (x, y) / (home, away) pair.
_PAIR_LEN = 2

_FONT_PATH = Path(__file__).parent / "assets" / "PressStart2P.ttf"


@dataclass(frozen=True)
class Player:
    """A single player's state within one replay frame."""

    id: int
    team: str
    role: str
    x: float
    y: float


@dataclass(frozen=True)
class Alert:
    """One coach alert about a player within a coach cycle."""

    player_id: str
    coach_reasoning: str
    player_decision: str
    override_written: bool


@dataclass(frozen=True)
class CoachCycle:
    """A coach decision cycle for one team, carrying its player alerts."""

    team: str
    alerts: list[Alert]


@dataclass(frozen=True)
class Frame:
    """A fully parsed replay frame (one logged tick)."""

    tick: int
    t: float
    players: list[Player]
    ball_x: float
    ball_y: float
    score: tuple[int, int]
    coach_cycles: list[CoachCycle]


def _parse_player(data: dict[str, Any]) -> Player:
    """Build a :class:`Player` from a raw player record."""
    return Player(
        id=int(data.get("id", -1)),
        team=str(data.get("team", "")),
        role=str(data.get("role", "")),
        x=float(data.get("x", 0.0)),
        y=float(data.get("y", 0.0)),
    )


def _parse_alert(data: dict[str, Any]) -> Alert:
    """Build an :class:`Alert` from a raw alert record."""
    return Alert(
        player_id=str(data.get("player_id", "")),
        coach_reasoning=str(data.get("coach_reasoning", "")),
        player_decision=str(data.get("player_decision", "")),
        override_written=bool(data.get("override_written", False)),
    )


def _parse_coach_cycles(raw: object) -> list[CoachCycle]:
    """Normalise a raw ``coach_cycle`` value into a list of :class:`CoachCycle`.

    Accepts ``None``, a single block dict, or a list of block dicts (the form
    written by :mod:`replay.logger`).

    Args:
        raw: The raw ``coach_cycle`` value from a replay line, if any.

    Returns:
        Parsed coach cycles; empty when ``raw`` is missing or empty.
    """
    if not raw:
        return []
    blocks = raw if isinstance(raw, list) else [raw]
    cycles: list[CoachCycle] = []
    for block in blocks:
        if not isinstance(block, dict):
            continue
        alerts = [_parse_alert(a) for a in block.get("alerts", []) if isinstance(a, dict)]
        cycles.append(CoachCycle(team=str(block.get("team", "")), alerts=alerts))
    return cycles


def _parse_ball(raw: object) -> tuple[float, float]:
    """Extract ball ``(x, y)`` from a dict ``{"x","y"}`` or an ``[x, y]`` list."""
    if isinstance(raw, dict):
        return float(raw.get("x", 0.0)), float(raw.get("y", 0.0))
    if isinstance(raw, (list, tuple)) and len(raw) >= _PAIR_LEN:
        return float(raw[0]), float(raw[1])
    return 0.0, 0.0


def parse_frame(data: dict[str, Any]) -> Frame:
    """Parse one replay line (already JSON-decoded) into a :class:`Frame`.

    Args:
        data: A decoded replay record.

    Returns:
        The parsed frame.
    """
    ball_x, ball_y = _parse_ball(data.get("ball"))
    score_raw = data.get("score", [0, 0])
    score = (int(score_raw[0]), int(score_raw[1])) if len(score_raw) >= _PAIR_LEN else (0, 0)
    return Frame(
        tick=int(data.get("tick", 0)),
        t=float(data.get("t", 0.0)),
        players=[_parse_player(p) for p in data.get("players", []) if isinstance(p, dict)],
        ball_x=ball_x,
        ball_y=ball_y,
        score=score,
        coach_cycles=_parse_coach_cycles(data.get("coach_cycle")),
    )


def load_replay(path: str | Path) -> list[Frame]:
    """Load and parse every frame from a JSONL replay file.

    Blank lines are skipped. Each remaining line must be a JSON object.

    Args:
        path: Path to the ``.jsonl`` replay file.

    Returns:
        The parsed frames in file order.

    Raises:
        FileNotFoundError: If ``path`` does not exist.
    """
    text = Path(path).read_text(encoding="utf-8")
    frames: list[Frame] = []
    for line in text.splitlines():
        stripped = line.strip()
        if not stripped:
            continue
        frames.append(parse_frame(json.loads(stripped)))
    return frames


def _to_screen(x: float, y: float) -> tuple[int, int]:
    """Map world ``(x, y)`` to integer screen pixel coordinates."""
    sx = PITCH_LEFT + (x + X_EXTENT) / (2 * X_EXTENT) * PITCH_W
    sy = PITCH_TOP + (y + Y_EXTENT) / (2 * Y_EXTENT) * PITCH_H
    return int(sx), int(sy)


def _player_label(player: Player) -> str:
    """Return the jersey label drawn on a player's square (the squad index)."""
    return str(player.id)


def _alert_matches_player(alert: Alert, player: Player) -> bool:
    """Best-effort match between a coach alert and a player on the pitch.

    The logger may identify players by squad index (``"5"``), by role, or by a
    descriptive id like ``"sterling_lw"``. Match on any of those.
    """
    pid = alert.player_id.lower()
    if not pid:
        return False
    role = player.role.lower().replace(" ", "")
    return pid == str(player.id) or pid == role or bool(role and role in pid)


@dataclass
class _Playback:
    """Mutable playback state for the replay loop."""

    index: int = 0
    paused: bool = False
    speed_idx: int = DEFAULT_SPEED_IDX
    accum: float = 0.0
    banner_left: int = 0
    banner_cycles: list[CoachCycle] = field(default_factory=list)


def _load_font(size: int) -> pygame.font.Font:
    """Load the bundled Press Start 2P font at ``size``, or a monospace fallback."""
    if _FONT_PATH.exists():
        return pygame.font.Font(str(_FONT_PATH), size)
    return pygame.font.SysFont("monospace", size, bold=True)


def _draw_pitch(surface: pygame.Surface) -> None:
    """Draw the green pitch background and all white markings."""
    surface.fill((20, 40, 12))
    pygame.draw.rect(surface, PITCH_GREEN, (PITCH_LEFT, PITCH_TOP, PITCH_W, PITCH_H))

    # Mowed stripes for a touch of 16-bit texture.
    stripes = 10
    for i in range(stripes):
        if i % 2 == 0:
            x = PITCH_LEFT + i * PITCH_W // stripes
            pygame.draw.rect(surface, PITCH_STRIPE, (x, PITCH_TOP, PITCH_W // stripes, PITCH_H))

    # Touchlines / goal lines.
    pygame.draw.rect(surface, LINE_WHITE, (PITCH_LEFT, PITCH_TOP, PITCH_W, PITCH_H), 2)

    # Halfway line.
    mid_top = _to_screen(0.0, -Y_EXTENT)
    mid_bot = _to_screen(0.0, Y_EXTENT)
    pygame.draw.line(surface, LINE_WHITE, mid_top, mid_bot, 2)

    # Centre circle (radius 0.15 in world x-units) and spot.
    centre = _to_screen(0.0, 0.0)
    radius_px = int(0.15 * PITCH_W / (2 * X_EXTENT))
    pygame.draw.circle(surface, LINE_WHITE, centre, radius_px, 2)
    pygame.draw.circle(surface, LINE_WHITE, centre, 3)

    _draw_boxes(surface)


def _draw_boxes(surface: pygame.Surface) -> None:
    """Draw both penalty areas and goal rectangles."""
    for sign in (-1.0, 1.0):
        # Penalty area: x in [0.78, 1.0] (mirrored), y in [-0.22, 0.22].
        near_x = sign * 0.78
        goal_x = sign * X_EXTENT
        corner = _to_screen(min(near_x, goal_x), -0.22)
        far = _to_screen(max(near_x, goal_x), 0.22)
        pygame.draw.rect(
            surface,
            LINE_WHITE,
            (corner[0], corner[1], far[0] - corner[0], far[1] - corner[1]),
            2,
        )

        # Goal: a small box just outside the goal line.
        gy_top = _to_screen(goal_x, -0.07)
        gy_bot = _to_screen(goal_x, 0.07)
        depth = 10 if sign > 0 else -10
        pygame.draw.rect(
            surface,
            LINE_WHITE,
            (min(gy_top[0], gy_top[0] + depth), gy_top[1], abs(depth), gy_bot[1] - gy_top[1]),
            2,
        )


def _draw_players(surface: pygame.Surface, frame: Frame, font: pygame.font.Font, glow: set[int]) -> None:
    """Draw every player square, jersey number, and override glow."""
    half = PLAYER_SIZE // 2
    for player in frame.players:
        cx, cy = _to_screen(player.x, player.y)
        rect = pygame.Rect(cx - half, cy - half, PLAYER_SIZE, PLAYER_SIZE)
        colour = HOME_RED if player.team == "home" else AWAY_BLUE
        pygame.draw.rect(surface, colour, rect)
        if player.id in glow:
            pygame.draw.rect(surface, OVERRIDE_YELLOW, rect.inflate(4, 4), 2)
        label = font.render(_player_label(player), True, TEXT_WHITE)
        surface.blit(label, label.get_rect(center=(cx, cy)))


def _draw_ball(surface: pygame.Surface, frame: Frame) -> None:
    """Draw the ball as a white circle."""
    centre = _to_screen(frame.ball_x, frame.ball_y)
    pygame.draw.circle(surface, BALL_WHITE, centre, BALL_RADIUS)
    pygame.draw.circle(surface, TEXT_DARK, centre, BALL_RADIUS, 1)


def _draw_hud(surface: pygame.Surface, frame: Frame, font: pygame.font.Font, speed: float) -> None:
    """Draw the score/clock header and the current playback speed."""
    minutes = int(frame.t // 60)
    seconds = int(frame.t % 60)
    score = f"{frame.score[0]}  -  {frame.score[1]}"
    clock = f"{minutes:02d}:{seconds:02d}"
    score_surf = font.render(score, True, TEXT_WHITE)
    clock_surf = font.render(clock, True, TEXT_WHITE)
    surface.blit(score_surf, score_surf.get_rect(center=(WIDTH // 2, 22)))
    surface.blit(clock_surf, clock_surf.get_rect(center=(WIDTH // 2, 46)))

    small = _load_font(9)
    legend = small.render(f"{speed:g}x  SPACE pause  <- -> scrub  +/- speed", True, TEXT_WHITE)
    surface.blit(legend, (PITCH_LEFT, HEIGHT - 18))


def _draw_coach_overlay(
    surface: pygame.Surface,
    frame: Frame,
    cycles: list[CoachCycle],
    font: pygame.font.Font,
) -> None:
    """Draw the coach banner and per-player decision bubbles."""
    if not cycles:
        return
    team = cycles[0].team
    colour = HOME_RED if team == "home" else AWAY_BLUE
    names = [a.player_id for c in cycles for a in c.alerts]
    banner = pygame.Rect(0, MARGIN_TOP, WIDTH, 26)
    pygame.draw.rect(surface, colour, banner)
    text = font.render(f"⚡ COACH → {', '.join(names) or team}", True, TEXT_WHITE)
    surface.blit(text, text.get_rect(center=(WIDTH // 2, MARGIN_TOP + 13)))

    small = _load_font(8)
    for cycle in cycles:
        for alert in cycle.alerts:
            match = next((p for p in frame.players if _alert_matches_player(alert, p)), None)
            if match is None or not alert.player_decision:
                continue
            _draw_bubble(surface, small, match, alert.player_decision)


def _draw_bubble(surface: pygame.Surface, font: pygame.font.Font, player: Player, text: str) -> None:
    """Draw a small decision bubble near ``player``."""
    cx, cy = _to_screen(player.x, player.y)
    label = font.render(text[:28], True, TEXT_DARK)
    bg = label.get_rect(topleft=(cx + 10, cy - 10)).inflate(6, 4)
    pygame.draw.rect(surface, BUBBLE_BG, bg)
    pygame.draw.rect(surface, TEXT_DARK, bg, 1)
    surface.blit(label, (bg.x + 3, bg.y + 2))


def _handle_event(event: pygame.event.Event, state: _Playback, n_frames: int) -> bool:
    """Apply one input event to ``state``. Returns ``False`` to request quit."""
    if event.type == pygame.QUIT:
        return False
    if event.type != pygame.KEYDOWN:
        return True
    if event.key == pygame.K_SPACE:
        state.paused = not state.paused
    elif event.key == pygame.K_LEFT:
        state.index = max(0, state.index - SCRUB_STEP)
    elif event.key == pygame.K_RIGHT:
        state.index = min(n_frames - 1, state.index + SCRUB_STEP)
    elif event.key in (pygame.K_PLUS, pygame.K_EQUALS, pygame.K_KP_PLUS):
        state.speed_idx = min(len(SPEEDS) - 1, state.speed_idx + 1)
    elif event.key in (pygame.K_MINUS, pygame.K_KP_MINUS):
        state.speed_idx = max(0, state.speed_idx - 1)
    return True


def _advance(state: _Playback, frames: list[Frame]) -> None:
    """Advance the frame index by the current speed and arm coach banners."""
    if state.paused:
        return
    state.accum += SPEEDS[state.speed_idx]
    while state.accum >= 1.0 and state.index < len(frames) - 1:
        state.index += 1
        state.accum -= 1.0
        if frames[state.index].coach_cycles:
            state.banner_left = BANNER_FRAMES
            state.banner_cycles = frames[state.index].coach_cycles


def _glow_ids(state: _Playback, frame: Frame) -> set[int]:
    """Return ids of players to glow (overridden, while the banner is active)."""
    if state.banner_left <= 0:
        return set()
    return {
        p.id
        for c in state.banner_cycles
        for a in c.alerts
        if a.override_written
        for p in frame.players
        if _alert_matches_player(a, p)
    }


def _render_frame(
    screen: pygame.Surface,
    frame: Frame,
    state: _Playback,
    fonts: tuple[pygame.font.Font, pygame.font.Font, pygame.font.Font],
) -> None:
    """Draw one full frame (pitch, players, ball, HUD, and coach overlay)."""
    hud_font, num_font, banner_font = fonts
    _draw_pitch(screen)
    _draw_players(screen, frame, num_font, _glow_ids(state, frame))
    _draw_ball(screen, frame)
    _draw_hud(screen, frame, hud_font, SPEEDS[state.speed_idx])
    if state.banner_left > 0:
        _draw_coach_overlay(screen, frame, state.banner_cycles, banner_font)
        state.banner_left -= 1


def run_replay(path: str | Path = "match/replay.jsonl") -> None:
    """Open a window and play back the replay at ``path``.

    Args:
        path: Path to the JSONL replay file.
    """
    frames = load_replay(path)
    if not frames:
        return

    pygame.init()
    screen = pygame.display.set_mode((WIDTH, HEIGHT))
    pygame.display.set_caption("agentic-soccer replay")
    clock = pygame.time.Clock()
    fonts = (_load_font(16), _load_font(8), _load_font(11))

    state = _Playback()
    if frames[0].coach_cycles:
        state.banner_left = BANNER_FRAMES
        state.banner_cycles = frames[0].coach_cycles

    running = True
    while running:
        for event in pygame.event.get():
            running = _handle_event(event, state, len(frames)) and running

        _render_frame(screen, frames[state.index], state, fonts)
        pygame.display.flip()
        _advance(state, frames)
        clock.tick(FPS)

    pygame.quit()


def main() -> None:
    """CLI entry point: replay the file given as ``argv[1]`` (or the default)."""
    path = sys.argv[1] if len(sys.argv) > 1 else "match/replay.jsonl"
    run_replay(path)


if __name__ == "__main__":
    main()
