"""ISS-style angled-perspective pygame replayer for agentic-soccer matches.

Reads a JSONL replay produced by :mod:`replay.logger` and plays it back in an
*International Superstar Soccer* style diagonal view: a tilted trapezoidal pitch
(far side narrow and high, near side wide and low), animated SNES-style player
sprites that run/idle and face their direction of travel, a ball with a shadow,
a score/clock HUD, and a coach banner overlay.

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

Perspective projection
-----------------------
:func:`project` maps world ``(x, y)`` to ``(screen_x, screen_y, depth_scale)``.
Let ``t = (y + Y_EXTENT) / (2 * Y_EXTENT)`` run 0 at the far touchline (top,
narrow) to 1 at the near touchline (bottom, wide). The pitch half-width and the
vertical position lerp linearly with ``t``, producing a trapezoid; ``depth_scale``
lerps from ``DEPTH_FAR`` (0.55) to ``DEPTH_NEAR`` (1.0). Sprites and the ball are
scaled by ``depth_scale`` and depth-sorted by projected ``screen_y`` (painter's
algorithm) so nearer figures overlap farther ones.

Run::

    uv run python -m replay.visualizer [match/replay.jsonl]
"""

from __future__ import annotations

import json
import math
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

# --- Window geometry (pixels) ------------------------------------------------
WIDTH = 960
HEIGHT = 600

# Trapezoid anchors: the far (y=-Y_EXTENT) touchline is high and narrow, the
# near (y=+Y_EXTENT) touchline is low and wide.
TOP_Y = 120
BOTTOM_Y = 560
FAR_HALF = 250.0  # half pitch width at the far touchline (narrow)
NEAR_HALF = 440.0  # half pitch width at the near touchline (wide)

DEPTH_FAR = 0.55
DEPTH_NEAR = 1.0

HUD_TOP = 64  # reserve the top band for the score/clock HUD

# --- Colours (16-bit palette) -----------------------------------------------
BG_DARK = (18, 30, 14)
PITCH_GREEN = (45, 90, 27)
PITCH_STRIPE = (52, 102, 31)
LINE_WHITE = (235, 235, 235)
HOME_RED = (204, 34, 34)
AWAY_BLUE = (34, 68, 204)
GOAL_WHITE = (220, 220, 220)
SHADOW = (0, 0, 0)
TEXT_WHITE = (240, 240, 240)
TEXT_DARK = (16, 16, 16)
BUBBLE_BG = (250, 250, 235)
OVERRIDE_YELLOW = (245, 224, 66)

# --- Sprite geometry ---------------------------------------------------------
SPRITE_W = 16
SPRITE_H = 24
SPRITE_SCALE = 4
RUN_INDICES = (1, 2, 3, 4)
IDLE_INDEX = 0
RUN_TICKS_PER_FRAME = 4  # advance the run cycle every N global ticks (~7.5fps @30)
MOVE_THRESHOLD = 0.004  # world-units of motion that counts as "running"

# --- Playback ----------------------------------------------------------------
FPS = 30
SPEEDS = (0.5, 1.0, 2.0, 4.0)
DEFAULT_SPEED_IDX = 1
SCRUB_STEP = 5
BANNER_FRAMES = 90  # 3 seconds at 30 fps

# Number of elements expected in an (x, y) / (home, away) pair.
_PAIR_LEN = 2

_ASSET_DIR = Path(__file__).parent / "assets"
_SPRITE_DIR = _ASSET_DIR / "sprites"
_FONT_PATH = _ASSET_DIR / "PressStart2P.ttf"


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


# --- Perspective projection --------------------------------------------------
def project(x: float, y: float) -> tuple[float, float, float]:
    """Project world ``(x, y)`` into the tilted pitch.

    Args:
        x: World x in ``[-1, 1]`` (home goal at -1, away goal at +1).
        y: World y in ``[-Y_EXTENT, Y_EXTENT]`` (far touchline at -Y_EXTENT).

    Returns:
        A ``(screen_x, screen_y, depth_scale)`` tuple. ``depth_scale`` lerps
        from :data:`DEPTH_FAR` at the far touchline to :data:`DEPTH_NEAR` at
        the near touchline.
    """
    t = (y + Y_EXTENT) / (2 * Y_EXTENT)
    t = max(0.0, min(1.0, t))
    half_w = FAR_HALF + (NEAR_HALF - FAR_HALF) * t
    depth = DEPTH_FAR + (DEPTH_NEAR - DEPTH_FAR) * t
    sx = WIDTH / 2 + x * half_w
    sy = TOP_Y + t * (BOTTOM_Y - TOP_Y)
    return sx, sy, depth


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


def _is_goalkeeper(player: Player) -> bool:
    """Return whether ``player`` should use the goalkeeper sprite sheet."""
    return player.role.upper() == "GK" or player.id == 0


# --- Sprite / asset loading --------------------------------------------------
@dataclass(frozen=True)
class Assets:
    """Loaded sprite sheets, ball graphics, and fonts for one render session.

    Bundled into a single object so render helpers stay within the project's
    ``max-args`` limit.
    """

    sheets: dict[str, list[pygame.Surface]]
    ball: pygame.Surface
    ball_shadow: pygame.Surface
    hud_font: pygame.font.Font
    banner_font: pygame.font.Font
    small_font: pygame.font.Font


def _load_image(name: str) -> pygame.Surface:
    """Load a sprite PNG (alpha preserved, no ``convert`` so it works headless)."""
    return pygame.image.load(str(_SPRITE_DIR / name))


def _slice_sheet(name: str) -> list[pygame.Surface]:
    """Slice an 80x24 sheet into its five 16x24 animation frames."""
    sheet = _load_image(name)
    frames: list[pygame.Surface] = []
    for i in range(5):
        frame = sheet.subsurface(pygame.Rect(i * SPRITE_W, 0, SPRITE_W, SPRITE_H))
        frames.append(frame.copy())
    return frames


def _load_font(size: int) -> pygame.font.Font:
    """Load the bundled Press Start 2P font at ``size``, or a monospace fallback."""
    if _FONT_PATH.exists():
        return pygame.font.Font(str(_FONT_PATH), size)
    return pygame.font.SysFont("monospace", size, bold=True)


def load_assets() -> Assets:
    """Load every sprite sheet, the ball graphics, and the HUD fonts.

    Returns:
        An :class:`Assets` bundle. Requires pygame (and its font module) to be
        initialised; works under the SDL ``dummy`` driver without a display.
    """
    if not pygame.font.get_init():
        pygame.font.init()
    return Assets(
        sheets={
            "home": _slice_sheet("home.png"),
            "away": _slice_sheet("away.png"),
            "goalkeeper": _slice_sheet("goalkeeper.png"),
        },
        ball=_load_image("ball.png"),
        ball_shadow=_load_image("ball_shadow.png"),
        hud_font=_load_font(16),
        banner_font=_load_font(11),
        small_font=_load_font(8),
    )


# --- Pitch drawing -----------------------------------------------------------
def _pitch_quad() -> list[tuple[float, float]]:
    """Return the four screen corners of the projected pitch trapezoid."""
    return [
        project(-X_EXTENT, -Y_EXTENT)[:2],
        project(X_EXTENT, -Y_EXTENT)[:2],
        project(X_EXTENT, Y_EXTENT)[:2],
        project(-X_EXTENT, Y_EXTENT)[:2],
    ]


def _draw_stripes(surface: pygame.Surface) -> None:
    """Draw vertical mowing stripes that converge with the perspective."""
    stripes = 10
    for i in range(0, stripes, 2):
        x0 = -X_EXTENT + (2 * X_EXTENT) * i / stripes
        x1 = -X_EXTENT + (2 * X_EXTENT) * (i + 1) / stripes
        poly = [
            project(x0, -Y_EXTENT)[:2],
            project(x1, -Y_EXTENT)[:2],
            project(x1, Y_EXTENT)[:2],
            project(x0, Y_EXTENT)[:2],
        ]
        pygame.draw.polygon(surface, PITCH_STRIPE, poly)


def _proj_line(surface: pygame.Surface, p0: tuple[float, float], p1: tuple[float, float]) -> None:
    """Draw a 2px white line between two world points already projected to screen."""
    pygame.draw.line(surface, LINE_WHITE, p0, p1, 2)


def _draw_ellipse_arc(surface: pygame.Surface, world_r: float, segments: int) -> None:
    """Draw the centre circle as a perspective ellipse by sampling its rim."""
    pts: list[tuple[float, float]] = []
    for i in range(segments):
        ang = 2 * math.pi * i / segments
        wx = world_r * math.cos(ang)
        # Scale y radius down so the world circle reads as round under the tilt.
        wy = world_r * 0.42 * math.sin(ang)
        pts.append(project(wx, wy)[:2])
    pygame.draw.polygon(surface, LINE_WHITE, pts, 2)


def _draw_box(surface: pygame.Surface, sign: float) -> None:
    """Draw one penalty area on the goal side given by ``sign`` (+1 away / -1 home)."""
    near_x = sign * 0.78
    goal_x = sign * X_EXTENT
    corners = [
        project(near_x, -0.22)[:2],
        project(goal_x, -0.22)[:2],
        project(goal_x, 0.22)[:2],
        project(near_x, 0.22)[:2],
    ]
    pygame.draw.polygon(surface, LINE_WHITE, corners, 2)


def _draw_goal(surface: pygame.Surface, sign: float) -> None:
    """Draw a simple 3D-ish goal frame at the goal line on side ``sign``."""
    goal_x = sign * X_EXTENT
    back_top = project(goal_x, -0.07)
    back_bot = project(goal_x, 0.07)
    depth = 18 * (1 if sign > 0 else -1)
    bt = (back_top[0], back_top[1])
    bb = (back_bot[0], back_bot[1])
    ft = (back_top[0] + depth, back_top[1] - 14)
    fb = (back_bot[0] + depth, back_bot[1] - 14)
    pygame.draw.polygon(surface, GOAL_WHITE, [bt, ft, fb, bb], 2)
    pygame.draw.line(surface, GOAL_WHITE, bt, ft, 2)
    pygame.draw.line(surface, GOAL_WHITE, bb, fb, 2)


def draw_pitch(surface: pygame.Surface) -> None:
    """Draw the tilted trapezoidal pitch and all perspective-correct markings."""
    surface.fill(BG_DARK)
    pygame.draw.polygon(surface, PITCH_GREEN, _pitch_quad())
    _draw_stripes(surface)
    # Touchlines + goal lines (the trapezoid outline).
    pygame.draw.polygon(surface, LINE_WHITE, _pitch_quad(), 2)
    # Halfway line.
    _proj_line(surface, project(0.0, -Y_EXTENT)[:2], project(0.0, Y_EXTENT)[:2])
    # Centre circle + spot.
    _draw_ellipse_arc(surface, 0.15, 28)
    cx, cy, _ = project(0.0, 0.0)
    pygame.draw.circle(surface, LINE_WHITE, (int(cx), int(cy)), 3)
    for sign in (-1.0, 1.0):
        _draw_box(surface, sign)
        _draw_goal(surface, sign)


# Backwards-compatible alias kept for any external callers / tests.
def _draw_pitch(surface: pygame.Surface) -> None:
    """Compatibility wrapper around :func:`draw_pitch`."""
    draw_pitch(surface)


def _draw_boxes(surface: pygame.Surface) -> None:
    """Compatibility wrapper drawing both penalty boxes and goals."""
    for sign in (-1.0, 1.0):
        _draw_box(surface, sign)
        _draw_goal(surface, sign)


# --- Player / ball drawing ---------------------------------------------------
def _shadow_ellipse(surface: pygame.Surface, cx: float, cy: float, depth: float) -> None:
    """Draw a soft translucent ground-shadow ellipse centred at ``(cx, cy)``."""
    w = max(6, int(18 * depth))
    h = max(3, int(7 * depth))
    shadow = pygame.Surface((w, h), pygame.SRCALPHA)
    pygame.draw.ellipse(shadow, (0, 0, 0, 90), shadow.get_rect())
    surface.blit(shadow, (int(cx - w / 2), int(cy - h / 2)))


def _select_frame(player: Player, prev: Player | None, tick: int) -> tuple[int, bool]:
    """Choose a sprite-sheet index and facing for ``player`` this ``tick``.

    Args:
        player: The player to animate.
        prev: The same player one frame earlier, if available.
        tick: A monotonically increasing global tick driving the run cycle.

    Returns:
        A ``(frame_index, flip_left)`` tuple. ``frame_index`` is the index into
        the player's sprite sheet; ``flip_left`` is True when moving left.
    """
    dx = (player.x - prev.x) if prev else 0.0
    dy = (player.y - prev.y) if prev else 0.0
    moving = (dx * dx + dy * dy) ** 0.5 > MOVE_THRESHOLD
    run_idx = RUN_INDICES[(tick // RUN_TICKS_PER_FRAME) % len(RUN_INDICES)]
    idx = run_idx if moving else IDLE_INDEX
    return idx, dx < 0


@dataclass(frozen=True)
class _Sprite:
    """A player resolved to its draw position, depth, sheet index, and facing."""

    sheet: list[pygame.Surface]
    index: int
    flip: bool
    sx: float
    sy: float
    depth: float
    glow: bool


def _resolve_sprite(
    player: Player,
    prev: Player | None,
    tick: int,
    assets: Assets,
    glow: bool,
) -> _Sprite:
    """Resolve a player into a positioned, animation-selected :class:`_Sprite`."""
    sx, sy, depth = project(player.x, player.y)
    if _is_goalkeeper(player):
        sheet = assets.sheets["goalkeeper"]
    else:
        sheet = assets.sheets[player.team if player.team in assets.sheets else "home"]
    index, flip = _select_frame(player, prev, tick)
    return _Sprite(sheet=sheet, index=index, flip=flip, sx=sx, sy=sy, depth=depth, glow=glow)


def _blit_sprite(surface: pygame.Surface, spr: _Sprite) -> None:
    """Draw one resolved sprite (shadow, scaled frame, optional override glow)."""
    _shadow_ellipse(surface, spr.sx, spr.sy, spr.depth)
    frame = spr.sheet[spr.index]
    if spr.flip:
        frame = pygame.transform.flip(frame, True, False)
    scale = SPRITE_SCALE * spr.depth
    w = max(1, int(SPRITE_W * scale))
    h = max(1, int(SPRITE_H * scale))
    scaled = pygame.transform.scale(frame, (w, h))
    # Anchor the fixed 16x24 frame by bottom-centre so feet sit on the pitch.
    bx = int(spr.sx - w / 2)
    by = int(spr.sy - h)
    if spr.glow:
        glow_rect = pygame.Rect(bx - 2, by - 2, w + 4, h + 4)
        pygame.draw.rect(surface, OVERRIDE_YELLOW, glow_rect, 2)
    surface.blit(scaled, (bx, by))


@dataclass(frozen=True)
class _AnimContext:
    """Inputs needed to animate one frame's players (keeps arg counts small)."""

    frame: Frame
    prev: Frame | None
    tick: int
    assets: Assets
    glow_ids: set[int]


def draw_players(surface: pygame.Surface, ctx: _AnimContext) -> None:
    """Draw every player, depth-sorted far-to-near (painter's algorithm)."""
    prev_by_id = {p.id: p for p in ctx.prev.players} if ctx.prev else {}
    sprites = [
        _resolve_sprite(p, prev_by_id.get(p.id), ctx.tick, ctx.assets, p.id in ctx.glow_ids)
        for p in ctx.frame.players
    ]
    for spr in sorted(sprites, key=lambda s: s.sy):
        _blit_sprite(surface, spr)


def draw_ball(surface: pygame.Surface, frame: Frame, assets: Assets) -> None:
    """Draw the ball with its shadow, scaled by depth."""
    sx, sy, depth = project(frame.ball_x, frame.ball_y)
    sw = max(2, int(assets.ball_shadow.get_width() * depth))
    sh = max(1, int(assets.ball_shadow.get_height() * depth))
    shadow = pygame.transform.scale(assets.ball_shadow, (sw, sh))
    surface.blit(shadow, (int(sx - sw / 2), int(sy - sh / 2)))
    bw = max(2, int(assets.ball.get_width() * depth * 1.5))
    bh = max(2, int(assets.ball.get_height() * depth * 1.5))
    ball = pygame.transform.scale(assets.ball, (bw, bh))
    surface.blit(ball, (int(sx - bw / 2), int(sy - bh - sh / 2)))


# --- HUD / coach overlay -----------------------------------------------------
def draw_hud(surface: pygame.Surface, frame: Frame, assets: Assets, speed: float) -> None:
    """Draw the score/clock header and the current playback speed legend."""
    minutes = int(frame.t // 60)
    seconds = int(frame.t % 60)
    score = f"{frame.score[0]}  -  {frame.score[1]}"
    clock = f"{minutes:02d}:{seconds:02d}"
    score_surf = assets.hud_font.render(score, True, TEXT_WHITE)
    clock_surf = assets.hud_font.render(clock, True, TEXT_WHITE)
    surface.blit(score_surf, score_surf.get_rect(center=(WIDTH // 2, 22)))
    surface.blit(clock_surf, clock_surf.get_rect(center=(WIDTH // 2, 46)))
    legend = assets.small_font.render(
        f"{speed:g}x  SPACE pause  <- -> scrub  +/- speed  ESC quit",
        True,
        TEXT_WHITE,
    )
    surface.blit(legend, (16, HEIGHT - 16))


def draw_coach_overlay(
    surface: pygame.Surface,
    frame: Frame,
    cycles: list[CoachCycle],
    assets: Assets,
) -> None:
    """Draw the coach banner and per-player decision bubbles."""
    if not cycles:
        return
    team = cycles[0].team
    colour = HOME_RED if team == "home" else AWAY_BLUE
    names = [a.player_id for c in cycles for a in c.alerts]
    banner = pygame.Rect(0, HUD_TOP, WIDTH, 26)
    pygame.draw.rect(surface, colour, banner)
    text = assets.banner_font.render(f"⚡ COACH → {', '.join(names) or team}", True, TEXT_WHITE)
    surface.blit(text, text.get_rect(center=(WIDTH // 2, HUD_TOP + 13)))
    for cycle in cycles:
        for alert in cycle.alerts:
            match = next((p for p in frame.players if _alert_matches_player(alert, p)), None)
            if match is None or not alert.player_decision:
                continue
            _draw_bubble(surface, assets.small_font, match, alert.player_decision)


def _draw_bubble(surface: pygame.Surface, font: pygame.font.Font, player: Player, text: str) -> None:
    """Draw a small decision bubble near ``player``."""
    sx, sy, _ = project(player.x, player.y)
    label = font.render(text[:28], True, TEXT_DARK)
    bg = label.get_rect(topleft=(int(sx) + 12, int(sy) - 40)).inflate(6, 4)
    pygame.draw.rect(surface, BUBBLE_BG, bg)
    pygame.draw.rect(surface, TEXT_DARK, bg, 1)
    surface.blit(label, (bg.x + 3, bg.y + 2))


# --- Headless single-frame render -------------------------------------------
def render_frame(
    surface: pygame.Surface,
    frames: list[Frame],
    index: int,
    assets: Assets | None = None,
    *,
    speed: float = 1.0,
) -> None:
    """Render frame ``index`` of ``frames`` onto ``surface``.

    This is the headless render seam: it works under ``SDL_VIDEODRIVER=dummy``
    and draws the coach overlay whenever the frame carries a cycle (independent
    of any live banner timer), so a saved PNG shows the banner. Animation uses
    ``frames[index - 1]`` to detect motion. Overridden players named in the
    frame's own coach cycle glow.

    Args:
        surface: Destination surface (e.g. ``pygame.Surface((WIDTH, HEIGHT))``).
        frames: All parsed frames.
        index: The frame index to draw.
        assets: A preloaded :class:`Assets` bundle; loaded on demand if omitted.
        speed: Playback speed shown in the HUD legend.
    """
    if not frames:
        return
    index = max(0, min(len(frames) - 1, index))
    if assets is None:
        assets = load_assets()
    frame = frames[index]
    prev = frames[index - 1] if index > 0 else None
    glow_ids = {
        p.id
        for c in frame.coach_cycles
        for a in c.alerts
        if a.override_written
        for p in frame.players
        if _alert_matches_player(a, p)
    }
    ctx = _AnimContext(frame=frame, prev=prev, tick=frame.tick, assets=assets, glow_ids=glow_ids)
    draw_pitch(surface)
    draw_players(surface, ctx)
    draw_ball(surface, frame, assets)
    draw_hud(surface, frame, assets, speed)
    if frame.coach_cycles:
        draw_coach_overlay(surface, frame, frame.coach_cycles, assets)


# --- Interactive playback ----------------------------------------------------
@dataclass
class _Playback:
    """Mutable playback state for the replay loop."""

    index: int = 0
    paused: bool = False
    speed_idx: int = DEFAULT_SPEED_IDX
    accum: float = 0.0
    banner_left: int = 0
    banner_cycles: list[CoachCycle] = field(default_factory=list)


def _handle_event(event: pygame.event.Event, state: _Playback, n_frames: int) -> bool:
    """Apply one input event to ``state``. Returns ``False`` to request quit."""
    if event.type == pygame.QUIT:
        return False
    if event.type != pygame.KEYDOWN:
        return True
    if event.key == pygame.K_ESCAPE:
        return False
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


def _coach_cycles_for(state: _Playback, frame: Frame) -> list[CoachCycle]:
    """Return coach cycles to overlay this frame, if the banner is live."""
    if frame.coach_cycles:
        return frame.coach_cycles
    if state.banner_left > 0:
        return state.banner_cycles
    return []


def _render_live(surface: pygame.Surface, frames: list[Frame], state: _Playback, assets: Assets) -> None:
    """Render the current playback frame with a live (timed) coach banner."""
    frame = frames[state.index]
    prev = frames[state.index - 1] if state.index > 0 else None
    ctx = _AnimContext(
        frame=frame, prev=prev, tick=frame.tick, assets=assets, glow_ids=_glow_ids(state, frame),
    )
    draw_pitch(surface)
    draw_players(surface, ctx)
    draw_ball(surface, frame, assets)
    draw_hud(surface, frame, assets, SPEEDS[state.speed_idx])
    cycles = _coach_cycles_for(state, frame)
    if cycles:
        draw_coach_overlay(surface, frame, cycles, assets)
    if state.banner_left > 0:
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
    assets = load_assets()

    state = _Playback()
    if frames[0].coach_cycles:
        state.banner_left = BANNER_FRAMES
        state.banner_cycles = frames[0].coach_cycles

    running = True
    while running:
        for event in pygame.event.get():
            running = _handle_event(event, state, len(frames)) and running
        _render_live(screen, frames, state, assets)
        pygame.display.flip()
        _advance(state, frames)
        clock.tick(FPS)

    pygame.quit()


def save_frame_png(path: str | Path, index: int, out: str | Path) -> None:
    """Render a single replay frame to a PNG (headless-friendly helper).

    Args:
        path: Path to the JSONL replay.
        index: Frame index to render.
        out: Output PNG path.
    """
    if not pygame.get_init():
        pygame.init()
    frames = load_replay(path)
    surface = pygame.Surface((WIDTH, HEIGHT))
    render_frame(surface, frames, index)
    pygame.image.save(surface, str(out))


def main() -> None:
    """CLI entry point: replay the file given as ``argv[1]`` (or the default)."""
    path = sys.argv[1] if len(sys.argv) > 1 else "match/replay.jsonl"
    run_replay(path)


if __name__ == "__main__":
    main()
