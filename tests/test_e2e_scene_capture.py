"""Scene-capture e2e tests: verify the RENDERED IMAGE is semantically correct.

Unlike :mod:`tests.test_e2e_replay` (which proves the pipeline runs without
raising and paints *something*), these tests render real frames headlessly and
then read the actual pixels back, asserting *semantic* properties of the picture
-- the pitch is green, both team kits are visible, the goal nets exist on both
sides, the bottom HUD carries the team-coloured score, the active-player marker
is drawn, the coach frame shows its banner + alert box, no player renders
off-frame, and near players are bigger than far ones (perspective depth scaling).

The assertions use colour *predicates / ranges* (tolerant to anti-aliasing and
the sprites' 3-tone shading), not fragile exact-RGB or pixel-diff comparisons.
Connected-component counting is done on a down-sampled boolean mask so each
shirt reads as a cluster regardless of internal shading splits.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import TYPE_CHECKING, Any, NamedTuple

import numpy as np
import pygame
import pytest

from replay.visualizer import (
    HEIGHT,
    HUD_BOTTOM_H,
    HUD_TOP,
    WIDTH,
    Frame,
    draw_pitch,
    load_assets,
    load_replay,
    parse_frame,
    project,
    render_frame,
)


if TYPE_CHECKING:
    from collections.abc import Callable, Iterator
    from types import ModuleType

# A colour-channel value: either a numpy int plane (vectorised) or a scalar int.
# Deliberately dynamic so one predicate works on whole-image arrays and on single
# ``surface.get_at`` pixels alike; aliased so it is not a bare ``Any`` annotation.
type Pixels = Any
type Mask = Any
type Predicate = Callable[[Pixels, Pixels, Pixels], Mask]

_REPLAY = "match/replay.jsonl"
_BAR_Y = HEIGHT - HUD_BOTTOM_H  # top edge of the bottom HUD bar


# ---------------------------------------------------------------------------
# Colour predicates (accept numpy arrays *or* scalar ints; tolerant ranges)
# ---------------------------------------------------------------------------
def is_green(r: Pixels, g: Pixels, b: Pixels) -> Mask:
    """Pitch green: clearly more green than red/blue."""
    return (g > r + 12) & (g > b + 12) & (g > 70)


def is_red(r: Pixels, g: Pixels, b: Pixels) -> Mask:
    """Home-kit red (shirt or HUD digit / coach banner)."""
    return (r > 110) & (r > g + 50) & (r > b + 50)


def is_blue(r: Pixels, g: Pixels, b: Pixels) -> Mask:
    """Away-kit blue (shirt or HUD digit)."""
    return (b > 110) & (b > r + 40) & (b > g + 40)


def is_white(r: Pixels, g: Pixels, b: Pixels) -> Mask:
    """Bright near-white (lines, posts, active marker)."""
    return (r > 185) & (g > 185) & (b > 185)


def is_net_mesh(r: Pixels, g: Pixels, b: Pixels) -> Mask:
    """Goal-net cross-hatch grey, specifically ``NET_GREY`` ~(206, 214, 224).

    It is light, slightly *blue* (b >= r), and dimmer than the pure-white pitch
    lines / posts (238 / 236). Excluding bright white (r < 232) is what makes the
    goal-net test mutation-proof: the touchlines, goal line, and penalty-box
    lines that pass through the goal screen region are pure white, so they do NOT
    satisfy this predicate -- only the actual mesh does.
    """
    return (r > 180) & (r < 232) & (g >= r) & (b >= r)


def is_yellow(r: Pixels, g: Pixels, b: Pixels) -> Mask:
    """Coach override alert-box yellow (high R+G, low B)."""
    return (r > 180) & (g > 160) & (b < 130)


# ---------------------------------------------------------------------------
# Pixel-analysis helpers
# ---------------------------------------------------------------------------
class Cluster(NamedTuple):
    """A connected blob of colour-matched pixels (centroid in screen px)."""

    size: int
    cx: float
    cy: float


def _channels(surface: pygame.Surface) -> tuple[Pixels, Pixels, Pixels]:
    """Return the (R, G, B) channel planes of ``surface`` as int arrays (W, H)."""
    arr = pygame.surfarray.array3d(surface).astype(int)
    return arr[:, :, 0], arr[:, :, 1], arr[:, :, 2]


def dominant_color(surface: pygame.Surface, rect: tuple[int, int, int, int]) -> tuple[int, int, int]:
    """Return the mean ``(r, g, b)`` colour inside ``rect`` (x0, y0, x1, y1)."""
    x0, y0, x1, y1 = rect
    arr = pygame.surfarray.array3d(surface).astype(int)
    mean = arr[x0:x1, y0:y1].reshape(-1, 3).mean(axis=0)
    return int(mean[0]), int(mean[1]), int(mean[2])


def region_has_color(
    surface: pygame.Surface,
    rect: tuple[int, int, int, int],
    predicate: Predicate,
    *,
    min_count: int = 30,
) -> bool:
    """Whether ``rect`` contains at least ``min_count`` pixels matching ``predicate``."""
    r, g, b = _channels(surface)
    x0, y0, x1, y1 = rect
    mask = predicate(r[x0:x1, y0:y1], g[x0:x1, y0:y1], b[x0:x1, y0:y1])
    return bool(np.asarray(mask).sum() >= min_count)


def count_color_clusters(
    surface: pygame.Surface,
    predicate: Predicate,
    *,
    min_size: int = 8,
    step: int = 3,
    region: tuple[int, int, int, int] | None = None,
) -> list[Cluster]:
    """Count connected blobs of ``predicate``-matching pixels (down-sampled BFS).

    ``min_size`` is in down-sampled cells (so it filters out anti-alias speckle);
    centroids are scaled back to full-resolution screen pixels. ``region`` limits
    analysis to a bbox so the HUD / off-pitch chrome can be excluded.
    """
    r, g, b = _channels(surface)
    mask = np.asarray(predicate(r, g, b), dtype=bool)
    if region is not None:
        x0, y0, x1, y1 = region
        keep = np.zeros_like(mask)
        keep[x0:x1, y0:y1] = True
        mask &= keep
    small = mask[::step, ::step]
    w, h = small.shape
    seen = np.zeros_like(small)
    clusters: list[Cluster] = []
    for i in range(w):
        for j in range(h):
            if not small[i, j] or seen[i, j]:
                continue
            stack = [(i, j)]
            seen[i, j] = True
            cells: list[tuple[int, int]] = []
            while stack:
                ci, cj = stack.pop()
                cells.append((ci, cj))
                for di, dj in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                    ni, nj = ci + di, cj + dj
                    if 0 <= ni < w and 0 <= nj < h and small[ni, nj] and not seen[ni, nj]:
                        seen[ni, nj] = True
                        stack.append((ni, nj))
            if len(cells) >= min_size:
                cx = sum(c[0] for c in cells) / len(cells) * step
                cy = sum(c[1] for c in cells) / len(cells) * step
                clusters.append(Cluster(size=len(cells), cx=cx, cy=cy))
    return clusters


# ---------------------------------------------------------------------------
# Synthetic frame builders (deterministic player positions for exact-ish counts)
# ---------------------------------------------------------------------------
def _player(pid: int, team: str, x: float, y: float) -> dict[str, Any]:
    return {"id": pid, "team": team, "role": "CM", "x": x, "y": y}


def _frame(players: list[dict[str, Any]], ball: tuple[float, float]) -> Frame:
    return parse_frame(
        {
            "tick": 0,
            "t": 1.0,
            "players": players,
            "ball": {"x": ball[0], "y": ball[1]},
            "score": [1, 2],
        },
    )


def _two_team_frame() -> Frame:
    """4 home + 4 away outfielders, well separated, away from pitch lines."""
    home = [(-0.6, -0.3), (-0.6, 0.3), (-0.2, -0.3), (-0.2, 0.3)]
    away = [(0.2, -0.3), (0.2, 0.3), (0.6, -0.3), (0.6, 0.3)]
    players = [_player(i + 1, "home", x, y) for i, (x, y) in enumerate(home)]
    players += [_player(i + 11, "away", x, y) for i, (x, y) in enumerate(away)]
    return _frame(players, ball=(-0.9, 0.0))


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------
@pytest.fixture
def headless_pygame() -> Iterator[ModuleType]:
    """Initialise pygame against the SDL dummy drivers (no display required)."""
    os.environ["SDL_VIDEODRIVER"] = "dummy"
    os.environ["SDL_AUDIODRIVER"] = "dummy"
    pg = pytest.importorskip("pygame")
    if pg.init()[1] > 0:
        pytest.skip("pygame could not initialise headlessly")
    try:
        yield pg
    finally:
        pg.quit()


@pytest.fixture
def real_surface(headless_pygame: ModuleType) -> pygame.Surface:
    """A rendered mid-play frame (index 200) of the recorded match replay."""
    if not Path(_REPLAY).exists():
        pytest.skip(f"{_REPLAY} not present")
    frames = load_replay(_REPLAY)
    surface: pygame.Surface = headless_pygame.Surface((WIDTH, HEIGHT))
    render_frame(surface, frames, 200, load_assets())
    return surface


# ---------------------------------------------------------------------------
# 1. Pitch is green
# ---------------------------------------------------------------------------
def test_pitch_centre_is_green(real_surface: pygame.Surface) -> None:
    """The central pitch region's dominant colour is clearly green."""
    r, g, b = dominant_color(real_surface, (430, 250, 530, 350))
    assert g > r + 20, f"centre not green (R≥G): {(r, g, b)}"
    assert g > b + 20, f"centre not green (B≥G): {(r, g, b)}"
    assert region_has_color(real_surface, (300, 200, 660, 460), is_green, min_count=500)


# ---------------------------------------------------------------------------
# 2. Both teams present (red + blue shirt clusters)
# ---------------------------------------------------------------------------
def test_both_teams_detected(headless_pygame: ModuleType) -> None:
    """A 4-home + 4-away scene yields several red AND several blue clusters."""
    surface = headless_pygame.Surface((WIDTH, HEIGHT))
    render_frame(surface, [_two_team_frame()], 0, load_assets())
    pitch = (0, HUD_TOP, WIDTH, _BAR_Y)
    red = count_color_clusters(surface, is_red, region=pitch)
    blue = count_color_clusters(surface, is_blue, region=pitch)
    # Shirts split into ~2 blobs under 3-tone shading, so we expect >= a few each
    # for 4 players a side -- never zero, and both kits must appear.
    assert len(red) >= 3, f"too few home(red) clusters: {len(red)}"
    assert len(blue) >= 3, f"too few away(blue) clusters: {len(blue)}"


# ---------------------------------------------------------------------------
# 3. Goal nets on both sides
# ---------------------------------------------------------------------------
def test_goal_nets_present_both_sides(real_surface: pygame.Surface) -> None:
    """Both goal boxes carry a dense NET_GREY mesh (not just stray white lines).

    Mutation-verified: with both ``_draw_goal_net`` calls stubbed the mesh count
    in each goal box drops to 0 (the pitch lines are pure white and excluded),
    so this assertion goes RED -- it genuinely requires the net to be drawn.
    Tight goal-box rects exclude the centre/halfway markings entirely.
    """
    left = region_has_color(real_surface, (0, 170, 150, 350), is_net_mesh, min_count=600)
    right = region_has_color(real_surface, (810, 170, 960, 350), is_net_mesh, min_count=600)
    assert left, "left goal net mesh missing"
    assert right, "right goal net mesh missing"


# ---------------------------------------------------------------------------
# 4. Bottom HUD carries the team-coloured score
# ---------------------------------------------------------------------------
def test_bottom_hud_has_team_colored_score(real_surface: pygame.Surface) -> None:
    """The bottom bar is non-blank and shows red (home) and blue (away) digits."""
    colours = {
        real_surface.get_at((x, y))[:3]
        for x in range(0, WIDTH, 12)
        for y in range(_BAR_Y, HEIGHT, 6)
    }
    assert len(colours) > 3, "HUD bar rendered near-blank"
    assert region_has_color(real_surface, (80, _BAR_Y, 300, HEIGHT), is_red, min_count=40)
    assert region_has_color(real_surface, (660, _BAR_Y, 900, HEIGHT), is_blue, min_count=40)


# ---------------------------------------------------------------------------
# 5. Active-player marker
# ---------------------------------------------------------------------------
def test_active_marker_drawn_near_player(headless_pygame: ModuleType) -> None:
    """The active marker adds white pixels above a player's head, on the pitch."""
    # One isolated home player who is the ball-carrier (ball at his feet); placed
    # off the halfway/centre lines so stray white can only be the marker.
    px, py = 0.35, 0.2
    frame = _frame([_player(7, "home", px, py)], ball=(px, py))
    assets = load_assets()
    sx, sy, depth = project(px, py)
    head_y = int(sy - assets.frame_h * assets.sprite_scale * depth - 8)
    box = (int(sx) - 16, head_y - 12, int(sx) + 16, head_y + 14)
    assert box[3] < _BAR_Y, "marker badge must sit on the pitch, not in the HUD"

    pitch_only = headless_pygame.Surface((WIDTH, HEIGHT))
    draw_pitch(pitch_only)
    before = region_has_color(pitch_only, box, is_white, min_count=20)

    rendered = headless_pygame.Surface((WIDTH, HEIGHT))
    render_frame(rendered, [frame], 0, assets)
    after = region_has_color(rendered, box, is_white, min_count=20)

    assert not before, "bare pitch already had white in the badge box (bad test region)"
    assert after, "active marker drew no white badge above the player"


# ---------------------------------------------------------------------------
# 6. Coach frame: top banner + alert box
# ---------------------------------------------------------------------------
def test_coach_frame_banner_and_alert(headless_pygame: ModuleType) -> None:
    """The coach frame paints a coloured top banner and a yellow alert box."""
    if not Path(_REPLAY).exists():
        pytest.skip(f"{_REPLAY} not present")
    frames = load_replay(_REPLAY)
    coach_idx = next((i for i, f in enumerate(frames) if f.coach_cycles), None)
    assert coach_idx is not None, "no coach_cycle frame in the replay"
    surface = headless_pygame.Surface((WIDTH, HEIGHT))
    render_frame(surface, frames, coach_idx, load_assets())

    # Banner spans the top band just under the reserved HUD_TOP rows.
    banner = (0, HUD_TOP, WIDTH, HUD_TOP + 28)
    assert region_has_color(surface, banner, is_red, min_count=1000), "coach banner missing"
    # Yellow override box somewhere on the pitch.
    assert region_has_color(surface, (0, HUD_TOP, WIDTH, _BAR_Y), is_yellow, min_count=80)


# ---------------------------------------------------------------------------
# 7. No off-frame players
# ---------------------------------------------------------------------------
def test_no_offframe_player_clusters(headless_pygame: ModuleType) -> None:
    """Every detected shirt cluster sits inside the window, above the HUD bar."""
    surface = headless_pygame.Surface((WIDTH, HEIGHT))
    render_frame(surface, [_two_team_frame()], 0, load_assets())
    # Restrict to the pitch band so the (legitimately team-coloured) HUD score
    # digits below _BAR_Y are not mistaken for players.
    pitch = (0, HUD_TOP, WIDTH, _BAR_Y)
    clusters = count_color_clusters(surface, is_red, region=pitch) + count_color_clusters(
        surface, is_blue, region=pitch,
    )
    assert clusters, "no player clusters detected at all"
    for c in clusters:
        assert 0 <= c.cx <= WIDTH, f"cluster off-frame in x: {c}"
        assert HUD_TOP <= c.cy <= _BAR_Y, f"cluster off-pitch in y: {c}"


# ---------------------------------------------------------------------------
# 8. Perspective: near players render larger than far players
# ---------------------------------------------------------------------------
def test_perspective_near_larger_than_far(headless_pygame: ModuleType) -> None:
    """A near-touchline home player paints more red than a far-touchline one."""
    near = _player(3, "home", 0.30, 0.40)  # near (bottom of pitch, large depth)
    far = _player(4, "home", 0.30, -0.40)  # far (top of pitch, small depth)
    surface = headless_pygame.Surface((WIDTH, HEIGHT))
    render_frame(surface, [_frame([near, far], ball=(-0.9, 0.0))], 0, load_assets())

    r, g, b = _channels(surface)

    def band_red(y0: int, y1: int) -> int:
        return int(np.asarray(is_red(r[:, y0:y1], g[:, y0:y1], b[:, y0:y1])).sum())

    near_px = band_red(400, _BAR_Y)  # bottom band (near player's body)
    far_px = band_red(15, 130)  # top band (far player's body)
    assert near_px > 0, f"near player not rendered: {near_px}"
    assert far_px > 0, f"far player not rendered: {far_px}"
    assert near_px > far_px * 1.5, f"near not clearly larger than far: {near_px} vs {far_px}"
