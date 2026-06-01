"""Jaleco "Goal!"-style footballer sprite-sheet generator for agentic-soccer.

Draws chunky pixel-art footballer sprites procedurally with pygame (no external
art), in the spirit of Jaleco's *Goal!* (Super Famicom): stocky bodies, boldly
**shaded** kits (a lit highlight tone, the base team colour, and a shadow tone,
lit from the upper-left), defined shorts + socks + boots, short hair over a skin
head, 1px dark outlines and crisp pixels (no anti-aliasing).

Outputs (to ``replay/assets/sprites/``):

* ``home.png`` / ``away.png`` / ``goalkeeper.png`` -- one sprite *sheet* per kit,
  a single row of 5 frames laid out as ``[idle, run0, run1, run2, run3]``.
  Frame size is 24w x 32h. Sheet size is 120w x 32h.
* ``ball.png`` -- 10x10 white ball with black pentagon hints and shading.
* ``ball_shadow.png`` -- 10x5 soft dark elliptical shadow blob (for the ball).
* ``manifest.json`` -- the authoritative description of frame size, layout,
  per-kit files, facing convention and the recommended scale factor.
* ``_preview.png`` -- all kits + all frames + ball, scaled up 6x with
  nearest-neighbour on a checkerboard background, for human eyeballing.

Sprites face RIGHT. The renderer flips them for left-facing movement
(``pygame.transform.flip(surf, True, False)``).

The renderer draws each player's elliptical **ground shadow** itself (see
``replay/visualizer.py::_shadow_ellipse``); sprites therefore carry NO baked
body shadow and keep a fully transparent background. ``ball_shadow.png`` is the
one exception -- it is the ball's drop shadow, blit under the ball by the
renderer.

All surfaces are ``SRCALPHA`` (transparent background).

Deterministic: no randomness; every pixel is drawn from fixed tables.

Run::

    uv run --no-sync python replay/sprite_gen.py
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any


# Headless: no display is available in CI / agent environments. Must be set
# before pygame video subsystem init.
os.environ.setdefault("SDL_VIDEODRIVER", "dummy")

import pygame


# --- Geometry ----------------------------------------------------------------
FRAME_W = 24
FRAME_H = 32
RUN_FRAMES = 4
FRAMES_PER_SHEET = 1 + RUN_FRAMES  # idle + run cycle
SHEET_W = FRAME_W * FRAMES_PER_SHEET
SHEET_H = FRAME_H
SCALE = 3  # recommended nearest-neighbour scale (24x32 is ~1.5x the old 16x24)

# --- Palette (RGB) -----------------------------------------------------------
TRANSPARENT = (0, 0, 0, 0)
OUTLINE = (20, 16, 24)        # near-black, used for all 1px outlines
SKIN = (242, 194, 152)
SKIN_HI = (255, 218, 180)
SKIN_SHADE = (202, 150, 112)
HAIR = (66, 42, 28)
HAIR_HI = (104, 70, 46)
BOOT = (28, 26, 32)
BOOT_HI = (66, 64, 74)

# Per-kit colours. Each kit gives a 3-tone shirt (highlight / base / shadow),
# a 2-tone shorts, a 2-tone sock and an accent patch (number/trim).
KITS: dict[str, dict[str, Any]] = {
    "home": {  # red shirt, white shorts
        "shirt_hi": (236, 96, 84),
        "shirt": (208, 48, 42),
        "shirt_shade": (150, 28, 26),
        "shorts_hi": (255, 255, 255),
        "shorts": (232, 232, 238),
        "shorts_shade": (186, 186, 198),
        "sock": (224, 224, 230),
        "sock_shade": (176, 176, 188),
        "patch": (250, 250, 250),
    },
    "away": {  # blue shirt, dark shorts
        "shirt_hi": (96, 138, 238),
        "shirt": (46, 86, 206),
        "shirt_shade": (28, 52, 150),
        "shorts_hi": (58, 64, 82),
        "shorts": (36, 40, 54),
        "shorts_shade": (20, 22, 32),
        "sock": (46, 86, 206),
        "sock_shade": (28, 52, 150),
        "patch": (248, 224, 92),
    },
    "goalkeeper": {  # teal/green shirt, dark shorts
        "shirt_hi": (92, 226, 188),
        "shirt": (42, 192, 150),
        "shirt_shade": (24, 138, 106),
        "shorts_hi": (54, 60, 70),
        "shorts": (32, 36, 44),
        "shorts_shade": (18, 20, 26),
        "sock": (42, 192, 150),
        "sock_shade": (24, 138, 106),
        "patch": (24, 26, 32),
    },
}

KIT_ORDER = ["home", "away", "goalkeeper"]


def _px(surf: pygame.Surface, x: int, y: int, color: tuple[int, int, int]) -> None:
    """Set a single opaque pixel (bounds-checked)."""
    if 0 <= x < surf.get_width() and 0 <= y < surf.get_height():
        surf.set_at((x, y), color)


def _rect(  # noqa: PLR0913 - a pixel-rect primitive; x/y/w/h/color are irreducible
    surf: pygame.Surface,
    x: int,
    y: int,
    w: int,
    h: int,
    color: tuple[int, int, int],
) -> None:
    """Fill an axis-aligned rectangle of solid pixels (no AA)."""
    for yy in range(y, y + h):
        for xx in range(x, x + w):
            _px(surf, xx, yy, color)


def _shaded_block(  # noqa: PLR0913 - a beveled-rect primitive; tones+geometry irreducible
    surf: pygame.Surface,
    x: int,
    y: int,
    w: int,
    h: int,
    base: tuple[int, int, int],
    hi: tuple[int, int, int],
    shade: tuple[int, int, int],
) -> None:
    """Fill a rectangle with 3-tone bevel shading, lit from the upper-left.

    The block is filled with ``base``; the top row and left column get the
    ``hi`` highlight tone, and the bottom row and right column get the ``shade``
    shadow tone. The result reads as a rounded, lit volume rather than flat fill.
    """
    _rect(surf, x, y, w, h, base)
    # Highlight: top edge + left edge.
    _rect(surf, x, y, w, 1, hi)
    _rect(surf, x, y, 1, h, hi)
    # Shadow: bottom edge + right edge.
    _rect(surf, x, y + h - 1, w, 1, shade)
    _rect(surf, x + w - 1, y, 1, h, shade)


def _outline_silhouette(surf: pygame.Surface) -> None:
    """Add a 1px dark outline around every opaque blob.

    For each transparent pixel orthogonally adjacent to an opaque one, paint the
    dark OUTLINE colour. Done as a post-pass so we never hand-place limb
    outlines. Deterministic.
    """
    w, h = surf.get_width(), surf.get_height()
    to_paint: list[tuple[int, int]] = []
    for y in range(h):
        for x in range(w):
            if surf.get_at((x, y)).a != 0:
                continue
            for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                nx, ny = x + dx, y + dy
                if 0 <= nx < w and 0 <= ny < h and surf.get_at((nx, ny)).a != 0:
                    to_paint.append((x, y))
                    break
    for x, y in to_paint:
        surf.set_at((x, y), OUTLINE)


# --- Footballer drawing ------------------------------------------------------
# Body drawn facing RIGHT, centred horizontally in the 24-wide frame.
# Vertical bands (base, before per-frame bob), in a 24x32 frame:
#   y 3-9    head + hair (7 tall)
#   y 10-19  torso / shirt (10 tall)
#   y 20-24  shorts (5 tall)
#   y 25-29  socks / legs
#   y 29-31  boots
# A "pose" describes the two legs (and arm swing) for a frame.

CX = 12  # nominal centre column


def _draw_head(s: pygame.Surface, top: int) -> None:
    """Draw the skin head + short hair cap + a single eye, facing right."""
    hx = CX - 3
    # Skin head, 6 wide x 6 tall.
    _shaded_block(s, hx, top, 6, 6, SKIN, SKIN_HI, SKIN_SHADE)
    # Hair cap over top + back (left) of the head.
    _rect(s, hx, top, 6, 2, HAIR)
    _rect(s, hx, top, 1, 4, HAIR)
    _px(s, hx, top, HAIR_HI)
    _px(s, hx + 1, top, HAIR_HI)
    # Ear hint on the back of the head.
    _px(s, hx, top + 3, SKIN_SHADE)
    # Eye (facing right).
    _px(s, hx + 4, top + 3, OUTLINE)
    # Jaw / chin shade.
    _px(s, hx + 1, top + 5, SKIN_SHADE)


def _draw_torso(s: pygame.Surface, kit: dict[str, Any], ty: int) -> None:
    """Draw the shaded shirt with a chest accent patch and a neck."""
    # Neck.
    _rect(s, CX - 1, ty - 1, 3, 1, SKIN_SHADE)
    # Shirt block, 10 wide x 10 tall.
    sx = CX - 5
    _shaded_block(s, sx, ty, 10, 10, kit["shirt"], kit["shirt_hi"], kit["shirt_shade"])
    # Extra shadow wedge under the right arm / lower-right torso.
    _rect(s, sx + 7, ty + 6, 2, 3, kit["shirt_shade"])
    # Chest accent patch (number/trim), 3 wide x 4 tall, slightly right-of-centre.
    _rect(s, CX, ty + 3, 3, 4, kit["patch"])
    _px(s, CX, ty + 3, kit["shirt_shade"])  # keep patch reading as on the shirt


def _draw_arm(s: pygame.Surface, kit: dict[str, Any], ty: int, forward: bool) -> None:
    """Draw the swinging near arm: forward (+x) or trailing back (-x)."""
    if forward:
        ax = CX + 5
        _shaded_block(s, ax, ty + 1, 2, 6, kit["shirt"], kit["shirt_hi"], kit["shirt_shade"])
        _rect(s, ax, ty + 7, 2, 2, SKIN)  # forearm/hand
        _px(s, ax + 1, ty + 8, SKIN_SHADE)
    else:
        ax = CX - 6
        _shaded_block(s, ax, ty + 1, 2, 6, kit["shirt"], kit["shirt_hi"], kit["shirt_shade"])
        _rect(s, ax, ty + 7, 2, 2, SKIN)  # forearm/hand
        _px(s, ax, ty + 8, SKIN_SHADE)


def _draw_leg(  # noqa: PLR0913 - a leg primitive; geometry + tones are irreducible
    s: pygame.Surface,
    kit: dict[str, Any],
    leg_x: int,
    leg_top: int,
    length: int,
    foot_dx: int,
    lead: bool,
) -> None:
    """Draw one leg: a thigh (shorts skin), a 2-tone sock and a boot.

    ``leg_x`` is the left column of the 2px-wide leg. ``foot_dx`` shifts the
    boot horizontally for stride. ``lead`` (front leg) uses the bright sock tone;
    the trailing leg uses the shaded sock tone so the legs read as alternating.
    """
    sock = kit["sock"] if lead else kit["sock_shade"]
    sock_sh = kit["sock_shade"]
    # Bare thigh just below the shorts (a touch of skin).
    _rect(s, leg_x, leg_top, 2, 1, SKIN_SHADE)
    # Sock column.
    for i in range(1, length):
        _rect(s, leg_x, leg_top + i, 2, 1, sock)
    _px(s, leg_x + 1, leg_top + length - 1, sock_sh)  # right-edge sock shade
    # Boot: 3px wide pointing in the stride direction, 2 tall.
    boot_y = leg_top + length
    bx = leg_x + min(0, foot_dx)
    bw = 3 + abs(foot_dx)
    _rect(s, bx, boot_y, bw, 2, BOOT)
    _rect(s, bx, boot_y, bw, 1, BOOT_HI)  # top sheen on the boot


# Each pose: (front_leg, back_leg, arm_forward, bob)
# leg tuple = (x, length, foot_dx, lead)
POSES: dict[str, dict[str, Any]] = {
    "idle": {
        "front": (CX + 1, 4, 0, True),
        "back": (CX - 3, 4, 0, False),
        "arm_forward": False,
        "bob": 0,
    },
    "run0": {  # right leg drives forward, left leg extends behind (ground contact)
        "front": (CX + 2, 3, 2, True),
        "back": (CX - 4, 4, -2, False),
        "arm_forward": True,
        "bob": 0,
    },
    "run1": {  # passing under the body, bob up
        "front": (CX, 5, 0, True),
        "back": (CX + 1, 4, 0, False),
        "arm_forward": False,
        "bob": -1,
    },
    "run2": {  # left leg drives forward, right leg extends behind (mirror stride)
        "front": (CX + 1, 4, 2, False),
        "back": (CX - 4, 3, -2, True),
        "arm_forward": False,
        "bob": 0,
    },
    "run3": {  # passing under the body again, bob up
        "front": (CX - 1, 5, 0, True),
        "back": (CX, 4, 0, False),
        "arm_forward": True,
        "bob": -1,
    },
}

POSE_SEQUENCE = ["idle", "run0", "run1", "run2", "run3"]


def _draw_player_frame(kit: dict[str, Any], pose_name: str) -> pygame.Surface:
    """Draw one 24x32 footballer frame onto a transparent surface."""
    s = pygame.Surface((FRAME_W, FRAME_H), pygame.SRCALPHA)
    s.fill(TRANSPARENT)
    pose = POSES[pose_name]
    bob = pose["bob"]

    head_top = 3 + bob
    torso_top = head_top + 7      # y~10
    shorts_top = torso_top + 10   # y~20
    legs_top = shorts_top + 5     # y~25

    # Back leg first (behind the body), then torso, then front leg on top.
    bx, blen, bdx, blead = pose["back"]
    _draw_leg(s, kit, bx, legs_top, blen, bdx, blead)

    # Shorts block, 12 wide x 5 tall.
    _shaded_block(
        s, CX - 6, shorts_top, 12, 5,
        kit["shorts"], kit["shorts_hi"], kit["shorts_shade"],
    )

    _draw_torso(s, kit, torso_top)
    _draw_head(s, head_top)
    _draw_arm(s, kit, torso_top, pose["arm_forward"])

    fx, flen, fdx, flead = pose["front"]
    _draw_leg(s, kit, fx, legs_top, flen, fdx, flead)

    _outline_silhouette(s)
    return s


def build_kit_sheet(kit_name: str) -> pygame.Surface:
    """Build the 120x32 sprite sheet for one kit (idle + 4 run frames)."""
    kit = KITS[kit_name]
    sheet = pygame.Surface((SHEET_W, SHEET_H), pygame.SRCALPHA)
    sheet.fill(TRANSPARENT)
    for idx, pose in enumerate(POSE_SEQUENCE):
        frame = _draw_player_frame(kit, pose)
        sheet.blit(frame, (idx * FRAME_W, 0))
    return sheet


def build_ball() -> pygame.Surface:
    """10x10 white ball with black pentagon hints and a 1px outline."""
    s = pygame.Surface((10, 10), pygame.SRCALPHA)
    s.fill(TRANSPARENT)
    white = (250, 250, 250)
    shade = (198, 204, 214)
    # Round body (rows of x-spans).
    rows = {
        1: (3, 6), 2: (2, 7), 3: (1, 8), 4: (1, 8),
        5: (1, 8), 6: (1, 8), 7: (2, 7), 8: (3, 6),
    }
    for y, (x0, x1) in rows.items():
        for x in range(x0, x1 + 1):
            _px(s, x, y, white)
    # Bottom-right shading crescent.
    for x, y in ((7, 5), (8, 4), (8, 5), (6, 6), (7, 6), (5, 7), (6, 7)):
        _px(s, x, y, shade)
    # Black pentagon hints.
    for x, y in ((4, 4), (5, 4), (4, 5), (5, 5)):
        _px(s, x, y, OUTLINE)
    _px(s, 6, 2, OUTLINE)
    _px(s, 2, 6, OUTLINE)
    _px(s, 7, 7, OUTLINE)
    _outline_silhouette(s)
    return s


def build_ball_shadow() -> pygame.Surface:
    """10x5 soft dark elliptical shadow blob (semi-transparent)."""
    s = pygame.Surface((10, 5), pygame.SRCALPHA)
    s.fill(TRANSPARENT)
    dark = (0, 0, 0, 80)
    darker = (0, 0, 0, 120)
    rows = {0: (3, 6), 1: (1, 8), 2: (1, 8), 3: (1, 8), 4: (3, 6)}
    for y, (x0, x1) in rows.items():
        for x in range(x0, x1 + 1):
            s.set_at((x, y), dark)
    for x in range(2, 8):
        s.set_at((x, 2), darker)
    return s


def _checkerboard(w: int, h: int, cell: int = 6) -> pygame.Surface:
    a = (58, 102, 64)   # pitch-ish greens so the kits read in context
    b = (70, 120, 76)
    s = pygame.Surface((w, h))
    for y in range(0, h, cell):
        for x in range(0, w, cell):
            color = a if ((x // cell + y // cell) % 2 == 0) else b
            s.fill(color, (x, y, cell, cell))
    return s


def build_preview(
    sheets: dict[str, pygame.Surface],
    ball: pygame.Surface,
    shadow: pygame.Surface,
    scale: int = 6,
) -> pygame.Surface:
    """Lay out all kit sheets + ball, scaled up nearest-neighbour."""
    pad = 8
    row_h = SHEET_H * scale + pad
    rows = len(KIT_ORDER) + 1  # kits + ball row
    pw = SHEET_W * scale + pad * 2
    ph = rows * row_h + pad
    bg = _checkerboard(pw, ph)
    y = pad
    for name in KIT_ORDER:
        big = pygame.transform.scale(sheets[name], (SHEET_W * scale, SHEET_H * scale))
        bg.blit(big, (pad, y))
        y += row_h
    # Ball row: shadow under the ball, both scaled up.
    bw, bh = ball.get_size()
    sw, sh = shadow.get_size()
    bball = pygame.transform.scale(ball, (bw * scale, bh * scale))
    bshadow = pygame.transform.scale(shadow, (sw * scale, sh * scale))
    bg.blit(bshadow, (pad, y + bh * scale))
    bg.blit(bball, (pad, y))
    return bg


def main() -> None:
    """Generate all sprite sheets, the ball, manifest, and preview PNG."""
    pygame.init()
    out_dir = Path(__file__).resolve().parent / "assets" / "sprites"
    out_dir.mkdir(parents=True, exist_ok=True)

    sheets: dict[str, pygame.Surface] = {}
    written: list[tuple[str, tuple[int, int]]] = []

    for name in KIT_ORDER:
        sheet = build_kit_sheet(name)
        sheets[name] = sheet
        path = out_dir / f"{name}.png"
        pygame.image.save(sheet, str(path))
        written.append((path.name, sheet.get_size()))

    ball = build_ball()
    pygame.image.save(ball, str(out_dir / "ball.png"))
    written.append(("ball.png", ball.get_size()))

    shadow = build_ball_shadow()
    pygame.image.save(shadow, str(out_dir / "ball_shadow.png"))
    written.append(("ball_shadow.png", shadow.get_size()))

    preview = build_preview(sheets, ball, shadow, scale=6)
    pygame.image.save(preview, str(out_dir / "_preview.png"))

    manifest = {
        "description": "Jaleco 'Goal!'-style footballer sprites for agentic-soccer replays.",
        "frame_width": FRAME_W,
        "frame_height": FRAME_H,
        "facing": "sprites face right; flip horizontally for left "
                  "(pygame.transform.flip(surf, True, False))",
        "scale_recommendation": SCALE,
        "outline_color": list(OUTLINE),
        "transparent_background": True,
        "ground_shadow": "drawn by the renderer (visualizer.py::_shadow_ellipse); "
                         "sprites carry no baked body shadow",
        "states": {
            "idle": {"frames": 1, "frame_indices": [0]},
            "run": {"frames": RUN_FRAMES, "frame_indices": [1, 2, 3, 4],
                    "cycle": "play 1->2->3->4->1; ~8-12 fps reads as running"},
        },
        "layout": {
            "type": "single_row",
            "frames_per_sheet": FRAMES_PER_SHEET,
            "order": ["idle", "run0", "run1", "run2", "run3"],
            "sheet_width": SHEET_W,
            "sheet_height": SHEET_H,
            "frame_x_for_index": "x = index * frame_width; y = 0",
        },
        "kits": {
            "home": {"file": "home.png", "shirt": "red", "shorts": "white"},
            "away": {"file": "away.png", "shirt": "blue", "shorts": "dark"},
            "goalkeeper": {"file": "goalkeeper.png", "shirt": "teal",
                           "shorts": "dark"},
        },
        "ball": {"file": "ball.png", "width": ball.get_width(),
                 "height": ball.get_height()},
        "ball_shadow": {"file": "ball_shadow.png", "width": shadow.get_width(),
                        "height": shadow.get_height(),
                        "note": "semi-transparent; blit under the ball"},
        "preview": "_preview.png",
    }
    with (out_dir / "manifest.json").open("w") as f:
        json.dump(manifest, f, indent=2)
    written.append(("manifest.json", (0, 0)))
    written.append(("_preview.png", preview.get_size()))

    # --- Report ---
    print(f"Wrote sprites to: {out_dir}")
    for fname, size in written:
        if fname == "manifest.json":
            print(f"  {fname}")
        elif fname.endswith(".png"):
            frames = FRAMES_PER_SHEET if size == (SHEET_W, SHEET_H) else 1
            print(f"  {fname}: {size[0]}x{size[1]}px  frames={frames}")
    print(f"Layout: single row [idle, run0, run1, run2, run3], "
          f"frame={FRAME_W}x{FRAME_H}, sheet={SHEET_W}x{SHEET_H}")
    print("Facing: right (flip horizontally for left)")
    print(f"Recommended scale: {SCALE}x nearest-neighbour")

    pygame.quit()


if __name__ == "__main__":
    main()
