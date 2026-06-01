"""SNES-style footballer sprite-sheet generator for agentic-soccer replays.

Draws pixel-art footballer sprites procedurally with pygame (no external art),
in the spirit of International Superstar Soccer / Super Soccer: bold shapes, a
limited palette, 1px dark outlines, and crisp pixels (no anti-aliasing).

Outputs (to ``replay/assets/sprites/``):

* ``home.png`` / ``away.png`` / ``goalkeeper.png`` -- one sprite *sheet* per kit,
  a single row of 5 frames laid out as ``[idle, run0, run1, run2, run3]``.
  Frame size is 16w x 24h. Sheet size is 80w x 24h.
* ``ball.png`` -- 8x8 white ball with a couple of black pentagon hints.
* ``ball_shadow.png`` -- 8x4 soft dark shadow blob.
* ``manifest.json`` -- the authoritative description of frame size, layout,
  per-kit files, facing convention and the recommended scale factor.
* ``_preview.png`` -- all kits + all frames + ball, scaled up 6x with
  nearest-neighbour on a checkerboard background, for human eyeballing.

Sprites face RIGHT. The renderer should horizontally flip them for left-facing
movement (``pygame.transform.flip(surf, True, False)``).

All surfaces are ``SRCALPHA`` (transparent background) so the renderer can blit
them straight onto the green pitch.

Deterministic: no randomness; every pixel is drawn from fixed tables.

Run::

    uv run --no-sync python replay/sprite_gen.py
"""

from __future__ import annotations

import json
import os
from pathlib import Path


# Headless: no display is available in CI / agent environments. Must be set
# before pygame video subsystem init.
os.environ.setdefault("SDL_VIDEODRIVER", "dummy")

import pygame


# --- Geometry ----------------------------------------------------------------
FRAME_W = 16
FRAME_H = 24
RUN_FRAMES = 4
FRAMES_PER_SHEET = 1 + RUN_FRAMES  # idle + run cycle
SHEET_W = FRAME_W * FRAMES_PER_SHEET
SHEET_H = FRAME_H
SCALE = 4  # recommended nearest-neighbour scale for the replayer

# --- Palette (RGB) -----------------------------------------------------------
TRANSPARENT = (0, 0, 0, 0)
OUTLINE = (24, 18, 28)        # near-black, used for all 1px outlines
SKIN = (240, 190, 150)
SKIN_SHADE = (205, 150, 115)
HAIR = (70, 45, 30)
BOOT = (30, 30, 36)
SOCK = (245, 245, 245)
SOCK_SHADE = (205, 205, 210)

# Per-kit colours: shirt, shirt-shade, shorts, sock, patch (number/stripe accent)
KITS = {
    "home": {
        "shirt": (210, 50, 45),
        "shirt_shade": (165, 35, 32),
        "shorts": (245, 245, 245),
        "shorts_shade": (205, 205, 210),
        "sock": (210, 50, 45),
        "patch": (245, 245, 245),
    },
    "away": {
        "shirt": (45, 75, 200),
        "shirt_shade": (32, 55, 155),
        "shorts": (30, 30, 36),
        "shorts_shade": (18, 18, 24),
        "sock": (45, 75, 200),
        "patch": (245, 245, 90),
    },
    "goalkeeper": {
        "shirt": (70, 210, 90),
        "shirt_shade": (45, 165, 70),
        "shorts": (35, 35, 40),
        "shorts_shade": (22, 22, 28),
        "sock": (70, 210, 90),
        "patch": (25, 25, 30),
    },
}

KIT_ORDER = ["home", "away", "goalkeeper"]


def _px(surf: pygame.Surface, x: int, y: int, color: tuple[int, int, int]) -> None:
    """Set a single opaque pixel (bounds-checked)."""
    if 0 <= x < surf.get_width() and 0 <= y < surf.get_height():
        surf.set_at((x, y), color)


def _rect(
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


def _outline_silhouette(surf: pygame.Surface) -> None:
    """Add a 1px dark outline around every opaque blob.

    For each transparent pixel orthogonally adjacent to an opaque one, paint the
    dark OUTLINE colour. Done as a post-pass so we never have to hand-place the
    outline of every limb. Deterministic.
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
# Body is drawn facing RIGHT, centred horizontally in the 16-wide frame.
# Vertical bands (base, before per-frame bob):
#   y 2-7   head + hair
#   y 8-15  torso (shirt)
#   y 16-19 shorts
#   y 20-22 socks/legs
#   y 22-23 boots
#
# A "pose" describes the two legs and two arms for a frame.


def _draw_player_frame(kit: dict, pose: str, bob: int) -> pygame.Surface:
    """Draw one 16x24 footballer frame onto a transparent surface.

    ``pose`` is one of: ``idle``, ``contactA``, ``passA``, ``contactB``, ``passB``.
    ``bob`` shifts the head+torso vertically by 1px on the 'passing' frames to
    sell the run cycle's body bob.
    """
    s = pygame.Surface((FRAME_W, FRAME_H), pygame.SRCALPHA)
    s.fill(TRANSPARENT)

    cx = 8  # nominal centre column
    shirt = kit["shirt"]
    shirt_shade = kit["shirt_shade"]
    shorts = kit["shorts"]
    shorts_shade = kit["shorts_shade"]
    sock = kit["sock"]
    patch = kit["patch"]

    top = 2 + bob  # head top

    # --- Head (skin) 4 wide ---
    _rect(s, cx - 2, top, 5, 5, SKIN)
    # face shading on the back (left) edge
    _rect(s, cx - 2, top + 1, 1, 3, SKIN_SHADE)
    # --- Hair: cap over the top + back of the head ---
    _rect(s, cx - 2, top, 5, 2, HAIR)
    _rect(s, cx - 2, top + 1, 1, 1, HAIR)  # sideburn back
    # eye (facing right) -- single dark pixel
    _px(s, cx + 2, top + 3, OUTLINE)

    # --- Torso / shirt ---
    ty = top + 5  # shirt top (y~7+bob)
    _rect(s, cx - 3, ty, 7, 7, shirt)
    # shirt shading down the left side
    _rect(s, cx - 3, ty, 1, 7, shirt_shade)
    # number / accent patch: a 2x2 block centred on the chest
    _rect(s, cx, ty + 2, 2, 3, patch)

    # --- Arms (swing opposite to legs) ---
    # Arm forward = drawn toward +x (right), arm back = toward -x (left).
    def arm(side_forward: bool) -> None:
        if side_forward:
            # front arm reaching forward/down on the right
            _rect(s, cx + 4, ty + 1, 1, 4, SKIN)
            _px(s, cx + 4, ty + 5, SKIN_SHADE)
        else:
            # back arm trailing on the left
            _rect(s, cx - 4, ty + 1, 1, 4, SKIN)
            _px(s, cx - 4, ty + 5, SKIN_SHADE)

    # --- Shorts ---
    sy = ty + 7  # shorts top (y~14+bob)
    _rect(s, cx - 3, sy, 7, 3, shorts)
    _rect(s, cx - 3, sy, 1, 3, shorts_shade)

    # --- Legs + socks + boots, per pose ---
    # leg_x are column positions of the two legs at the hip line.
    ly = sy + 3  # legs top (y~17+bob)

    def leg(x: int, length: int, foot_dx: int, lead: bool) -> None:
        """Draw a leg: sock column of given length, then a boot offset foot.

        ``foot_dx`` shifts the boot horizontally (stride). ``lead`` tints the
        sock the kit sock colour (front leg) vs a shaded sock (back leg).
        """
        col = sock if lead else SOCK_SHADE
        for i in range(length):
            _px(s, x, ly + i, col)
        # thigh joins to shorts (skin just below shorts hidden by sock here)
        boot_y = ly + length
        # boot: 2px wide pointing forward (+x)
        _rect(s, x + min(0, foot_dx), boot_y, 2 + abs(foot_dx), 1, BOOT)
        if boot_y + 1 < FRAME_H:
            _px(s, x + foot_dx, boot_y, BOOT)

    if pose == "idle":
        arm(False)
        # both legs together, slight stand
        leg(cx - 1, 4, 0, True)
        leg(cx + 1, 4, 0, False)
    elif pose == "contactA":
        # right leg forward (lead), left leg back -- ground contact
        arm(True)
        leg(cx + 2, 3, 1, True)   # front leg forward + boot reaching right
        leg(cx - 2, 4, -1, False)  # back leg extended behind
    elif pose == "passA":
        # legs passing under body (mid-stride), bob up
        arm(False)
        leg(cx, 4, 0, True)
        leg(cx + 1, 3, 0, False)
    elif pose == "contactB":
        # left leg forward, right leg back (mirror of contactA stride)
        arm(False)
        leg(cx + 1, 4, 1, False)
        leg(cx - 2, 3, -1, True)
    elif pose == "passB":
        arm(True)
        leg(cx - 1, 4, 0, True)
        leg(cx, 3, 0, False)

    _outline_silhouette(s)
    return s


POSE_SEQUENCE = ["idle", "contactA", "passA", "contactB", "passB"]
POSE_BOB = {"idle": 0, "contactA": 0, "passA": 1, "contactB": 0, "passB": 1}


def build_kit_sheet(kit_name: str) -> pygame.Surface:
    """Build the 80x24 sprite sheet for one kit (idle + 4 run frames)."""
    kit = KITS[kit_name]
    sheet = pygame.Surface((SHEET_W, SHEET_H), pygame.SRCALPHA)
    sheet.fill(TRANSPARENT)
    for idx, pose in enumerate(POSE_SEQUENCE):
        frame = _draw_player_frame(kit, pose, POSE_BOB[pose])
        sheet.blit(frame, (idx * FRAME_W, 0))
    return sheet


def build_ball() -> pygame.Surface:
    """8x8 white ball with black pentagon hints and a 1px outline."""
    s = pygame.Surface((8, 8), pygame.SRCALPHA)
    s.fill(TRANSPARENT)
    white = (250, 250, 250)
    shade = (200, 205, 215)
    # round-ish body
    body = [
        (2, 1), (3, 1), (4, 1), (5, 1),
        (1, 2), (2, 2), (3, 2), (4, 2), (5, 2), (6, 2),
        (1, 3), (2, 3), (3, 3), (4, 3), (5, 3), (6, 3),
        (1, 4), (2, 4), (3, 4), (4, 4), (5, 4), (6, 4),
        (1, 5), (2, 5), (3, 5), (4, 5), (5, 5), (6, 5),
        (2, 6), (3, 6), (4, 6), (5, 6),
    ]
    for x, y in body:
        _px(s, x, y, white)
    # bottom-right shading
    for x, y in ((5, 5), (6, 4), (4, 6), (5, 6)):
        _px(s, x, y, shade)
    # black pentagon hints
    for x, y in ((3, 3), (4, 3), (3, 4)):
        _px(s, x, y, OUTLINE)
    _px(s, 5, 2, OUTLINE)
    _px(s, 2, 5, OUTLINE)
    _outline_silhouette(s)
    return s


def build_ball_shadow() -> pygame.Surface:
    """8x4 soft dark elliptical shadow blob (semi-transparent)."""
    s = pygame.Surface((8, 4), pygame.SRCALPHA)
    s.fill(TRANSPARENT)
    dark = (0, 0, 0, 90)
    darker = (0, 0, 0, 130)
    rows = [
        (2, 5),  # y0: x2..x5
        (1, 6),  # y1
        (1, 6),  # y2
        (2, 5),  # y3
    ]
    for y, (x0, x1) in enumerate(rows):
        for x in range(x0, x1 + 1):
            s.set_at((x, y), dark)
    for x in range(2, 6):
        s.set_at((x, 1), darker)
        s.set_at((x, 2), darker)
    return s


def _checkerboard(w: int, h: int, cell: int = 6) -> pygame.Surface:
    a = (60, 64, 72)
    b = (96, 100, 110)
    s = pygame.Surface((w, h))
    for y in range(0, h, cell):
        for x in range(0, w, cell):
            color = a if ((x // cell + y // cell) % 2 == 0) else b
            s.fill(color, (x, y, cell, cell))
    return s


def build_preview(sheets: dict[str, pygame.Surface], ball: pygame.Surface,
                  shadow: pygame.Surface, scale: int = 6) -> pygame.Surface:
    """Lay out all kit sheets + ball, scaled up nearest-neighbour."""
    pad = 8
    label_h = 0
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
    # ball row: ball + shadow scaled up
    bball = pygame.transform.scale(ball, (8 * scale, 8 * scale))
    bshadow = pygame.transform.scale(shadow, (8 * scale, 4 * scale))
    bg.blit(bshadow, (pad, y + 8 * scale + 4))
    bg.blit(bball, (pad, y))
    _ = label_h
    return bg


def main() -> None:
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
        "description": "SNES-style footballer sprites for agentic-soccer replays.",
        "frame_width": FRAME_W,
        "frame_height": FRAME_H,
        "facing": "sprites face right; flip horizontally for left "
                  "(pygame.transform.flip(surf, True, False))",
        "scale_recommendation": SCALE,
        "outline_color": list(OUTLINE),
        "transparent_background": True,
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
            "away": {"file": "away.png", "shirt": "blue", "shorts": "black"},
            "goalkeeper": {"file": "goalkeeper.png", "shirt": "green",
                           "shorts": "black"},
        },
        "ball": {"file": "ball.png", "width": 8, "height": 8},
        "ball_shadow": {"file": "ball_shadow.png", "width": 8, "height": 4,
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
