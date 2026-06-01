# Design: "Goal!" Art-Direction Migration (whole-pitch)

**Date:** 2026-06-01
**Status:** Approved
**Reference:** `docs/superpowers/specs/goal-reference.jpg` (Jaleco *Goal!*, Super Famicom)

## Goal

Re-skin the existing pygame replay renderer so the **whole-pitch** view matches the
art direction of Jaleco's *Goal!* — steeper skewed perspective, larger shaded
sprites, detailed goal nets, brighter pitch, a bottom HUD, and an active-player
marker — **without** changing the camera model (we keep all 22 players visible for
tactical replay) and **without** breaking the existing machinery.

## Non-goals (YAGNI)

- **No ball-follow / scrolling camera.** User chose whole-pitch framing.
- **No crowd / stadium background.** Doesn't serve the tactical replay; high art cost.
- No change to `load_replay`, the replay JSON format, MCP/sim, or the 85 tests' contracts.

## Constraints / Invariants

- `replay/visualizer.py` keeps `load_replay(path)`, the `Frame` shape (`.players` with
  `.id/.team/.role/.x/.y`, `.ball`, `.score`, `.coach_cycle`), `render_frame`/
  `save_frame_png` headless seams, `run_replay(path)`, `main()`, and the playback
  controls (SPACE / ←→ / +- / ESC).
- Pitch coords unchanged: x∈[-1,1] (x=-1 home goal line, x=+1 away goal line),
  y∈[-0.42,0.42].
- Guard: `uv run --no-sync pytest -q` stays green (85+), `ruff check .` clean,
  `mypy .` clean. Use `--no-sync` always (plain uv run triggers a slow gfootball rebuild).
- Sprites face right; flip horizontally for left. Headless rendering must work under
  `SDL_VIDEODRIVER=dummy` (verification renders frames to PNG).

## Work units (each independently testable)

### Unit 1 — Pitch & skewed perspective (`visualizer.py`)
- Change `project(x, y) -> (sx, sy, depth)` from a symmetric trapezoid to a **skewed**
  parallelogram: vertical pitch lines lean diagonally (a horizontal shear that grows
  with depth), matching Goal!'s slanted look. Steeper tilt than current.
- Brighter Goal!-green base + finer mow stripes; remove the dark vignette/background so
  the field fills more of the frame.
- Widen the depth-scale range (near players clearly larger than far). Keep painter's-
  algorithm depth sort.
- Centre circle remains a projected ellipse; penalty boxes + halfway line follow the
  new shear.

### Unit 2 — Detailed goals (`visualizer.py`, new helper `_draw_goal_net`)
- Replace the line-frame goals at x=±1 with a perspective **net mesh**: white posts +
  crossbar and a diamond/cross-hatch grid receding in perspective (as in the reference's
  right-hand goal). Both goals.

### Unit 3 — Sprites v2 (`sprite_gen.py` regen, new assets)
- Larger base sprite (e.g. 20×30 or keep 16×24 but richer) with **3-tone shading**
  (highlight/midtone/shadow) on the shirt, defined shorts + socks, a stronger elliptical
  drop shadow, and clearer run-cycle frames (chunkier, more readable legs).
- Keep manifest contract (idle=frame0, run=1..4, face-right, blit rect) so the renderer
  blit code needs only the scale constant updated. Update `manifest.json` +
  regenerate `home.png/away.png/goalkeeper.png/ball.png/ball_shadow.png/_preview.png`.

### Unit 4 — Bottom HUD (`visualizer.py`, new helper `_draw_hud` rework)
- Move score to a **bottom bar** in the Goal! style: team codes (`HOM`/`AWY`), large
  outlined score digits, centered match clock (MM:SS), and `◀1 P▶` / `COM▶` control-hint
  glyphs. Pixel font (Press Start 2P if present, else monospace). Keep the ⚡ COACH
  banner (top) for coach_cycle frames.

### Unit 5 — Active-player marker (`visualizer.py`, new helper)
- Draw Goal!'s selected-player indicator: a white arc/`①`-style marker under one player —
  the ball-carrier (nearest player to the ball) by default, or the last-overridden player.
  Visually distinct from the coach's **yellow** alert box.

## Architecture / ownership

- **Two file-conflict domains**, so two agents on an agent team:
  - **sprite-agent** → Unit 3 only (`replay/sprite_gen.py` + `replay/assets/sprites/*`).
  - **render-agent** → Units 1, 2, 4, 5 (`replay/visualizer.py` + its tests).
  - These don't share files. render-agent consumes sprite-agent's manifest/assets, so
    sprite-agent lands first (or render-agent codes against the existing manifest, which
    is contract-compatible).
- **Team-lead** runs the visual-verification loop: render frames to PNG headlessly, Read
  them, compare to `goal-reference.jpg`, iterate until each unit matches.

## Verification

- Per unit: `save_frame_png(...)` → team-lead Reads the PNG → compare to reference →
  iterate. Use a mid-play frame (players spread), a near-goal frame, and a coach_cycle
  frame (frame 45 of `match/replay.jsonl`).
- Update `tests/test_e2e_replay.py` to keep covering load + headless render incl. new
  helpers (goal net, bottom HUD, active marker), non-blank assertions.
- Final gate: 85+ tests green, ruff clean, mypy clean, tree clean.

## Risks

- **Skewed projection can break line/box math** → verify centre circle + penalty boxes
  still render coherently after the shear (visual check, not just tests).
- **Bigger sprites → overlap clutter** at kickoff cluster → depth-sort must stay correct;
  check a clustered frame.
- **Net mesh perf** → keep it simple (drawn lines, not per-pixel) so 30fps holds.
