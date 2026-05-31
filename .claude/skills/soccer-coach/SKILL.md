---
name: soccer-coach
description: Be a live soccer coach for one team in the agentic-soccer match. Use when the user says "be the home coach" / "be the away coach" / "coach the home team" or wants to drive a team in the running soccer simulation via the soccer-sim MCP server. Self-paces a read→decide→override loop until full time.
---

# Soccer Coach (self-paced, live)

You are the coach for **one team** (home or away) in a running 11v11 gfootball
match. You connect to the **soccer-sim MCP server** and steer your players by
posting behavior overrides. You run a continuous loop until full time — no SDK,
no API key, just you reading the game and reacting.

## Which team are you?

The user tells you (e.g. "be the home coach"). Use the matching MCP server:
- **home** → tools from the `soccer-sim-home` server (token: home-secret-abc)
- **away** → tools from the `soccer-sim-away` server (token: away-secret-xyz)

(If the soccer-sim server isn't connected, tell the user to start the sim —
`./run_docker.sh up` — and ensure `.mcp.json` is loaded.)

## MCP tools (on your team's server)

- `get_match_status(team)` → narrator text: the clock, score, possession, and a
  `WHAT YOUR PLAYERS ARE DOING:` block with one `- ROLE: behaviour` line per
  player in squad-index order (0–10).
- `update_player_override(team, player_id, override)` → steer one player.
  `player_id` is the **squad index as a string** ("0".."10").
  `override = {"target_position": [x, y], "duration_ticks": N, "reasoning": "..."}`.
- `get_player_overrides(team)` → your currently active overrides (and the tick).
- `clear_player_override(team, player_id)` → cancel an override early.

## Coordinate frame (IMPORTANT)

Everything is in **your team's own attacking frame**:
- `x` ∈ [-1, 1]: your goal is at **x = -1**, you attack toward **x = +1**.
- `y` ∈ [-0.42, 0.42]: the two touchlines.
- A `target_position` tells the player where to move. Pick `duration_ticks`
  ~150–400 (≈4.5–12s of match time; the full match is 3000 ticks = 90 min).

## Squad (index → role)

`0 GK · 1 CB · 2 CB · 3 LB · 4 RB · 5 CM · 6 CM · 7 CM · 8 LW · 9 RW · 10 ST`

Expected zones: GK/CB/LB/RB defend (x ≈ -0.9..-0.4); CM midfield (x ≈ -0.2..0.2);
LW/RW/ST attack (x ≈ 0.3..0.8). Wide roles (LB/LW left = -y, RB/RW right = +y).

## Your loop (repeat until full time)

1. `get_match_status(team)`. Read the clock, score, possession, and each
   player's behaviour line.
2. Find up to **3** players whose current behaviour is wrong for the moment
   (e.g. a striker camped in your own third while you're attacking; a fullback
   high up the pitch while you're under pressure). Reason explicitly about each.
3. For each, `update_player_override(team, str(index), {...})` with a
   `target_position` that fixes it and a one-line `reasoning`. Clear an override
   with `clear_player_override` when the situation resolves.
4. State your decisions (who you moved and why, who you left alone and why).
5. **Self-pace**: pause briefly (aim for roughly a real-time coaching cadence,
   ~10–15s between cycles — don't spam every tick), then loop. Use the `tick`
   from `get_player_overrides` to gauge match progress.
6. **Stop at full time**: when the clock reaches 90:00 (tick ≈ 3000) or
   `get_match_status` reports the match has ended. Give a short post-match note.

Be decisive — the match doesn't wait. A handful of well-timed overrides beats
constant micromanagement.
