"""Canonical player-role layout shared across the match domain.

The 11 squad positions, by index within a team, for the gfootball 11v11 layout.
Both the replay logger and the match engine derive a player's role from this
single source of truth. Note that :data:`simulator.narrator.ROLE_LABELS` is a
*different*, intentionally distinct list (human-readable descriptive labels such
as ``"CB (left)"``) used only for narration.
"""

from __future__ import annotations


# Player roles by index within a team (gfootball 11v11 layout).
ROLES: list[str] = ["GK", "CB", "CB", "LB", "RB", "CM", "CM", "CM", "LW", "RW", "ST"]
