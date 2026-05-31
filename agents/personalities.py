"""Player personality profiles, keyed by squad index (0-10) per team.

The squad index lines up with the shared role contract used across the project
(``replay.logger.ROLES`` and ``simulator.narrator.ROLE_LABELS``):

    0=GK 1=CB 2=CB 3=LB 4=RB 5=CM 6=CM 7=CM 8=LW 9=RW 10=ST

Each profile carries an ``expected_zone`` band the coach uses to detect a player
drifting out of position, plus free-text ``traits``/``tendency`` that flavor the
prompt handed to the player sub-agent.
"""

from __future__ import annotations


# Coarse pitch bands a role is "supposed" to occupy, in the team's
# attacking-toward-+x frame (the same frame the narrator reports in).
ZONE_DEFENSIVE = "defensive"
ZONE_MIDFIELD = "midfield"
ZONE_ATTACKING = "attacking"

# Role label per squad index — must match simulator.narrator.ROLE_LABELS so the
# coach can line personalities up with parsed narrator lines.
_ROLE_LABELS: tuple[str, ...] = (
    "GK",
    "CB (left)",
    "CB (right)",
    "LB",
    "RB",
    "CM (left)",
    "CM (center)",
    "CM (right)",
    "LW",
    "RW",
    "ST",
)


def _person(pid: int, name: str, expected_zone: str, traits: list[str], tendency: str) -> dict:
    """Assemble one personality profile record (role is derived from ``pid``)."""
    return {
        "id": pid,
        "name": name,
        "role": _ROLE_LABELS[pid],
        "expected_zone": expected_zone,
        "traits": traits,
        "tendency": tendency,
    }


# Home squad: the named Liverpool-style XI from the task brief (Sterling at LW is
# off-era but kept verbatim per the brief).
_LIVERPOOL: dict[int, dict] = {
    0: _person(0, "Alisson", ZONE_DEFENSIVE, ["calm", "commanding"],
               "Sweeper-keeper; stays home but starts attacks with the ball at his feet."),
    1: _person(1, "Van Dijk", ZONE_DEFENSIVE, ["composed", "dominant"],
               "Anchors the back line; steps up to intercept but rarely strays from his half."),
    2: _person(2, "Gomez", ZONE_DEFENSIVE, ["quick", "recovery pace"],
               "Covers in behind; uses pace to recover rather than diving in."),
    3: _person(3, "Robertson", ZONE_DEFENSIVE, ["relentless", "high-energy"],
               "Overlaps hard down the left but must recover defensively when out of possession."),
    4: _person(4, "Alexander-Arnold", ZONE_DEFENSIVE, ["creative", "ambitious"],
               "Whips in early crosses; prone to being caught upfield after attacks break down."),
    5: _person(5, "Fabinho", ZONE_MIDFIELD, ["disciplined", "screening"],
               "Defensive anchor; shields the back four and seldom ventures forward."),
    6: _person(6, "Henderson", ZONE_MIDFIELD, ["box-to-box", "vocal"],
               "Drives between boxes; organizes the press."),
    7: _person(7, "Wijnaldum", ZONE_MIDFIELD, ["press-resistant", "progressive"],
               "Carries through pressure and links midfield to attack."),
    8: _person(8, "Sterling", ZONE_ATTACKING, ["direct", "fast"],
               "Runs in behind on the left; should stay high and wide to stretch the defense."),
    9: _person(9, "Salah", ZONE_ATTACKING, ["clinical", "incisive"],
               "Hugs the right touchline then cuts inside to shoot; lives in the final third."),
    10: _person(10, "Firmino", ZONE_ATTACKING, ["selfless", "link-up"],
                "False-nine who drops to link play, but leads the line in the attacking third."),
}

# Away squad: a distinct generic opposition XI so the two coaches manage
# different personalities.
_RIVALS: dict[int, dict] = {
    0: _person(0, "Keita-Keeper", ZONE_DEFENSIVE, ["shot-stopper", "vocal"],
               "Line-keeper; stays on his line and commands the box."),
    1: _person(1, "Stone", ZONE_DEFENSIVE, ["aggressive", "physical"],
               "Front-foot defender; steps out to win the ball early."),
    2: _person(2, "Walls", ZONE_DEFENSIVE, ["positional", "reads play"],
               "Holds the line and sweeps space behind."),
    3: _person(3, "Dash", ZONE_DEFENSIVE, ["pacey", "raiding"],
               "Bombs forward on the left; must track back when possession is lost."),
    4: _person(4, "Croft", ZONE_DEFENSIVE, ["steady", "tucks in"],
               "Conservative full-back; rarely overlaps."),
    5: _person(5, "Pivot", ZONE_MIDFIELD, ["holding", "metronome"],
               "Deep-lying playmaker who dictates tempo from the base."),
    6: _person(6, "Engine", ZONE_MIDFIELD, ["tireless", "two-way"],
               "Covers ground box-to-box."),
    7: _person(7, "Spark", ZONE_MIDFIELD, ["creative", "ball-progressing"],
               "Threads passes and drives into the half-space."),
    8: _person(8, "Flash", ZONE_ATTACKING, ["tricky", "1v1"],
               "Beats his man on the left and stays high."),
    9: _person(9, "Bolt", ZONE_ATTACKING, ["rapid", "in-behind"],
               "Stretches the line with runs behind the full-back."),
    10: _person(10, "Striker", ZONE_ATTACKING, ["poacher", "clinical"],
                "Plays on the shoulder; stays central in the box."),
}

# Public mapping: team -> squad index -> profile.
PERSONALITIES: dict[str, dict[int, dict]] = {
    "home": _LIVERPOOL,
    "away": _RIVALS,
}


def squad_for(team: str) -> dict[int, dict]:
    """Return the personality profiles for ``team`` ("home" or "away")."""
    try:
        return PERSONALITIES[team]
    except KeyError as err:
        msg = f"unknown team {team!r}; expected one of {tuple(PERSONALITIES)}"
        raise ValueError(msg) from err
