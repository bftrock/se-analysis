"""Reading LOGR | Solar diagnostic files, and the standard expectations for their rails.

A .diag file is the logger reporting on itself: one row a minute carrying the
Avg, Min, Max and SD of every internal supply rail and of the current drawn from
it. None of it measures the site, so none of it is interesting until something is
wrong -- which is exactly what makes a spec worth having, because nobody reads
these files until they are already looking for a fault.

SPEC is what a healthy file looks like. It is built by expanding tables of
nominal rail voltages and plausible currents across the four statistics rather
than written out column by column: the 68 columns are 17 signals seen four ways,
and a limit that applies to a rail applies to every statistic of it. Site-specific
deviations belong in the notebook, via SPEC.override(...).

The limits here are the logger's own, not physical ones, so unlike
se_analysis.protonode they live in this module rather than in
se_analysis.validation -- the exception being the site supply voltage, which is
the same 24 V rail a ProtoNode file reports and is held to the same band.
"""

import pandas as pd

from se_analysis.validation import LOGR_VOLTAGE, Spec

TIMESTAMP = "Stats_Timestamp"

# Each signal arrives as four columns, one per statistic. Avg, Min and Max are
# levels -- volts or milliamps the rail actually reached -- so they answer to the
# same limits. SD is a spread, and answers to its own.
LEVEL_STATS = ("Avg", "Min", "Max")
SPREAD_STAT = "SD"

# How far a regulated rail may sit from the volts it is named for. The usual
# tolerance band for a switching regulator, and wide next to what these rails
# actually do: across a full day the loosest of them stays within 1.4% of
# nominal, and most within half of that.
RAIL_TOLERANCE = 0.05

# How much a rail may wander within a single minute, as a fraction of nominal.
# Deliberately loose: over a healthy day the noisiest rail peaks at a quarter of
# a percent of nominal and the quietest at a thousandth, so this leaves even the
# tightest of them a factor of eight. It is here to catch a rail oscillating or
# an ADC rattling, not to police ordinary noise.
RIPPLE_TOLERANCE = 0.02

# Every voltage rail the logger reports, by the volts it is meant to hold. The
# sign is part of the nominal: the logger runs split supplies, so -15V_V reading
# -15.13 is healthy and the same rail reading +15.13 is the channel inverted.
RAIL_VOLTS = {
    "VIN_RP_V": 24.0,
    "+VIN_RP": 24.0,
    "-VIN_RP_V": -24.0,
    "13_3V_V": 13.3,
    "12_4V_V": 12.4,
    # The two COM excitation outputs are fed from the 12.4 V rail and read a few
    # tenths under it, which the tolerance band covers comfortably.
    "COMA EXC_V": 12.4,
    "COMB EXC_V": 12.4,
    "15V_V": 15.0,
    "-15V_V": -15.0,
    "12V_V": 12.0,
    "5V_V": 5.0,
    "3_3V_V": 3.3,
}

# Rails that are the site supply rather than something regulated down from it.
# They sag and float with the battery and the charge controller, so holding them
# to a regulator's tolerance would flag an ordinary night; they get the band the
# logger's supply voltage is held to everywhere else instead.
SUPPLY_RAILS = ("VIN_RP_V", "+VIN_RP", "-VIN_RP_V")

# What each rail may draw, in mA. Current is load, not regulation -- it moves
# with whatever the logger and its sensors are doing -- so these are sized from
# what a healthy logger draws with room above it, and catch a short or a dead
# rail rather than a change in how busy the logger is.
RAIL_CURRENT = (0, 500)
EXCITATION_CURRENT = (0, 200)

CURRENT_DRAW = {
    "12_4V_mA": RAIL_CURRENT,
    "3_3V_mA": RAIL_CURRENT,
    "12V_mA": RAIL_CURRENT,
    "COMA EXC_mA": EXCITATION_CURRENT,
    "COMB EXC_mA": EXCITATION_CURRENT,
}


def _band(rail: str) -> tuple[float, float]:
    """The volts `rail` is allowed to read, sign and all.

    Worked out in absolute volts and flipped at the end, so a negative rail gets
    the same band the positive one of that size would and nothing has to be
    written down twice with the signs swapped by hand.
    """
    nominal = RAIL_VOLTS[rail]
    lo, hi = (
        LOGR_VOLTAGE
        if rail in SUPPLY_RAILS
        else (
            abs(nominal) * (1 - RAIL_TOLERANCE),
            abs(nominal) * (1 + RAIL_TOLERANCE),
        )
    )
    lo, hi = round(lo, 3), round(hi, 3)
    return (-hi, -lo) if nominal < 0 else (lo, hi)


def _ripple(rail: str) -> tuple[float, float]:
    """How far `rail` may spread within one statistics interval.

    A standard deviation, so unsigned however the rail itself reads.
    """
    return (0.0, round(abs(RAIL_VOLTS[rail]) * RIPPLE_TOLERANCE, 3))


RANGES = {
    **{f"{rail}_{stat}": _band(rail) for rail in RAIL_VOLTS for stat in LEVEL_STATS},
    **{f"{rail}_{SPREAD_STAT}": _ripple(rail) for rail in RAIL_VOLTS},
    # Current SD answers to the same band as the current itself rather than to a
    # ripple fraction. The draw genuinely swings with whatever the logger is
    # doing, so there is no quiet level to hold it to -- but a spread wider than
    # the whole plausible range of the signal is not a reading at all.
    **{
        f"{channel}_{stat}": limits
        for channel, limits in CURRENT_DRAW.items()
        for stat in (*LEVEL_STATS, SPREAD_STAT)
    },
}

# Every signal in the file, in the order the logger writes them.
SIGNALS = tuple(RAIL_VOLTS) + tuple(CURRENT_DRAW)

# Min <= Avg <= Max, which holds by construction for every signal in the file and
# so says nothing about the hardware -- it says the row is intact. A LOGR data row
# that came up short shifts every later field one column left, and the values that
# land are all numeric and mostly in range (see logr.malformed_rows); the ordering
# of a signal's own statistics is what that shift breaks first.
LESSER = tuple(
    pair
    for signal in SIGNALS
    for pair in (
        (f"{signal}_Min", f"{signal}_Avg"),
        (f"{signal}_Avg", f"{signal}_Max"),
    )
)

# Signals that should never sit still. Only the currents qualify: a regulated
# rail sitting perfectly still is a regulated rail working, and on the quietest
# of them the Min and Max columns land on the same ADC code for the better part
# of an hour. Current moves with whatever the logger is doing, so a current that
# stops moving is a reading that stopped being taken.
DYNAMIC = tuple(
    f"{channel}_{stat}" for channel in CURRENT_DRAW for stat in ("Avg", SPREAD_STAT)
)

# Diagnostic rows arrive once a minute, well over validation's default, and these
# currents hold a value for two minutes at a time in a healthy file. Long enough
# to leave that alone, short enough that a frozen channel surfaces within the
# quarter hour.
MAX_STALL = pd.Timedelta("15min")

# No solar block: a .diag file carries no solar angles and no position to check
# them against. The site coordinates live in the header rather than the data --
# see logr.parse_site_info, and solar.get_normalization_window for reading the
# position straight out of it.
SPEC = Spec(
    ranges=RANGES,
    lesser=LESSER,
    dynamic=DYNAMIC,
    max_stall=MAX_STALL,
)


def indexed(data: pd.DataFrame) -> pd.DataFrame:
    """A diagnostic frame indexed by its statistics timestamp, oldest first.

    Takes `reader.data` from logr.read_files(directory, file_type="diag"), which
    is left to the caller so the reader -- and the Site_info on it -- stays to
    hand. Stamps are written with a Z and so come back tz-aware in UTC, which is
    what the rest of the checks want and what these files, unlike the ProtoNode
    exports, already are.
    """
    stamped = data.copy()
    stamped[TIMESTAMP] = pd.to_datetime(stamped[TIMESTAMP], utc=True)
    return stamped.set_index(TIMESTAMP).sort_index()
