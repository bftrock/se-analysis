"""Reading LOGR | Solar measurement files, and what their channels should read.

A measurement file is the logger reporting on the site rather than on itself: one
row per statistical interval carrying the Avg, Min, Max and SD of every channel
wired to the tower, or one row a second carrying the samples themselves.

Unlike a ProtoNode export or a .diag file, no two measurement files have the same
columns. The channels, the sensors on them and the names they are given are
whatever the site was configured with, so there is no SPEC to write down here --
spec_for() builds one per file instead.

What makes that possible is that the file says what each channel measures. Every
Sensor History block in the header carries a Description, sometimes a Measurand,
and Units; between them they identify a kind of measurement, and a kind is what
carries limits. So the limits are written down once per kind, and RULES does the
site-specific work of deciding which kind a channel is.

channels() is that spec read the other way round, by channel rather than by
check, and is the place to start on an unfamiliar file. A channel whose kind
cannot be read off the header gets no checks rather than guessed ones, and
unclassified() lists those, so nothing goes unchecked silently.

Two things about a day of site data that a short ProtoNode capture never raised:

Irradiance sits at zero all night, legitimately, and so does everything derived
from it. A stall check over a whole day would report every radiometer on the site
as stuck, and the albedo -- a ratio of two numbers that are both nearly zero --
wanders past 1 every night at every site. Those checks are held back for a second
pass over the daylight rows alone: spec_for(sunlit=True) is that pass's spec and
daylight() is its rows.

And the angles in these files are the logger's own geometric ones, computed
without refraction, which is why solar_angles() pins refract=False. Against them
se_analysis.solar agrees to a third of a degree in elevation across a full day
and three quarters of a degree in azimuth, so the tolerance here is a degree --
loose enough to leave two ephemerides alone, tight enough that a configured site
position out by a degree still shows.
"""

import re
from dataclasses import dataclass

import pandas as pd

from se_analysis import logr, solar, validation
from se_analysis.logr import STATISTICS
from se_analysis.validation import SolarAngles, Spec

TIMESTAMP = "Timestamp"

# The statistics a channel's columns are spread across. Avg, Min, Max and a Gust
# are levels -- what the signal actually reached -- and so are the Sum and Total
# a totalizer accumulates, so all of them answer to the channel's own band. SD is
# a spread and gets a band derived from it. A one-second file spreads a channel
# across nothing at all: its single column is the sample, and reads here as a
# level with no statistic to name, LEVEL_ONLY.
LEVEL_ONLY = ""
LEVELS = (LEVEL_ONLY, "Avg", "Min", "Max", "Gust", "Sum", "Total")
SPREAD = "SD"
EXTREMES = ("Min", "Max")
ACCUMULATIONS = ("Sum", "Total")

# The column that stands for a channel where only one of them can: its Avg, or
# in a one-second file the single column it has.
PRIMARY = ("Avg", LEVEL_ONLY)

# Rows arrive once a statistical interval -- a minute in every file seen here --
# so validation's 20-second default sits below the interval and would register
# every sample as a stall. This is the window a channel with nothing quantised
# about it gets: long enough to sit out a quiet stretch, short enough that a dead
# channel surfaces within the half hour.
MAX_STALL = pd.Timedelta("30min")

# Windows for the channels that hold a value legitimately, each sized above the
# longest run in a healthy day rather than off the sensor's resolution: 9 minutes
# of SR30 internal pressure, and a wind speed that can genuinely report its own
# offset through an hour of a calm night.
PRESSURE_STALL = pd.Timedelta("1h")
WIND_STALL = pd.Timedelta("2h")

# A sensor's housekeeping channels -- the temperature, humidity and speed inside
# its own housing -- move slowly and some are reported only to a tenth of a unit,
# so sitting on one value for a couple of hours is what a dry, ventilated
# instrument does: the longest healthy run across the sites here is 2h25m of an
# R2-D's internal humidity. This window is above that and still well under a
# file, because the fault worth catching is a Modbus sensor that dropped out and
# started reading zero -- which holds its value for the whole record.
HOUSEKEEPING_STALL = pd.Timedelta("4h")

# What a radiometer reads with nothing to measure. A thermopile carries its own
# thermal offset at night and a silicon cell an amplifier offset, either of which
# can sit a little under zero -- the reflected channels here reach -0.63 W/m^2,
# and a few W/m^2 is ordinary on a clear night. Zero is the physical floor but
# not the reading's, so the irradiance bands are dropped this far below it rather
# than reporting every clear night as out of range.
RADIOMETER_OFFSET = 5.0

# The same allowance for a voltage channel referred to its own zero, sized to the
# couple of millivolts the grid monitor here sits at when there is nothing on it.
VOLTAGE_OFFSET = 0.5

# How far a recorded solar angle may sit from the computed one. Wider than
# validation's default because these angles are the logger's own and a simpler
# calculation than se_analysis.solar's: across a full day at one site the two
# agree to 0.35 degrees in elevation and 0.77 in azimuth, which is the algorithm
# rather than the site. A degree still catches a configured position out by
# about a degree, latitude moving elevation one for one and four minutes of
# longitude moving it about as much.
SOLAR_TOLERANCE = 1.0

# How far above the horizon the sun has to be for a radiometer's reading to mean
# something. Low sun is where a pyranometer's cosine response is worst and where
# an albedo -- a ratio of two small numbers -- stops being a ratio at all. Ten
# degrees is past both and still leaves most of the day.
DAYLIGHT_ELEVATION = 10.0

# What Kind.sunlit may name: the checks that only mean anything with the sun up.
SUNLIT_CHECKS = ("range", "stall")


@dataclass(frozen=True)
class Kind:
    """One kind of measurement: the band it reads in, and how it behaves.

    `limits` is the band every level statistic of a channel answers to -- its
    Avg, its Min, its Max, a Gust, a Sum. The SD band is derived from it rather
    than written down per kind (see _spread): a spread wider than the whole
    plausible range of the signal is not a reading at all, and there is nothing
    tighter to say that holds for a minute of broken cloud as well as a minute of
    clear sky.

    `dynamic` marks a channel that should never sit still, and `stall_window` how
    long it may hold a value before that counts. A coarsely quantised channel
    legitimately sits on one code for a while -- an SR30 reports its internal
    humidity to a hundredth of a percent and moves it by less than that on a
    still night -- and wants a window of its own rather than a looser default
    that would also hide a dead irradiance channel.

    `extremes` is False for the channels whose Min and Max columns the logger
    does not fill in. A wind vane is the case to know about: it reports a mean
    direction and a gust direction and writes zero into Min and Max. Those zeros
    are not readings, so they are neither ranged nor ordered against the Avg --
    the alternative being 1440 spurious findings a day, one per row.

    `sunlit` names which of this kind's checks belong to the daylight pass rather
    than the whole-file one, from SUNLIT_CHECKS. An irradiance has a band that
    holds at midnight and a stall that means nothing there, so only its stall
    moves; an albedo's band does not hold at midnight either, so both do.
    """

    name: str
    limits: tuple
    dynamic: bool = False
    stall_window: pd.Timedelta | None = None
    extremes: bool = True
    sunlit: tuple = ()


def _floor(limits: tuple, offset: float) -> tuple:
    """`limits` with the bottom dropped by `offset`, for a zero a reading undershoots."""
    lo, hi = limits
    return (lo - offset, hi)


def _spread(limits: tuple) -> tuple:
    """The band an SD answers to, given the band the signal itself answers to."""
    lo, hi = limits
    return (0.0, float(hi - lo))


# Irradiance, and the two kinds of it that answer to a different ceiling: a
# channel facing the ground sees a fraction of what fell on it, and a
# pyrheliometer aimed at the sun sees more than any horizontal plane does.
IRRADIANCE = Kind(
    "irradiance",
    _floor(validation.GHI_IRRADIANCE, RADIOMETER_OFFSET),
    dynamic=True,
    sunlit=("stall",),
)
REFLECTED_IRRADIANCE = Kind(
    "reflected irradiance",
    _floor(validation.ALBEDO_IRRADIANCE, RADIOMETER_OFFSET),
    dynamic=True,
    sunlit=("stall",),
)
DIRECT_IRRADIANCE = Kind(
    "direct irradiance",
    _floor(validation.DNI_IRRADIANCE, RADIOMETER_OFFSET),
    dynamic=True,
    sunlit=("stall",),
)

# An albedo is reflected over incident, so it is a fraction by construction and
# nothing about it survives the sun going down: both terms fall to their own
# offsets and the ratio of those is whatever it happens to be. Band and stall
# both belong to the daylight pass.
ALBEDO = Kind("albedo", validation.RATIO, dynamic=True, sunlit=SUNLIT_CHECKS)

# A soiling station's own arithmetic: the ratio of what the soiled coupon makes
# to what the clean one does. Not dynamic -- the daily figures are computed once
# a day and hold that value until the next one.
SOILING_RATIO = Kind("soiling ratio", validation.ISC_SOILING_RATIO)

# A DustIQ reports soiling as the percentage of transmission it cost, so the band
# is the whole of a percentage.
SOILING = Kind("soiling", (0, 100))

# A day of irradiance integrated. Fifteen kWh/m^2 is more than the sunniest place
# on the planet receives on its longest day, and the channel holds yesterday's
# total until midnight, so it is not dynamic either.
INSOLATION = Kind("insolation", (0, 15))

TEMPERATURE = Kind("temperature", validation.TEMPERATURE, dynamic=True)

# A radiometer's own body temperature, which is a temperature like any other and
# answers to the same band, but which sits inside a housing rather than in the
# weather and so is allowed to stop moving for longer.
INSTRUMENT_TEMPERATURE = Kind(
    "instrument temperature",
    validation.TEMPERATURE,
    dynamic=True,
    stall_window=HOUSEKEEPING_STALL,
)

# Every humidity in these files is housekeeping: the sealed air inside a
# radiometer, or the ambient reading of a sensor that answers just as slowly.
RELATIVE_HUMIDITY = Kind(
    "relative humidity",
    validation.RELATIVE_HUMIDITY,
    dynamic=True,
    stall_window=HOUSEKEEPING_STALL,
)
BAROMETRIC_PRESSURE = Kind(
    "barometric pressure",
    validation.BAROMETRIC_PRESSURE,
    dynamic=True,
    stall_window=PRESSURE_STALL,
)
WIND_SPEED = Kind(
    "wind speed", validation.WIND_SPEED, dynamic=True, stall_window=WIND_STALL
)
WIND_DIRECTION = Kind(
    "wind direction",
    validation.WIND_DIRECTION,
    dynamic=True,
    stall_window=WIND_STALL,
    extremes=False,
)

# Tilt. A pyranometer measuring the global horizontal has to be level to half a
# degree; one aimed at the array reads the array's own tilt, as a magnitude.
LEVEL_TILT = Kind("level tilt", validation.GHI_TILT)
ARRAY_TILT = Kind("array tilt", validation.ARRAY_TILT)

# A soiling sensor reports its orientation as two signed angles about its own
# axes rather than as one tilt, so neither of them is a magnitude and the band
# has to take a negative. Wide on purpose: there is nothing to say about where a
# DustIQ should be pointed beyond it being pointed somewhere, the plane it sits in
# being the array's business rather than the sensor's.
AXIS_TILT = Kind("axis tilt", (-90, 90))

# A sensor facing the ground reads 180 degrees less the tilt of what it faces, so
# its band is the upward one mirrored -- widened a couple of degrees past level,
# because 180 is the middle of a downfacing sensor's noise rather than a limit it
# sits against. The reflected channels here read between 161 and 179 degrees.
DOWNWARD_TILT = Kind("downward tilt", (180 - validation.ARRAY_TILT[1], 182.0))

SOLAR_ZENITH = Kind("solar zenith", validation.SOLAR_ZENITH, dynamic=True)
SOLAR_ELEVATION = Kind("solar elevation", validation.SOLAR_ELEVATION, dynamic=True)
SOLAR_AZIMUTH = Kind("solar azimuth", validation.SOLAR_AZIMUTH, dynamic=True)

# What the NRG Grid Monitor accessory can report: its own 0 to 30 V span, not the
# 22 to 25 V a LOGR's own supply is held to. What is on the other end of this
# channel is a site supply that may legitimately be switched off, which is also
# why it is not dynamic -- at the site here it reads zero for most of the day.
MONITORED_VOLTAGE = Kind("monitored voltage", _floor((0, 30), VOLTAGE_OFFSET))

# A PV coupon's open-circuit voltage and short-circuit current, as a soiling
# station measures them. The coupons here run to 50 V and 2.4 A; the bands are
# wide enough to take a different module without being retuned, and catch only a
# reading that is not a reading. Voc barely moves with irradiance, where Isc is
# very nearly proportional to it, so only the current has to keep moving.
PV_VOLTAGE = Kind("open-circuit voltage", (0, 100))
PV_CURRENT = Kind("short-circuit current", (0, 15), dynamic=True, sunlit=("stall",))

# The ventilator on a Hukseflux radiometer runs near 8,800 rpm and the band is
# the decade it sits inside. Zero is in range deliberately: a stopped fan is what
# the stall check reports, and it reports it as a fan that stopped rather than as
# a speed no fan could reach.
FAN_SPEED = Kind("fan speed", (0, 12000), dynamic=True)

# Radiometer heater current, peaking at 340 mA here with the heater on. Not
# dynamic: a heater is meant to sit at zero for months, and does.
HEATER_CURRENT = Kind("heater current", (0, 500))

# Rainfall. A running total, reset hourly or at midnight, so a dry week holds it
# still and nothing about it is dynamic. Somewhere near 500 mm is the wettest day
# on record anywhere, so the band is one no real rainfall reaches rather than one
# a site approaches. A total is also a step function the logger does not
# summarise, so its Min and Max are not readings.
PRECIPITATION = Kind("precipitation", (0, 1000), extremes=False)
RAIN_INTENSITY = Kind("rainfall intensity", (0, 300))  # the gauges here peak at 96

# Unit spellings, which vary by file and by sensor: a statistical file writes an
# SR30's internal pressure as mbar where a one-second file writes the same
# channel as hPa, and the solar angle channels say Degrees where a tilt says deg.
# Matched stripped and lowercased, so only genuine differences are listed.
IRRADIANCE_UNITS = ("w/m^2", "w/sqm")
TEMPERATURE_UNITS = ("deg_c",)
PRESSURE_UNITS = ("hpa", "mbar")
ANGLE_UNITS = ("deg", "degrees")
PERCENT_UNITS = ("%",)
SPEED_UNITS = ("m/s",)
RAINFALL_UNITS = ("mm",)
RAINFALL_RATE_UNITS = ("mm/hour",)
INSOLATION_UNITS = ("kwh/m^2",)
FAN_UNITS = ("rpm",)
MILLIAMP_UNITS = ("ma",)
VOLT_UNITS = ("v",)
AMP_UNITS = ("a",)

# A channel the logger derives rather than reads carries no unit at all, or the
# literal "NA".
UNITLESS = ("", "na")

# A channel facing down at the ground rather than up at the sky. The R is for
# reflected: RHI is the reflected horizontal, RPOA the reflected plane of array.
# Every reflected channel across the sites here is named with one or the other.
REFLECTED = ("rhi", "rpoa")


def _present(text: str, want) -> bool:
    """Whether `want` -- a substring, or a tuple of alternatives -- is in `text`."""
    if isinstance(want, tuple):
        return any(_present(text, alternative) for alternative in want)
    return want in text


@dataclass(frozen=True)
class Rule:
    """One step of the classifier: what has to hold for `kind` to be the answer.

    `units` is the set of unit spellings a channel may carry for this rule to
    apply, empty meaning any. `words` are matched against the Description and
    Measurand together and all of them have to appear, an entry written as a
    tuple meaning any one of its alternatives will do.

    Rules are tried in order and the first match wins, so a specific rule goes
    ahead of the general one it is an exception to: a reflected irradiance before
    an irradiance, a level pyranometer's tilt before a tilt.
    """

    kind: Kind
    units: tuple = ()
    words: tuple = ()

    def matches(self, units: str, text: str) -> bool:
        if self.units and units not in self.units:
            return False
        return all(_present(text, word) for word in self.words)


RULES = (
    # The angles the logger works out for itself, named in the Description and
    # with no sensor or serial number behind them.
    Rule(SOLAR_ZENITH, ANGLE_UNITS, words=("zenith",)),
    Rule(SOLAR_ELEVATION, ANGLE_UNITS, words=("elevation angle",)),
    Rule(SOLAR_AZIMUTH, ANGLE_UNITS, words=("azimuth",)),
    Rule(REFLECTED_IRRADIANCE, IRRADIANCE_UNITS, words=(REFLECTED,)),
    Rule(DIRECT_IRRADIANCE, IRRADIANCE_UNITS, words=("dni",)),
    Rule(IRRADIANCE, IRRADIANCE_UNITS),
    # Which tilt band applies depends on what the sensor is pointed at, so the
    # named cases come before the general one.
    Rule(AXIS_TILT, ANGLE_UNITS, words=(("tilt x", "tilt y"),)),
    Rule(DOWNWARD_TILT, ANGLE_UNITS, words=("tilt", REFLECTED)),
    Rule(LEVEL_TILT, ANGLE_UNITS, words=("tilt", "ghi")),
    Rule(ARRAY_TILT, ANGLE_UNITS, words=("tilt",)),
    # The remaining angle is a direction, whether it comes off a vane of its own
    # or out of an all-in-one weather sensor.
    Rule(WIND_DIRECTION, ANGLE_UNITS, words=(("vane", "wind direction"),)),
    Rule(WIND_SPEED, SPEED_UNITS),
    # A sensor's own body temperature before a temperature of the weather: the
    # band is the same and only the stall window differs.
    Rule(
        INSTRUMENT_TEMPERATURE,
        TEMPERATURE_UNITS,
        words=(("body temp", "instrument temperature", "internal temp"),),
    ),
    Rule(TEMPERATURE, TEMPERATURE_UNITS),
    Rule(BAROMETRIC_PRESSURE, PRESSURE_UNITS),
    # A DustIQ reports a percentage of transmission lost; anything else
    # reporting a percentage is a humidity.
    Rule(SOILING, PERCENT_UNITS, words=("soiling",)),
    Rule(RELATIVE_HUMIDITY, PERCENT_UNITS),
    Rule(RAIN_INTENSITY, RAINFALL_RATE_UNITS),
    Rule(PRECIPITATION, RAINFALL_UNITS),
    Rule(INSOLATION, INSOLATION_UNITS),
    Rule(FAN_SPEED, FAN_UNITS),
    Rule(HEATER_CURRENT, MILLIAMP_UNITS),
    # Volts and amps belong to a soiling station's PV coupon, bar the one
    # accessory that monitors a supply.
    Rule(MONITORED_VOLTAGE, VOLT_UNITS, words=("grid monitor",)),
    Rule(PV_VOLTAGE, VOLT_UNITS, words=("voc",)),
    Rule(PV_CURRENT, AMP_UNITS, words=("isc",)),
    # The derived ratios, which carry no units to be told apart by.
    Rule(ALBEDO, UNITLESS, words=("albedo",)),
    Rule(SOILING_RATIO, UNITLESS, words=("soiling",)),
)

_CHANNEL_STAT = re.compile(r"^Ch(\d+)_([^_]+)")


def _value(block: dict[str, str], key: str) -> str:
    """One Sensor History field, with absent and unfilled fields both empty.

    A field the logger left out of a block can still be present carrying an empty
    value, which pandas before 3.0 renders as the string "nan" once the column is
    cast to str. All three mean "not given".
    """
    value = block.get(key, "").strip()
    return "" if value == "nan" else value


def channel_columns(data: pd.DataFrame) -> dict[str, dict[str, str]]:
    """{channel: {statistic: column}} for the `Ch<n>_` columns of `data`.

    Works before renaming and after, which is what lets a spec be built against
    either: a LOGR export puts the statistic in the token straight after the
    channel number both ways round -- `Ch1_Avg_deg_C` and
    `Ch1_Avg_Pt1000_deg_C`. Reading it by position and not by vocabulary alone
    matters, because a renamed column can carry a second statistic token inside
    the sensor's own name: `Ch302_Avg_Daily_Sum_mm` is an Avg, and only the first
    of its two tokens is this column's statistic.

    A one-second file gives a channel one column rather than several -- spelt
    `Ch1_Samples_deg_C` before renaming and `Ch1_Pt1000_deg_C` after -- and both
    land under LEVEL_ONLY, the sample itself having no statistic to name.
    """
    found: dict[str, dict[str, str]] = {}
    for column in data.columns:
        match = _CHANNEL_STAT.match(str(column))
        if not match:
            continue
        channel, token = match.groups()
        statistic = token if token in STATISTICS else LEVEL_ONLY
        found.setdefault(channel, {})[statistic] = str(column)
    return found


def channel_kinds(site_info: pd.DataFrame) -> dict[str, Kind]:
    """{channel: Kind} for every header block whose kind RULES can read.

    Channels RULES cannot place are absent rather than present carrying a guess,
    so a caller asking what a channel measures gets an answer it can trust, and
    unclassified() is what lists the rest.
    """
    kinds: dict[str, Kind | None] = {}
    for channel, block in logr.channel_blocks(site_info):
        units = _value(block, "Units").lower()
        text = " ".join(
            part
            for part in (_value(block, "Description"), _value(block, "Measurand"))
            if part
        ).lower()
        # last block wins, as in logr.channel_names: a channel reconfigured part
        # way through the record is whatever it was reconfigured into
        kinds[channel] = next(
            (rule.kind for rule in RULES if rule.matches(units, text)), None
        )
    return {channel: kind for channel, kind in kinds.items() if kind is not None}


def _primary(members: dict[str, str]) -> str | None:
    """The one column that stands for a channel, or None if it has no level at all."""
    return next((members[stat] for stat in PRIMARY if stat in members), None)


def solar_angles(data: pd.DataFrame, site_info: pd.DataFrame) -> SolarAngles | None:
    """The recorded solar angles, to check against the position in the header.

    None when the file carries no angle channels, or no position to check them
    against. Otherwise pinned three ways, all of them properties of these files
    rather than choices:

    The position is the header's Latitude and Longitude, passed as numbers. A
    measurement file states where the site is once, in its header, and never in
    its data -- so unlike a ProtoNode capture there is no column to name, and the
    check doubles as a check on the header.

    `utc_offset` is pinned to zero because these stamps are written with a Z and
    are UTC. That makes a logger clock error read as an angle error rather than
    as a timezone, which is the right way round here: there is no timezone left to
    discover. It also sidesteps the half interval a statistical stamp sits ahead
    of the average it labels, which as an inferred offset would read as thirty
    seconds of clock skew and as an angle is a tenth of a degree.

    `refract` is off because the logger computes geometric angles -- the sun's
    true position, not the one the atmosphere bends into view. That costs up to
    half a degree at the horizon, so it is worth getting right rather than
    absorbing into the tolerance.
    """
    kinds = channel_kinds(site_info)
    columns = channel_columns(data)

    named = {}
    for angle, kind in (
        ("zenith", SOLAR_ZENITH),
        ("elevation", SOLAR_ELEVATION),
        ("azimuth", SOLAR_AZIMUTH),
    ):
        for channel, members in columns.items():
            if kinds.get(channel) is kind and (column := _primary(members)):
                named[angle] = column
                break
    if not named:
        return None

    try:
        latitude, longitude = solar.site_coordinates(site_info)
    except (KeyError, ValueError):
        return None  # a header that cannot say where the site is

    return SolarAngles(
        latitude=latitude,
        longitude=longitude,
        tolerance=SOLAR_TOLERANCE,
        utc_offset=pd.Timedelta(0),
        refract=False,
        **named,
    )


def spec_for(data: pd.DataFrame, site_info: pd.DataFrame, sunlit: bool = False) -> Spec:
    """What a healthy measurement file looks like, built from its own header.

    `data` supplies the column names and `site_info` says what each channel
    measures, so every check in the returned spec is aimed at a column that
    exists. Either form of frame works, but the findings only read as anything on
    one indexed() has renamed.

    `sunlit` picks which of the two passes this spec is for. The whole-file pass,
    the default, runs every check that means something at midnight; the sunlit
    pass runs the rest and wants the rows of daylight() rather than the whole
    frame. No check is in both, so the two sets of findings concatenate into one
    report without double-counting anything.

    Ranges go on every level statistic a channel has, since each catches
    something different -- a Max out of band above an Avg that is inside it is a
    spike, not a dead channel. Stalls go on the Avg alone, because the Min and
    Max of a quietly quantised channel legitimately land on the same code for
    long stretches while the Avg carries on moving, and because one finding per
    stalled channel is what you go and look at.

    The Min <= Avg <= Max ordering holds by construction for every channel the
    logger populates it for, so it says nothing about the sensors -- it says the
    row is intact. A LOGR data row that came up short shifts every later field
    one column left (see logr.malformed_rows) and the values that land are all
    numeric and mostly in range; a signal's own statistics falling out of order is
    what that shift breaks first. Gust <= Max rides along with it: a gust is an
    average over a few seconds, so it cannot exceed the highest single sample.

    Site-specific deviations belong in the notebook, via spec_for(...).override().
    """
    kinds = channel_kinds(site_info)
    ranges, dynamic, lesser, stall_windows = {}, [], [], {}

    for channel, members in channel_columns(data).items():
        kind = kinds.get(channel)
        if kind is None:
            continue

        # A totalizer's Min, Max and SD are left at zero the way a vane's are:
        # what the logger has to report about an accumulation is the accumulation.
        extremes = kind.extremes and not any(stat in members for stat in ACCUMULATIONS)
        levels = [
            stat
            for stat in LEVELS
            if stat in members and (extremes or stat not in EXTREMES)
        ]

        if ("range" in kind.sunlit) == sunlit:
            for stat in levels:
                ranges[members[stat]] = kind.limits
            if SPREAD in members:
                ranges[members[SPREAD]] = _spread(kind.limits)

        if kind.dynamic and ("stall" in kind.sunlit) == sunlit:
            if primary := _primary(members):
                dynamic.append(primary)
                if kind.stall_window is not None:
                    stall_windows[primary] = kind.stall_window

        if not sunlit and extremes and {"Min", "Avg", "Max"} <= set(members):
            lesser.append((members["Min"], members["Avg"]))
            lesser.append((members["Avg"], members["Max"]))
            if "Gust" in members:
                lesser.append((members["Gust"], members["Max"]))

    return Spec(
        ranges=ranges,
        lesser=tuple(lesser),
        dynamic=tuple(dynamic),
        max_stall=MAX_STALL,
        stall_windows=stall_windows,
        # The daylight rows are missing every night by construction, so the gap
        # check belongs to the pass that has the whole record to look at. So do
        # the angles: they reconcile with the computed position all night as well
        # as all day, once refraction is out of the way.
        solar=None if sunlit else solar_angles(data, site_info),
        gaps=not sunlit,
    )


CHANNEL_COLUMNS = [
    "channel",
    "description",
    "measurand",
    "units",
    "statistics",
    "kind",
    "limits",
    "dynamic",
]


def channels(data: pd.DataFrame, site_info: pd.DataFrame) -> pd.DataFrame:
    """One row per channel in `data`: what the header says, and what RULES made of it.

    The spec spec_for() builds, read the other way round -- by channel rather
    than by check -- which is where to look first when a finding names a column
    you do not recognise, or when a channel you expected to be checked was not.
    A channel RULES could not place carries an empty kind and no limits; those
    rows on their own are unclassified().
    """
    kinds = channel_kinds(site_info)
    blocks = dict(logr.channel_blocks(site_info))
    rows = []
    for channel, members in channel_columns(data).items():
        block = blocks.get(channel, {})
        kind = kinds.get(channel)
        rows.append(
            {
                "channel": channel,
                "description": _value(block, "Description"),
                "measurand": _value(block, "Measurand"),
                "units": _value(block, "Units"),
                "statistics": " ".join(sorted(s for s in members if s)),
                "kind": kind.name if kind else "",
                "limits": kind.limits if kind else None,
                "dynamic": bool(kind and kind.dynamic),
            }
        )
    return pd.DataFrame(rows, columns=CHANNEL_COLUMNS)


def unclassified(data: pd.DataFrame, site_info: pd.DataFrame) -> pd.DataFrame:
    """The channels of `data` that RULES could not place, and what the header says.

    A kind is what carries limits, so these are the channels spec_for() left out
    altogether: present in the file, plotted like any other, and checked by
    nothing. Read them and either widen RULES, if the kind is one this module
    ought to know, or put the limits in the spec in the notebook, if it is one
    site's own. Either way it is here rather than passing quietly.
    """
    listing = channels(data, site_info)
    return listing[listing["kind"] == ""].reset_index(drop=True)


def indexed(data: pd.DataFrame, site_info: pd.DataFrame) -> pd.DataFrame:
    """A measurement frame under its channel labels, indexed by stamp, oldest first.

    Takes `reader.data` and `reader.Site_info` from
    logr.read_files(directory, file_type="statistical"), which is left to the
    caller so the reader stays to hand.

    Columns are renamed from `Ch<n>_<stat>_<units>` to the labels
    logr.channel_names() builds out of the header, because a finding against
    `Ch108_Avg_W/m^2` says nothing about where to go and look and one against
    `Ch108_Avg_Hukseflux_SR30-POA_Irradiance_W/m^2` says the POA pyranometer.
    Stamps are written with a Z and so come back tz-aware in UTC, which is what
    the checks want and what these files, unlike the ProtoNode exports, already
    are.
    """
    renamed = logr.rename_channels(data, logr.channel_names(site_info))
    renamed[TIMESTAMP] = pd.to_datetime(renamed[TIMESTAMP], utc=True)
    return renamed.set_index(TIMESTAMP).sort_index()


def daylight(
    data: pd.DataFrame,
    site_info: pd.DataFrame,
    min_elevation: float = DAYLIGHT_ELEVATION,
) -> pd.DataFrame:
    """The rows of `data` with the sun more than `min_elevation` above the horizon.

    The rows spec_for(sunlit=True) gets checked over. The sun's position is
    computed from the header coordinates rather than read out of the file's own
    angle channels, so a logger whose angles are wrong does not get to choose
    which rows its irradiance is judged on.

    Expects the UTC index indexed() returns.
    """
    latitude, longitude = solar.site_coordinates(site_info)
    position = solar.position(data.index, latitude, longitude)
    return data[position["elevation"].to_numpy() > min_elevation]
