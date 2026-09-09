"""When a mast's lightning rod puts its shadow on a boom-mounted pyranometer.

A short met tower carries its pyranometer out on a boom and its lightning rod
straight up off the top. Twice a day the sun lines up with the boom, and for a
few minutes the rod sits between the sun and the sensor -- a notch in an
otherwise clean irradiance trace, at nearly the same clock time every day, that
reads as a sensor fault until you work out it is the tower shading itself.

The geometry that decides it is entirely local: only structure standing *above*
the sensor's own height can ever occlude it, so on a tower whose boom sits at the
top the rod is the only candidate, and the tower body below never enters into it.
That makes the question one segment against one point, which is what this module
solves.

`shadow_track` walks a single date; `shadow_season` walks a span of them, which
is how the pattern's drift through the year becomes visible. Both come back with
a *fraction of the solar disk blocked* rather than a shaded/not-shaded flag,
because at these distances the answer is almost never all-or-nothing: a half-inch
rod seen from four feet away subtends about the same angle as the sun itself, so
the fully-shadowed core is a fraction of an inch wide and everything either side
of it is partial. `shadow_windows` reduces a track to the handful of intervals
worth looking at in the data.

What is modelled is the direct beam only. Diffuse light arrives from the whole
sky and the rod blocks a negligible slice of that, so a blocked fraction f
removes roughly f * DNI * cos(incidence) from the reading and no more -- on a
clear day at low sun that is most of the signal, under overcast it is nothing.
The sensor is treated as a point, which slightly sharpens the peak and shortens
the window against a real 10 mm thermopile that averages over its own aperture.
"""

import matplotlib.dates as mdates
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

from se_analysis import plot, solar

# ── the site's hardware, in feet ─────────────────────────────────────────────
# Defaults describe the tower this module was written for. Every one of them is
# a parameter; they live here so a notebook can state only what differs.
TOWER_HT = 8.0
ROD_LEN = 1.5
BOOM_LEN = 4.0

# A 1/2-inch solid lightning rod. This is the single most load-bearing number in
# the whole calculation and the easiest one to get wrong -- the blocked fraction
# scales with it directly, and the difference between 3/8" and 5/8" stock is the
# difference between a 60% notch and a full one. Measure the rod.
ROD_DIAMETER = 0.5 / 12.0

# East, because a boom "running east-west" still has to point one way, and which
# way decides whether the notch lands in the morning or the afternoon. An
# east-pointing boom is shaded by the afternoon sun, which is behind it.
BOOM_AZIMUTH = 90.0

# The sun is not a point: it spans about half a degree, and that is why the
# shadow has a soft edge at all. It breathes between 0.524 deg at aphelion and
# 0.542 deg at perihelion -- a 3% swing that moves no conclusion here, so the
# mean stands in for it rather than being computed per date.
SUN_ANGULAR_DIAMETER = 0.533

# Below this the notch is not worth chasing in the data: a percent of the direct
# beam is inside the noise of a clean pyranometer trace, let alone a real one.
DEFAULT_THRESHOLD = 0.01

# A single date is walked finely because the event is short -- the shadow crosses
# the sensor in a few minutes and a coarse step can step straight over the peak.
TRACK_FREQ = "10s"

# A season is walked coarsely, because it is 365 times as much work and the
# question has changed: not how deep the notch goes to the percent, but which
# weeks of the year have one at all.
SEASON_FREQ = "1min"


def _sun_vector(elevation, azimuth) -> np.ndarray:
    """Unit vectors pointing at the sun, as (east, north, up) rows.

    Azimuth is degrees clockwise from true north, matching solar.position().
    """
    elev = np.deg2rad(np.asarray(elevation, dtype=float))
    azi = np.deg2rad(np.asarray(azimuth, dtype=float))
    return np.column_stack(
        [np.cos(elev) * np.sin(azi), np.cos(elev) * np.cos(azi), np.sin(elev)]
    )


def _angular_miss(
    sun: np.ndarray, offset: np.ndarray, along: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """How far off the sun sits from a segment, seen from the sensor.

    The segment runs from `offset` to `offset + along`, both measured from the
    sensor. Returns the smallest angle (radians) between the sun direction and
    any point of the segment, plus the distance to whichever point that was --
    which is what turns the rod's thickness into an angular width.

    Working in angles rather than projecting a shadow onto the sensor's plane is
    what keeps the segment's *ends* honest. A projection has to treat the rod as
    infinitely long and then patch up the case where the shadow falls short;
    here the top of the rod simply stops being in the way, which is exactly what
    happens as the sun climbs past the line from sensor to rod tip.
    """
    # cos of the angle to the point at parameter t is (a + b t) / |offset + t along|,
    # whose stationary point solves out in closed form. c, e and g are fixed --
    # only the sun moves.
    a = sun @ offset
    b = sun @ along
    c = offset @ offset
    e = offset @ along
    g = along @ along

    denominator = b * e - a * g
    # Degenerate where the sun lies square on the segment's own direction; the
    # endpoint candidates below cover it, so any placeholder in range will do.
    stationary = np.where(
        np.abs(denominator) > 1e-12,
        (a * e - b * c) / np.where(np.abs(denominator) > 1e-12, denominator, 1.0),
        0.0,
    )

    best_cos = np.full(len(sun), -1.0)
    best_distance = np.full(len(sun), np.inf)
    # The extremum may fall outside the rod, in which case the nearest approach
    # is at whichever end -- so the ends are always candidates too.
    for t in (np.clip(stationary, 0.0, 1.0), np.zeros(len(sun)), np.ones(len(sun))):
        distance = np.sqrt(np.maximum(c + 2.0 * e * t + g * t * t, 1e-24))
        cosine = (a + b * t) / distance
        better = cosine > best_cos
        best_cos = np.where(better, cosine, best_cos)
        best_distance = np.where(better, distance, best_distance)

    return np.arccos(np.clip(best_cos, -1.0, 1.0)), best_distance


def _disk_blocked(
    miss: np.ndarray, half_width: np.ndarray, sun_radius: float
) -> np.ndarray:
    """Fraction of the sun's disk hidden by a straight bar across it.

    The disk has angular radius `sun_radius`; the bar is `half_width` wide either
    side of its axis, and that axis passes `miss` from the disk's centre (all
    radians). Integrating the disk's chords across the overlapping strip gives a
    closed form, so the soft edge of the shadow comes out as a curve rather than
    a guessed ramp.

    The bar is taken as running clean across the disk, which the rod does except
    in the last instant before its tip clears the sun -- there the true figure is
    smaller than this returns, for the few seconds the tip is cutting the disk
    corner-wise instead of edge to edge.
    """
    low = np.clip(miss - half_width, -sun_radius, sun_radius)
    high = np.clip(miss + half_width, -sun_radius, sun_radius)

    def integral(x):
        """Area of the disk left of x, up to a constant."""
        return x * np.sqrt(
            np.maximum(sun_radius**2 - x**2, 0.0)
        ) + sun_radius**2 * (np.arcsin(np.clip(x / sun_radius, -1.0, 1.0)))

    return np.clip((integral(high) - integral(low)) / (np.pi * sun_radius**2), 0.0, 1.0)


def shade(
    index,
    latitude: float,
    longitude: float,
    tower_ht: float = TOWER_HT,
    rod_len: float = ROD_LEN,
    boom_len: float = BOOM_LEN,
    boom_azimuth: float = BOOM_AZIMUTH,
    sensor_ht: float | None = None,
    rod_diameter: float = ROD_DIAMETER,
    sun_diameter: float = SUN_ANGULAR_DIAMETER,
) -> pd.DataFrame:
    """How much of the sun the lightning rod hides, at every stamp in `index`.

    The tower stands at the origin with the rod rising from `tower_ht` to
    `tower_ht + rod_len` on its axis; the pyranometer sits `boom_len` from that
    axis along `boom_azimuth`, at `sensor_ht` -- the top of the tower unless said
    otherwise. Lengths are all in the same unit, feet by default; only their
    ratios matter.

    Pass a tz-aware `index` and the result keeps that timezone, which is the only
    way the answer is readable as a time of day. A tz-naive index is read as UTC,
    as everywhere else in this package.

    Columns:
      elevation, azimuth  where the sun was, degrees (azimuth from true north)
      miss                angle from the sun's centre to the rod's axis, degrees
      rod_width           the rod's own angular half-width from there, degrees
      reach               how far the rod's shadow falls at the sensor's height,
                          in boom units -- under `boom_len` it lands short
      blocked             fraction of the solar disk the rod covers, 0 to 1

    Only structure above `sensor_ht` can shade the sensor, so with the boom at the
    top of the tower the rod is the whole story. Drop `sensor_ht` below
    `tower_ht` and the tower body starts shading too -- that is not modelled
    here, and `blocked` will read low.
    """
    if sensor_ht is None:
        sensor_ht = tower_ht

    stamps = pd.DatetimeIndex(index)
    sky = solar.position(stamps, latitude, longitude)
    elevation = sky["elevation"].to_numpy()
    azimuth = sky["azimuth"].to_numpy()

    bearing = np.deg2rad(boom_azimuth)
    sensor = np.array(
        [boom_len * np.sin(bearing), boom_len * np.cos(bearing), sensor_ht], dtype=float
    )
    # the rod as seen from the sensor: base offset, then the run to its tip
    offset = np.array([0.0, 0.0, tower_ht], dtype=float) - sensor
    along = np.array([0.0, 0.0, rod_len], dtype=float)

    miss, distance = _angular_miss(_sun_vector(elevation, azimuth), offset, along)
    half_width = np.arctan(0.5 * rod_diameter / distance)
    blocked = _disk_blocked(miss, half_width, np.deg2rad(0.5 * sun_diameter))

    # A sun below the horizon has no beam to block, whatever the geometry says.
    blocked = np.where(elevation > 0.0, blocked, 0.0)

    # Reach is not used by the test above -- it is reported because it is the
    # number that explains the result. The shadow of the rod's tip lands
    # rod_len / tan(elevation) from the mast, so once the sun is higher than
    # arctan(rod_len / boom_len) the shadow cannot get out to the sensor at all.
    with np.errstate(divide="ignore", invalid="ignore"):
        reach = np.where(
            elevation > 0.0, rod_len / np.tan(np.deg2rad(elevation)), np.inf
        )

    return pd.DataFrame(
        {
            "elevation": elevation,
            "azimuth": azimuth,
            "miss": np.rad2deg(miss),
            "rod_width": np.rad2deg(half_width),
            "reach": reach,
            "blocked": blocked,
        },
        index=stamps,
    )


def grazing_elevation(rod_len: float = ROD_LEN, boom_len: float = BOOM_LEN) -> float:
    """Sun elevation above which the rod's shadow can never reach the sensor.

    The rod's tip throws its shadow rod_len / tan(elevation) out from the mast, so
    this is simply the angle at which that equals the boom -- and it is the one
    number that says whether a tower has this problem at all. Below it the
    shadow can land on the boom given the right azimuth; above it, never, at any
    time of any year.
    """
    return float(np.rad2deg(np.arctan(rod_len / boom_len)))


def shadow_track(
    date,
    latitude: float,
    longitude: float,
    tz: str | None = None,
    freq: str = TRACK_FREQ,
    **geometry,
) -> pd.DataFrame:
    """`shade` across one local day, at a step fine enough to catch the notch.

    `date` is anything pandas reads as one -- "2026-06-21", a Timestamp, a
    date. `tz` names the site's timezone, so the index reads as local clock time
    and the day starts and ends where the operator's day does; without one the
    day is UTC.
    """
    start = pd.Timestamp(date).normalize()
    end = start + pd.Timedelta("1D")
    if tz is not None:
        # Localised as two midnights rather than counted off as a fixed number of
        # steps, so a day that clocks 23 or 25 hours over a DST change still runs
        # midnight to midnight instead of spilling into the next date or stopping
        # an hour short of the end of its own.
        start, end = start.tz_localize(tz), end.tz_localize(tz)
    # closed on the left, so consecutive dates tile instead of double-counting
    # the stroke of midnight
    index = pd.date_range(start, end, freq=freq, inclusive="left")
    return shade(index, latitude, longitude, **geometry)


def shadow_windows(
    track: pd.DataFrame, threshold: float = DEFAULT_THRESHOLD
) -> pd.DataFrame:
    """The intervals in a track where the rod blocks more than `threshold`.

    One row per unbroken run, which is what makes a track actionable: a day
    usually has one, and it is a few minutes long. `peak` is the deepest
    blocking reached, and the elevation and azimuth beside it say where the sun
    was when it happened -- a low peak elevation is the tell that the notch will
    be shallow in irradiance terms however complete the geometric blocking.

    Duration is measured from the first to the last sample over threshold plus
    one step, so a single-sample window comes back as one step long rather than
    zero.
    """
    over = track["blocked"] > threshold
    if not over.any():
        return pd.DataFrame(
            columns=["start", "end", "duration", "peak", "elevation", "azimuth"]
        )

    step = track.index.to_series().diff().median()
    # each False->True edge opens a new run, so a cumulative sum labels them
    runs = (over & ~over.shift(1, fill_value=False)).cumsum()[over]

    rows = []
    for _, part in track[over].groupby(runs):
        deepest = part["blocked"].idxmax()
        rows.append(
            {
                "start": part.index[0],
                "end": part.index[-1] + step,
                "duration": part.index[-1] - part.index[0] + step,
                "peak": part.loc[deepest, "blocked"],
                "elevation": part.loc[deepest, "elevation"],
                "azimuth": part.loc[deepest, "azimuth"],
            }
        )
    return pd.DataFrame(rows)


def shadow_season(
    start,
    end,
    latitude: float,
    longitude: float,
    tz: str | None = None,
    freq: str = SEASON_FREQ,
    threshold: float = DEFAULT_THRESHOLD,
    **geometry,
) -> pd.DataFrame:
    """One row per date from `start` to `end`, saying whether the notch happens.

    The whole span is walked in a single vectorised pass and then split by local
    date, which is why this is cheap enough to run over a year at a minute's
    resolution. That step is coarse against a notch a few minutes wide, so read
    `peak` and `minutes` as the shape of the season rather than as measurements
    -- re-run `shadow_track` on a date that matters.

    Columns: peak blocked fraction, the local time and sun position at that peak,
    and how many minutes of the day cleared `threshold`.
    """
    first = pd.Timestamp(start).normalize()
    last = pd.Timestamp(end).normalize()
    if tz is not None:
        first, last = first.tz_localize(tz), last.tz_localize(tz)
    index = pd.date_range(first, last + pd.Timedelta("1D"), freq=freq, inclusive="left")

    track = shade(index, latitude, longitude, **geometry)
    step_minutes = pd.Timedelta(freq).total_seconds() / 60.0

    by_date = track.groupby(track.index.date)
    peaks = track.loc[by_date["blocked"].idxmax()]
    return pd.DataFrame(
        {
            "peak": peaks["blocked"].to_numpy(),
            "at": peaks.index,
            "elevation": peaks["elevation"].to_numpy(),
            "azimuth": peaks["azimuth"].to_numpy(),
            "minutes": (
                by_date["blocked"].apply(lambda s: (s > threshold).sum()) * step_minutes
            ).to_numpy(),
        },
        index=pd.DatetimeIndex(list(by_date.groups), name="date"),
    )


# ── charts ───────────────────────────────────────────────────────────────────
# Both charts reuse plot.py's tokens and its two strongest categorical slots, so
# a shading figure sits beside a signal figure without re-teaching the reader
# what the ink means. Blue (4.30:1 on this surface) carries the answer -- the
# blocking itself -- and violet (8.33:1) the sun elevation that says how much
# irradiance the blocking is worth. They are separate panels sharing one x axis
# rather than two y scales on one axes.
BLOCKED_COLOR = plot.SIGNAL_PALETTE[0]
ELEVATION_COLOR = plot.SIGNAL_PALETTE[1]

# How much clear air to leave either side of the event when a day is zoomed in
# on. The notch is minutes long inside a 24-hour track, so plotting the day
# entire draws a spike one pixel wide; this frames it instead.
TRACK_MARGIN = pd.Timedelta("10min")


def _naive(index) -> pd.DatetimeIndex:
    """`index` with any timezone dropped, keeping local clock time as the label.

    Matplotlib's date handling and tz-aware indexes disagree often enough that
    it is not worth finding out which way; the numbers wanted on the axis are
    local wall-clock anyway.
    """
    idx = pd.DatetimeIndex(index)
    return idx.tz_localize(None) if idx.tz is not None else idx


def plot_track(
    track: pd.DataFrame,
    threshold: float = DEFAULT_THRESHOLD,
    title: str = "",
    margin: pd.Timedelta = TRACK_MARGIN,
    figsize: tuple = (10, 3.4),
    show: bool = True,
):
    """The day's notch, framed around the event rather than spread over 24 hours.

    Zooms to whatever cleared `threshold` plus `margin` either side, and marks
    the peak directly -- one label, on the one point that carries the finding.
    Returns the figure when `show` is False. A day with no shadow draws nothing
    and says so, rather than returning an empty pair of axes.
    """
    windows = shadow_windows(track, threshold)
    if windows.empty:
        print("no shadow on this date -- nothing to draw")
        return None

    span = track.loc[windows["start"].min() - margin : windows["end"].max() + margin]
    stamps = _naive(span.index)
    blocked = span["blocked"].to_numpy() * 100.0
    peak = int(blocked.argmax())

    with plt.rc_context(plot.chart_style()):
        figure, axes = plt.subplots(figsize=figsize, constrained_layout=True)
        axes.fill_between(stamps, blocked, color=BLOCKED_COLOR, alpha=0.16, linewidth=0)
        axes.plot(stamps, blocked, color=BLOCKED_COLOR, linewidth=2.0)
        axes.plot(
            stamps[peak],
            blocked[peak],
            marker="o",
            markersize=8,
            color=BLOCKED_COLOR,
            markeredgecolor=plot.SURFACE,
            markeredgewidth=1.6,
        )
        axes.annotate(
            f"{blocked[peak]:.0f}% of the beam at {stamps[peak]:%H:%M:%S}\n"
            f"sun {span['elevation'].iloc[peak]:.1f}° up, "
            f"{span['azimuth'].iloc[peak]:.1f}° azimuth",
            xy=(stamps[peak], blocked[peak]),
            xytext=(14, -10),
            textcoords="offset points",
            fontsize=8.5,
            color=plot.MUTED,
            va="top",
        )
        axes.set_ylim(0, 105)
        axes.set_ylabel("direct beam blocked (%)")
        axes.set_title(f"{title} lightning rod shadow on the pyranometer".strip())
        axes.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M"))
        plot._finish(axes)

        if show:
            plt.show()
            plt.close(figure)
            return None
    return figure


def plot_season(
    season: pd.DataFrame,
    rod_len: float = ROD_LEN,
    boom_len: float = BOOM_LEN,
    title: str = "",
    figsize: tuple = (10, 4.6),
    show: bool = True,
):
    """The year: which days have a notch, and how much sun is behind it.

    Two panels on a shared date axis. The top one is the finding -- minutes per
    day over threshold -- and the bottom one is the caveat, the sun's elevation
    when it happened, against the grazing angle no shadow can beat. Days near
    that line are the ones that matter: the sun is as high as it ever gets while
    still being blocked, so the beam it loses is worth the most.

    The elevation trace is drawn only on days that had an event, so it breaks
    into the two seasons rather than joining them across a winter that has none.
    """
    dates = _naive(season.index)
    minutes = season["minutes"].to_numpy(dtype=float)
    affected = minutes > 0
    # a peak elevation on a day with no event is just the argmax of a flat zero
    elevation = np.where(affected, season["elevation"].to_numpy(dtype=float), np.nan)
    grazing = grazing_elevation(rod_len, boom_len)

    with plt.rc_context(plot.chart_style()):
        figure, (top, bottom) = plt.subplots(
            2,
            1,
            sharex=True,
            constrained_layout=True,
            figsize=figsize,
            gridspec_kw={"height_ratios": [1.5, 1]},
        )

        top.fill_between(
            dates, minutes, color=BLOCKED_COLOR, alpha=0.16, linewidth=0, step="mid"
        )
        top.step(dates, minutes, where="mid", color=BLOCKED_COLOR, linewidth=1.8)
        top.set_ylabel("minutes shaded")
        top.set_title(
            f"{title} days the rod shades the pyranometer "
            f"({int(affected.sum())} of {len(season)})".strip()
        )

        bottom.axhline(grazing, color=plot.MUTED, linewidth=1.0, linestyle=(0, (4, 3)))
        bottom.annotate(
            f"{grazing:.1f}° -- shadow cannot reach the sensor above this",
            xy=(dates[0], grazing),
            xytext=(2, 4),
            textcoords="offset points",
            fontsize=8.5,
            color=plot.MUTED,
        )
        bottom.plot(dates, elevation, color=ELEVATION_COLOR, linewidth=2.0)
        bottom.set_ylim(0, grazing * 1.35)
        bottom.set_ylabel("sun elevation (°)")

        for axes in (top, bottom):
            plot._finish(axes)
        bottom.xaxis.set_major_formatter(mdates.DateFormatter("%b"))
        bottom.set_xlabel("")

        if show:
            plt.show()
            plt.close(figure)
            return None
    return figure
