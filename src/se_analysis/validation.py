"""Signal validation shared by ProtoNode, LOGR measurement and LOGR diagnostic data.

Every check takes a frame with a DatetimeIndex and wide numeric signal columns --
the shape all three sources reduce to -- and returns findings in one schema, so
results from different checks and different sources concatenate into one report.

A finding is one distinct problem. For most checks that is "signal X failed check
Y across N rows"; for stalls and gaps each run is a finding of its own, since the
individual runs are what you go and look at.

Missing data is its own check, not a symptom of another one: check_gaps reports
the stretches where the record stops, and the stall check works within them, so a
gap is never also reported as every signal holding its value at once.

Recorded solar angles get checked against a position computed independently from
the site latitude and longitude (see se_analysis.solar). File stamps are not
reliably UTC, so the offset they were written at is worked out from the angles and
reported in its own right -- solar_offset() is the number to watch as these files
move to UTC.

Findings only ever describe failures. To see what was checked and passed, roll
them up with summarize(), which enumerates the spec rather than the findings.
"""

from dataclasses import dataclass, field, replace

import numpy as np
import pandas as pd

from se_analysis import solar

# Physical limits, reusable across sources: a temperature is a temperature
# whether it came off a ProtoNode or a LOGR channel.
ALBEDO_IRRADIANCE = (0, 500)
ARRAY_TILT = (0, 75)
BAROMETRIC_PRESSURE = (800, 1200)
DNI_IRRADIANCE = (0, 1400)
GHI_TILT = (0, 0.5)
GHI_IRRADIANCE = (0, 1200)
ISC_SOILING_RATIO = (0, 2)  # may occasionally exceed 1
LATITUDE = (-90, 90)
LOGR_VOLTAGE = (22, 25)
LONGITUDE = (-180, 180)
RATIO = (0, 1)
RELATIVE_HUMIDITY = (0, 100)
SOLAR_AZIMUTH = (0, 360)
SOLAR_ELEVATION = (-90, 90)
SOLAR_ZENITH = (0, 180)
STATE = (0, 1)
TEMPERATURE = (-40, 70)
WIND_DIRECTION = (0, 360)
WIND_SPEED = (0, 50)  # m/s

FINDING_COLUMNS = [
    "check",
    "signal",
    "detail",
    "value",
    "start",
    "end",
    "duration",
    "n_samples",
]

DEFAULT_MAX_STALL = pd.Timedelta("20s")
DEFAULT_SENTINEL = -99

# A stamp-to-stamp step this many times the median interval means samples are
# missing. Loose enough for ordinary jitter, tight enough to catch a single
# dropped sample.
DEFAULT_GAP_FACTOR = 1.5

# Gaps stop every signal at once, so they are reported against the frame rather
# than once per column.
ALL_SIGNALS = "(all signals)"

RUN_COLUMNS = ["signal", "value", "start", "end", "n_samples", "duration"]

# How far a recorded solar angle may sit from the computed one. Loose next to the
# hundredths of a degree the two agree to in practice, because refraction near
# the horizon and whatever algorithm the logger runs are both worth a tenth or
# so, and neither is a fault.
DEFAULT_SOLAR_TOLERANCE = 0.5


def _findings(rows: list[dict]) -> pd.DataFrame:
    """A findings frame with the full schema, even when there is nothing to report."""
    return pd.DataFrame(rows, columns=FINDING_COLUMNS).reset_index(drop=True)


def _pair_label(left: str, right: str) -> str:
    return f"{left} vs {right}"


def _span(index: pd.Index, mask: np.ndarray) -> dict:
    """First/last stamp and row count for a boolean mask over `index`."""
    hit = index[mask]
    start, end = (hit[0], hit[-1]) if len(hit) else (pd.NaT, pd.NaT)
    duration = end - start if len(hit) else pd.NaT
    return {
        "start": start,
        "end": end,
        "duration": duration,
        "n_samples": int(mask.sum()),
    }


def check_ranges(data: pd.DataFrame, ranges: dict[str, tuple]) -> pd.DataFrame:
    """Flag signals straying outside [lo, hi]. NaN counts as out of range."""
    rows = []
    for signal, (lo, hi) in ranges.items():
        if signal not in data.columns:
            continue
        bad = ~data[signal].between(lo, hi)
        if not bad.any():
            continue
        rows.append(
            {
                "check": "range",
                "signal": signal,
                "detail": f"outside [{lo}, {hi}]",
                "value": data.loc[bad, signal].iloc[0],
                **_span(data.index, bad.to_numpy()),
            }
        )
    return _findings(rows)


def check_equivalent(data: pd.DataFrame, pairs) -> pd.DataFrame:
    """Flag pairs of signals that should carry the same value but do not."""
    rows = []
    for left, right in pairs:
        if left not in data.columns or right not in data.columns:
            continue
        bad = data[left] != data[right]
        if not bad.any():
            continue
        rows.append(
            {
                "check": "equivalent",
                "signal": _pair_label(left, right),
                "detail": f"{int(bad.sum())} of {len(data)} rows differ",
                "value": data.loc[bad, left].iloc[0],
                **_span(data.index, bad.to_numpy()),
            }
        )
    return _findings(rows)


def check_lesser(data: pd.DataFrame, pairs) -> pd.DataFrame:
    """Flag pairs where the left signal should stay below the right but does not."""
    rows = []
    for left, right in pairs:
        if left not in data.columns or right not in data.columns:
            continue
        bad = data[left] > data[right]
        if not bad.any():
            continue
        rows.append(
            {
                "check": "lesser",
                "signal": _pair_label(left, right),
                "detail": f"{left} exceeds {right} in {int(bad.sum())} of {len(data)} rows",
                "value": data.loc[bad, left].iloc[0],
                **_span(data.index, bad.to_numpy()),
            }
        )
    return _findings(rows)


def sampling_interval(index: pd.DatetimeIndex) -> pd.Timedelta:
    """The median stamp-to-stamp step -- the cadence data is meant to arrive at."""
    return pd.Series(index).diff().median()


def expected_periods(start, end, interval: pd.Timedelta) -> int:
    """How many samples a run at `interval` should have produced over whole days.

    The bounds are widened to day boundaries first -- `start` back to its own
    midnight, `end` forward to the midnight after it -- so a request written in
    dates covers those dates end to end, and the count does not move with the
    time of day the run happened to be started or stopped at.

    Widening `end` past midnight is unconditional: a day named as the end of the
    span is a day that should be complete, including one named as a bare date
    that arrives here already sitting on midnight.

    Intervals that do not divide the span evenly leave a remainder too short to
    hold another sample, so only whole periods are counted.
    """
    start = pd.Timestamp(start)
    end = pd.Timestamp(end)
    interval = pd.Timedelta(interval)
    if interval <= pd.Timedelta(0):
        raise ValueError(f"interval must be positive, got {interval}")
    if end < start:
        raise ValueError(f"end ({end}) precedes start ({start})")

    span = (end.floor("D") + pd.Timedelta(days=1)) - start.floor("D")
    return span // interval


def availability(
    start, end, data: pd.DataFrame, interval: pd.Timedelta | None = None
) -> tuple[float, pd.DataFrame]:
    """What fraction of the expected samples actually arrived, and what did not.

    Answers the question a delivery is judged on -- "did we get the days we asked
    for?" -- rather than the one gap_runs answers, which is "where does the
    record we did get stop?". The difference is the ends: a run that started late
    or stopped early leaves no step in its own index for gap_runs to find, but is
    missing data all the same, and shows up here.

    The expected stamps are the whole-day span of `expected_periods`, laid out at
    `interval` from the start of the first day. `interval` defaults to the
    frame's own median cadence. Real stamps carry jitter, so each one is snapped
    to the nearest slot on that grid; stamps landing outside the span, and
    several landing in one slot, each count once at most.

    Returns the availability as a percentage and one row per contiguous stretch
    of absent slots, with `end` being the stamp the record was due to resume at,
    so `duration` is the time actually unaccounted for.
    """
    index = data.index
    if not isinstance(index, pd.DatetimeIndex):
        # A bare number reads as an epoch offset rather than failing, so the row
        # numbers of a frame still carrying its stamps in a column would pass
        # silently and be measured as nanoseconds.
        stamps = [c for c in data.columns if "timestamp" in str(c).lower()]
        hint = f"; set_index({stamps[0]!r}) first" if stamps else ""
        raise TypeError(
            f"data must have a DatetimeIndex, got {type(index).__name__}{hint}"
        )

    if interval is None:
        interval = sampling_interval(index)
    if isinstance(interval, (int, float)):
        # pd.Timedelta reads a bare number as nanoseconds, which is never what
        # anyone means by a sampling interval.
        raise TypeError(
            f"interval must be a Timedelta or a string like '10min', not {interval!r}"
        )
    interval = pd.Timedelta(interval)
    if pd.isna(interval) or interval <= pd.Timedelta(0):
        raise ValueError(
            f"no usable interval ({interval}); pass one explicitly when the frame "
            "is too short to infer a cadence from"
        )

    n_expected = expected_periods(start, end, interval)
    if n_expected == 0:
        raise ValueError(f"interval ({interval}) is longer than the requested span")

    origin = pd.Timestamp(start).floor("D")
    # Bare dates are the natural way to ask for a span; match them to whatever
    # the stamps carry rather than making the caller do it.
    if index.tz is not None and origin.tz is None:
        origin = origin.tz_localize(index.tz)
    elif index.tz is None and origin.tz is not None:
        origin = origin.tz_localize(None)

    slot = ((index - origin) / interval).to_numpy(dtype="float64")
    slot = np.round(slot[np.isfinite(slot)]).astype("int64")
    present = np.unique(slot[(slot >= 0) & (slot < n_expected)])
    percent = 100.0 * len(present) / n_expected

    seen = np.zeros(n_expected, dtype=bool)
    seen[present] = True
    absent = np.flatnonzero(~seen)
    runs = np.split(absent, np.flatnonzero(np.diff(absent) != 1) + 1)
    rows = [
        {
            "start": origin + int(run[0]) * interval,
            "end": origin + int(run[-1] + 1) * interval,
            "duration": len(run) * interval,
            "n_missing": len(run),
        }
        for run in runs
        if len(run)
    ]
    return percent, pd.DataFrame(
        rows, columns=["start", "end", "duration", "n_missing"]
    )


def _segments(index: pd.DatetimeIndex, gap_factor: float) -> list[tuple[int, int]]:
    """Half-open [lo, hi) row ranges of a sorted index, split at every gap.

    Whatever spans a gap is not one stretch of data, so anything measured over
    elapsed time gets measured inside these ranges instead.
    """
    step = pd.Series(index).diff()
    interval = step.median()
    if pd.isna(interval) or interval == pd.Timedelta(0):
        return [(0, len(index))]
    breaks = np.flatnonzero((step > gap_factor * interval).to_numpy()).tolist()
    edges = [0, *breaks, len(index)]
    return list(zip(edges, edges[1:]))


def gap_runs(
    data: pd.DataFrame, gap_factor: float = DEFAULT_GAP_FACTOR
) -> pd.DataFrame:
    """Stretches where no sample arrived at all -- one row per gap in the index.

    A step wider than `gap_factor` times the median interval means at least one
    expected sample never showed up. Scaling off the median keeps this working
    for every source without configuration: a second for ProtoNode polls, a
    minute for LOGR records.
    """
    index = data.index.sort_values()
    step = pd.Series(index).diff()
    interval = step.median()
    if len(index) < 2 or pd.isna(interval) or interval == pd.Timedelta(0):
        return pd.DataFrame(columns=["start", "end", "duration", "n_missing"])

    pos = np.flatnonzero((step > gap_factor * interval).to_numpy())
    out = pd.DataFrame({"start": index[pos - 1], "end": index[pos]})
    out["duration"] = out["end"] - out["start"]
    # one step was due; everything beyond it is a sample that never came
    out["n_missing"] = (out["duration"] / interval).round().astype("Int64") - 1
    return out


def check_gaps(
    data: pd.DataFrame, gap_factor: float = DEFAULT_GAP_FACTOR
) -> pd.DataFrame:
    """Flag stretches where the record stops. One finding per gap.

    Reported against the frame, not per signal: when polling stops, every signal
    stops together, and one finding per gap is what you go and look at.
    """
    rows = [
        {
            "check": "gap",
            "signal": ALL_SIGNALS,
            "detail": f"no data for {run.duration} (~{run.n_missing} samples missing)",
            "value": run.n_missing,
            "start": run.start,
            "end": run.end,
            "duration": run.duration,
            "n_samples": 0,
        }
        for run in gap_runs(data, gap_factor=gap_factor).itertuples()
    ]
    return _findings(rows)


def _unchanged_runs(v: pd.Series, raw: pd.Series, signal: str) -> pd.DataFrame:
    """One row per stretch of unchanged value, within a series free of gaps.

    Compares on `v` and reports from `raw`, so rounding decides what counts as a
    change without rounding the value you are shown.
    """
    # new run wherever the value changes; NaN-to-NaN counts as unchanged
    prev = v.shift()
    is_start = ~(v.eq(prev) | (v.isna() & prev.isna())).to_numpy()
    is_start[0] = True  # the first sample opens a run even when it is NaN

    start = v.index[is_start]
    # each run ends where the next begins; the last one ends at the last sample
    end = start[1:].append(pd.DatetimeIndex([v.index[-1]]))
    pos = np.flatnonzero(is_start)

    out = pd.DataFrame(
        {
            "signal": signal,
            "value": raw.to_numpy()[is_start],
            "start": start,
            "end": end,
            "n_samples": np.diff(np.append(pos, len(v))),
        }
    )
    out["duration"] = out["end"] - out["start"]
    return out


def stall_runs(
    data: pd.DataFrame,
    signal: str,
    max_stall: pd.Timedelta = DEFAULT_MAX_STALL,
    decimals: int | None = None,
    gap_factor: float = DEFAULT_GAP_FACTOR,
) -> pd.DataFrame:
    """Runs where `signal` held one value longer than `max_stall`.

    Expects a DatetimeIndex. Returned one row per run with the value that was
    held -- this is the frame to eyeball when a stall finding needs explaining.

    A stall is a signal that stopped moving while the data kept coming, so runs
    are found within each gap-free stretch of the index (see gap_runs): a hold
    either side of a gap is never joined across it, and runs of NaN are left to
    the checks that report missing values rather than counted as stalls. Missing
    data is reported once, by check_gaps.

    `max_stall` has to exceed the sampling interval, or every single sample
    counts as a run that outlasted it. ProtoNode polls about once a second while
    LOGR statistical and diagnostic data arrive once a minute, so the default
    suits the former and needs raising for the latter.
    """
    s = data[signal].sort_index()
    if len(s) < 2:
        return pd.DataFrame(columns=RUN_COLUMNS)

    interval = sampling_interval(s.index)
    if pd.notna(interval) and max_stall < interval:
        raise ValueError(
            f"max_stall ({max_stall}) is below the {interval} sampling interval, "
            f"so every sample of {signal!r} would register as a stall. Raise "
            "max_stall above the interval."
        )

    v = s.round(decimals) if decimals is not None else s

    runs = [
        _unchanged_runs(v.iloc[lo:hi], s.iloc[lo:hi], signal)
        for lo, hi in _segments(s.index, gap_factor)
        if hi > lo
    ]
    if not runs:
        return pd.DataFrame(columns=RUN_COLUMNS)
    out = pd.concat(runs, ignore_index=True)
    held = out["value"].notna() & (out["duration"] > max_stall)
    return out[held].reset_index(drop=True)


def check_stalls(
    data: pd.DataFrame,
    signals,
    max_stall: pd.Timedelta = DEFAULT_MAX_STALL,
    stall_windows: dict | None = None,
    decimals: int | None = None,
    gap_factor: float = DEFAULT_GAP_FACTOR,
) -> pd.DataFrame:
    """Flag signals expected to keep moving that held a value. One finding per run.

    `max_stall` is the window every signal is held to; `stall_windows` maps the
    signals that need their own to the window they get. A coarsely quantised
    signal -- barometric pressure reported to 0.1 hPa, say -- legitimately sits
    on one value for minutes at a time, and wants a window of its own rather
    than a looser default that would also hide a dead irradiance channel.
    """
    stall_windows = stall_windows or {}
    rows = []
    for signal in signals:
        if signal not in data.columns:
            continue
        runs = stall_runs(
            data,
            signal,
            max_stall=stall_windows.get(signal, max_stall),
            decimals=decimals,
            gap_factor=gap_factor,
        )
        for run in runs.itertuples():
            rows.append(
                {
                    "check": "stall",
                    "signal": signal,
                    "detail": f"held {run.value} for {run.duration}",
                    "value": run.value,
                    "start": run.start,
                    "end": run.end,
                    "duration": run.duration,
                    "n_samples": int(run.n_samples),
                }
            )
    return _findings(rows)


def check_constants(
    data: pd.DataFrame,
    signals,
    zero_tol: float = 1e-9,
    eps: float = 0.0,
) -> pd.DataFrame:
    """Flag signals that should hold one non-zero value all file long but do not.

    Which value is not the question here -- a serial number reads whatever the
    logger was given. For a signal that has to sit at a value you can name, use
    check_expected instead.
    """
    rows = []

    for signal in signals:
        if signal not in data.columns:
            continue
        raw = data[signal]
        n_nan = int(raw.isna().sum())
        s = raw.dropna()

        if len(s) == 0:
            rows.append(
                {
                    "check": "constant",
                    "signal": signal,
                    "detail": "all NaN",
                    "value": None,
                    "start": data.index[0] if len(data) else pd.NaT,
                    "end": data.index[-1] if len(data) else pd.NaT,
                    "duration": pd.NaT,
                    "n_samples": n_nan,
                }
            )
            continue

        num = pd.to_numeric(s, errors="coerce")
        is_numeric = bool(num.notna().all())

        if is_numeric:
            constant = float(num.max() - num.min()) <= eps
            nonzero = float(num.abs().min()) > zero_tol
            value = num.iloc[0]
        else:
            constant = s.astype(str).nunique() == 1
            nonzero = str(s.iloc[0]).strip() not in {"", "0", "0.0", "None", "nan"}
            value = s.iloc[0]

        problems = list(
            filter(
                None,
                [
                    None if constant else "changes",
                    None if nonzero else "zero/blank",
                    None if n_nan == 0 else f"{n_nan} NaN",
                ],
            )
        )
        if not problems:
            continue

        rows.append(
            {
                "check": "constant",
                "signal": signal,
                "detail": "; ".join(problems),
                "value": value,
                "start": data.index[0],
                "end": data.index[-1],
                "duration": data.index[-1] - data.index[0],
                "n_samples": int(s.astype(str).nunique()),
            }
        )

    return _findings(rows)


def _matches(values: pd.Series, want, eps: float = 0.0) -> np.ndarray:
    """Elementwise "reads `want`", numerically where both sides are numbers.

    Missing and non-numeric values never match, so a dropout counts as a
    deviation rather than quietly passing.
    """
    num = pd.to_numeric(values, errors="coerce")
    target = pd.to_numeric(pd.Series([want]), errors="coerce").iloc[0]
    if pd.notna(target) and bool(num.notna().any()):
        return ((num - target).abs() <= eps).to_numpy()
    return (values.astype(str) == str(want)).to_numpy()


def check_expected(
    data: pd.DataFrame, expected: dict, eps: float = 0.0
) -> pd.DataFrame:
    """Flag signals that should read one named value throughout but do not.

    Where the constant check only asks that a signal never move, this says what
    it has to read: a ventilator fan wired to run all the time is broken whether
    it sits at 0 for the whole file or drops out for an hour in the middle.
    Numeric values compare within `eps`, anything else as text.

    One finding per signal, spanning the rows that deviate -- the rows
    themselves are expected_deviations().
    """
    rows = []
    for signal, want in expected.items():
        if signal not in data.columns:
            continue
        column = data[signal]
        bad = ~_matches(column, want, eps)
        n_bad = int(bad.sum())
        if n_bad == 0:
            continue

        off = column[bad]
        n_missing = int(off.isna().sum())
        others = sorted(off.dropna().astype(str).unique().tolist())[:5]
        detail = "; ".join(
            filter(
                None,
                [
                    (
                        f"{n_bad - n_missing} value(s) != {want} (saw {others})"
                        if others
                        else None
                    ),
                    f"{n_missing} missing" if n_missing else None,
                ],
            )
        )
        rows.append(
            {
                "check": "expected",
                "signal": signal,
                "detail": detail,
                "value": off.dropna().iloc[0] if n_bad > n_missing else None,
                **_span(data.index, bad),
            }
        )
    return _findings(rows)


def expected_deviations(
    data: pd.DataFrame, signal: str, value, eps: float = 0.0
) -> pd.DataFrame:
    """The rows where `signal` does not read `value` -- for eyeballing a failure."""
    return data.loc[~_matches(data[signal], value, eps), [signal]]


def check_sentinel(
    data: pd.DataFrame, signals, sentinel=DEFAULT_SENTINEL, eps: float = 0.0
) -> pd.DataFrame:
    """Flag signals that should read the invalid-data sentinel everywhere but do not.

    `sentinel` is a scalar, or a {signal: sentinel} mapping for mixed maps.
    """
    rows = []
    for signal in signals:
        if signal not in data.columns:
            continue
        expect = (
            sentinel.get(signal, DEFAULT_SENTINEL)
            if isinstance(sentinel, dict)
            else sentinel
        )
        num = pd.to_numeric(data[signal], errors="coerce")

        n_nan = int(num.isna().sum())  # missing, or non-numeric junk
        bad = (num - expect).abs() > eps  # NaN compares False, so counted apart
        n_bad = int(bad.sum())
        if n_bad == 0 and n_nan == 0:
            continue

        others = sorted(num[bad].dropna().unique().tolist())[:5]
        detail = "; ".join(
            filter(
                None,
                [
                    f"{n_bad} value(s) != {expect}" if n_bad else None,
                    f"{n_nan} NaN/non-numeric" if n_nan else None,
                ],
            )
        )
        finding = {
            "check": "sentinel",
            "signal": signal,
            "detail": f"{detail} (saw {others})" if others else detail,
            "value": others[0] if others else None,
            **_span(data.index, bad.to_numpy()),
        }
        finding["n_samples"] = n_bad + n_nan
        rows.append(finding)
    return _findings(rows)


def sentinel_deviations(
    data: pd.DataFrame, signal: str, sentinel=DEFAULT_SENTINEL, eps: float = 0.0
) -> pd.DataFrame:
    """The rows where `signal` is not the sentinel -- for eyeballing a failure."""
    num = pd.to_numeric(data[signal], errors="coerce")
    off = ((num - sentinel).abs() > eps) | num.isna()
    return data.loc[off.to_numpy(), [signal]]


@dataclass(frozen=True)
class SolarAngles:
    """Where the site is, and which columns carry the logger's own solar angles.

    Column names are source-specific, so a source supplies them -- see
    se_analysis.protonode.SOLAR.

    `latitude` and `longitude` are each either the name of a column carrying the
    position the file was configured with, or that position itself as a number.
    Both spellings occur: a ProtoNode export repeats the coordinates on every
    row, where a LOGR measurement file states them once in its header and never
    in its data (see se_analysis.measurement.solar_angles).

    `utc_offset` left as None means work the offset out from the angles
    themselves, which is what makes the check usable on files whose stamps are in
    whatever timezone the technician's laptop was set to. Pin it to
    pd.Timedelta(0) to insist on UTC and have anything else read as an angle
    error instead of a timezone.

    `refract` says whether the recorded angles are apparent or geometric -- that
    is, whether whatever computed them lifted the sun over the horizon the way
    the atmosphere does. It is a property of the logger's algorithm rather than a
    tolerance to be widened: the difference runs to half a degree near the
    horizon, which is the whole error budget and more.
    """

    latitude: str | float
    longitude: str | float
    zenith: str | None = None
    elevation: str | None = None
    azimuth: str | None = None
    tolerance: float = DEFAULT_SOLAR_TOLERANCE
    utc_offset: pd.Timedelta | None = None
    max_clock_skew: pd.Timedelta = solar.DEFAULT_MAX_CLOCK_SKEW
    refract: bool = True

    def override(self, **changes) -> "SolarAngles":
        """A copy with `changes` applied."""
        return replace(self, **changes)

    def angles(self) -> dict[str, str]:
        """The (computed angle -> recorded column) pairs this config compares."""
        named = {
            "zenith": self.zenith,
            "elevation": self.elevation,
            "azimuth": self.azimuth,
        }
        return {angle: column for angle, column in named.items() if column}

    def columns(self) -> list[str]:
        """Every column the check needs present.

        A coordinate given as a number is not a column and does not belong here:
        a file stating its position in its header rather than its data is not a
        file missing a signal.
        """
        position = [c for c in (self.latitude, self.longitude) if isinstance(c, str)]
        return [*position, *self.angles().values()]


def _coordinate(data: pd.DataFrame, where, limits: tuple) -> float | None:
    """One coordinate, read out of a column of `data` or given outright.

    A column has to be present, numeric and unchanging for a computed position
    to mean anything; a number has only to be a position. check_constants and
    check_ranges are what report a coordinate column being otherwise, so this
    only decides whether the solar check can run.
    """
    lo, hi = limits
    if isinstance(where, str):
        if where not in data.columns:
            return None
        values = pd.to_numeric(data[where], errors="coerce").dropna().unique()
        if len(values) != 1:
            return None
        where = values[0]
    value = float(where)
    return value if lo <= value <= hi else None


def _site_position(data: pd.DataFrame, angles: SolarAngles) -> tuple:
    """The latitude and longitude the recorded angles get checked against.

    (None, None) unless both resolve, since neither alone places the sun.
    """
    latitude = _coordinate(data, angles.latitude, LATITUDE)
    longitude = _coordinate(data, angles.longitude, LONGITUDE)
    if latitude is None or longitude is None:
        return None, None
    return latitude, longitude


def solar_offset(data: pd.DataFrame, angles: SolarAngles) -> pd.Timedelta:
    """How far the file's stamps run ahead of UTC, per its own solar angles.

    The number to watch while these files move to UTC: it should become zero.
    NaT when the file does not carry enough to work it out.
    """
    if angles.utc_offset is not None:
        return angles.utc_offset
    latitude, longitude = _site_position(data, angles)
    recorded = angles.angles()
    if latitude is None or "elevation" not in recorded or len(data) == 0:
        return pd.NaT
    azimuth = recorded.get("azimuth")
    return solar.infer_utc_offset(
        data.index,
        latitude,
        longitude,
        data[recorded["elevation"]],
        None if azimuth is None else data[azimuth],
        refract=angles.refract,
    )


def check_solar_angles(data: pd.DataFrame, angles: SolarAngles) -> pd.DataFrame:
    """Flag recorded solar angles that disagree with the site position and clock.

    Three separate problems, because they want separate answers:

    "solar" -- a recorded angle that no timezone reconciles with the computed
    position. A stale angle, a wrong latitude or a flipped azimuth convention
    shows up here; a clock offset does not.

    "utc" -- stamps that are not UTC. Expected for now, since files are written in
    whatever timezone the machine polling them is set to, and the thing to watch
    as that changes.

    "clock" -- an offset that is not on the quarter hour every real timezone sits
    on, so no timezone explains it. Either the logger clock is wrong or the
    position it was configured with is. Which of the two, a short file cannot say:
    four minutes of clock and a degree of longitude move the sun alike.

    The timezone findings are only raised once the angles do reconcile, because
    the offset is inferred from those same angles -- reading a timezone off angles
    that do not fit would be asserting a conclusion from the evidence that just
    failed. Fix the angles, run again, and the timezone question is waiting.
    """
    recorded = angles.angles()
    if not recorded or len(data) == 0:
        return _findings([])
    if any(column not in data.columns for column in angles.columns()):
        return _findings([])  # missing_signals is what reports absent columns

    whole_file = np.ones(len(data), dtype=bool)
    latitude, longitude = _site_position(data, angles)
    if latitude is None:
        return _findings(
            [
                {
                    "check": "solar",
                    "signal": ALL_SIGNALS,
                    "detail": (
                        f"{angles.latitude}/{angles.longitude} unusable, so the "
                        "angles were not checked"
                    ),
                    "value": None,
                    **_span(data.index, whole_file),
                }
            ]
        )

    offset = solar_offset(data, angles)
    if pd.isna(offset):
        return _findings(
            [
                {
                    "check": "solar",
                    "signal": ALL_SIGNALS,
                    "detail": "no solar position reconciles the recorded angles",
                    "value": None,
                    **_span(data.index, whole_file),
                }
            ]
        )

    computed = solar.position(
        data.index - offset, latitude, longitude, refract=angles.refract
    )
    misses = {}
    for angle, column in recorded.items():
        miss = (
            computed[angle].to_numpy()
            - pd.to_numeric(data[column], errors="coerce").to_numpy()
        )
        if angle == "azimuth":  # wraps at 360, so compare the short way round
            miss = (miss + 180.0) % 360.0 - 180.0
        misses[column] = (angle, np.abs(miss))

    rows = []
    for column, (angle, miss) in misses.items():
        bad = miss > angles.tolerance
        if not bad.any():
            continue
        worst = int(np.nanargmax(np.where(np.isnan(miss), -np.inf, miss)))
        rows.append(
            {
                "check": "solar",
                "signal": column,
                "detail": (
                    f"off computed {angle} by up to {miss[bad].max():.3f} deg "
                    f"(tolerance {angles.tolerance}) at the best-fit "
                    f"{solar.format_offset(offset)}"
                ),
                "value": data[column].iloc[worst],
                **_span(data.index, bad),
            }
        )
    if rows:
        return _findings(rows)

    timezone = solar.nearest_timezone(offset)
    skew = offset - timezone
    off_lattice = abs(skew) > angles.max_clock_skew

    # Reported apart, and both when both hold: stamps not being UTC is the
    # migration still outstanding, an offset that is no timezone is a fault. A
    # file can be guilty of each, and saying only the second would leave the
    # first reading as passed.
    if offset != pd.Timedelta(0):
        rows.append(
            {
                "check": "utc",
                "signal": ALL_SIGNALS,
                "detail": (
                    f"stamps are {solar.format_offset(offset)}, not UTC"
                    if off_lattice
                    else f"stamps are {solar.format_offset(timezone)}, not UTC"
                    + (f" (clock {solar.format_skew(skew)})" if skew else "")
                ),
                "value": offset if off_lattice else timezone,
                **_span(data.index, whole_file),
            }
        )
    if off_lattice:
        rows.append(
            {
                "check": "clock",
                "signal": ALL_SIGNALS,
                "detail": (
                    f"{solar.format_offset(offset)} is no timezone "
                    f"({solar.format_offset(timezone)} is nearest, "
                    f"{solar.format_skew(skew)} out); logger clock or configured "
                    "position is wrong"
                ),
                "value": offset,
                **_span(data.index, whole_file),
            }
        )
    return _findings(rows)


@dataclass(frozen=True)
class Spec:
    """Which signals get which checks, and with what limits.

    Sources ship a standard spec (see se_analysis.protonode.SPEC); notebooks call
    override() for whatever a given site does differently.

    `constant` lists the signals that must not move, whatever they happen to
    read; `expected` is the {signal: value} map for the ones that must read
    something specific. Being a dict, `expected` merges on override(), so a
    notebook can pin one more signal without restating the rest.

    `gaps` is the one check driven by a flag rather than by a list of signals,
    every frame having an index to find them in. Turn it off for a frame whose
    rows were selected rather than recorded: the daylight half of a day is
    missing every night by construction, and calling that a gap in the record
    would be reporting the question back as the answer. summarize() then leaves
    the check out rather than showing it passed.
    """

    ranges: dict = field(default_factory=dict)
    equivalent: tuple = ()
    lesser: tuple = ()
    dynamic: tuple = ()
    constant: tuple = ()
    sentinel: tuple = ()
    solar: "SolarAngles | None" = None
    expected: dict = field(default_factory=dict)
    max_stall: pd.Timedelta = DEFAULT_MAX_STALL
    stall_windows: dict = field(default_factory=dict)
    sentinel_value: object = DEFAULT_SENTINEL
    gap_factor: float = DEFAULT_GAP_FACTOR
    gaps: bool = True

    def override(self, **changes) -> "Spec":
        """A copy with `changes` applied: dict fields merge, sequence fields replace."""
        merged = {}
        for name, value in changes.items():
            current = getattr(self, name)
            merged[name] = (
                {**current, **value}
                if isinstance(current, dict) and isinstance(value, dict)
                else value
            )
        return replace(self, **merged)

    def signals(self) -> dict[str, list[str]]:
        """The (check -> labels) this spec tests, whether they pass or not."""
        # The timezone findings only arise from an offset that was inferred.
        # Pinning utc_offset asserts the timezone instead of reading it off the
        # angles, which leaves neither check anything to find -- and so nothing
        # to claim as passed.
        timezone = [ALL_SIGNALS] if self.solar and self.solar.utc_offset is None else []
        return {
            "gap": [ALL_SIGNALS] if self.gaps else [],
            "range": list(self.ranges),
            "equivalent": [_pair_label(a, b) for a, b in self.equivalent],
            "lesser": [_pair_label(a, b) for a, b in self.lesser],
            "stall": list(self.dynamic),
            "constant": list(self.constant),
            "expected": list(self.expected),
            "sentinel": list(self.sentinel),
            "utc": timezone,
            "clock": timezone,
            "solar": list(self.solar.angles().values()) if self.solar else [],
        }


def missing_signals(data: pd.DataFrame, spec: Spec) -> pd.DataFrame:
    """Flag signals the spec expects that the file does not contain."""
    wanted = (
        set(spec.ranges)
        | set(spec.dynamic)
        | set(spec.constant)
        | set(spec.expected)
        | set(spec.sentinel)
    )
    for left, right in list(spec.equivalent) + list(spec.lesser):
        wanted |= {left, right}
    if spec.solar:
        wanted |= set(spec.solar.columns())
    rows = [
        {
            "check": "missing",
            "signal": signal,
            "detail": "not present in file",
            "value": None,
            "start": pd.NaT,
            "end": pd.NaT,
            "duration": pd.NaT,
            "n_samples": 0,
        }
        for signal in sorted(wanted - set(data.columns))
    ]
    return _findings(rows)


def validate(data: pd.DataFrame, spec: Spec) -> pd.DataFrame:
    """Run every check in `spec` and return all findings in one frame."""
    return pd.concat(
        [
            missing_signals(data, spec),
            (
                check_gaps(data, gap_factor=spec.gap_factor)
                if spec.gaps
                else _findings([])
            ),
            check_ranges(data, spec.ranges),
            check_equivalent(data, spec.equivalent),
            check_lesser(data, spec.lesser),
            check_stalls(
                data,
                spec.dynamic,
                max_stall=spec.max_stall,
                stall_windows=spec.stall_windows,
                gap_factor=spec.gap_factor,
            ),
            check_constants(data, spec.constant),
            check_expected(data, spec.expected),
            check_sentinel(data, spec.sentinel, sentinel=spec.sentinel_value),
            check_solar_angles(data, spec.solar) if spec.solar else _findings([]),
        ],
        ignore_index=True,
    )


def summarize(findings: pd.DataFrame, spec: Spec) -> pd.DataFrame:
    """One row per (check, signal) the spec tests, the passing ones included."""
    counts = (
        findings.groupby(["check", "signal"]).size().to_dict() if len(findings) else {}
    )
    rows = []
    for check, signals in spec.signals().items():
        for signal in signals:
            n = counts.get((check, signal), 0)
            rows.append(
                {"check": check, "signal": signal, "n_findings": n, "pass": n == 0}
            )
    out = pd.DataFrame(rows, columns=["check", "signal", "n_findings", "pass"])
    return out.sort_values(["pass", "check", "signal"]).reset_index(drop=True)
