"""Where the sun actually was, so recorded solar angles can be checked against it.

A logger computes its own solar position from a configured latitude, longitude and
clock. Any of the three can be wrong, and the recorded angles are the only place
that shows up -- so this module computes the position independently and, because
file timestamps are not reliably UTC, works out what offset the recording was
made at.

position() follows the NOAA solar calculator (Meeus, Astronomical Algorithms) and
agrees with NREL SPA to about 0.01 degrees, far finer than any tolerance worth
validating a logger against. Angles are apparent -- refracted -- so they match
what an instrument sees; near the horizon refraction is worth a few tenths of a
degree, so tolerances should not be razor thin.
"""

import numpy as np
import pandas as pd

# Real-world UTC offsets land on quarter hours: whole hours mostly, :30 for India
# and Iran, :45 for Nepal and the Chathams. An inferred offset that misses the
# lattice is not a timezone, it is a broken clock or a bad configured position.
TIMEZONE_STEP = pd.Timedelta("15min")

# How far an inferred offset may sit off that lattice and still read as a
# timezone plus ordinary logger clock drift.
DEFAULT_MAX_CLOCK_SKEW = pd.Timedelta("2min")


def _utc_naive(index) -> pd.DatetimeIndex:
    """`index` as tz-naive UTC, the form the algorithm expects."""
    idx = pd.DatetimeIndex(index)
    return idx.tz_convert("UTC").tz_localize(None) if idx.tz is not None else idx


def _ecliptic(jd: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Solar declination (radians) and equation of time (minutes) at Julian days."""
    t = (jd - 2451545.0) / 36525.0

    mean_long = np.deg2rad((280.46646 + t * (36000.76983 + t * 0.0003032)) % 360.0)
    mean_anom = np.deg2rad(357.52911 + t * (35999.05029 - 0.0001537 * t))
    eccentricity = 0.016708634 - t * (0.000042037 + 0.0000001267 * t)
    center = (
        np.sin(mean_anom) * (1.914602 - t * (0.004817 + 0.000014 * t))
        + np.sin(2 * mean_anom) * (0.019993 - 0.000101 * t)
        + np.sin(3 * mean_anom) * 0.000289
    )

    # the moon's node wobbles apparent longitude and obliquity together
    node = np.deg2rad(125.04 - 1934.136 * t)
    app_long = np.deg2rad(
        np.rad2deg(mean_long) + center - 0.00569 - 0.00478 * np.sin(node)
    )
    obliquity = np.deg2rad(
        23.0
        + (26.0 + (21.448 - t * (46.815 + t * (0.00059 - t * 0.001813))) / 60.0) / 60.0
        + 0.00256 * np.cos(node)
    )

    declination = np.arcsin(np.sin(obliquity) * np.sin(app_long))

    y = np.tan(obliquity / 2.0) ** 2
    eot = 4.0 * np.rad2deg(
        y * np.sin(2 * mean_long)
        - 2 * eccentricity * np.sin(mean_anom)
        + 4 * eccentricity * y * np.sin(mean_anom) * np.cos(2 * mean_long)
        - 0.5 * y * y * np.sin(4 * mean_long)
        - 1.25 * eccentricity * eccentricity * np.sin(2 * mean_anom)
    )
    return declination, eot


def _hour_angle(idx: pd.DatetimeIndex, eot: np.ndarray, longitude: float) -> np.ndarray:
    """Hour angle (radians), negative before local solar noon and positive after."""
    minutes = (
        idx.hour * 60.0 + idx.minute + idx.second / 60.0 + idx.microsecond / 6e7
    ).to_numpy(dtype=float)
    true_solar_time = (minutes + eot + 4.0 * longitude) % 1440.0
    return np.deg2rad(true_solar_time / 4.0 - 180.0)


def refraction(elevation) -> np.ndarray:
    """Degrees the atmosphere lifts the sun by, at a true elevation in degrees."""
    e = np.asarray(elevation, dtype=float)
    tan_e = np.tan(np.deg2rad(np.where(np.abs(e) < 1e-9, 1e-9, e)))
    arcsec = np.where(
        e > 85.0,
        0.0,
        np.where(
            e > 5.0,
            58.1 / tan_e - 0.07 / tan_e**3 + 0.000086 / tan_e**5,
            np.where(
                e > -0.575,
                1735.0 + e * (-518.2 + e * (103.4 + e * (-12.79 + e * 0.711))),
                -20.772 / tan_e,
            ),
        ),
    )
    return arcsec / 3600.0


def position(
    index, latitude: float, longitude: float, refract: bool = True
) -> pd.DataFrame:
    """Apparent solar zenith, elevation and azimuth for stamps read as UTC.

    Azimuth is degrees clockwise from true north. Pass an index already shifted to
    UTC -- infer_utc_offset() is what tells you by how much.
    """
    idx = _utc_naive(index)
    declination, eot = _ecliptic(idx.to_julian_date().to_numpy(dtype=float))
    hour_angle = _hour_angle(idx, eot, longitude)

    lat = np.deg2rad(latitude)
    cos_zenith = np.clip(
        np.sin(lat) * np.sin(declination)
        + np.cos(lat) * np.cos(declination) * np.cos(hour_angle),
        -1.0,
        1.0,
    )
    zenith = np.arccos(cos_zenith)

    # degenerate with the sun overhead or at the pole, where azimuth means nothing
    denominator = np.cos(lat) * np.sin(zenith)
    safe = np.abs(denominator) > 1e-12
    azimuth = np.rad2deg(
        np.arccos(
            np.clip(
                np.where(
                    safe,
                    (np.sin(lat) * cos_zenith - np.sin(declination))
                    / np.where(safe, denominator, 1.0),
                    0.0,
                ),
                -1.0,
                1.0,
            )
        )
    )
    azimuth = np.where(
        hour_angle > 0, (azimuth + 180.0) % 360.0, (540.0 - azimuth) % 360.0
    )

    elevation = 90.0 - np.rad2deg(zenith)
    if refract:
        elevation = elevation + refraction(elevation)

    return pd.DataFrame(
        {"zenith": 90.0 - elevation, "elevation": elevation, "azimuth": azimuth},
        index=idx,
    )


def deviation(
    index,
    latitude: float,
    longitude: float,
    elevation=None,
    azimuth=None,
    refract: bool = True,
) -> np.ndarray:
    """Per-row angular miss between recorded angles and the position at `index`.

    Elevation and azimuth combined in quadrature, in degrees. Whichever of the
    two is given is used; azimuth wraps at 360, so it is compared the short way
    round.
    """
    calculated = position(index, latitude, longitude, refract=refract)
    misses = []
    if elevation is not None:
        misses.append(calculated["elevation"].to_numpy() - np.asarray(elevation, float))
    if azimuth is not None:
        gap = calculated["azimuth"].to_numpy() - np.asarray(azimuth, float)
        misses.append((gap + 180.0) % 360.0 - 180.0)
    return np.sqrt(np.sum(np.square(misses), axis=0))


# How far either side of the analytic estimate the refinement sweep looks, and
# how finely. Wide enough to cover the estimate going soft near solar noon.
_REFINE_SPAN = pd.Timedelta("5min")
_REFINE_STEP = pd.Timedelta("1s")

# The span real UTC offsets occupy, Baker Island to Kiritimati. Used to decide
# which whole-day alias of an inferred offset could be a timezone at all.
MIN_UTC_OFFSET = pd.Timedelta("-12h")
MAX_UTC_OFFSET = pd.Timedelta("14h")
_DAY = pd.Timedelta("24h")

# Step for the sweep of last resort, when the recorded angles cannot be inverted
# at all. Fine enough that the true minimum falls inside a refinement span of
# whichever grid point wins.
_COARSE_STEP = pd.Timedelta("2min")


def infer_utc_offset(
    index,
    latitude: float,
    longitude: float,
    elevation,
    azimuth=None,
    refract: bool = True,
) -> pd.Timedelta:
    """How far the stamps in `index` run ahead of UTC, per the recorded angles.

    Subtract the result from the index to get UTC.

    Estimated first by inverting the recorded elevation into an hour angle and
    comparing it against the hour angle the stamps imply -- the difference is the
    offset, at 15 degrees to the hour. That is solved per row and reduced by
    median, so a few bad rows cannot move the answer, and iterated because
    declination and the equation of time themselves depend on the offset they are
    helping to find. It also locks on from arbitrarily far away, which a local
    search would not.

    The estimate then gets a second-by-second sweep, because inverting elevation
    goes soft within a few degrees of due south: elevation is stationary at solar
    noon, so arccos there turns a hair of angle into minutes of clock. The sweep
    scores elevation and azimuth together, and azimuth swings fastest exactly
    where elevation stalls.

    Pass `azimuth` whenever there is one. Elevation on its own is mirror-symmetric
    about solar noon, and while both readings are tried and the better fit kept, a
    file lying wholly on one side of noon can fit its own mirror just as well --
    in which case the answer is right about the sun and can still be hours out
    about the clock.

    A time shift and a longitude error move the sun the same way -- four minutes
    of clock to the degree -- so over a short file the two cannot be told apart.
    What comes back is the offset that reconciles the angles with the *configured*
    position, nothing more.

    Returns NaT only when there is nothing to fit at all -- every angle missing.
    Angles the site could never see still come back with the offset that gets
    closest, so a caller can say how far off they stayed rather than only that
    they were impossible.
    """
    idx = _utc_naive(index)
    elev = np.asarray(elevation, dtype=float)
    if len(idx) == 0 or not np.isfinite(elev).any():
        return pd.NaT  # nothing to fit, so no honest answer to give
    true_elev = elev - refraction(elev) if refract else elev
    azi = None if azimuth is None else np.asarray(azimuth, dtype=float)
    lat = np.deg2rad(latitude)

    def estimate_for(afternoon) -> pd.Timedelta:
        """Analytic offset, given which side of solar noon each row sits."""
        offset_hours = 0.0
        for _ in range(3):
            shifted = idx - pd.Timedelta(seconds=offset_hours * 3600.0)
            declination, eot = _ecliptic(shifted.to_julian_date().to_numpy(dtype=float))
            calculated = _hour_angle(shifted, eot, longitude)

            cos_recorded = (
                np.sin(np.deg2rad(true_elev)) - np.sin(lat) * np.sin(declination)
            ) / (np.cos(lat) * np.cos(declination))
            # that is |hour angle|; afternoon is the half that carries the sign
            recorded = np.where(afternoon, 1.0, -1.0) * np.arccos(
                np.where(np.abs(cos_recorded) <= 1.0, cos_recorded, np.nan)
            )

            # the hour angle wraps every 24h; keep the nearest reading of the shift
            delta = np.rad2deg(calculated - recorded) / 15.0
            delta = (delta + 12.0) % 24.0 - 12.0
            if not np.isfinite(delta).any():
                return pd.NaT
            offset_hours += float(np.nanmedian(delta))
        return pd.Timedelta(seconds=offset_hours * 3600.0).round(_REFINE_STEP)

    # Elevation alone is mirror-symmetric about solar noon: the same height
    # happens once going up and once coming down, so it cannot say which. The
    # azimuth settles it outright; without one, both readings go forward as
    # candidates and the fit picks between them.
    if azi is not None:
        estimates = [estimate_for(azi > 180.0)]
    else:
        estimates = [estimate_for(True), estimate_for(False)]
    estimates = [e for e in estimates if not pd.isna(e)]

    # a handful of rows carries the shape of the sweep; sampling keeps the
    # second-by-second scan cheap on files with thousands of them
    step = max(1, len(idx) // 240)
    sample = slice(None, None, step)

    def score(candidate) -> float:
        missed = deviation(
            idx[sample] - pd.Timedelta(candidate),
            latitude,
            longitude,
            elevation=elev[sample],
            azimuth=None if azi is None else azi[sample],
            refract=refract,
        )
        usable = missed[np.isfinite(missed)]
        return float(usable.mean()) if usable.size else np.nan

    # An elevation the site cannot reach on that date inverts to nothing at all --
    # arccos runs out of domain -- and an angle that impossible is exactly what
    # wants quantifying. So sweep the whole plausible range instead and hand back
    # the closest the sun ever comes; the caller sees how far off it stayed.
    if not estimates:
        span = int((MAX_UTC_OFFSET - MIN_UTC_OFFSET) / _COARSE_STEP)
        coarse = MIN_UTC_OFFSET + np.arange(span + 1) * _COARSE_STEP
        scanned = [score(candidate) for candidate in coarse]
        if not np.isfinite(scanned).any():
            return pd.NaT
        estimates = [pd.Timedelta(coarse[int(np.nanargmin(scanned))])]

    steps = int(_REFINE_SPAN / _REFINE_STEP)

    def refine(around) -> tuple[float, pd.Timedelta]:
        """The best-scoring offset within a sweep either side of `around`."""
        tried = around + np.arange(-steps, steps + 1) * _REFINE_STEP
        scores = [score(candidate) for candidate in tried]
        if not np.isfinite(scores).any():
            return np.inf, pd.Timedelta(around)
        best = int(np.nanargmin(scores))
        return scores[best], pd.Timedelta(tried[best])

    # The hour angle wraps every 24h, so the estimate came back folded into
    # (-12h, +12h] and a genuine UTC+13 reads as UTC-11. Whole-day aliases are
    # told apart only by declination having moved overnight: decisive near an
    # equinox, nearly mute at a solstice and at the equator, so this settles a
    # real ambiguity rather than removing one. Each alias is refined before they
    # are compared, so none loses for being badly centred rather than wrong.
    # Offsets inside +/-10h have no alias in range and cost a single sweep. The
    # band is widened by the sweep, since an alias landing just outside it is one
    # the sweep can still walk back in -- the estimate need only be close.
    aliases = [
        candidate
        for estimate in estimates
        for shift in (-_DAY, pd.Timedelta(0), _DAY)
        if MIN_UTC_OFFSET - _REFINE_SPAN
        <= (candidate := estimate + shift)
        <= MAX_UTC_OFFSET + _REFINE_SPAN
    ]
    best = min((refine(candidate) for candidate in aliases), default=None)
    return estimates[0] if best is None or not np.isfinite(best[0]) else best[1]


def nearest_timezone(offset: pd.Timedelta) -> pd.Timedelta:
    """`offset` snapped to the quarter-hour lattice real UTC offsets sit on."""
    if pd.isna(offset):
        return pd.NaT
    return pd.Timedelta(round(offset / TIMEZONE_STEP) * TIMEZONE_STEP)


def format_offset(offset: pd.Timedelta) -> str:
    """An offset written the way a timezone is, as UTC+HH:MM."""
    if pd.isna(offset):
        return "UTC?"
    total = int(round(offset.total_seconds()))
    sign = "-" if total < 0 else "+"
    hours, minutes, seconds = (
        abs(total) // 3600,
        (abs(total) % 3600) // 60,
        abs(total) % 60,
    )
    stamp = f"UTC{sign}{hours:02d}:{minutes:02d}"
    return f"{stamp}:{seconds:02d}" if seconds else stamp


def format_skew(delta: pd.Timedelta) -> str:
    """A short signed duration, the way clock drift is quoted: -18s, +2m55s.

    Timedelta's own repr turns 18 seconds behind into "-1 days +23:59:42", which
    is no way to read a finding.
    """
    if pd.isna(delta):
        return "?"
    total = int(round(delta.total_seconds()))
    sign = "-" if total < 0 else "+"
    hours, minutes, seconds = (
        abs(total) // 3600,
        (abs(total) % 3600) // 60,
        abs(total) % 60,
    )
    spelled = f"{hours}h" if hours else ""
    spelled += f"{minutes}m" if minutes else ""
    spelled += f"{seconds}s" if seconds or not spelled else ""
    return sign + spelled
