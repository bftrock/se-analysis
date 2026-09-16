"""The ephemeris, checked against astronomy rather than against its own output.

solar.position() claims agreement with NREL SPA to about 0.01 degrees. Freezing
whatever it currently returns would pin that claim to itself, so instead these
assert facts that hold independently of any implementation: the obliquity of the
ecliptic at the solstices, the annual extremes of the equation of time, and the
identity that solar noon elevation is 90 - |latitude - declination|. If the
algorithm is ever rewritten, these still say whether it is right.

The offset inference is tested by round trip: synthesize the angles a logger
would record at a known offset, then ask for the offset back.
"""

import numpy as np
import pandas as pd
import pytest

from conftest import LATITUDE, LONGITUDE, site_info
from se_analysis import logr, solar

# Obliquity of the ecliptic in 2024, which is the declination the sun reaches at
# either solstice. It drifts by well under an arcminute a year, so it is good to
# four decimals for any date these tools see.
OBLIQUITY = 23.4386

# 2024 equinoxes and solstices, UTC, to the minute.
EQUINOX_MARCH = "2024-03-20 03:06"
SOLSTICE_JUNE = "2024-06-20 20:51"
EQUINOX_SEPTEMBER = "2024-09-22 12:44"
SOLSTICE_DECEMBER = "2024-12-21 09:21"


def declination_at(stamp) -> float:
    """Solar declination in degrees, for a single stamp read as UTC."""
    jd = pd.DatetimeIndex([pd.Timestamp(stamp)]).to_julian_date().to_numpy(float)
    return float(np.rad2deg(solar._ecliptic(jd)[0][0]))


def equation_of_time_at(stamp) -> float:
    """The equation of time in minutes, for a single stamp read as UTC."""
    jd = pd.DatetimeIndex([pd.Timestamp(stamp)]).to_julian_date().to_numpy(float)
    return float(solar._ecliptic(jd)[1][0])


class TestEcliptic:
    """Declination and the equation of time, the two terms everything rests on."""

    @pytest.mark.parametrize(
        "stamp,expected",
        [
            (EQUINOX_MARCH, 0.0),
            (SOLSTICE_JUNE, OBLIQUITY),
            (EQUINOX_SEPTEMBER, 0.0),
            (SOLSTICE_DECEMBER, -OBLIQUITY),
        ],
    )
    def test_declination_at_the_quarter_days(self, stamp, expected):
        # The sun crosses the celestial equator at an equinox and reaches the
        # obliquity at a solstice, by definition of those moments.
        assert declination_at(stamp) == pytest.approx(expected, abs=0.01)

    def test_equation_of_time_annual_extremes(self):
        # Published extremes: about -14.2 minutes in mid-February and +16.4 in
        # early November. This is the tightest independent check available on the
        # eccentricity and obliquity terms together.
        dates = pd.date_range("2024-01-01", "2024-12-31", freq="1D")
        _, eot = solar._ecliptic(dates.to_julian_date().to_numpy(float))

        assert eot.min() == pytest.approx(-14.2, abs=0.2)
        assert eot.max() == pytest.approx(16.4, abs=0.2)
        assert dates[eot.argmin()].month == 2
        assert dates[eot.argmax()].month == 11

    def test_equation_of_time_is_near_zero_at_its_crossings(self):
        # It passes through zero four times a year; mid-April and the start of
        # September are two of them.
        assert equation_of_time_at("2024-04-15 12:00") == pytest.approx(0.0, abs=0.5)
        assert equation_of_time_at("2024-09-01 12:00") == pytest.approx(0.0, abs=0.5)


class TestPosition:
    """Where the sun is, checked by the identities that hold at solar noon."""

    @pytest.mark.parametrize(
        "latitude,longitude,date",
        [
            (LATITUDE, LONGITUDE, EQUINOX_MARCH[:10]),
            (LATITUDE, LONGITUDE, SOLSTICE_JUNE[:10]),
            (LATITUDE, LONGITUDE, SOLSTICE_DECEMBER[:10]),
            (-33.9, 151.2, SOLSTICE_DECEMBER[:10]),  # Sydney
            (0.0, 0.0, EQUINOX_MARCH[:10]),  # the Gulf of Guinea
            (64.1, -21.9, SOLSTICE_JUNE[:10]),  # Reykjavik
        ],
    )
    def test_noon_elevation_is_ninety_less_the_zenith_distance(
        self, latitude, longitude, date
    ):
        # At the meridian crossing the sun's elevation is 90 degrees less the
        # angle between the site and the subsolar latitude. True at every site on
        # every date, and it ties position() to _ecliptic() independently.
        noon = solar._solar_noon(date, longitude).tz_localize("UTC")
        sky = solar.position(
            pd.DatetimeIndex([noon]), latitude, longitude, refract=False
        )

        expected = 90.0 - abs(latitude - declination_at(noon.tz_localize(None)))
        assert sky["elevation"].iloc[0] == pytest.approx(expected, abs=0.01)

    def test_sun_is_due_south_at_noon_north_of_the_tropics(self):
        noon = solar._solar_noon("2024-06-21", LONGITUDE).tz_localize("UTC")
        sky = solar.position(pd.DatetimeIndex([noon]), LATITUDE, LONGITUDE)

        # North of the Tropic of Cancer the sun is always south at midday.
        assert sky["azimuth"].iloc[0] == pytest.approx(180.0, abs=0.05)

    def test_sun_is_due_north_at_noon_south_of_the_tropics(self):
        noon = solar._solar_noon("2024-12-21", 151.2).tz_localize("UTC")
        sky = solar.position(pd.DatetimeIndex([noon]), -33.9, 151.2)

        # Azimuth is measured clockwise from north, so due north wraps.
        azimuth = sky["azimuth"].iloc[0] % 360.0
        assert min(azimuth, 360.0 - azimuth) == pytest.approx(0.0, abs=0.05)

    def test_noon_is_the_highest_the_sun_gets(self):
        day = pd.date_range("2024-04-01", periods=1440, freq="1min", tz="UTC")
        sky = solar.position(day, LATITUDE, LONGITUDE)
        noon = solar._solar_noon("2024-04-01", LONGITUDE)

        # The peak of the day should land on the computed meridian crossing.
        assert abs(sky["elevation"].idxmax() - noon) <= pd.Timedelta("1min")

    def test_the_returned_index_is_tz_naive_utc(self):
        # position() indexes its result on _utc_naive(index), so a tz-aware index
        # goes in and a naive one comes back. Worth pinning: comparing the result
        # index against a tz-aware stamp raises rather than returning a wrong
        # answer, and callers that only read the columns never notice either way.
        aware = pd.date_range("2024-04-01", periods=4, freq="1h", tz="US/Eastern")
        sky = solar.position(aware, LATITUDE, LONGITUDE)

        assert sky.index.tz is None
        assert (sky.index == aware.tz_convert("UTC").tz_localize(None)).all()

    def test_zenith_and_elevation_are_complementary(self):
        day = pd.date_range("2024-08-09", periods=200, freq="7min", tz="UTC")
        sky = solar.position(day, LATITUDE, LONGITUDE)

        assert np.allclose((sky["zenith"] + sky["elevation"]).to_numpy(), 90.0)

    def test_angles_stay_in_range_over_a_year(self):
        stamps = pd.date_range("2024-01-01", "2024-12-31", freq="53min", tz="UTC")
        sky = solar.position(stamps, LATITUDE, LONGITUDE)

        assert sky["azimuth"].between(0.0, 360.0).all()
        assert sky["elevation"].between(-90.0, 90.0).all()

    def test_a_naive_index_is_read_as_utc(self):
        naive = pd.date_range("2024-05-01", periods=24, freq="1h")

        assert np.allclose(
            solar.position(naive, LATITUDE, LONGITUDE)["elevation"].to_numpy(),
            solar.position(naive.tz_localize("UTC"), LATITUDE, LONGITUDE)[
                "elevation"
            ].to_numpy(),
        )

    def test_an_offset_index_converts_before_computing(self):
        utc = pd.date_range("2024-05-01 12:00", periods=8, freq="1h", tz="UTC")

        # Same instants, different wall clock: the sun cannot have moved.
        assert np.allclose(
            solar.position(utc, LATITUDE, LONGITUDE)["elevation"].to_numpy(),
            solar.position(utc.tz_convert("America/New_York"), LATITUDE, LONGITUDE)[
                "elevation"
            ].to_numpy(),
        )


class TestRefraction:
    """The atmosphere's lift, which is the whole error budget near the horizon."""

    def test_lift_at_the_horizon_is_about_half_a_degree(self):
        # The standard figure is ~34 arcmin for a sun on the *apparent* horizon,
        # which is a true elevation slightly below zero; at a true zero it is a
        # little under half a degree.
        assert solar.refraction(0.0) == pytest.approx(0.48, abs=0.05)
        assert solar.refraction(-0.5) == pytest.approx(0.56, abs=0.05)

    def test_lift_falls_away_as_the_sun_climbs(self):
        lift = solar.refraction(np.array([0.0, 1.0, 5.0, 10.0, 30.0, 60.0, 85.0]))

        assert np.all(np.diff(lift) < 0.0)

    def test_overhead_sun_is_not_refracted(self):
        assert solar.refraction(86.0) == 0.0
        assert solar.refraction(90.0) == 0.0

    def test_refraction_raises_the_apparent_sun(self):
        stamps = pd.date_range("2024-03-20 09:00", periods=120, freq="5min", tz="UTC")
        apparent = solar.position(stamps, LATITUDE, LONGITUDE, refract=True)
        geometric = solar.position(stamps, LATITUDE, LONGITUDE, refract=False)

        lift = (apparent["elevation"] - geometric["elevation"]).to_numpy()
        assert np.all(lift >= 0.0)
        # and it is worth having: a tenth of a degree or more somewhere in the day
        assert lift.max() > 0.1


class TestInferUtcOffset:
    """Recovering the offset a file's stamps were written at, by round trip."""

    @staticmethod
    def recorded(offset, date="2024-05-15", refract=True):
        """Stamps `offset` ahead of UTC, and the angles a logger would record."""
        utc = pd.date_range(date, periods=1440, freq="1min", tz="UTC")
        return utc + offset, solar.position(utc, LATITUDE, LONGITUDE, refract=refract)

    @pytest.mark.parametrize(
        "hours",
        [
            0,  # already UTC, the state these files are migrating to
            -5,  # US Eastern
            -8,
            5.5,  # India, off the whole hour
            5.75,  # Nepal, off the half hour
            -11,
            13,  # past +12, so the whole-day alias has to be resolved
            9.75,
        ],
    )
    def test_offset_is_recovered_from_elevation_and_azimuth(self, hours):
        offset = pd.Timedelta(hours=hours)
        stamps, truth = self.recorded(offset)

        found = solar.infer_utc_offset(
            stamps,
            LATITUDE,
            LONGITUDE,
            truth["elevation"].to_numpy(),
            truth["azimuth"].to_numpy(),
        )
        assert found == offset

    @pytest.mark.parametrize("hours", [0, -5, 5.5])
    def test_offset_is_recovered_from_elevation_alone_over_a_full_day(self, hours):
        # Elevation is mirror-symmetric about solar noon, so on its own it cannot
        # say which side of noon a row sits. A full day covers both sides, which
        # is what lets the fit settle it -- see the docstring's warning about a
        # file lying wholly on one side.
        offset = pd.Timedelta(hours=hours)
        stamps, truth = self.recorded(offset)

        found = solar.infer_utc_offset(
            stamps, LATITUDE, LONGITUDE, truth["elevation"].to_numpy()
        )
        assert abs(found - offset) <= pd.Timedelta("1min")

    def test_geometric_angles_are_recovered_with_refract_off(self):
        # A logger computing without refraction has to be read the same way; this
        # is the pin behind measurement.solar_angles(refract=False).
        offset = pd.Timedelta(hours=-5)
        stamps, truth = self.recorded(offset, refract=False)

        found = solar.infer_utc_offset(
            stamps,
            LATITUDE,
            LONGITUDE,
            truth["elevation"].to_numpy(),
            truth["azimuth"].to_numpy(),
            refract=False,
        )
        assert found == offset

    def test_nothing_to_fit_returns_nat(self):
        stamps = pd.date_range("2024-05-15", periods=10, freq="1min", tz="UTC")

        assert pd.isna(
            solar.infer_utc_offset(
                stamps, LATITUDE, LONGITUDE, np.full(len(stamps), np.nan)
            )
        )

    def test_an_empty_index_returns_nat(self):
        assert pd.isna(
            solar.infer_utc_offset(
                pd.DatetimeIndex([], tz="UTC"), LATITUDE, LONGITUDE, np.array([])
            )
        )

    def test_impossible_angles_still_come_back_with_a_best_fit(self):
        # An elevation the site can never reach inverts to nothing at all, and the
        # docstring promises the closest approach rather than NaT -- so a caller
        # can report how far off it stayed.
        stamps = pd.date_range("2024-12-21", periods=120, freq="1min", tz="UTC")
        impossible = np.full(len(stamps), 89.0)  # the sun is never overhead here

        found = solar.infer_utc_offset(stamps, LATITUDE, LONGITUDE, impossible)

        assert not pd.isna(found)
        assert solar.MIN_UTC_OFFSET <= found <= solar.MAX_UTC_OFFSET


class TestTimezoneHelpers:
    """Snapping an inferred offset to the lattice, and writing it down."""

    @pytest.mark.parametrize(
        "given,expected",
        [
            ("0s", "0h"),
            ("4h59m50s", "5h"),
            ("-4h58m", "-5h"),
            ("5h31m", "5h30m"),
            ("5h44m", "5h45m"),
            ("7m", "0h"),  # nearer UTC than any other timezone
        ],
    )
    def test_nearest_timezone_snaps_to_the_quarter_hour(self, given, expected):
        snapped = solar.nearest_timezone(pd.Timedelta(given))

        assert snapped == pd.Timedelta(expected)
        # every real UTC offset is a whole number of quarter hours
        assert snapped % solar.TIMEZONE_STEP == pd.Timedelta(0)

    def test_nearest_timezone_of_nat_is_nat(self):
        assert pd.isna(solar.nearest_timezone(pd.NaT))

    @pytest.mark.parametrize(
        "given,expected",
        [
            ("0s", "UTC+00:00"),
            ("-5h", "UTC-05:00"),
            ("5h30m", "UTC+05:30"),
            ("13h", "UTC+13:00"),
            ("-9h30m", "UTC-09:30"),
            ("1h2m3s", "UTC+01:02:03"),  # seconds only shown when there are some
        ],
    )
    def test_format_offset_reads_as_a_timezone(self, given, expected):
        assert solar.format_offset(pd.Timedelta(given)) == expected

    def test_format_offset_of_nat(self):
        assert solar.format_offset(pd.NaT) == "UTC?"

    @pytest.mark.parametrize(
        "given,expected",
        [
            ("0s", "+0s"),
            ("-18s", "-18s"),
            ("2m55s", "+2m55s"),
            ("-1h5m", "-1h5m"),
            ("3h", "+3h"),
        ],
    )
    def test_format_skew_is_short_and_signed(self, given, expected):
        # Timedelta's own repr turns 18 seconds behind into "-1 days +23:59:42",
        # which is the whole reason this function exists.
        assert solar.format_skew(pd.Timedelta(given)) == expected

    def test_format_skew_of_nat(self):
        assert solar.format_skew(pd.NaT) == "?"


class TestSiteCoordinates:
    """Reading the site's position out of a header, in either of its shapes."""

    def test_reads_the_two_column_frame(self, header):
        assert solar.site_coordinates(header) == (44.5, -73.2)

    def test_reads_the_parsed_dict(self, header):
        assert solar.site_coordinates(logr.parse_site_info(header)) == (44.5, -73.2)

    @pytest.mark.parametrize(
        "label", ["Latitude:", "latitude", "Site_Latitude", "LATITUDE"]
    )
    def test_labels_are_matched_loosely(self, label):
        info = site_info([(label, "44.5"), ("Longitude:", "-73.2")])

        assert solar.site_coordinates(info)[0] == 44.5

    def test_a_missing_coordinate_raises(self):
        info = site_info([("Latitude:", "44.5")])

        with pytest.raises(KeyError, match="longitude"):
            solar.site_coordinates(info)

    @pytest.mark.parametrize("value", ["", "n/a", "999", "-181"])
    def test_a_coordinate_that_is_not_a_position_raises(self, value):
        info = site_info([("Latitude:", "44.5"), ("Longitude:", value)])

        with pytest.raises(ValueError, match="not a position"):
            solar.site_coordinates(info)


class TestNormalizationWindow:
    """The hours either side of solar noon that irradiance is normalized over."""

    def test_window_is_centred_on_solar_noon(self, header):
        start, end = solar.get_normalization_window(header, "2024-06-21")
        noon = solar._solar_noon("2024-06-21", LONGITUDE).tz_localize("UTC")

        assert start + (end - start) / 2 == noon
        assert end - start == 2 * solar.NORMALIZATION_HALF_WIDTH

    def test_both_ends_come_back_in_utc(self, header):
        start, end = solar.get_normalization_window(header, "2024-06-21")

        assert str(start.tz) == "UTC"
        assert str(end.tz) == "UTC"

    def test_half_width_widens_both_ends(self, header):
        narrow = solar.get_normalization_window(header, "2024-06-21")
        wide = solar.get_normalization_window(
            header, "2024-06-21", half_width=pd.Timedelta("3h")
        )

        assert wide[0] < narrow[0]
        assert wide[1] > narrow[1]
        assert wide[1] - wide[0] == pd.Timedelta("6h")

    def test_a_header_with_no_position_fails_here(self):
        info = site_info([("Site Number:", "002682")])

        with pytest.raises(KeyError):
            solar.get_normalization_window(info, "2024-06-21")
