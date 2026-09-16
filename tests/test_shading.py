"""The lightning rod's shadow: geometry checked against closed forms and symmetry.

_disk_blocked() integrates the sun's chords across a strip, so it has an exact
answer to be checked against -- the area of a circle between two vertical lines,
which is elementary. That is the strongest test in the file and it pins the
soft-edged shadow curve that everything downstream reads as a depth.

The rest is symmetry. A boom pointing east is shaded by the afternoon sun and one
pointing west by the morning sun; the season a notch happens in is symmetric
about the equinoxes. Neither depends on the implementation, and both would break
under a sign error in the azimuth convention -- which is the mistake worth
guarding against here, since the answer would still look plausible.

Charts are not tested. They are checked by looking at them.
"""

import numpy as np
import pandas as pd
import pytest

from conftest import LATITUDE, LONGITUDE
from se_analysis import shading as sh

# The angular radius of the sun, which sets the scale of every soft edge here.
SUN_RADIUS = np.deg2rad(sh.SUN_ANGULAR_DIAMETER / 2.0)


def strip_of_disk(half_width: float, radius: float) -> float:
    """Exact fraction of a disk covered by a centred strip, from first principles.

    The area of a circle of radius r between x = -w and x = +w is
    2 * [w*sqrt(r^2 - w^2) + r^2 * arcsin(w/r)], over a total of pi*r^2. Written
    out independently so _disk_blocked() is checked against the mathematics
    rather than against itself.
    """
    w = min(half_width, radius)
    area = 2.0 * (w * np.sqrt(radius**2 - w**2) + radius**2 * np.arcsin(w / radius))
    return area / (np.pi * radius**2)


class TestGrazingElevation:
    """The one number that says whether a tower has this problem at all."""

    @pytest.mark.parametrize(
        "rod,boom", [(1.5, 4.0), (1.0, 1.0), (0.5, 10.0), (3.0, 2.0)]
    )
    def test_it_is_the_arctangent_of_rod_over_boom(self, rod, boom):
        # The rod's tip throws its shadow rod/tan(elevation) out from the mast, so
        # the grazing angle is where that equals the boom.
        assert sh.grazing_elevation(rod, boom) == pytest.approx(
            np.rad2deg(np.arctan(rod / boom))
        )

    def test_equal_rod_and_boom_graze_at_forty_five_degrees(self):
        assert sh.grazing_elevation(4.0, 4.0) == pytest.approx(45.0)

    def test_a_longer_boom_lowers_the_grazing_angle(self):
        # Reach the sensor out further and only a lower sun can ever shade it.
        assert sh.grazing_elevation(1.5, 8.0) < sh.grazing_elevation(1.5, 4.0)


class TestDiskBlocked:
    """The soft edge of the shadow, against the exact strip-of-a-circle area."""

    @pytest.mark.parametrize("fraction", [0.1, 0.25, 0.5, 0.75, 0.9, 1.0])
    def test_a_centred_strip_matches_the_closed_form(self, fraction):
        half_width = fraction * SUN_RADIUS
        blocked = sh._disk_blocked(np.array([0.0]), np.array([half_width]), SUN_RADIUS)[
            0
        ]

        assert blocked == pytest.approx(strip_of_disk(half_width, SUN_RADIUS), abs=1e-9)

    def test_an_infinitely_thin_bar_blocks_nothing(self):
        blocked = sh._disk_blocked(np.array([0.0]), np.array([0.0]), SUN_RADIUS)[0]

        assert blocked == 0.0

    def test_a_bar_as_wide_as_the_disk_blocks_all_of_it(self):
        blocked = sh._disk_blocked(np.array([0.0]), np.array([SUN_RADIUS]), SUN_RADIUS)[
            0
        ]

        assert blocked == pytest.approx(1.0)

    def test_a_bar_clear_of_the_disk_blocks_nothing(self):
        blocked = sh._disk_blocked(
            np.array([3.0 * SUN_RADIUS]), np.array([0.5 * SUN_RADIUS]), SUN_RADIUS
        )[0]

        assert blocked == 0.0

    def test_blocking_falls_away_as_the_bar_slides_off(self):
        misses = np.linspace(0.0, 2.0 * SUN_RADIUS, 25)
        half_width = np.full_like(misses, 0.3 * SUN_RADIUS)
        blocked = sh._disk_blocked(misses, half_width, SUN_RADIUS)

        assert np.all(np.diff(blocked) <= 1e-12)
        assert blocked[0] > 0.0
        assert blocked[-1] == 0.0

    def test_the_result_is_always_a_fraction(self):
        misses = np.linspace(-3.0 * SUN_RADIUS, 3.0 * SUN_RADIUS, 101)
        half_width = np.full_like(misses, 2.0 * SUN_RADIUS)
        blocked = sh._disk_blocked(misses, half_width, SUN_RADIUS)

        assert np.all((blocked >= 0.0) & (blocked <= 1.0))

    def test_a_thicker_rod_blocks_more(self):
        thin = sh._disk_blocked(
            np.array([0.0]), np.array([0.2 * SUN_RADIUS]), SUN_RADIUS
        )
        thick = sh._disk_blocked(
            np.array([0.0]), np.array([0.6 * SUN_RADIUS]), SUN_RADIUS
        )

        assert thick[0] > thin[0]


class TestAngularMiss:
    """How far the sun sits off the rod, seen from the sensor."""

    @staticmethod
    def geometry(boom_len=4.0, tower_ht=8.0, rod_len=1.5, boom_azimuth=90.0):
        bearing = np.deg2rad(boom_azimuth)
        sensor = np.array(
            [boom_len * np.sin(bearing), boom_len * np.cos(bearing), tower_ht]
        )
        offset = np.array([0.0, 0.0, tower_ht]) - sensor
        return offset, np.array([0.0, 0.0, rod_len])

    def test_a_sun_along_the_rod_base_direction_misses_by_nothing(self):
        offset, along = self.geometry()
        # look straight from the sensor at the rod's base
        direction = offset / np.linalg.norm(offset)

        miss, distance = sh._angular_miss(direction[None, :], offset, along)

        assert miss[0] == pytest.approx(0.0, abs=1e-9)
        assert distance[0] == pytest.approx(np.linalg.norm(offset))

    def test_the_closed_form_matches_a_brute_force_search(self):
        # _angular_miss solves the nearest-approach angle in closed form, which is
        # the one piece of algebra here that cannot be eyeballed. Sampling the rod
        # finely and taking the minimum gets the same answer the slow way, so this
        # is what says the stationary-point solution and its endpoint clamping are
        # right -- across a whole day, including the sun behind the tower.
        offset, along = self.geometry()
        day = pd.date_range("2024-04-01", periods=288, freq="5min", tz="UTC")
        sky = sh.shade(day, LATITUDE, LONGITUDE)
        sun = sh._sun_vector(sky["elevation"], sky["azimuth"])

        miss, distance = sh._angular_miss(sun, offset, along)

        # every point of the rod, as seen from the sensor
        t = np.linspace(0.0, 1.0, 4001)[:, None]
        points = offset + t * along
        units = points / np.linalg.norm(points, axis=1, keepdims=True)
        brute = np.arccos(np.clip(sun @ units.T, -1.0, 1.0)).min(axis=1)

        assert np.allclose(miss, brute, atol=1e-5)
        assert np.all(miss >= 0.0)
        assert np.all(miss <= np.pi)
        # the reported distance is to whichever point of the rod that was
        assert np.all(distance >= np.linalg.norm(points, axis=1).min() - 1e-9)
        assert np.all(distance <= np.linalg.norm(points, axis=1).max() + 1e-9)

    def test_the_nearest_point_can_be_an_endpoint(self):
        # The stationary point of the angle may fall outside the rod, in which
        # case the closest approach is at whichever end -- so the returned
        # distance never exceeds the further endpoint.
        offset, along = self.geometry()
        sun = sh._sun_vector([45.0], [270.0])

        _, distance = sh._angular_miss(sun, offset, along)

        assert distance[0] <= np.linalg.norm(offset + along) + 1e-9
        assert (
            distance[0]
            >= min(np.linalg.norm(offset), np.linalg.norm(offset + along)) - 1e-9
        )


class TestSunVector:
    """Unit vectors in east/north/up, matching solar.position()'s convention."""

    @pytest.mark.parametrize(
        "azimuth,expected",
        [
            (0.0, (0.0, 1.0, 0.0)),  # due north
            (90.0, (1.0, 0.0, 0.0)),  # due east
            (180.0, (0.0, -1.0, 0.0)),  # due south
            (270.0, (-1.0, 0.0, 0.0)),  # due west
        ],
    )
    def test_a_sun_on_the_horizon_points_along_the_compass(self, azimuth, expected):
        vector = sh._sun_vector([0.0], [azimuth])[0]

        assert vector == pytest.approx(np.array(expected), abs=1e-12)

    def test_the_zenith_points_straight_up(self):
        assert sh._sun_vector([90.0], [0.0])[0] == pytest.approx(
            np.array([0.0, 0.0, 1.0]), abs=1e-12
        )

    def test_they_are_unit_vectors(self):
        vectors = sh._sun_vector([0.0, 15.0, 45.0, 80.0], [10.0, 100.0, 200.0, 330.0])

        assert np.allclose(np.linalg.norm(vectors, axis=1), 1.0)


class TestShade:
    """The whole calculation, checked by physics rather than by stored numbers."""

    def test_a_sun_below_the_horizon_blocks_nothing(self):
        # Local midnight: whatever the geometry says, there is no beam to block.
        night = pd.date_range("2024-04-01 04:00", periods=60, freq="1min", tz="UTC")
        track = sh.shade(night, LATITUDE, LONGITUDE)

        assert (track["elevation"] < 0.0).all()
        assert (track["blocked"] == 0.0).all()

    def test_blocked_is_always_a_fraction(self):
        day = pd.date_range("2024-04-01", periods=1440, freq="1min", tz="UTC")
        track = sh.shade(day, LATITUDE, LONGITUDE)

        assert track["blocked"].between(0.0, 1.0).all()

    def test_a_tz_aware_index_keeps_its_timezone(self):
        # The only way the answer reads as a time of day.
        local = pd.date_range(
            "2024-04-01", periods=10, freq="1min", tz="America/New_York"
        )
        track = sh.shade(local, LATITUDE, LONGITUDE)

        assert str(track.index.tz) == "America/New_York"

    def test_an_east_boom_is_shaded_by_the_afternoon_sun(self):
        # The shadow falls away from the sun, so an east-pointing boom is shaded
        # when the sun is in the west. A sign error in the azimuth convention
        # would put this in the morning and still look plausible.
        track = sh.shadow_track(
            "2024-04-01", LATITUDE, LONGITUDE, tz="America/New_York"
        )
        shaded = track[track["blocked"] > sh.DEFAULT_THRESHOLD]

        assert not shaded.empty
        assert (shaded.index.hour >= 12).all()
        assert shaded["azimuth"].mean() == pytest.approx(270.0, abs=2.0)

    def test_a_west_boom_is_shaded_by_the_morning_sun(self):
        track = sh.shadow_track(
            "2024-04-01",
            LATITUDE,
            LONGITUDE,
            tz="America/New_York",
            boom_azimuth=270.0,
        )
        shaded = track[track["blocked"] > sh.DEFAULT_THRESHOLD]

        assert not shaded.empty
        assert (shaded.index.hour < 12).all()
        assert shaded["azimuth"].mean() == pytest.approx(90.0, abs=2.0)

    def test_shading_happens_near_the_grazing_elevation_at_most(self):
        # grazing_elevation() treats the rod's tip as a point; shade() gives the
        # rod a thickness and the sun a disk, so partial blocking survives a
        # little above that ideal angle -- by about the sun's own radius, not by
        # degrees. This pins the two against each other.
        track = sh.shadow_track(
            "2024-04-01", LATITUDE, LONGITUDE, tz="America/New_York"
        )
        shaded = track[track["blocked"] > sh.DEFAULT_THRESHOLD]

        assert shaded["elevation"].max() < sh.grazing_elevation() + 1.0

    def test_a_sensor_at_the_top_of_the_tower_is_the_default(self):
        day = pd.date_range("2024-04-01", periods=120, freq="1min", tz="UTC")
        implied = sh.shade(day, LATITUDE, LONGITUDE)
        explicit = sh.shade(day, LATITUDE, LONGITUDE, sensor_ht=sh.TOWER_HT)

        assert np.allclose(implied["blocked"], explicit["blocked"])

    def test_reach_explains_the_result(self):
        # Reach is reported rather than used: below the boom length the tip's
        # shadow lands short of the sensor.
        day = pd.date_range("2024-04-01", periods=1440, freq="1min", tz="UTC")
        track = sh.shade(day, LATITUDE, LONGITUDE)
        blocking = track[track["blocked"] > 0.5]

        assert (blocking["reach"] > sh.BOOM_LEN * 0.5).all()


class TestShadowTrack:
    def test_a_local_day_runs_midnight_to_midnight(self):
        track = sh.shadow_track(
            "2024-04-01", LATITUDE, LONGITUDE, tz="America/New_York", freq="1min"
        )

        assert track.index[0].hour == 0
        assert track.index[0].minute == 0
        # closed on the left, so consecutive dates tile rather than double-counting
        assert track.index[-1].date() == pd.Timestamp("2024-04-01").date()
        assert len(track) == 1440

    def test_a_dst_spring_forward_day_is_an_hour_short(self):
        # Localised as two midnights rather than counted off as a fixed number of
        # steps, so the day still runs midnight to midnight.
        track = sh.shadow_track(
            "2024-03-10", LATITUDE, LONGITUDE, tz="America/New_York", freq="1min"
        )

        assert len(track) == 1380  # 23 hours

    def test_a_dst_fall_back_day_is_an_hour_long(self):
        track = sh.shadow_track(
            "2024-11-03", LATITUDE, LONGITUDE, tz="America/New_York", freq="1min"
        )

        assert len(track) == 1500  # 25 hours

    def test_without_a_timezone_the_day_is_utc(self):
        track = sh.shadow_track("2024-04-01", LATITUDE, LONGITUDE, freq="1min")

        assert track.index.tz is None
        assert len(track) == 1440


class TestShadowWindows:
    def test_a_day_with_no_shadow_returns_the_empty_schema(self):
        # The solstice sun passes due west too high for the rod to reach out.
        track = sh.shadow_track(
            "2024-06-21", LATITUDE, LONGITUDE, tz="America/New_York"
        )
        windows = sh.shadow_windows(track)

        assert windows.empty
        assert list(windows.columns) == [
            "start",
            "end",
            "duration",
            "peak",
            "elevation",
            "azimuth",
        ]

    def test_one_row_per_unbroken_run(self):
        track = sh.shadow_track(
            "2024-04-01", LATITUDE, LONGITUDE, tz="America/New_York"
        )
        windows = sh.shadow_windows(track)

        assert len(windows) == 1
        assert windows.iloc[0]["duration"] > pd.Timedelta(0)
        assert 0.0 < windows.iloc[0]["peak"] <= 1.0

    def test_the_window_brackets_everything_over_threshold(self):
        track = sh.shadow_track(
            "2024-04-01", LATITUDE, LONGITUDE, tz="America/New_York"
        )
        windows = sh.shadow_windows(track)
        over = track[track["blocked"] > sh.DEFAULT_THRESHOLD]

        assert windows.iloc[0]["start"] == over.index[0]
        assert windows.iloc[0]["end"] >= over.index[-1]

    def test_duration_includes_one_step_so_a_single_sample_is_not_zero(self):
        # A window one sample wide is one step long, not instantaneous.
        index = pd.date_range("2024-04-01 18:00", periods=5, freq="10s", tz="UTC")
        track = pd.DataFrame(
            {
                "blocked": [0.0, 0.0, 0.5, 0.0, 0.0],
                "elevation": 7.0,
                "azimuth": 270.0,
            },
            index=index,
        )
        windows = sh.shadow_windows(track)

        assert len(windows) == 1
        assert windows.iloc[0]["duration"] == pd.Timedelta("10s")

    def test_a_higher_threshold_narrows_the_window(self):
        track = sh.shadow_track(
            "2024-04-01", LATITUDE, LONGITUDE, tz="America/New_York"
        )
        wide = sh.shadow_windows(track, threshold=0.01)
        narrow = sh.shadow_windows(track, threshold=0.9)

        assert narrow.iloc[0]["duration"] < wide.iloc[0]["duration"]

    def test_two_separate_runs_are_two_rows(self):
        index = pd.date_range("2024-04-01 18:00", periods=7, freq="10s", tz="UTC")
        track = pd.DataFrame(
            {
                "blocked": [0.5, 0.0, 0.0, 0.5, 0.5, 0.0, 0.5],
                "elevation": 7.0,
                "azimuth": 270.0,
            },
            index=index,
        )

        assert len(sh.shadow_windows(track)) == 3


class TestShadowSeason:
    """Which weeks of the year have a notch at all."""

    @pytest.fixture(scope="class")
    @classmethod
    def season(cls):
        # A year at a minute's resolution, walked once for the whole class.
        return sh.shadow_season(
            "2024-01-01", "2024-12-31", LATITUDE, LONGITUDE, tz="America/New_York"
        )

    def test_one_row_per_date(self, season):
        assert len(season) == 366  # 2024 is a leap year

    def test_the_notch_seasons_straddle_the_equinoxes(self, season):
        # An east-pointing boom is shaded when the setting sun crosses due west,
        # which happens around the equinoxes -- so the affected dates fall into
        # two blocks, one either side of the year's midpoint, and neither touches
        # the solstices.
        affected = season[season["minutes"] > 0].index

        assert not affected.empty
        months = set(affected.month)
        assert months <= {3, 4, 8, 9}
        assert 6 not in months  # never at the June solstice
        assert 12 not in months  # never at the December one

    def test_the_two_seasons_are_roughly_symmetric_about_the_solstice(self, season):
        affected = season[season["minutes"] > 0].index
        spring = affected[affected.month <= 6]
        autumn = affected[affected.month > 6]

        # Days either side of the June solstice, which the geometry is symmetric
        # about; a week of slack for the eccentricity of the orbit.
        solstice = pd.Timestamp("2024-06-20").dayofyear
        assert (
            abs(
                (solstice - spring.dayofyear.max())
                - (autumn.dayofyear.min() - solstice)
            )
            < 7
        )

    def test_peak_elevation_never_beats_the_grazing_angle_by_much(self, season):
        affected = season[season["minutes"] > 0]

        assert affected["elevation"].max() < sh.grazing_elevation() + 1.0

    def test_days_with_no_event_report_no_minutes(self, season):
        quiet = season[season["minutes"] == 0]

        assert not quiet.empty
        assert (quiet["peak"] <= sh.DEFAULT_THRESHOLD).all()

    def test_a_single_date_span_is_one_row(self):
        one = sh.shadow_season("2024-04-01", "2024-04-01", LATITUDE, LONGITUDE)

        assert len(one) == 1
