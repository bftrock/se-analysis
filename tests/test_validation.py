"""The check engine: the period arithmetic, and one test per kind of finding.

availability() and gap_runs() answer two different questions that are easy to
conflate, and the difference is the whole reason both exist: a run that started
late leaves no step in its own index for gap_runs to find, but is missing data
all the same. That distinction is pinned explicitly below.

expected_periods() is the other arithmetic worth pinning, because its
whole-day widening is deliberate and surprising -- a span written in dates
covers those dates end to end regardless of the time of day it was started at.
"""

import numpy as np
import pandas as pd
import pytest

from se_analysis import validation as v


class TestExpectedPeriods:
    """How many samples a span should have produced."""

    @pytest.mark.parametrize(
        "start,end,interval,expected",
        [
            ("2024-05-01", "2024-05-01", "10min", 144),
            ("2024-05-01", "2024-05-01", "1min", 1440),
            ("2024-05-01", "2024-05-02", "10min", 288),
            ("2024-05-01", "2024-05-07", "1h", 168),
        ],
    )
    def test_whole_days_at_a_cadence(self, start, end, interval, expected):
        assert v.expected_periods(start, end, interval) == expected

    def test_the_span_is_widened_to_day_boundaries(self):
        # A day named as the end of the span is a day that should be complete, so
        # the count does not move with the clock time the run was started at.
        assert v.expected_periods("2024-05-01 13:37", "2024-05-01 23:00", "1min") == (
            v.expected_periods("2024-05-01", "2024-05-01", "1min")
        )

    def test_a_bare_end_date_still_covers_its_whole_day(self):
        # Widening past midnight is unconditional, including for an end that
        # already sits on midnight.
        assert v.expected_periods("2024-05-01", "2024-05-01 00:00", "1h") == 24

    def test_an_interval_that_does_not_divide_evenly_counts_whole_periods(self):
        # The remainder is too short to hold another sample.
        assert v.expected_periods("2024-05-01", "2024-05-01", "7min") == 1440 // 7

    def test_end_before_start_raises(self):
        with pytest.raises(ValueError, match="precedes start"):
            v.expected_periods("2024-05-02", "2024-05-01", "1min")

    @pytest.mark.parametrize("interval", ["0min", "-10min"])
    def test_a_non_positive_interval_raises(self, interval):
        with pytest.raises(ValueError, match="must be positive"):
            v.expected_periods("2024-05-01", "2024-05-01", interval)


class TestAvailability:
    """Did we get the days we asked for -- as distinct from where the record stops."""

    def test_a_complete_day_is_a_hundred_percent(self, daily):
        percent, missing = v.availability("2024-05-01", "2024-05-01", daily)

        assert percent == 100.0
        assert missing.empty

    def test_a_hole_is_reported_as_one_run(self, daily):
        holed = daily.drop(daily.index[72:78])
        percent, missing = v.availability("2024-05-01", "2024-05-01", holed)

        assert percent == pytest.approx(100.0 * 138 / 144)
        assert len(missing) == 1
        assert missing.iloc[0]["n_missing"] == 6
        assert missing.iloc[0]["duration"] == pd.Timedelta("1h")
        # `end` is the stamp the record was due to resume at, so the duration is
        # the time actually unaccounted for
        assert missing.iloc[0]["start"] == pd.Timestamp("2024-05-01 12:00", tz="UTC")
        assert missing.iloc[0]["end"] == pd.Timestamp("2024-05-01 13:00", tz="UTC")

    def test_a_late_start_is_missing_data_that_gap_runs_cannot_see(self, daily):
        # The distinction the docstring draws: a run that started six hours late
        # has no step inside its own index, so gap_runs finds nothing, but a
        # quarter of the day never arrived.
        late = daily.iloc[36:]
        percent, missing = v.availability("2024-05-01", "2024-05-01", late)

        assert percent == pytest.approx(75.0)
        assert len(missing) == 1
        assert v.gap_runs(late).empty

    def test_an_early_stop_is_caught_the_same_way(self, daily):
        early = daily.iloc[:108]
        percent, _ = v.availability("2024-05-01", "2024-05-01", early)

        assert percent == pytest.approx(75.0)

    def test_several_stamps_in_one_slot_count_once(self, daily):
        # Real stamps carry jitter and are snapped to the grid; duplicates must
        # not inflate the figure past 100%.
        doubled = pd.concat([daily, daily.iloc[:10]]).sort_index()
        percent, _ = v.availability("2024-05-01", "2024-05-01", doubled)

        assert percent == 100.0

    def test_jittered_stamps_snap_to_their_slot(self, daily):
        jittered = daily.copy()
        jittered.index = daily.index + pd.Timedelta("7s")
        percent, _ = v.availability("2024-05-01", "2024-05-01", jittered)

        assert percent == 100.0

    def test_a_bare_date_is_matched_to_tz_aware_stamps(self, daily):
        # Bare dates are the natural way to ask for a span, so the caller is not
        # made to localize them by hand.
        percent, _ = v.availability("2024-05-01", "2024-05-01", daily)

        assert percent == 100.0

    def test_a_naive_index_works_too(self, daily):
        naive = daily.copy()
        naive.index = daily.index.tz_localize(None)

        assert v.availability("2024-05-01", "2024-05-01", naive)[0] == 100.0

    def test_stamps_outside_the_span_are_not_counted(self, daily):
        percent, _ = v.availability("2024-05-02", "2024-05-02", daily)

        assert percent == 0.0

    def test_a_frame_without_a_datetime_index_is_refused(self, daily):
        # A bare number would read as an epoch offset rather than failing, so the
        # row numbers of a frame still carrying its stamps in a column would pass
        # silently and be measured in nanoseconds.
        flat = daily.reset_index(names="Timestamp")

        with pytest.raises(TypeError, match="DatetimeIndex"):
            v.availability("2024-05-01", "2024-05-01", flat)

    def test_the_refusal_names_the_column_to_index_on(self, daily):
        flat = daily.reset_index(names="Timestamp")

        with pytest.raises(TypeError, match="set_index"):
            v.availability("2024-05-01", "2024-05-01", flat)

    def test_a_bare_number_interval_is_refused(self, daily):
        # pd.Timedelta reads a bare number as nanoseconds, which is never what
        # anyone means by a sampling interval.
        with pytest.raises(TypeError, match="Timedelta or a string"):
            v.availability("2024-05-01", "2024-05-01", daily, interval=10)

    def test_an_interval_longer_than_the_span_is_refused(self, daily):
        with pytest.raises(ValueError, match="longer than the requested span"):
            v.availability("2024-05-01", "2024-05-01", daily, interval="48h")

    def test_a_frame_too_short_to_infer_a_cadence_says_so(self):
        one = pd.DataFrame(
            {"x": [1.0]}, index=pd.DatetimeIndex(["2024-05-01"], tz="UTC")
        )

        with pytest.raises(ValueError, match="no usable interval"):
            v.availability("2024-05-01", "2024-05-01", one)


class TestSamplingInterval:
    def test_the_median_step_is_the_cadence(self, daily):
        assert v.sampling_interval(daily.index) == pd.Timedelta("10min")

    def test_a_few_odd_steps_do_not_move_the_median(self, daily):
        holed = daily.drop(daily.index[10:14])

        assert v.sampling_interval(holed.index) == pd.Timedelta("10min")


class TestGapRuns:
    """Where the record we did get stops."""

    def test_a_gap_free_frame_has_none(self, daily):
        assert v.gap_runs(daily).empty

    def test_a_gap_is_bounded_by_the_stamps_either_side(self, daily):
        holed = daily.drop(daily.index[72:78])
        gaps = v.gap_runs(holed)

        assert len(gaps) == 1
        assert gaps.iloc[0]["start"] == daily.index[71]
        assert gaps.iloc[0]["end"] == daily.index[78]
        # one step was due; everything beyond it never came
        assert gaps.iloc[0]["n_missing"] == 6

    def test_a_single_dropped_sample_is_caught(self, daily):
        holed = daily.drop(daily.index[50])
        gaps = v.gap_runs(holed)

        assert len(gaps) == 1
        assert gaps.iloc[0]["n_missing"] == 1

    def test_a_frame_of_one_row_has_no_gaps(self):
        one = pd.DataFrame(
            {"x": [1.0]}, index=pd.DatetimeIndex(["2024-05-01"], tz="UTC")
        )

        assert v.gap_runs(one).empty

    def test_gaps_are_reported_against_the_frame_not_per_signal(self, daily):
        # When polling stops every signal stops together, so one finding per gap
        # is what you go and look at.
        holed = daily.drop(daily.index[72:78])
        findings = v.check_gaps(holed)

        assert len(findings) == 1
        assert findings.iloc[0]["signal"] == v.ALL_SIGNALS
        assert findings.iloc[0]["check"] == "gap"


class TestCheckRanges:
    def test_a_signal_inside_its_band_reports_nothing(self, daily):
        assert v.check_ranges(daily, {"signal": (-1, 1000)}).empty

    def test_a_signal_outside_its_band_is_flagged_once(self, daily):
        findings = v.check_ranges(daily, {"signal": (0, 100)})

        assert len(findings) == 1
        assert findings.iloc[0]["check"] == "range"
        assert findings.iloc[0]["n_samples"] == 43  # values 101..143

    def test_nan_counts_as_out_of_range(self):
        data = pd.DataFrame(
            {"x": [1.0, np.nan, 2.0]},
            index=pd.date_range("2024-05-01", periods=3, freq="1min", tz="UTC"),
        )
        findings = v.check_ranges(data, {"x": (0, 10)})

        assert len(findings) == 1
        assert findings.iloc[0]["n_samples"] == 1

    def test_a_signal_the_file_does_not_carry_is_skipped(self, daily):
        # missing_signals() is what reports an absent column, not this.
        assert v.check_ranges(daily, {"absent": (0, 1)}).empty

    def test_the_findings_schema_is_complete_even_when_empty(self, daily):
        assert list(v.check_ranges(daily, {}).columns) == v.FINDING_COLUMNS


class TestCheckLesserAndEquivalent:
    @staticmethod
    def frame(**columns):
        rows = len(next(iter(columns.values())))
        index = pd.date_range("2024-05-01", periods=rows, freq="1min", tz="UTC")
        return pd.DataFrame(columns, index=index)

    def test_lesser_passes_when_the_order_holds(self):
        data = self.frame(dhi=[1.0, 2.0, 3.0, 4.0], ghi=[2.0, 3.0, 4.0, 5.0])

        assert v.check_lesser(data, [("dhi", "ghi")]).empty

    def test_lesser_flags_the_rows_that_invert(self):
        data = self.frame(dhi=[1.0, 9.0, 3.0, 4.0], ghi=[2.0, 3.0, 4.0, 5.0])
        findings = v.check_lesser(data, [("dhi", "ghi")])

        assert len(findings) == 1
        assert findings.iloc[0]["n_samples"] == 1
        assert findings.iloc[0]["signal"] == "dhi vs ghi"

    def test_equal_values_satisfy_lesser(self):
        data = self.frame(a=[1.0, 1.0], b=[1.0, 1.0])

        assert v.check_lesser(data, [("a", "b")]).empty

    def test_equivalent_flags_a_pair_that_disagrees(self):
        data = self.frame(left=[1.0, 1.0, 2.0, 1.0], right=[1.0, 1.0, 1.0, 1.0])
        findings = v.check_equivalent(data, [("left", "right")])

        assert len(findings) == 1
        assert findings.iloc[0]["check"] == "equivalent"

    def test_a_pair_with_a_missing_member_is_skipped(self):
        data = self.frame(left=[1.0, 1.0])

        assert v.check_equivalent(data, [("left", "absent")]).empty


class TestCheckStalls:
    """A signal that stopped moving while the data kept coming."""

    @staticmethod
    def frame(values, freq="1min"):
        index = pd.date_range("2024-05-01", periods=len(values), freq=freq, tz="UTC")
        return pd.DataFrame({"x": values}, index=index)

    def test_a_moving_signal_does_not_stall(self):
        data = self.frame(np.arange(60, dtype=float))

        assert v.check_stalls(data, ["x"], max_stall=pd.Timedelta("5min")).empty

    def test_a_held_value_is_one_finding_per_run(self):
        data = self.frame([1.0] * 30 + [2.0] * 30)
        findings = v.check_stalls(data, ["x"], max_stall=pd.Timedelta("5min"))

        assert len(findings) == 2
        assert findings.iloc[0]["check"] == "stall"
        assert findings.iloc[0]["value"] == 1.0

    def test_a_run_shorter_than_the_window_is_not_a_stall(self):
        data = self.frame([1.0] * 3 + list(np.arange(10.0, 60.0)))

        assert v.check_stalls(data, ["x"], max_stall=pd.Timedelta("10min")).empty

    def test_a_window_below_the_interval_is_refused(self):
        # Every single sample would otherwise register as a run that outlasted it.
        data = self.frame(np.arange(60, dtype=float))

        with pytest.raises(ValueError, match="below the"):
            v.stall_runs(data, "x", max_stall=pd.Timedelta("10s"))

    def test_a_hold_either_side_of_a_gap_is_not_joined_across_it(self):
        held = self.frame([5.0] * 20)
        later = held.copy()
        later.index = held.index + pd.Timedelta("6h")
        split = pd.concat([held, later])

        runs = v.stall_runs(split, "x", max_stall=pd.Timedelta("5min"))

        assert len(runs) == 2

    def test_runs_of_nan_are_left_to_the_gap_check(self):
        # Missing data is reported once, by check_gaps, rather than as every
        # signal holding its value at the same moment.
        data = self.frame([np.nan] * 40)

        assert v.check_stalls(data, ["x"], max_stall=pd.Timedelta("5min")).empty

    def test_a_per_signal_window_overrides_the_default(self):
        # A coarsely quantised signal wants its own window rather than a looser
        # default that would also hide a dead irradiance channel.
        data = self.frame([1.0] * 40)
        windows = {"x": pd.Timedelta("2h")}

        assert v.check_stalls(
            data, ["x"], max_stall=pd.Timedelta("5min"), stall_windows=windows
        ).empty

    def test_rounding_decides_what_counts_as_a_change(self):
        # A channel dithering in its last digit is not moving in any sense that
        # matters, but it never repeats a value either -- so `decimals` is what
        # turns that into a stall.
        data = self.frame(1.0 + np.arange(40) * 1e-6)
        window = pd.Timedelta("5min")

        assert v.check_stalls(data, ["x"], max_stall=window).empty
        assert len(v.check_stalls(data, ["x"], max_stall=window, decimals=3)) == 1

    def test_the_reported_value_is_not_the_rounded_one(self):
        # Comparison happens on the rounded series, but the value you are shown
        # comes off the raw one.
        data = self.frame([1.23456] * 40)
        runs = v.stall_runs(data, "x", max_stall=pd.Timedelta("5min"), decimals=2)

        assert runs.iloc[0]["value"] == pytest.approx(1.23456)


class TestCheckConstants:
    """Signals that must hold one non-zero value, whatever it happens to be."""

    @staticmethod
    def frame(values):
        index = pd.date_range("2024-05-01", periods=len(values), freq="1min", tz="UTC")
        return pd.DataFrame({"x": values}, index=index)

    def test_a_steady_non_zero_value_passes(self):
        assert v.check_constants(self.frame([12.0] * 5), ["x"]).empty

    def test_a_changing_value_is_flagged(self):
        findings = v.check_constants(self.frame([12.0, 12.0, 13.0]), ["x"])

        assert len(findings) == 1
        assert "changes" in findings.iloc[0]["detail"]

    def test_a_zero_value_is_flagged(self):
        findings = v.check_constants(self.frame([0.0] * 5), ["x"])

        assert "zero/blank" in findings.iloc[0]["detail"]

    def test_an_all_nan_signal_is_flagged(self):
        findings = v.check_constants(self.frame([np.nan] * 5), ["x"])

        assert findings.iloc[0]["detail"] == "all NaN"

    def test_some_nan_is_flagged_alongside(self):
        findings = v.check_constants(self.frame([12.0, np.nan, 12.0]), ["x"])

        assert "NaN" in findings.iloc[0]["detail"]

    def test_a_steady_text_value_passes(self):
        assert v.check_constants(self.frame(["Pecos"] * 4), ["x"]).empty

    def test_a_changing_text_value_is_flagged(self):
        findings = v.check_constants(self.frame(["Pecos", "Flats"]), ["x"])

        assert "changes" in findings.iloc[0]["detail"]

    @pytest.mark.parametrize("blank", ["", "0", "None"])
    def test_a_blank_text_value_is_flagged(self, blank):
        findings = v.check_constants(self.frame([blank] * 3), ["x"])

        assert "zero/blank" in findings.iloc[0]["detail"]


class TestCheckExpected:
    """Signals that must read one named value throughout."""

    @staticmethod
    def frame(values):
        index = pd.date_range("2024-05-01", periods=len(values), freq="1min", tz="UTC")
        return pd.DataFrame({"fan": values}, index=index)

    def test_the_expected_value_throughout_passes(self):
        assert v.check_expected(self.frame([1, 1, 1]), {"fan": 1}).empty

    def test_a_signal_sitting_at_the_wrong_value_is_flagged(self):
        # A ventilator wired to run all the time is broken whether it sits at 0
        # for the whole file or drops out for an hour in the middle.
        findings = v.check_expected(self.frame([0, 0, 0]), {"fan": 1})

        assert len(findings) == 1
        assert findings.iloc[0]["check"] == "expected"

    def test_a_dropout_is_counted_as_a_deviation(self):
        findings = v.check_expected(self.frame([1, np.nan, 1]), {"fan": 1})

        assert "missing" in findings.iloc[0]["detail"]

    def test_deviating_rows_can_be_listed_for_eyeballing(self):
        data = self.frame([1, 0, 1, 0])
        rows = v.expected_deviations(data, "fan", 1)

        assert len(rows) == 2
        assert list(rows.columns) == ["fan"]

    def test_eps_admits_a_tolerance(self):
        assert v.check_expected(self.frame([1.0, 1.001]), {"fan": 1}, eps=0.01).empty


class TestCheckSentinel:
    """Signals that should read the invalid-data sentinel everywhere."""

    @staticmethod
    def frame(values):
        index = pd.date_range("2024-05-01", periods=len(values), freq="1min", tz="UTC")
        return pd.DataFrame({"unused": values}, index=index)

    def test_the_sentinel_throughout_passes(self):
        assert v.check_sentinel(self.frame([-99, -99, -99]), ["unused"]).empty

    def test_a_real_reading_is_flagged(self):
        findings = v.check_sentinel(self.frame([-99, 4.2, -99]), ["unused"])

        assert len(findings) == 1
        assert findings.iloc[0]["check"] == "sentinel"

    def test_nan_is_counted_apart_from_a_wrong_value(self):
        findings = v.check_sentinel(self.frame([-99, np.nan]), ["unused"])

        assert "non-numeric" in findings.iloc[0]["detail"]

    def test_a_per_signal_sentinel_map(self):
        data = self.frame([-999, -999])

        assert v.check_sentinel(data, ["unused"], sentinel={"unused": -999}).empty

    def test_deviating_rows_can_be_listed(self):
        rows = v.sentinel_deviations(self.frame([-99, 4.2]), "unused")

        assert len(rows) == 1


class TestSpec:
    """Which signals get which checks, and how a notebook adjusts that."""

    def test_dict_fields_merge_on_override(self):
        # So a notebook can pin one more signal without restating the rest.
        spec = v.Spec(ranges={"a": (0, 1)}, expected={"fan": 1})
        wider = spec.override(ranges={"b": (0, 2)})

        assert wider.ranges == {"a": (0, 1), "b": (0, 2)}
        assert wider.expected == {"fan": 1}

    def test_sequence_fields_replace_on_override(self):
        spec = v.Spec(dynamic=("a", "b"))

        assert spec.override(dynamic=("c",)).dynamic == ("c",)

    def test_override_leaves_the_original_alone(self):
        spec = v.Spec(ranges={"a": (0, 1)})
        spec.override(ranges={"b": (0, 2)})

        assert spec.ranges == {"a": (0, 1)}

    def test_signals_enumerates_what_is_tested(self):
        spec = v.Spec(ranges={"a": (0, 1)}, lesser=(("a", "b"),), dynamic=("a",))
        tested = spec.signals()

        assert tested["range"] == ["a"]
        assert tested["lesser"] == ["a vs b"]
        assert tested["stall"] == ["a"]

    def test_turning_gaps_off_removes_the_check_entirely(self):
        # The daylight half of a day is missing every night by construction, so
        # calling that a gap would report the question back as the answer.
        assert v.Spec(gaps=False).signals()["gap"] == []
        assert v.Spec(gaps=True).signals()["gap"] == [v.ALL_SIGNALS]


class TestMissingSignals:
    def test_a_signal_the_spec_wants_but_the_file_lacks(self, daily):
        spec = v.Spec(ranges={"signal": (0, 200), "absent": (0, 1)})
        findings = v.missing_signals(daily, spec)

        assert len(findings) == 1
        assert findings.iloc[0]["signal"] == "absent"
        assert findings.iloc[0]["check"] == "missing"

    def test_every_check_contributes_its_signals(self, daily):
        spec = v.Spec(
            dynamic=("gone_dynamic",),
            constant=("gone_constant",),
            sentinel=("gone_sentinel",),
            expected={"gone_expected": 1},
            lesser=(("gone_left", "gone_right"),),
        )
        found = set(v.missing_signals(daily, spec)["signal"])

        assert found == {
            "gone_dynamic",
            "gone_constant",
            "gone_sentinel",
            "gone_expected",
            "gone_left",
            "gone_right",
        }


class TestValidateAndSummarize:
    """The whole pass, and the roll-up that shows what passed."""

    def test_validate_concatenates_every_check(self, daily):
        # max_stall has to clear the frame's ten-minute cadence; validation's own
        # default suits a ProtoNode's once-a-second polling, not this.
        spec = v.Spec(
            ranges={"signal": (0, 50)},
            dynamic=("signal",),
            max_stall=pd.Timedelta("30min"),
        )
        findings = v.validate(daily, spec)

        assert list(findings.columns) == v.FINDING_COLUMNS
        assert "range" in set(findings["check"])

    def test_the_default_stall_window_is_too_tight_for_a_ten_minute_frame(self, daily):
        # Worth pinning as the sharp edge it is: a spec that names a dynamic
        # signal without raising max_stall fails loudly on LOGR-cadence data
        # rather than reporting every sample as a stall.
        spec = v.Spec(dynamic=("signal",))

        with pytest.raises(ValueError, match="below the"):
            v.validate(daily, spec)

    def test_a_healthy_frame_produces_no_findings(self, daily):
        spec = v.Spec(ranges={"signal": (-1, 1000)})

        assert v.validate(daily, spec).empty

    def test_summarize_enumerates_the_spec_not_the_findings(self, daily):
        # Findings only ever describe failures, so the passing checks have to come
        # from the spec or they would not appear at all.
        spec = v.Spec(ranges={"signal": (-1, 1000), "other": (0, 1)})
        summary = v.summarize(v.validate(daily, spec), spec)

        ranges = summary[summary["check"] == "range"]
        assert set(ranges["signal"]) == {"signal", "other"}
        assert bool(ranges[ranges["signal"] == "signal"]["pass"].iloc[0]) is True

    def test_summarize_counts_findings_per_signal(self, daily):
        spec = v.Spec(ranges={"signal": (0, 50)})
        summary = v.summarize(v.validate(daily, spec), spec)
        row = summary[(summary["check"] == "range") & (summary["signal"] == "signal")]

        assert row["n_findings"].iloc[0] == 1
        assert bool(row["pass"].iloc[0]) is False

    def test_failures_sort_to_the_top(self, daily):
        spec = v.Spec(ranges={"signal": (0, 50), "other": (0, 1)})
        summary = v.summarize(v.validate(daily, spec), spec)

        assert bool(summary["pass"].iloc[0]) is False
