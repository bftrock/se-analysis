"""Header parsing and channel renaming -- where a silent mistake is worst.

rename_channels() is the riskiest function in the module: it decides which token
of a column name is the statistic, and getting that wrong does not raise, it
mislabels a column. A finding then names the wrong sensor, which is worse than
no finding at all. The cases pinned here are the ones the docstring calls out:
the two export layouts that put the statistic in different places, the
Total-versus-Sum collision, and a channel left with a single column.

malformed_rows() is the other one, because the corruption it looks for is
invisible downstream -- a short row shifts later channels into their neighbours'
columns and every value that lands is numeric and plausibly scaled.
"""

import gzip

import pandas as pd
import pytest

from conftest import site_info
from se_analysis import logr


class TestParseSiteInfo:
    """The flat top-level fields, and the trap in reading them."""

    def test_reads_the_top_level_fields(self, header):
        fields = logr.parse_site_info(header)

        assert fields["Site Number"] == "002682"
        assert fields["Latitude"] == "44.5"

    def test_trailing_colons_are_stripped(self, header):
        # Labels arrive as "Latitude:"; nobody wants to type the colon.
        assert "Latitude" in logr.parse_site_info(header)
        assert "Latitude:" not in logr.parse_site_info(header)

    def test_a_repeated_label_collapses_to_the_last_one(self, header):
        # The documented trap: "Serial Number" appears once for the logger and
        # again in every sensor block, so this yields the *final* sensor's serial
        # rather than the logger's. channel_blocks() is what reads it properly.
        assert logr.parse_site_info(header)["Serial Number"] == "SN-SR30-9"


class TestChannelBlocks:
    """Cutting the repeating Sensor History region into per-channel records."""

    def test_one_block_per_channel_row(self, header):
        blocks = logr.channel_blocks(header)

        assert [channel for channel, _ in blocks] == ["1", "108"]

    def test_fields_before_the_first_channel_are_not_in_any_block(self, header):
        blocks = dict(logr.channel_blocks(header))

        # the logger's own serial precedes the first "Channel:" row
        assert blocks["1"]["Serial Number"] == "SN-PT-1"
        assert "Site Number" not in blocks["1"]

    def test_values_arrive_stripped(self):
        info = site_info([("Channel:", " 3 "), ("Description:", "  Vane  ")])
        blocks = logr.channel_blocks(info)

        assert blocks == [("3", {"Description": "Vane"})]

    def test_a_reconfigured_channel_appears_twice(self):
        # The header is a transcript, not a mapping. A channel reconfigured part
        # way through the record shows up as two blocks, and the caller gets to
        # see that rather than have them silently merged.
        info = site_info(
            [
                ("Channel:", "1"),
                ("Description:", "NRG 40C"),
                ("Channel:", "1"),
                ("Description:", "NRG S1"),
            ]
        )
        blocks = logr.channel_blocks(info)

        assert len(blocks) == 2
        assert [block["Description"] for _, block in blocks] == ["NRG 40C", "NRG S1"]

    def test_within_one_block_the_first_value_wins(self):
        info = site_info(
            [
                ("Channel:", "1"),
                ("Units:", "deg_C"),
                ("Units:", "K"),
            ]
        )

        assert logr.channel_blocks(info)[0][1]["Units"] == "deg_C"


class TestChannelNames:
    """Building the label a finding will name a sensor by."""

    def test_description_is_preferred_over_sensor_type(self, header):
        # A Modbus channel lists both: the type is the model and the Description
        # the configured name, so preferring the Description keeps several
        # identical sensors on one site apart.
        assert logr.channel_names(header)["108"] == (
            "Ch108_Hukseflux_SR30-POA_Irradiance_W/m^2"
        )

    def test_a_channel_with_only_a_description(self, header):
        assert logr.channel_names(header)["1"] == "Ch1_Pt1000_deg_C"

    def test_sensor_type_is_the_logr_fallback(self):
        info = site_info(
            [("Channel:", "5"), ("Sensor Type:", "NRG 40C"), ("Units:", "m/s")]
        )

        assert logr.channel_names(info)["5"] == "Ch5_NRG_40C_m/s"

    def test_bare_type_is_the_cloud_export_fallback(self):
        # NRG Cloud spells the same field "Type"; the two never co-occur.
        info = site_info([("Channel:", "5"), ("Type:", "Anem"), ("Units:", "m/s")])

        assert logr.channel_names(info)["5"] == "Ch5_Anem_m/s"

    def test_whitespace_inside_a_field_becomes_underscores(self):
        info = site_info(
            [
                ("Channel:", "2"),
                ("Description:", "Pt1000  Ambient"),
                ("Units:", "deg_C"),
            ]
        )

        assert logr.channel_names(info)["2"] == "Ch2_Pt1000_Ambient_deg_C"

    def test_absent_fields_are_left_out_rather_than_spelled_nan(self):
        # A field the logger omitted can still be present as a missing value,
        # which older pandas renders as the string "nan" once cast to str.
        info = site_info(
            [("Channel:", "7"), ("Description:", "Tilt"), ("Units:", None)]
        )

        assert logr.channel_names(info)["7"] == "Ch7_Tilt"


class TestRenameChannels:
    """Which token is the statistic -- the decision that must not go wrong."""

    def test_statistical_columns_keep_the_statistic_after_the_channel(
        self, header, statistical
    ):
        names = logr.channel_names(header)
        renamed = logr.rename_channels(statistical, names)

        assert "Ch1_Avg_Pt1000_deg_C" in renamed.columns
        assert "Ch1_SD_Pt1000_deg_C" in renamed.columns

    def test_a_channel_left_with_one_column_takes_the_plain_label(
        self, header, statistical
    ):
        # Channel 108 has only an Avg here, so there is nothing to hold apart and
        # the statistic is not inserted.
        renamed = logr.rename_channels(statistical, logr.channel_names(header))

        assert "Ch108_Hukseflux_SR30-POA_Irradiance_W/m^2" in renamed.columns

    def test_onesecond_columns_take_the_plain_label(self, header):
        one_second = pd.DataFrame(
            columns=["Timestamp", "Ch1_Samples_deg_C", "Ch108_Samples_W/m^2"]
        )
        renamed = logr.rename_channels(one_second, logr.channel_names(header))

        assert list(renamed.columns) == [
            "Timestamp",
            "Ch1_Pt1000_deg_C",
            "Ch108_Hukseflux_SR30-POA_Irradiance_W/m^2",
        ]

    def test_the_notused_placeholder_is_dropped_by_default(self, header, statistical):
        renamed = logr.rename_channels(statistical, logr.channel_names(header))

        assert not [c for c in renamed.columns if "NotUsed" in c]

    def test_keeping_the_placeholder_is_how_you_inspect_it(self, header, statistical):
        # Dropping it hides a short row's corruption rather than causing it, so
        # there has to be a way to look.
        renamed = logr.rename_channels(
            statistical, logr.channel_names(header), drop_unused=False
        )

        assert "Ch1_NotUsed_Pt1000_deg_C" in renamed.columns

    def test_a_cloud_export_finds_the_statistic_by_vocabulary_not_position(self):
        # Ch4_Total_1.19m_E_Sum_mm is a Sum: "Total" is the sensor type
        # (Totalizer) and the statistic is the last vocabulary match, not the
        # first. Reading it positionally would label this column a Total.
        cloud = pd.DataFrame(
            columns=["Ch4_Total_1.19m_E_Sum_mm", "Ch4_Total_1.19m_E_Avg_mm"]
        )
        renamed = logr.rename_channels(cloud, {"4": "Ch4_Totalizer_Rain_mm"})

        assert list(renamed.columns) == [
            "Ch4_Sum_Totalizer_Rain_mm",
            "Ch4_Avg_Totalizer_Rain_mm",
        ]

    def test_columns_with_no_channel_prefix_are_untouched(self, header, statistical):
        renamed = logr.rename_channels(statistical, logr.channel_names(header))

        assert renamed.index.equals(statistical.index)

    def test_channels_absent_from_names_are_left_alone(self):
        data = pd.DataFrame(columns=["Ch1_Avg_deg_C", "Ch99_Avg_m/s"])
        renamed = logr.rename_channels(data, {"1": "Ch1_Pt1000_deg_C"})

        assert "Ch99_Avg_m/s" in renamed.columns

    def test_the_original_frame_is_not_modified(self, header, statistical):
        before = list(statistical.columns)
        logr.rename_channels(statistical, logr.channel_names(header))

        assert list(statistical.columns) == before

    def test_a_label_that_would_apply_twice_raises(self):
        # Reachable when a channel's columns carry a statistic outside the
        # vocabulary, leaving them nothing to tell each other apart. A silent
        # duplicate label is far worse than a loud failure.
        data = pd.DataFrame(columns=["Ch1_Bogus_deg_C", "Ch1_Other_deg_C"])

        with pytest.raises(ValueError, match="multiple columns"):
            logr.rename_channels(data, {"1": "Ch1_Pt1000_deg_C"})


class TestSignalGroups:
    """Grouping `<signal>_<stat>` columns, for picking explicit column lists."""

    def test_groups_statistics_under_their_signal(self):
        data = pd.DataFrame(
            columns=["VIN_RP_V_Avg", "VIN_RP_V_Min", "VIN_RP_V_Max", "VIN_RP_V_SD"]
        )

        assert logr.signal_groups(data) == {
            "VIN_RP_V": [
                "VIN_RP_V_Avg",
                "VIN_RP_V_Min",
                "VIN_RP_V_Max",
                "VIN_RP_V_SD",
            ]
        }

    def test_a_column_with_no_statistic_is_its_own_group(self):
        data = pd.DataFrame(columns=["Stats_Timestamp", "VIN_RP_V_Avg"])
        groups = logr.signal_groups(data)

        assert groups["Stats_Timestamp"] == ["Stats_Timestamp"]


class TestMalformedRows:
    """Short data rows, whose damage is invisible once pandas has padded them."""

    @staticmethod
    def write(path, columns, rows, marker=logr.DATA_MARKER, compress=False):
        """A minimal LOGR file: a header, the marker, column names, then rows."""
        lines = ["Site Number:\t002682", marker, "\t".join(columns)]
        lines += ["\t".join(str(v) for v in row) for row in rows]
        text = "\n".join(lines) + "\n"
        if compress:
            with gzip.open(path, "wt", encoding="utf-8") as handle:
                handle.write(text)
        else:
            path.write_text(text, encoding="utf-8")
        return path

    def test_a_well_formed_file_reports_nothing(self, tmp_path):
        path = self.write(
            tmp_path / "good.dat",
            ["Timestamp", "Ch1_Avg", "Ch2_Avg"],
            [("2024-05-01 00:00:00", 1.0, 2.0), ("2024-05-01 00:01:00", 1.1, 2.1)],
        )
        found = logr.malformed_rows([path])

        assert found.empty
        # the schema is there to concatenate against even when nothing is wrong
        assert list(found.columns) == ["file", "timestamp", "fields", "expected"]

    def test_a_short_row_is_reported_with_both_field_counts(self, tmp_path):
        path = self.write(
            tmp_path / "short.dat",
            ["Timestamp", "Ch1_Avg", "Ch2_Avg"],
            [
                ("2024-05-01 00:00:00", 1.0, 2.0),
                ("2024-05-01 00:01:00", 1.1),  # a channel wrote a short record
            ],
        )
        found = logr.malformed_rows([path])

        assert len(found) == 1
        assert found.iloc[0]["fields"] == 2
        assert found.iloc[0]["expected"] == 3
        assert found.iloc[0]["timestamp"] == "2024-05-01 00:01:00"
        assert found.iloc[0]["file"] == "short.dat"

    def test_a_long_row_is_reported_too(self, tmp_path):
        path = self.write(
            tmp_path / "long.dat",
            ["Timestamp", "Ch1_Avg"],
            [("2024-05-01 00:00:00", 1.0, 99.0)],
        )

        assert logr.malformed_rows([path]).iloc[0]["fields"] == 3

    def test_a_file_with_no_data_marker_is_skipped(self, tmp_path):
        path = self.write(
            tmp_path / "notlogr.dat",
            ["a", "b"],
            [("1",)],
            marker="Something Else",
        )

        assert logr.malformed_rows([path]).empty

    def test_blank_lines_are_not_rows(self, tmp_path):
        path = tmp_path / "trailing.dat"
        path.write_text(
            "Site Number:\t002682\nData\nTimestamp\tCh1_Avg\n"
            "2024-05-01 00:00:00\t1.0\n\n\n",
            encoding="utf-8",
        )

        assert logr.malformed_rows([path]).empty

    def test_gzipped_files_are_read_transparently(self, tmp_path):
        path = self.write(
            tmp_path / "short.dat.gz",
            ["Timestamp", "Ch1_Avg", "Ch2_Avg"],
            [("2024-05-01 00:01:00", 1.1)],
            compress=True,
        )

        assert len(logr.malformed_rows([path])) == 1

    def test_rows_from_several_files_are_reported_together(self, tmp_path):
        first = self.write(
            tmp_path / "a.dat",
            ["Timestamp", "Ch1_Avg"],
            [("2024-05-01 00:00:00",)],
        )
        second = self.write(
            tmp_path / "b.dat",
            ["Timestamp", "Ch1_Avg"],
            [("2024-05-02 00:00:00",)],
        )
        found = logr.malformed_rows([first, second])

        assert sorted(found["file"]) == ["a.dat", "b.dat"]


class TestReadFiles:
    """The wrapper exists to turn nrgpy's unhelpful failures into real messages."""

    def test_a_missing_directory_says_so(self, tmp_path):
        with pytest.raises(NotADirectoryError):
            logr.read_files(tmp_path / "nope")

    def test_an_unknown_malformed_action_is_rejected_up_front(self, tmp_path):
        # Checked before any reading, so a typo costs nothing.
        with pytest.raises(ValueError, match="malformed must be one of"):
            logr.read_files(tmp_path, malformed="explode")

    def test_an_empty_directory_names_what_it_found_instead(self, tmp_path):
        # nrgpy surfaces this as AttributeError on a missing attribute, which says
        # nothing about the directory you pointed it at.
        with pytest.raises(FileNotFoundError, match="the directory is empty"):
            logr.read_files(tmp_path)

    def test_a_directory_of_the_wrong_files_lists_the_suffixes(self, tmp_path):
        (tmp_path / "notes.txt").write_text("nothing to read", encoding="utf-8")
        (tmp_path / "export.csv").write_text("a,b", encoding="utf-8")

        with pytest.raises(FileNotFoundError, match=r"\.txt"):
            logr.read_files(tmp_path)
