"""Reading LOGR files, and the header metadata nrgpy exposes as `LogrRead.Site_info`.

`Site_info` is a flat two-column key/value listing: a block of top-level logger
fields, followed by one repeating block per channel under Sensor History. Which
helper you want depends on which of those two regions you're reading.
"""

import gzip
import re
import warnings
from collections import Counter
from pathlib import Path

import nrgpy
import pandas as pd

# what nrgpy will consider at all, whatever the file_type filter says
LOGR_SUFFIXES = (".dat", ".log", ".diag", ".dat.gz", ".log.gz", ".diag.gz")

# the line that ends the header and precedes the column names
DATA_MARKER = "Data"

MALFORMED_ACTIONS = ("drop", "raise", "keep")

# the statistic a channel's column holds, whichever export produced it
STATISTICS = ("Avg", "SD", "Min", "Max", "Gust", "GustDir", "Sum", "Total", "NotUsed")


def parse_site_info(site_info: pd.DataFrame) -> dict[str, str]:
    """Top-level LOGR header fields as {label: value}.

    Only meaningful for labels that appear once. Labels repeating per sensor
    block (Channel, Serial Number, Description, Units, Height, Scale Factor,
    ...) collapse to their *last* occurrence, so e.g. "Serial Number" yields the
    final sensor's serial rather than the logger's. Use channel_names() to read
    the per-channel blocks.
    """
    si = site_info.dropna(subset=[0])
    return dict(zip(si[0].str.strip().str.rstrip(":"), si[1]))


def _field(block: dict[str, str], key: str) -> str:
    """One Sensor History value, with absent and unfilled fields both empty.

    A field the logger left out of a block can still be present as an empty
    value, and pandas before 3.0 renders that missing value as the string "nan"
    when the column is cast to str. Treat all three as "not given".
    """
    value = block.get(key, "")
    return "" if value == "nan" else value


def channel_names(site_info: pd.DataFrame) -> dict[str, str]:
    """Parse reader.Site_info into {channel: 'Ch<n>_<sensor>_[<Measurand>_]<Units>'}.

    The sensor part is the block's Description, falling back to the sensor type
    for blocks that carry no Description. Modbus channels list both -- the type
    being the model ("Hukseflux SR30") and Description the configured name
    ("Hukseflux SR30-POA") -- so preferring Description keeps the several
    identical sensors on a site apart.

    LOGR files spell that fallback key "Sensor Type" and NRG Cloud exports spell
    it "Type", so both are tried. They do not overlap: a LOGR header has no bare
    "Type" key, and a cloud export has no "Sensor Type".
    """
    info = site_info.set_axis(["key", "value"], axis=1)
    keys = info["key"].astype(str).str.strip().str.rstrip(":")
    # fillna after the cast: pandas 3.0 keeps missing values missing under
    # astype(str), where earlier versions rendered them as the string "nan"
    values = info["value"].astype(str).str.strip().fillna("")

    # Sensor History is a flat key/value list; each "Channel:" row starts a new block
    blocks, current = [], None
    for key, value in zip(keys, values):
        if key == "Channel":
            current = {}
            blocks.append((value, current))
        elif current is not None:
            current.setdefault(key, value)

    names = {}
    for channel, block in blocks:
        sensor = (
            _field(block, "Description")
            or _field(block, "Sensor Type")  # LOGR
            or _field(block, "Type")  # NRG Cloud export
        )
        fields = [sensor, _field(block, "Measurand"), _field(block, "Units")]
        parts = [f"Ch{channel}"] + [re.sub(r"\s+", "_", f) for f in fields if f]
        names[channel] = "_".join(parts)
    return names


def rename_channels(
    data: pd.DataFrame, names: dict[str, str], drop_unused: bool = True
) -> pd.DataFrame:
    """Rename a LOGR data frame's `Ch<n>_*` columns to channel_names() labels.

    Columns with no `Ch<n>_` prefix (Timestamp) and channels absent from `names`
    are left as they are. Returns a new frame; `data` is not modified.

    Measurement files carry one column per channel, so those take the channel
    label unchanged: `Ch1_Samples_deg_C` -> `Ch1_Pt1000_deg_C`. Statistical files
    carry several (Avg/Min/Max/SD plus Gust, Sum or Total), which would all
    reduce to that one label, so there the statistic is kept right after the
    channel to hold them apart: `Ch1_Avg_deg_C` -> `Ch1_Avg_Pt1000_deg_C`.

    NRG Cloud exports read by nrgpy.sympro_txt_read work too: they order the
    column as `Ch<n>_<Type>_<Height>m_<Bearing>_<Stat>_<Units>` rather than
    putting the statistic first, and the statistic is located by name either way.

    Statistical files also carry a `Ch<n>_NotUsed` placeholder per channel, which
    `drop_unused` removes rather than renaming. Note that these are only reliably
    empty on well-formed rows: a short data row shifts real values into the
    NotUsed slot, so dropping the column hides that corruption rather than
    causing it. Pass drop_unused=False to keep the placeholders and inspect them.
    """
    # channel number and statistic token per renameable column
    parsed = {}
    for col in data.columns:
        m = re.match(r"Ch(\d+)_", col)
        if not m or m.group(1) not in names:
            continue
        # the statistic sits at a different offset per export: second token in
        # LOGR files (Ch1_Avg_deg_C), second from last in cloud exports
        # (Ch1_Anem_2.09m_NE_Avg_m/s). Find it by vocabulary instead, taking the
        # last match, because a cloud export names the sensor type first and
        # Totalizer collides: Ch4_Total_1.19m_E_Sum_mm is a Sum, not a Total.
        statistics = [t for t in col.split("_")[1:] if t in STATISTICS]
        parsed[col] = (m.group(1), statistics[-1] if statistics else "")

    unused = [col for col, (_, stat) in parsed.items() if stat == "NotUsed"]
    if drop_unused:
        for col in unused:
            del parsed[col]
    else:
        unused = []

    # counted after the drop, so a channel left with one column keeps the plain label
    columns_per_channel = Counter(channel for channel, _ in parsed.values())

    rename = {}
    for col, (channel, statistic) in parsed.items():
        label = names[channel]
        if columns_per_channel[channel] > 1 and statistic:
            prefix = f"Ch{channel}"
            label = (
                f"{prefix}_{statistic}{label[len(prefix):]}"
                if label.startswith(prefix)
                else f"{label}_{statistic}"
            )
        rename[col] = label

    # reachable when a channel's columns carry a statistic outside STATISTICS,
    # which leaves them nothing to tell each other apart; a silent duplicate
    # label is far worse than a loud failure
    collisions = {name for name, n in Counter(rename.values()).items() if n > 1}
    if collisions:
        name = sorted(collisions)[0]
        sources = sorted(col for col, new in rename.items() if new == name)
        raise ValueError(
            f"{len(collisions)} label(s) would apply to multiple columns, e.g. "
            f"{sources} all become {name!r}; this frame was left unrenamed."
        )

    return data.drop(columns=unused).rename(columns=rename)


def signal_groups(
    data: pd.DataFrame, stats: tuple = ("Avg", "Min", "Max", "SD")
) -> dict[str, list[str]]:
    """Group `<signal>_<stat>` columns by signal, for statistical and diag frames.

    Statistical measurement files and diagnostic files both spread one signal
    across several columns, one per statistic. Validation checks take explicit
    column lists, so this is how you pick out "the Avg column of every rail" or
    "every column belonging to VIN_RP_V".
    """
    groups: dict[str, list[str]] = {}
    suffixes = tuple(f"_{stat}" for stat in stats)
    for col in data.columns:
        key = next((col[: -len(s)] for s in suffixes if col.endswith(s)), col)
        groups.setdefault(key, []).append(col)
    return groups


def _directory_contents(directory: Path) -> str:
    """A count per suffix, to say what the directory holds instead."""
    counts = Counter(
        "".join(p.suffixes).lower() for p in directory.iterdir() if p.is_file()
    )
    if not counts:
        return "the directory is empty"
    return ", ".join(
        f"{n}x {suffix or 'no suffix'}"
        for suffix, n in sorted(counts.items(), key=lambda kv: -kv[1])
    )


def _open_text(path: Path):
    """Open a LOGR file as text, transparently for the .gz variants."""
    if path.name.lower().endswith(".gz"):
        return gzip.open(path, "rt", encoding="utf-8", errors="replace")
    return open(path, encoding="utf-8", errors="replace")


def malformed_rows(paths) -> pd.DataFrame:
    """Data rows whose field count disagrees with their own file's header.

    A LOGR data row carries one field per header column. A row carrying fewer
    means a channel wrote a short record, and the fields after it land under the
    wrong column: pandas fills from the left and pads the shortfall onto the
    *end* of the row, so a channel that came up short early silently shifts
    every later channel's values into a neighbouring channel's columns. The
    values stay numeric and plausibly scaled, which is what makes this worth
    rejecting rather than eyeballing -- a shifted barometer reads as a pressure,
    just not this row's pressure.

    Returns one row per malformed data row, with the file it came from, its
    timestamp as written, and the field counts found and expected. Rows are
    identified by position, so a row too short to hold its own timestamp still
    reports. Files with no `Data` marker are skipped as not LOGR data files.
    """
    found = []
    for path in paths:
        path = Path(path)
        with _open_text(path) as handle:
            lines = handle.read().split("\n")

        marker = next(
            (i for i, line in enumerate(lines) if line.strip() == DATA_MARKER), None
        )
        if marker is None or marker + 1 >= len(lines):
            continue
        expected = len(lines[marker + 1].split("\t"))

        for line in lines[marker + 2 :]:
            if not line.strip():
                continue
            fields = line.split("\t")
            if len(fields) != expected:
                found.append(
                    {
                        "file": path.name,
                        "timestamp": fields[0] if fields else "",
                        "fields": len(fields),
                        "expected": expected,
                    }
                )

    return pd.DataFrame(found, columns=["file", "timestamp", "fields", "expected"])


def _malformed_message(malformed: pd.DataFrame, rows: int) -> str:
    """What was rejected, and enough of where to go and look at a file."""
    counts = Counter(zip(malformed["fields"], malformed["expected"]))
    shapes = ", ".join(
        f"{n} row(s) with {got} fields, not {want}"
        for (got, want), n in sorted(counts.items())
    )
    files = sorted(set(malformed["file"]))
    named = ", ".join(files[:3]) + (
        f" and {len(files) - 3} more" if len(files) > 3 else ""
    )
    return (
        f"{len(malformed)} of {rows} rows are malformed ({shapes}) across "
        f"{len(files)} file(s): {named}. A short row shifts every later "
        "channel's values into the wrong columns."
    )


def _reject_malformed(reader: nrgpy.LogrRead, directory: Path, action: str) -> None:
    """Drop, report or keep the rows whose field count does not match the header.

    Only the files nrgpy actually concatenated are scanned, so whatever narrowed
    the selection -- file_type, file_filter, a start or end date -- narrows this
    the same way rather than flagging rows that never reached the frame.
    """
    if action == "keep":
        return

    malformed = malformed_rows(directory / name for name in reader.dat_file_names)
    if action == "drop":
        # set unconditionally, so `reader.malformed` is somewhere to look rather
        # than an attribute whose absence has to be interpreted
        reader.malformed = malformed
    if malformed.empty:
        return

    message = _malformed_message(malformed, len(reader.data))
    if action == "raise":
        raise ValueError(message)

    # match on the parsed stamp: nrgpy has already converted the column unless
    # text_timestamps was asked for, so neither side can be compared as written
    stamps = pd.to_datetime(malformed["timestamp"], format="mixed", utc=True)
    column = pd.to_datetime(reader.data[reader.timestamp_col], utc=True)
    reader.data = reader.data[~column.isin(set(stamps))].reset_index(drop=True)
    warnings.warn(f"{message} Dropped; see reader.malformed.", stacklevel=3)


def read_files(
    directory, file_type: str = "log", malformed: str = "drop", **kwargs
) -> nrgpy.LogrRead:
    """nrgpy's concat_txt, but saying so when the directory holds nothing to read.

    nrgpy assigns the object it concatenates into only while looping over matched
    files, then dereferences it unconditionally, so nothing to read surfaces as
    `AttributeError: 'LogrRead' object has no attribute 'base'` -- which says
    nothing about the directory you pointed it at. Translate that into the
    actual problem, and distinguish it from a first file that failed to parse.

    Rows whose field count disagrees with their file's header are rejected here,
    because nothing downstream can recognise them: the shifted values are all
    numeric and mostly in range, so they reach a plot or a range check looking
    like readings. `malformed` chooses what happens to them -- "drop" (the
    default) removes them and warns, leaving them on the reader as `.malformed`;
    "raise" refuses the whole read; "keep" restores the old behaviour of
    concatenating them silently. See malformed_rows() for the detail.

    Returns the reader, so `.data` and `.Site_info` stay available as usual.
    """
    directory = Path(directory)
    if not directory.is_dir():
        raise NotADirectoryError(f"{directory} is not a directory")
    if malformed not in MALFORMED_ACTIONS:
        raise ValueError(
            f"malformed must be one of {MALFORMED_ACTIONS}, not {malformed!r}"
        )

    kwargs.setdefault("drop_duplicates", False)
    reader = nrgpy.LogrRead()
    try:
        reader.concat_txt(dat_dir=str(directory), file_type=file_type, **kwargs)
    except AttributeError as exc:
        if "base" not in str(exc):
            raise
        # file_count is set before the loop, so it separates the two causes
        if getattr(reader, "file_count", 0):
            raise RuntimeError(
                f"nrgpy matched {reader.file_count} {file_type!r} file(s) in "
                f"{directory} but could not read the first one -- see the logged "
                "exception. A non-standard LOGR header does this."
            ) from exc
        raise FileNotFoundError(_no_match(directory, file_type)) from exc

    # a fixed nrgpy returns an empty frame here rather than raising
    if not getattr(reader, "file_count", 0):
        raise FileNotFoundError(_no_match(directory, file_type))

    _reject_malformed(reader, directory, malformed)

    return reader


def _no_match(directory: Path, file_type: str) -> str:
    return (
        f"No {file_type!r} files in {directory}. nrgpy wants a name containing "
        f"{file_type!r} that also ends in one of {', '.join(LOGR_SUFFIXES)}; "
        f"instead {_directory_contents(directory)}."
    )
