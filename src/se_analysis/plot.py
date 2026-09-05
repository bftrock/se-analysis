"""Plotting every signal in a frame for visual inspection.

Measurement files come in two shapes. ProtoNode exports and LOGR onesecond files
carry one column per signal, so one axes per column is all it takes. LOGR
statistical and diagnostic files carry several columns per signal -- Avg, Min,
Max, SD and sometimes Sum, Gust or Total -- which belong on shared axes so the
spread of one signal reads as one picture.

`plot_signals` sweeps a whole frame that way, one figure per signal. When the
question is how a handful of signals move against each other -- two temperatures
that should track, a rail against the load it feeds -- `plot_together` draws a
chosen subset on one axes instead, colouring by signal rather than by statistic.

Which statistics are present is not fixed, so it is discovered rather than
assumed: Cold Creek diagnostic files carry the full Avg/Min/Max/SD quartet while
Sundance ones carry only Min/Max, and statistical channels vary between Sum,
Gust, Total and nothing at all in their fifth slot.
"""

import re

import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt

# ── chart surface and ink ────────────────────────────────────────────────────
SURFACE = "#fcfcfb"
# 1.55:1 against the surface -- recessive, but actually visible. The first pass
# used #e1e0d9, which measures 1.29:1 and reads as no grid at all once the
# figure is scaled down; a grid that cannot be seen is a grid that is not there.
# Kept deliberately lighter than AXIS (1.75:1) so the spine stays the stronger
# line and the anatomy still has a hierarchy.
GRID = "#cfcdc1"      # hairline, solid; dashing a grid reads as data
AXIS = "#c3c2b7"
MUTED = "#898781"     # axis and tick text
INK = "#0b0b0b"       # titles

# Fixed colour and dash per statistic, so a statistic keeps its identity from one
# file to the next: Min stays violet whether the logger reports Min/Max alone
# (Sundance diagnostics) or the full quartet (Cold Creek). Colouring by column
# order instead would repaint every series whenever the set changed, which is
# exactly what makes a stack of QC charts unreadable.
#
# The hues are five documented categorical slots, picked by enumerating all 56
# five-subsets and keeping one that clears the ALL-PAIRS gates -- overlapping
# lines get compared in any combination, not just neighbouring ones, and the
# default slot order fails that test (its yellow-vs-orange pair measures 13.7,
# under the 15 floor). Validated on this surface: CVD dE 13.0 against a target
# of 8, normal-vision dE 16.3 against a floor of 15.
#
# Contrast against the surface is in genuine tension with that separation -- the
# five slots that all clear 3:1 collapse under CVD (orange vs green dE 3.2) -- so
# the contrast budget goes to the most-read series and the two thin ones take the
# documented relief of visible labels, which is why the legend is not optional.
# The dashes are secondary encoding: identity never rests on hue alone.
STATISTIC_STYLE = {
    "Avg":     ("#2a78d6", "-",  2.0),  # blue    4.30:1  read first, so drawn heaviest
    "Average": ("#2a78d6", "-",  2.0),
    "Min":     ("#4a3aa7", "--", 1.6),  # violet  8.33:1  dashed pair with Max --
    "Max":     ("#008300", "--", 1.6),  # green   4.82:1  together they read as the envelope
    "SD":      ("#e87ba4", ":",  1.6),  # magenta 2.62:1  relief: legend
    "StdDev":  ("#e87ba4", ":",  1.6),
    # The fifth slot a channel may carry. No channel ever carries two of these at
    # once, so they can share one hue without ever colliding on an axes.
    "Sum":     ("#eda100", "-.", 1.6),  # yellow  2.11:1  relief: legend
    "Total":   ("#eda100", "-.", 1.6),
    "Gust":    ("#eda100", "-.", 1.6),
    "NotUsed": ("#eda100", "-.", 1.6),
}

# A signal with no statistics is alone on its axes, so it takes slot 1 and needs
# no legend -- the title names it.
PLAIN_STYLE = ("#2a78d6", "-", 2.0)

# An unrecognised statistic is drawn deliberately grey rather than given a
# generated hue, which would break the validated set. Add it to
# STATISTIC_STYLE to give it an identity.
FALLBACK_STYLE = (MUTED, (0, (4, 2)), 1.6)

# Overlaying signals moves identity from the statistic to the signal, so the same
# five validated slots are reused as a plain categorical ramp -- they were chosen
# on the all-pairs gate, which is exactly the question an overlay asks. Order is
# by contrast against the surface, so a two-signal comparison gets the two
# strongest hues and the thin ones are only reached when the plot is busy anyway.
SIGNAL_PALETTE = ("#2a78d6", "#4a3aa7", "#008300", "#e87ba4", "#eda100")

# Past five signals the hues run out and the dash becomes identity rather than
# reinforcement, which is why `plot_together` keeps its dashes on dense data
# where `plot_signals` drops them.
SIGNAL_DASHES = ("-", "--", ":", "-.")
SIGNAL_WIDTH = 1.8

# Below this many samples a line is too sparse to read as a line, so it gets
# markers -- a statistical file covering a quarter of an hour holds 18 rows.
MARKER_THRESHOLD = 60

# Above this many samples the dashes stop helping and start hurting: on a signal
# that oscillates every sample, a dash pattern shreds the line into noise. The
# palette clears the CVD target on its own (dE 13.0 against a target of 8), so
# the dashes are reinforcement rather than the identity channel, and dropping
# them on dense data costs nothing. A day of one-minute diagnostics is 1440 rows.
DENSE_THRESHOLD = 400

# Tokens that name a statistic rather than part of a signal's name. Needed
# because position alone is ambiguous: the token after the channel number is the
# statistic in `Ch1_Avg_Pt1000_deg_C` but the sensor in `Ch1_Pt1000_deg_C`.
# "Samples" is deliberately absent -- a onesecond channel has one column, and
# reading it as a plain signal keeps the units in the title.
STATISTICS = (
    "Avg", "Average", "Min", "Max", "SD", "StdDev", "Sum", "Total", "Gust",
    "NotUsed",
)

_CHANNEL_PREFIXED = re.compile(r"^(Ch\d+)_([^_]+)(.*)$")

# SD is a spread, not a value, and it is usually near zero beside a signal that
# is not: sharing one y-range leaves the value's own variation under 5% of the
# axes on 65% of the signals measured here (median 1.6%), which flattens exactly
# the drift a QC pass is looking for. It gets its own panel below, sharing the x
# axis -- two panels, never two y-scales on one plot. Sum, Total and Gust stay on
# the value panel; they measure 1.0-1.5x the Avg, so they cost nothing.
SPREAD_STATISTICS = ("SD", "StdDev")

# Drawn last, so it sits on top where the statistics coincide -- on a stable
# voltage rail Avg, Min and Max fall on the same line and only the topmost shows.
PRIMARY_STATISTICS = ("Avg", "Average")

# Reading order for the legend, kept independent of draw order: the primary is
# drawn last for z-order but belongs first in the key.
LEGEND_ORDER = ("Avg", "Average", "Min", "Max", "Gust", "Sum", "Total",
                "NotUsed", "SD", "StdDev")


def chart_style() -> dict:
    """Seaborn's whitegrid retuned to the chart tokens, as rcParams.

    Applied per figure through plt.rc_context rather than sns.set_theme, so
    plotting a signal never reaches out and restyles the rest of a notebook.
    """
    style = sns.axes_style("whitegrid")
    style.update({
        "figure.facecolor": SURFACE,
        "axes.facecolor": SURFACE,
        "axes.edgecolor": AXIS,
        "axes.linewidth": 0.8,
        "axes.labelcolor": MUTED,
        "axes.titlecolor": INK,
        "axes.titlesize": 10.5,
        "axes.titlelocation": "left",
        "axes.titlepad": 8,
        "grid.color": GRID,
        "grid.linestyle": "-",   # solid hairline; a dashed grid competes with the data
        "grid.linewidth": 0.8,
        "xtick.color": MUTED,
        "ytick.color": MUTED,
        "xtick.labelsize": 8.5,
        "ytick.labelsize": 8.5,
        "legend.frameon": False,
        "legend.fontsize": 8.5,
        "legend.labelcolor": MUTED,   # text wears ink, the line-key carries identity
        "lines.solid_capstyle": "round",
        "lines.dash_capstyle": "round",
        "font.family": "sans-serif",
        "font.sans-serif": ["Segoe UI", "DejaVu Sans", "sans-serif"],
        "figure.dpi": 110,
    })
    return style


def _split_statistic(column: str, statistics) -> tuple[str, str]:
    """Split a column into (signal, statistic), with "" when it carries none.

    The signal is the column name with the statistic token excised, so
    `Ch1_Avg_Pt1000_deg_C` and `VIN_RP_V_Avg` label as `Ch1_Pt1000_deg_C` and
    `VIN_RP_V` -- keeping the units and description that make a title readable.
    """
    # channel-prefixed, statistic straight after the channel: Ch1_Avg_deg_C
    match = _CHANNEL_PREFIXED.match(column)
    if match:
        channel, token, rest = match.groups()
        if token in statistics:
            return f"{channel}{rest}", token

    # suffixed, statistic trailing: VIN_RP_V_Avg
    head, _, token = column.rpartition("_")
    if head and token in statistics:
        return head, token

    return column, ""


def group_signals(data: pd.DataFrame, statistics=STATISTICS) -> dict[str, dict[str, str]]:
    """Map each signal to its {statistic: column}, keeping the frame's column order.

    A signal with no statistics gets a single entry keyed by "". Statistics are
    matched against `statistics` rather than by position, so `CHG_STATE` stays
    one plain signal instead of becoming a "STATE" statistic of `CHG`.
    """
    groups: dict[str, dict[str, str]] = {}
    for column in data.columns:
        signal, statistic = _split_statistic(str(column), statistics)
        groups.setdefault(signal, {})[statistic] = column
    return groups


def _choose_statistic(members: dict[str, str], wanted: str | None):
    """Pick the (statistic, column) of one signal to stand for it in an overlay.

    Returns None when a named statistic is asked for and the signal carries
    statistics but not that one -- a Max was wanted and only Avg/Min exist. A
    signal that carries no statistics at all has only its one column to give, so
    it answers with that rather than dropping out of the comparison.
    """
    if wanted is not None:
        if wanted in members:
            return wanted, members[wanted]
        if set(members) == {""}:
            return "", members[""]
        return None

    for name in PRIMARY_STATISTICS:
        if name in members:
            return name, members[name]
    if "" in members:
        return "", members[""]
    first = next(iter(members))
    return first, members[first]


def _draw(axes, label: str, values: pd.Series, style: tuple, samples: int,
          dashes_are_identity: bool = False) -> None:
    """One line, styled by the caller and thinned by how dense the data is."""
    color, dashes, width = style

    marks = {}
    if samples <= MARKER_THRESHOLD:
        # sparse enough that the samples themselves are worth seeing, with a
        # surface ring so overlapping markers stay legible
        marks = dict(marker="o", markersize=5, markeredgecolor=SURFACE,
                     markeredgewidth=1.2)
    if samples > DENSE_THRESHOLD:
        # thinning always helps; flattening to solid only when the dash is
        # reinforcement, never when it is the only thing telling two lines apart
        width *= 0.75
        if not dashes_are_identity:
            dashes = "-"

    axes.plot(values.index, values.to_numpy(dtype=float), label=label,
              color=color, linestyle=dashes, linewidth=width,
              solid_capstyle="round", **marks)


def _legend_outside(axes, handles=None, labels=None) -> None:
    """The key, parked to the right so overlapping lines are never occluded."""
    args = () if handles is None else (handles, labels)
    axes.legend(*args, loc="upper left", bbox_to_anchor=(1.01, 1.0),
                borderaxespad=0)


def _finish(axes) -> None:
    """The trim every axes gets: no top or right spine, no offset notation."""
    sns.despine(ax=axes, top=True, right=True)
    try:
        axes.ticklabel_format(style="plain", axis="y", useOffset=False)
    except AttributeError:
        pass  # a non-scalar formatter, nothing to flatten


def _report_skipped(skipped) -> None:
    """Name what was dropped, so nothing disappears silently."""
    if skipped:
        print(f"skipped {len(skipped)} column(s) with no numeric data: "
              f"{', '.join(map(str, skipped[:6]))}"
              f"{' ...' if len(skipped) > 6 else ''}")


def plot_signals(data: pd.DataFrame, columns=None, statistics=STATISTICS,
                 title: str = "", figsize: tuple = (10, 3), show: bool = True):
    """One axes per signal, with every statistic of that signal drawn on it.

    `columns` restricts and orders what is plotted -- pass
    protonode.signal_columns(df) to skip the plumbing columns. `title` is
    prepended to each signal's name, for the site or logger the frame came from.

    Values are coerced to numbers rather than type-checked, because nrgpy's
    concat_txt hands back object columns full of perfectly good floats where a
    single-file read gives float64. Columns with nothing numeric in them -- the
    hex-coded CHG_STATE, a timestamp left as a column -- are skipped, and named
    so nothing disappears silently. With show=False the figures are returned
    instead of displayed, for saving them yourself.
    """
    frame = data if columns is None else data[list(columns)]
    groups = group_signals(frame, statistics)

    figures, skipped = [], []
    with plt.rc_context(chart_style()):
        for signal, members in groups.items():
            drawable = {}
            for statistic, column in members.items():
                values = pd.to_numeric(frame[column], errors="coerce")
                if values.notna().any():
                    drawable[statistic] = values
            if not drawable:
                skipped.extend(members.values())
                continue

            panels = {}
            for statistic, values in drawable.items():
                panel = "spread" if statistic in SPREAD_STATISTICS else "value"
                panels.setdefault(panel, {})[statistic] = values
            order = [name for name in ("value", "spread") if name in panels]

            stacked = len(order) > 1
            figure, drawn = plt.subplots(
                len(order), 1, sharex=True, constrained_layout=True,
                figsize=(figsize[0], figsize[1] * (1.35 if stacked else 1.0)),
                gridspec_kw={"height_ratios": [2.4, 1]} if stacked else None)
            all_axes = list(drawn) if stacked else [drawn]

            samples = len(frame.index)
            for axes, panel in zip(all_axes, order):
                series = panels[panel]
                # False sorts first, so the primary is drawn last and lands on top
                for statistic in sorted(series, key=lambda s: s in PRIMARY_STATISTICS):
                    style = (STATISTIC_STYLE.get(statistic, FALLBACK_STYLE)
                             if statistic else PLAIN_STYLE)
                    # a lone line is labelled by its signal, having no statistic
                    _draw(axes, statistic or signal, series[statistic], style,
                          samples)
                if len(series) > 1:
                    rank = {name: i for i, name in enumerate(LEGEND_ORDER)}
                    keys = sorted(zip(*axes.get_legend_handles_labels()[::-1]),
                                  key=lambda pair: rank.get(pair[0], len(rank)))
                    _legend_outside(axes, [handle for _, handle in keys],
                                    [label for label, _ in keys])
                elif panel == "spread":
                    # one series needs no legend; the axis label names it
                    axes.set_ylabel(next(iter(series)))
                _finish(axes)

            all_axes[0].set_title(f"{title} {signal}".strip())
            all_axes[-1].set_xlabel("")

            if show:
                plt.show()
                plt.close(figure)
            else:
                figures.append(figure)

    _report_skipped(skipped)

    return figures if not show else None


def plot_together(data: pd.DataFrame, columns, statistic: str | None = None,
                  statistics=STATISTICS, title: str = "",
                  figsize: tuple = (10, 4), show: bool = True):
    """Several signals overlaid on one axes, coloured by signal.

    For the questions `plot_signals` cannot answer because its signals live in
    separate figures: whether two temperatures track, whether a rail sags when a
    heater draws. Identity moves from the statistic to the signal, so each gets
    one line -- drawing the full Avg/Min/Max/SD quartet of four signals would put
    sixteen lines on one axes and answer nothing.

    `columns` names what to draw and in what order, as either signal names (as
    `group_signals` keys them) or exact column names. An exact column pins the
    statistic, which is how two statistics of one signal get compared:
    `["VIN_RP_V_Min", "VIN_RP_V_Max"]`. A signal name takes `statistic` if given
    and its Avg otherwise, and a signal that lacks the named statistic is
    reported rather than quietly dropped.

    Nothing rescales here: signals in different units share one y-axis, and one
    that runs an order of magnitude larger will flatten the rest. Overlay
    signals that are commensurate, and reach for `plot_signals` when they are
    not. With show=False the figure is returned instead of displayed.
    """
    names = [str(name) for name in columns]
    if not names:
        raise ValueError("plot_together needs at least one signal to draw")

    groups = group_signals(data, statistics)
    lookup = {str(column): column for column in data.columns}

    chosen, missing = [], []
    for name in names:
        if name in groups:
            picked = _choose_statistic(groups[name], statistic)
            if picked is None:
                missing.append(name)
                continue
            chosen.append((name, *picked))
        elif name in lookup:
            signal, found = _split_statistic(name, statistics)
            chosen.append((signal, found, lookup[name]))
        else:
            raise KeyError(f"{name!r} is neither a signal nor a column; "
                           f"group_signals(data) lists the signals")

    drawable, skipped = [], []
    for signal, found, column in chosen:
        values = pd.to_numeric(data[column], errors="coerce")
        if values.notna().any():
            drawable.append((signal, found, values))
        else:
            skipped.append(column)

    if missing:
        print(f"skipped {len(missing)} signal(s) carrying no {statistic}: "
              f"{', '.join(missing)}")
    _report_skipped(skipped)
    if not drawable:
        return None

    # When every line shows the same statistic the legend says so once, in the
    # title; when they differ, each label has to carry its own.
    drawn = {found for _, found, _ in drawable}
    mixed = len(drawn) > 1
    heading = title if mixed else f"{title} {drawn.pop()}".strip()

    # Past the palette the dash is the only thing separating a repeated hue, so
    # it has to survive the density rule that would otherwise flatten it.
    identity = len(drawable) > len(SIGNAL_PALETTE)
    samples = len(data.index)

    with plt.rc_context(chart_style()):
        figure, axes = plt.subplots(figsize=figsize, constrained_layout=True)
        for position, (signal, found, values) in enumerate(drawable):
            style = (SIGNAL_PALETTE[position % len(SIGNAL_PALETTE)],
                     SIGNAL_DASHES[position // len(SIGNAL_PALETTE)
                                   % len(SIGNAL_DASHES)],
                     SIGNAL_WIDTH)
            label = f"{signal} {found}".strip() if mixed else signal
            _draw(axes, label, values, style, samples, dashes_are_identity=identity)

        _legend_outside(axes)
        _finish(axes)
        axes.set_title(heading)
        axes.set_xlabel("")

        if show:
            plt.show()
            plt.close(figure)
            return None
    return figure
