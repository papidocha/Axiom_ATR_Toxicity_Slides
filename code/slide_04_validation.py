"""Slide 4: how general_tox is built and how far the predictions can be trusted.

A. The definition of general_tox.
B. Predicted against measured EC for the two compounds with experimental data (n = 2, probably in the model's
   training set: a sanity check, not a validation).
C. Percentile of gartisertib and tuvusertib in every column that enters general_tox. Percentiles are ranks, so
   compounds tied at the 200 µM ceiling share a mid-rank (hollow triangles).
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from matplotlib.backends.backend_agg import RendererAgg
from matplotlib.figure import Figure
from matplotlib.font_manager import FontProperties
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

import load_data as D
import plot_utils as pu



NAME = "slide_04_validation"

#: Size of all text on this slide except the panel titles (16 pt) and the equation
#: (12 pt, with 10 pt scripts), so the axis labels, legends, notes and tick labels agree.
_BODY = pu.SIZE_SMALL
_CM_IN = 1 / 2.54
#: Vertical distance between consecutive bullet points, inches (panels A and B).
_BULLET_PITCH_IN = 0.3
#: Panels A and B sit this many inches right of the slide's left edge, and panel C this many
#: inches left of its old position, which narrows the gap between the two columns.
_SHIFT = 0.0

_CALIBRATION_LIMITS = (0.2, 330.0)


# --------------------------------------------------------------------------- #
# Panel C: percentile profile
# --------------------------------------------------------------------------- #


def _panel_profile(ax, profile: pd.DataFrame) -> None:
    """Dumbbell per column: gartisertib against tuvusertib, general_tox on its own row."""
    columns = [c for c in profile.index if c != "general_tox"]
    labels = ["general_tox"] + [pu.column_label(c) for c in columns]
    positions = np.concatenate([[0.0], 1.6 + np.arange(len(columns))])
    order = ["general_tox", *columns]

    for position, key in zip(positions, order):
        row = profile.loc[key]
        ax.plot(
            [row["gartisertib"], row["tuvusertib"]],
            [position, position],
            color="#8C8C8C",
            linewidth=1.3,
            zorder=1,
        )
        for compound, colour in (("gartisertib", pu.GARTISERTIB), ("tuvusertib", pu.TUVUSERTIB)):
            if row[f"censored_{compound}"]:
                ax.scatter(
                    row[compound],
                    position,
                    s=62,
                    marker=">",
                    facecolor="white",
                    edgecolor=colour,
                    linewidth=1.6,
                    zorder=3,
                )
            else:
                size = 80 if key == "general_tox" else 46
                ax.scatter(row[compound], position, s=size, color=colour, zorder=3, linewidth=0)

    ax.axvline(0.5, color=pu.MUTED, linewidth=0.8, linestyle=(0, (2, 3)), zorder=0)
    for compound, colour in (("gartisertib", pu.GARTISERTIB), ("tuvusertib", pu.TUVUSERTIB)):
        ax.annotate(
            f"{profile.at['general_tox', compound]:.2f}",
            (profile.at["general_tox", compound], 0.0),
            xytext=(0, 9),
            textcoords="offset points",
            ha="center",
            fontsize=pu.SIZE_SMALL,
            fontweight="bold",
            color=colour,
        )

    ax.set_yticks(positions)
    ax.set_yticklabels(labels, fontsize=pu.SIZE_SMALL)
    ax.get_yticklabels()[0].set_fontweight("bold")
    ax.set_ylim(positions[-1] + 0.7, -0.9)
    ax.set_xlim(0.0, 1.02)
    ax.set_xticks([0, 0.25, 0.5, 0.75, 1.0])
    ax.set_xlabel("Predicted percentile (higher = safer)", fontsize=_BODY)
    ax.xaxis.grid(True)
    ax.tick_params(axis="y", length=0)


# --------------------------------------------------------------------------- #
# Equation typesetting: live text from measured glyph widths (an SVG has no maths layout)
# --------------------------------------------------------------------------- #

EQUATION_FONT = "Times New Roman"
_POINTS_PER_INCH = 72.0
_RENDERER = RendererAgg(1000, 1000, _POINTS_PER_INCH)


@dataclass(frozen=True)
class Run:
    """A piece of text on one baseline.

    Attributes:
        text: The characters.
        size: Font size in points (one of the figure's three sizes).
        italic: True for variables.
        dy: Baseline shift in inches, positive = up (sub- and superscripts).
    """

    text: str
    size: float = pu.SIZE_LABEL
    italic: bool = False
    dy: float = 0.0

    def shown(self) -> str:
        """The text with ordinary spaces made non-breaking, so SVG viewers keep them."""
        return self.text.replace(" ", "\u00a0")

    def width(self) -> float:
        """Advance width in inches."""
        props = FontProperties(family=EQUATION_FONT, style="italic" if self.italic else "normal", size=self.size)
        width_px, _, _ = _RENDERER.get_text_width_height_descent(self.shown(), props, ismath=False)
        return width_px / _POINTS_PER_INCH


@dataclass(frozen=True)
class Stack:
    """Runs stacked on a common centre: a fraction (with a bar) or a sum with limits.

    Attributes:
        centre: Runs on the baseline (the sigma of a sum); empty for a fraction.
        above: Runs centred over it (numerator, upper limit).
        below: Runs centred under it (denominator, lower limit).
        above_dy: Baseline of ``above`` relative to the main baseline, inches.
        below_dy: Baseline of ``below`` relative to the main baseline, inches.
        bar_dy: Height of the fraction bar above the baseline in inches; None for no bar.
    """

    above: list[Run]
    below: list[Run]
    centre: list[Run] = field(default_factory=list)
    above_dy: float = 0.0
    below_dy: float = 0.0
    bar_dy: float | None = None

    def width(self) -> float:
        pad = 0.06 if self.bar_dy is not None else 0.02
        return max(_row_width(self.above), _row_width(self.below), _row_width(self.centre)) + 2 * pad


Item = Run | Stack


def _row_width(items: list[Run]) -> float:
    return sum(r.width() for r in items)


def text_width(text: str, size: float, family: str = pu.FONT_FAMILY) -> float:
    """Width in inches of ``text`` set in ``family`` (Arial by default) at ``size`` points."""
    props = FontProperties(family=family, size=size)
    width_px, _, _ = _RENDERER.get_text_width_height_descent(text, props, ismath=False)
    return width_px / _POINTS_PER_INCH


def runs_width(items: list[Item]) -> float:
    """Total width in inches of a row of items."""
    return sum(i.width() for i in items)


def fraction(
    numerator: list[Run], denominator: list[Run], size: float = pu.SIZE_LABEL, lift: float = 0.0
) -> Stack:
    """A stacked fraction with a rule, sized for text of ``size`` points.

    ``lift`` raises the numerator by that many inches, for numerators whose subscripts
    would otherwise touch the rule.
    """
    em = size / _POINTS_PER_INCH
    return Stack(
        above=numerator,
        below=denominator,
        above_dy=0.50 * em + lift,
        below_dy=-0.62 * em,
        bar_dy=0.27 * em,
    )


def summation(upper: list[Run], lower: list[Run], sigma_size: float = pu.SIZE_HEADING) -> Stack:
    """A capital sigma with its limits above and below."""
    em = sigma_size / _POINTS_PER_INCH
    return Stack(
        centre=[Run("Σ", sigma_size)],
        above=upper,
        below=lower,
        above_dy=0.98 * em,
        below_dy=-0.42 * em - 0.05,
    )


def _place(fig: Figure, x: float, baseline: float, run: Run) -> None:
    fig.text(
        x / pu.SLIDE_W_IN,
        1.0 - (baseline - run.dy) / pu.SLIDE_H_IN,
        run.shown(),
        ha="left",
        va="baseline",
        fontsize=run.size,
        fontfamily=EQUATION_FONT,
        fontstyle="italic" if run.italic else "normal",
        color=pu.INK,
    )


def draw_runs(fig: Figure, x_in: float, baseline_in: float, items: list[Item]) -> float:
    """Draw a row of items starting at ``x_in`` on a baseline ``baseline_in`` from the top.

    Returns:
        The x (inches) where the row ends.
    """
    x = x_in
    for item in items:
        if isinstance(item, Run):
            _place(fig, x, baseline_in, item)
            x += item.width()
            continue
        total = item.width()
        for part, dy in ((item.centre, 0.0), (item.above, item.above_dy), (item.below, item.below_dy)):
            if not part:
                continue
            start = x + (total - _row_width(part)) / 2
            for run in part:
                _place(fig, start, baseline_in - dy, run)
                start += run.width()
        if item.bar_dy is not None:
            y = 1.0 - (baseline_in - item.bar_dy) / pu.SLIDE_H_IN
            fig.add_artist(
                Line2D(
                    [(x + 0.02) / pu.SLIDE_W_IN, (x + total - 0.02) / pu.SLIDE_W_IN],
                    [y, y],
                    transform=fig.transFigure,
                    color=pu.INK,
                    linewidth=0.8,
                    solid_capstyle="butt",
                )
            )
        x += total
    return x


# --------------------------------------------------------------------------- #
# Panel A: the definition (equation and the meaning of its symbols)
# --------------------------------------------------------------------------- #


def _sub(text: str, italic: bool = True) -> Run:
    """A 10 pt subscript run, followed by a hair of space so it does not touch the next glyph."""
    return Run(text + "\u2009", pu.SIZE_SMALL, italic=italic, dy=-0.05)


def _var(letter: str, index: str | None = None) -> list[Run]:
    """An italic variable, optionally with an italic subscript."""
    runs = [Run(letter, italic=True)]
    return runs + ([_sub(index)] if index else [])


def draw_equation(fig, x_in: float, y_in: float, width_in: float = 5.9) -> float:
    """The general_tox equation, set as live text; returns the y (inches) where it ends.

    Row 1: the mean over families and the mean within a family. Row 2: the z-score of
    one column, with its name beside it. Each row is centred in ``width_in``.
    """
    row_1, row_2 = y_in + 0.26, y_in + 1.1

    family_mean = [*_var("m", "f")]
    general = [
        Run("general_tox"),
        Run(" = "),
        fraction([Run("1")], [Run("9")]),
        summation([_sub("9", italic=False)], [_sub("f"), _sub(" = 1", italic=False)]),
        *family_mean,
    ]
    within = [
        *family_mean,
        Run(" = "),
        fraction([Run("1")], _var("n", "f")),
        summation([], [_sub("i"), _sub(" in ", italic=False), _sub("f")]),
        *_var("z", "i"),
    ]
    gap_1 = 0.55
    row_1_width = runs_width(general) + gap_1 + runs_width(within)
    assert row_1_width < width_in, "equation row 1 no longer fits the left column"
    start_1 = x_in + (width_in - row_1_width) / 2
    # Nudged left of the centred position: general_tox by 2 cm, m_f by 1 cm.
    start_general = max(x_in, start_1 - _CM_IN * 2)
    draw_runs(fig, start_general, row_1, general)
    draw_runs(fig, start_1 + runs_width(general) + gap_1 - _CM_IN, row_1, within)

    z_score = [
        *_var("z", "i"),
        Run(" = "),
        fraction(
            [
                Run("log"),
                _sub("10", italic=False),
                Run(" min(EC"),
                _sub("i"),
                Run(", 200 µM) − "),
                *_var("μ", "i"),
            ],
            _var("σ", "i"),
            lift=0.07,
        ),
    ]
    name, gap_2 = "(z-scoring)", 0.5
    row_2_width = runs_width(z_score) + gap_2 + text_width(name, _BODY)
    assert row_2_width < width_in, "equation row 2 no longer fits the left column"
    start_2 = x_in + (width_in - row_2_width) / 2
    end_z = draw_runs(fig, start_2, row_2, z_score)
    fig.text(
        (end_z + gap_2) / pu.SLIDE_W_IN,
        1.0 - row_2 / pu.SLIDE_H_IN,
        name,
        ha="left",
        va="baseline",
        fontsize=_BODY,
        color=pu.INK,
    )
    return row_2 + 0.42


def draw_symbol_notes(fig, x_in: float, y_in: float) -> float:
    """Bullets explaining m_f and n_f (typeset symbol, then Arial text); returns the end y."""
    notes = (
        (_var("m", "f"), ":  Mean z-score of family f (e.g. cell viability: Its EC20 and EC50 z averaged)"),
        (_var("n", "f"), ":  EC levels of family f in use (2 = EC20 and EC50, 1 = EC20 only)"),
    )
    bullet_width = text_width("•  ", _BODY)
    y = y_in
    for symbol, text in notes:
        baseline = y + 0.13
        fig.text(
            x_in / pu.SLIDE_W_IN,
            1.0 - baseline / pu.SLIDE_H_IN,
            "•",
            ha="left",
            va="baseline",
            fontsize=_BODY,
            color=pu.INK,
        )
        x_end = draw_runs(fig, x_in + bullet_width, baseline, symbol)
        fig.text(
            (x_end + 0.02) / pu.SLIDE_W_IN,
            1.0 - baseline / pu.SLIDE_H_IN,
            text,
            ha="left",
            va="baseline",
            fontsize=_BODY,
            color=pu.INK,
        )
        y += _BULLET_PITCH_IN
    return y


# --------------------------------------------------------------------------- #
# Panel B: calibration
# --------------------------------------------------------------------------- #


def _panel_calibration(ax, table: pd.DataFrame) -> None:
    """Predicted against measured effect concentration, both on log axes."""
    lo, hi = _CALIBRATION_LIMITS
    grid = np.array([lo, hi])
    ax.fill_between(grid, grid / 10.0, grid * 10.0, color="#EBEBEB", linewidth=0, zorder=0)
    ax.plot(grid, grid, color=pu.INK, linewidth=1.0, zorder=1)

    for compound, colour in ((D.GARTISERTIB, pu.GARTISERTIB), (D.TUVUSERTIB, pu.TUVUSERTIB)):
        for level, marker in (("ld20", "o"), ("ld50", "s")):
            part = table[(table["compound"] == compound) & (table["level"] == level)]
            good = part[part["comparable"]]
            bad = part[~part["comparable"]]
            ax.scatter(
                good["experimental_uM"],
                good["predicted_uM"],
                s=42,
                marker=marker,
                color=colour,
                linewidth=0.6,
                edgecolor="white",
                zorder=3,
            )
            ax.scatter(
                bad["experimental_uM"],
                bad["predicted_uM"],
                s=42,
                marker=marker,
                facecolor="none",
                edgecolor=colour,
                linewidth=1.3,
                zorder=3,
            )

    ax.axvline(D.CEILING_UM, color=pu.CENSORED, linewidth=0.8, linestyle=(0, (1, 2)), zorder=1)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlim(lo, hi)
    ax.set_ylim(lo, hi)
    ticks = [1, 10, 100]
    ax.set_xticks(ticks)
    ax.set_yticks(ticks)
    ax.set_xticklabels([str(t) for t in ticks])
    ax.set_yticklabels([str(t) for t in ticks])
    ax.minorticks_off()
    ax.set_xlabel("Measured EC (µM)", fontsize=_BODY)
    ax.set_ylabel("Predicted EC (µM)", fontsize=_BODY)
    ax.grid(True)
    ax.set_aspect("equal", adjustable="box")


def _calibration_legend(fig, x_in: float, y_in: float) -> None:
    """Legend for panel C as an absolutely placed block."""
    handles = [
        Line2D([], [], marker="o", linestyle="none", markersize=6, color=pu.GARTISERTIB, label="Gartisertib"),
        Line2D([], [], marker="o", linestyle="none", markersize=6, color=pu.TUVUSERTIB, label="Tuvusertib"),
        Line2D([], [], marker="o", linestyle="none", markersize=6, color=pu.MUTED, label="EC20"),
        Line2D([], [], marker="s", linestyle="none", markersize=6, color=pu.MUTED, label="EC50"),
        Line2D([], [], linestyle=(0, (1, 2)), color=pu.CENSORED, linewidth=1.0, label="200 µM ceiling"),
        Patch(facecolor="#EBEBEB", edgecolor="#C8C8C8", linewidth=0.6, label="within 10-fold"),
    ]
    fig.legend(
        handles=handles,
        loc="upper left",
        bbox_to_anchor=(x_in / pu.SLIDE_W_IN, 1.0 - y_in / pu.SLIDE_H_IN),
        fontsize=_BODY,
        labelspacing=0.3,
        handletextpad=0.4,
    )


# --------------------------------------------------------------------------- #
# Slide
# --------------------------------------------------------------------------- #


def slide_04_validation() -> None:
    series = D.load_lead_series()
    profile = D.percentile_profile(series)
    table = D.calibration_table()
    points = D.comparable_points(table)

    fig = pu.new_figure()

    # --- A: definition ---------------------------------------------------- #
    pu.panel_heading(fig, _SHIFT, 0.1, "A.  general_tox: A unified toxicity score")
    end_eq = draw_equation(fig, _SHIFT, 0.58)
    end_a = draw_symbol_notes(fig, _SHIFT, end_eq + 0.05)

    # --- B: calibration --------------------------------------------------- #
    top_b = end_a + 0.1
    pu.panel_heading(fig, _SHIFT, top_b, "B.  Predicted vs measured EC")
    plot_size = min(2.6, pu.SLIDE_H_IN - 0.65 - (top_b + 0.45))
    legend_top = top_b + 0.4
    # The two bullets sit midway between the end of the legend (about 0.95 in tall) and the
    # bottom of the plot, so the white space above and below them is about equal.
    stats_top = (legend_top + 0.95 + top_b + 0.45 + plot_size) / 2 - 0.365
    ax_b = pu.axes_in(fig, _SHIFT + 0.9, top_b + 0.45, plot_size, plot_size)
    _panel_calibration(ax_b, table)
    _calibration_legend(fig, _SHIFT + 3.45, legend_top)

    bullets = (
        f"•  Median fold error {points['fold_error'].median():.1f}×",
        f"•  {int(points['over_predicts_safety'].sum())} of {len(points)} predict a safer EC",
    )
    for k, text in enumerate(bullets):
        # Same 0.3 in pitch as the m_f / n_f bullets in panel A.
        fig.text(
            (_SHIFT + 3.5) / pu.SLIDE_W_IN,
            1.0 - (stats_top + k * _BULLET_PITCH_IN) / pu.SLIDE_H_IN,
            text,
            ha="left",
            va="top",
            fontsize=_BODY,
            color=pu.INK,
        )

    # --- C: percentile profile -------------------------------------------- #
    left = 6.15
    pu.panel_heading(fig, left, 0.1, "C.  Predicted percentile in the 553-compound series")
    fig.legend(
        handles=[
            Line2D([], [], marker="o", linestyle="none", markersize=6, color=pu.GARTISERTIB, label="Gartisertib"),
            Line2D([], [], marker="o", linestyle="none", markersize=6, color=pu.TUVUSERTIB, label="Tuvusertib"),
            Line2D(
                [],
                [],
                marker=">",
                linestyle="none",
                markersize=6,
                markerfacecolor="white",
                markeredgecolor=pu.MUTED,
                label="at the 200 µM ceiling",
            ),
        ],
        loc="upper left",
        bbox_to_anchor=(left / pu.SLIDE_W_IN, 1.0 - 0.6 / pu.SLIDE_H_IN),
        ncols=3,
        fontsize=_BODY,
        columnspacing=1.0,
        handletextpad=0.3,
    )
    ax_c = pu.axes_in(fig, left + 1.78, 1.2, 3.95, 4.15)
    _panel_profile(ax_c, profile)
    pu.save_svg(fig, NAME)
