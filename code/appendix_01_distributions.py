"""Appendix: distribution of all 20 predicted EC20 / EC50 columns.

Each panel is one readout at one effect level on a log10 concentration axis. Values at or above the 200 µM
ceiling are right-censored ("no effect reached below the ceiling") and drawn as a hatched count past an axis
break. Columns that do not enter general_tox are marked "(excl.)".
"""

from __future__ import annotations

import numpy as np
from matplotlib.patches import Patch
from matplotlib.ticker import MaxNLocator

import load_data as D
import plot_utils as pu



NAME = "appendix_01_distributions"

_LOG_CEILING = float(np.log10(D.CEILING_UM))
_BINS = np.linspace(-1.0, _LOG_CEILING, 28)
_BREAK_X = _LOG_CEILING + 0.14
_CENSORED_X = _LOG_CEILING + 0.40
_XLIM = (-0.85, _LOG_CEILING + 0.62)

#: Layout in inches: left edge of the plot grid, column width and gutter.
_GRID_LEFT, _COL_W, _COL_GAP = 0.66, 1.80, 0.46
_ROW_H, _ROW_GAP = 0.62, 0.50


def _uncensored_log(series: D.LeadSeries, column: str):
    """log10 values of the compounds below the ceiling; censored ones are removed."""
    return series.log_values.loc[~series.censored[column], column]


def _draw_panel(ax, series: D.LeadSeries, column: str) -> None:
    """One column: histogram below the ceiling plus a hatched count at the ceiling."""
    counts, _, _ = ax.hist(
        _uncensored_log(series, column),
        bins=_BINS,
        color=pu.SERIES,
        edgecolor="white",
        linewidth=0.3,
        zorder=2,
    )
    n_censored = int(series.censored[column].sum())
    height = max(float(counts.max()), float(n_censored), 1.0)

    if n_censored:
        ax.axvline(_BREAK_X, color=pu.MUTED, linewidth=0.8, linestyle=(0, (2, 2)))
        ax.bar(
            _CENSORED_X,
            n_censored,
            width=0.20,
            facecolor="white",
            edgecolor=pu.CENSORED,
            hatch="////",
            linewidth=1.0,
            zorder=3,
        )
        ax.annotate(
            str(n_censored),
            (_CENSORED_X, n_censored),
            xytext=(0, 2),
            textcoords="offset points",
            ha="center",
            va="bottom",
            fontsize=pu.SIZE_SMALL,
            color=pu.CENSORED,
            fontweight="bold",
        )

    ax.set_xlim(*_XLIM)
    ax.set_ylim(0, height * 1.38)
    ax.yaxis.set_major_locator(MaxNLocator(nbins=2, integer=True, steps=[1, 2, 5, 10]))
    ax.set_xticks([0, 1, 2, _CENSORED_X])
    ax.set_xticklabels(["1", "10", "100", f"≥{D.CEILING_UM:.0f}"])


def appendix_01_distributions() -> None:
    series = D.load_lead_series()
    excluded = set(series.excluded)
    fig = pu.new_figure()

    blocks = (
        ("EC20", D.LD20_COLUMNS, 0.62),
        ("EC50", D.LD50_COLUMNS, 3.62),
    )
    for level, columns, top in blocks:
        pu.panel_heading(fig, 0.0, top - 0.52, f"{level}  predicted concentration giving a "
                         f"{level[2:]}% shift in the readout")
        for index, column in enumerate(columns):
            row, col = divmod(index, 5)
            ax = pu.axes_in(
                fig,
                _GRID_LEFT + col * (_COL_W + _COL_GAP),
                top + 0.18 + row * (_ROW_H + _ROW_GAP),
                _COL_W,
                _ROW_H,
            )
            _draw_panel(ax, series, column)

            assay = column.rsplit("_", 1)[0]
            is_excluded = column in excluded
            ax.set_title(
                pu.SHORT_NAMES[assay] + ("  (excl.)" if is_excluded else ""),
                fontsize=pu.SIZE_LABEL,
                fontweight="bold",
                color=pu.MUTED if is_excluded else pu.INK,
                pad=4,
            )
            if col == 0:
                ax.set_ylabel("Compounds", fontsize=pu.SIZE_LABEL)
            if row == 1:
                ax.set_xlabel("Concentration (µM)", fontsize=pu.SIZE_SMALL, labelpad=2)

    handles = [
        Patch(facecolor=pu.SERIES, edgecolor="none", label="Predicted, below the 200 µM ceiling"),
        Patch(
            facecolor="white",
            edgecolor=pu.CENSORED,
            hatch="////",
            label="At or above the ceiling (censored)",
        ),
        Patch(facecolor="none", edgecolor="none", label="(excl.) = not used in general_tox"),
    ]
    fig.legend(
        handles=handles,
        loc="lower left",
        bbox_to_anchor=(0.0, -0.01),
        ncols=3,
        fontsize=pu.SIZE_LABEL,
        handlelength=1.6,
        columnspacing=1.8,
    )
    pu.save_svg(fig, NAME)
