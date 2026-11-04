"""Slide 3: graded potency from the source patent.

The spreadsheet's potency column has only "< 20 nM" (16 compounds) and "Inactive" (537). The patent
US 9,340,546 B2 (Table 5) gives Ki bins for the same compounds (+++ < 0.01 µM, ++ 0.01-1 µM, + 1-5 µM); they were
read from the PDF (``data/patent_ki_bins.csv``).
A. How many of the 553 have a bin. B. Spreadsheet label against patent bin.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from matplotlib.colors import PowerNorm

import load_data as D
import plot_utils as pu



NAME = "slide_03_potency_recovery"

#: Patent bins left to right, most potent first, then compounds without a patent value.
_BIN_COLUMNS: tuple[str, ...] = ("+++", "++", "+", "NA")
_BIN_LABELS: dict[str, str] = {"+++": "+++", "++": "++", "+": "+", "NA": "no value"}

#: The spreadsheet's own wording for its two classes (ACTIVE / UNKNOWN in this repo).
_SHEET_LABELS: dict[str, str] = {D.POTENCY_ACTIVE: "< 20 nM", D.POTENCY_UNKNOWN: "Inactive"}


def potency_tables() -> tuple[pd.DataFrame, pd.DataFrame]:
    """The patent Ki bins joined onto the 553, and the contingency table (rows ACTIVE, UNKNOWN; columns +++, ++, +, NA)."""
    joined = D.join_patent_bins()
    table = D.contingency(joined).reindex(index=list(D.POTENCY_ORDER), columns=list(_BIN_COLUMNS), fill_value=0)
    return joined, table


def _panel_coverage(ax, n_matched: int, n_total: int) -> None:
    """Panel A: how many of the 553 compounds got a patent Ki bin."""
    counts = [n_matched, n_total - n_matched]
    ax.bar([0, 1], counts, width=0.62, color=[pu.SERIES, "#9A9A9A"], zorder=2)
    for x, value in enumerate(counts):
        ax.annotate(
            f"{value}  ({value / n_total:.0%})",
            (x, value),
            xytext=(0, 4),
            textcoords="offset points",
            ha="center",
            va="bottom",
            fontsize=pu.SIZE_LABEL,
            fontweight="bold",
        )
    ax.set_xticks([0, 1])
    ax.set_xticklabels(["Ki bin found", "no bin found"], fontsize=pu.SIZE_LABEL)
    ax.set_ylabel("Compounds")
    ax.set_ylim(0, n_total * 1.12)
    ax.set_xlim(-0.6, 1.6)
    ax.yaxis.grid(True)
    ax.tick_params(axis="x", length=0)


def _panel_contingency(ax, table: pd.DataFrame) -> None:
    """Panel B: spreadsheet label against patent Ki bin, counts in the cells."""
    values = table.to_numpy()
    n_rows, n_cols = values.shape
    # pcolormesh draws each cell as a vector patch; imshow would embed a raster image.
    ax.pcolormesh(
        np.arange(n_cols + 1),
        np.arange(n_rows + 1),
        values,
        cmap="Blues",
        norm=PowerNorm(0.45, vmin=0, vmax=values.max()),
        edgecolors="white",
        linewidth=2.0,
    )
    ax.set_xlim(0, n_cols)
    ax.set_ylim(n_rows, 0)
    ax.set_xticks(np.arange(n_cols) + 0.5)
    ax.set_xticklabels([_BIN_LABELS[c] for c in table.columns], fontsize=pu.SIZE_LABEL)
    ax.set_yticks(np.arange(n_rows) + 0.5)
    ax.set_yticklabels(
        [f"'{_SHEET_LABELS[r]}'\n(n = {int(table.loc[r].sum())})" for r in table.index],
        fontsize=pu.SIZE_LABEL,
    )
    ax.set_xlabel("Patent Ki bin")
    ax.set_ylabel("Spreadsheet label")
    ax.tick_params(length=0)
    for spine in ax.spines.values():
        spine.set_visible(False)

    threshold = values.max() * 0.45
    for i in range(n_rows):
        for j in range(n_cols):
            ax.text(
                j + 0.5,
                i + 0.5,
                str(values[i, j]),
                ha="center",
                va="center",
                fontsize=pu.SIZE_LABEL,
                fontweight="bold",
                color="white" if values[i, j] > threshold else pu.INK,
            )


def slide_03_potency_recovery() -> None:
    joined, table = potency_tables()
    n_total = len(joined)
    n_matched = int(joined["ki_bin"].notna().sum())

    inactive = table.loc[D.POTENCY_UNKNOWN]
    n_inactive_with_value = int(inactive[["+++", "++", "+"]].sum())
    n_inactive_potent = int(inactive[["+++", "++"]].sum())

    fig = pu.new_figure()
    pu.panel_heading(fig, 0.0, 0.1, "A.  Patent coverage of the 553")
    pu.panel_heading(fig, 5.1, 0.1, "B.  Axiom Bio spreadsheet label vs patent Ki bin")

    # Both plots end on the same line (their axes' bottom edges).
    plot_bottom = 4.5
    ax_a = pu.axes_in(fig, 0.85, 0.95, 3.6, plot_bottom - 0.95)
    _panel_coverage(ax_a, n_matched, n_total)

    # The axes start 0.95 in right of the panel edge, so the "Spreadsheet label" axis title
    # lines up with the panel title and the sentence above the plot.
    ax_c = pu.axes_in(fig, 5.1 + 0.95, plot_bottom - 3.2, 5.0, 3.2)
    _panel_contingency(ax_c, table)

    fig.text(
        5.1 / pu.SLIDE_W_IN,
        1.0 - 0.68 / pu.SLIDE_H_IN,
        f"{n_inactive_potent} of the {n_inactive_with_value} 'Inactive' compounds that have a patent value "
        "have Ki < 1 µM.",
        ha="left",
        va="top",
        fontsize=pu.SIZE_LABEL,
        color=pu.INK,
        wrap=True,
    )
    pu.footnote(
        fig,
        0.0,
        5.4,
        "Ki bins read from US9340546B2: +++ Ki < 0.01 µM, ++ 0.01–1 µM, + 1–5 µM.\n"
        "The spreadsheet's labels come from a different table with different thresholds "
        "('< 20 nM' vs 'Inactive'), so the two columns are not directly comparable.",
    )
    pu.save_svg(fig, NAME)

