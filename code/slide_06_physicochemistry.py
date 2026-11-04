"""Slide 6: which molecular properties go with general_tox.

A. Basic nitrogen count (box plot), B. molecular weight, C. cLogP, D. Fsp3, E. rotatable bonds, F. H-bond donors,
G. TPSA, H. every Spearman rho ranked. Spearman is rank based, so the censored 200 µM ceiling does not distort it.
The correlations are marginal. Gartisertib and tuvusertib are marked.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from matplotlib.ticker import MaxNLocator

import load_data as D
import plot_utils as pu


NAME = "slide_06_physicochemistry"

#: Descriptor column -> axis label. Order is the order of the scatter panels.
_BASIC_N = "n_basic_nitrogen"
_SCATTER: tuple[tuple[str, str], ...] = (
    ("mw", "MW (Da)"),
    ("clogp", "cLogP"),
    ("fsp3", "Fsp³"),
    ("rotatable_bonds", "Rotatable bonds"),
    ("hbd", "H-bond donors"),
    ("tpsa", "TPSA (Å²)"),
)
_RANK_LABELS: dict[str, str] = {
    _BASIC_N: "Basic N",
    "mw": "MW",
    "clogp": "cLogP",
    "fsp3": "Fsp³",
    "rotatable_bonds": "Rotatable bonds",
    "hbd": "H-bond donors",
    "tpsa": "TPSA",
}
_HEADINGS = (
    "A.  Basic nitrogens",
    "B.  Molecular weight",
    "C.  cLogP",
    "D.  Fsp³",
    "E.  Rotatable bonds",
    "F.  H-bond donors",
    "G.  TPSA",
    "H.  Ranked by |ρ|",
)

#: Grid geometry, inches from the top-left of the canvas.
_COL_X = (0.62, 3.67, 6.72, 9.77)
_AXES_W = 2.45
_ROW_HEADING = (0.08, 3.03)
_ROW_TOP = (0.5, 3.45)
_AXES_H = 1.65

_POINT = pu.SERIES
_HIGHLIGHT = pu.CENSORED  #: basic-nitrogen median line and bar


def _minus(text: str) -> str:
    return text.replace("-", "−")


def physchem_frame() -> pd.DataFrame:
    """One row per compound: ``general_tox`` and every descriptor that is drawn."""
    series = D.load_lead_series()
    descriptors = D.load_descriptors()
    frame = descriptors[[_BASIC_N, *(name for name, _ in _SCATTER)]].copy()
    frame["general_tox"] = series.data.set_index("name")["general_tox"].reindex(frame.index)
    return frame


def _rho_note(ax, rho: float) -> None:
    ax.text(
        0.04,
        0.04,
        _minus(f"ρ = {rho:+.2f}"),
        transform=ax.transAxes,
        ha="left",
        va="bottom",
        fontsize=pu.SIZE_SMALL,
        fontweight="bold",
        color=pu.INK,
        bbox={"boxstyle": "round,pad=0.25", "facecolor": "white", "edgecolor": "none", "alpha": 0.85},
        zorder=8,
    )


def _mark_references(ax, x: pd.Series, y: pd.Series) -> None:
    for compound, colour in ((D.GARTISERTIB, pu.GARTISERTIB), (D.TUVUSERTIB, pu.TUVUSERTIB)):
        ax.scatter(
            x[compound], y[compound], s=60, color=colour, edgecolor="white", linewidth=1.1, zorder=7
        )


def _y_axis(ax, frame: pd.DataFrame, first_column: bool) -> None:
    ax.set_ylim(frame["general_tox"].min() - 0.25, frame["general_tox"].max() + 0.25)
    ax.set_yticks([-2, -1, 0, 1, 2, 3])
    ax.set_yticklabels(["−2", "−1", "0", "1", "2", "3"])
    ax.grid(axis="y")
    if first_column:
        ax.set_ylabel("general_tox")
    else:
        ax.tick_params(axis="y", labelleft=False)


def _panel_basic_n(fig, frame: pd.DataFrame, rho: float) -> None:
    """General_tox by basic-nitrogen count: one box per count, the points on top."""
    ax = pu.axes_in(fig, _COL_X[0], _ROW_TOP[0], _AXES_W, _AXES_H)
    rng = np.random.default_rng(D.RANDOM_SEED)
    groups = sorted(int(g) for g in frame[_BASIC_N].unique())
    counts: dict[int, int] = {}
    for group in groups:
        sample = frame.loc[frame[_BASIC_N] == group, "general_tox"].dropna()
        counts[group] = len(sample)
        ax.boxplot(
            sample.to_numpy(),
            positions=[group],
            widths=0.55,
            showfliers=False,
            patch_artist=True,
            boxprops={"facecolor": "none", "edgecolor": pu.MUTED, "linewidth": 1.0},
            medianprops={"color": _HIGHLIGHT, "linewidth": 2.0},
            whiskerprops={"color": pu.MUTED, "linewidth": 0.9},
            capprops={"color": pu.MUTED, "linewidth": 0.9},
        )
        ax.scatter(
            group + rng.normal(0, 0.075, size=len(sample)),
            sample.to_numpy(),
            s=7,
            color=_POINT,
            alpha=0.45,
            linewidth=0,
            zorder=3,
        )
    _mark_references(ax, frame[_BASIC_N], frame["general_tox"])
    ax.set_xticks(groups)
    ax.set_xticklabels([f"{g}\n(n = {counts[g]})" for g in groups])
    ax.set_xlim(min(groups) - 0.6, max(groups) + 0.6)
    ax.set_xlabel("Basic nitrogens per molecule")
    ax.grid(axis="x", visible=False)
    _y_axis(ax, frame, first_column=True)
    _rho_note(ax, rho)


def _panel_scatter(fig, frame: pd.DataFrame, index: int, column: str, label: str, rho: float) -> None:
    """One descriptor against general_tox (index 1 to 6 in reading order)."""
    row, col = divmod(index, 4)
    ax = pu.axes_in(fig, _COL_X[col], _ROW_TOP[row], _AXES_W, _AXES_H)
    values = frame[column]
    rng = np.random.default_rng(D.RANDOM_SEED + index)
    discrete = bool((values.dropna() % 1 == 0).all())
    x = values + rng.normal(0, 0.10, len(values)) if discrete else values
    ax.scatter(x, frame["general_tox"], s=7, color=_POINT, alpha=0.45, linewidth=0, zorder=2)
    _mark_references(ax, values, frame["general_tox"])
    ax.set_xlabel(label)
    if discrete:
        ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    ax.grid(axis="x", visible=False)
    _y_axis(ax, frame, first_column=col == 0)
    _rho_note(ax, rho)


def _panel_ranking(fig, rhos: dict[str, float]) -> None:
    """Every Spearman rho on one axis, the basic-nitrogen count highlighted."""
    ax = pu.axes_in(fig, _COL_X[3] + 0.75, _ROW_TOP[1], _AXES_W - 0.75, _AXES_H)
    ranked = pd.Series(rhos).sort_values(key=np.abs)  # weakest at the bottom
    colours = [_HIGHLIGHT if name == _BASIC_N else _POINT for name in ranked.index]
    ax.barh(range(len(ranked)), ranked.to_numpy(), color=colours, height=0.7)
    ax.axvline(0, color=pu.INK, linewidth=0.8)
    ax.set_yticks(range(len(ranked)))
    ax.set_yticklabels([_RANK_LABELS[name] for name in ranked.index])
    ax.tick_params(axis="y", length=0)
    for y, value in enumerate(ranked.to_numpy()):
        ax.annotate(
            _minus(f"{value:+.2f}"),
            (value, y),
            xytext=(3 if value > 0 else -3, 0),
            textcoords="offset points",
            ha="left" if value > 0 else "right",
            va="center",
            fontsize=pu.SIZE_SMALL,
            color=pu.INK,
        )
    ax.set_xlim(-1.3, 0.55)
    ax.set_xticks([-1, -0.5, 0])
    ax.set_xticklabels(["−1", "−0.5", "0"])
    ax.set_xlabel("Spearman ρ")
    ax.grid(axis="y", visible=False)
    ax.spines["left"].set_visible(False)


def _legend(fig) -> None:
    handles = [
        Line2D([], [], marker="o", linestyle="none", markersize=6, color=pu.GARTISERTIB, label="Gartisertib"),
        Line2D([], [], marker="o", linestyle="none", markersize=6, color=pu.TUVUSERTIB, label="Tuvusertib"),
    ]
    fig.legend(
        handles=handles,
        loc="center left",
        bbox_to_anchor=(0.62 / pu.SLIDE_W_IN, 1.0 - 6.1 / pu.SLIDE_H_IN),
        ncols=2,
        fontsize=pu.SIZE_SMALL,
        columnspacing=1.2,
        handletextpad=0.3,
    )
    pu.footnote(
        fig,
        3.67,
        6.03,
        "Spearman ρ over all 553 compounds. Higher general_tox = safer.",
    )


# --------------------------------------------------------------------------- #
# Slide
# --------------------------------------------------------------------------- #


def slide_06_physicochemistry() -> None:
    frame = physchem_frame()
    rhos = {
        name: D.spearman(frame[name], frame["general_tox"])[0]
        for name in (_BASIC_N, *(n for n, _ in _SCATTER))
    }

    fig = pu.new_figure()
    for k, text in enumerate(_HEADINGS):
        row, col = divmod(k, 4)
        pu.panel_heading(fig, 0.0 if col == 0 else _COL_X[col], _ROW_HEADING[row], text)

    _panel_basic_n(fig, frame, rhos[_BASIC_N])
    for index, (column, label) in enumerate(_SCATTER, start=1):
        _panel_scatter(fig, frame, index, column, label, rhos[column])
    _panel_ranking(fig, rhos)
    _legend(fig)
    pu.save_svg(fig, NAME)
