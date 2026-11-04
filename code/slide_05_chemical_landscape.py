"""Slide 5: the chemical-space map colored four ways.

The 553 compounds sit at the t-SNE coordinates of slide 2. A. general_tox, B. patent Ki bin, C. basic nitrogen count,
D. molecular weight. A miniature of the sub-series map shows where each sub-series lies.
"""

from __future__ import annotations

import pandas as pd
from matplotlib.colors import Normalize
from matplotlib.lines import Line2D

import load_data as D
import plot_utils as pu


NAME = "slide_05_chemical_landscape"

#: Patent Ki bins drawn least potent first, so the potent compounds end up on top.
_KI_ORDER: tuple[str, ...] = ("NA", "+", "++", "+++")
_KI_COLORS: dict[str, str] = {"+++": "#08519C", "++": "#4292C6", "+": "#9ECAE1", "NA": "#CFCFCF"}
_KI_LABELS: dict[str, str] = {
    "+++": "+++   < 0.01 µM",
    "++": "++   0.01–1 µM",
    "+": "+   1–5 µM",
    "NA": "no value",
}

#: Basic-nitrogen colour scale: viridis (0 = purple, 2 = yellow) with the whole gradient
#: spent between 0 and 2, where the data end.
_BASIC_N_CMAP = "viridis"
_BASIC_N_TOP = 2

#: Panel geometry, inches. All five maps (four panels and the miniature) share one size,
#: with the aspect of slide 2's map, so the layout looks the same everywhere.
_PANEL_W, _PANEL_H = 3.09, 2.5
_COL_X = (0.0, 4.6)
_ROW_Y = (0.55, 3.65)
_MINI_X = 12.33 - _PANEL_W

#: Offsets (points) of the labels on the miniature.
_MINI_OFFSETS: dict[str, tuple[float, float]] = {
    "Compound_I-G-32": (-2, -17),
    "Compound_I-C-79": (10, 8),
}
_LABELS = {name: meta["label"] for name, meta in D.REFERENCE_COMPOUNDS.items()}


# --------------------------------------------------------------------------- #
# Numbers
# --------------------------------------------------------------------------- #


def landscape_frame() -> pd.DataFrame:
    """One row per compound: t-SNE position and every variable that is drawn."""
    series = D.load_lead_series()
    layout = D.tsne_coordinates()
    descriptors = D.load_descriptors()
    joined = D.join_patent_bins().set_index("name")

    data = series.data.set_index("name").loc[layout.index]
    frame = layout.copy()
    frame["general_tox"] = data["general_tox"]
    frame["subseries"] = data["subseries"].astype(str)
    frame["ki_bin"] = joined["ki_bin"].reindex(frame.index).fillna("NA")
    frame["basic_n"] = descriptors["n_basic_nitrogen"].reindex(frame.index).astype(int)
    frame["mw"] = descriptors["mw"].reindex(frame.index)
    return frame


# --------------------------------------------------------------------------- #
# Panels
# --------------------------------------------------------------------------- #


def _ring_references(ax, frame: pd.DataFrame) -> None:
    """Ring the two reference compounds (named on the miniature)."""
    for name in _MINI_OFFSETS:
        ax.scatter(
            frame.at[name, "tsne_x"],
            frame.at[name, "tsne_y"],
            s=70,
            facecolor="none",
            edgecolor=pu.INK,
            linewidth=1.4,
            zorder=5,
        )


def _map(fig, frame: pd.DataFrame, x: float, y: float, order=None):
    """Axes with the shared t-SNE view; returns ``(ax, ordered frame)``."""
    ax = pu.axes_in(fig, x, y, _PANEL_W, _PANEL_H)
    pu.apply_view(ax, frame)
    return ax, (frame if order is None else frame.sort_values(order, kind="stable"))


def _colorbar(fig, handle, x: float, y: float, label: str, ticks=None, ticklabels=None, extend="neither"):
    cax = pu.axes_in(fig, x + _PANEL_W + 0.14, y + 0.3, 0.14, _PANEL_H - 0.6)
    bar = fig.colorbar(handle, cax=cax, extend=extend)
    # matplotlib rasterises colourbars with many levels; keep the gradient as vector paths.
    bar.solids.set_rasterized(False)
    bar.solids.set_edgecolor("face")
    bar.set_label(label, fontsize=pu.SIZE_SMALL)
    bar.ax.tick_params(labelsize=pu.SIZE_SMALL, length=2.5)
    bar.outline.set_visible(False)
    if ticks is not None:
        bar.set_ticks(ticks)
    if ticklabels is not None:
        bar.set_ticklabels(ticklabels)
    return bar


def _scatter(ax, frame: pd.DataFrame, colours, cmap=None, norm=None):
    return ax.scatter(
        frame["tsne_x"], frame["tsne_y"], c=colours, cmap=cmap, norm=norm, s=15, linewidth=0, alpha=0.92, zorder=2
    )


def _panel_general_tox(fig, frame: pd.DataFrame) -> None:
    x, y = _COL_X[0], _ROW_Y[0]
    pu.panel_heading(fig, x, y - 0.45, "A.  general_tox")
    ax, _ = _map(fig, frame, x, y)
    handle = _scatter(ax, frame, frame["general_tox"], cmap="viridis")
    _ring_references(ax, frame)
    _colorbar(fig, handle, x, y, "general_tox (higher = safer)")


def _panel_ki_bin(fig, frame: pd.DataFrame) -> None:
    x, y = _COL_X[1], _ROW_Y[0]
    pu.panel_heading(fig, x, y - 0.45, "B.  Patent Ki bin")
    ax, _ = _map(fig, frame, x, y)
    for key in _KI_ORDER:
        part = frame[frame["ki_bin"] == key]
        ax.scatter(part["tsne_x"], part["tsne_y"], s=15, color=_KI_COLORS[key], linewidth=0, alpha=0.95, zorder=2)
    _ring_references(ax, frame)
    handles = [
        Line2D([], [], marker="o", linestyle="none", markersize=6, color=_KI_COLORS[k], label=_KI_LABELS[k])
        for k in ("+++", "++", "+", "NA")
    ]
    fig.legend(
        handles=handles,
        loc="upper left",
        bbox_to_anchor=((x + _PANEL_W + 0.1) / pu.SLIDE_W_IN, 1.0 - (y + 0.3) / pu.SLIDE_H_IN),
        fontsize=pu.SIZE_SMALL,
        labelspacing=0.5,
        handletextpad=0.3,
        frameon=False,
    )


def _panel_basic_n(fig, frame: pd.DataFrame) -> None:
    x, y = _COL_X[0], _ROW_Y[1]
    pu.panel_heading(fig, x, y - 0.45, "C.  Basic nitrogen count")
    ax, ordered = _map(fig, frame, x, y, order="basic_n")
    norm = Normalize(vmin=0, vmax=_BASIC_N_TOP, clip=True)
    handle = _scatter(ax, ordered, ordered["basic_n"], cmap=_BASIC_N_CMAP, norm=norm)
    _ring_references(ax, frame)
    _colorbar(
        fig,
        handle,
        x,
        y,
        "Basic N",
        ticks=[0, 1, 2],
        ticklabels=["0", "1", str(_BASIC_N_TOP)],
    )


def _panel_mw(fig, frame: pd.DataFrame) -> None:
    x, y = _COL_X[1], _ROW_Y[1]
    pu.panel_heading(fig, x, y - 0.45, "D.  Molecular weight")
    ax, _ = _map(fig, frame, x, y)
    handle = _scatter(ax, frame, frame["mw"], cmap="viridis")
    _ring_references(ax, frame)
    _colorbar(fig, handle, x, y, "MW (Da)")


def _panel_miniature(fig, frame: pd.DataFrame) -> None:
    """The sub-series map of slide 2 in miniature, with the reference compounds named."""
    x, y = _MINI_X, _ROW_Y[0]
    fig.text(
        x / pu.SLIDE_W_IN,
        1.0 - (y - 0.25) / pu.SLIDE_H_IN,
        "Sub-series",
        ha="left",
        va="bottom",
        fontsize=pu.SIZE_LABEL,
        fontweight="bold",
        color=pu.INK,
    )
    ax, _ = _map(fig, frame, x, y)
    for name in D.SUBSERIES:
        part = frame[frame["subseries"] == name]
        ax.scatter(part["tsne_x"], part["tsne_y"], s=11, color=pu.SUBSERIES_COLORS[name], linewidth=0, alpha=0.92, zorder=2)
    for compound, offset in _MINI_OFFSETS.items():
        px, py = frame.loc[compound, ["tsne_x", "tsne_y"]]
        ax.scatter(
            px,
            py,
            s=55,
            color=pu.SUBSERIES_COLORS[frame.at[compound, "subseries"]],
            edgecolor=pu.INK,
            linewidth=1.3,
            zorder=4,
        )
        ax.annotate(
            _LABELS[compound],
            (px, py),
            xytext=offset,
            textcoords="offset points",
            fontsize=pu.SIZE_SMALL,
            fontweight="bold",
            color=pu.INK,
            ha="center",
            va="center",
            zorder=5,
            arrowprops={"arrowstyle": "-", "color": pu.INK, "linewidth": 0.7, "shrinkA": 0, "shrinkB": 4},
            bbox={"boxstyle": "round,pad=0.1", "facecolor": "white", "edgecolor": "none", "alpha": 0.85},
        )
    counts = frame["subseries"].value_counts()
    fig.legend(
        handles=[
            Line2D(
                [],
                [],
                marker="o",
                linestyle="none",
                markersize=6,
                color=pu.SUBSERIES_COLORS[name],
                label=f"{name}  (n = {counts[name]})",
            )
            for name in D.SUBSERIES
        ],
        loc="upper left",
        bbox_to_anchor=(x / pu.SLIDE_W_IN, 1.0 - (y + _PANEL_H + 0.08) / pu.SLIDE_H_IN),
        ncols=2,
        fontsize=pu.SIZE_SMALL,
        columnspacing=1.0,
        handletextpad=0.2,
        frameon=False,
    )
    pu.footnote(
        fig,
        x,
        y + _PANEL_H + 0.8,
        "Rings in A to D mark gartisertib and tuvusertib.\n"
        "t-SNE of Tanimoto distance, Morgan\n"
        "fingerprints (radius 2, 2048 bits)",
    )


# --------------------------------------------------------------------------- #
# Slide
# --------------------------------------------------------------------------- #


def slide_05_chemical_landscape() -> None:
    frame = landscape_frame()

    fig = pu.new_figure()
    _panel_general_tox(fig, frame)
    _panel_ki_bin(fig, frame)
    _panel_basic_n(fig, frame)
    _panel_mw(fig, frame)
    _panel_miniature(fig, frame)
    pu.save_svg(fig, NAME)
