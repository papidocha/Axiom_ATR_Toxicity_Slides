"""Slide 8: the 23 existing patent compounds recommended for testing alongside tuvusertib instead of gartisertib.

A. general_tox against patent Ki bin for all 553 compounds with the two thresholds of the selection rule
   (``load_data.select_shortlist``). B. The slide-2 map with the selected compounds. C. The three top-ranked
   candidates and the two references, drawn at slide 2's scale. All 23 are in ``output/shortlist_23_compounds.csv``.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from rdkit import Chem

import load_data as D
import plot_utils as pu



NAME = "slide_08_shortlist"

_SELECTED = "#009E73"  #: the selected compounds, in every panel
_POINT = pu.SERIES

#: Category positions of panel A: weakest Ki bin on the left, no value last.
_NO_VALUE = "no value"
_POSITIONS: dict[str, int] = {"+": 0, "++": 1, "+++": 2, _NO_VALUE: 3}

#: Structures shown in panel C, in order: the top three of the ranking, then the references.
_CANDIDATES = ("Compound_I-O-27", "Compound_I-C-3", "Compound_I-O-29")
_REFERENCES = (D.TUVUSERTIB, D.GARTISERTIB)
_NAMES = {D.GARTISERTIB: "Gartisertib", D.TUVUSERTIB: "Tuvusertib"}
_COLOURS = {D.GARTISERTIB: pu.GARTISERTIB, D.TUVUSERTIB: pu.TUVUSERTIB}

#: Geometry, inches from the top-left of the canvas.
_PLOT = (0.65, 0.5, 5.7, 2.55)  # left, top, width, height
_RIGHT_X = 6.9
_MAP_W, _MAP_H = 2.6, 2.1
_TEXT_X = 9.7

_C_HEADING_Y = 3.72
_LABEL_TOP = 4.12
_STRUCT_TOP = 4.52
_STRUCT_W, _STRUCT_H = 2.3, 1.75
_CELL_PITCH = 2.45
_FIRST_CENTRE = 1.2

_BASIC_HALO = "#F6C28B"  #: halo behind a basic nitrogen


def _percentile(value: float) -> str:
    """``0.9764`` -> ``p98``; only the single safest compound reads ``p100``."""
    rounded = round(value * 100)
    return f"p{rounded if value >= 1.0 else min(rounded, 99)}"


def _minus(text: str) -> str:
    return text.replace("-", "−")


# --------------------------------------------------------------------------- #
# Numbers
# --------------------------------------------------------------------------- #


def slide_data() -> dict[str, object]:
    """Everything the slide draws or prints, from the selection rule."""
    table = D.candidate_table()
    selected = D.select_shortlist(table)
    cutoff, bar = D.thresholds(table)
    category = table["ki_bin"].fillna(_NO_VALUE)
    return {
        "table": table,
        "selected": selected,
        "cutoff": cutoff,
        "bar": bar,
        "category": category,
        "n_by_bin": category.value_counts().to_dict(),
        "selected_by_bin": selected["ki_bin"].value_counts().to_dict(),
    }


# --------------------------------------------------------------------------- #
# Panel A: predicted safety against the patent Ki bin
# --------------------------------------------------------------------------- #


def _panel_landscape(fig, data: dict[str, object]) -> None:
    table, selected = data["table"], data["selected"]
    category = data["category"]
    ax = pu.axes_in(fig, *_PLOT)
    jitter = np.random.default_rng(D.RANDOM_SEED).uniform(-0.3, 0.3, len(table))
    x = pd.Series(category.map(_POSITIONS).to_numpy() + jitter, index=table.index)
    y = table["general_tox"]

    ax.scatter(x, y, s=9, color=_POINT, alpha=0.45, linewidth=0, zorder=2)
    ax.scatter(x[selected.index], y[selected.index], s=26, color=_SELECTED, edgecolor=pu.INK, linewidth=0.5, zorder=4)

    for ki_bin, level in ((D.SECOND_BIN, data["bar"]), (D.TOP_BIN, data["cutoff"])):
        centre = _POSITIONS[ki_bin]
        ax.hlines(level, centre - 0.42, centre + 0.42, color=pu.INK, linewidth=1.3, linestyle="--", zorder=3)

    for compound, offset in ((D.GARTISERTIB, (9, -9)), (D.TUVUSERTIB, (-9, 9))):
        ax.scatter(x[compound], y[compound], s=70, color=_COLOURS[compound], edgecolor="white", linewidth=1.1, zorder=6)
        ax.annotate(
            _NAMES[compound],
            (x[compound], y[compound]),
            xytext=offset,
            textcoords="offset points",
            fontsize=pu.SIZE_SMALL,
            fontweight="bold",
            color=_COLOURS[compound],
            ha="left" if compound == D.GARTISERTIB else "right",
            va="center",
            zorder=7,
            bbox={"boxstyle": "round,pad=0.1", "facecolor": "white", "edgecolor": "none", "alpha": 0.85},
        )

    n_by_bin, selected_by_bin = data["n_by_bin"], data["selected_by_bin"]
    labels = []
    for name in _POSITIONS:
        definition = D.KI_RANGE.get(name)
        head = f"{name}  {definition}" if definition else name
        labels.append(f"{head}\nn = {n_by_bin.get(name, 0)}\n{selected_by_bin.get(name, 0)} selected")
    ax.set_xticks(list(_POSITIONS.values()))
    ax.set_xticklabels(labels)
    ax.set_xlim(-0.55, len(_POSITIONS) - 0.45)
    ax.set_ylim(-2.3, 2.75)
    ax.set_yticks([-2, -1, 0, 1, 2])
    ax.set_yticklabels(["−2", "−1", "0", "1", "2"])
    ax.set_ylabel("general_tox  (higher = safer)")
    ax.grid(axis="y")
    ax.grid(axis="x", visible=False)

    handles = [
        Line2D([], [], marker="o", linestyle="none", markersize=5, color=_POINT, label=f"All compounds (n = {len(table)})"),
        Line2D([], [], marker="o", linestyle="none", markersize=6, markerfacecolor=_SELECTED, markeredgecolor=pu.INK,
               label=f"Selected (n = {len(selected)})"),
        Line2D([], [], color=pu.INK, linestyle="--", linewidth=1.3, label="Threshold of that bin"),
    ]
    ax.legend(handles=handles, loc="upper left", ncols=2, fontsize=pu.SIZE_SMALL, columnspacing=1.0,
              handletextpad=0.3, borderaxespad=0.2)


# --------------------------------------------------------------------------- #
# Panel B: where the selected compounds sit in chemical space
# --------------------------------------------------------------------------- #


def _text(fig, x_in: float, y_in: float, text: str, colour: str = pu.INK) -> None:
    """Running text of panel B: one size (10 pt) throughout."""
    fig.text(
        x_in / pu.SLIDE_W_IN,
        1.0 - y_in / pu.SLIDE_H_IN,
        text,
        ha="left",
        va="top",
        fontsize=pu.SIZE_SMALL,
        color=colour,
        linespacing=1.5,
    )


def _panel_map(fig, data: dict[str, object]) -> None:
    layout = D.tsne_coordinates()
    selected = data["selected"]
    ax = pu.axes_in(fig, _RIGHT_X, 0.5, _MAP_W, _MAP_H)
    pu.apply_view(ax, layout)
    others = layout.index.difference(selected.index)
    ax.scatter(layout.loc[others, "tsne_x"], layout.loc[others, "tsne_y"], s=11, color="#C9C9C9", linewidth=0, zorder=2)
    ax.scatter(
        layout.loc[selected.index, "tsne_x"],
        layout.loc[selected.index, "tsne_y"],
        s=26,
        color=_SELECTED,
        edgecolor=pu.INK,
        linewidth=0.5,
        zorder=4,
    )
    for compound in _REFERENCES:
        ax.scatter(
            layout.at[compound, "tsne_x"],
            layout.at[compound, "tsne_y"],
            s=64,
            color=_COLOURS[compound],
            edgecolor="white",
            linewidth=1.1,
            zorder=6,
        )

    handles = [
        Line2D([], [], marker="o", linestyle="none", markersize=6, markerfacecolor=_SELECTED, markeredgecolor=pu.INK,
               label=f"Selected (n = {len(selected)})"),
        Line2D([], [], marker="o", linestyle="none", markersize=6, color="#C9C9C9", label="Other compounds"),
        Line2D([], [], marker="o", linestyle="none", markersize=7, color=pu.GARTISERTIB, label="Gartisertib"),
        Line2D([], [], marker="o", linestyle="none", markersize=7, color=pu.TUVUSERTIB, label="Tuvusertib"),
    ]
    fig.legend(
        handles=handles,
        loc="upper left",
        bbox_to_anchor=(_TEXT_X / pu.SLIDE_W_IN, 1.0 - 0.45 / pu.SLIDE_H_IN),
        fontsize=pu.SIZE_SMALL,
        labelspacing=0.45,
        handletextpad=0.3,
    )

    by_sub = selected["subseries"].value_counts()
    bar, cutoff = _minus(f"{data['bar']:+.2f}"), _minus(f"{data['cutoff']:+.2f}")
    _text(
        fig,
        _TEXT_X,
        1.55,
        f"++:  general_tox ≥ {bar} (tuvusertib)\n+++:  general_tox ≥ {cutoff} (p90)",
    )
    _text(
        fig,
        _TEXT_X,
        2.05,
        f"{len(selected)} compounds: {', '.join(f'{n} {name}' for name, n in by_sub.items())}\n"
        f"Basic N: 0 in all {len(selected)}\n"
        f"MW {selected['mw'].min():.0f}–{selected['mw'].max():.0f} Da",
    )
    _text(
        fig,
        _TEXT_X,
        2.72,
        "p = general_tox percentile among\nthe 553 compounds.",
        colour=pu.MUTED,
    )


# --------------------------------------------------------------------------- #
# Panel C: structures
# --------------------------------------------------------------------------- #


def _drawn(name: str, reference: Chem.Mol, mols: dict[str, Chem.Mol]) -> Chem.Mol:
    """A copy of the compound drawn in the orientation of the shared core."""
    mol = Chem.Mol(mols[name])
    pu.align_to(mol, reference)
    return mol


def _panel_structures(fig, data: dict[str, object]) -> list[str]:
    table, selected = data["table"], data["selected"]
    mols = D.load_molecules()
    reference = pu.reference_core()
    size_px = (int(_STRUCT_W * pu.PX_PER_IN), int(_STRUCT_H * pu.PX_PER_IN))

    pu.panel_heading(fig, 0.0, _C_HEADING_Y, "C.  Top candidates and the two references")
    extras: list[str] = []
    for index, name in enumerate((*_CANDIDATES, *_REFERENCES)):
        centre = _FIRST_CENTRE + index * _CELL_PITCH
        is_reference = name in _REFERENCES
        is_selected = name in selected.index
        record = table.loc[name]
        if is_reference:
            label = _NAMES[name]
            if is_selected:
                label = f"#{int(selected.at[name, 'rank'])}  {label}"
            colour = _COLOURS[name]
        else:
            label = f"#{int(selected.at[name, 'rank'])}  {name.replace('Compound_', '')}"
            colour = pu.INK
        detail = f"Ki {record['ki_bin']},  {_percentile(record['general_tox_percentile'])}"
        if not is_selected:
            detail += ",  not selected"
        for y_in, text, size, weight, text_colour in (
            (_LABEL_TOP, label, pu.SIZE_LABEL, "bold", colour),
            (_LABEL_TOP + 0.2, detail, pu.SIZE_SMALL, "normal", pu.INK),
        ):
            fig.text(
                centre / pu.SLIDE_W_IN,
                1.0 - y_in / pu.SLIDE_H_IN,
                text,
                ha="center",
                va="top",
                fontsize=size,
                fontweight=weight,
                color=text_colour,
            )
        mol = _drawn(name, reference, mols)
        halo = {m[0]: _BASIC_HALO for m in mol.GetSubstructMatches(D.basic_nitrogen_pattern())}
        svg = pu.draw_structure(
            mol,
            size_px=size_px,
            highlight_atoms=halo,
            bond_width=pu.BOND_WIDTH,
            font_px=pu.FONT_PX,
            bond_length_px=pu.BOND_LENGTH_PX,
        )
        extras.append(pu.place(svg, centre - _STRUCT_W / 2, _STRUCT_TOP, _STRUCT_W, f"structure_{index}"))

    # A rule between the candidates and the two references.
    divider_x = (_FIRST_CENTRE + 2.5 * _CELL_PITCH) / pu.SLIDE_W_IN
    fig.add_artist(
        Line2D(
            [divider_x, divider_x],
            [1.0 - 4.15 / pu.SLIDE_H_IN, 1.0 - 6.2 / pu.SLIDE_H_IN],
            color=pu.GRID,
            linewidth=1.0,
        )
    )
    return extras


# --------------------------------------------------------------------------- #
# Slide
# --------------------------------------------------------------------------- #


def slide_08_shortlist() -> None:
    data = slide_data()

    fig = pu.new_figure()
    pu.panel_heading(fig, 0.0, 0.1, "A.  Predicted safety by patent Ki bin")
    pu.panel_heading(fig, _RIGHT_X, 0.1, "B.  Selected compounds in chemical space")
    _panel_landscape(fig, data)
    _panel_map(fig, data)
    extras = _panel_structures(fig, data)
    pu.save_svg(fig, NAME, extras=extras)
