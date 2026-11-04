"""Slide 9: six unmade compounds predicted to be safe and potent (virtual screen and XGBoost, see ``design.py``).

A. general_tox against potency level. Potency is the predicted ordinal score (+ = -1, ++ = 0, +++ = +1), plotted on
   the axis of the bins (score + 2): made compounds sit on their bin, virtual compounds at their predicted score.
   The two safety thresholds of slide 8 are drawn.
B. The six structures, each with its predicted general_tox and potency score.
The tables are read from ``output/design`` (written by ``design.py``).
"""

from __future__ import annotations


import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from rdkit import Chem

import load_data as D
import plot_utils as pu



NAME = "slide_09_designs"
_CANVAS = (3.0, 2.8)  #: larger than D5's long tail, so no structure is shrunk below slide 2's bond length

_GREEN = pu.DESIGN_GREEN  #: slide 8's colour for the selected compounds
_DESIGN = pu.lighten(_GREEN, 0.3)  #: the six designs, in both panels (orange is kept for basic amines)
_POINT = pu.SERIES
_CANDIDATE = "#8C8C8C"

#: Geometry, inches from the top-left of the canvas.
_PLOT = (1.4, 0.55, 4.1, 4.75)  # left, top, width, height
_GRID_X = 5.95
_PITCH = 2.1
#: Each structure gets a canvas larger than any of the six, so RDKit never has to shrink
#: one to fit: every bond is exactly ``pu.BOND_LENGTH_PX`` (checked in ``bond_lengths``).
_STRUCT_W, _STRUCT_H = 2.3, 2.0
_ROW_TOP = (0.7, 3.3)  # label tops of the two rows
_STRUCT_OFFSET = 0.45  # structure top below the label top

#: Potency level of the y axis: ``+`` = 1, ``++`` = 2, ``+++`` = 3.
_LEVEL = {"+": 1, "++": 2, "+++": 3}

#: Label positions of panel A, in axes data units (x offset from the diamond, fixed y column).
_LABEL_X = 1.95


def _minus(text: str) -> str:
    return text.replace("-", "−")


# --------------------------------------------------------------------------- #
# Numbers
# --------------------------------------------------------------------------- #


def slide_data() -> dict[str, object]:
    """Everything the slide draws or prints."""
    table = D.candidate_table()
    cutoff, bar = D.thresholds(table)
    made = table[table["ki_bin"].notna()].copy()
    made["level"] = made["ki_bin"].map(_LEVEL)
    picks = pd.read_csv(D.DESIGN_DIR / "design_picks_xgb.csv")
    candidates = pd.read_csv(D.DESIGN_DIR / "design_candidates_xgb.csv")
    # The table stores the sulfoxide in its charge-separated form; draw it as S=O.
    picks["smiles"] = picks["smiles"].str.replace("[S+]([O-])", "S(=O)", regex=False)
    for frame in (picks, candidates):
        frame["level"] = frame["potency"] + 2  # ``+`` = 1, ``++`` = 2, ``+++`` = 3, as on the axis
    return {"table": table, "made": made, "picks": picks, "candidates": candidates, "cutoff": cutoff, "bar": bar}


# --------------------------------------------------------------------------- #
# Panel A: safety against potency
# --------------------------------------------------------------------------- #


def _panel_space(fig, data: dict[str, object], plot: tuple[float, float, float, float] = _PLOT) -> None:
    made, picks, candidates = data["made"], data["picks"], data["candidates"]
    table = data["table"]
    ax = pu.axes_in(fig, *plot)
    jitter = np.random.default_rng(D.RANDOM_SEED).normal(0, 0.035, len(made))

    ax.scatter(made["general_tox"], made["level"] + jitter, s=9, color=_POINT, alpha=0.45, linewidth=0, zorder=2)
    ax.scatter(candidates["tox"], candidates["level"], s=14, marker="D", facecolor="none",
               edgecolor=_CANDIDATE, linewidth=0.5, alpha=0.5, zorder=1)
    ax.axvline(data["cutoff"], color=pu.INK, linestyle="--", linewidth=1.1, zorder=3)
    ax.axvline(data["bar"], color=pu.TUVUSERTIB, linestyle=":", linewidth=1.3, zorder=3)

    for compound, colour, label, side in ((D.TUVUSERTIB, pu.TUVUSERTIB, "Tuvusertib", 1), (D.GARTISERTIB, pu.GARTISERTIB, "Gartisertib", -1)):
        x, y = table.at[compound, "general_tox"], _LEVEL[table.at[compound, "ki_bin"]]
        ax.scatter(x, y, s=70, color=colour, edgecolor="white", linewidth=1.1, zorder=6)
        ax.annotate(label, (x, y), xytext=(8 * side, -13 if side > 0 else 11), textcoords="offset points",
                    fontsize=pu.SIZE_SMALL, fontweight="bold", color=colour, ha="left" if side > 0 else "right",
                    va="center", zorder=7,
                    bbox={"boxstyle": "round,pad=0.1", "facecolor": "white", "edgecolor": "none", "alpha": 0.85})

    # The six diamonds nearly coincide: labels are fanned out in a column to the right.
    ordered = picks.sort_values("level", ascending=False).reset_index(drop=True)
    label_y = np.linspace(ordered["level"].max() + 0.18, ordered["level"].min() - 0.18, len(ordered))
    for (_, d), y_label in zip(ordered.iterrows(), label_y):
        ax.scatter(d["tox"], d["level"], s=60, marker="D", color=_DESIGN, edgecolor=pu.INK, linewidth=0.8, zorder=6)
        ax.annotate(d["design"], (d["tox"], d["level"]), xytext=(_LABEL_X, y_label), textcoords="data",
                    fontsize=pu.SIZE_SMALL, fontweight="bold", va="center", ha="left", zorder=7,
                    arrowprops={"arrowstyle": "-", "color": pu.MUTED, "linewidth": 0.6, "shrinkA": 1, "shrinkB": 3})

    ax.set_xlim(-2.4, 2.2)
    ax.set_ylim(0.75, 3.35)
    ax.set_yticks([1, 2, 3])
    ax.set_yticklabels([f"{k}\n{D.KI_RANGE[k]}" for k in ("+", "++", "+++")])
    ax.set_xticks([-2, -1, 0, 1, 2])
    ax.set_xticklabels(["−2", "−1", "0", "1", "2"])
    ax.set_xlabel("general_tox  (higher = safer)")
    ax.set_ylabel("Potency")
    ax.grid(True)

    ax.text(data["cutoff"] - 0.04, 0.82, f"p90  {_minus(f'{data['cutoff']:+.2f}')}", fontsize=pu.SIZE_SMALL,
            color=pu.INK, ha="right", va="bottom")
    ax.text(data["bar"] + 0.04, 0.82, f"tuvusertib  {_minus(f'{data['bar']:+.2f}')}", fontsize=pu.SIZE_SMALL,
            color=pu.TUVUSERTIB, ha="left", va="bottom")

    handles = [
        Line2D([], [], marker="o", linestyle="none", markersize=5, color=_POINT, label=f"Patent compounds (n = {len(made)})"),
        Line2D([], [], marker="D", linestyle="none", markersize=5, markerfacecolor="none", markeredgecolor=_CANDIDATE,
               label=f"Virtual screens (n = {len(candidates):,})"),
        Line2D([], [], marker="D", linestyle="none", markersize=6, markerfacecolor=_DESIGN, markeredgecolor=pu.INK,
               label="Selected designs (n = 6)"),
    ]
    ax.legend(handles=handles, loc="upper left", fontsize=pu.SIZE_SMALL, labelspacing=0.3, handletextpad=0.3,
              borderaxespad=0.2)


def _text(fig, x_in: float, y_in: float, text: str, colour: str = pu.INK, size: float = pu.SIZE_SMALL) -> None:
    """Running text: 10 pt, absolute position."""
    fig.text(x_in / pu.SLIDE_W_IN, 1.0 - y_in / pu.SLIDE_H_IN, text, ha="left", va="top",
             fontsize=size, color=colour, linespacing=1.5)


# --------------------------------------------------------------------------- #
# Panel B: the six structures
# --------------------------------------------------------------------------- #


def _core_offset_in(mol: Chem.Mol, reference: Chem.Mol) -> np.ndarray:
    """Where the core's first atom lies relative to the drawing's centre, in inches (x right, y down)."""
    conformer = mol.GetConformer()
    xy = np.array([[conformer.GetAtomPosition(i).x, conformer.GetAtomPosition(i).y] for i in range(mol.GetNumAtoms())])
    bond = mol.GetBondWithIdx(0)
    unit = np.linalg.norm(xy[bond.GetBeginAtomIdx()] - xy[bond.GetEndAtomIdx()])
    centre = (xy.min(axis=0) + xy.max(axis=0)) / 2
    offset = (xy[mol.GetSubstructMatch(reference)[0]] - centre) * (pu.BOND_LENGTH_PX / unit) / pu.PX_PER_IN
    return offset * np.array([1.0, -1.0])


def _panel_structures(fig, data: dict[str, object]) -> list[str]:
    picks = data["picks"]
    reference = pu.reference_core()
    canvas_w, canvas_h = _CANVAS
    size_px = (int(canvas_w * pu.PX_PER_IN), int(canvas_h * pu.PX_PER_IN))
    extras: list[str] = []
    mols = {}
    for _, d in picks.iterrows():
        mol = Chem.MolFromSmiles(d["smiles"])
        pu.align_to(mol, reference)
        mols[d["design"]] = mol
    # RDKit centres each drawing on its own bounding box, so a longer tail would shift the
    # core. Each drawing is moved back so the shared core sits at the same place in every cell.
    shifts = {name: _core_offset_in(mol, reference) for name, mol in mols.items()}
    mean_shift = np.mean(list(shifts.values()), axis=0)
    for index, d in picks.iterrows():
        row, col = divmod(index, 3)
        x = _GRID_X + col * _PITCH
        top = _ROW_TOP[row]
        lines = (
            (d["design"], pu.SIZE_LABEL, "bold", pu.INK),
            (f"general_tox  {_minus(f'{d['tox']:+.2f}')}", pu.SIZE_SMALL, "normal", pu.INK),
            (f"potency  {_minus(f'{d['potency']:+.2f}')}", pu.SIZE_SMALL, "normal", pu.INK),
        )
        offsets = (0.0, 0.25, 0.43)
        for (text, size, weight, colour), dy in zip(lines, offsets):
            fig.text(x / pu.SLIDE_W_IN, 1.0 - (top + dy) / pu.SLIDE_H_IN, text, ha="left", va="top",
                     fontsize=size, fontweight=weight, color=colour)
        mol = mols[d["design"]]
        shared = set(mol.GetSubstructMatch(pu.DESIGN_SCAFFOLD))
        # Halos only behind drawn letters (N, O, S); carbons are traced by their bonds alone.
        atoms = {a.GetIdx(): pu.DESIGN_HALO for a in mol.GetAtoms()
                 if a.GetIdx() not in shared and a.GetAtomicNum() != 6}
        bonds = {b.GetIdx(): pu.DESIGN_HALO for b in mol.GetBonds()
                 if b.GetBeginAtomIdx() not in shared or b.GetEndAtomIdx() not in shared}
        svg = pu.draw_structure(mol, size_px=size_px, highlight_atoms=atoms, highlight_bonds=bonds,
                                      bond_width=pu.BOND_WIDTH, font_px=pu.FONT_PX, bond_length_px=pu.BOND_LENGTH_PX)
        dx, dy = mean_shift - shifts[d["design"]]
        extras.append(pu.place(svg, x + 1.0 - canvas_w / 2 + dx, top + _STRUCT_OFFSET + (_STRUCT_H - canvas_h) / 2 + dy, canvas_w,
                                     f"structure_{d['design']}"))
    return extras


# --------------------------------------------------------------------------- #
# Slide
# --------------------------------------------------------------------------- #


def slide_09_designs() -> None:
    data = slide_data()

    fig = pu.new_figure()
    pu.panel_heading(fig, 0.0, 0.1, "A.  Safety and potency, predicted")
    pu.panel_heading(fig, _GRID_X, 0.1, "B.  Proposed designs")
    _panel_space(fig, data)
    extras = _panel_structures(fig, data)

    _text(
        fig,
        0.0,
        5.85,
        "Virtual screens scored by XGBoost on Morgan fingerprints,\n"
        "trained on the patent compounds. Potency score: + = −1, ++ = 0, +++ = +1.",
        colour=pu.MUTED,
    )
    pu.save_svg(fig, NAME, extras=extras)
