"""Slide 1: gartisertib through the DMTA pipeline (design, make, test), then a clinical failure."""

from __future__ import annotations

from matplotlib.patches import Polygon
from rdkit import Chem

import load_data as D
import plot_utils as pu

NAME = "slide_01_title"

_PIPE_X, _PITCH, _STEP_W, _CY = 0.0, 1.75, 1.6, 3.74  # chevron bottom edge level with the bottom of the structure
_STRUCTURE_X, _STRUCTURE_W = 7.3, 5.0
_MERCK_STEP = pu.lighten(pu.AXIOM, 0.5)  # Merck's make and test: Axiom's green, lighter, until Phase I contradicts them
_PIPERAZINE = "N1CCNCC1"  # the part of gartisertib's R2 tail that is marked


def chevron(ax, x: float, cy: float, label: str, *, fill: str, colour: str, w: float, h: float = 0.62, notch: float = 0.22) -> None:
    """A pipeline step: a flat chevron pointing right; ``(x, cy)`` is its left edge and vertical centre."""
    top, bottom = cy - h / 2, cy + h / 2
    ax.add_patch(Polygon([(x, top), (x + w - notch, top), (x + w, cy), (x + w - notch, bottom), (x, bottom), (x + notch, cy)],
                         closed=True, facecolor=fill, edgecolor="none", zorder=3))
    pu.text(ax, x + w / 2 + notch / 4, cy, label, weight="bold", colour=colour, ha="center", va="center_baseline")


def structure(smiles: str, x: float, y: float, w: float, h: float, group: str, colour: str, scale: float, highlight: str) -> str:
    """The structure in the core orientation of slide 2, with the SMARTS ``highlight`` marked: a pale band behind its
    bonds and a round halo behind its basic nitrogen (not the amide one), plus the bond that nitrogen's third
    substituent hangs on. ``scale`` enlarges bonds and labels together."""
    mol, _ = pu.aligned(smiles, pu.reference_core())
    marked = set(mol.GetSubstructMatch(Chem.MolFromSmarts(highlight)))
    amide_n = {i for i in marked if any(n.GetAtomicNum() == 6 and any(b.GetBondType() == Chem.BondType.DOUBLE
               and b.GetOtherAtom(n).GetAtomicNum() == 8 for b in n.GetBonds()) for n in mol.GetAtomWithIdx(i).GetNeighbors())}
    basic_n = {i for i in marked if mol.GetAtomWithIdx(i).GetAtomicNum() == 7} - amide_n
    tint = pu.lighten(colour, 0.6)
    svg = pu.draw_structure(
        mol,
        size_px=(int(w * pu.PX_PER_IN), int(h * pu.PX_PER_IN)),
        bond_width=pu.BOND_WIDTH * scale,
        font_px=round(pu.FONT_PX * scale),
        bond_length_px=pu.BOND_LENGTH_PX * scale,
        highlight_atoms={i: tint for i in basic_n},
        highlight_bonds={b.GetIdx(): tint for b in mol.GetBonds()
                         if (b.GetBeginAtomIdx() in marked and b.GetEndAtomIdx() in marked)
                         or b.GetBeginAtomIdx() in basic_n or b.GetEndAtomIdx() in basic_n},
        highlight_bond_width=3.0,
    )
    return pu.place(svg, x, y, w, group)


def slide_01_title() -> None:
    fig = pu.new_figure()
    ax = pu.overlay(fig)

    pu.text(ax, _PIPE_X, _CY - 0.68, "Merck KGaA's ATR inhibitor program, oncology", colour=pu.MUTED)
    pu.text(ax, _STRUCTURE_X + _STRUCTURE_W / 2, 0.55, "Gartisertib (M4344)", size=pu.SIZE_HEADING, weight="bold",
            colour=pu.FAIL, ha="center")
    steps = ("Design", "Make", "Test", "Phase I")
    for i, step in enumerate(steps):
        last, first = i == len(steps) - 1, i == 0
        chevron(ax, _PIPE_X + i * _PITCH, _CY, step, fill=pu.FAIL if last else (pu.AXIOM if first else _MERCK_STEP),
                colour="white" if last or first else pu.INK, w=_STEP_W)
    pu.text(ax, _PIPE_X + 3 * _PITCH + _STEP_W / 2, _CY + 0.55, "Clinical failure", weight="bold", colour=pu.FAIL, ha="center")

    ax.plot([0, 12.33], [5.2, 5.2], color=pu.AXIOM, linewidth=1.4)

    smiles = D.REFERENCE_COMPOUNDS[D.GARTISERTIB]["smiles"]
    extras = [structure(smiles, _STRUCTURE_X, 0.5, _STRUCTURE_W, 4.0, "gartisertib", pu.FAIL, 1.9, _PIPERAZINE)]
    pu.save_svg(fig, NAME, extras=extras)

