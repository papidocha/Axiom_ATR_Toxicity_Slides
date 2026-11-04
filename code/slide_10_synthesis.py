"""Slide 10: how three of the designs (D2, D4, D6 of slide 9) would be made.

Every design is one amide: the acid component common to all (A) coupled to a 3-aminopyridine that carries the tail
(B). Everything up to the aminopyridine is chemistry documented in the patent US 9,340,546 B2 (blue, with page
numbers of the PDF); the final coupling with this acid and this amine is the one new step (green). The three designs
stand for the three ways the tail is attached: C-O (ether), C-N and C-C (Suzuki).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem import rdDepictor

import load_data as D
import plot_utils as pu


NAME = "slide_10_synthesis"

_PATENT = pu.SERIES  #: documented in the patent
_NEW = "#009E73"  #: new: slide 8's selected-compound green, as for the designs on slide 9

_W, _H = pu.SLIDE_W_IN, pu.SLIDE_H_IN
_INCH_PER_PX = 1.0 / pu.PX_PER_IN
_LABEL_X = 0.0
_ROW_X = 0.55  #: where each row's first structure starts
_GAP = 0.1  #: between a structure and its neighbour (plus sign or arrow)
_PLUS_W = 0.34
_BLUE_ARROW = 2.3  #: every patent arrow has this length, and every new arrow the next one
_GREEN_ARROW = 1.4
_LINE = 0.175  #: line pitch of the text on an arrow, in inches
_ACID_Y = 0.92
_HEADING_B_Y = 2.14  #: the heading of panel B sits 0.5 in above this
_BOXES_TOP = 1.86  #: top of the first green box
_BOX_GAP = 0.06
_LEGEND_X = 8.7
_LEGEND_Y = (0.55, 0.8)  #: top right, beside the precursor synthesis

# --------------------------------------------------------------------------- #
# Chemistry: structures, reagents and the pages of US 9,340,546 B2 they come from
# --------------------------------------------------------------------------- #

_ACID_ESTER = "C=CCOC(=O)c1c(N)n[nH]c1N"
_NITRILE = "N#CCC(C(OC)OC)C(OC)OC"
_ACTIVE_ESTER = "N#CCc1cnc2c(C(=O)On3nnc4ccccc43)c(N)nn2c1"
_CHLORONITROPYRIDINE = "[O-][N+](=O)c1cnccc1Cl"
_ACID_STEPS = (("a) PTSA, DMSO/H2O, 85 °C", "b) PhSiH3, Pd(PPh3)4, DCM, RT"), ("c) TBTU, Et3N, DCM, RT",))
_ACID_PAGES = "p. 103–104"
_COUPLING = (("6c",), ("NMP, 100 °C",))
_COUPLING_PAGE = "p. 104"

#: tail (R2) -> (starting material, aminopyridine, conditions above the arrow, conditions below it, patent pages).
#: The thioether of D2 is oxidised (Prep. O-4) before the nitro group is reduced.
_ROUTES = {
    "O=S1(=O)CCC(O[*:2])CC1": (
        "OC1CCSCC1", "Nc1cnccc1OC1CCS(=O)(=O)CC1",
        ("a) NaH, THF, 0 °C → RT", "b) mCPBA, DCM, RT"), ("c) H2, Pd/C, MeOH, RT",), "p. 182, 188"),
    "COCCS(=O)(=O)C1CCN([*:2])CC1": (
        "COCCS(=O)(=O)C1CCNCC1", "Nc1cnccc1N1CCC(S(=O)(=O)CCOC)CC1",
        ("a) DIPEA, dioxane, 80 °C",), ("b) H2, Pd/C, MeOH, RT",), "p. 111"),
    "C1CC([*:2])CO1": (
        "CC1(C)OB(OC1(C)C)C1=CCOC1", "Nc1cnccc1C1CCOC1",
        ("a) Pd2(dba)3, PPh3, Na2CO3,", "toluene/EtOH, 75 °C"), ("b) H2, Pd/C, MeOH, RT",), "p. 198"),
}


def _flat(smiles: str) -> str:
    return Chem.MolToSmiles(Chem.MolFromSmiles(smiles), isomericSmiles=False)


# --------------------------------------------------------------------------- #
# Drawing
# --------------------------------------------------------------------------- #


def _oriented(smiles: str, left: str | None = None) -> Chem.Mol:
    """A molecule drawn as flat as possible and, if ``left`` is given, with that part on the left."""
    mol = Chem.MolFromSmiles(smiles)
    rdDepictor.SetPreferCoordGen(True)
    rdDepictor.Compute2DCoords(mol)
    conformer = mol.GetConformer()
    xy = np.array([[conformer.GetAtomPosition(i).x, conformer.GetAtomPosition(i).y] for i in range(mol.GetNumAtoms())])
    centred = xy - xy.mean(axis=0)
    # The flattest orientation: the turn (in 2 degree steps) that gives the drawing the least height.
    heights = []
    for degrees in range(0, 180, 2):
        angle = np.radians(degrees)
        turn = np.array([[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]])
        heights.append(np.ptp((centred @ turn.T)[:, 1]))
    angle = np.radians(2 * int(np.argmin(heights)))
    rotation = np.array([[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]])
    turned = centred @ rotation.T
    if left is not None:
        part = list(mol.GetSubstructMatch(Chem.MolFromSmarts(left)))
        if part and turned[part, 0].mean() > 0:
            turned[:, 0] *= -1
    for i, (x, y) in enumerate(turned):
        conformer.SetAtomPosition(i, (float(x), float(y), 0.0))
    return mol


def _align_to_pattern(smiles: str, reference: Chem.Mol, pattern: Chem.Mol) -> Chem.Mol:
    """A molecule drawn with the part it shares with ``reference`` (``pattern``) in the same position."""
    mol = Chem.MolFromSmiles(smiles)
    rdDepictor.SetPreferCoordGen(True)
    rdDepictor.GenerateDepictionMatching2DStructure(mol, reference, refPatt=pattern)
    return mol


def _extent_in(mol: Chem.Mol) -> tuple[float, float]:
    """Width and height of the drawn atoms in inches at the slide's bond length."""
    conformer = mol.GetConformer()
    xy = np.array([[conformer.GetAtomPosition(i).x, conformer.GetAtomPosition(i).y] for i in range(mol.GetNumAtoms())])
    bond = mol.GetBondWithIdx(0)
    unit = np.linalg.norm(xy[bond.GetBeginAtomIdx()] - xy[bond.GetEndAtomIdx()])
    scale = pu.BOND_LENGTH_PX / unit * _INCH_PER_PX
    width, height = (xy.max(axis=0) - xy.min(axis=0)) * scale
    return float(width) + 0.2, float(height) + 0.2  # room for atom labels


def _tail_highlight(mol: Chem.Mol) -> tuple[dict[int, str], dict[int, str]]:
    """Slide 9's highlight: the atoms and bonds beyond the shared scaffold, halos only behind lettered atoms."""
    shared = set(mol.GetSubstructMatch(pu.DESIGN_SCAFFOLD))
    atoms = {a.GetIdx(): pu.DESIGN_HALO for a in mol.GetAtoms() if a.GetIdx() not in shared and a.GetAtomicNum() != 6}
    bonds = {b.GetIdx(): pu.DESIGN_HALO for b in mol.GetBonds()
             if b.GetBeginAtomIdx() not in shared or b.GetEndAtomIdx() not in shared}
    return atoms, bonds


class _Row:
    """A line of structures, plus signs and arrows, laid out left to right on one centre line."""

    def __init__(self, ax, extras: list[str], y: float, x: float = _ROW_X) -> None:
        self.ax, self.extras, self.y, self.x = ax, extras, y, x

    def molecule(self, smiles: str | Chem.Mol, name: str, left: str | None = None,
                 tail: bool = False) -> tuple[float, float, float]:
        mol = smiles if isinstance(smiles, Chem.Mol) else _oriented(smiles, left)
        width, height = _extent_in(mol)
        canvas_w, canvas_h = width + 0.8, height + 0.8  # larger than the molecule: RDKit never shrinks it
        px = (int(canvas_w * pu.PX_PER_IN), int(canvas_h * pu.PX_PER_IN))
        atoms, bonds = _tail_highlight(mol) if tail else ({}, {})
        svg = pu.draw_structure(mol, size_px=px, highlight_atoms=atoms, highlight_bonds=bonds,
                                      bond_width=pu.BOND_WIDTH, font_px=pu.FONT_PX, bond_length_px=pu.BOND_LENGTH_PX)
        centre = self.x + width / 2
        canvas_w, canvas_h = px[0] * _INCH_PER_PX, px[1] * _INCH_PER_PX
        self.extras.append(pu.place(svg, centre - canvas_w / 2, self.y - canvas_h / 2, canvas_w, name))
        left_edge = self.x
        self.x += width + _GAP
        return left_edge, width, height

    def plus(self) -> None:
        self.ax.text(self.x + _PLUS_W / 2, self.y, "+", ha="center", va="center", fontsize=pu.SIZE_LABEL, color=pu.INK)
        self.x += _PLUS_W + _GAP

    def arrow(self, above: tuple[str, ...], below: tuple[str, ...], page: str, colour: str) -> None:
        length = _BLUE_ARROW if colour == _PATENT else _GREEN_ARROW
        self.ax.annotate("", xy=(self.x + length, self.y), xytext=(self.x, self.y),
                         arrowprops={"arrowstyle": "-|>", "linewidth": 1.6, "color": colour, "shrinkA": 0, "shrinkB": 0})
        centre = self.x + length / 2
        for k, line in enumerate(reversed(above)):
            self.ax.text(centre, self.y - 0.07 - k * _LINE, line, ha="center", va="bottom", fontsize=pu.SIZE_SMALL, color=colour)
        for k, line in enumerate(below):
            self.ax.text(centre, self.y + 0.08 + k * _LINE, line, ha="center", va="top", fontsize=pu.SIZE_SMALL, color=colour)
        self.ax.text(centre, self.y + 0.08 + len(below) * _LINE, page, ha="center", va="top", fontsize=pu.SIZE_SMALL, color=pu.MUTED)
        self.x += length + _GAP


# --------------------------------------------------------------------------- #
# Slide
# --------------------------------------------------------------------------- #


def slide_10_synthesis() -> None:
    picks = pd.read_csv(D.DESIGN_DIR / "design_picks_xgb.csv")
    picks["flat"] = [_flat(s.replace("[S+]([O-])", "S(=O)")) for s in picks["smiles"]]
    design_of = dict(zip(picks["R2"], picks["design"]))
    product_of = dict(zip(picks["R2"], picks["flat"]))

    fig = pu.new_figure()
    ax = pu.axes_in(fig, 0, 0, _W, _H)
    ax.set_xlim(0, _W)
    ax.set_ylim(_H, 0)
    ax.axis("off")
    pu.panel_heading(fig, 0.0, 0.1, "A.  Precursor synthesis in patent")
    pu.panel_heading(fig, 0.0, _HEADING_B_Y - 0.5, "B.  Routes to three designs")

    extras: list[str] = []

    row = _Row(ax, extras, _ACID_Y)
    row.molecule(_ACID_ESTER, "acid_pyrazole")
    row.plus()
    row.molecule(_NITRILE, "acid_nitrile")
    row.arrow(_ACID_STEPS[0], _ACID_STEPS[1], _ACID_PAGES, _PATENT)
    left_edge, width, height = row.molecule(_ACTIVE_ESTER, "active_ester_6c", left="Nc1nn2cc(CC#N)cnc2c1")
    ax.text(left_edge + width / 2, _ACID_Y + height / 2, "6c", ha="center", va="top", fontsize=pu.SIZE_LABEL, fontweight="bold", color=pu.INK)

    product_left = "Nc1nn2cc(CC#N)cnc2c1"
    # Every product shares the core of the first, and every aminopyridine the pyridine of the first, so
    # the three rows can be compared by eye.
    first = next(iter(_ROUTES))
    product_ref = _oriented(Chem.MolToSmiles(Chem.MolFromSmiles(product_of[first])), product_left)
    amine_ref = _oriented(_ROUTES[first][1], "[NH2]c1cnccc1")
    core = Chem.MolFromSmiles("Nc1nn2cc(CC#N)cnc2c1C(=O)Nc1cnccc1")
    pyridine = Chem.MolFromSmiles("Nc1cnccc1")
    products = {r2: product_ref if r2 == first else _align_to_pattern(Chem.MolToSmiles(Chem.MolFromSmiles(product_of[r2])), product_ref, core)
                for r2 in _ROUTES}
    amines = {r2: amine_ref if r2 == first else _align_to_pattern(_ROUTES[r2][1], amine_ref, pyridine) for r2 in _ROUTES}
    sizes = {r2: _extent_in(products[r2]) for r2 in _ROUTES}
    # Columns shared by the rows (the arrows have fixed lengths, so the columns follow from them): the first
    # patent arrow starts after the widest starting-material cluster, the new arrow after the widest aminopyridine.
    cluster_end = max(
        _ROW_X + _extent_in(_oriented(sm))[0] + _GAP + _PLUS_W + _GAP + _extent_in(_oriented(_CHLORONITROPYRIDINE))[0] + _GAP
        for sm, *_ in _ROUTES.values()
    )
    amine_x = cluster_end + _BLUE_ARROW + _GAP
    top = _BOXES_TOP

    for r2, (sm, _, above, below, pages) in _ROUTES.items():
        design = design_of[r2]
        box_h = sizes[r2][1] + 0.1
        y = top + box_h / 2
        top += box_h + _BOX_GAP
        ax.text(_LABEL_X, y, design, ha="left", va="center", fontsize=pu.SIZE_LABEL, fontweight="bold", color=pu.INK)
        row = _Row(ax, extras, y)
        row.molecule(sm, f"{design}_start")
        row.plus()
        row.molecule(_CHLORONITROPYRIDINE, f"{design}_chloronitropyridine")
        row.x = cluster_end
        row.arrow(above, below, pages, _PATENT)
        row.molecule(amines[r2], f"{design}_amine")
        row.x = amine_x + max(_extent_in(m)[0] for m in amines.values()) + _GAP
        row.arrow(_COUPLING[0], _COUPLING[1], _COUPLING_PAGE, _NEW)
        row.molecule(products[r2], f"{design}_product", tail=True)

    for y, colour, text in ((_LEGEND_Y[0], _PATENT, "Patent chemistry (page in US 9,340,546 B2)"),
                            (_LEGEND_Y[1], _NEW, "New step and new compound")):
        ax.annotate("", xy=(_LEGEND_X + 0.45, y), xytext=(_LEGEND_X, y),
                    arrowprops={"arrowstyle": "-|>", "linewidth": 1.6, "color": colour, "shrinkA": 0, "shrinkB": 0})
        ax.text(_LEGEND_X + 0.55, y, text, ha="left", va="center", fontsize=pu.SIZE_SMALL, color=colour)

    pu.save_svg(fig, NAME, extras=extras)
