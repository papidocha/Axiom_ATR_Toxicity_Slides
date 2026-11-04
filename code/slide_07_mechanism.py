"""Slide 7: how a cyclic basic amine can turn toxic, what failed in the clinic, and what to test next.

A. Bioactivation of gartisertib's piperazine: a CYP enzyme oxidizes the carbon next to the basic nitrogen (via the
   carbinolamine, which loses water) to an iminium ion; water adds and the ring opens to an amino-aldehyde. Both are
   electrophiles that can bind proteins. A hypothesis from the literature, not tested here.
B. Gartisertib inhibits UGT1A1 in human liver microsomes only (Burris et al., Br J Cancer 2024; 130: 1131) and what
   the delivered predictions can and cannot say about it.
C. The experiments that would explain the toxicity of the series and what to test on the recommended compounds.
"""

from __future__ import annotations

import textwrap

from matplotlib.patches import FancyArrowPatch
from rdkit import Chem
from rdkit.Chem import rdDepictor

import load_data as D
import plot_utils as pu


NAME = "slide_07_mechanism"

_LINE = 0.215  #: inches per 12 pt line
_RIGHT_X = 6.5
_BOTTOM_TOP = 3.05
_PIECES = {
    "amine": "*C(=O)N1CCN(CC1)C1COC1",
    "iminium": "*C(=O)N1CC=[N+](CC1)C1COC1",
    "aldehyde": "*C(=O)N(CC=O)CCNC1COC1",
}
#: Gartisertib's UGT1A1 inhibition, IC50 in µM (Burris et al. 2024): human, dog and rat liver microsomes.
#: Dog and rat are reported as "> 50".
_UGT1A1 = (("Human", 8.36, False), ("Dog", 50.0, True), ("Rat", 50.0, True))


def slide_data() -> dict[str, float]:
    """The numbers the slide's text quotes."""
    series = D.load_lead_series()
    frame = D.load_descriptors().copy()
    frame["general_tox"] = series.data.set_index("name")["general_tox"].reindex(frame.index)
    basic = frame["n_basic_nitrogen"]
    profile = D.percentile_profile(series)
    return {
        "rho_basic": D.spearman(basic, frame["general_tox"])[0],
        "median_zero": float(frame.loc[basic == 0, "general_tox"].median()),
        "median_one": float(frame.loc[basic == 1, "general_tox"].median()),
        "gart_ldh_pct": float(profile.at["ldh_ld20", "gartisertib"]),
        "tuvu_ldh_pct": float(profile.at["ldh_ld20", "tuvusertib"]),
    }


def _minus(value: str) -> str:
    return value.replace("-", "−")


def _ordinal(share: float) -> str:
    n = round(share * 100)
    return f"{n}{'th' if 10 <= n % 100 <= 20 else {1: 'st', 2: 'nd', 3: 'rd'}.get(n % 10, 'th')}"


def _bullets(ax, x: float, y: float, items: list[str], wrap: int) -> float:
    """Bullet points in 12 pt with a hanging indent; returns the y below the last one."""
    for item in items:
        lines = textwrap.fill(item, wrap).split("\n")
        pu.text(ax, x, y, "•", va="top")
        pu.text(ax, x + 0.2, y, "\n".join(lines), va="top")
        y += len(lines) * _LINE + 0.06
    return y


def _piece(name: str, coordinates: Chem.Mol | None = None) -> Chem.Mol:
    mol = Chem.MolFromSmiles(_PIECES[name])
    if coordinates is None:
        rdDepictor.SetPreferCoordGen(True)
        rdDepictor.Compute2DCoords(mol)
    else:  # same atom order, only a bond order and a charge differ
        mol.AddConformer(Chem.Conformer(coordinates.GetConformer()), assignId=True)
    mol.GetAtomWithIdx(0).SetProp("atomLabel", "R")
    return mol


def _draw(mol: Chem.Mol, centre_x: float, centre_y: float, width: float, height: float, group: str) -> str:
    svg = pu.draw_structure(
        mol,
        size_px=(int(width * pu.PX_PER_IN), int(height * pu.PX_PER_IN)),
        bond_width=pu.BOND_WIDTH,
        font_px=pu.FONT_PX,
        bond_length_px=pu.BOND_LENGTH_PX,
    )
    return pu.place(svg, centre_x - width / 2, centre_y - height / 2, width, group)


def _arrow(ax, x0: float, x1: float, y: float, label: str) -> None:
    ax.add_patch(FancyArrowPatch((x0, y), (x1, y), arrowstyle="-|>", mutation_scale=14, color=pu.INK, linewidth=1.8, zorder=3))
    pu.text(ax, (x0 + x1) / 2, y - 0.1, label, size=pu.SIZE_SMALL, colour=pu.MUTED, ha="center", va="bottom")


def _bioactivation(ax) -> list[str]:
    """Amine -> iminium ion -> ring-opened amino-aldehyde; both intermediates are electrophiles that bind proteins."""
    amine = _piece("amine")
    iminium = _piece("iminium", amine)
    aldehyde = _piece("aldehyde")
    y = 1.9
    x_amine, x_iminium, x_aldehyde, x_adduct = 1.9, 5.2, 8.5, 11.0
    pu.text(ax, 0.0, 0.8, "Bioactivation by CYP enzymes: Two reactive intermediates, both able to bind proteins", weight="bold", va="top")
    for x, name in ((x_iminium, "iminium ion"), (x_aldehyde, "amino-aldehyde")):
        pu.text(ax, x, y + 0.5, name, weight="bold", colour=pu.FAIL, ha="center", va="top")
    _arrow(ax, 2.85, 4.15, y, "CYP\noxidation")
    _arrow(ax, 6.2, 7.45, y, "+ H2O\nring opens")
    _arrow(ax, 9.5, 10.2, y, "protein\nCys, Lys")
    pu.text(ax, x_adduct, y, "covalent\nprotein adducts", weight="bold", colour=pu.FAIL, ha="center", va="center")
    ax.add_patch(FancyArrowPatch((x_adduct, y + 0.34), (x_adduct, y + 0.56), arrowstyle="-|>", mutation_scale=14,
                                 color=pu.INK, linewidth=1.8, shrinkA=0, shrinkB=0, zorder=3))
    pu.text(ax, x_adduct, y + 0.6, "hepatocyte injury", weight="bold", colour=pu.FAIL, ha="center", va="top")
    pu.text(ax, x_adduct, y + 0.82, "ALT/AST (clinic), LDH release (Axiom)", size=pu.SIZE_SMALL, colour=pu.MUTED, ha="center", va="top")
    pu.text(ax, 0.0, y + 0.6, "R = rest of gartisertib", size=pu.SIZE_SMALL, colour=pu.MUTED, va="top")
    return [_draw(amine, x_amine, y, 1.55, 1.3, "amine"), _draw(iminium, x_iminium, y, 1.55, 1.3, "iminium"),
            _draw(aldehyde, x_aldehyde, y, 1.75, 1.3, "aldehyde")]


def _ugt1a1_chart(fig, left: float, top: float) -> None:
    ax = pu.axes_in(fig, left, top, 4.0, 0.62)
    positions = [2, 1, 0]
    values = [value for _, value, _ in _UGT1A1]
    open_ended = [flag for _, _, flag in _UGT1A1]
    ax.barh(positions, values, color=[("#D0D0D0" if flag else pu.FAIL) for flag in open_ended], height=0.7)
    for position, value, flag in zip(positions, values, open_ended):
        ax.text(value + 1.2, position, "> 50 µM" if flag else f"{value:.1f} µM", va="center", ha="left",
                fontsize=pu.SIZE_SMALL, fontweight="bold", color=pu.MUTED if flag else pu.FAIL)
    ax.set_yticks(positions, [name for name, _, _ in _UGT1A1], fontsize=pu.SIZE_SMALL)
    ax.set_xlim(0, 75)
    ax.set_xticks([])
    ax.tick_params(axis="y", length=0)
    ax.spines["bottom"].set_visible(False)


def slide_07_mechanism() -> None:
    d = slide_data()
    fig = pu.new_figure()
    ax = pu.overlay(fig)
    ax.plot([0, 12.33], [_BOTTOM_TOP - 0.1, _BOTTOM_TOP - 0.1], color=pu.GRID, linewidth=1.0)
    ax.plot([6.2, 6.2], [_BOTTOM_TOP, 5.95], color=pu.GRID, linewidth=1.0)

    # ---- 1: bioactivation ------------------------------------------------------------------- #
    pu.text(ax, 0.0, 0.1, "A.  How a cyclic basic amine can turn toxic", size=pu.SIZE_HEADING, weight="bold", va="top")
    pu.text(ax, 0.0, 0.5, f"In Axiom Bio's predicted data a basic N lowers general_tox (median {_minus(f'{d['median_zero']:+.2f}')} without, "
         f"{_minus(f'{d['median_one']:+.2f}')} with one; \u03c1 {_minus(f'{d['rho_basic']:.2f}')}). Bioactivation is one way a cyclic amine could do this.",
         size=pu.SIZE_SMALL, colour=pu.MUTED, va="top")
    extras = _bioactivation(ax)

    # ---- 2: what failed -------------------------------------------------------------------- #
    pu.text(ax, 0.0, _BOTTOM_TOP, "B.  What failed in the clinic", size=pu.SIZE_HEADING, weight="bold", va="top")
    pu.text(ax, 0.0, _BOTTOM_TOP + 0.45, "Gartisertib inhibits UGT1A1 in human liver only", weight="bold", va="top")
    _ugt1a1_chart(fig, 0.85, _BOTTOM_TOP + 0.72)
    _bullets(ax, 0.0, _BOTTOM_TOP + 1.47, [
        f"ALT/AST rises: Hepatocyte injury that bioactivation could explain and Axiom's LDH readout sees "
        f"(gartisertib {_ordinal(d['gart_ldh_pct'])} vs tuvusertib {_ordinal(d['tuvu_ldh_pct'])} percentile).",
        "Bilirubin rise (UGT1A1): No readout in Axiom's model; linked to bioactivation only if M26 forms that way (unproven).",
    ], wrap=66)

    # ---- 3: next steps ---------------------------------------------------------------------- #
    pu.text(ax, _RIGHT_X, _BOTTOM_TOP, "C.  What to test next", size=pu.SIZE_HEADING, weight="bold", va="top")
    y = _BOTTOM_TOP + 0.45
    pu.text(ax, _RIGHT_X, y, "To understand the gartisertib series", weight="bold", va="top")
    y = _bullets(ax, _RIGHT_X, y + _LINE + 0.02, [
        "M26: UGT1A1 IC50, structure, formation",
        "Reactive metabolites: LC-MS/MS trapping and GSH depletion",
        "OATP1B1/1B3 inhibition (also raises indirect bilirubin)",
    ], wrap=60) + 0.04
    pu.text(ax, _RIGHT_X, y, "For the recommended candidates (no basic N is no guarantee)", weight="bold", va="top")
    _bullets(ax, _RIGHT_X, y + _LINE + 0.02, [
        "Generate experimental data in Axiom Bio's hepatocyte systems",
        "Run UGT1A1 and OATP1B1/1B3 assays, gartisertib as reference",
        "Human systems only; rat and dog missed this signal",
    ], wrap=60)

    pu.text(ax, 0.0, 6.05, "Burris et al., British Journal of Cancer 2024, 130, 1131\u20131140", size=pu.SIZE_SMALL, colour=pu.MUTED, va="top")

    pu.save_svg(fig, NAME, extras=extras)
