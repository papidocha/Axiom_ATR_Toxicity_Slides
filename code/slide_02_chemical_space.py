"""Slide 2: one chemotype, four sub-series.

A. t-SNE map of the 553 compounds (Morgan fingerprint Tanimoto distance) colored by sub-series, with the two
   reference compounds labeled.
B. The core shared by every compound, with its attachment points R1 to R3.
C. What defines each sub-series: how the R2 tail is attached to the pyridine.
Every number is computed from the R-group decomposition. Compound_I-C-50 is corrected when loading (aromatic pyrazole).
"""

from __future__ import annotations

import re

import pandas as pd
from matplotlib.lines import Line2D
from rdkit import Chem

import load_data as D
import plot_utils as pu



NAME = "slide_02_chemical_space"

#: Core with the three variable attachment points labelled as the decomposition does.
#: R4 (pyridine C6) is H in 99 % of compounds and is not drawn.
_CORE_R123 = "Nc1nn2cc([*:1])cnc2c1C(=O)Nc1cncc([*:3])c1[*:2]"

#: The linker that defines each sub-series, as a SMILES fragment hanging off the
#: pyridine C4. ``[*:2]`` is the rest of the solubilising tail (drawn R'); the I-N nitrogen
#: and the I-C carbon carry a second one, ``[*:5]`` (drawn R''). The I-C bond to R' is drawn
#: solid-plus-dashed (see ``_dash_one_tail_bond``).
_LINKERS: dict[str, str] = {
    "I-N": "N([*:2])[*:5]",
    "I-G": "N1CCC(CC1)C(=O)[*:2]",
    "I-O": "O[*:2]",
    "I-C": "C([*:5])[*:2]",
}

#: An attachment point (R1, R3, R4) is drawn on a sub-series structure only if at least
#: this share of its compounds carry something other than hydrogen there.
_POSITION_MIN_SHARE = 0.10

_HALOGENS = {"F[*:1]", "Cl[*:1]", "Br[*:1]", "I[*:1]", "F[*:3]", "Cl[*:3]", "Br[*:3]", "I[*:3]"}
_PIPERIDINE_CARBONYL = Chem.MolFromSmarts("[#6](=O)C1CCN(*)CC1")

#: Offsets (points) of the reference labels.
_LABEL_OFFSETS: dict[str, tuple[float, float]] = {
    "Compound_I-G-32": (-4, -26),
    "Compound_I-C-79": (14, 10),
}
_LABELS = {name: meta["label"] for name, meta in D.REFERENCE_COMPOUNDS.items()}


# --------------------------------------------------------------------------- #
# Numbers
# --------------------------------------------------------------------------- #


def _attachment(smiles: str | None) -> tuple[str, bool] | None:
    """Element and ring membership of the atom the R-group is attached through."""
    if not isinstance(smiles, str):
        return None
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    for atom in mol.GetAtoms():
        if atom.GetAtomicNum() == 0:
            neighbour = atom.GetNeighbors()[0]
            return neighbour.GetSymbol(), neighbour.IsInRing()
    return None


def _has_piperidine_carbonyl(smiles: str | None) -> bool:
    if not isinstance(smiles, str):
        return False
    mol = Chem.MolFromSmiles(smiles)
    return mol is not None and mol.HasSubstructMatch(_PIPERIDINE_CARBONYL)


def subseries_facts(series, rgroups: pd.DataFrame) -> dict[str, dict[str, object]]:
    """For each sub-series: its size and how many compounds show its defining linker.

    The defining feature is read off the R2 fragment of the decomposition: the element
    the tail is attached through (N in a ring, O, C) and, for I-G, the
    piperidine-4-carbonyl motif.
    """
    data = series.data.set_index("name")
    sub = data.loc[rgroups.index, "subseries"]
    facts: dict[str, dict[str, object]] = {}
    for name in D.SUBSERIES:
        members = rgroups.index[sub == name]
        total = int((data["subseries"] == name).sum())
        attachment = rgroups.loc[members, "R2"].map(_attachment)
        if name == "I-N":
            hits = int(attachment.map(lambda a: a == ("N", True)).sum())
        elif name == "I-G":
            hits = int(rgroups.loc[members, "R2"].map(_has_piperidine_carbonyl).sum())
        elif name == "I-O":
            hits = int(attachment.map(lambda a: a is not None and a[0] == "O").sum())
        else:
            hits = int(attachment.map(lambda a: a is not None and a[0] == "C").sum())
        facts[name] = {"n": total, "n_decomposed": len(members), "n_defining": hits}
    return facts


def core_position_facts(rgroups: pd.DataFrame) -> dict[str, float]:
    """The R1 / R2 / R3 statements printed next to the core, computed from the data."""
    n = len(rgroups)
    counts = rgroups["R2"].value_counts()
    return {
        "n_core": n,
        "r1_f": float((rgroups["R1"] == "F[*:1]").mean()),
        "r1_cl": float((rgroups["R1"] == "Cl[*:1]").mean()),
        "r2_distinct": int(counts.size),
        "r2_seen_once": float((counts == 1).mean()),
        "r3_h": float((rgroups["R3"] == "[H][*:3]").mean()),
        "r3_halogen": float(rgroups["R3"].isin(_HALOGENS).mean()),
    }


def drawn_positions(series, rgroups: pd.DataFrame) -> dict[str, tuple[int, ...]]:
    """Which of R1, R3, R4 appear in each sub-series, and so get drawn on its structure.

    A position appears when at least ``_POSITION_MIN_SHARE`` of the sub-series' compounds
    carry a non-hydrogen substituent there. R2 is the tail itself and is always drawn.
    """
    sub = series.data.set_index("name").loc[rgroups.index, "subseries"]
    shown: dict[str, tuple[int, ...]] = {}
    for name in D.SUBSERIES:
        members = rgroups[sub == name]
        positions = []
        for k in (1, 3, 4):
            fragment = members[f"R{k}"]
            non_hydrogen = (fragment.notna() & (fragment != f"[H][*:{k}]")).mean()
            if non_hydrogen >= _POSITION_MIN_SHARE:
                positions.append(k)
        shown[name] = tuple(positions)
    return shown


# --------------------------------------------------------------------------- #
# Structures
# --------------------------------------------------------------------------- #


def _label_dummies(mol: Chem.Mol, labels: dict[int, str] | str) -> None:
    """Give each dummy atom a visible label (``R1``... or a fixed ``R``)."""
    for atom in mol.GetAtoms():
        if atom.GetAtomicNum() != 0:
            continue
        text = labels if isinstance(labels, str) else labels[atom.GetAtomMapNum()]
        atom.SetAtomMapNum(0)
        # R1..R4 are drawn with the number as a superscript (RDKit label markup).
        atom.SetProp("atomLabel", re.sub(r"R(\d)", r"R<sup>\1</sup>", text))


def draw_core(reference: Chem.Mol, size_px: tuple[int, int]) -> str:
    """The shared core with R1-R3 labelled."""
    mol, _ = pu.aligned(_CORE_R123, reference)
    _label_dummies(mol, {1: "R1", 2: "R2", 3: "R3"})
    return pu.draw_structure(
        mol, size_px=size_px, bond_width=2.2, font_px=22, bond_length_px=pu.BOND_LENGTH_PX
    )


def _template_smiles(name: str, positions: tuple[int, ...]) -> str:
    """SMILES of the core with the sub-series linker and the attachment points that appear.

    Atom order along the pyridine matches ``CORE_NO_R`` (NH-bearing carbon, CH, N, C6,
    C5, C4), so the template aligns to the reference drawing.
    """
    bicycle = "Nc1nn2cc([*:1])cnc2c1" if 1 in positions else "Nc1nn2cccnc2c1"
    c6 = "c([*:4])" if 4 in positions else "c"
    c5 = "c([*:3])" if 3 in positions else "c"
    return f"{bicycle}C(=O)Nc1cn{c6}{c5}c1{_LINKERS[name]}"


#: Share of white mixed into the sub-series colour for the highlight on the structures, so
#: the black atom labels and bonds stay readable on top of it.
_HIGHLIGHT_LIGHTEN = 0.55


#: Sub-series whose R' / R'' stubs are highlighted along with the linker atom.
_TAIL_HIGHLIGHTED = ("I-N", "I-G", "I-O", "I-C")


def _dash_one_tail_bond(mol: Chem.Mol) -> None:
    """Draw one of the two R' bonds on the I-C carbon as a solid-plus-dashed bond.

    The I-C carbon starts a tail that can be a chain, a ring or a fused ring, so its
    bonds to R' may be single or aromatic. Only the depiction changes; the coordinates
    are already set.
    """
    for carbon in mol.GetAtoms():
        stubs = [n for n in carbon.GetNeighbors() if n.GetAtomicNum() == 0]
        if carbon.GetSymbol() == "C" and len(stubs) == 2:
            dashed = next(n for n in stubs if n.GetProp("atomLabel") == "R'")
            bond = mol.GetBondBetweenAtoms(carbon.GetIdx(), dashed.GetIdx())
            bond.SetBondType(Chem.BondType.AROMATIC)
            bond.SetIsAromatic(True)
            return


def draw_subseries(
    name: str, reference: Chem.Mol, size_px: tuple[int, int], positions: tuple[int, ...]
) -> str:
    """The core plus the linker that defines ``name``, linker highlighted in its colour.

    Attachment points that appear in the sub-series (``positions``) are drawn and
    labelled R1, R3 or R4; the tail stubs are labelled R' (not R2, which names a position on the core in panel B).
    """
    mol, new_atoms = pu.aligned(_template_smiles(name, positions), reference)
    _label_dummies(mol, {1: "R1", 2: "R'", 3: "R3", 4: "R4", 5: "R''"})
    if name == "I-C":
        _dash_one_tail_bond(mol)

    colour = pu.lighten(pu.SUBSERIES_COLORS[name], _HIGHLIGHT_LIGHTEN)
    new = set(new_atoms)
    atoms = {i: colour for i in new_atoms if mol.GetAtomWithIdx(i).GetAtomicNum() != 0}
    bonds = {
        b.GetIdx(): colour
        for b in mol.GetBonds()
        if (b.GetBeginAtomIdx() in new or b.GetEndAtomIdx() in new)
        and not (
            mol.GetAtomWithIdx(b.GetBeginAtomIdx()).GetAtomicNum() == 0
            or mol.GetAtomWithIdx(b.GetEndAtomIdx()).GetAtomicNum() == 0
        )
    }
    if name in _TAIL_HIGHLIGHTED:  # the R' / R'' stubs and their bonds carry the colour too
        stubs = {a.GetIdx() for a in mol.GetAtoms() if a.GetAtomicNum() == 0 and a.GetProp("atomLabel").startswith("R'")}
        atoms.update({i: colour for i in stubs})
        bonds.update(
            {
                b.GetIdx(): colour
                for b in mol.GetBonds()
                if b.GetBeginAtomIdx() in stubs or b.GetEndAtomIdx() in stubs
            }
        )
    if name == "I-C":  # a bare carbon is not labelled by default; label it.
        for i in atoms:
            if mol.GetAtomWithIdx(i).GetAtomicNum() != 0:
                mol.GetAtomWithIdx(i).SetProp("atomLabel", "C")
    # Halos only behind drawn letters (N, O, S, R, the labelled C); unlabelled carbons are traced by their bonds alone.
    atoms = {i: c for i, c in atoms.items() if mol.GetAtomWithIdx(i).GetAtomicNum() != 6 or mol.GetAtomWithIdx(i).HasProp("atomLabel")}
    return pu.draw_structure(
        mol,
        size_px=size_px,
        highlight_atoms=atoms,
        highlight_bonds=bonds,
        bond_width=2.2,
        font_px=22,
        bond_length_px=pu.BOND_LENGTH_PX,
    )


# --------------------------------------------------------------------------- #
# Panel A: the map
# --------------------------------------------------------------------------- #


def _panel_map(ax, series, layout: pd.DataFrame) -> None:
    """t-SNE map coloured by sub-series; reference compounds ringed and labelled."""
    data = series.data.set_index("name").loc[layout.index]
    for name in D.SUBSERIES:
        mask = (data["subseries"] == name).to_numpy()
        ax.scatter(
            layout.loc[mask, "tsne_x"],
            layout.loc[mask, "tsne_y"],
            s=15,
            color=pu.SUBSERIES_COLORS[name],
            linewidth=0,
            alpha=0.9,
            zorder=2,
        )

    for compound, offset in _LABEL_OFFSETS.items():
        x, y = layout.loc[compound, ["tsne_x", "tsne_y"]]
        colour = pu.SUBSERIES_COLORS[data.at[compound, "subseries"]]
        ax.scatter(x, y, s=95, color=colour, edgecolor=pu.INK, linewidth=1.6, zorder=4)
        ax.annotate(
            _LABELS[compound],
            (x, y),
            xytext=offset,
            textcoords="offset points",
            fontsize=pu.SIZE_LABEL,
            fontweight="bold",
            color=pu.INK,
            ha="center",
            va="center",
            zorder=5,
            arrowprops={"arrowstyle": "-", "color": pu.INK, "linewidth": 0.8, "shrinkA": 0, "shrinkB": 5},
            bbox={"boxstyle": "round,pad=0.12", "facecolor": "white", "edgecolor": "none", "alpha": 0.85},
        )
    # t-SNE coordinates have no units, so the layout is stretched to fill the panel
    # instead of keeping an equal aspect, which would leave the panel half empty. Slide 5
    # uses the same view.
    pu.apply_view(ax, layout)


# --------------------------------------------------------------------------- #
# Slide
# --------------------------------------------------------------------------- #


def slide_02_chemical_space() -> None:
    series = D.load_lead_series()
    layout = D.tsne_coordinates()
    rgroups, _ = D.rgroup_decomposition()
    facts = subseries_facts(series, rgroups)
    core = core_position_facts(rgroups)
    reference = pu.reference_core()

    fig = pu.new_figure()

    # --- A -------------------------------------------------------------- #
    pu.panel_heading(fig, 0.0, 0.1, "A.  Chemical space of the 553 compounds")
    ax = pu.axes_in(fig, 0.0, 1.0, 6.0, 4.85)
    _panel_map(ax, series, layout)
    fig.legend(
        handles=[
            Line2D(
                [],
                [],
                marker="o",
                linestyle="none",
                markersize=7,
                color=pu.SUBSERIES_COLORS[name],
                label=f"{name}  (n = {facts[name]['n']})",
            )
            for name in D.SUBSERIES
        ],
        loc="upper left",
        bbox_to_anchor=(0.0, 1.0 - 0.55 / pu.SLIDE_H_IN),
        ncols=4,
        fontsize=pu.SIZE_LABEL,
        columnspacing=1.2,
        handletextpad=0.2,
    )
    pu.footnote(
        fig,
        0.0,
        6.02,
        "t-SNE of Tanimoto distance, Morgan fingerprints (radius 2, 2048 bits)",
        size=pu.SIZE_LABEL,
    )

    # --- B -------------------------------------------------------------- #
    left = 6.45
    pu.panel_heading(fig, left, 0.1, "B.  Shared core in all compounds")
    px_core = (440, 235)
    core_w_in = 2.7
    core_svg = draw_core(reference, px_core)
    extras = [pu.place(core_svg, left, 0.45, core_w_in, "core")]
    r_text = (
        f"R1: F {core['r1_f']:.0%}, Cl {core['r1_cl']:.0%}\n"
        f"R2: {core['r2_distinct']} distinct ({core['r2_seen_once']:.0%})\n"
        f"R3: H {core['r3_h']:.0%}, halogen {core['r3_halogen']:.0%}"
    )
    fig.text(
        (left + core_w_in + 0.1) / pu.SLIDE_W_IN,
        1.0 - 0.95 / pu.SLIDE_H_IN,
        r_text,
        ha="left",
        va="top",
        fontsize=pu.SIZE_LABEL,
        color=pu.INK,
        linespacing=1.6,
    )

    # --- C -------------------------------------------------------------- #
    pu.panel_heading(fig, left, 1.98, "C.  What defines each sub-series")
    cell_w, cell_h = 2.95, 1.88
    px_cell = (440, 250)
    structure_w_in = 2.85
    positions = drawn_positions(series, rgroups)
    for index, name in enumerate(D.SUBSERIES):
        row, col = divmod(index, 2)
        x = left + col * (cell_w + 0.05)
        y = 2.5 + row * cell_h
        n_def, n_all = facts[name]["n_defining"], facts[name]["n_decomposed"]
        fig.text(
            x / pu.SLIDE_W_IN,
            1.0 - y / pu.SLIDE_H_IN,
            f"{name}",
            ha="left",
            va="top",
            fontsize=pu.SIZE_LABEL,
            fontweight="bold",
            color=pu.SUBSERIES_COLORS[name],
        )
        fig.text(
            (x + 0.5) / pu.SLIDE_W_IN,
            1.0 - (y + 0.02) / pu.SLIDE_H_IN,
            f"{n_def} of {n_all} compounds",
            ha="left",
            va="top",
            fontsize=pu.SIZE_LABEL,
            color=pu.INK,
        )
        extras.append(
            pu.place(
                draw_subseries(name, reference, px_cell, positions[name]),
                x,
                y + 0.26,
                structure_w_in,
                f"subseries_{name}",
            )
        )

    pu.save_svg(fig, NAME, extras=extras)
