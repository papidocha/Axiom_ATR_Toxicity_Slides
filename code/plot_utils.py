"""Shared plotting code of the slides: style, SVG output, RDKit structure drawing and the t-SNE map view.

Every slide figure is drawn on one canvas and saved by :func:`save_svg`, which

* writes vector SVG with live text (Arial; plain font attributes that PowerPoint reads),
* places the figure on a white 33.867 x 19.05 cm slide with the title from :data:`TITLES`
  above it, and keeps the corner of PowerPoint's page number free (:data:`PAGE_NUMBER_ZONE`),
* asserts that no font size other than 24 / 16 / 12 / 10 pt is used.

Matplotlib writes SVG in points, so 1 SVG unit = 1 pt and a "12" in the file is 12 pt on the slide.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Final

import matplotlib as mpl

mpl.use("Agg")  # figures are only written to file

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np
import pandas as pd
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from rdkit import Chem
from rdkit.Chem import rdDepictor
from rdkit.Chem.Draw import rdMolDraw2D

import load_data as D

REFERENCE_COLORS: Final[dict[str, str]] = {D.GARTISERTIB: "#D55E00", D.TUVUSERTIB: "#0072B2"}
SUBSERIES_COLORS: Final[dict[str, str]] = {"I-N": "#0072B2", "I-G": "#D55E00", "I-O": "#009E73", "I-C": "#CC79A7"}
LEVEL_LABELS: Final[dict[str, str]] = {"ld20": "EC20", "ld50": "EC50"}

# --------------------------------------------------------------------------- #
# Canvas, font, sizes
# --------------------------------------------------------------------------- #

#: Slide-content canvas, in inches (16:9 slide minus side margins and a title band).
SLIDE_W_IN: Final[float] = 12.33
SLIDE_H_IN: Final[float] = 6.30

FONT_FAMILY: Final[str] = "Arial"
FONT_FALLBACK: Final[str] = "Arial, Helvetica, sans-serif"
#: The equation on slide 4 is the one text set in a serif face.
EQUATION_FAMILY: Final[str] = "Times New Roman"
EQUATION_FALLBACK: Final[str] = "Times New Roman, Times, serif"

#: The whole slide, in inches (33.867 x 19.05 cm).
PAGE_W_IN: Final[float] = 33.867 / 2.54
PAGE_H_IN: Final[float] = 19.05 / 2.54
#: Space above the figure on slides that carry a title; a slide without one is centered vertically.
TITLE_BAND_IN: Final[float] = 1.0
#: Where PowerPoint puts its page number (inches from the slide's top-left): left, top, right, bottom.
#: About 11 mm from the right edge and 7 mm from the bottom, plus some room around it.
PAGE_NUMBER_ZONE: Final[tuple[float, float, float, float]] = (12.0, 6.85, PAGE_W_IN, PAGE_H_IN)

#: Slide titles, by figure name. Neighbouring slides may share one.
TITLES: Final[dict[str, str]] = {
    "appendix_01_distributions": "Appendix: Distribution of the 20 predicted EC columns",
    "slide_02_chemical_space": "553 compounds, one shared core, four sub-series",
    "slide_03_potency_recovery": "Patent Ki bins fill the gap in Axiom's potency labels",
    "slide_04_validation": "general_tox: One score, checked on two references",
    "slide_05_chemical_landscape": "Predicted safety across chemical space",
    "slide_06_physicochemistry": "Basic nitrogen count is the strongest correlate",
    "slide_07_mechanism": "Bioactivation hypothesis and what to test next",
    "slide_08_shortlist": "Shortlist of 23 safer patent compounds",
    "slide_09_designs": "Six proposed designs, predicted safe and potent",
    "slide_10_synthesis": "Designs D2, D4 and D6 reachable with patent chemistry",
}

#: The only sizes, in points, used by any slide figure (the title is the one larger size).
SIZE_TITLE: Final[int] = 24  #: slide title
SIZE_HEADING: Final[int] = 16  #: panel headings ("A.  ...")
SIZE_LABEL: Final[int] = 12  #: axis labels, legends, panel titles, annotations
SIZE_SMALL: Final[int] = 10  #: tick labels, data labels, footnotes
ALLOWED_SIZES: Final[frozenset[int]] = frozenset({SIZE_TITLE, SIZE_HEADING, SIZE_LABEL, SIZE_SMALL})
#: Sub- and superscripts (K_i, EC_20, R^1) are set at this share of their line's size.
SCRIPT_SHRINK: Final[float] = 0.7
ALLOWED_FILE_SIZES: Final[frozenset[int]] = ALLOWED_SIZES | frozenset(
    round(size * SCRIPT_SHRINK) for size in ALLOWED_SIZES
)

# --------------------------------------------------------------------------- #
# Colour
# --------------------------------------------------------------------------- #

INK: Final[str] = "#222222"
MUTED: Final[str] = "#6B6B6B"
GRID: Final[str] = "#E3E3E3"
SERIES: Final[str] = "#4C72B0"
CENSORED: Final[str] = "#B5483C"
GARTISERTIB: Final[str] = REFERENCE_COLORS[D.GARTISERTIB]
TUVUSERTIB: Final[str] = REFERENCE_COLORS[D.TUVUSERTIB]
SUBSERIES_COLORS: Final[dict[str, str]] = dict(SUBSERIES_COLORS)

FIGURES_DIR: Final[Path] = D.SLIDES_DIR

#: Short assay names for panel titles and row labels (the long ones live in ``config``).
SHORT_NAMES: Final[dict[str, str]] = {
    "cell_viability": "Cell viability",
    "ldh": "LDH release",
    "cell_death": "Cell death",
    "nuclei_size": "Nuclei size",
    "cytoplasm_area": "Cytoplasm area",
    "mitochondria": "Mitochondria",
    "er_stress": "ER stress",
    "ros": "ROS",
    "bioactivity1": "Bioactivity 1",
    "bioactivity2": "Bioactivity 2",
}


def column_label(column: str) -> str:
    """``cell_viability_ld20`` -> ``Cell viability EC20``."""
    assay, _, level = column.rpartition("_")
    return f"{SHORT_NAMES[assay]} {LEVEL_LABELS[level]}"


# --------------------------------------------------------------------------- #
# matplotlib
# --------------------------------------------------------------------------- #


def apply_style() -> None:
    """Install the slide style. Call once before creating a figure."""
    mpl.rcParams.update(
        {
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "savefig.facecolor": "white",
            "font.family": "sans-serif",
            "font.sans-serif": [FONT_FAMILY, "Helvetica", "DejaVu Sans"],
            "font.size": SIZE_LABEL,
            "axes.titlesize": SIZE_LABEL,
            "axes.titleweight": "bold",
            "axes.titlelocation": "left",
            "axes.labelsize": SIZE_LABEL,
            "axes.edgecolor": INK,
            "axes.labelcolor": INK,
            "axes.linewidth": 0.8,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.axisbelow": True,
            "axes.grid": False,
            "grid.color": GRID,
            "grid.linewidth": 0.6,
            "xtick.labelsize": SIZE_SMALL,
            "ytick.labelsize": SIZE_SMALL,
            "xtick.color": INK,
            "ytick.color": INK,
            "xtick.major.size": 3.0,
            "ytick.major.size": 3.0,
            "legend.fontsize": SIZE_LABEL,
            "legend.frameon": False,
            "text.color": INK,
            "mathtext.fontset": "custom",
            "mathtext.rm": FONT_FAMILY,
            "mathtext.it": f"{FONT_FAMILY}:italic",
            "mathtext.bf": f"{FONT_FAMILY}:bold",
            "mathtext.fallback": "stix",
            "svg.fonttype": "none",
            "svg.hashsalt": "axiom-atr-slides",
            "hatch.linewidth": 1.0,
        }
    )


def new_figure() -> Figure:
    """A blank slide-content canvas."""
    apply_style()
    return plt.figure(figsize=(SLIDE_W_IN, SLIDE_H_IN))


def panel_heading(fig: Figure, x_in: float, y_in: float, text: str) -> None:
    """A bold panel heading placed by absolute position (inches from the top-left)."""
    fig.text(
        x_in / SLIDE_W_IN,
        1.0 - y_in / SLIDE_H_IN,
        text,
        ha="left",
        va="top",
        fontsize=SIZE_HEADING,
        fontweight="bold",
        color=INK,
    )


def axes_in(fig: Figure, left: float, top: float, width: float, height: float, **kw):
    """Add axes by absolute inches from the top-left corner of the canvas."""
    return fig.add_axes(
        (
            left / SLIDE_W_IN,
            1.0 - (top + height) / SLIDE_H_IN,
            width / SLIDE_W_IN,
            height / SLIDE_H_IN,
        ),
        **kw,
    )


def footnote(fig: Figure, x_in: float, y_in: float, text: str, size: float = SIZE_SMALL) -> None:
    """Small grey note at an absolute position; used sparingly."""
    fig.text(
        x_in / SLIDE_W_IN,
        1.0 - y_in / SLIDE_H_IN,
        text,
        ha="left",
        va="top",
        fontsize=size,
        color=MUTED,
        linespacing=1.25,
    )


# --------------------------------------------------------------------------- #
# SVG output
# --------------------------------------------------------------------------- #

_TEXT_TAG = re.compile(r"<(text|tspan)([^>]*)>")
_STYLE_ATTR = re.compile(r'\s*style="([^"]*)"')
_SIZE_IN_FILE = re.compile(r'font-size="([\d.]+)"')
#: CSS properties moved from ``style`` into plain presentation attributes.
_TEXT_PROPERTIES = (
    "font-size",
    "font-family",
    "font-weight",
    "font-style",
    "text-anchor",
    "fill",
    "white-space",
)


def _promote_text_properties(tag: re.Match[str]) -> str:
    """Move font and fill properties of one ``<text>`` / ``<tspan>`` into attributes.

    Matplotlib writes them inside ``style="..."``; Office reads plain attributes more
    reliably. The family list is replaced by Arial, or by the equation face for the equation text.
    """
    name, attributes = tag.groups()
    style = _STYLE_ATTR.search(attributes)
    if style is None:
        return tag.group(0)

    moved: dict[str, str] = {}
    kept: list[str] = []
    for declaration in (d.strip() for d in style.group(1).split(";") if d.strip()):
        key, _, value = declaration.partition(":")
        key, value = key.strip(), value.strip()
        if key in _TEXT_PROPERTIES:
            moved[key] = value
        else:
            kept.append(declaration)

    if "font-size" in moved:
        moved["font-size"] = moved["font-size"].removesuffix("px")
    if "font-family" in moved:
        is_equation = EQUATION_FAMILY in moved["font-family"]
        moved["font-family"] = EQUATION_FALLBACK if is_equation else FONT_FALLBACK
    remaining = attributes.replace(style.group(0), "")
    extra = f' style="{"; ".join(kept)}"' if kept else ""
    promoted = "".join(f' {k}="{v}"' for k, v in moved.items())
    return f"<{name}{promoted}{extra}{remaining}>"


def _normalise_text_styles(svg: str) -> str:
    """Rewrite every text element's font properties as plain SVG attributes."""
    return _TEXT_TAG.sub(_promote_text_properties, svg)


#: Text that gets a sub- or superscript: K_i, EC_20 / EC_50, R^1 to R^4, the 2 of CH2CN and the digits of a few
#: reagent formulas (Et3N, H2, Pd(PPh3)4 ...). One alternation
#: so that neighbouring matches cannot overlap.
_SCRIPT_TARGET = re.compile(
    r"(?<![A-Za-z0-9])(?:(K)(i)|(EC|IC)(20|50)|(R)([1-4]))(?![A-Za-z0-9])|(?<![A-Za-z0-9])(CH)(2)(?=CN(?![A-Za-z0-9]))"
    r"|(?<![A-Za-z0-9])(Pd\(PPh3\)4|Pd2\(dba\)3|PPh3|PhSiH3|Na2CO3|Et3N|H2O|H2)(?![A-Za-z0-9])"
)
#: Reagent formulas of slide 10: every digit in them is a subscript.
_FORMULA_DIGITS = re.compile(r"(\d+)")
_PLAIN_TEXT_ELEMENT = re.compile(r"(<text\b[^>]*>)([^<]*)(</text>)")
_SIZE_ATTRIBUTE = re.compile(r'font-size="([\d.]+)"')
_SUBSCRIPT_DROP = 0.22  #: em, downwards
_SUPERSCRIPT_RISE = 0.38  #: em, upwards
_KI_HEADING_DX = 0.06  #: em, to the right: keeps the i of K_i in 16 pt headings off the K when PowerPoint re-lays out the text


def _scripted_content(content: str, size: float) -> str:
    """``content`` with K_i, EC_20 / EC_50 and R^1..R^4 rewritten as sub/superscript tspans.

    Each script is a ``<tspan>`` at 70 % of the line's size moved by ``dy``; the text after
    it is wrapped in a ``<tspan>`` that moves back, so the line stays one live text element.
    """
    pieces: list[tuple[str, float, float]] = []  # (text, size, vertical offset in em, + = down)
    position = 0
    for match in _SCRIPT_TARGET.finditer(content):
        base, script, offset = None, None, 0.0
        if match.group(1):
            base, script, offset = match.group(1), match.group(2), _SUBSCRIPT_DROP
        elif match.group(3):
            base, script, offset = match.group(3), match.group(4), _SUBSCRIPT_DROP
        elif match.group(5):
            base, script, offset = match.group(5), match.group(6), -_SUPERSCRIPT_RISE
        elif match.group(9):
            parts = _FORMULA_DIGITS.split(match.group(9))  # text, digits, text, digits, ...
            pieces.append((content[position : match.start()] + parts[0], size, 0.0))
            for index in range(1, len(parts), 2):
                pieces.append((parts[index], size * SCRIPT_SHRINK, _SUBSCRIPT_DROP))
                if parts[index + 1]:
                    pieces.append((parts[index + 1], size, 0.0))
            position = match.end()
            continue
        else:
            base, script, offset = match.group(7), match.group(8), _SUBSCRIPT_DROP
        pieces.append((content[position : match.start()] + base, size, 0.0))
        pieces.append((script, size * SCRIPT_SHRINK, offset))
        position = match.end()
    if not pieces:
        return content
    if content[position:]:
        pieces.append((content[position:], size, 0.0))

    out: list[str] = []
    current = 0.0
    for index, (text, piece_size, offset) in enumerate(pieces):
        if index == 0 and offset == 0.0:
            out.append(text)  # leading text stays a plain text node
            continue
        dy = (offset - current) * size
        size_attribute = f' font-size="{piece_size:g}"' if piece_size != size else ""
        nudge = text == "i" and size == SIZE_HEADING and pieces[index - 1][0].endswith("K")
        dx_attribute = f' dx="{_KI_HEADING_DX * size:.3f}"' if nudge else ""
        out.append(f'<tspan{size_attribute}{dx_attribute} dy="{dy:.3f}">{text}</tspan>')
        current = offset
    return "".join(out)


def _apply_scripts(svg: str) -> str:
    """Rewrite every plain ``<text>`` element that mentions K_i, EC_20 / EC_50 or R^1..R^4."""

    def rewrite(match: re.Match[str]) -> str:
        opening, content, closing = match.groups()
        size = _SIZE_ATTRIBUTE.search(opening)
        if size is None or not _SCRIPT_TARGET.search(content):
            return match.group(0)
        return f"{opening}{_scripted_content(content, float(size.group(1)))}{closing}"

    return _PLAIN_TEXT_ELEMENT.sub(rewrite, svg)


def assert_font_sizes(svg: str, name: str) -> set[int]:
    """Fail if the file uses any size other than the three allowed ones (or their scripts).

    Returns:
        The set of sizes found, rounded to whole points.
    """
    found = {round(float(s)) for s in _SIZE_IN_FILE.findall(svg)}
    stray = found - ALLOWED_FILE_SIZES
    if stray:
        raise AssertionError(f"{name}: font sizes {sorted(stray)} are not in {sorted(ALLOWED_FILE_SIZES)}")
    return found


_SVG_ROOT = re.compile(r"<svg [^>]*>")


def _place_on_page(svg: str, title: str | None) -> str:
    """Put the content-area SVG on a white full-slide page, with the title above it.

    The figure's own coordinates are untouched: it is wrapped in a translated group.
    """
    left = (PAGE_W_IN - SLIDE_W_IN) / 2
    top = TITLE_BAND_IN if title else (PAGE_H_IN - SLIDE_H_IN) / 2
    root = _SVG_ROOT.search(svg)
    assert root is not None
    size = f'width="{PAGE_W_IN * 72:.2f}pt" height="{PAGE_H_IN * 72:.2f}pt" viewBox="0 0 {PAGE_W_IN * 72:.2f} {PAGE_H_IN * 72:.2f}"'
    parts = [
        f'<svg xmlns:xlink="http://www.w3.org/1999/xlink" xmlns="http://www.w3.org/2000/svg" version="1.1" {size}>',
        f' <rect x="0" y="0" width="{PAGE_W_IN * 72:.2f}" height="{PAGE_H_IN * 72:.2f}" fill="#ffffff"/>',
    ]
    if title:
        parts.append(
            f' <text x="{left * 72:.2f}" y="{0.62 * 72:.2f}" font-size="{SIZE_TITLE}" font-family="{FONT_FALLBACK}" '
            f'font-weight="bold" fill="{INK}">{title}</text>'
        )
    parts.append(f' <g transform="translate({left * 72:.2f} {top * 72:.2f})">')
    inner = svg[root.end() : svg.rindex("</svg>")]
    return svg[: root.start()] + "\n".join(parts) + inner + "</g>\n</svg>\n"


def save_svg(fig: Figure, name: str, extras: list[str] | None = None) -> Path:
    """Write ``slides/<name>.svg`` (a whole slide) and return its path.

    Args:
        fig: A figure created by :func:`new_figure`.
        name: File stem, e.g. ``slide_03_potency_recovery``.
        extras: Pre-built SVG groups to place on top of the
            figure, used for the RDKit structure drawings.
    """
    import io

    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    buffer = io.StringIO()
    fig.savefig(buffer, format="svg", metadata={"Date": None, "Creator": "Axiom_ATR_Tox_Slides"})
    plt.close(fig)

    svg = _apply_scripts(_normalise_text_styles(buffer.getvalue()))
    if extras:
        svg = inject(svg, extras)
    svg = _apply_scripts(_place_on_page(svg, TITLES.get(name)))
    assert_font_sizes(svg, name)

    path = FIGURES_DIR / f"{name}.svg"
    path.write_text(svg, encoding="utf-8")
    print(f"  wrote {path.name}")
    return path


# --------------------------------------------------------------------------- #
# RDKit structures as SVG groups
# --------------------------------------------------------------------------- #


_RDKIT_ROOT = re.compile(r"<svg\b([^>]*)>", re.S)
_VIEWBOX = re.compile(r"""viewBox=['"]\s*([-\d.]+)[ ,]+([-\d.]+)[ ,]+([-\d.]+)[ ,]+([-\d.]+)\s*['"]""")


def hex_to_rgb(colour: str) -> tuple[float, float, float]:
    """``'#D55E00'`` -> ``(0.835, 0.369, 0.0)``."""
    colour = colour.lstrip("#")
    return tuple(int(colour[i : i + 2], 16) / 255.0 for i in (0, 2, 4))  # type: ignore[return-value]


def unwrap(svg: str) -> tuple[str, float, float]:
    """Inner markup of an SVG document and its drawing size in user units."""
    root = _RDKIT_ROOT.search(svg)
    if root is None:
        raise ValueError("not an SVG document")
    box = _VIEWBOX.search(root.group(1))
    if box is None:
        raise ValueError("SVG root has no viewBox")
    width, height = float(box.group(3)), float(box.group(4))
    inner = svg[root.end() : svg.rindex("</svg>")]
    inner = re.sub(r"<metadata>.*?</metadata>", "", inner, flags=re.S)
    inner = re.sub(r"<!--.*?-->", "", inner, flags=re.S)
    return inner, width, height


def place(svg: str, x_in: float, y_in: float, width_in: float, group_id: str) -> str:
    """Wrap an SVG document as a group placed at (x, y) inches with the given width."""
    inner, width, _ = unwrap(svg)
    scale = width_in * 72.0 / width
    return (
        f'<g id="{group_id}" transform="translate({x_in * 72.0:.2f} {y_in * 72.0:.2f}) '
        f'scale({scale:.5f})">{inner}</g>'
    )


def inject(svg: str, groups: list[str]) -> str:
    """Insert groups just before the closing ``</svg>`` so they draw on top."""
    cut = svg.rindex("</svg>")
    return svg[:cut] + "\n".join(groups) + "\n" + svg[cut:]


def draw_structure(
    mol: Chem.Mol,
    *,
    size_px: tuple[int, int],
    highlight_atoms: dict[int, str] | None = None,
    highlight_bonds: dict[int, str] | None = None,
    bond_width: float = 2.0,
    font_px: int = 20,
    bond_length_px: float = 0.0,
    highlight_bond_width: float = 0.0,
) -> str:
    """Render a molecule to an SVG string in black and white, with optional highlights.

    Args:
        mol: Molecule with 2-D coordinates. Atoms carrying an ``atomLabel`` property
            (the R-group stubs) are drawn with that label.
        size_px: Drawing size in RDKit pixels; the aspect ratio is kept when placed.
        highlight_atoms: Atom index -> colour hex, drawn as a filled halo.
        highlight_bonds: Bond index -> colour hex.
        bond_width: Line width of the bonds, in RDKit pixels.
        font_px: Atom-label size, in RDKit pixels.
        highlight_bond_width: If positive, the band behind a highlighted bond is this many times the bond's
            line width (RDKit's default is 8).
        bond_length_px: If positive, every bond is drawn this long (RDKit pixels), so
            structures on one slide share a scale instead of each filling its canvas.
    """
    drawer = rdMolDraw2D.MolDraw2DSVG(*size_px)
    options = drawer.drawOptions()
    options.useBWAtomPalette()
    options.clearBackground = False
    options.bondLineWidth = bond_width
    options.fixedFontSize = font_px
    options.padding = 0.04
    options.fillHighlights = True
    options.highlightRadius = 0.42
    if bond_length_px > 0:
        options.fixedBondLength = bond_length_px
    if highlight_bond_width > 0:
        options.highlightBondWidthMultiplier = int(round(highlight_bond_width))

    atoms = highlight_atoms or {}
    bonds = highlight_bonds or {}
    drawer.DrawMolecule(
        mol,
        highlightAtoms=list(atoms),
        highlightAtomColors={i: hex_to_rgb(c) for i, c in atoms.items()},
        highlightBonds=list(bonds),
        highlightBondColors={i: hex_to_rgb(c) for i, c in bonds.items()},
    )
    drawer.FinishDrawing()
    return drawer.GetDrawingText()


def align_to(template: Chem.Mol, reference: Chem.Mol) -> None:
    """Give ``template`` 2-D coordinates that reuse ``reference``'s on the shared core.

    All the structures on one slide are then drawn in the same orientation, so the eye
    compares what differs instead of re-reading a rotated skeleton.
    """
    rdDepictor.SetPreferCoordGen(True)
    rdDepictor.GenerateDepictionMatching2DStructure(template, reference)


# --------------------------------------------------------------------------- #
# The t-SNE map view shared by slides 2, 5 and 8
# --------------------------------------------------------------------------- #


def map_limits(layout: pd.DataFrame) -> tuple[tuple[float, float], tuple[float, float]]:
    """``((xmin, xmax), (ymin, ymax))`` of the map, with a margin for the labels on the left."""
    x_range = float(layout["tsne_x"].max() - layout["tsne_x"].min())
    y_range = float(layout["tsne_y"].max() - layout["tsne_y"].min())
    return (
        (float(layout["tsne_x"].min()) - 0.10 * x_range, float(layout["tsne_x"].max()) + 0.04 * x_range),
        (float(layout["tsne_y"].min()) - 0.04 * y_range, float(layout["tsne_y"].max()) + 0.04 * y_range),
    )


def apply_view(ax, layout: pd.DataFrame) -> None:
    """Give ``ax`` the shared limits, stretch to the panel and hide the axes."""
    xlim, ylim = map_limits(layout)
    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim)
    ax.set_aspect("auto")
    ax.set_axis_off()


# --------------------------------------------------------------------------- #
# Free placement by inches (title slide, mechanism slide)
# --------------------------------------------------------------------------- #

AXIOM = "#009E73"  #: green: the colour for what Axiom selects (slides 1, 8, 9, 10)
FAIL = GARTISERTIB


def overlay(fig: Figure) -> Axes:
    """An axes over the whole canvas in inches from the top-left, so shapes are placed like text."""
    ax = fig.add_axes((0, 0, 1, 1))
    ax.set_xlim(0, SLIDE_W_IN)
    ax.set_ylim(SLIDE_H_IN, 0)
    ax.axis("off")
    return ax


def text(ax: Axes, x: float, y: float, content: str, *, size: int = SIZE_LABEL, weight: str = "normal",
         colour: str = INK, ha: str = "left", va: str = "center", spacing: float = 1.25) -> None:
    ax.text(x, y, content, fontsize=size, fontweight=weight, color=colour, ha=ha, va=va, linespacing=spacing)


# --------------------------------------------------------------------------- #
# Structure drawing shared by slides 1, 2, 8, 9 and 10
# --------------------------------------------------------------------------- #

#: Every structure is drawn with this bond length and label size (RDKit pixels), so all share one scale.
BOND_LENGTH_PX = 30.0
PX_PER_IN = 440 / 2.85
FONT_PX = 22
BOND_WIDTH = 2.2
#: The shared core without its attachment points; every drawing is aligned to it.
CORE_NO_R = "Nc1nn2cccnc2c1C(=O)Nc1cnccc1"

#: Atom indices of ``CORE_NO_R``: the pyrazolo[1,5-a]pyrimidine bicycle and the pyridine.
_BICYCLE_ATOMS = tuple(range(1, 10))
_PYRIDINE_ATOMS = tuple(range(13, 19))


def reference_core() -> Chem.Mol:
    """The bare core, drawn with the bicycle on the left and the pyridine on the right.

    This is the alignment target for every structure on the slide, so all of them share
    one orientation and the eye compares only what differs.
    """
    core = Chem.MolFromSmiles(CORE_NO_R)
    rdDepictor.SetPreferCoordGen(True)
    rdDepictor.Compute2DCoords(core)

    conformer = core.GetConformer()
    xy = np.array([[conformer.GetAtomPosition(i).x, conformer.GetAtomPosition(i).y] for i in range(core.GetNumAtoms())])
    direction = xy[list(_PYRIDINE_ATOMS)].mean(axis=0) - xy[list(_BICYCLE_ATOMS)].mean(axis=0)
    angle = -np.arctan2(direction[1], direction[0])
    rotation = np.array([[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]])
    centre = xy.mean(axis=0)
    rotated = (xy - centre) @ rotation.T

    # Flip the pyridine across its para axis (atom 13, the NH-bearing C3, to atom 16, C6)
    # so its C4/C5 substituents point away from the amide oxygen. Otherwise the R2 label
    # of the I-N, I-G, I-O and I-C drawings lands next to the carbonyl "O".
    axis_start, axis_end = rotated[13], rotated[16]
    axis = (axis_end - axis_start) / np.linalg.norm(axis_end - axis_start)
    for i in _PYRIDINE_ATOMS:
        offset = rotated[i] - axis_start
        along = np.dot(offset, axis) * axis
        rotated[i] = axis_start + along - (offset - along)

    for i, (x, y) in enumerate(rotated):
        conformer.SetAtomPosition(i, (float(x), float(y), 0.0))
    return core


def aligned(smiles: str, reference: Chem.Mol) -> tuple[Chem.Mol, list[int]]:
    """Parse ``smiles`` and draw it in the reference orientation.

    Returns:
        The molecule with coordinates, and the indices of the atoms that are *not* part
        of the shared core (the sub-series linker), for highlighting.
    """
    mol = Chem.MolFromSmiles(smiles)
    align_to(mol, reference)
    in_core = set(mol.GetSubstructMatch(reference))
    new_atoms = [a.GetIdx() for a in mol.GetAtoms() if a.GetIdx() not in in_core]
    return mol, new_atoms


def lighten(hex_colour: str, amount: float) -> str:
    """``hex_colour`` mixed with white; ``amount`` 0 keeps it, 1 gives white."""
    rgb = [int(hex_colour.lstrip("#")[i : i + 2], 16) for i in (0, 2, 4)]
    return "#" + "".join(f"{round(c + (255 - c) * amount):02x}" for c in rgb)



#: Colours of the designs (slides 9 and 10): the green of slide 8's selection, lightened.
DESIGN_GREEN = "#009E73"
DESIGN_SCAFFOLD = Chem.MolFromSmarts("Nc1nn2cc(CC#N)cnc2c1C(=O)Nc1cnccc1")  #: scaffold shared by all designs
DESIGN_HALO = lighten(DESIGN_GREEN, 0.6)  #: halo on the R2 residue, as slide 2 does for the linker
