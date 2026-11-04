"""Build every slide: ``python make_slides.py`` writes ``slides/*.svg`` and ``output/shortlist_23_compounds.csv``.

The designs of slide 9 are read from ``output/design``; ``python code/design.py`` recomputes them (about a minute).
"""

import importlib
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "code"))

SLIDES = (
    "slide_01_title", "slide_02_chemical_space", "slide_03_potency_recovery", "slide_04_validation",
    "slide_05_chemical_landscape", "slide_06_physicochemistry", "slide_07_mechanism", "slide_08_shortlist",
    "slide_09_designs", "slide_10_synthesis", "appendix_01_distributions",
)

if __name__ == "__main__":
    import load_data

    for name in SLIDES:
        getattr(importlib.import_module(name), name)()
    load_data.write_shortlist_csv()
