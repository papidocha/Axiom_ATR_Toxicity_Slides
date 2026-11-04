# Axiom ATR lead series: slide figures

Code and figures behind the slides on the predicted liver-toxicity profile of the 553-compound ATR-inhibitor lead series
(Axiom Bio's predictions, gartisertib and tuvusertib as references). `COMMENTS.md` explains what the slides leave out: which prediction
columns enter `general_tox`, and how the 23 recommended compounds (slide 8) differ from the 6 designed ones (slide 9).

```
pip install -r requirements.txt
python make_slides.py          # writes slides/*.svg and output/shortlist_23_compounds.csv (about 10 s)
python code/design.py          # optional: recomputes the designs of slide 9 (about 1 min; results are included)
```

## Layout

| Path | Content |
|---|---|
| `code/slide_NN_*.py`, `appendix_01_*.py` | One self-contained script per slide; each draws its figure and returns the numbers it shows |
| `code/load_data.py` | Reads the inputs; computes `general_tox`, descriptors, patent Ki bins, the shortlist rule and the predicted-versus-measured table |
| `code/design.py` | Virtual screen and XGBoost models behind slide 9 |
| `code/plot_utils.py` | Style, SVG page layout and titles, RDKit structure drawing shared by several slides |
| `data/` | Inputs (below) |
| `slides/` | The slide SVGs |
| `output/` | Other results: `shortlist_23_compounds.csv`, `design/` (from `design.py`, read by slides 9 and 10) and the t-SNE layout |

## Slides

| | Content |
|---|---|
| 1 | Title: gartisertib through design, make, test and Phase I |
| 2 | Chemical space of the 553 compounds, shared core, the four sub-series |
| 3 | Potency: graded Ki bins from the source patent against the spreadsheet's two labels |
| 4 | Definition of `general_tox`, predicted against measured EC, gartisertib and tuvusertib in every column |
| 5 | The chemical-space map colored by `general_tox`, Ki bin, basic nitrogen count and molecular weight |
| 6 | Molecular properties against `general_tox` (basic nitrogen count is the strongest) |
| 7 | Bioactivation hypothesis, what failed in the clinic, what to test next |
| 8 | The 23 recommended existing compounds |
| 9 | The six designed compounds (virtual screen, XGBoost) |
| 10 | How three of the designs would be made |
| Appendix | Distribution of all 20 predicted EC columns |

## Inputs (`data/`, as provided at the start of the study)

| File | Content |
|---|---|
| `US 20140163000 A1_553.xlsx` | The 553 compounds with Axiom Bio's predictions, as delivered |
| `Experimental_data_Gartisertib_Tuvusertib.xlsx` | Measured dose-response for the two reference compounds |
| `gartisertib_prediction_column_mapping.csv` | Axiom Bio's mapping of prediction columns to experimental readouts (written into `READOUT_MAP` in `load_data.py`) |
| `patent_ki_bins.csv` | Ki bins read from Table 5 of US 9,340,546 B2 |

`output/chemical_space_embedding.parquet` is the seeded t-SNE layout of the map; `load_data.tsne_coordinates` recomputes it if missing.

## Conventions

* `_ld20` / `_ld50` columns are EC20 / EC50 effect concentrations in µM, not lethal doses. Predictions at or above the
  200 µM ceiling are censored (no effect reached), not measurements.
* Compound_I-C-50 is drawn without its aromatic pyrazole circle in the patent file; the corrected SMILES is applied in code
  (`SMILES_CORRECTIONS`), the data file is untouched.
