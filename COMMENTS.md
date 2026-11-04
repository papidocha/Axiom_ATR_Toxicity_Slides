# Comments on slides 4, 8 and 9

Only the points that cannot be read from the slides are mentiond here. "Predicted" means Axiom Bio's GNN output unless stated otherwise.

## Slide 4: what goes into general_tox

`general_tox` is built from the GNN's predicted effect concentrations (EC20 and EC50, in µM, columns ending `_ld20` and `_ld50`;
not lethal doses). **Panel C shows exactly the 15 columns that enter the score, and no others.**

| Assay (prediction column) | EC20 | EC50 | Why |
|---|---|---|---|
| Cell viability | used | used | |
| LDH release | used | **not used** | EC50: 547 of 553 predictions sit at the 200 µM ceiling |
| Cell death | used | **not used** | EC50: 420 at the ceiling |
| Nuclei size | used | **not used** | EC50: 519 at the ceiling |
| Cytoplasm area | used | used | |
| Mitochondria | used | **not used** | EC50: 437 at the ceiling |
| ER stress | used | used | |
| ROS | used | **not used** | EC50: 292 of 553 (53 %) at the ceiling; excluded by decision, because it would also leave gartisertib and tuvusertib tied |
| Bioactivity 1 | used | used | |
| Bioactivity 2 | used | used | |
| Halos | **not used** | **not used** | Excluded after discussion with Chengpeng|

* **Excluded columns (5 of the 20 EC columns, plus halos):** `ldh_ld50`, `cell_death_ld50`, `nuclei_size_ld50`, `mitochondria_ld50`,
  `ros_ld50`. For the first four, 76 to 99 % of the values are at the 200 µM ceiling, so they have no spread and carry no ranking
  information. They appear in the appendix, marked "excl.", but not on slide 4.
* **Used columns (15):** `cell_viability_ld20/_ld50`, `ldh_ld20`, `cell_death_ld20`, `nuclei_size_ld20`, `cytoplasm_area_ld20/_ld50`,
  `mitochondria_ld20`, `er_stress_ld20/_ld50`, `ros_ld20`, `bioactivity1_ld20/_ld50`, `bioactivity2_ld20/_ld50`.
* **Construction:** log10 of each EC (clipped at 200 µM) is z-scored over the 553 compounds, so a larger EC (safer) gives a larger z;
  the z-scores are averaged within each assay family, and the nine families are averaged. Bioactivity 1 and 2 count as one family,
  hence nine families for ten assays. Higher = safer.
* **Reading panel C:** percentiles are ranks within the 553 compounds. Gartisertib is below tuvusertib in all 15 columns
  (general_tox 0.46 against 0.97). Hollow triangles are tuvusertib values at the ceiling.
* **Panel B** is only a sanity check: two compounds, both likely in the model's training set; predicted values are on average about 6.8-fold
  safer than measured. This is why percentiles, not absolute µM values, are used for comparisons.

## Slides 8 and 9: two different kinds of recommendation

| | Slide 8: recommended candidates | Slide 9: designed compounds |
|---|---|---|
| What | 23 compounds that **already exist** in the patent series | 6 compounds that have **not been made** |
| Basis | Axiom Bio's GNN (`general_tox`) and the patent's measured Ki bins, used directly | A surrogate model trained on the series, because the GNN cannot be run on new structures from my side |
| Data | `output/shortlist_23_compounds.csv` (all 23: rank, ID, sub-series, Ki bin, general_tox and percentile, basic N, MW, cLogP, SMILES) | `output/design/design_picks_xgb.csv` (six designs) and `output/design/design_candidates_xgb.csv` (all 2,125 ranked candidates) |

### Slide 8: selection rule (existing patent space)

Predicted safety first, measured patent potency second. A compound is recommended if it is in Ki bin ++ (0.01 to 1 µM) and its
`general_tox` is at least tuvusertib's (+1.47), or in bin +++ (Ki < 0.01 µM) and its `general_tox` is in the top 10 % of the series
(≥ +1.18, the 90th percentile). Result: 23 compounds (17 in +++, 6 in ++), all without a basic nitrogen, none of them gartisertib.
The values are Axiom Bio's predictions and the patent's measurements, nothing is modeled by Sven.

### Slide 9: virtual screen and XGBoost model

Reproduced by `python code/design.py` (about a minute).

* **Why a surrogate model:** I don't have access to run Axiom Bio's GNN beyond the structures in provided with the task, so a model was trained to
  reproduce its `general_tox` output (and the patent potency) from the structure, and used to score new compounds.
* **Virtual library:** built from the series' own pieces. Every observed R1 substituent (7) × every observed R2 tail (401) × R3 in
  {H, F, Cl}, with R4 = H, minus the compounds that were already made. No new fragment is invented, so each design is a new combination
  of known parts. All designs keep the shared core, so they remain one coupling step from intermediates the patent already makes (slide 10).
* **Evidence filter:** a candidate is kept only if its combination of R1, R3 and sub-series (I-N, I-G, I-O, and I-C) has been made at least twice, so that
  potency is interpolated between measured neighbors, not guessed. This leaves 2,125 unmade candidates.
* **Features and learner:** Morgan count fingerprint of the whole molecule (radius 2, 1,024 bits), XGBoost regression, the same for both the potency and safety endpoint.
* **Training labels:**
  * *Safety:* `general_tox` of the 552 compounds that decompose into the R-group scheme, that is, **the GNN's predictions, not experimental
    toxicity**. Any error of Axiom Bio's GNN passes into the design.
  * *Potency:* an ordinal score from the **experimental** patent Ki bin: + (1 to 5 µM) = −1, ++ (0.01 to 1 µM) = 0, +++ (< 0.01 µM) = +1,
    for the 543 compounds with a bin. The model predicts a number on this scale, and boosted trees keep it within about −1 to +1.
* **Potency on the slide:** the number under each structure is the model's predicted potency score on the scale of the y-axis, where
  0 is the ++ bin and +1 is the +++ bin; it is a coding of ordered bins, not a Ki value. D1 to D3 (+0.95 to +0.98) are predicted +++.
  D4 to D6 (+0.45 to +0.54) lie between ++ and +++: the model cannot say which bin, so these trade potency for predicted safety.
* **Accuracy (5-fold cross-validation within the series):** `general_tox` R² 0.87, RMSE 0.33; potency score R² 0.33, but AUC 0.93 for
  separating +++ from the rest (0.80 within the 40 compounds with R1 = CH2CN). The folds are random, so neighbors of every test compound
  are in the training set: this shows the model interpolates within the series, not how it does on truly new chemistry.

## What would make this better

The designs are only as good as the surrogate, which learned from the GNN's outputs and not from toxicity measurements. Direct access to
Axiom Bio's GNN would allow scoring the virtual library with the real model and would remove the surrogate's error. Better still would be to
**retrain the GNN with more compounds from this series that have experimental in-house toxicity data**: measured data exist for only
two of the 553 compounds (gartisertib and tuvusertib), too few to check, let alone tune, the predictions inside this chemical series.
