"""Virtual screen and designs of slide 9: six unmade compounds predicted safe and potent.

    python code/design.py

Axiom Bio's model cannot be run on structures that are not in the delivered file, so two XGBoost
regressors are trained on the series from the Morgan count fingerprint (radius 2, 1,024 bits):

* safety: ``general_tox`` of the compounds, which is Axiom Bio's own prediction, not a measurement;
* potency: an ordinal score from the experimental patent Ki bin (+ = -1, ++ = 0, +++ = +1).

The virtual library is every R1 x R2 tail x R3 in {H, F, Cl} combination of the series' own fragments
that has not been made. Each candidate gets a predicted score and a spread (12 bootstrap refits combined
with the cross-validated RMSE). Two probabilities follow, assuming a normal distribution: P(+++) that the
potency score is above +0.5 (midway between ++ and +++), and P(safe) that general_tox reaches the 90th
percentile of the series. Candidates are ranked by their product, and six are picked greedily.
Writes ``output/design/design_picks_xgb.csv`` and ``output/design/design_candidates_xgb.csv``.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from rdkit import Chem, DataStructs
from rdkit.Chem import Descriptors, rdFingerprintGenerator
from scipy.stats import norm
from sklearn.model_selection import KFold
from xgboost import XGBRegressor

import load_data as D

CORE = "Nc1nn2cc([*:1])cnc2c1C(=O)Nc1cnc([*:4])c([*:3])c1[*:2]"
R1_OPTIONS = ("F[*:1]", "Cl[*:1]", "N#CC[*:1]", "CC(C#N)[*:1]", "CC(C)(C#N)[*:1]", "CCC(C#N)[*:1]", "N#CC1([*:1])CC1")
R3_OPTIONS = ("[H][*:3]", "F[*:3]", "Cl[*:3]")
R4_H = "[H][*:4]"
POTENCY_SCORE = {"+": -1.0, "++": 0.0, "+++": 1.0}
TOP_CUT = 0.5  # a score above it counts as predicted +++
MIN_SUPPORT = 2  # combination of R1, R3 and sub-series made at least this often, so potency is interpolated
N_PICKS, MAX_PER_SUBSERIES, MAX_SIMILARITY = 6, 3, 0.85
N_BOOTSTRAP, SEED = 12, D.RANDOM_SEED
#: Left out of the training set, as in the study the slide shows: its R-group decomposition failed before the SMILES correction.
LEFT_OUT = "Compound_I-C-50"


def assemble(r1: str, r2: str, r3: str, r4: str = R4_H) -> Chem.Mol | None:
    """Core plus four R-group SMILES; ``None`` if a fragment does not parse."""
    mol = Chem.MolFromSmiles(CORE)
    for fragment in (r1, r2, r3, r4):
        group = Chem.MolFromSmiles(fragment)
        if group is None:
            return None
        mol = Chem.molzip(mol, group)
    mol = Chem.RemoveHs(mol)
    Chem.SanitizeMol(mol)
    return mol


def flat_smiles(mol: Chem.Mol) -> str:
    return Chem.MolToSmiles(mol, isomericSmiles=False)


def training_table() -> pd.DataFrame:
    """The compounds with R-groups, sub-series, Ki bin, general_tox and flat SMILES."""
    candidates = D.candidate_table()
    names = [n for n in candidates.index if n != LEFT_OUT]
    rgroups, _ = D.rgroup_decomposition(names)
    frame = rgroups.join(candidates[["subseries", "ki_bin", "general_tox", "smiles"]])
    frame["flat_smiles"] = [flat_smiles(Chem.MolFromSmiles(s)) for s in frame["smiles"]]
    return frame


def virtual_library(train: pd.DataFrame) -> pd.DataFrame:
    """Every observed R2 tail combined with the R1 and R3 options, minus the compounds already made."""
    tails = train.groupby("R2")["subseries"].agg(lambda s: s.mode().iat[0]).rename("subseries").reset_index()
    made = set(train["flat_smiles"])
    pattern = D.basic_nitrogen_pattern()
    rows = []
    for r1 in R1_OPTIONS:
        for r3 in R3_OPTIONS:
            for r2, subseries in zip(tails["R2"], tails["subseries"]):
                mol = assemble(r1, r2, r3)
                if mol is None or "[*:3]" in r2:
                    continue
                smiles = flat_smiles(mol)
                if smiles not in made:
                    rows.append({"R1": r1, "R2": r2, "R3": r3, "R4": R4_H, "subseries": subseries, "flat_smiles": smiles,
                                 "mw": Descriptors.MolWt(mol), "n_basic_nitrogen": len(mol.GetSubstructMatches(pattern))})
    return pd.DataFrame(rows).drop_duplicates("flat_smiles").reset_index(drop=True)


_FINGERPRINT = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=1024)


def fingerprints(smiles: pd.Series) -> np.ndarray:
    return np.array([_FINGERPRINT.GetCountFingerprintAsNumPy(Chem.MolFromSmiles(s)) for s in smiles], dtype=float)


def regressor(seed: int) -> XGBRegressor:
    return XGBRegressor(n_estimators=300, learning_rate=0.05, max_depth=4, min_child_weight=2,
                        subsample=0.8, colsample_bytree=0.5, reg_lambda=1.0, n_jobs=-1, random_state=seed)


def fit_endpoint(x: np.ndarray, y: np.ndarray) -> tuple[list[XGBRegressor], float]:
    """Bootstrap refits and the cross-validated RMSE (5 folds, 3 repeats) of one endpoint."""
    predictions = np.zeros((3, len(y)))
    for repeat in range(3):
        for train_idx, test_idx in KFold(5, shuffle=True, random_state=repeat).split(y):
            predictions[repeat, test_idx] = regressor(SEED + repeat).fit(x[train_idx], y[train_idx]).predict(x[test_idx])
    rmse = float(np.mean([np.sqrt(((y - p) ** 2).mean()) for p in predictions]))
    rng = np.random.default_rng(SEED)
    fits = []
    for k in range(N_BOOTSTRAP):
        idx = rng.integers(0, len(y), len(y))
        fits.append(regressor(SEED + k).fit(x[idx], y[idx]))
    return fits, rmse


def predict(endpoint: tuple[list[XGBRegressor], float], x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Mean prediction and its spread (bootstrap variance plus cross-validated RMSE squared)."""
    fits, rmse = endpoint
    stack = np.vstack([m.predict(x) for m in fits])
    return stack.mean(axis=0), np.sqrt(stack.var(axis=0) + rmse**2)


def score_library(train: pd.DataFrame, library: pd.DataFrame) -> pd.DataFrame:
    """Predicted general_tox and potency of every candidate, with P(+++), P(safe) and their product."""
    x_train = fingerprints(train["flat_smiles"])
    known = train["ki_bin"].notna().to_numpy()
    safety = fit_endpoint(x_train, train["general_tox"].to_numpy())
    potency = fit_endpoint(x_train[known], train.loc[known, "ki_bin"].map(POTENCY_SCORE).to_numpy())

    x_library = fingerprints(library["flat_smiles"])
    scored = library.copy()
    scored["tox"], scored["tox_sd"] = predict(safety, x_library)
    scored["potency"], scored["potency_sd"] = predict(potency, x_library)
    scored["p_top"] = norm.sf((TOP_CUT - scored["potency"]) / scored["potency_sd"])
    scored["p_ge2"] = norm.sf((-TOP_CUT - scored["potency"]) / scored["potency_sd"])
    cutoff, _ = D.thresholds(D.candidate_table())
    scored["p_safe"] = norm.cdf((scored["tox"] - cutoff) / scored["tox_sd"])
    scored["p_joint"] = scored["p_safe"] * scored["p_top"]

    # Evidence: how often the combination of R1, R3 and sub-series has been made, and how often as +++.
    key = train["R1"] + "|" + train["R3"] + "|" + train["subseries"]
    library_key = scored["R1"] + "|" + scored["R3"] + "|" + scored["subseries"]
    scored["n_support"] = library_key.map(key.value_counts()).fillna(0).astype(int)
    scored["n_support_top"] = library_key.map(key[train["ki_bin"] == "+++"].value_counts()).fillna(0).astype(int)

    # The made compound with the same tail (same R3 if possible) and the highest general_tox: the pair the design extends.
    by_tail = {tail: group for tail, group in train.groupby("R2")}
    parents = []
    for tail in scored["R2"]:
        group = by_tail[tail]
        group = group[group["R3"] == R3_OPTIONS[0]] if (group["R3"] == R3_OPTIONS[0]).any() else group
        parents.append(group.sort_values("general_tox", ascending=False).iloc[0])
    scored["parent"] = [p.name for p in parents]
    scored["parent_ki_bin"] = [p["ki_bin"] for p in parents]
    scored["parent_tox"] = [p["general_tox"] for p in parents]
    return scored


def pick_designs(scored: pd.DataFrame) -> pd.DataFrame:
    """Greedy pick by P(both) among candidates with measured neighbors: at most three per sub-series,
    no two more similar than a Morgan Tanimoto of 0.85."""
    pool = scored[scored["n_support"] >= MIN_SUPPORT].sort_values("p_joint", ascending=False)
    generator = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048)
    chosen, chosen_fps, per_subseries = [], [], {}
    for idx, row in pool.iterrows():
        if per_subseries.get(row["subseries"], 0) >= MAX_PER_SUBSERIES:
            continue
        fp = generator.GetFingerprint(Chem.MolFromSmiles(row["flat_smiles"]))
        if any(DataStructs.TanimotoSimilarity(fp, other) > MAX_SIMILARITY for other in chosen_fps):
            continue
        chosen.append(idx)
        chosen_fps.append(fp)
        per_subseries[row["subseries"]] = per_subseries.get(row["subseries"], 0) + 1
        if len(chosen) == N_PICKS:
            break
    picks = scored.loc[chosen].copy()
    picks.insert(0, "design", [f"D{i + 1}" for i in range(len(picks))])
    picks["smiles"] = [Chem.MolToSmiles(assemble(r.R1, r.R2, r.R3, r.R4)) for r in picks.itertuples()]
    return picks


def main() -> None:
    train = training_table()
    library = virtual_library(train)
    scored = score_library(train, library)
    picks = pick_designs(scored)
    D.DESIGN_DIR.mkdir(parents=True, exist_ok=True)
    picks[["design", "subseries", "R2", "smiles", "tox", "tox_sd", "potency", "potency_sd", "p_top", "p_ge2", "p_safe", "p_joint",
           "n_support", "n_support_top", "parent", "parent_ki_bin", "parent_tox", "n_basic_nitrogen", "mw"]].round(3).to_csv(
        D.DESIGN_DIR / "design_picks_xgb.csv", index=False)
    eligible = scored[scored["n_support"] >= MIN_SUPPORT].sort_values("p_joint", ascending=False)
    eligible[["subseries", "tox", "potency", "p_top", "p_ge2", "p_joint"]].round(3).to_csv(
        D.DESIGN_DIR / "design_candidates_xgb.csv", index=False)
    print(f"{len(library):,} virtual compounds, {len(eligible):,} with measured neighbors; designs: {', '.join(picks['design'])}")


if __name__ == "__main__":
    main()
