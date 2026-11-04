"""Virtual screen and designs of slide 9: six unmade compounds predicted safe and potent.

    python code/design.py

Axiom Bio's model cannot be run on structures that are not in the delivered file, so two XGBoost
regressors are trained on the series from the Morgan count fingerprint (radius 2, 1,024 bits):

* safety: ``general_tox`` of the compounds, which is Axiom Bio's own prediction, not a measurement;
* potency: an ordinal score from the experimental patent Ki bin (+ = -1, ++ = 0, +++ = +1).

The virtual library is every R1 x R2 tail x R3 in {H, F, Cl} combination of the series' own fragments
that has not been made. Each candidate gets a predicted general_tox and a predicted potency score. Candidates
with general_tox at the 90th percentile of the series or above are ranked by the sum of the two predictions,
and six are picked greedily.
Writes ``output/design/design_picks_xgb.csv`` and ``output/design/design_candidates_xgb.csv``.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from rdkit import Chem, DataStructs
from rdkit.Chem import Descriptors, rdFingerprintGenerator
from sklearn.model_selection import KFold
from xgboost import XGBRegressor

import load_data as D

CORE = "Nc1nn2cc([*:1])cnc2c1C(=O)Nc1cnc([*:4])c([*:3])c1[*:2]"
R1_OPTIONS = ("F[*:1]", "Cl[*:1]", "N#CC[*:1]", "CC(C#N)[*:1]", "CC(C)(C#N)[*:1]", "CCC(C#N)[*:1]", "N#CC1([*:1])CC1")
R3_OPTIONS = ("[H][*:3]", "F[*:3]", "Cl[*:3]")
R4_H = "[H][*:4]"
POTENCY_SCORE = {"+": -1.0, "++": 0.0, "+++": 1.0}
MIN_SUPPORT = 2  # combination of R1, R3 and sub-series made at least this often, so potency is interpolated
N_PICKS, MAX_PER_SUBSERIES, MAX_SIMILARITY = 6, 3, 0.85
SEED = D.RANDOM_SEED
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


def cross_validate(x: np.ndarray, y: np.ndarray) -> tuple[float, float]:
    """RMSE and R^2 of 5-fold cross-validation (3 shuffles, averaged)."""
    rmse, r2 = [], []
    for repeat in range(3):
        predicted = np.zeros(len(y))
        for train_idx, test_idx in KFold(5, shuffle=True, random_state=repeat).split(y):
            predicted[test_idx] = regressor(SEED + repeat).fit(x[train_idx], y[train_idx]).predict(x[test_idx])
        rmse.append(np.sqrt(((y - predicted) ** 2).mean()))
        r2.append(1 - ((y - predicted) ** 2).sum() / ((y - y.mean()) ** 2).sum())
    return float(np.mean(rmse)), float(np.mean(r2))


def score_library(train: pd.DataFrame, library: pd.DataFrame) -> pd.DataFrame:
    """Predicted general_tox and potency of every candidate, and how well the combination has been tried before."""
    x_train = fingerprints(train["flat_smiles"])
    known = train["ki_bin"].notna().to_numpy()
    y_safety = train["general_tox"].to_numpy()
    y_potency = train.loc[known, "ki_bin"].map(POTENCY_SCORE).to_numpy()
    for name, x, y in (("general_tox", x_train, y_safety), ("potency score", x_train[known], y_potency)):
        rmse, r2 = cross_validate(x, y)
        print(f"{name}: 5-fold cross-validation RMSE {rmse:.2f}, R2 {r2:.2f}")
    safety = regressor(SEED).fit(x_train, y_safety)
    potency = regressor(SEED).fit(x_train[known], y_potency)

    x_library = fingerprints(library["flat_smiles"])
    scored = library.copy()
    scored["tox"] = safety.predict(x_library)
    scored["potency"] = potency.predict(x_library)
    scored["score"] = scored["tox"] + scored["potency"]

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
    """Greedy pick by predicted general_tox + potency among candidates with measured neighbors and general_tox at the
    series' 90th percentile or above: at most three per sub-series, no two more similar than a Morgan Tanimoto of 0.85."""
    cutoff, _ = D.thresholds(D.candidate_table())
    pool = scored[(scored["n_support"] >= MIN_SUPPORT) & (scored["tox"] >= cutoff)].sort_values("score", ascending=False)
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
    picks[["design", "subseries", "R2", "smiles", "tox", "potency", "score",
           "n_support", "n_support_top", "parent", "parent_ki_bin", "parent_tox", "n_basic_nitrogen", "mw"]].round(3).to_csv(
        D.DESIGN_DIR / "design_picks_xgb.csv", index=False)
    eligible = scored[scored["n_support"] >= MIN_SUPPORT].sort_values("score", ascending=False)
    eligible[["subseries", "tox", "potency", "score"]].round(3).to_csv(
        D.DESIGN_DIR / "design_candidates_xgb.csv", index=False)
    print(f"{len(library):,} virtual compounds, {len(eligible):,} with measured neighbors; designs: {', '.join(picks['design'])}")


if __name__ == "__main__":
    main()
