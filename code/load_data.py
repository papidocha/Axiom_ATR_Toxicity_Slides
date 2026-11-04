"""Input data for the slides: the 553-compound series, general_tox, patent Ki bins, descriptors,
the shortlist rule and the predicted-versus-measured comparison.

Column names ending in ``_ld20`` / ``_ld50`` are EC20 / EC50 effect concentrations in µM
(predicted by Axiom Bio's model), not lethal doses.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd
from rdkit import Chem, DataStructs, RDLogger
from rdkit.Chem import Descriptors, rdFingerprintGenerator, rdMolDescriptors
from scipy import stats

RDLogger.DisableLog("rdApp.*")

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
OUTPUT_DIR = ROOT / "output"
SLIDES_DIR = ROOT / "slides"
LEAD_SERIES_XLSX = DATA_DIR / "US 20140163000 A1_553.xlsx"
EXPERIMENTAL_XLSX = DATA_DIR / "Experimental_data_Gartisertib_Tuvusertib.xlsx"
PATENT_KI_CSV = DATA_DIR / "patent_ki_bins.csv"
DESIGN_DIR = OUTPUT_DIR / "design"  #: the designs of slide 9, written by design.py
EMBEDDING_FILE = OUTPUT_DIR / "chemical_space_embedding.parquet"

RANDOM_SEED = 20251013
CEILING_UM = 200.0  #: predictions at or above it are right-censored ("no effect reached")

GARTISERTIB = "Compound_I-G-32"
TUVUSERTIB = "Compound_I-C-79"
REFERENCE_COMPOUNDS = {
    GARTISERTIB: {
        "label": "Gartisertib",
        "smiles": "NC1=NN2C=C(F)C=NC2=C1C(=O)NC1=C(N2CCC(CC2)C(=O)N2CCN(CC2)C2COC2)C(F)=CN=C1",
    },
    TUVUSERTIB: {
        "label": "Tuvusertib",
        "smiles": "CN1C=NC=C1C2=C(F)C=NC=C2NC(=O)C3=C4N=CC(F)=CN4N=C3N",
    },
}
SUBSERIES = ("I-N", "I-G", "I-O", "I-C")

# --------------------------------------------------------------------------- #
# Prediction columns and general_tox
# --------------------------------------------------------------------------- #

ASSAYS = (
    "cell_viability", "ldh", "cell_death", "nuclei_size", "cytoplasm_area",
    "mitochondria", "er_stress", "ros", "bioactivity1", "bioactivity2",
)
TOX_COLUMNS = tuple(f"{assay}_{level}" for assay in ASSAYS for level in ("ld20", "ld50"))
LD20_COLUMNS = tuple(c for c in TOX_COLUMNS if c.endswith("_ld20"))
LD50_COLUMNS = tuple(c for c in TOX_COLUMNS if c.endswith("_ld50"))
#: Assay families of general_tox: bioactivity 1 and 2 are two normalizations of one measurement.
FAMILIES = {
    "bioactivity": ("bioactivity1", "bioactivity2"),
    "cell_viability": ("cell_viability",), "er_stress": ("er_stress",),
    "cytoplasm_area": ("cytoplasm_area",), "ros": ("ros",), "ldh": ("ldh",),
    "cell_death": ("cell_death",), "nuclei_size": ("nuclei_size",), "mitochondria": ("mitochondria",),
}
IQR_MIN = 0.1  #: a column enters general_tox only if the IQR of its log10 values exceeds this
MANUAL_EXCLUSIONS = ("ros_ld50",)  #: left out as well: 53 % at the ceiling, ties the two references

#: Compound_I-C-50 is drawn without the aromatic pyrazole circle in the data file and the patent.
SMILES_CORRECTIONS = {
    "Compound_I-C-50": (
        "NC1NN2C=C(F)C=NC2C1C(=O)NC1=C(C=CN=C1)C1=CC=C(C=C1F)C(=O)N1CCN2CCC1CC2",
        "Nc1nn2cc(F)cnc2c1C(=O)Nc1cnccc1-c1ccc(cc1F)C(=O)N1CCN2CCC1CC2",
    ),
}
CORE_SMARTS = "Nc1nn2cc(*)cnc2c1C(=O)Nc1cnccc1"
#: Basic nitrogen: protonatable at physiological pH. Excludes amides, sulfonamides, N-aryl, nitriles, imines, aromatic N.
BASIC_NITROGEN_SMARTS = "[NX3;H2,H1,H0;!$(NC=O);!$(NS=O);!$(N-a);!$(N#*);!$(N=*);!a]"

POTENCY_ACTIVE, POTENCY_UNKNOWN = "ACTIVE", "UNKNOWN"
POTENCY_ORDER = (POTENCY_ACTIVE, POTENCY_UNKNOWN)


@dataclass(frozen=True)
class LeadSeries:
    """The series with its censoring mask.

    Attributes:
        data: One row per compound: name, compound_id, compound_id_alt, subseries, smiles, spreadsheet
            potency class, the 20 columns in µM clipped at the ceiling, general_tox and its percentile.
        censored: True where a prediction is at or above the ceiling.
        raw_values: The 20 columns as in the file.
        log_values: log10 of the clipped columns.
        included, excluded: Columns that enter / do not enter general_tox.
    """

    data: pd.DataFrame
    censored: pd.DataFrame
    raw_values: pd.DataFrame
    log_values: pd.DataFrame
    included: tuple[str, ...]
    excluded: tuple[str, ...]


def _canonical(smiles: str) -> str:
    return Chem.MolToSmiles(Chem.MolFromSmiles(smiles))


def _general_tox(raw_values: pd.DataFrame) -> tuple[pd.Series, tuple[str, ...], tuple[str, ...]]:
    """general_tox (higher = safer): z-scored log10 EC clipped at the ceiling, averaged within each
    assay family (EC20 and EC50 columns together) and then over the nine families.

    Returns:
        ``(score, included_columns, excluded_columns)``.
    """
    log_values = np.log10(raw_values[list(TOX_COLUMNS)].clip(upper=CEILING_UM))
    iqr = log_values.quantile(0.75) - log_values.quantile(0.25)
    included = tuple(c for c in TOX_COLUMNS if iqr[c] > IQR_MIN and c not in MANUAL_EXCLUSIONS)
    excluded = tuple(c for c in TOX_COLUMNS if c not in included)
    z = (log_values - log_values.mean()) / log_values.std(ddof=0)
    family_scores = {}
    for family, assays in FAMILIES.items():
        per_level = []
        for level in ("ld20", "ld50"):
            members = [f"{a}_{level}" for a in assays if f"{a}_{level}" in included]
            if members:
                per_level.append(z[members].mean(axis=1))
        family_scores[family] = pd.concat(per_level, axis=1).mean(axis=1)
    return pd.DataFrame(family_scores).mean(axis=1), included, excluded


@lru_cache(maxsize=1)
def load_lead_series() -> LeadSeries:
    """Read the 553 compounds, correct the one bad SMILES and compute general_tox."""
    raw = pd.read_excel(LEAD_SERIES_XLSX, sheet_name="toxicity_predictions").dropna(how="all").reset_index(drop=True)
    raw = raw.drop(columns=["halos_ld20", "halos_ld50"])  # halos: excluded on vendor advice (see COMMENTS.md)
    assert len(raw) == 553

    # Nine rows carry two patent IDs ("Compound_A; Compound_B"): the first is the name.
    parts = raw["name"].str.split(";")
    primary = parts.str[0].str.strip()
    alt = parts.str[1:].map(lambda rest: rest[0].strip().replace("Compound_", "") if rest else None)
    frame = pd.concat([pd.DataFrame({
        "name": primary,
        "compound_id": primary.str.replace("Compound_", "", regex=False),
        "compound_id_alt": alt,
    }), raw.drop(columns=["name"])], axis=1)
    for name, (as_delivered, corrected) in SMILES_CORRECTIONS.items():
        row = frame.index[frame["name"] == name][0]
        assert _canonical(frame.at[row, "smiles"]) == _canonical(as_delivered), name
        frame.at[row, "smiles"] = corrected
    frame["subseries"] = frame["compound_id"].str.extract(r"^(I-[NGCO])")[0]
    label = frame["atr_binding_potency"].str.strip()
    frame["potency_class"] = pd.Categorical(
        np.where(label == "< 20 nM", POTENCY_ACTIVE, POTENCY_UNKNOWN), categories=list(POTENCY_ORDER), ordered=True
    )

    raw_values = frame[list(TOX_COLUMNS)].copy()
    censored = raw_values >= CEILING_UM
    frame[list(TOX_COLUMNS)] = raw_values.clip(upper=CEILING_UM)
    frame["n_censored"] = censored.sum(axis=1)
    score, included, excluded = _general_tox(raw_values)
    frame["general_tox"] = score
    frame["general_tox_pct"] = score.rank(pct=True)
    return LeadSeries(frame, censored, raw_values, np.log10(frame[list(TOX_COLUMNS)]), included, excluded)


def percentile(values: pd.Series) -> pd.Series:
    """Fractional rank in [0, 1]; ties share the mid-rank."""
    return values.rank(pct=True)


def spearman(x, y) -> tuple[float, float]:
    """Spearman rho and p-value, dropping pairs with a missing value."""
    pairs = pd.DataFrame({"x": np.asarray(x, float), "y": np.asarray(y, float)}).dropna()
    result = stats.spearmanr(pairs["x"], pairs["y"])
    return float(result.statistic), float(result.pvalue)


def percentile_profile(series: LeadSeries) -> pd.DataFrame:
    """Percentile of gartisertib and tuvusertib in every general_tox column (first row: general_tox)."""
    log_values = series.log_values.set_axis(series.data["name"])
    censored = series.censored.set_axis(series.data["name"])
    percentiles = log_values.apply(percentile)
    rows = {}
    for column in series.included:
        rows[column] = {
            "gartisertib": float(percentiles.at[GARTISERTIB, column]),
            "tuvusertib": float(percentiles.at[TUVUSERTIB, column]),
            "censored_gartisertib": bool(censored.at[GARTISERTIB, column]),
            "censored_tuvusertib": bool(censored.at[TUVUSERTIB, column]),
        }
    data = series.data.set_index("name")
    rows["general_tox"] = {
        "gartisertib": float(data.at[GARTISERTIB, "general_tox_pct"]),
        "tuvusertib": float(data.at[TUVUSERTIB, "general_tox_pct"]),
        "censored_gartisertib": False,
        "censored_tuvusertib": False,
    }
    return pd.DataFrame.from_dict(rows, orient="index")


# --------------------------------------------------------------------------- #
# Structures: molecules, descriptors, R-groups, chemical space
# --------------------------------------------------------------------------- #


@lru_cache(maxsize=1)
def load_molecules() -> dict[str, Chem.Mol]:
    """RDKit molecules keyed by compound name."""
    series = load_lead_series()
    return {name: Chem.MolFromSmiles(smiles) for name, smiles in zip(series.data["name"], series.data["smiles"])}


@lru_cache(maxsize=1)
def basic_nitrogen_pattern() -> Chem.Mol:
    return Chem.MolFromSmarts(BASIC_NITROGEN_SMARTS)


@lru_cache(maxsize=1)
def load_descriptors() -> pd.DataFrame:
    """The descriptors the slides use, one row per compound."""
    pattern = basic_nitrogen_pattern()
    rows = [
        {
            "mw": Descriptors.MolWt(mol),
            "clogp": Descriptors.MolLogP(mol),
            "tpsa": Descriptors.TPSA(mol),
            "hbd": rdMolDescriptors.CalcNumHBD(mol),
            "rotatable_bonds": rdMolDescriptors.CalcNumRotatableBonds(mol),
            "fsp3": rdMolDescriptors.CalcFractionCSP3(mol),
            "n_basic_nitrogen": len(mol.GetSubstructMatches(pattern)),
        }
        for mol in load_molecules().values()
    ]
    return pd.DataFrame(rows, index=pd.Index(list(load_molecules()), name="name"))


def rgroup_decomposition(names: list[str] | None = None) -> tuple[pd.DataFrame, list[str]]:
    """R-group decomposition against the shared core.

    Returns:
        ``(table, failed)``: one SMILES column per attachment point (R1..R4) indexed by compound name,
        and the compounds that could not be placed.
    """
    from rdkit.Chem import rdRGroupDecomposition as rgd

    mols = load_molecules()
    if names is not None:
        mols = {n: mols[n] for n in names}
    params = rgd.RGroupDecompositionParameters()
    params.removeAllHydrogenRGroups = True
    params.onlyMatchAtRGroups = False
    decomposer = rgd.RGroupDecomposition(Chem.MolFromSmarts(CORE_SMARTS), params)
    placed, failed = [], []
    for name, mol in mols.items():
        (placed if decomposer.Add(mol) >= 0 else failed).append(name)
    decomposer.Process()
    table = pd.DataFrame(decomposer.GetRGroupsAsColumns(asSmiles=True), index=pd.Index(placed, name="name"))
    return table[[c for c in table.columns if c.startswith("R")]], failed


def tsne_coordinates() -> pd.DataFrame:
    """t-SNE layout of the compounds in Morgan (ECFP4, 2048 bits) Tanimoto-distance space.

    Read from ``output/chemical_space_embedding.parquet`` so the map is identical on every run; if the file
    is missing it is recomputed with the fixed seed.
    """
    if EMBEDDING_FILE.exists():
        return pd.read_parquet(EMBEDDING_FILE)[["tsne_x", "tsne_y"]]
    from sklearn.manifold import TSNE

    mols = load_molecules()
    generator = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048)
    fingerprints = [generator.GetFingerprint(m) for m in mols.values()]
    distance = np.zeros((len(fingerprints),) * 2, dtype=np.float32)
    for i in range(1, len(fingerprints)):
        similarity = np.asarray(DataStructs.BulkTanimotoSimilarity(fingerprints[i], fingerprints[:i]), dtype=np.float32)
        distance[i, :i] = distance[:i, i] = 1.0 - similarity
    layout = TSNE(metric="precomputed", init="random", random_state=RANDOM_SEED, perplexity=30.0).fit_transform(
        distance.astype(np.float64)
    )
    frame = pd.DataFrame(layout, columns=["tsne_x", "tsne_y"], index=pd.Index(list(mols), name="name"))
    OUTPUT_DIR.mkdir(exist_ok=True)
    frame.to_parquet(EMBEDDING_FILE)
    return frame


# --------------------------------------------------------------------------- #
# Patent potency and the shortlist
# --------------------------------------------------------------------------- #

KI_BIN_ORDER = ("+++", "++", "+")
KI_RANGE = {"+++": "< 0.01 µM", "++": "0.01–1 µM", "+": "1–5 µM"}
TOP_BIN, SECOND_BIN = KI_BIN_ORDER[:2]
PERCENTILE_FOR_TOP_BIN = 0.90


def join_patent_bins() -> pd.DataFrame:
    """The graded Ki bin of the source patent (US 9,340,546 B2, Table 5) joined onto the series.

    The table was read from the patent by OCR (``data/patent_ki_bins.csv``). Nine compounds have two IDs;
    both are matched.
    """
    lookup = pd.read_csv(PATENT_KI_CSV).set_index("compound_id")["ki_bin"].to_dict()
    frame = load_lead_series().data[["name", "compound_id", "compound_id_alt", "potency_class", "general_tox"]].copy()
    primary = frame["compound_id"].map(lookup)
    alternate = frame["compound_id_alt"].map(lambda v: lookup.get(v) if v else None)
    frame["ki_bin"] = primary.where(primary.notna(), alternate)
    return frame


def contingency(joined: pd.DataFrame) -> pd.DataFrame:
    """Spreadsheet potency label against the patent Ki bin."""
    return pd.crosstab(joined["potency_class"].astype(str), joined["ki_bin"].fillna("NA"), dropna=False)


def candidate_table() -> pd.DataFrame:
    """One row per compound (indexed by name) with what the shortlist rule needs."""
    data = load_lead_series().data.set_index("name")
    descriptors = load_descriptors().reindex(data.index)
    joined = join_patent_bins().set_index("name")
    return pd.DataFrame({
        "compound_id": data["compound_id"],
        "compound_id_alt": data["compound_id_alt"],
        "subseries": data["subseries"].astype(str),
        "ki_bin": joined["ki_bin"].reindex(data.index),
        "general_tox": data["general_tox"],
        "general_tox_percentile": data["general_tox_pct"],
        "n_basic_nitrogen": descriptors["n_basic_nitrogen"].astype(int),
        "mw": descriptors["mw"],
        "clogp": descriptors["clogp"],
        "smiles": data["smiles"],
    })


def thresholds(table: pd.DataFrame) -> tuple[float, float]:
    """``(+++ cutoff, ++ bar)`` in general_tox units: the 90th percentile of the series and tuvusertib's value."""
    reaches = table["general_tox_percentile"] >= PERCENTILE_FOR_TOP_BIN
    return float(table.loc[reaches, "general_tox"].min()), float(table.at[TUVUSERTIB, "general_tox"])


def select_shortlist(table: pd.DataFrame | None = None) -> pd.DataFrame:
    """Recommended compounds, safest first: bin +++ with general_tox in the top 10 % of the series, or bin ++
    with general_tox at least tuvusertib's. Compounds in bin + or without a bin are never selected."""
    table = candidate_table() if table is None else table
    _, bar = thresholds(table)
    top = (table["ki_bin"] == TOP_BIN) & (table["general_tox_percentile"] >= PERCENTILE_FOR_TOP_BIN)
    second = (table["ki_bin"] == SECOND_BIN) & (table["general_tox"] >= bar)
    selected = table[top | second].copy()
    selected["selection_basis"] = selected["ki_bin"].map({
        TOP_BIN: f"Ki {TOP_BIN} and general_tox percentile >= {PERCENTILE_FOR_TOP_BIN:.2f}",
        SECOND_BIN: f"Ki {SECOND_BIN} and general_tox >= tuvusertib ({bar:+.2f})",
    })
    order = selected["ki_bin"].map({b: i for i, b in enumerate(KI_BIN_ORDER)})
    selected = selected.assign(_bin=order).sort_values(["general_tox", "_bin"], ascending=[False, True]).drop(columns="_bin")
    selected["rank"] = range(1, len(selected) + 1)
    return selected


def write_shortlist_csv() -> pd.DataFrame:
    """Write ``output/shortlist_23_compounds.csv`` and return it."""
    out = select_shortlist().reset_index().rename(columns={"name": "compound"})
    out["ki_range_uM"] = out["ki_bin"].map(KI_RANGE).str.replace(" µM", "", regex=False).str.replace("–", "-", regex=False)
    out["is_benchmark_tuvusertib"] = out["compound"] == TUVUSERTIB
    out = out[["rank", "compound", "compound_id", "compound_id_alt", "subseries", "ki_bin", "ki_range_uM", "selection_basis",
               "is_benchmark_tuvusertib", "general_tox", "general_tox_percentile", "n_basic_nitrogen", "mw", "clogp", "smiles"]]
    OUTPUT_DIR.mkdir(exist_ok=True)
    out.round({"general_tox": 4, "general_tox_percentile": 4, "mw": 2, "clogp": 2}).to_csv(
        OUTPUT_DIR /"shortlist_23_compounds.csv", index=False, encoding="utf-8"
    )
    return out


# --------------------------------------------------------------------------- #
# Predicted versus measured effect concentrations (gartisertib and tuvusertib only)
# --------------------------------------------------------------------------- #

EXPERIMENTAL_COMPOUNDS = {"Tuvusertib": TUVUSERTIB, "VX-803 (M4344)": GARTISERTIB}
#: Assay family -> the experimental readout(s) the vendor mapped it to.
READOUT_MAP = {
    "cell_viability": ("mtt_ridge_norm_inv",),
    "ldh": ("ldh_ridge_norm",),
    "cell_death": ("mean_nuclei_count_ridge_norm_inv",),
    "nuclei_size": ("mean_nuclei_area_mean_ridge_norm_inv",),
    "er_stress": ("vacuole_10_sum_ratio",),
    "cytoplasm_area": ("cyto_area_ratio_ridge_norm_inv",),
    "ros": ("ros_stress_granules_sum_ratio",),
    "mitochondria": ("mito_v2_puncta_ratio_ridge_norm",),
    "bioactivity": ("bioactivity_platenorm", "bioactivity_regnorm"),
}


def _geometric_mean(values: np.ndarray) -> float:
    return float(10 ** np.mean(np.log10(values)))


def calibration_table() -> pd.DataFrame:
    """Predicted against measured EC, one row per compound x assay family x EC level (2 x 9 x 2 = 36).

    Both sides are clipped at the ceiling; families with two members are collapsed by geometric mean.
    A row is *comparable* when the measured value is below the ceiling.
    """
    fits = pd.read_excel(EXPERIMENTAL_XLSX, sheet_name="ld20_ld50")
    fits["compound"] = fits["compound_name"].map(EXPERIMENTAL_COMPOUNDS)
    fits = fits.set_index(["compound", "readout"])
    series = load_lead_series()
    predictions = series.raw_values.set_index(series.data["name"]).clip(upper=CEILING_UM)
    rows = []
    for compound in EXPERIMENTAL_COMPOUNDS.values():
        for family, readouts in READOUT_MAP.items():
            assays = FAMILIES.get(family) or (family,)
            for level in ("ld20", "ld50"):
                measured = np.array([fits.loc[(compound, r), f"{level}_uM"] for r in readouts], float)
                predicted = np.array([predictions.loc[compound, f"{a}_{level}"] for a in assays], float)
                rows.append({
                    "compound": compound, "family": family, "level": level,
                    "predicted_uM": _geometric_mean(predicted),
                    "experimental_uM": _geometric_mean(np.minimum(measured, CEILING_UM)),
                    "predicted_censored": bool((predicted >= CEILING_UM).any()),
                    "experimental_censored": bool((measured >= CEILING_UM).any()),
                })
    table = pd.DataFrame(rows)
    table["comparable"] = ~table["experimental_censored"]
    table["log10_ratio"] = np.log10(table["predicted_uM"] / table["experimental_uM"])
    table["fold_error"] = 10 ** table["log10_ratio"].abs()
    table["over_predicts_safety"] = table["predicted_uM"] > table["experimental_uM"]
    return table


def comparable_points(table: pd.DataFrame) -> pd.DataFrame:
    """The rows of a calibration table with a measured value below the ceiling."""
    return table[table["comparable"]].reset_index(drop=True)
