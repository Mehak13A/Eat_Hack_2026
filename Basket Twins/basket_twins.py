#!/usr/bin/env python3
"""
BASKET TWINS: find the London neighbourhoods that shop like your best one
=========================================================================
STEP 3 OF 3 in the pipeline:
    trend_agent.py  ->  launch_inputs_agent.py  ->  basket_twins.py (this file)

WHAT IT DOES (in plain words)
  A brand says where a product already sells (or, for new trends, where it already
  appears in London). This script then:
    1. builds a "basket fingerprint" for every London ward from Tesco 2015 shopping data,
    2. finds the wards whose fingerprints look most like that place ("basket twins"),
    3. explains each match (which features line up, which differ),
    4. flags near-misses (similar overall but weak in the product's own category),
    5. suggests a launch month from the 12 monthly files,
    6. designs a small trial: launch wards, each paired with a similar comparison ward,
    7. checks itself (hidden-feature tests, census variables never used, stability).
  If a product has no known area, a weaker "fallback" ranking is used instead
  (how much the ward buys the product's category, nudged by nutrition fit).

NO AI, NO API, NO INTERNET: this file is plain statistics (pandas, numpy, scikit-learn).

HOW TO RUN (PyCharm or terminal)
  1. pip install pandas numpy scikit-learn matplotlib
  2. Check the three paths in SETTINGS below. If a path does not exist on your
     computer, the script automatically uses the folder this file is in, i.e. it
     expects:   data/ (Tesco files + ward_lookup.csv), launch_products.csv, outputs/
  3. Run:   python basket_twins.py         (or right-click > Run in PyCharm)
  Other modes:  python basket_twins.py --check      only inspect data files and columns
                python basket_twins.py --synthetic  practice run on FAKE generated data

INPUTS
  data/year_osward_grocery.csv     Tesco Grocery 1.0, full year 2015, one row per ward (required)
  data/Jan_osward_grocery.csv ...  the 12 monthly files (optional; used for launch timing)
  data/ward_lookup.csv             ward code -> ward name, borough, centre coordinates
  launch_products.csv              one row per product (made by launch_inputs_agent.py);
                                   key columns: product_name, tesco_category, sells_well_in
                                   ("Ward (Borough); Ward (Borough)"), nutrition per 100 g/ml

OUTPUTS (in outputs/)
  summary.csv                 one row per product: launch wards, comparison wards,
                              near-misses, launch month, hidden-category p-value
  all_launch_plans.md         every launch plan in one document
  neighbourhood_types.png     wards grouped into shopping "types" (context only)
  validation_holdout.csv/png  the global self-check
  <product folder>/           launch_plan.md, 1_twins_ranking.png, 2_why_they_match.png
                              (twins mode only), 3_timing.png, ranking_top.csv

KEY IDEAS (glossary)
  ward              small London area (638 in the data; 574 kept after the coverage filter)
  fingerprint       25 numbers per ward: shares of 17 food categories + 8 nutrition values
                    of the typical item bought
  z-score           how many standard deviations a ward is above/below the London average
  cosine similarity 1 = same shopping pattern, 0 = unrelated (shape of the basket, not size)
  anchor            the ward(s) where the product already sells/appears (averaged)
  near-miss         twin that buys at least 1 SD less of the product's category
  comparison ward   similar ward that does NOT get the product, so sales can be compared
  p-value           chance that random wards would match the hidden category as well as
                    the twins do (small = strong evidence; above 0.05 = not conclusive)
  stability         share of the top-10 twins that survive when one feature is dropped

Data: Tesco Grocery 1.0 (Aiello et al., Scientific Data, 2020), CC BY 4.0
      https://doi.org/10.6084/m9.figshare.c.4769354.v2
Limits: 2015 data; Tesco Clubcard shoppers only; uneven store coverage; results show
        where a trial is most likely to fit, not guaranteed sales.
"""

from __future__ import annotations

import argparse
import difflib
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

import matplotlib

matplotlib.use("Agg")  # save charts to files instead of opening windows
import matplotlib.pyplot as plt  # noqa: E402
from sklearn.cluster import KMeans  # noqa: E402
from sklearn.metrics import silhouette_score  # noqa: E402
from sklearn.metrics.pairwise import cosine_similarity  # noqa: E402
from sklearn.preprocessing import StandardScaler  # noqa: E402

# =============================================================================
# SETTINGS (change these, no need to touch anything below)
# =============================================================================
SCRIPT_DIR = Path(__file__).resolve().parent

# Plain text paths (full paths on your computer). Keep the r before the quotes.
# On another computer you do not need to edit these: if a path does not exist,
# the script falls back to the folder this file is in (data/, launch_products.csv, outputs/).
DATA_DIR = r"/Users/mehakagrawal/Desktop/Basket Twins/data"                     # Tesco files + ward_lookup.csv
PRODUCTS_FILE = r"/Users/mehakagrawal/Desktop/Basket Twins/outputs_launch/launch_products.csv"  # input: one row per product
OUTPUT_DIR = r"/Users/mehakagrawal/Desktop/Basket Twins/outputs_basket"                 # charts and reports go here

YEAR_FILE_NAME = None    # None = auto-detect "year_*ward*grocery*.csv"
LOOKUP_FILE_NAME = None  # None = auto-detect a CSV with "lookup" in its name

CHECK_DATA_ONLY = False      # True = only inspect files and columns, then stop
USE_SYNTHETIC_DATA = False   # True = practice on fake data (never for the demo)

MIN_COVERAGE_QUANTILE = 0.10  # drop the 10% of wards with the lowest Tesco coverage
N_TWINS_SHOWN = 10            # twins listed in the ranking chart and report
N_LAUNCH_WARDS = 5            # wards recommended for the trial launch
CANDIDATE_POOL = 30           # near-misses and comparison wards come from this many top twins
NEAR_MISS_GAP_SD = 1.0        # near-miss = product category this many SDs below the anchor
MAX_LAUNCH_GAP_SD = 0.5       # launch wards may be at most this far below the anchor
PREFER_OTHER_BOROUGH_FOR_COMPARISON = True

CLUSTER_K_OPTIONS = [4, 5, 6, 7]  # number of neighbourhood types to try
TRIBE_NAMES: dict[int, str] = {}  # optional renames after looking at the chart,
                                  # e.g. {0: "Grab-and-go drinkers", 3: "Home cooks"}

N_PERMUTATIONS = 2000
RANDOM_SEED = 42

# =============================================================================
# Column definitions (flexible matching: case, spaces and underscores ignored)
# =============================================================================
ID_ALIASES = ["area_id", "area_code", "ward_code", "code", "gss_code"]
COVERAGE_ALIASES = ["representativeness_norm", "representativeness"]

CATEGORY_ALIASES = {
    "fruit_veg": ["f_fruit_veg", "f_fruitveg", "f_fruit_and_veg"],
    "grains": ["f_grains"],
    "meat_red": ["f_meat_red", "f_red_meat"],
    "poultry": ["f_poultry"],
    "fish": ["f_fish"],
    "dairy": ["f_dairy"],
    "eggs": ["f_eggs"],
    "fats_oils": ["f_fats_oils", "f_fats_and_oils"],
    "sweets": ["f_sweets"],
    "readymade": ["f_readymade", "f_ready_made", "f_readymeals"],
    "sauces": ["f_sauces"],
    "tea_coffee": ["f_tea_coffee"],
    "soft_drinks": ["f_soft_drinks", "f_softdrinks"],
    "water": ["f_water"],
    "beer": ["f_beer"],
    "wine": ["f_wine"],
    "spirits": ["f_spirits"],
}
NUTRIENT_ALIASES = {
    "energy": ["energy_tot", "energy"],
    "fat": ["fat"],
    "saturates": ["saturate", "saturates", "saturated_fat"],
    "sugar": ["sugar"],
    "protein": ["protein"],
    "carbs": ["carb", "carbs", "carbohydrate"],
    "fibre": ["fibre", "fiber"],
    "salt": ["salt"],
}
CENSUS_ALIASES = {  # never used for matching: only for validation
    "avg_age": ["avg_age", "average_age", "mean_age"],
    "density": ["people_per_sq_km", "density", "pop_density"],
}

LABELS = {
    "fruit_veg": "fruit & veg", "grains": "grains", "meat_red": "red meat",
    "poultry": "poultry", "fish": "fish", "dairy": "dairy", "eggs": "eggs",
    "fats_oils": "fats & oils", "sweets": "sweets", "readymade": "ready meals",
    "sauces": "sauces", "tea_coffee": "tea & coffee", "soft_drinks": "soft drinks",
    "water": "water", "beer": "beer", "wine": "wine", "spirits": "spirits",
    "energy": "calories", "fat": "fat", "saturates": "saturated fat",
    "sugar": "sugar", "protein": "protein", "carbs": "carbs", "fibre": "fibre",
    "salt": "salt", "avg_age": "average age (census)", "density": "population density (census)",
}
DRINK_CATEGORIES = {"soft_drinks", "water", "beer", "wine", "spirits"}
MONTHS = ["Jan", "Feb", "Mar", "Apr", "May", "Jun",
          "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]

# UK front-of-pack "traffic light" thresholds per 100g (food) or 100ml (drinks).
# (low_at_or_below, high_above). Used only in the fallback ranking.
FSA_THRESHOLDS = {
    "food": {"sugar": (5.0, 22.5), "fat": (3.0, 17.5), "saturates": (1.5, 5.0), "salt": (0.3, 1.5)},
    "drink": {"sugar": (2.5, 11.25), "fat": (1.5, 8.75), "saturates": (0.75, 2.5), "salt": (0.3, 0.75)},
}

PRODUCT_COLUMNS = [
    "brand", "product_name", "tesco_category",
    "energy_kcal_100g", "sugar_g_100g", "fat_g_100g", "saturates_g_100g",
    "protein_g_100g", "fibre_g_100g", "salt_g_100g",
    "sells_well_in", "sells_well_source", "channel", "price_gbp", "notes",
]
PRODUCT_NUTRIENT_COLUMNS = {
    "sugar": "sugar_g_100g", "fat": "fat_g_100g",
    "saturates": "saturates_g_100g", "salt": "salt_g_100g",
}


class BasketTwinsError(Exception):
    """A problem with a clear, human-readable fix."""


# =============================================================================
# Small helpers
# =============================================================================
def use_path(text: str, fallback_name: str) -> Path:
    """Use the path from SETTINGS if it (or its parent folder) exists on this computer.
    Otherwise use a file/folder with `fallback_name` next to this script, so the
    project runs on any computer without editing paths."""
    p = Path(text)
    if p.exists() or p.parent.exists():
        return p
    fallback = SCRIPT_DIR / fallback_name
    say(f"  Note: '{text}' not found on this computer, using {fallback}")
    return fallback


def norm(text) -> str:
    """Lower-case and remove everything except letters and digits."""
    return re.sub(r"[^a-z0-9]", "", str(text).lower())


def find_column(columns, aliases):
    """Return the real column name that matches any alias (case, spaces and underscores
    are ignored), or None if the file has no such column."""
    lookup = {norm(c): c for c in columns}
    for alias in aliases:
        if norm(alias) in lookup:
            return lookup[norm(alias)]
    return None


def slugify(text: str) -> str:
    """Turn a product name into a safe folder name, e.g. 'Hojicha Latte' -> 'hojicha_latte'."""
    return re.sub(r"[^a-z0-9]+", "_", str(text).lower()).strip("_")[:60] or "product"


def label(feature: str) -> str:
    """Readable label for a feature code, e.g. 'meat_red' -> 'red meat'."""
    return LABELS.get(feature, feature)


def say(text: str = "") -> None:
    """Print straight away (flush), so progress appears in PyCharm while the script runs."""
    print(text, flush=True)


# =============================================================================
# Finding and loading files
# =============================================================================
def find_year_file(data_dir: Path) -> Path:
    """Find the full-year Tesco ward file (year_*ward*grocery*.csv) in the data folder,
    with a clear error message if it is missing."""
    if YEAR_FILE_NAME:
        path = data_dir / YEAR_FILE_NAME
        if not path.exists():
            raise BasketTwinsError(f"YEAR_FILE_NAME is set to '{YEAR_FILE_NAME}' but {path} does not exist.")
        return path
    if not data_dir.exists():
        raise BasketTwinsError(
            f"Data folder not found: {data_dir}\n"
            "  Create it and put the Tesco ward files inside (e.g. year_osward_grocery.csv).\n"
            "  Download: https://doi.org/10.6084/m9.figshare.c.4769354.v2\n"
            "  Or practise first with:  python basket_twins.py --synthetic")
    matches = [p for p in data_dir.glob("*.csv")
               if p.name.lower().startswith("year") and "ward" in p.name.lower()
               and "grocery" in p.name.lower()]
    if not matches:
        found = ", ".join(sorted(p.name for p in data_dir.glob("*.csv"))) or "no CSV files"
        raise BasketTwinsError(
            f"No full-year ward file found in {data_dir} (looked for 'year_*ward*grocery*.csv').\n"
            f"  Files found: {found}\n"
            "  If the file has another name, set YEAR_FILE_NAME in the SETTINGS.")
    if len(matches) > 1:
        say(f"  Note: several year files found, using {matches[0].name}")
    return sorted(matches)[0]


def find_month_files(data_dir: Path) -> dict[str, Path]:
    """Find the monthly Tesco ward files (Jan_..., Feb_..., ...). Returns {month: path}.
    Missing months are allowed; timing is skipped unless all 12 are present."""
    found = {}
    for path in data_dir.glob("*.csv"):
        name = path.name.lower()
        for month in MONTHS:
            if name.startswith(month.lower() + "_") and "ward" in name and "grocery" in name:
                found[month] = path
    return found


def find_lookup_file(data_dir: Path):
    """Find the ward lookup CSV (any CSV with 'lookup' in its name). It turns ward codes
    into ward names and boroughs; without it, outputs show codes only."""
    if LOOKUP_FILE_NAME:
        path = data_dir / LOOKUP_FILE_NAME
        if not path.exists():
            raise BasketTwinsError(f"LOOKUP_FILE_NAME is set but {path} does not exist.")
        return path
    for path in sorted(data_dir.glob("*.csv")):
        if "lookup" in path.name.lower() or "ward_names" in path.name.lower():
            return path
    return None


def read_area_file(path: Path):
    """Read one Tesco file and rename its columns to short standard names."""
    try:
        raw = pd.read_csv(path)
    except Exception as exc:  # noqa: BLE001
        raise BasketTwinsError(f"Could not read {path.name}: {exc}") from exc

    id_col = find_column(raw.columns, ID_ALIASES)
    if id_col is None:  # fall back to a column whose values look like ONS codes
        for col in raw.columns:
            if raw[col].astype(str).str.match(r"^[A-Z]\d{8}$").mean() > 0.8:
                id_col = col
                break
    if id_col is None:
        raise BasketTwinsError(
            f"{path.name}: could not find the area code column.\n"
            f"  First columns are: {list(raw.columns[:8])}\n"
            "  Add its name to ID_ALIASES in the script.")

    out = pd.DataFrame(index=raw[id_col].astype(str).str.strip())
    out.index.name = "code"
    report = {"id": id_col, "categories": {}, "nutrients": {}, "census": {}, "coverage": None}
    for group, aliases_map in (("categories", CATEGORY_ALIASES),
                               ("nutrients", NUTRIENT_ALIASES),
                               ("census", CENSUS_ALIASES)):
        for canon, aliases in aliases_map.items():
            col = find_column(raw.columns, aliases)
            if col is not None:
                out[canon] = pd.to_numeric(raw[col], errors="coerce").to_numpy()
                report[group][canon] = col
    cov = find_column(raw.columns, COVERAGE_ALIASES)
    if cov is not None:
        out["coverage"] = pd.to_numeric(raw[cov], errors="coerce").to_numpy()
        report["coverage"] = cov
    out = out[~out.index.duplicated()]
    return out, report, list(raw.columns)


def print_column_report(path: Path, report: dict, all_columns: list, n_rows: int) -> None:
    """Print which real column was used for each category, nutrient and census variable,
    so a missing or renamed column is visible immediately."""
    say(f"\nFile: {path.name}  ({n_rows} areas, {len(all_columns)} columns)")
    say(f"  Area code column : {report['id']}")
    say(f"  Coverage column  : {report['coverage'] or 'NOT FOUND (no coverage filter will be applied)'}")
    for group, names in (("categories", CATEGORY_ALIASES), ("nutrients", NUTRIENT_ALIASES),
                         ("census", CENSUS_ALIASES)):
        found = report[group]
        missing = [c for c in names if c not in found]
        say(f"  {group:<10} found {len(found)}/{len(names)}: "
            + ", ".join(f"{k}<-{v}" for k, v in found.items()))
        if missing:
            say(f"  {group:<10} MISSING: {', '.join(missing)}")
    used = set(report["categories"].values()) | set(report["nutrients"].values()) \
        | set(report["census"].values()) | {report["id"], report["coverage"]}
    f_unused = [c for c in all_columns if str(c).lower().startswith("f_") and c not in used
                and "energy" not in str(c).lower() and not str(c).lower().endswith("_weight")]
    if f_unused:
        say("  Unmatched 'f_' columns (if any is a food category, add it to CATEGORY_ALIASES):")
        say("    " + ", ".join(map(str, f_unused)))


def load_lookup(path: Path, codes: pd.Index):
    """Ward code -> (name, borough). Works with ONS or London Datastore style files."""
    try:
        raw = pd.read_csv(path, dtype=str, encoding_errors="replace")
    except Exception as exc:  # noqa: BLE001
        raise BasketTwinsError(f"Could not read lookup file {path.name}: {exc}") from exc
    code_set = set(codes)
    best_col, best_hits = None, 0
    for col in raw.columns:
        hits = raw[col].astype(str).str.strip().isin(code_set).sum()
        if hits > best_hits:
            best_col, best_hits = col, hits
    if best_col is None or best_hits == 0:
        say(f"  Warning: {path.name} has no column matching the Tesco ward codes. Names disabled.")
        return {}, {}
    name_col = find_column(raw.columns, ["ward_name", "wardname", "WD15NM", "WD14NM", "WD16NM",
                                         "WD13NM", "WD11NM", "NAME", "ward"])
    if name_col is None:
        name_col = next((c for c in raw.columns if c != best_col and norm(c).endswith("nm")
                         and not norm(c).startswith("lad")), None)
    borough_col = find_column(raw.columns, ["borough", "borough_name", "LAD15NM", "LAD14NM",
                                            "LAD16NM", "LAD13NM", "LAD11NM", "local_authority",
                                            "district", "lad_name"])
    raw = raw.drop_duplicates(subset=best_col).set_index(raw[best_col].astype(str).str.strip())
    names = raw[name_col].to_dict() if name_col else {}
    boroughs = raw[borough_col].to_dict() if borough_col else {}
    rate = 100 * best_hits / max(len(code_set), 1)
    say(f"  Lookup {path.name}: code<-{best_col}, name<-{name_col}, borough<-{borough_col}; "
        f"matched {best_hits}/{len(code_set)} Tesco wards ({rate:.0f}%)")
    if rate < 80:
        say("  Warning: under 80% of wards matched. The lookup may use a different ward "
            "boundary year from the Tesco data (2015). Codes will be shown where names are missing.")
    return names, boroughs


# =============================================================================
# Preparing the ward data
# =============================================================================
@dataclass
class WardData:
    table: pd.DataFrame           # all wards, standard column names
    candidates: pd.Index          # wards that pass the coverage filter
    features: list[str]           # columns used for matching
    categories: list[str]
    census: list[str]
    z: pd.DataFrame               # standardised features for all usable wards
    names: dict = field(default_factory=dict)
    boroughs: dict = field(default_factory=dict)

    def name(self, code: str) -> str:
        return str(self.names.get(code, code))

    def full_name(self, code: str) -> str:
        b = self.boroughs.get(code)
        return f"{self.name(code)} ({b})" if b else self.name(code)


def prepare_wards(table: pd.DataFrame, report: dict, names: dict, boroughs: dict) -> WardData:
    """Build the 'basket fingerprint' of every ward.

    - Features: the 17 category shares + 8 nutrition values (25 in total).
    - Drops wards with missing values, then keeps only wards with enough Tesco
      coverage (the lowest 10% are removed, as the dataset's authors recommend).
    - Standardises every feature into z-scores (fitted on the trusted wards only),
      so no feature dominates just because its numbers are bigger."""
    categories = [c for c in CATEGORY_ALIASES if c in report["categories"]]
    nutrients = [n for n in NUTRIENT_ALIASES if n in report["nutrients"]]
    census = [c for c in CENSUS_ALIASES if c in report["census"]]
    if len(categories) < 8:
        raise BasketTwinsError(
            f"Only {len(categories)} of 17 food category columns were recognised.\n"
            "  Run with --check to see the real column names, then add them to CATEGORY_ALIASES.")
    features = categories + nutrients

    usable = table.dropna(subset=features)
    dropped = len(table) - len(usable)
    if dropped:
        say(f"  Dropped {dropped} wards with missing values in the matching columns.")
    if "coverage" in usable.columns and usable["coverage"].notna().any():
        cutoff = usable["coverage"].quantile(MIN_COVERAGE_QUANTILE)
        candidates = usable.index[usable["coverage"] >= cutoff]
        say(f"  Coverage filter: kept {len(candidates)}/{len(usable)} wards "
            f"(dropped the lowest {MIN_COVERAGE_QUANTILE:.0%}, coverage < {cutoff:.3f}).")
    else:
        candidates = usable.index
        say("  No coverage column found, so no coverage filter was applied.")
    if len(candidates) < 30:
        raise BasketTwinsError(f"Only {len(candidates)} wards left after filtering; too few to match.")

    scaler = StandardScaler().fit(usable.loc[candidates, features])  # fit on trusted wards only
    z = pd.DataFrame(scaler.transform(usable[features]), index=usable.index, columns=features)
    say(f"  Matching on {len(features)} features: {len(categories)} categories + {len(nutrients)} nutrients.")
    return WardData(table=table, candidates=candidates, features=features, categories=categories,
                    census=census, z=z, names=names, boroughs=boroughs)


# =============================================================================
# Neighbourhood types (k-means), supporting context only
# =============================================================================
@dataclass
class Tribes:
    labels: pd.Series
    names: dict
    centres: pd.DataFrame
    k: int
    silhouette: float
    model: KMeans


def build_tribes(wd: WardData) -> Tribes:
    """Group wards into 4-7 'neighbourhood types' with k-means and keep the k with the
    best silhouette score. Types are named after their strongest features.
    This is background context for the reports; the twins do the real work."""
    x = wd.z.loc[wd.candidates].to_numpy()
    best = None
    for k in CLUSTER_K_OPTIONS:
        model = KMeans(n_clusters=k, n_init=10, random_state=RANDOM_SEED).fit(x)
        score = silhouette_score(x, model.labels_)
        say(f"    k={k}: silhouette {score:.3f}")
        if best is None or score > best[1]:
            best = (model, score)
    model, score = best
    centres = pd.DataFrame(model.cluster_centers_, columns=wd.features)
    auto_names = {}
    for i, row in centres.iterrows():
        top = row.sort_values(ascending=False)
        highs = [label(f) for f in top.index[:2] if top[f] > 0.3]
        lows = [label(f) for f in top.index[::-1][:1] if top[f] < -0.3]
        name = "High " + " & ".join(highs) if highs else "Low " + " & ".join(lows) if lows else "Average"
        auto_names[i] = TRIBE_NAMES.get(i, f"T{i}: {name}")
    say(f"  Chose k={model.n_clusters} neighbourhood types (silhouette {score:.3f}; "
        "0 = overlapping groups, 1 = perfectly separate).")
    return Tribes(labels=pd.Series(model.labels_, index=wd.candidates), names=auto_names,
                  centres=centres, k=model.n_clusters, silhouette=score, model=model)


def tribe_of(tribes: Tribes, wd: WardData, code: str) -> str:
    """Name of the neighbourhood type a ward belongs to (predicted for filtered-out wards)."""
    if code in tribes.labels.index:
        return tribes.names[int(tribes.labels[code])]
    pred = tribes.model.predict(wd.z.loc[[code]].to_numpy())[0]
    return tribes.names[int(pred)]


# =============================================================================
# Global validation: do twins share things they were never matched on?
# =============================================================================
def holdout_validation(wd: WardData, k: int = 10) -> pd.DataFrame:
    """For each feature: hide it, find every ward's top-k twins, and check whether
    the twins are closer on the hidden feature than a random ward would be.
    Census variables are never used for matching, so they are always 'hidden'."""
    z = wd.z.loc[wd.candidates]
    x = z.to_numpy()
    n = len(x)
    rows = []

    def score(sim, values, feature, kind):
        np.fill_diagonal(sim, -np.inf)
        top = np.argpartition(-sim, k, axis=1)[:, :k]
        twin_gap = np.abs(values[:, None] - values[top]).mean(axis=1)
        rand_gap = np.abs(values[:, None] - values[None, :]).sum(axis=1) / (n - 1)
        rows.append({
            "feature": feature, "kind": kind,
            "closer_than_random_pct": 100 * (1 - np.median(twin_gap / rand_gap)),
            "wards_where_twins_win_pct": 100 * np.mean(twin_gap < rand_gap),
        })

    for j, feature in enumerate(wd.features):
        keep = [i for i in range(x.shape[1]) if i != j]
        score(cosine_similarity(x[:, keep]), x[:, j], feature, "hidden while matching")

    full_sim = cosine_similarity(x)
    for c in wd.census:
        values = wd.table.loc[wd.candidates, c]
        if values.notna().sum() < 0.9 * n:
            continue
        v = values.fillna(values.median()).to_numpy()
        v = (v - v.mean()) / (v.std() or 1)
        score(full_sim.copy(), v, c, "never used (census)")
    return pd.DataFrame(rows)


# =============================================================================
# Products
# =============================================================================
def canonical_category(text: str):
    """Map what the user typed (e.g. 'soft drinks', 'f_soft_drinks') to one of the
    17 category keys used in this script, or None if it is not recognised."""
    key = norm(text)
    for canon, aliases in CATEGORY_ALIASES.items():
        options = [canon] + aliases + [a[2:] for a in aliases]
        if key in {norm(o) for o in options}:
            return canon
    return None


def write_blank_template(path: Path) -> None:
    """Create an empty launch_products.csv with the 15 expected column headers."""
    pd.DataFrame(columns=PRODUCT_COLUMNS).to_csv(path, index=False)


def load_products(path: Path, wd: WardData) -> list[dict]:
    """Read launch_products.csv and return one dict per usable product.
    Rows with an unknown category or non-numeric nutrition values are reported
    and skipped (or the bad value ignored), instead of crashing the run."""
    if not path.exists():
        write_blank_template(path)
        raise BasketTwinsError(f"No products file found, so a blank one was created at:\n  {path}\n"
                               "  Fill in one row per product and run again.")
    df = pd.read_csv(path, dtype=str).fillna("")
    df.columns = [norm(c) for c in df.columns]
    needed = ["brand", "productname", "tescocategory"]
    missing = [c for c in needed if c not in df.columns]
    if missing:
        raise BasketTwinsError(f"{path.name} is missing columns: {missing}. "
                               f"Expected columns: {PRODUCT_COLUMNS}")
    products = []
    for i, row in df.iterrows():
        line = i + 2  # spreadsheet row number (header is row 1)
        if not row.get("productname", "").strip():
            continue
        cat = canonical_category(row["tescocategory"])
        if cat is None:
            say(f"  Row {line} ({row['productname']}): unknown tesco_category "
                f"'{row['tescocategory']}'. Use one of: {', '.join(CATEGORY_ALIASES)}. Skipped.")
            continue
        if cat not in wd.categories:
            say(f"  Row {line}: category '{cat}' is not in the data files. Skipped.")
            continue
        nutrients = {}
        for key, col in PRODUCT_NUTRIENT_COLUMNS.items():
            value = row.get(norm(col), "").strip()
            if value:
                try:
                    nutrients[key] = float(value)
                except ValueError:
                    say(f"  Row {line}: '{col}' = '{value}' is not a number; ignored.")
        products.append({
            "brand": row["brand"].strip(), "product": row["productname"].strip(),
            "category": cat, "nutrients": nutrients,
            "sells_well_in": row.get("sellswellin", "").strip(),
            "source": row.get("sellswellsource", "").strip(),
        })
    if not products:
        raise BasketTwinsError(f"{path.name} has no usable product rows yet.")
    say(f"  Loaded {len(products)} product(s) from {path.name}.")
    return products


def resolve_places(text: str, wd: WardData):
    """Turn 'Islington; Hackney Central' into ward codes. Separate places with ';'."""
    usable = set(wd.z.index)
    code_map = {norm(c): c for c in usable}
    ward_map, borough_map, ward_in_borough = {}, {}, {}
    for code in usable:
        if code in wd.names:
            ward_map.setdefault(norm(wd.names[code]), []).append(code)
            if code in wd.boroughs:  # "Highgate (Camden)" or "Highgate, Camden"
                ward_in_borough[norm(wd.names[code]) + norm(wd.boroughs[code])] = code
        if code in wd.boroughs:
            borough_map.setdefault(norm(wd.boroughs[code]), []).append(code)
    codes, notes = [], []
    for part in [p.strip() for p in re.split(r"[;|]", text) if p.strip()]:
        key = norm(part)
        if key in code_map:
            codes.append(code_map[key])
            notes.append(f"'{part}' = ward code")
        elif key in ward_in_borough:
            codes.append(ward_in_borough[key])
            notes.append(f"'{part}' = ward")
        elif key in borough_map:
            codes += borough_map[key]
            notes.append(f"'{part}' = borough ({len(borough_map[key])} wards averaged)")
        elif key in ward_map:
            codes += ward_map[key]
            extra = (" (this name exists in several boroughs, so all were used; write e.g. "
                     f"'{part} ({wd.boroughs.get(ward_map[key][0], 'Borough')})' to pick one)"
                     if len(ward_map[key]) > 1 else "")
            notes.append(f"'{part}' = ward{extra}")
        else:
            keys = list(borough_map) + list(ward_map)
            close = difflib.get_close_matches(key, keys, n=3, cutoff=0.6)
            display = {**{k: wd.boroughs[v[0]] for k, v in borough_map.items()},
                       **{k: wd.names[v[0]] for k, v in ward_map.items()}}
            hint = f" Did you mean: {', '.join(display[c] for c in close)}?" if close else ""
            if not wd.names:
                hint += " (No ward lookup loaded, so only ward codes like E05000123 work.)"
            notes.append(f"'{part}' NOT FOUND.{hint}")
    return list(dict.fromkeys(codes)), notes


# =============================================================================
# The launch plan for one product
# =============================================================================
@dataclass
class Plan:
    product: dict
    mode: str                      # "twins" or "fallback"
    anchor_codes: list
    place_notes: list
    ranking: pd.DataFrame          # all candidate wards, scored
    launch: list
    comparisons: dict              # launch code -> comparison code
    near_misses: list
    timing: dict | None = None
    permutation: dict | None = None
    stability: dict | None = None


def explain_match(anchor: np.ndarray, ward: np.ndarray, features: list[str]):
    """Explain one twin in words. 'Matches on' = the 3 features where both the anchor
    and the ward are strongly on the same side of the London average; 'differs on' =
    the 2 features with the biggest gap."""
    shared = anchor * ward  # positive when both are above average or both below
    order = np.argsort(-shared)
    matches = []
    for i in order[:3]:
        if shared[i] <= 0:
            break
        direction = "high" if anchor[i] > 0 else "low"
        matches.append(f"both {direction} {label(features[i])}")
    diff = ward - anchor
    differs = []
    for i in np.argsort(-np.abs(diff))[:2]:
        differs.append(f"{'more' if diff[i] > 0 else 'less'} {label(features[i])}")
    return "; ".join(matches) or "close to the London average overall", "; ".join(differs)


def pick_comparisons(wd: WardData, launch: list, pool: list) -> dict:
    """Pair each launch ward with the most similar unused twin (ideally another borough)."""
    pairs, used = {}, set(launch)
    for code in launch:
        options = [c for c in pool if c not in used]
        if PREFER_OTHER_BOROUGH_FOR_COMPARISON and wd.boroughs:
            other = [c for c in options if wd.boroughs.get(c) != wd.boroughs.get(code)]
            options = other or options
        if not options:
            continue
        sims = cosine_similarity(wd.z.loc[options].to_numpy(), wd.z.loc[[code]].to_numpy()).ravel()
        best = options[int(np.argmax(sims))]
        pairs[code] = best
        used.add(best)
    return pairs


def fallback_scores(wd: WardData, product: dict) -> tuple[pd.Series, list[str]]:
    """No known selling area: rank wards by how much they buy the product's category,
    nudged by nutrient fit (areas already buying low-sugar items for a low-sugar product).
    This is a heuristic and much weaker evidence than basket twins."""
    z = wd.z.loc[wd.candidates]
    score = z[product["category"]].copy()
    reasons = [f"share of {label(product['category'])}"]
    kind = "drink" if product["category"] in DRINK_CATEGORIES else "food"
    signals = []
    for nutrient, value in product["nutrients"].items():
        if nutrient not in z.columns or nutrient not in FSA_THRESHOLDS[kind]:
            continue
        low, high = FSA_THRESHOLDS[kind][nutrient]
        if value <= low:
            signals.append(-z[nutrient])
            reasons.append(f"low {label(nutrient)} (buys lower-{label(nutrient)} items)")
        elif value > high:
            signals.append(z[nutrient])
            reasons.append(f"high {label(nutrient)} (buys higher-{label(nutrient)} items)")
    if signals:
        score = score + 0.5 * pd.concat(signals, axis=1).mean(axis=1)
    return score, reasons


def make_plan(wd: WardData, product: dict, month_tables: dict) -> Plan:
    """Make the launch plan for one product.

    Twins mode (the product has an anchor area):
      1. anchor fingerprint = average z-scores of the anchor wards
      2. cosine similarity of every other trusted ward to the anchor -> ranking
      3. near-misses: top-30 twins at least NEAR_MISS_GAP_SD lower on the product's category
      4. launch wards: best twins no more than MAX_LAUNCH_GAP_SD below on that category
      5. comparison wards: the most similar remaining twins (ideally another borough)
      6. trust checks: hidden-category permutation test and stability test
    Fallback mode (no anchor): rank wards by category share + nutrition fit.
    Both modes: launch timing from the monthly files."""
    rng = np.random.default_rng(RANDOM_SEED)
    cat = product["category"]
    codes, notes = resolve_places(product["sells_well_in"], wd) if product["sells_well_in"] else ([], [])

    if codes:
        anchor = wd.z.loc[codes].mean().to_numpy()
        low_cov = [c for c in codes if c not in set(wd.candidates)]
        if low_cov:
            notes.append(f"{len(low_cov)} anchor ward(s) have low Tesco coverage; treat with care.")
        pool_codes = [c for c in wd.candidates if c not in set(codes)]
        zp = wd.z.loc[pool_codes]
        sims = cosine_similarity(zp.to_numpy(), anchor.reshape(1, -1)).ravel()
        ranking = pd.DataFrame({"similarity": sims}, index=pd.Index(pool_codes, name="code"))
        ranking["category_gap_sd"] = zp[cat].to_numpy() - anchor[wd.features.index(cat)]
        ranking = ranking.sort_values("similarity", ascending=False)
        explanations = [explain_match(anchor, wd.z.loc[c].to_numpy(), wd.features)
                        for c in ranking.index[:CANDIDATE_POOL]]
        ranking["matches_on"] = ""
        ranking["differs_on"] = ""
        ranking.iloc[:CANDIDATE_POOL, ranking.columns.get_loc("matches_on")] = [e[0] for e in explanations]
        ranking.iloc[:CANDIDATE_POOL, ranking.columns.get_loc("differs_on")] = [e[1] for e in explanations]
        top_pool = ranking.head(CANDIDATE_POOL)
        near = top_pool.index[top_pool["category_gap_sd"] <= -NEAR_MISS_GAP_SD].tolist()
        eligible = [c for c in top_pool.index if top_pool.at[c, "category_gap_sd"] >= -MAX_LAUNCH_GAP_SD]
        launch = eligible[:N_LAUNCH_WARDS]
        comparisons = pick_comparisons(wd, launch, eligible[N_LAUNCH_WARDS:])
        plan = Plan(product, "twins", codes, notes, ranking, launch, comparisons, near)
        plan.permutation = permutation_check(wd, anchor, codes, cat, rng)
        plan.stability = stability_check(wd, anchor, codes)
    else:
        if product["sells_well_in"]:
            notes.append("No listed place could be matched, so the fallback ranking was used.")
        score, reasons = fallback_scores(wd, product)
        ranking = pd.DataFrame({"fit_score": score}).sort_values("fit_score", ascending=False)
        ranking["reason"] = "fit on: " + ", ".join(reasons)
        launch = ranking.index[:N_LAUNCH_WARDS].tolist()
        comparisons = pick_comparisons(wd, launch, ranking.index[N_LAUNCH_WARDS:CANDIDATE_POOL].tolist())
        plan = Plan(product, "fallback", [], notes, ranking, launch, comparisons, [])

    plan.timing = timing_for(wd, month_tables, cat, launch)
    return plan


def permutation_check(wd: WardData, anchor: np.ndarray, anchor_codes: list, cat: str, rng) -> dict:
    """Hide the product's own category, find twins again, and ask: are these twins
    closer to the anchor on the hidden category than random sets of wards?"""
    j = wd.features.index(cat)
    keep = [i for i in range(len(wd.features)) if i != j]
    pool = [c for c in wd.candidates if c not in set(anchor_codes)]
    zp = wd.z.loc[pool].to_numpy()
    sims = cosine_similarity(zp[:, keep], anchor[keep].reshape(1, -1)).ravel()
    k = N_TWINS_SHOWN
    top = np.argsort(-sims)[:k]
    observed = np.abs(zp[top, j] - anchor[j]).mean()
    random_gaps = np.array([np.abs(zp[rng.choice(len(zp), k, replace=False), j] - anchor[j]).mean()
                            for _ in range(N_PERMUTATIONS)])
    p = (np.sum(random_gaps <= observed) + 1) / (N_PERMUTATIONS + 1)
    return {"category": cat, "observed_gap": observed, "random_gap": random_gaps.mean(),
            "beats_random_pct": 100 * np.mean(random_gaps > observed), "p_value": p}


def stability_check(wd: WardData, anchor: np.ndarray, anchor_codes: list) -> dict:
    """Drop one feature at a time: how much of the top-10 list survives?"""
    pool = [c for c in wd.candidates if c not in set(anchor_codes)]
    zp = wd.z.loc[pool].to_numpy()
    k = N_TWINS_SHOWN
    base = set(np.argsort(-cosine_similarity(zp, anchor.reshape(1, -1)).ravel())[:k])
    overlaps = {}
    for j, feature in enumerate(wd.features):
        keep = [i for i in range(len(wd.features)) if i != j]
        top = set(np.argsort(-cosine_similarity(zp[:, keep], anchor[keep].reshape(1, -1)).ravel())[:k])
        overlaps[feature] = len(base & top) / k
    worst = min(overlaps, key=overlaps.get)
    return {"mean_overlap": float(np.mean(list(overlaps.values()))),
            "worst_feature": worst, "worst_overlap": overlaps[worst]}


def timing_for(wd: WardData, month_tables: dict, cat: str, launch: list):
    """When do the launch wards buy the product's category? For each month, the
    category's share is indexed to its yearly average (100 = average). The best
    3-month stretch is the 'season'; the suggested launch month is the month before it.
    A season counts as real only if it is at least 5% above average."""
    if len(month_tables) < 12 or not launch:
        return None
    twins, london = [], []
    for m in MONTHS:
        t = month_tables[m]
        if cat not in t.columns:
            return None
        twins.append(t[cat].reindex(launch).mean())
        london.append(t[cat].reindex(wd.candidates).mean())
    twins, london = np.array(twins, dtype=float), np.array(london, dtype=float)
    if np.isnan(twins).any():
        return None
    twin_idx = 100 * twins / twins.mean()
    london_idx = 100 * london / np.nanmean(london)
    peak = int(np.argmax(twin_idx))
    swing = twin_idx.max() - twin_idx.min()
    # Best sustained season: the 3 consecutive months (wrapping Dec->Jan) with the highest average.
    # A single spike (e.g. Christmas mixers) is reported separately, not used for the launch month.
    windows = [np.mean([twin_idx[(s + i) % 12] for i in range(3)]) for s in range(12)]
    start = int(np.argmax(windows))
    season = [MONTHS[(start + i) % 12] for i in range(3)]
    return {"twin_index": twin_idx, "london_index": london_idx, "peak": MONTHS[peak],
            "peak_value": twin_idx[peak], "swing": swing, "season": season,
            "season_value": windows[start], "launch_month": MONTHS[(start - 1) % 12],
            "spike_outside_season": MONTHS[peak] not in season,
            "seasonal": windows[start] >= 105}


# =============================================================================
# Charts
# =============================================================================
def watermark(fig, synthetic: bool) -> None:
    """Stamp 'SYNTHETIC' on charts made in practice mode, so they are never mistaken for results."""
    if synthetic:
        fig.text(0.5, 0.5, "SYNTHETIC PRACTICE DATA", fontsize=28, color="red",
                 alpha=0.15, ha="center", va="center", rotation=20)


def chart_ranking(plan: Plan, wd: WardData, path: Path, synthetic: bool) -> None:
    """Bar chart of the best-matching wards (or fallback fit scores), coloured by role:
    launch, comparison, near-miss or other."""
    col = "similarity" if plan.mode == "twins" else "fit_score"
    comp = set(plan.comparisons.values())
    keep = list(plan.ranking.index[:N_TWINS_SHOWN]) + plan.launch + list(comp) + plan.near_misses[:3]
    top = plan.ranking.loc[list(dict.fromkeys(keep))].sort_values(col, ascending=False)
    colours, tags = [], []
    for code in top.index:
        if code in plan.launch:
            colours.append("#1f6f5c"); tags.append("launch")
        elif code in comp:
            colours.append("#7fb8a8"); tags.append("comparison")
        elif code in plan.near_misses:
            colours.append("#d9822b"); tags.append("near-miss")
        elif plan.mode == "twins" and plan.ranking.at[code, "category_gap_sd"] < -MAX_LAUNCH_GAP_SD:
            colours.append("#e8c9a8"); tags.append(f"weaker in {label(plan.product['category'])}")
        else:
            colours.append("#c9c9c9"); tags.append("")
    fig, ax = plt.subplots(figsize=(9, 0.45 * len(top) + 1.5))
    y = np.arange(len(top))[::-1]
    ax.barh(y, top[col], color=colours)
    ax.set_yticks(y, [wd.full_name(c) for c in top.index], fontsize=8)
    for yi, value, tag in zip(y, top[col], tags):
        ax.text(value, yi, f" {value:.2f} {tag}", va="center", fontsize=7)
    title = (f"Basket twins of {plan.product['sells_well_in']}" if plan.mode == "twins"
             else "Fallback fit ranking (no known selling area)")
    ax.set_xlim(0, top[col].max() * 1.35 if top[col].max() > 0 else 1)
    ax.set_title(f"{plan.product['product']}: {title}", fontsize=10)
    ax.set_xlabel("Basket similarity (1 = identical mix)" if plan.mode == "twins" else "Fit score")
    watermark(fig, synthetic)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def chart_why(plan: Plan, wd: WardData, path: Path, synthetic: bool) -> None:
    """Twins mode only: compare the anchor, the top launch ward and (if any) a near-miss
    on the 10 features that most define the anchor, plus the product's own category."""
    if plan.mode != "twins" or not plan.launch:
        return
    anchor = wd.z.loc[plan.anchor_codes].mean()
    twin = wd.z.loc[plan.launch[0]]
    feats = list(anchor.abs().sort_values(ascending=False).index[:10])
    if plan.product["category"] not in feats:
        feats.append(plan.product["category"])
    series = [("Where it sells well", anchor, "#333333"), (f"Top launch ward: {wd.name(plan.launch[0])}", twin, "#1f6f5c")]
    if plan.near_misses:
        series.append((f"Near-miss: {wd.name(plan.near_misses[0])}", wd.z.loc[plan.near_misses[0]], "#d9822b"))
    fig, ax = plt.subplots(figsize=(9, 6))
    y = np.arange(len(feats))
    h = 0.8 / len(series)
    for i, (name, values, colour) in enumerate(series):
        ax.barh(y + i * h, values[feats], height=h, label=name, color=colour)
    ax.set_yticks(y + h * (len(series) - 1) / 2,
                  [label(f) + ("  <- product's category" if f == plan.product["category"] else "")
                   for f in feats], fontsize=8)
    ax.axvline(0, color="black", lw=0.8)
    ax.invert_yaxis()
    ax.set_xlabel("Compared with the London average (0) in standard deviations")
    ax.set_title(f"{plan.product['product']}: why these areas match", fontsize=10)
    ax.legend(fontsize=8, loc="lower right")
    watermark(fig, synthetic)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def chart_timing(plan: Plan, path: Path, synthetic: bool) -> None:
    """Line chart of the monthly index for the launch wards vs all London wards,
    with the best 3-month season shaded."""
    t = plan.timing
    if not t:
        return
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(MONTHS, t["twin_index"], marker="o", color="#1f6f5c", label="Launch wards")
    ax.plot(MONTHS, t["london_index"], marker=".", color="#999999", ls="--", label="All London wards")
    ax.axhline(100, color="black", lw=0.6)
    idx = [MONTHS.index(m) for m in t["season"]]
    for i in idx:
        ax.axvspan(i - 0.5, i + 0.5, color="#1f6f5c", alpha=0.08)
    ax.text(idx[1], ax.get_ylim()[1], "best 3-month season", ha="center", va="top", fontsize=8)
    ax.set_ylabel("Share of category vs yearly average (100 = average)")
    ax.set_title(f"{plan.product['product']}: when do these areas buy "
                 f"{label(plan.product['category'])}? (2015)", fontsize=10)
    ax.legend(fontsize=8)
    watermark(fig, synthetic)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def chart_tribes(tribes: Tribes, path: Path, synthetic: bool) -> None:
    """Heat map of each neighbourhood type's average z-scores (red = above London average)."""
    c = tribes.centres
    fig, ax = plt.subplots(figsize=(11, 0.6 * tribes.k + 2.5))
    im = ax.imshow(c.to_numpy(), cmap="RdBu_r", vmin=-2, vmax=2, aspect="auto")
    ax.set_xticks(range(c.shape[1]), [label(f) for f in c.columns], rotation=60, ha="right", fontsize=8)
    counts = tribes.labels.value_counts()
    ax.set_yticks(range(tribes.k), [f"{tribes.names[i]} ({counts.get(i, 0)} wards)" for i in range(tribes.k)],
                  fontsize=8)
    fig.colorbar(im, ax=ax, label="vs London average (SD)")
    ax.set_title(f"Neighbourhood types from basket mix (k={tribes.k}, silhouette {tribes.silhouette:.2f})",
                 fontsize=10)
    watermark(fig, synthetic)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def chart_validation(val: pd.DataFrame, path: Path, synthetic: bool) -> None:
    """Bar chart of the global hold-out check: how much closer twins are than random
    wards on features hidden from matching and on census variables never used."""
    v = val.sort_values("closer_than_random_pct")
    colours = ["#d9822b" if k.startswith("never") else "#1f6f5c" for k in v["kind"]]
    fig, ax = plt.subplots(figsize=(8, 0.3 * len(v) + 1.8))
    ax.barh([label(f) for f in v["feature"]], v["closer_than_random_pct"], color=colours)
    ax.axvline(0, color="black", lw=0.8)
    ax.set_xlabel("How much closer twins are than random wards on the HIDDEN feature (%)")
    ax.set_title("Hold-out check: green = hidden while matching, orange = census, never used", fontsize=9)
    ax.tick_params(axis="y", labelsize=8)
    watermark(fig, synthetic)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


# =============================================================================
# Reports
# =============================================================================
def plan_report(plan: Plan, wd: WardData, tribes: Tribes, synthetic: bool) -> str:
    """Write the launch plan for one product as Markdown: anchor, trial table, top twins,
    near-misses, timing and how much to trust the result."""
    p = plan.product
    L = []
    if synthetic:
        L.append("> **SYNTHETIC PRACTICE DATA. These results are not real.**\n")
    L.append(f"# Launch plan: {p['product']} ({p['brand']})\n")
    L.append(f"Category: **{label(p['category'])}**  ")
    if plan.mode == "twins":
        src = f" (source: {p['source']})" if p["source"] else ""
        L.append(f"Sells well in: **{p['sells_well_in']}**{src}  ")
        L.append("Matched as: " + "; ".join(plan.place_notes) + "  ")
        counts = pd.Series([tribe_of(tribes, wd, c) for c in plan.anchor_codes]).value_counts()
        L.append("Neighbourhood type(s) of that area: "
                 + ", ".join(f"{t} ({n} ward{'s' if n > 1 else ''})" for t, n in counts.items()) + "\n")
    else:
        L.append("**No known selling area**, so this uses the weaker fallback ranking "
                 "(category share plus nutrient fit). " + " ".join(plan.place_notes) + "\n")

    L.append("## Recommended trial\n")
    L.append("| Launch ward | Comparison ward (no launch) | Why the launch ward fits |")
    L.append("|---|---|---|")
    for code in plan.launch:
        comp = plan.comparisons.get(code)
        why = plan.ranking.at[code, "matches_on"] if plan.mode == "twins" else plan.ranking.at[code, "reason"]
        L.append(f"| {wd.full_name(code)} | {wd.full_name(comp) if comp else 'none available'} | {why} |")
    L.append("\nRun the product in the launch wards only, then compare its sales against the paired "
             "comparison wards. Because each pair has similar baskets, a clear gap is better evidence "
             "than a launch with nothing to compare against.\n")

    if plan.mode == "twins":
        L.append(f"## Top {N_TWINS_SHOWN} basket twins\n")
        L.append("| # | Ward | Similarity | Role | Type | Matches on | Differs on |")
        L.append("|---|---|---|---|---|---|---|")
        comp = set(plan.comparisons.values())
        for i, code in enumerate(plan.ranking.index[:N_TWINS_SHOWN], 1):
            r = plan.ranking.loc[code]
            role = ("launch" if code in plan.launch else "comparison" if code in comp
                    else "near-miss" if code in plan.near_misses
                    else f"weaker in {label(p['category'])}" if r["category_gap_sd"] < -MAX_LAUNCH_GAP_SD
                    else "reserve")
            L.append(f"| {i} | {wd.full_name(code)} | {r['similarity']:.2f} | {role} | {tribe_of(tribes, wd, code)} "
                     f"| {r['matches_on']} | {r['differs_on']} |")
        L.append(f"\nLaunch wards must be no more than {MAX_LAUNCH_GAP_SD} SD below the selling area on "
                 f"{label(p['category'])}; that is why some close twins are not chosen.")
        L.append("")
        L.append("## Near-misses (why not here?)\n")
        if plan.near_misses:
            L.append(f"These wards look like {p['sells_well_in']} overall but buy far less "
                     f"{label(p['category'])}, so the product may struggle there:\n")
            for code in plan.near_misses[:5]:
                r = plan.ranking.loc[code]
                L.append(f"- {wd.full_name(code)}: similarity {r['similarity']:.2f}, "
                         f"{label(p['category'])} {abs(r['category_gap_sd']):.1f} SD lower")
        else:
            L.append(f"None of the top {CANDIDATE_POOL} twins is much weaker in {label(p['category'])}.")
        L.append("")

    L.append("## Timing\n")
    t = plan.timing
    if t is None:
        L.append("Monthly files not available, so no timing advice.\n")
    elif t["seasonal"]:
        L.append(f"In the launch wards, {label(p['category'])} is strongest over "
                 f"**{', '.join(t['season'])}** ({t['season_value'] - 100:+.0f}% vs their yearly average). "
                 f"A launch in **{t['launch_month']}** puts the product on shelf as that season starts "
                 "(rule of thumb).")
        if t["spike_outside_season"]:
            L.append(f"There is also a one-month spike in **{t['peak']}** ({t['peak_value'] - 100:+.0f}%), "
                     "which may reflect a specific occasion (e.g. Christmas) rather than demand for "
                     "this kind of product; check whether it fits before chasing it.")
        L.append("Based on one year (2015), so a one-off event could look like a season.\n")
    else:
        L.append(f"No strong seasonal pattern (best 3-month stretch only {t['season_value'] - 100:+.0f}% "
                 "above average), so location matters more than timing for this category.\n")

    L.append("## How much to trust this\n")
    if plan.permutation:
        pc = plan.permutation
        L.append(f"- **Hidden-category test:** with {label(pc['category'])} removed from matching, "
                 f"the twins were still closer to the anchor on it than {pc['beats_random_pct']:.0f}% "
                 f"of random sets of wards (p = {pc['p_value']:.3f}).")
    if plan.stability:
        s = plan.stability
        L.append(f"- **Stability:** dropping one feature at a time kept on average "
                 f"{100 * s['mean_overlap']:.0f}% of the top-{N_TWINS_SHOWN} list "
                 f"(lowest {100 * s['worst_overlap']:.0f}%, when {label(s['worst_feature'])} was dropped).")
    L.append("- **Limits:** 2015 data; Tesco Clubcard shoppers only; store coverage is uneven across "
             "London; areas are described by their typical item, not individual shoppers; 'sells well "
             "in' is a place, not sales figures. This suggests where a trial is most likely to fit, "
             "not guaranteed sales.")
    return "\n".join(L) + "\n"


# =============================================================================
# Synthetic practice data (fake, for testing the pipeline before the real files)
# =============================================================================
def make_synthetic_data(folder: Path) -> None:
    """Generate FAKE ward files and an example products file for practice (--synthetic).
    Never use these results for decisions."""
    rng = np.random.default_rng(7)
    folder.mkdir(parents=True, exist_ok=True)
    cats = list(CATEGORY_ALIASES)
    base = np.array([.17, .13, .03, .03, .02, .12, .01, .02, .20, .06, .04, .02, .09, .03, .01, .01, .01])
    base = base / base.sum()
    n_types = 5
    type_profiles = [base * rng.lognormal(0, 0.45, len(base)) for _ in range(n_types)]
    type_profiles = [t / t.sum() for t in type_profiles]
    type_age = rng.normal(37, 4, n_types)
    type_density = rng.lognormal(8.5, 0.5, n_types)

    rows, lookup = [], []
    for b in range(33):
        mix = rng.dirichlet(np.ones(n_types) * 0.6)
        for w in range(18):
            code = f"SYN{b:02d}{w:03d}"
            t = rng.choice(n_types, p=mix)
            shares = rng.dirichlet(type_profiles[t] * 400)
            row = {"area_id": code}
            for c, s in zip(cats, shares):
                row[f"f_{c}"] = s
            d = dict(zip(cats, shares))
            row["sugar"] = 4 + 30 * d["sweets"] + 25 * d["soft_drinks"] + rng.normal(0, 0.4)
            row["fat"] = 4 + 25 * d["fats_oils"] + 12 * d["sweets"] + 20 * d["meat_red"] + rng.normal(0, 0.3)
            row["saturate"] = 0.4 * row["fat"] + rng.normal(0, 0.1)
            row["protein"] = 3 + 40 * (d["meat_red"] + d["poultry"] + d["fish"]) + 8 * d["dairy"] + rng.normal(0, 0.2)
            row["carb"] = 10 + 40 * d["grains"] + 30 * d["sweets"] + rng.normal(0, 0.5)
            row["fibre"] = 1 + 5 * d["fruit_veg"] + 3 * d["grains"] + rng.normal(0, 0.05)
            row["salt"] = 0.3 + 3 * d["sauces"] + 2 * d["readymade"] + rng.normal(0, 0.03)
            row["energy_tot"] = 9 * row["fat"] + 4 * (row["carb"] + row["protein"]) + rng.normal(0, 3)
            row["representativeness_norm"] = rng.beta(4, 2)
            row["population"] = int(rng.normal(12000, 2500))
            row["avg_age"] = type_age[t] + rng.normal(0, 1.5)
            row["people_per_sq_km"] = type_density[t] * rng.lognormal(0, 0.2)
            rows.append(row)
            lookup.append({"WD15CD": code, "WD15NM": f"Syn Ward {b:02d}-{w:02d}",
                           "LAD15NM": f"Syn Borough {b:02d}"})
    year = pd.DataFrame(rows)
    year.to_csv(folder / "year_osward_grocery.csv", index=False)
    pd.DataFrame(lookup).to_csv(folder / "ward_lookup.csv", index=False)

    peak_month = {"soft_drinks": 6, "water": 6, "beer": 6, "fruit_veg": 6, "sweets": 11, "wine": 11,
                  "spirits": 11, "tea_coffee": 0}
    for m, month in enumerate(MONTHS):
        monthly = year.copy()
        for c in cats:
            if c in peak_month:
                mult = 1 + 0.25 * np.cos(2 * np.pi * (m - peak_month[c]) / 12)
            else:
                mult = 1.0
            monthly[f"f_{c}"] = year[f"f_{c}"] * mult * rng.lognormal(0, 0.03, len(year))
        fcols = [f"f_{c}" for c in cats]
        monthly[fcols] = monthly[fcols].div(monthly[fcols].sum(axis=1), axis=0)
        monthly.to_csv(folder / f"{month}_osward_grocery.csv", index=False)

    example = pd.DataFrame([
        ["Example Brew", "Raspberry Kombucha", "soft_drinks", 18, 3.9, 0, 0, 0, 0, 0.01,
         "Syn Borough 04", "brand rep (verbal)", "", "", "synthetic example"],
        ["Example Bars", "Peanut Protein Bar", "sweets", 410, 4.5, 18, 5, 30, 8, 0.4,
         "Syn Ward 10-03; Syn Ward 10-05", "brand website stockist list", "", "", "synthetic example"],
        ["Example Sauce Co", "Smoky Hot Sauce", "sauces", 60, 3, 0.5, 0.1, 1, 1, 2.1,
         "", "", "", "", "no known selling area: tests the fallback"],
    ], columns=PRODUCT_COLUMNS)
    example.to_csv(folder / "launch_products_example.csv", index=False)


# =============================================================================
# Main
# =============================================================================
def main() -> int:
    """Run the whole pipeline: load data [1/6], prepare wards [2/6], neighbourhood types
    [3/6], global validation [4/6], one launch plan per product [5/6], summary [6/6]."""
    parser = argparse.ArgumentParser(description="Basket Twins launch planner")
    parser.add_argument("--check", action="store_true", help="only inspect data files and columns")
    parser.add_argument("--synthetic", action="store_true", help="practice run on fake data")
    args = parser.parse_args()
    check_only = args.check or CHECK_DATA_ONLY
    synthetic = args.synthetic or USE_SYNTHETIC_DATA

    # Use the SETTINGS paths if they exist on this computer, otherwise the script's own folder.
    data_dir = use_path(DATA_DIR, "data")
    products_file = use_path(PRODUCTS_FILE, "launch_products.csv")
    out_dir = use_path(OUTPUT_DIR, "outputs")
    if synthetic:
        data_dir = SCRIPT_DIR / "data_synthetic"
        products_file = data_dir / "launch_products_example.csv"
        out_dir = SCRIPT_DIR / "outputs_synthetic"
        say("*** SYNTHETIC PRACTICE MODE: generating fake data. Do not use these results. ***")
        make_synthetic_data(data_dir)

    say("\n[1/6] Loading data")
    year_path = find_year_file(data_dir)
    table, report, all_cols = read_area_file(year_path)
    print_column_report(year_path, report, all_cols, len(table))
    month_paths = find_month_files(data_dir)
    say(f"\n  Monthly ward files found: {len(month_paths)}/12"
        + (f" (missing: {', '.join(m for m in MONTHS if m not in month_paths)})" if len(month_paths) < 12 else ""))
    lookup_path = find_lookup_file(data_dir)
    names, boroughs = load_lookup(lookup_path, table.index) if lookup_path else ({}, {})
    if not lookup_path:
        say("  No ward lookup file: outputs will show ward codes, and 'sells_well_in' must use codes.")

    if check_only:
        if not products_file.exists():
            write_blank_template(products_file)
            say(f"\n  Created blank products file: {products_file}")
        say("\nCheck finished. If a category or nutrient is MISSING above, add its real column "
            "name to the matching ALIASES list near the top of the script.")
        return 0

    month_tables = {}
    for month, path in month_paths.items():
        month_tables[month], _, _ = read_area_file(path)

    say("\n[2/6] Preparing wards")
    wd = prepare_wards(table, report, names, boroughs)

    out_dir.mkdir(parents=True, exist_ok=True)
    say("\n[3/6] Building neighbourhood types (k-means)")
    tribes = build_tribes(wd)
    chart_tribes(tribes, out_dir / "neighbourhood_types.png", synthetic)

    say("\n[4/6] Validation: do twins share features they were never matched on?")
    val = holdout_validation(wd)
    val.to_csv(out_dir / "validation_holdout.csv", index=False)
    chart_validation(val, out_dir / "validation_holdout.png", synthetic)
    hidden = val[val["kind"].str.startswith("hidden")]
    census = val[val["kind"].str.startswith("never")]
    say(f"  Hidden features: twins are a median {hidden['closer_than_random_pct'].median():.0f}% closer "
        f"than random wards (twins win for {hidden['wards_where_twins_win_pct'].median():.0f}% of wards).")
    for _, r in census.iterrows():
        say(f"  {label(r['feature'])}: twins {r['closer_than_random_pct']:.0f}% closer than random.")

    say("\n[5/6] Making launch plans")
    products = load_products(products_file, wd)
    summary, all_reports = [], []
    for product in products:
        plan = make_plan(wd, product, month_tables)
        folder = out_dir / slugify(f"{product['brand']}_{product['product']}")
        folder.mkdir(parents=True, exist_ok=True)
        chart_ranking(plan, wd, folder / "1_twins_ranking.png", synthetic)
        chart_why(plan, wd, folder / "2_why_they_match.png", synthetic)
        chart_timing(plan, folder / "3_timing.png", synthetic)
        text = plan_report(plan, wd, tribes, synthetic)
        (folder / "launch_plan.md").write_text(text, encoding="utf-8")
        plan.ranking.head(CANDIDATE_POOL).assign(ward=lambda d: [wd.full_name(c) for c in d.index]) \
            .to_csv(folder / "ranking_top.csv")
        all_reports.append(text)
        summary.append({
            "brand": product["brand"], "product": product["product"], "mode": plan.mode,
            "launch_wards": "; ".join(wd.full_name(c) for c in plan.launch),
            "comparison_wards": "; ".join(wd.full_name(plan.comparisons[c]) for c in plan.launch
                                          if c in plan.comparisons),
            "near_misses": "; ".join(wd.full_name(c) for c in plan.near_misses[:3]),
            "launch_month": plan.timing["launch_month"] if plan.timing and plan.timing["seasonal"] else "",
            "hidden_category_p": round(plan.permutation["p_value"], 4) if plan.permutation else "",
        })
        say(f"  {product['product']}: {plan.mode}; launch in {', '.join(wd.name(c) for c in plan.launch)}"
            + (f"; month {plan.timing['launch_month']}" if plan.timing and plan.timing["seasonal"] else ""))
        for note in plan.place_notes:
            if "NOT FOUND" in note:
                say(f"    {note}")

    say("\n[6/6] Writing summary")
    pd.DataFrame(summary).to_csv(out_dir / "summary.csv", index=False)
    (out_dir / "all_launch_plans.md").write_text("\n\n---\n\n".join(all_reports), encoding="utf-8")
    say(f"\nDone. Open {out_dir} for the charts and launch plans.")
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except BasketTwinsError as err:
        say(f"\nSTOPPED: {err}")
        sys.exit(1)