"""
TREND TO TILL, STEP 2a: turns the trends from step 1 into products the RL model can score
==================================================================================
Pipeline:  step1_find_trends.py -> step2a_build_success_inputs.py (this file) -> step2b_predict_success.py predict
           -> step2c_select_for_london.py -> step3_basket_twins.py

WHAT IT DOES
  The RL model (step2b_predict_success.py) scores product ideas from 10 attributes,
  each scored 0-10 (5 = average):
      search_trend_growth, workaround_evidence, competitor_dissatisfaction,
      price_competitiveness, quality_score, brand_strength, sustainability_score,
      innovation_level, marketing_spend, distribution_reach
  This file writes one row per trend into success_model/data/candidate_products.csv.

WHERE THE SCORES COME FROM (honestly)
  - search_trend_growth = trend_score / 10, from step 1 (e.g. a trend score of 72 -> 7.2).
    This is the only attribute our data measures.
  - The other 9 attributes describe the brand's plans and the competition (price, quality,
    marketing budget, distribution...). We do not know them for a product idea, so they
    default to 5 = "average" (which has no effect on the score).
  - If a brand knows them, put them in inputs/product_attribute_scores.csv (optional): one row per
    product_name, any attribute columns you know. Those values replace the defaults.

HOW TO RUN
  1. pip install pandas
  2. Paths are relative to this file's folder, so no editing is needed.
  3. python step2a_build_success_inputs.py   (or just run main.py)
"""

import sys
from pathlib import Path

import pandas as pd

# =============================================================================
# SETTINGS (plain text paths; keep the r before the quotes)
# =============================================================================
TRENDS_CSV = r"outputs/1_trends/uk_trending_products.csv"
ATTRIBUTES_CSV = r"success_model/data/attributes.csv"
BRAND_INPUTS_CSV = r"inputs/product_attribute_scores.csv"   # optional
OUTPUT_CSV = r"outputs/2_success/candidate_products.csv"

LAUNCH_YEAR, LAUNCH_MONTH = 2026, 11     # planned launch date written for every candidate
DEFAULT_SCORE = 5.0                      # 5 = average, i.e. "unknown, no effect"

SCRIPT_DIR = Path(__file__).resolve().parent


def use_path(text, fallback_name=None):
    """Turn a SETTINGS path into a full path. Relative paths (the default) are read
    from this script's folder, so the project runs wherever it is unzipped, on any
    computer. An absolute path (e.g. r"/Users/me/data") is used exactly as written."""
    p = Path(text)
    return p if p.is_absolute() else SCRIPT_DIR / p


def main():
    trends_path = use_path(TRENDS_CSV, "outputs/1_trends/uk_trending_products.csv")
    attrs_path = use_path(ATTRIBUTES_CSV, "success_model/data/attributes.csv")
    inputs_path = use_path(BRAND_INPUTS_CSV, "inputs/product_attribute_scores.csv")
    out_path = use_path(OUTPUT_CSV, "success_model/data/candidate_products.csv")

    for p in (trends_path, attrs_path):
        if not p.is_file():
            sys.exit(f"Can't find {p}. Run step1_find_trends.py first, and keep success_model/ next to this file.")

    trends = pd.read_csv(trends_path)
    attributes = pd.read_csv(attrs_path)["attribute"].tolist()

    # Optional brand inputs: product_name + any attribute columns (blank = unknown)
    brand = {}
    if inputs_path.is_file():
        b = pd.read_csv(inputs_path)
        for _, r in b.iterrows():
            known = {a: float(r[a]) for a in attributes if a in b.columns and pd.notna(r[a])}
            if known:
                brand[str(r["product_name"]).strip().lower()] = known
        print(f"Using brand inputs for {len(brand)} product(s) from {inputs_path.name}.")

    rows = []
    for i, t in trends.reset_index(drop=True).iterrows():
        row = {"product_id": f"T{i + 1:03d}", "product_name": t["trend"],
               "category": t["product_category"],
               "launch_year": LAUNCH_YEAR, "launch_month": LAUNCH_MONTH}
        for a in attributes:
            row[a] = DEFAULT_SCORE                                     # unknown -> average
        row["search_trend_growth"] = round(min(max(t["trend_score"] / 10, 0), 10), 1)  # from step 1
        for a, v in brand.get(str(t["trend"]).strip().lower(), {}).items():
            row[a] = min(max(v, 0), 10)                                # brand's own numbers win
        rows.append(row)

    out = pd.DataFrame(rows)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(out_path, index=False)
    print(f"Wrote {len(out)} candidate product(s) to {out_path}")
    print(out[["product_id", "product_name", "search_trend_growth"]].to_string(index=False))
    print("\nNext: python step2b_predict_success.py predict (see main.py)")


if __name__ == "__main__":
    main()
