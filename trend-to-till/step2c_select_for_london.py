"""
TREND TO TILL, STEP 2c: turns the RL predictions into the input for Basket Twins
=========================================================================
Pipeline:  step1_find_trends.py -> step2a_build_success_inputs.py -> step2b_predict_success.py predict
           -> step2c_select_for_london.py (this file) -> step3_basket_twins.py

WHAT IT DOES
  1. Reads success_model/model/predictions.csv (the RL output): for every product, a UK
     success probability, a verdict, and a success probability per UK region,
     including London (column p_London).
  2. Decides which products go on to Basket Twins. Basket Twins only covers London,
     so a product goes forward if:
        - its London probability is at least MIN_LONDON_PROB (default 0.5), and
        - its UK verdict is not "Likely fail" (unless KEEP_LIKELY_FAIL = True).
     Every decision and its reason is written to london_selection.csv.
  3. Writes basket_twins_input.csv in exactly Basket Twins' 15-column format:
        - tesco_category: mapped from the product's category (CATEGORY_TO_TESCO below)
        - sells_well_in (the anchor) and nutrition: taken from inputs/london_early_adopter_wards.csv if it has the
          product (optional). Without an anchor, Basket Twins uses its weaker
          "fallback" ranking (how much each ward buys that category).
        - notes: the RL probabilities and verdict, so they travel with the product.

HOW TO RUN
  1. pip install pandas
  2. Paths are relative to this file's folder, so no editing is needed.
  3. python step2c_select_for_london.py   then: python step3_basket_twins.py   (or just main.py)
"""

import sys
from pathlib import Path

import pandas as pd

# =============================================================================
# SETTINGS (plain text paths; keep the r before the quotes)
# =============================================================================
PREDICTIONS_CSV = r"success_model/model/predictions.csv"
CANDIDATES_CSV = r"outputs/2_success/candidate_products.csv"
ANCHORS_CSV = r"inputs/london_early_adopter_wards.csv"                 # optional
OUTPUT_CSV = r"outputs/2_success/basket_twins_input.csv"          # Basket Twins reads this
SELECTION_CSV = r"outputs/2_success/london_selection.csv"      # why each product was kept or not

MIN_LONDON_PROB = 0.5          # London success probability needed to go on to Basket Twins
KEEP_LIKELY_FAIL = False       # True = also send "Likely fail" products forward
DEFAULT_BRAND = "Concept (trend)"

# Step 1's product categories -> the 17 Tesco categories Basket Twins uses (judgement calls)
CATEGORY_TO_TESCO = {
    "Milk-based drinks": "tea_coffee",            # e.g. iced lattes
    "Tea and coffee": "tea_coffee",
    "Soft drinks (with added sugar)": "soft_drinks",
    "Soft drinks (no added sugar)": "soft_drinks",
    "Savoury snacks": "sauces",                   # no snacks category; pickles closest to sauces
    "Breakfast cereals": "grains",
    "Confectionery": "sweets",
    "Ice cream and frozen desserts": "dairy",
    "Cakes": "sweets",
    "Sweet biscuits and bars": "sweets",
    "Morning goods and pastries": "grains",
    "Desserts and puddings": "sweets",
    "Sweetened yoghurt": "dairy",
    "Pizza": "readymade",
    "Potato products": "fruit_veg",
    "Ready meals and sandwiches": "readymade",
    "Sauces and condiments": "sauces",
    "Nuts and seeds": "fruit_veg",
    "Dairy products": "dairy",
    "Other food": "readymade",
}
LAUNCH_COLUMNS = ["brand", "product_name", "tesco_category", "energy_kcal_100g", "sugar_g_100g",
                  "fat_g_100g", "saturates_g_100g", "protein_g_100g", "fibre_g_100g",
                  "salt_g_100g", "sells_well_in", "sells_well_source", "channel", "price_gbp",
                  "notes"]
CARRY_FROM_ANCHORS = ["energy_kcal_100g", "sugar_g_100g", "fat_g_100g", "saturates_g_100g",
                      "protein_g_100g", "fibre_g_100g", "salt_g_100g", "sells_well_in",
                      "sells_well_source", "channel", "price_gbp"]

SCRIPT_DIR = Path(__file__).resolve().parent


def use_path(text, fallback_name=None):
    """Turn a SETTINGS path into a full path. Relative paths (the default) are read
    from this script's folder, so the project runs wherever it is unzipped, on any
    computer. An absolute path (e.g. r"/Users/me/data") is used exactly as written."""
    p = Path(text)
    return p if p.is_absolute() else SCRIPT_DIR / p


def key(name):
    """Forgiving product-name key for joining files."""
    return str(name).strip().lower()


def main():
    pred_path = use_path(PREDICTIONS_CSV, "success_model/model/predictions.csv")
    cand_path = use_path(CANDIDATES_CSV, "success_model/data/candidate_products.csv")
    anchors_path = use_path(ANCHORS_CSV, "inputs/london_early_adopter_wards.csv")
    out_path = use_path(OUTPUT_CSV, "basket_twins_input.csv")
    sel_path = use_path(SELECTION_CSV, "london_selection.csv")
    if not pred_path.is_file():
        sys.exit(f"Can't find {pred_path}. Run: python step2b_predict_success.py predict ...")

    preds = pd.read_csv(pred_path)
    if "p_London" not in preds.columns:
        sys.exit("predictions.csv has no p_London column; the RL model must include London.")
    cats = {}
    if cand_path.is_file():
        c = pd.read_csv(cand_path)
        cats = dict(zip(c["product_id"], c["category"]))
    anchors = {}
    if anchors_path.is_file():
        a = pd.read_csv(anchors_path, dtype=str, keep_default_na=False)
        anchors = {key(r["product_name"]): r for _, r in a.iterrows()}
        print(f"Anchors/nutrition available for {len(anchors)} product(s) from {anchors_path.name}.")

    rows, selection = [], []
    for _, p in preds.iterrows():
        name, p_lon = p["product_name"], float(p["p_London"])
        category = cats.get(p["product_id"], "")
        tesco = CATEGORY_TO_TESCO.get(category, "")
        # --- decide whether the product goes on to Basket Twins ---
        if p_lon < MIN_LONDON_PROB:
            keep, reason = False, f"London probability {p_lon:.2f} below {MIN_LONDON_PROB}"
        elif p["uk_verdict"] == "Likely fail" and not KEEP_LIKELY_FAIL:
            keep, reason = False, "UK verdict is Likely fail"
        elif not tesco:
            keep, reason = False, f"category '{category}' has no Tesco mapping (add it to CATEGORY_TO_TESCO)"
        else:
            keep, reason = True, f"London probability {p_lon:.2f}; UK verdict {p['uk_verdict']}"
        selection.append({"product_id": p["product_id"], "product_name": name,
                          "uk_success_probability": p["uk_success_probability"],
                          "uk_verdict": p["uk_verdict"], "p_London": p_lon,
                          "sent_to_basket_twins": "yes" if keep else "no", "reason": reason})
        if not keep:
            continue
        # --- build the Basket Twins row ---
        row = {c: "" for c in LAUNCH_COLUMNS}
        row.update({"brand": DEFAULT_BRAND, "product_name": name, "tesco_category": tesco})
        extra = anchors.get(key(name))
        if extra is not None:
            for col in CARRY_FROM_ANCHORS:
                if col in extra.index and str(extra[col]).strip():
                    row[col] = extra[col]
        row["notes"] = (f"RL: UK success probability {p['uk_success_probability']:.2f} "
                        f"({p['uk_verdict']}); London {p_lon:.2f}; drivers: {p['top_drivers_uk']}")
        rows.append(row)

    pd.DataFrame(selection).to_csv(sel_path, index=False)
    launch = pd.DataFrame(rows, columns=LAUNCH_COLUMNS)
    launch.to_csv(out_path, index=False)

    print("\nPRODUCTS SENT TO BASKET TWINS")
    print("=" * 70)
    for s in selection:
        print(f"  [{s['sent_to_basket_twins']:>3}] {s['product_name'][:42]:<43} {s['reason']}")
    for _, r in launch.iterrows():
        mode = "anchor: " + r["sells_well_in"] if r["sells_well_in"] else "no anchor -> fallback"
        print(f"      {r['product_name'][:40]:<41} [{r['tesco_category']}] {mode}")
    print(f"\nSaved {out_path.name} ({len(launch)} product(s)) and {sel_path.name}.")
    print("Next: python step3_basket_twins.py")


if __name__ == "__main__":
    main()
