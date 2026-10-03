"""
TREND TO TILL: from trend to till, with evidence
================================================
Trend to Till finds WHAT to sell next (a UK food and drink trend), checks WHETHER it is
likely to succeed (a success model), and finds WHERE to launch it first (the London
neighbourhoods that shop like its early adopters, found by our "Basket Twins" method).

Run the whole project with one command:
    python main.py

What it runs, in order (each step can also be run on its own):
  1.  step1_find_trends.py             what's trending in UK food and drink (trend score 0-100)
  2a. step2a_build_success_inputs.py   trends -> products described by 10 attributes
  2b. step2b_predict_success.py        success probability, UK + 9 regions (RL model)
  2c. step2c_select_for_london.py      keeps products likely to succeed in London
  3.  step3_basket_twins.py            where in London to launch, and when
  ->  final_recommendations.csv        one row per product, joining every step

No API key or internet needed: step 1 replays a saved live run, and the RL model
in success_model/model/ is already trained. Results are written to outputs/.
"""

import shutil
import subprocess
import sys
from pathlib import Path

import pandas as pd

HERE = Path(__file__).resolve().parent
MODEL_DIR = HERE / "success_model" / "model"
RL_DATA = HERE / "success_model" / "data"
OUT = HERE / "outputs"


def run(args, title):
    """Run one step with the same Python; stop if it fails."""
    print(f"\n{'=' * 70}\n{title}\n{'=' * 70}", flush=True)
    if subprocess.run([sys.executable] + args, cwd=HERE).returncode != 0:
        sys.exit(f"\nStopped: '{title}' failed (see the message above).")


def ensure_model():
    """The trained RL model ships with the project. If it is missing, retrain it
    (needs the training files products.csv, historical_sales.csv, incoming_sales.csv)."""
    if (MODEL_DIR / "model_state.json").is_file():
        return
    needed = ["products.csv", "historical_sales.csv", "incoming_sales.csv"]
    if any(not (RL_DATA / f).is_file() for f in needed):
        sys.exit("The trained model success_model/model/model_state.json is missing, and the "
                 f"training files ({', '.join(needed)}) are not in success_model/data/ to retrain it.")
    rl = "step2b_predict_success.py"
    run([rl, "train", "--data", str(RL_DATA), "--model", str(MODEL_DIR)], "Training the RL model (first time only)")
    run([rl, "update", "--data", str(RL_DATA), "--model", str(MODEL_DIR),
         "--incoming", str(RL_DATA / "incoming_sales.csv")], "Updating the RL model with newer sales (first time only)")


def final_summary():
    """Join every step into final_recommendations.csv (one row per product)."""
    trends = pd.read_csv(OUT / "1_trends" / "uk_trending_products.csv")
    sel = pd.read_csv(OUT / "2_success" / "london_selection.csv")
    summary = OUT / "3_basket_twins" / "summary.csv"
    plans = pd.read_csv(summary) if summary.is_file() else pd.DataFrame(columns=["product"])
    df = (sel.merge(trends[["trend", "trend_score"]], left_on="product_name", right_on="trend", how="left")
             .merge(plans, left_on="product_name", right_on="product", how="left"))
    out = df[["product_name", "trend_score", "uk_success_probability", "uk_verdict", "p_London",
              "sent_to_basket_twins", "reason"]].copy()
    for col in ["mode", "launch_wards", "comparison_wards", "near_misses", "launch_month"]:
        out[col] = df[col] if col in df.columns else ""
    out = out.rename(columns={"mode": "basket_twins_mode"}).sort_values("uk_success_probability", ascending=False)
    out.to_csv(HERE / "final_recommendations.csv", index=False)
    print(f"\n{'=' * 70}\nFINAL RECOMMENDATIONS  ->  final_recommendations.csv\n{'=' * 70}")
    for _, r in out.iterrows():
        where = r["launch_wards"] if isinstance(r["launch_wards"], str) else "not sent to Basket Twins"
        print(f"  {r['product_name'][:40]:<41} UK {r['uk_success_probability']:.2f}  "
              f"London {r['p_London']:.2f}  ->  {where[:55]}")


def main():
    run(["step1_find_trends.py"], "STEP 1: what's trending")
    run(["step2a_build_success_inputs.py"], "STEP 2a: describe each trend with 10 attributes")
    ensure_model()
    run(["step2b_predict_success.py", "predict", "--data", str(RL_DATA), "--model", str(MODEL_DIR),
         "--products", str(OUT / "2_success" / "candidate_products.csv")], "STEP 2b: predict success (UK + regions)")
    shutil.copy(MODEL_DIR / "predictions.csv", OUT / "2_success" / "success_predictions.csv")  # visible copy
    run(["step2c_select_for_london.py"], "STEP 2c: select products for London")
    run(["step3_basket_twins.py"], "STEP 3: Basket Twins (where in London, and when)")
    final_summary()


if __name__ == "__main__":
    main()
