"""
TREND AGENT: an AI agent that finds UK food and drink trends and scores them
============================================================================
TREND TO TILL, STEP 1 OF 3 (run everything with: python main.py):
    step1_find_trends.py (this file) -> step2a_build_success_inputs.py -> step2b_predict_success.py predict
    -> step2c_select_for_london.py -> step3_basket_twins.py

>>> ORGANISERS / JUDGES: RUN THIS IN REPLAY MODE (the default). <<<
    Replay mode needs NO API key, NO internet and costs nothing. It re-scores the
    saved live run in inputs/trend_agent_saved_run.json and reproduces our submitted results exactly.

THE TWO MODES (set MODE in SETTINGS)
  "replay"  Reads a saved agent run (inputs/trend_agent_saved_run.json) and does steps 3-4 below.
            Same input always gives the same output.
  "live"    Calls the Anthropic API so Claude can search the web (steps 1-4). Needs:
              - an Anthropic API key with credit, in the ANTHROPIC_API_KEY environment
                variable (PyCharm: Run > Edit Configurations > Environment variables),
              - web search enabled for the organisation in the Claude Console.
            Each live run saves its raw output as agent_run_<date>_<time>.json, which
            can then be replayed. Results differ between live runs because the web changes.

WHAT HAPPENS
  Step 1  DISCOVER     (live only) Claude searches retailer reports, sales panels, trade
                       press, TikTok, Reddit, X and lists emerging UK food/drink trends
                       (generic product types, no brand names).
  Step 2  INVESTIGATE  (live only) for each trend, more searches return 4-8 evidence items:
                       source type, rising/falling, UK or not, date, stated growth %, URL,
                       plus how easy the product is to buy in UK shops.
  Step 3  VERIFY       Python checks every evidence URL against the URLs the search tool
                       actually returned. Unverified evidence counts half; no URL = dropped.
  Step 4  SCORE        Python (not the AI) computes a 0-100 trend score from 7 components:
                       momentum, breadth, social buzz, UK relevance, recency, credibility,
                       UK whitespace; reduced if evidence shows the trend fading or if fewer
                       than 4 items are verified. Weights are in SETTINGS.

HOW TO RUN
  1. pip install pandas matplotlib anthropic   (anthropic is only needed for live mode)
  2. Paths are relative to this file's folder, so no editing is needed.
  3. Run:  python step1_find_trends.py   (or just run main.py, which runs every step)

INPUT    inputs/trend_agent_saved_run.json (replay mode)
OUTPUTS  outputs/1_trends/uk_trending_products.csv        one row per trend: rank, trend,
                                                        description, category, location,
                                                        trend_score, uk_evidence_count,
                                                        latest_uk_evidence_date
         outputs/1_trends/uk_trending_products_chart.png  bar chart of the scores
Only food and drink trends with at least one UK source are listed.
"""

import json
import math
import os
import re
import sys
import textwrap
from datetime import date, datetime
from pathlib import Path

import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# =============================================================================
# SETTINGS (the only part you may need to change)
# =============================================================================
# Paths: relative to this file's folder (keep the r before the quotes). You can also
# write a full path such as r"/Users/me/some_folder".
MODE = "replay"                        # "live" or "replay"
OUTPUT_FOLDER = r"outputs/1_trends"
REPLAY_FILE = r"inputs/trend_agent_saved_run.json"   # used in replay mode

API_KEY = os.environ.get("ANTHROPIC_API_KEY", "")   # or paste your key here as a string
MODEL = "claude-sonnet-5-5"
MAX_TRENDS = 8                        # trends to investigate (cost grows with this)
SEARCHES_DISCOVER = 8                 # web searches allowed in step 1
SEARCHES_PER_TREND = 6                # web searches allowed per trend in step 2

FOCUS = "food and drink products (no cosmetics, beauty, supplements or alcohol) sold in UK shops"
EXCLUDE_TERMS = []                    # e.g. brand names on The Shelf: ["BrandA", "BrandB"]

# Score weights (they are normalised, so they don't need to add up to 1)
WEIGHTS = {"momentum": 0.25, "breadth": 0.15, "social": 0.10, "uk_relevance": 0.15,
           "recency": 0.10, "credibility": 0.10, "whitespace": 0.15}
SOURCE_CREDIBILITY = {"retail_sales": 1.0, "search_data": 0.9, "retailer_report": 0.8,
                      "market_report": 0.6, "trade_press": 0.6, "social_media": 0.5,
                      "prediction": 0.4}
WHITESPACE = {"none": 1.0, "limited": 0.7, "widespread": 0.2}
RECENCY_HALF_LIFE_DAYS = 180          # evidence loses half its weight every 6 months
GROWTH_FOR_FULL_MOMENTUM = 500        # +500% growth (or more) = full momentum score
UNVERIFIED_WEIGHT = 0.5               # evidence whose URL wasn't in the search results
FULL_EVIDENCE_ITEMS = 4               # fewer verified items than this shrinks the score
TODAY = date.today()
SCRIPT_DIR = Path(__file__).resolve().parent   # folder this file is in (used for path fallback)

CATEGORIES = ["soft_drinks_added_sugar", "soft_drinks_no_added_sugar", "milk_based_drinks",
              "tea_coffee", "savoury_snacks", "breakfast_cereals", "confectionery", "ice_cream",
              "cakes", "sweet_biscuits_bars", "morning_goods", "desserts_puddings",
              "sweetened_yoghurt", "pizza", "potato_products", "ready_meals_sandwiches",
              "sauces_condiments", "nuts_seeds", "dairy_products", "other_food", "out_of_scope"]
FOOD_DRINK_CATEGORIES = [c for c in CATEGORIES if c != "out_of_scope"]   # only these are listed
SOURCE_TYPES = list(SOURCE_CREDIBILITY)

# =============================================================================
# PART 1: THE AGENT (Claude + web search)
# =============================================================================
DISCOVER_PROMPT = f"""You are a retail trend analyst. Today is {TODAY.isoformat()}.
Use web search to find EMERGING consumer trends for {FOCUS}.
Look across different kinds of sources: UK retailer reports and press releases,
grocery sales panels (e.g. NielsenIQ, Worldpanel), search-trend coverage, TikTok,
Reddit, X and Instagram coverage, and food/beauty trade press. Prefer the last 12 months.
Focus on specific product trends (e.g. "pandan drinks"), not vague themes.
Describe each trend as a GENERIC product type with NO brand or company names.
Prioritise trends that are growing abroad or online but are NOT yet widely
available in UK shops (that is where the opportunity is).
{"Do not include these brands or anything named after them: " + ", ".join(EXCLUDE_TERMS) if EXCLUDE_TERMS else ""}

Return up to {MAX_TRENDS} trends as JSON inside <json></json> tags, as a list of objects:
{{"trend_name": str, "category": one of {CATEGORIES},
  "short_name": 2 to 5 word label, e.g. "Pandan iced lattes",
  "origin_country": str (where it is biggest or started),
  "why": two sentences describing the product trend: what the products are,
         and what is happening in the UK, written in the third person (no "I")}}
Return nothing after the closing tag."""

INVESTIGATE_PROMPT = """You are a retail trend analyst. Today is {today}.
Investigate this consumer trend for the UK market: "{trend}" (origin: {origin}).
Use web search to collect 4 to 8 pieces of evidence from DIFFERENT source types.
Prioritise UK sources (UK retailers, UK sales data, UK press, UK social media);
including UK sales or retailer data if it exists, and social media (TikTok, Reddit, X)
if it exists. Also look for evidence that the trend is FADING, and record it if found.
Also judge how easy it is to buy this kind of product in UK shops today.

Only use facts from pages you actually found. Do not invent numbers or URLs.
Keep "trend_name" generic, with no brand or company names.
Return JSON inside <json></json> tags:
{{"trend_name": str,
  "uk_availability": "none" | "limited" | "widespread",
  "availability_reason": str,
  "evidence": [
    {{"source_type": one of {types},
      "direction": "rising" | "falling" | "flat",
      "geography": "UK" | "other",
      "date": "YYYY-MM" (when the data/article is from),
      "growth_pct": number or null (only if the source states a % change),
      "metric_text": short description in your own words,
      "source_name": str,
      "url": str}}
  ]}}
Return nothing after the closing tag."""


def get_client():
    """Create the Anthropic API client (live mode only), with clear errors if the
    library or the API key is missing."""
    try:
        import anthropic
    except ImportError:
        sys.exit("Install the SDK first:  pip install anthropic")
    if not API_KEY:
        sys.exit("No API key. Set the ANTHROPIC_API_KEY environment variable or API_KEY in SETTINGS.")
    return anthropic.Anthropic(api_key=API_KEY)


def run_agent(client, prompt, max_searches):
    """
    One agent task. The API runs the web searches itself (server-side loop).
    Returns (final_text, set_of_urls_the_search_tool_returned).
    """
    tools = [{"type": "web_search_20250305", "name": "web_search", "max_uses": max_searches,
              "user_location": {"type": "approximate", "city": "London", "country": "GB",
                                "timezone": "Europe/London"}}]
    messages = [{"role": "user", "content": prompt}]
    texts, urls = [], set()
    for _ in range(5):                                  # continue if the API pauses a long turn
        response = client.messages.create(model=MODEL, max_tokens=8000,
                                          messages=messages, tools=tools)
        for block in response.content:
            if block.type == "text":
                texts.append(block.text)
            elif block.type == "web_search_tool_result":
                results = getattr(block, "content", [])
                if isinstance(results, list):
                    for r in results:
                        url = getattr(r, "url", None)
                        if url:
                            urls.add(url)
        if response.stop_reason == "pause_turn":
            messages.append({"role": "assistant", "content": response.content})
            continue
        break
    return "".join(texts), urls


def extract_json(text):
    """Pull the JSON answer out of Claude's reply (it is asked to put it between
    <json></json> tags) and parse it."""
    match = re.search(r"<json>(.*?)</json>", text, re.S)
    raw = match.group(1) if match else text
    raw = raw.strip().strip("`").removeprefix("json").strip()
    start = min([i for i in (raw.find("["), raw.find("{")) if i >= 0], default=-1)
    if start < 0:
        raise ValueError("no JSON found")
    return json.loads(raw[start:raw.rfind("]" if raw[start] == "[" else "}") + 1])


def live_run():
    """Live mode: Step 1 (discover trends), then Step 2 (investigate each trend).
    Returns the raw run as a dict, which main() saves as JSON."""
    client = get_client()
    print("Step 1: discovering trends (this takes a minute)...")
    text, urls = run_agent(client, DISCOVER_PROMPT, SEARCHES_DISCOVER)
    try:
        trends = extract_json(text)
    except Exception as error:
        sys.exit(f"Could not read the discovery output ({error}). Raw text:\n{text[:2000]}")
    trends = [t for t in trends if not any(x.lower() in t["trend_name"].lower() for x in EXCLUDE_TERMS)]
    print(f"  Found {len(trends)} trends: " + "; ".join(t["trend_name"] for t in trends))

    run = {"run_date": TODAY.isoformat(), "model": MODEL, "trends": []}
    for i, t in enumerate(trends[:MAX_TRENDS], start=1):
        print(f"Step 2: investigating {i}/{min(len(trends), MAX_TRENDS)}: {t['trend_name']}")
        prompt = INVESTIGATE_PROMPT.format(today=TODAY.isoformat(), trend=t["trend_name"],
                                           origin=t.get("origin_country", ""), types=SOURCE_TYPES)
        text, found_urls = run_agent(client, prompt, SEARCHES_PER_TREND)
        try:
            detail = extract_json(text)
        except Exception as error:
            print(f"  Skipped (could not read output: {error})")
            continue
        run["trends"].append({**t, **detail, "search_urls": sorted(found_urls | urls)})
    return run


# =============================================================================
# PART 2: VERIFY AND SCORE (plain Python, fully explainable)
# =============================================================================
def parse_month(text):
    """Turn '2026-08', '2026-08-15' or '2026' into a date (or None if unreadable)."""
    for fmt in ("%Y-%m-%d", "%Y-%m", "%Y"):
        try:
            return datetime.strptime(str(text)[:len(fmt) + 2].strip(), fmt).date()
        except ValueError:
            continue
    return None


def norm_url(url):
    """Normalise a URL (no https://, no www., no trailing slash) so the same page
    always compares equal during verification."""
    return re.sub(r"^https?://(www\.)?", "", str(url)).split("#")[0].rstrip("/").lower()


def score_trend(t):
    """Returns (score 0-100, components dict, cleaned evidence list)."""
    found = {norm_url(u) for u in t.get("search_urls", [])}
    evidence, seen = [], set()
    for e in t.get("evidence", []):
        url = e.get("url", "")
        if not url or norm_url(url) in seen:
            continue                                   # drop items without a URL, and duplicates
        seen.add(norm_url(url))
        verified = norm_url(url) in found
        when = parse_month(e.get("date"))
        age = (TODAY - when).days if when else 365
        evidence.append({**e, "verified": verified, "age_days": max(age, 0),
                         "weight": 1.0 if verified else UNVERIFIED_WEIGHT})
    if not evidence:
        return 0.0, {k: 0.0 for k in WEIGHTS} | {"fading_share": 0.0, "sufficiency": 0.0}, []

    w_total = sum(e["weight"] for e in evidence)
    rising = [e for e in evidence if e.get("direction") == "rising"]
    falling_share = sum(e["weight"] for e in evidence if e.get("direction") == "falling") / w_total

    # 1 Momentum: median stated growth among rising evidence, on a log scale
    growths = sorted(float(e["growth_pct"]) for e in rising
                     if isinstance(e.get("growth_pct"), (int, float)) and e["growth_pct"] > 0)
    if growths:
        g = growths[len(growths) // 2]
        momentum = min(math.log10(1 + g / 100) / math.log10(1 + GROWTH_FOR_FULL_MOMENTUM / 100), 1)
    else:
        momentum = 0.3 if len(rising) > len(evidence) / 2 else 0.0

    # 2 Breadth: how many different kinds of source agree (5+ types = full marks)
    breadth = min(len({e.get("source_type") for e in evidence}) / 5, 1)
    # 3 Social buzz: social media evidence (2+ items = full marks)
    social = min(sum(e["weight"] for e in evidence if e.get("source_type") == "social_media") / 2, 1)
    # 4 UK relevance: share of evidence about the UK
    uk = sum(e["weight"] for e in evidence if e.get("geography") == "UK") / w_total
    # 5 Recency: evidence halves in value every RECENCY_HALF_LIFE_DAYS
    recency = sum(e["weight"] * 0.5 ** (e["age_days"] / RECENCY_HALF_LIFE_DAYS)
                  for e in evidence) / w_total
    # 6 Credibility: sales data counts more than predictions
    cred = sum(e["weight"] * SOURCE_CREDIBILITY.get(e.get("source_type"), 0.3)
               for e in evidence) / w_total
    # 7 Whitespace: easier to win if few UK products exist yet
    white = WHITESPACE.get(str(t.get("uk_availability", "")).lower(), 0.5)

    comp = {"momentum": momentum, "breadth": breadth, "social": social, "uk_relevance": uk,
            "recency": recency, "credibility": cred, "whitespace": white}
    base = sum(WEIGHTS[k] * v for k, v in comp.items()) / sum(WEIGHTS.values())
    # Evidence sufficiency: 1 verified item keeps 70% of the score, 4+ keep 100%,
    # so one eye-catching number can't push a thinly evidenced trend to the top.
    verified = sum(e["verified"] for e in evidence)
    sufficiency = min(1.0, 0.6 + 0.4 * verified / FULL_EVIDENCE_ITEMS)
    score = 100 * base * (1 - 0.5 * falling_share) * sufficiency   # fading evidence pulls it down
    comp["fading_share"] = falling_share
    comp["sufficiency"] = sufficiency
    return round(score, 1), {k: round(v, 2) for k, v in comp.items()}, evidence


def confidence(evidence):
    """Confidence label from the number of verified evidence items:
    high = 5+, medium = 3-4, low = fewer."""
    verified = sum(e["verified"] for e in evidence)
    if verified >= 5:
        return "high"
    return "medium" if verified >= 3 else "low"


# =============================================================================
# PART 3: OUTPUTS
# =============================================================================
CATEGORY_LABELS = {
    "soft_drinks_added_sugar": "Soft drinks (with added sugar)",
    "soft_drinks_no_added_sugar": "Soft drinks (no added sugar)",
    "milk_based_drinks": "Milk-based drinks", "tea_coffee": "Tea and coffee",
    "savoury_snacks": "Savoury snacks", "breakfast_cereals": "Breakfast cereals",
    "confectionery": "Confectionery", "ice_cream": "Ice cream and frozen desserts",
    "cakes": "Cakes", "sweet_biscuits_bars": "Sweet biscuits and bars",
    "morning_goods": "Morning goods and pastries", "desserts_puddings": "Desserts and puddings",
    "sweetened_yoghurt": "Sweetened yoghurt", "pizza": "Pizza",
    "potato_products": "Potato products", "ready_meals_sandwiches": "Ready meals and sandwiches",
    "sauces_condiments": "Sauces and condiments", "nuts_seeds": "Nuts and seeds",
    "dairy_products": "Dairy products", "other_food": "Other food",
}
OUTPUT_CSV = "uk_trending_products.csv"
OUTPUT_CHART = "uk_trending_products_chart.png"
MIN_UK_EVIDENCE = 1          # a trend needs at least this many UK evidence items to be listed


def short_name(t):
    """A short label for charts: the agent's short_name, or the name cut at '(' or ','."""
    if t.get("short_name"):
        return t["short_name"]
    return re.split(r"[(,]", t["trend_name"])[0].strip()


def write_outputs(run, out):
    """Score every trend and write the outputs: keeps only food/drink trends with
    at least one UK source, then writes uk_trending_products.csv and the chart."""
    out.mkdir(parents=True, exist_ok=True)
    rows = []
    for t in run["trends"]:
        score, comp, evidence = score_trend(t)
        if t.get("category") not in FOOD_DRINK_CATEGORIES:
            print(f"  Not listed (not a food or drink category): {t['trend_name']}")
            continue
        uk = [e for e in evidence if e.get("geography") == "UK"]
        if len(uk) < MIN_UK_EVIDENCE:
            print(f"  Not listed (no UK evidence): {t['trend_name']}")
            continue
        uk_dates = sorted(d for d in (parse_month(e.get("date")) for e in uk) if d)
        rows.append({
            "trend": short_name(t),
            "product_trend_description": t.get("why") or t["trend_name"],
            "product_category": CATEGORY_LABELS.get(t.get("category", ""), t.get("category", "")),
            "location": "United Kingdom",
            "trend_score": score,
            "uk_evidence_count": len(uk),
            "latest_uk_evidence_date": uk_dates[-1].strftime("%Y-%m") if uk_dates else "",
        })
    if not rows:
        sys.exit("No trend had UK evidence, so nothing to list.")

    table = pd.DataFrame(rows).sort_values("trend_score", ascending=False)
    table.insert(0, "rank", range(1, len(table) + 1))
    table.to_csv(out / OUTPUT_CSV, index=False)

    # Chart: trend score per trend, coloured by product category, with UK evidence counts
    cats = list(dict.fromkeys(table["product_category"]))
    palette = plt.get_cmap("tab10")
    colours = {c: palette(i % 10) for i, c in enumerate(cats)}
    data = table.iloc[::-1]
    labels = [textwrap.fill(n, 30) for n in data["trend"]]          # wrap long names
    fig, ax = plt.subplots(figsize=(12, 0.7 * len(table) + 1.6))
    ax.barh(labels, data["trend_score"], color=[colours[c] for c in data["product_category"]])
    for y, (s, n, d) in enumerate(zip(data["trend_score"], data["uk_evidence_count"],
                                      data["latest_uk_evidence_date"])):
        ax.text(s + 1, y, f"{s:.0f}  ({n} UK sources, latest {d})", va="center", fontsize=8)
    ax.set_xlim(0, 135)                       # room for the labels after each bar
    ax.set_xticks(range(0, 101, 20))
    ax.spines[["top", "right"]].set_visible(False)
    ax.set_xlabel("Trend score (0-100)")
    ax.set_title(f"What's trending in the UK ({run.get('run_date', '')})")
    handles = [plt.Rectangle((0, 0), 1, 1, color=colours[c]) for c in cats]
    ax.legend(handles, cats, title="Product category", fontsize=8, title_fontsize=8,
              loc="center left", bbox_to_anchor=(1.01, 0.5))
    fig.tight_layout()
    fig.savefig(out / OUTPUT_CHART, dpi=150, bbox_inches="tight")
    plt.close(fig)

    print("\nWHAT'S TRENDING IN THE UK")
    print("=" * 70)
    for _, r in table.iterrows():
        print(f"  {r['rank']:>2}. {r['trend'][:38]:<39} score {r['trend_score']:5.1f} | "
              f"{r['uk_evidence_count']} UK sources | latest {r['latest_uk_evidence_date']}")
    print(f"\nSaved to {out}: {OUTPUT_CSV}, {OUTPUT_CHART}")


def use_path(text, fallback_name=None):
    """Turn a SETTINGS path into a full path. Relative paths (the default) are read
    from this script's folder, so the project runs wherever it is unzipped, on any
    computer. An absolute path (e.g. r"/Users/me/data") is used exactly as written."""
    p = Path(text)
    return p if p.is_absolute() else SCRIPT_DIR / p


def main():
    """Live: run the agent and save its raw output; replay: load a saved run.
    Then score the trends and write the CSV and chart."""
    out = use_path(OUTPUT_FOLDER)
    if MODE == "live":
        run = live_run()
        out.mkdir(parents=True, exist_ok=True)
        saved = out / f"agent_run_{datetime.now():%Y%m%d_%H%M}.json"
        saved.write_text(json.dumps(run, indent=2), encoding="utf-8")
        print(f"Raw agent output saved to {saved} (use it with MODE = 'replay')")
    elif MODE == "replay":
        path = use_path(REPLAY_FILE, "inputs/trend_agent_saved_run.json")
        if not path.is_file():
            sys.exit(f"Can't find the replay file: {path}")
        run = json.loads(path.read_text(encoding="utf-8"))
    else:
        sys.exit("MODE must be 'live' or 'replay'")
    if not run.get("trends"):
        sys.exit("No trends to score.")
    write_outputs(run, out)


if __name__ == "__main__":
    main()
