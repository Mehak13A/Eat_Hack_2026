"""
LAUNCH INPUTS AGENT: builds the input file (launch_products.csv) for Basket Twins
================================================================================
STEP 2 OF 3 in the pipeline:
    trend_agent.py  ->  launch_inputs_agent.py (this file)  ->  basket_twins.py

>>> ORGANISERS / JUDGES: RUN THIS IN REPLAY MODE (the default). <<<
    Replay mode needs NO API key, NO internet and costs nothing. It rebuilds
    launch_products.csv from the saved research in launch_inputs_demo.json.

WHY THIS STEP EXISTS
  Basket Twins needs an "anchor" for each product: London wards where it already sells.
  Trend products are not yet in UK shops, so nobody can say "it sells well in X".
  The next best evidence is where the product ALREADY APPEARS in London: cafes,
  restaurants and specialist shops serving it ("early-adopter venues").

THE TWO MODES (set MODE in SETTINGS)
  "replay"  Reads saved research (launch_inputs_demo.json) and does steps 2-4 below.
            NOTE: launch_inputs_demo.json was NOT produced by this agent. It holds venues
            found by Claude's web searches in a chat on 3 Oct 2026 (our API credit had run
            out). Coordinates are approximate, nutrition values are typical estimates,
            and pickle snacks / tea fizzy drinks were not researched for venues.
            The file itself says this in its "note" field.
  "live"    Calls the Anthropic API so Claude can search the web (step 1). Needs an API
            key with credit in the ANTHROPIC_API_KEY environment variable and web search
            enabled in the Claude Console. Each live run saves launch_inputs_run_<time>.json,
            which can then be replayed.

WHAT HAPPENS FOR EACH PRODUCT
  Step 1  (live only) Claude searches for up to 6 London venues selling/serving it, the
          closest of the 17 Tesco categories, and the nutrition of a similar product.
  Step 2  Python keeps only venues whose web page really appeared in the searches.
  Step 3  Python maps each venue to its 2015 ward: the nearest ward centre (from
          ward_lookup.csv) to the venue's coordinates, within its borough when known.
  Step 4  Python writes the row in exactly Basket Twins' 15-column format. Up to 4 wards
          with the most venues become the anchor ("Ward (Borough); Ward (Borough)").
          No venue found -> no anchor -> Basket Twins uses its nutrition fallback.

HOW TO RUN
  1. pip install pandas anthropic   (anthropic is only needed for live mode)
  2. Check the paths in SETTINGS. If a path does not exist on your computer, the script
     uses the folder this file is in (launch_inputs_demo.json, data/ward_lookup.csv,
     outputs_trends/uk_trending_products.csv, and it writes next to this file).
  3. Run:  python launch_inputs_agent.py   (or right-click > Run in PyCharm)
  4. Then run basket_twins.py, which reads launch_products.csv from the same folder.

INPUTS   launch_inputs_demo.json (replay) or uk_trending_products.csv (live: the product
         list; any CSV with a product_name or trend column works), data/ward_lookup.csv
OUTPUTS  launch_products.csv         the Basket Twins input (one row per product)
         launch_products_venues.csv  every venue: address, source URL, verified or not,
                                     the ward it maps to, and the distance to that ward's centre
"""

import json
import math
import os
import re
import sys
from datetime import date, datetime
from pathlib import Path

import pandas as pd

# =============================================================================
# SETTINGS (full paths on your computer; plain text, keep the r before the quotes)
# If a path does not exist on this computer, the script falls back to its own folder.
# =============================================================================
MODE = "replay"                       # "live" or "replay"
INPUT_PRODUCTS_CSV = r"/Users/mehakagrawal/Desktop/Hackathon/outputs_trends/uk_trending_products.csv"
WARD_LOOKUP_CSV = r"/Users/mehakagrawal/Desktop/Hackathon/data/ward_lookup.csv"
OUTPUT_FOLDER = r"/Users/mehakagrawal/Desktop/Hackathon/outputs_launch"    # where basket_twins.py lives
REPLAY_FILE = r"/Users/mehakagrawal/Desktop/Hackathon/launch_inputs_demo.json"

API_KEY = os.environ.get("ANTHROPIC_API_KEY", "")
MODEL = "claude-sonnet-5-5"
SEARCHES_PER_PRODUCT = 5

DEFAULT_BRAND = "Concept (trend)"     # products have no brand yet
MAX_ANCHOR_WARDS = 4                  # use the wards with the most venues, up to this many
TODAY = date.today()
SCRIPT_DIR = Path(__file__).resolve().parent   # folder this file is in (used for path fallback)

# The 17 categories exactly as basket_twins.py spells them
TESCO_CATEGORIES = ["fruit_veg", "grains", "meat_red", "poultry", "fish", "dairy", "eggs",
                    "fats_oils", "sweets", "readymade", "sauces", "tea_coffee", "soft_drinks",
                    "water", "beer", "wine", "spirits"]
# The 15 columns exactly as basket_twins.py expects them
LAUNCH_COLUMNS = ["brand", "product_name", "tesco_category", "energy_kcal_100g", "sugar_g_100g",
                  "fat_g_100g", "saturates_g_100g", "protein_g_100g", "fibre_g_100g",
                  "salt_g_100g", "sells_well_in", "sells_well_source", "channel", "price_gbp",
                  "notes"]
NUTRITION_KEYS = {"energy_kcal": "energy_kcal_100g", "sugar_g": "sugar_g_100g",
                  "fat_g": "fat_g_100g", "saturates_g": "saturates_g_100g",
                  "protein_g": "protein_g_100g", "fibre_g": "fibre_g_100g", "salt_g": "salt_g_100g"}

# =============================================================================
# PART 1: THE AGENT (Claude + web search)
# =============================================================================
PROMPT = """Today is {today}. A food and drink brand wants to launch this product in London:
"{product}"
Description: {description}

Use web search to find:
1. Up to 6 specific London venues (cafes, restaurants, food halls, specialist shops)
   that CURRENTLY sell or serve this product or a very close version. Prefer recent
   sources. For each, give the street address with postcode, the London borough, and
   approximate latitude/longitude of the address.
2. The closest Tesco food category, one of: {categories}
3. Typical nutrition per 100 g (or 100 ml for drinks) of a similar existing product.

Only use facts from pages you actually found. Do not invent venues, numbers or URLs.
If you find no London venues, return an empty list.
Return JSON inside <json></json> tags:
{{"tesco_category": str, "category_reason": short str,
  "channel": short str (where it is typically sold), "price_gbp": number or null,
  "venues": [{{"name": str, "address": str, "borough": str,
              "latitude": number or null, "longitude": number or null,
              "url": str, "evidence": short str in your own words}}],
  "nutrition": {{"similar_product": str, "url": str or null,
                "energy_kcal": n, "sugar_g": n, "fat_g": n, "saturates_g": n,
                "protein_g": n, "fibre_g": n, "salt_g": n}}}}
Use null for any nutrition value you could not find. Return nothing after the closing tag."""


def run_agent(client, prompt):
    """One agent task; the API runs the searches. Returns (text, urls_found)."""
    tools = [{"type": "web_search_20250305", "name": "web_search",
              "max_uses": SEARCHES_PER_PRODUCT,
              "user_location": {"type": "approximate", "city": "London", "country": "GB",
                                "timezone": "Europe/London"}}]
    messages = [{"role": "user", "content": prompt}]
    texts, urls = [], set()
    for _ in range(5):
        response = client.messages.create(model=MODEL, max_tokens=6000,
                                          messages=messages, tools=tools)
        for block in response.content:
            if block.type == "text":
                texts.append(block.text)
            elif block.type == "web_search_tool_result":
                for r in getattr(block, "content", []) or []:
                    if getattr(r, "url", None):
                        urls.add(r.url)
        if response.stop_reason == "pause_turn":
            messages.append({"role": "assistant", "content": response.content})
            continue
        break
    return "".join(texts), urls


def extract_json(text):
    """Pull the JSON answer out of Claude's reply (between <json></json> tags) and parse it."""
    match = re.search(r"<json>(.*?)</json>", text, re.S)
    raw = (match.group(1) if match else text).strip().strip("`").removeprefix("json").strip()
    return json.loads(raw[raw.find("{"):raw.rfind("}") + 1])


def live_run(products):
    """Live mode: one agent task per product (Step 1). Returns the raw run as a dict,
    which main() saves as JSON for later replay."""
    try:
        import anthropic
    except ImportError:
        sys.exit("Install the SDK first:  pip install anthropic")
    if not API_KEY:
        sys.exit("No API key: set ANTHROPIC_API_KEY in the PyCharm run configuration.")
    client = anthropic.Anthropic(api_key=API_KEY)
    run = {"run_date": TODAY.isoformat(), "model": MODEL, "products": []}
    for i, p in enumerate(products, start=1):
        print(f"Researching {i}/{len(products)}: {p['product_name']}")
        prompt = PROMPT.format(today=TODAY.isoformat(), product=p["product_name"],
                               description=p["description"] or "(none)",
                               categories=", ".join(TESCO_CATEGORIES))
        text, urls = run_agent(client, prompt)
        try:
            found = extract_json(text)
        except Exception as error:
            print(f"  Could not read the agent's answer ({error}); this product gets no anchor.")
            found = {}
        run["products"].append({**p, **found, "search_urls": sorted(urls)})
    return run


# =============================================================================
# PART 2: VENUES -> WARDS (plain Python)
# =============================================================================
def norm(text):
    """Lower-case and keep only letters and digits, for forgiving name matching."""
    return re.sub(r"[^a-z0-9]", "", str(text).lower())


def norm_url(url):
    """Normalise a URL (no https://, no www., no trailing slash) for verification."""
    return re.sub(r"^https?://(www\.)?", "", str(url)).split("#")[0].rstrip("/").lower()


def km_between(lat1, lon1, lat2, lon2):
    """Great-circle distance in km (haversine formula)."""
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dp, dl = p2 - p1, math.radians(lon2 - lon1)
    a = math.sin(dp / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dl / 2) ** 2
    return 6371 * 2 * math.asin(math.sqrt(a))


def load_lookup(path):
    """Read ward_lookup.csv (ward_code, ward_name, borough, longitude, latitude).
    The coordinates are each ward's centre, used to place venues in wards."""
    p = Path(path)
    if not p.is_file():
        sys.exit(f"Can't find the ward lookup: {p}")
    df = pd.read_csv(p)
    need = {"ward_code", "ward_name", "borough", "longitude", "latitude"}
    if not need.issubset(df.columns):
        sys.exit(f"ward_lookup.csv needs columns {sorted(need)}")
    return df.dropna(subset=["latitude", "longitude"])


def to_ward(venue, lookup):
    """Return (ward_label 'Ward (Borough)', method, distance_km) or (None, reason, None)."""
    lat, lon = venue.get("latitude"), venue.get("longitude")
    if not isinstance(lat, (int, float)) or not isinstance(lon, (int, float)):
        return None, "no coordinates", None
    cands = lookup
    borough = norm(venue.get("borough", ""))
    same = lookup[lookup["borough"].map(norm) == borough] if borough else lookup.iloc[0:0]
    if len(same):
        cands, method = same, "nearest ward centre in its borough"
    else:
        method = "nearest ward centre (borough not matched)"
    dist = cands.apply(lambda r: km_between(lat, lon, r["latitude"], r["longitude"]), axis=1)
    best = cands.loc[dist.idxmin()]
    if dist.min() > 3:
        return None, f"too far from any ward centre ({dist.min():.1f} km); outside London?", None
    return f"{best['ward_name']} ({best['borough']})", method, round(float(dist.min()), 2)


def build_rows(run, lookup):
    """Steps 2-4 for every product: verify venues, map them to wards, pick the anchor
    wards, copy category and nutrition, and build the Basket Twins row.
    Also returns a table of every venue for checking."""
    rows, details = [], []
    for p in run["products"]:
        name = p["product_name"]
        found_urls = {norm_url(u) for u in p.get("search_urls", [])}
        ward_counts, venue_names = {}, []
        for v in p.get("venues", []) or []:
            verified = bool(v.get("url")) and norm_url(v["url"]) in found_urls
            ward, method, km = to_ward(v, lookup) if verified else (None, "URL not in search results", None)
            details.append({"product_name": name, "venue": v.get("name", ""),
                            "address": v.get("address", ""), "borough": v.get("borough", ""),
                            "url": v.get("url", ""), "verified": verified,
                            "ward": ward or "", "ward_method": method, "km_to_ward_centre": km,
                            "evidence": v.get("evidence", "")})
            if ward:
                ward_counts[ward] = ward_counts.get(ward, 0) + 1
                venue_names.append(v.get("name", ""))
        anchors = [w for w, _ in sorted(ward_counts.items(), key=lambda kv: -kv[1])][:MAX_ANCHOR_WARDS]

        cat = str(p.get("tesco_category", "")).strip()
        if cat not in TESCO_CATEGORIES:
            print(f"  {name}: category '{cat}' is not a Tesco category; please fix it in the CSV.")
        nut = p.get("nutrition", {}) or {}
        row = {c: "" for c in LAUNCH_COLUMNS}
        row.update({
            "brand": p.get("brand") or DEFAULT_BRAND,
            "product_name": name,
            "tesco_category": cat,
            "sells_well_in": "; ".join(anchors),
            "sells_well_source": (f"{len(venue_names)} London venue(s) found by web search "
                                  f"({run.get('run_date', '')}): " + ", ".join(venue_names[:6]))
                                 if anchors else "",
            "channel": p.get("channel", "") or "",
            "price_gbp": p.get("price_gbp") if p.get("price_gbp") is not None else "",
            "notes": "; ".join(x for x in [
                "anchor = early-adopter venues (ward = nearest ward centre)" if anchors
                else "no London venue found: Basket Twins will use its nutrition fallback",
                f"nutrition from: {nut.get('similar_product')}" if nut.get("similar_product") else "",
                p.get("notes_extra", "")] if x),
        })
        for key, col in NUTRITION_KEYS.items():
            if isinstance(nut.get(key), (int, float)):
                row[col] = nut[key]
        rows.append(row)
    return pd.DataFrame(rows, columns=LAUNCH_COLUMNS), pd.DataFrame(details)


# =============================================================================
# MAIN
# =============================================================================
def load_products(path):
    """Read the product list for live mode. Needs a 'product_name' (or 'trend') column;
    a 'description' or 'product_trend_description' column is used as extra context."""
    p = Path(path)
    if not p.is_file():
        sys.exit(f"Can't find the products file: {p}")
    df = pd.read_csv(p, dtype=str, keep_default_na=False)
    name_col = "product_name" if "product_name" in df.columns else "trend" if "trend" in df.columns else None
    if not name_col:
        sys.exit("The products file needs a 'product_name' (or 'trend') column.")
    desc_col = next((c for c in ["description", "product_trend_description"] if c in df.columns), None)
    return [{"product_name": r[name_col].strip(),
             "description": r[desc_col].strip() if desc_col else "",
             "brand": r.get("brand", "").strip()} for _, r in df.iterrows() if r[name_col].strip()]


def use_path(text, fallback_name):
    """Use the path from SETTINGS if it (or its parent folder) exists on this computer;
    otherwise use `fallback_name` next to this script, so it runs anywhere unedited."""
    p = Path(text)
    if p.exists() or p.parent.exists():
        return p
    fallback = SCRIPT_DIR / fallback_name
    print(f"Note: '{text}' not found on this computer, using {fallback}")
    return fallback


def main():
    """Live: research each product with the agent and save the raw run; replay: load
    the saved research. Then map venues to wards and write launch_products.csv."""
    out = use_path(OUTPUT_FOLDER, ".")
    out.mkdir(parents=True, exist_ok=True)
    lookup = load_lookup(use_path(WARD_LOOKUP_CSV, "data/ward_lookup.csv"))

    if MODE == "live":
        products = load_products(use_path(INPUT_PRODUCTS_CSV, "outputs_trends/uk_trending_products.csv"))
        print(f"{len(products)} products to research.\n")
        run = live_run(products)
        saved = out / f"launch_inputs_run_{datetime.now():%Y%m%d_%H%M}.json"
        saved.write_text(json.dumps(run, indent=2), encoding="utf-8")
        print(f"\nRaw agent output saved to {saved} (use it with MODE = 'replay')")
    elif MODE == "replay":
        path = use_path(REPLAY_FILE, "launch_inputs_demo.json")
        if not path.is_file():
            sys.exit(f"Can't find the replay file: {path}")
        run = json.loads(path.read_text(encoding="utf-8"))
        if run.get("note"):
            print("NOTE: " + run["note"] + "\n")
    else:
        sys.exit("MODE must be 'live' or 'replay'")

    launch, details = build_rows(run, lookup)
    launch.to_csv(out / "launch_products.csv", index=False)
    details.to_csv(out / "launch_products_venues.csv", index=False)

    print("LAUNCH PRODUCTS FOR BASKET TWINS")
    print("=" * 70)
    for _, r in launch.iterrows():
        mode = f"anchor: {r['sells_well_in']}" if r["sells_well_in"] else "no anchor -> nutrition fallback"
        print(f"  {r['product_name'][:38]:<39} [{r['tesco_category']}]\n      {mode}")
    print(f"\nSaved to {out}:\n  launch_products.csv         (the Basket Twins input)"
          f"\n  launch_products_venues.csv  (every venue found and the ward it maps to)"
          f"\nNext: run basket_twins.py.")


if __name__ == "__main__":
    main()