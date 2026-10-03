# Trend to Till

**Authors: Mehak Agrawal and Anitha Irene**

**From trend to till, with evidence.**

Trend to Till finds **what** to sell next (an emerging UK food and drink trend), checks **whether** it is
likely to succeed (a product-success model), and finds **where** to launch it first (the London
neighbourhoods that shop most like its early adopters, using our **Basket Twins** method).

EAT_HACK 2026 (Really Good Culture), Track 2: Retail Futures.

---

## 1. The problem

Food and drink brands, especially small challenger brands, face three expensive guesses:

1. **What should we make next?** Trends move fast and the evidence is scattered across retailer
   reports, sales panels, trade press and social media.
2. **Is it likely to succeed?** Many trends fade, and most new products fail.
3. **Where should we launch it?** A small brand cannot launch everywhere. The wrong neighbourhoods
   mean poor early sales and a de-listed product, even when the product was right.

Large companies answer these with research teams and paid data. Challenger brands usually go on instinct.

## 2. Aim and objectives

**Aim:** give a food and drink brand an evidence-based, explainable answer to *what to make next*,
*how likely it is to succeed*, and *where in London to test it first*.

**Objectives:**
1. Find emerging UK food and drink trends from the web and score how strong each one is.
2. Estimate each product's chance of success, in the UK overall and in each UK region, with a
   model that keeps learning as new sales arrive.
3. For products likely to succeed in London, find the neighbourhoods whose shoppers buy most like
   the product's early adopters ("basket twins"), and recommend where and when to launch.
4. Make every recommendation traceable and honest about how much to trust it.

---

## 3. The solution at a glance

```
main.py runs everything in this order:

step1_find_trends.py             What's trending?        -> trend score 0-100
step2a_build_success_inputs.py   (connector) trends -> products described by 10 attributes
step2b_predict_success.py        Will it succeed?        -> success probability, UK + 9 regions (RL)
step2c_select_for_london.py      (connector) keeps products likely to succeed in London
step3_basket_twins.py            Where in London, when?  -> launch wards, comparison wards, month
final_recommendations.csv        one row per product, joining every step
```

**In plain words:** we find what is starting to trend, estimate how likely each product is to
succeed (and in which regions), and for the ones likely to work in London, find the neighbourhoods
that shop like its early adopters and design a small, fair test there.

### How the three parts are connected

The three steps were built separately and "speak different languages", so two small connector
files translate between them:

| Connector | Takes | Gives | How |
|---|---|---|---|
| `step2a_build_success_inputs.py` | `outputs/1_trends/uk_trending_products.csv` | `outputs/2_success/candidate_products.csv` | Describes each trend with the RL model's 10 attributes (0-10). `search_trend_growth` = trend score / 10. Other attributes come from `inputs/product_attribute_scores.csv` if known, otherwise 5 (= average, no effect) |
| `step2c_select_for_london.py` | `success_model/model/predictions.csv` | `outputs/2_success/basket_twins_input.csv` | Keeps products whose **London** success probability is at least 0.5 and whose UK verdict is not "Likely fail". Maps each category to a Tesco category, and adds early-adopter anchor wards from `inputs/london_early_adopter_wards.csv` when available |

`main.py` runs everything in order and writes `final_recommendations.csv`.

---

## 4. Quick start (no API key needed)

```bash
pip install -r requirements.txt
python main.py
```

That is all. It runs the five scripts above in order (about a minute) and prints the final
recommendations. You can also run each script on its own, in the same order.

- **Replay mode (default):** `step1_find_trends.py` re-scores a saved live agent run
  (`inputs/trend_agent_saved_run.json`), so no API key, internet or cost is needed, and results are identical every time.
- **Live mode (optional):** set `MODE = "live"` in `step1_find_trends.py`. Claude then searches the web via
  the Anthropic API (needs an API key with credit in `ANTHROPIC_API_KEY` and web search enabled in the
  Claude Console). Each live run saves a JSON file that can be replayed.
- **Paths:** every path is relative to the project folder, so it runs wherever you unzip it,
  with **no editing**.
- **The RL model** is already trained (`success_model/model/model_state.json`). To retrain it, add the
  training files (`products.csv`, `historical_sales.csv`, `incoming_sales.csv`) to `success_model/data/`
  and delete `model_state.json`; `main.py` then retrains it automatically.

---

## 5. Datasets

| Dataset | What it is | Size | Used in |
|---|---|---|---|
| **Trend evidence** (`inputs/trend_agent_saved_run.json`) | A live Trend Agent run on 3 Oct 2026 (claude-sonnet-5-5): trends with verified evidence from ~39 websites | 8 trends, 59 evidence items (6 food/drink trends used) | Step 1 |
| **Attribute evidence** (`inputs/product_attribute_scores.csv`, `inputs/product_attribute_scores_evidence.csv`) | Two RL attributes measured for our trends: workaround evidence (DIY recipe websites) and innovation level (UK availability) | 6 products | Connector 1 |
| **RL training data** (not included; the trained model is) | **Synthetic sample data** generated by the RL team's `generate_sample_data.py`: 200 products with 10 attribute scores, monthly sales in 9 UK regions (2021-2025), plus 2026 sales for online learning | 17,739 + 3,321 sales rows | Step 2 (training only) |
| **Tesco Grocery 1.0** (`tesco_data/`) | Aiello et al., *Scientific Data* (2020), CC BY 4.0. 2015 purchases of 1.6M Clubcard holders (420M items, 411 stores), aggregated by London ward | 13 files (year + 12 months), 638 wards x 202 columns | Step 3 |
| **Ward lookup** (`tesco_data/ward_lookup.csv`) | Ward code, name, borough and centre coordinates (2015 boundaries) | 638 wards, 100% match | Step 3 |
| **Early-adopter anchors** (`inputs/london_early_adopter_wards.csv`) | London venues already serving each product, mapped to wards (found by web search on 3 Oct 2026); nutrition values are typical estimates | 3 products with anchors | Connector 2 |

---

## 6. Methodology

### Step 1: Trend Agent (`step1_find_trends.py`)
1. **Discover** (live): Claude searches retailer reports, sales panels, trade press, TikTok, Reddit,
   X and news for emerging UK food and drink trends, favouring ones not yet widely sold in UK shops.
2. **Investigate** (live): 4-8 evidence items per trend: source type, rising/falling, UK or not,
   date, stated growth, URL, plus how easy the product is to buy in UK shops.
3. **Verify** (Python): every URL is checked against the pages the search tool actually returned.
4. **Score** (Python): 0-100 from momentum (0.25), breadth (0.15), social buzz (0.10), UK relevance
   (0.15), recency (0.10), credibility (0.10) and UK whitespace (0.15); reduced for fading evidence
   and for fewer than 4 verified items.

### Step 2: Product Success RL (`step2b_predict_success.py`)
A **contextual bandit trained with policy gradient (REINFORCE)**:
- **Context:** a product's 10 attribute scores (0-10, centred so 5 = 0) and the market
  (one of 9 UK regions, or the UK overall).
- **Action:** back the product (predict success) or pass.
- **Reward:** once a product has 6 months of sales in a market, a correct call earns a positive
  reward and a wrong call a negative one, scaled by how decisive the outcome was.
- **Success:** 6-month units at or above that market's historical median (fixed at training time).
- **Policy:** P(back) = sigmoid((global weights + small regional adjustments) · attributes + regional bias).
- **Training:** replays historical outcomes in time order (60 epochs), then calibrates the
  probabilities so "0.8" means about 80% of such products succeeded.
- **Online learning:** `python step2b_predict_success.py update --data success_model/data --model success_model/model --incoming <new sales>` updates the
  weights as each product reaches 6 months of sales.
- **Prediction:** a calibrated success probability for the UK and each region, a verdict
  (Likely success >= 0.65; Uncertain 0.35-0.65; Likely fail < 0.35) and the top 3 driving attributes.
- **The 10 attributes:** search_trend_growth, workaround_evidence, competitor_dissatisfaction,
  price_competitiveness, quality_score, brand_strength, sustainability_score, innovation_level,
  marketing_spend, distribution_reach.

### Step 3: Basket Twins (`step3_basket_twins.py`)
1. **Basket fingerprint per ward:** 25 features (17 food category shares + 8 nutrition values of the
   typical item), as z-scores; the 10% of wards with the lowest Tesco coverage are removed (574 remain).
2. **Anchor:** the wards where the product already appears (from `inputs/london_early_adopter_wards.csv`), averaged.
3. **Basket twins:** cosine similarity between the anchor and every ward (same shopping pattern =
   high similarity). Distance and demographics are not used.
4. **Product checks:** launch wards must buy enough of the product's category; near-misses are
   flagged; each launch ward is paired with a similar comparison ward for a fair trial.
5. **Timing:** the best 3-month season for the category from the monthly files; launch the month before.
6. **Fallback:** with no anchor, wards are ranked by category share and nutrition fit (weaker evidence).
7. **Self-checks:** hidden-feature test, census variables never used for matching, per-product
   permutation test (p-value), and stability of the top-10 list.

---

## 7. Results

### Final recommendations (`final_recommendations.csv`)

| Product | Trend score | UK success prob. | Verdict | London prob. | Launch wards (Basket Twins) |
|---|---|---|---|---|---|
| Hojicha | 56.7 | 0.98 | Likely success | 0.99 | Colville, Golborne, Bayswater, Queen's Gate, Pembridge (twins) |
| Cottage-cheese ice cream | 58.6 | 0.98 | Likely success | 0.99 | Queensbury, Aldersgate, Abbey Road, Regent's Park, Canons (fallback) |
| Sparkling cold brew coffee tonics | 50.5 | 0.97 | Likely success | 0.98 | Abbey (Merton), Limehouse, Clerkenwell, Merton Park, Parsons Green (twins, 1 venue) |
| Tea-flavoured fizzy drinks | 56.0 | 0.80 | Likely success | 0.86 | Addiscombe East, Chadwell Heath, Worcester Park, Cricket Green, Stamford Hill West (fallback) |
| Pandan iced lattes | 72.0 | 0.72 | Likely success | 0.79 | Bayswater, Pembridge, Queen's Gate, Colville, Notting Dale (twins) |
| Pickle snacks | 64.5 | 0.18 | Likely fail | 0.20 | not sent to Basket Twins |

**Key finding:** hojicha and pandan first appear in central and east London (Islington, Spitalfields,
Hackney, West End), but their basket twins are in **west London** (Notting Hill, Bayswater, Kensington).

### How much to trust it

| Check | Result |
|---|---|
| RL holdout test (synthetic data) | Accuracy 0.74 → **0.81**, AUC 0.81 → **0.90** vs starting weights only |
| RL online accuracy (synthetic data) | **0.87** on 370 new outcomes before each update |
| Basket Twins hidden features | Twins a median **49% closer** than random wards (win for 94% of wards) |
| Basket Twins census check | Twins **40% closer** on age, **37% closer** on density (never used for matching) |
| Hojicha / pandan twins | Stability 89% / 97%; hidden-category p = 0.11 / 0.16 (not conclusive) |

**How to read this:** the probabilities rank products for testing; they are not real odds, because the
RL model is trained on synthetic data. Launch wards are "the best places to test first", not
"where it will sell". Recommended next step: a free desk check, then low-cost demand tests in twin vs
comparison wards, then a 1 + 1 mini-trial before the full 5 + 5 trial.

---

## 8. Outputs

| File | Written by | Contents |
|---|---|---|
| `final_recommendations.csv` | `main.py` | **The headline result:** one row per product joining every step |
| `outputs/1_trends/uk_trending_products.csv` + `_chart.png` | Step 1 | Ranked trends with score, category, UK evidence count and date |
| `outputs/2_success/candidate_products.csv` | Step 2a | Our trends described by the 10 attributes |
| `outputs/2_success/success_predictions.csv` | Step 2b | UK probability, verdict, likely regions, top drivers, probability per region |
| `outputs/2_success/london_selection.csv` | Step 2c | Which products went to Basket Twins, and why |
| `outputs/2_success/basket_twins_input.csv` | Step 2c | Basket Twins input (15 columns) |
| `outputs/3_basket_twins/` | Step 3 | `summary.csv`, `all_launch_plans.md`, charts, and a folder per product (launch plan, twins ranking, why they match, timing) |

---

## 9. Limitations

- **The RL model is trained on synthetic, non-food sample data.** Its probabilities for our food
  trends are illustrative until it is retrained on real food sales (same file format).
- **Only 3 of the 10 RL attributes are measured** (search trend growth, workaround evidence, innovation).
  The other 7 are brand decisions (price, quality, marketing, distribution...) and default to average.
- **Rough attribute measurement:** workaround evidence comes from one web search per product; because it
  is the model's strongest weight, it swings results a lot (e.g. pickle snacks).
- **Time gap:** Tesco data is from 2015; trends are from 2026.
- **Coverage:** Tesco Clubcard shoppers only; uneven store coverage across London.
- **Limited comparison factors:** twins are matched on 17 food categories and 8 nutrition values only.
- **Coarse categories:** e.g. pickle snacks mapped to "sauces".
- **Proxy anchors:** early-adopter venues stand in for sales data; coffee tonics rests on one venue.
- **AI reading:** the URL check catches invented links, not misread figures.
- **No sales validation yet** of the launch recommendations.

---

## 10. Future work

1. **User input:** a simple form or web app where a brand enters its product, its plans (price, quality,
   marketing, distribution) and known selling areas, and gets a full launch plan back.
2. **More comparison factors for the twins:** behavioural characteristics (shopping frequency, online vs
   in-store, promotion sensitivity) and economic factors (income, local prices, inflation rate, footfall),
   plus product-specific feature weights.
3. **Sales comparison:** after launch, compare sales in each launch ward against its comparison ward,
   and feed the result back into the RL model (`update`) so it learns from real outcomes.
4. **Real training data for the RL model:** past food and drink launches with real regional sales.
5. **Automated attribute measurement:** workaround evidence from YouTube/Reddit APIs over time,
   innovation from Open Food Facts product counts, benchmarks for price and sustainability.
6. **Current shopping data and beyond London:** recent retailer data; other UK cities.
7. **Stronger trend evidence and backtesting:** official platform data, repeated runs over time, and
   checking whether the pipeline would have spotted past trends early.
8. **Regulation checks:** UK less-healthy-food advertising rules, sugar levy, caffeine, allergens, novel foods.

---

## 11. Files

```
trend-to-till/
  main.py                              runs everything in order
  step1_find_trends.py                 step 1: trend agent
  step2a_build_success_inputs.py       connector: trends -> 10 attributes
  step2b_predict_success.py            step 2: product success RL model
  step2c_select_for_london.py          connector: RL predictions -> Basket Twins input
  step3_basket_twins.py                step 3: Basket Twins
  requirements.txt
  inputs/
    trend_agent_saved_run.json         saved live trend agent run (replayed by step 1)
    product_attribute_scores.csv       measured attribute values (workaround, innovation)
    product_attribute_scores_evidence.csv   where those values came from
    london_early_adopter_wards.csv     early-adopter anchor wards + nutrition estimates
  success_model/
    data/attributes.csv                the 10 attributes and starting weights
    model/model_state.json             the trained RL model
  tesco_data/                          Tesco Grocery 1.0 ward files (year + 12 months) + ward_lookup.csv
```
Every script starts with a plain-English explanation and every function has a short description.

## 12. Troubleshooting

| Message | Fix |
|---|---|
| `No module named ...` | `pip install -r requirements.txt` |
| `Can't find ...` | keep all files and folders together as shown in section 11 |
| `Data folder not found` | keep `tesco_data/` next to the scripts |
| A step fails in `main.py` | run that script on its own to see the full message |
| `No API key` (live mode only) | set `ANTHROPIC_API_KEY`, or use `MODE = "replay"` |

## 13. Credits

Tesco Grocery 1.0: Aiello, L.M., Quercia, D., Schifanella, R. and Del Prete, L. (2020), *Scientific Data*,
CC BY 4.0. AI agent built with the Anthropic API (Claude) and its web search tool.
