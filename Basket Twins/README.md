# Basket Twins

**Find out what to make next, and where in London to launch it first.**

EAT_HACK 2026 (Really Good Culture), Track 2: Retail Futures.

---

## 1. The problem

Food and drink brands, especially small challenger brands, face two expensive guesses:

1. **What should we make next?** Trends move fast, and the evidence is scattered across
   retailer reports, sales panels, trade press and social media. By the time a trend is
   obvious, the shelf is already crowded.
2. **Where should we launch it?** A small brand cannot launch everywhere. Choosing the wrong
   neighbourhoods means poor early sales and a de-listed product, even when the product itself
   was right.

Large companies answer these questions with research teams and paid data. Challenger brands
usually go on instinct.

## 2. Aim and objectives

**Aim:** give a food and drink brand an evidence-based, explainable answer to *what to make
next* and *where in London to test it first*, using an AI agent, open data and transparent maths.

**Objectives:**
1. Find emerging UK food and drink trends from the web and score how strong each one is.
2. Find where in London each trend already appears today (its early adopters).
3. Find the London neighbourhoods whose shoppers buy most like those early adopters
   (their "basket twins"), and recommend where to launch next.
4. Make every recommendation traceable and honest: show the evidence, the reasoning,
   and how much to trust it.
5. Keep the first test small and fair: pair each launch area with a similar comparison area.

## 3. The solution at a glance

Basket Twins is a three-step pipeline:

| Step | File | Question | Method | Main output |
|---|---|---|---|---|
| 1 | `trend_agent.py` | What food and drink is trending in the UK? | AI agent (Claude + web search) collects evidence; Python verifies it and calculates a 0-100 trend score | `outputs_trends/uk_trending_products.csv` + chart |
| 2 | `launch_inputs_agent.py` | Where in London does each product already appear? | AI agent finds cafes and shops; Python maps them to wards | `launch_products.csv` + `launch_products_venues.csv` |
| 3 | `basket_twins.py` | Where should it launch next, and when? | Statistics on 2015 Tesco shopping data (no AI) | `outputs/`: a launch plan per product, `summary.csv`, charts |

**The core idea:** if a product already sells (or appears) in one part of London, the best next
places are the ones where people *shop the same way*. Basket Twins measures that using what
people actually buy, not who they are on paper.

---

## 4. Quick start (5 minutes, no API key needed)

```bash
pip install -r requirements.txt
python trend_agent.py           # step 1: what's trending
python launch_inputs_agent.py   # step 2: where it already appears in London
python basket_twins.py          # step 3: where to launch next
```

- Run the scripts **in this order**.
- Everything runs in **replay mode** by default: no API key, no internet, no cost, and the
  same results every time.
- The file paths in each script point to the author's computer. If they don't exist on yours,
  each script automatically uses its own folder, so **no editing is needed**. You'll see a line
  like `Note: '...' not found on this computer, using ...`; that is expected.

### Replay mode vs live mode (steps 1 and 2 only)

- **Replay (default):** reads a saved run (`agent_run_demo.json`, `launch_inputs_demo.json`)
  and does all verification, scoring and mapping in Python.
- **Live:** set `MODE = "live"` at the top of the file. Claude searches the web through the
  Anthropic API. This needs an Anthropic API key with credit in the `ANTHROPIC_API_KEY`
  environment variable, and web search enabled in the Claude Console. Each live run saves a
  new JSON file that can be replayed later. Live results change over time, because the web changes.

Step 3 never uses the API.

---

## 5. Datasets

### 5.1 Tesco Grocery 1.0 (the main dataset)
- **Source:** Aiello et al., *Scientific Data* (2020), CC BY 4.0,
  https://doi.org/10.6084/m9.figshare.c.4769354.v2
- **What it is:** aggregated 2015 purchases by Tesco Clubcard holders in Greater London.
  According to the authors, it is built from **420 million food items** bought by
  **1.6 million** Clubcard holders at **411 stores**.
- **What we use:** the **ward-level** files: one full-year file and 12 monthly files
  (13 files, about 20 MB). Each has **638 wards** and **202 columns**.
- **Columns we use (per ward):**
  - **17 category shares:** the fraction of food items in fruit & veg, grains, red meat,
    poultry, fish, dairy, eggs, fats & oils, sweets, ready meals, sauces, tea & coffee,
    soft drinks, water, beer, wine, spirits.
  - **8 nutrition values of the typical item bought:** energy, fat, saturated fat, sugar,
    protein, carbohydrates, fibre, salt.
  - **Coverage** (`representativeness_norm`): how well Tesco data represents the ward.
  - **2 census variables, for validation only:** average age and population density.

### 5.2 Ward lookup (`data/ward_lookup.csv`)
- 638 London wards (2015 boundaries) across 33 boroughs: ward code, ward name, borough,
  and the coordinates of each ward's centre.
- Matches **100%** of the Tesco ward codes. Used to turn codes into names, and venue
  locations into wards.

### 5.3 Trend evidence (`agent_run_demo.json`)
- A **live Trend Agent run** on 3 October 2026 (model `claude-sonnet-5-5`), rebuilt from that
  run's output files.
- **8 trends** and **59 evidence items, all verified** against the pages the search tool
  returned, drawn from about 39 websites (trade press, retailer reports, sales and search data,
  market research, social media, news).
- **6 food and drink trends** are used (two beauty trends are filtered out), with 44 evidence
  items, 22 of them UK-specific.

### 5.4 Early-adopter venues (`launch_inputs_demo.json`)
- **Not an agent run.** API credit ran out during the event, so these venues were found by
  Claude's web searches in a chat on 3 October 2026, and saved in the agent's format.
- **12 London venues** from 8 source pages: pandan drinks (5), hojicha (6), coffee tonics (1).
  Sources are listed in `launch_products_venues.csv`.
- Coordinates are approximate. Nutrition values are typical estimates, not label values.
  Pickle snacks and tea fizzy drinks were not researched for venues; cottage-cheese ice cream
  was searched, but no London venue was found.

---

## 6. Methodology

### Step 1: Trend Agent (`trend_agent.py`)
1. **Discover** (live only): Claude searches retailer reports, grocery sales panels, trade press,
   TikTok, Reddit, X and news for emerging UK food and drink trends. It is told to describe
   generic product types (no brand names), and to prioritise trends that are growing abroad
   or online but are **not yet widely available in UK shops**.
2. **Investigate** (live only): for each trend, more searches return 4 to 8 evidence items:
   source type, rising/falling/flat, UK or not, date, stated growth %, short description, URL.
   Claude also judges how easy the product is to buy in UK shops (none / limited / widespread).
3. **Verify** (Python): every evidence URL is checked against the URLs the search tool actually
   returned. Unmatched evidence counts half; evidence without a URL, and duplicates, are dropped.
4. **Score** (Python, not the AI): a 0-100 trend score from seven components:

   | Component | Meaning | Weight |
   |---|---|---|
   | Momentum | median stated growth of rising evidence, on a log scale (+500% = full marks) | 0.25 |
   | Breadth | number of different kinds of source (5 = full marks) | 0.15 |
   | Social buzz | social media evidence (2 items = full marks) | 0.10 |
   | UK relevance | share of evidence about the UK | 0.15 |
   | Recency | evidence loses half its value every 6 months | 0.10 |
   | Credibility | sales data counts most (1.0), predictions least (0.4) | 0.10 |
   | Whitespace | UK availability: none 1.0, limited 0.7, widespread 0.2 | 0.15 |

   **Score = 100 × weighted average × (1 − 0.5 × share of "fading" evidence) × evidence sufficiency**,
   where sufficiency is 70% with 1 verified item, rising to 100% with 4 or more.
5. **Output:** only food and drink trends with at least one UK source are listed.

### Step 2: Launch Inputs Agent (`launch_inputs_agent.py`)
New trend products are not yet sold in UK shops, so nobody can say "it sells well in X".
The next best evidence is **where it already appears**: London cafes, restaurants and
specialist shops serving it.
1. **Find venues** (live only): Claude searches for up to 6 London venues per product, plus the
   closest Tesco category and the nutrition of a similar product.
2. **Verify** (Python): only venues whose web page appeared in the searches are kept.
3. **Map to wards** (Python): each venue is placed in the 2015 ward whose centre is nearest to
   its coordinates, within its borough when known (haversine distance).
4. **Build the row:** up to 4 wards with the most venues become the **anchor**. Products with
   no venue get no anchor and use Basket Twins' nutrition fallback.

### Step 3: Basket Twins (`basket_twins.py`)
1. **Prepare wards:** drop wards with missing values, then remove the **10% with the lowest
   Tesco coverage** (as the dataset's authors recommend): **574 of 638 wards** remain.
2. **Build a basket fingerprint** for each ward: the **25 features** (17 category shares +
   8 nutrition values), each converted to a **z-score** (standard deviations above or below the
   London average), so no feature dominates because of its units.
3. **Anchor:** the anchor wards' fingerprints are averaged into one.
4. **Find basket twins:** **cosine similarity** compares the anchor with every other ward.
   It measures whether two fingerprints point the same way (1 = same shopping pattern,
   0 = unrelated): the *shape* of the basket, not its size. Distance and demographics are not used.
5. **Explain each match:** the 3 features where both are strongly on the same side of the London
   average ("both high in fish"), and the 2 that differ most.
6. **Product checks:**
   - **Launch wards:** the best twins that are no more than 0.5 SD below the anchor on the
     product's own category (e.g. tea & coffee for a latte). 5 are chosen.
   - **Near-misses:** close twins at least 1 SD lower on that category: similar shoppers, but a
     risky place for this product.
   - **Comparison wards:** each launch ward is paired with the most similar remaining twin,
     preferably in another borough. The comparison ward does not get the product, so sales
     can be compared fairly.
7. **Timing:** from the 12 monthly files, the product category's monthly share in the launch
   wards is indexed to its yearly average. The best 3-month stretch is the "season"; the launch
   month is the month before it. A season only counts if it is at least 5% above average.
8. **Fallback (no anchor):** wards are ranked by how much they buy the product's category,
   nudged by nutrition fit using UK front-of-pack thresholds (e.g. a low-sugar product suits
   wards that buy lower-sugar items). This is weaker evidence and is labelled as such.
9. **Self-checks:**
   - **Global hold-out test:** hide one feature, find twins again, and check whether twins are
     still closer on the hidden feature than random wards.
   - **Census check:** are twins closer on average age and population density, which are never
     used for matching?
   - **Per-product hidden-category test:** hide the product's category and compare the twins with
     2,000 random sets of wards (gives a p-value).
   - **Stability:** drop one feature at a time and measure how much of the top-10 list survives.
10. **Neighbourhood types (context only):** k-means groups wards into 4 shopping styles
    (chosen by silhouette score) to give each ward a readable label.

---

## 7. Results

### What's trending (step 1)

| Rank | Trend | Category | Trend score | UK sources |
|---|---|---|---|---|
| 1 | Pandan and coconut flavoured iced lattes | Milk-based drinks | 72.0 | 4 |
| 2 | Pickle-flavoured and sweet-sour pickle snacks | Savoury snacks | 64.5 | 4 |
| 3 | Cottage-cheese-based ice cream and frozen dessert tubs | Ice cream and frozen desserts | 58.6 | 3 |
| 4 | Hojicha | Tea and coffee | 56.7 | 3 |
| 5 | Tea-flavoured fizzy drinks | Soft drinks (with added sugar) | 56.0 | 5 |
| 6 | Sparkling cold brew coffee tonics | Tea and coffee | 50.5 | 3 |

### Where to launch (step 3)

| Product | Mode | Launch wards |
|---|---|---|
| Pandan iced lattes | Twins (5 venues) | Bayswater, Pembridge, Queen's Gate, Colville, Notting Dale |
| Hojicha | Twins (6 venues) | Colville, Golborne, Bayswater, Queen's Gate, Pembridge |
| Coffee tonics | Twins (1 venue) | Abbey (Merton), Limehouse, Clerkenwell, Merton Park, Parsons Green and Walham |
| Pickle snacks | Fallback | Isleworth, Hatch Lane, Monkhams, Endlebury, Larkswood |
| Cottage-cheese ice cream | Fallback | Queensbury, Aldersgate, Abbey Road, Regent's Park, Canons |
| Tea fizzy drinks | Fallback | Addiscombe East, Chadwell Heath, Worcester Park, Cricket Green, Stamford Hill West |

**Key finding:** pandan and hojicha first appear in central and east London (West End, Holborn,
Spitalfields, Hackney), but their basket twins are in **west London** (Bayswater, Notting Hill,
Kensington): a non-obvious, testable recommendation.

### How much to trust it

| Check | Result |
|---|---|
| Hidden features (all wards) | Twins a median **49% closer** than random wards; twins win for **94%** of wards |
| Census variables never used | Twins **40% closer** on average age, **37% closer** on population density |
| Pandan | Stability **97%**; hidden-category p = 0.155 (not conclusive) |
| Hojicha | Stability **89%**; hidden-category p = 0.106 (not conclusive) |
| Coffee tonics | Hidden-category p = **0.0025** (strong), but based on a single venue |
| Neighbourhood types | k = 4, silhouette 0.20 (groups overlap; used as labels only) |

**How to read this:** the method works in general, and the twin lists are stable. For pandan and
hojicha, twins shop like the anchor overall, but we cannot prove they are specifically stronger
tea and coffee buyers. So the recommendation is **"the best places to test first"**, not
"where it will sell".

**Recommended way to act on it:** test in stages, spending more only when the previous step
works: a free desk check that the 2015 pattern still holds today → low-cost demand signals
(geo-targeted ads or sampling) in twin vs comparison wards → a mini-trial in 1 launch ward and
1 comparison ward → the full 5 + 5 trial.

---

## 8. Outputs

**Step 1** (`outputs_trends/`)
- `uk_trending_products.csv`: rank, trend, description, category, location, trend score,
  UK evidence count, latest UK evidence date.
- `uk_trending_products_chart.png`: bar chart of trend scores by category.

**Step 2** (next to the scripts)
- `launch_products.csv`: the Basket Twins input (15 columns, one row per product).
- `launch_products_venues.csv`: every venue, its source URL, whether it was verified, its ward,
  and its distance to that ward's centre.

**Step 3** (`outputs/`)
- `summary.csv`: one row per product: launch wards, comparison wards, near-misses,
  launch month, hidden-category p-value.
- `all_launch_plans.md`: every launch plan in one document.
- `neighbourhood_types.png`, `validation_holdout.csv/png`: context and global self-check.
- One folder per product: `launch_plan.md`, `1_twins_ranking.png`, `2_why_they_match.png`
  (twins mode only), `3_timing.png`, `ranking_top.csv`.

---

## 9. Limitations

- **Time gap:** shopping data is from 2015; trends are from 2026. We assume neighbourhood
  shopping habits change slowly, which may not hold everywhere (e.g. heavily redeveloped areas).
- **Coverage:** Tesco Clubcard shoppers only; store coverage is uneven across London.
- **Limited comparison factors:** twins are matched on only 17 food categories and 8 nutrition
  values. Behaviour, income, prices and wider economic factors are not included.
- **Coarse categories:** e.g. pickle snacks had to be mapped to "sauces"; there is no snacks category.
- **Proxy anchors:** early-adopter venues stand in for sales data. One product (coffee tonics)
  rests on a single venue.
- **Demo research:** the step 2 demo file came from chat-based searches, not an agent run;
  coordinates are approximate and nutrition values are estimates.
- **AI reading:** the URL check catches invented links, not misread figures.
- **No sales validation yet:** recommendations have not been tested against real launch sales.
- **Scores are relative:** a trend score means "worth a closer look"; launch wards mean
  "best place to test first". Neither is a sales forecast.

---

## 10. Future work

1. **User input:** let a brand enter its own product, description and known selling areas
   (through a simple form or web app) and get a launch plan back, instead of editing CSV files.
2. **More comparison factors for the twins:** add behavioural characteristics (e.g. shopping
   frequency, online vs in-store, promotion sensitivity) and economic factors (income, local
   prices, inflation rate, footfall) to the 8 nutrition values and 17 food categories used now.
   Product-specific weights could also prioritise the features that matter most for each
   product (e.g. tea & coffee, dairy and sugar for a latte).
3. **Sales comparison:** once a product launches, compare its sales in each launch ward against
   its paired comparison ward. This turns the trial design into measured evidence, and the results
   can be fed back to improve future recommendations.
4. **Current data:** replace the 2015 Tesco data with recent transaction or loyalty data from a
   retail partner; the method stays the same.
5. **Beyond London:** extend to other UK cities, or other countries, with equivalent area-level data.
6. **Stronger trend evidence:** use official platform data where available (search trends, social
   media), block low-quality sources (sellers, marketplaces), and repeat runs over time to show
   which trends are new, rising or fading.
7. **Backtesting:** check whether the agents would have spotted past trends (e.g. Dubai-style
   chocolate) before they reached UK shelves, and whether twins predicted where they spread.
8. **Regulation checks:** add UK rule checks (less-healthy-food advertising rules, the sugar levy,
   caffeine limits, allergens, novel foods) between steps 1 and 2.
9. **Scale and privacy:** scheduled weekly runs with cost controls; all data stays aggregated by
   area, with no personal data.

---

## 11. Files

```
trend_agent.py              step 1
agent_run_demo.json         saved step 1 run (replay input)
launch_inputs_agent.py      step 2
launch_inputs_demo.json     saved step 2 research (replay input)
basket_twins.py             step 3
data/                       Tesco Grocery 1.0 ward files (year + 12 months) + ward_lookup.csv
requirements.txt            Python packages
```
Every script starts with a plain-English explanation (what it does, inputs, outputs, how to run),
and every function has a short description.

## 12. Troubleshooting

| Message | Fix |
|---|---|
| `No module named ...` | `pip install -r requirements.txt` |
| `Can't find the replay file` | keep the `.json` files in the same folder as the scripts |
| `Data folder not found` | keep the `data/` folder next to `basket_twins.py` |
| `No API key` (live mode only) | set `ANTHROPIC_API_KEY`, or use `MODE = "replay"` |
| `credit balance is too low` (live mode only) | add credit in the Claude Console, or use replay |

## 13. Credits

Tesco Grocery 1.0: Aiello, L.M., Quercia, D., Schifanella, R. and Del Prete, L. (2020),
*Scientific Data*, CC BY 4.0. AI agents built with the Anthropic API (Claude) and its web search tool.
