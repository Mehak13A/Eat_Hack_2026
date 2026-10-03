# Trend to Launch: what to make next, and where in London to launch it

EAT_HACK 2026, Track 2: Retail Futures.

A three-step pipeline:

1. **Trend Agent** (`trend_agent.py`): an AI agent (Claude + web search) finds emerging
   food and drink trends and gives each a transparent 0-100 trend score.
2. **Launch Inputs Agent** (`launch_inputs_agent.py`): for each trend, an AI agent finds
   the London venues where it already appears and maps them to wards (the "anchor").
3. **Basket Twins** (`basket_twins.py`): finds the wards whose 2015 Tesco shopping
   baskets look most like the anchor, then suggests launch wards, near-misses,
   a launch month and a matched trial. No AI; plain statistics.

## Folder layout
```
Hackathon/
  trend_agent.py              step 1
  agent_run_demo.json         saved step 1 run (replay)
  launch_inputs_agent.py      step 2
  launch_inputs_demo.json     saved step 2 research (replay)
  basket_twins.py             step 3
  data/                       Tesco Grocery 1.0 ward files (year + 12 months) and ward_lookup.csv
```

## How to run (replay mode: no API, no cost)
1. `pip install pandas numpy scikit-learn matplotlib anthropic`
2. Run `trend_agent.py` → `outputs_trends/uk_trending_products.csv` and chart
3. Run `launch_inputs_agent.py` → `launch_products.csv` (+ `launch_products_venues.csv`)
4. Run `basket_twins.py` → `outputs/` (launch plan per product, charts, `summary.csv`)

Paths are set in the SETTINGS section at the top of steps 1 and 2.

## Live mode
Set `MODE = "live"` in `trend_agent.py` and/or `launch_inputs_agent.py`, with an Anthropic
API key in the `ANTHROPIC_API_KEY` environment variable (web search enabled in the Claude
Console). Each live run saves a JSON file that can be replayed later.

## Trend score (step 1)
Weighted mix of momentum (stated growth, log scale), breadth (source types), social buzz,
UK relevance, recency (half-life 6 months), source credibility and UK whitespace (how hard
the product is to buy in UK shops); reduced when evidence shows the trend fading and when
fewer than 4 evidence items are verified. Every evidence URL is checked against the URLs the
search tool actually returned.

## Data and provenance
- Tesco Grocery 1.0 (Aiello et al., Scientific Data, 2020), CC BY 4.0,
  https://doi.org/10.6084/m9.figshare.c.4769354.v2
- `agent_run_demo.json`: a live Trend Agent run on 3 Oct 2026 (claude-sonnet-5-5, 8 trends,
  59 verified evidence items), rebuilt from that run's output files. Only the 6 food and
  drink trends are used.
- `launch_inputs_demo.json`: NOT an agent run. Venues were found by Claude web searches in a
  chat on 3 Oct 2026 (sources listed in `launch_products_venues.csv`). Coordinates are
  approximate; nutrition values are typical estimates, not label values; pickle snacks and
  tea fizzy drinks were not researched for venues (they use Basket Twins' nutrition fallback).

## Limitations
- Shopping data is from 2015; trends are from 2026. We assume neighbourhood shopping habits
  change slowly.
- Tesco Clubcard shoppers only; store coverage is uneven across London.
- The 17 Tesco categories are coarse (e.g. pickle snacks mapped to "sauces").
- Early-adopter venues are a proxy for "sells well in", not sales data.
- AI agents read sources; the URL check catches invented links, not misread figures.
- A trend score means "worth a closer look", not "will sell".

## Disclosure
[Fill in: which parts were prepared before EAT_HACK (e.g. Basket Twins core, agent code)
and which were built or run during the event (live agent runs, launch-input research,
integration of the three steps, analysis).]
