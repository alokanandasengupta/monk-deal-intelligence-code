# Monk Deal Intelligence

A data investigation for an influencer talent agency: which creators and campaign choices worked, which did not, and how sure the data is.

**All data in this repo is simulated.** The 7,500 deals, 700 creators, brands and briefs are made up (the generator is in `pipeline/01_generate_data.py`). A few real-world effects were written in on purpose so the analysis could be graded on whether it finds them and correctly ignores the ones that were not written in. The method is real. The effect sizes need re-measuring on real deals.

## Run it

```bash
pip install -r requirements.txt
python app.py            # then open http://127.0.0.1:5050
```

The web page has four tabs:

| Tab | What it does |
|---|---|
| Start here | Who built it, the problem, a peek at the data and a column guide |
| Playground | Pick a brand brief and get an allocation (Plan a campaign), or paste a script and see how it is read (Check a script) |
| Proof | Six findings with charts (Findings) and the scorecard with believable ranges (Evidence) |
| How it works | The method behind each finding, with the charts |

The earlier seven-tab layout is at `/classic`. A denser view is at `/advanced`.

## Repo layout

```
app.py              Flask server (routes: / /classic /advanced /api/*)
static/             The web page (index.html), earlier layouts, chart images
src/                Code the app uses
  pitch_model.py      Quantile model (p10/p50/p90) with creator-grouped validation and conformal calibration
  pitch_builder.py    Turns a brief into a budget split and a creator list
  ai_layer.py         Script reader: reads opening style and ask position (rules offline, optional LLM)
pipeline/           Numbered scripts that rebuild everything from scratch
data/               monk_deals_simulated.csv (generated, fixed seed)
outputs/            Generated results: cleaned deals, trained model, tests, charts
docs/               Decision memo, model card, example pitch, Excel workbook of the data
scripts/            demo.py (terminal demo), verify_data.py (sanity checks on the data)
```

## Rebuild from scratch

Run from anywhere; each script sets its own paths.

```bash
python pipeline/01_generate_data.py        # simulated deals -> data/
python pipeline/02_analysis.py             # benchmarks, charts, creator scorecard -> outputs/
python pipeline/03_train_model.py          # prediction model and its validation
python pipeline/04_robustness.py           # creator-level resampling for believable ranges
python pipeline/05_whatif_check.py         # change one factor, hold the rest equal
python pipeline/06_eval_tagger.py          # accuracy of the script reader
python pipeline/07_build_evidence.py       # evidence.json read by the web page
python pipeline/08_model_charts_and_card.py
python pipeline/09_build_memo.py
python pipeline/10_build_workbook.py
```

The data regenerates byte for byte, and the model metrics reproduce.

## Method in brief

- **Measure:** return per rupee (revenue over fee, a ratio of sums), engagement rate, click rate, by creator size, topic, opening style, ask position.
- **Test:** resample creators (not deals) 400 times to get a believable range for each gap. Creator-level rank tests for factors that may not matter, after adjusting for creator size and format.
- **Model:** gradient-boosted quantile regression, scored on creators the model has never seen. Ranges are calibrated so the 80% range holds about 80% of the time on held-out creators.
- **Plan:** budget split across creator sizes with caps, then a creator list filtered for topic match, track record and overbooking.
- **Script reader:** the only AI step. Offline rules by default. An optional Claude-based reader runs when `ANTHROPIC_API_KEY` is set (not tested here).

## Limits

- Simulated data. The mid-video ask effect was written in strongly and will likely be smaller on real deals.
- Posting-time effect: evening posts return more, with no engagement change. This was first missed and corrected.
- The script reader is accurate on templated scripts (91% opening, 100% ask position) and weak on messy ones (25% opening, 88% ask position). It asks the user when unsure.
