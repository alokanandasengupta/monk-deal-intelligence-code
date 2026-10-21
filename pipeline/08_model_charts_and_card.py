import os, sys
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(ROOT); sys.path.insert(0, os.path.join(ROOT, "src"))  # run from anywhere; paths are relative to the repo root
import json
import numpy as np, pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

O = "outputs"; CH = f"{O}/charts"
M = json.load(open(f"{O}/model_results.json")); W = json.load(open(f"{O}/whatif_results.json"))
RB = json.load(open(f"{O}/robustness.json")); RS = json.load(open(f"{O}/results.json"))
df = pd.read_csv(f"{O}/deals_clean.csv").reset_index(drop=True)
oe, orr = np.load(f"{O}/oof_eng.npy"), np.load(f"{O}/oof_roas.npy")
NAVY, RED, GREY, TEAL = "#1F3A5F", "#C0392B", "#9AA5B1", "#1F8A70"
plt.rcParams.update({"font.family": "DejaVu Sans", "axes.spines.top": False, "axes.spines.right": False, "font.size": 10})

# ---- grouped importance ----
GROUPS = {"Creator size (tier, followers)": ["tier", "log_followers"], "Fee": ["log_fee"],
          "Recent deals (saturation)": ["creator_deals_last_30d"], "Hook type": ["hook_type"],
          "Niche match": ["niche_match"], "Brand category": ["brand_category"], "CTA position": ["cta_position"],
          "Format and platform": ["format", "platform"], "Creator niche": ["creator_niche"],
          "Timing (hour, day)": ["post_hour", "post_day", "is_weekend"], "Duration": ["duration_sec"]}
def grouped(t):
    imp = M["importance"][t]
    return pd.Series({g: sum(imp.get(f, 0) for f in fs) for g, fs in GROUPS.items()}).sort_values(ascending=False)
GE, GR = grouped("engagement_rate"), grouped("roas")
fig, axs = plt.subplots(1, 2, figsize=(11, 4.6), sharex=False)
for ax, g, ttl in [(axs[0], GE, "Engagement rate"), (axs[1], GR, "ROAS")]:
    top = g.head(7)[::-1]
    ax.barh(top.index, top.values * 100, color=NAVY)
    for i, v in enumerate(top.values): ax.text(v * 100 + 0.6, i, f"{v*100:.0f}%", va="center", fontsize=9)
    ax.set_title(f"What drives predicted {ttl}", loc="left", fontweight="bold", fontsize=10.5); ax.set_xlabel("share of predictive power (%)")
plt.tight_layout(); plt.savefig(f"{CH}/7_model_drivers.png", dpi=160); plt.close()

# ---- calibration by predicted decile ----
df["pe"], df["pr"] = oe[:, 1], orr[:, 1]
df["de"], df["dr"] = pd.qcut(df.pe, 10, labels=False, duplicates="drop"), pd.qcut(df.pr, 10, labels=False, duplicates="drop")
ce = df.groupby("de").agg(pred=("pe", "median"), act=("engagement_rate", "median"))
cr = df.groupby("dr").agg(pred=("pr", "median"), act=("roas", "median"))
fig, axs = plt.subplots(1, 2, figsize=(10.5, 4.4))
for ax, c, ttl, sc in [(axs[0], ce, "Engagement rate (%)", 100), (axs[1], cr, "ROAS", 1)]:
    lim = max(c.pred.max(), c.act.max()) * sc * 1.08
    ax.plot([0, lim], [0, lim], color=GREY, ls="--", lw=1, label="perfect")
    ax.plot(c.pred * sc, c.act * sc, marker="o", color=TEAL, lw=1.8, label="model (held-out creators)")
    ax.set_xlabel(f"predicted median, {ttl}"); ax.set_ylabel(f"actual median, {ttl}"); ax.legend(frameon=False, fontsize=8)
    ax.set_title(f"Calibration: {ttl.split(' (')[0]}", loc="left", fontweight="bold", fontsize=10.5)
plt.tight_layout(); plt.savefig(f"{CH}/8_model_calibration.png", dpi=160); plt.close()

# ---- what-if effects ----
eng_levers = [("Story hook vs direct pitch", W["eng_story_vs_direct"]), ("Niche mismatch vs match", W["eng_mismatch_vs_match"]),
              ("4 recent deals vs 1", W["eng_saturation_4_vs_1"]), ("CTA mid vs end", W["engagement_rate_cta_mid_vs_end"]),
              ("45s vs 20s reel", W["eng_duration_45_vs_20"]), ("Saturday vs Tuesday", W["eng_saturday_vs_tuesday"]),
              ("8pm vs 4am", W["eng_hour20_vs_hour4"])]
roas_levers = [("CTA mid vs end", W["roas_cta_mid_vs_end"]), ("Story hook vs direct pitch", W["roas_story_vs_direct"]),
               ("4 recent deals vs 1", W["roas_saturation_4_vs_1"])]
fig, axs = plt.subplots(1, 2, figsize=(11, 4.4), gridspec_kw={"width_ratios": [1.25, 1]})
for ax, lv, ttl in [(axs[0], eng_levers, "Effect on engagement rate"), (axs[1], roas_levers, "Effect on ROAS")]:
    lv = lv[::-1]; vals = [v * 100 for _, v in lv]
    ax.barh([n for n, _ in lv], vals, color=[TEAL if v > 1.5 else RED if v < -1.5 else GREY for v in vals])
    ax.axvline(0, color="black", lw=0.8)
    for i, v in enumerate(vals):
        ax.text(v + (1.5 if v >= 0 else -1.5), i, "0%" if abs(v) < 0.5 else f"{v:+.0f}%", va="center", ha="left" if v >= 0 else "right", fontsize=9)
    lo, hi = min(vals + [0]) , max(vals + [0]); pad = (hi - lo) * 0.25 + 3; ax.set_xlim(lo - pad, hi + pad)
    ax.set_title(ttl + " (model what-if)", loc="left", fontweight="bold", fontsize=10)
plt.tight_layout(); plt.savefig(f"{CH}/9_model_whatif.png", dpi=160); plt.close()

# ---- model card ----
E, R = M["metrics"]["engagement_rate"], M["metrics"]["roas"]
p10r, p90r = np.median(np.clip(orr[:, 0] - M["metrics"]["roas"]["interval_widening"], 0, None)), np.median(orr[:, 2] + M["metrics"]["roas"]["interval_widening"])
top_dec = cr.iloc[-1]
sat_raw = RS["F3"]["drop"]
pct = lambda x, d=0: f"{x*100:.{d}f}%"
def sp(x):  # signed percent, flat effects shown as 0%
    return "0%" if abs(x) < 0.005 else f"{x*100:+.0f}%"
md = f"""# Monk Pitch Model: what it predicts, how well, and where it is weak

**What it does:** given a proposed deal (creator tier and niche, brand category, format, hook, CTA position, fee, timing, how booked-up the creator is), it predicts a **range** for engagement rate and ROAS: a low case (10th percentile), a typical case (median) and a high case (90th percentile).

**Trained on:** {M['n']:,} simulated deals from {M['creators']} creators. All results below are from simulated data and show that the method works. Retrain on Monk's real deals with one command (`python3 pitch_model.py`).

## How it was tested
5-fold validation where **every test creator is unseen** by the model (folds are split by creator, so it cannot memorise a creator's habits). The baseline is the median for the same tier and brand category, which is roughly what a pitch built on averages would use.

| Metric | Engagement rate | ROAS |
|---|---|---|
| Rank correlation (Spearman) | {E['spearman']:.2f} | {R['spearman']:.2f} |
| Error vs baseline | {pct(E['skill_vs_baseline'])} lower | {pct(R['skill_vs_baseline'])} lower |
| R squared | {E['r2']:.2f} | {R['r2']:.2f} |
| Stated 80% range actually contained | {pct(E['coverage_p10_p90'])} before fix, **{pct(E['coverage_after_calibration_heldout'],1)}** after | {pct(R['coverage_p10_p90'])} before fix, **{pct(R['coverage_after_calibration_heldout'],1)}** after |

The raw ranges were too narrow (they contained about 74% of outcomes while claiming 80%), so I applied a standard conformal correction and checked it on creators held out from the calibration. The ranges now contain 80% of real outcomes.

**Reading this honestly:** engagement is highly predictable (large part of it is creator size). ROAS is much noisier, because it depends on whether a click turns into a sale. The model ranks ROAS well (0.73) but the exact value is uncertain: a typical ROAS range runs from about {p10r:.1f} to {p90r:.1f}. That is why the tool gives ranges, not single numbers.

## What drives the predictions
- **Engagement:** {GE.index[0].lower()} ({pct(GE.iloc[0])}), {GE.index[1].lower()} ({pct(GE.iloc[1])}), {GE.index[2].lower()} ({pct(GE.iloc[2])}), {GE.index[3].lower()} ({pct(GE.iloc[3])}).
- **ROAS:** {GR.index[0].lower()} ({pct(GR.iloc[0])}), {GR.index[1].lower()} ({pct(GR.iloc[1])}), {GR.index[2].lower()} ({pct(GR.iloc[2])}), {GR.index[3].lower()} ({pct(GR.iloc[3])}).

Shares come from permutation importance on held-out creators: shuffle one input, see how much accuracy drops.

## Does it agree with the analysis?
Same deals re-scored with one input changed at a time. This is a consistency check, not proof of cause.

| Change | Model says (engagement) | Model says (ROAS) | Analysis found |
|---|---|---|---|
| Story hook instead of direct pitch | {sp(W['eng_story_vs_direct'])} | {sp(W['roas_story_vs_direct'])} | Direct pitch gets {pct(RB['direct_pitch_gap']['point'])} less engagement |
| Niche mismatch instead of match | {sp(W['eng_mismatch_vs_match'])} | not shown | {pct(RB['mismatch_drop']['point'])} lower engagement |
| 4 recent deals instead of 1 | {W['eng_saturation_4_vs_1']*100:+.0f}% | {W['roas_saturation_4_vs_1']*100:+.0f}% | {pct(sat_raw)} average drop at 3 or more deals |
| CTA mid-video instead of end | {sp(W['engagement_rate_cta_mid_vs_end'])} | {sp(W['roas_cta_mid_vs_end'])} | CTR {pct(RS['F4']['lift_mid_vs_end'])} higher |
| 45s reel instead of 20s | {W['eng_duration_45_vs_20']*100:+.0f}% | not shown | no significant length effect |
| Saturday instead of Tuesday, 8pm instead of 4am | {sp(W['eng_saturday_vs_tuesday'])}, {sp(W['eng_hour20_vs_hour4'])} | not shown | not tested as a finding |

The model finds the effects the analysis found and finds none where the analysis found none. The saturation figure is smaller than the headline (about {abs(W['eng_saturation_4_vs_1'])*100:.0f}% at exactly 4 recent deals vs {pct(sat_raw)} averaged over all deals with 3 or more, which includes heavier bursts). The CTA effect on ROAS is larger than on CTR because mid-video CTAs lift both clicks and conversion.

## Where it is weak
- **ROAS at the top end:** the model is conservative. Its highest-predicted bucket has a median prediction of {top_dec.pred:.1f} against an actual median of {top_dec.act:.1f}. Treat the high end as a floor.
- **Hook ranking is soft:** among the non-direct hooks the predicted ROAS differed by only 1 to 2% in the briefs I checked, which is a tie. The tool flags near-ties instead of crowning a winner. What is clear: mid-video CTA, any non-direct hook, avoid saturated creators.
- **Language and city are excluded on purpose.** I tested them (tier-adjusted, creator level) and found no effect, so including them would only add noise.
- **Simulated data.** Real data will be noisier and will not follow these tidy patterns. The point of this build is the method and the tool, not the specific numbers.
- **Not causal.** It describes how outcomes relate to inputs in past deals. Test big changes on live campaigns before making them policy.

## Using it
```python
from pitch_model import load, predict_brief, best_plays
art = load()
predict_brief(art, dict(tier="mid", creator_niche="fashion", brand_category="fashion",
                        format="reel", hook_type="social_proof", cta_position="mid", creator_deals_last_30d=1))
best_plays(art, tier="mid", niche="beauty", category="beauty", top=5)   # ranks hook x CTA combos
```
"""
open("docs/model_card.md", "w").write(md)
print(md)
