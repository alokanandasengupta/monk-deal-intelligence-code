import os, sys
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(ROOT); sys.path.insert(0, os.path.join(ROOT, "src"))  # run from anywhere; paths are relative to the repo root
import json
import numpy as np, pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

O = "outputs"
R = json.load(open(f"{O}/results.json"))
df = pd.read_csv(f"{O}/deals_clean.csv")
df["big"] = df.tier.isin(["macro", "mega"])

def cr(x):  # INR to crore string
    return f"Rs {abs(x)/1e7:.2f} cr"
def pct(x, d=0):
    return f"{x*100:.{d}f}%"

# ---- chart 5 redo: language ROAS, small tiers vs macro/mega (shows it is tier mix) ----
def roas(x): return x.revenue_inr.sum() / x.deal_value_inr.sum()
lang = pd.DataFrame({
    "small": df[~df.big].groupby("creator_language").apply(roas, include_groups=False),
    "big": df[df.big].groupby("creator_language").apply(roas, include_groups=False)}).sort_values("small")
plt.rcParams.update({"font.family": "DejaVu Sans", "axes.spines.top": False, "axes.spines.right": False, "font.size": 10})
fig, ax = plt.subplots(figsize=(8, 4.8)); y = np.arange(len(lang))
ax.barh(y + 0.2, lang.small, 0.4, color="#1F8A70", label="nano, micro, mid")
ax.barh(y - 0.2, lang.big, 0.4, color="#C0392B", label="macro, mega")
ax.axvline(1, color="black", lw=0.8, ls="--"); ax.set_yticks(y); ax.set_yticklabels(lang.index)
ax.set_xlabel("ROAS (1.0 = break-even)"); ax.legend(frameon=False)
ax.set_title("Language does not drive ROAS, creator tier does", loc="left", fontweight="bold")
plt.tight_layout(); plt.savefig(f"{O}/charts/5_roas_by_language.png", dpi=160); plt.close()

F1, F2, F3, F4, S, fit = R["F1"], R["F2"], R["F3"], R["F4"], R["scorecard"], R["fit"]
T = R["tier"]
tot_spend = df.deal_value_inr.sum()
losers = pd.read_csv(f"{O}/quietly_losing_money.csv")
H = pd.DataFrame(F2["table"])
sig_hooks = int((H.p < 0.05).sum())
_hp = pd.read_csv(f"{O}/hook_by_category_median_engagement.csv", index_col=0)
n_last = int((_hp.rank(axis=1, ascending=False)["direct_pitch"] == _hp.shape[1]).sum())
_cs = pd.read_csv(f"{O}/creator_scorecard.csv")
_w = _cs[(_cs.roas > 1.5) & (_cs.deals >= 6)]
n_w, n_w_small = len(_w), int(_w.tier.isin(["nano", "micro", "mid"]).sum())
from scipy import stats as _st
ROB = json.load(open(f"{O}/robustness.json"))
def ci(k, as_pct=True, d=0):
    v = ROB[k]; f = (lambda x: f"{x*100:.{d}f}%") if as_pct else (lambda x: f"{x:.2f}")
    return f"{f(v['lo'])} to {f(v['hi'])}"
_hook_names = {"social_proof": "social proof", "direct_pitch": "direct pitch"}
best_list = ", ".join(f"{r.category} {_hook_names.get(r.best_hook, r.best_hook)}" for r in H.itertuples())
n_min, n_max = int(H.n_best.min()), int(H.n_best.max())
n_bot2 = int((_hp.rank(axis=1, ascending=False)["direct_pitch"] >= _hp.shape[1] - 1).sum())
direct_less = 1 - F2["direct_med"] / F2["other_med"]
# creator-level, tier-adjusted tests for language / city (creator is the unit, deals within a creator are correlated)
_d = df.copy(); _d["rk"] = _d.groupby("tier").roas.rank(pct=True)
_cr = _d.groupby(["creator_id", "creator_language", "creator_city", "city_type"]).agg(rk=("rk", "mean"), deals=("deal_id", "count")).reset_index()
_cr = _cr[_cr.deals >= 4]
def _kw(by):
    gs = [g.rk.values for _, g in _cr.groupby(by) if len(g) >= 8]
    return _st.kruskal(*gs)[1] if len(gs) > 2 else _st.mannwhitneyu(*gs)[1]
p_lang, p_city, p_ctype = _kw("creator_language"), _kw("creator_city"), _kw("city_type")
_raw = {k: v["roas"] for k, v in R["lang"].items()}
lang_hi, lang_lo = max(_raw, key=_raw.get), min(_raw, key=_raw.get)
_F5 = pd.DataFrame(R["F5"]); sig_dur = _F5[_F5.kruskal_p < 0.05].category.tolist()
dur_line = (f"only {' and '.join(sig_dur)} show a significant length effect. For the other categories the data does not support a length rule, so I would not recommend one."
            if sig_dur else "no category shows a statistically significant effect of video length on engagement, so I would not recommend a length rule.")
F5 = pd.DataFrame(R["F5"]); sig_dur = F5[F5.kruskal_p < 0.05].category.tolist()
small = df[~df.big]
lang_small_min, lang_small_max = lang.small.min(), lang.small.max()

md = f"""# Monk Deal Intelligence: what the data says and what I would do

**Dataset:** {R['rows_clean']:,} simulated influencer deals, {df.creator_id.nunique()} creators, Oct 2024 to Sep 2026, total spend {cr(tot_spend)}. All numbers below are from simulated data and show the method. The same code runs on Monk's real deal log.

## Headline
Across the book, every rupee spent returned Rs {R['overall_roas']:.2f} in attributed revenue, and **{pct(R['pct_deals_losing'])} of deals lose money**. The losses are concentrated, which means they are fixable.

## Five decisions

**1. Move budget out of macro and mega, into mid tier.**
Macro and mega take {pct(F1['bigshare'])} of spend and return {F1['big_roas']:.2f} ROAS; {pct(F1['big_lose'])} of those deals lose money. Mid tier returns {F1['mid_roas']:.2f} ROAS with {pct(F1['mid_lose'])} losing. The ROAS gap is {ROB['tier_roas_gap']['point']:.2f} (95% CI {ci('tier_roas_gap', False)}).
Illustrative impact: shifting 20% of macro and mega spend ({cr(F1['whatif_moved_spend'])}) into mid tier would add about {cr(F1['whatif_revenue_delta'])} of attributed revenue, **if** mid-tier ROAS holds at scale. Treat that as an upper bound, since returns usually fall as you scale a tier.

**2. Stop shipping direct-pitch scripts.**
{pct(F2['direct_share_of_videos'])} of video deals open with a plain "this video is sponsored by" style hook. They get {pct(direct_less)} less engagement than the other hooks combined ({F2['direct_med']*100:.2f}% vs {F2['other_med']*100:.2f}% median; 95% CI {ci('direct_pitch_gap')}). Direct pitch ranks last in {n_last} of 8 categories{" and second to last in the rest" if n_bot2 == 8 and n_last < 8 else (f" and in the bottom two in {n_bot2}" if n_last < 8 else "")}.
The best hook in each category beats direct pitch significantly in {sig_hooks} of {len(H)} categories. Best hook by category: {best_list}. Each of these rests on {n_min} to {n_max} videos, and the gaps among the non-direct hooks are smaller, so use this as a starting playbook and keep testing it.

**3. Cap how often one creator is booked.**
Creators with 3 or more deals in the prior 30 days get {pct(F3['drop'])} lower engagement (95% CI {ci('saturation_drop')}; {pct(F3['tier_controlled_drop'])} after controlling for tier). {F3['n_sat']:,} deals ({pct(F3['pct_deals_sat'])} of the book, {cr(F3['spend_sat'])} of spend) were booked into saturated creators and returned {F3['roas_sat']:.2f} ROAS against {F3['roas_ok']:.2f} for the rest.
Rule: a cooldown flag at 3 deals in 30 days, built into the pitch tool.

**4. Put the call to action in the middle of the video.**
Mid-video CTAs get {pct(F4['lift_mid_vs_end'])} higher CTR than end CTAs ({F4['mid_ctr']*100:.2f}% vs {F4['end_ctr']*100:.2f}%; 95% CI {ci('cta_mid_vs_end_lift')}), and clicks convert better too ({pct(F4['c2c_mid'],1)} vs {pct(F4['c2c_end'],1)}). Half of all video deals ({pct(F4['pct_deals_end'])}) still put it at the end. This is the cheapest change on the list: it costs nothing.

**5. Match creator niche to brand category.**
{pct(fit['pct_mismatch'])} of deals pair a creator with a category outside their niche. Those get {pct(1 - fit['mismatch']/fit['match'])} lower engagement ({fit['mismatch']*100:.2f}% vs {fit['match']*100:.2f}%; 95% CI {ci('mismatch_drop')}). Add a niche-fit check to every pitch.

## Creators to review
{len(losers)} large creators account for {cr(losers.spend.sum())} of spend and a net loss of {cr(losers.net_inr.sum())} (ROAS below 0.7, top 40% by spend). Full list in `quietly_losing_money.csv`. Top of the list:

| Creator | Tier | Niche | Deals | Spend | ROAS |
|---|---|---|---|---|---|
""" + "\n".join(f"| {r.creator_name} ({r.creator_handle}) | {r.tier} | {r.creator_niche} | {r.deals} | {cr(r.spend)} | {r.roas:.2f} |" for r in losers.head(5).itertuples()) + f"""

On the other side, {n_w} creators have ROAS above 1.5 with 6+ deals, and {n_w_small} of them are nano, micro or mid tier. Single-creator ROAS rests on a limited number of deals each, so treat both lists as shortlists for review, not verdicts.

## What I tested and did not headline
- **Language:** raw ROAS ranges from {_raw[lang_lo]:.2f} ({lang_lo}) to {_raw[lang_hi]:.2f} ({lang_hi}), but that gap does not survive scrutiny. Comparing creators (not deals), adjusted for tier, there is no detectable language effect (p = {p_lang:.2f}). The raw gap comes partly from tier mix (languages with more macro and mega spend look worse) and partly from chance variation across small groups of creators. Language is a targeting filter, not a performance driver.
- **City:** same result. Tier-adjusted, city makes no detectable difference (p = {p_city:.2f}), and metro vs tier 2 and 3 creators perform alike (p = {p_ctype:.2f}).
- **Video length:** {dur_line}

## Limits
- Simulated data. Real data will be noisier, and some of these effects will shrink.
- ROAS depends on attribution (tracked links and codes). Brands that do not share conversion data will need a proxy such as CPV or engagement.
- Deals from the same creator are not independent, so every confidence interval above comes from resampling creators (400 bootstrap rounds), not deals. All five headline effects stay clearly away from zero.
- Associations, not experiments. The cooldown rule and the hook rule should be tested on live deals before they become policy.

## What I would do in the first 30 days
1. Load Monk's real deal log into the same schema and rerun this analysis.
2. Ship the pitch tool: enter a brief, get creator mix, hook angles, CTA placement and budget split by tier.
3. Run a clean test of the hook and CTA changes on live campaigns.
"""
open("docs/decision_memo.md", "w").write(md)
print(md)
