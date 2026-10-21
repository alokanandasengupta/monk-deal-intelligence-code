import os, sys
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(ROOT); sys.path.insert(0, os.path.join(ROOT, "src"))  # run from anywhere; paths are relative to the repo root
"""
Monk Deal Intelligence: analytics core.
Run: python3 pipeline/02_analysis.py   (reads data/monk_deals_simulated.csv, writes outputs/)
Works unchanged on real data with the same column names.
"""
import os, json
import numpy as np, pandas as pd
from scipy import stats
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

OUT = "outputs"; CH = f"{OUT}/charts"
os.makedirs(CH, exist_ok=True)
df = pd.read_csv("data/monk_deals_simulated.csv", parse_dates=["deal_date"])
R = {}  # results dict for memo

# ---------------- 1. clean + helper columns ----------------
n0 = len(df)
df = df.drop_duplicates("deal_id")
df = df[(df.views > 0) & (df.deal_value_inr > 0)].copy()
R["rows_in"], R["rows_clean"] = n0, len(df)
df["is_video"] = df.format.isin(["reel", "youtube_integration", "youtube_dedicated"])
df["dur_bucket"] = pd.cut(df.duration_sec, [0, 20, 30, 45, 90, 1000], labels=["<=20s", "21-30s", "31-45s", "46-90s", "90s+"])
df["saturation"] = pd.cut(df.creator_deals_last_30d, [-1, 2, 100], labels=["0-2 recent deals", "3+ recent deals"])
df["profitable"] = df.roas >= 1
df["month"] = df.deal_date.dt.to_period("M").astype(str)
tier_order = ["nano", "micro", "mid", "macro", "mega"]
df["tier"] = pd.Categorical(df.tier, tier_order, ordered=True)

def bench(by):
    g = df.groupby(by, observed=True)
    t = g.agg(deals=("deal_id", "count"), spend_inr=("deal_value_inr", "sum"), revenue_inr=("revenue_inr", "sum"),
              med_eng_rate=("engagement_rate", "median"), med_ctr=("ctr", "median"),
              med_cpv=("cpv_inr", "median"), pct_profitable=("profitable", "mean"))
    t["roas"] = t.revenue_inr / t.spend_inr
    t["spend_share"] = t.spend_inr / t.spend_inr.sum()
    return t.round(4)

B = {k: bench(k) for k in ["tier", "brand_category", "format", "platform"]}
with pd.ExcelWriter(f"{OUT}/benchmarks.xlsx") as xw:
    for k, t in B.items():
        t.to_excel(xw, sheet_name=f"by_{k}")
overall_roas = df.revenue_inr.sum() / df.deal_value_inr.sum()
R["overall_roas"] = overall_roas
R["pct_deals_losing"] = 1 - df.profitable.mean()
R["tier"] = B["tier"][["deals", "roas", "spend_share", "med_eng_rate", "med_cpv", "pct_profitable"]].round(3).to_dict("index")

# ---------------- 2. statistical findings ----------------
def mw(a, b):
    u, p = stats.mannwhitneyu(a, b, alternative="two-sided")
    return p

# F1 budget: mega+macro vs mid
big = df[df.tier.isin(["macro", "mega"])]; mid = df[df.tier == "mid"]
R["F1"] = dict(
    bigshare=float(big.deal_value_inr.sum() / df.deal_value_inr.sum()),
    big_roas=float(big.revenue_inr.sum() / big.deal_value_inr.sum()),
    mid_roas=float(mid.revenue_inr.sum() / mid.deal_value_inr.sum()),
    big_lose=float((big.roas < 1).mean()), mid_lose=float((mid.roas < 1).mean()),
    p=float(mw(big.roas, mid.roas)))
# what-if: move 20% of macro+mega spend into mid at mid's median ROAS-of-aggregate
moved = 0.20 * big.deal_value_inr.sum()
R["F1"]["whatif_revenue_delta"] = float(moved * (R["F1"]["mid_roas"] - R["F1"]["big_roas"]))
R["F1"]["whatif_moved_spend"] = float(moved)

# F2 hooks: direct_pitch vs the best non-direct hook, per category, plus overall
vid = df[df.is_video].copy()
hp = vid.pivot_table(index="brand_category", columns="hook_type", values="engagement_rate", aggfunc="median")
hp.to_csv(f"{OUT}/hook_by_category_median_engagement.csv")
rows = []
for cat, g in vid.groupby("brand_category"):
    med = g.groupby("hook_type").engagement_rate.median().sort_values(ascending=False)
    best = med.index[0]
    dp = g[g.hook_type == "direct_pitch"].engagement_rate; bs = g[g.hook_type == best].engagement_rate
    rows.append(dict(category=cat, best_hook=best, best_eng=med.iloc[0], direct_pitch_eng=med.get("direct_pitch", np.nan),
                     lift_vs_direct=med.iloc[0] / med.get("direct_pitch", np.nan) - 1,
                     n_best=len(bs), n_direct=len(dp), p=mw(bs, dp)))
H = pd.DataFrame(rows).round(4); H.to_csv(f"{OUT}/best_hook_per_category.csv", index=False)
others = vid[vid.hook_type != "direct_pitch"].engagement_rate; dpall = vid[vid.hook_type == "direct_pitch"].engagement_rate
R["F2"] = dict(table=H.to_dict("records"), direct_med=float(dpall.median()), other_med=float(others.median()),
               gap=float(others.median() / dpall.median() - 1), p=float(mw(others, dpall)),
               direct_share_of_videos=float((vid.hook_type == "direct_pitch").mean()),
               kruskal_p=float(stats.kruskal(*[g.engagement_rate.values for _, g in vid.groupby("hook_type")])[1]))

# F3 saturation: same creator before vs after heavy bursts is complex; use band comparison + within-tier control
sat = df.groupby("saturation", observed=True).agg(n=("deal_id", "count"), eng=("engagement_rate", "median"),
                                                   ctr=("ctr", "median"), roas=("roas", "median"))
a = df[df.creator_deals_last_30d <= 2]; b = df[df.creator_deals_last_30d >= 3]
# control for tier: ratio of median eng within each tier, averaged
ratios = []
for t in tier_order:
    x = a[a.tier == t].engagement_rate; y = b[b.tier == t].engagement_rate
    if len(y) >= 10:
        ratios.append(y.median() / x.median())
R["F3"] = dict(eng_ok=float(a.engagement_rate.median()), eng_sat=float(b.engagement_rate.median()),
               drop=float(1 - b.engagement_rate.median() / a.engagement_rate.median()),
               tier_controlled_drop=float(1 - np.mean(ratios)), p=float(mw(a.engagement_rate, b.engagement_rate)),
               n_sat=int(len(b)), pct_deals_sat=float(len(b) / len(df)),
               spend_sat=float(b.deal_value_inr.sum()),
               roas_ok=float(a.revenue_inr.sum() / a.deal_value_inr.sum()), roas_sat=float(b.revenue_inr.sum() / b.deal_value_inr.sum()))
by_n = df.groupby(df.creator_deals_last_30d.clip(upper=6)).engagement_rate.median()

# F4 CTA position
cta = vid.groupby("cta_position").agg(n=("deal_id", "count"), ctr=("ctr", "median"), conv_rate=("conversions", "sum"),
                                       clicks=("link_clicks", "sum"))
cta["click_to_conv"] = cta.conv_rate / cta.clicks
mid_ctr = vid[vid.cta_position == "mid"].ctr; end_ctr = vid[vid.cta_position == "end"].ctr
R["F4"] = dict(mid_ctr=float(mid_ctr.median()), end_ctr=float(end_ctr.median()),
               start_ctr=float(vid[vid.cta_position == "start"].ctr.median()),
               lift_mid_vs_end=float(mid_ctr.median() / end_ctr.median() - 1), p=float(mw(mid_ctr, end_ctr)),
               n_mid=int(len(mid_ctr)), n_end=int(len(end_ctr)),
               pct_deals_end=float((vid.cta_position == "end").mean()),
               c2c_mid=float(cta.loc["mid", "click_to_conv"]), c2c_end=float(cta.loc["end", "click_to_conv"]))

# F5 duration sweet spot per category (reels only)
reels = df[df.format == "reel"]
dp_rows = []
for cat, g in reels.groupby("brand_category"):
    t = g.groupby("dur_bucket", observed=True).engagement_rate.agg(["median", "count"])
    t = t[t["count"] >= 15]
    if len(t) < 2: continue
    best = t["median"].idxmax(); worst = t["median"].idxmin()
    dp_rows.append(dict(category=cat, best_bucket=str(best), best_eng=t["median"].max(), worst_bucket=str(worst),
                        worst_eng=t["median"].min(), lift=t["median"].max() / t["median"].min() - 1,
                        kruskal_p=float(stats.kruskal(*[x.engagement_rate.values for _, x in g.groupby("dur_bucket", observed=True) if len(x) >= 15])[1])))
D = pd.DataFrame(dp_rows).round(4); D.to_csv(f"{OUT}/duration_sweet_spot.csv", index=False)
R["F5"] = D.to_dict("records")

# fit effect (bonus)
fit = df.assign(fit=df.creator_niche == df.brand_category).groupby("fit").engagement_rate.median()
R["fit"] = dict(match=float(fit[True]), mismatch=float(fit[False]), p=float(mw(df[df.creator_niche == df.brand_category].engagement_rate,
                                                                                  df[df.creator_niche != df.brand_category].engagement_rate)),
                pct_mismatch=float((df.creator_niche != df.brand_category).mean()))

# ---------------- 3. creator scorecard ----------------
cs = df.groupby(["creator_id", "creator_name", "creator_handle", "creator_language", "creator_city", "tier", "creator_niche"],
                observed=True).agg(deals=("deal_id", "count"), spend=("deal_value_inr", "sum"), revenue=("revenue_inr", "sum"),
                                   med_eng=("engagement_rate", "median"), eng_sd=("engagement_rate", "std"),
                                   max_burst=("creator_deals_last_30d", "max"),
                                   sat_deals=("creator_deals_last_30d", lambda s: (s >= 3).sum())).reset_index()
cs["roas"] = cs.revenue / cs.spend
cs["consistency"] = 1 - (cs.eng_sd / cs.med_eng).clip(0, 1).fillna(0.5)
cs["saturated_flag"] = cs.sat_deals / cs.deals >= 0.25
cs["net_inr"] = cs.revenue - cs.spend
cs = cs[cs.deals >= 5].copy()
cs["value_score"] = (cs.roas.rank(pct=True) * 0.6 + cs.consistency.rank(pct=True) * 0.25 + (~cs.saturated_flag) * 0.15) * 100
cs = cs.sort_values("value_score", ascending=False).round(3)
cs.to_csv(f"{OUT}/creator_scorecard.csv", index=False)
losers = cs[(cs.roas < 0.7) & (cs.spend >= cs.spend.quantile(0.6))].sort_values("net_inr").head(10)
losers.to_csv(f"{OUT}/quietly_losing_money.csv", index=False)
winners = cs[(cs.roas > 1.5) & (cs.deals >= 6)].sort_values("roas", ascending=False).head(10)
R["scorecard"] = dict(n=int(len(cs)), top5=cs.head(5)[["creator_name", "creator_handle", "tier", "creator_niche", "roas", "deals"]].to_dict("records"),
                      losers=losers.head(5)[["creator_name", "creator_handle", "tier", "creator_niche", "roas", "spend", "net_inr", "deals"]].to_dict("records"),
                      losers_total_loss=float(losers.net_inr.sum()), losers_total_spend=float(losers.spend.sum()),
                      pct_creators_saturated=float(cs.saturated_flag.mean()),
                      winners=winners.head(5)[["creator_name", "creator_handle", "tier", "creator_niche", "roas", "deals"]].to_dict("records"))

# ---------------- 4. language and city ----------------
def reg(by, minn=40):
    g = df.groupby(by, observed=True).agg(deals=("deal_id", "count"), spend=("deal_value_inr", "sum"), revenue=("revenue_inr", "sum"),
                                          med_eng=("engagement_rate", "median"), med_cpv=("cpv_inr", "median"))
    g["roas"] = g.revenue / g.spend
    g["avg_fee"] = g.spend / g.deals
    return g[g.deals >= minn].sort_values("roas", ascending=False).round(3)
LG = reg("creator_language"); LG.to_csv(f"{OUT}/by_language.csv")
CT = reg("creator_city", 60); CT.to_csv(f"{OUT}/by_city.csv")
metros = {"Delhi", "Mumbai", "Bengaluru", "Hyderabad", "Chennai", "Kolkata", "Pune", "Gurugram", "Noida"}
df["city_type"] = np.where(df.creator_city.isin(metros), "metro", "tier2/3")
cty = df.groupby("city_type").agg(deals=("deal_id", "count"), spend=("deal_value_inr", "sum"), revenue=("revenue_inr", "sum"),
                                  med_eng=("engagement_rate", "median"), avg_fee=("deal_value_inr", "mean"))
cty["roas"] = cty.revenue / cty.spend
R["lang"] = LG[["deals", "roas", "med_eng", "avg_fee"]].to_dict("index")
R["city_type"] = cty.round(3).to_dict("index")
R["city_type_p_eng"] = float(mw(df[df.city_type == "metro"].engagement_rate, df[df.city_type != "metro"].engagement_rate))

# ---------------- 5. charts ----------------
plt.rcParams.update({"font.family": "DejaVu Sans", "axes.spines.top": False, "axes.spines.right": False, "font.size": 10})
NAVY, RED, GREY, TEAL = "#1F3A5F", "#C0392B", "#9AA5B1", "#1F8A70"

# c1 spend share vs ROAS by tier
t = B["tier"].reindex(tier_order)
fig, ax = plt.subplots(figsize=(8, 4.2))
x = np.arange(len(t)); ax.bar(x - 0.2, t.spend_share * 100, 0.4, color=GREY, label="% of spend")
ax2 = ax.twinx(); ax2.spines["right"].set_visible(True)
ax2.bar(x + 0.2, t.roas, 0.4, color=[RED if r < 1 else TEAL for r in t.roas], label="ROAS")
ax2.axhline(1, color="black", lw=0.8, ls="--"); ax2.set_ylabel("ROAS (1.0 = break-even)"); ax.set_ylabel("% of spend")
ax.set_xticks(x); ax.set_xticklabels(tier_order); ax.set_title("Where the money goes vs what it returns, by creator tier", loc="left", fontweight="bold")
for i, r in enumerate(t.roas): ax2.text(i + 0.2, r + 0.03, f"{r:.2f}", ha="center", fontsize=9)
plt.tight_layout(); plt.savefig(f"{CH}/1_spend_vs_roas_by_tier.png", dpi=160); plt.close()

# c2 hook heatmap
fig, ax = plt.subplots(figsize=(8.5, 4.6))
hp2 = hp[["story", "question", "shock", "social_proof", "challenge", "demo", "direct_pitch"]] * 100
im = ax.imshow(hp2.values, cmap="Blues", aspect="auto")
ax.set_xticks(range(hp2.shape[1])); ax.set_xticklabels(hp2.columns, rotation=30, ha="right")
ax.set_yticks(range(hp2.shape[0])); ax.set_yticklabels(hp2.index)
for i in range(hp2.shape[0]):
    for j in range(hp2.shape[1]):
        v = hp2.values[i, j]; star = "*" if v == hp2.iloc[i].max() else ""
        ax.text(j, i, f"{v:.1f}{star}", ha="center", va="center", fontsize=8, color="white" if v > np.nanmax(hp2.values) * 0.7 else "black")
ax.set_title("Median engagement rate (%) by hook and category (* = best)", loc="left", fontweight="bold")
plt.tight_layout(); plt.savefig(f"{CH}/2_hook_heatmap.png", dpi=160); plt.close()

# c3 saturation
fig, ax = plt.subplots(figsize=(6.5, 4))
ax.bar(by_n.index.astype(int), by_n.values * 100, color=[TEAL if i <= 2 else RED for i in by_n.index])
ax.set_xlabel("Creator's deals in prior 30 days (6 = 6+)"); ax.set_ylabel("Median engagement rate (%)")
ax.set_title("Engagement decays when a creator is oversold", loc="left", fontweight="bold")
plt.tight_layout(); plt.savefig(f"{CH}/3_saturation_decay.png", dpi=160); plt.close()

# c4 CTA
fig, ax = plt.subplots(figsize=(5.5, 4))
o = ["start", "mid", "end"]; v = [vid[vid.cta_position == k].ctr.median() * 100 for k in o]
ax.bar(o, v, color=[GREY, TEAL, RED]); ax.set_ylabel("Median CTR (%)"); ax.set_title("CTA placement and CTR", loc="left", fontweight="bold")
for i, x_ in enumerate(v): ax.text(i, x_ + 0.02, f"{x_:.2f}%", ha="center")
plt.tight_layout(); plt.savefig(f"{CH}/4_cta_position.png", dpi=160); plt.close()

# c5 language roas
fig, ax = plt.subplots(figsize=(7.5, 4.4))
l = LG.sort_values("roas")
ax.barh(l.index, l.roas, color=[RED if r < 1 else TEAL for r in l.roas]); ax.axvline(1, color="black", lw=0.8, ls="--")
ax.set_xlabel("ROAS"); ax.set_title("ROAS by creator language", loc="left", fontweight="bold")
plt.tight_layout(); plt.savefig(f"{CH}/5_roas_by_language.png", dpi=160); plt.close()

# c6 duration
if len(D):
    fig, ax = plt.subplots(figsize=(8, 4.2))
    piv = reels.pivot_table(index="brand_category", columns="dur_bucket", values="engagement_rate", aggfunc="median", observed=True) * 100
    for c in piv.index: ax.plot(piv.columns.astype(str), piv.loc[c], marker="o", label=c, lw=1.4)
    ax.set_ylabel("Median engagement rate (%)"); ax.legend(ncol=4, fontsize=8, frameon=False)
    ax.set_title("Reel length vs engagement, by category", loc="left", fontweight="bold")
    plt.tight_layout(); plt.savefig(f"{CH}/6_duration_by_category.png", dpi=160); plt.close()

json.dump(R, open(f"{OUT}/results.json", "w"), indent=2, default=float)
df.to_csv(f"{OUT}/deals_clean.csv", index=False)
print("done", R["rows_clean"], "rows")
