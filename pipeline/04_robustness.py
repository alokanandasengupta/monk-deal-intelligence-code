import os, sys
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(ROOT); sys.path.insert(0, os.path.join(ROOT, "src"))  # run from anywhere; paths are relative to the repo root
"""Creator-clustered bootstrap: resample creators (not deals) to get honest 95% CIs on the headline effects."""
import json
import numpy as np, pandas as pd

O = "outputs"
df = pd.read_csv(f"{O}/deals_clean.csv")
rng = np.random.default_rng(7)
df = df.reset_index(drop=True)
idx_by_creator = [g.index.values for _, g in df.groupby("creator_id")]
nC = len(idx_by_creator)

def stats_of(d):
    big = d[d.tier.isin(["macro", "mega"])]; mid = d[d.tier == "mid"]
    vid = d[d.is_video]
    dp = vid[vid.hook_type == "direct_pitch"].engagement_rate.median()
    oth = vid[vid.hook_type != "direct_pitch"].engagement_rate.median()
    ok = d[d.creator_deals_last_30d <= 2].engagement_rate.median()
    sat = d[d.creator_deals_last_30d >= 3].engagement_rate.median()
    mid_c = vid[vid.cta_position == "mid"].ctr.median(); end_c = vid[vid.cta_position == "end"].ctr.median()
    fit = d[d.creator_niche == d.brand_category].engagement_rate.median()
    mis = d[d.creator_niche != d.brand_category].engagement_rate.median()
    return dict(
        tier_roas_gap=mid.revenue_inr.sum() / mid.deal_value_inr.sum() - big.revenue_inr.sum() / big.deal_value_inr.sum(),
        direct_pitch_gap=1 - dp / oth, saturation_drop=1 - sat / ok, cta_mid_vs_end_lift=mid_c / end_c - 1,
        mismatch_drop=1 - mis / fit)

point = stats_of(df)
B = []
for _ in range(400):
    pick = rng.integers(0, nC, nC)
    rows = np.concatenate([idx_by_creator[i] for i in pick])
    B.append(stats_of(df.iloc[rows]))
B = pd.DataFrame(B)
out = {k: dict(point=float(point[k]), lo=float(B[k].quantile(0.025)), hi=float(B[k].quantile(0.975))) for k in point}
json.dump(out, open(f"{O}/robustness.json", "w"), indent=2)
for k, v in out.items():
    print(f"{k:22s} point {v['point']:.3f}  95% CI [{v['lo']:.3f}, {v['hi']:.3f}]")
