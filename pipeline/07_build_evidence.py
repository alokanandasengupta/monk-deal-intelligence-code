import os, sys
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(ROOT); sys.path.insert(0, os.path.join(ROOT, "src"))  # run from anywhere; paths are relative to the repo root
"""Builds outputs/evidence.json: the 'does it matter?' tests and the evidence table. Run: python pipeline/07_build_evidence.py"""
import json, numpy as np, pandas as pd
from scipy.stats import kruskal
O = "outputs"
d = pd.read_csv(f"{O}/deals_clean.csv", parse_dates=["deal_date"])
d["fmt_group"] = d.is_video.map({True: "video", False: "static"})
for col in ["engagement_rate", "roas"]:
    med = d.groupby(["tier", "fmt_group"])[col].transform("median")
    d[col + "_idx"] = d[col] / med          # 1.00 = typical for that creator size and format
d["hour_band"] = pd.cut(d.post_hour, [-1, 4, 10, 16, 21, 24], labels=["Late night", "Morning", "Afternoon", "Evening", "Late night "]).astype(str).str.strip()
d["weekend"] = np.where(d.post_day.isin(["Saturday", "Sunday"]), "Weekend", "Weekday")
d["language_g"] = d.creator_language
d["city_g"] = d.city_type.map({"metro": "Metro", "tier2/3": "Tier 2 and 3 cities"})
d["dur_g"] = d.dur_bucket

def test(col, title, order=None, video_only=False):
    x = d[d.is_video] if video_only else d
    x = x[x[col].notna() & (x[col] != "nan")]
    groups = []
    for g, s in x.groupby(col):
        groups.append(dict(name=str(g), n=int(len(s)), eng_idx=float(s.engagement_rate_idx.median()), roas_idx=float(s.roas_idx.median())))
    if order: groups.sort(key=lambda r: order.index(r["name"]) if r["name"] in order else 99)
    else: groups.sort(key=lambda r: -r["n"])
    p = {}
    for m in ["engagement_rate_idx", "roas_idx"]:
        # creator-level: average each creator's deals within a group, then compare groups
        cl = x.groupby(["creator_id", col])[m].mean().reset_index()
        arrs = [v[m].values for _, v in cl.groupby(col) if len(v) > 5]
        p[m] = float(kruskal(*arrs).pvalue)
    return dict(title=title, groups=groups, p_eng=p["engagement_rate_idx"], p_roas=p["roas_idx"], creators=int(x.creator_id.nunique()))

myths = [test("language_g", "Creator's language"), test("city_g", "Creator's city type"),
         test("dur_g", "Video length", order=["<=20s", "21-30s", "31-45s", "46-90s", "90s+"], video_only=True),
         test("hour_band", "Time of posting", order=["Morning", "Afternoon", "Evening", "Late night"]),
         test("weekend", "Day of week", order=["Weekday", "Weekend"])]
rob, wi, mr, tg = [json.load(open(f"{O}/{f}.json")) for f in ["robustness", "whatif_results", "model_results", "tagger_eval"]]
res = json.load(open(f"{O}/results.json"))
import pitch_model as pm
art = pm.load(); rr, ee = [], []
for cat in sorted(d.brand_category.unique()):
    for tier in ["micro", "mid", "macro"]:
        base = dict(tier=tier, creator_niche=cat, brand_category=cat, format="reel", hook_type="story", cta_position="mid")
        a = pm.predict_brief(art, dict(base, post_hour=20)); b = pm.predict_brief(art, dict(base, post_hour=13))
        rr.append(a["roas"]["p50"] / b["roas"]["p50"] - 1); ee.append(a["engagement_rate"]["p50"] / b["engagement_rate"]["p50"] - 1)
hour_whatif = dict(roas=float(np.median(rr)), eng=float(np.median(ee)), n=len(rr))
print("hour what-if (8pm vs 1pm, same everything else): roas %+.1f%% eng %+.1f%%" % (100*hour_whatif["roas"], 100*hour_whatif["eng"]))
ev = d[d.hour_band == "Evening"]; ot = d[d.hour_band != "Evening"]
hour_raw = float(ev.roas_idx.median() / ot.roas_idx.median() - 1); print("raw evening vs rest roas index:", round(hour_raw, 3))
out = dict(hooks_by_cat=json.load(open(f'{O}/results.json'))['F2']['table'], hour_whatif=hour_whatif, hour_raw=hour_raw, myths=myths, robustness=rob, whatif=wi, model=mr, tagger=tg,
           facts=dict(n_sat=res["F3"]["n_sat"], pct_sat=res["F3"]["pct_deals_sat"], spend_sat=res["F3"]["spend_sat"], mismatch_share=res["fit"]["pct_mismatch"],
                      match_eng=res["fit"]["match"], mismatch_eng=res["fit"]["mismatch"], n_mid=res["F4"]["n_mid"], n_end=res["F4"]["n_end"], c2c_mid=res["F4"]["c2c_mid"], c2c_end=res["F4"]["c2c_end"]))
json.dump(out, open(f"{O}/evidence.json", "w"), indent=1)
for m in myths:
    print(m["title"], "| p eng %.3f roas %.3f" % (m["p_eng"], m["p_roas"]), "|", [(g["name"], g["n"], round(g["eng_idx"], 2), round(g["roas_idx"], 2)) for g in m["groups"]])
