import os, sys
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(ROOT); sys.path.insert(0, os.path.join(ROOT, "src"))  # run from anywhere; paths are relative to the repo root
import json, numpy as np, pandas as pd
import pitch_model as pm
art = pm.load(); O = pm.O
df = pd.read_csv(f"{O}/deals_clean.csv").reset_index(drop=True)
rng = np.random.default_rng(1)
cats = art["cats"]; models = art["models"]
def p50(d, t):
    X = pm.featurize(d, cats); return models[t][0.5].predict(X)
def effect(d, change, t):
    a = p50(d, t); d2 = change(d.copy()); b = p50(d2, t)
    return float(np.mean(b) / np.mean(a) - 1)
out = {}
vid = df[df.format.isin(pm.VIDEO)].sample(1500, random_state=1)
reel = df[df.format == "reel"].sample(1200, random_state=2)
allx = df.sample(2000, random_state=3)
# 1 hook: direct_pitch -> story, on deals that currently are neither (so we compare same population)
def set_hook(h):
    def f(d): d["hook_type"] = h; return d
    return f
out["hook_direct_to_story_eng"] = effect(vid, set_hook("story"), "engagement_rate") if False else None
base = vid.copy()
a = p50(base.assign(hook_type="direct_pitch"), "engagement_rate"); b = p50(base.assign(hook_type="story"), "engagement_rate")
out["eng_story_vs_direct"] = float(b.mean() / a.mean() - 1)
a = p50(base.assign(hook_type="direct_pitch"), "roas"); b = p50(base.assign(hook_type="story"), "roas")
out["roas_story_vs_direct"] = float(b.mean() / a.mean() - 1)
# 2 saturation 1 vs 4 recent deals
a = p50(allx.assign(creator_deals_last_30d=1), "engagement_rate"); b = p50(allx.assign(creator_deals_last_30d=4), "engagement_rate")
out["eng_saturation_4_vs_1"] = float(b.mean() / a.mean() - 1)
a = p50(allx.assign(creator_deals_last_30d=1), "roas"); b = p50(allx.assign(creator_deals_last_30d=4), "roas")
out["roas_saturation_4_vs_1"] = float(b.mean() / a.mean() - 1)
# 3 niche fit: brand category = creator niche vs a different category
other = {c: cats["brand_category"][(i + 3) % len(cats["brand_category"])] for i, c in enumerate(cats["brand_category"])}
a = p50(allx.assign(brand_category=allx.creator_niche), "engagement_rate"); b = p50(allx.assign(brand_category=allx.creator_niche.map(other)), "engagement_rate")
out["eng_mismatch_vs_match"] = float(b.mean() / a.mean() - 1)
# 4 CTA mid vs end (ROAS), should help; on engagement should be ~0
for t in ["roas", "engagement_rate"]:
    a = p50(vid.assign(cta_position="end"), t); b = p50(vid.assign(cta_position="mid"), t)
    out[f"{t}_cta_mid_vs_end"] = float(b.mean() / a.mean() - 1)
# 5 duration 20s vs 45s on reels, expected ~0
a = p50(reel.assign(duration_sec=20), "engagement_rate"); b = p50(reel.assign(duration_sec=45), "engagement_rate")
out["eng_duration_45_vs_20"] = float(b.mean() / a.mean() - 1)
# 6 post hour evening vs 4am, and weekend vs weekday day (should be small)
a = p50(allx.assign(post_hour=4), "engagement_rate"); b = p50(allx.assign(post_hour=20), "engagement_rate")
out["eng_hour20_vs_hour4"] = float(b.mean() / a.mean() - 1)
a = p50(allx.assign(post_day="Tuesday"), "engagement_rate"); b = p50(allx.assign(post_day="Saturday"), "engagement_rate")
out["eng_saturday_vs_tuesday"] = float(b.mean() / a.mean() - 1)
# 7 language/city were excluded from the model on purpose
json.dump(out, open(f"{O}/whatif_results.json", "w"), indent=2)
for k, v in out.items():
    if v is not None: print(f"{k:28s} {v*100:+.1f}%")
print("\nbest plays: beauty creator, mid tier, beauty brand, reel")
bp = pm.best_plays(art, "mid", "beauty", "beauty", top=5); print(bp.round(3).to_string(index=False))
print("\nbest plays: tech creator, mid tier, tech brand, reel")
print(pm.best_plays(art, "mid", "tech", "tech", top=3).round(3).to_string(index=False))
print("\nSingle brief example:")
print(json.dumps(pm.predict_brief(art, dict(tier="mid", creator_niche="fashion", brand_category="fashion", format="reel", hook_type="social_proof", duration_sec=22, cta_position="mid", creator_deals_last_30d=1)), indent=1))
