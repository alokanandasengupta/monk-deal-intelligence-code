import os, sys
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(ROOT); sys.path.insert(0, os.path.join(ROOT, "src"))  # run from anywhere; paths are relative to the repo root
"""Run: python scripts/demo.py   (needs outputs/ folder from this project). Shows the three things the app will do."""
import pandas as pd, pitch_model as pm, pitch_builder as pb, ai_layer as ai
art = pm.load(); df, sc = pb.load_data()
line = lambda t: print("\n" + "=" * 78 + f"\n{t}\n" + "=" * 78)

line("1. BENCHMARKS: where does the money go, and what does it return?")
t = df.groupby("tier").agg(deals=("deal_id", "count"), spend_cr=("deal_value_inr", lambda x: x.sum() / 1e7), roas=("roas", "median"))
t["spend_share_%"] = (t.spend_cr / t.spend_cr.sum() * 100).round(0)
print(t.reindex(["nano", "micro", "mid", "macro", "mega"]).round(2).to_string())
L = pd.read_csv(f"{pm.O}/quietly_losing_money.csv")
print(f"\nCreators 'quietly losing money': {len(L)}. Worst 5:")
print(L.head(5).iloc[:, :8].to_string(index=False))

line("2. HOOK PLAYBOOK: best opening + CTA for a mid-tier beauty reel (model, p50 ROAS)")
bp = pm.best_plays(art, "mid", "beauty", "beauty", fmt="reel", top=6)
print(bp[["hook", "cta", "roas_p10", "roas_p50", "roas_p90", "roas_vs_top", "near_tie_with_top"]].round(2).to_string(index=False))

line("3. SCRIPT SCORER: paste a creator script, get a score and a fix")
brief = dict(tier="mid", creator_niche="beauty", brand_category="beauty", format="reel", creator_deals_last_30d=1)
script = "Hello everyone, today I want to tell you about this new serum. It has niacinamide. Link in bio."
s = ai.score_script(art, script, brief, backend="rules")
print("SCRIPT :", script)
print("TAGS   :", s["tags"]["hook_type"], "| CTA", s["tags"]["cta_position"])
print("SCORE  :", s["score"], "/ 100")
for r in s["recommendations"]: print(" -", r["change"], f"(ROAS {r['roas_effect']:+.0%})")
print("REWRITE:", s["rewrite_offline"])

line("4. PITCH BUILDER: brief in, plan out")
r = pb.build_pitch(art, dict(brand="Glowvera", brand_category="beauty", budget_inr=1_500_000, goal="sales"), df, sc)
print(r["markdown"].split("## Creator shortlist")[0])
print("(+ creator shortlist, creative guidance, expected outcome)")
