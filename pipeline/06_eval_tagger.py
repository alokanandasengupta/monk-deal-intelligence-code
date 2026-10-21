import os, sys
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(ROOT); sys.path.insert(0, os.path.join(ROOT, "src"))  # run from anywhere; paths are relative to the repo root
import json, collections
import numpy as np, pandas as pd
import ai_layer as ai
sys.path.insert(0, os.path.join(ROOT, "pipeline"))
from tagger_testset import TESTSET

O = ai.O
# A) simulator scripts (templated, so this is a plumbing check, NOT evidence of real-world accuracy)
df = pd.read_csv(f"{O}/deals_clean.csv")
v = df[df.script_text.notna() & (df.script_text != "")].sample(1000, random_state=0)
sim = [ai.tag_rules(s) for s in v.script_text]
sim_hook = np.mean([t["hook_type"] == h for t, h in zip(sim, v.hook_type)])
sim_cta = np.mean([t["cta_position"] == c for t, c in zip(sim, v.cta_position)])

# B) hand-written messy set
res = [ai.tag_rules(s) for s, _, _ in TESTSET]
hook_ok = [r["hook_type"] == h for r, (_, h, _) in zip(res, TESTSET)]
cta_ok = [r["cta_position"] == c for r, (_, _, c) in zip(res, TESTSET)]
unclear = sum(r["hook_type"] == "unclear" for r in res)
by_hook = collections.defaultdict(list)
for ok, (_, h, _) in zip(hook_ok, TESTSET): by_hook[h].append(ok)
conf = collections.Counter((h, r["hook_type"]) for r, (_, h, _) in zip(res, TESTSET) if r["hook_type"] != h)
out = dict(sim_n=len(v), sim_hook_acc=float(sim_hook), sim_cta_acc=float(sim_cta), hand_n=len(TESTSET),
           hand_hook_acc=float(np.mean(hook_ok)), hand_cta_acc=float(np.mean(cta_ok)), hand_unclear=int(unclear),
           hook_acc_by_type={k: float(np.mean(x)) for k, x in by_hook.items()},
           top_confusions=[dict(true=a, predicted=b, n=n) for (a, b), n in conf.most_common(8)],
           language_style_hinglish_detected=int(sum(r["language_style"] == "hinglish" for r in res)))
json.dump(out, open(f"{O}/tagger_eval.json", "w"), indent=2)
print(json.dumps(out, indent=1))
print("\nMISSES:")
for r, (s, h, c) in zip(res, TESTSET):
    if r["hook_type"] != h or r["cta_position"] != c:
        print(f"  true=({h},{c}) got=({r['hook_type']},{r['cta_position']}) :: {s[:80]}")
