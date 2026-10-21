"""Monk Pitch Copilot: local web app.  Run:  python app.py   then open http://127.0.0.1:5050
All data is SIMULATED."""
import os, sys, subprocess, json, math
import numpy as np, pandas as pd
from flask import Flask, jsonify, request, send_from_directory

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "src"))
import pitch_model as pm, pitch_builder as pb, ai_layer as ai

TIERS = ["nano", "micro", "mid", "macro", "mega"]
HOOKS = ["story", "question", "shock", "social_proof", "challenge", "demo", "direct_pitch"]


def load_model():
    try:
        return pm.load()
    except Exception as e:  # missing file or scikit-learn version mismatch: retrain from the data (about a minute)
        print(f"Model file could not be loaded ({type(e).__name__}). Retraining from the data, this takes about a minute...")
        subprocess.check_call([sys.executable, os.path.join(HERE, "src", "pitch_model.py")], cwd=HERE)
        return pm.load()


art = load_model()
df, sc = pb.load_data()
app = Flask(__name__, static_folder=os.path.join(HERE, "static"), static_url_path="/static")


def clean(o):
    if isinstance(o, dict): return {k: clean(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)): return [clean(v) for v in o]
    if isinstance(o, pd.DataFrame): return clean(o.to_dict("records"))
    if isinstance(o, (np.floating, float)): return None if (math.isnan(o) or math.isinf(o)) else float(o)
    if isinstance(o, np.integer): return int(o)
    if isinstance(o, np.bool_): return bool(o)
    return o


def ok(o): return jsonify(clean(o))


@app.get("/")
def index(): return send_from_directory(app.static_folder, "index.html")


EVIDENCE = json.load(open(os.path.join(pm.O, "evidence.json")))
_prof = df.groupby("creator_id").agg(prof=("roas", lambda x: float((x >= 1).mean())), n_deals=("deal_id", "count"))
_maxd = df.deal_date.max()
_m = df.groupby("creator_id")
_stats = pd.DataFrame(dict(
    followers=_m.followers.first(), platform=_m.platform.first(), language=_m.creator_language.first(),
    avg_fee=_m.deal_value_inr.mean(), avg_views=_m.views.mean(), eng_rate=_m.engagement_rate.median(),
    view_rate=_m.apply(lambda g: float((g.views / g.followers).median())),
    likes=_m.likes.mean(), comments=_m.comments.mean(), saves=_m.saves.mean(), shares=_m.shares.mean(),
    ctr=_m.ctr.median(), conv_rate=_m.apply(lambda g: float(g.conversions.sum() / max(g.link_clicks.sum(), 1))),
    cpv=_m.apply(lambda g: float(g.deal_value_inr.sum() / max(g.views.sum(), 1))),
    cpe=_m.apply(lambda g: float(g.deal_value_inr.sum() / max(g.engagements.sum(), 1))),
    cpa=_m.apply(lambda g: float(g.deal_value_inr.sum() / g.conversions.sum()) if g.conversions.sum() > 0 else None),
    best_hook=_m.apply(lambda g: g.groupby("hook_type").roas.median().idxmax()),
    mid_share=_m.apply(lambda g: float((g.cta_position == "mid").mean())),
    last_deal=_m.deal_date.max().dt.strftime("%Y-%m-%d"),
    brands=_m.brand.nunique(),
))
_recent = df[df.deal_date > _maxd - pd.Timedelta(days=30)].groupby("creator_id").size()


@app.get("/api/evidence")
def evidence(): return ok(EVIDENCE)


@app.get("/api/creators")
def creators():
    cat = request.args.get("category", "all")
    s = sc.copy().merge(_prof[["prof"]], left_on="creator_id", right_index=True).merge(_stats, left_on="creator_id", right_index=True)
    s["recent_30d"] = s.creator_id.map(_recent).fillna(0).astype(int)
    if cat != "all": s = s[s.creator_niche == cat]
    s = s[s.deals >= 4]
    cols = ["creator_name", "creator_handle", "creator_city", "tier", "creator_niche", "deals", "spend", "revenue", "roas", "prof", "consistency", "sat_deals", "recent_30d", "net_inr", "value_score", "saturated_flag"] + list(_stats.columns)
    backed = s[(s.roas >= 1) & (~s.saturated_flag)].sort_values("value_score", ascending=False).head(8)
    not_backed = s[(s.roas < 1) | s.saturated_flag].sort_values("net_inr").head(8)
    return ok(dict(category=cat, n_eligible=int(len(s)), n_backed=int(((s.roas >= 1) & (~s.saturated_flag)).sum()), n_not=int(((s.roas < 1) | s.saturated_flag).sum()),
                   backed=backed[cols], not_backed=not_backed[cols]))


@app.get("/api/sample")
def sample():
    cols = ["deal_date", "creator_name", "platform", "followers", "tier", "creator_niche", "brand", "brand_category", "format", "duration_sec", "hook_type", "cta_position", "post_day", "post_hour", "creator_deals_last_30d", "deal_value_inr", "views", "likes", "comments", "saves", "shares", "link_clicks", "conversions", "revenue_inr", "engagement_rate", "ctr", "roas"]
    s = df.sample(10, random_state=7).sort_values("deal_date")[cols].copy(); s["deal_date"] = s.deal_date.dt.strftime("%Y-%m-%d")
    return ok(dict(columns=cols, rows=s, n_rows=len(df), n_cols=len(df.columns)))


@app.get("/classic")
def classic(): return send_from_directory(app.static_folder, "index_7tabs.html")


@app.get("/advanced")
def advanced(): return send_from_directory(app.static_folder, "advanced.html")


@app.get("/api/meta")
def meta():
    return ok(dict(tiers=TIERS, hooks=HOOKS, categories=sorted(df.brand_category.unique()), niches=sorted(df.creator_niche.unique()),
                   formats=sorted(df.format.unique()), languages=sorted(df.creator_language.unique()),
                   hook_labels=ai.HOOK_LABEL, llm_available=bool(os.environ.get("ANTHROPIC_API_KEY")),
                   n_deals=len(df), n_creators=int(df.creator_id.nunique())))


@app.get("/api/benchmarks")
def benchmarks():
    spend, rev = df.deal_value_inr.sum(), df.revenue_inr.sum()
    t = df.groupby("tier").agg(deals=("deal_id", "count"), spend=("deal_value_inr", "sum"), revenue=("revenue_inr", "sum"),
                               median_roas=("roas", "median"), pct_profitable=("roas", lambda x: (x >= 1).mean()))
    t["roas"] = t.revenue / t.spend; t["spend_share"] = t.spend / spend
    t = t.reindex(TIERS).reset_index()
    big = float(t[t.tier.isin(["macro", "mega"])].spend_share.sum())
    heat = df.pivot_table(index="hook_type", columns="brand_category", values="engagement_rate", aggfunc="median").reindex(HOOKS)
    v = df[df.is_video & df.cta_position.isin(["start", "mid", "end"])]
    cta = v.groupby("cta_position").agg(ctr=("ctr", "median"), roas=("roas", "median"), deals=("deal_id", "count")).reindex(["start", "mid", "end"]).reset_index()
    d2 = df.copy(); d2["b"] = pd.cut(d2.creator_deals_last_30d, [-1, 1, 2, 3, 99], labels=["0-1", "2", "3", "4+"])
    sat = d2.groupby("b", observed=True).agg(eng=("engagement_rate", "median"), roas=("roas", "median"), deals=("deal_id", "count")).reset_index()
    sat["b"] = sat.b.astype(str)
    lose = pd.read_csv(f"{pm.O}/quietly_losing_money.csv")
    return ok(dict(
        headline=dict(deals=len(df), spend=spend, roas=rev / spend, pct_losing=float((df.roas < 1).mean()), big_share=big,
                      big_roas=float(t[t.tier.isin(["macro", "mega"])].revenue.sum() / t[t.tier.isin(["macro", "mega"])].spend.sum()),
                      mid_roas=float(t.loc[t.tier == "mid", "roas"].iloc[0]),
                      start=str(df.deal_date.min().date()), end=str(df.deal_date.max().date())),
        tiers=t, heat=dict(rows=HOOKS, cols=list(heat.columns), values=heat.values.tolist()), cta=cta, saturation=sat,
        losing=lose[["creator_name", "creator_handle", "tier", "creator_niche", "creator_city", "deals", "spend", "revenue", "roas", "net_inr"]]))


@app.get("/api/playbook")
def playbook():
    a = request.args
    tier, cat, fmt = a.get("tier", "mid"), a.get("category", "beauty"), a.get("format", "reel")
    p = pm.best_plays(art, tier, cat, cat, fmt=fmt, top=50)
    return ok(dict(plays=p, video=fmt in pm.VIDEO))


@app.post("/api/score")
def score():
    j = request.get_json(force=True)
    script = (j.get("script") or "").strip()
    if len(script) < 15: return ok(dict(error="Paste a script of at least a couple of sentences."))
    brief = dict(tier=j.get("tier", "mid"), creator_niche=j.get("niche", "beauty"), brand_category=j.get("category", "beauty"),
                 format=j.get("format", "reel"), brand=j.get("brand") or "this", creator_deals_last_30d=int(j.get("recent_deals", 1)))
    try:
        r = ai.score_script(art, script, brief, hook_override=j.get("hook_override") or None, backend=j.get("backend", "auto"))
    except Exception as e:
        return ok(dict(error=f"{type(e).__name__}: {e}"))
    return ok(r)


@app.post("/api/pitch")
def pitch():
    j = request.get_json(force=True)
    try:
        if "category" not in j or "budget_inr" not in j: return ok(dict(error="Both category and budget_inr are required."))
        brief = dict(brand=j.get("brand") or "the brand", brand_category=j["category"], budget_inr=float(j["budget_inr"]),
                     goal=j.get("goal", "sales"), format=j.get("format", "reel"))
        if j.get("languages"): brief["languages"] = j["languages"]
        r = pb.build_pitch(art, brief, df, sc)
    except Exception as e:
        return ok(dict(error=f"{type(e).__name__}: {e}"))
    r.pop("brief", None)
    return ok(r)


if __name__ == "__main__":
    print("\nMonk Pitch Copilot running at http://127.0.0.1:5050  (simulated data)\n")
    app.run(host="127.0.0.1", port=int(os.environ.get("PORT", 5050)), debug=False)
