"""
Monk Pitch Model: predicts a RANGE (p10 / p50 / p90) of engagement rate and ROAS for a proposed deal.
Train + validate:  python3 src/pitch_model.py   (or python3 pipeline/03_train_model.py)
Use in the app:    from pitch_model import load, predict_brief, best_plays
Validation is grouped by creator, so every score is on creators the model has never seen.
"""
import json
import numpy as np, pandas as pd, joblib
from scipy.stats import spearmanr
from sklearn.ensemble import HistGradientBoostingRegressor as HGB
from sklearn.model_selection import GroupKFold
from sklearn.inspection import permutation_importance

import os as _os
O = _os.path.join(_os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))), "outputs")
CAT_COLS = ["tier", "creator_niche", "platform", "brand_category", "format", "hook_type", "cta_position", "post_day"]
NUM_COLS = ["log_followers", "duration_sec", "post_hour", "creator_deals_last_30d", "log_fee", "niche_match", "is_weekend"]
FEATS = CAT_COLS + NUM_COLS
QUANTS = [0.1, 0.5, 0.9]
TARGETS = ["engagement_rate", "roas"]
VIDEO = ["reel", "youtube_integration", "youtube_dedicated"]


def featurize(d, cats):
    X = pd.DataFrame(index=d.index)
    X["tier"] = d["tier"].astype(str)
    X["creator_niche"] = d["creator_niche"]; X["platform"] = d["platform"]
    X["brand_category"] = d["brand_category"]; X["format"] = d["format"]
    X["hook_type"] = d["hook_type"]
    X["cta_position"] = d["cta_position"].fillna("none").replace("", "none")
    X["post_day"] = d["post_day"]
    X["log_followers"] = np.log(d["followers"].astype(float))
    X["duration_sec"] = d["duration_sec"].astype(float)
    X["post_hour"] = d["post_hour"].astype(float)
    X["creator_deals_last_30d"] = d["creator_deals_last_30d"].astype(float)
    X["log_fee"] = np.log(d["deal_value_inr"].astype(float))
    X["niche_match"] = (d["creator_niche"] == d["brand_category"]).astype(float)
    X["is_weekend"] = d["post_day"].isin(["Saturday", "Sunday"]).astype(float)
    for c in CAT_COLS:
        X[c] = pd.Categorical(X[c], categories=cats[c])
    return X[FEATS]


def make_model(q):
    return HGB(loss="quantile", quantile=q, max_iter=250, learning_rate=0.05, max_depth=4,
               min_samples_leaf=40, l2_regularization=1.0, categorical_features="from_dtype", random_state=0)


def fit_all(X, df):
    return {t: {q: make_model(q).fit(X, df[t]) for q in QUANTS} for t in TARGETS}


def predict_df(models, X):
    out = {}
    for t in TARGETS:
        P = np.column_stack([models[t][q].predict(X) for q in QUANTS])
        P = np.sort(P, axis=1)  # no crossing quantiles
        if t == "roas":
            P = np.clip(P, 0, None)
        out[t] = P
    return out


def validate(df, X, k=5):
    groups = df["creator_id"].values
    oof = {t: np.zeros((len(df), 3)) for t in TARGETS}
    base = {t: np.zeros(len(df)) for t in TARGETS}
    imp = {t: [] for t in TARGETS}
    for tr, te in GroupKFold(n_splits=k).split(X, groups=groups):
        m = fit_all(X.iloc[tr], df.iloc[tr])
        P = predict_df(m, X.iloc[te])
        for t in TARGETS:
            oof[t][te] = P[t]
            g = df.iloc[tr].groupby(["tier", "brand_category"])[t].median()
            fb = df.iloc[tr].groupby("tier")[t].median()
            keys = list(zip(df.iloc[te]["tier"], df.iloc[te]["brand_category"]))
            base[t][te] = [g.get(kk, fb[kk[0]]) for kk in keys]
            r = permutation_importance(m[t][0.5], X.iloc[te], df.iloc[te][t], n_repeats=3, random_state=0,
                                       scoring="neg_mean_absolute_error")
            imp[t].append(r.importances_mean)
    res = {}
    for t in TARGETS:
        y = df[t].values; p50 = oof[t][:, 1]
        sse = ((y - p50) ** 2).sum(); sst = ((y - y.mean()) ** 2).sum()
        mae, mae_b = np.abs(y - p50).mean(), np.abs(y - base[t]).mean()
        res[t] = dict(r2=float(1 - sse / sst), mae=float(mae), mae_baseline=float(mae_b),
                      skill_vs_baseline=float(1 - mae / mae_b),
                      spearman=float(spearmanr(y, p50)[0]),
                      coverage_p10_p90=float(((y >= oof[t][:, 0]) & (y <= oof[t][:, 2])).mean()),
                      below_p10=float((y < oof[t][:, 0]).mean()), above_p90=float((y > oof[t][:, 2]).mean()))
    imp_df = {}
    for t in TARGETS:
        v = np.mean(imp[t], axis=0)
        v = np.clip(v, 0, None); imp_df[t] = pd.Series(v / v.sum(), index=FEATS).sort_values(ascending=False)
    return res, oof, imp_df


def cqr_width(y, P, level=0.8):
    """Conformalized quantile regression: how much to widen [p10, p90] so it truly covers `level` of outcomes."""
    s = np.maximum(P[:, 0] - y, y - P[:, 2]); n = len(y)
    k = int(np.ceil((n + 1) * level))
    return float(np.sort(s)[min(k, n) - 1])


def widen(P, q, nonneg=True):
    P = P.copy(); P[:, 0] -= q; P[:, 2] += q
    if nonneg: P[:, 0] = np.clip(P[:, 0], 0, None)
    return P


def crossfit_coverage(df, oof, seed=0):
    """Calibrate the widening on one random half of creators, measure coverage on the other half (and swap)."""
    rng = np.random.default_rng(seed)
    ids = df["creator_id"].unique(); rng.shuffle(ids)
    A = df["creator_id"].isin(ids[: len(ids) // 2]).values
    out = {}
    for t in TARGETS:
        y = df[t].values; P = oof[t]; cov = []
        for tr, te in [(A, ~A), (~A, A)]:
            q = cqr_width(y[tr], P[tr]); W = widen(P[te], q)
            cov.append(((y[te] >= W[:, 0]) & (y[te] <= W[:, 2])).mean())
        out[t] = float(np.mean(cov))
    return out


def load():
    return joblib.load(f"{O}/pitch_model.joblib")


def predict_brief(art, brief):
    """brief: dict. Required: tier, creator_niche, brand_category, format, hook_type.
    Optional: platform, duration_sec, cta_position, post_hour, post_day, creator_deals_last_30d, followers, fee_inr"""
    b = dict(brief)
    t = b["tier"]
    b.setdefault("followers", art["tier_followers"][t]); b.setdefault("deal_value_inr", art["tier_fee"][t])
    if "fee_inr" in b: b["deal_value_inr"] = b.pop("fee_inr")
    b.setdefault("platform", "youtube" if b["format"].startswith("youtube") else "instagram")
    is_video = b["format"] in VIDEO
    b.setdefault("duration_sec", (30 if b["format"] == "reel" else 60) if is_video else np.nan)
    b.setdefault("cta_position", "mid" if is_video else "none")
    b.setdefault("post_hour", 20); b.setdefault("post_day", "Thursday"); b.setdefault("creator_deals_last_30d", 1)
    X = featurize(pd.DataFrame([b]), art["cats"])
    P = predict_df(art["models"], X)
    qh = art.get("qhat", {})
    if qh:
        P = {t_: widen(P[t_], qh[t_]) for t_ in TARGETS}
    return {t_: dict(p10=float(P[t_][0, 0]), p50=float(P[t_][0, 1]), p90=float(P[t_][0, 2])) for t_ in TARGETS}


def best_plays(art, tier, niche, category, fmt="reel", top=5, **kw):
    """Rank hook x CTA-position combinations for a brief. Ranked by expected ROAS (CTA position moves revenue, not
    engagement). Duration is deliberately not varied: the data shows no reliable length effect."""
    rows = []
    for hook in ["story", "question", "shock", "social_proof", "challenge", "demo", "direct_pitch"]:
        for cta in (["start", "mid", "end"] if fmt in VIDEO else ["none"]):
            br = dict(tier=tier, creator_niche=niche, brand_category=category, format=fmt, hook_type=hook,
                      cta_position=cta, **kw)
            p = predict_brief(art, br)
            rows.append(dict(hook=hook, cta=cta, eng_p10=p["engagement_rate"]["p10"], eng_p50=p["engagement_rate"]["p50"],
                             eng_p90=p["engagement_rate"]["p90"], roas_p10=p["roas"]["p10"], roas_p50=p["roas"]["p50"],
                             roas_p90=p["roas"]["p90"]))
    R = pd.DataFrame(rows).sort_values(["roas_p50", "eng_p50"], ascending=False).reset_index(drop=True)
    R["roas_vs_top"] = R.roas_p50 / R.roas_p50.iloc[0] - 1
    R["eng_vs_top"] = R.eng_p50 / R.eng_p50.max() - 1
    R["near_tie_with_top"] = (R.roas_vs_top > -0.03) & (R.eng_vs_top > -0.05)  # within 3% ROAS and 5% engagement
    return R.head(top)


if __name__ == "__main__":
    df = pd.read_csv(f"{O}/deals_clean.csv").reset_index(drop=True)
    cats = {c: sorted(df[c].fillna("none").replace("", "none").astype(str).unique()) for c in CAT_COLS}
    X = featurize(df, cats)
    res, oof, imp = validate(df, X)
    models = fit_all(X, df)
    qhat = {t: cqr_width(df[t].values, oof[t]) for t in TARGETS}
    cov_after = crossfit_coverage(df, oof)
    for t in TARGETS:
        res[t]["coverage_after_calibration_heldout"] = cov_after[t]
        res[t]["interval_widening"] = qhat[t]
    art = dict(models=models, cats=cats, feats=FEATS, qhat=qhat,
               tier_followers=df.groupby("tier").followers.median().to_dict(),
               tier_fee=df.groupby("tier").deal_value_inr.median().to_dict())
    joblib.dump(art, f"{O}/pitch_model.joblib")
    json.dump(dict(metrics=res, importance={t: imp[t].round(4).to_dict() for t in TARGETS}, n=len(df),
                   creators=int(df.creator_id.nunique())), open(f"{O}/model_results.json", "w"), indent=2)
    np.save(f"{O}/oof_eng.npy", oof["engagement_rate"]); np.save(f"{O}/oof_roas.npy", oof["roas"])
    for t in TARGETS:
        print(t, {k: round(v, 4) for k, v in res[t].items()})
        print("  top drivers:", imp[t].head(6).round(3).to_dict())
