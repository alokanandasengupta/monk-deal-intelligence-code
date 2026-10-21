"""
Brief in, pitch out.
build_pitch(art, brief) -> dict + markdown pitch for a brand.
brief keys: brand, brand_category, budget_inr, goal ("sales" | "engagement" | "reach"); optional: format (default "reel"),
            languages (list; a targeting filter, language does not drive performance), big_tier_cap (share for macro+mega)
Every number comes from the trained model or from past deals; nothing is typed in by hand.
"""
import numpy as np, pandas as pd
import pitch_model as pm
from ai_layer import STARTER, HOOK_LABEL

O = pm.O
TIERS = ["nano", "micro", "mid", "macro", "mega"]
BASE_CAPS = {"nano": 0.15, "micro": 0.35, "mid": 0.60}   # operational caps: many tiny deals are hard to run


def load_data():
    df = pd.read_csv(f"{O}/deals_clean.csv", parse_dates=["deal_date"])
    sc = pd.read_csv(f"{O}/creator_scorecard.csv")
    return df, sc


def water_fill(weights, caps):
    shares = {t: 0.0 for t in weights}; free = {t for t in weights if weights[t] > 0 and caps[t] > 0}; rem = 1.0
    while free and rem > 1e-9:
        ws = sum(weights[t] for t in free)
        alloc = {t: rem * weights[t] / ws for t in free}
        over = [t for t in free if shares[t] + alloc[t] > caps[t] + 1e-12]
        if not over:
            for t in free: shares[t] += alloc[t]
            rem = 0
        else:
            for t in over:
                rem -= caps[t] - shares[t]; shares[t] = caps[t]; free.discard(t)
    return shares


def inr(x):
    return f"Rs {x/1e7:.2f} cr" if x >= 1e7 else f"Rs {x/1e5:.1f} lakh"


def build_pitch(art, brief, df=None, sc=None):
    if df is None or sc is None: df, sc = load_data()
    cat, goal, budget, fmt = brief["brand_category"], brief.get("goal", "sales"), float(brief["budget_inr"]), brief.get("format", "reel")
    brand = brief.get("brand", "the brand")
    big_cap = brief.get("big_tier_cap", 0.5 if goal == "reach" else 0.25)
    caps = dict(BASE_CAPS, macro=0.6 * big_cap, mega=0.4 * big_cap)

    # --- creative: hooks and CTA, from the model ---
    plays = pm.best_plays(art, "mid", cat, cat, fmt=fmt, top=100)
    mid = plays[(plays.cta == "mid" if fmt in pm.VIDEO else plays.cta == "none") & (plays.hook != "direct_pitch")]
    near = mid[mid.roas_vs_top > -0.03] if len(mid) else mid
    hooks = list(near.sort_values("roas_p50", ascending=False).hook[:3]) or [plays.iloc[0].hook]
    dp = plays[(plays.cta == ("mid" if fmt in pm.VIDEO else "none")) & (plays.hook == "direct_pitch")]
    dp_gap = float(dp.roas_vs_top.iloc[0]) if len(dp) else None
    lead_hook = hooks[0]
    cta = "mid" if fmt in pm.VIDEO else "none"

    # --- per-tier expectations under the recommended creative (matched niche, creator not oversold) ---
    per = {}
    cat_df = df[df.brand_category == cat]
    for t in TIERS:
        p = pm.predict_brief(art, dict(tier=t, creator_niche=cat, brand_category=cat, format=fmt, hook_type=lead_hook, cta_position=cta,
                                       creator_deals_last_30d=1))
        sub = cat_df[cat_df.tier == t]; sub = sub if len(sub) >= 30 else df[df.tier == t]
        per[t] = dict(roas=p["roas"], eng=p["engagement_rate"], fee=float(art["tier_fee"][t]),
                      views_per_inr=float((sub.views / sub.deal_value_inr).median()), eng_per_inr=float((sub.engagements / sub.deal_value_inr).median()))
    # --- candidate creators: niche match, enough history, not booked up ---
    maxd = df.deal_date.max()
    recent = df[df.deal_date > maxd - pd.Timedelta(days=30)].groupby("creator_id").size()
    fee = df.groupby("creator_id").deal_value_inr.median().rename("typical_fee")
    c = sc[(sc.creator_niche == cat) & (sc.deals >= 4) & (~sc.saturated_flag)].copy()
    c["booked_last_30d"] = c.creator_id.map(recent).fillna(0).astype(int)
    c = c[c.booked_last_30d < 3].merge(fee, on="creator_id")
    if brief.get("languages"): c = c[c.creator_language.isin(brief["languages"])]
    if goal == "sales": c = c[c.roas >= 1.0]
    c = c.sort_values("value_score", ascending=False)
    have = set(c.tier.astype(str))

    # --- budget split ---
    if goal == "sales": w = {t: per[t]["roas"]["p50"] ** 2 for t in TIERS}
    elif goal == "engagement": w = {t: per[t]["eng_per_inr"] ** 2 for t in TIERS}
    else: w = {t: per[t]["views_per_inr"] ** 2 for t in TIERS}
    w = {t: (w[t] if t in have else 0.0) for t in TIERS}
    if sum(w.values()) == 0: raise ValueError("No eligible creators in this category and filters.")
    shares = water_fill(w, caps)
    target = {t: shares[t] * budget for t in TIERS}

    # --- pick creators within each tier's target; second pass reuses leftover without breaking caps ---
    picked, spent = [], {t: 0.0 for t in TIERS}
    def take(t, limit):
        for _, r in c[c.tier.astype(str) == t].iterrows():
            if r.creator_id in {p["creator_id"] for p in picked}: continue
            if spent[t] + r.typical_fee > limit: continue
            picked.append(dict(creator_id=r.creator_id, creator=r.creator_name, handle=r.creator_handle, language=r.creator_language,
                               city=r.creator_city, tier=t, past_roas=float(r.roas), past_deals=int(r.deals), fee=float(r.typical_fee)))
            spent[t] += r.typical_fee
    for t in sorted(TIERS, key=lambda x: -shares[x]): take(t, target[t])
    for _ in range(2):
        left = budget - sum(spent.values())
        if left < min([per[t]["fee"] for t in TIERS if t in have] or [0]) * 0.5: break
        for t in sorted(TIERS, key=lambda x: -w[x]):
            if w[t] > 0: take(t, min(caps[t] * budget, spent[t] + left))
    # enforce the macro+mega cap on the FINAL selection (share of what is actually spent), dropping the weakest big-tier creators first
    trimmed = 0
    while picked:
        tot = sum(p["fee"] for p in picked); big = sum(p["fee"] for p in picked if p["tier"] in ("macro", "mega"))
        if big <= big_cap * tot + 1e-6: break
        cand = [p for p in picked if p["tier"] in ("macro", "mega")]
        drop = min(cand, key=lambda p: p["past_roas"]); picked.remove(drop); spent[drop["tier"]] -= drop["fee"]; trimmed += 1
    P = pd.DataFrame(picked)
    total_spent = float(P.fee.sum()) if len(P) else 0.0
    real_share = {t: spent[t] / total_spent if total_spent else 0 for t in TIERS}

    # --- outcome: model estimate, same creative, recommended mix vs Monk's historical mix in this category ---
    hist_share = (cat_df.groupby("tier").deal_value_inr.sum() / cat_df.deal_value_inr.sum()).reindex(TIERS).fillna(0).to_dict()
    blend = lambda sh, key: sum(sh[t] * per[t][key]["p50"] for t in TIERS)
    rec_roas, base_roas = blend(real_share, "roas"), blend(hist_share, "roas")
    worst = pm.predict_brief  # creative-off scenario: direct pitch, CTA at the end, same mix
    off = {t: pm.predict_brief(art, dict(tier=t, creator_niche=cat, brand_category=cat, format=fmt, hook_type="direct_pitch",
                                         cta_position="end" if fmt in pm.VIDEO else "none", creator_deals_last_30d=1))["roas"]["p50"] for t in TIERS}
    rec_off = sum(real_share[t] * off[t] for t in TIERS)
    out = dict(brief=brief, hooks=hooks, lead_hook=lead_hook, shares=real_share, target_shares=shares, per_tier=per, creators=P,
               total_spent=total_spent, unallocated=budget - total_spent, rec_roas_p50=rec_roas, hist_mix_roas_p50=base_roas,
               rec_mix_off_creative_roas_p50=rec_off, hist_share=hist_share, caps=caps, n_candidates=len(c))

    # --- markdown ---
    L = []
    L += [f"# Proposed plan for {brand}", "",
          f"**Objective:** {dict(sales='sales (return on spend)', engagement='engagement per rupee', reach='views per rupee')[goal]}  |  "
          f"**Category:** {cat}  |  **Budget:** {inr(budget)}  |  **Format:** {fmt}", "",
          "## Recommended budget split", "", "| Tier | Share | Spend | Creators | Typical ROAS (median deal) | 80% range |", "|---|---|---|---|---|---|"]
    for t in TIERS:
        n = int((P.tier == t).sum()) if len(P) else 0
        if real_share[t] > 0:
            L.append(f"| {t} | {real_share[t]*100:.0f}% | {inr(spent[t])} | {n} | {per[t]['roas']['p50']:.2f} | {per[t]['roas']['p10']:.2f} to {per[t]['roas']['p90']:.2f} |")
    L += ["", f"Macro and mega are capped at {big_cap*100:.0f}% combined because they return less per rupee in the past data; nano is capped at "
              f"{caps['nano']*100:.0f}% and micro at {caps['micro']*100:.0f}% because very many small deals are hard to run.", ""]
    if out["unallocated"] > budget * 0.03:
        L += [f"**Note:** {inr(out['unallocated'])} ({out['unallocated']/budget*100:.0f}%) is left unallocated. There are only {len(c)} eligible creators "
              f"in {cat} (niche match, 4+ past deals, not booked up), each used once, and the macro/mega cap stops the big tiers absorbing the rest. "
              "Options: widen languages, add creators, allow repeat bookings, or accept a smaller campaign.", ""]
    L += ["## Creator shortlist", "", "Matched to the category, not currently booked up, ranked by past value for money.", "",
          "| Creator | Tier | Language | City | Past ROAS | Past deals | Typical fee |", "|---|---|---|---|---|---|---|"]
    for r in P.sort_values(["tier", "past_roas"], ascending=[True, False]).head(15).itertuples():
        L.append(f"| {r.creator} ({r.handle}) | {r.tier} | {r.language} | {r.city} | {r.past_roas:.2f} | {r.past_deals} | {inr(r.fee)} |")
    if len(P) > 15: L.append(f"\n{len(P)-15} more creators in the full list.")
    L += ["", "## Creative guidance", "",
          f"- **Open with one of:** {', '.join(HOOK_LABEL[h] for h in hooks)}. The predicted difference among these is small, so test two of them.",
          (f"- **Do not open with a plain sponsor introduction** (\"this video is sponsored by...\"). The model puts it {abs(dp_gap)*100:.0f}% below the best hook in {cat}."
           if dp_gap is not None and dp_gap < -0.05 else "- Avoid a plain sponsor introduction as the opening line."),
          f"- **Put the call to action in the middle of the video**, not the end." if fmt in pm.VIDEO else "- Keep the call to action clear and singular.",
          "- **Book each creator once in this window** and avoid creators with 3 or more deals in the past 30 days.", "",
          "Starter opening lines to adapt (the creator must only keep claims that are true for them):", ""]
    for h in hooks[:3]:
        L.append(f"- {HOOK_LABEL[h]}: \"{STARTER['english'][h][0].format(b=brand, c=cat)}\"")
    L += ["", "## What to expect (model estimate)", "",
          f"- Typical deal ROAS with this mix and creative: **{rec_roas:.2f}** (median outcome; the average is usually higher because a few deals do very well).",
          f"- The same creative on Monk's historical tier mix for {cat}: {base_roas:.2f}. The mix change is worth {(rec_roas/base_roas-1)*100:+.0f}%.",
          f"- This mix with a plain sponsor opening and the CTA at the end: {rec_off:.2f}. The creative changes are worth {(rec_roas/rec_off-1)*100:+.0f}%.",
          f"- Rough revenue at the median: {inr(total_spent * rec_roas)} on {inr(total_spent)} spend.", "",
          "These come from a model trained on simulated data. Real results will differ; the CTA effect in particular is likely smaller in real campaigns. "
          "Treat this as a starting plan and measure it."]
    out["markdown"] = "\n".join(L)
    return out
