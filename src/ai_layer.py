"""
Monk AI layer: script tagging, script scoring/rewrite, and brief-to-pitch.

Tagger backends
  - "llm":   Claude via the Anthropic API. Needs ANTHROPIC_API_KEY and `pip install anthropic`. Handles messy, mixed-language scripts.
  - "rules": offline keyword rules. Fast and free, weaker on unusual wording. Used when no key is set.
"""
import os, re, json
import numpy as np, pandas as pd
import pitch_model as pm

O = pm.O
HOOKS = ["story", "question", "shock", "social_proof", "challenge", "demo", "direct_pitch"]

# ---------------------------------------------------------------- rule-based tagger
CTA_RE = re.compile(r"(link (is )?(in|on|below|niche|description)|link bio|bio mein|in my bio|use (my )?(code|link)|code [A-Z0-9]{3,}|"
                    r"\bcode\b.*\b(use|karo)\b|tap the link|tap (the )?link|grab (it|yours|a pair|the|via)|order (now|karo|from|karne)|"
                    r"shop now|swipe up|download (it|the app)|check (the )?link|details in the link|"
                    r"link mein|link description|use my link|discount ke liye|check kar lo|join karo|jaldi lo|products linked|before it sells out)", re.I)
HOOK_PATTERNS = [  # order = priority (most specific first; question last because many hooks contain a '?')
    ("direct_pitch", r"(sponsored by|in collaboration with|paid partnership|partnering with|today i('m| am) (sharing|talking|going to tell|introducing)|"
                     r"i want to tell you about|hi guys|hello everyone|hello dosto|namaste dosto|aaj main aapko .{0,40}(batane|bataunga)|"
                     r"aaj ki video .{0,30} ke saath|this video is)"),
    ("challenge", r"(challenge|day (one|1|\d+)\b|for (seven|7|\d+|one|a) (days|week|weeks)|(seven|7|\d+) din tak|din tak sirf|try this with me|no cheating|"
                  r"i (only )?used only|i only used|i only (drink|ate|wore)|thirty days)"),
    ("demo", r"(watch (how|me)|let me show|here('s| is) (how|the full|my)|step by step|start to finish|tutorial|how i (use|do|set)|dekho kaise|"
             r"dikhata hoon|dikhati hoon|kaise banta)"),
    ("social_proof", r"(everyone (in|is|keeps)|everybody|all my (friends|followers)|half my friends|[\d,]+ of you|ten thousand of you|you guys asked|"
                     r"you asked|my dms|mere dms|dms mein|sold out|sab log|sabne|thousands of reviews|trending|viral|best.?selling|asking about)"),
    ("shock", r"(stop scrolling|not a typo|can'?t believe|cannot believe|did not expect|didn'?t expect|unbelievable|insane|unreal|crazy|you won'?t believe|"
              r"shocking|believe nahi|dimaag ghoom|not a drill|wait,|bhai ye kya|ye kya hai)"),
    ("story", r"(last (month|year|week|summer|winter)|years? ago|i was this close|i almost|one day|when i |i remember|story time|i used to|"
              r"pichle|pehle|kuch saal pehle|ek baar|us din|i couldn'?t|i was about to)"),
    ("question", r"(\?|^(why|what|how|is|are|do|did|can|ever wondered|kya|kyun|kaise|kab)\b)"),
]
HINGLISH_WORDS = set("hai hain kya nahi aap dosto yaar bhai mast ekdum bahut bohot karo dekho toh mein ke liye pehle phir maine aaj ye wala wali sab log".split())


def split_sentences(text):
    parts = re.split(r"(?<=[.!?])\s+|\n+", text.strip())
    return [p.strip() for p in parts if p.strip()]


def tag_rules(script):
    sents = split_sentences(script)
    n = len(sents)
    cta_idx = [i for i, s in enumerate(sents) if CTA_RE.search(s)]
    # hook = first sentence that is not a CTA line
    hook_sent = next((s for i, s in enumerate(sents) if i not in cta_idx), sents[0] if sents else "")
    hook_type = "unclear"
    for name, pat in HOOK_PATTERNS:
        if re.search(pat, hook_sent, re.I):
            hook_type = name; break
    if hook_type == "unclear":  # allow a match anywhere in the first two non-CTA sentences
        early = " ".join([s for i, s in enumerate(sents) if i not in cta_idx][:2])
        for name, pat in HOOK_PATTERNS[:-1]:
            if re.search(pat, early, re.I):
                hook_type = name; break
    if cta_idx:  # position = share of the script's words spoken before the first CTA sentence (a proxy for timing)
        wc = [len(re.findall(r"\w+", x)) for x in sents]
        frac = sum(wc[:cta_idx[0]]) / max(sum(wc), 1)
        if cta_idx[0] == n - 1 and n > 1:  # the CTA closes the script, however long the sentence is
            cta_pos = "end"
        else:
            cta_pos = "start" if frac < 0.15 else ("mid" if frac < 0.55 else "end")
    else:
        cta_pos = "none"
    words = re.findall(r"\w+", script.lower())
    hing = sum(w in HINGLISH_WORDS for w in words)
    return dict(hook_type=hook_type, hook_line=hook_sent, cta_position=cta_pos,
                cta_line=sents[cta_idx[0]] if cta_idx else "", language_style="hinglish" if hing >= 2 else "english",
                energy="high" if script.count("!") >= 2 else ("medium" if script.count("!") == 1 else "calm"),
                est_duration_sec=int(round(len(words) / 2.5)), confidence="rules", backend="rules")


# ---------------------------------------------------------------- LLM tagger (needs a key)
LLM_PROMPT = """You label influencer video scripts for a marketing agency. Return ONLY JSON with keys:
hook_type: one of story, question, shock, social_proof, challenge, demo, direct_pitch
  (direct_pitch = plain introduction of the brand or sponsor, e.g. "this video is sponsored by", "hi guys today I'm sharing")
hook_line: the exact opening line the viewer hears first (ignore a leading discount-code line)
cta_position: where the first call to action sits: start, mid, end, or none
cta_line: the call-to-action sentence, or ""
language_style: english, hinglish, or hindi
energy: calm, medium, or high
The script may mix English and Hindi in Roman script.

SCRIPT:
{script}"""


def tag_llm(script, model="claude-sonnet-5-5"):
    import anthropic  # pip install anthropic ; needs ANTHROPIC_API_KEY
    client = anthropic.Anthropic()
    msg = client.messages.create(model=model, max_tokens=400, messages=[{"role": "user", "content": LLM_PROMPT.format(script=script)}])
    d = parse_llm_json(msg.content[0].text)
    words = re.findall(r"\w+", script.lower())
    d.update(est_duration_sec=int(round(len(words) / 2.5)), confidence="llm", backend="llm")
    return d


def parse_llm_json(txt):
    """Pull the JSON object out of an LLM reply and sanity-check the labels."""
    d = json.loads(re.search(r"\{.*\}", txt, re.S).group(0))
    if d.get("hook_type") not in HOOKS: d["hook_type"] = "unclear"
    if d.get("cta_position") not in ("start", "mid", "end", "none"): d["cta_position"] = "none"
    d.setdefault("hook_line", ""); d.setdefault("cta_line", "")
    return d


def tag_script(script, backend="auto"):
    if backend == "llm" or (backend == "auto" and os.environ.get("ANTHROPIC_API_KEY")):
        try:
            return tag_llm(script)
        except Exception as e:  # fall back, but say so
            t = tag_rules(script); t["note"] = f"LLM tagging failed ({type(e).__name__}); used offline rules"; return t
    return tag_rules(script)


# ---------------------------------------------------------------- script scorer + rewrite
STARTER = {  # generic opening lines per hook type; {b}=brand, {c}=category. Placeholders to adapt, not finished copy.
    "english": {
        "story": ["Last month I almost gave up on {c}, and then {b} changed that.", "Three years ago I would never have tried {b}, but here we are."],
        "question": ["Is {b} actually worth your money? I tested it for two weeks.", "Why does nobody talk about this {c} trick?"],
        "shock": ["I did not expect {b} to work this well.", "Stop scrolling, this price is not a typo."],
        "social_proof": ["Everyone in my DMs keeps asking about {b}, so here it is.", "Over 10,000 of you asked me to review this."],
        "challenge": ["I used only {b} for seven days. Here is what happened.", "Day one of my {b} challenge, no cheating."],
        "demo": ["Watch how fast this comes together with {b}.", "Let me show you exactly how I use {b} every day."]},
    "hinglish": {
        "story": ["Pichle mahine main {c} se pareshan thi, phir {b} try kiya.", "Kuch saal pehle main {b} kabhi try nahi karti, par ab dekho."],
        "question": ["Kya {b} sach mein paisa vasool hai? Maine do hafte test kiya.", "{c} ka ye trick koi kyun nahi batata?"],
        "shock": ["Mujhe nahi laga tha {b} itna accha nikalega.", "Ruko, ye price typo nahi hai."],
        "social_proof": ["Mere DMs mein sab {b} ke baare mein pooch rahe the, toh ye raha review.", "Aap mein se 10,000 logon ne ye review maanga tha."],
        "challenge": ["Maine 7 din sirf {b} use kiya. Dekho kya hua.", "Day one of my {b} challenge, no cheating."],
        "demo": ["Dekho kaise main roz {b} use karti hoon.", "Chalo main dikhati hoon ki {b} kitni jaldi ho jata hai."]}}

HOOK_LABEL = {"story": "story", "question": "question", "shock": "shock", "social_proof": "social proof",
              "challenge": "challenge", "demo": "demo", "direct_pitch": "direct pitch"}


def _pct(a, b):
    return float(a / b - 1) if b else 0.0


def rewrite_offline(script, tags, rec_hook, brief):
    """Mechanical rewrite: swap in a starter hook for the recommended type and move the CTA to the middle. Keeps the body."""
    sents = split_sentences(script)
    cta_idx = [i for i, s in enumerate(sents) if CTA_RE.search(s)]
    cta_block = [sents[i] for i in cta_idx] or ["[Add a call to action: link or code]"]
    hook_i = next((i for i in range(len(sents)) if i not in cta_idx), 0)
    body = [s for i, s in enumerate(sents) if i not in cta_idx and i != hook_i]
    style = "hinglish" if tags.get("language_style") == "hinglish" else "english"
    brand = brief.get("brand", "this")
    line = STARTER[style][rec_hook][0].format(b=brand, c=brief.get("brand_category", "this category"))
    k = (len(body) + 1) // 2  # CTA lands after roughly half of the body
    return " ".join([line] + body[:k] + cta_block + body[k:])


def rewrite_llm(script, tags, rec_hook, brief):
    """LLM rewrite that keeps facts and tone but applies the data-backed structure. Returns None without an API key."""
    if not os.environ.get("ANTHROPIC_API_KEY"):
        return None
    import anthropic
    prompt = (f"Rewrite this influencer video script for the brand {brief.get('brand', 'the brand')} ({brief.get('brand_category', '')}).\n"
              f"Rules: open with a {HOOK_LABEL[rec_hook]} hook (not a plain sponsor introduction); put the call to action in the MIDDLE of the script, "
              f"not the end; keep every factual claim, price and code unchanged; do NOT invent usage claims, results or personal experiences that the original does not support (where the new hook needs a personal experience, write [CREATOR: add your real experience] instead); keep the creator's voice and the same language mix "
              f"({tags.get('language_style', 'english')}); keep roughly the same length. Return only the rewritten script.\n\nSCRIPT:\n{script}")
    msg = anthropic.Anthropic().messages.create(model="claude-sonnet-5-5", max_tokens=700, messages=[{"role": "user", "content": prompt}])
    return msg.content[0].text.strip()


def score_script(art, script, brief, hook_override=None, cta_override=None, backend="auto"):
    """brief needs: tier, creator_niche, brand_category. Optional: brand, format (default reel), creator_deals_last_30d, fee_inr."""
    tags = tag_script(script, backend)
    if hook_override: tags["hook_type"], tags["confidence"] = hook_override, "user"
    if cta_override: tags["cta_position"] = cta_override
    if tags["hook_type"] == "unclear":
        return dict(tags=tags, needs_input=True,
                    message="Could not tell what kind of hook this opens with. Pick one (story, question, shock, social_proof, challenge, demo, direct_pitch) "
                            "and call again with hook_override, or use the LLM tagger.")
    b = {k: v for k, v in brief.items() if k != "brand"}
    b.setdefault("format", "reel")
    plays = pm.best_plays(art, b["tier"], b["creator_niche"], b["brand_category"], fmt=b["format"], top=100,
                          **{k: v for k, v in b.items() if k in ("creator_deals_last_30d", "fee_inr")})
    video = b["format"] in pm.VIDEO  # image formats have no CTA position: every option is cta="none"
    mid = "mid" if video else "none"
    cta = (tags["cta_position"] if tags["cta_position"] != "none" else "end") if video else "none"
    cur = plays[(plays.hook == tags["hook_type"]) & (plays.cta == cta)].iloc[0]
    rank = int(cur.name) + 1; n = len(plays)
    score = int(round(100 * (1 - (rank - 1) / (n - 1))))
    top = plays.iloc[0]
    good_hooks = [h for h in plays[(plays.cta == mid) & (plays.hook != "direct_pitch") & (plays.roas_vs_top > -0.03)].hook]
    _mid = plays[(plays.cta == mid) & (plays.hook.isin(good_hooks))]
    rec_hook = tags["hook_type"] if (tags["hook_type"] != "direct_pitch" and tags["hook_type"] in good_hooks) else \
        (_mid.sort_values("eng_p50", ascending=False).iloc[0].hook if len(_mid) else top.hook)
    best = plays[(plays.hook == rec_hook) & (plays.cta == mid)].iloc[0]
    recs = []
    if tags["hook_type"] == "direct_pitch":
        alt = plays[(plays.hook == rec_hook) & (plays.cta == cta)].iloc[0]
        recs.append(dict(change=f"Open with a {HOOK_LABEL[rec_hook]} hook instead of a plain brand introduction",
                         engagement_effect=_pct(alt.eng_p50, cur.eng_p50), roas_effect=_pct(alt.roas_p50, cur.roas_p50)))
    if not video:
        pass
    elif tags["cta_position"] == "none":
        recs.append(dict(change="No call to action found: add one in the middle of the video", engagement_effect=0.0,
                         roas_effect=_pct(plays[(plays.hook == tags["hook_type"]) & (plays.cta == "mid")].iloc[0].roas_p50, cur.roas_p50)))
    elif cta != "mid":
        alt = plays[(plays.hook == tags["hook_type"]) & (plays.cta == "mid")].iloc[0]
        recs.append(dict(change=f"Move the call to action from the {cta} to the middle", engagement_effect=_pct(alt.eng_p50, cur.eng_p50),
                         roas_effect=_pct(alt.roas_p50, cur.roas_p50)))
    sat = int(b.get("creator_deals_last_30d", 1))
    if sat >= 3:
        one = pm.predict_brief(art, dict(b, hook_type=tags["hook_type"], cta_position=cta, creator_deals_last_30d=1))
        now = pm.predict_brief(art, dict(b, hook_type=tags["hook_type"], cta_position=cta))
        recs.append(dict(change=f"This creator already has {sat} deals in the last 30 days: consider a less-booked creator or wait",
                         engagement_effect=_pct(one["engagement_rate"]["p50"], now["engagement_rate"]["p50"]),
                         roas_effect=_pct(one["roas"]["p50"], now["roas"]["p50"]), note="effect shown is the gain from using a creator with 1 recent deal"))
    return dict(
        tags=tags, needs_input=False, score=score, rank=rank, n_options=n,
        score_meaning=f"Where this hook and CTA placement rank among the {n} combinations for this brief (100 = best predicted ROAS, 0 = worst).",
        current=dict(eng=[cur.eng_p10, cur.eng_p50, cur.eng_p90], roas=[cur.roas_p10, cur.roas_p50, cur.roas_p90]),
        recommended=dict(hook=rec_hook, cta=mid, eng=[best.eng_p10, best.eng_p50, best.eng_p90], roas=[best.roas_p10, best.roas_p50, best.roas_p90],
                         roas_uplift=_pct(best.roas_p50, cur.roas_p50), eng_uplift=_pct(best.eng_p50, cur.eng_p50),
                         near_tied_hooks=good_hooks),
        recommendations=recs, rewrite_offline=rewrite_offline(script, tags, rec_hook, brief), rewrite_llm=rewrite_llm(script, tags, rec_hook, brief),
        rewrite_note="The rewrite uses template hook lines. Any personal experience or result in the new hook must be true for the creator and the product; replace or delete it otherwise.",
        caveat="Ranges are 80% intervals from a model trained on simulated data. ROAS ranges are wide by nature.")
