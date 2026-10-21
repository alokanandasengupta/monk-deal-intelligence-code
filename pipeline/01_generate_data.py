import os, sys
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(ROOT); sys.path.insert(0, os.path.join(ROOT, "src"))  # run from anywhere; paths are relative to the repo root
"""
Monk Deal Intelligence: simulated deal data generator.
All data is SIMULATED. Brand and creator names are fictional.
Hidden patterns planted (so analysis can rediscover them):
  1. Hook x category fit (story wins in beauty, question in tech, shock in gaming,
     social_proof in fashion, challenge in fitness, demo in food). direct_pitch loses everywhere.
  2. Ideal video duration differs by category (fashion short, tech long).
  3. Creator saturation: 3+ deals in the prior 30 days decays performance.
  4. Macro/mega tier is poor value in beauty and food; mid tier is best value.
  5. CTA placed mid-video beats end for CTR and conversions.
  6. Post timing: 7-10pm best; weekends help fashion/travel/lifestyle, weekdays help tech.
  7. Creator-category mismatch lowers performance.
"""
import numpy as np
import pandas as pd

rng = np.random.default_rng(42)
N_CREATORS, N_DEALS = 700, 7500
START, END = pd.Timestamp("2024-10-01"), pd.Timestamp("2026-09-30")

NICHES = ["beauty", "fashion", "food", "tech", "fitness", "travel", "gaming", "lifestyle"]
TIERS = ["nano", "micro", "mid", "macro", "mega"]
TIER_W = [0.15, 0.30, 0.30, 0.18, 0.07]
FOLLOWERS = {"nano": (5e3, 2e4), "micro": (2e4, 1e5), "mid": (1e5, 5e5),
             "macro": (5e5, 2e6), "mega": (2e6, 8e6)}
FEE = {"nano": 3000, "micro": 10000, "mid": 32000, "macro": 110000, "mega": 380000}
REACH = {"nano": 0.45, "micro": 0.35, "mid": 0.24, "macro": 0.12, "mega": 0.07}
ENG = {"nano": 0.060, "micro": 0.046, "mid": 0.034, "macro": 0.021, "mega": 0.013}
NICHE_FEE = dict(beauty=1.0, fashion=1.1, food=0.85, tech=1.25, fitness=0.95,
                 travel=1.15, gaming=1.0, lifestyle=0.9)
AOV = dict(beauty=650, fashion=1400, food=450, tech=2200, fitness=900,
           travel=3500, gaming=600, lifestyle=800)
NICHE_CVR = dict(beauty=1.1, fashion=1.0, food=1.2, tech=0.7, fitness=0.9,
                 travel=0.5, gaming=0.8, lifestyle=1.0)
SAVE_SHARE = dict(beauty=0.14, fashion=0.14, food=0.16, tech=0.06, fitness=0.10,
                  travel=0.12, gaming=0.04, lifestyle=0.08)

HOOKS = ["story", "question", "shock", "social_proof", "challenge", "demo", "direct_pitch"]
HOOK_W = [0.14, 0.14, 0.12, 0.13, 0.10, 0.13, 0.24]
HOOK_BEST = dict(beauty="story", tech="question", gaming="shock", fashion="social_proof",
                 fitness="challenge", food="demo")
IDEAL_DUR = dict(beauty=(32, 14), fashion=(20, 10), food=(24, 10), tech=(50, 20),
                 fitness=(30, 12), travel=(28, 12), gaming=(40, 15), lifestyle=(30, 12))

BRANDS = {
    "beauty": ["Glowvera", "Petalpure", "Kohl&Co", "Lumeskin", "Rosewater Labs"],
    "fashion": ["Threadhaus", "Kurta Lane", "Denimwala", "Saree Studio", "Urbanloom"],
    "food": ["Chaiwala Co", "Masala Box", "Snackbar 91", "Ghee Gold", "Brewtale"],
    "tech": ["Pixelon", "Boltbuds", "Gizmoo", "Zenwatch", "Nexa Mobile"],
    "fitness": ["Flexfuel", "Repsy", "Yogaroots", "Sprintly", "ProteinPad"],
    "travel": ["Wanderkart", "Trekora", "StayNest", "Hopon Tours", "Jetly"],
    "gaming": ["Gamezilla", "Quest Cafe", "Pixel Pass", "Rage Rigs", "LootBox India"],
    "lifestyle": ["Homely", "Candlecraft", "Plantify", "Nestora", "Daily Decor"],
}


POOLS = {
 "Hindi": dict(w=0.44,
   first=["Aarohi","Kritika","Neha","Riya","Shruti","Pooja","Tanvi","Muskan","Pragya","Divya","Sakshi","Nidhi","Aanchal","Ritika","Komal","Isha","Khushi","Aman","Rohit","Karan","Vikas","Ankit","Harsh","Mohit","Shubham","Yash","Abhishek","Rahul","Nitin","Saurabh","Tushar","Kunal","Gaurav","Vivek","Piyush","Deepak"],
   last=["Sharma","Verma","Gupta","Singh","Yadav","Mishra","Tiwari","Pandey","Agarwal","Jain","Chauhan","Saxena","Kapoor","Malhotra","Arora","Bansal","Mehra","Rastogi","Srivastava","Joshi"],
   cities=["Delhi","Mumbai","Lucknow","Jaipur","Noida","Gurugram","Bhopal","Indore","Patna","Kanpur","Dehradun"]),
 "Punjabi": dict(w=0.05,
   first=["Simran","Gurleen","Harleen","Manpreet","Jasleen","Navjot","Amrit","Harpreet","Gagan","Jashan","Sukhdeep","Arshdeep","Taranjeet","Ishmeet","Kiranjot","Mehar"],
   last=["Kaur","Singh","Gill","Sandhu","Dhillon","Bajwa","Grewal","Sidhu","Brar","Randhawa","Bedi","Chadha"],
   cities=["Chandigarh","Ludhiana","Amritsar","Jalandhar","Mohali"]),
 "Bengali": dict(w=0.06,
   first=["Ishani","Sreeja","Aditi","Rituparna","Tanushree","Debolina","Madhurima","Sayantani","Arnab","Soumya","Subhajit","Anirban","Dipankar","Abhijit","Sourav","Koushik","Ritwik","Aritra","Priyanka"],
   last=["Banerjee","Chatterjee","Mukherjee","Ghosh","Das","Sen","Bose","Dutta","Roy","Chakraborty","Bhattacharya","Majumdar","Saha"],
   cities=["Kolkata","Siliguri","Durgapur","Howrah"]),
 "Tamil": dict(w=0.09,
   first=["Meenakshi","Kaviya","Harini","Dhanya","Janani","Swetha","Priyadharshini","Lakshmi","Nandhini","Vignesh","Karthik","Arun","Sriram","Prasanth","Dinesh","Surya","Naveen","Aravind"],
   last=["Iyer","Iyengar","Krishnan","Subramanian","Raman","Natarajan","Pillai","Murugan","Rajan","Venkatesh","Balasubramanian","Chandran"],
   cities=["Chennai","Coimbatore","Madurai","Trichy"]),
 "Telugu": dict(w=0.10,
   first=["Sravya","Pranathi","Harika","Lasya","Sirisha","Anusha","Bhavana","Teja","Saikiran","Charan","Venkat","Pavan","Naresh","Vamsi","Rakesh","Sandeep","Manoj"],
   last=["Reddy","Naidu","Rao","Chowdary","Varma","Goud","Raju","Prasad","Kumar","Yadav"],
   cities=["Hyderabad","Vijayawada","Visakhapatnam","Guntur"]),
 "Kannada": dict(w=0.07,
   first=["Chaitra","Spandana","Meghana","Ramya","Rakshita","Nikhil","Darshan","Varun","Prajwal","Sharath","Harsha","Kiran","Vinay","Ananya","Pooja"],
   last=["Gowda","Shetty","Hegde","Bhat","Rao","Kulkarni","Naik","Patil","Acharya","Kamath"],
   cities=["Bengaluru","Mysuru","Mangaluru","Hubballi"]),
 "Malayalam": dict(w=0.05,
   first=["Anjana","Athira","Nimisha","Gayathri","Meera","Neethu","Aparna","Arjun","Akhil","Anoop","Jithin","Sreejith","Vishnu","Midhun","Nikhil"],
   last=["Nair","Menon","Pillai","Kurup","Varghese","Thomas","Mathew","Joseph","Warrier","Panicker"],
   cities=["Kochi","Thiruvananthapuram","Kozhikode","Thrissur"]),
 "Marathi": dict(w=0.08,
   first=["Sayali","Rutuja","Pranjal","Mrunal","Shruti","Aishwarya","Snehal","Omkar","Prathamesh","Swapnil","Akshay","Tejas","Ninad","Siddhesh","Sarang"],
   last=["Patil","Deshmukh","Kulkarni","Joshi","Jadhav","Pawar","Shinde","Kadam","Gokhale","Bhosale","Deshpande","Naik"],
   cities=["Pune","Mumbai","Nagpur","Nashik","Thane"]),
 "Gujarati": dict(w=0.06,
   first=["Dhruvi","Hetal","Krupa","Mansi","Riddhi","Jinal","Urvi","Jay","Dhaval","Parth","Harsh","Darshan","Chirag","Meet","Smit"],
   last=["Patel","Shah","Mehta","Desai","Trivedi","Joshi","Parekh","Vyas","Modi","Thakkar","Gandhi","Bhatt"],
   cities=["Ahmedabad","Surat","Vadodara","Rajkot"]),
}
METROS = {"Delhi","Mumbai","Bengaluru","Hyderabad","Chennai","Kolkata","Pune","Gurugram","Noida"}
SUFFIX = dict(beauty=["beauty","glam","makeup","skin"], fashion=["styles","wardrobe","ootd","fits"],
              food=["eats","kitchen","foodie","bites"], tech=["tech","gadgets","techtalks","reviews"],
              fitness=["fit","fitness","gymlife","wellness"], travel=["travels","wanders","explores","trips"],
              gaming=["gaming","plays","gamer","live"], lifestyle=["vlogs","life","daily","diaries"])
_lang_names = list(POOLS)
_lang_w = np.array([POOLS[l]["w"] for l in _lang_names]); _lang_w = _lang_w / _lang_w.sum()

def make_creator_identity(niche, used_names, used_handles):
    while True:
        lang_region = str(rng.choice(_lang_names, p=_lang_w))
        P = POOLS[lang_region]
        first, last = str(rng.choice(P["first"])), str(rng.choice(P["last"]))
        name = f"{first} {last}"
        if name in used_names:
            continue
        city = str(rng.choice(P["cities"]))
        suf = str(rng.choice(SUFFIX[niche]))
        style = int(rng.integers(0, 5))
        base = {0: f"{first}{last}", 1: f"{first}.{suf}", 2: f"the{first}{suf}", 3: f"{first}_{last}", 4: f"{first}{suf}"}[style].lower()
        if rng.random() < 0.3:
            base += str(int(rng.integers(1, 99)))
        handle = "@" + base
        if handle in used_handles:
            continue
        language = lang_region
        if city in METROS and rng.random() < 0.18:
            language = "English"
        elif lang_region == "Hindi" and city in METROS and rng.random() < 0.5:
            language = "Hinglish"
        used_names.add(name); used_handles.add(handle)
        return name, handle, language, city

# ---------- creators ----------
creators = []
heavy_ids = set(rng.choice(N_CREATORS, 100, replace=False))
used, used_h = set(), set()
for i in range(N_CREATORS):
    niche = str(rng.choice(NICHES))
    nm, handle, language, city = make_creator_identity(niche, used, used_h)
    tier = rng.choice(TIERS, p=TIER_W)
    lo, hi = FOLLOWERS[tier]
    followers = int(np.exp(rng.uniform(np.log(lo), np.log(hi))))
    creators.append(dict(
        creator_id=f"C{i+1:03d}", creator_name=nm, handle=handle, language=language, city=city,
        tier=tier, followers=followers,
        niche=niche, platform=rng.choice(["instagram", "youtube"], p=[0.7, 0.3]),
        quality=float(np.exp(rng.normal(0, 0.25))), heavy=i in heavy_ids))
cdf = pd.DataFrame(creators)

# ---------- deal skeleton (creator, date) ----------
span = (END - START).days
w = np.where(cdf["heavy"], 4.0, 1.0)
w = w / w.sum()
bursts = {c: START + pd.Timedelta(days=int(rng.integers(20, span - 20))) for c in cdf.index if cdf.loc[c, "heavy"]}
bursts2 = {c: START + pd.Timedelta(days=int(rng.integers(20, span - 20))) for c in bursts}
rows = []
for _ in range(N_DEALS):
    ci = int(rng.choice(len(cdf), p=w))
    if cdf.loc[ci, "heavy"] and rng.random() < 0.65:
        center = bursts[ci] if rng.random() < 0.5 else bursts2[ci]
        d = center + pd.Timedelta(days=int(rng.integers(-12, 13)))
    else:
        d = START + pd.Timedelta(days=int(rng.integers(0, span)))
    d = min(max(d, START), END)
    rows.append((ci, d))
sk = pd.DataFrame(rows, columns=["ci", "deal_date"]).sort_values(["ci", "deal_date"]).reset_index(drop=True)
sk["deals_last_30d"] = 0
for ci, g in sk.groupby("ci"):
    dates = g["deal_date"].values
    for idx, d in zip(g.index, dates):
        sk.loc[idx, "deals_last_30d"] = int(((dates < d) & (dates >= d - np.timedelta64(30, "D"))).sum())
sk = sk.sort_values("deal_date").reset_index(drop=True)

# ---------- script text ----------
HOOK_LINES = {
    "story": ["Last month my skin broke out right before a shoot, and this saved me.",
              "I almost returned {b}, then something unexpected happened.",
              "Three years ago I would never have tried this, but here we are."],
    "question": ["Why does nobody talk about this {c} trick?",
                 "Is {b} actually worth your money? I tested it for two weeks.",
                 "What if you could fix this in under a minute?"],
    "shock": ["I did not expect this to actually work.",
              "Stop scrolling, this price is not a typo.",
              "This is the most underrated thing I have tried all year."],
    "social_proof": ["Everyone in my DMs keeps asking about {b}, so here it is.",
                     "Over 10,000 of you asked me to review this.",
                     "My friends all switched to {b} after trying it once."],
    "challenge": ["I used only {b} for seven days. Here is what happened.",
                  "Try this with me for one week and tell me the difference.",
                  "Day one of my {b} challenge, no cheating."],
    "demo": ["Watch how fast this comes together with {b}.",
             "Let me show you exactly how I use {b} every day.",
             "Here is the full process, start to finish, with {b}."],
    "direct_pitch": ["Hi guys, today I am sharing {b} with you.",
                     "This video is sponsored by {b}.",
                     "Hello everyone, I want to tell you about {b}."],
}
BODY = ["What I liked most is how easy it is to fit into a normal day.",
        "The quality surprised me, and the price is fair for what you get.",
        "I have been using it for a while now and I keep coming back to it."]
CTA_LINE = ["Link is in my bio, use my code for a discount.",
            "Tap the link below and grab it before the offer ends."]

recs = []
for i, r in sk.iterrows():
    cr = cdf.loc[r["ci"]]
    cat = cr["niche"] if rng.random() < 0.75 else rng.choice(NICHES)
    brand = rng.choice(BRANDS[cat])
    hook = rng.choice(HOOKS, p=HOOK_W)
    fmt = (rng.choice(["reel", "static", "story_set"], p=[0.6, 0.15, 0.25]) if cr["platform"] == "instagram"
           else rng.choice(["youtube_integration", "youtube_dedicated"], p=[0.55, 0.45]))
    video = fmt in ("reel", "youtube_integration", "youtube_dedicated")
    if video:
        base_d = IDEAL_DUR[cat][0]
        if fmt == "reel":
            dur = int(np.clip(rng.normal(base_d, 16), 8, 90))
        elif fmt == "youtube_integration":
            dur = int(np.clip(rng.normal(max(base_d, 40), 20), 20, 120))
        else:
            dur = int(np.clip(rng.normal(480, 180), 240, 900))
    else:
        dur = np.nan
    cta_pos = rng.choice(["start", "mid", "end"], p=[0.2, 0.3, 0.5])
    _hw = np.array([1, 1, 1, 1, 1, 1, 2, 3, 4, 4, 4, 5, 6, 6, 5, 5, 5, 6, 8, 10, 10, 9, 5, 2], dtype=float)
    hour = int(rng.choice(range(24), p=_hw / _hw.sum()))
    d = r["deal_date"]
    dow = d.dayofweek
    weekend = dow >= 5

    # multipliers
    best = HOOK_BEST.get(cat)
    if hook == "direct_pitch":
        hm = 0.78
    elif hook == best:
        hm = 1.35
    else:
        hm = 1.0 if rng.random() < 0.7 else 1.08
    if video and fmt == "reel":
        ideal, spread = IDEAL_DUR[cat]
        dm = 0.7 + 0.3 * np.exp(-0.5 * ((dur - ideal) / spread) ** 2)
    else:
        dm = 1.0
    n30 = int(r["deals_last_30d"])
    sat = 1.0 if n30 <= 2 else max(0.55, 1 - 0.14 * (n30 - 2))
    hourm = 1.15 if 19 <= hour <= 22 else (1.0 if 12 <= hour <= 14 else 0.88)
    if cat in ("fashion", "travel", "lifestyle"):
        dowm = 1.10 if weekend else 1.0
    elif cat == "tech":
        dowm = 1.0 if weekend else 1.10
    else:
        dowm = 1.0
    fit = 1.0 if cat == cr["niche"] else 0.80
    fmt_view = dict(reel=1.0, static=0.45, story_set=0.35, youtube_integration=0.8, youtube_dedicated=1.1)[fmt]
    tier_pen = 0.78 if (cr["tier"] in ("macro", "mega") and cat in ("beauty", "food")) else 1.0
    q = cr["quality"]
    noise = np.exp(rng.normal(0, 0.30))

    followers = cr["followers"]
    views = followers * REACH[cr["tier"]] * fmt_view * hm * dm * sat * hourm * dowm * q * fit * noise
    views = int(max(views, 200))
    eng_rate = float(np.clip(ENG[cr["tier"]] * hm ** 0.7 * q ** 0.5 * sat * fit * tier_pen
                             * np.exp(rng.normal(0, 0.15)), 0.004, 0.2))
    engs = int(views * eng_rate)
    saves = int(engs * SAVE_SHARE[cat])
    comments = int(engs * 0.05)
    shares = int(engs * 0.05)
    likes = max(engs - saves - comments - shares, 0)

    cta_ctr = dict(start=0.95, mid=1.25, end=0.8)[cta_pos]
    ctr = 0.016 * hm ** 0.5 * cta_ctr * sat * np.exp(rng.normal(0, 0.2))
    clicks = int(views * ctr)
    cvr = 0.09 * dict(start=0.95, mid=1.2, end=0.85)[cta_pos] * NICHE_CVR[cat] * np.exp(rng.normal(0, 0.25))
    conv = int(rng.binomial(clicks, min(cvr, 0.5))) if clicks > 0 else 0
    revenue = conv * AOV[cat] * float(np.exp(rng.normal(0, 0.1)))

    value = FEE[cr["tier"]] * NICHE_FEE[cat] * dict(reel=1.0, static=0.55, story_set=0.45,
            youtube_integration=1.0, youtube_dedicated=1.6)[fmt] * float(np.exp(rng.normal(0, 0.25)))
    value = int(round(value / 500) * 500)

    hl = rng.choice(HOOK_LINES[hook]).format(b=brand, c=cat)
    cta = rng.choice(CTA_LINE)
    body = rng.choice(BODY)
    parts = {"start": [cta, hl, body], "mid": [hl, cta, body], "end": [hl, body, cta]}[cta_pos]
    script = " ".join(p for p in parts if p) if video else ""

    recs.append(dict(
        deal_id=f"D{i+1:04d}", deal_date=d.date(), creator_id=cr["creator_id"],
        creator_name=cr["creator_name"], creator_handle=cr["handle"],
        creator_language=cr["language"], creator_city=cr["city"],
        platform=cr["platform"], tier=cr["tier"],
        creator_niche=cr["niche"], followers=followers, brand=brand, brand_category=cat,
        format=fmt, duration_sec=dur, hook_type=hook, hook_line=hl if video else "",
        cta_position=cta_pos if video else "", script_text=script,
        post_day=d.day_name(), post_hour=hour, creator_deals_last_30d=n30,
        deal_value_inr=value, views=views, likes=likes, comments=comments, saves=saves,
        shares=shares, link_clicks=clicks, conversions=conv, revenue_inr=round(revenue)))

df = pd.DataFrame(recs)
df["engagements"] = df[["likes", "comments", "saves", "shares"]].sum(axis=1)
df["engagement_rate"] = (df["engagements"] / df["views"]).round(4)
df["ctr"] = (df["link_clicks"] / df["views"]).round(5)
df["cpv_inr"] = (df["deal_value_inr"] / df["views"]).round(3)
df["cpe_inr"] = (df["deal_value_inr"] / df["engagements"].replace(0, np.nan)).round(2)
df["cost_per_conversion_inr"] = (df["deal_value_inr"] / df["conversions"].replace(0, np.nan)).round(0)
df["roas"] = (df["revenue_inr"] / df["deal_value_inr"]).round(3)
df.to_csv("data/monk_deals_simulated.csv", index=False)
print(df.shape)
print(df.head(3).T)
