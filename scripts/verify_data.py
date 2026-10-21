import os, sys
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(ROOT); sys.path.insert(0, os.path.join(ROOT, "src"))  # run from anywhere; paths are relative to the repo root
import pandas as pd, numpy as np
df = pd.read_csv("data/monk_deals_simulated.csv")
pd.set_option("display.width", 200)
print("ROAS median/mean:", df.roas.median().round(2), df.roas.mean().round(2))
print("Total spend INR:", df.deal_value_inr.sum(), "Revenue:", df.revenue_inr.sum())
print("\nBy tier:\n", df.groupby("tier").agg(n=("deal_id","count"), eng=("engagement_rate","mean"), cpv=("cpv_inr","median"), roas=("roas","median")).round(3))
print("\nHook x category engagement (mean):\n", df.pivot_table(index="brand_category", columns="hook_type", values="engagement_rate", aggfunc="mean").round(3))
print("\nSaturation:\n", df.assign(sat=pd.cut(df.creator_deals_last_30d,[-1,2,3,10],labels=["0-2","3","4+"])).groupby("sat",observed=True).agg(n=("deal_id","count"), eng=("engagement_rate","mean"), ctr=("ctr","mean")).round(4))
print("\nCTA position:\n", df[df.cta_position!=""].groupby("cta_position").agg(n=("deal_id","count"), ctr=("ctr","mean"), conv=("conversions","mean")).round(4))
reels = df[(df.format=="reel")]
print("\nReel duration bucket x category (engagement):\n", reels.assign(b=pd.cut(reels.duration_sec,[0,20,30,45,90])).pivot_table(index="brand_category", columns="b", values="engagement_rate", aggfunc="mean", observed=True).round(3))
print("\nMacro/mega vs mid in beauty/food, cost per engagement:\n", df[df.brand_category.isin(["beauty","food"])].groupby("tier").cpe_inr.median().round(1))
print("\nFit (creator niche == category):\n", df.assign(fit=df.creator_niche==df.brand_category).groupby("fit").engagement_rate.mean().round(4))
