# Monk Pitch Model: what it predicts, how well, and where it is weak

**What it does:** given a proposed deal (creator tier and niche, brand category, format, hook, CTA position, fee, timing, how booked-up the creator is), it predicts a **range** for engagement rate and ROAS: a low case (10th percentile), a typical case (median) and a high case (90th percentile).

**Trained on:** 7,500 simulated deals from 700 creators. All results below are from simulated data and show that the method works. Retrain on Monk's real deals with one command (`python3 pitch_model.py`).

## How it was tested
5-fold validation where **every test creator is unseen** by the model (folds are split by creator, so it cannot memorise a creator's habits). The baseline is the median for the same tier and brand category, which is roughly what a pitch built on averages would use.

| Metric | Engagement rate | ROAS |
|---|---|---|
| Rank correlation (Spearman) | 0.90 | 0.73 |
| Error vs baseline | 31% lower | 23% lower |
| R squared | 0.78 | 0.34 |
| Stated 80% range actually contained | 74% before fix, **80.1%** after | 74% before fix, **80.0%** after |

The raw ranges were too narrow (they contained about 74% of outcomes while claiming 80%), so I applied a standard conformal correction and checked it on creators held out from the calibration. The ranges now contain 80% of real outcomes.

**Reading this honestly:** engagement is highly predictable (large part of it is creator size). ROAS is much noisier, because it depends on whether a click turns into a sale. The model ranks ROAS well (0.73) but the exact value is uncertain: a typical ROAS range runs from about 0.4 to 2.6. That is why the tool gives ranges, not single numbers.

## What drives the predictions
- **Engagement:** creator size (tier, followers) (58%), recent deals (saturation) (25%), hook type (8%), niche match (7%).
- **ROAS:** fee (35%), creator size (tier, followers) (30%), brand category (12%), cta position (9%).

Shares come from permutation importance on held-out creators: shuffle one input, see how much accuracy drops.

## Does it agree with the analysis?
Same deals re-scored with one input changed at a time. This is a consistency check, not proof of cause.

| Change | Model says (engagement) | Model says (ROAS) | Analysis found |
|---|---|---|---|
| Story hook instead of direct pitch | +24% | +44% | Direct pitch gets 19% less engagement |
| Niche mismatch instead of match | -18% | not shown | 20% lower engagement |
| 4 recent deals instead of 1 | -29% | -44% | 40% average drop at 3 or more deals |
| CTA mid-video instead of end | 0% | +90% | CTR 59% higher |
| 45s reel instead of 20s | +1% | not shown | no significant length effect |
| Saturday instead of Tuesday, 8pm instead of 4am | +1%, 0% | not shown | not tested as a finding |

The model finds the effects the analysis found and finds none where the analysis found none. The saturation figure is smaller than the headline (about 29% at exactly 4 recent deals vs 40% averaged over all deals with 3 or more, which includes heavier bursts). The CTA effect on ROAS is larger than on CTR because mid-video CTAs lift both clicks and conversion.

## Where it is weak
- **ROAS at the top end:** the model is conservative. Its highest-predicted bucket has a median prediction of 2.9 against an actual median of 4.1. Treat the high end as a floor.
- **Hook ranking is soft:** among the non-direct hooks the predicted ROAS differed by only 1 to 2% in the briefs I checked, which is a tie. The tool flags near-ties instead of crowning a winner. What is clear: mid-video CTA, any non-direct hook, avoid saturated creators.
- **Language and city are excluded on purpose.** I tested them (tier-adjusted, creator level) and found no effect, so including them would only add noise.
- **Simulated data.** Real data will be noisier and will not follow these tidy patterns. The point of this build is the method and the tool, not the specific numbers.
- **Not causal.** It describes how outcomes relate to inputs in past deals. Test big changes on live campaigns before making them policy.

## Using it
```python
from pitch_model import load, predict_brief, best_plays
art = load()
predict_brief(art, dict(tier="mid", creator_niche="fashion", brand_category="fashion",
                        format="reel", hook_type="social_proof", cta_position="mid", creator_deals_last_30d=1))
best_plays(art, tier="mid", niche="beauty", category="beauty", top=5)   # ranks hook x CTA combos
```
