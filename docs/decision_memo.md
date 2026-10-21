# Monk Deal Intelligence: what the data says and what I would do

**Dataset:** 7,500 simulated influencer deals, 700 creators, Oct 2024 to Sep 2026, total spend Rs 41.54 cr. All numbers below are from simulated data and show the method. The same code runs on Monk's real deal log.

## Headline
Across the book, every rupee spent returned Rs 1.09 in attributed revenue, and **51% of deals lose money**. The losses are concentrated, which means they are fixable.

## Five decisions

**1. Move budget out of macro and mega, into mid tier.**
Macro and mega take 76% of spend and return 0.87 ROAS; 67% of those deals lose money. Mid tier returns 1.79 ROAS with 46% losing. The ROAS gap is 0.93 (95% CI 0.69 to 1.16).
Illustrative impact: shifting 20% of macro and mega spend (Rs 6.32 cr) into mid tier would add about Rs 5.86 cr of attributed revenue, **if** mid-tier ROAS holds at scale. Treat that as an upper bound, since returns usually fall as you scale a tier.

**2. Stop shipping direct-pitch scripts.**
25% of video deals open with a plain "this video is sponsored by" style hook. They get 19% less engagement than the other hooks combined (2.71% vs 3.35% median; 95% CI 16% to 22%). Direct pitch ranks last in 7 of 8 categories and second to last in the rest.
The best hook in each category beats direct pitch significantly in 8 of 8 categories. Best hook by category: beauty story, fashion social proof, fitness challenge, food demo, gaming shock, lifestyle demo, tech question, travel challenge. Each of these rests on 66 to 106 videos, and the gaps among the non-direct hooks are smaller, so use this as a starting playbook and keep testing it.

**3. Cap how often one creator is booked.**
Creators with 3 or more deals in the prior 30 days get 40% lower engagement (95% CI 35% to 45%; 39% after controlling for tier). 1,557 deals (21% of the book, Rs 8.83 cr of spend) were booked into saturated creators and returned 0.51 ROAS against 1.24 for the rest.
Rule: a cooldown flag at 3 deals in 30 days, built into the pitch tool.

**4. Put the call to action in the middle of the video.**
Mid-video CTAs get 59% higher CTR than end CTAs (1.87% vs 1.18%; 95% CI 56% to 61%), and clicks convert better too (10.1% vs 7.2%). Half of all video deals (50%) still put it at the end. This is the cheapest change on the list: it costs nothing.

**5. Match creator niche to brand category.**
22% of deals pair a creator with a category outside their niche. Those get 20% lower engagement (2.69% vs 3.36%; 95% CI 17% to 23%). Add a niche-fit check to every pitch.

## Creators to review
10 large creators account for Rs 7.02 cr of spend and a net loss of Rs 4.69 cr (ROAS below 0.7, top 40% by spend). Full list in `quietly_losing_money.csv`. Top of the list:

| Creator | Tier | Niche | Deals | Spend | ROAS |
|---|---|---|---|---|---|
| Vivek Arora (@vivekarora) | mega | food | 34 | Rs 1.31 cr | 0.28 |
| Swetha Iyer (@swetha.gamer) | mega | gaming | 32 | Rs 1.03 cr | 0.21 |
| Madhurima Dutta (@themadhurimaglam) | mega | beauty | 25 | Rs 0.82 cr | 0.31 |
| Yash Yadav (@yash.beauty) | mega | beauty | 24 | Rs 0.74 cr | 0.46 |
| Arjun Kurup (@arjun_kurup) | mega | gaming | 8 | Rs 0.52 cr | 0.24 |

On the other side, 239 creators have ROAS above 1.5 with 6+ deals, and 215 of them are nano, micro or mid tier. Single-creator ROAS rests on a limited number of deals each, so treat both lists as shortlists for review, not verdicts.

## What I tested and did not headline
- **Language:** raw ROAS ranges from 0.81 (English) to 1.34 (Kannada), but that gap does not survive scrutiny. Comparing creators (not deals), adjusted for tier, there is no detectable language effect (p = 0.93). The raw gap comes partly from tier mix (languages with more macro and mega spend look worse) and partly from chance variation across small groups of creators. Language is a targeting filter, not a performance driver.
- **City:** same result. Tier-adjusted, city makes no detectable difference (p = 0.63), and metro vs tier 2 and 3 creators perform alike (p = 0.79).
- **Video length:** no category shows a statistically significant effect of video length on engagement, so I would not recommend a length rule.

## Limits
- Simulated data. Real data will be noisier, and some of these effects will shrink.
- ROAS depends on attribution (tracked links and codes). Brands that do not share conversion data will need a proxy such as CPV or engagement.
- Deals from the same creator are not independent, so every confidence interval above comes from resampling creators (400 bootstrap rounds), not deals. All five headline effects stay clearly away from zero.
- Associations, not experiments. The cooldown rule and the hook rule should be tested on live deals before they become policy.

## What I would do in the first 30 days
1. Load Monk's real deal log into the same schema and rerun this analysis.
2. Ship the pitch tool: enter a brief, get creator mix, hook angles, CTA placement and budget split by tier.
3. Run a clean test of the hook and CTA changes on live campaigns.
