import numpy as np
rng=np.random.default_rng(3)
# ==========================================================================
# HOLDOVER LEAK MODEL
# Barrages seed holdovers (~15% of ignitions smolder rather than flame).
# Each holdover, per day: dies naturally (q=0.10), flares on an ordinary
# day (r=0.15 -> handled by normal network response), or persists.
# WIND DAY (the Bush Creek trigger): surviving holdovers flare with p=0.7,
# at wind-driven ROS where drops are ineffective -> each = megafire exposure.
# NIGHT IR PATROL: per-night detection prob p_det per smoldering spot;
# detected spot killed with 1-2 drops (trivial cost).
# Season: 3 barrages (day 0,15,30; e.g. ~7 holdovers each on the zone,
# scaled from 40 prompt fires x 15/85), wind event day 35.
# ==========================================================================
def season(p_det,n_rep=40000):
    exposure=np.zeros(n_rep); patrol_kills=np.zeros(n_rep)
    for b_day,n_mean in [(0,7),(15,7),(30,7)]:
        n=rng.poisson(n_mean,n_rep)
        days=35-b_day
        # per-day survival against death, ordinary flare, and patrol
        p_survive=(1-0.10)*(1-0.15)*(1-p_det)
        # expected patrol kills: caught before dying/flaring
        # simulate per rep aggregate binomially day by day (vectorized)
        alive=n.astype(float)
        for d in range(days):
            det=rng.binomial(np.maximum(alive.astype(int),0),p_det)
            patrol_kills+=det; alive=alive-det
            gone=rng.binomial(np.maximum(alive.astype(int),0),0.10+0.15*0.9)
            alive=np.maximum(alive-gone,0)
        exposure+=rng.binomial(np.maximum(alive.astype(int),0),0.7)
    return exposure.mean(), (exposure>=1).mean(), patrol_kills.mean()

print("WIND-DAY MEGAFIRE EXPOSURE vs night-patrol detection capability")
print("(3 lightning barrages before a day-35 wind event; ~21 holdovers seeded)\n")
print(f"{'patrol p/night':>15}{'armed flares on wind day':>26}{'P(>=1)':>9}{'patrol kills':>14}")
for p in [0.0,0.10,0.20,0.35,0.60]:
    m,p1,k=season(p)
    lbl='none' if p==0 else f'{p:.2f}'
    print(f"{lbl:>15}{m:>26.2f}{p1:>9.1%}{k:>14.1f}")
# expected cost framing: armed wind-day flare ~ campaign fire ~$10M (conservative)
print("\nExpected wind-day cost at $10M per armed flare (conservative):")
for p in [0.0,0.20,0.35]:
    m,_,_=season(p)
    print(f"  patrol {p:.2f}: ${m*10:.1f}M exposure per wind event")
