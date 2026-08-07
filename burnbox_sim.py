import numpy as np
rng=np.random.default_rng(5)
# ==========================================================================
# BURN BOX: wasps holding the perimeter of a prescribed burn
# Burn: polygon of A_ha, active perimeter fraction 0.4 (strip ignition).
# Wind: mean U (km/h) + gust episodes (Poisson, 1.6x-2.4x mean, 5-15 min).
# Slop-overs: spot/creep across line, Poisson per km of active line per hr:
#   lam = 0.05 * (U_now/10)^3   [cubic in wind, ~spotting physics]
# Wasps on station: preloaded; hit slop-over in t_hit = 1.5 min transit
#   + queue; each kill = 1 drop + 6 min refill cycle before reusable.
# Slop-over becomes ESCAPE if not hit within 12 min (dry-day young-fire
#   window) or if >3 simultaneous unserved (crew overwhelm proxy).
# Burn day: 8 h active. Escape prob per burn day vs (A_ha, U, N wasps).
# ==========================================================================
def burn_day(A_ha,U_mean,N,n_rep=800):
    P_km=0.4*4*np.sqrt(A_ha*1e4)/1e3     # active line km (square-ish)
    esc=0
    for _ in range(n_rep):
        t=0.0; wfree=np.zeros(N); fail=False
        pend=[]  # spawn times of unserved slop-overs
        gust_until=0.0; gust_mult=1.0
        while t<480.0:
            if t>=gust_until and rng.random()<1.0/90.0:   # gust episode ~every 1.5h
                gust_until=t+rng.uniform(5,15); gust_mult=rng.uniform(1.6,2.4)
            U=U_mean*(gust_mult if t<gust_until else 1.0)
            lam=0.05*(U/10.0)**3*P_km/60.0
            for _ in range(rng.poisson(lam)):
                pend.append(t)
            pend2=[]
            for ts in pend:
                if N==0:
                    if t-ts>12.0: fail=True; break
                    pend2.append(ts); continue
                idx=np.argmin(wfree)
                if wfree[idx]<=t:
                    wfree[idx]=t+1.5+6.0   # engage + refill cycle
                elif t-ts>12.0:
                    fail=True; break
                else: pend2.append(ts)
            pend=pend2
            if fail or len(pend)>3: fail=True; break
            t+=1.0
        esc+=fail
    return esc/n_rep

print("ESCAPE PROBABILITY per 8-h burn day  (rows: mean wind; cols: wasps on station)")
for A in [50,200]:
    print(f"\n--- burn size {A} ha (active line {0.4*4*np.sqrt(A*1e4)/1e3:.1f} km) ---")
    print(f"{'wind km/h':>10}"+"".join(f"{f'N={n}':>8}" for n in [0,2,4,6,8]))
    for U in [8,12,16,20,25]:
        row=f"{U:>10}"
        for N in [0,2,4,6,8]:
            e=burn_day(A,U,N)
            row+=f"{e:>8.1%}"
        print(row)

# burnable-days widening: daily mean wind ~ Weibull(k=2, scale=13 km/h)
# conventional window: 5 <= U <= 12 (escape risk cap, no holding force)
# drone window at N wasps: 5 <= U <= U_max(N) where escape prob < 2% at 200 ha
print("\nBURNABLE-DAYS MULTIPLIER (200 ha burn, escape tolerance 2%/day):")
days=rng.weibull(2.0,200000)*13.0
base=np.mean((days>=5)&(days<=12))
for N in [2,4,6,8]:
    umax=5.0
    for U in np.arange(6,30,1.0):
        if burn_day(200,U,N,n_rep=300)<0.02: umax=U
    frac=np.mean((days>=5)&(days<=umax))
    print(f"  N={N}: safe up to ~{umax:.0f} km/h -> {frac/base:.2f}x burnable days (window {frac:.0%} of days vs {base:.0%})")
