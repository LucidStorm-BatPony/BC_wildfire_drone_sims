import numpy as np
rng=np.random.default_rng(9)
# ==========================================================================
# SNIFFER PATROL — can a $200 nose beat IR on smoldering holdovers?
# PART A: nighttime plume physics.
# Source: duff spot smoldering ~1 kg fuel/h. Smoldering CO emission factor
#   ~200 g/kg (2-3x flaming) -> Q = 0.055 g/s CO. PM2.5 EF ~30 g/kg.
# Night stable boundary layer (Pasquill class F), drainage flow u=1 m/s
#   channeled down-valley. Ground-level centerline: C = Q/(pi*u*sy*sz).
# Channeling: valley walls cap lateral spread at ~W/4 (W=200 m valley).
# Sensor: electrochemical CO cell, resolution ~5 ppb, background ~120+-15 ppb
#   -> detection needs ~20 ppb excess (3 sigma of background fluctuation).
# 1 ppb CO = 1.145 ug/m3.
# ==========================================================================
def sigmas_F(x):   # Pasquill-Gifford class F approximations (x in m)
    sy=0.04*x*(1+0.0001*x)**-0.5
    sz=0.016*x*(1+0.0003*x)**-1.0
    return sy,sz
Q=0.055; u=1.0
print("PART A — CO excess (ppb) at valley floor vs down-drainage distance:")
print(f"{'dist (km)':>10}{'open terrain':>14}{'channeled valley':>18}")
for x in [0.5e3,1e3,2e3,4e3,8e3]:
    sy,sz=sigmas_F(x)
    C_open=Q/(np.pi*u*sy*sz)*1e6/1.145
    syc=min(sy,50.0)   # valley cap
    C_chan=Q/(np.pi*u*syc*sz)*1e6/1.145
    print(f"{x/1e3:>10.1f}{C_open:>14.1f}{C_chan:>18.1f}")
print("  (detection threshold ~20 ppb excess; IR needs near-overhead LOS)")

# ==========================================================================
# PART B — per-night detection probability for the patrol
# Zone catchment model: holdover lands in a catchment; plume flows to the
# main stem. Patrol flies main-stem transects (54 km/base/night at 10 m/s,
# 12 bases -> ~650 km, covering ~75% of major drainage stems).
# Detection per pass: crossing the plume where excess > 20 ppb.
#   d = spot's down-drainage distance to flown stem: U(0.3, 6) km.
#   p_cross = 0.8 if excess>60 ppb, 0.6 if >20, 0.15 if >8 (marginal), else 0.
#   Terrain: 75% of spots in channeled catchments w/ flown stems;
#            15% plateau/no channeling (open-terrain dilution, transect 2 km);
#            10% unmonitored micro-catchments (nose misses; IR-only).
# Fusion: nose cue -> next-pass IR localization succeeds p=0.85 (cued search,
#   gradient ascent up-drainage); else re-cue next night.
# IR-only baseline: p_det = 0.15/night (hard target under canopy).
# ==========================================================================
def excess_ppb(d,channeled=True):
    sy,sz=sigmas_F(d)
    if channeled: sy=min(sy,50.0)
    return Q/(np.pi*u*sy*sz)*1e6/1.145
def p_cross(e):
    if e>60: return 0.8
    if e>20: return 0.6
    if e>8: return 0.15
    return 0.0
def nightly_pdet(n=200000):
    r=rng.random(n); d=rng.uniform(300,6000,n)
    p=np.zeros(n)
    chan=r<0.75; plat=(r>=0.75)&(r<0.90)
    e=np.array([excess_ppb(x,True) for x in d])
    eo=np.array([excess_ppb(min(x,2000),False) for x in d])
    p[chan]=[p_cross(v) for v in e[chan]]
    p[plat]=[p_cross(v) for v in eo[plat]]
    # cue -> kill requires IR localization 0.85; failed localization re-cues
    p_kill=p*0.85
    # IR contributes independently everywhere:
    p_fused=1-(1-p_kill)*(1-0.15)
    return p_kill.mean(),p_fused.mean()
pk,pf=nightly_pdet()
print(f"\nPART B — per-night kill probability of a smoldering holdover:")
print(f"  IR-only patrol:        0.15")
print(f"  nose-only (cue+loc):   {pk:.2f}")
print(f"  fused nose+IR:         {pf:.2f}")

# ==========================================================================
# PART C — season consequence (same model as holdover brief:
# 3 barrages, ~21 sleepers, wind day 35)
# ==========================================================================
def season(p_det,n_rep=40000):
    exposure=np.zeros(n_rep)
    for b_day,n_mean in [(0,7),(15,7),(30,7)]:
        n=rng.poisson(n_mean,n_rep); alive=n.astype(float)
        for dd in range(35-b_day):
            det=rng.binomial(np.maximum(alive.astype(int),0),p_det)
            alive-=det
            gone=rng.binomial(np.maximum(alive.astype(int),0),0.10+0.15*0.9)
            alive=np.maximum(alive-gone,0)
        exposure+=rng.binomial(np.maximum(alive.astype(int),0),0.7)
    return exposure.mean(),(exposure>=1).mean()
print(f"\nPART C — wind-day megafire exposure (3 barrages, ~21 sleepers):")
print(f"{'patrol':>22}{'armed flares':>14}{'P(>=1)':>9}")
for lbl,p in [("no patrol",0.0),("IR only (0.15)",0.15),(f"fused ({pf:.2f})",pf)]:
    m,p1=season(p)
    print(f"{lbl:>22}{m:>14.2f}{p1:>9.1%}")

# ==========================================================================
# PART D — where the leverage hides
# Knob 1: dual-species discrimination (CO + guaiacol VOC fingerprint via
#   BME688) rejects false positives -> usable threshold drops 20 -> 8 ppb.
# Knob 2: source strength. 1 kg/h is a minimal spot; established sleepers
#   smolder 3-5 kg/h -> proportionally stronger plume.
# ==========================================================================
def nightly2(thresh,Qmult,n=100000):
    def pc(e):
        if e>3*thresh: return 0.8
        if e>thresh: return 0.6
        if e>0.4*thresh: return 0.15
        return 0.0
    r=rng.random(n); d=rng.uniform(300,6000,n); p=np.zeros(n)
    chan=r<0.75; plat=(r>=0.75)&(r<0.90)
    e=np.array([excess_ppb(x,True) for x in d])*Qmult
    eo=np.array([excess_ppb(min(x,2000),False) for x in d])*Qmult
    p[chan]=[pc(v) for v in e[chan]]
    p[plat]=[pc(v) for v in eo[plat]]
    return (1-(1-p*0.85)*(1-0.15)).mean()
print("\nPART D — fused per-night kill prob under upgrades:")
for lbl,th,qm in [("baseline (20 ppb, 1 kg/h)",20,1.0),
                  ("dual-species (8 ppb)",8,1.0),
                  ("bigger sleeper (3 kg/h)",20,3.0),
                  ("both",8,3.0)]:
    pn=nightly2(th,qm)
    m,p1=season(pn)
    print(f"  {lbl:<28} p/night {pn:.2f} -> wind-day P(>=1) {p1:.0%}")
print("\n10-night cumulative detection (survival pressure):")
for pn in [0.15,0.31,0.55]:
    print(f"  p/night {pn:.2f}: {1-(1-pn)**10:.0%} cumulative")
