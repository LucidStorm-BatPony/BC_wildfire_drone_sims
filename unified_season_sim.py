import numpy as np
rng=np.random.default_rng(21)
# ==========================================================================
# UNIFIED BC SEASON SIM — all doctrine layers, April 1 to Oct 31 (214 d)
# Composes calibrated response functions from the component sims:
#   staging (brief 1), kill-cliff/double-tap + siege capacity, night pincer,
#   wind-day inventory, holdover/sniffer patrol, prescribed-burn box.
# Zone: 100x70 km Kamloops-FC-like region, ~300 starts core season.
# ==========================================================================
DAYS=214
def danger_curve():
    d=np.arange(DAYS)
    base=0.25+0.75*np.exp(-((d-120)/55.0)**2)      # peak late July
    ar=np.zeros(DAYS); 
    for i in range(1,DAYS): ar[i]=0.85*ar[i-1]+rng.normal(0,0.25)
    return np.clip(base*(1+0.5*ar),0.05,2.2)

# component response functions -----------------------------------------
def siege_response(starts,fleet):
    # from siege sim: escapes appear past capacity; escalation frac ~5%
    cap = 150.0 if fleet<=24 else (220.0 if fleet<=48 else 320.0)
    over=max(starts-cap,0)
    escapes=over*0.6
    escal=rng.binomial(int(max(starts-over,0)),0.05)
    return escapes,escal
ROS_CEIL_NIGHT=16.0
def pincer_resolves(ros): return ros<ROS_CEIL_NIGHT
def cost_escape(n): return rng.lognormal(np.log(0.6e6),1.5,n).sum() if n>0 else 0.0
def cost_megafire(n): return rng.lognormal(np.log(10e6),1.0,n).sum() if n>0 else 0.0

def run_season(nb,dpb,p_patrol,rx_on,n_rep=300):
    fleet=nb*dpb
    tot=np.zeros(n_rep); treated=np.zeros(n_rep)
    for rep in range(n_rep):
        dang=danger_curve()
        # schedule wind events: 3 draws weighted by danger, min 12 d apart
        wd=[]; pr=dang/dang.sum()
        while len(wd)<3:
            c=rng.choice(DAYS,p=pr)
            if all(abs(c-w)>12 for w in wd): wd.append(c)
        wind=set(wd)
        # lightning barrages: 3, danger-weighted
        bar=set(rng.choice(DAYS,3,replace=False,p=pr))
        cost=0.0; inv=0; sleepers=0; rx_ha=0.0
        for d in range(DAYS):
            # ignitions
            base=rng.poisson(1.05*dang[d])
            if d in bar:
                base+=rng.poisson(38)
                sleepers+=rng.poisson(38*15/85)
            # network initial attack (double-tap + EDF baked into response fn)
            esc,escal=siege_response(base,fleet)
            cost+=cost_escape(int(round(esc)))
            cost+=escal*40e3                       # pincer op cost
            # escalated fires: night pincer closes unless extreme ROS
            for _ in range(escal):
                if not pincer_resolves(rng.lognormal(np.log(1.2),1.0)*4):
                    inv+=1
            # standing inventory: pincer keeps working it down nightly
            if inv>0: inv-=rng.binomial(inv,0.5)
            # sleepers: nightly patrol + natural death/ordinary flare
            if sleepers>0:
                k=rng.binomial(sleepers,p_patrol); sleepers-=k
                g=rng.binomial(sleepers,0.10+0.135); sleepers-=g
            # wind day: inventory + armed sleepers blow up
            if d in wind and dang[d]>0.6:
                armed=rng.binomial(sleepers,0.7)+inv
                cost+=cost_megafire(armed)
                sleepers=max(sleepers-armed+inv,0)//1; inv=0
            # prescribed burn box (shoulder seasons, spare capacity)
            if rx_on and (d<60 or d>168) and dang[d]<0.55:
                if rng.random()<0.80:               # held-window fraction
                    squads=max(nb//3,1)             # a third of bases run burns
                    rx_ha+=squads*55.0              # ha/day per burn squad
        netcost=fleet*1.5e6*0.25+fleet*0.2e6+nb*0.15e6+fleet*200
        tot[rep]=cost+netcost; treated[rep]=rx_ha
    return tot.mean(),treated.mean()

# conventional baseline for the same season structure ------------------
def run_conv(n_rep=300):
    tot=np.zeros(n_rep)
    for rep in range(n_rep):
        dang=danger_curve()
        pr=dang/dang.sum()
        wd=[]; 
        while len(wd)<3:
            c=rng.choice(DAYS,p=pr)
            if all(abs(c-w)>12 for w in wd): wd.append(c)
        bar=set(rng.choice(DAYS,3,replace=False,p=pr))
        cost=0.0; inv=0.0; sleepers=0
        for d in range(DAYS):
            base=rng.poisson(1.05*dang[d])
            if d in bar:
                base+=rng.poisson(38); sleepers+=rng.poisson(38*15/85)
            esc=rng.binomial(base,0.217)            # calibrated escape rate
            cost+=cost_escape(esc)
            inv+=esc*0.25                           # quarter linger as inventory
            inv=max(inv-inv*0.12,0)                 # slow conventional closeout
            if sleepers>0:
                g=rng.binomial(sleepers,0.10+0.135); sleepers-=g
            if d in set(wd) and dang[d]>0.6:
                armed=rng.binomial(sleepers,0.7)+int(inv)
                cost+=cost_megafire(armed); sleepers=max(sleepers-armed,0); inv=0
        tot[rep]=cost
    return tot.mean()

conv=run_conv()
print(f"Conventional baseline season cost (zone): ${conv/1e6:.0f}M\n")
print(f"{'config':<10}{'sniffer':<14}{'Rx':<5}{'season cost':>12}{'saved':>9}{'ROI':>6}{'Rx ha':>8}")
best=None
for nb,dpb in [(6,4),(9,4),(12,4),(12,6),(16,6)]:
    for sn_lbl,p in [('none',0.0),('IR only',0.15),('nose+IR',0.31),('dual-spec',0.56)]:
        if sn_lbl in ('none','IR only') and (nb,dpb) not in [(6,4),(12,6)]: continue
        for rx in ([False,True] if p>=0.31 else [False]):
            c,ha=run_season(nb,dpb,p,rx)
            netcost=nb*dpb*1.5e6*0.25+nb*dpb*0.2e6+nb*0.15e6
            saved=conv-c+(ha*2500 if rx else 0)     # $2.5K/ha treatment value
            roi=saved/netcost
            print(f"{nb}x{dpb:<7}{sn_lbl:<14}{('Y' if rx else 'N'):<5}"
                  f"{c/1e6:>10.0f}M{saved/1e6:>8.0f}M{roi:>6.1f}{ha:>8.0f}")
            if best is None or saved>best[0]: best=(saved,nb,dpb,sn_lbl,rx)
print(f"\nMax net value: {best[1]}x{best[2]}, sniffer={best[3]}, Rx={best[4]}: ${best[0]/1e6:.0f}M saved")
