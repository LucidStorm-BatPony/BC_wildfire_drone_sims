import numpy as np, heapq
rng=np.random.default_rng(7)
ZX,ZY=100e3,70e3; V=24.7; CYC=5.2
def radius_m(a,ros):
    te=max(a-15*(1-np.exp(-a/30)),0.05); return ros*te*0.6
def drops_needed(a,f):
    ros,fd,dose=f; return max(int(np.ceil(2*np.pi*radius_m(a,ros)*fd*dose/350.0)),1)
def ha_at(a,f):
    r=radius_m(a,f[0]); return np.pi*r*r*1.4/1e4
LIGHT=(2.0,0.8,2.0); HEAVY=(4.0,2.0,4.0)
def age_to_4ha(f):
    lo,hi=1.0,600.0
    for _ in range(30):
        m=(lo+hi)/2
        if ha_at(m,f)<4.0: lo=m
        else: hi=m
    return lo
A4L,A4H=age_to_4ha(LIGHT),age_to_4ha(HEAVY)
def gen(n,hours):
    fires=[]
    for c in range(max(int(round(n/13)),1)):
        t0=rng.uniform(0,max(hours-10,1)); cx,cy=rng.uniform(0,ZX),rng.uniform(0,ZY)
        vx,vy=rng.normal(0,2.2),rng.normal(0,1.0)
        for _ in range(rng.poisson(13)+1):
            dt=rng.uniform(0,10)
            fires.append([t0+dt,np.clip(cx+vx*1e3*dt+rng.normal(0,9e3),0,ZX),
                          np.clip(cy+vy*1e3*dt+rng.normal(0,9e3),0,ZY),
                          LIGHT if rng.random()<0.6 else HEAVY])
    return sorted(fires,key=lambda f:f[0])[:n]

def sim(nb,dpb,n_starts,hours,detect=8.0,n_rep=40):
    nx=int(np.ceil(np.sqrt(nb*ZX/ZY))); ny=int(np.ceil(nb/nx))
    bx=(np.arange(nx)+0.5)*ZX/nx; by=(np.arange(ny)+0.5)*ZY/ny
    BX,BY=np.meshgrid(bx,by); B=np.c_[BX.ravel(),BY.ravel()][:nb]
    ESC=ESCAL=KILL=0
    for _ in range(n_rep):
        fires=gen(n_starts,hours); NF=len(fires)
        Nw=nb*dpb
        wfree=np.zeros(Nw)                     # time each wasp frees
        wbase=np.repeat(np.arange(nb),dpb)
        arrive=[(f[0]+detect/60.0,k) for k,f in enumerate(fires)]; arrive.sort()
        pend={}; ai=0; esc=escal=kill=0; t=0.0
        while ai<len(arrive) or pend:
            # advance to next moment something can happen
            tw=np.partition(wfree,0)[0]
            ta=arrive[ai][0] if ai<len(arrive) else 1e9
            if not pend: t=max(ta,tw) if ta<1e9 else tw
            else: t=max(min(ta,tw),t)
            while ai<len(arrive) and arrive[ai][0]<=t+1e-9:
                pend[arrive[ai][1]]=True; ai+=1
            avail=np.where(wfree<=t+1e-9)[0]
            if len(avail)==0 or not pend:
                if ai>=len(arrive) and pend and len(avail)==0:
                    t=np.min(wfree[wfree>t]) if np.any(wfree>t) else t+0.1
                continue
            # EDF: least slack to 4 ha
            k2=min(pend,key=lambda c:fires[c][0]+ (A4L if fires[c][3] is LIGHT else A4H)/60.0)
            del pend[k2]
            # nearest available wasp
            d_all=np.hypot(fires[k2][1]-B[wbase[avail],0],fires[k2][2]-B[wbase[avail],1])
            wi=avail[np.argmin(d_all)]; d=np.min(d_all)
            arr=t+4/60.0+d/V/3600.0
            age=(arr-fires[k2][0])*60.0
            if ha_at(age,fires[k2][3])>=4.0:
                helpers=avail[np.argsort(d_all)][:6]
                if len(helpers)>=6:
                    wfree[helpers]=arr+1.5; escal+=1
                else:
                    esc+=1; wfree[wi]=t+0.05   # scout only, fire lost
                continue
            nd=drops_needed(age,fires[k2][3])+1
            dur=4/60.0+(2*d/V+nd*CYC*60)/3600.0
            wfree[wi]=t+dur
            if rng.random()<0.97: kill+=1
            else: arrive.append((t+dur+25/60.0,k2)); arrive.sort()
        ESC+=esc; ESCAL+=escal; KILL+=kill
    return ESC/n_rep, ESCAL/n_rep

print("BREAKING POINT v2 (EDF + overkill doctrine, corrected accounting):")
print(f"{'scenario':<28}{'24 wasps':>18}{'72 wasps':>18}")
print(f"{'':<28}{'esc/escalated':>18}{'esc/escalated':>18}")
for n,h,lbl in [(40,36,'July 2026 replay 40/36h'),(75,24,'150/day warning 75/24h'),
                (150,24,'2x warning 150/24h'),(150,12,'compressed 150/12h'),
                (300,24,'apocalypse 300/24h')]:
    a=sim(6,4,n,h); b=sim(12,6,n,h)
    print(f"{lbl:<28}{a[0]:>9.1f}/{a[1]:<8.1f}{b[0]:>9.1f}/{b[1]:<8.1f}")
print("\nDetection sensitivity (72 wasps, 150/24h):")
for det in [8.0,15.0,25.0]:
    e,es=sim(12,6,150,24,detect=det)
    print(f"  detect {det:>4.0f} min: escapes {e:>5.1f}  escalated {es:>5.1f}")
