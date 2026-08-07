import numpy as np

# ============================================================
# CONTAINMENT RACE: anchor-and-flank pincer on an established fire
# Squad of N Fire Wasps vs elliptical fire, diurnal intensity cycle
# ============================================================
# Fire geometry: ellipse, head ROS r, flank 0.35r, back 0.1r
# Byram fireline intensity I = H * w * ROS ; H=18 MJ/kg, w=1.5 kg/m^2
# Drop-effectiveness ceiling: segments with I > 2500 kW/m un-attackable
# Diurnal: night ROS multiplier 0.35 (humidity recovery), day 1.0
# Drop: 350 L -> effective secured line per drop (intensity-dependent):
#   L_drop = 45 m at I<500, 30 m at I<1500, 18 m at I<2500 kW/m
# Cycle: 2*d_water/V + 150 s overhead; V=24.7 m/s
# Rekindle: leading-edge wet segment holds 8 min (day) / 25 min (night);
#   line is permanent once fire burns out behind it (anchor logic) --
#   modelled as: line only at the two advancing pincer fronts is at risk;
#   if gap since last drop on a front > rekindle time, that front resets 150 m.
# Containment: total secured line >= perimeter at that moment, pincers meet.
H, w = 18000.0, 1.5     # kJ/kg, kg/m^2
V = 24.7
def intensity(ros_mmin): return H*w*(ros_mmin/60.0)   # kW/m
def drop_len(I):
    if I < 500: return 45.0
    if I < 1500: return 30.0
    if I < 2500: return 18.0
    return 0.0
def perimeter(a_head, b_flank):  # Ramanujan
    a,b = max(a_head,1.0), max(b_flank,1.0)
    h=((a-b)/(a+b))**2
    return np.pi*(a+b)*(1+3*h/(10+np.sqrt(4-3*h)))

def run(A0_ha, r_day, N, d_water=2000.0, t_start_hr=14.0, night_ops=True, T_max_hr=48):
    dt=60.0  # s
    cycle = 2*d_water/V + 150.0
    pulse = cycle/N          # seconds between drops arriving (staggered squad)
    # initial ellipse matching A0: A = pi*a*b, a/b ~ 2.5
    ratio=2.5
    b=np.sqrt(A0_ha*1e4/(np.pi*ratio)); a=ratio*b
    secured=0.0; t=0.0; next_drop=0.0
    front_last=[0.0,0.0]   # last drop time per pincer front
    turn=0
    while t < T_max_hr*3600:
        hr=(t_start_hr+t/3600.0)%24.0
        night = (hr>=21.0)or(hr<6.0)
        mult = 0.35 if night else 1.0
        r = r_day*mult
        rek = 1500.0 if night else 480.0
        # grow unsecured perimeter: head+flanks grow, scaled by fraction unsecured
        P=perimeter(a,b)
        frac_open=max(1.0-secured/P,0.0)
        a += (r/60.0)*dt*frac_open
        b += (0.35*r/60.0)*dt*frac_open
        # drops
        while next_drop<=t and (night_ops or not night):
            # pincer fronts work flanks toward head; head attackable only if I<ceiling
            I_here=intensity(r*0.6)  # flank-to-head transition intensity
            # if only head remains (secured > 80% P), must attack head itself
            if secured> 0.8*P: I_here=intensity(r)
            L=drop_len(I_here)
            f=turn%2; turn+=1
            if L>0:
                if t-front_last[f]>rek and front_last[f]>0:
                    secured=max(secured-150.0,0.0)   # front rekindled, lose ground
                secured+=L; front_last[f]=t
            next_drop+=pulse
        if not night_ops and night:
            next_drop=max(next_drop,t+dt)  # stand down; also rekindle losses accrue
            for f in(0,1):
                if t-front_last[f]>rek and front_last[f]>0:
                    secured=max(secured-150.0,0.0); front_last[f]=t
        P=perimeter(a,b)
        if secured>=P: return True, t/3600.0, np.pi*a*b/1e4
        t+=dt
    return False, T_max_hr, np.pi*a*b/1e4

print("PHASE BOUNDARY: largest fire size at attack (ha) a squad can close out")
print("(head ROS in m/min; water 2 km; success = pincer closes within 48 h)\n")
hdr=f"{'':>14}"+"".join(f"{f'N={n}':>8}" for n in [2,4,6,8,12])
for label,night_ops in [("24/7 ops",True),("day-only",False)]:
    print(f"--- {label} ---"); print(hdr)
    for r in [1.0,2.5,5.0,10.0,20.0]:
        row=f"ROS {r:>4.1f} m/min"
        for N in [2,4,6,8,12]:
            lo,hi=0.05,200.0; best=0.0
            for _ in range(18):
                mid=np.sqrt(lo*hi)
                ok,_,_=run(mid,r,N,night_ops=night_ops)
                if ok: best=mid; lo=mid
                else: hi=mid
            row+=f"{best:>8.1f}" if best>=0.05 else f"{'--':>8}"
        print(row)
    print()
# time-to-contain detail: 6 wasps, 5 ha fire, ROS 5, attack at 2pm vs 2am
for t0,lbl in [(14.0,"attack 14:00"),(2.0,"attack 02:00")]:
    ok,tt,Af=run(5.0,5.0,6,t_start_hr=t0)
    print(f"6 wasps, 5 ha @ ROS 5, {lbl}: contained={ok}, {tt:.1f} h, final {Af:.1f} ha")

print("\nCRITICAL ROS (m/min) — fastest fire a squad can close out (10 ha at attack):")
print(f"{'':>10}"+"".join(f"{f'N={n}':>8}" for n in [2,3,4,6,8,12]))
for label,night_ops in [("24/7",True),("day-only",False)]:
    row=f"{label:>10}"
    for N in [2,3,4,6,8,12]:
        lo,hi=0.3,40.0; best=0.0
        for _ in range(20):
            mid=np.sqrt(lo*hi)
            ok,_,_=run(10.0,mid,N,night_ops=night_ops)
            if ok: best=mid; lo=mid
            else: hi=mid
        row+=f"{best:>8.1f}"
    print(row)

print("\nFINAL BURNED AREA vs attack size (6 wasps, 24/7, ROS 5, attack 14:00):")
for A0 in [0.1,0.5,2.0,5.0,10.0,25.0]:
    ok,tt,Af=run(A0,5.0,6)
    print(f"  attack at {A0:>5.1f} ha -> contained {str(ok):>5} in {tt:>4.1f} h at {Af:>6.1f} ha  (growth x{Af/A0:.1f})")
