import numpy as np
rng = np.random.default_rng(11)

# v4: acceleration-limited early growth + accessibility-weighted response.
V=24.7; PAYLOAD=350.0
ZONE_X, ZONE_Y = 100e3, 70e3
N=8000; STARTS=300

def sample_ros(n):
    return np.minimum(rng.lognormal(np.log(1.2), 1.0, n), 40.0)
def area_ha(ros,t):
    # acceleration: fires take ~30 min to reach steady ROS (McAlpine-Wakimoto style)
    t_eff = np.maximum(t - 15.0*(1-np.exp(-t/30.0)), 1.0)
    return 0.35*np.pi*(ros*t_eff)**2/1e4
def required_lpm(A): return 80.0+90.0*A
def squad_lpm(n_dr,d_w): return n_dr*PAYLOAD/((2*d_w/V+150.0)/60.0)

ros=sample_ros(N); d_water=rng.uniform(500,6000,N)
# conventional: 60% of starts accessible (fast response), 40% remote
accessible = rng.random(N)<0.60
is_night = rng.random(N)<0.35
day_t  = np.where(accessible, rng.lognormal(np.log(25),0.5,N), rng.lognormal(np.log(55),0.5,N))
night_t= np.where(accessible, rng.lognormal(np.log(120),0.5,N), rng.lognormal(np.log(300),0.4,N))
t_conv = np.where(is_night, night_t, day_t)
A_conv = area_ha(ros,t_conv); esc_conv=A_conv>=4.0
print("CALIBRATION v4:")
print(f"  escape(>4ha): {esc_conv.mean():.1%} (target ~15%) | held<1ha: {(A_conv<1).mean():.1%} (target ~79%)")
print(f"  day escape {esc_conv[~is_night].mean():.1%} | night escape {esc_conv[is_night].mean():.1%}")
def cost_of_fire(A):
    return np.where(A<4.0, rng.lognormal(np.log(15e3),0.8,len(A)),
                           rng.lognormal(np.log(0.6e6),1.5,len(A)))
c_conv=cost_of_fire(A_conv); season_base=c_conv.mean()*STARTS
print(f"  avg $/fire ${c_conv.mean()/1e3:.0f}K | provincial scale ${c_conv.mean()*1400/1e6:.0f}M (target 510-1000) | zone ${season_base/1e6:.0f}M")
print()
print(f"{'config':<22}{'fleet':>6}{'escape':>8}{'held<1ha':>9}{'net$':>7}{'saved':>8}{'ROI':>6}")
results={}
for nb,dpb in [(6,4),(9,4),(12,4),(12,6),(16,6),(20,6)]:
    nx=int(np.ceil(np.sqrt(nb*ZONE_X/ZONE_Y))); ny=int(np.ceil(nb/nx))
    bx=(np.arange(nx)+0.5)*ZONE_X/nx; by=(np.arange(ny)+0.5)*ZONE_Y/ny
    BX,BY=np.meshgrid(bx,by); B=np.c_[BX.ravel(),BY.ravel()][:nb]
    P=np.c_[rng.uniform(0,ZONE_X,N),rng.uniform(0,ZONE_Y,N)]
    d=np.min(np.linalg.norm(P[:,None,:]-B[None,:,:],axis=2),axis=1)
    t_net=8.0+4.0+d/V/60.0
    A_net=area_ha(ros,t_net); lpm=squad_lpm(dpb,d_water)
    net_c=(A_net<4.0)&(lpm>required_lpm(A_net))
    A_final=np.where(net_c, np.minimum(A_net,A_conv), A_conv)
    c_final=cost_of_fire(A_final); season=c_final.mean()*STARTS
    fleet=nb*dpb; netcost=fleet*1.5e6*0.25+fleet*0.2e6+nb*0.15e6
    saved=season_base-season-netcost
    esc_f=(A_final>=4.0).mean()
    results[(nb,dpb)]=(esc_f,saved,netcost)
    print(f"{nb:>3} bases x {dpb} wasps   {fleet:>6}{esc_f:>8.1%}{(A_final<1).mean():>9.1%}"
          f"{netcost/1e6:>6.0f}M{saved/1e6:>7.0f}M{saved/netcost:>6.1f}x")
# decomposition for 12x6
nb,dpb=12,6
nx=4;ny=3
bx=(np.arange(nx)+0.5)*ZONE_X/nx; by=(np.arange(ny)+0.5)*ZONE_Y/ny
BX,BY=np.meshgrid(bx,by); B=np.c_[BX.ravel(),BY.ravel()][:nb]
P=np.c_[rng.uniform(0,ZONE_X,N),rng.uniform(0,ZONE_Y,N)]
d=np.min(np.linalg.norm(P[:,None,:]-B[None,:,:],axis=2),axis=1)
t_net=8.0+4.0+d/V/60.0; A_net=area_ha(ros,t_net); lpm=squad_lpm(dpb,d_water)
net_c=(A_net<4.0)&(lpm>required_lpm(A_net)); rescued=net_c&esc_conv
print()
print(f"12x6 decomposition: rescued {rescued.mean():.1%} of starts from escape;")
print(f"  {(rescued&is_night).sum()/max(rescued.sum(),1):.0%} of rescues at night/smoke; "
      f"{(rescued&~accessible).sum()/max(rescued.sum(),1):.0%} in remote terrain")
print(f"  half-season sensitivity: at 150 starts, saved = ${(season_base/2 - (season_base-results[(12,6)][1]-results[(12,6)][2])/2 - results[(12,6)][2])/1e6:.0f}M (still positive? check)")
