"""Generates figures/season_economics.png for the README.
Left: why response time is the lever (suppression cost convexity).
Right: season net value by network configuration (unified_season_sim.py output,
       dual-species sniffer + prescribed-burn program).
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

fig,(ax1,ax2)=plt.subplots(1,2,figsize=(13,5.2),dpi=150)
fig.patch.set_facecolor("white")

# ---- LEFT: convexity of suppression cost vs fire size at containment ----
sizes=np.logspace(-1,4,200)                      # 0.1 ha .. 10,000 ha
cost=8e3*(sizes/0.1)**1.15                        # calibrated anchors: $8K@0.1ha -> ~$50M campaign
cost=np.minimum(cost,6e7)
ax1.loglog(sizes,cost,color="#8B2500",lw=2.8)
ax1.axvspan(0.1,1.0,color="#2e7d32",alpha=0.15)
ax1.axvspan(200,10000,color="#8B2500",alpha=0.10)
ax1.annotate("drone network\nintercept window\n(median attack 17 min)",xy=(0.35,1.2e6),
             ha="center",fontsize=9,color="#1b5e20",fontweight="bold")
ax1.annotate("campaign fire\n\\$10M – \\$50M+",xy=(1400,1.3e7),ha="center",fontsize=9,
             color="#8B2500",fontweight="bold")
ax1.set_xlabel("Fire size at containment (ha)")
ax1.set_ylabel("Suppression cost ($)")
ax1.set_title("Why response time is the lever",fontsize=12,fontweight="bold")
ax1.grid(alpha=0.25,which="both",lw=0.4)

# ---- RIGHT: season net value by configuration (from unified_season_sim.py) ----
cfg=["6×4\n(24 a/c)","9×4\n(36 a/c)","12×4\n(48 a/c)","12×6\n(72 a/c)","16×6\n(96 a/c)"]
saved=[114,117,122,107,105]                       # $M, dual-spec sniffer + Rx program
netcost=[16,24,32,46,59]                          # $M network cost/season
colors=["#c8875f","#c8875f","#8B2500","#c8875f","#c8875f"]
x=np.arange(5)
b=ax2.bar(x,saved,color=colors,width=0.62)
ax2.bar(x,netcost,color="none",edgecolor="#444",width=0.62,hatch="///",lw=0.8)
for i,(s,c) in enumerate(zip(saved,netcost)):
    ax2.text(i,s+2,f"${s}M\n{s/c:.1f}×",ha="center",fontsize=9,
             fontweight="bold" if i==2 else "normal")
ax2.set_xticks(x); ax2.set_xticklabels(cfg,fontsize=9)
ax2.set_ylabel("Season net value ($M, modelled zone)")
ax2.set_title("Season-optimal network: 12 bases × 4 aircraft",fontsize=12,fontweight="bold")
ax2.set_ylim(0,145)
ax2.legend([b[2],plt.Rectangle((0,0),1,1,fc="none",ec="#444",hatch="///")],
           ["net value saved (with sniffer + burn program)","network cost/season"],
           fontsize=8.5,loc="upper left",frameon=False)
ax2.grid(alpha=0.25,axis="y",lw=0.4)

fig.suptitle("BC wildfire drone network — season-scale economics (Monte Carlo, BCWS-calibrated)",
             fontsize=13,fontweight="bold",y=1.0)
fig.text(0.5,0.005,"Ben Watson (Lucid) · independent research · github.com/LucidStorm-BatPony/BC_wildfire_drone_sims · costs are flagged placeholders pending vendor figures",
         ha="center",fontsize=7.5,color="#777")
plt.tight_layout(rect=[0,0.02,1,0.97])
plt.savefig("season_economics.png",bbox_inches="tight",facecolor="white")
print("saved")
