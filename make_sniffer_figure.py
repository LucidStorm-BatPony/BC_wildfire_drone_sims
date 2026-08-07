"""Generates figures/sniffer_exposure.png — README header chart.
Wind-day megafire exposure vs sniffer configuration (from sniffer_patrol_sim.py:
3 lightning barrages, ~21 holdover sleepers, wind event day 35).
Spec: 1200px wide, no in-image title, phone-readable axis labels.
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

fig,ax=plt.subplots(figsize=(8,4.5),dpi=150)
fig.patch.set_facecolor("white")

cfg=["No night\npatrol","IR-only\npatrol","+ $200 CO/VOC pod\n(nose + IR fused)","+ dual-species\ndiscrimination\n(same pod, software)"]
expo=[73,43,18,2]
colors=["#7a7a7a","#b0765a","#c8552e","#1b7a3d"]
x=np.arange(4)
bars=ax.bar(x,expo,color=colors,width=0.62)
for i,v in enumerate(expo):
    ax.text(i,v+2.5,f"{v}%",ha="center",fontsize=17,fontweight="bold",
            color=colors[i])
ax.annotate("",xy=(3,10),xytext=(0,68),
            arrowprops=dict(arrowstyle="->",lw=2.2,color="#1b7a3d",
                            connectionstyle="arc3,rad=-0.25"))
ax.text(1.55,62,"one ~$200 sensor pod\nper aircraft",ha="center",fontsize=13,
        fontweight="bold",color="#1b7a3d")
ax.set_xticks(x); ax.set_xticklabels(cfg,fontsize=12)
ax.set_ylabel("Wind-day megafire exposure\nP(≥1 sleeper flares)  %",fontsize=14)
ax.set_ylim(0,88)
ax.tick_params(axis="y",labelsize=12)
ax.grid(alpha=0.25,axis="y",lw=0.5)
for s in ["top","right"]: ax.spines[s].set_visible(False)
plt.tight_layout()
plt.savefig("sniffer_exposure.png",bbox_inches="tight",facecolor="white")
print("saved")
