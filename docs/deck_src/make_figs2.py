import sys, json, shutil
from pathlib import Path
import numpy as np, matplotlib
matplotlib.use("Agg"); import matplotlib.pyplot as plt
WC=Path("/Users/srivathsan/Desktop/smart-buildings/working-code"); KIT=WC.parent/"smart-buildings-starter-kit"
OUT=Path(sys.argv[1]); OUT.mkdir(exist_ok=True)
sys.path.insert(0,str(WC/"code"))
from discovery.episodes import geometry
plt.rcParams.update({"font.size":12,"axes.spines.top":False,"axes.spines.right":False,"font.family":"DejaVu Sans"})
ORA,BLU,GRY,INK="#D9692A","#2F72B5","#8A93A3","#13213A"
for f in ["overview.png","floorplan_variability.png","device_locations.png"]:
    shutil.copy(KIT/"images"/f, OUT/("kit_"+f))
for f in ["progress.png","g002_i1_0022_horizon.png"]:
    shutil.copy(WC/"outputs/discovery/selftest/plots"/f, OUT/("selftest_"+f.split("_")[-1]))
O=WC/"outputs"
def L(n): return np.load(O/f"roomdata_{n}.npz",allow_pickle=True)
t=lambda n: np.arange(n)/288
# 1 weather
fig,ax=plt.subplots(figsize=(11,4))
for n,c in [("building_train_1_fairbanks_1_alloff","#2F72B5"),("building_train_4_denver_2_alloff","#5B6676"),("building_train_2_atlanta_3_alloff","#3a9a5b"),("building_train_7_phoenix_3_alloff",ORA)]:
    z=L(n); ax.plot(t(len(z["Tamb"])),z["Tamb"]-273.15,color=c,lw=2,label=n.split("_")[3].capitalize())
ax.axhspan(21,24,color="#cfe8d4",alpha=.6,lw=0); ax.text(0.02,22.2,"comfort band (day)",fontsize=10,color="#3a7a4b")
ax.set_xlabel("days"); ax.set_ylabel("outside air, °C"); ax.legend(ncol=4,frameon=False,loc="upper center",bbox_to_anchor=(.5,1.12))
fig.tight_layout(); fig.savefig(OUT/"weather.png",dpi=130); plt.close()
# 2 rooms under pid_water, exterior vs interior
def ext_mask(z):
    rooms=[str(r) for r in z["rooms"]]; g=geometry(z["floor_plan"],z["zone_map"],rooms,{"a":rooms}); return g
z=L("building_train_2_atlanta_3_pidwater"); g=ext_mask(z); T=z["Tsensor"]-273.15; n=len(T)
ext=g["facade_faces"]>0
fig,ax=plt.subplots(figsize=(11,4.4))
ax.plot(t(n),T[:,~ext],color=BLU,lw=.6,alpha=.45); ax.plot(t(n),T[:,ext],color=ORA,lw=.6,alpha=.45)
ax.plot([],[],color=ORA,label=f"rooms with an outside wall ({ext.sum()})"); ax.plot([],[],color=BLU,label=f"interior rooms ({(~ext).sum()})")
ax2=ax.twinx(); ax2.plot(t(n),z["Tamb"]-273.15,color=GRY,lw=1.2); ax2.set_ylabel("outside °C",color=GRY); ax2.spines["right"].set_visible(True)
ax.set_xlabel("days"); ax.set_ylabel("room temperature, °C"); ax.legend(frameon=False,loc="lower left")
ax.set_title("building_train_2, Atlanta, pid_water baseline: 85 rooms, same controller",fontsize=11,loc="left")
fig.tight_layout(); fig.savefig(OUT/"rooms_pid.png",dpi=130); plt.close()
# 3 same building, three behaviours
fig,ax=plt.subplots(figsize=(11,4))
for p,c,lab in [("alloff",GRY,"HVAC off"),("random",BLU,"random commands"),("pidwater",ORA,"pid_water baseline")]:
    z=L(f"building_train_2_atlanta_3_{p}"); T=z["Tsensor"]-273.15
    ax.plot(t(len(T)),T.mean(1),color=c,lw=2,label=lab); ax.fill_between(t(len(T)),np.percentile(T,10,1),np.percentile(T,90,1),color=c,alpha=.12,lw=0)
ax.set_xlabel("days"); ax.set_ylabel("room temperature, °C (mean, 10–90 %)"); ax.legend(frameon=False,ncol=3)
fig.tight_layout(); fig.savefig(OUT/"policies.png",dpi=130); plt.close()
# 4 geometry predicts behaviour: night drop vs facade per area (HVAC off)
fig,ax=plt.subplots(figsize=(7,4.6))
for n,c in [("building_train_2_atlanta_3_alloff",ORA),("building_train_1_fairbanks_1_alloff",BLU),("building_train_8_fairbanks_8_alloff","#3a9a5b")]:
    z=L(n); g=ext_mask(z); T=z["Tsensor"]; d=(T[0]-T[143])/12  # K/h over first 12 h
    rel=d/np.maximum(T[0]-z["Tamb"][:144].mean(),1)
    ax.scatter(g["facade_faces"]*0.2/g["area"],rel*100,s=18,color=c,alpha=.75,label=n.split("_")[2]+" "+n.split("_")[3])
ax.set_xlabel("outside-wall length per m² of floor  (from the floor plan)"); ax.set_ylabel("cooling rate, % of indoor–outdoor gap per hour")
ax.legend(frameon=False,fontsize=10); fig.tight_layout(); fig.savefig(OUT/"geometry_explains.png",dpi=130); plt.close()
# 5 baselines
B=json.load(open(KIT/"baseline_returns.json")); pols=["bang_bang","fixed_setpoint","pid","pid_water"]
vals={p:[-v[p] for v in B.values() if p in v] for p in pols}
fig,ax=plt.subplots(figsize=(7,4))
ax.boxplot([vals[p] for p in pols],vert=False,labels=pols,showfliers=False); ax.set_xscale("log")
ax.set_xlabel("loss over the episode (lower is better, log scale)"); ax.set_title(f"kit baselines on the {len(B)} shipped scenarios",fontsize=11,loc="left")
fig.tight_layout(); fig.savefig(OUT/"baselines.png",dpi=130); plt.close()
print(sorted(p.name for p in OUT.iterdir()))
