"""Real-data physics figures from the 30-day kit runs (scratch), replayed with our starterkit_data."""
import sys, json
from pathlib import Path
import numpy as np, pandas as pd, matplotlib
matplotlib.use("Agg"); import matplotlib.pyplot as plt
WC=Path("/Users/srivathsan/Desktop/smart-buildings/working-code"); KIT=WC.parent/"smart-buildings-starter-kit"
OUT=Path(sys.argv[1]); RES=Path(sys.argv[2]); JS=Path(sys.argv[3])
sys.path.insert(0,str(WC/"code"))
from discovery.starterkit_data import load_starterkit_episode
plt.rcParams.update({"font.size":12,"axes.spines.top":False,"axes.spines.right":False})
COL=["#2F72B5","#3a9a5b","#8A93A3","#D9692A","#8e44ad","#13213A"]
SC=[("building_train_1","fairbanks_1"),("building_train_2","atlanta_3"),("building_train_4","denver_2"),("building_train_5","phoenix_1"),("building_train_7","singapore_6"),("building_train_8","atlanta_1")]
SD=KIT/"smart_buildings/training_scenarios"; cache={}; res={}
# 1 LINEAR IN TEMPERATURE: HVAC off, nights (no comfort mode), building-mean hourly change vs indoor-outdoor gap
fig,axs=plt.subplots(1,2,figsize=(13,4.6))
fits={}
for (b,w),c in zip(SC,COL):
    e,_=load_starterkit_episode(SD,RES,b,w,"all_off",cache)
    X=np.vstack([np.full((1,e.V),e.T0),e.T_meas]); Tm=X.mean(1); To=e.inputs["T_out"]; night=e.inputs["comfort"]<0.5
    k=np.arange(288,e.n-12,12); m=night[k]&night[k+11]&night[np.maximum(k-12,0)]   # skip the first hour after 19:00 (gains drop to 20%)
    m19=night[k]&night[k+11]&~night[np.maximum(k-12,0)]
    Tk=np.array([To[i:i+12].mean() for i in k])-Tm[k]
    axs[0].scatter(Tk[m19],(Tm[k+12]-Tm[k])[m19],s=12,marker="x",color=c,alpha=.6)
    dT=(Tm[k+12]-Tm[k])[m]; gap=(To[k:k+12].mean() if False else np.array([To[i:i+12].mean() for i in k])-Tm[k])[m]
    axs[0].scatter(gap,dT,s=5,alpha=.35,color=c,label=f"{b.replace('building_','')} {w}")
    A=np.c_[gap,np.ones_like(gap)]; coef,*_=np.linalg.lstsq(A,dT,rcond=None); r=np.corrcoef(gap,dT)[0,1]; fits[f"{b}/{w}"]=dict(slope_per_h=float(coef[0]),r=float(r))
    xx=np.linspace(gap.min(),gap.max(),10); axs[0].plot(xx,coef[0]*xx+coef[1],color=c,lw=1.5)
axs[0].set_xlabel("outside − mean room temperature, K"); axs[0].set_ylabel("change of mean room temperature per hour, K"); axs[0].legend(frameon=False,fontsize=8,markerscale=3)
axs[0].set_title("HVAC off, nights: one straight line per building (x = the hour after 19:00, when lights drop to 20%)",fontsize=10,loc="left")
nm=[k.replace("building_","").replace("/"," ") for k in fits]; axs[1].barh(nm,[fits[k]["r"] for k in fits],color=COL); axs[1].set_xlim(0,1); axs[1].set_xlabel("correlation r of the straight-line fit")
for i,k in enumerate(fits): axs[1].text(fits[k]["r"]+.01,i,f"r = {fits[k]['r']:.2f}",va="center",fontsize=10)
fig.tight_layout(); fig.savefig(OUT/"real_linear.png",dpi=130); plt.close(); res["linear"]=fits
# 2 NONLINEAR IN COMMANDS: damper command -> air stream g, reheat valve -> reheat heat (pid_water, train_2 Atlanta)
e,chk,F=load_starterkit_episode(SD,RES,"building_train_2","atlanta_3","pid_water",cache,full=True); act=F["act"]
ra=F["ctx"].room_ahu; P=act["ahu_pressure"][:,ra]; dmp=act["damper"]; g=e.inputs["g"]; rh=act["reheat"]; Q=e.inputs["Q_reheat"]
fig,axs=plt.subplots(1,2,figsize=(13,4.6))
idx=np.random.default_rng(0).choice(dmp.size,size=min(60000,dmp.size),replace=False)
sc=axs[0].scatter(dmp.ravel()[idx],g.ravel()[idx],c=P.ravel()[idx],s=3,cmap="viridis",alpha=.5); cb=fig.colorbar(sc,ax=axs[0]); cb.set_label("AHU static pressure, Pa")
axs[0].set_xlabel("damper position (command, floor 0.1)"); axs[0].set_ylabel("air stream into the room g, W/K")
axs[0].set_title("Air: product of damper and √pressure, a floor, and 0 when the AHU is off",fontsize=10,loc="left")
e2,_,F2=load_starterkit_episode(SD,RES,"building_train_1","fairbanks_1","pid_water",cache,full=True)
rh2=F2["act"]["reheat"]; Q2=e2.inputs["Q_reheat"]; j=np.random.default_rng(1).choice(rh2.size,size=min(60000,rh2.size),replace=False)
axs[1].scatter(rh2.ravel()[j],Q2.ravel()[j]/1000,s=3,alpha=.4,color="#D9692A"); axs[1].set_xlabel("reheat valve (command)"); axs[1].set_ylabel("reheat heat into the room, kW")
axs[1].set_title("Reheat (building_train_1, Fairbanks): capped, not proportional",fontsize=10,loc="left")
fig.suptitle("Real 30-day pid_water episodes: commands replayed, heat from the kit's equipment laws (left: building_train_2 Atlanta)",fontsize=11)
fig.tight_layout(); fig.savefig(OUT/"real_commands.png",dpi=130); plt.close()
res["commands_check"]=chk
# 3 FAST THEN SLOW: every weekday evening the baseline stops at 19:00; average room trajectory afterwards
fig,axs=plt.subplots(1,2,figsize=(13,4.6))
stats={}
for (b,w),c in zip(SC,COL):
    e,_=load_starterkit_episode(SD,RES,b,w,"pid_water",cache); X=np.vstack([np.full((1,e.V),e.T0),e.T_meas]); Tm=X.mean(1)
    com=e.inputs["comfort"]>0.5; stops=np.where(com[:-1]&~com[1:])[0]+1
    curves=[]
    for s in stops:
        if s+120<e.n: curves.append(Tm[s:s+121]-Tm[s])
    if not curves: continue
    C=np.array(curves); mcur=C.mean(0); t=np.arange(121)*5/60
    axs[0].plot(t,mcur,color=c,lw=2,label=f"{b.replace('building_','')} {w} ({len(C)} evenings)")
    fast=-(mcur[6]-mcur[0]); slow=-(mcur[120]-mcur[24])/8
    stats[f"{b}/{w}"]=dict(first_30min_K=float(fast),per_hour_after_2h_K=float(slow),evenings=len(C))
axs[0].axvline(0.5,color="#8A93A3",ls=":",lw=1); axs[0].set_xlabel("hours after the comfort period ends (19:00)"); axs[0].set_ylabel("mean room temperature change, K")
axs[0].legend(frameon=False,fontsize=8); axs[0].set_title("Every weekday evening, averaged",fontsize=11,loc="left")
nm=list(stats); x=np.arange(len(nm))
axs[1].bar(x-.2,[stats[k]["first_30min_K"]*2 for k in nm],.4,color="#D9692A",label="first 30 min (K per hour)")
axs[1].bar(x+.2,[stats[k]["per_hour_after_2h_K"] for k in nm],.4,color="#2F72B5",label="after 2 h (K per hour)")
axs[1].set_xticks(x); axs[1].set_xticklabels([k.replace("building_train_","b").replace("/"," ") for k in nm],rotation=25,ha="right",fontsize=9)
axs[1].set_ylabel("cooling rate, K per hour"); axs[1].legend(frameon=False,fontsize=9); axs[1].set_title("Fast first, then slow",fontsize=11,loc="left")
fig.tight_layout(); fig.savefig(OUT/"real_fastslow.png",dpi=130); plt.close(); res["fastslow"]=stats
json.dump(res,open(JS,"w"),indent=1,default=float); print(json.dumps(res,indent=1,default=float)[:3000])
