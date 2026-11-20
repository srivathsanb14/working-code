"""30-day figures: the 6 shipped buildings, one weather each, pid_water and all_off, 8,640 steps (kit evaluate.py into a
scratch folder), replayed into episodes with our starterkit_data.load_starterkit_episode, seeds scored with closedform."""
import sys, json, re
from pathlib import Path
import numpy as np, pandas as pd, matplotlib
matplotlib.use("Agg"); import matplotlib.pyplot as plt
WC=Path("/Users/srivathsan/Desktop/smart-buildings/working-code"); KIT=WC.parent/"smart-buildings-starter-kit"
OUT=Path(sys.argv[1]); RES=Path(sys.argv[2]); JS=Path(sys.argv[3])
sys.path.insert(0,str(WC/"code"))
from discovery.starterkit_data import load_starterkit_episode
from discovery import closedform as cf
plt.rcParams.update({"font.size":12,"axes.spines.top":False,"axes.spines.right":False})
ORA,BLU,GRY,INK,GRN="#D9692A","#2F72B5","#8A93A3","#13213A","#3a9a5b"
SC=[("building_train_1","fairbanks_1"),("building_train_2","atlanta_3"),("building_train_4","denver_2"),("building_train_5","phoenix_1"),("building_train_7","singapore_6"),("building_train_8","atlanta_1")]
SD=KIT/"smart_buildings/training_scenarios"; U,_=cf.universal_constants(); res={}
eps={}; cache={}
for b,w in SC:
    for p in ("pid_water","all_off"):
        e,chk=load_starterkit_episode(SD,RES,b,w,p,cache); eps[(b,w,p)]=e
        res.setdefault("replay",{})[e.name]=dict(steps=e.n,rooms=e.V,**chk)
        print(e.name,e.n,e.V,{k:round(v,4) for k,v in chk.items()},flush=True)
day=lambda n: np.arange(n)/288
def band(e):
    c=e.inputs["comfort"]>0.5; return np.where(c,294.0,289.0),np.where(c,297.0,298.0)
# 1 profiles, 6 buildings, 30 days
fig,axs=plt.subplots(3,2,figsize=(14,11),sharex=True)
for ax,(b,w) in zip(axs.flat,SC):
    e=eps[(b,w,"pid_water")]; X=np.vstack([np.full((1,e.V),e.T0),e.T_meas])-273.15; t=day(len(X)); lo,hi=band(e)
    ax.fill_between(day(e.n),lo-273.15,hi-273.15,color="#cfe8d4",alpha=.6,lw=0,step="post")
    ext=e.geo["facade_faces"]>0
    ax.plot(t,X[:,ext],color=ORA,lw=.25,alpha=.35); ax.plot(t,X[:,~ext],color=BLU,lw=.25,alpha=.35)
    ax.plot(t,np.median(X,1),color=INK,lw=1)
    ax2=ax.twinx(); ax2.plot(day(e.n),e.inputs["T_out"]-273.15,color=GRY,lw=.6,alpha=.7); ax2.tick_params(labelsize=8,colors=GRY)
    ax.set_title(f"{b} · {w} · {e.V} rooms",fontsize=11,loc="left"); ax.set_ylabel("°C")
for ax in axs[-1]: ax.set_xlabel("day of the 30-day episode")
axs[0,0].plot([],[],color=ORA,label="rooms with an outside wall"); axs[0,0].plot([],[],color=BLU,label="interior rooms"); axs[0,0].plot([],[],color=INK,label="median room")
axs[0,0].legend(frameon=False,fontsize=9,loc="lower left")
fig.suptitle("30-day episodes, pid_water baseline: every room (green = active comfort band; grey, right axis = outside air)",fontsize=12)
fig.tight_layout(); fig.savefig(OUT/"ep30_profiles.png",dpi=110); plt.close()
# 2 one week zoom
e=eps[("building_train_2","atlanta_3","pid_water")]; X=np.vstack([np.full((1,e.V),e.T0),e.T_meas])-273.15; ext=e.geo["facade_faces"]>0
a,bnd=7*288,14*288; t=day(len(X))
fig,ax=plt.subplots(figsize=(12,4.4)); lo,hi=band(e)
ax.fill_between(day(e.n)[a:bnd],lo[a:bnd]-273.15,hi[a:bnd]-273.15,color="#cfe8d4",alpha=.6,lw=0,step="post")
ax.plot(t[a:bnd],X[a:bnd][:,ext],color=ORA,lw=.5,alpha=.4); ax.plot(t[a:bnd],X[a:bnd][:,~ext],color=BLU,lw=.5,alpha=.4)
ax2=ax.twinx(); ax2.plot(day(e.n)[a:bnd],e.inputs["T_out"][a:bnd]-273.15,color=GRY,lw=1); ax2.set_ylabel("outside °C",color=GRY)
ax.set_xlabel("day"); ax.set_ylabel("room °C"); ax.set_title("One week of the episode: building_train_2, Atlanta, pid_water (orange: outside wall, blue: interior)",fontsize=11,loc="left")
fig.tight_layout(); fig.savefig(OUT/"ep30_week.png",dpi=130); plt.close()
# 3 weather over 30 days
fig,ax=plt.subplots(figsize=(12,4.2))
for (b,w),c in zip(SC,[BLU,"#3a9a5b",GRY,ORA,"#8e44ad",INK]):
    e=eps[(b,w,"pid_water")]; ax.plot(day(e.n),e.inputs["T_out"]-273.15,color=c,lw=1,label=w)
ax.set_xlabel("day"); ax.set_ylabel("outside air, °C"); ax.legend(frameon=False,ncol=6,fontsize=9,loc="upper center",bbox_to_anchor=(.5,1.13))
fig.tight_layout(); fig.savefig(OUT/"ep30_weather.png",dpi=130); plt.close()
# 4 reward terms per day, 6 buildings
fig,axs=plt.subplots(2,3,figsize=(14,6.5),sharex=True)
summ={}
for ax,(b,w) in zip(axs.flat,SC):
    rc=pd.read_csv(RES/b/w/"pid_water"/"reward_components.csv"); n=len(rc); d=np.arange(n)//288
    cm=pd.Series(-0.6*rc["normalized_productivity_regret"].values).groupby(d).mean(); en=pd.Series(0.2*rc["normalized_energy_cost"].values).groupby(d).mean(); ca=pd.Series(0.2*rc["normalized_carbon_emission"].values).groupby(d).mean()
    ax.bar(cm.index,cm,color=ORA,label="comfort"); ax.bar(en.index,en,bottom=cm,color=BLU,label="energy"); ax.bar(ca.index,ca,bottom=cm+en,color=GRY,label="carbon")
    s=json.load(open(RES/b/w/"pid_water"/"summary.json")); summ[f"{b}/{w}"]=dict(score=-s["avg_reward"],sec_per_step=s.get("seconds_per_step"))
    ax.set_title(f"{b} · {w} · score {-s['avg_reward']:.4f}",fontsize=10,loc="left")
axs[0,0].legend(frameon=False,fontsize=9); 
for ax in axs[-1]: ax.set_xlabel("day")
for ax in axs[:,0]: ax.set_ylabel("mean loss per step")
fig.suptitle("Where the loss comes from, day by day, over the 30-day episode (pid_water; kit reward_components.csv)",fontsize=12)
fig.tight_layout(); fig.savefig(OUT/"ep30_reward.png",dpi=110); plt.close(); res["pid_water_scores"]=summ
# 5 comfort deviation, K outside band, per day
fig,ax=plt.subplots(figsize=(12,4))
for (b,w),c in zip(SC,[BLU,"#3a9a5b",GRY,ORA,"#8e44ad",INK]):
    e=eps[(b,w,"pid_water")]; lo,hi=band(e); T=e.T_meas; dev=np.maximum(lo[:,None]-T,0)+np.maximum(T-hi[:,None],0)
    com=e.inputs["comfort"]>0.5; d=np.arange(e.n)//288
    daily=pd.Series(np.where(com,dev.mean(1),np.nan)).groupby(d).mean()
    ax.plot(daily.index,daily.values,"o-",color=c,ms=3,lw=1,label=b.replace("building_","")+" "+w)
ax.set_xlabel("day"); ax.set_ylabel("mean K outside band, comfort hours"); ax.legend(frameon=False,fontsize=8,ncol=3)
ax.set_title("Comfort deviation, measured as the reward does (unweighted by occupants here)",fontsize=11,loc="left")
fig.tight_layout(); fig.savefig(OUT/"ep30_comfort.png",dpi=130); plt.close()
# 6 HVAC off vs pid_water, mean room temperature, 30 days
fig,axs=plt.subplots(1,2,figsize=(14,4))
for ax,(b,w) in zip(axs,[SC[1],SC[0]]):
    for p,c,l in (("all_off",GRY,"HVAC off"),("pid_water",ORA,"pid_water")):
        e=eps[(b,w,p)]; T=e.T_meas-273.15; ax.plot(day(e.n),T.mean(1),color=c,lw=1.3,label=l); ax.fill_between(day(e.n),np.percentile(T,10,1),np.percentile(T,90,1),color=c,alpha=.15,lw=0)
    ax.set_title(f"{b} · {w}",fontsize=11,loc="left"); ax.set_xlabel("day"); ax.set_ylabel("room °C (mean, 10–90 %)")
axs[0].legend(frameon=False)
fig.tight_layout(); fig.savefig(OUT/"ep30_policies.png",dpi=130); plt.close()
# 7 geometry explains: HVAC off, first 12 h, 6 buildings
fig,ax=plt.subplots(figsize=(7.5,4.6))
for (b,w),c in zip(SC,[BLU,"#3a9a5b",GRY,ORA,"#8e44ad",INK]):
    e=eps[(b,w,"all_off")]; X=np.vstack([np.full((1,e.V),e.T0),e.T_meas]); gap=X[0]-e.inputs["T_out"][:144].mean()
    rate=(X[0]-X[144])/12/np.where(np.abs(gap)<1,np.nan,gap)*100
    ax.scatter(e.geo["facade_faces"]*0.2/e.geo["area"],rate,s=14,color=c,alpha=.75,label=b.replace("building_","")+" "+w)
ax.set_xlabel("outside-wall length per m² of floor (from the floor plan)"); ax.set_ylabel("cooling, % of indoor–outdoor gap per hour")
ax.legend(frameon=False,fontsize=8); fig.tight_layout(); fig.savefig(OUT/"ep30_geometry.png",dpi=130); plt.close()
# 8 seeds on the 30-day episodes (nothing fitted)
seeds={p.stem:p.read_text() for p in sorted((WC/"code/discovery/seeds_cf").glob("*.py"))}
allep=list(eps.values()); H=(12,36,72); tab={}
for nm,src in seeds.items():
    r=cf.evaluate(src,allep[:1],allep,U,horizons=H,every=12,screen_factor=1e9,max_train=1); tab[nm]=r
    print(nm,r["status"],r.get("reason"),{k:round(v,3) for k,v in r.items() if k.startswith("val_") and isinstance(v,float)},flush=True)
ceil={e.name:cf.ceiling(e,U,H,12) for e in allep}
res["seeds"]={nm:{k:v for k,v in r.items() if k.startswith(("val_","stage1")) and isinstance(v,float)} for nm,r in tab.items()}
res["ceiling"]={k:float(np.mean([c[k] for c in ceil.values()])) for k in next(iter(ceil.values()))}
res["by_episode_3h"]={nm:{en:v["36"] for en,v in r["val_by_episode"].items()} for nm,r in tab.items()}
res["ceiling_by_episode_3h"]={k:v["3h"] for k,v in ceil.items()}
fig,ax=plt.subplots(figsize=(7.5,4.3)); hh=[h/12 for h in H]; r0=next(iter(tab.values()))
ax.plot(hh,[r0[f"val_persistence_{cf._hname(h)}_mean"] for h in H],"s--",color=GRY,label="persistence")
for (nm,r),c in zip(tab.items(),[ORA,BLU]): ax.plot(hh,[r[f"val_{cf._hname(h)}_mean"] for h in H],"o-",color=c,label=nm.replace("seed_cf_","seed: ")+" (no fitting)")
ax.plot(hh,[res["ceiling"][cf._hname(h)] for h in H],"^:",color=GRN,label="ceiling: fitted per room (diagnostic)")
ax.set_xticks(hh); ax.set_xlabel("forecast horizon, hours"); ax.set_ylabel("RMSE, K"); ax.legend(frameon=False,fontsize=9); ax.grid(alpha=.3)
ax.set_title("12 episodes × 30 days: 6 buildings × (pid_water, HVAC off)",fontsize=11,loc="left")
fig.tight_layout(); fig.savefig(OUT/"ep30_horizon.png",dpi=130); plt.close()
# per-episode bars
names=[e.name for e in allep]; best=min(tab,key=lambda k:tab[k]["score"])
fig,ax=plt.subplots(figsize=(13,4.2)); x=np.arange(len(names)); wdt=.27
ax.bar(x-wdt,[tab[best]["val_by_episode"][n]["persistence_36"] for n in names],wdt,color=GRY,label="persistence")
ax.bar(x,[tab[best]["val_by_episode"][n]["36"] for n in names],wdt,color=ORA,label=best.replace("seed_cf_","seed: "))
ax.bar(x+wdt,[ceil[n].get("3h",np.nan) for n in names],wdt,color=GRN,label="ceiling")
ax.set_xticks(x); ax.set_xticklabels([n.replace("building_train_","b").replace("pid_water","pid").replace("all_off","off") for n in names],rotation=30,ha="right",fontsize=9)
ax.set_ylabel("3 h rollout RMSE, K"); ax.legend(frameon=False,ncol=3); fig.tight_layout(); fig.savefig(OUT/"ep30_by_episode.png",dpi=130); plt.close()
# measured vs predicted, one week
e=eps[("building_train_2","atlanta_3","pid_water")]; P=cf.profile(seeds[best],e,U,36); X=P["X"]; t=day(len(X)); a,bnd=7*288,14*288
fig,axs=plt.subplots(2,1,figsize=(12,6.4),sharex=True)
for ax,i,lab in zip(axs,P["rooms"],["most exposed exterior room","largest interior room"]):
    ax.plot(t[a:bnd],X[a:bnd,i]-273.15,color=INK,lw=1.6,label="simulator")
    m=(P["hz_t"]>=a)&(P["hz_t"]<bnd); ax.plot(P["hz_t"][m]/288,P["hz"][m,i]-273.15,color=ORA,lw=1.1,label="seed, 3 h ahead from measured history")
    ax.plot(t[a+1:bnd],P["one"][a:bnd-1,i]-273.15,color=BLU,lw=.8,ls=":",label="seed, 5 min ahead")
    ax.set_title(f"room #{i}: {lab}",fontsize=10,loc="left"); ax.set_ylabel("°C")
axs[0].legend(frameon=False,fontsize=9,ncol=3,loc="upper left"); axs[1].set_xlabel("day of the episode")
fig.suptitle(f"Week 2 of the 30-day episode: building_train_2, Atlanta, pid_water — {best.replace('seed_cf_','seed ')}, nothing fitted",fontsize=11)
fig.tight_layout(); fig.savefig(OUT/"ep30_profile_pred.png",dpi=130); plt.close()
json.dump(res,open(JS,"w"),indent=1,default=float); print("done")
