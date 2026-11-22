"""Evidence figures for the decision slides (existing 30-day results; nothing new generated)."""
import sys, json, pickle
from pathlib import Path
import numpy as np, pandas as pd, matplotlib
matplotlib.use("Agg"); import matplotlib.pyplot as plt
WC=Path("/Users/srivathsan/Desktop/smart-buildings/working-code"); S=Path(sys.argv[1]); OUT=Path(sys.argv[2])
sys.path.insert(0,str(WC/"code"))
from discovery import closedform as cf
from control import validate_costs as vc
from control.occupancy import OccupancyForecaster
plt.rcParams.update({"font.size":12,"axes.spines.top":False,"axes.spines.right":False})
ORA,BLU,GRY,INK,GRN,PUR="#D9692A","#2F72B5","#8A93A3","#13213A","#3a9a5b","#8e44ad"
res={}
# (a) three seeds on the 12 thirty-day episodes
eps=pickle.load(open(S/"ep30_eps.pkl","rb")); U,_=cf.universal_constants(); H=(12,36,72)
seeds={"one_node":"seed_cf_one_node","history_slab":"seed_cf_history_slab","network_observer":"seed_cf_network_observer"}
for k,f in seeds.items():
    r=cf.evaluate((WC/f"code/discovery/seeds_cf/{f}.py").read_text(),eps[:1],eps,U,every=12,screen_factor=1e9,max_train=1)
    res[k]={kk:v for kk,v in r.items() if kk.startswith("val_") and isinstance(v,float)}; res[k]["status"]=r["status"]; print(k,r["status"],flush=True)
ceil={"1h":0.312,"3h":0.459,"6h":0.617}
fig,axs=plt.subplots(1,2,figsize=(13,4.6),gridspec_kw=dict(width_ratios=[1.4,1]))
hh=[1,3,6]; p0=res["network_observer"]
axs[0].plot(hh,[p0[f"val_persistence_{h}h_mean"] for h in hh],"s--",color=GRY,label="persistence")
for (k,lab),c in zip([("network_observer","stateful network (measurement-corrected)"),("history_slab","history buffer"),("one_node","one node, nominal")],[ORA,BLU,PUR]):
    axs[0].plot(hh,[res[k][f"val_{h}h_mean"] for h in hh],"o-",color=c,label=lab)
axs[0].plot(hh,[ceil[f"{h}h"] for h in hh],"^:",color=GRN,label="ceiling (fitted per room, diagnostic)")
axs[0].set_ylim(0,2.2); axs[0].set_xticks(hh); axs[0].set_xlabel("horizon, h"); axs[0].set_ylabel("RMSE, K (true future inputs)"); axs[0].legend(frameon=False,fontsize=9)
axs[0].set_title("Seeds on 12 thirty-day episodes (one-node off the chart: 3.6 / 6.6 / 7.9 K)",fontsize=10,loc="left")
names=["network_observer","history_slab"]; x=np.arange(2)
axs[1].bar(x-0.2,[res[k]["val_3h_mean"] for k in names],0.4,color=BLU,label="true future inputs (diagnostic)")
axs[1].bar(x+0.2,[res[k]["val_3h_deploy_mean"] for k in names],0.4,color=ORA,label="test-time: future weather/occupancy unknown (the score)")
axs[1].axhline(p0["val_persistence_3h_mean"],color=GRY,ls="--",lw=1); axs[1].text(1.45,p0["val_persistence_3h_mean"]+0.02,"persistence",fontsize=9,color=GRY,ha="right")
axs[1].set_xticks(x); axs[1].set_xticklabels(["stateful network","history buffer"]); axs[1].set_ylabel("3 h RMSE, K"); axs[1].legend(frameon=False,fontsize=8,loc="lower right")
fig.tight_layout(); fig.savefig(OUT/"seeds_compare.png",dpi=130); plt.close()
# (b) cost model vs the kit's logged costs
fig,axs=plt.subplots(1,2,figsize=(13,4.6)); R=S/"ep30/results"
for run,c in [("building_train_2/atlanta_3/pid_water",BLU),("building_train_1/fairbanks_1/pid_water",ORA)]:
    st,A=vc.check(R,run,arrays=True); res["cost_"+run]={k:v for k,v in st.items()}
    lab=run.split("/")[0].replace("building_train_","b")+" "+run.split("/")[1]
    axs[0].scatter(A["kit_e"]/1e3,A["est_e"]/1e3,s=3,alpha=.25,color=c,label=f"{lab}: r = {st['electric W'][0]:.3f}")
    axs[1].scatter(A["kit_g"]/1e3,A["est_g"]/1e3,s=3,alpha=.25,color=c,label=f"{lab}: r = {st['gas W'][0]:.3f}")
for ax,t in zip(axs,["electric power (fans, cooling, pump), kW","gas power (reheat), kW"]):
    lim=ax.get_xlim()[1]; ax.plot([0,lim],[0,lim],color=GRY,lw=1); ax.set_xlabel("implied by the kit's logged cost, "+t.split(",")[-1].strip()); ax.set_ylabel("our estimate from commands"); ax.set_title(t,fontsize=11,loc="left"); ax.legend(frameon=False,fontsize=9,markerscale=4)
fig.tight_layout(); fig.savefig(OUT/"cost_validation.png",dpi=130); plt.close()
# (c) occupancy forecast vs actual (30 days)
b,w="building_train_2","atlanta_3"; rc=pd.read_csv(R/b/w/"pid_water"/"reward_components.csv"); occ=rc.total_occupancy.values; ts=pd.to_datetime(rc.timestamp)-pd.Timedelta(minutes=5)
f=OccupancyForecaster(85); fc=np.full(len(occ),np.nan)
for k in range(len(occ)-36):
    if k>=2:
        t2=ts.iloc[k-2]; f.observe(occ[k-2],(t2.hour*60+t2.minute)//5,t2.dayofweek<5)
    t=ts.iloc[k+36]; fc[k+36]=f.total((t.hour*60+t.minute)//5,t.dayofweek<5,1)[0]
fig,ax=plt.subplots(figsize=(12,3.8)); d=np.arange(len(occ))/288
ax.plot(d,occ,color=INK,lw=1,label="observed head count"); ax.plot(d,fc,color=ORA,lw=1,label="forecast made 3 h earlier")
ax.set_xlabel("day"); ax.set_ylabel("people in the building"); ax.legend(frameon=False,ncol=2)
ax.set_title("Occupancy forecast: schedule prior, then the building's own weekday profile (building_train_2, atlanta_3)",fontsize=10,loc="left")
fig.tight_layout(); fig.savefig(OUT/"occupancy_forecast.png",dpi=130); plt.close()
# (d) controllers vs pid_water (2-day checks, kit harness)
vals={"pid_water":(0.02497,0.02309),"feed-forward":(0.02497,0.02268),"MPC":(0.02501,0.02089)}
fig,ax=plt.subplots(figsize=(8,4.2)); x=np.arange(2)
for i,(k,v) in enumerate(vals.items()):
    ax.bar(x+(i-1)*0.26,v,0.26,color=[GRY,BLU,ORA][i],label=k)
    for j,y in enumerate(v): ax.text(x[j]+(i-1)*0.26,y+0.0003,f"{y:.5f}",ha="center",fontsize=8,rotation=90)
ax.set_xticks(x); ax.set_xticklabels(["building_train_8 · atlanta_1","building_train_1 · fairbanks_1"]); ax.set_ylabel("score (mean loss per step, lower is better)")
ax.set_ylim(0,0.031); ax.legend(frameon=False,ncol=3,loc="upper center"); ax.set_title("2-day checks through the kit's harness (576 steps)",fontsize=10,loc="left")
fig.tight_layout(); fig.savefig(OUT/"controllers.png",dpi=130); plt.close()
json.dump(res,open(S/"deck/figs6_results.json","w"),indent=1,default=float); print(json.dumps({k:{kk:(round(vv,3) if isinstance(vv,float) else vv) for kk,vv in v.items()} for k,v in res.items() if not k.startswith("cost")},indent=0)[:1500])
