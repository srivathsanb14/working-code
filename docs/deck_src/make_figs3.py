"""Data-grounded figures. Sources: working-code/outputs (kept simulator audit files), scratch kit runs (gen/), kit files."""
import sys, json, glob, re, os
from pathlib import Path
import numpy as np, pandas as pd, matplotlib
matplotlib.use("Agg"); import matplotlib.pyplot as plt
WC=Path("/Users/srivathsan/Desktop/smart-buildings/working-code"); KIT=WC.parent/"smart-buildings-starter-kit"
OUT=Path(sys.argv[1]); GEN=Path(sys.argv[2])
sys.path.insert(0,str(WC/"code"))
from discovery.episodes import Episode, geometry
from discovery import closedform as cf
plt.rcParams.update({"font.size":12,"axes.spines.top":False,"axes.spines.right":False})
ORA,BLU,GRY,INK,GRN="#D9692A","#2F72B5","#8A93A3","#13213A","#3a9a5b"
U,_=cf.universal_constants()
res={}
# ---------------------------------------------------------------- real episodes from the audit files
def episode(path):
    z=np.load(path,allow_pickle=True); names=[str(x) for x in z["names"]]; ix={n:i for i,n in enumerate(names)}
    rooms=[str(r) for r in z["rooms"]]; ahu=json.loads(str(z["ahu_json"]))
    geo=geometry(z["floor_plan"],z["zone_map"],rooms,ahu)
    O=z["obs"][:-1]; n=len(z["Tamb"])
    def col(p): return O[:,[i for nm,i in ix.items() if re.fullmatch(p,nm)][0]]
    inp=dict(T_out=z["Tamb"],g=z["g"],T_supply=np.where(z["g"]>0,z["Tsup"],294.0),Q_reheat=z["Qroom"],Q_gain=np.zeros_like(z["Qroom"]),
             occupants=np.zeros(n),comfort=col(r"comfort_mode_now").astype(float)>0,hour_sin=col(r"hod_sin_000"),hour_cos=col(r"hod_cos_000"),
             dow_sin=col(r"dow_sin_000"),dow_cos=col(r"dow_cos_000"))
    inp["comfort"]=inp["comfort"].astype(float)
    nm=Path(path).stem.replace("roomdata_","")
    b="_".join(nm.split("_")[:3]); w="_".join(nm.split("_")[3:5]); p=nm.split("_")[-1]
    return Episode(name=nm,building=b,weather=w,policy=p,scenario=f"{b}/{w}",geo=geo,inputs=inp,T_meas=z["Ttrue_after"],T0=float(np.mean(z["T0"]))), z
paths=sorted(glob.glob(str(WC/"outputs/roomdata_*.npz")))
eps=[]; raw={}
for p in paths:
    e,z=episode(p); eps.append(e); raw[e.name]=z
print("episodes",len(eps))
seeds={p.stem:p.read_text() for p in sorted((WC/"code/discovery/seeds_cf").glob("*.py"))}
H=(12,36,72); tab={}
for nm,src in seeds.items():
    r=cf.evaluate(src,eps[:2],eps,U,horizons=H,every=6,screen_factor=1e9)
    tab[nm]=r; print(nm,r["status"],{k:round(v,3) for k,v in r.items() if k.startswith("val_") and isinstance(v,float)})
ceil=[cf.ceiling(e,U,H,6) for e in eps]
res["seeds_real"]={nm:{k:v for k,v in r.items() if k.startswith(("val_","stage1")) and isinstance(v,float)} for nm,r in tab.items()}
res["ceiling_real"]={k:float(np.mean([c[k] for c in ceil])) for k in ceil[0]}
by_pol={}
for nm,r in tab.items():
    for en,v in r["val_by_episode"].items():
        by_pol.setdefault(nm,{}).setdefault(en.split("_")[-1],[]).append(v["36"])
res["seed_3h_by_policy"]={nm:{p:float(np.mean(v)) for p,v in d.items()} for nm,d in by_pol.items()}
# horizon plot on real data
fig,ax=plt.subplots(figsize=(7.5,4.3)); hh=[h/12 for h in H]
r0=next(iter(tab.values()))
ax.plot(hh,[r0[f"val_persistence_{cf._hname(h)}_mean"] for h in H],"s--",color=GRY,label="persistence (temperature stays)")
for (nm,r),c in zip(tab.items(),[ORA,BLU]):
    ax.plot(hh,[r[f"val_{cf._hname(h)}_mean"] for h in H],"o-",color=c,label=nm.replace("seed_cf_","seed: ")+" (no fitting)")
ax.plot(hh,[res["ceiling_real"][cf._hname(h)] for h in H],"^:",color=GRN,label="ceiling: fitted per room (diagnostic)")
ax.set_xlabel("forecast horizon, hours"); ax.set_ylabel("RMSE vs simulator, K"); ax.set_xticks(hh); ax.legend(frameon=False,fontsize=10); ax.grid(alpha=.3)
ax.set_title(f"{len(eps)} simulator episodes, 6 shipped buildings, 3 policies",fontsize=11,loc="left")
fig.tight_layout(); fig.savefig(OUT/"real_horizon.png",dpi=130); plt.close()
# real profile: best seed on building_train_2 atlanta pidwater
best=min(tab,key=lambda k:tab[k]["score"]); e=[x for x in eps if x.name=="building_train_2_atlanta_3_pidwater"][0]
P=cf.profile(seeds[best],e,U,36); X=P["X"]; t=np.arange(len(X))/288
fig,axs=plt.subplots(2,1,figsize=(11,6.2),sharex=True)
for ax,i,lab in zip(axs,P["rooms"],["most exposed exterior room","largest interior room"]):
    on=e.inputs["g"].mean(1)>50
    for s0,e0 in [(a,b) for a,b in zip(np.where(np.diff(np.r_[0,on.astype(int)])==1)[0],np.where(np.diff(np.r_[on.astype(int),0])==-1)[0]+1)]:
        ax.axvspan(s0/288,e0/288,color="#f2c14e",alpha=.15,lw=0)
    ax.plot(t,X[:,i]-273.15,color=INK,lw=1.8,label="simulator (measured)")
    ax.plot(t[1:],P["one"][:,i]-273.15,color=BLU,lw=1,ls=":",label="seed, 5 min ahead")
    ax.plot(P["hz_t"]/288,P["hz"][:,i]-273.15,color=ORA,lw=1.2,label="seed, 3 h ahead from measured history")
    ax.set_ylabel("°C"); ax.set_title(f"room #{i}: {lab}",fontsize=10,loc="left")
axs[0].legend(frameon=False,fontsize=9,ncol=3,loc="upper left"); axs[1].set_xlabel("days (shaded: HVAC air on)")
fig.suptitle(f"Real simulator data: building_train_2, Atlanta, pid_water — {best.replace('seed_cf_','seed ')}, nothing fitted",fontsize=11)
fig.tight_layout(); fig.savefig(OUT/"real_profile.png",dpi=130); plt.close()
res["profile_episode"]=e.name; res["profile_seed"]=best
# ---------------------------------------------------------------- all observation variable groups (one real episode)
z=raw["building_train_2_atlanta_3_pidwater"]; names=[str(x) for x in z["names"]]; O=z["obs"]; n=len(O); tt=np.arange(n)/288
groups=[("room air temperature (V)",r"VAV_\d+_zone_air_temperature_sensor"),("damper position (V)",r"VAV_\d+_supply_air_damper_percentage_command"),
        ("reheat valve (V)",r"VAV_\d+_reheat_valve_percentage_command"),("comfort band: heating setpoint (V)",r"VAV_\d+_zone_air_heating_temperature_setpoint_sensor"),
        ("AHU supply-air temperature (K)",r"ahs_ahu_\d+_supply_air_temperature_sensor"),("AHU supply airflow (K)",r"ahs_ahu_\d+_supply_air_flowrate_sensor"),
        ("AHU static pressure (K)",r"ahs_ahu_\d+_supply_air_static_pressure_sensor"),("AHU recirculation (K)",r"ahs_ahu_\d+_recirculation_percentage_command"),
        ("boiler supply water",r"hws_supply_water_temperature_sensor"),("outside air",r"ahs_ahu_1_outside_air_temperature_sensor"),
        ("occupants (building total)",r"num_occupants"),("comfort mode now / soon",r"comfort_mode_(now|soon)"),("hour of day (sin, cos)",r"hod_(sin|cos)_000"),("day of week (sin, cos)",r"dow_(sin|cos)_000")]
fig,axs=plt.subplots(7,2,figsize=(13,13),sharex=True)
for ax,(lab,pat) in zip(axs.T.flat,groups):
    idx=[i for i,nm in enumerate(names) if re.fullmatch(pat,nm)]
    ax.plot(tt,O[:,idx],lw=.6,alpha=.6 if len(idx)>3 else 1,color=BLU if len(idx)>3 else INK)
    ax.set_title(f"{lab}  [{len(idx)} field{'s' if len(idx)>1 else ''}]",fontsize=10,loc="left"); ax.tick_params(labelsize=8)
for ax in axs[-1]: ax.set_xlabel("days")
fig.suptitle("Every observation group, as the policy receives it (normalised), building_train_2 Atlanta pid_water",fontsize=12)
fig.tight_layout(); fig.savefig(OUT/"obs_groups.png",dpi=110); plt.close()
# ---------------------------------------------------------------- big generated building (kit run into scratch)
big=GEN/"results/building_0903/atlanta_1/pid_water"; ts=pd.read_csv(big/"timeseries.csv"); rc=pd.read_csv(big/"reward_components.csv")
Z=ts[[c for c in ts.columns if re.fullmatch(r"zone_\d+_temp",c)]].values-273.15; t2=np.arange(len(Z))/288
fig,ax=plt.subplots(figsize=(11,4.2)); ax.plot(t2,Z,lw=.4,alpha=.35,color=BLU)
ax.plot(t2,np.median(Z,1),color=INK,lw=2,label="median room"); ax.axhspan(21,24,color="#cfe8d4",alpha=.5,lw=0)
ax.set_xlabel("days"); ax.set_ylabel("room temperature, °C"); ax.legend(frameon=False)
ax.set_title(f"Generated building seed 903: {Z.shape[1]} rooms, 21 AHUs, pid_water, Atlanta (kit evaluate.py)",fontsize=11,loc="left")
fig.tight_layout(); fig.savefig(OUT/"big_building.png",dpi=130); plt.close()
fig,axs=plt.subplots(3,1,figsize=(11,5.6),sharex=True); t3=np.arange(len(rc))/288
axs[0].plot(t3,rc["agent_reward"],color=INK); axs[0].set_ylabel("reward")
axs[1].plot(t3,0.6*rc["normalized_productivity_regret"],color=ORA,label="0.6 × comfort regret"); axs[1].plot(t3,-0.2*rc["normalized_energy_cost"],color=BLU,label="−0.2 × energy"); axs[1].plot(t3,-0.2*rc["normalized_carbon_emission"],color=GRY,label="−0.2 × carbon"); axs[1].legend(frameon=False,fontsize=9,ncol=3); axs[1].set_ylabel("terms")
axs[2].plot(t3,rc["total_occupancy"],color=GRN); axs[2].set_ylabel("occupants"); axs[2].set_xlabel("days")
fig.suptitle("Where the reward comes from, step by step (same run; kit reward_components.csv)",fontsize=11)
fig.tight_layout(); fig.savefig(OUT/"reward_terms.png",dpi=130); plt.close()
# ---------------------------------------------------------------- generator distribution
B=json.load(open(GEN/"sample_buildings.json"))
shipped={"building_train_8":(28,3),"building_train_1":(39,4),"building_train_2":(85,9)}
fig,axs=plt.subplots(1,3,figsize=(13,3.8))
axs[0].hist([b["V"] for b in B],bins=15,color=BLU); axs[0].set_xlabel("rooms (V)"); axs[0].set_ylabel("buildings")
axs[1].hist([b["K"] for b in B],bins=range(3,23),color=ORA); axs[1].set_xlabel("air handlers (K)")
Uf=[1/(0.1/b["exterior_cv_conductivity"]+0.299) for b in B]; Cap=[b["exterior_cv_density"]*b["exterior_cv_heat_capacity"]/1e6 for b in B]
axs[2].scatter(Uf,Cap,color=INK,s=18); axs[2].set_xlabel("facade U, W/m²K"); axs[2].set_ylabel("facade heat capacity, MJ/m³K"); axs[2].set_yscale("log")
for ax in axs[:2]:
    for nm,(v,k) in shipped.items(): pass
axs[0].axvline(85,color=GRY,ls="--"); axs[0].text(87,axs[0].get_ylim()[1]*.85,"largest shipped (85)",fontsize=9,color=GRY)
axs[1].axvline(9,color=GRY,ls="--"); axs[1].text(9.3,axs[1].get_ylim()[1]*.85,"largest shipped (9)",fontsize=9,color=GRY)
fig.suptitle(f"{len(B)} buildings from the kit generator (seeds 900–959, {60-len(B)} of 60 seeds rejected by the kit)",fontsize=11)
fig.tight_layout(); fig.savefig(OUT/"generator_dist.png",dpi=130); plt.close()
# ---------------------------------------------------------------- comfort deviation, measured as the reward does (real episode)
z=raw["building_train_2_atlanta_3_pidwater"]; ix={nm:i for i,nm in enumerate(names)}
T=z["Tsensor"]; rooms=[str(r) for r in z["rooms"]]
lo=np.stack([z["obs"][1:,ix[f"VAV_{r.split('_')[1]}_zone_air_heating_temperature_setpoint_sensor"]] for r in rooms],1)
hi=np.stack([z["obs"][1:,ix[f"VAV_{r.split('_')[1]}_zone_air_cooling_temperature_setpoint_sensor"]] for r in rooms],1)
dev=np.maximum(lo-T,0)+np.maximum(T-hi,0); occ=z["occ_true"]
fig,ax=plt.subplots(figsize=(11,3.8)); t4=np.arange(len(T))/288
ax.plot(t4,(dev*occ).sum(1)/np.maximum(occ.sum(1),1e-9),color=ORA,lw=1.5,label="occupancy-weighted K outside the band")
ax.plot(t4,dev.mean(1),color=GRY,lw=1,label="unweighted mean over rooms")
ax.set_xlabel("days"); ax.set_ylabel("K outside comfort band"); ax.legend(frameon=False)
ax.set_title("How comfort deviation is measured (real episode: building_train_2 Atlanta pid_water)",fontsize=11,loc="left")
fig.tight_layout(); fig.savefig(OUT/"comfort_deviation.png",dpi=130); plt.close()
json.dump(res,open(OUT.parent.parent/"figs3_results.json","w"),indent=1)
print(json.dumps(res,indent=1)[:2500])
