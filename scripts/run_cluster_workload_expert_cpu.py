"""Compare shared and per-workload expert cluster-I/O forecasts."""
import argparse, hashlib, json, time
from pathlib import Path
import numpy as np

H = (1, 5, 10)
WORKLOADS = ("ai_inference_ses", "ai_training_ses", "bigdata_baleen", "graph_zipf", "hpc_wrf")

def sha(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def metric(y, p):
    y=np.asarray(y,float); p=np.asarray(p,float); e=np.abs(p-y); pos=y>0
    yc=y-y.mean(); pc=p-p.mean(); ss=float((yc**2).sum())
    return {"mae":float(e.mean()),"rmse":float(np.sqrt((e**2).mean())),
      "mape":float(100*(e[pos]/y[pos]).mean()) if pos.any() else None,
      "wape":float(100*e.sum()/y.sum()) if y.sum() else None,
      "r2":float(1-(e**2).sum()/ss) if ss else None,
      "pearson":float((yc@pc)/np.sqrt((yc**2).sum()*(pc**2).sum())) if ss and (pc**2).sum() else None,
      "n":int(len(y))}

def scores(y,p,work):
    groups={w:{str(h):metric(y[work==w,:h].mean(1),p[work==w,:h].mean(1)) for h in H} for w in WORKLOADS}
    macro={str(h):{k:float(np.mean([groups[w][str(h)][k] for w in WORKLOADS])) for k in ("mae","rmse","mape","wape","r2","pearson")} for h in H}
    pooled={str(h):metric(y[:,:h].mean(1),p[:,:h].mean(1)) for h in H}
    return groups,macro,pooled

def join_rows(i1_ns,horizon_ns):
    pos={int(v):i for i,v in enumerate(i1_ns)}; idx=np.array([pos.get(int(v),-1) for v in horizon_ns],dtype=int)
    if (idx<0).any(): raise ValueError(f"unmatched timestamps: {(idx<0).sum()}")
    if len(np.unique(idx))!=len(idx): raise ValueError("duplicate timestamp join")
    return idx

def fit_predict(xfit,yfit,xdev,wfit,wdev,experts,trees):
    from sklearn.ensemble import ExtraTreesRegressor
    p=np.zeros((len(xdev),10),dtype=float); models={}
    def make(): return ExtraTreesRegressor(n_estimators=trees,max_depth=8,min_samples_leaf=3,random_state=2026,criterion="squared_error",n_jobs=4)
    if experts:
        for w in WORKLOADS:
            m=make(); m.fit(xfit[wfit==w],yfit[wfit==w]); p[wdev==w]=m.predict(xdev[wdev==w]); models[w]=m
    else:
        m=make(); m.fit(xfit,yfit); p=m.predict(xdev); models["shared"]=m
    return p,models

def main():
    ap=argparse.ArgumentParser(); ap.add_argument("--i1",type=Path,required=True); ap.add_argument("--horizon",type=Path,required=True); ap.add_argument("--metadata",type=Path,required=True); ap.add_argument("--sample-ids",type=Path,required=True); ap.add_argument("--output",type=Path,required=True); ap.add_argument("--verify-only",action="store_true"); a=ap.parse_args()
    if a.output.exists(): raise ValueError("output exists; use a new versioned path")
    zi=np.load(a.i1,allow_pickle=False); zh=np.load(a.horizon,allow_pickle=False); meta=json.loads(a.metadata.read_text(encoding="utf-8")); ids=json.loads(a.sample_ids.read_text(encoding="utf-8"))
    fi=join_rows(zi["fit_available_ns"],zh["fit_available_ns"]); di=join_rows(zi["internal_dev_available_ns"],zh["internal_dev_available_ns"])
    def workload(split): return np.asarray([meta[r]["workload"] for r,_ in ids[split]])
    wf=workload("fit"); wd=workload("internal_dev")
    if set(wf)!=set(WORKLOADS) or set(wd)!=set(WORKLOADS): raise ValueError("workload set mismatch")
    def features(name):
        ks=["fit_features"]; ds=["internal_dev_features"]
        if name=="CBP": ks += ["fit_B","fit_P"]; ds += ["internal_dev_B","internal_dev_P"]
        xf=np.concatenate([zi[k][fi].reshape(len(fi),-1) for k in ks],axis=1).astype(np.float32)
        xd=np.concatenate([zi[k][di].reshape(len(di),-1) for k in ds],axis=1).astype(np.float32)
        return xf,xd
    yfit=zh["fit_seconds"].astype(float); ydev=zh["internal_dev_seconds"].astype(float)
    if a.verify_only:
        result={"state":"verified_preparation","fit_samples":len(yfit),"internal_dev_samples":len(ydev),"fit_workload_counts":{w:int((wf==w).sum()) for w in WORKLOADS},"internal_dev_workload_counts":{w:int((wd==w).sum()) for w in WORKLOADS},"feature_dimensions":{"C":int(8*258),"CBP":int(8*(258+180+80))},"target_shape":{"fit":list(yfit.shape),"internal_dev":list(ydev.shape)},"join":{"fit_exact":True,"internal_dev_exact":True,"fit_index_sha256":hashlib.sha256(fi.tobytes()).hexdigest(),"internal_dev_index_sha256":hashlib.sha256(di.tobytes()).hexdigest()},"input_sha256":{"i1":sha(a.i1),"horizon":sha(a.horizon),"metadata":sha(a.metadata),"sample_ids":sha(a.sample_ids)},"fit_performed":False}
        a.output.write_text(json.dumps(result,indent=2,ensure_ascii=False)+"\n",encoding="utf-8"); print(json.dumps(result,indent=2,ensure_ascii=False)); return
    out={"state":"partial","input_sha256":{"i1":sha(a.i1),"horizon":sha(a.horizon),"metadata":sha(a.metadata),"sample_ids":sha(a.sample_ids)},"fit_samples":len(yfit),"internal_dev_samples":len(ydev),"workload_counts":{"fit":{w:int((wf==w).sum()) for w in WORKLOADS},"internal_dev":{w:int((wd==w).sum()) for w in WORKLOADS}},"join":{"fit_exact":True,"internal_dev_exact":True,"fit_index_sha256":hashlib.sha256(fi.tobytes()).hexdigest(),"internal_dev_index_sha256":hashlib.sha256(di.tobytes()).hexdigest()},"targets":{"source":"cluster_horizon/prepared.npz","units":"cluster IOPS","horizons_seconds":[1,5,10],"aggregation":"mean of 1-second trajectory"},"models":{},"formal_acceptance":False,"new_external_validation_evaluations":0}
    for name in ("C","CBP"):
        xf,xd=features(name)
        for experts,trees,suffix in ((False,64,"shared_64"),(True,64,"workload_experts_64"),(False,320,"shared_capacity_320")):
            key=f"{name}_{suffix}"; started=time.monotonic(); p,models=fit_predict(xf,yfit,xd,wf,wd,experts,trees); groups,macro,pooled=scores(ydev,p,wd); pp=a.output.parent/(key+"_predictions.npy"); np.save(pp,p)
            out["models"][key]={"experts":experts,"trees_per_model":trees,"total_trees":trees*len(models),"feature_dim":int(xf.shape[1]),"fit_seconds":time.monotonic()-started,"groups":groups,"macro":macro,"pooled":pooled,"prediction_sha256":sha(pp),"prediction_path":pp.name}
            print(key,json.dumps(macro,ensure_ascii=False),flush=True)
    out["state"]="completed"; a.output.write_text(json.dumps(out,indent=2,ensure_ascii=False)+"\n",encoding="utf-8"); print(json.dumps(out,indent=2,ensure_ascii=False))

if __name__=="__main__": main()
