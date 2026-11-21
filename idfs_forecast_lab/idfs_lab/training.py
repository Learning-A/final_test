"""Train only explicit, disjoint runs. Heavy dependencies are optional, never auto-installed."""
import csv
import hashlib
import json
from pathlib import Path
import pickle
import math

from .evaluation import score
from .io import save

FEATURES=["lag1","lag2","lag3","lag4","lag5","rolling_mean"]


def read_dataset(path, target):
    with Path(path).open(encoding="utf-8",newline="") as f:
        rows=[r for r in csv.DictReader(f) if all(str(r[k])==str(target[k]) for k in ("scope","entity","component"))]
    if not rows:raise ValueError("no rows for target")
    for r in rows:
        if int(r["feature_end_ns"])>=int(r["window_start_ns"]) or int(r["features_available_ns"])>=int(r["window_start_ns"]):
            raise ValueError("noncausal training example")
    return rows


def train(train_paths, validation_paths, target, model_name, destination, protocol,epochs=100,cpu_threads=4):
    train_rows=[r for p in train_paths for r in read_dataset(p,target)]
    val_rows=[r for p in validation_paths for r in read_dataset(p,target)]
    train_ids={r["run_id"] for r in train_rows};val_ids={r["run_id"] for r in val_rows}
    if train_ids & val_ids:raise ValueError("train/validation run leakage")
    # Duplicate paths cannot disguise copies of the same run/window.
    keys=[(r["run_id"],r["window_start_ns"]) for r in train_rows+val_rows]
    if len(set(keys))!=len(keys):raise ValueError("duplicated training/validation examples")
    from .placement import require_training_host
    require_training_host()
    x=[[float(r[k]) for k in FEATURES] for r in train_rows];y=[float(r["target_iops"]) for r in train_rows]
    vx=[[float(r[k]) for k in FEATURES] for r in val_rows]
    if not all(math.isfinite(v) and v>=0 for row in x+vx for v in row) or not all(math.isfinite(v) and v>=0 for v in y+[float(r["target_iops"]) for r in val_rows]):
        raise ValueError("training features/targets must be finite and nonnegative")
    estimator=None;extra={}
    destination=Path(destination);destination.mkdir(parents=True,exist_ok=False)
    if model_name=="last_value":pred=[r[0] for r in vx]
    elif model_name=="rolling_mean":pred=[r[-1] for r in vx]
    elif model_name in ("gru","tcn"):
        from .models_torch import fit_sequence
        artifact=destination/"model.ts"
        pred,extra=fit_sequence(x,y,vx,[float(r["target_iops"]) for r in val_rows],model_name,artifact,epochs,cpu_threads)
    else:
        if model_name=="ridge":
            from sklearn.pipeline import make_pipeline
            from sklearn.preprocessing import StandardScaler
            from sklearn.linear_model import Ridge
            estimator=make_pipeline(StandardScaler(),Ridge(alpha=1.0))
        elif model_name=="hist_gradient_boosting":
            from sklearn.ensemble import HistGradientBoostingRegressor
            estimator=HistGradientBoostingRegressor(random_state=2026,max_iter=200)
        elif model_name=="lightgbm":
            from lightgbm import LGBMRegressor
            estimator=LGBMRegressor(n_estimators=400,num_leaves=31,random_state=2026,n_jobs=cpu_threads,verbosity=-1)
        elif model_name=="xgboost":
            from xgboost import XGBRegressor
            estimator=XGBRegressor(n_estimators=300,max_depth=6,random_state=2026,n_jobs=cpu_threads)
        else:raise ValueError("model inventory entry has no training adapter: "+model_name)
        import numpy as np
        estimator.fit(np.asarray(x),np.asarray(y))
        pred=list(estimator.predict(np.asarray(vx)))
    metrics=score([(max(0,float(p)),float(r["target_iops"])) for p,r in zip(pred,val_rows)],protocol["mask"])
    if model_name in ("gru","tcn"):
        pass
    elif estimator is None:
        artifact=destination/"model.json";save(artifact,{"model":model_name,"features":FEATURES})
    else:
        artifact=destination/"model.pkl"
        with artifact.open("wb") as f:pickle.dump(estimator,f,protocol=4)
    metadata={"model":model_name,"artifact":artifact.name,"sha256":hashlib.sha256(artifact.read_bytes()).hexdigest(),
              "target":target,"feature_order":FEATURES,"training_runs":sorted(train_ids),
              "validation_runs":sorted(val_ids),"validation":metrics,"test_runs":[],
              "data_inputs":[{"path":str(p),"sha256":hashlib.sha256(Path(p).read_bytes()).hexdigest()} for p in train_paths+validation_paths],
              "formal_acceptance":False,"state":"trained_on_supplied_data_not_formal_test",
              "training_host":"lmj@b507","training_details":extra,"cpu_threads":cpu_threads}
    save(destination/"model_manifest.json",metadata)
    return metadata
