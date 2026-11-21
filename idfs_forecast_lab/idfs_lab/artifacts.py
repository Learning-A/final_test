"""Export trained models to CPU inference and gate native/exported equivalence."""
import hashlib
import json
from pathlib import Path
import pickle
import tarfile

from .io import load,save
from .runtime import Predictor
from .training import read_dataset,FEATURES


def lightgbm_tree(node):
    if "leaf_value" in node:return {"leaf":float(node["leaf_value"])}
    if node.get("decision_type")!="<=":raise ValueError("only numeric LightGBM splits supported")
    return {"feature":int(node["split_feature"]),"threshold":float(node["threshold"]),
            "left":lightgbm_tree(node["left_child"]),"right":lightgbm_tree(node["right_child"])}


def xgboost_tree(node):
    if "leaf" in node:return {"leaf":float(node["leaf"])}
    # XGBoost uses '<'; nextafter preserves the strict comparison for finite doubles.
    import math
    feature=str(node["split"])
    if feature.startswith("f"):feature=feature[1:]
    if not feature.isdigit():raise ValueError("numeric XGBoost feature indices required")
    children={x["nodeid"]:x for x in node["children"]}
    return {"feature":int(feature),"threshold":math.nextafter(float(node["split_condition"]),-math.inf),
            "left":xgboost_tree(children[node["yes"]]),"right":xgboost_tree(children[node["no"]])}


def histgb_tree(nodes,index=0):
    node=nodes[index]
    if node["is_leaf"]:return {"leaf":float(node["value"])}
    if node["is_categorical"]:raise ValueError("categorical HistGB export unsupported")
    return {"feature":int(node["feature_idx"]),"threshold":float(node["num_threshold"]),
            "left":histgb_tree(nodes,int(node["left"])),"right":histgb_tree(nodes,int(node["right"]))}


def export_cpu(model_dir,validation_paths,destination):
    model_dir,destination=Path(model_dir),Path(destination)
    meta=load(model_dir/"model_manifest.json")
    artifact=model_dir/meta["artifact"]
    if hashlib.sha256(artifact.read_bytes()).hexdigest()!=meta["sha256"]:raise ValueError("model checksum mismatch")
    rows=[r for p in validation_paths for r in read_dataset(p,meta["target"])]
    if {r["run_id"] for r in rows}!=set(meta["validation_runs"]):raise ValueError("same frozen validation runs required for export gate")
    destination.mkdir(parents=True,exist_ok=False)
    x=[[float(r[k]) for k in FEATURES] for r in rows]
    name=meta["model"]
    if name in ("last_value","rolling_mean"):
        model={"kind":name}
        native=[v[0] if name=="last_value" else v[5] for v in x]
    elif name in ("gru","tcn"):
        import torch
        original=torch.jit.load(str(artifact),map_location="cpu").eval()
        original.save(str(destination/"weights.ts"))
        model={"kind":"torchscript","artifact":"weights.ts","torch_version":torch.__version__}
        with torch.inference_mode():native=original(torch.tensor(x,dtype=torch.float32)).reshape(-1).tolist()
    else:
        with artifact.open("rb") as f:estimator=pickle.load(f)
        import numpy as np
        native=estimator.predict(np.asarray(x)).tolist()
        if name=="ridge":
            scale,regressor=estimator.steps[0][1],estimator.steps[-1][1]
            coefficients=(regressor.coef_/scale.scale_).tolist()
            model={"kind":"linear","coefficients":coefficients,
                   "intercept":float(regressor.intercept_-sum(a*b for a,b in zip(coefficients,scale.mean_)))}
        elif name=="lightgbm":
            dump=estimator.booster_.dump_model()
            model={"kind":"tree_sum","base":0,"trees":[lightgbm_tree(t["tree_structure"]) for t in dump["tree_info"]]}
        elif name=="hist_gradient_boosting":
            model={"kind":"tree_sum","base":float(estimator._baseline_prediction.reshape(-1)[0]),
                   "trees":[histgb_tree(p.nodes) for iteration in estimator._predictors for p in iteration]}
        elif name=="xgboost":
            booster=estimator.get_booster();config=json.loads(booster.save_config())
            base=config["learner"]["learner_model_param"]["base_score"]
            parsed=json.loads(base) if str(base).startswith("[") else float(base)
            if isinstance(parsed,list):parsed=parsed[0]
            model={"kind":"tree_sum","base":float(parsed),"trees":[xgboost_tree(json.loads(t)) for t in booster.get_dump(dump_format="json")]}
        else:raise ValueError("CPU export adapter unavailable: "+name)
    save(destination/"cpu_model.json",model)
    predictor=Predictor(destination/"cpu_model.json")
    exported=[predictor.predict(row) for row in x]
    errors=[abs(max(0,float(a))-b) for a,b in zip(native,exported)]
    # Numeric tree exports must pass prediction equivalence, not merely parse.
    tolerance=1e-4 if name=="xgboost" else 1e-6
    equivalent=all(error<=tolerance*max(1,abs(float(a))) for error,a in zip(errors,native))
    gate={"passed":equivalent,"samples":len(errors),"max_abs_error":max(errors,default=0),"relative_tolerance":tolerance}
    save(destination/"equivalence.json",gate)
    if not equivalent:raise ValueError("CPU export equivalence failed; artifact is not deployable")
    save(destination/"parity_vectors.json",[{"features":row,"expected":value} for row,value in list(zip(x,exported))[:16]])
    deployment={"model":name,"target":meta["target"],"feature_order":FEATURES,
                "source_model_sha256":meta["sha256"],"training_runs":meta["training_runs"],
                "validation_runs":meta["validation_runs"],"validation":meta["validation"],
                "runtime":"torch_cpu" if name in ("gru","tcn") else "python_standard_library_cpu",
                "equivalence":gate,"fit_on_cluster":False,
                "files":{p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in destination.iterdir() if p.is_file()}}
    save(destination/"deployment.json",deployment)
    return deployment


INFERENCE_MODULES=["__init__.py","io.py","counters.py","dataset.py","joint.py","evaluation.py","journal.py","runtime.py","online.py","seal.py"]


def bundle_inference(export_dir,protocol_path,destination):
    export_dir,destination=Path(export_dir),Path(destination)
    deployment=load(export_dir/"deployment.json")
    if not deployment["equivalence"]["passed"]:raise ValueError("export gate not passed")
    destination.parent.mkdir(parents=True,exist_ok=True)
    if destination.exists():raise ValueError("bundle destination exists")
    content=[]
    for name,digest in deployment["files"].items():
        p=export_dir/name
        if hashlib.sha256(p.read_bytes()).hexdigest()!=digest:raise ValueError("export file checksum mismatch")
        content.append((p,"model/"+name))
    content.append((export_dir/"deployment.json","model/deployment.json"))
    content.append((Path(protocol_path),"config/protocol.json"))
    for name in INFERENCE_MODULES:
        p=Path(__file__).with_name(name)
        if not p.is_file():raise ValueError("missing inference runtime: "+name)
        content.append((p,"idfs_lab/"+name))
    with tarfile.open(destination,"x:gz") as archive:
        for source,name in content:archive.add(source,arcname=name,recursive=False)
    manifest={"bundle":destination.name,"sha256":hashlib.sha256(destination.read_bytes()).hexdigest(),
              "kind":"inference_only","contains_training_code":False,"runtime":deployment["runtime"],
              "files":[name for _,name in content]}
    save(str(destination)+".json",manifest)
    return manifest
