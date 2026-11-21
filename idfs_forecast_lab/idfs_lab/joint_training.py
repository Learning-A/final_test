"""GPU-only MoE fitting/export. No fit is performed by preparing a job plan."""
import copy
import hashlib
import json
import math
from pathlib import Path
from .io import load, read_events, save, canonical_hash
from .joint import TARGET, FEATURE_COUNT, topology
from .evaluation import score


def read_split(train_paths, validation_paths, study, site, target=None):
    target=target or TARGET
    splits = []
    seen = set()
    registered = {r["run_id"]: r for r in study["rounds"]}
    th = canonical_hash(topology(site))
    for paths, role in ((train_paths, "train"), (validation_paths, "validation")):
        rows = [r for path in paths for r in read_events(path) if r.get("target")==target]
        if not rows:
            raise ValueError("empty joint split")
        expected = {r["run_id"] for r in study["rounds"] if r["role"] == role}
        if {r["run_id"] for r in rows} != expected:
            raise ValueError("all registered 15 training / 5 validation runs required for pooled MoE")
        for row in rows:
            run = registered.get(row["run_id"], {})
            if run.get("role") != role or row.get("role") != role or row.get("workload") != run.get("workload"):
                raise ValueError("formal data or incorrect role/workload in fit input")
            key = (row["run_id"], row["window_start_ns"])
            if "start_ns" in run and not run["start_ns"]<=row["window_start_ns"]<run["end_ns"]:
                raise ValueError("example outside registered chronological smoke partition")
            if key in seen:
                raise ValueError("duplicate or leaked run/window")
            seen.add(key)
            if row.get("target") != target or row.get("topology_hash") != th:
                raise ValueError("target/topology mismatch")
            if row.get("attribution") != "storage_context_only":
                raise ValueError("unverified client-to-OSS attribution")
            start = row["window_start_ns"]
            if row["feature_end_ns"] > start-1000000000 or row["features_available_ns"] > start-30000000:
                raise ValueError("noncausal or late features")
            x, y = row["features"], row["target_read_write"]
            if len(x) != FEATURE_COUNT or len(y) != 2 or any(not math.isfinite(v) or v < 0 for v in x+y):
                raise ValueError("invalid joint tensor")
            masks = x[18:399:4]
            if any(v not in (0, 1) for v in masks+x[407:415]) or sum(masks) < 92:
                raise ValueError("invalid observation masks/coverage")
            if not math.isclose(sum(y), row["target_iops"], rel_tol=1e-8):
                raise ValueError("inconsistent target total")
        splits.append(rows)
    return splits


def train_joint(train_paths, validation_paths, destination, protocol, study, site, epochs=100, threads=4,target=None):
    target=target or TARGET
    from .placement import require_training_host
    root = require_training_host()
    if root not in Path(destination).resolve().parents:
        raise ValueError("output outside isolated lmj job")
    train, validation = read_split(train_paths, validation_paths, study, site,target)
    import torch
    from torch.nn import functional as F
    from .moe import JointMoE, training_scaler
    if not torch.cuda.is_available():
        raise RuntimeError("idle visible GPU required; no CPU fitting fallback")
    if not 1 <= threads <= 4 or not 1 <= epochs <= 1000:
        raise ValueError("invalid thread/epoch budget")
    torch.set_num_threads(threads); torch.set_num_interop_threads(1)
    torch.manual_seed(2026); torch.cuda.manual_seed_all(2026)
    x = torch.tensor([r["features"] for r in train], dtype=torch.float32, device="cuda:0")
    y = torch.tensor([r["target_read_write"] for r in train], dtype=torch.float32, device="cuda:0")
    vx = torch.tensor([r["features"] for r in validation], dtype=torch.float32, device="cuda:0")
    vy = torch.tensor([r["target_read_write"] for r in validation], dtype=torch.float32, device="cuda:0")
    mean, std = training_scaler(x)
    model = JointMoE(topology(site), mean, std).to("cuda:0")
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    best, best_loss, stale, history = None, float("inf"), 0, []

    def loss_for(detail, truth):
        total = F.smooth_l1_loss(torch.log1p(detail[:, 0]), torch.log1p(truth.sum(dim=1)))
        rw = F.smooth_l1_loss(torch.log1p(detail[:, 1:3]), torch.log1p(truth))
        balance = (detail[:, 3:6].mean(dim=0)-1.0/3).square().mean()
        return total+0.25*rw+0.001*balance

    for epoch in range(epochs):
        model.train(); order = torch.randperm(len(x), device=x.device)
        total_loss = 0.0
        for start in range(0, len(x), 128):
            batch = order[start:start+128]
            optimizer.zero_grad(set_to_none=True)
            loss = loss_for(model.details(x[batch]), y[batch])
            if not torch.isfinite(loss):
                raise RuntimeError("non-finite MoE training loss")
            loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0); optimizer.step()
            total_loss += float(loss.detach())*len(batch)
        model.eval()
        with torch.inference_mode():
            val_loss = float(loss_for(model.details(vx), vy))
        if not math.isfinite(val_loss):
            raise RuntimeError("non-finite MoE validation loss")
        history.append({"epoch": epoch+1, "train_loss": total_loss/len(x), "validation_loss": val_loss})
        print(json.dumps(history[-1]), flush=True)
        if val_loss < best_loss:
            best_loss, best, stale = val_loss, copy.deepcopy(model.state_dict()), 0
        else:
            stale += 1
        if stale >= 12:
            break
    model.load_state_dict(best); model.eval()
    with torch.inference_mode():
        fp32_native = model(vx).cpu().tolist()
    # Preserve learned weights while evaluating/exporting in float64. Small
    # log-output differences can be magnified by expm1 on poorly scaled smoke
    # inputs; use higher precision instead of widening the equivalence gate.
    model=model.double()
    with torch.inference_mode():
        native = model(vx.double()).cpu().tolist()
    model = model.cpu().eval()
    scripted = torch.jit.script(model)
    cpu_x = vx.cpu().double()
    with torch.inference_mode():
        cpu_values = scripted(cpu_x).tolist()
        diagnostics = scripted.details(cpu_x).tolist()
    parity = max(abs(a-b)/max(1, abs(a)) for a, b in zip(native, cpu_values))
    if parity > 1e-5:
        raise ValueError("MoE GPU/CPU export parity failed: relative error="+str(parity))
    destination = Path(destination); destination.mkdir(parents=True, exist_ok=False)
    export = destination/"cpu_export"; export.mkdir()
    scripted.save(str(export/"weights.ts"))
    weights_hash = hashlib.sha256((export/"weights.ts").read_bytes()).hexdigest()
    metrics = score(list(zip(cpu_values, [r["target_iops"] for r in validation])), protocol["mask"])
    by_run={run:score([(p,r["target_iops"]) for p,r in zip(cpu_values,validation) if r["run_id"]==run],protocol["mask"])
            for run in sorted({r["run_id"] for r in validation})}
    by_workload={workload:score([(p,r["target_iops"]) for p,r in zip(cpu_values,validation) if r["workload"]==workload],protocol["mask"])
                 for workload in sorted({r["workload"] for r in validation})}
    metadata = {"model": "joint_moe", "artifact": "cpu_export/weights.ts", "sha256": weights_hash,
                "target": target, "training_runs": sorted({r["run_id"] for r in train}),
                "validation_runs": sorted({r["run_id"] for r in validation}), "validation": metrics,
                "validation_by_run":by_run,"validation_by_workload":by_workload,
                "formal_acceptance": False, "feature_schema": "joint_context_v1", "feature_count": FEATURE_COUNT,
                "topology_hash": canonical_hash(topology(site)), "attribution": "storage_context_only",
                "training_host": "lmj@b507", "history": history, "torch_version": torch.__version__,
                "training_precision":"float32","inference_precision":"float64",
                "normalization_log_std_floor":0.25,
                "fp32_vs_fp64_max_relative_difference":max(abs(a-b)/max(1,abs(b)) for a,b in zip(fp32_native,native)),
                "input_sha256": {str(p): hashlib.sha256(Path(p).read_bytes()).hexdigest() for p in train_paths+validation_paths},
                "study_hash": canonical_hash(study), "protocol_hash": canonical_hash(protocol),
                "data_partition_kind":study.get("partition_kind","independent_runs"),
                "experiment_kind":study.get("experiment_kind","development")}
    save(destination/"model_manifest.json", metadata)
    save(export/"cpu_model.json", {"kind": "torchscript", "artifact": "weights.ts",
                                   "feature_count": FEATURE_COUNT, "diagnostics": "moe_read_write_gates"})
    save(export/"site.json", site)
    save(export/"parity_vectors.json", [{"features": r["features"], "expected": p} for r, p in list(zip(validation, cpu_values))[:16]])
    save(destination/"validation_predictions.json", [{"run_id": r["run_id"], "window_start_ns": r["window_start_ns"],
         "actual": r["target_iops"], "total_read_write_gates": d} for r, d in zip(validation, diagnostics)])
    gate = {"passed": True, "relative_tolerance": 1e-5, "gpu_cpu_max_relative_error": parity, "samples": len(validation)}
    deployment = dict(metadata, source_model_sha256=weights_hash, runtime="torch_cpu", equivalence=gate,
                      fit_on_cluster=False, files={p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in export.iterdir()})
    save(export/"deployment.json", deployment)
    from .runtime import check_runtime
    check_runtime(export)
    return metadata, deployment
