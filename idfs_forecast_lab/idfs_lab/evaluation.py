from decimal import Decimal, localcontext
import hashlib
import math
from pathlib import Path

from .io import canonical_hash, load, read_events

SECOND_NS = 1000000000


def score(pairs, mask):
    """One function for provisional display and final independent recomputation.

    Mask is fixed to the mean of this evaluation interval, not a running historical mean.
    Decimal avoids display rounding determining a strict >90 boundary.
    """
    with localcontext() as ctx:
        ctx.prec=40
        values=[(Decimal(str(p)),Decimal(str(a))) for p,a in pairs]
        if any(not p.is_finite() or not a.is_finite() or p<0 or a<0 for p,a in values):
            raise ValueError("invalid prediction/actual")
        if not values:
            return {"accuracy_percent":None,"n_eval":0,"excluded":0,"threshold_iops":None}
        mean=sum(a for _,a in values)/len(values)
        if mask["mode"]=="evaluation_window_fixed":
            threshold=max(Decimal(mask["minimum_iops"]),Decimal(mask["mean_fraction"])*mean)
        elif mask["mode"]=="none":
            threshold=Decimal(0)
        else:
            raise ValueError("unknown mask mode")
        chosen=[(p,a) for p,a in values if a>=threshold and a>0]
        accuracy=None
        if chosen:
            accuracy=str((1-sum(abs(p-a)/a for p,a in chosen)/len(chosen))*100)
        return {"accuracy_percent":accuracy,"accuracy_display":format(Decimal(accuracy),'.4f') if accuracy else None,
                "n_eval":len(chosen),"excluded":len(values)-len(chosen),"threshold_iops":str(threshold)}


def verify_round(manifest, predictions, actuals, protocol):
    errors=[]
    ph=canonical_hash(protocol)
    target=manifest["target"]
    allowed=protocol.get("formal_targets",[protocol["formal_target"]] if protocol.get("formal_target") else [])
    if allowed and target not in allowed:errors.append("formal_target_mismatch")
    if manifest.get("protocol_hash")!=ph:errors.append("protocol_hash_mismatch")
    if manifest.get("t0_source") not in ("tool_first_io", "recording_start"):
        errors.append("t0_source_missing")
    if not manifest.get("selection_frozen_before_formal_test"):errors.append("selection_not_frozen")
    if not manifest.get("selection_hash"):errors.append("selection_hash_missing")
    if manifest.get("data_kind")!="measured":errors.append("not_measured_experiment")
    clock=manifest.get("clock_alignment",{})
    if (not clock.get("verified") or not clock.get("evidence_sha256") or
            abs(clock.get("max_relative_offset_ms",float('inf')))>protocol["max_clock_offset_ms"] or
            clock.get("uncertainty_ms",float('inf'))>protocol["max_clock_uncertainty_ms"]):
        errors.append("clock_alignment_unverified")
    t0=int(manifest["t0_epoch_ns"])
    lo=t0+protocol["evaluation_from_s"]*SECOND_NS
    hi=t0+protocol["evaluation_to_s"]*SECOND_NS
    first=((lo+SECOND_NS-1)//SECOND_NS)*SECOND_NS
    expected=list(range(first,hi-SECOND_NS+1,SECOND_NS))
    predmap={};actualmap={};commits={}
    for item in predictions:
        if item.get("event")=="prediction_commit" and item.get("target")==target:
            start=item["window_start_ns"]
            if start in commits:errors.append("duplicate_commit:%d"%start)
            commits[start]=item["committed_epoch_ns"]
    for p in predictions:
        if p.get("target")!=target:continue
        if p.get("event")=="prediction_commit":continue
        start=p["window_start_ns"]
        if start in predmap:errors.append("duplicate_prediction:%d"%start)
        predmap[start]=p
        if p.get("protocol_hash")!=ph or p.get("model_sha256")!=manifest.get("model_sha256"):
            errors.append("prediction_version_mismatch:%d"%start)
        if (p["issue_epoch_ns"]>=start or p.get("feature_end_ns",start)>p["issue_epoch_ns"] or
                p.get("features_available_ns",start)>p.get("compute_start_ns",p["issue_epoch_ns"]) or
                p.get("compute_start_ns",0)>p["issue_epoch_ns"]):
            errors.append("late_or_noncausal_prediction:%d"%start)
        if p.get("window_end_ns")!=start+SECOND_NS:errors.append("wrong_prediction_horizon")
        committed=commits.get(start)
        if committed is None or committed<p["issue_epoch_ns"] or committed>=start:
            errors.append("missing_or_late_durable_prediction:%d"%start)
    for a in actuals:
        if a.get("scope")!=target["scope"] or str(a.get("entity"))!=str(target["entity"]):continue
        start=a["window_start_ns"]
        if start in actualmap:errors.append("duplicate_actual:%d"%start)
        actualmap[start]=a
    # Evaluate first prediction ISSUE time, not a settlement-completion time one second later.
    issue_times=[p["issue_epoch_ns"] for p in predmap.values()]
    if not issue_times or abs(min(issue_times)-lo)>protocol["prediction_start_tolerance_s"]*SECOND_NS:
        errors.append("prediction_start_outside_tolerance")
    pairs=[];settlements=[]
    for start in expected:
        p,a=predmap.get(start),actualmap.get(start)
        if not p:
            errors.append("missing_prediction:%d"%start);continue
        if not a or not a.get("complete") or a.get("n_ticks")!=10:
            errors.append("incomplete_actual:%d"%start);continue
        if a.get("window_end_ns")!=start+SECOND_NS or a.get("available_epoch_ns",0)<start+SECOND_NS:
            errors.append("actual_horizon_or_availability_invalid:%d"%start);continue
        observed=a[target["component"]+"_iops"]
        pairs.append((p["prediction_iops"],observed))
        settlements.append({"window_start_ns":start,"window_end_ns":start+SECOND_NS,
                            "issue_epoch_ns":p["issue_epoch_ns"],"prediction_iops":p["prediction_iops"],
                            "actual_iops":observed,"target":target})
    metrics=score(pairs,protocol["mask"])
    passed=(not errors and len(pairs)==len(expected) and metrics["n_eval"]>0 and
            Decimal(metrics["accuracy_percent"])>Decimal(protocol["threshold_percent"]))
    return dict(metrics,round_id=manifest["round_id"],round_index=manifest.get("round_index"),
                workload=manifest["workload"],target=target,model_sha256=manifest.get("model_sha256"),
                protocol_hash=ph,selection_hash=manifest.get("selection_hash"),
                workload_commit=manifest.get("workload_commit"),
                expected_windows=len(expected),observed_complete_windows=len(pairs),
                passed=passed,errors=sorted(set(errors)),settlements=settlements)


def verify_files(run_dir, protocol):
    run_dir=Path(run_dir)
    manifest=load(run_dir/"round_manifest.json")
    t0_evidence=manifest.get("first_io_evidence",{})
    t0_path=run_dir/t0_evidence.get("file", "first_io.json")
    if hashlib.sha256(t0_path.read_bytes()).hexdigest()!=t0_evidence.get("sha256"):
        raise ValueError("first I/O evidence checksum mismatch")
    first_io=load(t0_path)
    if (first_io.get("event") not in ("tool_first_io", "recording_start") or
            first_io.get("event") != manifest.get("t0_source") or
            first_io.get("epoch_ns")!=manifest.get("t0_epoch_ns")):
        raise ValueError("first I/O evidence does not match t0")
    selection=load(run_dir/"selection.json")
    if canonical_hash(selection)!=manifest.get("selection_hash"):
        raise ValueError("frozen selection checksum mismatch")
    if selection.get("frozen_epoch_ns",float('inf'))>=manifest["t0_epoch_ns"]:
        raise ValueError("selection was not frozen before first I/O")
    if manifest["round_id"] in set(selection.get("training_runs",[])+selection.get("validation_runs",[])):
        raise ValueError("formal round overlaps train/validation")
    for key in ("target","model_sha256","workload","workload_commit","protocol_hash"):
        if selection.get(key)!=manifest.get(key):raise ValueError("selection mismatch: "+key)
    clock_path=run_dir/"clock_alignment.json"
    clock_hash=hashlib.sha256(clock_path.read_bytes()).hexdigest()
    if manifest.get("clock_alignment",{}).get("evidence_sha256")!=clock_hash:
        raise ValueError("clock evidence checksum mismatch")
    clock=load(clock_path)
    for key in ("verified","max_relative_offset_ms","uncertainty_ms"):
        if manifest["clock_alignment"].get(key)!=clock.get(key):raise ValueError("clock evidence value mismatch")
    for name in ("predictions.jsonl","windows.jsonl"):
        expected=manifest.get("log_sha256",{}).get(name)
        actual=hashlib.sha256((run_dir/name).read_bytes()).hexdigest()
        if not expected or expected!=actual:
            raise ValueError("missing or mismatched sealed log hash: "+name)
    return verify_round(manifest,list(read_events(run_dir/"predictions.jsonl")),
                        list(read_events(run_dir/"windows.jsonl")),protocol)


def summarize_rounds(reports,protocol):
    errors=[]
    if len(reports)!=protocol["formal_rounds"]:errors.append("exactly_three_rounds_required")
    if len({r["round_id"] for r in reports})!=len(reports):errors.append("duplicate_round_id")
    if {r.get("round_index") for r in reports}!={1,2,3}:errors.append("round_indices_must_be_1_2_3")
    for field in ("target","model_sha256","protocol_hash","selection_hash","workload","workload_commit"):
        if len({canonical_hash(r.get(field)) for r in reports})!=1 or any(r.get(field) is None for r in reports):
            errors.append("inconsistent_or_missing:"+field)
    ph=canonical_hash(protocol)
    for r in reports:
        if r.get("protocol_hash")!=ph:errors.append("protocol_mismatch")
        if r.get("errors") or not r.get("passed") or r.get("accuracy_percent") is None or Decimal(r["accuracy_percent"])<=Decimal(protocol["threshold_percent"]):
            errors.append("round_failed:"+r["round_id"])
    return {"passed":not errors,"rule":"each_of_three_rounds_strictly_above_90",
            "errors":sorted(set(errors)),"rounds":[{k:r.get(k) for k in ("round_id","accuracy_percent","n_eval","passed")} for r in reports]}
