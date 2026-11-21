"""Explicit local preprocessing command; never connects or starts collection."""
import argparse
import hashlib
from pathlib import Path
from .io import load, read_events, save, append_event
from .joint import feature_rows, host_windows, topology


def export(windows_path, study_path, site_path, run_id, destination):
    study, site = load(study_path), load(site_path)
    found = [r for r in study["rounds"] if r["run_id"] == run_id]
    if len(found) != 1:
        raise ValueError("run must be registered in the study plan")
    run = found[0]
    if run["role"] == "formal":
        raise ValueError("formal held-out rounds must not enter training feature export")
    quality=load(Path(windows_path).parent/"quality.json")
    source_hash=hashlib.sha256(Path(windows_path).read_bytes()).hexdigest()
    if quality.get("source_kind")!="counter_collection" or quality.get("windows_sha256")!=source_hash:
        raise ValueError("processed counter-collection provenance and matching windows hash required")
    windows = list(read_events(windows_path))
    rows = list(feature_rows(windows, site, run_id, run["role"]))
    if not rows:
        raise ValueError("no complete causal joint examples")
    destination = Path(destination)
    destination.mkdir(parents=True, exist_ok=False)
    for row in rows:
        row["workload"] = run["workload"]
        row["source_windows_sha256"] = source_hash
        append_event(destination/"joint.jsonl", row)
    for row in host_windows(windows, site):
        append_event(destination/"host_windows.jsonl", row)
    result = {"run": run, "examples": len(rows), "source_sha256": hashlib.sha256(Path(windows_path).read_bytes()).hexdigest(),
              "study_sha256": hashlib.sha256(Path(study_path).read_bytes()).hexdigest(),
              "source_quality": quality,
              "attribution": "storage_context_only", "normalization_fitted": False,
              "timing_note": "offline availability uses recorded times; online host reception is authoritative",
              "topology": topology(site), "training_performed": False}
    save(destination/"preprocessing.json", result)
    return result


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--windows", required=True); p.add_argument("--study", required=True)
    p.add_argument("--site", required=True); p.add_argument("--run-id", required=True)
    p.add_argument("--out", required=True)
    a = p.parse_args()
    export(a.windows, a.study, a.site, a.run_id, a.out)


if __name__ == "__main__":
    main()
