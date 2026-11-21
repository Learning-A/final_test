"""Seal a stopped inference run for independent recomputation; no training imports."""
import argparse
import hashlib
from pathlib import Path

from .io import canonical_hash,load,save


def seal_run(directory,round_id,round_index,formal=False):
    directory=Path(directory)
    if not (directory/"service_finished.json").is_file():raise ValueError("inference service has not closed its logs")
    selection=load(directory/"selection.json");first=load(directory/"first_io.json")
    clock_file=directory/"clock_alignment.json";clock=load(clock_file)
    manifest={"schema_version":1,"round_id":round_id,"round_index":round_index,
              "workload":selection["workload"],"workload_commit":selection["workload_commit"],
              "target":selection["target"],"model_sha256":selection["model_sha256"],
              "protocol_hash":selection["protocol_hash"],"selection_hash":canonical_hash(selection),
              "selection_frozen_before_formal_test":selection["frozen_epoch_ns"]<first["epoch_ns"],
              "data_kind":"measured" if formal else "measured_exploration",
              "t0_epoch_ns":first["epoch_ns"],"t0_source":first.get("event","recording_start"),
              "first_io_evidence":{"file":"first_io.json","sha256":hashlib.sha256((directory/"first_io.json").read_bytes()).hexdigest()},
              "clock_alignment":{k:clock.get(k) for k in ("verified","max_relative_offset_ms","uncertainty_ms")},
              "log_sha256":{name:hashlib.sha256((directory/name).read_bytes()).hexdigest() for name in ("predictions.jsonl","windows.jsonl")}}
    manifest["clock_alignment"]["evidence_sha256"]=hashlib.sha256(clock_file.read_bytes()).hexdigest()
    if (directory/"round_manifest.json").exists():raise ValueError("round already sealed")
    save(directory/"round_manifest.json",manifest)
    return manifest


if __name__=="__main__":
    p=argparse.ArgumentParser();p.add_argument("--run",required=True);p.add_argument("--round-id",required=True)
    p.add_argument("--round-index",type=int,required=True);p.add_argument("--formal",action="store_true")
    a=p.parse_args();seal_run(a.run,a.round_id,a.round_index,a.formal)
