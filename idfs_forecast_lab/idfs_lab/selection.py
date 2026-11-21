import hashlib
from pathlib import Path
import time

from .io import canonical_hash, load, save


def freeze_selection(model_manifest, workload, workload_commit, protocol, destination):
    metadata=load(model_manifest)
    if metadata.get("experiment_kind")=="smoke_not_acceptance" or metadata.get("data_partition_kind")=="chronological_same_capture_not_independent":
        raise ValueError("smoke model cannot be frozen as a formal independent-validation selection")
    allowed=protocol.get("formal_targets",[protocol["formal_target"]] if protocol.get("formal_target") else [])
    if allowed and metadata["target"] not in allowed:
        raise ValueError("target is outside the registered physical storage hosts")
    if not metadata.get("training_runs") or not metadata.get("validation_runs"):
        raise ValueError("explicit current-data train/validation runs required")
    if set(metadata["training_runs"]) & set(metadata["validation_runs"]):
        raise ValueError("overlapping train/validation runs")
    artifact=Path(model_manifest).parent/metadata["artifact"]
    if hashlib.sha256(artifact.read_bytes()).hexdigest()!=metadata["sha256"]:
        raise ValueError("model artifact checksum mismatch")
    result={"schema_version":1,"frozen_epoch_ns":time.time_ns(),"workload":workload,
            "workload_commit":workload_commit,"target":metadata["target"],
            "model_sha256":metadata["sha256"],"model_manifest_sha256":hashlib.sha256(Path(model_manifest).read_bytes()).hexdigest(),
            "training_runs":metadata["training_runs"],"validation_runs":metadata["validation_runs"],
            "protocol_hash":canonical_hash(protocol),"validation":metadata["validation"],
            "rule":"one_selection_for_all_three_formal_rounds"}
    if Path(destination).exists():raise ValueError("selection already frozen; choose a new version")
    save(destination,result)
    return dict(result,selection_hash=canonical_hash(result))
