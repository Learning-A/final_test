"""Rebuild local preparation files only. No network/collection/training calls."""
import json
from pathlib import Path
from idfs_lab.io import canonical_hash
from idfs_lab.joint import TARGET

ROOT = Path(__file__).resolve().parent


def load(name):
    return json.loads((ROOT/name).read_text(encoding="utf-8"))


def save(name, value):
    (ROOT/name).write_text(json.dumps(value, ensure_ascii=False, indent=2)+"\n", encoding="utf-8")


def main():
    if any((ROOT/"runs").glob("*/collection_manifest.json")):
        raise RuntimeError("preparation regeneration is disabled after measured collection starts")
    requirements = load("config/confirmed_requirements.json")
    requirements.update(formal_target=TARGET, choose_target_on_validation_then_freeze=False,
                        joint_strategy="client7414_target_with_storage_context",
                        exact_client_oss_attribution="unverified_disabled",
                        training_rounds_per_workload=3, validation_rounds_per_workload=1,
                        formal_rounds_per_workload=3, total_rounds=35,
                        collection_started=False, training_started=False)
    save("config/confirmed_requirements.json", requirements)
    protocol = load("config/protocol.json")
    protocol.update(formal_target=TARGET,
                    formal_selection="fixed_client7414_total_choose_model_on_validation_then_freeze_before_formal",
                    mask_status="user_confirmed_zero_actual_only",
                    mask={"mode":"none", "zero_actual_policy":"exclude_and_disclose", "disclose_threshold_and_exclusions":True})
    save("config/protocol.json", protocol)
    requirements["accuracy_mask"]="exclude_only_zero_actual_keep_all_positive"
    save("config/confirmed_requirements.json", requirements)
    gates=load("config/preflight_gates.json")
    for gate in gates:
        if gate["id"]=="numeric_near_zero_policy":
            gate.update(status="passed",evidence="用户已确认：只排除实际 IOPS 为零的窗口，保留全部正值并披露排除量")
    save("config/preflight_gates.json",gates)
    cases = load("config/workloads.json")["formal_cases"]
    rounds = []
    for workload in cases:
        for role, count in (("train", 3), ("validation", 1), ("formal", 3)):
            for index in range(1, count+1):
                rounds.append({"run_id": "%s_%s_%02d" % (workload, role, index),
                               "workload": workload, "role": role, "role_index": index,
                               "workload_seconds": 600, "planned_collection_seconds": 660,
                               "state": "planned_not_started"})
    study = {"schema_version": 1, "authority": "user_confirmed_2026-11-21",
             "formal_target": TARGET, "joint_strategy": "storage_context_only",
             "workload_commit": load("config/workloads.json")["commit"],
             "order": ["collect_15_train_and_5_validation", "preprocess", "train_on_lmj_gpu",
                       "choose_on_validation_and_freeze", "stage_cpu_inference", "collect_15_formal_and_recompute"],
             "total_rounds": 35, "workload_minutes": 350, "collection_minutes_without_gaps": 385,
             "sampling_budget_note": "660s is planning allowance; first-I/O must fall inside capture and all required windows must be covered",
             "formal_data_may_train_or_select": False, "rounds": rounds}
    save("config/study_plan.json", study)
    spec = {"job_id": "joint_moe_v1_after_measured_collection", "model": "joint_moe", "target": TARGET,
            "site": "config/site.json", "study": "config/study_plan.json", "cpu_threads": 4,
            "epochs": 100, "gpu_uuid": None}
    for role in ("train", "validation"):
        spec[role] = ["prepared/%s/joint.jsonl" % r["run_id"] for r in rounds if r["role"] == role]
    save("config/gpu_job_moe.example.json", spec)
    save("config/joint_contract.json", {"schema": "joint_context_v1", "target": TARGET,
        "feature_count": 415, "temporal_shape": [5, 3], "temporal_order": "oldest_to_newest_read_write_total",
        "spatial_shape": [96, 4], "spatial_order": "numeric_oss_id_read_write_total_observed_mask",
        "cross_values": ["client_read", "client_write", "client_total", "observed_oss_read_sum",
                         "observed_oss_write_sum", "oss_coverage", "observed_other_client_total", "other_client_coverage"],
        "cross_masks": 8, "minimum_observed_oss": 92, "minimum_oss_status": "engineering_setting_validate_on_development_data",
        "history_end_before_target_ms": 1000, "issue_margin_ms": 30,
        "graph": "same_physical_host_OSS_graph_not_client_request_attribution",
        "normalization": "log1p_masked_train_only_mean_std_embedded_in_TorchScript",
        "exact_client_oss_attribution": {"enabled": False, "required_evidence":
            ["client_identity", "oss_identity", "cumulative_read_write_count", "reset_identity", "sampling_time", "semantic_and_coverage_validation"]},
        "experts": ["GRU_temporal", "physical_host_GCN_spatial", "cross_layer_MLP"],
        "gates": "learned_hierarchical_client_vs_storage_then_spatial_vs_cross",
        "output": "read_plus_write_total_future_one_second",
        "current_task": "client_layer_1s_only_no_multihorizon_training_claim",
        "not_yet_done": ["measured_data_fit", "GPU_execution", "target_CPU_parity", "accuracy_test"]})
    template = load("config/round_manifest.template.json")
    template.update(target=TARGET, protocol_hash=canonical_hash(protocol))
    save("config/round_manifest.template.json", template)
    inventory = load("models/inventory.json")
    inventory["models"] = [m for m in inventory["models"] if m["name"] != "joint_moe"]
    inventory["models"].append({"name": "joint_moe", "source_path": str(ROOT/"idfs_lab/moe.py"),
        "source_exists": True, "weights": None, "status": "architecture_prepared_waiting_measured_data",
        "targets": ["client_nfs_rpc"], "local_training_adapter": True,
        "training_location": "lmj@b507 only", "feature_schema": "joint_context_v1"})
    save("models/inventory.json", inventory)


if __name__ == "__main__":
    main()
