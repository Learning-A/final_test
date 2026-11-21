"""Conservative local readiness report; no connection or experimental action."""
import argparse
from pathlib import Path
from .io import load, save


def report(root):
    root = Path(root)
    study = load(root/"config/study_plan.json")
    requirements = load(root/"config/confirmed_requirements.json")
    rounds = study["rounds"]
    if len(rounds) != 35 or len({r["run_id"] for r in rounds}) != 35:
        raise ValueError("invalid confirmed round plan")
    if requirements["formal_target"] != study["formal_target"]:
        raise ValueError("formal target conflict")
    gates = load(root/"config/preflight_gates.json")
    return {"state": requirements.get("latest_phase","preparation_only"), "formal_target": study["formal_target"],
            "formal_targets":requirements.get("formal_targets",[study["formal_target"]]),
            "old_round_plan_state":study.get("state","planned"),
            "roles": {role: sum(r["role"] == role for r in rounds) for role in ("train", "validation", "formal")},
            "collection_ready": all(g["status"] == "passed" for g in gates if g["phase"] == "collection"),
            "formal_ready": all(g["status"] == "passed" for g in gates),
            "outstanding": [g for g in gates if g["status"] != "passed"],
            "network_accessed_by_this_report": False,
            "collection_started":requirements.get("collection_started",False),
            "training_started":requirements.get("training_started",False)}


def main():
    p = argparse.ArgumentParser(); p.add_argument("--root", default="."); p.add_argument("--out")
    a = p.parse_args(); result = report(a.root)
    if a.out:
        save(a.out, result)
    import json
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
