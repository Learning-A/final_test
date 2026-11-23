"""Measure offline node bucket coverage under hypothetical deadline margins."""

import argparse
import gzip
import json
from collections import Counter
from pathlib import Path


SECOND = 10**9
BUCKET = SECOND // 10
MARGINS_MS = (0, 25, 50, 75, 100, 200, 500)


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def audit_capture(original, fine_path):
    anchor = read_json(original / "recording_start.json")["epoch_ns"]
    slack = {}
    with gzip.open(fine_path, "rt", encoding="utf-8") as stream:
        for line in stream:
            row = json.loads(line)
            if row["scope"] == "host_oss" and row["sample_ok"]:
                end = row["bucket_end_ns"]
                slack[row["entity"], end - BUCKET] = max(end, row["available_epoch_ns"])
    counts = {margin: Counter() for margin in MARGINS_MS}
    labels = 0
    with (original / "processed/windows.jsonl").open(encoding="utf-8") as stream:
        for line in stream:
            row = json.loads(line)
            start = row["window_start_ns"]
            if (row["scope"] != "host_oss" or not row["complete"]
                    or start < anchor + 16 * SECOND
                    or row["window_end_ns"] > anchor + 600 * SECOND):
                continue
            labels += 1
            available = [slack.get((row["entity"], start - lag * BUCKET)) for lag in range(1, 21)]
            for margin, tally in counts.items():
                cutoff = start - (30 + margin) * 10**6
                usable = [observed is not None and observed <= cutoff for observed in available]
                tally["usable_slots"] += sum(usable)
                tally["no_usable_slots"] += not any(usable)
                tally["at_least_one_last_500ms_slot"] += any(usable[:5])
    if not labels:
        raise ValueError(f"No eligible node labels in {original.name}")
    return {
        "eligible_node_windows": labels,
        "margins": {
            str(margin): {**tally, "mean_usable_slots": tally["usable_slots"] / labels}
            for margin, tally in counts.items()
        },
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("index", "original", "core", "reference", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    args = parser.parse_args()
    index = read_json(args.index)
    result = {
        "request_id": index["request_id"],
        "scope": "15 train captures, complete node labels within recording_start+[16s,600s); coverage only",
        "time_evidence": "source read-end; additional margins are hypothetical, not clock bounds",
        "published_deadline_before_target_ms": 30,
        "captures": {},
    }
    for capture in index["captures"]:
        name = capture["dataset_id"]
        fine = (args.reference if capture["delivery"]["kind"] == "existing_release"
                else args.core / name)
        result["captures"][name] = audit_capture(args.original / name, fine / "fine_context.jsonl.gz")
    with args.output.open("x", encoding="utf-8") as output:
        json.dump(result, output, indent=2)
        output.write("\n")


if __name__ == "__main__":
    main()
