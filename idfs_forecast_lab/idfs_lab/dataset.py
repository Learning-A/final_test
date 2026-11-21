from collections import defaultdict
import csv
import hashlib
from pathlib import Path

from .counters import GRID_NS, aggregate_cluster, counter_delta
from .io import append_event, read_events, save, BufferedJSONLog

SECOND_NS = 1000000000


def build_buckets(samples, expected_ids, jitter_ms=25):
    grouped = defaultdict(list)
    seen = set()
    for row in samples:
        key = (row["scope"], row["entity"], row["scheduled_epoch_ns"])
        if key in seen:
            raise ValueError("duplicate source/grid sample: " + str(key))
        seen.add(key)
        grouped[(row["scope"], row["entity"])].append(row)
    buckets = []
    for rows in grouped.values():
        rows.sort(key=lambda r: r["scheduled_epoch_ns"])
        for previous, current in zip(rows, rows[1:]):
            bucket=counter_delta(previous, current, jitter_ns=int(jitter_ms*1e6))
            if "received_inference_epoch_ns" in current:
                bucket["available_epoch_ns"]=max(bucket["available_epoch_ns"],current["received_inference_epoch_ns"])
            buckets.append(bucket)
    return buckets + aggregate_cluster(buckets, expected_ids)


def build_windows(buckets):
    grouped = defaultdict(dict)
    invalid = set()
    for row in buckets:
        start = row.get("bucket_start_ns", row["bucket_end_ns"] - GRID_NS)
        if start % GRID_NS:
            raise ValueError("bucket not on absolute 100ms grid")
        second = start // SECOND_NS * SECOND_NS
        key = (row["scope"], row["entity"], second)
        if start in grouped[key]:
            raise ValueError("duplicate bucket in window")
        grouped[key][start] = row
        if not row["sample_ok"]:
            invalid.add(key)
    windows = []
    for (scope, entity, second), ticks in sorted(grouped.items()):
        expected = set(range(second, second + SECOND_NS, GRID_NS))
        complete = set(ticks) == expected and (scope, entity, second) not in invalid
        row = {"scope":scope, "entity":entity, "window_start_ns":second,
               "window_end_ns":second+SECOND_NS, "complete":complete,
               "n_ticks":len(ticks), "unit":"ops/s"}
        if complete:
            for part in ("read", "write", "total"):
                row[part+"_iops"] = sum(v[part+"_count"] for v in ticks.values())
            row["available_epoch_ns"] = max(v["available_epoch_ns"] for v in ticks.values())
        else:
            row["error"] = "incomplete_or_invalid_100ms_window"
        windows.append(row)
    return windows


def process(directory, output, protocol):
    directory, output = Path(directory), Path(output)
    if (directory/"probe.json").exists():
        raise ValueError("bounded probe snapshots are not experimental training data")
    if not (directory/"collection_manifest.json").exists():
        raise ValueError("collection manifest required")
    paths = sorted(directory.glob("*.jsonl"))
    if not paths:
        raise ValueError("no counter logs")
    samples = [row for path in paths for row in read_events(path)]
    buckets = build_buckets(samples, protocol["expected_oss_ids"], protocol["max_sample_jitter_ms"])
    windows = build_windows(buckets)
    if protocol.get("host_oss_groups"):
        from .joint import host_windows
        topology_site={"storage_nodes":[{"hostname":host,"oss_ids":ids} for host,ids in protocol["host_oss_groups"].items()]}
        windows+=list(host_windows(windows,topology_site))
    output.mkdir(parents=True, exist_ok=False)
    with BufferedJSONLog(output/"buckets.jsonl") as writer:
        for row in buckets:writer.append(row)
    with BufferedJSONLog(output/"windows.jsonl") as writer:
        for row in windows:writer.append(row)
    summary = {"raw_samples":len(samples), "buckets":len(buckets),
               "good_buckets":sum(bool(x["sample_ok"]) for x in buckets),
               "windows":len(windows), "complete_windows":sum(x["complete"] for x in windows)}
    from .io import load
    summary.update(source_kind=load(directory/"collection_manifest.json").get("kind"),
                   availability_evidence="inference_host_reception" if all("received_inference_epoch_ns" in r for r in samples) else "source_read_end_only_not_online_reception_proof",
                   source_directory=str(directory.resolve()),
                   source_manifest_sha256=hashlib.sha256((directory/"collection_manifest.json").read_bytes()).hexdigest(),
                   raw_files_sha256={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in paths},
                   windows_sha256=hashlib.sha256((output/"windows.jsonl").read_bytes()).hexdigest())
    save(output/"quality.json", summary)
    return summary


def feature_rows(windows, lookback=5):
    """Each target is the NEXT complete second; missing history is never zero-filled."""
    grouped = defaultdict(dict)
    for row in windows:
        key = (row["scope"], row["entity"])
        if row["window_start_ns"] in grouped[key]:
            raise ValueError("duplicate second/entity")
        grouped[key][row["window_start_ns"]] = row
    for (scope, entity), series in sorted(grouped.items()):
        for start, target in sorted(series.items()):
            history = [series.get(start-i*SECOND_NS) for i in range(1,lookback+1)]
            if not target["complete"] or any(r is None or not r["complete"] for r in history):
                continue
            # A real forecast requires the history to have arrived before target start.
            available = max(r["available_epoch_ns"] for r in history)
            if available >= start:
                # The last sample at the boundary may arrive after it; retain a full
                # second of issue margin by excluding that latest window, not by
                # backdating prediction timestamps.
                history = [series.get(start-i*SECOND_NS) for i in range(2,lookback+2)]
                if any(r is None or not r["complete"] for r in history):
                    continue
                available = max(r["available_epoch_ns"] for r in history)
            if available >= start:
                continue
            for component in ("read","write","total"):
                values = [r[component+"_iops"] for r in history]
                yield {"scope":scope,"entity":entity,"component":component,
                       "window_start_ns":start,"window_end_ns":target["window_end_ns"],
                       "feature_end_ns":history[0]["window_end_ns"],
                       "features_available_ns":available,"lags":values,
                       "rolling_mean":sum(values)/len(values),
                       "target_iops":target[component+"_iops"]}


def export_features(windows_path, destination, run_id):
    rows = list(feature_rows(list(read_events(windows_path))))
    if not rows:
        raise ValueError("no complete causal training examples")
    destination = Path(destination)
    destination.parent.mkdir(parents=True,exist_ok=True)
    with destination.open("x",encoding="utf-8",newline="") as f:
        fields=["run_id","scope","entity","component","window_start_ns","window_end_ns",
                "feature_end_ns","features_available_ns"]+["lag%d"%i for i in range(1,6)]+["rolling_mean","target_iops"]
        writer=csv.DictWriter(f,fieldnames=fields)
        writer.writeheader()
        for row in rows:
            lags=row.pop("lags")
            row.update({"lag%d"%(i+1):v for i,v in enumerate(lags)})
            writer.writerow(dict(row,run_id=run_id))
    return {"rows":len(rows),"output":str(destination)}
