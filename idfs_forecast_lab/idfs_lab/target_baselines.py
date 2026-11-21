"""Development-only node/cluster baselines. No fitting or remote operations."""
import math

SECOND = 1_000_000_000
SCHEMA = "paired_host_cluster_baseline_v1"


def paired_baselines(windows, hosts, issue_margin_ms=30):
    """Evaluate one run, using five seconds ending one second before target.

    Call separately for each run: history must never cross capture boundaries.
    Host windows must come from the existing complete-12-OSS aggregation.
    Both targets share the same complete labels and timely history. Coverage
    includes rejected target windows; results are not formal acceptance.
    """
    hosts = tuple(hosts)
    if len(hosts) != 8 or len(set(hosts)) != 8:
        raise ValueError("eight distinct physical host names required")
    if not math.isfinite(issue_margin_ms) or not 0 < issue_margin_ms < 1000:
        raise ValueError("issue margin must be between zero and one second")
    index = {}
    for row in windows:
        if row.get("scope") != "host_oss" or row.get("entity") not in hosts:
            continue
        start = row["window_start_ns"]
        if not isinstance(start, int) or start % SECOND:
            raise ValueError("unaligned host window")
        key = (row["entity"], start)
        if key in index:
            raise ValueError("duplicate host window; pass one run at a time")
        if row.get("complete"):
            if row["window_end_ns"] != start+SECOND or row["n_ticks"] != 10:
                raise ValueError("invalid complete host window")
            value = row["total_iops"]
            if not math.isfinite(value) or value < 0:
                raise ValueError("invalid host IOPS")
            if row["available_epoch_ns"] < row["window_end_ns"]:
                raise ValueError("actual available before window ends")
        index[key] = row
    starts = sorted({start for _, start in index})
    pairs = []
    rejected = {"incomplete_label": 0, "missing_history": 0, "late_history": 0}
    for start in starts:
        labels = [index.get((h, start)) for h in hosts]
        if any(r is None or not r.get("complete") for r in labels):
            rejected["incomplete_label"] += 1
            continue
        history = [[index.get((h, start-i*SECOND)) for i in range(2, 7)] for h in hosts]
        if any(r is None or not r.get("complete") for series in history for r in series):
            rejected["missing_history"] += 1
            continue
        available = max(r["available_epoch_ns"] for series in history for r in series)
        deadline = start-int(issue_margin_ms*1_000_000)
        if available > deadline:
            rejected["late_history"] += 1
            continue
        last = [series[0]["total_iops"] for series in history]
        mean = [sum(r["total_iops"] for r in series)/5 for series in history]
        pairs.append({"window_start_ns": start, "feature_end_ns": start-SECOND,
                      "features_available_ns": available, "deadline_ns": deadline,
                      "node_actual": [r["total_iops"] for r in labels],
                      "cluster_actual": sum(r["total_iops"] for r in labels),
                      "node_last_available": last, "node_mean_five": mean,
                      "cluster_last_available": sum(last), "cluster_mean_five": sum(mean)})
    return {"schema": SCHEMA, "formal_acceptance": False, "fitting_performed": False,
            "hosts": list(hosts), "candidate_windows": len(starts),
            "paired_windows": len(pairs), "rejected": rejected, "pairs": pairs}


def metrics(pairs):
    """(prediction, actual) pairs; MAPE excludes only actual zero."""
    pairs = list(pairs)
    if any(not math.isfinite(v) or v < 0 for pair in pairs for v in pair):
        raise ValueError("nonfinite or negative prediction/actual")
    positive = [(p, a) for p, a in pairs if a > 0]
    absolute = sum(abs(p-a) for p, a in pairs)
    actual = sum(a for _, a in pairs)
    mape = 100*sum(abs(p-a)/a for p, a in positive)/len(positive) if positive else None
    return {"n": len(pairs), "zero_actual": len(pairs)-len(positive),
            "mae_iops": absolute/len(pairs) if pairs else None,
            "wape_percent": 100*absolute/actual if actual else None,
            "mape_percent": mape, "accuracy_percent": 100-mape if mape is not None else None}
