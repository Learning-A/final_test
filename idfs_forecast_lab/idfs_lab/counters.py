import math
import re

GRID_NS = 100000000


def parse_mountstats(text, mountpoint):
    sections = [x for x in text.split("device ") if
                " mounted on %s " % mountpoint in x.split("\n", 1)[0]]
    if len(sections) != 1:
        raise ValueError("expected exactly one NFS mount section")
    section = sections[0]
    source = section.split()[0]
    operations = {}
    for line in section.splitlines()[1:]:
        name, sep, rest = line.strip().partition(":")
        if name in ("READ", "WRITE", "COMMIT", "GETATTR"):
            operations[name] = [int(v) for v in rest.split()]
    if any(len(operations.get(k, [])) < 8 for k in ("READ", "WRITE")):
        raise ValueError("READ/WRITE cumulative RPC counters missing")
    return {"source_id": source + "|" + mountpoint,
            "read_count": operations["READ"][0], "write_count": operations["WRITE"][0],
            "rpc_fields": operations, "count_semantics": "NFS per-mount RPC operations"}


def parse_gnfs_client_io(text):
    result = {}
    for name in ("READ", "WRITE"):
        match = re.search(r"^\s*" + name + r"\s*:\s*(\d+)\s+(\d+)\s+(\d+)", text, re.M)
        if not match:
            raise ValueError("GNFS client statistics missing " + name)
        total, errors, transferred = map(int, match.groups())
        result[name.lower() + "_count"] = total
        result[name.lower() + "_errors"] = errors
        result[name.lower() + "_bytes"] = transferred
    return result


def valid_count(value):
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value) and value >= 0


def counter_delta(previous, current, grid_ns=GRID_NS, jitter_ns=25000000):
    """No interpolation: require two adjacent scheduled and observed samples."""
    keys = ("scope", "entity", "host", "source_id", "boot_id", "instance_id", "sample_interval_ns")
    result = {"scope": current["scope"], "entity": current["entity"],
              "host": current["host"], "source_id": current.get("source_id"),
              "bucket_end_ns": current["scheduled_epoch_ns"], "sample_ok": False}
    for key in keys:
        if previous.get(key) != current.get(key):
            return dict(result, error="identity_changed:" + key)
    if not previous.get("sample_ok") or not current.get("sample_ok"):
        return dict(result, error="source_sample_failed")
    expected_interval = current.get("sample_interval_ns", grid_ns)
    if expected_interval != grid_ns:
        return dict(result, error="source_grid_mismatch")
    if current["scheduled_epoch_ns"] - previous["scheduled_epoch_ns"] != grid_ns:
        return dict(result, error="nonadjacent_grid")
    for sample in (previous, current):
        if sample.get("clock_step", False):
            return dict(result, error="clock_step")
        if abs(sample["read_end_epoch_ns"] - sample["scheduled_epoch_ns"]) > jitter_ns:
            return dict(result, error="sampling_jitter")
    elapsed = current["read_end_mono_ns"] - previous["read_end_mono_ns"]
    if elapsed <= 0 or abs(elapsed - grid_ns) > jitter_ns:
        return dict(result, error="invalid_observed_interval")
    counts = {}
    for name in ("read_count", "write_count"):
        a, b = previous.get(name), current.get(name)
        if not valid_count(a) or not valid_count(b):
            return dict(result, error="missing_counter:" + name)
        if b < a:
            return dict(result, error="counter_reset:" + name)
        counts[name] = b - a
    return dict(result, **counts, total_count=sum(counts.values()), sample_ok=True,
                bucket_start_ns=current["scheduled_epoch_ns"] - grid_ns,
                observed_interval_ns=elapsed,
                available_epoch_ns=current["read_end_epoch_ns"],
                read_iops=counts["read_count"] * 1e9 / elapsed,
                write_iops=counts["write_count"] * 1e9 / elapsed)


def aggregate_cluster(buckets, expected_ids):
    """Aggregate ONLY OSS counters; a partial cluster never becomes a zero-filled target."""
    expected_ids = set(str(x) for x in expected_ids)
    grouped = {}
    for row in buckets:
        if row["scope"] != "oss":
            continue
        grouped.setdefault(row["bucket_start_ns"] if row.get("sample_ok") else
                           row["bucket_end_ns"] - GRID_NS, []).append(row)
    result = []
    for start, rows in sorted(grouped.items()):
        found = {str(r["entity"]) for r in rows}
        ok = len(rows) == len(found) and found == expected_ids and all(r["sample_ok"] for r in rows)
        row = {"scope": "cluster_oss", "entity": "all", "bucket_start_ns": start,
               "bucket_end_ns": start + GRID_NS, "sample_ok": ok,
               "expected_oss": len(expected_ids), "observed_oss": len(found)}
        if ok:
            row.update(read_count=sum(r["read_count"] for r in rows),
                       write_count=sum(r["write_count"] for r in rows),
                       total_count=sum(r["total_count"] for r in rows),
                       available_epoch_ns=max(r["available_epoch_ns"] for r in rows))
        else:
            row["error"] = "incomplete_oss_coverage"
        result.append(row)
    return result
