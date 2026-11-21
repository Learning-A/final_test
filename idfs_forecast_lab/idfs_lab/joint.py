"""Identical causal, masked storage-context inputs for offline and online use.

An OSS total is never attributed to a client. All numbers are operations/second.
"""
import math
from .dataset import SECOND_NS
from .io import canonical_hash

TARGET = {"scope": "host_oss", "entity": "idfs44", "component": "total"}
LOOKBACK = 5
FEATURE_COUNT = 415  # 5*3 temporal + 96*4 spatial + 8 values + 8 masks
COMPONENTS = ("read", "write", "total")


def index_windows(windows):
    result = {}
    for row in windows:
        key = (row["scope"], str(row["entity"]), row["window_start_ns"])
        if key in result:
            raise ValueError("duplicate window")
        result[key] = row
    return result


def topology(site):
    groups = [list(map(str, node["oss_ids"])) for node in site["storage_nodes"]]
    flat = [v for group in groups for v in group]
    if len(flat) != 96 or set(flat) != {str(i) for i in range(96)}:
        raise ValueError("exact physical-host partition of 96 OSS required")
    # Shared physical server, including self edges; not client request routing.
    graph = [[0.0]*96 for _ in range(96)]
    for group in groups:
        for i in group:
            for j in group:
                graph[int(i)][int(j)] = 1.0/len(group)
    return graph


def causal_features(index, start, site, issue_margin_ms=30, min_oss=92, target=None):
    """Use five seconds ending one second before the future target window.

    The guard checks actual availability as well as source window end. Missing
    spatial/auxiliary observations have explicit masks; target history may not.
    """
    target=target or TARGET
    deadline = start-int(issue_margin_ms*1e6)
    used = []

    def get(scope, entity, second):
        row = index.get((scope, str(entity), second))
        if row is None or not row.get("complete"):
            return None
        if row["window_end_ns"] > start-SECOND_NS or row["available_epoch_ns"] > deadline:
            return None
        values = [row[c+"_iops"] for c in COMPONENTS]
        if any(not math.isfinite(v) or v < 0 for v in values):
            raise ValueError("invalid observed IOPS")
        if not math.isclose(values[0]+values[1], values[2], rel_tol=1e-8, abs_tol=1e-6):
            raise ValueError("read/write total mismatch")
        used.append(row)
        return values

    temporal = [get(target["scope"], target["entity"], start-i*SECOND_NS)
                for i in range(6, 1, -1)]
    if any(v is None for v in temporal):
        return None
    spatial = []
    for oss in range(96):
        values = get("oss", oss, start-2*SECOND_NS)
        spatial.append(values+[1.0] if values is not None else [None, None, None, 0.0])
    observed = [v for v in spatial if v[3]]
    if len(observed) < min_oss:
        return None
    auxiliary = []
    for client in site["clients"]:
        entity = client["hostname"]
        if target["scope"]!="client_nfs_rpc" or entity != target["entity"]:
            value = get("client_nfs_rpc", entity, start-2*SECOND_NS)
            if value is not None:
                auxiliary.append(value[2])
    aux_count = len(site["clients"])-int(target["scope"]=="client_nfs_rpc")
    cross = temporal[-1]+[sum(v[0] for v in observed), sum(v[1] for v in observed),
                            len(observed)/96.0,
                            sum(auxiliary) if auxiliary else None,
                            len(auxiliary)/aux_count if aux_count else 0.0]
    masks = [1.0 if value is not None else 0.0 for value in cross]
    # Zero is a tensor placeholder only; nullable observations remain in JSON.
    flattened = [v for row in temporal for v in row]
    flattened += [0.0 if v is None else v for row in spatial for v in row]
    flattened += [0.0 if v is None else v for v in cross]+masks
    assert len(flattened) == FEATURE_COUNT
    return {"features": flattened, "spatial_observations": spatial, "cross_observations": cross,
            "feature_end_ns": max(r["window_end_ns"] for r in used),
            "features_available_ns": max(r["available_epoch_ns"] for r in used),
            "observed_oss": len(observed), "observed_aux_clients": len(auxiliary),
            "attribution": "storage_context_only", "topology_hash": canonical_hash(topology(site))}


def feature_rows(windows, site, run_id, role, issue_margin_ms=30, target=None):
    target_identity=target or TARGET
    index = index_windows(windows)
    for (scope, entity, start), target in sorted(index.items()):
        if (scope, entity) != (target_identity["scope"], target_identity["entity"]) or not target.get("complete"):
            continue
        result = causal_features(index, start, site, issue_margin_ms,target=target_identity)
        if result is None:
            continue
        y = [target[c+"_iops"] for c in ("read", "write")]
        if any(not math.isfinite(v) or v < 0 for v in y):
            raise ValueError("invalid target")
        yield dict(result, run_id=run_id, role=role, target=target_identity,
                   window_start_ns=start, window_end_ns=target["window_end_ns"],
                   target_read_write=y, target_iops=sum(y))


def host_windows(windows, site):
    """Emit physical-host sums only when every member OSS is observed."""
    index = index_windows(windows)
    starts = sorted({t for s, e, t in index if s == "oss"})
    for node in site["storage_nodes"]:
        for start in starts:
            rows = [index.get(("oss", str(i), start)) for i in node["oss_ids"]]
            complete = all(r is not None and r.get("complete") for r in rows)
            out = {"scope": "host_oss", "entity": node["hostname"],
                   "window_start_ns": start, "window_end_ns": start+SECOND_NS, "complete": complete,
                   "n_ticks":10 if complete else 0,"unit":"ops/s"}
            if complete:
                out.update({c+"_iops": sum(r[c+"_iops"] for r in rows) for c in COMPONENTS})
                out["available_epoch_ns"] = max(r["available_epoch_ns"] for r in rows)
            else:
                out["error"] = "missing_or_invalid_member_oss"
            yield out
