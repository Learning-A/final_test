"""Stream read-only counters to stdout. No remote files, daemons, signals or resets.

This file is sent as Python source through SSH -c, never installed remotely.
Keep compatible with the storage node's Python 3.6 standard library.
"""
import concurrent.futures
import glob
import json
import math
import os
import re
import socket
import sys
import time


def ns():
    return int(time.time() * 1000000000)


def mono():
    return int(time.monotonic() * 1000000000)


def decode_loose(raw):
    text = raw.decode("utf-8", errors="strict")
    # Match JSON scalar values only, never quoted strings or names.
    fixed, changes = re.subn(r'(?<=:)\s*[-+]?(?:nan|inf(?:inity)?)(?=\s*[,}])',
                             ' null', text, flags=re.I)
    return json.loads(fixed), changes


def read_exact(s, size):
    result = b""
    while len(result) < size:
        chunk = s.recv(min(size - len(result), 65536))
        if not chunk:
            raise IOError("truncated admin socket reply")
        result += chunk
    return result


def read_perf(path):
    s = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    s.settimeout(0.8)
    try:
        s.connect(path)
        # The live command schema advertises the optional "logger" filter.
        # Read only OSS counters instead of serializing/parsing every engine group.
        s.sendall(b'{"prefix":"perf dump","logger":"oss"}\n')
        length = int.from_bytes(read_exact(s, 4), "big")
        if length > 5000000:
            raise ValueError("oversized admin socket reply")
        body = read_exact(s, length)
        value, count = decode_loose(body)
        return value, count, len(body)
    finally:
        s.close()


def instance(path):
    s = os.stat(path)
    return str(s.st_ino) + ":" + str(int(s.st_mtime * 1e9))


def mount_counters(target):
    with open("/proc/self/mountstats") as f:
        text = f.read()
    sections = [s for s in text.split("device ") if
                " mounted on " + target + " " in s.split("\n", 1)[0]]
    if len(sections) != 1:
        raise ValueError("expected one NFS mount")
    section = sections[0]
    fields = {}
    for line in section.splitlines():
        name, _, rest = line.strip().partition(":")
        if name in ("READ", "WRITE", "COMMIT", "GETATTR"):
            fields[name] = [int(x) for x in rest.split()]
    if any(len(fields.get(k, [])) < 8 for k in ("READ", "WRITE")):
        raise ValueError("READ/WRITE fields unavailable")
    # mount ID identifies a remount even when source/target are unchanged.
    with open("/proc/self/mountinfo") as f:
        ids = [line.split()[0] for line in f if line.split()[4] == target]
    if len(ids) != 1:
        raise ValueError("ambiguous mount ID")
    return dict(read_count=fields["READ"][0], write_count=fields["WRITE"][0],
                source_id=section.split()[0] + "|" + target,
                instance_id=ids[0], rpc_fields=fields)


def main(config):
    interval_ns = int(config.get("interval_ms", 100) * 1000000)
    count = int(config.get("samples", 2))
    if interval_ns < 100000000 or count < 1:
        raise ValueError("minimum interval is 100ms and samples must be positive")
    kind = config["kind"]
    if kind not in ("oss", "client_nfs_rpc"):
        raise ValueError("unsupported source")
    host = socket.gethostname()
    with open("/proc/sys/kernel/random/boot_id") as f:
        boot = f.read().strip()
    items = [(str(int(re.search(r'idfs-oss\.(\d+)\.asok$', p).group(1))), p)
             for p in glob.glob("/var/run/idfs/idfs-oss.*.asok")] if kind == "oss" else [(config.get("entity", host), config["mountpoint"])]
    if not items:
        raise ValueError("no OSS sockets found")
    start_wall = ((ns() // interval_ns) + 2) * interval_ns
    start_mono = mono() + (start_wall - ns())
    previous_offset = None

    def sample(item, scheduled):
        entity, path = item
        begin_wall, begin_mono = ns(), mono()
        row = dict(schema_version=1, scope=kind, entity=entity, host=host,
                   boot_id=boot, scheduled_epoch_ns=scheduled,
                   sample_interval_ns=interval_ns, read_start_epoch_ns=begin_wall,
                   read_start_mono_ns=begin_mono, sample_ok=False)
        try:
            if kind == "oss":
                identity_before = instance(path)
                obj, changes, length = read_perf(path)
                identity_after = instance(path)
                if identity_before != identity_after:
                    raise ValueError("daemon socket changed during read")
                core = obj.get("oss", {})
                row.update(source_id=path, instance_id=identity_after,
                           read_count=core.get("op_r"), write_count=core.get("op_w"),
                           read_bytes=core.get("op_out_bytes"), write_bytes=core.get("op_in_bytes"),
                           latency={k: core.get(k) for k in ("op_latency", "op_r_latency", "op_w_latency")},
                           op_wip=core.get("op_wip"), nonfinite_replaced=changes,
                           response_bytes=length, source_query="perf dump logger=oss")
            else:
                row.update(mount_counters(path))
            for key in ("read_count", "write_count"):
                value = row.get(key)
                if not isinstance(value, (int, float)) or isinstance(value, bool) or not math.isfinite(value) or value < 0:
                    raise ValueError("invalid core counter " + key)
            row["sample_ok"] = True
        except Exception as exc:
            row["error"] = type(exc).__name__ + ": " + str(exc)
        row.update(read_end_epoch_ns=ns(), read_end_mono_ns=mono())
        row["read_duration_ns"] = row["read_end_mono_ns"] - begin_mono
        return row

    with concurrent.futures.ThreadPoolExecutor(max_workers=min(12, len(items))) as pool:
        for tick in range(count):
            scheduled = start_wall + tick * interval_ns
            deadline = start_mono + tick * interval_ns
            wait = (deadline - mono()) / 1e9
            if wait > 0:
                time.sleep(wait)
            offset = ns() - mono()
            step = previous_offset is not None and abs(offset - previous_offset) > 25000000
            previous_offset = offset
            rows = list(pool.map(lambda item: sample(item, scheduled), items))
            for row in rows:
                row["clock_step"] = step
                # Non-finite ancillary values remain null, with an explicit replacement count.
                print(json.dumps(row, allow_nan=False, separators=(",", ":")), flush=True)


if __name__ == "__main__":
    main(json.loads(sys.argv[1]))
