import hashlib
import json
import os
import time
from pathlib import Path


def canonical_hash(value):
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True,
                                    separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def load(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def save(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    os.replace(str(temporary), str(path))


def append_event(path, event):
    """Do not round measurements or timestamps; fsync each audit event."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8", newline="\n") as stream:
        stream.write(json.dumps(event, ensure_ascii=False, separators=(",", ":"), allow_nan=False) + "\n")
        stream.flush()
        os.fsync(stream.fileno())


def read_events(path):
    with Path(path).open(encoding="utf-8") as stream:
        for number, line in enumerate(stream, 1):
            if line.strip():
                try:
                    yield json.loads(line)
                except ValueError as exc:
                    raise ValueError("invalid JSONL at %s:%d: %s" % (path, number, exc))


class BufferedJSONLog:
    """Batch raw counter fsyncs; prediction audit events still use append_event."""
    def __init__(self,path,interval_s=0.1):
        self.path=Path(path);self.path.parent.mkdir(parents=True,exist_ok=True)
        self.file=self.path.open("x",encoding="utf-8",newline="\n")
        self.interval=interval_s;self.last=time.monotonic()

    def append(self,value):
        self.file.write(json.dumps(value,ensure_ascii=False,separators=(",",":"),allow_nan=False)+"\n")
        self.flush()

    def flush(self,force=False):
        if force or time.monotonic()-self.last>=self.interval:
            self.file.flush();os.fsync(self.file.fileno());self.last=time.monotonic()

    def __enter__(self):return self

    def __exit__(self,*unused):
        self.flush(force=True);self.file.close()
