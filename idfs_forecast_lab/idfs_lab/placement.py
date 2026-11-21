import getpass
import os
from pathlib import Path
import socket


def require_training_host():
    """No fitting on Windows or IDFS nodes, even if training dependencies exist."""
    if getpass.getuser()!="lmj" or socket.gethostname().split(".")[0]!="b507":
        raise RuntimeError("training is restricted to lmj@b507 (192.168.1.13)")
    root=os.environ.get("IDFS_GPU_JOB_ROOT")
    if not root:raise RuntimeError("use the isolated GPU job worker")
    resolved=Path(root).resolve()
    allowed=Path("/home/lmj/idfs_forecast_lab").resolve()
    if allowed not in resolved.parents or resolved.stat().st_uid!=os.getuid():
        raise RuntimeError("training job must be in an lmj-owned project directory")
    return resolved
