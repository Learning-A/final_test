"""Local A/B resource slots compatible with the original exclusive training gate."""
from contextlib import contextmanager
import json
import os
from pathlib import Path
import re
import signal
import stat
import subprocess
import time

from .placement import require_training_host
from .offline_job import choose_idle_gpu

OWNERS = {
    '01a1fd50-435a-7d31-891c-88f94e0ea7eb': ('A', 'GPU-8923f552-5691-b49b-d2c8-87cb9292ae92'),
    '01a203f5-3aa6-7b21-8a1f-8fece4787432': ('B', 'GPU-ac63ac15-888b-c3fb-c42f-eb1cb26487c2'),
}
COORD = Path('/home/lmj/idfs_forecast_lab/model_b_20261122T074416Z/coordination')


@contextmanager
def slot_locks(coord, gpu_uuid, owner_thread):
    """Linux locks only; also used with isolated directories by integration tests."""
    import fcntl
    coord = Path(coord)
    if (coord.is_symlink() or not coord.is_dir() or coord.stat().st_uid != os.getuid()
            or not re.fullmatch(r'GPU-[a-f0-9-]{36}', gpu_uuid)
            or not re.fullmatch(r'[a-f0-9-]{36}', owner_thread)):
        raise ValueError('owned existing coordination directory and UUIDs required')
    fds = []
    try:
        for name, mode, create in (
                ('gpu_training.lock', fcntl.LOCK_SH, False),
                ('gpu-' + gpu_uuid + '.lock', fcntl.LOCK_EX, True),
                ('owner-' + owner_thread + '.lock', fcntl.LOCK_EX, True)):
            fd = os.open(str(coord / name), os.O_RDWR | os.O_NOFOLLOW |
                         (os.O_CREAT if create else 0), 0o600)
            fds.append(fd)
            info = os.fstat(fd)
            if info.st_uid != os.getuid() or not stat.S_ISREG(info.st_mode):
                raise ValueError('lock must be an owned regular file')
            fcntl.flock(fd, mode | fcntl.LOCK_NB)
        yield tuple(fds)
    finally:
        # No explicit LOCK_UN: inherited descriptors protect children even if
        # their supervisor is killed. Last close releases each lock.
        for fd in reversed(fds):
            os.close(fd)


def run_leased(command, *, job_id, owner_thread, code_sha, data_hashes,
               budget_seconds=1800, use_gpu=False):
    """Synchronous supervisor: locks outlive the entire owned child process group."""
    job = require_training_host()
    owner, gpu = OWNERS[owner_thread]
    if job.name != job_id or not re.fullmatch(r'[a-zA-Z0-9_-]+', job_id):
        raise ValueError('unique owned job identity required')
    if not re.fullmatch('[a-f0-9]{40}', code_sha):
        raise ValueError('fixed source SHA required')
    if (not data_hashes or any(not re.fullmatch('[a-f0-9]{64}', v)
                               for v in data_hashes.values())):
        raise ValueError('fixed data hashes required')
    if type(budget_seconds) is not int or not 0 < budget_seconds <= 1800:
        raise ValueError('finite approved budget required')
    if any(os.environ.get(k) != '4' for k in
           ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS')):
        raise ValueError('CPU4 worker environment required')
    expected_cuda = gpu if use_gpu else ''
    if os.environ.get('CUDA_VISIBLE_DEVICES') != expected_cuda:
        raise ValueError('assigned single GPU or CPU-only environment required')
    if not isinstance(command, list) or not command or not all(isinstance(s, str) for s in command):
        raise ValueError('explicit argv required; shell execution prohibited')
    record = COORD / (job_id + '.json')
    with slot_locks(COORD, gpu, owner_thread) as fds:
        if use_gpu and choose_idle_gpu(gpu) != gpu:
            raise RuntimeError('assigned GPU is occupied')
        state = dict(job_id=job_id, owner=owner, owner_thread=owner_thread,
                     gpu_uuid=gpu if use_gpu else None, resource_slot=gpu,
                     cpu_threads=4, workers=0, code_sha=code_sha,
                     data_sha256=data_hashes, budget_seconds=budget_seconds,
                     supervisor_pid=os.getpid(), state='reserved', started_epoch=time.time())
        with record.open('x', encoding='utf-8') as f:
            json.dump(state, f, indent=2)
        child = None
        def save():
            temporary = record.with_suffix('.json.tmp')
            temporary.write_text(json.dumps(state, indent=2) + '\n', encoding='utf-8')
            temporary.replace(record)
        def stop_group():
            if child is None:
                return
            # Only the group created by this supervisor, never other job PIDs.
            try:
                os.killpg(child.pid, signal.SIGTERM)
            except ProcessLookupError:
                pass
            try:
                child.wait(timeout=5)
            except subprocess.TimeoutExpired:
                pass
            try:
                os.killpg(child.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            child.wait()
        old_handlers = {}
        def interrupted(signum, frame):
            raise KeyboardInterrupt('supervisor signal ' + str(signum))
        try:
            for sig in (signal.SIGTERM, signal.SIGINT):
                old_handlers[sig] = signal.signal(sig, interrupted)
            with (job / 'worker.stdout.log').open('x') as out, (job / 'worker.stderr.log').open('x') as err:
                child = subprocess.Popen(command, cwd=job, stdout=out, stderr=err,
                                         start_new_session=True, pass_fds=fds)
                state.update(state='running', pid=child.pid)
                save()
                code = child.wait(timeout=budget_seconds)
                if code:
                    raise RuntimeError('owned worker exited ' + str(code))
                state.update(state='completed', exit_code=code)
        except subprocess.TimeoutExpired:
            state['state'] = 'timed_out'
            raise
        except BaseException:
            state['state'] = 'failed_or_interrupted'
            raise
        finally:
            stop_group()
            for sig, previous in old_handlers.items():
                signal.signal(sig, previous)
            state['ended_epoch'] = time.time()
            save()
    return state
