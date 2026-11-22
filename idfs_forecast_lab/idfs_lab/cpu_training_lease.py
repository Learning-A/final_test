"""CPU fits use the already published GPU-training flock, without reserving a GPU."""
from contextlib import contextmanager
import json
import os
import re
import time
from .placement import require_training_host


@contextmanager
def cpu_lease(job_id, code_sha, data_hashes, budget_seconds=1800):
    job=require_training_host()
    if job.name!=job_id or not re.fullmatch('[0-9a-f]{40}',code_sha):
        raise ValueError('new owned job and fixed source SHA required')
    if (not isinstance(data_hashes,dict) or not data_hashes
            or any(not isinstance(k,str) or not k or not isinstance(v,str)
                   or not re.fullmatch('[0-9a-f]{64}',v) for k,v in data_hashes.items())):
        raise ValueError('pinned data checksums required')
    if type(budget_seconds) is not int or not 0<budget_seconds<=1800:
        raise ValueError('approved finite budget required')
    if (any(os.environ.get(k)!='4' for k in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS'))
            or os.environ.get('CUDA_VISIBLE_DEVICES')!=''):
        raise ValueError('CPU4 and GPU-invisible worker environment required')
    import fcntl
    coord=job.parent/'coordination'
    lock=coord/'gpu_training.lock'
    if coord.is_symlink() or not coord.is_dir() or coord.stat().st_uid!=os.getuid() or not lock.is_file():
        raise ValueError('existing owned shared coordination lock required')
    fd=os.open(str(lock),os.O_RDWR|os.O_NOFOLLOW)
    record=coord/(job_id+'.json');state=None;created=False
    try:
        fcntl.flock(fd,fcntl.LOCK_EX|fcntl.LOCK_NB)
        if os.fstat(fd).st_uid!=os.getuid():
            raise ValueError('wrong shared lock owner')
        state=dict(job_id=job_id,owner='B',executor='B',pid=os.getpid(),gpu_uuid=None,
                   cpu_threads=4,workers=0,code_sha=code_sha,data_sha256=dict(data_hashes),
                   budget_seconds=budget_seconds,state='reserved',started_epoch=time.time(),coordination_issue=19)
        with record.open('x',encoding='utf-8') as f:
            json.dump(state,f,indent=2)
        created=True
        os.ftruncate(fd,0);os.write(fd,json.dumps(state).encode())
        yield json.loads(json.dumps(state))
        state['state']='released'
    except BaseException:
        if state is not None:
            state['state']='failed_or_interrupted'
        raise
    finally:
        try:
            if state is not None and created:
                state['ended_epoch']=time.time()
                record.write_text(json.dumps(state,indent=2),encoding='utf-8')
        finally:
            os.close(fd)
