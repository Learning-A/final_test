"""Published cross-end GPU reservation; hold flock for entire child training."""
from contextlib import contextmanager
import json,os,time
from pathlib import Path
from .placement import require_training_host
from .offline_job import choose_idle_gpu


@contextmanager
def lease(job_id):
    import fcntl
    job=require_training_host()
    if job.name!=job_id:raise ValueError('lease identity differs from owned job')
    coord=job.parent/'coordination';coord.mkdir(exist_ok=True)
    if coord.is_symlink() or coord.stat().st_uid!=os.getuid():raise ValueError('unsafe coordination directory')
    fd=os.open(str(coord/'gpu_training.lock'),os.O_RDWR|os.O_CREAT|os.O_NOFOLLOW,0o600)
    record=coord/(job_id+'.json');state=None;record_created=False
    try:
        fcntl.flock(fd,fcntl.LOCK_EX|fcntl.LOCK_NB)
        if os.fstat(fd).st_uid!=os.getuid():raise ValueError('wrong lock owner')
        gpu=choose_idle_gpu()
        state={'job_id':job_id,'executor':'B','pid':os.getpid(),'gpu_uuid':gpu,'state':'reserved','started_epoch':time.time(),'coordination_issue':17}
        with record.open('x') as f:json.dump(state,f)
        record_created=True
        os.ftruncate(fd,0);os.write(fd,json.dumps(state).encode())
        # Recheck while holding the common lock, immediately before child launch.
        if choose_idle_gpu(gpu)!=gpu:raise RuntimeError('GPU became busy')
        yield gpu
        state['state']='released'
    except BaseException:
        if state is not None:state['state']='failed_or_interrupted'
        raise
    finally:
        if state is not None and record_created:
            state['ended_epoch']=time.time();record.write_text(json.dumps(state,indent=2))
        os.close(fd)
