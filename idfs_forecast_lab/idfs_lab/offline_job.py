"""Foreground training worker. Only its own new lmj job directory is writable."""
import argparse
import getpass
import hashlib
import json
import os
from pathlib import Path
import socket
import subprocess

from .io import load,save


def choose_idle_gpu(requested=None):
    command=["nvidia-smi","--query-gpu=uuid,memory.used,utilization.gpu","--format=csv,noheader,nounits"]
    lines=subprocess.check_output(command,text=True).splitlines()
    busy_output=subprocess.check_output(["nvidia-smi","--query-compute-apps=gpu_uuid","--format=csv,noheader,nounits"],text=True)
    busy={v.strip() for v in busy_output.splitlines() if v.strip()}
    available=[]
    for line in lines:
        uuid,used,util=[v.strip() for v in line.split(",")]
        if uuid not in busy and int(used)<=64 and int(util)==0 and (not requested or requested==uuid):available.append(uuid)
    if not available:raise RuntimeError("no idle requested GPU; no existing process will be touched")
    return available[0]


def run_job(job_path):
    job_path=Path(job_path).resolve();root=job_path.parent
    if getpass.getuser()!="lmj" or socket.gethostname().split(".")[0]!="b507":
        raise RuntimeError("offline worker only runs as lmj@b507")
    if Path("/home/lmj/idfs_forecast_lab") not in root.parents or root.stat().st_uid!=os.getuid():
        raise RuntimeError("job is outside the lmj project root")
    job=load(job_path)
    threads=int(job.get("cpu_threads",4))
    if not 1<=threads<=4:raise ValueError("default shared-server CPU limit is 1..4 threads")
    for name in ("OMP_NUM_THREADS","MKL_NUM_THREADS","OPENBLAS_NUM_THREADS","NUMEXPR_NUM_THREADS"):
        os.environ[name]=str(threads)
    os.environ.update(IDFS_GPU_JOB_ROOT=str(root),PYTHONDONTWRITEBYTECODE="1",
                      CUBLAS_WORKSPACE_CONFIG=":4096:8",CUDA_DEVICE_ORDER="PCI_BUS_ID")
    os.nice(10)  # affects this new worker only
    lock=root/"worker.lock"
    fd=os.open(str(lock),os.O_WRONLY|os.O_CREAT|os.O_EXCL,0o600)
    os.write(fd,str(os.getpid()).encode());os.close(fd)
    result={"job_id":job["job_id"],"user":"lmj","pid":os.getpid(),"other_processes_modified":False}
    try:
        if (root/"results").exists() or (root/"results").is_symlink():raise ValueError("results already exist; use a new job directory")
        (root/"results").mkdir(mode=0o700)
        scratch=root/"scratch";scratch.mkdir(mode=0o700,exist_ok=False)
        for key,name in {"TMPDIR":"tmp","XDG_CACHE_HOME":"cache","CUDA_CACHE_PATH":"cuda",
                         "TORCH_HOME":"torch","TORCH_EXTENSIONS_DIR":"torch_extensions",
                         "HF_HOME":"huggingface","JOBLIB_TEMP_FOLDER":"joblib"}.items():
            path=scratch/name;path.mkdir(mode=0o700);os.environ[key]=str(path)
        for entry in job["inputs"]:
            path=(root/entry["path"]).resolve()
            if root not in path.parents:raise ValueError("input outside job directory")
            if hashlib.sha256(path.read_bytes()).hexdigest()!=entry["sha256"]:raise ValueError("input checksum mismatch")
        gpu=choose_idle_gpu(job.get("gpu_uuid")) if job["model"] in ("gru","tcn","joint_moe") else None
        os.environ["CUDA_VISIBLE_DEVICES"]=gpu or ""
        result["gpu_uuid"]=gpu
        from .training import train
        from .artifacts import export_cpu,bundle_inference
        if job["model"]=="joint_moe":
            from .joint_training import train_joint
            metadata,exported=train_joint([str(root/p) for p in job["train"]],[str(root/p) for p in job["validation"]],
                root/"results/model",load(root/"protocol.json"),load(root/"study.json"),load(root/"site.json"),
                int(job.get("epochs",100)),threads,target=job["target"])
            export_dir=root/"results/model/cpu_export"
        else:
            metadata=train([str(root/p) for p in job["train"]],[str(root/p) for p in job["validation"]],
                           job["target"],job["model"],root/"results/model",load(root/"protocol.json"),
                           int(job.get("epochs",100)),threads)
            export_dir=root/"results/cpu_export"
            exported=export_cpu(root/"results/model",[str(root/p) for p in job["validation"]],export_dir)
        bundle=bundle_inference(export_dir,root/"protocol.json",root/"results/inference.tar.gz")
        result.update(state="completed",model=metadata,export=exported,bundle=bundle)
    except Exception as exc:
        result.update(state="failed",error=type(exc).__name__+": "+str(exc))
        raise
    finally:
        save(root/"job_result.json",result)
        # Never signal a PID or unlink a pre-existing lock: this file was created above.
        if lock.read_text()==str(os.getpid()):lock.unlink()
    return result


def main():
    p=argparse.ArgumentParser();p.add_argument("--job",required=True);a=p.parse_args()
    print(json.dumps(run_job(a.job),ensure_ascii=False),flush=True)


if __name__=="__main__":main()
