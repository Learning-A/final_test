"""Explicit, plan-first handoffs between Windows, lmj GPU training and CPU inference."""
import hashlib
import json
from pathlib import Path,PurePosixPath
import re
import shlex
import stat

from .gpu import gpu_connection,query
from .io import load,save

TRAIN_MODULES=["__init__.py","io.py","evaluation.py","counters.py","dataset.py","journal.py",
               "runtime.py","online.py","training.py","models_torch.py","placement.py",
               "artifacts.py","offline_job.py","seal.py","joint.py","joint_training.py","moe.py"]


def digest(path):
    h=hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda:f.read(1024*1024),b""):h.update(block)
    return h.hexdigest()


def scoped_path(path,home,workspace):
    p=PurePosixPath(path);base=PurePosixPath(workspace);home=PurePosixPath(home)
    if not p.is_absolute() or ".." in p.parts or "\\" in str(p):raise ValueError("unsafe remote path")
    if home not in base.parents or (p!=base and base not in p.parents):raise ValueError("path outside isolated user workspace")
    return str(p)


def owned_directory(sftp,path,home,workspace,uid):
    """Only mkdir below the named private root; never chmod/chown existing paths."""
    scoped_path(path,home,workspace)
    current=PurePosixPath(home)
    entry=sftp.lstat(str(current))
    if not stat.S_ISDIR(entry.st_mode) or entry.st_uid!=uid:raise ValueError("home owner/type mismatch")
    remainder=PurePosixPath(path).relative_to(current)
    for part in remainder.parts:
        current=current/part
        try:entry=sftp.lstat(str(current))
        except FileNotFoundError:
            sftp.mkdir(str(current),mode=0o700);entry=sftp.lstat(str(current))
        if not stat.S_ISDIR(entry.st_mode) or entry.st_uid!=uid:raise ValueError("existing path is not an owned directory")


def upload_new(sftp,local,remote):
    with Path(local).open("rb") as source,sftp.open(remote,"wx") as target:
        if hasattr(target,"set_pipelined"):target.set_pipelined(True)
        for block in iter(lambda:source.read(1024*1024),b""):target.write(block)


def job_plan(host_config,spec,protocol_path):
    job_id=spec["job_id"]
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,90}",job_id):raise ValueError("invalid job_id")
    models={"last_value","rolling_mean","ridge","hist_gradient_boosting","lightgbm","xgboost","gru","tcn","joint_moe"}
    if spec["model"] not in models:raise ValueError("model needs a separate adapter")
    if not 1<=int(spec.get("cpu_threads",4))<=4:raise ValueError("CPU thread budget must be 1..4")
    if not 1<=int(spec.get("epochs",100))<=1000:raise ValueError("epochs must be in 1..1000")
    files=[];inputs=[];train=[];validation=[];missing=[]
    for role in ("train","validation"):
        for index,value in enumerate(spec[role]):
            extension="jsonl" if spec["model"]=="joint_moe" else "csv"
            local=Path(value).resolve();relative="data/%s_%d.%s"%(role,index,extension)
            if not local.is_file():missing.append(str(local))
            else:
                files.append({"local":str(local),"remote_relative":relative,"sha256":digest(local),"bytes":local.stat().st_size})
                inputs.append({"path":relative,"sha256":digest(local)})
            (train if role=="train" else validation).append(relative)
    for name in TRAIN_MODULES:
        local=Path(__file__).with_name(name)
        files.append({"local":str(local),"remote_relative":"src/idfs_lab/"+name,"sha256":digest(local),"bytes":local.stat().st_size})
    protocol=Path(protocol_path).resolve()
    files.append({"local":str(protocol),"remote_relative":"protocol.json","sha256":digest(protocol),"bytes":protocol.stat().st_size})
    inputs.append({"path":"protocol.json","sha256":digest(protocol)})
    if spec["model"]=="joint_moe":
        if spec["target"]["scope"]!="host_oss" or spec["target"]["component"]!="total":
            raise ValueError("joint MoE target must be a physical host OSS total")
        for key in ("site","study"):
            local=Path(spec[key]).resolve()
            if not local.is_file():raise ValueError("missing joint "+key+" configuration")
            files.append({"local":str(local),"remote_relative":key+".json","sha256":digest(local),"bytes":local.stat().st_size})
            inputs.append({"path":key+".json","sha256":digest(local)})
    job={"job_id":job_id,"model":spec["model"],"target":spec["target"],"train":train,"validation":validation,
         "inputs":inputs,"cpu_threads":min(int(spec.get("cpu_threads",4)),4),"epochs":int(spec.get("epochs",100)),
         "gpu_uuid":spec.get("gpu_uuid"),"data_loader_workers":0}
    root=host_config["remote_workspace"]+"/jobs/"+job_id
    scoped_path(root,"/home/lmj",host_config["remote_workspace"])
    python=host_config["python_by_family"]["neural" if spec["model"] in ("gru","tcn","joint_moe") else "tabular"]
    bootstrap="import sys;sys.path.insert(0,"+repr(root+"/src")+");from idfs_lab.offline_job import main;main()"
    command=shlex.join([python,"-I","-B","-c",bootstrap,"--job",root+"/job.json"])
    return {"role":"offline_training_on_gpu_host","user":"lmj","host":host_config["host"],
            "remote_job_root":root,"python":python,"files":files,"job":job,"command":command,
            "missing_inputs":missing,"upload_bytes":sum(f["bytes"] for f in files),
            "execute_ready":not missing,"changes_other_users":False,"starts_training":False}


def submit_job(host_config,spec,protocol_path,local_output,execute=False):
    plan=job_plan(host_config,spec,protocol_path)
    if not execute:return plan
    if not plan["execute_ready"]:raise ValueError("training inputs are missing")
    if plan["upload_bytes"]>host_config["max_upload_bytes"]:raise ValueError("upload exceeds project data budget")
    local_output=Path(local_output);local_output.mkdir(parents=True,exist_ok=False)
    save(local_output/"submission.json",plan)
    client=gpu_connection(host_config)
    try:
        identity=query(client,"owner","id -un; id -u")
        if identity["stdout"].splitlines()!=["lmj",str(host_config["observed_uid"])]:raise ValueError("training identity changed")
        sftp=client.open_sftp();workspace=host_config["remote_workspace"];root=plan["remote_job_root"]
        try:
            command=shlex.join([plan["python"],"-I","-B","-c",
                "import os;v=os.statvfs('/home/lmj');print(v.f_bavail*v.f_frsize)"])
            disk=query(client,"lmj_disk_free",command)
            if disk["exit"]:raise RuntimeError("could not inspect lmj free space")
            free=int(disk["stdout"].strip())
            if free-plan["upload_bytes"]<host_config["minimum_remaining_bytes"]:raise ValueError("insufficient free space under lmj home; no cleanup will be attempted")
            try:sftp.lstat(root)
            except FileNotFoundError:pass
            else:raise ValueError("job already exists; use a new job_id")
            owned_directory(sftp,root,"/home/lmj",workspace,host_config["observed_uid"])
            for entry in plan["files"]:
                target=root+"/"+entry["remote_relative"]
                owned_directory(sftp,str(PurePosixPath(target).parent),"/home/lmj",workspace,host_config["observed_uid"])
                upload_new(sftp,entry["local"],target)
            with sftp.open(root+"/job.json","wx") as f:f.write(json.dumps(plan["job"],ensure_ascii=False).encode())
        finally:sftp.close()
        _,out,err=client.exec_command(plan["command"],timeout=86400)
        with (local_output/"training_stdout.jsonl").open("x",encoding="utf-8") as log:
            for line in out:
                print(line.rstrip(),flush=True);log.write(line);log.flush()
        stderr=err.read().decode(errors="replace");code=out.channel.recv_exit_status()
        save(local_output/"training_exit.json",{"exit":code,"stderr":stderr,"remote_job_root":root})
        if code:raise RuntimeError("lmj training job failed; see its logs")
        return {"state":"completed","remote_job_root":root,"local_log":str(local_output)}
    finally:client.close()


def fetch_results(host_config,job_id,destination,execute=False):
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,90}",job_id):raise ValueError("invalid job_id")
    root=host_config["remote_workspace"]+"/jobs/"+job_id
    scoped_path(root,"/home/lmj",host_config["remote_workspace"])
    if not execute:return {"state":"plan_only","from":root+"/results","to":str(destination),"user":"lmj"}
    destination=Path(destination);destination.mkdir(parents=True,exist_ok=False)
    client=gpu_connection(host_config)
    try:
        sftp=client.open_sftp();downloaded=[];total=0
        def fetch(remote,local,depth=0):
            nonlocal total
            if depth>4:raise ValueError("unexpected result directory depth")
            entry=sftp.lstat(remote)
            if entry.st_uid!=host_config["observed_uid"] or stat.S_ISLNK(entry.st_mode):raise ValueError("result owner/type mismatch")
            if stat.S_ISDIR(entry.st_mode):
                local.mkdir(parents=True,exist_ok=True)
                for child in sftp.listdir(remote):fetch(remote+"/"+child,local/child,depth+1)
            elif stat.S_ISREG(entry.st_mode):
                total+=entry.st_size
                if total>2*1024**3:raise ValueError("result download budget exceeded")
                with sftp.open(remote,"rb") as src,local.open("xb") as dst:
                    for chunk in iter(lambda:src.read(1024*1024),b""):dst.write(chunk)
                downloaded.append({"file":str(local.relative_to(destination)),"sha256":digest(local)})
            else:raise ValueError("non-regular result")
        fetch(root+"/results",destination/"results")
        fetch(root+"/job_result.json",destination/"job_result.json")
        save(destination/"download_manifest.json",downloaded)
        return {"state":"downloaded","files":len(downloaded),"bytes":total}
    finally:client.close()


def stage_inference(site,bundle,execute=False):
    from .transport import SiteConnection
    bundle=Path(bundle).resolve();checksum=digest(bundle)
    destination="/home/hust/idfs_forecast_inference/releases/"+checksum[:16]
    plan={"state":"plan_only","bundle":str(bundle),"sha256":checksum,"host":site["control_host"],
          "user":"hust","destination":destination,"starts_service":False,"contains_training_code":False}
    if not execute:return plan
    metadata=load(str(bundle)+".json")
    if metadata.get("sha256")!=checksum or metadata.get("contains_training_code") is not False:raise ValueError("not a sealed inference-only bundle")
    with SiteConnection(site) as connection:
        client=connection.client(site["control_host"])
        try:
            r=query(client,"owner","id -un; id -u")
            if r["stdout"].splitlines()!=["hust","1000"]:raise ValueError("inference owner mismatch")
            sftp=client.open_sftp()
            try:
                owned_directory(sftp,destination,"/home/hust","/home/hust/idfs_forecast_inference",1000)
                upload_new(sftp,bundle,destination+"/inference.tar.gz")
            finally:sftp.close()
            source=r'''
import os,sys,tarfile,hashlib,pathlib
root=pathlib.Path(sys.argv[1]).resolve();archive=root/'inference.tar.gz'
if os.getuid()!=1000 or pathlib.Path('/home/hust/idfs_forecast_inference') not in root.parents:raise RuntimeError('owner/root mismatch')
if hashlib.sha256(archive.read_bytes()).hexdigest()!=sys.argv[2]:raise RuntimeError('bundle hash mismatch')
with tarfile.open(str(archive),'r:gz') as f:
 members=f.getmembers()
 if sum(m.size for m in members)>512*1024**2:raise RuntimeError('expanded bundle too large')
 for m in members:
  relative=pathlib.PurePosixPath(m.name)
  if relative.is_absolute() or '..' in relative.parts or not m.isfile():raise RuntimeError('unsafe archive member')
  if any(v in relative.parts for v in ['training.py','offline_job.py','models_torch.py','joint_training.py','moe.py']):raise RuntimeError('training code forbidden')
  target=root/str(relative);target.parent.mkdir(parents=True,exist_ok=True)
  with f.extractfile(m) as src,target.open('xb') as dst:dst.write(src.read())
print('staged_inference_only',str(root))
'''
            command=shlex.join(["python3","-I","-B","-c",source,destination,checksum])
            _,out,err=client.exec_command(command,timeout=30)
            message=out.read().decode();error=err.read().decode();code=out.channel.recv_exit_status()
            if code:raise RuntimeError(error)
            runtime_python=site.get("inference_python","python3")
            if runtime_python!="python3":scoped_path(runtime_python,"/home/hust","/home/hust/idfs_forecast_inference")
            runtime="cd "+shlex.quote(destination)+" && "+shlex.join([runtime_python,"-B","-m","idfs_lab.runtime","--check",destination+"/model","--out",destination+"/runtime_check.json"])
            _,out,err=client.exec_command(runtime,timeout=60)
            check=out.read().decode();error=err.read().decode();code=out.channel.recv_exit_status()
            if code:raise RuntimeError("inference staged but runtime preflight failed; service was not started: "+error)
            return dict(plan,state="staged_runtime_verified_not_started",message=message,runtime_check=check)
        finally:client.close()
