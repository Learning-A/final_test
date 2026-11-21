"""One real smoke experiment, explicitly excluded from formal acceptance."""
import argparse
import hashlib
import json
from pathlib import Path
import shlex
import threading
import time

from .io import load,save,read_events,BufferedJSONLog,canonical_hash
from .joint import feature_rows,host_windows
from .transport import SiteConnection
from .workflow import owned_directory,upload_new,submit_job,fetch_results,stage_inference
from .probe import collect
from .evaluation import score

NAME="host_smoke_20261121"
BASE=Path("runs")/NAME


def prepare():
    site=load("config/site.json"); protocol=load("config/protocol_smoke.json")
    windows=list(read_events("runs/"+NAME+"_processed/windows.jsonl"))
    complete=[r for r in windows if r["scope"]=="host_oss" and r["complete"]]
    origin=min(r["window_start_ns"] for r in complete)
    train_start=origin+8*10**9;train_end=origin+80*10**9
    val_start=origin+90*10**9;val_end=origin+115*10**9
    study={"experiment_kind":"smoke_not_acceptance",
           "partition_kind":"chronological_same_capture_not_independent",
           "capture_id":NAME+"_capture","formal_acceptance":False,
           "rounds":[{"run_id":NAME+"_train_segment","role":"train","workload":"background_io_smoke",
                      "start_ns":train_start,"end_ns":train_end},
                     {"run_id":NAME+"_validation_segment","role":"validation","workload":"background_io_smoke",
                      "start_ns":val_start,"end_ns":val_end}]}
    BASE.mkdir(parents=True,exist_ok=False)
    save(BASE/"study.json",study)
    source_sha=hashlib.sha256(Path("runs/"+NAME+"_processed/windows.jsonl").read_bytes()).hexdigest()
    report={"experiment_kind":"smoke_not_acceptance","targets":[],"sampling_quality":load("runs/"+NAME+"_processed/quality.json")}
    for node in site["storage_nodes"]:
        target={"scope":"host_oss","entity":node["hostname"],"component":"total"}
        paths={};counts={}
        for segment in study["rounds"]:
            path=BASE/(node["hostname"]+"_"+segment["role"]+".jsonl")
            rows=[r for r in feature_rows(windows,site,segment["run_id"],segment["role"],target=target)
                  if segment["start_ns"]<=r["window_start_ns"]<segment["end_ns"]]
            if len(rows)<10:raise ValueError("insufficient valid smoke examples for "+node["hostname"])
            with BufferedJSONLog(path) as out:
                for row in rows:
                    row.update(workload="background_io_smoke",source_windows_sha256=source_sha,capture_id=study["capture_id"])
                    out.append(row)
            paths[segment["role"]]=str(path);counts[segment["role"]]=len(rows)
        spec={"job_id":NAME+"_"+node["hostname"],"model":"joint_moe","target":target,
              "train":[paths["train"]],"validation":[paths["validation"]],"site":"config/site.json",
              "study":str(BASE/"study.json"),"epochs":12,"cpu_threads":4,"gpu_uuid":None}
        save(BASE/(node["hostname"]+"_job.json"),spec)
        report["targets"].append({"target":target,"examples":counts})
    save(BASE/"preparation.json",report)
    return report


def runtime_setup(wheel_directory):
    """Offline wheels installed in a new private target, not the system Python."""
    site=load("config/site.json"); root="/home/hust/idfs_forecast_inference/runtime/"+NAME
    wheels=sorted(Path(wheel_directory).glob("*.whl"))
    if not wheels or not any(p.name.startswith("torch-") for p in wheels):raise ValueError("CPU wheels missing")
    with SiteConnection(site) as connection:
        client=connection.client(site["control_host"])
        try:
            sftp=client.open_sftp()
            try:
                try:sftp.lstat(root);raise ValueError("runtime already exists; do not overwrite")
                except FileNotFoundError:pass
                for sub in (root+"/wheels",root+"/tmp"):
                    owned_directory(sftp,sub,"/home/hust","/home/hust/idfs_forecast_inference",1000)
                for wheel in wheels:
                    print("upload CPU wheel",wheel.name,flush=True)
                    upload_new(sftp,wheel,root+"/wheels/"+wheel.name)
                manifest={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in wheels}
                with sftp.open(root+"/wheel_sha256.json","wx") as out:out.write(json.dumps(manifest))
            finally:sftp.close()
            source=r'''
import sys,os,json,hashlib,pathlib
root=pathlib.Path(sys.argv[1]).resolve()
if os.getuid()!=1000 or pathlib.Path('/home/hust/idfs_forecast_inference') not in root.parents:raise RuntimeError('scope mismatch')
for name,expected in json.loads((root/'wheel_sha256.json').read_text()).items():
 if hashlib.sha256((root/'wheels'/name).read_bytes()).hexdigest()!=expected:raise RuntimeError('wheel checksum mismatch')
os.environ.update(TMPDIR=str(root/'tmp'),PYTHONDONTWRITEBYTECODE='1',PIP_NO_CACHE_DIR='1')
sys.path.insert(0,str(next((root/'wheels').glob('pip-*.whl'))))
from pip._internal.cli.main import main
raise SystemExit(main(['install','--no-index','--find-links',str(root/'wheels'),'--target',str(root/'lib'),
 '--no-compile','--no-cache-dir','--disable-pip-version-check','torch==2.4.1+cpu','numpy==1.24.4']))
'''
            command=shlex.join(["python3","-B","-c",source,root])
            _,out,err=client.exec_command(command,timeout=300)
            stdout=out.read().decode();stderr=err.read().decode();code=out.channel.recv_exit_status()
            save(BASE/"runtime_install.json",{"exit":code,"stdout":stdout,"stderr":stderr,"root":root,"system_environment_modified":False})
            if code:raise RuntimeError("private CPU runtime installation failed; see runtime_install.json")
            wrapper="#!/bin/sh\nexport PYTHONPATH="+shlex.quote(root+"/lib")+"\nexport PYTHONNOUSERSITE=1\nexport OMP_NUM_THREADS=1\nexport MKL_NUM_THREADS=1\nexec /usr/bin/python3 -B \"$@\"\n"
            sftp=client.open_sftp()
            try:
                with sftp.open(root+"/python","wx") as out:out.write(wrapper)
                sftp.chmod(root+"/python",0o700)
            finally:sftp.close()
            command=shlex.join([root+"/python","-c","import torch,numpy;print(torch.__version__,numpy.__version__);print(torch.cuda.is_available())"])
            _,out,err=client.exec_command(command,timeout=60)
            result={"stdout":out.read().decode(),"stderr":err.read().decode(),"exit":out.channel.recv_exit_status(),"python":root+"/python"}
            save(BASE/"runtime_check.json",result)
            if result["exit"]:raise RuntimeError("private CPU runtime import failed")
            site["inference_python"]=root+"/python";save("config/site_smoke.json",site)
            return result
        finally:client.close()


def train_all():
    host=load("config/training_host.json"); results=[]
    for node in load("config/site.json")["storage_nodes"]:
        name=node["hostname"];spec=load(BASE/(name+"_job.json"))
        print("training own lmj GPU job",name,flush=True)
        result=submit_job(host,spec,"config/protocol_smoke.json",BASE/(name+"_submission"),execute=True)
        fetch_results(host,spec["job_id"],BASE/(name+"_returned"),execute=True)
        results.append({"host":name,"result":result})
        save(BASE/"training_status.json",results)
    return results


def stage_all():
    site=load("config/site_smoke.json");results=[]
    for node in site["storage_nodes"]:
        name=node["hostname"]
        bundle=BASE/(name+"_returned")/"results/inference.tar.gz"
        print("staging CPU inference",name,flush=True)
        results.append({"storage_host":name,**stage_inference(site,bundle,execute=True)})
        save(BASE/"staging.json",results)
    return results


def live(seconds=70):
    site=load("config/site_smoke.json");protocol=load("config/protocol_smoke.json")
    releases=load(BASE/"staging.json"); directory=BASE/"live"
    if len({r["storage_host"] for r in releases})!=8:raise ValueError("eight distinct storage targets required")
    directory.mkdir(parents=True,exist_ok=False)
    root="/home/hust/idfs_forecast_inference"
    services=[];locks={};errors=[]
    with SiteConnection(site) as connection:
        client=connection.client(site["control_host"])
        try:
            sftp=client.open_sftp()
            try:owned_directory(sftp,root+"/runs","/home/hust",root,1000)
            finally:sftp.close()
            for item in releases:
                name=item["storage_host"];release=item["destination"];output=root+"/runs/"+NAME+"_"+name
                command="cd "+shlex.quote(release)+" && "+shlex.join([site["inference_python"],"-m","idfs_lab.online",
                    "--model",release+"/model","--protocol",release+"/config/protocol.json","--out",output])
                stdin,stdout,stderr=client.exec_command(command,timeout=seconds+120)
                services.append({"host":name,"remote":output,"stdin":stdin,"stdout":stdout,"stderr":stderr})
                locks[name]=threading.Lock()
            _,out,err=client.exec_command("python3 -B -c 'import time;print(time.time_ns())'",timeout=15)
            anchor=int(out.read().decode().strip())
            def send(message):
                line=json.dumps(message,separators=(",",":"))+"\n"
                for service in services:
                    try:
                        with locks[service["host"]]:service["stdin"].write(line);service["stdin"].flush()
                    except Exception as exc:
                        errors.append({"host":service["host"],"error":str(exc)})
                        raise
            send({"type":"observation_start","epoch_ns":anchor,"evidence":{"kind":"background_observation_not_workload_first_IO"}})
            try:
                collected=collect(site,protocol,directory/"raw",seconds,execute=True,
                                  on_sample=lambda row:send({"type":"sample","record":row}))
            finally:
                for service in services:service["stdin"].channel.shutdown_write()
            results=[]
            for service in services:
                name=service["host"];stdout=service["stdout"].read().decode();stderr=service["stderr"].read().decode()
                code=service["stdout"].channel.recv_exit_status()
                result={"host":name,"exit":code,"stdout":stdout,"stderr":stderr,"remote":service["remote"]}
                target_dir=directory/name;target_dir.mkdir()
                sftp=client.open_sftp()
                try:
                    for entry in sftp.listdir_attr(service["remote"]):
                        if entry.filename.endswith((".json",".jsonl")):
                            with sftp.open(service["remote"]+"/"+entry.filename,"rb") as src,(target_dir/entry.filename).open("xb") as dst:
                                for chunk in iter(lambda:src.read(1024*1024),b""):dst.write(chunk)
                finally:sftp.close()
                save(target_dir/"download_sha256.json",{p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in target_dir.iterdir()})
                results.append(result)
            status={"experiment_kind":"smoke_not_acceptance","observation_epoch_ns":anchor,"services":results,
                    "collection":collected,"errors":errors,"real_tool_first_IO_claimed":False}
            save(directory/"status.json",status)
            return status
        finally:client.close()


def recompute():
    directory=BASE/"live";status=load(directory/"status.json");protocol=load("config/protocol_smoke.json")
    site=load("config/site.json");results=[]
    for node in site["storage_nodes"]:
        name=node["hostname"];root=directory/name
        hashes=load(root/"download_sha256.json")
        integrity=all(hashlib.sha256((root/p).read_bytes()).hexdigest()==v for p,v in hashes.items())
        predictions=list(read_events(root/"predictions.jsonl")) if (root/"predictions.jsonl").exists() else []
        windows=list(read_events(root/"windows.jsonl"))
        actual={r["window_start_ns"]:r for r in windows if r["scope"]=="host_oss" and r["entity"]==name and r["complete"]}
        source=index={(r["scope"],str(r["entity"]),r["window_start_ns"]):r for r in windows}
        aggregate_errors=[]
        for start,row in actual.items():
            members=[source.get(("oss",str(i),start)) for i in node["oss_ids"]]
            if any(r is None or not r["complete"] for r in members) or any(sum(r[c+"_iops"] for r in members)!=row[c+"_iops"] for c in ("read","write","total")):
                aggregate_errors.append(start)
        commits={r["window_start_ns"]:r["committed_epoch_ns"] for r in predictions if r.get("event")=="prediction_commit"}
        issued=[r for r in predictions if r.get("event")=="prediction"]
        pairs=[(r["prediction_iops"],actual[r["window_start_ns"]]["total_iops"]) for r in issued if r["window_start_ns"] in actual]
        late=sum(r["issue_epoch_ns"]>=r["window_start_ns"] or commits.get(r["window_start_ns"],10**30)>=r["window_start_ns"] for r in issued)
        service=next(r for r in status["services"] if r["host"]==name)
        result={"host":name,"integrity":integrity,"predictions":len(issued),"settled_pairs":len(pairs),"late_predictions":late,
                "host_aggregation_errors":len(aggregate_errors),"diagnostics_present":all(r.get("diagnostics") for r in issued),
                "accuracy_informational_only":score(pairs,protocol["mask"]),
                "functional_pass":service["exit"]==0 and integrity and not aggregate_errors and len(pairs)>=10 and late==0,
                "formal_acceptance":False}
        results.append(result)
    report={"experiment_kind":"smoke_not_acceptance","functional_pass":all(r["functional_pass"] for r in results),
            "nodes":results,"source_clock_precision_verified":False,
            "training_split":"chronological_same_capture_not_independent","extra_workload_started":False,
            "formal_accuracy_claim":False,"other_work_processes_modified":False}
    save(BASE/"smoke_report.json",report)
    return report


def main():
    p=argparse.ArgumentParser();p.add_argument("phase",choices=["prepare","runtime","train","stage","live","recompute"])
    a=p.parse_args()
    result={"prepare":prepare,"runtime":lambda:runtime_setup("evidence/smoke_cpu_wheels"),"train":train_all,
            "stage":stage_all,"live":live,"recompute":recompute}[a.phase]()
    print(json.dumps(result,ensure_ascii=False,indent=2))


if __name__=="__main__":main()
