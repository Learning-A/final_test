"""Forward read-only counter streams to a trained CPU inference service over SSH."""
import hashlib
import json
from pathlib import Path
import re
import shlex
import threading
import time

from .io import load,save
from .probe import collect
from .transport import SiteConnection
from .workflow import owned_directory,upload_new,scoped_path


def relay(site,protocol,release,selection_path,clock_path,first_io_path,directory,
          run_id,round_index,seconds=660,formal=False,execute=False):
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,90}",run_id):raise ValueError("invalid run ID")
    root="/home/hust/idfs_forecast_inference"
    scoped_path(release,"/home/hust",root)
    output=root+"/runs/"+run_id
    plan={"state":"plan_only","role":"cluster_cpu_inference_no_training","release":release,
          "output":output,"seconds":seconds,"first_io_file":str(first_io_path),"formal":formal,
          "starts_workload":False,"modifies_existing_pids":False}
    if not execute:return plan
    if Path(first_io_path).exists():raise ValueError("first-I/O event path must be new for this run")
    directory=Path(directory);directory.mkdir(parents=True,exist_ok=False)
    save(directory/"relay_plan.json",plan)
    incoming=root+"/inputs/"+run_id
    selection_path,clock_path=Path(selection_path),Path(clock_path)
    # A recording anchor removes the workload-first-IO requirement, not the
    # need for bounded relative source clocks in a formal accuracy claim.
    if formal and not load(clock_path).get("verified"):
        raise ValueError("formal run requires verified clock evidence")
    with SiteConnection(site) as connection:
        client=connection.client(site["control_host"])
        try:
            sftp=client.open_sftp()
            try:
                owned_directory(sftp,incoming,"/home/hust",root,1000)
                owned_directory(sftp,root+"/runs","/home/hust",root,1000)
                upload_new(sftp,selection_path,incoming+"/selection.json")
                upload_new(sftp,clock_path,incoming+"/clock_alignment.json")
            finally:sftp.close()
            command="cd "+shlex.quote(release)+" && "+shlex.join(["python3","-B","-m","idfs_lab.online",
                "--model",release+"/model","--protocol",release+"/config/protocol.json","--out",output,
                "--selection",incoming+"/selection.json","--clock",incoming+"/clock_alignment.json"])
            stdin,stdout,stderr=client.exec_command(command,timeout=seconds+150)
            lock=threading.Lock();stop=threading.Event();t0_sent=threading.Event();errors=[]
            def send(message):
                with lock:
                    stdin.write(json.dumps(message,separators=(",",":"))+"\n");stdin.flush()
            def watch_first_io():
                while not stop.wait(0.1):
                    p=Path(first_io_path)
                    if p.is_file():
                        try:
                            value=load(p)
                            if value.get("event") not in ("tool_first_io", "recording_start") or not isinstance(value.get("epoch_ns"),int) or value["epoch_ns"]<=0:
                                raise ValueError("recording_start or tool_first_io event required")
                            send({"type":"t0","epoch_ns":value["epoch_ns"],
                                  "evidence":{"file":str(p.resolve()),"sha256":hashlib.sha256(p.read_bytes()).hexdigest(),"details":value}})
                            t0_sent.set();return
                        except Exception as exc:errors.append(str(exc));return
            watcher=threading.Thread(target=watch_first_io,daemon=True);watcher.start()
            try:
                collected=collect(site,protocol,directory/"raw",seconds,execute=True,
                                  on_sample=lambda row:send({"type":"sample","record":row}))
            finally:
                stop.set();watcher.join(timeout=3);stdin.channel.shutdown_write()
            out=stdout.read().decode();err=stderr.read().decode();code=stdout.channel.recv_exit_status()
            status={"exit":code,"stdout":out,"stderr":err,"first_io_sent":t0_sent.is_set(),
                    "input_errors":errors,"collection":collected,"remote_run":output}
            save(directory/"inference_exit.json",status)
            if code or errors or not t0_sent.is_set():raise RuntimeError("inference run incomplete; see local/remote logs")
            seal="cd "+shlex.quote(release)+" && "+shlex.join(["python3","-B","-m","idfs_lab.seal",
                 "--run",output,"--round-id",run_id,"--round-index",str(round_index)]+(["--formal"] if formal else []))
            _,o,e=client.exec_command(seal,timeout=30);o.read();err=e.read().decode();code=o.channel.recv_exit_status()
            if code:raise RuntimeError("log sealing failed: "+err)
            sftp=client.open_sftp()
            try:
                for name in ["round_manifest.json","selection.json","clock_alignment.json","first_io.json",
                             "predictions.jsonl","windows.jsonl","settlement.jsonl","service_finished.json"]:
                    with sftp.open(output+"/"+name,"rb") as src,(directory/name).open("xb") as dst:
                        for chunk in iter(lambda:src.read(1024*1024),b""):dst.write(chunk)
            finally:sftp.close()
            return dict(plan,state="closed_sealed_downloaded",remote_run=output)
        finally:client.close()
