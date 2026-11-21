import concurrent.futures
import json
from pathlib import Path
import shlex
import time

from .io import append_event, canonical_hash, save,BufferedJSONLog
from .transport import SiteConnection


def targets(site):
    return ([{"kind": "oss", "host": n["management_ip"], "entity": n["hostname"]}
             for n in site["storage_nodes"]] +
            [{"kind": "client_nfs_rpc", "host": n["management_ip"], "entity": n["hostname"]}
             for n in site["clients"]])


def refresh(site, directory):
    """Bounded snapshots only: three samples per source, all artifacts local."""
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=False)
    save(directory/"site_used.json", site)
    records = []
    result = {"kind": "read_only_probe", "start_local_epoch_ns": time.time_ns(),
              "site_hash": canonical_hash(site), "targets": [], "not_training_data": True}
    with SiteConnection(site) as connection:
        leader = targets(site)[0]
        result["cluster"] = connection.run(leader, "timeout 12 idfs -s; timeout 12 idfs node ls all -f json")
        control = {"kind": "client_nfs_rpc", "host": site["control_host"]}
        metadata = r'''
import os,json,hashlib,glob,time,subprocess
base='/home/hust'
releases=[]
for path in glob.glob(base+'/idfs-workloads-*'):
 commit=path+'/COMMIT'
 if os.path.isfile(commit):
  releases.append({'path':path,'commit':open(commit).read().strip(),'mtime':os.stat(commit).st_mtime})
releases.sort(key=lambda v:v['mtime'],reverse=True)
out={'hostname':os.uname().nodename,'epoch_ns':int(time.time()*1e9),'releases':releases,'workloads':[]}
if releases:
 for p in sorted(glob.glob(releases[0]['path']+'/config/*/rendered/manifest.json')):
  raw=open(p,'rb').read();m=json.loads(raw.decode());f=m.get('preparation',{}).get('file') or m.get('data_file')
  entry={'id':m.get('id'),'path':p,'sha256':hashlib.sha256(raw).hexdigest(),'data_file':f,'capacity_bytes':m.get('capacity_bytes'),'runtime':m.get('runtime'),'profiles':[]}
  if f:
   try:
    st=os.stat(f);entry.update(actual_bytes=st.st_size,inode=st.st_ino)
   except Exception as e:entry['stat_error']=str(e)
  for v in m.get('profiles',{}).get('current',[]):
   entry['profiles'].append({k:v.get(k) for k in ['name','seconds','read_share','write_share','object_selection','object_sampling']})
  entry['phase_names']=[[v.get('name') for v in x.get('business_phases',[])] for x in m.get('profiles',{}).get('current',[])]
  out['workloads'].append(entry)
out['active_workload_processes']=subprocess.check_output(['ps','-eo','user,pid,ppid,etime,comm'],universal_newlines=True)
print(json.dumps(out))
'''
        result["control"] = connection.run(control, "python3 -c " + shlex.quote(metadata))
        if result["control"]["exit"] == 0:
            result["control_metadata"] = json.loads(result["control"]["stdout"])

        def one(target):
            local = []
            before = time.time_ns()
            try:
                config = dict(kind=target["kind"], entity=target["entity"], samples=3,
                              interval_ms=100, mountpoint=site["mountpoint"])
                reply = connection.stream(target, config, local.append, 35)
                return dict(target=target, records=local, reply=reply,
                            local_start_ns=before, local_end_ns=time.time_ns())
            except Exception as exc:
                return dict(target=target, records=local, error=type(exc).__name__+": "+str(exc),
                            local_start_ns=before, local_end_ns=time.time_ns())

        with concurrent.futures.ThreadPoolExecutor(max_workers=8) as pool:
            for item in pool.map(one, targets(site)):
                records.extend(item.pop("records"))
                result["targets"].append(item)
                print(item["target"]["host"], item.get("error", "read-only counters OK"), flush=True)
    for row in records:
        append_event(directory/"counter_snapshots.jsonl", row)
    result.update(end_local_epoch_ns=time.time_ns(), sample_records=len(records),
                  good_records=sum(bool(r.get("sample_ok")) for r in records),
                  observed_oss_ids=sorted({int(r["entity"]) for r in records if r["scope"]=="oss" and r.get("sample_ok")}),
                  observed_clients=sorted({r["entity"] for r in records if r["scope"]=="client_nfs_rpc" and r.get("sample_ok")}))
    save(directory/"probe.json", result)
    return result


def collect(site, protocol, directory, seconds, execute=False,on_sample=None):
    """Future operation: stdout-only remote samplers, locally persisted counters.

    No remote output files, workload start, daemonization, sudo or process control.
    """
    if not execute:
        return {"state":"plan_only", "seconds":seconds, "remote_disk_writes":False,
                "targets":targets(site), "instruction":"Use --execute only when collection is authorized."}
    if seconds <= 0 or seconds > 86400:
        raise ValueError("duration must be in (0,86400] seconds")
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=False)
    save(directory/"collection_manifest.json", {"kind":"counter_collection", "site":site,
         "protocol":protocol,"protocol_hash":canonical_hash(protocol), "duration_s":seconds,
         "no_workload_launched":True,"start_local_ns":time.time_ns()})
    jobs = targets(site)
    with SiteConnection(site) as connection:
        def one(target):
            name=target["kind"]+"_"+target["host"]+".jsonl"
            with BufferedJSONLog(directory/name) as writer:
                def emit(row):
                    writer.append(row)
                    if on_sample:on_sample(row)
                return connection.stream(target,dict(kind=target["kind"], entity=target["entity"],
                    samples=int(seconds*1000/protocol["sample_grid_ms"])+1,
                    interval_ms=protocol["sample_grid_ms"],mountpoint=site["mountpoint"]),
                    emit,seconds+90)
        # All targets must be active concurrently; never queue entire collection runs.
        with concurrent.futures.ThreadPoolExecutor(max_workers=len(jobs)) as pool:
            futures={pool.submit(one,t):t for t in jobs}
            outcomes=[]
            for future in concurrent.futures.as_completed(futures):
                target=futures[future]
                try:outcomes.append({"target":target,"result":future.result()})
                except Exception as exc:outcomes.append({"target":target,"error":str(exc)})
    save(directory/"collection_finished.json",{"outcomes":outcomes,"end_local_ns":time.time_ns()})
    return {"state":"complete" if all("error" not in o for o in outcomes) else "incomplete", "outcomes":outcomes}
