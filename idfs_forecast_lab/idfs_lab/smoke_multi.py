"""Bounded shared-stream smoke retry; only new private directories are created."""
import hashlib
import json
from pathlib import Path
import shlex
import threading
import queue
import time
from .io import load,save,read_events
from .transport import SiteConnection
from .workflow import owned_directory,upload_new
from .probe import collect
from .artifacts import INFERENCE_MODULES
from .smoke import BASE,NAME
from .evaluation import score


def run(seconds=80,version='v3'):
    site=load('config/site_smoke.json');protocol=load('config/protocol_smoke.json')
    protocol.update(issue_margin_ms=200,evaluation_from_s=20,evaluation_to_s=60,
                    service_kind='single_stream_eight_models',issue_margin_note='smoke scheduling budget for sequential CPU models; not a formal protocol change')
    directory=BASE/('live_'+version);directory.mkdir(parents=True,exist_ok=False)
    save(directory/'protocol.json',protocol)
    entries=[{'host':r['storage_host'],'model':r['destination']+'/model'} for r in load(BASE/('staging_'+version+'.json'))]
    save(directory/'models.json',entries)
    root='/home/hust/idfs_forecast_inference';service=root+'/services/'+NAME+'_'+version;remote=root+'/runs/'+NAME+'_'+version
    with SiteConnection(site) as connection:
        client=connection.client(site['control_host'])
        try:
            sftp=client.open_sftp()
            try:
                owned_directory(sftp,service+'/idfs_lab','/home/hust',root,1000)
                for name in INFERENCE_MODULES+['multi_online.py']:
                    upload_new(sftp,Path(__file__).with_name(name),service+'/idfs_lab/'+name)
                for name in ['protocol.json','models.json']:upload_new(sftp,directory/name,service+'/'+name)
            finally:sftp.close()
            command='cd '+shlex.quote(service)+' && '+shlex.join([site['inference_python'],'-m','idfs_lab.multi_online',
                '--models',service+'/models.json','--protocol',service+'/protocol.json','--out',remote])
            stdin,stdout,stderr=client.exec_command(command,timeout=seconds+120)
            ready=stdout.readline()
            if not ready or json.loads(ready).get('event')!='ready':raise RuntimeError('shared CPU service did not become ready: '+stderr.read().decode())
            save(directory/'ready.json',json.loads(ready))
            _,out,err=client.exec_command("python3 -B -c 'import time;print(time.time_ns())'",timeout=15)
            anchor=int(out.read().decode());lock=threading.Lock()
            def send(message):
                if (directory/'STOP_THIS_SMOKE').exists():raise RuntimeError('own smoke stop requested')
                with lock:stdin.write(json.dumps(message,separators=(',',':'))+'\n');stdin.flush()
            send({'type':'observation_start','epoch_ns':anchor})
            outgoing=queue.Queue(maxsize=5000);sender_errors=[]
            def sender():
                pending=[];deadline=time.monotonic()+0.05
                try:
                    while True:
                        try:row=outgoing.get(timeout=max(0.001,deadline-time.monotonic()))
                        except queue.Empty:row='flush'
                        if row is None:
                            if pending:send({'type':'samples','records':pending})
                            return
                        if row!='flush':pending.append(row)
                        if pending and (len(pending)>=128 or time.monotonic()>=deadline):
                            send({'type':'samples','records':pending});pending=[];deadline=time.monotonic()+0.05
                        elif not pending:deadline=time.monotonic()+0.05
                except Exception as exc:sender_errors.append(str(exc))
            writer=threading.Thread(target=sender,daemon=True);writer.start()
            def enqueue(row):
                if sender_errors:raise RuntimeError(sender_errors[0])
                outgoing.put(row,timeout=10)
            try:
                collection=collect(site,protocol,directory/'raw',seconds,execute=True,
                    on_sample=enqueue)
            finally:
                outgoing.put(None,timeout=10);writer.join(timeout=30);stdin.channel.shutdown_write()
            if writer.is_alive() or sender_errors:raise RuntimeError('batch sender did not finish: '+str(sender_errors))
            tail=stdout.read().decode();error=stderr.read().decode();code=stdout.channel.recv_exit_status()
            save(directory/'status.json',{'exit':code,'stdout':tail,'stderr':error,'remote':remote,'collection':collection,
                'observation_epoch_ns':anchor,'first_tool_io_claimed':False,'formal_acceptance':False})
            sftp=client.open_sftp()
            try:
                def download(remote_dir,local_dir):
                    local_dir.mkdir(parents=True,exist_ok=True)
                    for entry in sftp.listdir_attr(remote_dir):
                        import stat
                        if stat.S_ISDIR(entry.st_mode):download(remote_dir+'/'+entry.filename,local_dir/entry.filename)
                        elif entry.filename!='raw_counters.jsonl' and entry.filename.endswith(('.json','.jsonl')):
                            # Shared local collector already archived all raw source counters.
                            sftp.get(remote_dir+'/'+entry.filename,str(local_dir/entry.filename),prefetch=True)
                download(remote,directory/'inference')
            finally:sftp.close()
            save(directory/'download_sha256.json',{p.relative_to(directory/'inference').as_posix():hashlib.sha256(p.read_bytes()).hexdigest()
                 for p in (directory/'inference').rglob('*') if p.is_file()})
            if code:raise RuntimeError('shared service failed; see status.json')
        finally:client.close()
    return recompute(version)


def recompute(version='v3'):
    directory=BASE/('live_'+version);root=directory/'inference';status=load(directory/'status.json');protocol=load(directory/'protocol.json')
    hashes=load(directory/'download_sha256.json')
    integrity=all(hashlib.sha256((root/p).read_bytes()).hexdigest()==h for p,h in hashes.items())
    windows=list(read_events(root/'windows.jsonl'));index={(r['scope'],str(r['entity']),r['window_start_ns']):r for r in windows}
    results=[]
    for node in load('config/site.json')['storage_nodes']:
        name=node['hostname'];logs=root/name
        rows=list(read_events(logs/'predictions.jsonl')) if (logs/'predictions.jsonl').exists() else []
        issued=[r for r in rows if r.get('event')=='prediction']
        commits={r['window_start_ns']:r['committed_epoch_ns'] for r in rows if r.get('event')=='prediction_commit'}
        actual={r['window_start_ns']:r for r in windows if r['scope']=='host_oss' and r['entity']==name and r['complete']}
        aggregate_errors=0
        for start,row in actual.items():
            members=[index.get(('oss',str(i),start)) for i in node['oss_ids']]
            if any(r is None or not r['complete'] for r in members) or any(sum(r[c+'_iops'] for r in members)!=row[c+'_iops'] for c in ('read','write','total')):aggregate_errors+=1
        pairs=[(r['prediction_iops'],actual[r['window_start_ns']]['total_iops']) for r in issued if r['window_start_ns'] in actual]
        late=sum(r['issue_epoch_ns']>=r['window_start_ns'] or commits.get(r['window_start_ns'],10**30)>=r['window_start_ns'] for r in issued)
        causal=all(r['causal'] for r in issued)
        result={'host':name,'predictions':len(issued),'settled_pairs':len(pairs),'late_predictions':late,'causal':causal,
                'host_aggregation_errors':aggregate_errors,'integrity':integrity,
                'diagnostics_present':bool(issued) and all(r.get('diagnostics') for r in issued),
                'accuracy_informational_only':score(pairs,protocol['mask']),
                'functional_pass':status['exit']==0 and integrity and not aggregate_errors and len(pairs)>=10 and late==0 and causal,
                'formal_acceptance':False}
        results.append(result)
    result={'experiment_kind':'smoke_not_acceptance','functional_pass':all(r['functional_pass'] for r in results) and status['collection']['state']=='complete',
            'nodes':results,'single_shared_counter_stream':True,'collection_state':status['collection']['state'],
            'source_clock_precision_verified':False,'training_split':'chronological_same_capture_not_independent',
            'first_failed_attempt':'live: duplicate eight streams caused backpressure; only its own controller was ended',
            'batch_forwarding':True,'attempt':version,
            'formal_accuracy_claim':False,'extra_workload_started':False,'other_work_processes_modified':False}
    save(BASE/'smoke_report.json',result)
    return result


def recompute_raw(version='v3'):
    """Rebuild node actuals from cumulative source counters, not logged actuals."""
    from .dataset import process
    directory=BASE/('live_'+version);offline=directory/'offline_recompute'
    protocol=load(directory/'protocol.json')
    quality=process(directory/'raw',offline,protocol)
    live=[r for r in read_events(directory/'inference/windows.jsonl') if r['scope']=='host_oss' and r['complete']]
    rebuilt=[r for r in read_events(offline/'windows.jsonl') if r['scope']=='host_oss' and r['complete']]
    key=lambda r:(r['entity'],r['window_start_ns'])
    left={key(r):r for r in live};right={key(r):r for r in rebuilt}
    errors=[]
    if set(left)!=set(right):errors.append('complete_node_window_set_differs')
    for k in set(left)&set(right):
        if any(left[k][c+'_iops']!=right[k][c+'_iops'] for c in ('read','write','total')):errors.append('node_count_differs:'+str(k))
    report=load(BASE/'smoke_report.json')
    for node in report['nodes']:
        issued=[r for r in read_events(directory/'inference'/node['host']/'predictions.jsonl') if r.get('event')=='prediction']
        pairs=[(r['prediction_iops'],right[(node['host'],r['window_start_ns'])]['total_iops']) for r in issued
               if (node['host'],r['window_start_ns']) in right]
        metrics=score(pairs,protocol['mask'])
        matches=metrics==node['accuracy_informational_only']
        if not matches:errors.append('accuracy_recomputation_differs:'+node['host'])
        node['raw_counter_recomputed_pairs']=len(pairs);node['raw_counter_accuracy_matches']=matches
    summary={'passed':not errors,'errors':errors,'complete_node_windows':len(right),'raw_samples':quality['raw_samples'],
             'source_kind':'cumulative_measured_counters','formal_acceptance':False}
    report['raw_counter_recomputation']=summary
    report['functional_pass']=report['functional_pass'] and summary['passed']
    save(BASE/'smoke_report.json',report);save(directory/'raw_recomputation.json',summary)
    return summary


if __name__=='__main__':print(json.dumps(run(),ensure_ascii=False,indent=2))
