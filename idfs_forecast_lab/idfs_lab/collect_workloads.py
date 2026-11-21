"""Sequential authorized development captures with pinned existing workloads."""
import concurrent.futures
import hashlib
import json
from pathlib import Path
import shutil
import shlex
import threading
import time

from .io import load, save
from .probe import collect
from .transport import SiteConnection
from .dataset import process


def development_rounds():
    aliases = [('ai_training', 'ai_training_ses'), ('ai_inference', 'ai_inference_ses'),
               ('baleen', 'bigdata_baleen'), ('graph', 'graph_zipf'), ('wrf', 'hpc_wrf')]
    return [dict(alias=alias, workload=workload, role=role, index=index,
                 run_id=workload+'_'+role+'_%02d' % index)
            for role, index in [('train', 1), ('train', 2), ('train', 3), ('validation', 1)]
            for alias, workload in aliases]


def run(root, site_path='config/site_smoke.json', protocol_path='config/protocol.json',
        workloads_path='config/workloads.json', rounds=None):
    root = Path(root)
    root.mkdir(parents=True, exist_ok=False)
    site = load(site_path)
    protocol = load(protocol_path)
    release = load(workloads_path)
    rounds = development_rounds() if rounds is None else rounds
    if not rounds:
        raise ValueError('at least one round required')
    save(root/'plan.json', dict(rounds=rounds, seconds=660, workload_seconds=600,
         development_only=True, formal=False, workload_commit=release['commit'],
         order='interleave workloads across repetitions; independent validation last'))
    outcomes = []
    try:
        for spec in rounds:
            if shutil.disk_usage(root).free < 25*2**30:
                raise RuntimeError('less than 25 GiB local free space')
            dest = root/spec['run_id']
            dest.mkdir(exist_ok=False)
            remote = '/home/hust/idfs_forecast_inference/development/'+root.name+'/'+spec['run_id']
            manifest = next(c for c in release['cases'] if c['id'] == spec['workload'])
            with SiteConnection(site) as connection:
                target = dict(kind='client_nfs_rpc', host=site['control_host'])
                health_target = dict(kind='oss', host=site['storage_nodes'][0]['management_ip'])
                health = connection.run(health_target, 'timeout 12 idfs -s', timeout=15)
                save(dest/'cluster_before.json', health)
                if health['exit'] or 'HEALTH_OK' not in health['stdout'] or '96 up' not in health['stdout']:
                    raise RuntimeError('cluster not healthy with 96 OSS up; no workload launched')
                # Query process names only; never signal or stop a pre-existing task.
                active = connection.run(target,
                    "ps -u hust -o comm=,args= | awk '$1 == \"java\" || $1 == \"vdbench\" || "
                    "($1 ~ /^python/ && ($0 ~ /idfs_forecast_inference|idfs[-_]workloads?|python3 -B -/)) {print}'")
                if active['exit'] or active['stdout'].strip():
                    raise RuntimeError('existing workload/preparation/copy/collector task: refuse overlapping load')
                ready = connection.run(target, shlex.join([release['release']+'/workloads.sh',
                                              'verify', '--case', spec['alias']]), timeout=60)
                save(dest/'data_readiness.json', ready)
                if ready['exit']:
                    raise RuntimeError('existing dataset not ready; no automatic preparation')
                check = connection.run(target, 'cat '+shlex.quote(release['release']+'/COMMIT')+
                                       '; sha256sum '+shlex.quote(manifest['path']))
                if check['exit'] or check['stdout'].splitlines()[0].strip()!=release['commit'] or \
                        check['stdout'].splitlines()[1].split()[0]!=manifest['sha256']:
                    raise RuntimeError('release or workload manifest changed')
                client = connection.client(site['control_host'])
                try:
                    sftp=client.open_sftp()
                    try:
                        with sftp.open(manifest['path'],'rb') as src:
                            data=src.read()
                        (dest/'workload_manifest.json').write_bytes(data)
                    finally:sftp.close()
                    seen=set();lock=threading.Lock();stable=threading.Event()
                    def sample(row):
                        if row.get('sample_ok'):
                            with lock:
                                seen.add((row['scope'],str(row['entity'])))
                                if len(seen)==112:stable.set()
                    save(root/'status.json', dict(state='collecting', current=spec, completed=outcomes))
                    print('CAPTURE '+spec['run_id'],flush=True)
                    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
                        capture=pool.submit(collect,site,protocol,dest/'raw',660,True,sample)
                        if not stable.wait(45):
                            raise RuntimeError('all 112 counters did not become ready; load not launched')
                        # t0 is the cluster controller recording time, independent of load first IO.
                        anchor=connection.run(target, "python3 -c 'import time; print(time.time_ns())'")
                        if anchor['exit']:raise RuntimeError('recording timestamp unavailable')
                        save(dest/'recording_start.json',dict(event='recording_start',
                             epoch_ns=int(anchor['stdout'].strip()),clock_host=site['control_host'],
                             first_io_claimed=False))
                        command=shlex.join([release['release']+'/workloads.sh','run','--case',
                                           spec['alias'],'--results',remote,'--execute'])
                        save(dest/'execution.json',dict(**spec,command=command,remote_results=remote,
                             manifest_sha256=hashlib.sha256(data).hexdigest(),development_only=True))
                        print('LOAD '+spec['run_id'],flush=True)
                        channel=client.get_transport().open_session(timeout=15)
                        channel.set_combine_stderr(True)
                        channel.exec_command(command)
                        health_events=[];next_health=time.monotonic()+30
                        with (dest/'workload.log').open('wb') as log:
                            while True:
                                if time.monotonic()>=next_health:
                                    try:
                                        health=connection.run(health_target,'timeout 12 idfs -s',timeout=15)
                                    except Exception as exc:
                                        # A transient health query failure must not abandon
                                        # the channel or lose the load's natural exit evidence.
                                        health={'exit':-1,'stdout':'','stderr':type(exc).__name__}
                                    health_events.append(dict(observed_local_ns=time.time_ns(),**health))
                                    save(dest/'health_during_load.json',health_events)
                                    next_health=time.monotonic()+30
                                if channel.recv_ready():
                                    chunk=channel.recv(65536);log.write(chunk);log.flush()
                                elif channel.exit_status_ready():break
                                else:time.sleep(.1)
                            while channel.recv_ready():log.write(channel.recv(65536))
                        code=channel.recv_exit_status();channel.close()
                        save(dest/'workload_exit.json',dict(exit=code,end_local_ns=time.time_ns()))
                        collected=capture.result()
                    try:
                        final_health=connection.run(health_target,'timeout 12 idfs -s',timeout=15)
                    except Exception as exc:
                        final_health={'exit':-1,'stdout':'','stderr':type(exc).__name__}
                    health_events.append(dict(observed_local_ns=time.time_ns(),**final_health))
                    save(dest/'health_during_load.json',health_events)
                    collectors_ok = len(collected['outcomes']) == 24 and all(
                        'error' not in item and item['result']['exit'] == 0
                        for item in collected['outcomes'])
                    if code or collected['state']!='complete' or not collectors_ok:
                        raise RuntimeError('workload/capture failure; do not continue to next load')
                    if any(h['exit'] or 'HEALTH_OK' not in h['stdout'] or '96 up' not in h['stdout']
                           for h in health_events):
                        save(dest/'excluded_from_normal_training.json',dict(reason='cluster health changed during load'))
                        raise RuntimeError('cluster health changed during load; evidence retained, next round blocked')
                finally:client.close()
            quality=process(dest/'raw',dest/'processed',protocol)
            summary=dict(**spec,state='captured_processed',raw_samples=quality['raw_samples'],
                         complete_windows=quality['complete_windows'],remote_results=remote)
            outcomes.append(summary)
            save(root/'status.json',dict(state='between_rounds',completed=outcomes))
            print('DONE '+json.dumps(summary),flush=True)
        save(root/'status.json',dict(state='complete',completed=outcomes,formal=False))
        return outcomes
    except Exception as exc:
        save(root/'status.json',dict(state='failed',current=spec,completed=outcomes,error=str(exc)))
        raise


if __name__=='__main__':
    import argparse
    parser=argparse.ArgumentParser();parser.add_argument('--out',required=True)
    run(parser.parse_args().out)
