import hashlib
import json
import platform
import sys
import zipfile
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'idfs_forecast_lab'))
from idfs_lab.target_baselines import paired_baselines, metrics, SECOND

import argparse
p=argparse.ArgumentParser()
p.add_argument('--archive', required=True)
p.add_argument('--output', required=True)
args=p.parse_args()
root=Path(__file__).resolve().parents[1]
archive=Path(args.archive)
assert hashlib.sha256(archive.read_bytes()).hexdigest()=='0e36d0a5a54e5a74f86a4327688ed09ad62c38bea48317a9da68672ac810ebd6'
z=zipfile.ZipFile(archive)
prefix='ai_training_fault_exploratory_v2/'
manifest_bytes=z.read(prefix+'handoff_manifest.json')
assert hashlib.sha256(manifest_bytes).hexdigest()=='4951b72711a72bd7cf66ecea9b9f700cea702345d2def0fa68475fc23e859d5d'
m=json.loads(manifest_bytes)
for f in m['files']:
    b=z.read(prefix+f['path'])
    assert len(b)==f['bytes'] and hashlib.sha256(b).hexdigest()==f['sha256']
rows=[json.loads(l) for l in z.read(prefix+'processed/windows.jsonl').splitlines()]
t0=json.loads(z.read(prefix+'recording_start.json'))['epoch_ns']
hosts=['idfs'+str(i) for i in (44,45,46,47,64,65,66,67)]
# Full history retained, but labels restricted to complete one-second windows
# inside the authorized 600-second recording interval.
result=paired_baselines(rows, hosts)
pairs=[r for r in result.pop('pairs') if t0<=r['window_start_ns'] and r['window_start_ns']+SECOND<=t0+600*SECOND]
result['paired_windows_entire_capture']=result.pop('paired_windows')
result.update(dataset_id=m['dataset_id'],run_id=m['source_run_id'],role='diagnostic',fault_affected=True,
              independent_validation=False,environment=platform.platform(),python=platform.python_version(),
              source_commit='1a112eeb922a10690d0c38425398c6c0f113d7d2',candidate_code_state='uncommitted candidate; hashes below',
              archive_sha256=hashlib.sha256(archive.read_bytes()).hexdigest(),
              manifest_sha256=hashlib.sha256(manifest_bytes).hexdigest(),source_windows_sha256=m['source_windows_sha256'],
              feature_schema=m['feature_schema'],protocol='development replay: five seconds ending target-1s, source availability <= target-30ms; not online arrival proof',
              workload_interval_paired_windows=len(pairs),metrics={})
for method in ('last_available','mean_five'):
    result['metrics'][method]={
        'nodes':{h:metrics((r['node_'+method][i],r['node_actual'][i]) for r in pairs) for i,h in enumerate(hosts)},
        'cluster':metrics((r['cluster_'+method],r['cluster_actual']) for r in pairs)}
result['code_sha256']={p:hashlib.sha256((root/p).read_bytes()).hexdigest() for p in ['idfs_forecast_lab/idfs_lab/target_baselines.py']}
out=Path(args.output)
out.write_text(json.dumps(result,indent=2)+'\n',encoding='utf-8')
print(json.dumps({'paired':len(pairs),'metrics':result['metrics']}))
