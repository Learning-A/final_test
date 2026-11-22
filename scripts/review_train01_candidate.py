"""Verify a new training capture and describe it; no fitting/certification."""
import argparse
import hashlib
import json
from pathlib import Path
import sys
import zipfile
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'idfs_forecast_lab'))
from idfs_lab.target_baselines import paired_baselines, metrics

p=argparse.ArgumentParser(description=__doc__)
p.add_argument('--archive',required=True)
p.add_argument('--output',required=True)
p.add_argument('--archive-sha',default='4505b0240625bfa16ae840ec66f7d82fc4cf67a8a821805e84e50db52cf59314')
p.add_argument('--manifest-sha',default='1b8196d7a0da8ed218936224ff5370af294a5dd04e5ad1fcedfe31e2004611a3')
args=p.parse_args()
sha=hashlib.sha256(Path(args.archive).read_bytes()).hexdigest()
if sha!=args.archive_sha:
    raise ValueError('candidate archive checksum mismatch')
with zipfile.ZipFile(args.archive) as z:
    names=[n for n in z.namelist() if n.endswith('/handoff_manifest.json')]
    if len(names)!=1: raise ValueError('ambiguous handoff manifest')
    prefix=names[0].rsplit('/',1)[0]+'/'
    raw=z.read(prefix+'handoff_manifest.json')
    manifest_sha=hashlib.sha256(raw).hexdigest()
    if manifest_sha!=args.manifest_sha:
        raise ValueError('candidate manifest checksum mismatch')
    m=json.loads(raw)
    for f in m['files']:
        b=z.read(prefix+f['path'])
        if len(b)!=f['bytes'] or hashlib.sha256(b).hexdigest()!=f['sha256']:
            raise ValueError('member checksum mismatch')
    rows=[json.loads(l) for l in z.read(prefix+'processed/windows.jsonl').splitlines()]
    t0=json.loads(z.read(prefix+'recording_start.json'))['epoch_ns']
    exit_event=json.loads(z.read(prefix+'workload_exit.json'))
    quality=json.loads(z.read(prefix+'processed/quality.json'))
hosts=['idfs'+str(i) for i in (44,45,46,47,64,65,66,67)]
def inside(row):
    return t0<=row['window_start_ns'] and row['window_end_ns']<=t0+600_000_000_000
out=dict(dataset_id=m['dataset_id'],source_run_id=m['source_run_id'],role=m['role'],
    source_batch=m['source_batch'],source_attempt=m['source_attempt'],
    fault_affected=m['fault_affected'],normal_training_certified=False,formal_acceptance=False,
    independent_validation=False,fitting_performed=False,review_status=m['review_status'],
    archive_sha256=sha,manifest_sha256=manifest_sha,members_verified=len(m['files']),
    source_windows_sha256=next(f['sha256'] for f in m['files'] if f['path']=='processed/windows.jsonl'),
    review_code_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    baseline_code_sha256=hashlib.sha256((Path(__file__).resolve().parents[1]/'idfs_forecast_lab/idfs_lab/target_baselines.py').read_bytes()).hexdigest(),
    interval='provisional recording_start+[0,600)s; actual workload boundary review pending',
    t0_epoch_ns=t0,workload_exit=exit_event,availability_evidence=quality.get('availability_evidence'),
    local_end_minus_source_record_start_seconds=(exit_event['end_local_ns']-t0)/1e9,
    cross_clock_duration_certified=False,nodes={})
for host in hosts:
    rs=[r for r in rows if r['scope']=='host_oss' and r['entity']==host and inside(r)]
    good=[r for r in rs if r['complete']]
    values=sorted(r['total_iops'] for r in good)
    out['nodes'][host]=dict(observed_seconds=len(rs),complete_seconds=len(good),incomplete_seconds=len(rs)-len(good),
        zero_seconds=sum(v==0 for v in values),min_positive=min(v for v in values if v>0),max_iops=max(values),
        mean_iops=sum(values)/len(values),median_iops=values[len(values)//2])
baseline=paired_baselines(rows,hosts)
paired=[r for r in baseline['pairs'] if t0<=r['window_start_ns'] and r['window_start_ns']+1_000_000_000<=t0+600_000_000_000]
out['common_causal_windows']=len(paired)
out['baseline_metrics']={method:{'cluster':metrics((r['cluster_'+method],r['cluster_actual']) for r in paired),
    'nodes':{h:metrics((r['node_'+method][i],r['node_actual'][i]) for r in paired) for i,h in enumerate(hosts)}}
    for method in ('last_available','mean_five')}
out['notes']=['Do not replace pinned fault diagnostic training data with this new candidate.',
              'Health check count is package metadata, not independent raw health-log verification.',
              'Absolute timestamps from different hosts do not certify duration or clock alignment.']
Path(args.output).write_text(json.dumps(out,indent=2)+'\n',encoding='utf-8')
print(json.dumps(out))
