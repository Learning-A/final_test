"""Pinned first-round loader; original common sample policy, no fitting."""
import hashlib,json,zipfile
from pathlib import Path
from .multiscale_inputs import build_input,SECOND
from .joint import index_windows

def load_fixed_data(archive):
    sha=hashlib.sha256(Path(archive).read_bytes()).hexdigest()
    if sha!='f28fc45ba4e6d80c80da815b06d4e05573a7e9c788cbe199e0a1f01898b40b49':raise ValueError('ZIP checksum')
    hosts=['idfs'+str(i) for i in (44,45,46,47,64,65,66,67)]
    datasets={};required={'ai_training_ses','ai_inference_ses','bigdata_baleen','graph_zipf','hpc_wrf'}
    with zipfile.ZipFile(archive) as z:
     idx=json.loads(z.read('dataset_index.json'))
     if {d['workload'] for d in idx['datasets']}!=required or len(idx['datasets'])!=5:raise ValueError('five workloads required')
     for d in idx['datasets']:
      prefix=d['dataset_id']+'/'
      raw=z.read(prefix+'handoff_manifest.json')
      if hashlib.sha256(raw).hexdigest()!=d['handoff_manifest_sha256']:raise ValueError('manifest checksum')
      manifest=json.loads(raw)
      if manifest['role']!='train' or manifest['fault_affected']:raise ValueError('wrong source role/fault')
      for item in manifest['files']:
       b=z.read(prefix+item['path'])
       if len(b)!=item['bytes'] or hashlib.sha256(b).hexdigest()!=item['sha256']:raise ValueError('member checksum')
      rows=[json.loads(l) for l in z.read(prefix+'processed/windows.jsonl').splitlines()]
      t0=json.loads(z.read(prefix+'recording_start.json'))['epoch_ns'];index=index_windows(rows)
      capture=d['dataset_id']+':'+d['handoff_manifest_sha256']
      split={'train':[],'development':[]};counts={'candidate_labels':0,'rejected_history_or_label':0,'purged_or_outside':0}
      starts=sorted({r['window_start_ns'] for r in rows if r['scope']=='host_oss' and t0<=r['window_start_ns'] and r['window_end_ns']<=t0+600*SECOND})
      for start in starts:
       counts['candidate_labels']+=1
       part=next((name for name,lo,hi in [('train',0,360),('development',420,600)] if start-16*SECOND>=t0+lo*SECOND and start+SECOND<=t0+hi*SECOND),None)
       if part is None:counts['purged_or_outside']+=1;continue
       labels=[index.get(('host_oss',h,start)) for h in hosts]
       fs=[build_input(index,start,h,hosts,capture,scales=(5,15)) for h in hosts]
       if any(r is None or not r['complete'] for r in labels) or any(f is None for f in fs):
        counts['rejected_history_or_label']+=1;continue
       x=[f['features'] for f in fs];y=[r['total_iops'] for r in labels]
       # Direct cluster observes the same node contexts; no target values in X.
       cluster_x=sum([f[:50] for f in x],[])+x[0][50:]
       split[part].append(dict(start=start,x=x+[cluster_x],y=y+[sum(y)],
         last=[f['features'][2] for f in fs],mean=[f['features'][25] for f in fs],
         available=max(f['features_available_ns'] for f in fs)))
      if not all(split.values()):raise ValueError('empty partition: '+d['workload'])
      datasets[d['workload']]=dict(source=d,split=split,coverage=counts)
    return sha,datasets
