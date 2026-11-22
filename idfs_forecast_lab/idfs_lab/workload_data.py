"""Verified15-train capture loader reused by controlled follow-up jobs; no fit."""
import hashlib,json,math,zipfile
from pathlib import Path
from types import SimpleNamespace
from .masked_inputs import build_masked_input,SECOND
from .causal_io_context import build_io_context
from .joint import index_windows
HOSTS=['idfs'+str(i) for i in (44,45,46,47,64,65,66,67)]+['cluster']

def load_train_inputs(train,index,topology):
    a=SimpleNamespace(train=train,index=index,topology=topology)
    expected={'train':'889d303141802a9c826497eb3cf5f31ddac7c7a5d4db63c48698c5894d1d4425','index':'52e0bf86ff62079e6104a78b58126f59d0f63210ecb405d86b1614d98e92fd2d','topology':'81ddb8350790e2a1e165757e01c9c3d6cdcd328d956179778f1b7298230eb98b'}
    for key,sha in expected.items():
        if hashlib.sha256(Path(getattr(a,key)).read_bytes()).hexdigest()!=sha:raise ValueError(key+' checksum')
    ds=[d for d in json.loads(Path(a.index).read_text())['datasets'] if d['role']=='train']
    if len(ds)!=15 or any(d['index'] not in (1,2,3) for d in ds) or len({(d['batch'],d['run_id'],d['attempt']) for d in ds})!=15:raise ValueError('fifteen unique train captures required')
    topology={r['entity']:r['oss_ids'] for r in json.loads(Path(a.topology).read_text())['storage_nodes']}
    if set(topology)!=set(HOSTS[:8]) or sorted(sum(topology.values(),[]))!=list(range(96)):raise ValueError('invalid topology')
    fit=[];dev=[];summaries={}
    with zipfile.ZipFile(a.train) as z:
        for d in ds:
            prefix=d['dataset_id']+'/';raw=z.read(prefix+'handoff_manifest.json')
            if hashlib.sha256(raw).hexdigest()!=d['manifest_sha256']:raise ValueError('manifest checksum')
            m=json.loads(raw)
            if m['role']!='train' or m['fault_affected'] or m['source_run_id']!=d['run_id']:raise ValueError('role/fault mismatch')
            for f in m['files']:
                raw=z.read(prefix+f['path'])
                if len(raw)!=f['bytes'] or hashlib.sha256(raw).hexdigest()!=f['sha256']:raise ValueError('member checksum')
            rows=[json.loads(l) for l in z.read(prefix+'processed/windows.jsonl').splitlines()];ix=index_windows(rows)
            t0=json.loads(z.read(prefix+'recording_start.json'))['epoch_ns'];floor=((t0+SECOND-1)//SECOND)*SECOND
            part=dev if d['index']==3 else fit;count={'candidate':0,'warmup':0,'bad_label':0,'bad_history':0,'accepted':0}
            starts=sorted({r['window_start_ns'] for r in rows if r['scope']=='host_oss' and t0<=r['window_start_ns'] and r['window_end_ns']<=t0+600*SECOND})
            for start in starts:
                count['candidate']+=1
                if start-16*SECOND<floor:count['warmup']+=1;continue
                labels=[ix.get(('host_oss',h,start)) for h in HOSTS[:8]]
                if any(r is None or not r['complete'] for r in labels):count['bad_label']+=1;continue
                fs=[build_masked_input(ix,start,h,HOSTS[:8],d['dataset_id'],floor) for h in HOSTS[:8]]
                if any(f is None for f in fs):count['bad_history']+=1;continue
                extras=[build_io_context(ix,start,topology[h]) for h in HOSTS[:8]]
                y=[r['total_iops'] for r in labels]
                if any(not math.isfinite(v) or v<0 for v in y):raise ValueError('bad actual')
                available=max([f['features_available_ns'] for f in fs]+[f['features_available_ns'] or 0 for f in extras])
                if available>start-30000000:raise ValueError('late feature')
                part.append(dict(workload=d['workload'],run_id=d['run_id'],dataset_id=d['dataset_id'],source_role='train',development_partition='internal_run03' if d['index']==3 else 'fit_run01_02',start=start,x=[f['features'] for f in fs],extra=[f['features'] for f in extras],y=y+[sum(y)],available=available))
                count['accepted']+=1
            summaries[d['dataset_id']]={'source':d,'partition':'internal_run03' if d['index']==3 else 'fit_run01_02','coverage':count}
            print(json.dumps({'processed_capture':d['dataset_id'],'accepted':count['accepted']}),flush=True)
    if not fit or not dev:raise ValueError('empty partition')
    return fit,dev,summaries,expected
