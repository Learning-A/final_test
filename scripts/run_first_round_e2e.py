"""One bounded server-side CPU baseline E2E, never launch IDFS loads."""
import argparse,hashlib,json,os,pickle,sys,time,zipfile
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'idfs_forecast_lab'))
from idfs_lab.multiscale_inputs import build_input,SECOND
from idfs_lab.joint import index_windows
from idfs_lab.target_baselines import metrics
from idfs_lab.placement import require_training_host

def main():
 p=argparse.ArgumentParser();p.add_argument('--archive',required=True);p.add_argument('--output',required=True)
 args=p.parse_args();jobroot=require_training_host()
 dest=Path(args.output).resolve()
 if jobroot not in dest.parents:raise ValueError('outside owned job root')
 dest.mkdir(exist_ok=False)
 import fcntl
 lock=open(jobroot.parent/'all-workloads-first-round-e2e-v1.lock','a')
 fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
 import numpy as np
 from sklearn.ensemble import ExtraTreesRegressor
 from sklearn.linear_model import Ridge
 from sklearn.pipeline import make_pipeline
 from sklearn.preprocessing import StandardScaler
 import sklearn
 sha=hashlib.sha256(Path(args.archive).read_bytes()).hexdigest()
 if sha!='f28fc45ba4e6d80c80da815b06d4e05573a7e9c788cbe199e0a1f01898b40b49':raise ValueError('ZIP checksum')
 hosts=['idfs'+str(i) for i in (44,45,46,47,64,65,66,67)]
 datasets={};required={'ai_training_ses','ai_inference_ses','bigdata_baleen','graph_zipf','hpc_wrf'}
 with zipfile.ZipFile(args.archive) as z:
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
 report=dict(test_id='all-workloads-first-round-e2e-v1',state='completed',scope='preliminary recording-interval development replay',
  archive_sha256=sha,normal_training_certified=False,formal_acceptance=False,independent_validation=False,
  time_boundary_review='pending workload clock evidence; recording_start interval only',
  online_30ms_verified=False,feature_schema='first_e2e_node82_cluster432_v1',seed=2026,threads=4,
  sklearn=sklearn.__version__,python=sys.version,code_commit=os.environ.get('E2E_CODE_SHA'),datasets={},models={})
 for name,data in datasets.items():
  report['datasets'][name]=dict(source=data['source'],coverage=data['coverage'],train=len(data['split']['train']),development=len(data['split']['development']))
 train=[r for d in datasets.values() for r in d['split']['train']]
 predictions={}
 for family in ('ridge','extra_trees'):
  predictions[family]={name:[] for name in datasets}
  for target,h in enumerate(hosts+['cluster']):
   model=make_pipeline(StandardScaler(),Ridge(alpha=1.0)) if family=='ridge' else ExtraTreesRegressor(n_estimators=64,max_depth=8,min_samples_leaf=3,random_state=2026,n_jobs=4)
   x=np.asarray([r['x'][target] for r in train]);y=np.asarray([r['y'][target] for r in train]);model.fit(x,y)
   path=dest/(family+'_'+h+'.pkl');path.write_bytes(pickle.dumps(model,protocol=4));loaded=pickle.loads(path.read_bytes())
   allrows=[r for d in datasets.values() for r in d['split']['development']];vx=np.asarray([r['x'][target] for r in allrows])
   a=np.maximum(model.predict(vx),0);b=np.maximum(loaded.predict(vx),0)
   parity=float(np.max(np.abs(a-b)/np.maximum(1,np.abs(a))))
   if parity>1e-5:raise ValueError('CPU serialized parity')
   loaded.predict(vx[:1]);lat=[]
   for _ in range(10):
    t=time.perf_counter();loaded.predict(vx[:1]);lat.append((time.perf_counter()-t)*1000)
   report['models'][family+'_'+h]=dict(sha256=hashlib.sha256(path.read_bytes()).hexdigest(),cpu_serialization_max_relative_error=parity,batch1_latency_ms_median=float(np.median(lat)))
   offset=0
   for name,d in datasets.items():
    rs=d['split']['development'];vals=b[offset:offset+len(rs)].tolist();offset+=len(rs)
    predictions[family][name].append(vals)
 report['metrics']={};logs=[]
 for name,d in datasets.items():
  rs=d['split']['development'];report['metrics'][name]={}
  for family in ('last_available','mean_five','ridge','extra_trees'):
   if family in ('last_available','mean_five'):
    key='last' if family=='last_available' else 'mean';pred=[[r[key][i] for r in rs] for i in range(8)];pred.append([sum(r[key]) for r in rs])
   else:pred=predictions[family][name]
   group={h:metrics(zip(pred[i],[r['y'][i] for r in rs])) for i,h in enumerate(hosts+['cluster'])}
   group['summed_node_cluster']=metrics((sum(pred[i][k] for i in range(8)),r['y'][8]) for k,r in enumerate(rs))
   report['metrics'][name][family]=group
   for k,r in enumerate(rs):
    if r['available']>r['start']-30_000_000:raise ValueError('noncausal replay')
    logs.append(dict(workload=name,family=family,window_start_ns=r['start'],features_available_ns=r['available'],predictions=[pred[i][k] for i in range(9)],actual=r['y']))
 (dest/'predictions.json').write_text(json.dumps(logs),encoding='utf-8')
 report['predictions_sha256']=hashlib.sha256((dest/'predictions.json').read_bytes()).hexdigest()
 (dest/'report.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
 print(json.dumps({'state':'completed','output':str(dest),'counts':{k:(v['train'],v['development']) for k,v in report['datasets'].items()}}))
if __name__=='__main__':main()
