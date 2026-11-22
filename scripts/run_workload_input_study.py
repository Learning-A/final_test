"""Fixed causal input increments on10 fit/5 internal train captures, no val ZIP."""
import argparse,hashlib,json,math,os,pickle,sys,time,zipfile
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'idfs_forecast_lab'))
from idfs_lab.placement import require_training_host
from idfs_lab.masked_inputs import build_masked_input,SECOND
from idfs_lab.causal_io_context import build_io_context
from idfs_lab.joint import index_windows
from idfs_lab.target_baselines import metrics
HOSTS=['idfs'+str(i) for i in (44,45,46,47,64,65,66,67)]+['cluster']


def main():
    p=argparse.ArgumentParser();p.add_argument('--train',required=True);p.add_argument('--index',required=True);p.add_argument('--topology',required=True);p.add_argument('--output',required=True);a=p.parse_args()
    root=require_training_host();dest=Path(a.output).resolve()
    if root not in dest.parents or dest.exists():raise ValueError('fresh owned output')
    expected={'train':'889d303141802a9c826497eb3cf5f31ddac7c7a5d4db63c48698c5894d1d4425','index':'52e0bf86ff62079e6104a78b58126f59d0f63210ecb405d86b1614d98e92fd2d','topology':'81ddb8350790e2a1e165757e01c9c3d6cdcd328d956179778f1b7298230eb98b'}
    for key,sha in expected.items():
        if hashlib.sha256(Path(getattr(a,key)).read_bytes()).hexdigest()!=sha:raise ValueError(key+' checksum')
    ds=[d for d in json.loads(Path(a.index).read_text())['datasets'] if d['role']=='train']
    if len(ds)!=15 or any(d['index'] not in (1,2,3) for d in ds) or len({(d['batch'],d['run_id'],d['attempt']) for d in ds})!=15:raise ValueError('fifteen unique train captures required')
    topology={r['entity']:r['oss_ids'] for r in json.loads(Path(a.topology).read_text())['storage_nodes']}
    if set(topology)!=set(HOSTS[:8]) or sorted(sum(topology.values(),[]))!=list(range(96)):raise ValueError('invalid topology')
    import numpy as np
    import sklearn
    from sklearn.ensemble import ExtraTreesRegressor
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
    def vector(r,target,extra_count):
        if target<8:return r['x'][target]+r['extra'][target][:extra_count]
        base=sum([v[:112] for v in r['x']],[])+r['x'][0][112:]
        if extra_count>=20:base+=r['extra'][0][:20]
        if extra_count==38:base+=sum([v[20:] for v in r['extra']],[])
        return base
    dest.mkdir();report=dict(experiment_id='workload-causal-input-v1',state='running',code_commit=os.environ['E2E_CODE_SHA'],hashes=expected,
        source_role='train',fit_capture_count=10,internal_dev_capture_count=5,external_validation_capture_count_used=0,
        fit_seconds=len(fit),internal_dev_seconds=len(dev),datasets=summaries,formal_acceptance=False,time_alignment='pending',
        seed=2026,threads=4,workers=0,sklearn=sklearn.__version__,loss='raw squared error',calibration='none',arms={},models={})
    logs=[]
    for arm,extra_count in [('base144',0),('client164',20),('client_oss182',38)]:
        predictions=[]
        for ti,h in enumerate(HOSTS):
            x=np.asarray([vector(r,ti,extra_count) for r in fit]);y=np.asarray([r['y'][ti] for r in fit]);vx=np.asarray([vector(r,ti,extra_count) for r in dev])
            model=ExtraTreesRegressor(n_estimators=64,max_depth=8,min_samples_leaf=3,random_state=2026,n_jobs=4)
            t=time.monotonic();model.fit(x,y);wall=time.monotonic()-t;model.n_jobs=1
            path=dest/(arm+'_'+h+'.pkl');path.write_bytes(pickle.dumps(model,protocol=4));loaded=pickle.loads(path.read_bytes())
            raw=model.predict(vx);pred=loaded.predict(vx);parity=float(np.max(np.abs(raw-pred)/np.maximum(1,np.abs(raw))))
            if parity>1e-10:raise ValueError('CPU serialization parity')
            report['models'][arm+'_'+h]={'sha256':hashlib.sha256(path.read_bytes()).hexdigest(),'feature_count':x.shape[1],'fit_seconds':wall,'train_mse':float(np.mean((model.predict(x)-y)**2)),'cpu_serialization_relative_error':parity}
            predictions.append(pred.tolist())
        report['arms'][arm]={}
        for w in sorted({r['workload'] for r in dev}):
            ids=[k for k,r in enumerate(dev) if r['workload']==w]
            ms={h:metrics((predictions[i][k],dev[k]['y'][i]) for k in ids) for i,h in enumerate(HOSTS)}
            ms['summed_node_cluster']=metrics((sum(predictions[i][k] for i in range(8)),dev[k]['y'][8]) for k in ids)
            report['arms'][arm][w]=ms
        for k,r in enumerate(dev):logs.append({k2:r[k2] for k2 in ('workload','run_id','dataset_id','source_role','development_partition')}|dict(arm=arm,window_start_ns=r['start'],features_available_ns=r['available'],actual=r['y'],raw_predictions=[v[k] for v in predictions],final_predictions=[v[k] for v in predictions]))
        (dest/'progress.json').write_text(json.dumps(report,indent=2));print(json.dumps({'arm_completed':arm}),flush=True)
    sample_ids={p:[(r['dataset_id'],r['start']) for r in rs] for p,rs in [('fit',fit),('internal_dev',dev)]}
    (dest/'sample_ids.json').write_text(json.dumps(sample_ids,sort_keys=True));(dest/'predictions.json').write_text(json.dumps(logs));report.update(state='completed',predictions_sha256=hashlib.sha256((dest/'predictions.json').read_bytes()).hexdigest(),sample_ids_sha256=hashlib.sha256((dest/'sample_ids.json').read_bytes()).hexdigest())
    (dest/'report.json').write_text(json.dumps(report,indent=2));print('COMPLETED',flush=True)


if __name__=='__main__':main()
