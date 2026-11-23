"""Bounded six-arm follow-up; never run the first E2E test again."""
import argparse,hashlib,json,os,pickle,sys,time,math
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'idfs_forecast_lab'))
from idfs_lab.placement import require_training_host
from idfs_lab.input_loss_study import ARMS,select_features,relative_weights
from idfs_lab.target_baselines import metrics
from idfs_lab.first_round_data import load_fixed_data


def main():
    p=argparse.ArgumentParser();p.add_argument('--archive',required=True);p.add_argument('--reference',required=True);p.add_argument('--output',required=True)
    a=p.parse_args();root=require_training_host();dest=Path(a.output).resolve()
    if root not in dest.parents:raise ValueError('outside owned job')
    import fcntl
    lock=open(root.parent/'first-round-input-loss-v1.lock','a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    dest.mkdir(exist_ok=False)
    import numpy as np
    import sklearn
    from sklearn.ensemble import ExtraTreesRegressor
    sha,data=load_fixed_data(a.archive)
    reference=json.loads(Path(a.reference).read_text())
    if reference['archive_sha256']!=sha:raise ValueError('reference dataset mismatch')
    hosts=['idfs'+str(i) for i in (44,45,46,47,64,65,66,67)]+['cluster']
    train=[r for d in data.values() for r in d['split']['train']]
    dev=[dict(r,workload=w) for w,d in data.items() for r in d['split']['development']]
    ids={w:{part:[r['start'] for r in d['split'][part]] for part in ('train','development')} for w,d in data.items()}
    sample_hash=hashlib.sha256(json.dumps(ids,sort_keys=True).encode()).hexdigest()
    report=dict(experiment_id='first-round-input-loss-v1',state='running',code_commit=os.environ['E2E_CODE_SHA'],archive_sha256=sha,
        sample_ids_sha256=sample_hash,seed=2026,threads=4,workers=0,sklearn=sklearn.__version__,python=sys.version,
        formal_acceptance=False,independent_validation=False,online_30ms_verified=False,time_boundary_review='pending',
        train_seconds=len(train),development_seconds=len(dev),selection_count=6,calibration='none',
        baseline_reference_sha256=hashlib.sha256(Path(a.reference).read_bytes()).hexdigest(),arms={},models={})
    logs=[]
    for name,group,criterion,islog,weighted in ARMS:
        bytarget=[]
        for i,h in enumerate(hosts):
            x=np.asarray([select_features(r['x'][i],i,group) for r in train]);y=np.asarray([r['y'][i] for r in train])
            vx=np.asarray([select_features(r['x'][i],i,group) for r in dev])
            fit_y=np.log1p(y) if islog else y
            model=ExtraTreesRegressor(n_estimators=64,max_depth=8,min_samples_leaf=3,random_state=2026,n_jobs=4,criterion=criterion)
            t=time.monotonic();model.fit(x,fit_y,sample_weight=relative_weights(y.tolist()) if weighted else None);wall=time.monotonic()-t
            model.n_jobs=1
            path=dest/(name+'_'+h+'.pkl');path.write_bytes(pickle.dumps(model,protocol=4));loaded=pickle.loads(path.read_bytes())
            raw=model.predict(vx);out=loaded.predict(vx)
            parity=float(np.max(np.abs(raw-out)/np.maximum(1,np.abs(raw))))
            if parity>1e-5:raise ValueError('serialized parity')
            pred=np.maximum(np.expm1(out) if islog else out,0)
            truth=np.asarray([r['y'][i] for r in dev])
            trainpred=np.maximum(np.expm1(model.predict(x)) if islog else model.predict(x),0)
            trainloss=float(np.average(np.abs(trainpred-y),weights=relative_weights(y.tolist()))) if weighted else float(np.mean(np.abs(trainpred-y))) if criterion=='absolute_error' else float(np.mean((model.predict(x)-fit_y)**2))
            report['models'][name+'_'+h]=dict(sha256=hashlib.sha256(path.read_bytes()).hexdigest(),fit_seconds=wall,training_loss=trainloss,loss_scale='weighted raw absolute' if weighted else 'raw absolute' if criterion=='absolute_error' else 'log1p squared' if islog else 'raw squared',cpu_serialization_relative_error=parity,feature_count=x.shape[1])
            bytarget.append(pred.tolist())
        report['arms'][name]={'input':group,'criterion':criterion,'target_log1p':islog,'relative_training_weights':weighted,'metrics':{}}
        for w,d in data.items():
            ix=[k for k,r in enumerate(dev) if r['workload']==w]
            ms={h:metrics((bytarget[i][k],dev[k]['y'][i]) for k in ix) for i,h in enumerate(hosts)}
            ms['summed_node_cluster']=metrics((sum(bytarget[i][k] for i in range(8)),dev[k]['y'][8]) for k in ix)
            if name=='spatial_mse':
                for h in hosts:
                    for key in ('mape_percent','mae_iops','wape_percent','n','zero_actual'):
                        if not math.isclose(ms[h][key],reference['metrics'][w]['extra_trees'][h][key],rel_tol=1e-10,abs_tol=1e-8):raise ValueError('baseline/sample regression')
            report['arms'][name]['metrics'][w]=ms
        for k,r in enumerate(dev):
            if r['available']>r['start']-30000000:raise ValueError('late feature')
            logs.append(dict(arm=name,workload=r['workload'],window_start_ns=r['start'],features_available_ns=r['available'],raw_predictions=[bytarget[i][k] for i in range(9)],final_predictions=[bytarget[i][k] for i in range(9)],actual=r['y']))
        print(json.dumps({'arm_completed':name}),flush=True)
        (dest/'progress.json').write_text(json.dumps(report,indent=2))
    report['state']='completed';report['baseline_regression_passed']=True
    (dest/'predictions.json').write_text(json.dumps(logs))
    report['predictions_sha256']=hashlib.sha256((dest/'predictions.json').read_bytes()).hexdigest()
    (dest/'sample_ids.json').write_text(json.dumps(ids,sort_keys=True))
    (dest/'report.json').write_text(json.dumps(report,indent=2))
    print(json.dumps({'state':'completed','train':len(train),'development':len(dev),'sample_ids_sha256':sample_hash}),flush=True)


if __name__=='__main__':main()
