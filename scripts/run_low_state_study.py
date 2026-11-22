"""Predict next-second low state from historical inputs, correct frozen regression."""
import argparse,hashlib,json,os,pickle,sys,math
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'idfs_forecast_lab'))
from idfs_lab.placement import require_training_host
from idfs_lab.workload_data import load_train_inputs,HOSTS
from idfs_lab.low_state import quiet_state,cluster_input
from idfs_lab.target_baselines import metrics


def main():
    p=argparse.ArgumentParser();p.add_argument('--train',required=True);p.add_argument('--index',required=True);p.add_argument('--topology',required=True);p.add_argument('--regressors',required=True);p.add_argument('--output',required=True);a=p.parse_args()
    root=require_training_host();dest=Path(a.output).resolve()
    if root not in dest.parents or dest.exists():raise ValueError('fresh owned output required')
    import numpy as np
    import sklearn
    from sklearn.ensemble import ExtraTreesClassifier
    from sklearn.metrics import roc_auc_score,average_precision_score,brier_score_loss,precision_score,recall_score
    fit,dev,datasets,hashes=load_train_inputs(a.train,a.index,a.topology)
    if (len(fit),len(dev))!=(5328,2661):raise ValueError('sample regression')
    x=np.asarray([cluster_input(r) for r in fit]);vx=np.asarray([cluster_input(r) for r in dev])
    y=np.asarray([quiet_state(r['y'][8],sum(v[2] for v in r['x'])) for r in fit]);vy=np.asarray([quiet_state(r['y'][8],sum(v[2] for v in r['x'])) for r in dev])
    if len(set(y))!=2:raise ValueError('two train classes required')
    model=ExtraTreesClassifier(n_estimators=64,max_depth=8,min_samples_leaf=3,class_weight='balanced',random_state=2026,n_jobs=4)
    model.fit(x,y);model.n_jobs=1;dest.mkdir();path=dest/'state_classifier.pkl';path.write_bytes(pickle.dumps(model,protocol=4));loaded=pickle.loads(path.read_bytes())
    probability=loaded.predict_proba(vx)[:,list(loaded.classes_).index(1)];reference=model.predict_proba(vx)[:,list(model.classes_).index(1)]
    if not np.allclose(probability,reference,rtol=1e-12,atol=1e-12):raise ValueError('classifier serialization')
    quiet=np.median(np.asarray([r['y'] for r,label in zip(fit,y) if label==1]),axis=0)
    source=Path(a.regressors);prior=json.loads((source/'report.json').read_text());baseline=[];source_hashes={}
    for i,h in enumerate(HOSTS):
        p=source/('client_oss182_'+h+'.pkl');sha=hashlib.sha256(p.read_bytes()).hexdigest()
        if sha!=prior['models']['client_oss182_'+h]['sha256']:raise ValueError('regressor checksum')
        source_hashes[h]=sha;m=pickle.loads(p.read_bytes())
        xx=np.asarray([r['x'][i]+r['extra'][i] for r in dev]) if i<8 else vx
        baseline.append(m.predict(xx))
    baseline=np.asarray(baseline).T
    hard=np.where((probability>=.5)[:,None],quiet[None,:],baseline)
    soft=(1-probability[:,None])*baseline+probability[:,None]*quiet[None,:]
    report=dict(experiment_id='causal-low-state-v1',state='completed',code_commit=os.environ['E2E_CODE_SHA'],hashes=hashes,source_regressor_hashes=source_hashes,
        auxiliary_state='actual_cluster <= max(10,0.1*last_available_cluster)',threshold=.5,threshold_selected=False,
        quiet_target_training_medians=quiet.tolist(),fit_seconds=len(fit),internal_dev_seconds=len(dev),
        external_validation_used=0,source_role='train',formal_acceptance=False,time_boundary_review='pending',sklearn=sklearn.__version__,
        training_quiet_count=int(y.sum()),classifier_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),classification={},metrics={})
    logs=[]
    for w in sorted({r['workload'] for r in dev}):
        ids=np.asarray([i for i,r in enumerate(dev) if r['workload']==w]);truth=vy[ids];pr=probability[ids]
        report['classification'][w]={'n':len(ids),'quiet_count':int(truth.sum()),'prevalence':float(truth.mean()),'average_precision':float(average_precision_score(truth,pr)),
            'roc_auc':float(roc_auc_score(truth,pr)) if len(set(truth))==2 else None,'brier':float(brier_score_loss(truth,pr)),
            'precision_at_fixed_half':float(precision_score(truth,pr>=.5,zero_division=0)),'recall_at_fixed_half':float(recall_score(truth,pr>=.5,zero_division=0)),
            'predicted_quiet_count':int(sum(pr>=.5))}
        report['metrics'][w]={}
        for variant,values in [('frozen_regression',baseline),('hard_state',hard),('soft_state',soft)]:
            ms={h:metrics((float(values[k,i]),dev[k]['y'][i]) for k in ids) for i,h in enumerate(HOSTS)}
            ms['summed_node_cluster']=metrics((float(values[k,:8].sum()),dev[k]['y'][8]) for k in ids)
            if variant=='frozen_regression':
                for h in HOSTS:
                    for key in ('n','mape_percent','mae_iops','wape_percent'):
                        if not math.isclose(ms[h][key],prior['arms']['client_oss182'][w][h][key],rel_tol=1e-10,abs_tol=1e-7):raise ValueError('frozen regressor/sample regression')
            report['metrics'][w][variant]=ms
            for k in ids:
                r=dev[k];logs.append(dict(workload=w,run_id=r['run_id'],source_role='train',partition='internal_run03',variant=variant,window_start_ns=r['start'],features_available_ns=r['available'],actual=r['y'],raw_predictions=baseline[k].tolist(),final_predictions=values[k].tolist(),quiet_probability=float(probability[k]),quiet_actual=int(vy[k])))
    (dest/'predictions.json').write_text(json.dumps(logs));report['predictions_sha256']=hashlib.sha256((dest/'predictions.json').read_bytes()).hexdigest();(dest/'report.json').write_text(json.dumps(report,indent=2));print(json.dumps({'state':'completed','train_quiet':int(y.sum()),'internal_quiet':int(vy.sum())}))


if __name__=='__main__':main()
