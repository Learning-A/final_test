"""One-second frozen-output metric audit, no fitting or neural inference."""
from collections import defaultdict
import gzip
import hashlib
import json
from pathlib import Path

import numpy as np

ROOT=Path(__file__).resolve().parents[1]
WORK=ROOT/'.local/metric_feasibility'
HOSTS=['idfs'+str(i) for i in (44,45,46,47,64,65,66,67)]


def digest(path):
    with path.open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()


def read(path):
    with gzip.open(path,'rt',encoding='utf-8') as f:return list(map(json.loads,f))


def metrics(y,p):
    y=np.asarray(y,dtype=float);p=np.asarray(p,dtype=float)
    assert y.shape==p.shape and np.isfinite(y).all() and np.isfinite(p).all() and (y>=0).all() and (p>=0).all()
    absolute=np.abs(p-y);positive=y>0
    ape=absolute[positive]/y[positive]
    smape=np.divide(absolute,p+y,out=np.zeros_like(y),where=p+y>0)
    ratio=np.divide(absolute,y,out=np.full_like(y,np.inf),where=positive)
    ratio[(y==0)&(p==0)]=0
    variance=float(np.sum((y-y.mean())**2));squared=float(np.sum((p-y)**2))
    le=np.log1p(p)-np.log1p(y)
    return dict(n=len(y),zero_actual=int((~positive).sum()),
        mape_percent=float(100*ape.mean()),wape_percent=float(100*absolute.sum()/y.sum()),
        mae_iops=float(absolute.mean()),rmse_iops=float(np.sqrt(np.mean((p-y)**2))),
        r2=float(1-squared/variance) if variance>0 else None,
        smape200_percent=float(200*smape.mean()),maape_radians=float(np.arctan(ratio).mean()),
        mdape_percent=float(100*np.median(ape)),q90_ape_positive_percent=float(100*np.quantile(ape,.9)),
        log_mae=float(np.abs(le).mean()),rmsle=float(np.sqrt(np.mean(le**2))),
        tolerance_hits_percent={str(int(t*100)):float(100*np.mean(absolute<=t*y)) for t in (.1,.2,.3)},
        mixed_20pct_absolute_hits_percent={str(a):float(100*np.mean(absolute<=np.maximum(a,.2*y))) for a in (10,50,100)},
        capacity_required_for_90_nmae_iops=float(10*absolute.mean()))


def aggregate(groups,pooled):
    keys=('mape_percent','wape_percent','mae_iops','rmse_iops','r2','smape200_percent','maape_radians','mdape_percent','log_mae','rmsle')
    mean={k:float(np.mean([g[k] for g in groups.values()])) for k in keys}
    mean['wape_score_percent']=100-mean['wape_percent']
    mean['pred20_percent']=float(np.mean([g['tolerance_hits_percent']['20'] for g in groups.values()]))
    return dict(groups=groups,macro=mean,
        worst_wape_percent=max(g['wape_percent'] for g in groups.values()),
        lowest_pred20_percent=min(g['tolerance_hits_percent']['20'] for g in groups.values()),
        groups_wape_below10=sum(g['wape_percent']<10 for g in groups.values()),
        groups_pred20_over90=sum(g['tolerance_hits_percent']['20']>90 for g in groups.values()),
        pooled=metrics(*pooled),pooling_note='pooled R2 can be inflated by between-workload levels; use each workload R2')


def main():
    WORK.mkdir(exist_ok=True)
    pinpaths=[ROOT/'.local/r1-frozen/results/predictions.jsonl.gz']
    r1=read(pinpaths[0]);original={};workloads={};models={}
    for margin,arm,name in [(50,'F0','R1_F0'),(50,'F0_F1','R1_F1_50'),(100,'F0_F1','R1_F1_100'),(200,'F0_F1','R1_F1_200')]:
        rows=[r for r in r1 if r['margin_ms']==margin and r['arm']==arm]
        assert len(rows)==2661
        values={}
        for r in rows:
            key=(r['dataset_id'],r['window_start_ns']);y=r['actual'];p=r['final_predictions']
            assert len(y)==len(p)==9 and sum(y[:8])==y[8] and r['raw_predictions']==p
            assert r['features_available_upper_ns']<=r['feature_deadline_ns']
            assert r['source_role']=='train' and not r['online_time_eligibility']
            assert key not in values
            if key in original:assert original[key]==y[:8]
            original[key]=y[:8];workloads[key]=r['workload'];values[key]=p
        models[name]=dict(values=values,direct_cluster=True)
    keys=sorted(original);assert len(keys)==2661
    def add(name,path,rows=None):
        if rows is None:rows=read(path)
        pinpaths.append(path);values={}
        for r in rows:
            key=(r['dataset_id'],r['window_start_ns']);y=r['actual']
            p=r.get('final_prediction',r.get('final_predictions'))
            assert y==original[key] and len(p)==8 and key not in values
            values[key]=p
        assert set(values)==set(keys)
        models[name]=dict(values=values,direct_cluster=False)
    for arm in ('L0','L1','L2','L3'):
        add('R2_'+arm,ROOT/f'.local/r2-results/results/{arm}/predictions.jsonl.gz')
    ipath=ROOT/'.local/v2_i1/fitted/predictions.jsonl.gz';irows=read(ipath)
    for arm in ('CB','CBP'):add('I1_'+arm,ipath,[r for r in irows if r['arm']==arm])
    for i in (1,2,3):add(f'V2_M{i}',ROOT/f'.local/v2_m{i}/fitted/predictions.jsonl.gz')
    for arm in ('T1','T2'):add(arm,ROOT/f'.local/data_transform/{arm}/{arm}/predictions.jsonl.gz')
    y=np.asarray([original[k] for k in keys]);labels=np.asarray([workloads[k] for k in keys])
    result={}
    for name,model in models.items():
        p=np.asarray([model['values'][k] for k in keys]);nodes={};sums={};direct={}
        for workload in sorted(set(labels)):
            rows=labels==workload;ys,ps=y[rows],p[rows]
            for i,h in enumerate(HOSTS):nodes[workload+'/'+h]=metrics(ys[:,i],ps[:,i])
            sums[workload]=metrics(ys.sum(axis=1),ps[:,:8].sum(axis=1))
            if model['direct_cluster']:direct[workload]=metrics(ys.sum(axis=1),ps[:,8])
        record=dict(nodes=aggregate(nodes,(y.flatten(),p[:,:8].flatten())),
                    summed_node_cluster=aggregate(sums,(y.sum(axis=1),p[:,:8].sum(axis=1))))
        if direct:record['direct_cluster']=aggregate(direct,(y.sum(axis=1),p[:,8]))
        record['fit_performed']=False
        result[name]=record
    expected={pinpaths[0]:json.loads((ROOT/'reports/r1_input_study_v1.json').read_text())['predictions_sha256']}
    old=json.loads((ROOT/'.local/r2-results/results/report.json').read_text())
    for arm in ('L0','L1','L2','L3'):
        expected[ROOT/f'.local/r2-results/results/{arm}/predictions.jsonl.gz']=old['arms'][arm]['predictions_sha256']
    for i,report in [(1,'v2_m1_local_residual.json'),(2,'v2_m2_workload_adapter.json'),(3,'v2_m3_short_causal_tcn.json')]:
        frozen=json.loads((ROOT/'reports'/report).read_text())
        expected[ROOT/f'.local/v2_m{i}/fitted/predictions.jsonl.gz']=(frozen if i==1 else frozen['result'])['predictions_sha256']
    expected[ipath]=json.loads((ROOT/'reports/v2_i1_input_study.json').read_text())['predictions_sha256']
    for arm in ('T1','T2'):
        expected[ROOT/f'.local/data_transform/{arm}/{arm}/predictions.jsonl.gz']=json.loads((ROOT/f'reports/data_transform_{arm}.json').read_text())['training']['predictions_sha256']
    assert set(expected)==set(pinpaths)
    manifest={}
    for path in sorted(set(pinpaths)):
        actual_sha=digest(path)
        assert actual_sha==expected[path], 'frozen input pin differs: '+str(path)
        manifest[str(path.relative_to(ROOT))]=dict(sha256=actual_sha,bytes=path.stat().st_size,prior_frozen_report_pin_matched=True)
    report=dict(scope='Existing 2661 train03 seconds, 40 node/workload and five cluster/workload groups. User prioritizes next complete 1-second cluster total. No fitting, new model inference, target smoothing or external validation evaluations.',
                formal_metric_replaced=False,checkpoint_reselected=False,new_external_validation_model_evaluations=0,
                sample_keys_sha256=hashlib.sha256(json.dumps(keys,separators=(',',':')).encode()).hexdigest(),
                limitations=['All diagnostics are development data; metric selection after seeing this audit requires prospective confirmation.','No information-theoretic irreducible-error bound estimated.','Source-clock offline eligibility only; service arrival and cross-host clock bounds unverified.'],
                input_manifest=manifest,models=result)
    out=ROOT/'reports/cluster_metric_feasibility_20261123.json'
    out.write_text(json.dumps(report,indent=2,allow_nan=False)+'\n',encoding='utf-8',newline='\n')
    for name,r in result.items():
        for target in ('summed_node_cluster','direct_cluster'):
            if target in r:
                s=r[target]
                print(name,target,json.dumps(dict(mape=s['macro']['mape_percent'],wape=s['macro']['wape_percent'],
                    score=s['macro']['wape_score_percent'],pred20=s['macro']['pred20_percent'],
                    smape200=s['macro']['smape200_percent'],r2=s['macro']['r2'],
                    all_workload_wape_pass=s['groups_wape_below10'],all_workload_pred20_pass=s['groups_pred20_over90'])))


if __name__=='__main__':main()
