"""Supervised b507-only I1: reuse C predictions, fit exactly two new input arms."""
import argparse
import gzip
import hashlib
import json
import os
from pathlib import Path
import pickle
import sys
import time

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'idfs_forecast_lab'))
from idfs_lab.placement import require_training_host
from idfs_lab.resource_slots import run_leased,COORD
from idfs_lab.r2_scaling import HOSTS

OWNER='01a203f5-3aa6-7b21-8a1f-8fece4787432'
JOB='b19_v2_i1_aux_inputs_v1_a01'
BUDGET=1709  # Original 1800-second I1 card less 91 charged input-preparation seconds.


def sha(path):
    with Path(path).open('rb') as f:
        return hashlib.file_digest(f,'sha256').hexdigest()


def checked(condition,message):
    if not condition:
        raise ValueError(message)


def save(path,obj):
    tmp=path.with_suffix(path.suffix+'.tmp')
    with tmp.open('w',encoding='utf-8') as f:
        json.dump(obj,f,indent=2,allow_nan=False)
        f.write('\n')
    tmp.replace(path)


def checked_registration(job):
    reg=json.loads((job/'registration.json').read_text())
    checked(job.name==JOB and reg['job_id']==JOB and reg['budget_seconds']==BUDGET,'Fixed I1 job and budget required')
    checked(reg['owner_thread']==OWNER and reg['code_sha']==os.environ['E2E_CODE_SHA'],'I1 owner or source differs')
    for name,meta in reg['inputs'].items():
        checked(Path(meta['file']).name==meta['file'],'Simple input filename required')
        checked(sha(job/'input'/meta['file'])==meta['sha256'],'Input checksum differs: '+name)
    checked(os.environ['CUDA_VISIBLE_DEVICES']=='','CPU-only tree fitting required')
    return reg


def summary(np,pred,actual,workloads):
    groups={}
    cluster={}
    for workload in sorted(set(workloads)):
        which=np.asarray([w==workload for w in workloads])
        ys,ps=actual[which],pred[which]
        groups[workload]={}
        for i,host in enumerate(HOSTS):
            y,p=ys[:,i],ps[:,i]
            pos=y>0
            groups[workload][host]=dict(count=len(y),positive_count=int(pos.sum()),
                zero_count=int((~pos).sum()),mape_percent=float(np.mean(100*np.abs(p[pos]-y[pos])/y[pos])) if pos.any() else None,
                mae=float(np.mean(np.abs(p-y))),bias=float(np.mean(p-y)),
                wape_percent=float(100*np.abs(p-y).sum()/y.sum()) if y.sum()>0 else None)
        cy,cp=ys.sum(axis=1),ps.sum(axis=1)
        positive=cy>0
        cluster[workload]=dict(count=len(cy),positive_count=int(positive.sum()),
            mape_percent=float(np.mean(100*np.abs(cp[positive]-cy[positive])/cy[positive])) if positive.any() else None,
            mae=float(np.mean(np.abs(cp-cy))),bias=float(np.mean(cp-cy)),
            wape_percent=float(100*np.abs(cp-cy).sum()/cy.sum()) if cy.sum()>0 else None)
    node_scores=[g['mape_percent'] for w in groups.values() for g in w.values()]
    checked(len(node_scores)==40 and all(x is not None for x in node_scores),'Forty positive-label groups required')
    return dict(groups=groups,summed_node_cluster=cluster,macro_mape_percent=float(np.mean(node_scores)),worst_group_mape_percent=max(node_scores),
        workload_mean_node_mae={w:float(np.mean([g['mae'] for g in gs.values()])) for w,gs in groups.items()},
        groups_mape_lt10=sum(x<10 for x in node_scores))


def screen(candidate,baseline):
    ratios={w:candidate['workload_mean_node_mae'][w]/baseline['workload_mean_node_mae'][w] for w in baseline['workload_mean_node_mae']}
    improvement=1-candidate['macro_mape_percent']/baseline['macro_mape_percent']
    worst=candidate['worst_group_mape_percent']/baseline['worst_group_mape_percent']
    return dict(relative_macro_improvement=improvement,worst_group_ratio=worst,workload_mae_ratios=ratios,
                passes=improvement>=.05 and worst<=1.05 and max(ratios.values())<=1.05)


def worker():
    import numpy as np
    import sklearn
    from sklearn.ensemble import ExtraTreesRegressor
    job=require_training_host()
    reg=checked_registration(job)
    lease=json.loads((COORD/(JOB+'.json')).read_text())
    checked(lease['state']=='running' and lease['pid']==os.getpid() and lease['owner_thread']==OWNER
            and lease['code_sha']==reg['code_sha'] and lease['budget_seconds']==BUDGET,'Live inherited I1 lease required')
    checked(sklearn.__version__==reg['sklearn_version'],'Match original tree library version')
    dest=job/'results'
    checked(not dest.exists(),'Fresh result directory required')
    dest.mkdir()
    t=time.monotonic()
    receipt=json.loads((job/'input'/reg['inputs']['receipt']['file']).read_text())
    checked(receipt['state']=='completed_no_fit_input_receipt' and not receipt['external_validation_read'],'No-fit training-role receipt required')
    checked(receipt['input_array_sha256']==reg['inputs']['arrays']['sha256'],'Array receipt mismatch')
    data=np.load(job/'input'/reg['inputs']['arrays']['file'],allow_pickle=False)
    ids=json.loads((job/'input'/reg['inputs']['samples']['file']).read_text())
    checked(len(ids['fit'])==5328 and len(ids['internal_dev'])==2661,'Fixed samples required')
    checked(not {tuple(k) for k in ids['fit']}.intersection(map(tuple,ids['internal_dev'])),'Training and internal split overlap')
    meta=json.loads((job/'input'/reg['inputs']['metadata']['file']).read_text())
    workloads=[meta[k[0]]['workload'] for k in ids['internal_dev']]
    for part,n in (('fit',5328),('internal_dev',2661)):
        for name,width in (('features',258),('B',180),('P',80)):
            checked(data[part+'_'+name].shape==(n,8,width) and np.isfinite(data[part+'_'+name]).all(),'Feature shape/value differs')
        checked(data[part+'_actual'].shape==(n,8),'Target shape differs')
    with gzip.open(job/'input'/reg['inputs']['baseline']['file'],'rt') as stream:
        frozen=[json.loads(line) for line in stream]
    baseline_map={}
    for row in frozen:
        key=row['dataset_id'],row['window_start_ns']
        checked(key not in baseline_map and row['arm']=='F0_F1' and row['margin_ms']==50,'Frozen C identity differs')
        baseline_map[key]=row
    checked(set(baseline_map)==set(map(tuple,ids['internal_dev'])),'Frozen C sample keys differ')
    baseline=np.asarray([baseline_map[tuple(k)]['final_predictions'][:8] for k in ids['internal_dev']])
    actual=np.asarray([baseline_map[tuple(k)]['actual'][:8] for k in ids['internal_dev']])
    checked(np.array_equal(actual,data['internal_dev_actual']),'Frozen original labels differ')
    report=dict(job_id=JOB,state='running',code_sha=reg['code_sha'],budget_seconds=BUDGET,
        card_total_budget_seconds=1800,input_preparation_charged_seconds=91,
        python=sys.version.split()[0],numpy=np.__version__,sklearn=sklearn.__version__,
        seed=2026,cpu_threads=4,workers=0,online_time_eligibility=False,external_validation_evaluations=0,
        inputs=reg['inputs'],feature_names=receipt['names'],host_order=list(HOSTS),
        coverage=receipt['coverage'],new_fit_count=0,baseline_reused_without_fit=True,
        arms={'C':summary(np,baseline,actual,workloads)},models={},comparisons={})
    all_predictions={'C':baseline}
    params=dict(n_estimators=64,max_depth=8,min_samples_leaf=3,random_state=2026,n_jobs=4,
        criterion='squared_error',max_features=1.0,min_samples_split=2,bootstrap=False,
        min_weight_fraction_leaf=0.0,max_leaf_nodes=None,min_impurity_decrease=0.0,
        oob_score=False,warm_start=False,ccp_alpha=0.0,max_samples=None,monotonic_cst=None)
    report['tree_parameters']=params
    save(dest/'progress.json',report)
    for arm,extras in (('CB',('B',)),('CBP',('B','P'))):
        pred=np.empty((2661,8),dtype=np.float64)
        for i,host in enumerate(HOSTS):
            x=np.concatenate([data['fit_features'][:,i]]+[data['fit_'+n][:,i] for n in extras],axis=1)
            vx=np.concatenate([data['internal_dev_features'][:,i]]+[data['internal_dev_'+n][:,i] for n in extras],axis=1)
            model=ExtraTreesRegressor(**params)
            fit_start=time.monotonic()
            model.fit(x,data['fit_actual'][:,i])
            fit_seconds=time.monotonic()-fit_start
            model.n_jobs=1
            path=dest/(arm+'_'+host+'.pkl')
            path.write_bytes(pickle.dumps(model,protocol=4))
            original=model.predict(vx)
            restored=pickle.loads(path.read_bytes()).predict(vx)
            checked(np.allclose(original,restored,rtol=1e-10,atol=0),'Frozen model serialization differs')
            pred[:,i]=restored
            report['new_fit_count']+=1
            report['models'][arm+'_'+host]=dict(sha256=sha(path),features=x.shape[1],fit_seconds=fit_seconds)
            save(dest/'progress.json',report)
            print(json.dumps(dict(arm=arm,node=host,fit_seconds=fit_seconds)),flush=True)
        report['arms'][arm]=summary(np,pred,actual,workloads)
        all_predictions[arm]=pred
        reference='C' if arm=='CB' else 'CB'
        report['comparisons'][arm+'_vs_'+reference]=screen(report['arms'][arm],report['arms'][reference])
        if arm=='CBP':
            report['comparisons']['CBP_vs_C']=screen(report['arms']['CBP'],report['arms']['C'])
        save(dest/'progress.json',report)
    predictions=dest/'predictions.jsonl.gz'
    with gzip.open(predictions,'wt',encoding='utf-8') as f:
        for arm,matrix in all_predictions.items():
            for i,key in enumerate(ids['internal_dev']):
                row=dict(dataset_id=key[0],window_start_ns=key[1],arm=arm,source_role='train',
                    workload=workloads[i],actual=actual[i].tolist(),raw_predictions=matrix[i].tolist(),
                    final_predictions=matrix[i].tolist(),online_time_eligibility=False,
                    development_partition='internal_run03',
                    features_available_upper_ns=int(data['internal_dev_available_ns'][i]),
                    feature_deadline_ns=key[1]-80_000_000,publication_deadline_ns=key[1]-30_000_000)
                f.write(json.dumps(row,allow_nan=False)+'\n')
    checked(report['new_fit_count']==16,'Exactly sixteen new node fits required')
    report.update(state='completed',elapsed_seconds=time.monotonic()-t,predictions_sha256=sha(predictions))
    save(dest/'report.json',report)
    print('COMPLETED',flush=True)


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--worker',action='store_true')
    args=parser.parse_args()
    if args.worker:
        worker()
    else:
        job=require_training_host()
        reg=checked_registration(job)
        checked(not (job/'results').exists(),'Do not repeat completed I1')
        run_leased([sys.executable,'-B',str(Path(__file__).resolve()),'--worker'],job_id=JOB,
            owner_thread=OWNER,code_sha=reg['code_sha'],data_hashes={k:v['sha256'] for k,v in reg['inputs'].items()},
            budget_seconds=BUDGET,use_gpu=False)


if __name__=='__main__':
    main()
