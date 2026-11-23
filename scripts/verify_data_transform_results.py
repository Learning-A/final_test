"""Separate frozen-output arithmetic only; no model load, fit or new inference."""
import argparse
import gzip
import hashlib
import json
from pathlib import Path

import numpy as np


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_rows(path):
    with gzip.open(path,'rt',encoding='utf-8') as f:
        return list(map(json.loads,f))


def verify(root,baseline_path):
    original=read_rows(baseline_path)
    by_key={(r['dataset_id'],r['window_start_ns']):r for r in original}
    assert len(by_key)==len(original)==2661
    verified={}
    for arm in ('T1','T2'):
        directory=root/arm
        report=json.loads((directory/'report.json').read_text())
        result=report['training'];lease=json.loads((directory/'lease.json').read_text())
        assert lease['state']=='completed' and lease['exit_code']==0
        assert lease['ended_epoch']-lease['started_epoch']<=1800
        for file,key in [('checkpoint.pt','checkpoint_sha256'),('predictions.jsonl.gz','predictions_sha256'),('epochs.json','epochs_sha256')]:
            assert sha(directory/arm/file)==result[key]
        assert sha(directory/'transform.json')==report['transform_sha256']
        assert sha(directory/arm/'log_predictions.jsonl.gz')==report['log_predictions_sha256']
        assert sha(directory/'input_context.npz')==report['input_context_sha256']
        rows=read_rows(directory/arm/'log_predictions.jsonl.gz')
        keys=[(r['dataset_id'],r['window_start_ns']) for r in rows]
        assert len(keys)==len(set(keys))==2661 and set(keys)==set(by_key)
        y=np.asarray([r['actual'] for r in rows]);p=np.asarray([r['final_prediction'] for r in rows])
        assert y.shape==p.shape==(2661,8) and np.isfinite(p).all() and (p>=0).all()
        assert np.array_equal(y,np.asarray([by_key[k]['actual'] for k in keys]))
        assert np.array_equal(p,np.asarray([r['raw_prediction'] for r in rows]))
        np.testing.assert_allclose(np.asarray([r['target_z'] for r in rows]),np.log1p(y),rtol=0,atol=1e-12)
        np.testing.assert_allclose(np.asarray([r['prediction_z'] for r in rows]),np.log1p(p),rtol=0,atol=1e-12)
        group_mape=[];log_mae=[];rmsle=[];workload_mae={}
        for capture in sorted({k[0] for k in keys}):
            mask=np.asarray([k[0]==capture for k in keys]);ys,ps=y[mask],p[mask]
            group_mape.extend([float(100*np.mean(np.abs(ps[:,i][ys[:,i]>0]-ys[:,i][ys[:,i]>0])/ys[:,i][ys[:,i]>0])) for i in range(8)])
            le=np.log1p(ps)-np.log1p(ys)
            log_mae.extend(np.abs(le).mean(axis=0).tolist());rmsle.extend(np.sqrt((le**2).mean(axis=0)).tolist())
            workload_mae[capture]=float(np.abs(ps-ys).mean())
        assert len(group_mape)==40
        np.testing.assert_allclose(np.mean(group_mape),report['arms'][arm]['macro_mape_percent'],atol=1e-9,rtol=0)
        np.testing.assert_allclose(np.mean(log_mae),report['auxiliary'][arm]['macro_log_mae'],atol=1e-12,rtol=0)
        np.testing.assert_allclose(np.mean(rmsle),report['auxiliary'][arm]['macro_rmsle'],atol=1e-12,rtol=0)
        assert sum(v<10 for v in group_mape)==report['arms'][arm]['groups_mape_lt10']
        epochs=json.loads((directory/arm/'epochs.json').read_text())
        best=min(epochs,key=lambda e:(e['dev_macro_mape'],e['epoch']))
        assert best['epoch']==result['best_epoch'] and len(epochs)==result['internal_checkpoint_evaluations']
        base=report['arms']['T0_reused']
        improvement=1-np.mean(group_mape)/base['macro_mape_percent']
        ratios=report['comparison']['workload_mae_ratios']
        for capture,value in workload_mae.items():
            matches=[w for w in base['workload_mean_node_mae'] if w in capture]
            assert len(matches)==1
            w=matches[0]
            np.testing.assert_allclose(value/base['workload_mean_node_mae'][w],ratios[w],atol=1e-12,rtol=0)
        passes=improvement>=.05 and max(group_mape)/base['worst_group_mape_percent']<=1.05 and max(ratios.values())<=1.05
        assert passes==report['comparison']['passes']
        context=np.load(directory/'input_context.npz')
        assert context['internal_dev_preserved'].shape==(2661,8,77)
        assert all(int(t)<=key[1]-80_000_000 for t,key in zip(context['available_ns'],keys))
        verified[arm]=dict(report_sha256=sha(directory/'report.json'),samples=2661,original_labels=21288,
                           groups=40,macro_mape=float(np.mean(group_mape)),macro_log_mae=float(np.mean(log_mae)),
                           macro_rmsle=float(np.mean(rmsle)),passes=bool(passes),lease_seconds=lease['ended_epoch']-lease['started_epoch'])
    assert sha(root/'T1/transform.json')==sha(root/'T2/transform.json')
    state=json.loads((root/'T1/transform.json').read_text())
    assert not state['t3_admitted'] and state['t3_eligible_columns']==0
    return dict(state='B_frozen_outputs_verified_A_review_pending',arms=verified,
                input_transform_shared_exactly=True,T3='skipped_training_only_gate_not_met',
                new_fit=False,model_load=False,inference=False,training_statistics_refitted=False)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--root',type=Path,required=True)
    parser.add_argument('--baseline',type=Path,required=True);parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args();result=verify(args.root,args.baseline)
    args.output.write_text(json.dumps(result,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(result,indent=2))
