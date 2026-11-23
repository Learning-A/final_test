"""Independently replay frozen preprocessing-study outputs without fitting."""
from collections import defaultdict
import gzip
import hashlib
import json
import math
from pathlib import Path
import statistics
import zipfile

import numpy as np


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read(path):
    return json.loads(path.read_bytes())


def rows(path):
    with gzip.open(path,'rt',encoding='utf8') as stream:
        return list(map(json.loads,stream))


def metrics(predictions):
    groups=defaultdict(list)
    for r in predictions:
        assert len(r['actual'])==len(r['final_prediction'])==8
        assert r['raw_prediction']==r['final_prediction']
        for i,(actual,pred) in enumerate(zip(r['actual'],r['final_prediction'])):
            assert actual>=0 and pred>=0 and math.isfinite(pred)
            groups[r['dataset_id'],i].append((actual,pred))
    maps=[];maes=defaultdict(list);logs=[];rms=[];wapes=[]
    assert len(groups)==40
    for (dataset,node),pairs in groups.items():
        maps.append(statistics.mean(abs(p-y)/y*100 for y,p in pairs if y>0))
        maes[dataset].append(statistics.mean(abs(p-y) for y,p in pairs))
        residual=[math.log1p(p)-math.log1p(y) for y,p in pairs]
        logs.append(statistics.mean(abs(v) for v in residual))
        rms.append(math.sqrt(statistics.mean(v*v for v in residual)))
        wapes.append(100*sum(abs(p-y) for y,p in pairs)/sum(y for y,p in pairs))
    return dict(macro_mape=statistics.mean(maps),worst=max(maps),groups_lt10=sum(v<10 for v in maps),
        workload_mae={d:statistics.mean(v) for d,v in maes.items()},
        macro_log_mae=statistics.mean(logs),macro_rmsle=statistics.mean(rms),macro_wape=statistics.mean(wapes))


def main():
    root=Path('../work/idfs-model-b/.local/data_transform')
    inputs=Path('../work/idfs-model-b/.local/v2_i1/inputs')
    archive=next(Path('../work/idfs-model-b/.local/r2-results').glob('*.zip'))
    assert sha(archive)=='b671e0029e8d0703b34a34a8b26f36624af7be1ed19a72f8726990add077bcf6'
    with zipfile.ZipFile(archive) as z:
        baseline=list(map(json.loads,gzip.decompress(z.read('results/L0/predictions.jsonl.gz')).splitlines()))
        sample_ids=json.loads(z.read('sample_ids.json'))
    keys=[tuple(x) for x in sample_ids['internal_dev']]
    original={ (r['dataset_id'],r['window_start_ns']):r['actual'] for r in baseline}
    assert len(original)==len(keys)==2661
    assert sha(inputs/'inputs.npz')=='ffc01763d54e6185eb62772769e489eed63b6eef817bf79171adfbbb24d41170'
    source=np.load(inputs/'inputs.npz',allow_pickle=False)
    column_contract=read(Path('reports/data_transform_A_column_contract.json'))['columns']
    preserved=[x['column'] for x in column_contract['preserved']]
    results={'T0':metrics(baseline)}
    evidence={}
    for arm in ('T1','T2'):
        d=root/arm
        report=read(d/'report.json');state=read(d/'transform.json');lease=read(d/'lease.json')
        assert report['state']=='completed' and report['new_fit_count']==1
        assert report['baseline_fit_count']==0 and report['external_validation_evaluations']==0
        assert sha(d/'transform.json')==report['transform_sha256']
        assert lease['state']=='completed' and lease['exit_code']==0
        assert lease['ended_epoch']-lease['started_epoch']<=1800
        contract=state['contract']
        assert contract['nonnegative']==[x['column'] for x in column_contract['nonnegative_log1p']]
        assert contract['signed']==[30,31,32] and contract['preserved']==preserved
        assert state['source_role']=='train' and state['partition']=='fit_run01_02' and state['fitted_on']=='lmj@b507'
        assert state['method']=='log1p'
        assert all(state['center'][i]==0 and state['scale'][i]==1 for i in preserved)
        assert all(math.isfinite(v) and v>0 for v in state['scale'])
        eligible=sum(x['observed']>=1000 and x['distinct']>=100 and abs(x['log_skew'])>2 for x in state['diagnostics'])
        assert eligible==state['t3_eligible_columns'] and state['t3_admitted']==(eligible>=18)
        if arm=='T2':
            assert (d/'transform.json').read_bytes()==(root/'T1/transform.json').read_bytes()
        pred=d/arm/'predictions.jsonl.gz';logs=d/arm/'log_predictions.jsonl.gz'
        assert sha(pred)==report['training']['predictions_sha256']
        assert sha(logs)==report['log_predictions_sha256']
        rr=rows(pred);zz=rows(logs)
        assert [(r['dataset_id'],r['window_start_ns']) for r in rr]==keys
        assert len(rr)==len(zz)==2661
        for r,z,k in zip(rr,zz,keys):
            assert r['actual']==original[k] and z['actual']==r['actual']
            assert (z['dataset_id'],z['window_start_ns'])==k
            assert r['final_prediction']==z['final_prediction']
            for y,p,ty,pz in zip(r['actual'],r['final_prediction'],z['target_z'],z['prediction_z']):
                assert math.isclose(ty,math.log1p(y),abs_tol=1e-12)
                assert math.isclose(pz,math.log1p(p),abs_tol=1e-12)
        context=np.load(d/'input_context.npz',allow_pickle=False)
        assert sha(d/'input_context.npz')==report['input_context_sha256']
        assert np.array_equal(context['internal_dev_preserved'],source['internal_dev_features'][:,:,preserved])
        assert np.array_equal(context['available_ns'],source['internal_dev_available_ns'])
        epochs=read(d/arm/'epochs.json')
        assert sha(d/arm/'epochs.json')==report['training']['epochs_sha256']
        assert [x['epoch'] for x in epochs]==list(range(len(epochs)))
        chosen=min(epochs,key=lambda x:(x['dev_macro_mape'],x['epoch']))
        assert chosen['epoch']==report['training']['best_epoch']
        m=metrics(rr);base=results['T0']
        assert math.isclose(m['macro_mape'],report['arms'][arm]['macro_mape_percent'],rel_tol=1e-12)
        for name in ('macro_log_mae','macro_rmsle'):
            assert math.isclose(m[name],report['auxiliary'][arm][name],rel_tol=1e-12)
        gain=1-m['macro_mape']/base['macro_mape']
        ratios={k:v/base['workload_mae'][k] for k,v in m['workload_mae'].items()}
        screen=gain>=.05 and m['worst']<=base['worst']*1.05 and all(v<=1.05 for v in ratios.values())
        assert screen==report['comparison']['passes']
        m.update(relative_mape_improvement=gain,workload_mae_ratios=ratios,screen=screen)
        results[arm]=m
        evidence[arm]=dict(report_sha256=sha(d/'report.json'),transform_sha256=sha(d/'transform.json'),
            predictions_sha256=sha(pred),best_epoch=chosen['epoch'],scoring_count=len(epochs),
            preserved_values_checked=2661*8*77,t3_eligible_columns=eligible,t3_admitted=state['t3_admitted'],
            lease_seconds=lease['ended_epoch']-lease['started_epoch'])
    result=dict(state='A_independent_frozen_T1_T2_metrics_verified',arms=results,evidence=evidence,
        original_labels_matched=21288,prediction_pairs_checked=42624,new_A_fit_count=0,
        scope='Immutable local reports/prediction hashes, labels, log values, preserved context, epoch selection, stored T3 gate; no transform statistics refit or weight load',
        formal_metric_replaced=False)
    Path('reports/data_transform_A_independent_review.json').write_text(json.dumps(result,indent=2)+'\n',encoding='utf8')
    print(json.dumps(result))


if __name__=='__main__':
    main()
