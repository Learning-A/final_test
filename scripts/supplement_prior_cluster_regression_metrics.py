"""Supplement the prior 2661-origin frozen audit with per-workload Pearson r."""
import gzip
import hashlib
import json
from pathlib import Path

from supplement_cluster_regression_metrics import ROOT,metric,summarize


def read(path):
    with gzip.open(ROOT/path,'rt') as f:return list(map(json.loads,f))


def main():
    old=json.loads((ROOT/'reports/cluster_metric_feasibility_20261123.json').read_text())
    for path,pin in old['input_manifest'].items():
        assert hashlib.sha256((ROOT/path).read_bytes()).hexdigest()==pin['sha256']
    r1=read('.local/r1-frozen/results/predictions.jsonl.gz')
    lookup={(r['dataset_id'],r['window_start_ns']):r['workload'] for r in r1 if r['margin_ms']==50 and r['arm']=='F0'}
    rows_by_model={}
    for margin,arm,name in [(50,'F0','R1_F0'),(50,'F0_F1','R1_F1_50'),(100,'F0_F1','R1_F1_100'),(200,'F0_F1','R1_F1_200')]:
        rows_by_model[name]=[r for r in r1 if r['margin_ms']==margin and r['arm']==arm]
    for arm in ('L0','L1','L2','L3'):
        rows_by_model['R2_'+arm]=read(f'.local/r2-results/results/{arm}/predictions.jsonl.gz')
    irows=read('.local/v2_i1/fitted/predictions.jsonl.gz')
    for arm in ('CB','CBP'):rows_by_model['I1_'+arm]=[r for r in irows if r['arm']==arm]
    for i in (1,2,3):rows_by_model[f'V2_M{i}']=read(f'.local/v2_m{i}/fitted/predictions.jsonl.gz')
    for arm in ('T1','T2'):rows_by_model[arm]=read(f'.local/data_transform/{arm}/{arm}/predictions.jsonl.gz')
    result={}
    for name,rows in rows_by_model.items():
        assert len(rows)==2661 and {(r['dataset_id'],r['window_start_ns']) for r in rows}==set(lookup)
        targets={}
        for kind in ('direct_cluster','summed_node_cluster'):
            if kind not in old['models'][name]:continue
            groups={}
            for workload in sorted(set(lookup.values())):
                selected=[r for r in rows if lookup[(r['dataset_id'],r['window_start_ns'])]==workload]
                y=[sum(r['actual'][:8]) for r in selected]
                native=[r.get('final_predictions',r.get('final_prediction')) for r in selected]
                p=[v[8] if kind=='direct_cluster' else sum(v[:8]) for v in native]
                groups[workload]=metric(y,p)
                expected=old['models'][name][kind]['groups'][workload]
                for key in ('mae_iops','rmse_iops','r2'):
                    assert abs(groups[workload][key]-expected[key])<1e-8
            targets[kind]=dict(groups=groups,macro=summarize(groups))
        result[name]=targets
    data=dict(scope='Previous2661 train03 origins;15 immutable configurations and19 direct/summed paths, not the new2297-origin horizon cohort. No model fit, load or inference.',
              input_hashes_matched_prior_audit=True,independent_stdlib_moments_verified=True,models=result)
    (ROOT/'reports/prior_cluster_regression_metrics.json').write_text(json.dumps(data,indent=2)+'\n',encoding='utf-8',newline='\n')
    print('Verified15 configurations,19 paths; added within-workload Pearson without changing old scores.')


if __name__=='__main__':main()
