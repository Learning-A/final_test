"""Independent frozen arithmetic and unit/identity checks, no model loading."""
from collections import defaultdict
import csv
import gzip
import hashlib
import json
import math
from pathlib import Path
import statistics

import numpy as np

ROOT=Path(__file__).resolve().parents[1]
HORIZONS=(1,3,5,10)


def sha(p):
    with p.open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()


def metric(y,p):
    err=[abs(t-q) for t,q in zip(y,p)];ape=[e/t for e,t in zip(err,y) if t>0]
    return dict(n=len(y),zero_count=sum(t==0 for t in y),mape_percent=100*statistics.mean(ape) if ape else None,
        wape_percent=100*math.fsum(err)/math.fsum(y) if math.fsum(y)>0 else None,
        mae_iops=statistics.mean(err),bias_iops=statistics.mean([q-t for t,q in zip(y,p)]),
        mdape_percent=100*statistics.median(ape) if ape else None,
        pred20_percent=100*sum(e<=.2*t for e,t in zip(err,y))/len(y),
        log_mae=statistics.mean([abs(math.log1p(q)-math.log1p(t)) for t,q in zip(y,p)]))


def same(a,b):
    if isinstance(a,dict):
        assert set(a)==set(b)
        for k in a:same(a[k],b[k])
    elif a is None:assert b is None
    else:assert math.isclose(a,b,rel_tol=1e-10,abs_tol=1e-8),(a,b)


def main():
    root=ROOT/'.local/cluster_horizon';keys=json.loads((root/'sample_ids.json').read_text())
    raw=np.load(root/'prepared.npz',allow_pickle=False);positions={tuple(k):i for i,k in enumerate(keys['internal_dev'])}
    meta=json.loads((root/'metadata.json').read_text());receipt=json.loads((root/'receipt.json').read_text())
    for entry in receipt['outputs'].values():assert sha(root/entry['file'])==entry['sha256']
    reports={};outputs={};result={};persistence={}
    for arm in ('direct_seconds','fine100ms'):
        dest=root/arm;r=json.loads((dest/'report.json').read_text());lease=json.loads((dest/'lease.json').read_text())
        assert r['state']=='completed' and lease['state']=='completed' and lease['exit_code']==0
        assert lease['ended_epoch']-lease['started_epoch']<=1500
        for file,key in [('predictions.jsonl.gz','predictions_sha256'),('checkpoint.pt','checkpoint_sha256'),('epochs.json','epochs_sha256')]:assert sha(dest/file)==r[key]
        epochs=json.loads((dest/'epochs.json').read_text());best=min(epochs,key=lambda e:(e['checkpoint_wape'],e['epoch']))
        assert best['epoch']==r['best_epoch'] and len(epochs)==r['internal_checkpoint_evaluations']
        rows=[]
        with gzip.open(dest/'predictions.jsonl.gz','rt') as f:
            for row in map(json.loads,f):
                key=(row['dataset_id'],row['window_start_ns']);i=positions[key]
                assert row['workload']==meta[key[0]]['workload']
                counts=row['actual_100ms_counts'];ys=row['actual_seconds'];ps=row['prediction_seconds'];native=row['prediction_native_rates']
                assert len(counts)==100 and len(ys)==len(ps)==10 and len(native)==(10 if arm=='direct_seconds' else 100)
                assert counts==raw['internal_dev_counts'][i].tolist() and ys==raw['internal_dev_seconds'][i].tolist()
                assert all(math.isfinite(v) and v>=0 for v in counts+ys+ps+native)
                for j in range(10):
                    assert math.isclose(math.fsum(counts[10*j:10*j+10]),ys[j],abs_tol=1e-8)
                    predicted=native[j] if arm=='direct_seconds' else math.fsum(native[10*j:10*j+10])/10
                    assert math.isclose(predicted,ps[j],abs_tol=1e-8)
                for j,h in enumerate(HORIZONS):
                    ty=math.fsum(ys[:h])/h;pp=math.fsum(ps[:h])/h
                    for k,v in [('actual_window_means',ty),('prediction_window_means',pp),('actual_window_operations',ty*h),('prediction_window_operations',pp*h)]:
                        assert math.isclose(row[k][j],v,abs_tol=1e-8)
                assert row['features_available_ns']==int(raw['internal_dev_available_ns'][i])<=row['source_input_deadline_ns']==key[1]-80_000_000
                rows.append(row)
        assert [(r['dataset_id'],r['window_start_ns']) for r in rows]==[tuple(k) for k in keys['internal_dev']]
        group=defaultdict(list)
        for row in rows:group[row['workload']].append(row)
        table={};trajectory={}
        for w,rr in group.items():
            table[w]={str(h):metric([r['actual_window_means'][j] for r in rr],[r['prediction_window_means'][j] for r in rr]) for j,h in enumerate(HORIZONS)}
            trajectory[w]={str(j+1):metric([r['actual_seconds'][j] for r in rr],[r['prediction_seconds'][j] for r in rr]) for j in range(10)}
        same(table,r['summary']['window_groups']);same(trajectory,r['summary']['trajectory_groups'])
        for h in HORIZONS:
            for name,v in r['summary']['window_macro'][str(h)].items():
                assert math.isclose(statistics.mean([g[str(h)][name] for g in table.values()]),v,abs_tol=1e-8)
        assert math.isclose(statistics.mean([v['wape_percent'] for g in table.values() for v in g.values()]),best['checkpoint_wape'],abs_tol=1e-8)
        reports[arm]=r;outputs[arm]=rows
        result[arm]=dict(state='frozen_arithmetic_verified',sample_count=len(rows),cluster_100ms_labels=len(rows)*100,
            future_second_labels=len(rows)*10,metrics_groups=5*4+5*10,report_sha256=sha(dest/'report.json'),
            lease_seconds=lease['ended_epoch']-lease['started_epoch'],best_epoch=r['best_epoch'],
            window_macro=r['summary']['window_macro'])
    direct,fine=reports.values()
    for field in ('learned_parameters','initial_parameter_sha256','batch_permutations_sha256','training_target_scale','fit_samples','internal_dev_samples','fit_run_counts'):
        assert direct[field]==fine[field],field
    # Exact frozen reference, no fit/reinference; report both full and narrowed cohorts separately.
    old={}
    with gzip.open(ROOT/'.local/r1-frozen/results/predictions.jsonl.gz','rt') as f:
        for r in map(json.loads,f):
            if r['margin_ms']==50 and r['arm']=='F0_F1':old[(r['dataset_id'],r['window_start_ns'])]=r
    reference={}
    for w in direct['summary']['window_groups']:
        kk=[k for k in positions if meta[k[0]]['workload']==w]
        for key in kk:
            i=positions[key];assert old[key]['actual'][8]==raw['internal_dev_seconds'][i,0]
        reference[w]=metric([old[k]['actual'][8] for k in kk],[old[k]['final_predictions'][8] for k in kk])
    # Each example is the first common-key sample of that workload, selected without inspecting errors.
    examples=[]
    for w in sorted(direct['summary']['window_groups']):
        i=next(i for i,row in enumerate(outputs['direct_seconds']) if row['workload']==w)
        d=outputs['direct_seconds'][i];f=outputs['fine100ms'][i]
        examples.append(dict(workload=w,dataset_id=d['dataset_id'],window_start_ns=d['window_start_ns'],
            actual_seconds=d['actual_seconds'],direct_prediction_seconds=d['prediction_seconds'],
            fine_prediction_seconds=f['prediction_seconds'],actual_window_means=d['actual_window_means'],
            direct_window_means=d['prediction_window_means'],fine_window_means=f['prediction_window_means']))
    report=dict(state='B_frozen_two_path_verification_completed',arms=result,equal_capacity_initialization_batches_and_scale=True,
        original_frozen_R1_direct_1s_on_common_keys=dict(groups=reference,macro_wape_percent=statistics.mean(g['wape_percent'] for g in reference.values())),
        no_fit=True,no_model_load=True,no_inference=True,external_validation_new_evaluations=0,
        cohort_notice='All future 10s labels complete; 2297 current seconds differ from old2661. Overlapping windows are not independent trials.')
    for name,data in [('cluster_horizon_B_verification.json',report),('cluster_horizon_examples.json',examples)]:
        (ROOT/'reports'/name).write_text(json.dumps(data,indent=2)+'\n',encoding='utf-8',newline='\n')
    out=ROOT.parent.parent/'outputs/cluster_horizon_20261123';out.mkdir(parents=True,exist_ok=True)
    with (out/'cluster_future_10seconds.csv').open('w',encoding='utf-8-sig',newline='') as f:
        writer=csv.writer(f);writer.writerow(['dataset_id','window_start_ns','workload','lead_second','actual_iops','direct_iops','fine100ms_aggregated_iops'])
        for d,q in zip(outputs['direct_seconds'],outputs['fine100ms']):
            for j in range(10):writer.writerow([d['dataset_id'],d['window_start_ns'],d['workload'],j+1,d['actual_seconds'][j],d['prediction_seconds'][j],q['prediction_seconds'][j]])
    with (out/'cluster_window_means.csv').open('w',encoding='utf-8-sig',newline='') as f:
        writer=csv.writer(f);writer.writerow(['dataset_id','window_start_ns','workload','window_seconds','actual_average_iops','direct_average_iops','fine_average_iops'])
        for d,q in zip(outputs['direct_seconds'],outputs['fine100ms']):
            for j,h in enumerate(HORIZONS):writer.writerow([d['dataset_id'],d['window_start_ns'],d['workload'],h,d['actual_window_means'][j],d['prediction_window_means'][j],q['prediction_window_means'][j]])
    print(json.dumps(report,indent=2))


if __name__=='__main__':main()
