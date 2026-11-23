"""Recompute I1 groups and error strata directly from stored predictions."""
import argparse
from collections import defaultdict
import gzip
import hashlib
import json
from pathlib import Path
import zipfile

import numpy as np


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--results',type=Path,required=True)
    parser.add_argument('--inputs',type=Path,required=True)
    parser.add_argument('--r1-archive',type=Path,required=True)
    parser.add_argument('--phases',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    if args.output.exists():
        raise ValueError('Do not repeat completed independent verification')
    report=json.loads((args.results/'report.json').read_text(encoding='utf-8'))
    predictions=args.results/'predictions.jsonl.gz'
    if sha(predictions)!=report['predictions_sha256']:
        raise ValueError('Prediction checksum differs')
    if report['state']!='completed' or report['new_fit_count']!=16 or report['external_validation_evaluations']!=0:
        raise ValueError('Execution count/state differs from card')
    ids=json.loads((args.inputs/'sample_ids.json').read_text(encoding='utf-8'))
    keys=set(map(tuple,ids['internal_dev']))
    if len(keys)!=2661:
        raise ValueError('Different held-out keys')
    if sha(args.r1_archive)!='76c81d14092b84bd80aaeadd69c477739a4c15df17e17a2c3fc43f1738655eb0':
        raise ValueError('R1 archive pin differs')
    with zipfile.ZipFile(args.r1_archive) as z:
        old=[json.loads(line) for line in gzip.decompress(z.read('results/predictions.jsonl.gz')).splitlines()]
    original={(r['dataset_id'],r['window_start_ns']):r for r in old if r['arm']=='F0_F1' and r['margin_ms']==50}
    with gzip.open(args.phases,'rt',encoding='utf-8') as stream:
        phases={(r['dataset_id'],r['window_start_ns']):r['phase'] for r in map(json.loads,stream)}
    all_rows=defaultdict(dict)
    with gzip.open(predictions,'rt',encoding='utf-8') as stream:
        for row in map(json.loads,stream):
            key=(row['dataset_id'],row['window_start_ns'])
            arm=row['arm']
            if key not in keys or key in all_rows[arm]:
                raise ValueError('Unexpected/duplicate prediction key')
            if row['actual']!=original[key]['actual'][:8] or row['workload']!=original[key]['workload']:
                raise ValueError('Original label or workload mismatch')
            if row['source_role']!='train' or row['features_available_upper_ns']>row['feature_deadline_ns']:
                raise ValueError('Role or source-time cutoff differs')
            if row['raw_predictions']!=row['final_predictions']:
                raise ValueError('Undeclared postprocessing')
            if arm=='C' and row['final_predictions']!=original[key]['final_predictions'][:8]:
                raise ValueError('Reused C prediction differs')
            all_rows[arm][key]=row
    if set(all_rows)!=set(('C','CB','CBP')) or any(set(rows)!=keys for rows in all_rows.values()):
        raise ValueError('Arms or complete sample coverage differs')
    out={}
    mag_labels=('zero','positive_le10','10_lt_y_le100','100_lt_y_le1000','gt1000')
    for arm,rows in all_rows.items():
        group_mape=[]
        workload_mae={}
        bands=defaultdict(float)
        phase_totals=defaultdict(float)
        for workload in sorted({r['workload'] for r in rows.values()}):
            rs=[r for r in rows.values() if r['workload']==workload]
            y=np.array([r['actual'] for r in rs],dtype=float)
            p=np.array([r['final_predictions'] for r in rs],dtype=float)
            if y.shape[1]!=8 or not np.isfinite(p).all() or (p<0).any():
                raise ValueError('Invalid prediction values')
            positive=y>0
            n=positive.sum(0)
            ape=np.zeros_like(y)
            np.divide(100*np.abs(p-y),y,out=ape,where=positive)
            group_mape.extend((ape.sum(0)/n).tolist())
            workload_mae[workload]=float(np.abs(p-y).mean())
            for label,mask in zip(mag_labels,(y==0,(y>0)&(y<=10),(y>10)&(y<=100),(y>100)&(y<=1000),y>1000)):
                bands[label]+=float((np.where(mask,ape,0).sum(0)/n).sum()/40)
            phase_labels=[phases.get((r['dataset_id'],r['window_start_ns']),'unknown') for r in rs]
            for label in set(phase_labels):
                mask=np.array([p==label for p in phase_labels])
                phase_totals[label]+=float((ape[mask].sum(0)/n).sum()/40)
        macro=float(np.mean(group_mape))
        saved=report['arms'][arm]
        if abs(macro-saved['macro_mape_percent'])>1e-9 or abs(max(group_mape)-saved['worst_group_mape_percent'])>1e-9:
            raise ValueError('Macro or worst-group score differs')
        for w,value in workload_mae.items():
            if abs(value-saved['workload_mean_node_mae'][w])>1e-9:
                raise ValueError('Workload MAE differs')
        if abs(sum(bands.values())-macro)>1e-9 or abs(sum(phase_totals.values())-macro)>1e-9:
            raise ValueError('Attribution not additive')
        out[arm]=dict(macro_mape_percent=macro,worst_group_mape_percent=max(group_mape),
            groups_mape_lt10=sum(m<10 for m in group_mape),workload_mean_node_mae=workload_mae,
            magnitude_contributions_pp=dict(bands),phase_contributions_pp=dict(phase_totals))
    for key,candidate,base in (('CB_vs_C','CB','C'),('CBP_vs_CB','CBP','CB'),('CBP_vs_C','CBP','C')):
        a,b=out[candidate],out[base]
        gain=1-a['macro_mape_percent']/b['macro_mape_percent']
        worst=a['worst_group_mape_percent']/b['worst_group_mape_percent']
        ratio=max(a['workload_mean_node_mae'][w]/b['workload_mean_node_mae'][w] for w in b['workload_mean_node_mae'])
        decision=gain>=.05 and worst<=1.05 and ratio<=1.05
        if decision!=report['comparisons'][key]['passes']:
            raise ValueError('Promotion screen differs')
    result=dict(state='verified_stored_predictions_metrics_and_screen',node_prediction_labels=63864,
        original_unique_node_labels=21288,prediction_rows=7983,source_role='train',
        report_sha256=sha(args.results/'report.json'),predictions_sha256=sha(predictions),
        source_phase_sha256=sha(args.phases),runner_sha256=sha(Path(__file__)),
        C_predictions_exact=True,new_fit_count=0,arms=out,any_screen_passes=False,
        scope='Frozen source labels and stored predictions, without model deserialization or timing replay')
    args.output.write_text(json.dumps(result,indent=2)+'\n',encoding='utf-8',newline='\n')
    print(json.dumps({k:v for k,v in result.items() if k not in ('arms',)}))


if __name__=='__main__':
    main()
