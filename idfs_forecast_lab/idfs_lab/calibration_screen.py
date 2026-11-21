"""Causal calibration ablation using previously received smoke settlements."""
from pathlib import Path
import itertools
import statistics

from .io import read_events, save


def metrics(pairs):
    errors = [abs(p-a) for p,a in pairs]
    positive = [(p,a) for p,a in pairs if a>0]
    return {'mae_iops':statistics.mean(errors),
            'wape_percent':100*sum(errors)/sum(a for _,a in pairs),
            'mape_percent':100*statistics.mean(abs(p-a)/a for p,a in positive),
            'zero_excluded_from_mape':len(pairs)-len(positive)}


def screen(directory):
    streams=[]
    baseline={name:[] for name in ('last_available','mean_five_available','raw_moe','published_moe')}
    for file in sorted(Path(directory).glob('*/predictions.jsonl')):
        issued={r['window_start_ns']:r for r in read_events(file) if r['event']=='prediction'}
        settled=sorted(read_events(file.with_name('settlement.jsonl')),key=lambda r:r['settled_epoch_ns'])
        ordered=sorted(settled,key=lambda r:issued[r['window_start_ns']]['compute_start_ns'])
        streams.append((issued,settled,ordered))
        for row in ordered:
            event=issued[row['window_start_ns']];x=event['feature_vector'];a=row['actual_iops']
            for name,p in [('last_available',x[14]),('mean_five_available',statistics.mean(x[2:15:3])),
                           ('raw_moe',event['model_raw_iops']),('published_moe',event['prediction_iops'])]:
                baseline[name].append((p,a))
    candidates=[]
    for alpha,cap in itertools.product([.02,.05,.1,.2,.5,1.],[25,50,100,250,500,1000]):
        pairs=[]
        for issued,settled,ordered in streams:
            position=0;ewma=0.;n=0
            for row in ordered:
                event=issued[row['window_start_ns']]
                # Apply only settlements that had arrived before this computation.
                while position<len(settled) and settled[position]['settled_epoch_ns']<event['compute_start_ns']:
                    old=settled[position];raw=issued[old['window_start_ns']]['model_raw_iops']
                    residual=old['actual_iops']-raw
                    ewma=residual if n==0 else alpha*residual+(1-alpha)*ewma
                    n+=1;position+=1
                correction=max(-cap,min(cap,ewma)) if n>=8 else 0.
                pairs.append((max(0,event['model_raw_iops']+correction),row['actual_iops']))
        candidates.append(dict(alpha=alpha,cap_iops=cap,**metrics(pairs)))
    return {'kind':'causal_smoke_calibration_screen','selection_data':'existing smoke only',
            'independent_validation':False,'formal_acceptance':False,
            'baselines':{name:metrics(pairs) for name,pairs in baseline.items()},
            'candidates_ranked_by_mae':sorted(candidates,key=lambda r:r['mae_iops']),
            'note':'Screening candidates requires independent workload validation before deployment.'}


if __name__=='__main__':
    import argparse
    p=argparse.ArgumentParser();p.add_argument('--input',required=True);p.add_argument('--out',required=True)
    args=p.parse_args();save(args.out,screen(args.input))
