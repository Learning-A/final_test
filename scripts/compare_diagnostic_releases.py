"""Compare pinned diagnostic prediction releases; JSON arithmetic only, no fit."""
import argparse
import hashlib
import json
from pathlib import Path
import sys
import zipfile
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'idfs_forecast_lab'))
from idfs_lab.target_baselines import metrics

p=argparse.ArgumentParser(description=__doc__)
for name in ('reference','candidate','reference-sha','candidate-sha','output'):
    p.add_argument('--'+name,required=True)
args=p.parse_args()
releases=[]
for path,expected in ((args.reference,args.reference_sha),(args.candidate,args.candidate_sha)):
    if hashlib.sha256(Path(path).read_bytes()).hexdigest()!=expected:
        raise ValueError('archive checksum mismatch')
    release={}
    with zipfile.ZipFile(path) as z:
        for target in ('idfs44','cluster'):
            names=[n for n in z.namelist() if '_'+target+'_' in n and n.endswith('/results/diagnostic_model_manifest.json')]
            if len(names)!=1: raise ValueError('ambiguous target manifest')
            name=names[0];prefix=name.rsplit('/',1)[0]+'/'
            m=json.loads(z.read(name))
            if m.get('source_role')!='diagnostic' or m.get('fault_affected') is not True or m.get('formal_acceptance') is not False:
                raise ValueError('diagnostic provenance required')
            if hashlib.sha256(z.read(prefix+'model.ts')).hexdigest()!=m['artifact_sha256']:
                raise ValueError('weights checksum mismatch')
            rows=json.loads(z.read(prefix+'validation_predictions.json'))
            score=metrics((r['prediction_iops'],r['target_iops']) for r in rows)
            for k,v in score.items():
                if (v is None)!=(m['validation_metrics'][k] is None): raise ValueError('metric mismatch')
                if v is not None and abs(v-m['validation_metrics'][k])>1e-7: raise ValueError('metric mismatch')
            release[target]=(m,rows,score)
    releases.append(release)
out=dict(role='diagnostic',fault_affected=True,independent_validation=False,formal_acceptance=False,
         fitting_performed=False,reference_sha256=args.reference_sha,candidate_sha256=args.candidate_sha,targets={})
for target in ('idfs44','cluster'):
    m1,a,s1=releases[0][target];m2,b,s2=releases[1][target]
    if len(a)!=len(b): raise ValueError('unpaired rows')
    for x,y in zip(a,b):
        if any(x[k]!=y[k] for k in ('window_start_ns','features','target_iops','feature_end_ns','features_available_ns')):
            raise ValueError('changed inputs/labels/availability')
    worst=sorted((i for i in range(len(b)) if b[i]['target_iops']>0),
                 key=lambda i:abs(b[i]['prediction_iops']-b[i]['target_iops'])/b[i]['target_iops'],reverse=True)[:5]
    h=m2['training_details']['history']
    out['targets'][target]=dict(reference=s1,candidate=s2,mae_delta=s2['mae_iops']-s1['mae_iops'],
        mape_delta=s2['mape_percent']-s1['mape_percent'],candidate_commit=m2['candidate_commit'],code_dirty=m2['code_dirty'],
        output_structure=m2['training_details'].get('output_structure','absolute_log_v1'),
        epochs=len(h),best_validation_epoch=min(h,key=lambda r:r['validation_loss'])['epoch'],
        candidate_range=[min(r['prediction_iops'] for r in b),max(r['prediction_iops'] for r in b)],
        last_available=metrics((r['features'][0],r['target_iops']) for r in b),
        mean_five=metrics((r['features'][5],r['target_iops']) for r in b),
        mae_improved_windows=sum(abs(y['prediction_iops']-y['target_iops'])<abs(x['prediction_iops']-x['target_iops']) for x,y in zip(a,b)),
        worst_positive_ape=[dict(window_start_ns=b[i]['window_start_ns'],actual=b[i]['target_iops'],lags=b[i]['features'][:5],
                                reference_prediction=a[i]['prediction_iops'],candidate_prediction=b[i]['prediction_iops']) for i in worst])
Path(args.output).write_text(json.dumps(out,indent=2)+'\n',encoding='utf-8')
print(json.dumps(out))
