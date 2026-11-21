"""Recompute pilot errors and fixed value bands; no model loading or fitting."""
import argparse
import hashlib
import json
from pathlib import Path
import sys
import zipfile
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'idfs_forecast_lab'))
from idfs_lab.target_baselines import metrics

p=argparse.ArgumentParser(description=__doc__)
p.add_argument('--archive',required=True)
p.add_argument('--output',required=True)
args=p.parse_args()
raw=Path(args.archive).read_bytes()
sha=hashlib.sha256(raw).hexdigest()
if sha!='b459e2c1ce363f68713a6d76cdf47f0b58db4ccb644128f0990d1f7e335c9b44':
    raise ValueError('pilot archive checksum mismatch')
out={'archive_sha256':sha,'role':'diagnostic','fault_affected':True,
     'independent_validation':False,'formal_acceptance':False,'fitting_performed':False,'targets':{}}
with zipfile.ZipFile(args.archive) as bundle:
    for target in ('idfs44','cluster'):
        prefix='diagnostic_sequence_v1_'+target+'_20261122_a03/'
        manifest=json.loads(bundle.read(prefix+'results/diagnostic_model_manifest.json'))
        model_hash=hashlib.sha256(bundle.read(prefix+'results/model.ts')).hexdigest()
        if model_hash!=manifest['artifact_sha256']:
            raise ValueError('weights checksum mismatch')
        rows=json.loads(bundle.read(prefix+'results/validation_predictions.json'))
        score=metrics((r['prediction_iops'],r['target_iops']) for r in rows)
        for key,value in score.items():
            expected=manifest['validation_metrics'][key]
            if value is None:
                assert expected is None
            else:
                assert abs(value-expected)<=1e-8*max(1,abs(value))
        errors=[abs(r['prediction_iops']-r['target_iops'])/r['target_iops'] for r in rows if r['target_iops']>0]
        total_ape=sum(errors)
        bands=[]
        for label,lo,hi in (('zero',-1,0),('(0,10]',0,10),('(10,100]',10,100),('(100,1000]',100,1000),('>1000',1000,float('inf'))):
            selected=[r for r in rows if lo<r['target_iops']<=hi]
            ape=sum(abs(r['prediction_iops']-r['target_iops'])/r['target_iops'] for r in selected if r['target_iops']>0)
            bands.append(dict(band=label,metrics=metrics((r['prediction_iops'],r['target_iops']) for r in selected),
                              ape_share=ape/total_ape if total_ape else None,
                              mean_residual=sum(r['prediction_iops']-r['target_iops'] for r in selected)/len(selected) if selected else None))
        history=manifest['training_details']['history']
        best=min(history,key=lambda r:r['validation_loss'])
        out['targets'][target]=dict(metrics=score,weight_sha256=model_hash,
            candidate_commit=manifest['candidate_commit'],code_dirty=manifest['code_dirty'],
            training_examples=manifest['train_examples'],validation_examples=len(rows),
            training_log_mean=manifest['training_details']['target_log_mean'],
            training_log_std=manifest['training_details']['target_log_std'],
            epochs=len(history),best_validation_epoch=best['epoch'],first_epoch=history[0],last_epoch=history[-1],
            actual_range=[min(r['target_iops'] for r in rows),max(r['target_iops'] for r in rows)],
            prediction_range=[min(r['prediction_iops'] for r in rows),max(r['prediction_iops'] for r in rows)],
            top_five_ape_share=sum(sorted(errors,reverse=True)[:5])/total_ape,
            last_available_metrics=metrics((r['features'][0],r['target_iops']) for r in rows),
            mean_five_metrics=metrics((r['features'][5],r['target_iops']) for r in rows),bands=bands)
Path(args.output).write_text(json.dumps(out,indent=2)+'\n',encoding='utf-8')
print(json.dumps(out))
