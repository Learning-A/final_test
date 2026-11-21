"""Paired JSON error analysis of two fixed releases; never fit/load a model."""
import argparse
import hashlib
import json
from pathlib import Path
import sys
import zipfile
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'idfs_forecast_lab'))
from idfs_lab.target_baselines import metrics

p=argparse.ArgumentParser(description=__doc__)
p.add_argument('--v1',required=True)
p.add_argument('--v2',required=True)
p.add_argument('--output',required=True)
args=p.parse_args()
sources=[(args.v1,'b459e2c1ce363f68713a6d76cdf47f0b58db4ccb644128f0990d1f7e335c9b44','diagnostic_sequence_v1_','20261122_a03'),
         (args.v2,'e50c529e55616137120ab0d50b01f396266f111b36e78e2afa0ad7224ca3bdcd','diagnostic_inputnorm_v2_','20261122_a01')]
data=[]
for path,expected,prefix,suffix in sources:
    if hashlib.sha256(Path(path).read_bytes()).hexdigest()!=expected:
        raise ValueError('archive checksum mismatch')
    values={}
    with zipfile.ZipFile(path) as bundle:
        for target in ('idfs44','cluster'):
            root=prefix+target+'_'+suffix+'/results/'
            m=json.loads(bundle.read(root+'diagnostic_model_manifest.json'))
            if hashlib.sha256(bundle.read(root+'model.ts')).hexdigest()!=m['artifact_sha256']:
                raise ValueError('model checksum mismatch')
            rows=json.loads(bundle.read(root+'validation_predictions.json'))
            score=metrics((r['prediction_iops'],r['target_iops']) for r in rows)
            for key,v in score.items():
                assert (v is None and m['validation_metrics'][key] is None) or abs(v-m['validation_metrics'][key])<1e-7
            values[target]=(m,rows,score)
    data.append(values)
out={'role':'diagnostic','fault_affected':True,'independent_validation':False,'formal_acceptance':False,
     'fitting_performed':False,'archives':[r[1] for r in sources],'targets':{}}
for target in ('idfs44','cluster'):
    m1,a,s1=data[0][target];m2,b,s2=data[1][target]
    if len(a)!=len(b): raise ValueError('unpaired lengths')
    for x,y in zip(a,b):
        for key in ('window_start_ns','features','target_iops','feature_end_ns','features_available_ns'):
            if x[key]!=y[key]: raise ValueError('changed data or alignment')
    h=m2['training_details']['history']
    worst=sorted((i for i in range(len(b)) if b[i]['target_iops']>0),
                 key=lambda i:abs(b[i]['prediction_iops']-b[i]['target_iops'])/b[i]['target_iops'],reverse=True)[:5]
    result=dict(v1=s1,v2=s2,mae_delta=s2['mae_iops']-s1['mae_iops'],mape_delta=s2['mape_percent']-s1['mape_percent'],
                v2_candidate=m2['candidate_commit'],v2_code_dirty=m2['code_dirty'],
                v2_input_transform=m2['training_details']['input_transform'],
                v2_input_mean=m2['training_details']['input_log_mean'],v2_input_std=m2['training_details']['input_log_std'],
                v2_epochs=len(h),v2_best_epoch=min(h,key=lambda r:r['validation_loss'])['epoch'],
                v1_range=[min(r['prediction_iops'] for r in a),max(r['prediction_iops'] for r in a)],
                v2_range=[min(r['prediction_iops'] for r in b),max(r['prediction_iops'] for r in b)],
                mae_improved_windows=sum(abs(y['prediction_iops']-y['target_iops'])<abs(x['prediction_iops']-x['target_iops']) for x,y in zip(a,b)),
                worst_v2=[dict(window_start_ns=b[i]['window_start_ns'],actual=b[i]['target_iops'],
                               lags=b[i]['features'][:5],v1_prediction=a[i]['prediction_iops'],v2_prediction=b[i]['prediction_iops']) for i in worst])
    out['targets'][target]=result
Path(args.output).write_text(json.dumps(out,indent=2)+'\n',encoding='utf-8')
print(json.dumps(out))
