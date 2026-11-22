"""CPU replay completed full-target arms after failed pilot; never refit."""
import argparse,hashlib,json,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'idfs_forecast_lab'))
from idfs_lab.placement import require_training_host
from idfs_lab.first_round_data import load_fixed_data
from idfs_lab.joint import index_windows
from idfs_lab.target_baselines import metrics
from run_legacy_core_pilot import HOSTS,ARMS
import zipfile


def main():
    p=argparse.ArgumentParser();p.add_argument('--archive',required=True);p.add_argument('--source',required=True);p.add_argument('--output',required=True);a=p.parse_args()
    root=require_training_host();source=Path(a.source).resolve();dest=Path(a.output).resolve()
    if root not in dest.parents or dest.exists():raise ValueError('fresh owned output required')
    if (source/'report.json').exists():raise ValueError('source already complete; do not recover or duplicate')
    prior=json.loads((source/'progress.json').read_text())
    import torch
    torch.set_num_threads(4);torch.set_num_interop_threads(1)
    sha,data=load_fixed_data(a.archive)
    if sha!=prior['archive_sha256']:raise ValueError('data changed')
    dev=[]
    with zipfile.ZipFile(a.archive) as z:
        for w,d in data.items():
            prefix=d['source']['dataset_id']+'/';ix=index_windows([json.loads(l) for l in z.read(prefix+'processed/windows.jsonl').splitlines()])
            for r in d['split']['development']:
                seq=[[ix[('host_oss',h,r['start']-i*10**9)]['total_iops'] for i in range(16,1,-1)] for h in HOSTS[:8]]
                seq.append([sum(s[j] for s in seq) for j in range(15)])
                dev.append(dict(r,sequence=seq,workload=w))
    result=dict(prior,state='partial_results_recovered',fit_performed=False,source_job_complete=False,metrics={},recovery_source_progress_sha256=hashlib.sha256((source/'progress.json').read_bytes()).hexdigest())
    result['source_exports']={k:v['weights_sha256'] for k,v in prior['models'].items()}
    logs=[];skipped=[]
    with torch.inference_mode():
        for arm,_,_ in ARMS:
            for sel in ('wape','mape'):
                variant=arm+'_'+sel;keys=[variant+'_'+h for h in HOSTS]
                if not all(k in prior['models'] for k in keys):skipped.append(variant);continue
                preds=[]
                for i,key in enumerate(keys):
                    path=source/(key+'.ts')
                    if hashlib.sha256(path.read_bytes()).hexdigest()!=prior['models'][key]['weights_sha256']:raise ValueError('weight checksum')
                    m=torch.jit.load(str(path),map_location='cpu').eval();x=torch.tensor([[[v] for v in r['sequence'][i]] for r in dev],dtype=torch.double)
                    preds.append(m(x).tolist())
                groups={}
                for w in data:
                    ids=[k for k,r in enumerate(dev) if r['workload']==w]
                    groups[w]={h:metrics((preds[i][k],dev[k]['y'][i]) for k in ids) for i,h in enumerate(HOSTS)}
                    groups[w]['summed_node_cluster']=metrics((sum(preds[i][k] for i in range(8)),dev[k]['y'][8]) for k in ids)
                result['metrics'][variant]=groups
                for k,r in enumerate(dev):logs.append(dict(variant=variant,workload=r['workload'],window_start_ns=r['start'],features_available_ns=r['available'],actual=r['y'],raw_predictions=[v[k] for v in preds],final_predictions=[v[k] for v in preds]))
    if not result['metrics']:raise ValueError('no complete full-target arm')
    dest.mkdir();(dest/'predictions.json').write_text(json.dumps(logs));result.update(skipped_incomplete_variants=skipped,predictions_sha256=hashlib.sha256((dest/'predictions.json').read_bytes()).hexdigest())
    (dest/'report.json').write_text(json.dumps(result,indent=2));print(json.dumps({'state':result['state'],'replayed_variants':list(result['metrics']),'skipped':skipped,'fit_performed':False}))


if __name__=='__main__':main()
