"""Bounded CPU component timing on frozen train03 samples, with no fitting."""
import argparse
import gzip
import hashlib
import json
from pathlib import Path
import pickle
import sys
import time
import zipfile

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'idfs_forecast_lab'))
from idfs_lab.fine_inputs import index_fine,build_fine_input,SECOND
from idfs_lab.masked_inputs import build_masked_input
from idfs_lab.joint import index_windows
from idfs_lab.placement import require_training_host


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--job',required=True);p.add_argument('--samples',required=True);p.add_argument('--output',required=True)
    a=p.parse_args();job=Path(a.job).resolve()
    if require_training_host()!=job:raise ValueError('owned authorized job required')
    report=json.loads((job/'results/report.json').read_text())
    hosts=['idfs'+str(i) for i in (44,45,46,47,64,65,66,67)];did='dev_20261122_graph_zipf_train_03_a01'
    samples=json.loads(Path(a.samples).read_text())
    if hashlib.sha256(Path(a.samples).read_bytes()).hexdigest()!=report['baseline_sample_ids_sha256']:raise ValueError('samples differ')
    starts=[t for name,t in samples['internal_dev'] if name==did][:100]
    with zipfile.ZipFile(job/'input/train.zip') as z:
        windows=[json.loads(line) for line in z.read(did+'/processed/windows.jsonl').splitlines()]
        t0=json.loads(z.read(did+'/recording_start.json'))['epoch_ns']
    with zipfile.ZipFile(job/'input/graph-r0.zip') as z:
        fine=[json.loads(line) for line in gzip.decompress(z.read('fine_context.jsonl.gz')).splitlines()]
    ix=index_windows(windows);fi=index_fine(fine);floor=((t0+SECOND-1)//SECOND)*SECOND
    models=[]
    for h in hosts+['cluster']:
        path=job/'results'/('margin50_'+h+'.pkl')
        if hashlib.sha256(path.read_bytes()).hexdigest()!=report['models']['margin50_'+h]['sha256']:raise ValueError('model differs')
        model=pickle.loads(path.read_bytes());model.n_jobs=1;models.append(model)
    import numpy as np
    construction=[];inference=[];total=[]
    for start in starts:
        t=time.perf_counter_ns()
        base=[build_masked_input(ix,start,h,hosts,did,floor) for h in hosts]
        near=[build_fine_input(fi,start,h,did,floor,processing_budget_ns=50_000_000,clock_uncertainty_ns=None) for h in hosts]
        vectors=[np.asarray([b['features']+f['features']]) for b,f in zip(base,near)]
        vectors.append(np.asarray([sum((b['features'][:112] for b in base),[])+base[0]['features'][112:]+
                                   sum((f['features'] for f in near),[])]))
        split=time.perf_counter_ns()
        for model,x in zip(models,vectors):model.predict(x)
        end=time.perf_counter_ns()
        construction.append((split-t)/1e6);inference.append((end-split)/1e6);total.append((end-t)/1e6)
    def stats(v):return {name:float(np.percentile(v,q)) for name,q in [('p50_ms',50),('p95_ms',95),('p99_ms',99)]}
    result=dict(scope='pre-indexed CPU feature construction plus nine sequential model calls; offline only',
        fitting_performed=False,source_role='train',partition='internal_run03',dataset_id=did,sample_count=len(starts),
        model_code_commit=report['code_commit'],script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        input_zip_parsing_and_indexing_included=False,network_arrival_and_publication_included=False,
        cuda_visible_devices='',feature_construction=stats(construction),nine_model_calls=stats(inference),
        combined=stats(total),formal_acceptance=False,online_time_eligibility=False)
    Path(a.output).write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))


if __name__=='__main__':main()
