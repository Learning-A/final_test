"""No-fit parity and bounded timing for the pre-registered frozen R1 forests."""
import argparse
import gzip
import hashlib
import json
import os
from pathlib import Path
import pickle
import sys
import time
import zipfile

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'idfs_forecast_lab'))
from idfs_lab.placement import require_training_host
from idfs_lab.fine_paired_dataset import paired_capture, vector
from idfs_lab.fine_inputs import index_fine, build_fine_input, SECOND
from idfs_lab.masked_inputs import build_masked_input
from idfs_lab.joint import index_windows
from idfs_lab.frozen_forests import FrozenForests
from verify_r1_training_core_batch import PINS

HOSTS=['idfs'+str(i) for i in (44,45,46,47,64,65,66,67)]+['cluster']
FROZEN_SHA='76c81d14092b84bd80aaeadd69c477739a4c15df17e17a2c3fc43f1738655eb0'
REPORT_SHA='902011ab459e54c66cfb0350cf6600a98debc0185736d4c645eceb21386b5541'
PREDICTIONS_SHA='2e01f0cac2f049fa98b29f67482aab2cb11c5df6a80d58ef9d116e0a2ba4a0dc'
SCHEMA_INDEX_SHA='f88ec8872529a97f001629a28fdcbcf738806eb96167ac201a3a2037256b73e2'


def sha_bytes(raw):return hashlib.sha256(raw).hexdigest()


def file_sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda:f.read(1024*1024),b''):h.update(block)
    return h.hexdigest()


def require(ok,message):
    if not ok:raise ValueError(message)


def timing_summary(values):
    import numpy as np
    require(len(values)>0 and all(np.isfinite(x) and x>=0 for x in values),'valid nonempty timings required')
    return {name:float(np.percentile(values,q)) for name,q in [('p50_ms',50),('p95_ms',95),('p99_ms',99)]}


def checked_vectors(ix,fi,start,did,floor):
    import numpy as np
    base=[build_masked_input(ix,start,h,HOSTS[:8],did,floor) for h in HOSTS[:8]]
    require(all(x is not None for x in base),'same F0 history required')
    near=[build_fine_input(fi,start,h,did,floor,clock_uncertainty_ns=None,
                          processing_budget_ns=50_000_000) for h in HOSTS[:8]]
    cutoff=start-80_000_000
    require(all(x['features_available_ns']<=cutoff for x in base),'late base features')
    require(all(x['features_available_ns'] is None or x['features_available_ns']<=cutoff for x in near),'late fine features')
    require(not any(x['online_ready'] for x in near),'online evidence must remain unknown')
    features=[b['features']+n['features'] for b,n in zip(base,near)]
    features.append(sum((b['features'][:112] for b in base),[])+base[0]['features'][112:]+
                    sum((n['features'] for n in near),[]))
    return [np.asarray(v,dtype=np.float64) for v in features]


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('frozen','train','original-index','core-index','archive','reference-archive','output'):
        p.add_argument('--'+name,required=True)
    a=p.parse_args();job=require_training_host();dest=Path(a.output).resolve()
    require(job in dest.parents and not dest.exists(),'fresh authorized output required')
    for name,pin in [('train','train'),('original_index','original_index'),('core_index','index'),
                     ('archive','archive'),('reference_archive','reference_archive')]:
        require(file_sha(getattr(a,name))==PINS[pin],'pinned input differs: '+name)
    require(file_sha(a.frozen)==FROZEN_SHA,'frozen package differs')
    import numpy as np
    import sklearn
    started=time.monotonic();models=[]
    with zipfile.ZipFile(a.frozen) as z:
        report_raw=z.read('results/report.json');require(sha_bytes(report_raw)==REPORT_SHA,'frozen report differs')
        frozen=json.loads(report_raw)
        sample_raw=z.read('results/sample_ids.json');require(sha_bytes(sample_raw)==PINS['sample_ids'],'sample keys differ')
        samples=json.loads(sample_raw)
        raw=z.read('results/predictions.jsonl.gz');require(sha_bytes(raw)==PREDICTIONS_SHA,'frozen predictions differ')
        reference={}
        for line in gzip.decompress(raw).splitlines():
            row=json.loads(line)
            if row['margin_ms']==50 and row['arm']=='F0_F1':
                key=(row['dataset_id'],row['window_start_ns'])
                require(key not in reference,'duplicate reference key');reference[key]=row
        raw=z.read('execution/schema-index.json');require(sha_bytes(raw)==SCHEMA_INDEX_SHA,'source schema index differs')
        schema=json.loads(raw)
        for name,expected in schema['schema_modules_sha256'].items():
            require(Path(name).name==name,'unsafe schema name')
            require(file_sha(Path(__file__).resolve().parents[1]/'idfs_forecast_lab/idfs_lab'/name)==expected,'feature implementation differs')
        for i,h in enumerate(HOSTS):
            raw=z.read('results/margin50_'+h+'.pkl')
            require(sha_bytes(raw)==frozen['models']['margin50_'+h]['sha256'],'model bytes differ')
            model=pickle.loads(raw)
            require((model.n_estimators,model.max_depth,model.min_samples_leaf,model.random_state,model.n_features_in_,model.n_outputs_)==
                    (64,8,3,2026,258 if i<8 else 1840,1),'frozen model specification differs')
            model.n_jobs=1;models.append(model)
    expected_keys=[tuple(k) for k in samples['internal_dev']]
    require(len(expected_keys)==len(reference)==2661 and set(expected_keys)==set(reference),'exact internal sample scope required')
    dest.mkdir();t=time.monotonic();runtime=FrozenForests.from_models(models)
    runtime.save(dest/'primary_forest.npz');loaded=FrozenForests.load(dest/'primary_forest.npz')
    export_load_seconds=time.monotonic()-t
    original={d['dataset_id']:d for d in json.loads(Path(a.original_index).read_text())['datasets'] if d['role']=='train'}
    core_index=json.loads(Path(a.core_index).read_text());matrices=[[] for _ in HOSTS];keys=[];graph=None
    by_key={};captures=[]
    with zipfile.ZipFile(a.train) as train,zipfile.ZipFile(a.archive) as core,zipfile.ZipFile(a.reference_archive) as ref:
        for c in core_index['captures']:
            if c['index']!=3:continue
            did=c['dataset_id'];d=original[did]
            require((c['source_role'],c['source_run_id'],c['workload'],c['index'])==('train',d['run_id'],d['workload'],d['index']),'capture identity differs')
            source=ref if c['delivery']['kind']=='existing_release' else core
            prefix='' if source is ref else c['delivery']['directory']+'/'
            raw=source.read(prefix+'export_manifest.json');require(sha_bytes(raw)==c['export_manifest_sha256'],'fine manifest differs')
            manifest=json.loads(raw);raw=source.read(prefix+'fine_context.jsonl.gz')
            pin=next(x for x in manifest['files'] if x['name']=='fine_context.jsonl.gz')
            require(len(raw)==pin['bytes'] and sha_bytes(raw)==pin['sha256'],'fine rows differ')
            fine=[json.loads(line) for line in gzip.decompress(raw).splitlines()]
            windows=[json.loads(line) for line in train.read(did+'/processed/windows.jsonl').splitlines()]
            t0=json.loads(train.read(did+'/recording_start.json'))['epoch_ns']
            pinned=[t for name,t in expected_keys if name==did]
            paired=paired_capture(windows,fine,capture_id=did,run_id=d['run_id'],source_role='train',run_index=3,
                                  workload=d['workload'],hosts=HOSTS[:8],recording_start_ns=t0,
                                  expected_starts=pinned,processing_budget_ns=50_000_000,clock_uncertainty_ns=None)
            for row in paired['samples']:
                key=(did,row['window_start_ns']);require(key not in by_key,'duplicate paired key')
                require(row['actual']==reference[key]['actual'],'original labels differ')
                by_key[key]=row
            captures.append(did)
            if did=='dev_20261122_graph_zipf_train_03_a01':
                graph=(index_windows(windows),index_fine(fine),((t0+SECOND-1)//SECOND)*SECOND,did)
            print(json.dumps({'paired_capture':did,'samples':len(paired['samples'])}),flush=True)
    require(len(captures)==5 and len(by_key)==2661 and graph is not None,'five internal captures required')
    ordered=[by_key[k] for k in expected_keys]
    matrices=[np.asarray([vector(r,i,'F0_F1') for r in ordered]) for i in range(9)]
    expected=np.asarray([reference[k]['raw_predictions'] for k in expected_keys])
    pred=loaded.predict_batch(matrices)
    relative=np.abs(pred-expected)/np.maximum(1,np.abs(expected))
    require(float(relative.max())<=1e-10,'frozen output parity failed')
    parity=dict(sample_count=2661,target_count=9,max_relative_error=float(relative.max()),
                per_target_max_relative_error={h:float(relative[:,i].max()) for i,h in enumerate(HOSTS)})
    (dest/'parity.json').write_text(json.dumps(parity,indent=2)+'\n')
    print(json.dumps({'all_frozen_output_parity':parity}),flush=True)
    ix,fi,floor,did=graph;starts=[t for name,t in expected_keys if name==did][:100]
    def infer(arm,x):
        if arm=='arrays':return loaded.predict_row(x)
        return np.asarray([model.predict(v[None,:])[0] for model,v in zip(models,x)])
    for t in starts[:5]:
        x=checked_vectors(ix,fi,t,did,floor)
        for arm in ('sklearn','arrays'):infer(arm,x)
    timings=[]
    for rnd in range(3):
        for k,t in enumerate(starts):
            order=('sklearn','arrays') if (rnd+k)%2==0 else ('arrays','sklearn')
            for position,arm in enumerate(order):
                begin=time.perf_counter_ns();x=checked_vectors(ix,fi,t,did,floor);mid=time.perf_counter_ns()
                y=infer(arm,x);end=time.perf_counter_ns()
                error=float(np.max(np.abs(y-np.asarray(reference[(did,t)]['raw_predictions']))/
                                   np.maximum(1,np.abs(reference[(did,t)]['raw_predictions']))))
                require(error<=1e-10,'timed prediction parity failed')
                timings.append(dict(round=rnd,position=position,arm=arm,dataset_id=did,window_start_ns=t,
                    feature_ms=(mid-begin)/1e6,model_ms=(end-mid)/1e6,combined_ms=(end-begin)/1e6,
                    max_relative_error=error))
        print(json.dumps({'timing_round_completed':rnd}),flush=True)
    stats={arm:{component:timing_summary([r[component] for r in timings if r['arm']==arm])
                for component in ('feature_ms','model_ms','combined_ms')} for arm in ('sklearn','arrays')}
    per_round={str(rnd):{arm:timing_summary([r['combined_ms'] for r in timings if r['round']==rnd and r['arm']==arm])
                         for arm in ('sklearn','arrays')} for rnd in range(3)}
    with gzip.open(dest/'timings.jsonl.gz','wt',encoding='utf-8') as f:
        for row in timings:f.write(json.dumps(row)+'\n')
    result=dict(state='completed',experiment_id='b19-r1-frozen-inference-v1',code_sha=os.environ['E2E_CODE_SHA'],
        fitting_performed=False,new_models=0,new_accuracy_selections=0,new_external_validation_evaluations=0,
        source_role='train',partition='internal_run03',source_frozen_sha256=FROZEN_SHA,source_report_sha256=REPORT_SHA,
        source_predictions_sha256=PREDICTIONS_SHA,sample_ids_sha256=PINS['sample_ids'],
        source_schema_index_sha256=SCHEMA_INDEX_SHA,
        source_model_sha256={h:frozen['models']['margin50_'+h]['sha256'] for h in HOSTS},
        exported_npz_sha256=file_sha(dest/'primary_forest.npz'),schema_modules_sha256=schema['schema_modules_sha256'],
        export_and_load_seconds=export_load_seconds,parity=parity,timing_samples_per_round=len(starts),rounds=3,
        warmup_samples=5,timing_scope='pre-indexed features plus nine frozen model outputs only',stats=stats,per_round=per_round,
        timings_sha256=file_sha(dest/'timings.jsonl.gz'),cpu_threads=4,workers=0,gpu_used=False,
        python=sys.version.split()[0],numpy=np.__version__,sklearn=sklearn.__version__,budget_seconds=180,
        component_p95_below_50ms=(stats['arrays']['combined_ms']['p95_ms']<50 and
                                 all(per_round[str(i)]['arrays']['p95_ms']<50 for i in range(3))),
        network_receipt_and_publication_included=False,clock_uncertainty_ns=None,online_time_eligibility=False,
        formal_acceptance=False,wall_seconds=time.monotonic()-started)
    (dest/'report.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result,indent=2),flush=True)


if __name__=='__main__':main()
