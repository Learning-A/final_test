"""Fixed R1 source-clock sensitivity experiment; all fitting is b507-only."""
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

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'idfs_forecast_lab'))
from idfs_lab.placement import require_training_host
from idfs_lab.fine_paired_dataset import paired_capture, vector
from idfs_lab.target_baselines import metrics
from verify_r1_training_core_batch import PINS

HOSTS = ['idfs'+str(i) for i in (44,45,46,47,64,65,66,67)] + ['cluster']
MARGINS = (50,100,200)
BASELINE_SHA = '16b13ccd4a0181edc2b266fe531c983c0eb4f28136991180683bf177ae63422b'


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda: f.read(1024*1024), b''):
            h.update(chunk)
    return h.hexdigest()


def require(ok, message):
    if not ok:
        raise ValueError(message)


def group_metrics(predicted, actual):
    out = metrics(zip(predicted, actual))
    bands = {}
    for name, lo, hi in (('positive_le_10',0,10),('positive_10_to_100',10,100),
                         ('positive_100_to_1000',100,1000),('above_1000',1000,float('inf'))):
        pairs = [(float(p),float(y)) for p,y in zip(predicted,actual) if lo < y <= hi]
        bands[name] = dict(n=len(pairs), ape_percent_sum=sum(100*abs(p-y)/y for p,y in pairs))
    out['positive_bands'] = bands
    return out


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ('train','original-index','core-index','archive','reference-archive','baseline','output'):
        p.add_argument('--'+name, required=True)
    a = p.parse_args()
    job = require_training_host()
    dest = Path(a.output).resolve()
    require(job in dest.parents and not dest.exists(), 'fresh owned output required')
    for name, pin in (('train','train'),('original_index','original_index'),
                      ('core_index','index'),('archive','archive'),('reference_archive','reference_archive')):
        require(sha(getattr(a,name)) == PINS[pin], 'pinned '+name+' differs')
    baseline = Path(a.baseline).resolve()
    require(sha(baseline/'report.json') == BASELINE_SHA, 'frozen P2 report differs')
    old_report = json.loads((baseline/'report.json').read_text())
    require(old_report['state'] == 'completed', 'baseline not complete')
    require(sha(baseline/'predictions.json') == old_report['predictions_sha256'], 'baseline predictions differ')
    require(sha(baseline/'sample_ids.json') == PINS['sample_ids'], 'sample ids differ')
    samples = json.loads((baseline/'sample_ids.json').read_text())
    old_predictions = {(r['dataset_id'],r['window_start_ns']):r for r in
                       json.loads((baseline/'predictions.json').read_text()) if r['arm']=='base144'}
    ci = json.loads(Path(a.core_index).read_text())
    originals = {d['dataset_id']:d for d in json.loads(Path(a.original_index).read_text())['datasets'] if d['role']=='train'}
    require(len(ci['captures']) == len(originals) == 15, 'exact fifteen train captures required')
    require({c['dataset_id'] for c in ci['captures']} == set(originals), 'capture identity differs')
    require(len(samples['fit']) == 5328 and len(samples['internal_dev']) == 2661, 'fixed partition differs')
    import numpy as np
    import sklearn
    from sklearn.ensemble import ExtraTreesRegressor
    models = []
    for h in HOSTS:
        file = baseline / ('base144_'+h+'.pkl')
        require(sha(file) == old_report['models']['base144_'+h]['sha256'], 'frozen model differs')
        models.append(pickle.loads(file.read_bytes()))
    dest.mkdir()
    report = dict(experiment_id='b19-r1-causal-inputs-offline-v1',state='running',
                  code_commit=os.environ['E2E_CODE_SHA'],source_role='train',
                  fit_seconds=5328,internal_dev_seconds=2661,external_validation_capture_count_used=0,
                  primary_margin_ms=50,sensitivity_margins_ms=[100,200],
                  clock_uncertainty_ns=None,online_time_eligibility=False,formal_acceptance=False,
                  seed=2026,threads=4,workers=0,sklearn=sklearn.__version__,
                  numpy=np.__version__,python=sys.version.split()[0],loss='raw squared error',
                  calibration='none',budget_seconds=1800,
                  data_hashes={k:sha(getattr(a,k)) for k in ('train','original_index','core_index','archive','reference_archive')},
                  baseline_report_sha256=BASELINE_SHA,baseline_sample_ids_sha256=PINS['sample_ids'],
                  margins={},models={},limitations=[
                      'Source-clock margins are hypothetical sensitivity budgets, not measured clock/arrival bounds.',
                      'No model-service arrival proof, live 30ms publication or formal acceptance.',
                      'All three variants are reported; only 50ms is primary, no best-margin selection.',
                      'Five train03 captures are reused internal development; external validation is untouched.'])
    all_logs=[]
    start_wall=time.monotonic()
    for margin in MARGINS:
        fit=[];dev=[];coverage={}
        with zipfile.ZipFile(a.train) as train, zipfile.ZipFile(a.archive) as core, zipfile.ZipFile(a.reference_archive) as ref:
            for c in ci['captures']:
                did=c['dataset_id'];d=originals[did]
                require((c['source_run_id'],c['source_role'],c['index'],c['workload']) ==
                        (d['run_id'],'train',d['index'],d['workload']), 'role or identity differs')
                require(c['clock_uncertainty_ns'] is None and c['online_receive_proof'] is False,
                        'source timing qualification differs')
                source = ref if c['delivery']['kind']=='existing_release' else core
                prefix = '' if source is ref else c['delivery']['directory']+'/'
                manifest_raw=source.read(prefix+'export_manifest.json')
                require(hashlib.sha256(manifest_raw).hexdigest()==c['export_manifest_sha256'], 'export manifest differs')
                manifest=json.loads(manifest_raw)
                fine_raw=source.read(prefix+'fine_context.jsonl.gz')
                member=next(f for f in manifest['files'] if f['name']=='fine_context.jsonl.gz')
                require(len(fine_raw)==member['bytes'] and hashlib.sha256(fine_raw).hexdigest()==member['sha256'], 'fine payload differs')
                original_raw=train.read(did+'/handoff_manifest.json')
                require(hashlib.sha256(original_raw).hexdigest()==d['manifest_sha256']==c['original_handoff_manifest_sha256'], 'original identity differs')
                original_manifest=json.loads(original_raw)
                require(original_manifest['role']=='train' and not original_manifest['fault_affected'], 'source role/fault differs')
                window_raw=train.read(did+'/processed/windows.jsonl')
                window_sha=next(f['sha256'] for f in original_manifest['files'] if f['path']=='processed/windows.jsonl')
                require(hashlib.sha256(window_raw).hexdigest()==window_sha,'original windows differ')
                windows=[json.loads(line) for line in window_raw.splitlines()]
                fine=[json.loads(line) for line in gzip.decompress(fine_raw).splitlines()]
                t0=json.loads(train.read(did+'/recording_start.json'))['epoch_ns']
                part='internal_dev' if d['index']==3 else 'fit'
                pinned=[t for name,t in samples[part] if name==did]
                paired=paired_capture(windows,fine,capture_id=did,run_id=d['run_id'],source_role='train',
                    run_index=d['index'],workload=d['workload'],hosts=HOSTS[:8],recording_start_ns=t0,
                    expected_starts=pinned,processing_budget_ns=margin*1_000_000,clock_uncertainty_ns=None)
                rows=paired['samples'];(dev if part=='internal_dev' else fit).extend(rows)
                coverage[did]=dict(seconds=len(rows),rejected=paired['rejected'],
                                   observed_fine_slots=sum(sum(r['fine_observed']) for r in rows),
                                   possible_fine_slots=len(rows)*8*20,
                                   empty_fine_node_windows=sum(n==0 for r in rows for n in r['fine_observed']))
                print(json.dumps(dict(margin_ms=margin,capture=did,paired_seconds=len(rows))),flush=True)
        by_key=lambda rs: {(r['dataset_id'],r['window_start_ns']):r for r in rs}
        fit_map=by_key(fit);dev_map=by_key(dev)
        require(len(fit_map)==5328 and len(dev_map)==2661, 'sample count differs')
        fit=[fit_map[tuple(k)] for k in samples['fit']];dev=[dev_map[tuple(k)] for k in samples['internal_dev']]
        for r in dev:
            require(r['actual']==old_predictions[(r['dataset_id'],r['window_start_ns'])]['actual'], 'baseline label differs')
        ys=np.asarray([r['actual'] for r in fit]);vys=np.asarray([r['actual'] for r in dev])
        arm_predictions={'F0':[],'F0_F1':[]};timings=[]
        for ti,h in enumerate(HOSTS):
            vx=np.asarray([vector(r,ti,'F0') for r in dev])
            frozen=models[ti];frozen.n_jobs=1
            pred=frozen.predict(vx)
            expected=np.asarray([old_predictions[(r['dataset_id'],r['window_start_ns'])]['raw_predictions'][ti] for r in dev])
            relative=float(np.max(np.abs(pred-expected)/np.maximum(1,np.abs(expected))))
            require(relative<=1e-10, 'F0 frozen prediction mismatch')
            arm_predictions['F0'].append(pred)
            x=np.asarray([vector(r,ti,'F0_F1') for r in fit]);vx=np.asarray([vector(r,ti,'F0_F1') for r in dev])
            model=ExtraTreesRegressor(n_estimators=64,max_depth=8,min_samples_leaf=3,random_state=2026,n_jobs=4)
            t=time.monotonic();model.fit(x,ys[:,ti]);wall=time.monotonic()-t;model.n_jobs=1
            file=dest/('margin%d_%s.pkl'%(margin,h));file.write_bytes(pickle.dumps(model,protocol=4))
            loaded=pickle.loads(file.read_bytes());raw=model.predict(vx);pred=loaded.predict(vx)
            parity=float(np.max(np.abs(raw-pred)/np.maximum(1,np.abs(raw))))
            require(parity<=1e-10,'CPU serialization mismatch')
            arm_predictions['F0_F1'].append(pred)
            lat=[]
            for k in range(min(100,len(vx))):
                t=time.perf_counter_ns();loaded.predict(vx[k:k+1]);lat.append((time.perf_counter_ns()-t)/1e6)
            report['models']['margin%d_%s'%(margin,h)]=dict(sha256=sha(file),feature_count=x.shape[1],
                fit_seconds=wall,train_mse=float(np.mean((model.predict(x)-ys[:,ti])**2)),
                cpu_serialization_relative_error=parity,frozen_F0_prediction_relative_error=relative,
                predict_only_ms_p50=float(np.percentile(lat,50)),predict_only_ms_p95=float(np.percentile(lat,95)))
            timings.append(lat)
            print(json.dumps(dict(margin_ms=margin,model_completed=h,fit_seconds=wall)),flush=True)
        mr=dict(coverage=coverage,arms={})
        for arm,ps in arm_predictions.items():
            matrix=np.asarray(ps).T
            scores={}
            for workload in sorted({r['workload'] for r in dev}):
                ids=[k for k,r in enumerate(dev) if r['workload']==workload]
                scores[workload]={h:group_metrics(matrix[ids,i],vys[ids,i]) for i,h in enumerate(HOSTS)}
                scores[workload]['summed_node_cluster']=group_metrics(np.sum(matrix[ids,:8],axis=1),vys[ids,8])
            mr['arms'][arm]=scores
            for k,r in enumerate(dev):
                all_logs.append({name:r[name] for name in ('dataset_id','run_id','workload','source_role','development_partition',
                    'window_start_ns','features_available_upper_ns','feature_deadline_ns','publication_deadline_ns','fine_observed')} |
                    dict(margin_ms=margin,arm=arm,actual=r['actual'],raw_predictions=matrix[k].tolist(),
                         final_predictions=matrix[k].tolist(),online_time_eligibility=False))
        # Sequential latency accounting for the actual nine model calls, with
        # feature construction and upstream receipt deliberately excluded.
        summed=np.sum(np.asarray(timings),axis=0)
        mr['nine_models_predict_only_ms_p50']=float(np.percentile(summed,50))
        mr['nine_models_predict_only_ms_p95']=float(np.percentile(summed,95))
        base_scores=mr['arms']['F0'];near_scores=mr['arms']['F0_F1']
        b=[base_scores[w][h]['mape_percent'] for w in base_scores for h in HOSTS[:8]]
        n=[near_scores[w][h]['mape_percent'] for w in near_scores for h in HOSTS[:8]]
        mae_ratios=[sum(near_scores[w][h]['mae_iops'] for h in HOSTS[:8]) /
                    sum(base_scores[w][h]['mae_iops'] for h in HOSTS[:8]) for w in base_scores]
        mr['summary']=dict(node_macro_mape_percent=float(np.mean(n)),F0_node_macro_mape_percent=float(np.mean(b)),
            worst_node_group_mape_percent=max(n),F0_worst_node_group_mape_percent=max(b),
            node_groups_strictly_over_90_accuracy=sum(x<10 for x in n),
            macro_relative_improvement=1-float(np.mean(n))/float(np.mean(b)),
            worst_group_ratio=max(n)/max(b),max_workload_node_macro_mae_ratio=max(mae_ratios))
        mr['passes_prespecified_screen']=(mr['summary']['macro_relative_improvement']>=.05 and
            mr['summary']['worst_group_ratio']<=1.05 and max(mae_ratios)<=1.05)
        report['margins'][str(margin)]=mr
        (dest/'progress.json').write_text(json.dumps(report,indent=2))
        print(json.dumps(dict(margin_completed=margin,summary=mr['summary'])),flush=True)
    pred_path=dest/'predictions.jsonl.gz'
    with gzip.open(pred_path,'wt',encoding='utf-8') as f:
        for row in all_logs:f.write(json.dumps(row)+'\n')
    report.update(state='completed',wall_seconds=time.monotonic()-start_wall,predictions_sha256=sha(pred_path),
                  primary_passes_screen=report['margins']['50']['passes_prespecified_screen'],new_model_count=27)
    (dest/'report.json').write_text(json.dumps(report,indent=2)+'\n')
    print('COMPLETED',flush=True)


if __name__=='__main__':
    main()
