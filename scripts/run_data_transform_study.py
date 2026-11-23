"""One bounded T1/T2/T3 card per registered b507 GPU lease; T0 stays frozen."""
import argparse
import gzip
import hashlib
import json
import os
from pathlib import Path
import sys
import time

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'idfs_forecast_lab'))
import numpy as np
import torch

from idfs_lab.data_transform_controls import (LogTargetMLP, apply_input_transform,
    fit_input_transform, transform_contract)
from idfs_lab.placement import require_training_host
from idfs_lab.resource_slots import COORD, OWNERS, run_leased
from idfs_lab.r2_loss_controls import R2SharedMLP, weighted_loss
from idfs_lab.r2_scaling import HOSTS, feature_contract, run_equal_node_weights
import run_r2_loss_study as r2
from run_v2_i1_study import summary, screen

OWNER='01a203f5-3aa6-7b21-8a1f-8fece4787432'
GPU=OWNERS[OWNER][1]
BUDGET=1800


def sha(path):
    with Path(path).open('rb') as f:
        return hashlib.file_digest(f,'sha256').hexdigest()


def check(value,message):
    if not value:
        raise ValueError(message)


def configuration(arm):
    check(arm in ('T1','T2','T3'),'fixed arm required')
    return dict(arm=arm, job_id=f'b19_data_transform_{arm.lower()}_v1_a01',
        transform_contract=transform_contract(), method='yeojohnson' if arm=='T3' else 'log1p',
        target='log1p(s)*softplus(h);expm1(z)' if arm=='T2' else 's*softplus(h)',
        target_reference_iops=1., loss='log_MAE' if arm=='T2' else 'R2_L0_scaled_MAE',
        baseline='frozen_R2_L0', backbone_parameters=21089, seed=2026, batch=128,
        max_epoch=60, patience=8, learning_rate=.001, weight_decay=.0001,
        gradient_clip=5., epoch0='original_R2_training_s', overflow='fail_if_z_gt_80_no_clip',
        checkpoint='original_40_group_macro_MAPE_earlier_epoch_on_tie',
        fit_samples=5328, internal_dev_samples=2661, cpu_threads=4, workers=0,
        budget_seconds=BUDGET, gpu_uuid=GPU, external_validation_evaluations=0,
        training_run_weight=.1, training_node_weight=.125)


def registration(job,arm):
    reg=json.loads((job/'registration.json').read_text())
    check(reg['configuration']==configuration(arm),'fixed configuration differs')
    check(job.name==reg['configuration']['job_id'],'job differs')
    check(reg['owner_thread']==OWNER and reg['code_sha']==os.environ['E2E_CODE_SHA'],'owner/code differs')
    check(os.environ.get('CUDA_VISIBLE_DEVICES')==GPU,'B assigned GPU required')
    for name,entry in reg['inputs'].items():
        check(Path(entry['file']).name==entry['file'],'simple filename required')
        check(sha(job/'input'/entry['file'])==entry['sha256'],'input pin differs: '+name)
    return reg


def extra_metrics(pred,actual,workloads):
    groups={}
    for workload in sorted(set(workloads)):
        rows=np.asarray(workloads)==workload
        for node,host in enumerate(HOSTS):
            error=np.log1p(pred[rows,node])-np.log1p(actual[rows,node])
            groups[workload+'/'+host]=dict(log_mae=float(np.abs(error).mean()),
                                         rmsle=float(np.sqrt(np.mean(error**2))))
    return dict(groups=groups,macro_log_mae=float(np.mean([g['log_mae'] for g in groups.values()])),
                macro_rmsle=float(np.mean([g['rmsle'] for g in groups.values()])),
                reduction='each node/workload group first, then equal mean of 40; zeros included')


def worker(arm):
    started=time.monotonic()
    job=require_training_host();reg=registration(job,arm)
    lease=json.loads((COORD/(job.name+'.json')).read_text())
    check(lease['state']=='running' and lease['pid']==os.getpid() and lease['owner_thread']==OWNER
          and lease['code_sha']==reg['code_sha'] and lease['gpu_uuid']==GPU,'live lease required')
    check(torch.cuda.device_count()==1,'one GPU required')
    torch.set_num_threads(4);torch.set_num_interop_threads(1);torch.use_deterministic_algorithms(True)
    device=torch.device('cuda:0');dest=job/'results';dest.mkdir()
    def inp(name):return job/'input'/reg['inputs'][name]['file']
    old=json.loads(inp('r2_report').read_text());r2state=json.loads(inp('preprocessing').read_text())
    check(sha(inp('preprocessing'))==old['preprocessing_sha256'],'R2 state pin mismatch')
    check(str(torch.__version__)==old['torch'] and np.__version__==old['numpy'],'match original numeric environment')
    keys=json.loads(inp('samples').read_text());meta=json.loads(inp('metadata').read_text())
    raw=np.load(inp('arrays'),allow_pickle=False);names=feature_contract()['names']
    check(r2state['feature_contract']==feature_contract(),'C258 contract differs')
    if arm=='T1':
        state=fit_input_transform(raw['fit_features'].reshape(-1,258),names,method='log1p',
                                  source_role='train',partition='fit_run01_02')
    else:
        t1=json.loads(inp('t1_state').read_text())
        check(t1['method']=='log1p' and t1['contract']==transform_contract(),'fixed T1 state required')
        if arm=='T3':
            check(t1['t3_admitted'],'T3 gate not met; do not fit')
            state=fit_input_transform(raw['fit_features'].reshape(-1,258),names,method='yeojohnson',
                                      source_role='train',partition='fit_run01_02')
        else:
            state=t1
    r2.write_json(dest/'transform.json',state)
    fit,dev={},{}
    check(not set(map(tuple,keys['fit'])).intersection(map(tuple,keys['internal_dev'])),'split leakage')
    for part,n,split in [('fit',5328,fit),('internal_dev',2661,dev)]:
        check(len(keys[part])==len(set(map(tuple,keys[part])))==n,'fixed keys mismatch')
        check(raw[part+'_features'].shape==(n,8,258) and raw[part+'_actual'].shape==(n,8),'raw shape mismatch')
        check(all(meta[k[0]]['role']=='train' and meta[k[0]]['index'] in ((1,2) if part=='fit' else (3,)) for k in keys[part]),'source role mismatch')
        check(all(int(t)<=k[1]-80_000_000 for t,k in zip(raw[part+'_available_ns'],keys[part])),'source cutoff mismatch')
        x=apply_input_transform(raw[part+'_features'].reshape(-1,258),names,state).astype(np.float32)
        check(np.isfinite(x).all(),'finite float32 features required')
        split.update(keys=keys[part],actual=raw[part+'_actual'],x=torch.from_numpy(x).to(device),
                     nodes=torch.arange(8,device=device).repeat(n))
    np.savez_compressed(dest/'input_context.npz',internal_dev_preserved=raw['internal_dev_features'][:,:,state['contract']['preserved']],
                        available_ns=raw['internal_dev_available_ns'])
    fit['y']=torch.from_numpy(fit['actual'].reshape(-1).astype(np.float32)).to(device)
    weights=run_equal_node_weights([k[0] for k in fit['keys']])
    fit['weights']=torch.from_numpy(weights.reshape(-1).astype(np.float32)).to(device)
    dev['workloads']=[meta[k[0]]['workload'] for k in dev['keys']]
    with gzip.open(inp('baseline'),'rt') as f:
        baseline_rows=list(map(json.loads,f))
    baseline_by_key={(r['dataset_id'],r['window_start_ns']):r for r in baseline_rows}
    check(len(baseline_by_key)==len(baseline_rows)==len(dev['keys']),'baseline duplicates/count')
    check(set(baseline_by_key)==set(map(tuple,dev['keys'])),'baseline key mismatch')
    baseline_rows=[baseline_by_key[tuple(k)] for k in dev['keys']]
    check(np.array_equal(np.asarray([r['actual'] for r in baseline_rows]),dev['actual']),'baseline labels mismatch')
    baseline=np.asarray([r['final_prediction'] for r in baseline_rows]);base_summary=summary(np,baseline,dev['actual'],dev['workloads'])
    check(abs(base_summary['macro_mape_percent']-old['arms']['L0']['macro_mape'])<1e-9,'T0 mismatch')
    target=r2state['target_scales'];scales=torch.tensor(target['s'],dtype=torch.float32,device=device)
    floors=torch.tensor(target['tau'],dtype=torch.float32,device=device)
    model=(LogTargetMLP if arm=='T2' else R2SharedMLP)(target['s']).to(device)
    check(sum(p.numel() for p in model.parameters())==21089,'network capacity differs')
    init=r2.state_on_cpu(model);epoch0=r2.predict(model,dev['x'],dev['nodes'])
    check(np.allclose(epoch0,np.tile(target['s'],(2661,1)),rtol=1e-6,atol=1e-6),'epoch0 differs from original s')
    orders,order_sha=r2.make_orders(len(fit['x']))
    check(order_sha==old['batch_permutations_sha256'],'R2 batch order differs')
    def loss(kind,prediction,actual,scale,floor,batch_weights,*,population_size):
        if kind=='T2':
            value=population_size*torch.mean(batch_weights*torch.abs(torch.log1p(prediction)-torch.log1p(actual)))
            return value,{'log_mae':value}
        return weighted_loss('L0',prediction,actual,scale,floor,batch_weights,population_size=population_size)
    r2.weighted_loss=loss
    info=dict(state='partial',job_id=job.name,code_sha=reg['code_sha'],configuration=configuration(arm),inputs=reg['inputs'],
        transform_sha256=sha(dest/'transform.json'),python=sys.version.split()[0],torch=str(torch.__version__),numpy=np.__version__,
        fit_samples=5328,internal_dev_samples=2661,baseline_fit_count=0,new_fit_count=1,
        epoch0_max_abs_error=float(np.max(np.abs(epoch0-np.asarray(target['s'])))),
        batch_permutations_sha256=order_sha,source_role='train',external_validation_evaluations=0,
        formal_acceptance=False,online_time_eligibility=False,clock_uncertainty_ns=None,
        arms={'T0_reused':base_summary},auxiliary={'T0_reused':extra_metrics(baseline,dev['actual'],dev['workloads'])})
    r2.write_json(dest/'report.json',info)
    result=r2.train_arm(arm,model,init,hashlib.sha256(epoch0.tobytes()).hexdigest(),orders,fit,dev,scales,floors,
                        r2.selection_masks(dev['actual'],dev['workloads']),dest)
    prediction=r2.predict(model,dev['x'],dev['nodes'])
    with gzip.open(dest/arm/'predictions.jsonl.gz','rt') as f:
        rows=list(map(json.loads,f))
    with gzip.open(dest/arm/'log_predictions.jsonl.gz','wt',encoding='utf-8') as f:
        for row,actual,pred in zip(rows,dev['actual'],prediction):
            row.update(target_z=np.log1p(actual).tolist(),prediction_z=np.log1p(pred).tolist())
            f.write(json.dumps(row,allow_nan=False)+'\n')
    current=summary(np,prediction,dev['actual'],dev['workloads'])
    info['arms'][arm]=current;info['auxiliary'][arm]=extra_metrics(prediction,dev['actual'],dev['workloads'])
    info.update(state='completed',comparison=screen(current,base_summary),training=result,
                log_predictions_sha256=sha(dest/arm/'log_predictions.jsonl.gz'),
                input_context_sha256=sha(dest/'input_context.npz'),elapsed_seconds=time.monotonic()-started)
    r2.write_json(dest/'report.json',info)
    print('COMPLETED',arm,flush=True)


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--arm',required=True,choices=('T1','T2','T3'))
    parser.add_argument('--worker',action='store_true');parser.add_argument('--configuration',action='store_true')
    args=parser.parse_args()
    if args.configuration:
        print(json.dumps(configuration(args.arm),indent=2));return
    if args.worker:
        worker(args.arm);return
    job=require_training_host();reg=registration(job,args.arm)
    check(not (job/'results').exists(),'new card directory required; no repeats')
    check(os.environ.get('CUBLAS_WORKSPACE_CONFIG')==':4096:8','deterministic GPU config required')
    run_leased([sys.executable,'-B',str(Path(__file__).resolve()),'--arm',args.arm,'--worker'],job_id=job.name,
               owner_thread=OWNER,code_sha=reg['code_sha'],data_hashes={k:v['sha256'] for k,v in reg['inputs'].items()},
               budget_seconds=BUDGET,use_gpu=True)


if __name__=='__main__':
    main()
