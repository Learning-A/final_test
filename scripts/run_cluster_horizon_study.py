"""Two equal-capacity one-shot cluster forecast paths under bounded B GPU leases."""
import argparse
import gzip
import hashlib
import json
import os
from pathlib import Path
import sys
import time

import numpy as np
import torch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'idfs_forecast_lab'))
from idfs_lab.cluster_horizon import ClusterHorizonMLP,HORIZONS,window_means
from idfs_lab.placement import require_training_host
from idfs_lab.resource_slots import COORD,OWNERS,run_leased
import run_r2_loss_study as r2

OWNER='01a203f5-3aa6-7b21-8a1f-8fece4787432'
GPU=OWNERS[OWNER][1]
ARMS=('direct_seconds','fine100ms')


def sha(path):
    with Path(path).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()


def config(arm):
    if arm not in ARMS:raise ValueError('registered arm')
    return dict(arm=arm,job_id='b19_cluster_horizon_'+arm+'_v1_a01',input_features=1840,
        target_steps=10 if arm=='direct_seconds' else 100,target_width_seconds=1 if arm=='direct_seconds' else .1,
        output_unit='IOPS; 100ms prediction is a rate, aggregate ten with mean',
        window_horizons=list(HORIZONS),seed=2026,max_epoch=40,patience=6,batch=128,
        optimizer='AdamW',learning_rate=.001,weight_decay=.0001,gradient_clip=5.,
        encoder=[1840,64,64],decoder=[66,32,1],time_coordinates='midpoint/10,width_seconds',
        output='training_cluster_scale*softplus(h)',epoch0='training_cluster_scale',
        loss='training_run_equal_mean_scaled_MAE_over_all_future_steps',
        checkpoint='equal mean of 5 workloads x 4 cumulative horizon WAPE; tie earlier epoch',
        cpu_threads=4,workers=0,gpu_uuid=GPU,full_card_budget_seconds=1800,
        external_validation_evaluations=0,online_time_eligibility=False)


def registration(job,arm):
    r=json.loads((job/'registration.json').read_text())
    if job.name!=config(arm)['job_id'] or r['configuration']!=config(arm):raise ValueError('job/config mismatch')
    if r['code_sha']!=os.environ['E2E_CODE_SHA'] or r['owner_thread']!=OWNER:raise ValueError('owner/code')
    if os.environ.get('CUDA_VISIBLE_DEVICES')!=GPU:raise ValueError('fixed B GPU')
    if r['budget_seconds']+r['preparation_charged_seconds']!=1800:raise ValueError('full-card budget')
    for name,entry in r['inputs'].items():
        if Path(entry['file']).name!=entry['file'] or sha(job/'input'/entry['file'])!=entry['sha256']:
            raise ValueError('input pin mismatch '+name)
    return r


def inspect(y,p,workloads):
    def metric(actual,pred):
        err=np.abs(pred-actual);pos=actual>0
        return dict(n=len(actual),zero_count=int((~pos).sum()),
            mape_percent=float(100*np.mean(err[pos]/actual[pos])) if pos.any() else None,
            wape_percent=float(100*err.sum()/actual.sum()) if actual.sum()>0 else None,
            mae_iops=float(err.mean()),bias_iops=float((pred-actual).mean()),
            mdape_percent=float(100*np.median(err[pos]/actual[pos])) if pos.any() else None,
            pred20_percent=float(100*np.mean(err<=.2*actual)),
            log_mae=float(np.abs(np.log1p(pred)-np.log1p(actual)).mean()))
    true_window,pred_window=window_means(y),window_means(p)
    groups={};trajectory={}
    for workload in sorted(set(workloads)):
        rows=np.asarray(workloads)==workload
        groups[workload]={str(h):metric(true_window[rows,i],pred_window[rows,i]) for i,h in enumerate(HORIZONS)}
        trajectory[workload]={str(i+1):metric(y[rows,i],p[rows,i]) for i in range(10)}
    macro={str(h):{name:float(np.mean([g[str(h)][name] for g in groups.values()]))
                    for name in ('mape_percent','wape_percent','mae_iops','mdape_percent','pred20_percent','log_mae')}
           for h in HORIZONS}
    return dict(window_groups=groups,window_macro=macro,trajectory_groups=trajectory,
                checkpoint_wape=float(np.mean([g['wape_percent'] for w in groups.values() for g in w.values()])))


def forecast(model,x,steps):
    model.eval()
    with torch.inference_mode():out=torch.cat([model(part) for part in x.split(512)]).cpu().numpy().astype(np.float64)
    if out.shape!=(len(x),steps) or not np.isfinite(out).all() or (out<0).any():raise ValueError('invalid model predictions')
    return out if steps==10 else out.reshape(-1,10,10).mean(axis=-1)


def worker(arm):
    started=time.monotonic();job=require_training_host();reg=registration(job,arm)
    lease=json.loads((COORD/(job.name+'.json')).read_text())
    if lease['state']!='running' or lease['pid']!=os.getpid() or lease['owner_thread']!=OWNER or lease['code_sha']!=reg['code_sha']:
        raise ValueError('live inherited B lease required')
    torch.set_num_threads(4);torch.set_num_interop_threads(1);torch.use_deterministic_algorithms(True)
    if torch.cuda.device_count()!=1:raise ValueError('one visible GPU required')
    device=torch.device('cuda:0');dest=job/'results';dest.mkdir()
    inp=lambda n:job/'input'/reg['inputs'][n]['file']
    raw=np.load(inp('arrays'),allow_pickle=False);keys=json.loads(inp('samples').read_text());meta=json.loads(inp('metadata').read_text())
    receipt=json.loads(inp('receipt').read_text())
    if receipt['preprocessing_fit'] or receipt['model_fit'] or receipt['external_validation_read']:raise ValueError('preparation provenance')
    if set(k[0] for k in keys['fit'])&set(k[0] for k in keys['internal_dev']):raise ValueError('run overlap')
    nfit=len(keys['fit']);ndev=len(keys['internal_dev'])
    for part,indexes in [('fit',(1,2)),('internal_dev',(3,))]:
        if len(keys[part])!=len(set(map(tuple,keys[part]))):raise ValueError('sample duplicates')
        if not all(meta[k[0]]['role']=='train' and meta[k[0]]['index'] in indexes for k in keys[part]):raise ValueError('split identity')
        if not all(int(t)<=k[1]-80_000_000 for t,k in zip(raw[part+'_available_ns'],keys[part])):raise ValueError('input cutoff')
    ids=np.asarray([k[0] for k in keys['fit']]);unique=sorted(set(ids))
    if len(unique)!=10:raise ValueError('ten fitting runs required')
    counts={did:int(np.sum(ids==did)) for did in unique}
    weights=np.asarray([1/(10*counts[did]) for did in ids],dtype=np.float32)
    if abs(float(weights.sum())-1)>1e-6:raise ValueError('run weights')
    target_seconds=raw['fit_seconds'];scale=max(1.,float(np.median(target_seconds[:,0][target_seconds[:,0]>0])))
    x=torch.from_numpy(raw['fit_x']).to(device);devx=torch.from_numpy(raw['internal_dev_x']).to(device)
    steps=config(arm)['target_steps']
    target=target_seconds if steps==10 else raw['fit_counts']*10.
    y=torch.from_numpy(target.astype(np.float32)).to(device);weights=torch.from_numpy(weights).to(device)
    workloads=[meta[k[0]]['workload'] for k in keys['internal_dev']];actual=raw['internal_dev_seconds']
    model=ClusterHorizonMLP(scale,steps).to(device)
    with torch.random.fork_rng(devices=[]):
        generator=torch.Generator().manual_seed(2026);orders=[torch.randperm(nfit,generator=generator) for _ in range(40)]
    order_sha=hashlib.sha256(b''.join(v.numpy().astype('<i8').tobytes() for v in orders)).hexdigest()
    initial=r2.state_on_cpu(model);torch.save(initial,dest/'epoch0.pt')
    init_hash=hashlib.sha256(b''.join(p.detach().cpu().numpy().tobytes() for p in model.parameters())).hexdigest()
    optimizer=torch.optim.AdamW(model.parameters(),lr=.001,weight_decay=.0001)
    prediction=forecast(model,devx,steps);summary=inspect(actual,prediction,workloads)
    best,best_epoch=summary['checkpoint_wape'],0;torch.save(initial,dest/'checkpoint.pt')
    epochs=[dict(epoch=0,checkpoint_wape=best,selected=True)]
    out=dict(state='partial',job_id=job.name,code_sha=reg['code_sha'],configuration=config(arm),inputs=reg['inputs'],
        fit_samples=nfit,internal_dev_samples=ndev,source_role='train',new_fit_count=1,
        training_target_scale=scale,target_scale_source='median positive first-second training cluster label; min1',
        fit_run_counts=counts,training_run_weights={d:.1 for d in unique},
        learned_parameters=sum(p.numel() for p in model.parameters()),initial_parameter_sha256=init_hash,
        batch_permutations_sha256=order_sha,python=sys.version.split()[0],torch=str(torch.__version__),numpy=np.__version__,
        external_validation_evaluations=0,formal_acceptance=False,online_time_eligibility=False,
        clock_uncertainty_ns=None,epoch0_checkpoint_sha256=sha(dest/'epoch0.pt'))
    r2.write_json(dest/'report.json',out);r2.write_json(dest/'epochs.json',epochs)
    for epoch,order in enumerate(orders,1):
        model.train();total=0.
        for indices in order.to(device).split(128):
            pred=model(x[indices]);per_row=torch.abs((pred-y[indices])/scale).mean(dim=1)
            loss=nfit*torch.mean(weights[indices]*per_row)
            if not bool(torch.isfinite(loss)):raise ValueError('nonfinite train loss')
            optimizer.zero_grad(set_to_none=True);loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(),5.,error_if_nonfinite=True);optimizer.step()
            total+=float(loss.detach())*len(indices)/nfit
        prediction=forecast(model,devx,steps);summary=inspect(actual,prediction,workloads)
        score=summary['checkpoint_wape'];selected=score<best
        if selected:
            best,best_epoch=score,epoch;torch.save(r2.state_on_cpu(model),dest/'checkpoint.pt')
        epochs.append(dict(epoch=epoch,checkpoint_wape=score,selected=selected,train_scaled_mae=total))
        r2.write_json(dest/'epochs.json',epochs);print(json.dumps(epochs[-1]),flush=True)
        if epoch-best_epoch>=6:break
    model.load_state_dict(torch.load(dest/'checkpoint.pt',map_location='cpu',weights_only=True))
    prediction=forecast(model,devx,steps);summary=inspect(actual,prediction,workloads)
    if abs(summary['checkpoint_wape']-best)>1e-9:raise ValueError('checkpoint reload score')
    with torch.inference_mode():
        future_rates=torch.cat([model(part) for part in devx.split(512)]).cpu().numpy().astype(np.float64)
    with gzip.open(dest/'predictions.jsonl.gz','wt',encoding='utf-8') as f:
        for i,key in enumerate(keys['internal_dev']):
            row=dict(dataset_id=key[0],window_start_ns=key[1],workload=workloads[i],
                actual_100ms_counts=raw['internal_dev_counts'][i].tolist(),actual_seconds=actual[i].tolist(),
                prediction_native_rates=future_rates[i].tolist(),prediction_seconds=prediction[i].tolist(),
                actual_window_means=window_means(actual[i]).tolist(),prediction_window_means=window_means(prediction[i]).tolist(),
                actual_window_operations=(window_means(actual[i])*np.asarray(HORIZONS)).tolist(),
                prediction_window_operations=(window_means(prediction[i])*np.asarray(HORIZONS)).tolist(),
                features_available_ns=int(raw['internal_dev_available_ns'][i]),source_input_deadline_ns=key[1]-80_000_000)
            f.write(json.dumps(row,allow_nan=False)+'\n')
    # Frozen no-fit reference: persist last complete historical second, supplied by pure prep.
    baseline=np.repeat(raw['internal_dev_last_second'][:,None],10,axis=1)
    out.update(state='completed',summary=summary,persistence_baseline=inspect(actual,baseline,workloads),
        best_epoch=best_epoch,epochs_trained=epochs[-1]['epoch'],internal_checkpoint_evaluations=len(epochs),
        checkpoint_sha256=sha(dest/'checkpoint.pt'),predictions_sha256=sha(dest/'predictions.jsonl.gz'),
        epochs_sha256=sha(dest/'epochs.json'),elapsed_seconds=time.monotonic()-started)
    r2.write_json(dest/'report.json',out);print('COMPLETED',arm,flush=True)


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--arm',choices=ARMS,required=True)
    parser.add_argument('--worker',action='store_true');parser.add_argument('--configuration',action='store_true');a=parser.parse_args()
    if a.configuration:print(json.dumps(config(a.arm),indent=2));return
    if a.worker:worker(a.arm);return
    job=require_training_host();r=registration(job,a.arm)
    if (job/'results').exists():raise ValueError('do not repeat completed or partial job')
    if os.environ.get('CUBLAS_WORKSPACE_CONFIG')!=':4096:8':raise ValueError('deterministic environment')
    run_leased([sys.executable,'-B',str(Path(__file__).resolve()),'--arm',a.arm,'--worker'],job_id=job.name,owner_thread=OWNER,
               code_sha=r['code_sha'],data_hashes={k:v['sha256'] for k,v in r['inputs'].items()},budget_seconds=r['budget_seconds'],use_gpu=True)


if __name__=='__main__':main()
