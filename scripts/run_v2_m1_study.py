"""One registered local-scale/residual model; reuse the exact R2 L0 baseline."""
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

from idfs_lab.placement import require_training_host
from idfs_lab.resource_slots import COORD,OWNERS,run_leased
from idfs_lab.r2_scaling import HOSTS,feature_contract,run_equal_node_weights
from idfs_lab.r2_loss_controls import weighted_loss
from idfs_lab.v2_local_residual import LOCAL_NAMES,LocalResidualMLP,prepare_features
import run_r2_loss_study as r2
from run_v2_i1_study import summary,screen

OWNER='01a203f5-3aa6-7b21-8a1f-8fece4787432'
GPU=OWNERS[OWNER][1]
JOB='b19_v2_m1_local_residual_v1_a01'
BUDGET=1800


def sha(path):
    with Path(path).open('rb') as f:
        return hashlib.file_digest(f,'sha256').hexdigest()


def check(value,message):
    if not value:
        raise ValueError(message)


def configuration():
    return dict(job_id=JOB,source_feature_count=258,descriptor_count=4,node_embedding=4,
        hidden=[64,64],parameters=21345,history_total_columns=[2,5,8,11,14],history_seconds=5,
        base_floor=1.0,scale_absolute_floor=1.0,scale_training_s_fraction_floor=.05,
        output='relu(b+sigma*delta)',epoch0='max(five_history_mean,1)',
        loss='same_R2_L0_scaled_MAE',seed=2026,max_epoch=60,patience=8,batch=128,
        AdamW_lr=.001,AdamW_weight_decay=.0001,gradient_clip=5.,budget_seconds=BUDGET,
        baseline='frozen_R2_L0_no_fit',new_network_fits=1,new_preprocessing_fits=0)


def registration(job):
    cfg=json.loads((job/'registration.json').read_text())
    check(job.name==JOB and cfg['job_id']==JOB and cfg['budget_seconds']==BUDGET,'Registered M1 job required')
    check(cfg['owner_thread']==OWNER and cfg['code_sha']==os.environ['E2E_CODE_SHA'],'Owner/code differs')
    check(cfg['configuration']==configuration(),'M1 configuration differs')
    check(os.environ['CUDA_VISIBLE_DEVICES']==GPU,'B assigned GPU UUID required')
    for name,entry in cfg['inputs'].items():
        check(Path(entry['file']).name==entry['file'],'Unsafe input filename')
        check(sha(job/'input'/entry['file'])==entry['sha256'],'Input pin differs: '+name)
    return cfg


def worker():
    started=time.monotonic()
    job=require_training_host()
    reg=registration(job)
    lease=json.loads((COORD/(JOB+'.json')).read_text())
    check(lease['state']=='running' and lease['pid']==os.getpid() and lease['owner_thread']==OWNER
          and lease['gpu_uuid']==GPU and lease['code_sha']==reg['code_sha'],'Live M1 lease required')
    check(torch.cuda.device_count()==1,'Exactly one assigned GPU required')
    torch.set_num_threads(4)
    torch.set_num_interop_threads(1)
    torch.use_deterministic_algorithms(True)
    device=torch.device('cuda:0')
    dest=job/'results'
    dest.mkdir()
    def inp(name):return job/'input'/reg['inputs'][name]['file']
    old_report=json.loads(inp('r2_report').read_text())
    state=json.loads(inp('preprocessing').read_text())
    check(sha(inp('preprocessing'))==old_report['preprocessing_sha256'],'Frozen R2 preprocessing mismatch')
    check(str(torch.__version__)==old_report['torch'],'Match original torch version')
    keys=json.loads(inp('samples').read_text())
    meta=json.loads(inp('metadata').read_text())
    raw=np.load(inp('arrays'),allow_pickle=False)
    names=feature_contract()['names']
    check(state['feature_contract']==feature_contract(),'Frozen feature contract differs')
    check([names[i] for i in (2,5,8,11,14)]==['lag%d_total'%i for i in range(1,6)],'Historical columns differ')
    fit,dev={},{}
    check(not set(map(tuple,keys['fit'])).intersection(map(tuple,keys['internal_dev'])),'Partition leakage')
    local_diagnostics={}
    for part,n,destination in (('fit',5328,fit),('internal_dev',2661,dev)):
        check(len(keys[part])==len(set(map(tuple,keys[part])))==n,'Fixed sample keys differ')
        check(raw[part+'_features'].shape==(n,8,258) and raw[part+'_actual'].shape==(n,8),'Raw C/target shape differs')
        check(all(meta[k[0]]['role']=='train' and meta[k[0]]['index'] in ((1,2) if part=='fit' else (3,)) for k in keys[part]),'Source role/run split differs')
        check(np.isfinite(raw[part+'_actual']).all() and (raw[part+'_actual']>=0).all(),'Invalid original labels')
        augmented,base,scale=prepare_features(raw[part+'_features'],names,state)
        destination.update(keys=keys[part],actual=raw[part+'_actual'],
            x=torch.from_numpy(augmented.reshape(-1,262).astype(np.float32)).to(device),
            nodes=torch.arange(8,device=device).repeat(n),
            available_ns=raw[part+'_available_ns'])
        check(all(int(t)<=k[1]-80_000_000 for t,k in zip(destination['available_ns'],keys[part])),'Core source cutoff differs')
        check(bool(torch.isfinite(destination['x']).all()),'Nonfinite M1 inputs')
        local_diagnostics[part]=dict(zero_history_rows=int(augmented[:,:,-1].sum()),
            baseline_min=float(base.min()),scale_min=float(scale.min()))
    fit['y']=torch.from_numpy(fit['actual'].reshape(-1).astype(np.float32)).to(device)
    weights=run_equal_node_weights([k[0] for k in fit['keys']])
    check(abs(weights.sum()-1)<1e-10,'Run weights do not sum to one')
    run_weights={did:float(weights[np.array([k[0]==did for k in fit['keys']])].sum()) for did in {k[0] for k in fit['keys']}}
    check(len(run_weights)==10 and all(abs(w-.1)<1e-10 for w in run_weights.values()),'Fitting run balance differs')
    fit['weights']=torch.from_numpy(weights.reshape(-1).astype(np.float32)).to(device)
    dev['workloads']=[meta[k[0]]['workload'] for k in dev['keys']]
    with gzip.open(inp('baseline'),'rt') as f:
        original={}
        for row in map(json.loads,f):
            key=(row['dataset_id'],row['window_start_ns'])
            check(key not in original,'Duplicate R2 baseline')
            original[key]=row
    check(set(original)==set(map(tuple,dev['keys'])),'Baseline key mismatch')
    actual=np.asarray([original[tuple(k)]['actual'] for k in dev['keys']])
    baseline=np.asarray([original[tuple(k)]['final_prediction'] for k in dev['keys']])
    check(np.array_equal(actual,dev['actual']),'Frozen labels differ')
    baseline_summary=summary(np,baseline,actual,dev['workloads'])
    check(abs(baseline_summary['macro_mape_percent']-old_report['arms']['L0']['macro_mape'])<1e-9,'Reused L0 score differs')
    groups=r2.selection_masks(actual,dev['workloads'])
    scales=torch.as_tensor(state['target_scales']['s'],dtype=torch.float32,device=device)
    floors=torch.as_tensor(state['target_scales']['tau'],dtype=torch.float32,device=device)
    model=LocalResidualMLP(state['target_scales']['s']).to(device)
    check(sum(p.numel() for p in model.parameters())==21345,'Parameter count differs')
    orders,order_sha=r2.make_orders(len(fit['x']))
    check(order_sha==old_report['batch_permutations_sha256'],'Batch permutation differs from R2 L0')
    optimizer=torch.optim.AdamW(model.parameters(),lr=.001,weight_decay=.0001)
    output=dest/'M1';output.mkdir()
    epoch0=r2.predict(model,dev['x'],dev['nodes'])
    expected=dev['x'][:,-4].cpu().numpy().reshape(-1,8)*np.array(state['target_scales']['s'],dtype=np.float32)
    check(np.allclose(epoch0,expected,rtol=1e-6,atol=1e-6),'Epoch0 is not the frozen historical base')
    best_epoch,best_score=0,r2.macro_mape(epoch0,groups)
    checkpoint=output/'checkpoint.pt'
    torch.save(r2.state_on_cpu(model),checkpoint)
    logs=[dict(epoch=0,dev_macro_mape=best_score,checkpoint_selected=True,dev_zero_prediction_rows=int((epoch0==0).sum()))]
    r2.write_json(output/'epochs.json',logs)
    metadata=dict(state='partial',job_id=JOB,code_sha=reg['code_sha'],configuration=configuration(),
        inputs=reg['inputs'],python=sys.version.split()[0],numpy=np.__version__,torch=str(torch.__version__),
        feature_names=names+list(LOCAL_NAMES),host_order=list(HOSTS),batch_permutations_sha256=order_sha,
        reused_preprocessing_sha256=sha(inp('preprocessing')),local_diagnostics=local_diagnostics,
        fit_run_weights=run_weights,source_role='train',fit_samples=5328,internal_dev_samples=2661,
        arms={'M0_reused':baseline_summary},baseline_fit_count=0,new_fit_count=1,new_preprocessing_fit_count=0,
        external_validation_evaluations=0,online_time_eligibility=False,clock_uncertainty_ns=None)
    r2.write_json(dest/'report.json',metadata)
    population=len(fit['x'])
    for epoch,order in enumerate(orders,start=1):
        model.train()
        train_loss=0.
        train_zero_prediction_rows=0
        for indices in order.to(device).split(128):
            nodes=fit['nodes'][indices]
            prediction=model(fit['x'][indices],nodes)
            train_zero_prediction_rows+=int((prediction.detach()==0).sum())
            loss,_=weighted_loss('L0',prediction,fit['y'][indices],scales[nodes],floors[nodes],fit['weights'][indices],population_size=population)
            check(bool(torch.isfinite(loss)),'Nonfinite M1 loss')
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(),5.,error_if_nonfinite=True)
            optimizer.step()
            train_loss+=float(loss.detach())*len(indices)/population
        prediction=r2.predict(model,dev['x'],dev['nodes'])
        score=r2.macro_mape(prediction,groups)
        chosen=r2.keep_checkpoint(score,best_score)
        if chosen:
            best_epoch,best_score=epoch,score
            tmp=output/'checkpoint.tmp'
            torch.save(r2.state_on_cpu(model),tmp);tmp.replace(checkpoint)
        logs.append(dict(epoch=epoch,dev_macro_mape=score,checkpoint_selected=chosen,train_loss=train_loss,
            train_zero_prediction_rows=train_zero_prediction_rows,dev_zero_prediction_rows=int((prediction==0).sum())))
        r2.write_json(output/'epochs.json',logs)
        print(json.dumps(logs[-1]),flush=True)
        if epoch-best_epoch>=8:
            break
    model.load_state_dict(torch.load(checkpoint,map_location='cpu',weights_only=True))
    prediction=r2.predict(model,dev['x'],dev['nodes'])
    check(abs(r2.macro_mape(prediction,groups)-best_score)<1e-8,'Restored checkpoint differs')
    predictions=output/'predictions.jsonl.gz'
    r2.save_predictions(predictions,dev['keys'],actual,prediction)
    current=summary(np,prediction,actual,dev['workloads'])
    metadata.update(state='completed',best_epoch=best_epoch,epochs_trained=logs[-1]['epoch'],
        internal_checkpoint_evaluations=len(logs),comparison=screen(current,baseline_summary),
        checkpoint_sha256=sha(checkpoint),predictions_sha256=sha(predictions),epochs_sha256=sha(output/'epochs.json'),
        selected_zero_prediction_rows=int((prediction==0).sum()),
        selected_zero_prediction_positive_actual_rows=int(((prediction==0)&(actual>0)).sum()),
        elapsed_seconds=time.monotonic()-started)
    metadata['arms']['M1']=current
    r2.write_json(dest/'report.json',metadata)
    print('COMPLETED',flush=True)


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--worker',action='store_true')
    parser.add_argument('--configuration',action='store_true')
    args=parser.parse_args()
    if args.configuration:
        print(json.dumps(configuration(),indent=2));return
    if args.worker:
        worker();return
    job=require_training_host()
    reg=registration(job)
    check(not (job/'results').exists(),'Do not repeat completed M1')
    check(os.environ.get('CUBLAS_WORKSPACE_CONFIG')==':4096:8','Deterministic GPU workspace required')
    run_leased([sys.executable,'-B',str(Path(__file__).resolve()),'--worker'],job_id=JOB,owner_thread=OWNER,
        code_sha=reg['code_sha'],data_hashes={k:v['sha256'] for k,v in reg['inputs'].items()},budget_seconds=BUDGET,use_gpu=True)


if __name__=='__main__':
    main()
