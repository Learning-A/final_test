"""Fixed four-arm GPU pilot, masked144 data, no collection or formal testing."""
import argparse,copy,hashlib,json,os,sys,time,zipfile,math
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'idfs_forecast_lab'))
from idfs_lab.placement import require_training_host
from idfs_lab.first_round_data import load_fixed_data
from idfs_lab.masked_inputs import build_masked_input,SECOND
from idfs_lab.joint import index_windows
from idfs_lab.target_baselines import metrics

ARMS=[('single_log','single',False),('uniform_log','uniform',False),('moe_log','moe',False),('moe_combo','moe',True)]
HOSTS=['idfs'+str(i) for i in (44,45,46,47,64,65,66,67)]+['cluster']


def data_from(archive):
    sha,strict=load_fixed_data(archive) # verifies ZIP, all manifests/members and source roles
    data={}
    with zipfile.ZipFile(archive) as z:
        for w,d in strict.items():
            identity=d['source'];prefix=identity['dataset_id']+'/'
            rows=[json.loads(l) for l in z.read(prefix+'processed/windows.jsonl').splitlines()]
            ix=index_windows(rows);t0=json.loads(z.read(prefix+'recording_start.json'))['epoch_ns']
            splits={'train':[],'development':[]}
            for start in sorted({r['window_start_ns'] for r in rows if r['scope']=='host_oss' and t0<=r['window_start_ns'] and r['window_end_ns']<=t0+600*SECOND}):
                part=next(((n,lo) for n,lo,hi in [('train',0,360),('development',420,600)] if start-16*SECOND>=t0+lo*SECOND and start+SECOND<=t0+hi*SECOND),None)
                if part is None:continue
                labels=[ix.get(('host_oss',h,start)) for h in HOSTS[:8]]
                if any(r is None or not r['complete'] for r in labels):continue
                floor=((t0+part[1]*SECOND+SECOND-1)//SECOND)*SECOND
                fs=[build_masked_input(ix,start,h,HOSTS[:8],identity['dataset_id']+':'+identity['handoff_manifest_sha256'],floor) for h in HOSTS[:8]]
                if any(f is None for f in fs):continue
                y=[r['total_iops'] for r in labels]
                common=start in {r['start'] for r in d['split'][part[0]]}
                splits[part[0]].append(dict(start=start,x=[f['features'] for f in fs],y=y+[sum(y)],workload=w,
                    available=max(f['features_available_ns'] for f in fs),common=common))
            data[w]={'source':identity,'split':splits}
    return sha,data


def main():
    p=argparse.ArgumentParser();p.add_argument('--archive',required=True);p.add_argument('--output',required=True);a=p.parse_args()
    root=require_training_host();dest=Path(a.output).resolve()
    if root not in dest.parents or dest.exists():raise ValueError('fresh owned output required')
    import torch
    from idfs_lab.masked_moe import MaskedForecast,training_scaler,objective
    if not torch.cuda.is_available() or torch.cuda.device_count()!=1:raise RuntimeError('one reserved idle visible GPU required')
    torch.set_num_threads(4);torch.set_num_interop_threads(1)
    torch.use_deterministic_algorithms(True);torch.backends.cudnn.benchmark=False
    dest.mkdir();sha,data=data_from(a.archive)
    train=[r for d in data.values() for r in d['split']['train']];dev=[r for d in data.values() for r in d['split']['development']]
    if (len(train),len(dev),sum(r['common'] for r in dev))!=(1514,762,697):raise ValueError('fixed sample regression')
    x=torch.tensor([r['x'] for r in train],dtype=torch.float32,device='cuda');y=torch.tensor([r['y'] for r in train],dtype=torch.float32,device='cuda')
    vx=torch.tensor([r['x'] for r in dev],dtype=torch.float32,device='cuda');vy=torch.tensor([r['y'] for r in dev],dtype=torch.float32,device='cuda')
    mean,std=training_scaler(x);ystd=torch.log1p(y).std(0).clamp_min(.25)
    scale=y.median(0).values.clamp_min(1);relfloor=torch.quantile(y,.1,dim=0).clamp_min(1)
    subsets={w:[k for k,r in enumerate(dev) if r['workload']==w] for w in data}
    ids={w:{p:[r['start'] for r in d['split'][p]] for p in ('train','development')} for w,d in data.items()}
    report=dict(experiment_id='masked-multigroup-moe-v1',state='running',archive_sha256=sha,code_commit=os.environ['E2E_CODE_SHA'],
        feature_schema='masked_host_multiscale_v1',feature_count=144,train_seconds=len(train),development_seconds=len(dev),common_development_seconds=697,
        independent_validation=False,formal_acceptance=False,online_30ms_verified=False,time_boundary_review='pending',calibration='none',
        source_datasets={w:d['source'] for w,d in data.items()},seed=2026,threads=4,workers=0,epoch_budget=60,patience=12,
        selection_metric='mean over workload x nine targets of positive-only MAPE',early_stop_metric='same as selection',
        target_scale=scale.tolist(),relative_training_floor=relfloor.tolist(),torch=torch.__version__,cuda=torch.version.cuda,python=sys.version,
        gpu_uuid=os.environ.get('CUDA_VISIBLE_DEVICES'),selection_configurations=4,arms={})
    def selection(pred):
        vals=[]
        for inds in subsets.values():
            truth=vy[inds];positive=truth>0
            vals.append(((((pred[inds]-truth).abs()/truth.clamp_min(1e-12))*positive).sum(0)/positive.sum(0).clamp_min(1)).mean())
        return float(torch.stack(vals).mean()*100)
    logs=[]
    for name,mode,combo in ARMS:
        torch.manual_seed(2026);torch.cuda.manual_seed_all(2026)
        model=MaskedForecast(mean.clone(),std.clone(),ystd.clone(),mode).cuda();opt=torch.optim.Adam(model.parameters(),lr=.001)
        model.eval()
        with torch.inference_mode():best_metric=selection(model(vx))
        best=copy.deepcopy(model.state_dict());best_epoch=0;stale=0;history=[]
        for epoch in range(1,61):
            model.train();order=torch.randperm(len(x),device='cuda');total=0.
            for start in range(0,len(x),128):
                ix=order[start:start+128];opt.zero_grad(set_to_none=True);pred,lower,upper=model.details(x[ix])
                loss=objective(pred,y[ix],lower,upper,scale,relfloor,combo,mode=='moe')
                if not torch.isfinite(loss):raise ValueError('nonfinite training loss')
                loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),5.);opt.step();total+=float(loss.detach())*len(ix)
            model.eval()
            with torch.inference_mode():
                vp,lo,up=model.details(vx);value=selection(vp);vl=float(objective(vp,vy,lo,up,scale,relfloor,combo,mode=='moe'))
            if not math.isfinite(value+vl):raise ValueError('nonfinite validation')
            history.append({'epoch':epoch,'train_loss':total/len(x),'development_loss':vl,'selection_mape_percent':value})
            if value<best_metric:best_metric=value;best=copy.deepcopy(model.state_dict());best_epoch=epoch;stale=0
            else:stale+=1
            if stale>=12:break
        model.load_state_dict(best);model=model.double().eval()
        with torch.inference_mode():native=model(vx.double()).cpu()
        model=model.cpu();exported=torch.jit.script(model);cx=vx.cpu().double()
        with torch.inference_mode():pred,lower,upper=exported.details(cx)
        parity=float(((native-pred).abs()/native.abs().clamp_min(1)).max())
        if not math.isfinite(parity) or parity>1e-5:raise ValueError('GPU/CPU parity')
        artifact=dest/(name+'.ts');exported.save(str(artifact))
        torch.save(best,dest/(name+'_checkpoint.pt'))
        lat=[]
        with torch.inference_mode():
            exported(cx[:1])
            for _ in range(30):
                t=time.perf_counter();exported(cx[:1]);lat.append((time.perf_counter()-t)*1000)
        groups={}
        for scope in ('all_available','common_strict15'):
            groups[scope]={}
            for w in data:
                ix=[k for k,r in enumerate(dev) if r['workload']==w and (scope=='all_available' or r['common'])]
                ms={h:metrics((float(pred[k,i]),dev[k]['y'][i]) for k in ix) for i,h in enumerate(HOSTS)}
                ms['summed_node_cluster']=metrics((float(pred[k,:8].sum()),dev[k]['y'][8]) for k in ix)
                groups[scope][w]=ms
        report['arms'][name]=dict(mode=mode,combined_loss=combo,history=history,checkpoint_epoch=best_epoch,best_selection_mape_percent=best_metric,
            metrics=groups,weights_sha256=hashlib.sha256(artifact.read_bytes()).hexdigest(),checkpoint_sha256=hashlib.sha256((dest/(name+'_checkpoint.pt')).read_bytes()).hexdigest(),gpu_cpu_relative_error=parity,
            cpu_batch1_nine_outputs_ms_median=sorted(lat)[len(lat)//2],cpu_batch1_nine_outputs_ms_p95=sorted(lat)[28],
            lower_gate_mean=lower.mean((0,1)).tolist(),upper_gate_mean=upper.mean((0,1)).tolist(),
            lower_entropy=float(-(lower*lower.clamp_min(1e-12).log()).sum(-1).mean()),upper_entropy=float(-(upper*upper.clamp_min(1e-12).log()).sum(-1).mean()))
        for k,r in enumerate(dev):
            if r['available']>r['start']-30000000:raise ValueError('late input')
            vals=pred[k].tolist();logs.append(dict(arm=name,workload=r['workload'],window_start_ns=r['start'],features_available_ns=r['available'],common_strict15=r['common'],actual=r['y'],raw_predictions=vals,final_predictions=vals))
        (dest/'progress.json').write_text(json.dumps(report,indent=2));print(json.dumps({'arm_completed':name,'best_epoch':best_epoch,'selection_mape_percent':best_metric}),flush=True)
        del model,exported,opt;torch.cuda.empty_cache()
    (dest/'predictions.json').write_text(json.dumps(logs));(dest/'sample_ids.json').write_text(json.dumps(ids,sort_keys=True))
    report.update(state='completed',predictions_sha256=hashlib.sha256((dest/'predictions.json').read_bytes()).hexdigest(),sample_ids_sha256=hashlib.sha256((dest/'sample_ids.json').read_bytes()).hexdigest())
    (dest/'report.json').write_text(json.dumps(report,indent=2));print('COMPLETED',flush=True)


if __name__=='__main__':main()
