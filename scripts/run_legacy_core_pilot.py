"""Three fixed legacy-core/current1s adaptations; no load or collection."""
import argparse,copy,hashlib,importlib.util,json,math,os,sys,time,zipfile
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'idfs_forecast_lab'))
from idfs_lab.placement import require_training_host
from idfs_lab.first_round_data import load_fixed_data
from idfs_lab.joint import index_windows
from idfs_lab.target_baselines import metrics
HOSTS=['idfs'+str(i) for i in (44,45,46,47,64,65,66,67)]+['cluster']
ARMS=[('gru_global_mae','gru',False),('gru_window_mae','gru',True),('tcn_window_mae','tcn',True)]


def main():
    p=argparse.ArgumentParser();p.add_argument('--archive',required=True);p.add_argument('--core',required=True);p.add_argument('--output',required=True);a=p.parse_args()
    root=require_training_host();dest=Path(a.output).resolve()
    if root not in dest.parents or dest.exists():raise ValueError('fresh owned output required')
    corepath=Path(a.core)
    expected='9510dfb820d770394a91b4d18d7ac29974a6db7303f64cdc9b17d9111eec2685'
    if hashlib.sha256(corepath.read_bytes()).hexdigest()!=expected:raise ValueError('legacy core source changed')
    import torch
    from idfs_lab.legacy_adapter import LegacyAdapter
    if not torch.cuda.is_available() or torch.cuda.device_count()!=1:raise RuntimeError('one reserved GPU required')
    torch.set_num_threads(4);torch.set_num_interop_threads(1);torch.use_deterministic_algorithms(True);torch.backends.cudnn.benchmark=False
    spec=importlib.util.spec_from_file_location('legacy_models',corepath);old=importlib.util.module_from_spec(spec);sys.modules[spec.name]=old;spec.loader.exec_module(old)
    dest.mkdir();sha,data=load_fixed_data(a.archive)
    with zipfile.ZipFile(a.archive) as z:
        for w,d in data.items():
            prefix=d['source']['dataset_id']+'/';ix=index_windows([json.loads(l) for l in z.read(prefix+'processed/windows.jsonl').splitlines()])
            for part in d['split'].values():
                for r in part:
                    sequences=[[ix[('host_oss',h,r['start']-i*10**9)]['total_iops'] for i in range(16,1,-1)] for h in HOSTS[:8]]
                    sequences.append([sum(s[j] for s in sequences) for j in range(15)])
                    r.update(sequence=sequences,workload=w)
    train=[r for d in data.values() for r in d['split']['train']];dev=[r for d in data.values() for r in d['split']['development']]
    if (len(train),len(dev))!=(1215,697):raise ValueError('sample regression')
    report=dict(experiment_id='legacy-core-current1s-pilot-v1',state='running',code_commit=os.environ['E2E_CODE_SHA'],legacy_core_sha256=expected,
        archive_sha256=sha,source_datasets={w:d['source'] for w,d in data.items()},train_seconds=1215,development_seconds=697,
        formal_acceptance=False,independent_validation=False,time_boundary_review='pending',calibration='none',
        adaptation_differences=['current1s15steps vs old100ms256steps','current targets and split','40epoch pilot vs old150','single seed vs old ens3'],
        seed=2026,threads=4,workers=0,epoch_budget=40,training_loss='MAE on normalized original-scale target',
        early_stop_metric='development WAPE',checkpoint_metrics=['WAPE','positive-only MAPE'],patience=10,
        torch=torch.__version__,cuda=torch.version.cuda,gpu_uuid=os.environ.get('CUDA_VISIBLE_DEVICES'),models={},metrics={})
    predictions={};logs=[]
    for arm,kind,window in ARMS:
        for sel in ('wape','mape'):predictions[arm+'_'+sel]=[]
        for ti,h in enumerate(HOSTS):
            torch.manual_seed(2026);torch.cuda.manual_seed_all(2026)
            x=torch.tensor([[ [v] for v in r['sequence'][ti]] for r in train],dtype=torch.float32,device='cuda')
            y=torch.tensor([r['y'][ti] for r in train],dtype=torch.float32,device='cuda')
            vx=torch.tensor([[[v] for v in r['sequence'][ti]] for r in dev],dtype=torch.float32,device='cuda')
            vy=torch.tensor([r['y'][ti] for r in dev],dtype=torch.float32,device='cuda')
            cy=y.cpu();mean=float(cy.mean());std=float(cy.std(unbiased=False)+1e-8)
            core=old.GRUNet(hidden=128,layers=2,dropout=.1) if kind=='gru' else old.TCNNet(channels=32,levels=6,dropout=.1)
            model=LegacyAdapter(core,window,mean,std).cuda();opt=torch.optim.AdamW(model.parameters(),lr=.001,weight_decay=1e-4)
            sched=torch.optim.lr_scheduler.ReduceLROnPlateau(opt,factor=.5,patience=4)
            best={k:{'metric':float('inf'),'epoch':0,'state':None} for k in ('wape','mape')};hist=[]
            for epoch in range(1,41):
                model.train();order=torch.randperm(len(x),device='cuda');loss_sum=0.
                for start in range(0,len(x),128):
                    ids=order[start:start+128];opt.zero_grad(set_to_none=True);pn,mu,sd=model.normalized(x[ids]);loss=(pn-(y[ids]-mu)/sd).abs().mean()
                    if not torch.isfinite(loss):raise ValueError('nonfinite train loss')
                    loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),5.);opt.step();loss_sum+=float(loss.detach())*len(ids)
                model.eval()
                with torch.inference_mode():
                    pred=model(vx);ae=(pred-vy).abs();vw=float(ae.sum()/vy.sum()*100);positive=vy>0;vm=float((ae[positive]/vy[positive]).mean()*100)
                if not math.isfinite(vw+vm):raise ValueError('nonfinite validation')
                hist.append({'epoch':epoch,'train_loss':loss_sum/len(x),'development_wape_percent':vw,'development_mape_percent':vm});sched.step(vw)
                for sel,value in [('wape',vw),('mape',vm)]:
                    if value<best[sel]['metric']:best[sel]={'metric':value,'epoch':epoch,'state':copy.deepcopy(model.state_dict())}
                if epoch-best['wape']['epoch']>=10:break
            for sel in ('wape','mape'):
                model=model.cuda();model.load_state_dict(best[sel]['state']);model=model.double().eval()
                with torch.inference_mode():native=model(vx.double()).cpu()
                model=model.cpu();cx=vx.cpu().double();export=torch.jit.script(model)
                with torch.inference_mode():cp=export(cx)
                parity=float(((native-cp).abs()/native.abs().clamp_min(1)).max())
                if parity>1e-5 or not math.isfinite(parity):raise ValueError('GPU/CPU parity')
                key=arm+'_'+sel+'_'+h;path=dest/(key+'.ts');export.save(str(path));torch.save(best[sel]['state'],dest/(key+'.pt'))
                with torch.inference_mode():
                    export(cx[:1]);t=time.perf_counter();export(cx[:1]);lat=(time.perf_counter()-t)*1000
                report['models'][key]=dict(history=hist,checkpoint_epoch=best[sel]['epoch'],checkpoint_metric=sel,gpu_cpu_relative_error=parity,
                    weights_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),cpu_batch1_ms=lat,global_mean=mean,global_std=std,window_normalization=window)
                predictions[arm+'_'+sel].append(cp.tolist())
            print(json.dumps({'completed_arm':arm,'target':h}),flush=True)
            (dest/'progress.json').write_text(json.dumps(report,indent=2));del model,opt,sched;torch.cuda.empty_cache()
    for variant,pred in predictions.items():
        report['metrics'][variant]={}
        for w in data:
            ids=[k for k,r in enumerate(dev) if r['workload']==w]
            ms={h:metrics((pred[i][k],dev[k]['y'][i]) for k in ids) for i,h in enumerate(HOSTS)}
            ms['summed_node_cluster']=metrics((sum(pred[i][k] for i in range(8)),dev[k]['y'][8]) for k in ids)
            report['metrics'][variant][w]=ms
        for k,r in enumerate(dev):logs.append(dict(variant=variant,workload=r['workload'],window_start_ns=r['start'],features_available_ns=r['available'],actual=r['y'],raw_predictions=[pred[i][k] for i in range(9)],final_predictions=[pred[i][k] for i in range(9)]))
    (dest/'predictions.json').write_text(json.dumps(logs));report.update(state='completed',predictions_sha256=hashlib.sha256((dest/'predictions.json').read_bytes()).hexdigest())
    (dest/'report.json').write_text(json.dumps(report,indent=2));print('COMPLETED',flush=True)


if __name__=='__main__':main()
