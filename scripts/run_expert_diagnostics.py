"""CPU inference of frozen MoE checkpoints; no fitting or new selection."""
import argparse,hashlib,json,sys,os
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'idfs_forecast_lab'))
from idfs_lab.placement import require_training_host
from idfs_lab.target_baselines import metrics
from run_masked_moe_study import data_from,HOSTS


def main():
    p=argparse.ArgumentParser();p.add_argument('--archive',required=True);p.add_argument('--models',required=True);p.add_argument('--output',required=True);a=p.parse_args()
    root=require_training_host();dest=Path(a.output).resolve()
    if root not in dest.parents or dest.exists():raise ValueError('fresh owned output required')
    dest.mkdir()
    import torch
    from idfs_lab.masked_moe import MaskedForecast
    from idfs_lab.expert_diagnostics import routed_prediction
    torch.set_num_threads(4);torch.set_num_interop_threads(1)
    source=Path(a.models);prior=json.loads((source/'report.json').read_text())
    sha,data=data_from(a.archive);dev=[r for d in data.values() for r in d['split']['development']]
    if len(dev)!=762:raise ValueError('sample regression')
    x=torch.tensor([r['x'] for r in dev],dtype=torch.double)
    report=dict(experiment_id='frozen-moe-group-diagnostics-v1',code_commit=os.environ['E2E_CODE_SHA'],archive_sha256=sha,
        state='running',fit_performed=False,new_model_selection=False,formal_acceptance=False,independent_validation=False,
        limitation='fixed-weight route removal and renormalization, not refitted ablation or causal feature importance',arms={})
    logs=[]
    for name in ('moe_log','moe_combo'):
        meta=prior['arms'][name];checkpoint=source/(name+'_checkpoint.pt');artifact=source/(name+'.ts')
        if hashlib.sha256(checkpoint.read_bytes()).hexdigest()!=meta['checkpoint_sha256'] or hashlib.sha256(artifact.read_bytes()).hexdigest()!=meta['weights_sha256']:raise ValueError('weight checksum')
        state=torch.load(checkpoint,map_location='cpu',weights_only=True)
        m=MaskedForecast(state['mean'],state['std'],state['target_log_std'],'moe');m.load_state_dict(state);m=m.double().eval()
        exported=torch.jit.load(str(artifact)).eval()
        report['arms'][name]={'checkpoint_sha256':meta['checkpoint_sha256'],'variants':{}}
        for removed in (-1,0,1,2):
            with torch.inference_mode():pred,weights=routed_prediction(m,x,removed)
            if removed==-1:
                with torch.inference_mode():reference=exported(x)
                parity=float(((pred-reference).abs()/reference.abs().clamp_min(1)).max())
                if parity>1e-10:raise ValueError('native/helper/export regression')
                report['arms'][name]['reference_cpu_parity']=parity
            tag=('full','without_short','without_long','without_spatial')[removed+1];groups={}
            for scope in ('all_available','common_strict15'):
                groups[scope]={}
                for w in data:
                    ix=[k for k,r in enumerate(dev) if r['workload']==w and (scope=='all_available' or r['common'])]
                    ms={h:metrics((float(pred[k,i]),dev[k]['y'][i]) for k in ix) for i,h in enumerate(HOSTS)}
                    ms['summed_node_cluster']=metrics((float(pred[k,:8].sum()),dev[k]['y'][8]) for k in ix)
                    groups[scope][w]=ms
            report['arms'][name]['variants'][tag]=groups
            for k,r in enumerate(dev):logs.append(dict(model=name,variant=tag,workload=r['workload'],window_start_ns=r['start'],features_available_ns=r['available'],common_strict15=r['common'],prediction=pred[k].tolist(),actual=r['y']))
    (dest/'predictions.json').write_text(json.dumps(logs));report.update(state='completed',predictions_sha256=hashlib.sha256((dest/'predictions.json').read_bytes()).hexdigest())
    (dest/'report.json').write_text(json.dumps(report,indent=2));print(json.dumps({'state':'completed','fit_performed':False,'variants':8,'development_seconds':len(dev)}))


if __name__=='__main__':main()
