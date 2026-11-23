"""One frozen-model development-round evaluation, no training/selection."""
import argparse,hashlib,json,os,sys,zipfile,time
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'idfs_forecast_lab'))
from idfs_lab.placement import require_training_host
from idfs_lab.multiscale_inputs import build_input,SECOND
from idfs_lab.masked_inputs import build_masked_input
from idfs_lab.joint import index_windows
from idfs_lab.target_baselines import metrics
HOSTS=['idfs'+str(i) for i in (44,45,46,47,64,65,66,67)]+['cluster']


def main():
    p=argparse.ArgumentParser();p.add_argument('--index',required=True);p.add_argument('--validation',required=True);p.add_argument('--gru-source',required=True);p.add_argument('--moe-source',required=True);p.add_argument('--output',required=True);a=p.parse_args()
    root=require_training_host();dest=Path(a.output).resolve()
    if root not in dest.parents or dest.exists():raise ValueError('fresh owned output required')
    if hashlib.sha256(Path(a.index).read_bytes()).hexdigest()!='52e0bf86ff62079e6104a78b58126f59d0f63210ecb405d86b1614d98e92fd2d':raise ValueError('index checksum')
    if hashlib.sha256(Path(a.validation).read_bytes()).hexdigest()!='0c7bcbf682db0b895569211966eed82cf37bd77663c126ab29269aae8cb05ecf':raise ValueError('validation checksum')
    idx=json.loads(Path(a.index).read_text());ds=[d for d in idx['datasets'] if d['role']=='validation']
    if len(ds)!=5 or len({d['workload'] for d in ds})!=5:raise ValueError('five heldout workloads required')
    if {d['run_id'] for d in ds}&{d['run_id'] for d in idx['datasets'] if d['role']=='train'}:raise ValueError('run overlap')
    import torch
    torch.set_num_threads(4);torch.set_num_interop_threads(1)
    gru_source=Path(a.gru_source);moe_source=Path(a.moe_source)
    gm=json.loads((gru_source/'progress.json').read_text());mm=json.loads((moe_source/'report.json').read_text())
    models=[];weight_hashes={}
    for h in HOSTS:
        key='gru_global_mae_mape_'+h;path=gru_source/(key+'.ts');sha=hashlib.sha256(path.read_bytes()).hexdigest()
        if sha!=gm['models'][key]['weights_sha256']:raise ValueError('frozen GRU checksum')
        weight_hashes[key]=sha;models.append(torch.jit.load(str(path),map_location='cpu').eval())
    path=moe_source/'moe_combo.ts';sha=hashlib.sha256(path.read_bytes()).hexdigest()
    if sha!=mm['arms']['moe_combo']['weights_sha256']:raise ValueError('frozen MoE checksum')
    weight_hashes['moe_combo']=sha;moe=torch.jit.load(str(path),map_location='cpu').eval()
    report=dict(experiment_id='frozen-first-independent-rounds-v1',state='running',fit_performed=False,checkpoint_selection_performed=False,
        independent_capture_roles=True,formal_acceptance=False,time_boundary_review='pending',online_30ms_verified=False,
        code_commit=os.environ['E2E_CODE_SHA'],frozen_weight_hashes=weight_hashes,validation_use_count=1,
        limitation='development validation only; sources trained on first-round partial time spans, not all15train runs',datasets={},metrics={})
    logs=[]
    with zipfile.ZipFile(a.validation) as z,torch.inference_mode():
        for d in ds:
            prefix=d['dataset_id']+'/';raw=z.read(prefix+'handoff_manifest.json')
            if hashlib.sha256(raw).hexdigest()!=d['manifest_sha256']:raise ValueError('manifest checksum')
            manifest=json.loads(raw)
            if manifest['role']!='validation' or manifest['fault_affected'] or manifest['source_run_id']!=d['run_id']:raise ValueError('role/fault mismatch')
            for f in manifest['files']:
                raw=z.read(prefix+f['path'])
                if len(raw)!=f['bytes'] or hashlib.sha256(raw).hexdigest()!=f['sha256']:raise ValueError('member checksum')
            t0=json.loads(z.read(prefix+'recording_start.json'))['epoch_ns'];floor=((t0+SECOND-1)//SECOND)*SECOND
            rows=[json.loads(l) for l in z.read(prefix+'processed/windows.jsonl').splitlines()];ix=index_windows(rows)
            samples=[];reject={'candidate_seconds':0,'warmup':0,'incomplete_label':0,'missing_or_late_15s':0}
            starts=sorted({r['window_start_ns'] for r in rows if r['scope']=='host_oss' and t0<=r['window_start_ns'] and r['window_end_ns']<=t0+600*SECOND})
            for start in starts:
                reject['candidate_seconds']+=1
                if start-16*SECOND<floor:reject['warmup']+=1;continue
                labels=[ix.get(('host_oss',h,start)) for h in HOSTS[:8]]
                if any(r is None or not r['complete'] for r in labels):reject['incomplete_label']+=1;continue
                strict=[build_input(ix,start,h,HOSTS[:8],d['dataset_id'],scales=(5,15)) for h in HOSTS[:8]]
                if any(f is None for f in strict):reject['missing_or_late_15s']+=1;continue
                fs=[build_masked_input(ix,start,h,HOSTS[:8],d['dataset_id'],floor) for h in HOSTS[:8]]
                if any(f is None for f in fs):raise ValueError('strict must imply masked')
                seq=[[ix[('host_oss',h,start-i*SECOND)]['total_iops'] for i in range(16,1,-1)] for h in HOSTS[:8]]
                seq.append([sum(s[k] for s in seq) for k in range(15)])
                y=[r['total_iops'] for r in labels];samples.append(dict(start=start,x=[f['features'] for f in fs],seq=seq,y=y+[sum(y)],available=max(f['features_available_ns'] for f in fs)))
            if not samples:raise ValueError('empty validation workload')
            pred={}
            pred['last_available']=[[r['seq'][i][-1] for i in range(9)] for r in samples]
            pred['mean_five']=[[sum(r['seq'][i][-5:])/5 for i in range(9)] for r in samples]
            gx=[m(torch.tensor([[[v] for v in r['seq'][i]] for r in samples],dtype=torch.double)).tolist() for i,m in enumerate(models)]
            pred['frozen_gru']=[[gx[i][k] for i in range(9)] for k in range(len(samples))]
            pred['frozen_moe']=moe(torch.tensor([r['x'] for r in samples],dtype=torch.double)).tolist()
            w=d['workload'];report['datasets'][w]={'source':d,'coverage':reject,'evaluated_seconds':len(samples)};report['metrics'][w]={}
            for name,values in pred.items():
                ms={h:metrics((v[i],r['y'][i]) for v,r in zip(values,samples)) for i,h in enumerate(HOSTS)}
                ms['summed_node_cluster']=metrics((sum(v[:8]),r['y'][8]) for v,r in zip(values,samples));report['metrics'][w][name]=ms
                for v,r in zip(values,samples):
                    if r['available']>r['start']-30000000:raise ValueError('late feature')
                    logs.append(dict(model=name,workload=w,run_id=d['run_id'],source_role='validation',window_start_ns=r['start'],features_available_ns=r['available'],actual=r['y'],raw_predictions=v,final_predictions=v))
    dest.mkdir();(dest/'predictions.json').write_text(json.dumps(logs));report.update(state='completed',predictions_sha256=hashlib.sha256((dest/'predictions.json').read_bytes()).hexdigest())
    (dest/'report.json').write_text(json.dumps(report,indent=2));print(json.dumps({'state':'completed','fit_performed':False,'seconds':{w:d['evaluated_seconds'] for w,d in report['datasets'].items()}}))


if __name__=='__main__':main()
