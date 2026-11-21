"""Explicit fault-preserving diagnostic sequence training; A executes on b507."""
import hashlib
import json
import math
import os
from pathlib import Path
import subprocess
from .diagnostic_split import HOSTS, read_release
from .target_baselines import metrics


def examples(rows, target):
    """Use common eight-node timestamps for fair node/cluster comparison.

    New schema diagnostic_total_lags_v1: newest-first five total IOPS and mean.
    Existing joint_context_v1 temporal totals are oldest-first at 2:15:3.
    """
    if target not in (*HOSTS, 'cluster'):
        raise ValueError('unknown development target')
    grouped={}
    for row in rows:
        start=row['window_start_ns']; host=row['target']['entity']
        group=grouped.setdefault(start,{})
        if host in group:
            raise ValueError('duplicate diagnostic host/window')
        group[host]=row
    output=[]
    for start, group in sorted(grouped.items()):
        if set(group) != set(HOSTS):
            continue
        selected=list(group.values()) if target=='cluster' else [group[target]]
        history=[sum(r['features'][i] for r in selected) for i in (14,11,8,5,2)]
        output.append(dict(window_start_ns=start, features=history+[sum(history)/5],
                           target_iops=sum(r['target_iops'] for r in selected),
                           feature_end_ns=max(r['feature_end_ns'] for r in selected),
                           features_available_ns=max(r['features_available_ns'] for r in selected)))
    if not output:
        raise ValueError('no common eight-node examples')
    return output


def fit(archive, destination, target, model='gru', epochs=40, normalize_inputs=False):
    from .placement import require_training_host
    root=require_training_host()  # before dependency import, data access or writes
    destination=Path(destination).resolve()
    if root not in destination.parents:
        raise ValueError('destination must be inside isolated job root')
    if destination.exists():
        raise ValueError('fresh output directory required')
    if model not in ('gru','tcn') or not isinstance(epochs,int) or not 1<=epochs<=100:
        raise ValueError('unsupported architecture or epoch budget')
    if any(int(os.environ.get(k,'4')) > 4 for k in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS')):
        raise ValueError('CPU thread environment exceeds four')
    visible=os.environ.get('CUDA_VISIBLE_DEVICES','')
    if not visible or ',' in visible:
        raise ValueError('A must select exactly one idle visible GPU')
    import torch
    if not torch.cuda.is_available() or torch.cuda.device_count()!=1:
        raise RuntimeError('exactly one visible GPU required; no CPU fitting fallback')
    splits, provenance=read_release(archive)
    train=examples(splits['train'],target)
    validation=examples(splits['validation'],target)
    x=[r['features'] for r in train]; y=[r['target_iops'] for r in train]
    vx=[r['features'] for r in validation]; vy=[r['target_iops'] for r in validation]
    destination.mkdir(parents=True,exist_ok=False)
    artifact=destination/'model.ts'
    from .models_torch import fit_sequence
    prediction, details=fit_sequence(x,y,vx,vy,model,artifact,epochs=epochs,threads=4,normalize_inputs=normalize_inputs)
    if len(prediction)!=len(vy) or any(not math.isfinite(v) or v<0 for v in prediction):
        raise ValueError('invalid diagnostic predictions')
    from .io import save
    report=dict(provenance,model=model,development_target=target,source_role='diagnostic',
                fault_affected=True,normal_training_certified=False,formal_acceptance=False,
                feature_schema='diagnostic_total_lags_v2' if normalize_inputs else 'diagnostic_total_lags_v1',
                source_feature_schema='joint_context_v1',normalize_inputs=normalize_inputs,
                training_host='lmj@b507',seed=2026,threads=4,dataloader_workers=0,
                epochs_budget=epochs,loss='train-standardized-log1p-smooth-l1',calibration='none',
                train_examples=len(train),validation_examples=len(validation),training_details=details,
                validation_metrics=metrics(zip(prediction,vy)),
                last_available_metrics=metrics((r['features'][0],r['target_iops']) for r in validation),
                mean_five_metrics=metrics((r['features'][5],r['target_iops']) for r in validation),
                artifact_sha256=hashlib.sha256(artifact.read_bytes()).hexdigest(),
                torch_version=torch.__version__,cuda_version=torch.version.cuda,
                execution_environment='A must attach job_id, selected GPU and wall-time log')
    repo=Path(__file__).resolve().parents[2]
    report['candidate_commit']=subprocess.check_output(['git','-C',str(repo),'rev-parse','HEAD'],text=True).strip()
    report['code_dirty']=bool(subprocess.check_output(['git','-C',str(repo),'status','--porcelain'],text=True).strip())
    save(destination/'diagnostic_model_manifest.json',report)
    save(destination/'validation_predictions.json',[
        dict(row,prediction_iops=float(p)) for row,p in zip(validation,prediction)])
    return report


def main():
    import argparse
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--archive',required=True)
    p.add_argument('--output',required=True)
    p.add_argument('--target',required=True,choices=(*HOSTS,'cluster'))
    p.add_argument('--model',choices=('gru','tcn'),default='gru')
    p.add_argument('--epochs',type=int,default=40)
    p.add_argument('--normalize-inputs',action='store_true',help='Train-only log history centering/scaling (diagnostic v2)')
    args=p.parse_args()
    fit(args.archive,args.output,args.target,args.model,args.epochs,args.normalize_inputs)


if __name__=='__main__': main()
