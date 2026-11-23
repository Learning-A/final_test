"""RMSE, MAE, predictive R2 and Pearson r on immutable horizon predictions."""
import csv
import gzip
import hashlib
import json
import math
from pathlib import Path
import statistics

import numpy as np

ROOT=Path(__file__).resolve().parents[1]


def metric(actual,prediction):
    y=np.asarray(actual,dtype=np.float64);p=np.asarray(prediction,dtype=np.float64)
    if y.shape!=p.shape or y.ndim!=1 or not np.isfinite(y).all() or not np.isfinite(p).all():
        raise ValueError('matching finite vectors required')
    error=p-y;yc=y-y.mean();pc=p-p.mean()
    sy=float(yc@yc);sp=float(pc@pc);sse=float(error@error)
    result=dict(n=len(y),mae_iops=float(np.abs(error).mean()),rmse_iops=float(np.sqrt(np.mean(error**2))),
        r2=float(1-sse/sy) if sy>0 else None,
        pearson_r=float(np.clip((yc@pc)/np.sqrt(sy*sp),-1,1)) if sy>0 and sp>0 else None,
        actual_std_iops=float(np.sqrt(sy/len(y))),prediction_std_iops=float(np.sqrt(sp/len(y))),
        bias_iops=float(error.mean()))
    # Separate stdlib moments: no regression or learned calibration is fitted.
    ym=statistics.mean(actual);pm=statistics.mean(prediction)
    std_sy=math.fsum((v-ym)**2 for v in actual);std_sp=math.fsum((v-pm)**2 for v in prediction)
    checks=dict(mae_iops=statistics.mean(abs(q-t) for t,q in zip(actual,prediction)),
        rmse_iops=math.sqrt(statistics.mean((q-t)**2 for t,q in zip(actual,prediction))),
        r2=1-math.fsum((q-t)**2 for t,q in zip(actual,prediction))/std_sy if std_sy>0 else None,
        pearson_r=math.fsum((t-ym)*(q-pm) for t,q in zip(actual,prediction))/math.sqrt(std_sy*std_sp) if std_sy>0 and std_sp>0 else None)
    for name,value in checks.items():
        if value is None:assert result[name] is None
        else:assert math.isclose(result[name],value,rel_tol=1e-9,abs_tol=1e-10),(name,result[name],value)
    return result


def summarize(groups):
    keys=('mae_iops','rmse_iops','r2','pearson_r','actual_std_iops','prediction_std_iops','bias_iops')
    return {k:dict(mean=float(statistics.mean([g[k] for g in groups.values() if g[k] is not None])),
                   defined_groups=sum(g[k] is not None for g in groups.values())) for k in keys}


def main():
    result={};csvrows=[]
    for arm in ('direct_seconds','fine100ms'):
        root=ROOT/'.local/cluster_horizon'/arm
        report=json.loads((root/'report.json').read_text());path=root/'predictions.jsonl.gz'
        digest=hashlib.sha256(path.read_bytes()).hexdigest();assert digest==report['predictions_sha256']
        with gzip.open(path,'rt') as f:rows=list(map(json.loads,f))
        assert len(rows)==2297
        workloads=sorted({r['workload'] for r in rows});window={};trajectory={}
        for kind,output,steps,field in [('window',window,(1,3,5,10),'window_means'),('trajectory',trajectory,range(1,11),'seconds')]:
            for j,step in enumerate(steps):
                groups={}
                for workload in workloads:
                    selected=[r for r in rows if r['workload']==workload]
                    ys=[r['actual_'+field][j] for r in selected];ps=[r['prediction_'+field][j] for r in selected]
                    groups[workload]=metric(ys,ps)
                    old=(report['summary']['window_groups'] if kind=='window' else report['summary']['trajectory_groups'])[workload][str(step)]
                    assert math.isclose(groups[workload]['mae_iops'],old['mae_iops'],abs_tol=1e-8)
                    csvrows.append(dict(arm=arm,output_type=kind,seconds=step,workload=workload,**groups[workload]))
                pooled=metric([r['actual_'+field][j] for r in rows],[r['prediction_'+field][j] for r in rows])
                output[str(step)]=dict(groups=groups,macro=summarize(groups),pooled=pooled)
        result[arm]=dict(source_prediction_sha256=digest,window=window,trajectory=trajectory)
    data=dict(scope='Frozen2297 train03 origins; original per-workload partitions, all future horizons already evaluated. Supplemental descriptive statistics only; no fit, model loading, new inference or external validation.',
        definitions=dict(MAE='mean(abs(pred-y)), IOPS',RMSE='sqrt(mean((pred-y)^2)), IOPS',
            R2='1-SSE/sum((y-mean(y))^2); not square of prediction correlation',
            Pearson='sum((y-ymean)*(pred-predmean))/sqrt(sum((y-ymean)^2)*sum((pred-predmean)^2))',
            macro='Five workload metrics computed separately, then arithmetic mean. RMSE macro is mean of five RMSEs; Pearson macro is a descriptive mean, not pooled correlation.',
            degenerate='undefined R2 or correlation is null; never replaced with zero.',
            pooled='All five workloads mixed; level differences can inflate pooled R2/r and mask weak within-workload forecasts.'),
        independent_stdlib_moments_verified=True,models=result)
    path=ROOT/'reports/cluster_horizon_regression_metrics.json'
    path.write_text(json.dumps(data,indent=2,allow_nan=False)+'\n',encoding='utf-8',newline='\n')
    csvpath=ROOT.parent.parent/'outputs/cluster_horizon_20261123/regression_metrics_by_workload.csv'
    with csvpath.open('w',encoding='utf-8-sig',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(csvrows[0]));writer.writeheader();writer.writerows(csvrows)
    for arm,r in result.items():
        print(arm)
        for h in (1,3,5,10):
            m=r['window'][str(h)]['macro'];p=r['window'][str(h)]['pooled']
            print(h,{k:round(m[k]['mean'],6) for k in ('mae_iops','rmse_iops','r2','pearson_r')},'pooledr',round(p['pearson_r'],6),'pooledR2',round(p['r2'],6))
        print('LEAD10',{k:r['trajectory']['10']['macro'][k]['mean'] for k in ('mae_iops','rmse_iops','r2','pearson_r')})


if __name__=='__main__':main()
