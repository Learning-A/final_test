"""No-fit auxiliary tolerance curves; never redefine formal acceptance."""
import argparse,hashlib,json,math,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'idfs_forecast_lab'))
from idfs_lab.target_baselines import metrics


def main():
    p=argparse.ArgumentParser();p.add_argument('--predictions',required=True);p.add_argument('--source-report',required=True);p.add_argument('--arm',default='moe_combo');p.add_argument('--output',required=True);a=p.parse_args()
    dest=Path(a.output)
    if dest.exists():raise ValueError('fresh output required')
    data=Path(a.predictions).read_bytes();source=json.loads(Path(a.source_report).read_text(encoding='utf-8'))
    sha=hashlib.sha256(data).hexdigest()
    if sha!=source['predictions_sha256']:raise ValueError('prediction checksum')
    rows=[r for r in json.loads(data) if r['arm']==a.arm]
    if not rows:raise ValueError('missing arm')
    targets=['idfs'+str(i) for i in (44,45,46,47,64,65,66,67)]+['cluster']
    absolutes=[0,1,5,10,25,50,100];relative=.1;out={}
    for w in sorted({r['workload'] for r in rows}):
        rs=[r for r in rows if r['workload']==w];out[w]={}
        for i,h in enumerate(targets):
            pairs=[(r['final_predictions'][i],r['actual'][i]) for r in rs]
            ms=metrics(pairs)
            curve=[{'absolute_tolerance_iops':v,'relative_tolerance':relative,'fraction_within_tolerance_percent':100*sum(abs(p-y)<=max(v,relative*y) for p,y in pairs)/len(pairs)} for v in absolutes]
            # The zero/zero case has no error; all valid windows enter this auxiliary statistic.
            smape=100*sum(2*abs(p-y)/(abs(p)+abs(y)) if p+y else 0 for p,y in pairs)/len(pairs)
            out[w][h]={'reference_metrics':ms,'smape_percent':smape,'tolerance_curve':curve}
    report={'source_predictions_sha256':sha,'arm':a.arm,'scope':'auxiliary sensitivity only; includes true-zero windows','formal_definition_changed':False,
        'approved_business_tolerances':None,'candidate_selection_performed':False,'formal_acceptance':False,'groups':out}
    dest.write_text(json.dumps(report,indent=2),encoding='utf-8');print(json.dumps({'groups':45,'formal_definition_changed':False,'output':str(dest)}))


if __name__=='__main__':main()
