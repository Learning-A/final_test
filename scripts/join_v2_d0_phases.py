"""Complete the phase-only attribution after A delivers per-second labels."""
import argparse
from collections import defaultdict
import gzip
import json
from pathlib import Path
import time

from analyze_v2_d0_frozen import add, finish, load_frozen, new_stats, sha, HOSTS

PHASES = {'known_onset','quiet_left_censored_start','quiet_continuation',
          'known_recovery','ordinary','unknown'}


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--registration',required=True,type=Path)
    args=p.parse_args()
    reg=json.loads(args.registration.read_text(encoding='utf-8'))
    if sha(__file__) != reg['runner_sha256']:
        raise ValueError('Join runner hash mismatch')
    dest=Path(reg['output'])
    if dest.exists():
        raise ValueError('Refusing to repeat completed phase join')
    start=time.monotonic()
    base=Path(reg['base_report'])
    if sha(base) != reg['base_sha256'] or sha(reg['phase_path']) != reg['phase_sha256']:
        raise ValueError('Report or phase source hash mismatch')
    report=json.loads(base.read_text(encoding='utf-8'))
    source_reg=json.loads(Path(reg['source_registration']).read_text(encoding='utf-8'))
    keys,arms,canonical,_=load_frozen(source_reg)
    phases={}
    with gzip.open(reg['phase_path'],'rt',encoding='utf-8') as f:
        for line in f:
            row=json.loads(line)
            key=row['dataset_id'],row['window_start_ns']
            if not isinstance(key[1],int) or isinstance(key[1],bool):
                raise ValueError('Noninteger phase timestamp')
            if key in phases or row['phase'] not in PHASES:
                raise ValueError('Duplicate key or invalid phase')
            phases[key]=row
    matched=keys & set(phases)
    for key in matched:
        row=phases[key]
        actual=sum(canonical[key]['actual'][:8])
        if row['actual_cluster'] is not None and row['actual_cluster'] != actual:
            raise ValueError('A cluster actual differs from frozen actual')
        if row['actual_cluster'] is None and row['phase']!='unknown':
            raise ValueError('Known phase without complete actual')
    for arm,rows in arms.items():
        values=defaultdict(new_stats)
        full={(g['workload'],g['node_id']):g['positive_count'] for g in report['arms'][arm]['groups']}
        for key in sorted(keys):
            row,info=rows[key],canonical[key]
            label=phases[key]['phase'] if key in phases else 'unknown'
            pred=row.get('final_predictions',row.get('final_prediction'))
            for i,host in enumerate(HOSTS):
                add(values[(info['workload'],host,label)],row['actual'][i],pred[i])
        detail=[dict(workload=w,node_id=h,stratum=label,**finish(s,full[(w,h)])) for (w,h,label),s in sorted(values.items())]
        totals={}
        for label in PHASES:
            rs=[r for r in detail if r['stratum']==label]
            totals[label]=dict(count=sum(r['count'] for r in rs),positive_count=sum(r['positive_count'] for r in rs),
                macro_mape_contribution_pp=sum(r['macro_mape_contribution_pp'] for r in rs))
        summed=sum(v['macro_mape_contribution_pp'] for v in totals.values())
        if abs(summed-report['arms'][arm]['macro_mape_percent'])>1e-9:
            raise ValueError('Phase contributions not additive')
        report['arms'][arm]['strata']['phase']=dict(detail=detail,totals=totals)
    report.update(state='completed',phase_join_status='A_phase_labels_left_joined_no_score_row_filtering',
        phase_join=dict(source_sha256=reg['phase_sha256'],base_report_sha256=reg['base_sha256'],
            matched_sample_keys=len(matched),missing_keys=len(keys-matched),source_rows=len(phases),
            known_phase_sample_keys=sum(phases[k]['phase']!='unknown' for k in matched),
            registration_sha256=sha(args.registration),runner_sha256=sha(__file__),elapsed_seconds=time.monotonic()-start))
    if report['phase_join']['elapsed_seconds']>reg['budget_seconds']:
        raise TimeoutError('Phase join budget exceeded')
    dest.write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(report['phase_join']))


if __name__=='__main__':
    main()
