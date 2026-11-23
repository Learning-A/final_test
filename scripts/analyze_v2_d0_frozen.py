"""Attribute existing held-out prediction errors; never fit or infer a model."""
from __future__ import annotations

import argparse
from collections import defaultdict
import gzip
import hashlib
import json
import math
from pathlib import Path
import platform
import time
import zipfile

HOSTS = ('idfs44', 'idfs45', 'idfs46', 'idfs47', 'idfs64', 'idfs65', 'idfs66', 'idfs67')
BINS = ('zero', 'positive_le10', '10_lt_y_le100', '100_lt_y_le1000', 'gt1000')


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def bucket(y):
    if y == 0:
        return BINS[0]
    if y <= 10:
        return BINS[1]
    if y <= 100:
        return BINS[2]
    return BINS[3] if y <= 1000 else BINS[4]


def new_stats():
    return dict(count=0, positive_count=0, zero_count=0, actual_sum=0.0,
                prediction_sum=0.0, absolute_error_sum=0.0,
                signed_error_sum=0.0, ape_percent_sum=0.0,
                zero_actual_positive_prediction_count=0)


def add(s, y, p):
    if not math.isfinite(y) or not math.isfinite(p) or y < 0 or p < 0:
        raise ValueError('Nonfinite/negative frozen actual or prediction')
    s['count'] += 1
    s['actual_sum'] += y
    s['prediction_sum'] += p
    s['absolute_error_sum'] += abs(p-y)
    s['signed_error_sum'] += p-y
    if y > 0:
        s['positive_count'] += 1
        s['ape_percent_sum'] += 100*abs(p-y)/y
    else:
        s['zero_count'] += 1
        s['zero_actual_positive_prediction_count'] += p > 0


def finish(s, full_positive=None, group_count=40):
    ans = dict(s)
    n, pos, total = s['count'], s['positive_count'], s['actual_sum']
    ans.update(mape_percent=s['ape_percent_sum']/pos if pos else None,
               mae=s['absolute_error_sum']/n if n else None,
               bias=s['signed_error_sum']/n if n else None,
               wape_percent=100*s['absolute_error_sum']/total if total else None)
    if full_positive is not None:
        ans['macro_mape_contribution_pp'] = (
            s['ape_percent_sum']/full_positive/group_count if full_positive else None)
    return ans


def keyed(rows, keys):
    out = {}
    for r in rows:
        key = (r['dataset_id'], r['window_start_ns'])
        if not isinstance(key[1], int) or isinstance(key[1], bool):
            raise ValueError('Timestamp must be integer ns')
        if key in out:
            raise ValueError('Duplicate prediction key')
        if key not in keys:
            raise ValueError('Unexpected prediction key')
        out[key] = r
    if set(out) != keys:
        raise ValueError('Missing prediction key; refusing an intersection')
    return out


def read_archive(path, expected_sha):
    if sha(path) != expected_sha:
        raise ValueError('Registered archive hash mismatch')
    z = zipfile.ZipFile(path)
    return z


def load_frozen(reg):
    members = {}
    with read_archive(reg['r1_archive'], reg['r1_sha256']) as z:
        ids = json.loads(z.read('results/sample_ids.json'))
        if len(ids['internal_dev']) != 2661 or len(ids['fit']) != 5328:
            raise ValueError('Sample count mismatch')
        keys = set(map(tuple, ids['internal_dev']))
        if len(keys) != 2661 or keys.intersection(map(tuple, ids['fit'])):
            raise ValueError('Duplicate or overlapping partitions')
        r1_rows = list(map(json.loads, gzip.decompress(z.read('results/predictions.jsonl.gz')).splitlines()))
        arms = {f'R1_{arm}_50': keyed((r for r in r1_rows if r['margin_ms'] == 50 and r['arm'] == arm), keys)
                for arm in ('F0', 'F0_F1')}
        for m in ('results/sample_ids.json', 'results/predictions.jsonl.gz'):
            members['R1/'+m] = hashlib.sha256(z.read(m)).hexdigest()
    with read_archive(reg['r2_archive'], reg['r2_sha256']) as z:
        ids2 = json.loads(z.read('sample_ids.json'))
        if ids2 != ids:
            raise ValueError('R1 and R2 sample IDs/order differ')
        for arm in ('L0', 'L1', 'L2', 'L3'):
            m = f'results/{arm}/predictions.jsonl.gz'
            arms['R2_'+arm] = keyed(map(json.loads, gzip.decompress(z.read(m)).splitlines()), keys)
            members['R2/'+m] = hashlib.sha256(z.read(m)).hexdigest()
        members['R2/sample_ids.json'] = hashlib.sha256(z.read('sample_ids.json')).hexdigest()
    canonical = arms['R1_F0_50']
    for arm, rows in arms.items():
        for key, row in rows.items():
            expected = canonical[key]
            n = 9 if arm.startswith('R1_') else 8
            if len(row['actual']) != n or row['actual'][:8] != expected['actual'][:8]:
                raise ValueError('Frozen source labels/order differ')
            pred = row['final_predictions'] if n == 9 else row['final_prediction']
            if len(pred) != n:
                raise ValueError('Prediction width differs')
            if n == 9 and (row['source_role'] != 'train' or row['development_partition'] != 'internal_run03'):
                raise ValueError('Not the registered training-role held-out scope')
    if len({v['workload'] for v in canonical.values()}) != 5:
        raise ValueError('Missing workload')
    return keys, arms, canonical, members


def analyze(keys, arms, canonical, phases=None):
    result = {}
    for arm, rows in arms.items():
        groups = defaultdict(new_stats)
        strata = {dim: defaultdict(new_stats) for dim in ('magnitude', 'phase', 'run', 'feature_observation_count')}
        cluster = {mode: defaultdict(new_stats) for mode in ('node_sum', 'direct_head')}
        for key in sorted(keys):
            row, info = rows[key], canonical[key]
            pred = row.get('final_predictions', row.get('final_prediction'))
            ys = row['actual'][:8]
            phase = phases.get(key, 'unknown') if phases is not None else 'unknown'
            for i, host in enumerate(HOSTS):
                g = (info['workload'], host)
                y,p = ys[i], pred[i]
                add(groups[g], y, p)
                dims = dict(magnitude=bucket(y), phase=phase, run=info['run_id'],
                            feature_observation_count=str(info['fine_observed'][i]))
                for dim, label in dims.items():
                    add(strata[dim][(*g, label)], y, p)
            add(cluster['node_sum'][info['workload']], sum(ys), sum(pred[:8]))
            if len(pred) == 9:
                if sum(ys) != row['actual'][8]:
                    raise ValueError('R1 cluster actual does not equal node sum')
                add(cluster['direct_head'][info['workload']], row['actual'][8], pred[8])
        if len(groups) != 40 or any(s['count'] == 0 or s['positive_count'] == 0 for s in groups.values()):
            raise ValueError('Missing positive-label workload/node group')
        group_rows = [dict(workload=g[0],node_id=g[1],**finish(s,s['positive_count'])) for g,s in sorted(groups.items())]
        by_stratum = {}
        for dim, values in strata.items():
            detail = [dict(workload=w,node_id=h,stratum=label,**finish(s,groups[(w,h)]['positive_count']))
                      for (w,h,label),s in sorted(values.items())]
            totals = {}
            for label in sorted({r['stratum'] for r in detail}):
                rs = [r for r in detail if r['stratum'] == label]
                totals[label] = dict(count=sum(r['count'] for r in rs),
                    positive_count=sum(r['positive_count'] for r in rs),
                    macro_mape_contribution_pp=sum(r['macro_mape_contribution_pp'] for r in rs),
                    mean_group_absolute_error_contribution=sum(r['absolute_error_sum']/groups[(r['workload'],r['node_id'])]['count']/40 for r in rs))
            by_stratum[dim] = dict(detail=detail,totals=totals)
        macro = sum(g['mape_percent'] for g in group_rows)/40
        for dim,values in by_stratum.items():
            if not math.isclose(sum(t['macro_mape_contribution_pp'] for t in values['totals'].values()), macro,abs_tol=1e-9):
                raise ValueError('Stratum contributions do not sum to macro')
        result[arm] = dict(node_rows=sum(g['count'] for g in group_rows),
            macro_mape_percent=macro,worst_group_mape_percent=max(g['mape_percent'] for g in group_rows),
            groups_mape_lt10=sum(g['mape_percent']<10 for g in group_rows),
            workload_mean_node_mae={w:sum(g['mae'] for g in group_rows if g['workload']==w)/8 for w in sorted({g['workload'] for g in group_rows})},
            groups=group_rows,strata=by_stratum,
            cluster={mode:{w:finish(s) for w,s in values.items()} for mode,values in cluster.items()})
    return result


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--registration',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    args=p.parse_args()
    if args.output.exists():
        raise SystemExit('Refusing to overwrite a completed analysis')
    started=time.monotonic()
    reg=json.loads(args.registration.read_text(encoding='utf-8'))
    if sha(__file__) != reg['runner_sha256']:
        raise ValueError('Runner hash differs from registration')
    keys,arms,canonical,members=load_frozen(reg)
    report=dict(schema='v2_d0_frozen_error_v1',job_id=reg['job_id'],state='frozen_error_complete_phase_pending',
        data_scope='existing internal train03 frozen predictions; no model inference or fitting',
        samples=2661, node_order=HOSTS,group_count=40,source_role='train',
        independent_new_validation=False,fit_count=0,new_model_inference_count=0,
        online_time_eligibility=False,clock_uncertainty_ns=None,model_service_arrival_proof=False,
        phase_join_status='pending_A_delivery_all_unknown_without_row_filtering',
        feature_observation_scope='R1 primary F1 input coverage, reused for matching sample diagnosis; not R2 receipt evidence',
        registration_sha256=sha(args.registration),runner_sha256=sha(__file__),python=platform.python_version(),
        input_member_sha256=members,
        metric='100*abs(pred-y)/y for y>0; per workload/node mean then equal macro over 40; stratum contributions use full group positive denominators',
        arms=analyze(keys,arms,canonical))
    report['elapsed_seconds']=time.monotonic()-started
    if report['elapsed_seconds']>reg['budget_seconds']:
        raise TimeoutError('Registered analysis budget exceeded')
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    print(json.dumps({k:report[k] for k in ('job_id','state','samples','elapsed_seconds')}))


if __name__ == '__main__':
    main()
