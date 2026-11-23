"""Recompute the frozen M3 comparison and D0 phase contributions."""
import argparse
from collections import defaultdict
import gzip
import hashlib
import json
import math
from pathlib import Path
from statistics import mean


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_rows(path):
    rows = {}
    with gzip.open(path, 'rt', encoding='utf-8') as source:
        for row in map(json.loads, source):
            key = row['dataset_id'], row['window_start_ns']
            if key in rows:
                raise ValueError(f'Duplicate result key in {path}')
            rows[key] = row
    return rows


def scores(rows, original, phases, metadata, keys):
    groups = defaultdict(list)
    for key in keys:
        row = rows[key]
        actual = row['actual']
        prediction = row['final_prediction']
        stage = phases[key]
        if (actual != original[key]['actual'] or sum(actual) != stage['actual_cluster']
                or len(actual) != 8 or len(prediction) != 8 or prediction != row['raw_prediction']):
            raise ValueError('Labels, node coverage or raw predictions differ')
        workload = metadata[key[0]]['workload']
        for i, (y, p) in enumerate(zip(actual, prediction)):
            if not math.isfinite(p) or p < 0:
                raise ValueError('Invalid frozen prediction')
            groups[workload, i].append((y, p, stage['phase']))
    if len(groups) != 40:
        raise ValueError('Expected the original 40 groups')
    phase_contributions = defaultdict(float)
    mae = defaultdict(list)
    group_scores = []
    for (workload, _), observations in groups.items():
        positive = [100 * abs(p - y) / y for y, p, _ in observations if y > 0]
        group_scores.append(mean(positive))
        mae[workload].append(mean(abs(p - y) for y, p, _ in observations))
        for y, p, phase in observations:
            if y > 0:
                phase_contributions[phase] += 100 * abs(p - y) / y / len(positive) / len(groups)
    return dict(macro_mape=mean(group_scores), worst=max(group_scores),
                groups_below10=sum(score < 10 for score in group_scores),
                workload_mae={w: mean(v) for w, v in mae.items()},
                phase_contributions_pp=dict(phase_contributions))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('fitted', 'original-input', 'phases', 'lease', 'output'):
        parser.add_argument('--' + name, type=Path, required=True)
    args = parser.parse_args()
    report = json.loads((args.fitted / 'report.json').read_bytes())
    lease = json.loads(args.lease.read_bytes())
    if (report['state'] != 'completed' or lease['state'] != 'completed'
            or lease['job_id'] != report['job_id'] or lease['code_sha'] != report['code_sha']
            or report['external_validation_evaluations'] != 0 or report['new_fit_count'] != 1):
        raise ValueError('Expected one completed, registered fit')
    result = report['result']
    for name, filename in (('checkpoint', 'checkpoint.pt'), ('predictions', 'predictions.jsonl.gz'),
                           ('epochs', 'epochs.json')):
        if digest(args.fitted / filename) != result[name + '_sha256']:
            raise ValueError(f'{name} hash mismatch')
    epochs = json.loads((args.fitted / 'epochs.json').read_bytes())
    if (len(epochs) != result['internal_checkpoint_evaluations']
            or min(epochs, key=lambda epoch: epoch['dev_macro_mape'])['epoch'] != result['best_epoch']):
        raise ValueError('Checkpoint selection differs')
    ids = json.loads((args.original_input / 'sample_ids.json').read_bytes())
    keys = list(map(tuple, ids['internal_dev']))
    metadata = json.loads((args.original_input / 'metadata.json').read_bytes())
    baseline_path = args.original_input / 'r2_L0_predictions.jsonl.gz'
    if digest(baseline_path) != report['inputs']['baseline']['sha256']:
        raise ValueError('Baseline pin differs')
    original = read_rows(baseline_path)
    candidate = read_rows(args.fitted / 'predictions.jsonl.gz')
    phases = read_rows(args.phases)
    if len(keys) != 2661 or not set(keys) == set(original) == set(candidate) or not set(keys) <= set(phases):
        raise ValueError('Exact common keys and complete D0 phase coverage required')
    arms = {name: scores(rows, original, phases, metadata, keys)
            for name, rows in (('M0_reused', original), ('M3', candidate))}
    for name, info in arms.items():
        raw = report['arms'][name]
        if not (math.isclose(info['macro_mape'], raw['macro_mape_percent'], abs_tol=1e-9)
                and math.isclose(info['worst'], raw['worst_group_mape_percent'], abs_tol=1e-9)
                and all(math.isclose(info['workload_mae'][w], v, abs_tol=1e-9)
                        for w, v in raw['workload_mean_node_mae'].items())):
            raise ValueError('Reported metrics differ from predictions')
    base, current = arms['M0_reused'], arms['M3']
    gain = 1 - current['macro_mape'] / base['macro_mape']
    passed = (gain >= .05 and current['worst'] <= 1.05 * base['worst']
              and all(v <= 1.05 * base['workload_mae'][w] for w, v in current['workload_mae'].items()))
    if passed != report['comparison']['passes']:
        raise ValueError('Reported gate decision differs')
    verified = dict(state='verified_frozen_M3_metrics', samples=2661, original_labels=21288,
        groups_recomputed=80, arms=arms, relative_macro_improvement=gain, screen_passes=passed,
        phase_source_sha256=digest(args.phases), additional_fits=0, model_loaded=False,
        lease_elapsed_seconds=lease['ended_epoch']-lease['started_epoch'])
    with args.output.open('x', encoding='utf-8', newline='\n') as output:
        json.dump(verified, output, indent=2)
        output.write('\n')
    print(json.dumps(verified))


if __name__ == '__main__':
    main()
