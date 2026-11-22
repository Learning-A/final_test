"""Check R2 artifacts and recompute the offline comparison without loading models."""
import argparse
from collections import defaultdict
import gzip
import json
import math
from pathlib import Path
import zipfile

from verify_r1_results import compare, file_sha, metrics


def require(condition, message):
    if not condition:
        raise ValueError(message)


def source_labels(train, datasets, expected, hosts):
    labels = {}
    keys = defaultdict(list)
    for dataset, start in expected:
        keys[dataset].append(start)
    with zipfile.ZipFile(train) as source:
        for dataset, starts in keys.items():
            info = datasets[dataset]
            require(info['role'] == 'train' and info['index'] == 3, 'internal train03 required')
            windows = [json.loads(row) for row in source.read(f'{dataset}/processed/windows.jsonl').splitlines()]
            lookup = {(row['entity'], row['window_start_ns']): row for row in windows if row['scope'] == 'host_oss'}
            for start in starts:
                actual = [lookup[(host, start)] for host in hosts]
                require(all(row['complete'] and row['n_ticks'] == 10 and row['window_end_ns'] == start + 1_000_000_000
                            for row in actual), 'incomplete original label')
                labels[(dataset, start)] = [row['total_iops'] for row in actual]
    return labels


def verify(root, train, index, samples):
    report = json.loads((root / 'report.json').read_bytes())
    require(report['state'] == 'completed' and report['external_validation_evaluations'] == 0
            and not report['formal_acceptance'] and not report['online_time_eligibility'], 'completed offline R2 required')
    for path, field in ((train, 'train'), (index, 'original_index'), (samples, 'sample_ids')):
        require(file_sha(path) == report['input_sha256'][field], f'{field} hash differs')
    for name, field in (('preprocessing.json', 'preprocessing_sha256'), ('epoch0.pt', 'epoch0_checkpoint_sha256')):
        require(file_sha(root / name) == report[field], f'{name} hash differs')

    sample_ids = json.loads(samples.read_bytes())
    expected = [tuple(key) for key in sample_ids['internal_dev']]
    hosts = report['hosts']
    require(len(hosts) == 8 and len(expected) == report['internal_dev_sample_count'] == 2661
            and len(sample_ids['fit']) == report['fit_sample_count'] == 5328, 'sample or node scope differs')
    datasets = {d['dataset_id']: d for d in json.loads(index.read_bytes())['datasets']}
    labels = source_labels(train, datasets, expected, hosts)
    require(set(report['arms']) == set(report['loss_arms']) == {'L0', 'L1', 'L2', 'L3'}, 'arm scope differs')

    summary = {}
    for arm, arm_report in report['arms'].items():
        arm_dir = root / arm
        for name, field in (('checkpoint.pt', 'checkpoint_sha256'), ('epochs.json', 'epochs_sha256'),
                            ('predictions.jsonl.gz', 'predictions_sha256')):
            require(file_sha(arm_dir / name) == arm_report[field], f'{arm}/{name} hash differs')
        logs = json.loads((arm_dir / 'epochs.json').read_bytes())
        require(len(logs) == arm_report['internal_checkpoint_evaluations'] == arm_report['epochs_trained'] + 1
                and [e['epoch'] for e in logs] == list(range(len(logs))), 'checkpoint evaluation count differs')
        selected = min(logs, key=lambda epoch: epoch['dev_macro_mape'])
        require(selected['epoch'] == arm_report['best_epoch'], 'checkpoint selection differs')

        keys = []
        groups = defaultdict(lambda: defaultdict(list))
        with gzip.open(arm_dir / 'predictions.jsonl.gz', 'rt', encoding='utf8') as stream:
            for line in stream:
                row = json.loads(line)
                key = (row['dataset_id'], row['window_start_ns'])
                keys.append(key)
                actual, predicted = row['actual'], row['raw_prediction']
                require(actual == labels[key] and predicted == row['final_prediction'] and len(predicted) == len(hosts),
                        'sample label or prediction shape differs')
                require(all(math.isfinite(value) and value >= 0 for value in predicted), 'invalid forecast')
                target_groups = groups[datasets[key[0]]['workload']]
                for host, a, p in zip(hosts, actual, predicted):
                    target_groups[host].append((p, a))
                target_groups['cluster_sum'].append((sum(predicted), sum(actual)))
        require(keys == expected, 'frozen sample order differs')
        group_mape = []
        mean_mae = {}
        for workload, targets in sorted(groups.items()):
            node_mae = []
            for host, pairs in targets.items():
                computed = metrics(pairs)
                source = arm_report['workload_metrics'][workload]
                compare(computed, source['cluster_sum'] if host == 'cluster_sum' else source['nodes'][host])
                if host != 'cluster_sum':
                    group_mape.append(computed['mape_percent'])
                    node_mae.append(computed['mae_iops'])
            mean_mae[workload] = sum(node_mae) / len(node_mae)
        require(len(group_mape) == 40, 'node × workload group count differs')
        macro = sum(group_mape) / len(group_mape)
        compare(macro, arm_report['macro_mape'])
        compare(macro, selected['dev_macro_mape'])
        summary[arm] = {'macro_mape': macro, 'worst_group_mape': max(group_mape), 'workload_node_mean_mae': mean_mae,
                        'groups_mape_lt_10': sum(mape < 10 for mape in group_mape),
                        'best_epoch': selected['epoch'], 'internal_checkpoint_evaluations': len(logs)}

    baseline = summary['L0']
    for arm, result in summary.items():
        improvement = 1 - result['macro_mape'] / baseline['macro_mape'] if baseline['macro_mape'] else None
        passes = (arm != 'L0' and improvement is not None and improvement >= .05
                  and result['worst_group_mape'] <= baseline['worst_group_mape'] * 1.05
                  and all(value <= baseline['workload_node_mean_mae'][workload] * 1.05
                          for workload, value in result['workload_node_mean_mae'].items()))
        computed = {'macro_mape_relative_improvement': improvement, 'passes': passes,
                    **{key: result[key] for key in ('worst_group_mape', 'workload_node_mean_mae', 'groups_mape_lt_10')}}
        compare(computed, report['arms'][arm]['screen'])
        result.update(macro_mape_relative_improvement=improvement, screen_passes=passes)
    return {'state': 'verified_offline_metrics', 'job_id': report['job_id'], 'code_sha': report['code_sha'],
            'report_sha256': file_sha(root / 'report.json'), 'samples': len(expected),
            'original_node_labels_checked': len(labels) * len(hosts), 'prediction_rows': len(expected) * len(summary),
            'arms': summary, 'online_time_eligibility': False, 'formal_acceptance': False}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('results', 'train', 'index', 'samples', 'output'):
        parser.add_argument('--' + name, type=Path, required=True)
    args = parser.parse_args()
    result = verify(args.results, args.train, args.index, args.samples)
    with args.output.open('x', encoding='utf8') as stream:
        json.dump(result, stream, indent=2, allow_nan=False)
        stream.write('\n')
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
