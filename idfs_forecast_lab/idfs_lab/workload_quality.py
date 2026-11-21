"""Read-only QA of completed development rounds; no accuracy claims."""
import argparse
import math
from pathlib import Path
import statistics

from .io import load, read_events, save
from .joint import causal_features, index_windows


def report(directory, site):
    directory = Path(directory)
    anchor = load(directory/'recording_start.json')['epoch_ns']
    execution = load(directory/'execution.json')
    finished = load(directory/'raw/collection_finished.json')
    raw_ok = len(finished['outcomes']) == 24 and all(
        'error' not in item and item['result']['exit'] == 0 for item in finished['outcomes'])
    windows = list(read_events(directory/'processed/windows.jsonl'))
    index = index_windows(windows)
    result = {'run_id': execution['run_id'], 'role': execution['role'],
              'workload': execution['workload'], 'all_24_collectors_ok': raw_ok,
              'workload_exit': load(directory/'workload_exit.json')['exit'],
              't0_source': 'recording_start', 'nodes': [], 'formal_acceptance': False}
    for node in site['storage_nodes']:
        entity = node['hostname']
        rows = [r for r in windows if r['scope'] == 'host_oss' and r['entity'] == entity
                and r['window_start_ns'] >= anchor
                and r['window_end_ns'] <= anchor+600*10**9]
        complete = [r for r in rows if r['complete']]
        values = [r['total_iops'] for r in complete]
        target = dict(scope='host_oss', entity=entity, component='total')
        features = sum(causal_features(index, r['window_start_ns'], site, target=target) is not None
                       for r in complete)
        result['nodes'].append(dict(entity=entity, complete_seconds=len(complete),
            invalid_seconds=len(rows)-len(complete), causal_feature_rows=features,
            min_iops=min(values) if values else None, max_iops=max(values) if values else None,
            mean_iops=statistics.mean(values) if values else None,
            std_iops=statistics.pstdev(values) if values else None,
            zero_seconds=values.count(0)))
    result['excluded_by_environment'] = (directory/'excluded_from_normal_training.json').exists()
    result['candidate_for_training_QA'] = raw_ok and result['workload_exit'] == 0 and not result['excluded_by_environment']
    result['note'] = 'Timestamp quality and workload log boundaries still require review; this is not a formal pass.'
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--run', required=True)
    parser.add_argument('--site', default='config/site_smoke.json')
    parser.add_argument('--out', required=True)
    args = parser.parse_args()
    save(args.out, report(args.run, load(args.site)))
