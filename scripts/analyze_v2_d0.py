"""Read-only whole-run diagnostics under the reviewed V2-D0 contract; no fitting."""
import argparse
from collections import Counter, defaultdict
import gzip
import hashlib
import json
from pathlib import Path
import statistics
import time

SECOND = 1_000_000_000
GRID = 100_000_000
HOSTS = ['idfs44', 'idfs45', 'idfs46', 'idfs47', 'idfs64', 'idfs65', 'idfs66', 'idfs67']


def read(path):
    return json.loads(path.read_bytes())


def sha(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def dump(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n', encoding='utf8')


def visible(row, cutoff):
    return (row is not None and row.get('sample_ok', False)
            and max(row['bucket_end_ns'], row['available_epoch_ns']) <= cutoff)


def probe(buckets, start):
    cutoff = start - 80_000_000
    end = (cutoff // GRID) * GRID
    current = buckets.get(end)
    result = dict(bucket_end_ns=end, alert=False, observable=False, reason='missing_or_late_current')
    if not visible(current, cutoff):
        return result
    reference = [buckets.get(end-i*GRID) for i in range(1, 10)]
    valid = [r for r in reference if visible(r, cutoff)]
    result.update(v=current['total_count'], reference_valid=len(valid),
                  reference_ends=[end-i*GRID for i in range(1, 10)])
    if len(valid) < 6:
        result['reason'] = 'insufficient_reference'
        return result
    median = statistics.median(r['total_count'] for r in valid)
    result['median'] = median
    if median <= 0:
        result['reason'] = 'zero_reference'
        return result
    upper = max(max(r['bucket_end_ns'], r['available_epoch_ns']) for r in valid+[current])
    result.update(observable=True, reason='eligible', alert=current['total_count'] <= .1*median,
                  source_available_upper_ns=upper, source_lead_ns=start-upper)
    assert upper <= cutoff
    return result


def label_states(cluster):
    rows = {}
    for start, target in sorted(cluster.items()):
        reference = None
        for lag in range(2, 7):
            prior = cluster.get(start-lag*SECOND)
            if prior and start-5*SECOND <= prior['end'] < start and prior['available'] <= start-80_000_000:
                reference = prior
                break
        rows[start] = {'quiet': None if reference is None else target['total'] <= max(10, .1*reference['total']),
                       'actual_cluster': target['total'], 'reference_total': None if reference is None else reference['total']}
    for start, row in rows.items():
        quiet = row['quiet']
        previous = rows.get(start-SECOND, {}).get('quiet')
        row['phase'] = ('unknown' if quiet is None else
                        'quiet_continuation' if quiet and previous is True else
                        'known_onset' if quiet and previous is False else
                        'quiet_left_censored_start' if quiet else
                        'known_recovery' if previous is True else 'ordinary')
    return rows


def segments(rows):
    events = []
    for start, row in sorted(rows.items()):
        if row['quiet'] is not True:
            continue
        if events and events[-1]['end_start_ns']+SECOND == start:
            events[-1]['end_start_ns'] = start
            events[-1]['seconds'] += 1
        else:
            events.append(dict(start_ns=start, end_start_ns=start, seconds=1,
                               left_censored=row['phase'] != 'known_onset'))
    for event in events:
        following = rows.get(event['end_start_ns']+SECOND, {}).get('quiet')
        event['right_censored'] = following is not False
    return events


def score(rows, probes):
    counts = Counter()
    for start, row in rows.items():
        p = probes[start]
        quiet = row['quiet']
        counts['all_alerts'] += p['alert']
        if quiet is None:
            counts['unknown_seconds'] += 1
            counts['alert_unknown_label'] += p['alert']
            continue
        counts['known_seconds'] += 1
        counts['observable_known'] += p['observable']
        counts['quiet_seconds' if quiet else 'nonquiet_seconds'] += 1
        counts['known_alerts'] += p['alert']
        counts['true_positive' if quiet else 'false_positive'] += p['alert']
        if row['phase'] == 'known_onset':
            counts['onsets'] += 1
            counts['onset_hits'] += p['alert']
        if p['alert'] and not quiet:
            prior = probes.get(start-SECOND, {})
            prior_state = rows.get(start-SECOND, {}).get('quiet')
            counts['false_positive_segments'] += not (prior.get('alert') and prior_state is False)
    def ratio(a, b):
        return counts[a]/counts[b] if counts[b] else None
    return dict(counts= dict(counts), recall=ratio('true_positive', 'quiet_seconds'),
                precision=ratio('true_positive', 'known_alerts'), onset_recall=ratio('onset_hits', 'onsets'),
                false_positive_per_minute=60*ratio('false_positive', 'nonquiet_seconds') if counts['nonquiet_seconds'] else None,
                observable_coverage=ratio('observable_known', 'known_seconds'),
                reasons=dict(Counter(p['reason'] for p in probes.values())))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=Path, default=Path('.'))
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    began = time.monotonic()
    root, out = args.root, args.output
    out.mkdir(parents=True, exist_ok=False)
    dump(out/'registration.json', dict(job_id='a_v2_d0_20261123_a01', state='running',
         spec_commit='840f6dd837ca81d9be9eca2067fa3e34531ade5e', script_sha256=sha(Path(__file__)),
         hard_budget_seconds=1800, fitting=False, external_validation=False))
    index_path = root/'.local/r0_artifacts/r1_training_core_index_v1.json'
    index = read(index_path)
    train = root/'.local/handoffs/development-all-20-rounds-20261122-v1/train'
    all_results, sources = {}, []
    with gzip.open(out/'v2_d0_phases.jsonl.gz', 'wt', encoding='utf8') as phases, gzip.open(out/'v2_d0_probes.jsonl.gz', 'wt', encoding='utf8') as probes_out, gzip.open(out/'v2_d0_events.jsonl.gz', 'wt', encoding='utf8') as events_out:
        for capture in index['captures']:
            assert time.monotonic()-began < 1800
            dataset = capture['dataset_id']
            assert capture['source_role'] == 'train'
            directory = (root/'.local/v2_graph_pinned' if capture['source_run_id']=='graph_zipf_train_03'
                         else root/'.local/r0_artifacts/r1-train-core'/dataset)
            manifest = directory/'export_manifest.json'
            assert sha(manifest) == capture['export_manifest_sha256']
            fine = directory/'fine_context.jsonl.gz'
            assert sha(fine) == next(f['sha256'] for f in read(manifest)['files'] if f['name']==fine.name)
            buckets, clients = defaultdict(dict), {}
            with gzip.open(fine, 'rt', encoding='utf8') as stream:
                for line in stream:
                    row = json.loads(line)
                    end = row['bucket_end_ns']
                    if row['scope']=='host_oss':
                        assert end not in buckets[row['entity']]
                        buckets[row['entity']][end] = row
                    elif row['entity']=='client7414':
                        assert end not in clients
                        clients[end] = row
            assert set(buckets)==set(HOSTS)
            cluster_buckets = {}
            for end in sorted(set().union(*(b.keys() for b in buckets.values()))):
                members = [buckets[h].get(end) for h in HOSTS]
                if all(r and r['sample_ok'] for r in members):
                    cluster_buckets[end] = dict(sample_ok=True, bucket_end_ns=end,
                         available_epoch_ns=max(r['available_epoch_ns'] for r in members),
                         total_count=sum(r['total_count'] for r in members))
            window_path = train/dataset/'processed/windows.jsonl'
            handoff = read(train/dataset/'handoff_manifest.json')
            assert sha(window_path)==next(f['sha256'] for f in handoff['files'] if f['path']=='processed/windows.jsonl')
            labels = defaultdict(dict)
            distribution = defaultdict(Counter)
            with window_path.open(encoding='utf8') as stream:
                for line in stream:
                    r = json.loads(line)
                    if r['scope']!='host_oss':
                        continue
                    start, host = r['window_start_ns'], r['entity']
                    assert host not in labels[start]
                    labels[start][host] = r
                    if r['complete']:
                        ticks = [buckets[host].get(start+(i+1)*GRID) for i in range(10)]
                        assert r['n_ticks']==10 and all(t and t['sample_ok'] for t in ticks)
                        assert sum(t['total_count'] for t in ticks)==r['total_iops']
                        y=r['total_iops']; band='zero' if y==0 else 'positive_le_10' if y<=10 else 'positive_10_to_100' if y<=100 else 'positive_100_to_1000' if y<=1000 else 'above_1000'
                        distribution[host][band] += 1
                    else:
                        distribution[host]['incomplete'] += 1
            cluster = {s:dict(total=sum(r['total_iops'] for r in ms.values()),end=s+SECOND,
                        available=max(r['available_epoch_ns'] for r in ms.values()))
                       for s,ms in labels.items() if set(ms)==set(HOSTS) and all(r['complete'] for r in ms.values())}
            states = label_states(cluster)
            probe_sets = {'oss':{s:probe(cluster_buckets,s) for s in states},
                          'client7414':{s:probe(clients,s) for s in states}}
            events = segments(states)
            for event in events:
                events_out.write(json.dumps(dict(dataset_id=dataset, **event))+'\n')
            # Emit every target second; incomplete cluster labels have an explicit unknown phase.
            for s in sorted(labels):
                row=states.get(s,dict(quiet=None,phase='unknown',actual_cluster=None,reference_total=None))
                phases.write(json.dumps(dict(dataset_id=dataset, window_start_ns=s, **row))+'\n')
            for kind, items in probe_sets.items():
                for s,p in items.items():
                    probes_out.write(json.dumps(dict(dataset_id=dataset,window_start_ns=s,kind=kind,**p))+'\n')
            all_results[dataset] = dict(workload=capture['workload'],index=capture['index'],
                complete_cluster_seconds=len(cluster), unknown_states=sum(v['quiet'] is None for v in states.values()),
                node_distribution={h:dict(c) for h,c in distribution.items()},
                events=events, probes={k:score(states,v) for k,v in probe_sets.items()})
            sources.append(dict(dataset_id=dataset,windows_sha256=sha(window_path),fine_sha256=sha(fine),export_manifest_sha256=sha(manifest)))
            print(dataset, len(cluster), len(events), flush=True)
    dump(out/'v2_d0_source_index.json',dict(index_sha256=sha(index_path),sources=sources,online_time_eligibility=False,
         offline_source_clock_only=True,clock_uncertainty_ns=None,model_service_arrival_proof=False))
    dump(out/'v2_d0_report.json',dict(state='completed',job_id='a_v2_d0_20261123_a01',elapsed_seconds=time.monotonic()-began,
         script_sha256=sha(Path(__file__)),runs=all_results,new_fit_count=0,external_validation_evaluations=0))


if __name__ == '__main__':
    main()
