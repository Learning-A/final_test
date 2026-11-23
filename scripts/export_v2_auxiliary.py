"""Stream existing OSS snapshots into masked node auxiliary buckets; no fitting."""
import argparse
from collections import defaultdict
import gzip
import hashlib
import json
import math
from pathlib import Path
import sys
import time


def number(value):
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value) and value >= 0


def byte_delta(previous, current, name, core_ok):
    a, b = previous.get(name), current.get(name)
    valid = core_ok and all(isinstance(v, int) and not isinstance(v, bool) and v >= 0 for v in (a,b)) and b >= a
    return b-a if valid else None


def aggregate(dataset, host, end, members, expected_ids):
    ids = set(members)
    expected = set(expected_ids)
    complete = ids == expected and all(m['core_ok'] for m in members.values())
    row = dict(dataset_id=dataset, scope='host_oss', entity=host, bucket_start_ns=end-100_000_000,
               bucket_end_ns=end, sample_ok=complete, expected_oss=len(expected),
               observed_oss=sum(m['core_ok'] for m in members.values()),
               available_epoch_ns=max((m['source_end'] for m in members.values()), default=end),
               offline_source_clock_only=True)
    for direction in ('read','write'):
        name=direction+'_bytes'
        valid=complete and all(m[name] is not None for m in members.values())
        row[name+'_valid']=valid
        row[name]=sum(m[name] for m in members.values()) if valid else None
        count=sum(m[direction+'_count'] for m in members.values()) if complete else None
        row[direction+'_count']=count
        row[direction+'_bytes_per_operation']=row[name]/count if valid and count else None
    valid=complete and all(number(m['op_wip']) for m in members.values())
    row['op_wip_valid']=valid
    row['op_wip_sum']=sum(m['op_wip'] for m in members.values()) if valid else None
    row['op_wip_max']=max(m['op_wip'] for m in members.values()) if valid else None
    return row


def export_file(path, expected_hash, dataset, host, expected_ids, output, delta_fn):
    digest=hashlib.sha256()
    previous={}
    frame={}
    tick=None
    counts=defaultdict(int)
    with path.open('rb') as stream:
        for line in stream:
            digest.update(line)
            current=json.loads(line)
            if current['scope']!='oss' or current['host']!=host:
                raise ValueError('unexpected scope or host')
            end=current['scheduled_epoch_ns']
            if tick is not None and end!=tick:
                if end<tick:
                    raise ValueError('source time order moved backwards')
                row=aggregate(dataset,host,tick,frame,expected_ids)
                output.write(json.dumps(row,allow_nan=False,separators=(',',':'))+'\n')
                counts['rows']+=1
                counts['core_valid']+=row['sample_ok']
                for name in ('read_bytes_valid','write_bytes_valid','op_wip_valid'):
                    counts[name]+=row[name]
                frame={}
            tick=end
            entity=str(current['entity'])
            if entity not in expected_ids or entity in frame:
                raise ValueError('unexpected or duplicate OSS entity')
            a=previous.get(entity)
            d=delta_fn(a,current) if a else {'sample_ok':False}
            ok=d['sample_ok']
            frame[entity]=dict(core_ok=ok,source_end=current.get('read_end_epoch_ns',end),
                read_count=d.get('read_count'),write_count=d.get('write_count'),
                read_bytes=byte_delta(a or {},current,'read_bytes',ok),
                write_bytes=byte_delta(a or {},current,'write_bytes',ok),op_wip=current.get('op_wip'))
            previous[entity]=current
        if tick is not None:
            row=aggregate(dataset,host,tick,frame,expected_ids)
            output.write(json.dumps(row,allow_nan=False,separators=(',',':'))+'\n')
            counts['rows']+=1
            counts['core_valid']+=row['sample_ok']
            for name in ('read_bytes_valid','write_bytes_valid','op_wip_valid'):
                counts[name]+=row[name]
    if digest.hexdigest()!=expected_hash:
        raise ValueError('raw source hash differs: '+path.name)
    return dict(counts, source_sha256=digest.hexdigest())


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--dataset',help='One original train dataset for reviewed pilot, otherwise all15')
    p.add_argument('--budget-seconds',type=int,default=1800)
    args=p.parse_args()
    args.output.mkdir(parents=True,exist_ok=False)
    started=time.monotonic()
    sys.path.insert(0,str(Path('../idfs_forecast_lab').resolve()))
    from idfs_lab.counters import counter_delta
    base=Path('.local/handoffs/development-all-20-rounds-20261122-v1')
    datasets=json.loads((base/'dataset_index.json').read_bytes())['datasets']
    selected=[d for d in datasets if d['role']=='train' and (not args.dataset or d['dataset_id']==args.dataset)]
    assert len(selected)==(1 if args.dataset else 15)
    topology=json.loads((base/'telemetry/topology.json').read_bytes())
    # Topology shape is deliberately validated before any raw export.
    groups={row['entity']:row['oss_ids'] for row in topology['storage_nodes']}
    assert len(groups)==8 and all(len(ids)==12 for ids in groups.values())
    results=[]
    for dataset in selected:
        if time.monotonic()-started>args.budget_seconds:
            raise TimeoutError('export card budget exhausted')
        name=dataset['dataset_id']
        q=json.loads((base/'train'/name/'processed/quality.json').read_bytes())
        out=args.output/(name+'.jsonl.gz')
        record={'dataset_id':name,'hosts':{}}
        with gzip.open(out,'wt',encoding='utf8',compresslevel=3) as stream:
            for source, expected_hash in q['raw_files_sha256'].items():
                if not source.startswith('oss_'):
                    continue
                host='idfs'+source.split('.')[-2]
                ids={str(i) for i in groups[host]}
                record['hosts'][host]=export_file(Path(q['source_directory'])/source,expected_hash,name,host,ids,stream,counter_delta)
                if time.monotonic()-started>args.budget_seconds:
                    raise TimeoutError('export card budget exhausted')
        record.update(file=out.name,sha256=hashlib.sha256(out.read_bytes()).hexdigest(),bytes=out.stat().st_size)
        results.append(record)
        (args.output/'manifest.json').write_text(json.dumps(dict(state='partial',datasets=results),indent=2)+'\n',encoding='utf8')
        print(name,'exported',flush=True)
    manifest=dict(state='completed',datasets=results,elapsed_seconds=time.monotonic()-started,
                  script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                  latency_included=False,new_fit_count=0,online_time_eligibility=False)
    (args.output/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n',encoding='utf8')


if __name__=='__main__':
    main()
