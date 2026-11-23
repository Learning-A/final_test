"""Independently replay bounded raw slices for A's auxiliary-field sample."""
import argparse
from collections import Counter, defaultdict
import hashlib
import json
import math
from pathlib import Path
import re
import time

GRID=100_000_000
EXPECTED_SAMPLE_SHA='d9396c7923dc715d7c581a5ea3086ca3cc24db9bdb4d51f8bf8e7e9d7decf550'


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def number(v):
    return isinstance(v,(int,float)) and not isinstance(v,bool) and math.isfinite(v) and v>=0


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--a-root',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    args=p.parse_args()
    if args.output.exists():
        raise ValueError('Completed sample verification must not be repeated')
    started=time.monotonic()
    sample_dir=args.a_root/'.local/v2_d1_sample_a01'
    sample=sample_dir/'sample_pairs.jsonl'
    if sha(sample)!=EXPECTED_SAMPLE_SHA:
        raise ValueError('D1 sample pin differs')
    audit=json.loads((sample_dir/'report.json').read_text(encoding='utf-8'))
    if audit['sample_sha256']!=EXPECTED_SAMPLE_SHA or len(audit['datasets'])!=3:
        raise ValueError('Sample audit differs')
    delivered={}
    for line in sample.read_text(encoding='utf-8').splitlines():
        r=json.loads(line)
        key=(r['dataset_id'],r['scope'],r['host'],str(r['entity']),r['bucket_end_ns'])
        if key in delivered:
            raise ValueError('Duplicate sample key')
        delivered[key]=r
    base=args.a_root/'.local/handoffs/development-all-20-rounds-20261122-v1'
    tick_pattern=re.compile(rb'"scheduled_epoch_ns":\s*(\d+)')
    expected=set()
    totals=Counter()
    node_buckets=defaultdict(lambda:dict(entities=set(),read_bytes=0,write_bytes=0,read_count=0,write_count=0,op_wip_sum=0.0,op_wip_max=0.0))
    datasets=[]
    for item in audit['datasets']:
        dataset=item['dataset_id']
        directory=base/'train'/dataset
        quality=json.loads((directory/'processed/quality.json').read_text(encoding='utf-8'))
        recording=json.loads((directory/'recording_start.json').read_text(encoding='utf-8'))
        start=(recording['epoch_ns']//GRID+300)*GRID
        if start!=item['sample_start_ns'] or item['sample_end_ns']!=start+4*GRID:
            raise ValueError('The sample is not the registered 400ms slice')
        channels=defaultdict(list)
        source_slices={}
        for source in item['sources']:
            name=source['file']
            if source['expected_sha256']!=quality['raw_files_sha256'][name]:
                raise ValueError('Inherited source inventory pin differs')
            selected_bytes=bytearray()
            n=0
            with (Path(quality['source_directory'])/name).open('rb') as stream:
                for line in stream:
                    m=tick_pattern.search(line)
                    if not m:
                        continue
                    tick=int(m[1])
                    if tick>start+4*GRID:
                        break
                    if tick<start:
                        continue
                    raw=json.loads(line)
                    channels[(raw['scope'],raw['host'],str(raw['entity']))].append(raw)
                    selected_bytes.extend(line)
                    n+=1
            if n!=source['sample_rows']:
                raise ValueError('Raw sample row count differs')
            source_slices[name]=hashlib.sha256(selected_bytes).hexdigest()
        if len(channels)!=112:
            raise ValueError('Expected 96 OSS and 16 client channels')
        counts=Counter()
        for (scope,host,entity),rows in channels.items():
            rows.sort(key=lambda r:r['scheduled_epoch_ns'])
            if len(rows)!=5 or len({r['scheduled_epoch_ns'] for r in rows})!=5:
                raise ValueError('Incomplete/duplicate sample history')
            for a,b in zip(rows,rows[1:]):
                identity=('scope','entity','host','source_id','boot_id','instance_id','sample_interval_ns')
                if any(a.get(k)!=b.get(k) for k in identity):
                    raise ValueError('Identity discontinuity in sample')
                if not all(r['sample_ok'] and not r.get('clock_step',False) and
                           r['sample_interval_ns']==GRID and abs(r['read_end_epoch_ns']-r['scheduled_epoch_ns'])<=25_000_000 for r in (a,b)):
                    raise ValueError('Invalid raw core sample')
                interval=b['read_end_mono_ns']-a['read_end_mono_ns']
                if b['scheduled_epoch_ns']-a['scheduled_epoch_ns']!=GRID or abs(interval-GRID)>25_000_000:
                    raise ValueError('Nonadjacent grid or invalid observed interval')
                key=(dataset,scope,host,entity,b['scheduled_epoch_ns'])
                expected.add(key)
                row=delivered[key]
                if not row['sample_ok'] or row['source_available_ns']!=b['read_end_epoch_ns'] or row['observed_interval_ns']!=interval:
                    raise ValueError('Availability or interval differs')
                for direction in ('read','write'):
                    name=direction+'_count'
                    if not all(number(r[name]) for r in (a,b)) or b[name]<a[name] or row[name]!=b[name]-a[name]:
                        raise ValueError('Core count difference mismatch')
                counts['valid_pairs']+=1
                if scope=='oss':
                    agg=node_buckets[(dataset,host,b['scheduled_epoch_ns'])]
                    if entity in agg['entities']:
                        raise ValueError('Duplicate OSS in node bucket')
                    agg['entities'].add(entity)
                    for direction in ('read','write'):
                        field=direction+'_bytes'
                        if not all(type(r[field]) is int and r[field]>=0 for r in (a,b)) or b[field]<a[field]:
                            raise ValueError('Invalid/reset byte counter')
                        diff=b[field]-a[field]
                        ops=row[direction+'_count']
                        ratio=diff/ops if ops else None
                        if row[field]!=diff or row[direction+'_bytes_per_operation']!=ratio:
                            raise ValueError('Byte difference or ratio mismatch')
                        agg[field]+=diff
                        agg[direction+'_count']+=ops
                        counts['byte_pairs']+=1
                    if not number(b['op_wip']) or row['op_wip_last']!=b['op_wip']:
                        raise ValueError('Gauge is not the raw endpoint snapshot')
                    agg['op_wip_sum']+=b['op_wip']
                    agg['op_wip_max']=max(agg['op_wip_max'],b['op_wip'])
                    counts['gauge_pairs']+=1
                    for name in ('op_latency','op_r_latency','op_w_latency'):
                        left,right=a['latency'][name],b['latency'][name]
                        dc,ds=right['count']-left['count'],right['sum']-left['sum']
                        latency_row=row[name]
                        if latency_row['count_delta']!=dc or latency_row['sum_delta']!=ds or latency_row['mean_raw_unit']!=(ds/dc if dc>0 and ds>=0 else None):
                            raise ValueError('Diagnostic latency difference mismatch')
                        if latency_row['unit']!='unknown_pending_source_confirmation':
                            raise ValueError('Unsupported latency unit assertion')
                        counts['latency_pairs']+=1
        if dict(counts)!={k:v for k,v in item['checks'].items() if k!='latency_nonmonotonic'}:
            raise ValueError('Audit aggregate counts differ')
        totals.update(counts)
        datasets.append(dict(dataset_id=dataset,channels=112,checks=dict(counts),raw_slice_sha256=source_slices))
    if expected!=set(delivered) or len(expected)!=1344:
        raise ValueError('Unexpected or missing delivered sample row')
    for r in node_buckets.values():
        if len(r['entities'])!=12:
            raise ValueError('Incomplete twelve-OSS aggregation')
    report=dict(state='verified_limited_sample_schema',sample_sha256=EXPECTED_SAMPLE_SHA,
        source_hash_scope='inherited previously verified whole-file inventory plus independently re-read bounded slice hashes',
        scope='three fixed 400ms slices; no assertion of full-round coverage',
        datasets=datasets,checks=dict(totals),complete_node_buckets=len(node_buckets),
        latency_training_eligible=False,online_time_eligibility=False,fit_count=0,
        elapsed_seconds=time.monotonic()-started,runner_sha256=sha(Path(__file__)))
    args.output.write_text(json.dumps(report,indent=2)+'\n',encoding='utf-8',newline='\n')
    print(json.dumps({k:v for k,v in report.items() if k not in ('datasets',)}))


if __name__=='__main__':
    main()
