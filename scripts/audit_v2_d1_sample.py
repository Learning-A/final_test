"""Export a bounded existing-raw sample for byte/pressure semantics; no fitting."""
import argparse
from collections import defaultdict
import hashlib
import json
from pathlib import Path
import re
import sys


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--output', type=Path, required=True)
    args = p.parse_args()
    args.output.mkdir(exist_ok=False, parents=True)
    sys.path.insert(0, str(Path('../idfs_forecast_lab').resolve()))
    from idfs_lab.counters import counter_delta
    base = Path('.local/handoffs/development-all-20-rounds-20261122-v1')
    index = json.loads((base/'dataset_index.json').read_bytes())
    selected = [d for d in index['datasets'] if d['role']=='train' and
                (d['workload'], d['index']) in {('ai_inference_ses', 3), ('graph_zipf', 1), ('hpc_wrf', 1)}]
    pattern = re.compile(rb'"scheduled_epoch_ns":\s*(\d+)')
    report = dict(state='sample_only_pending_B_review', unit_status='bytes_and_operations_named_by_collector; latency_seconds_not_yet_source_confirmed', datasets=[])
    with (args.output/'sample_pairs.jsonl').open('x', encoding='utf8') as output:
        for item in selected:
            directory=base/'train'/item['dataset_id']
            quality=json.loads((directory/'processed/quality.json').read_bytes())
            raw=Path(quality['source_directory'])
            recording=json.loads((directory/'recording_start.json').read_bytes())
            start=((recording['epoch_ns']//10**8)+300)*10**8
            end=start+4*10**8
            groups=defaultdict(list)
            sources=[]
            for name in quality['raw_files_sha256']:
                path=raw/name
                count=0
                # A narrow time slice; the whole-file hash is inherited, not re-read here.
                with path.open('rb') as stream:
                    for line in stream:
                        match=pattern.search(line)
                        if not match:
                            continue
                        tick=int(match[1])
                        if tick>end:
                            break
                        if tick<start:
                            continue
                        row=json.loads(line)
                        groups[row['scope'],row['host'],row['entity']].append(row)
                        count+=1
                sources.append(dict(file=name,expected_sha256=quality['raw_files_sha256'][name],hash_scope='previously_verified_inventory_not_rehashed_for_sample',sample_rows=count))
            checks=defaultdict(int)
            for key, rows in groups.items():
                rows.sort(key=lambda r:r['scheduled_epoch_ns'])
                assert len({r['scheduled_epoch_ns'] for r in rows})==len(rows)
                for a,b in zip(rows,rows[1:]):
                    delta=counter_delta(a,b)
                    evidence=dict(dataset_id=item['dataset_id'],scope=key[0],host=key[1],entity=key[2],bucket_end_ns=b['scheduled_epoch_ns'],sample_ok=delta['sample_ok'])
                    if not delta['sample_ok']:
                        evidence['error']=delta['error']
                        checks['invalid_pairs']+=1
                    else:
                        checks['valid_pairs']+=1
                        evidence.update(read_count=delta['read_count'],write_count=delta['write_count'],
                            source_available_ns=delta['available_epoch_ns'], observed_interval_ns=delta['observed_interval_ns'])
                        for direction in ('read','write'):
                            field=direction+'_bytes'
                            if field in a and field in b and b[field]>=a[field]:
                                value=b[field]-a[field]
                                ops=delta[direction+'_count']
                                evidence[field]=value
                                evidence[direction+'_bytes_per_operation']=value/ops if ops else None
                                checks['byte_pairs']+=1
                        if 'op_wip' in b:
                            evidence['op_wip_last']=b['op_wip']
                            checks['gauge_pairs']+=1
                        for name in ('op_latency','op_r_latency','op_w_latency'):
                            la=a.get('latency',{}).get(name,{})
                            lb=b.get('latency',{}).get(name,{})
                            if all('count' in v and 'sum' in v for v in (la,lb)):
                                dc=lb['count']-la['count'];ds=lb['sum']-la['sum']
                                evidence[name]=dict(count_delta=dc,sum_delta=ds,mean_raw_unit=ds/dc if dc>0 and ds>=0 else None,unit='unknown_pending_source_confirmation')
                                checks['latency_pairs']+=1
                                checks['latency_nonmonotonic']+=dc<0 or ds<0
                    output.write(json.dumps(evidence,allow_nan=False)+'\n')
            report['datasets'].append(dict(dataset_id=item['dataset_id'],sample_start_ns=start,sample_end_ns=end,
                channels=len(groups),sources=sources,checks=dict(checks)))
            print(item['dataset_id'],dict(checks),flush=True)
    report['sample_sha256']=hashlib.sha256((args.output/'sample_pairs.jsonl').read_bytes()).hexdigest()
    (args.output/'report.json').write_text(json.dumps(report,indent=2)+'\n',encoding='utf8')


if __name__=='__main__':
    main()
