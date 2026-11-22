"""Offline export of EXISTING capture artifacts. Never collect/connect/train."""
import argparse,gzip,hashlib,json,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'idfs_forecast_lab'))
from idfs_lab.fine_export import FineAccumulator
from idfs_lab.dataset import build_windows


def sha(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda:f.read(1024*1024),b''):h.update(block)
    return h.hexdigest()


def main():
    p=argparse.ArgumentParser();p.add_argument('--handoff-manifest',required=True);p.add_argument('--quality',required=True);p.add_argument('--windows',required=True);p.add_argument('--raw',required=True);p.add_argument('--buckets',required=True);p.add_argument('--topology',required=True);p.add_argument('--event-start-ns',type=int,action='append',required=True);p.add_argument('--output',required=True);a=p.parse_args()
    manifest=Path(a.handoff_manifest);quality=Path(a.quality);raw=Path(a.raw);buckets=Path(a.buckets);dest=Path(a.output)
    if dest.exists():raise ValueError('fresh output required')
    m=json.loads(manifest.read_text(encoding='utf-8'));q=json.loads(quality.read_text(encoding='utf-8'))
    if m['role'] not in ('train','validation','diagnostic') or m.get('formal_acceptance'):raise ValueError('no formal/test data export')
    expected_quality=next(f['sha256'] for f in m['files'] if f['path']=='processed/quality.json')
    if sha(quality)!=expected_quality or q['source_kind']!='counter_collection':raise ValueError('quality provenance mismatch')
    expected_windows=next(f['sha256'] for f in m['files'] if f['path']=='processed/windows.jsonl')
    if sha(Path(a.windows))!=expected_windows or q['windows_sha256']!=expected_windows:raise ValueError('original label checksum')
    if sha(raw/'collection_manifest.json')!=q['source_manifest_sha256']:raise ValueError('raw identity mismatch')
    if (raw/'probe.json').exists():raise ValueError('probe is not capture data')
    acc=FineAccumulator(json.loads(Path(a.topology).read_text(encoding='utf-8')));n=0
    with buckets.open(encoding='utf-8') as f:
        for line in f:
            if line.strip():acc.add(json.loads(line));n+=1
    if n!=q['buckets']:raise ValueError('processed bucket count mismatch')
    replay={(r['scope'],r['entity'],r['window_start_ns']):r for r in build_windows(acc.rows())}
    compared=0
    with Path(a.windows).open(encoding='utf-8') as f:
        for line in f:
            r=json.loads(line)
            if r['scope']!='host_oss' and not (r['scope']=='client_nfs_rpc' and r['entity']=='client7414'):continue
            p=replay.get((r['scope'],r['entity'],r['window_start_ns']))
            if p is None or p['complete']!=r['complete']:raise ValueError('fine export completeness differs from original')
            if r['complete']:
                for component in ('read','write','total'):
                    if p[component+'_iops']!=r[component+'_iops']:raise ValueError('fine export labels differ from original')
                compared+=1
    dest.mkdir();fine=dest/'fine_context.jsonl.gz';fine_rows=0
    with gzip.open(fine,'wt',encoding='utf-8') as f:
        for r in acc.rows():f.write(json.dumps(r,allow_nan=False,separators=(',',':'))+'\n');fine_rows+=1
    allowed={'schema_version','scope','entity','host','boot_id','source_id','instance_id','scheduled_epoch_ns','sample_interval_ns','read_start_epoch_ns','read_end_epoch_ns','read_start_mono_ns','read_end_mono_ns','read_duration_ns','received_inference_epoch_ns','sample_ok','clock_step','error','read_count','write_count','read_bytes','write_bytes','latency','op_wip','rpc_fields','source_query'}
    event=dest/'raw_event_context.jsonl.gz';raw_hashes={};selected=0
    with gzip.open(event,'wt',encoding='utf-8') as out:
        for name,expected in q['raw_files_sha256'].items():
            if Path(name).name!=name:raise ValueError('unsafe raw filename')
            path=raw/name
            if sha(path)!=expected:raise ValueError('raw checksum mismatch: '+name)
            raw_hashes[name]=expected
            with path.open(encoding='utf-8') as f:
                for line in f:
                    r=json.loads(line);t=r['scheduled_epoch_ns']
                    if r['scope']=='oss' or (r['scope']=='client_nfs_rpc' and r['entity']=='client7414'):
                        if any(target-5*10**9<=t<=target+6*10**9 for target in a.event_start_ns):
                            out.write(json.dumps({k:v for k,v in r.items() if k in allowed},allow_nan=False,separators=(',',':'))+'\n');selected+=1
    report={'dataset_id':m['dataset_id'],'source_run_id':m['source_run_id'],'source_role':m['role'],'fault_affected':m['fault_affected'],'formal_acceptance':False,
        'input_manifest_sha256':sha(manifest),'quality_sha256':sha(quality),'topology_sha256':sha(Path(a.topology)),'processed_buckets_sha256':sha(buckets),'source_raw_files_sha256':raw_hashes,
        'fine_rows':fine_rows,'raw_event_rows':selected,'original_complete_windows_recomputed':compared,'diagnostic_event_start_ns':a.event_start_ns,'diagnostic_radius':'minus5sec to plus6sec, never a future predictor input',
        'availability_evidence':q['availability_evidence'],'auxiliary_units':'preserve raw fields; latency/rpc_fields semantics require audit before deriving rates',
        'collection_started':False,'fitting_performed':False,'original_labels_modified':False,'files':[{'name':p.name,'bytes':p.stat().st_size,'sha256':sha(p)} for p in (fine,event)]}
    (dest/'export_manifest.json').write_text(json.dumps(report,indent=2),encoding='utf-8');print(json.dumps(report,ensure_ascii=False))


if __name__=='__main__':main()
