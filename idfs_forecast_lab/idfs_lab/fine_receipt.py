"""Verify a pinned, unpacked existing-data export against pinned original labels."""
import gzip
import hashlib
import json
from pathlib import Path
from .fine_inputs import index_fine, SECOND, GRID_NS


def file_sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda:stream.read(1024*1024),b''):
            h.update(block)
    return h.hexdigest()


def verify_delivery(directory, manifest_sha256, windows, windows_sha256, dataset_id,
                    run_id, source_role, hosts, event_start_ns):
    root=Path(directory).resolve()
    manifest=root/'export_manifest.json'
    if file_sha(manifest)!=manifest_sha256 or file_sha(windows)!=windows_sha256:
        raise ValueError('pinned manifest or original windows checksum mismatch')
    m=json.loads(manifest.read_text(encoding='utf-8'))
    if source_role not in ('train','validation','diagnostic') or m.get('formal_acceptance') is not False:
        raise ValueError('formal data forbidden')
    if (m.get('dataset_id'),m.get('source_run_id'),m.get('source_role'))!=(dataset_id,run_id,source_role):
        raise ValueError('capture identity or original role differs')
    if m.get('fault_affected') is not False:
        raise ValueError('fault data requires a separate diagnostic receipt protocol')
    if any(m.get(k) is not False for k in ('collection_started','fitting_performed','original_labels_modified')):
        raise ValueError('existing-data-only provenance not declared')
    if len(hosts)!=8 or len(set(hosts))!=8 or not event_start_ns:
        raise ValueError('eight distinct hosts and explicit events required')
    if any(type(t) is not int or t%SECOND for t in event_start_ns) or len(set(event_start_ns))!=len(event_start_ns):
        raise ValueError('distinct integer second events required')
    files={}
    for item in m['files']:
        name=item['name']
        if Path(name).name!=name or name in files:
            raise ValueError('duplicate or unsafe manifest filename')
        p=(root/name).resolve()
        if p.parent!=root or not p.is_file() or p.stat().st_size!=item['bytes'] or file_sha(p)!=item['sha256']:
            raise ValueError('export file checksum or path mismatch')
        files[name]=p
    if not {'fine_context.jsonl.gz','raw_event_context.jsonl.gz'} <= set(files):
        raise ValueError('required export files absent')
    with gzip.open(files['fine_context.jsonl.gz'],'rt',encoding='utf-8') as f:
        fine=index_fine(json.loads(line) for line in f if line.strip())
    if len(fine)!=m['fine_rows']:
        raise ValueError('fine row count differs')
    if {entity for scope,entity,_ in fine if scope=='host_oss'}!=set(hosts):
        raise ValueError('node set differs from topology')
    events=set(event_start_ns)
    if not events <= set(m.get('diagnostic_event_start_ns',[])):
        raise ValueError('requested diagnostic events missing from manifest')
    originals={}
    with Path(windows).open(encoding='utf-8') as f:
        for line in f:
            r=json.loads(line)
            if r['scope']=='host_oss' and r['window_start_ns'] in events:
                key=(r['entity'],r['window_start_ns'])
                if key in originals:
                    raise ValueError('duplicate original label')
                originals[key]=r
    compared=0;cluster={}
    for t in event_start_ns:
        actual_cluster=0
        for host in hosts:
            original=originals.get((host,t))
            ticks=[fine.get(('host_oss',host,t+i*GRID_NS)) for i in range(10)]
            if original is None or not original['complete'] or original['n_ticks']!=10:
                raise ValueError('requested original event label is incomplete')
            if any(r is None or not r['sample_ok'] for r in ticks):
                raise ValueError('sample lacks ten valid node ticks for event')
            for part in ('read','write','total'):
                actual=sum(r[part+'_count'] for r in ticks)
                if actual!=original[part+'_iops']:
                    raise ValueError('fine counts do not reproduce original '+part+' label')
            compared+=1;actual_cluster+=original['total_iops']
        cluster[str(t)]=actual_cluster
    raw_count=0
    with gzip.open(files['raw_event_context.jsonl.gz'],'rt',encoding='utf-8') as f:
        for line in f:
            if not line.strip():continue
            r=json.loads(line);t=r['scheduled_epoch_ns']
            if type(t) is not int or not any(e-5*SECOND<=t<=e+6*SECOND for e in m['diagnostic_event_start_ns']):
                raise ValueError('raw event outside declared diagnostic neighborhood')
            raw_count+=1
    if raw_count!=m['raw_event_rows']:
        raise ValueError('raw row count differs')
    return dict(status='verified',scope='hashes_identity_and_requested_event_labels_only',dataset_id=dataset_id,
                source_role=source_role,manifest_sha256=manifest_sha256,windows_sha256=windows_sha256,
                fine_rows=len(fine),raw_event_rows=raw_count,node_event_labels_recomputed=compared,
                cluster_actual=cluster,availability_evidence=m.get('availability_evidence','unknown'),
                online_reception_clock_and_auxiliary_units_verified=False,training_authorized_by_receipt=False,
                future_neighborhood_use='diagnostic_only',fitting_performed=False)
