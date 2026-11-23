"""Read-only first-round handoff gate; no collection, normalization or fitting."""
import hashlib
import json
from pathlib import PurePosixPath
import zipfile

TEST_ID='all-workloads-first-round-e2e-v1'
WORKLOADS=('ai_training_ses','ai_inference_ses','bigdata_baleen','graph_zipf','hpc_wrf')


def check_packages(entries):
    """Entries describe A-owned data evidence, not permission to re-label data.

    Each entry pins workload/dataset_id/archive/ZIP hash/manifest hash and two
    explicit review flags. Missing evidence is pending, not capture failure.
    Review flags must come from recorded data QA, never guessed from HEALTH_OK.
    """
    by_workload={}
    for entry in entries:
        workload=entry['workload']
        if workload not in WORKLOADS or workload in by_workload:
            raise ValueError('unknown or duplicate first-round workload')
        by_workload[workload]=entry
    report=dict(test_id=TEST_ID,state='awaiting_handoff',ready=False,formal_acceptance=False,
                independent_validation=False,fitting_performed=False,workloads={})
    identities=set()
    for workload in WORKLOADS:
        entry=by_workload.get(workload)
        if entry is None:
            report['workloads'][workload]={'state':'awaiting_handoff'}
            continue
        path=entry['archive']
        with open(path,'rb') as handle: digest=hashlib.sha256(handle.read()).hexdigest()
        if digest!=entry['archive_sha256']:raise ValueError('archive checksum mismatch: '+workload)
        with zipfile.ZipFile(path) as bundle:
            names=bundle.namelist()
            if len(names)!=len(set(names)):raise ValueError('duplicate ZIP member')
            candidates=[n for n in names if n.endswith('/handoff_manifest.json')]
            if len(candidates)!=1:raise ValueError('ambiguous handoff manifest')
            prefix=candidates[0].rsplit('/',1)[0]+'/'
            raw=bundle.read(candidates[0])
            if hashlib.sha256(raw).hexdigest()!=entry['manifest_sha256']:
                raise ValueError('manifest checksum mismatch')
            m=json.loads(raw)
            if m['dataset_id']!=entry['dataset_id'] or m.get('role')!='train' or m.get('fault_affected') is not False:
                raise ValueError('wrong identity/role or fault capture')
            listed=set()
            for item in m['files']:
                rel=PurePosixPath(item['path'])
                if rel.is_absolute() or '..' in rel.parts or '\\' in item['path'] or ':' in item['path']:
                    raise ValueError('unsafe member path')
                if item['path'] in listed:raise ValueError('duplicate manifest member')
                listed.add(item['path'])
                data=bundle.read(prefix+item['path'])
                if len(data)!=item['bytes'] or hashlib.sha256(data).hexdigest()!=item['sha256']:
                    raise ValueError('member checksum mismatch')
            required={'processed/windows.jsonl','processed/quality.json','recording_start.json','workload_manifest.json','workload_exit.json','workload_quality.json'}
            if not required.issubset(listed):raise ValueError('required evidence missing from manifest')
            wm=json.loads(bundle.read(prefix+'workload_manifest.json'))
            if wm.get('id')!=workload:raise ValueError('workload provenance mismatch')
            recording=json.loads(bundle.read(prefix+'recording_start.json'))
            if recording.get('event')!='recording_start':raise ValueError('wrong recording origin')
            exit_event=json.loads(bundle.read(prefix+'workload_exit.json'))
            if exit_event.get('exit')!=0:raise ValueError('failed workload exit')
        identity=(m['dataset_id'],m.get('source_batch'),m.get('source_attempt'),entry['manifest_sha256'])
        if identity in identities:raise ValueError('same capture supplied more than once')
        identities.add(identity)
        reviewed=(entry.get('time_boundary_reviewed') is True
                  and entry.get('quality_approved_for_development') is True
                  and bool(entry.get('review_evidence_reference')))
        report['workloads'][workload]=dict(state='ready' if reviewed else 'pending_quality_review',
            dataset_id=m['dataset_id'],source_run_id=m['source_run_id'],source_batch=m.get('source_batch'),
            source_attempt=m.get('source_attempt'),archive_sha256=digest,
            manifest_sha256=entry['manifest_sha256'],members_verified=len(listed),
            review_evidence_reference=entry.get('review_evidence_reference'),
            normal_training_certified=m.get('normal_training_certified',False))
    states=[item['state'] for item in report['workloads'].values()]
    if all(s=='ready' for s in states):report.update(state='ready',ready=True)
    elif 'awaiting_handoff' not in states:report['state']='pending_quality_review'
    return report
