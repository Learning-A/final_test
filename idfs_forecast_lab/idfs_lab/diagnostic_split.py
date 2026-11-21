"""Explicit diagnostic partitions. Never relabel diagnostic rows as normal runs."""
import hashlib
import json
import math
import zipfile

SECOND = 1_000_000_000
ARCHIVE_SHA256 = '0e36d0a5a54e5a74f86a4327688ed09ad62c38bea48317a9da68672ac810ebd6'
MANIFEST_SHA256 = '4951b72711a72bd7cf66ecea9b9f700cea702345d2def0fa68475fc23e859d5d'
HOSTS = ('idfs44','idfs45','idfs46','idfs47','idfs64','idfs65','idfs66','idfs67')


def partition_rows(rows, t0, run_id, source_windows_sha256):
    """Fixed train [0,360), validation [420,600); retain history within each.

    Boundaries use recording_start, never observed errors or fault outcomes.
    Five one-second histories start at target-6s and end at target-1s.
    Source rows keep their actual role/run_id. partition is a separate field.
    """
    output = {'train': [], 'validation': []}
    dropped = {'boundary_or_embargo': 0}
    seen = set()
    for row in rows:
        if (row.get('role') != 'diagnostic' or row.get('fault_affected') is not True
                or row.get('normal_training_allowed') is not False
                or row.get('exploratory_development_allowed') is not True):
            raise ValueError('explicit diagnostic authorization and fault flags required')
        if row.get('run_id') != run_id or row.get('source_windows_sha256') != source_windows_sha256:
            raise ValueError('source provenance mismatch')
        target = row['target']
        if (target.get('scope') != 'host_oss' or target.get('entity') not in HOSTS
                or target.get('component') != 'total'):
            raise ValueError('unexpected diagnostic target')
        start = row['window_start_ns']
        key = (target['entity'], start)
        if key in seen:
            raise ValueError('duplicate source node/window')
        seen.add(key)
        if (start % SECOND or row['window_end_ns'] != start+SECOND
                or row['feature_end_ns'] != start-SECOND
                or row['features_available_ns'] > start-30_000_000):
            raise ValueError('noncausal diagnostic example')
        x, y = row['features'], row['target_read_write']
        if len(x) != 415 or len(y) != 2 or any(not math.isfinite(v) or v < 0 for v in x+y):
            raise ValueError('invalid diagnostic tensor')
        if not math.isclose(sum(y), row['target_iops'], rel_tol=1e-8, abs_tol=1e-6):
            raise ValueError('inconsistent total label')
        if row.get('attribution') != 'storage_context_only':
            raise ValueError('invalid context attribution')
        masks = x[18:399:4]
        if any(v not in (0,1) for v in masks+x[407:415]) or sum(masks) < 92:
            raise ValueError('invalid observation coverage')
        partition = None
        for name, lo, hi in (('train',0,360), ('validation',420,600)):
            if start-6*SECOND >= t0+lo*SECOND and row['window_end_ns'] <= t0+hi*SECOND:
                partition = name
        if partition is None:
            dropped['boundary_or_embargo'] += 1
        else:
            output[partition].append(dict(row, diagnostic_partition=partition))
    for group in output.values():
        group.sort(key=lambda r:(r['window_start_ns'],r['target']['entity']))
    if not all(output.values()):
        raise ValueError('empty diagnostic partition')
    return output, dropped


def read_release(archive):
    """Validate pinned release and members in memory, with no extraction or fit."""
    with open(archive,'rb') as handle:
        digest=hashlib.sha256(handle.read()).hexdigest()
    if digest != ARCHIVE_SHA256:
        raise ValueError('diagnostic archive checksum mismatch')
    prefix='ai_training_fault_exploratory_v2/'
    with zipfile.ZipFile(archive) as bundle:
        raw=bundle.read(prefix+'handoff_manifest.json')
        if hashlib.sha256(raw).hexdigest() != MANIFEST_SHA256:
            raise ValueError('diagnostic manifest checksum mismatch')
        manifest=json.loads(raw)
        for item in manifest['files']:
            data=bundle.read(prefix+item['path'])
            if len(data) != item['bytes'] or hashlib.sha256(data).hexdigest() != item['sha256']:
                raise ValueError('diagnostic member checksum mismatch')
        t0=json.loads(bundle.read(prefix+'recording_start.json'))['epoch_ns']
        rows=[json.loads(line) for host in HOSTS
              for line in bundle.read(prefix+'features/'+host+'.jsonl').splitlines()]
    split, dropped=partition_rows(rows,t0,manifest['source_run_id'],manifest['source_windows_sha256'])
    return split, dict(dataset_id=manifest['dataset_id'], source_run_id=manifest['source_run_id'],
                      source_windows_sha256=manifest['source_windows_sha256'], t0_epoch_ns=t0,
                      archive_sha256=digest, manifest_sha256=MANIFEST_SHA256,
                      partition_kind='single_capture_chronological_diagnostic',
                      independent_validation=False, formal_acceptance=False,
                      normalization_fitted=False, dropped=dropped)
