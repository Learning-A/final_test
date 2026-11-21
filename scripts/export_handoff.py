"""Allow-listed dataset transfer: no credentials, connection manifests or weights."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil

ALLOW = ('processed/windows.jsonl', 'processed/quality.json', 'recording_start.json',
         'workload_manifest.json', 'workload_exit.json', 'workload_quality.json',
         'excluded_from_normal_training.json')


def digest(path):
    result = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024*1024), b''):
            result.update(block)
    return result.hexdigest()


def verify(root):
    root = Path(root).resolve()
    manifest = json.loads((root/'handoff_manifest.json').read_text(encoding='utf-8'))
    for item in manifest['files']:
        path = (root/item['path']).resolve()
        if root not in path.parents or not path.is_file() or path.is_symlink():
            raise ValueError('invalid handoff file path')
        if digest(path) != item['sha256'] or path.stat().st_size != item['bytes']:
            raise ValueError('checksum/size mismatch: '+item['path'])
    return manifest


def export(run, output, allow_diagnostic=False):
    run, output = Path(run).resolve(), Path(output).resolve()
    fault = (run/'excluded_from_normal_training.json').is_file()
    if fault and not allow_diagnostic:
        raise ValueError('fault-affected round: use --allow-diagnostic only for explicit fault analysis')
    if output == run or run in output.parents:
        raise ValueError('output must be separate from source run')
    for name in ('processed/windows.jsonl', 'processed/quality.json', 'recording_start.json',
                 'workload_manifest.json', 'workload_exit.json'):
        if not (run/name).is_file():
            raise ValueError('required file missing: '+name)
    if json.loads((run/'workload_exit.json').read_text(encoding='utf-8'))['exit'] != 0:
        raise ValueError('workload did not complete successfully')
    # Exporting does not certify environment quality or independent validation.
    output.mkdir(parents=True, exist_ok=False)
    files = []
    for name in ALLOW:
        src = run/name
        if not src.exists():
            continue
        if src.is_symlink() or run not in src.resolve().parents:
            raise ValueError('symlink/outside source refused')
        dst = output/name
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(src, dst)
        files.append(dict(path=name, bytes=dst.stat().st_size, sha256=digest(dst)))
    manifest = dict(schema_version=1, source_run_id=run.name, dataset_id=output.name,
                    role='diagnostic' if fault else 'requires_catalog_role_review',
                    fault_affected=fault, normal_training_certified=False,
                    formal_acceptance=False, files=files)
    (output/'handoff_manifest.json').write_text(json.dumps(manifest, ensure_ascii=False, indent=2)+'\n', encoding='utf-8')
    verify(output)
    return manifest


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--run')
    parser.add_argument('--output')
    parser.add_argument('--allow-diagnostic', action='store_true')
    parser.add_argument('--verify')
    args = parser.parse_args()
    if args.verify:
        result = verify(args.verify)
    else:
        if not args.run or not args.output:
            parser.error('--run and --output required for export')
        result = export(args.run, args.output, args.allow_diagnostic)
    print(json.dumps({'dataset_id':result['dataset_id'], 'files':len(result['files']),
                      'fault_affected':result['fault_affected'], 'verified':True}))
