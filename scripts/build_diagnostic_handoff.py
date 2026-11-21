"""Package an explicitly authorized fault round for exploratory model work."""
import argparse
import json
from pathlib import Path
import sys
import zipfile

from export_handoff import export, digest, verify

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'idfs_forecast_lab'))
from idfs_lab.io import load, read_events, save
from idfs_lab.joint import feature_rows


def build(run, output, site_path):
    run, output = Path(run).resolve(), Path(output).resolve()
    quality=load(run/'processed/quality.json')
    source_hash=digest(run/'processed/windows.jsonl')
    if quality.get('windows_sha256')!=source_hash:
        raise ValueError('source windows provenance mismatch')
    if not (run/'excluded_from_normal_training.json').exists():
        raise ValueError('this exporter is only for explicitly authorized diagnostic rounds')
    if load(run/'workload_manifest.json').get('id')!='ai_training_ses':
        raise ValueError('this diagnostic package is pinned to the AI-training capture')
    site=load(site_path)
    manifest=export(run,output,allow_diagnostic=True)
    windows=list(read_events(output/'processed/windows.jsonl'))
    anchor=load(output/'recording_start.json')['epoch_ns']
    counts={}
    for node in site['storage_nodes']:
        entity=node['hostname']
        path=output/'features'/(entity+'.jsonl');path.parent.mkdir(exist_ok=True)
        count=0
        with path.open('x',encoding='utf8') as stream:
            for row in feature_rows(windows,site,run.name,'diagnostic',
                                    target=dict(scope='host_oss',entity=entity,component='total')):
                if row['window_start_ns']<anchor or row['window_end_ns']>anchor+600*10**9:
                    continue
                row.update(workload='ai_training_ses',source_windows_sha256=source_hash,
                           fault_affected=True,normal_training_allowed=False,
                           exploratory_development_allowed=True)
                stream.write(json.dumps(row,ensure_ascii=False,allow_nan=False)+'\n');count+=1
        counts[entity]=count
    instructions=(
        'This is a single fault-affected AI-training workload capture, explicitly authorized '
        'for exploratory model optimization. It is NOT a normal workload training round or '
        'independent acceptance dataset. Preserve masks and fault labels. Any chronological '
        'split remains within one capture; purge overlapping history/labels at split boundaries. '
        'Do not fit normalization on validation windows. Runtime fitting remains assigned to '
        'experiment A on the authorized GPU server. B may inspect distributions, code, and '
        'inference locally and submit candidate training requests. No normal or formal accuracy '
        'claim may be based on this package. Availability times are offline source-read evidence, '
        'not proof of live feature arrival. Workload source: https://github.com/ChrisLzt/idfs-workloads '
        'commit fb83711a1226f59899ed034b9195402156ef57d6.\n')
    (output/'USAGE.txt').write_text(instructions,encoding='utf8')
    manifest.update(exploratory_development_allowed=True,normal_training_certified=False,
                    independent_validation=False,feature_schema='joint_context_v1',
                    normalization_fitted=False,training_performed=False,
                    examples_by_node=counts,source_windows_sha256=source_hash,
                    workload_source={'url':'https://github.com/ChrisLzt/idfs-workloads',
                                     'commit':'fb83711a1226f59899ed034b9195402156ef57d6'})
    existing={f['path'] for f in manifest['files']}
    for path in sorted(output.rglob('*')):
        if not path.is_file() or path.name=='handoff_manifest.json':continue
        rel=path.relative_to(output).as_posix()
        if rel not in existing:
            manifest['files'].append(dict(path=rel,bytes=path.stat().st_size,sha256=digest(path)))
    save(output/'handoff_manifest.json',manifest)
    verify(output)
    archive=output.with_suffix('.zip')
    with zipfile.ZipFile(archive,'x',compression=zipfile.ZIP_DEFLATED) as z:
        for path in sorted(output.rglob('*')):
            if path.is_file():z.write(path,(Path(output.name)/path.relative_to(output)).as_posix())
    checksum=archive.with_suffix('.zip.sha256')
    checksum.write_text(digest(archive)+'  '+archive.name+'\n',encoding='ascii')
    return dict(archive=str(archive),sha256=digest(archive),bytes=archive.stat().st_size,
                manifest_sha256=digest(output/'handoff_manifest.json'),examples_by_node=counts,
                files=len(manifest['files']))


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--run',required=True);parser.add_argument('--output',required=True)
    parser.add_argument('--site',default='idfs_forecast_lab/config/site.json')
    args=parser.parse_args()
    print(json.dumps(build(args.run,args.output,args.site),ensure_ascii=False))
