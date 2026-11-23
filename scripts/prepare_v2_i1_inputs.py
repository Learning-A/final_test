"""Reconstruct C+B/P for fixed train keys, without fitting any parameter."""
import argparse
from collections import Counter
import gzip
import hashlib
import json
from pathlib import Path
import zipfile
import time

import numpy as np

from prepare_r2_inputs import load_r2_inputs, checked_member
from idfs_lab.r2_scaling import HOSTS, feature_contract
from idfs_lab.v2_auxiliary_inputs import index_auxiliary, auxiliary_features


def sha(path):
    with Path(path).open('rb') as f:
        return hashlib.file_digest(f,'sha256').hexdigest()


def load_i1_inputs(paths, auxiliary_dir, manifest_sha):
    root=Path(auxiliary_dir)
    manifest_path=root/'manifest.json'
    if sha(manifest_path)!=manifest_sha:
        raise ValueError('Auxiliary manifest checksum differs')
    manifest=json.loads(manifest_path.read_text(encoding='utf-8'))
    if manifest['state']!='completed' or manifest['latency_included'] or manifest['online_time_eligibility']:
        raise ValueError('Completed offline byte/gauge-only manifest required')
    source=json.loads(Path(paths['original_index']).read_text(encoding='utf-8'))
    datasets={d['dataset_id']:d for d in source['datasets'] if d['role']=='train'}
    entries={d['dataset_id']:d for d in manifest['datasets']}
    if len(entries)!=len(manifest['datasets']) or len(entries)!=15 or set(entries)!=set(datasets):
        raise ValueError('Exact fifteen original training captures required')
    splits,evidence=load_r2_inputs(paths)
    positions={part:{tuple(key):i for i,key in enumerate(data['keys'])} for part,data in splits.items()}
    for data in splits.values():
        data['B']=np.zeros((*data['actual'].shape,180),dtype=np.float64)
        data['P']=np.zeros((*data['actual'].shape,80),dtype=np.float64)
    names=None
    coverage={}
    core_index=json.loads(Path(paths['index']).read_text(encoding='utf-8'))
    captures={d['dataset_id']:d for d in core_index['captures']}
    with zipfile.ZipFile(paths['train']) as training, zipfile.ZipFile(paths['archive']) as fine_core, zipfile.ZipFile(paths['reference_archive']) as fine_reference:
        for did,entry in entries.items():
            path=root/entry['file']
            if Path(entry['file']).name!=entry['file'] or sha(path)!=entry['sha256']:
                raise ValueError('Unsafe member name or auxiliary checksum mismatch')
            if set(entry['hosts'])!=set(HOSTS):
                raise ValueError('Auxiliary host set differs')
            with gzip.open(path,'rt',encoding='utf-8') as f:
                index=index_auxiliary(map(json.loads,f),did,HOSTS)
            capture=captures[did]
            reference=capture['delivery']['kind']=='existing_release'
            archive,prefix=(fine_reference,'') if reference else (fine_core,capture['delivery']['directory']+'/')
            original_manifest=json.loads(checked_member(archive,prefix+'export_manifest.json',capture['export_manifest_sha256']))
            core_member=next(x for x in original_manifest['files'] if x['name']=='fine_context.jsonl.gz')
            core_payload=checked_member(archive,prefix+core_member['name'],core_member['sha256'])
            valid_core=set()
            for row in map(json.loads,gzip.decompress(core_payload).splitlines()):
                if row['scope']!='host_oss':
                    continue
                key=row['entity'],row['bucket_end_ns']
                other=index.get(key)
                if row['sample_ok']:
                    valid_core.add(key)
                    if other is None or not other['sample_ok'] or any(row[k]!=other[k] for k in ('read_count','write_count','available_epoch_ns')):
                        raise ValueError('Auxiliary core differs from frozen valid core')
                elif other is not None and other['sample_ok']:
                    raise ValueError('An invalid core bucket became valid')
            if {k for k,v in index.items() if v['sample_ok']}!=valid_core:
                raise ValueError('Extra valid auxiliary sample outside old core')
            recording=json.loads(training.read(did+'/recording_start.json'))['epoch_ns']
            floor=((recording+999_999_999)//1_000_000_000)*1_000_000_000
            part='internal_dev' if datasets[did]['index']==3 else 'fit'
            assigned=0
            counts=Counter()
            for key,i in positions[part].items():
                if key[0]!=did:
                    continue
                data=splits[part]
                for n,host in enumerate(HOSTS):
                    aux=auxiliary_features(index,key[1],host,floor)
                    name_pair=(aux['B_names'],aux['P_names'])
                    if names is None:
                        names=name_pair
                    elif name_pair!=names:
                        raise ValueError('Auxiliary field order varies between samples')
                    data['B'][i,n]=aux['B']
                    data['P'][i,n]=aux['P']
                    upper=aux['source_available_upper_ns']
                    if upper is not None and upper>key[1]-80_000_000:
                        raise ValueError('Auxiliary timing predicate failed')
                    if upper is not None:
                        data['available_ns'][i]=max(data['available_ns'][i],upper)
                    counts['byte_observed_components']+=sum(aux['B'][j] for j in range(1,180,9))+sum(aux['B'][j] for j in range(3,180,9))
                    counts['pressure_observed_slots']+=sum(aux['P'][j] for j in range(2,80,4))
                assigned+=1
            if assigned==0:
                raise ValueError('Dataset without registered model samples')
            coverage[did]=dict(part=part,seconds=assigned,valid_core_buckets=len(valid_core),**counts)
    c_names=feature_contract()['names']
    schema=dict(C=c_names,CB=c_names+names[0],CBP=c_names+names[0]+names[1])
    if list(map(len,(schema['C'],schema['CB'],schema['CBP'])))!=[258,438,518]:
        raise ValueError('Frozen I1 dimensions differ')
    for data in splits.values():
        if any(not np.isfinite(data[k]).all() for k in ('features','B','P')):
            raise ValueError('Nonfinite new inputs')
    return splits,dict(core=evidence,auxiliary_manifest_sha256=manifest_sha,
        auxiliary_members_sha256={did:e['sha256'] for did,e in entries.items()},
        names=schema,coverage=coverage,fit_count=0,external_validation_read=False,
        online_time_eligibility=False)


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--registration',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    if args.output.exists():
        raise ValueError('Unique input receipt output required')
    reg=json.loads(args.registration.read_text(encoding='utf-8'))
    if sha(__file__)!=reg['loader_sha256']:
        raise ValueError('Input loader hash differs from registration')
    module=Path(__file__).resolve().parents[1]/'idfs_forecast_lab/idfs_lab/v2_auxiliary_inputs.py'
    if sha(module)!=reg['feature_module_sha256']:
        raise ValueError('Auxiliary feature contract changed')
    started=time.monotonic()
    data,evidence=load_i1_inputs(reg['paths'],reg['auxiliary_dir'],reg['auxiliary_manifest_sha256'])
    args.output.mkdir()
    arrays={}
    for part,source in data.items():
        for field in ('features','B','P','actual','available_ns'):
            arrays[part+'_'+field]=source[field]
    npz=args.output/'inputs.npz'
    np.savez_compressed(npz,**arrays)
    ids={part:source['keys'] for part,source in data.items()}
    (args.output/'sample_ids.json').write_text(json.dumps(ids)+'\n',encoding='utf-8',newline='\n')
    evidence.update(state='completed_no_fit_input_receipt',job_id=reg['job_id'],
        elapsed_seconds=time.monotonic()-started,input_array_sha256=sha(npz),
        loader_sha256=sha(__file__),feature_module_sha256=sha(module),
        registration_sha256=sha(args.registration),numpy=np.__version__)
    if evidence['elapsed_seconds']>reg['budget_seconds']:
        raise TimeoutError('Input receipt budget exceeded')
    (args.output/'report.json').write_text(json.dumps(evidence,ensure_ascii=False,indent=2)+'\n',encoding='utf-8',newline='\n')
    print(json.dumps({k:evidence[k] for k in ('state','elapsed_seconds','input_array_sha256')}))


if __name__=='__main__':
    main()
