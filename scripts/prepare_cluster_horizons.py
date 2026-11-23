"""Pure frozen-feature application and future-label construction; no fitting."""
from collections import Counter
import gzip
import hashlib
import json
from pathlib import Path
import sys
import time
import zipfile

import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'idfs_forecast_lab'))
from idfs_lab.cluster_horizon import collect_future_counts,seconds_from_counts,cluster_features
from idfs_lab.fine_inputs import index_fine
from idfs_lab.r2_scaling import HOSTS,feature_contract,transform


def sha(path):
    with Path(path).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()


def main():
    started=time.monotonic();root=Path(__file__).resolve().parents[1]
    dest=root/'.local/cluster_horizon';dest.mkdir(exist_ok=True)
    if (dest/'prepared.npz').exists():raise ValueError('prepared output exists; do not rebuild')
    sources=root.parent.parent/'collaboration_repo/.local/r0_artifacts'
    training=root.parent.parent/'collaboration_repo/.local/handoffs/development-all-20-rounds-20261122-v1/train.zip'
    paths=dict(core=sources/'r1-train-core.zip',reference=sources/'graph-fine-context-r0-20261122-v1.zip',
               index=sources/'r1_training_core_index_v1.json',train=training,
               arrays=root/'.local/v2_i1/inputs/inputs.npz',samples=root/'.local/r1-frozen/baseline/sample_ids.json',
               metadata=root/'.local/v2_i1/inputs/metadata.json',preprocessing=root/'.local/r2-results/results/preprocessing.json')
    expected=dict(core='d6fc9b0e6227d4ea67fbc53c914702c278cf1d4ec2b8d68ec189a538eb9bd0f9',
        reference='9f25bff3f4870ccb9f2b7e96e82d79d1d544dc4b627101061d881564a78d6752',
        index='b4efd63c33aa44dd124c3553233857a5d9d896e6f4c53e42b2ed8f59747381ce',
        train='889d303141802a9c826497eb3cf5f31ddac7c7a5d4db63c48698c5894d1d4425')
    # Sample digest is taken from the authoritative prior checkpoint to avoid hand transcription.
    prior=json.loads((root/'reports/b19_current_checkpoint.json').read_text(encoding='utf-8'))
    expected['samples']=prior['current']['sample_ids_sha256']
    frozen=json.loads((root/'reports/r1_input_study_v1.json').read_text())
    expected['train']=frozen['data_hashes']['train']
    expected['preprocessing']=json.loads((root/'.local/r2-results/results/report.json').read_text())['preprocessing_sha256']
    manifest={name:dict(path=str(path),sha256=sha(path),bytes=path.stat().st_size) for name,path in paths.items()}
    for name,pin in expected.items():assert manifest[name]['sha256']==pin,name+' checksum mismatch'
    i1reg=json.loads((root/'.local/v2_i1/registration.json').read_text())
    assert manifest['arrays']['sha256']==i1reg['inputs']['arrays']['sha256']
    assert json.loads(paths['samples'].read_text())==json.loads((root/'.local/v2_i1/inputs/sample_ids.json').read_text())
    keys=json.loads(paths['samples'].read_text());meta=json.loads(paths['metadata'].read_text())
    raw=np.load(paths['arrays'],allow_pickle=False);state=json.loads(paths['preprocessing'].read_text())
    index=json.loads(paths['index'].read_text());captures={c['dataset_id']:c for c in index['captures']}
    by_capture={}
    for part in ('fit','internal_dev'):
        for position,(did,t) in enumerate(keys[part]):by_capture.setdefault(did,[]).append((part,position,t))
    out={part:dict(keys=[],positions=[],counts=[],available=[]) for part in ('fit','internal_dev')}
    rejected={};source_evidence={}
    with zipfile.ZipFile(paths['core']) as core,zipfile.ZipFile(paths['reference']) as reference,zipfile.ZipFile(paths['train']) as train:
        for did,positions in by_capture.items():
            c=captures[did];is_ref=c['delivery']['kind']=='existing_release'
            archive=reference if is_ref else core;prefix='' if is_ref else c['delivery']['directory']+'/'
            mbytes=archive.read(prefix+'export_manifest.json')
            assert hashlib.sha256(mbytes).hexdigest()==c['export_manifest_sha256']
            m=json.loads(mbytes);member=next(r for r in m['files'] if r['name']=='fine_context.jsonl.gz')
            contents=archive.read(prefix+member['name'])
            assert len(contents)==member['bytes'] and hashlib.sha256(contents).hexdigest()==member['sha256']
            fine=index_fine(json.loads(line) for line in gzip.decompress(contents).splitlines())
            recording_start=json.loads(train.read(did+'/recording_start.json'))['epoch_ns']
            reject=Counter()
            for part,pos,t in positions:
                assert meta[did]['role']=='train' and meta[did]['index'] in ((1,2) if part=='fit' else (3,))
                counts,reason=collect_future_counts(fine,t,HOSTS,recording_start)
                if reason:reject[reason]+=1;continue
                # Every complete future second must equal the independently sealed original node labels.
                if not np.array_equal(seconds_from_counts(counts)[0],raw[part+'_actual'][pos].sum()):
                    raise ValueError('future first-second label differs from frozen originals')
                assert raw[part+'_available_ns'][pos]<=t-80_000_000
                out[part]['keys'].append([did,t]);out[part]['positions'].append(pos);out[part]['counts'].append(counts)
                out[part]['available'].append(int(raw[part+'_available_ns'][pos]))
            rejected[did]=dict(reject);source_evidence[did]=dict(fine_sha256=member['sha256'],recording_start_ns=recording_start)
    arrays={};newkeys={};counts={}
    for part in ('fit','internal_dev'):
        result=out[part];order=np.argsort(result['positions'])
        positions=np.asarray(result['positions'])[order]
        node_x=raw[part+'_features'][positions]
        fixed=transform(node_x.reshape(-1,258),feature_contract()['names'],state['scaling']).reshape(-1,8,258)
        arrays[part+'_x']=cluster_features(fixed).astype(np.float32)
        arrays[part+'_counts']=np.asarray(result['counts'])[order]
        arrays[part+'_seconds']=seconds_from_counts(arrays[part+'_counts'])
        arrays[part+'_available_ns']=np.asarray(result['available'],dtype=np.int64)[order]
        arrays[part+'_last_second']=node_x[:,:,2].sum(axis=1)
        newkeys[part]=[result['keys'][i] for i in order]
        counts[part]=len(positions)
        assert np.isfinite(arrays[part+'_x']).all()
    np.savez_compressed(dest/'prepared.npz',**arrays)
    (dest/'sample_ids.json').write_text(json.dumps(newkeys,indent=2)+'\n',encoding='utf-8')
    (dest/'metadata.json').write_text(json.dumps(meta,indent=2)+'\n',encoding='utf-8')
    receipt=dict(state='pure_preparation_completed',source_manifest=manifest,source_evidence=source_evidence,
        original_samples={p:len(keys[p]) for p in keys},accepted=counts,rejected_by_capture=rejected,
        future_seconds=10,future_100ms_buckets=100,label_units='100ms operation counts; 1s=sum of ten counts; Hs mean=sum first 10H counts/H',
        source_cutoff='S-80ms',input_feature_count=1840,preprocessing_fit=False,model_fit=False,
        source_role='train',external_validation_read=False,source_clock_only=True,
        clock_alignment_note='No retrospective 20261123 clock offset applied to historical 20261122 training data.',
        elapsed_seconds=time.monotonic()-started)
    receipt['outputs']={name:dict(file=name,sha256=sha(dest/name),bytes=(dest/name).stat().st_size)
                        for name in ('prepared.npz','sample_ids.json','metadata.json')}
    (dest/'receipt.json').write_text(json.dumps(receipt,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(receipt,indent=2))


if __name__=='__main__':main()
