"""Check exported auxiliary fields against pinned core counts and zero semantics."""
from collections import Counter
import gzip
import hashlib
import json
from pathlib import Path


def main():
    output=Path('.local/v2_d1_export_a01')
    manifest=json.loads((output/'manifest.json').read_bytes())
    assert manifest['state']=='completed' and len(manifest['datasets'])==15
    checks=[]
    for item in manifest['datasets']:
        dataset=item['dataset_id']
        fine=(Path('.local/v2_graph_pinned/fine_context.jsonl.gz') if 'graph_zipf_train_03' in dataset
              else Path('.local/r0_artifacts/r1-train-core')/dataset/'fine_context.jsonl.gz')
        core={}
        with gzip.open(fine,'rt',encoding='utf8') as stream:
            for r in map(json.loads,stream):
                if r['scope']=='host_oss':
                    core[r['entity'],r['bucket_end_ns']]=r
        path=output/item['file']
        assert hashlib.sha256(path.read_bytes()).hexdigest()==item['sha256']
        counts=Counter();seen=set()
        with gzip.open(path,'rt',encoding='utf8') as stream:
            for r in map(json.loads,stream):
                key=r['entity'],r['bucket_end_ns']
                assert key not in seen;seen.add(key)
                counts['rows']+=1
                old=core.get(key)
                if old and old['sample_ok']:
                    assert r['sample_ok'],key
                    assert all(r[d+'_count']==old[d+'_count'] for d in ('read','write')),key
                    assert r['available_epoch_ns']==old['available_epoch_ns'],key
                    counts['original_valid_core_matched']+=1
                if r['sample_ok']:
                    assert old and old['sample_ok'],key
                for d in ('read','write'):
                    value=r[d+'_bytes'];valid=r[d+'_bytes_valid'];ops=r[d+'_count'];ratio=r[d+'_bytes_per_operation']
                    assert valid==(value is not None)
                    if valid:
                        assert isinstance(value,int) and value>=0
                        assert ratio==value/ops if ops else ratio is None
                    else:
                        assert ratio is None
                    counts[d+'_bytes_valid']+=valid
                assert r['op_wip_valid']==(r['op_wip_sum'] is not None and r['op_wip_max'] is not None)
                counts['op_wip_valid']+=r['op_wip_valid']
        assert counts['original_valid_core_matched']==sum(r['sample_ok'] for r in core.values())
        checks.append(dict(dataset_id=dataset,counts=dict(counts)))
        print(dataset,dict(counts),flush=True)
    result=dict(state='verified_against_pinned_core_all15',datasets=checks,
                manifest_sha256=hashlib.sha256((output/'manifest.json').read_bytes()).hexdigest(),
                scope='all valid core counts, availability, field masks and null/zero semantics; original raw hashes verified by exporter',
                fit_count=0,latency_included=False,online_time_eligibility=False)
    Path('reports/v2_d1_auxiliary_A_verification.json').write_text(json.dumps(result,indent=2)+'\n',encoding='utf8')


if __name__=='__main__':
    main()
