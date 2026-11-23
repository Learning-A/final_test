"""A independent standard-library metric replay of the frozen I1 delivery."""
from collections import defaultdict
import gzip
import hashlib
import json
import math
from pathlib import Path
import statistics
import zipfile


def main():
    package=Path('../work/idfs-model-b/.local/v2_i1/b19-v2-i1-aux-inputs-20261123-v1.zip')
    sha=lambda data:hashlib.sha256(data).hexdigest()
    assert sha(package.read_bytes())=='77d96a5c8bf033f0237ecaac0de095794a6d05aa9dcb0485e90a9e57e86b3131'
    old={}
    with gzip.open('.local/a_r1_review/package/results/predictions.jsonl.gz','rt') as stream:
        for r in map(json.loads,stream):
            if r['arm']=='F0_F1' and r['margin_ms']==50:
                old[r['dataset_id'],r['window_start_ns']]=r
    assert len(old)==2661
    with gzip.open('.local/v2_d0/v2_d0_phases.jsonl.gz','rt') as stream:
        phases={(r['dataset_id'],r['window_start_ns']):r['phase'] for r in map(json.loads,stream)}
    with zipfile.ZipFile(package) as z:
        manifest=json.loads(z.read('MANIFEST.json'))
        for m in manifest:
            data=z.read(m['path'])
            assert len(data)==m['bytes'] and sha(data)==m['sha256']
        # Lease was added after the manifest; it is covered by the pinned whole ZIP.
        assert set(z.namelist())=={m['path'] for m in manifest}|{'MANIFEST.json','execution/lease.json'}
        lease=json.loads(z.read('execution/lease.json'))
        assert lease['state']=='completed' and lease['exit_code']==0 and lease['cpu_threads']==4
        report=json.loads(z.read('results/report.json'))
        assert report['new_fit_count']==16 and report['external_validation_evaluations']==0
        hosts=report['host_order']
        seen=defaultdict(set)
        groups=defaultdict(list)
        for r in map(json.loads,gzip.decompress(z.read('results/predictions.jsonl.gz')).splitlines()):
            arm=r['arm'];key=r['dataset_id'],r['window_start_ns']
            assert key not in seen[arm] and key in old
            seen[arm].add(key)
            assert r['actual']==old[key]['actual'][:8] and r['raw_predictions']==r['final_predictions']
            assert r['source_role']=='train' and r['features_available_upper_ns']<=key[1]-80_000_000
            assert r['feature_deadline_ns']==key[1]-80_000_000
            if arm=='C':
                assert r['final_predictions']==old[key]['final_predictions'][:8]
            for host,actual,pred in zip(hosts,r['actual'],r['final_predictions']):
                assert math.isfinite(pred) and pred>=0
                groups[arm,r['workload'],host].append((actual,pred,phases[key]))
        assert set(seen)=={'C','CB','CBP'} and all(keys==set(old) for keys in seen.values())
        summaries={}
        for arm in seen:
            scores=[];maes=defaultdict(list);contribution=defaultdict(float)
            for (ar,workload,host),rows in groups.items():
                if ar!=arm:
                    continue
                positive=sum(a>0 for a,p,phase in rows)
                mape=sum(abs(p-a)/a*100 for a,p,phase in rows if a>0)/positive
                mae=statistics.mean(abs(p-a) for a,p,phase in rows)
                scores.append(mape);maes[workload].append(mae)
                target=report['arms'][arm]['groups'][workload][host]
                assert math.isclose(mape,target['mape_percent'],rel_tol=1e-12)
                assert math.isclose(mae,target['mae'],rel_tol=1e-12)
                for a,p,phase in rows:
                    if a>0:
                        contribution[phase]+=100*abs(p-a)/a/positive/40
            summaries[arm]=dict(macro_mape=statistics.mean(scores),worst=max(scores),
                groups_below10=sum(v<10 for v in scores),workload_mae={w:statistics.mean(v) for w,v in maes.items()},
                phase_contributions=dict(contribution))
        baseline=summaries['C']
        for arm,s in summaries.items():
            s['improvement_vs_C']=1-s['macro_mape']/baseline['macro_mape']
            s['screen_vs_C']=(arm!='C' and s['improvement_vs_C']>=.05 and s['worst']<=baseline['worst']*1.05
                and all(v<=baseline['workload_mae'][w]*1.05 for w,v in s['workload_mae'].items()))
        result=dict(state='A_independently_verified_I1_no_promotion',zip_sha256=sha(package.read_bytes()),
            manifest_members=len(manifest),lease_hash_scope='whole pinned ZIP; not separately listed in manifest',
            prediction_rows=sum(len(v) for v in seen.values()),original_node_labels_checked=2661*8,
            source_label_evidence='exact equality to independently verified R1 original labels; no fresh raw replay',
            node_prediction_pairs_checked=sum(len(v) for v in seen.values())*8,groups_recomputed=len(groups),
            summaries=summaries,models_loaded=False,new_fit_count=0,online_time_eligibility=False,formal_acceptance=False)
        Path('reports/v2_i1_A_independent_review.json').write_text(json.dumps(result,indent=2)+'\n',encoding='utf8')
        print(json.dumps(result))


if __name__=='__main__':
    main()
