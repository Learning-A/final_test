"""Pure single-capture R1 pairing: fixed labels/samples, separate F0/F1 inputs."""
import math
from .joint import index_windows
from .masked_inputs import build_masked_input
from .fine_inputs import index_fine, build_fine_input, SECOND
from .counters import valid_count


def paired_capture(windows, fine_rows, *, capture_id, run_id, source_role, run_index,
                   workload, hosts, recording_start_ns, expected_starts=None,
                   clock_uncertainty_ns=None, processing_budget_ns=0,
                   availability_evidence='source_read_end_only_not_online_reception_proof'):
    """No fitting; callers must verify source hashes/provenance before invoking.

    Reproduce the prior P2 warm-up and history protocol. Fine missingness never
    removes an accepted sample; disagreement with pinned starts fails closed.
    """
    hosts=tuple(hosts)
    if (source_role!='train' or type(run_index) is not int or run_index not in (1,2,3)
            or not capture_id or not run_id or not workload or len(hosts)!=8
            or len(set(hosts))!=8 or type(recording_start_ns) is not int):
        raise ValueError('explicit train capture, run partition and eight hosts required')
    ix=index_windows(windows)
    fine=index_fine(fine_rows)
    if any(scope=='host_oss' and host not in hosts for scope,host,_ in fine):
        raise ValueError('fine node outside declared topology')
    if expected_starts is not None:
        expected_starts=list(expected_starts)
        if (len(set(expected_starts))!=len(expected_starts)
                or any(type(t) is not int or t%SECOND for t in expected_starts)):
            raise ValueError('pinned starts must be unique integer seconds')
    floor=((recording_start_ns+SECOND-1)//SECOND)*SECOND
    starts=sorted({r['window_start_ns'] for r in ix.values()
                   if r['scope']=='host_oss' and r['entity'] in hosts
                   and recording_start_ns<=r['window_start_ns']
                   and r['window_end_ns']<=recording_start_ns+600*SECOND})
    output=[];rejected={'warmup':0,'label':0,'history':0,'processing_deadline':0}
    for start in starts:
        if start-16*SECOND<floor:
            rejected['warmup']+=1;continue
        labels=[ix.get(('host_oss',h,start)) for h in hosts]
        if any(r is None or not r.get('complete') for r in labels):
            rejected['label']+=1;continue
        for row in labels:
            if row['window_start_ns']!=start or row['window_end_ns']!=start+SECOND or row['n_ticks']!=10:
                raise ValueError('malformed complete label')
            vals=[row[k+'_iops'] for k in ('read','write','total')]
            if any(not valid_count(v) for v in vals) or not math.isclose(vals[0]+vals[1],vals[2]):
                raise ValueError('invalid actual label')
        base=[build_masked_input(ix,start,h,hosts,capture_id,floor) for h in hosts]
        if any(f is None for f in base):
            rejected['history']+=1;continue
        near=[build_fine_input(fine,start,h,capture_id,floor,
                              clock_uncertainty_ns=clock_uncertainty_ns,
                              processing_budget_ns=processing_budget_ns,
                              availability_evidence=availability_evidence) for h in hosts]
        uncertainty=clock_uncertainty_ns or 0
        feature_deadline=near[0]['feature_deadline_ns']
        if any(f['features_available_ns']+uncertainty>feature_deadline for f in base):
            rejected['processing_deadline']+=1;continue
        actual=[r['total_iops'] for r in labels]
        fine_available=[f['features_available_ns'] for f in near if f['features_available_ns'] is not None]
        source_available=max(f['features_available_ns']+uncertainty for f in base)
        output.append(dict(dataset_id=capture_id,run_id=run_id,workload=workload,source_role=source_role,
                           development_partition='internal_run03' if run_index==3 else 'fit_run01_02',
                           window_start_ns=start,history_start_ns=floor,host_order=list(hosts),
                           actual=actual+[sum(actual)],base=[f['features'] for f in base],
                           fine=[f['features'] for f in near],fine_observed=[f['observed_buckets'] for f in near],
                           base_available_upper_ns=source_available,
                           features_available_upper_ns=max([source_available]+fine_available),
                           feature_deadline_ns=feature_deadline,publication_deadline_ns=start-30_000_000,
                           fine_evidence_online_ready=all(f['online_ready'] for f in near),
                           formal_acceptance=False))
    accepted=[r['window_start_ns'] for r in output]
    if expected_starts is not None and accepted!=sorted(expected_starts):
        raise ValueError('paired sample set differs from pinned baseline; register a protocol revision')
    return dict(schema='paired_F0_F1_v1',capture_id=capture_id,source_role=source_role,
                samples=output,rejected=rejected,candidate_seconds=len(starts),accepted_seconds=len(output),
                hosts=list(hosts),fitting_performed=False)


def vector(sample, target, arm):
    if type(target) is not int or not 0<=target<=8 or arm not in ('F0','F0_F1'):
        raise ValueError('fixed node/cluster target and registered arm required')
    if len(sample['base'])!=8 or any(len(x)!=144 for x in sample['base']):
        raise ValueError('base144 feature schema differs')
    if len(sample['fine'])!=8 or any(len(x)!=114 for x in sample['fine']):
        raise ValueError('fine114 feature schema differs')
    if target<8:
        return sample['base'][target]+(sample['fine'][target] if arm=='F0_F1' else [])
    # Match the existing P2 cluster layout: own112 per node, then one spatial32.
    base=sum((x[:112] for x in sample['base']),[])+sample['base'][0][112:]
    return base+(sum(sample['fine'],[]) if arm=='F0_F1' else [])
