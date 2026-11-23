"""No-fit inputs from existing fine buckets; labels are never constructed here."""
import math
import statistics
from .counters import GRID_NS, valid_count

SECOND = 1_000_000_000
SCHEMA = 'causal_fine20_mask_age_v1'


def index_fine(rows):
    out = {}
    for row in rows:
        scope = row['scope']
        if scope not in ('host_oss', 'client_nfs_rpc'):
            raise ValueError('only node and separate client context accepted')
        end = row['bucket_end_ns']
        start = row.get('bucket_start_ns', end-GRID_NS)
        if type(start) is not int or type(end) is not int or start % GRID_NS or end-start != GRID_NS:
            raise ValueError('integer absolute 100ms grid required')
        if type(row['sample_ok']) is not bool:
            raise ValueError('explicit validity required')
        key = (scope, row['entity'], start)
        if key in out:
            raise ValueError('duplicate fine identity/grid')
        if row['sample_ok']:
            values = [row[k+'_count'] for k in ('read', 'write', 'total')]
            if any(not valid_count(v) for v in values) or not math.isclose(values[0]+values[1], values[2], rel_tol=1e-9, abs_tol=1e-9):
                raise ValueError('invalid fine counts')
            if type(row['available_epoch_ns']) is not int:
                raise ValueError('availability timestamp required')
            if scope == 'host_oss':
                if row.get('expected_oss') != 12 or row.get('observed_oss') != 12:
                    raise ValueError('complete twelve OSS required')
                if row.get('unit') != 'operations per nominal100ms bucket':
                    raise ValueError('node bucket unit is not nominal counts')
        out[key] = dict(row, bucket_start_ns=start)
    return out


def _boundary(target_ns, history_start_ns, clock_uncertainty_ns):
    if (type(target_ns) is not int or target_ns % SECOND or type(history_start_ns) is not int
            or history_start_ns % SECOND or history_start_ns > target_ns):
        raise ValueError('aligned target and run/partition history floor required')
    if clock_uncertainty_ns is not None and (type(clock_uncertainty_ns) is not int or clock_uncertainty_ns < 0):
        raise ValueError('clock uncertainty must be a nonnegative integer or unknown')


def build_fine_input(index, target_ns, entity, capture_id, history_start_ns,
                     scope='host_oss', clock_uncertainty_ns=None,
                     availability_evidence='source_read_end_only_not_online_reception_proof',
                     require_online=False, processing_budget_ns=0):
    _boundary(target_ns, history_start_ns, clock_uncertainty_ns)
    if not capture_id or scope not in ('host_oss', 'client_nfs_rpc'):
        raise ValueError('capture identity and context scope required')
    if type(processing_budget_ns) is not int or not 0 <= processing_budget_ns < SECOND:
        raise ValueError('nonnegative subsecond processing budget required')
    online = (availability_evidence == 'inference_host_reception'
              and clock_uncertainty_ns is not None and processing_budget_ns > 0)
    if require_online and not online:
        raise ValueError('online reception, bounded clocks and processing budget not proved')
    deadline = target_ns-30_000_000
    feature_deadline = deadline-processing_budget_ns
    uncertainty = clock_uncertainty_ns or 0
    names, values, consumed = [], [], []
    totals = []
    def add(name, value):
        names.append(name); values.append(float(value))
    for lag in range(1, 21):
        start = target_ns-lag*GRID_NS
        row = index.get((scope, entity, start)) if start >= history_start_ns else None
        ok = bool(row and row['sample_ok'] and row['bucket_end_ns']+uncertainty <= feature_deadline
                  and row['available_epoch_ns']+uncertainty <= feature_deadline)
        counts = [row[k+'_count'] for k in ('read', 'write', 'total')] if ok else [0, 0, 0]
        for component, value in zip(('read', 'write', 'total'), counts):
            add('fine_lag%d_%s_count' % (lag, component), value)
        add('fine_lag%d_observed' % lag, ok)
        add('fine_lag%d_age_seconds' % lag, (deadline-row['bucket_end_ns'])/SECOND if ok else 0)
        totals.append(counts[2] if ok else None)
        if ok:
            consumed.append(row)
    add('fine_observed_fraction', len(consumed)/20)
    add('fine_latest_observed_age_seconds', min((deadline-r['bucket_end_ns'])/SECOND for r in consumed) if consumed else 0)
    for ticks in (3, 5, 10, 20):
        observed = [v for v in totals[:ticks] if v is not None]
        add('fine_%dticks_observed_count_sum' % ticks, sum(observed))
        add('fine_%dticks_observed_fraction' % ticks, len(observed)/ticks)
        add('fine_%dticks_zero_observed_fraction' % ticks, sum(v == 0 for v in observed)/len(observed) if observed else 0)
    return dict(schema=SCHEMA, capture_id=capture_id, scope=scope, entity=entity,
                target_start_ns=target_ns, deadline_ns=deadline, history_start_ns=history_start_ns,
                feature_names=names, features=values, feature_count=len(values),
                feature_end_ns=max((r['bucket_end_ns'] for r in consumed), default=None),
                features_available_ns=max((r['available_epoch_ns']+uncertainty for r in consumed), default=None),
                availability_evidence=availability_evidence, clock_uncertainty_ns=clock_uncertainty_ns,
                feature_deadline_ns=feature_deadline, processing_budget_ns=processing_budget_ns,
                online_ready=online, observed_buckets=len(consumed), normalization_fitted=False)


def build_long_input(index, target_ns, entity, capture_id, history_start_ns,
                     clock_uncertainty_ns=None, processing_budget_ns=0):
    """Optional sixty own seconds, ending target-1s; no required history expansion."""
    _boundary(target_ns, history_start_ns, clock_uncertainty_ns)
    if not capture_id:
        raise ValueError('capture identity required')
    if type(processing_budget_ns) is not int or not 0 <= processing_budget_ns < SECOND:
        raise ValueError('nonnegative subsecond processing budget required')
    deadline = target_ns-30_000_000
    feature_deadline=deadline-processing_budget_ns
    uncertainty = clock_uncertainty_ns or 0
    names, features, observed, consumed = [], [], [], []
    def add(name, value):
        names.append(name); features.append(float(value))
    for lag in range(1, 61):
        start = target_ns-(lag+1)*SECOND
        row = index.get(('host_oss', entity, start)) if start >= history_start_ns else None
        ok = bool(row and row.get('complete') and row['window_end_ns']+uncertainty <= target_ns-SECOND
                  and row['available_epoch_ns']+uncertainty <= feature_deadline)
        vals = [row[k+'_iops'] for k in ('read', 'write', 'total')] if ok else [0, 0, 0]
        if ok:
            if row['window_start_ns'] != start or row['window_end_ns'] != start+SECOND or row['n_ticks'] != 10:
                raise ValueError('malformed long-history window')
            if any(not valid_count(v) for v in vals) or not math.isclose(vals[0]+vals[1], vals[2]):
                raise ValueError('invalid long-history counts')
            observed.append((lag, vals)); consumed.append(row)
        for component, value in zip(('read', 'write', 'total'), vals):
            add('long_lag%d_%s' % (lag, component), value)
        add('long_lag%d_observed' % lag, ok)
        add('long_lag%d_age_seconds' % lag, (deadline-row['window_end_ns'])/SECOND if ok else 0)
    for scale in (5, 15, 60):
        rows = [v for lag, v in observed if lag <= scale]
        add('long_%ds_observed_fraction' % scale, len(rows)/scale)
        for j, component in enumerate(('read', 'write', 'total')):
            vs = [v[j] for v in rows]
            for stat, val in [('mean', statistics.mean(vs) if vs else 0), ('std', statistics.pstdev(vs) if vs else 0)]:
                add('long_%ds_%s_%s' % (scale, component, stat), val)
    return dict(schema='causal_long60_mask_age_v1', capture_id=capture_id, entity=entity,
                feature_names=names, features=features, feature_count=len(features),
                deadline_ns=deadline, observed_seconds=len(observed),
                feature_deadline_ns=feature_deadline, processing_budget_ns=processing_budget_ns,
                features_available_ns=max((r['available_epoch_ns']+uncertainty for r in consumed), default=None),
                clock_uncertainty_ns=clock_uncertainty_ns, online_ready=False, normalization_fitted=False)
