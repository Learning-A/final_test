"""Causal short/long/spatial inputs; missing history is masked, never a label."""
import math
import statistics
from .multiscale_inputs import build_input, SECOND, COMPONENTS

SCHEMA = 'masked_host_multiscale_v1'


def build_masked_input(index, start, host, hosts, capture_id, history_start_ns, margin_ms=30):
    """Require five timely own seconds; optional 15s context cannot cross split.

    Each optional lag contains three placeholder values AND an observed mask.
    Statistics use observed values only and carry coverage and age. This
    function neither reads target labels nor certifies actual network arrival.
    """
    if type(history_start_ns) is not int or history_start_ns % SECOND:
        raise ValueError('aligned partition history floor required')
    if start-6*SECOND < history_start_ns:
        return None
    short = build_input(index, start, host, hosts, capture_id, scales=(5,), margin_ms=margin_ms)
    if short is None:
        return None
    context_start = short['feature_groups']['spatial_context'][0]
    names = short['feature_names'][:context_start]
    features = short['features'][:context_start]
    groups = {'short': [0, len(features)]}
    lo = len(features)
    observed = []
    end = short['feature_end_ns']
    available = short['features_available_ns']

    def add(name, value):
        names.append(name)
        features.append(float(value))

    for lag in range(1, 16):
        second = start-(lag+1)*SECOND
        row = index.get(('host_oss', host, second)) if second >= history_start_ns else None
        values = None
        if row is not None and row.get('complete'):
            if row['window_start_ns'] != second or row['window_end_ns'] != second+SECOND or row['n_ticks'] != 10:
                raise ValueError('invalid complete long-history window')
            if row['available_epoch_ns'] < row['window_end_ns']:
                raise ValueError('observation before settlement')
            if row['available_epoch_ns'] <= short['deadline_ns']:
                values = [row[c+'_iops'] for c in COMPONENTS]
                if any(not math.isfinite(v) or v < 0 for v in values):
                    raise ValueError('invalid observed history')
                if not math.isclose(values[0]+values[1], values[2], rel_tol=1e-8, abs_tol=1e-6):
                    raise ValueError('inconsistent history total')
                observed.append((lag, values))
                end = max(end, row['window_end_ns'])
                available = max(available, row['available_epoch_ns'])
        for component, value in zip(COMPONENTS, values if values is not None else (0, 0, 0)):
            add('optional_lag%d_%s' % (lag, component), value)
        add('optional_lag%d_observed' % lag, int(values is not None))
    for j, component in enumerate(COMPONENTS):
        vs = [values[j] for _, values in observed]
        for stat, value in [('mean', statistics.mean(vs)), ('std', statistics.pstdev(vs)),
                            ('min', min(vs)), ('max', max(vs)), ('median', statistics.median(vs))]:
            add('observed_long_%s_%s' % (component, stat), value)
    add('long_observed_fraction', len(observed)/15)
    add('oldest_observed_lag_seconds', max(lag+1 for lag, _ in observed))
    groups['long'] = [lo, len(features)]
    lo = len(features)
    names.extend(short['feature_names'][context_start:])
    features.extend(short['features'][context_start:])
    groups['spatial'] = [lo, len(features)]
    return dict(schema=SCHEMA, capture_id=capture_id, host=host, window_start_ns=start,
                history_start_ns=history_start_ns, feature_end_ns=end, features_available_ns=available,
                deadline_ns=short['deadline_ns'], feature_names=names, features=features,
                feature_groups=groups, feature_count=len(features), normalization_fitted=False,
                host_order=list(hosts), long_observed_count=len(observed))
