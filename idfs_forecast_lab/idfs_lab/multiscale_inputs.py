"""No-fit, availability-aware input groups from ONE capture's host windows."""
import math
import statistics

SECOND=1_000_000_000
SCHEMA='host_multiscale_causal_v1'
COMPONENTS=('read','write','total')


def build_input(index,start,host,hosts,capture_id,scales=(5,15,60),margin_ms=30):
    """Labels never participate. Input index must contain one capture only.

    Latest history starts target-2s, ends target-1s. Every consumed row must
    actually be available by target-margin. Neighbor absence uses explicit
    tensor placeholder/mask and never creates a zero actual label.
    """
    hosts=tuple(hosts);scales=tuple(scales)
    if len(hosts)!=8 or len(set(hosts))!=8 or host not in hosts or not capture_id:
        raise ValueError('explicit capture identity and eight distinct hosts required')
    if (not isinstance(start,int) or start%SECOND or not scales or 5 not in scales
            or tuple(sorted(set(scales)))!=scales or any(type(n)!=int or n<5 or n>120 for n in scales)
            or not math.isfinite(margin_ms) or not 0<margin_ms<1000):
        raise ValueError('invalid context or deadline')
    deadline=start-int(margin_ms*1_000_000)
    consumed=[]
    def read(entity,second):
        row=index.get(('host_oss',entity,second))
        if row is None or not row.get('complete'):return None
        if row['window_start_ns']!=second or row['window_end_ns']!=second+SECOND or row['n_ticks']!=10:
            raise ValueError('invalid complete host window')
        available=row['available_epoch_ns']
        if available<row['window_end_ns']:raise ValueError('observation available before settlement')
        if row['window_end_ns']>start-SECOND or available>deadline:return None
        values=[row[c+'_iops'] for c in COMPONENTS]
        if any(not math.isfinite(v) or v<0 for v in values):raise ValueError('invalid historical IOPS')
        if not math.isclose(values[0]+values[1],values[2],rel_tol=1e-8,abs_tol=1e-6):
            raise ValueError('inconsistent read/write total')
        consumed.append(row)
        return values
    history=[read(host,start-i*SECOND) for i in range(2,max(scales)+2)]
    if any(v is None for v in history):return None
    names=[];features=[];groups={}
    def add(name,value):
        names.append(name);features.append(float(value))
    lo=len(features)
    for lag,row in enumerate(history[:5],1):
        for component,value in zip(COMPONENTS,row):add('lag%d_%s'%(lag,component),value)
    groups['read_write_history']=[lo,len(features)]
    lo=len(features)
    for scale in scales:
        for j,component in enumerate(COMPONENTS):
            values=[v[j] for v in history[:scale]]
            for stat,value in [('mean',statistics.mean(values)),('std',statistics.pstdev(values)),
                               ('min',min(values)),('max',max(values)),('median',statistics.median(values))]:
                add('%s_%ds_%s'%(component,scale,stat),value)
    for j,component in enumerate(COMPONENTS):
        add(component+'_difference',history[0][j]-history[1][j])
    total=history[0][2]
    add('latest_read_fraction',history[0][0]/total if total else 0)
    add('latest_total_zero',int(total==0))
    groups['multiscale']=[lo,len(features)]
    lo=len(features)
    for neighbor in hosts:
        values=read(neighbor,start-2*SECOND)
        for component,value in zip(COMPONENTS,values if values is not None else (0,0,0)):
            add('context_'+neighbor+'_'+component,value)
        add('context_'+neighbor+'_observed',int(values is not None))
    groups['spatial_context']=[lo,len(features)]
    return dict(schema=SCHEMA,capture_id=capture_id,host=host,window_start_ns=start,
                feature_end_ns=max(r['window_end_ns'] for r in consumed),
                features_available_ns=max(r['available_epoch_ns'] for r in consumed),
                deadline_ns=deadline,context_seconds=max(scales),scales=list(scales),host_order=list(hosts),
                feature_names=names,features=features,feature_groups=groups,
                feature_count=len(features),normalization_fitted=False)
