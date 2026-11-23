"""38 causal client/OSS features; source semantics remain separate from labels."""
import math,statistics
from .multiscale_inputs import SECOND,COMPONENTS


def build_io_context(index,start,oss_ids,client='client7414',margin_ms=30):
    if len(oss_ids)!=12 or len(set(oss_ids))!=12:raise ValueError('explicit twelve-OSS topology required')
    if type(start) is not int or start%SECOND or not 0<margin_ms<1000:raise ValueError('bad deadline')
    deadline=start-int(margin_ms*1e6);consumed=[];features=[];names=[]
    def read(scope,entity,second):
        r=index.get((scope,str(entity),second))
        if r is None or not r.get('complete'):return None
        if r['window_start_ns']!=second or r['window_end_ns']!=second+SECOND or r['n_ticks']!=10:raise ValueError('bad historical window')
        if r['available_epoch_ns']<r['window_end_ns']:raise ValueError('available before settlement')
        if r['available_epoch_ns']>deadline:return None
        v=[r[c+'_iops'] for c in COMPONENTS]
        if any(not math.isfinite(x) or x<0 for x in v) or not math.isclose(v[0]+v[1],v[2],rel_tol=1e-8,abs_tol=1e-6):raise ValueError('bad observed IOPS')
        consumed.append(r);return v
    def add(name,v):names.append(name);features.append(float(v))
    for lag in range(1,6):
        v=read('client_nfs_rpc',client,start-(lag+1)*SECOND)
        for c,value in zip(COMPONENTS,v if v is not None else (0,0,0)):add('client_lag%d_%s'%(lag,c),value)
        add('client_lag%d_observed'%lag,int(v is not None))
    values=[v for entity in oss_ids if (v:=read('oss',entity,start-2*SECOND)) is not None]
    for j,c in enumerate(COMPONENTS):
        vs=[v[j] for v in values]
        for stat,value in [('mean',statistics.mean(vs) if vs else 0),('std',statistics.pstdev(vs) if vs else 0),('min',min(vs) if vs else 0),('max',max(vs) if vs else 0),('active_fraction',sum(v>0 for v in vs)/len(vs) if vs else 0)]:add('oss_'+c+'_'+stat,value)
    total=sum(v[2] for v in values)
    add('oss_observed_fraction',len(values)/12)
    add('oss_total_concentration',sum(v[2]**2 for v in values)/total**2 if total else 0)
    add('oss_observed_read_fraction',sum(v[0] for v in values)/total if total else 0)
    return {'schema':'causal_client_oss38_v1','features':features,'feature_names':names,'feature_count':len(features),
            'features_available_ns':max((r['available_epoch_ns'] for r in consumed),default=None),
            'semantics':'client RPC context and observed OSS distribution, never added to actual node labels'}
