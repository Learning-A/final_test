"""Explicit R2 observation contract. Fit statistics only on b507."""
from collections import Counter
import hashlib
import json

import numpy as np

from .placement import require_training_host

HOSTS = tuple('idfs'+str(i) for i in (44,45,46,47,64,65,66,67))
COMPONENTS = ('read','write','total')
STATS = ('mean','std','min','max','median')


def feature_contract(hosts=HOSTS):
    hosts = tuple(hosts)
    if hosts != HOSTS:
        raise ValueError('fixed eight-node feature order required')
    names=[]
    for lag in range(1, 6):
        names.extend(f'lag{lag}_{component}' for component in COMPONENTS)
    for component in COMPONENTS:
        names.extend(f'{component}_5s_{stat}' for stat in STATS)
    names.extend(c+'_difference' for c in COMPONENTS)
    names.extend(('latest_read_fraction','latest_total_zero'))
    for lag in range(1,16):
        names.extend('optional_lag%d_%s'%(lag,c) for c in COMPONENTS)
        names.append('optional_lag%d_observed'%lag)
    for component in COMPONENTS:
        names.extend(f'observed_long_{component}_{stat}' for stat in STATS)
    names.extend(('long_observed_fraction','oldest_observed_lag_seconds'))
    for h in hosts:
        names.extend('context_%s_%s'%(h,c) for c in COMPONENTS)
        names.append('context_'+h+'_observed')
    for lag in range(1,21):
        names.extend('fine_lag%d_%s_count'%(lag,c) for c in COMPONENTS)
        names.extend(('fine_lag%d_observed'%lag,'fine_lag%d_age_seconds'%lag))
    names.extend(('fine_observed_fraction','fine_latest_observed_age_seconds'))
    for ticks in (3,5,10,20):
        names.extend(('fine_%dticks_observed_count_sum'%ticks,
                      'fine_%dticks_observed_fraction'%ticks,
                      'fine_%dticks_zero_observed_fraction'%ticks))
    continuous = list(range(33)) + list(range(95, 110))
    observed = {}
    optional_lags = [35 + 4*i for i in range(15)]
    spatial = [112 + 4*i for i in range(8)]
    fine_lags = [144 + 5*i for i in range(20)]
    for start in optional_lags + spatial + fine_lags:
        for column in range(start, start + 3):
            continuous.append(column)
            observed[str(column)] = start + 3
    for i in range(4):
        column = 246 + 3*i
        continuous.append(column)
        observed[str(column)] = column + 1
    continuous=sorted(continuous)
    mask = [34] + [start + 3 for start in optional_lags + spatial + fine_lags]
    fractions=[33,110,244]+[247+3*i for i in range(4)]+[248+3*i for i in range(4)]
    ages=[111,245]+[148+5*i for i in range(20)]
    if len(names)!=258 or len(set(names))!=258 or len(continuous)!=181:
        raise AssertionError('internal schema specification differs')
    contract=dict(schema='r2_observed_scaling_v1',hosts=list(hosts),names=names,
                  continuous=continuous,preserved=[i for i in range(258) if i not in continuous],
                  observed_by_column=observed,binary_columns=mask,fraction_columns=fractions,
                  age_columns=ages,source_schema='masked_host_multiscale_v1+causal_fine20_mask_age_v1',
                  node_feature_count=258,std_ddof=0,min_scale=1)
    return contract


def contract_sha(contract):
    return hashlib.sha256(json.dumps(contract,sort_keys=True,separators=(',',':')).encode()).hexdigest()


def checked_input(values,names,hosts=HOSTS):
    hosts = tuple(hosts)
    contract=feature_contract(hosts)
    if list(names)!=contract['names']:
        raise ValueError('semantic feature names/order differ')
    x=np.asarray(values,dtype=np.float64)
    if x.ndim!=2 or x.shape[1]!=258 or not np.all(np.isfinite(x)):
        raise ValueError('finite feature matrix required')
    if np.any((x[:,contract['binary_columns']]!=0)&(x[:,contract['binary_columns']]!=1)):
        raise ValueError('explicit binary observation masks required')
    if np.any((x[:,contract['fraction_columns']]<0)|(x[:,contract['fraction_columns']]>1)):
        raise ValueError('coverage and fractions must be in [0,1]')
    if np.any(x[:,contract['age_columns']]<0) or np.any(x[:,110]<=0):
        raise ValueError('valid long coverage and nonnegative ages required')
    nonnegative=[i for i in range(258) if i not in (30,31,32)]
    if np.any(x[:,nonnegative]<0):
        raise ValueError('only explicitly signed differences may be negative')
    for column,mask in contract['observed_by_column'].items():
        if np.any(x[x[:,mask]==0,int(column)]!=0):
            raise ValueError('missing observations must retain placeholder zero')
    for start in (144+5*i for i in range(20)):
        if np.any(x[x[:,start+3]==0,start+4]!=0):
            raise ValueError('missing fine age must be zero')
    if np.any(x[x[:,244]==0,245]!=0):
        raise ValueError('no fine observations must retain absent-age placeholder')
    for i in range(4):
        if np.any(x[x[:,247+3*i]==0,248+3*i]!=0):
            raise ValueError('undefined zero fraction must use coverage mask')
    return x,contract


def fit_observed_scaling(values,names,*,source_role,partition,hosts=HOSTS):
    """Only invoke inside a registered server lease, after data/sample checks."""
    require_training_host()
    hosts = tuple(hosts)
    if source_role!='train' or partition!='fit_run01_02':
        raise ValueError('fitting-role input only; no internal development scales')
    x,c=checked_input(values,names,hosts)
    if len(x) == 0:
        raise ValueError('nonempty fitting features required')
    center = np.zeros(258)
    scale = np.ones(258)
    count = np.zeros(258, dtype=np.int64)
    for column in c['continuous']:
        observed=c['observed_by_column'].get(str(column))
        v=x[:,column] if observed is None else x[x[:,observed]>0,column]
        count[column]=len(v)
        if len(v):
            mean = float(np.mean(v))
            std = float(np.std(v, ddof=0))
            if not np.isfinite(mean) or not np.isfinite(std):
                raise ValueError('nonfinite fitting statistics')
            center[column] = mean
            scale[column] = max(1, std)
    return dict(schema=c['schema'],contract_sha256=contract_sha(c),hosts=list(hosts),
                source_role='train',partition='fit_run01_02',center=center.tolist(),
                scale=scale.tolist(),observed_count=count.tolist(),fitting_rows=len(x),
                fitted_on='lmj@b507',normalization_fitted=True)


def transform(values,names,state,hosts=HOSTS):
    """Apply already fixed statistics; safe for local forward-only checks."""
    hosts = tuple(hosts)
    x,c=checked_input(values,names,hosts)
    if (state.get('schema')!=c['schema'] or state.get('contract_sha256')!=contract_sha(c)
            or state.get('hosts')!=list(hosts) or state.get('source_role')!='train'
            or state.get('partition')!='fit_run01_02'):
        raise ValueError('pinned fitting-state identity differs')
    center = np.asarray(state['center'], dtype=np.float64)
    scale = np.asarray(state['scale'], dtype=np.float64)
    counts=np.asarray(state['observed_count'])
    if (center.shape!=(258,) or scale.shape!=(258,) or counts.shape!=(258,)
            or not np.all(np.isfinite(center)) or not np.all(np.isfinite(scale))
            or np.any(scale<1) or counts.dtype.kind not in 'iu' or np.any(counts<0)):
        raise ValueError('finite fixed center/scale and integer counts required')
    keep=c['preserved']
    if np.any(center[keep]!=0) or np.any(scale[keep]!=1) or np.any(counts[keep]!=0):
        raise ValueError('mask/fraction/age columns must remain unscaled')
    empty=[i for i in c['continuous'] if counts[i]==0]
    if np.any(center[empty]!=0) or np.any(scale[empty]!=1):
        raise ValueError('unobserved training columns require fixed defaults')
    out = x.copy()
    columns = c['continuous']
    out[:, columns] = (x[:, columns] - center[columns]) / scale[columns]
    for column, mask in c['observed_by_column'].items():
        out[x[:, mask] == 0, int(column)] = 0
    if not np.all(np.isfinite(out)):
        raise ValueError('nonfinite transformed features')
    return out


def fit_target_scales(actual,*,source_role,partition):
    require_training_host()
    if source_role!='train' or partition!='fit_run01_02':
        raise ValueError('fitting-role labels only')
    y=np.asarray(actual,dtype=np.float64)
    if y.ndim!=2 or y.shape[1]!=8 or len(y)==0 or not np.all(np.isfinite(y)) or np.any(y<0):
        raise ValueError('finite eight-node fitting labels required')
    scales = []
    floors = []
    counts = []
    for column in y.T:
        positive=column[column>0]
        if not len(positive):
            raise ValueError('no positive training target; stop this card')
        counts.append(len(positive))
        scales.append(max(1, float(np.quantile(positive, .5, method='linear'))))
        floors.append(max(1, float(np.quantile(positive, .1, method='linear'))))
    return dict(hosts=list(HOSTS),source_role=source_role,partition=partition,
                s=scales,tau=floors,positive_counts=counts,quantile_method='linear',fitted_on='lmj@b507')


def run_equal_node_weights(run_ids):
    """Fixed run balancing from identities, with no learned label parameters."""
    run_ids = list(run_ids)
    if any(not isinstance(run, str) or not run for run in run_ids):
        raise ValueError('nonempty fitting-run identities required')
    counts=Counter(run_ids)
    if len(counts) != 10:
        raise ValueError('ten verified fitting-run identities required')
    return np.asarray([[1/(10*counts[run]*8)]*8 for run in run_ids],dtype=np.float64)
