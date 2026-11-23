"""Fixed 20-slot byte and pressure inputs from reviewed node buckets, without fit."""
import math

GRID=100_000_000
SECOND=1_000_000_000
BYTE_NAMES=('read_bytes','write_bytes','read_bytes_per_operation','write_bytes_per_operation')
PRESSURE_NAMES=('op_wip_sum','op_wip_max')


def number(x):
    return isinstance(x,(int,float)) and not isinstance(x,bool) and math.isfinite(x) and x>=0


def index_auxiliary(rows,dataset_id,hosts):
    index={}
    for row in rows:
        end=row['bucket_end_ns']
        if (row['dataset_id']!=dataset_id or row['scope']!='host_oss' or row['entity'] not in hosts
                or type(end) is not int or end%GRID or row['bucket_start_ns']!=end-GRID):
            raise ValueError('Auxiliary identity or absolute 100ms grid differs')
        if type(row['sample_ok']) is not bool or row['expected_oss']!=12 or type(row['available_epoch_ns']) is not int:
            raise ValueError('Explicit validity/availability and twelve OSS required')
        if row['sample_ok'] and row['observed_oss']!=12:
            raise ValueError('Incomplete node cannot be core-valid')
        for direction in ('read','write'):
            name=direction+'_bytes'
            if type(row[name+'_valid']) is not bool:
                raise ValueError('Per-direction byte mask required')
            if row[name+'_valid']:
                if not row['sample_ok'] or type(row[name]) is not int or row[name]<0 or not number(row[direction+'_count']):
                    raise ValueError('Invalid byte aggregate or core counter')
            elif row[name] is not None:
                raise ValueError('Masked byte value must be null')
        if type(row['op_wip_valid']) is not bool:
            raise ValueError('Pressure mask required')
        if row['op_wip_valid']:
            if not row['sample_ok'] or not all(number(row[n]) for n in PRESSURE_NAMES):
                raise ValueError('Invalid pressure snapshot')
            if row['op_wip_max']>row['op_wip_sum']:
                raise ValueError('Maximum pressure exceeds sum')
        elif any(row[n] is not None for n in PRESSURE_NAMES):
            raise ValueError('Masked pressure must remain null')
        key=row['entity'],end
        if key in index:
            raise ValueError('Duplicate auxiliary bucket')
        index[key]=row
    return index


def auxiliary_features(index,target_ns,host,history_start_ns):
    if (type(target_ns) is not int or target_ns%SECOND or type(history_start_ns) is not int
            or history_start_ns%SECOND or history_start_ns>target_ns):
        raise ValueError('Aligned target and capture-local history required')
    cutoff=target_ns-80_000_000
    publication=target_ns-30_000_000
    names_b,values_b,names_p,values_p=[],[],[],[]
    used=[]
    for lag in range(1,21):
        # Same lag layout as fine_inputs: lag1 ends at S and is unavailable.
        end=target_ns-(lag-1)*GRID
        row=index.get((host,end)) if end-GRID>=history_start_ns else None
        visible=bool(row and row['sample_ok'] and max(end,row['available_epoch_ns'])<=cutoff)
        raw_b={}
        if visible:
            for direction in ('read','write'):
                name=direction+'_bytes'
                ok=row[name+'_valid']
                raw_b[name]=(row[name] if ok else 0,ok)
                count=row[direction+'_count']
                ratio_ok=ok and count>0
                raw_b[direction+'_bytes_per_operation']=(row[name]/count if ratio_ok else 0,ratio_ok)
        for name in BYTE_NAMES:
            value,observed=raw_b.get(name,(0,False))
            names_b.extend((f'aux_lag{lag}_{name}',f'aux_lag{lag}_{name}_observed'))
            values_b.extend((float(value),float(observed)))
        names_b.append(f'aux_lag{lag}_byte_age_seconds')
        byte_observed=any(ok for _,ok in raw_b.values())
        values_b.append((publication-end)/SECOND if byte_observed else 0.0)
        pressure_ok=visible and row['op_wip_valid']
        for name in PRESSURE_NAMES:
            names_p.append(f'aux_lag{lag}_{name}')
            values_p.append(float(row[name]) if pressure_ok else 0.0)
        names_p.extend((f'aux_lag{lag}_op_wip_observed',f'aux_lag{lag}_op_wip_age_seconds'))
        values_p.extend((float(pressure_ok),(publication-end)/SECOND if pressure_ok else 0.0))
        if byte_observed or pressure_ok:
            used.append(max(end,row['available_epoch_ns']))
    assert len(values_b)==180 and len(values_p)==80
    return dict(B_names=names_b,B=values_b,P_names=names_p,P=values_p,
                source_available_upper_ns=max(used) if used else None,
                feature_deadline_ns=cutoff,online_time_eligibility=False,
                normalization_fitted=False)
