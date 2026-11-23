"""Descriptive comparison of four captures per workload; no model evaluation."""
from collections import defaultdict
import hashlib
import itertools
import json
from pathlib import Path
import statistics

import numpy as np


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    root=Path('.local/handoffs/development-all-20-rounds-20261122-v1')
    index=json.loads((root/'dataset_index.json').read_bytes())
    rounds=[]
    traces={}
    for d in index['datasets']:
        folder=root/d['role']/d['dataset_id']
        manifest=json.loads((folder/'handoff_manifest.json').read_bytes())
        windows=folder/'processed/windows.jsonl'
        assert sha(windows)==next(x['sha256'] for x in manifest['files'] if x['path']=='processed/windows.jsonl')
        wm=json.loads((folder/'workload_manifest.json').read_bytes())
        recording=json.loads((folder/'recording_start.json').read_bytes())['epoch_ns']
        # Integer complete seconds strictly inside the first 600 recording seconds.
        # No claim that recording_start is native RD start or matching business phase.
        start=((recording+10**9-1)//10**9)*10**9
        finish=((recording+600*10**9)//10**9)*10**9
        labels=defaultdict(dict)
        with windows.open(encoding='utf8') as stream:
            for row in map(json.loads,stream):
                s=row['window_start_ns']
                if row['scope']=='host_oss' and start<=s<finish:
                    assert row['entity'] not in labels[s]
                    labels[s][row['entity']]=row
        joint={s:v for s,v in labels.items() if len(v)==8 and all(r['complete'] and r['n_ticks']==10 for r in v.values())}
        assert joint
        hosts=sorted(next(iter(joint.values())))
        data=np.array([[joint[s][h]['total_iops'] for h in hosts] for s in sorted(joint)])
        read=np.array([[joint[s][h]['read_iops'] for h in hosts] for s in sorted(joint)])
        total=data.sum(axis=1)
        mean=data.mean(axis=0)
        shares=data.sum(axis=0)/data.sum()
        name=('train'+str(d['index'])) if d['role']=='train' else 'dev_validation'
        row={'dataset_id':d['dataset_id'],'workload':d['workload'],'round':name,'role':d['role'],
             'window_scope':'complete joint-node seconds inside recording_start+[0,600s); not native RD phase alignment',
             'start_offset_ns':start-recording,'valid_joint_seconds':len(joint),'possible_seconds':(finish-start)//10**9,
             'mean_node_iops':float(data.mean()),'mean_cluster_iops':float(total.mean()),
             'median_cluster_iops':float(np.median(total)),'p10_cluster_iops':float(np.quantile(total,.1)),
             'p90_cluster_iops':float(np.quantile(total,.9)),
             'cluster_cv':float(total.std()/total.mean()),
             'node_zero_labels':int((data==0).sum()),'node_positive_le10_labels':int(((data>0)&(data<=10)).sum()),
             'cluster_le10_seconds':int((total<=10).sum()),'oss_read_operation_fraction':float(read.sum()/data.sum()),
             'node_mean_iops':dict(zip(hosts,map(float,mean))), 'node_volume_shares':dict(zip(hosts,map(float,shares))),
             'model_sha256':wm['model_sha256'],'run_config_sha256':wm['config_sha256']['run_current.vdb'],
             'layout_sha256':wm['layout_sha256'],'asset_sha256':wm['asset_sha256'],'data_root':wm['data_root'],
             'windows_sha256':sha(windows)}
        rounds.append(row)
        traces[d['workload'],name]={int((s-start)//10**9):sum(v[h]['total_iops'] for h in hosts) for s,v in joint.items()}
    comparisons={}
    for workload in sorted({r['workload'] for r in rounds}):
        rs=[r for r in rounds if r['workload']==workload]
        pairs=[]
        for a,b in itertools.combinations(rs,2):
            x,y=traces[workload,a['round']],traces[workload,b['round']]
            keys=sorted(set(x)&set(y))
            one=np.array([x[k] for k in keys]);two=np.array([y[k] for k in keys])
            # Do not tune time shift or align on observed dips.
            corr=float(np.corrcoef(one,two)[0,1]) if one.std()>0 and two.std()>0 else None
            smooth=[]
            for start in range(0,600,10):
                ks=list(range(start,start+10))
                if all(k in x and k in y for k in ks):smooth.append((statistics.mean(x[k] for k in ks),statistics.mean(y[k] for k in ks)))
            corr10=float(np.corrcoef(np.array(smooth).T)[0,1]) if len(smooth)>2 and all(np.std(np.array(smooth),axis=0)>0) else None
            shares_a=a['node_volume_shares'];shares_b=b['node_volume_shares']
            pairs.append({'a':a['round'],'b':b['round'],'matched_recording_offsets':len(keys),
                          'cluster_trace_corr_1s':corr,'matched_10s_blocks':len(smooth),'cluster_trace_corr_10s':corr10,
                          'max_node_share_change_percentage_points':max(abs(shares_a[h]-shares_b[h])*100 for h in shares_a)})
        comparisons[workload]={'mean_node_iops_range':[min(r['mean_node_iops'] for r in rs),max(r['mean_node_iops'] for r in rs)],
            'mean_node_iops_max_over_min':max(r['mean_node_iops'] for r in rs)/min(r['mean_node_iops'] for r in rs),
            'same_model_sha':len({r['model_sha256'] for r in rs})==1,'same_run_config_sha':len({r['run_config_sha256'] for r in rs})==1,
            'same_layout_sha':len({r['layout_sha256'] for r in rs})==1,'same_data_root':len({r['data_root'] for r in rs})==1,
            'asset_hashes_equal':all(r['asset_sha256']==rs[0]['asset_sha256'] for r in rs),'pairs':pairs}
    result={'state':'descriptive_four_round_analysis_completed','model_fit_count':0,'new_model_evaluations':0,
        'validation_use':'user-requested descriptive data analysis only; validation remains previously viewed, not a pristine holdout',
        'round_count':len(rounds),'index_sha256':sha(root/'dataset_index.json'), 'rounds':rounds,'comparisons':comparisons,
        'limitations':['Recording offsets are not proven native RD alignment; trace correlations are exploratory descriptive statistics.',
                       'Startup/shutdown/cache and max-rate scheduling can affect traces; no causation inferred.',
                       'No fit or parameter search; validation data description does not authorize model selection on it.']}
    Path('reports/workload_four_round_comparison.json').write_text(json.dumps(result,indent=2,allow_nan=False)+'\n',encoding='utf8')
    for w,c in comparisons.items():
        print(w,'mean ratio',round(c['mean_node_iops_max_over_min'],4),'same cfg/model',c['same_run_config_sha'],c['same_model_sha'])
        for r in [v for v in rounds if v['workload']==w]:
            print(r['round'],r['valid_joint_seconds'],round(r['mean_node_iops'],2),r['node_positive_le10_labels'],r['node_zero_labels'],round(r['oss_read_operation_fraction'],4))
        print('corr1', [round(p['cluster_trace_corr_1s'],3) for p in c['pairs']], 'corr10',[round(p['cluster_trace_corr_10s'],3) for p in c['pairs']])


if __name__=='__main__':
    main()
