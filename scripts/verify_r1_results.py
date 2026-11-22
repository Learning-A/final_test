"""Independent stdlib-only replay of frozen R1 predictions and original labels."""
import argparse
from collections import defaultdict
import gzip
import hashlib
import json
import math
from pathlib import Path
import zipfile


def file_sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def metrics(pairs):
    count=len(pairs);positive=[(p,y) for p,y in pairs if y>0]
    absolute=sum(abs(p-y) for p,y in pairs);actual=sum(y for _,y in pairs)
    mape=100*sum(abs(p-y)/y for p,y in positive)/len(positive) if positive else None
    result=dict(n=count,zero_actual=count-len(positive),mae_iops=absolute/count,
                wape_percent=100*absolute/actual if actual else None,
                mape_percent=mape,accuracy_percent=100-mape if mape is not None else None)
    for name,lo,hi in (('positive_le_10',0,10),('positive_10_to_100',10,100),
                       ('positive_100_to_1000',100,1000),('above_1000',1000,float('inf'))):
        ps=[(p,y) for p,y in pairs if lo<y<=hi]
        result.setdefault('positive_bands',{})[name]=dict(n=len(ps),ape_percent_sum=sum(100*abs(p-y)/y for p,y in ps))
    return result


def compare(a,b,path=''):
    if isinstance(a,dict):
        if set(a)!=set(b):raise ValueError('metric keys differ '+path)
        for k,v in a.items():compare(v,b[k],path+'/'+k)
    elif a is None or b is None:
        if a!=b:raise ValueError('null differs '+path)
    elif not math.isclose(a,b,rel_tol=1e-10,abs_tol=1e-8):
        raise ValueError('metric differs '+path)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('results','train','index','samples','output'):p.add_argument('--'+name,required=True)
    a=p.parse_args();root=Path(a.results)
    report=json.loads((root/'report.json').read_text())
    if report['state']!='completed' or report['formal_acceptance'] or report['online_time_eligibility']:
        raise ValueError('completed offline results required')
    if file_sha(a.train)!=report['data_hashes']['train'] or file_sha(a.index)!=report['data_hashes']['original_index']:
        raise ValueError('original data differs')
    if file_sha(a.samples)!=report['baseline_sample_ids_sha256']:
        raise ValueError('fixed sample list differs')
    if file_sha(root/'predictions.jsonl.gz')!=report['predictions_sha256']:
        raise ValueError('prediction archive differs')
    samples=json.loads(Path(a.samples).read_text());expected=[tuple(k) for k in samples['internal_dev']]
    hosts=['idfs'+str(i) for i in (44,45,46,47,64,65,66,67)]+['cluster']
    datasets={d['dataset_id']:d for d in json.loads(Path(a.index).read_text())['datasets'] if d['role']=='train'}
    labels={}
    with zipfile.ZipFile(a.train) as z:
        for did in {key[0] for key in expected}:
            rows=[json.loads(line) for line in z.read(did+'/processed/windows.jsonl').splitlines()]
            lookup={(r['entity'],r['window_start_ns']):r for r in rows if r['scope']=='host_oss'}
            for name,t in expected:
                if name!=did:continue
                rr=[lookup[(h,t)] for h in hosts[:8]]
                if not all(r['complete'] and r['n_ticks']==10 and r['window_end_ns']==t+1_000_000_000 for r in rr):
                    raise ValueError('original label incomplete')
                ys=[r['total_iops'] for r in rr];labels[(did,t)]=ys+[sum(ys)]
    groups=defaultdict(list);keys=defaultdict(list);frozen={};diagnostics={}
    with gzip.open(root/'predictions.jsonl.gz','rt') as f:
        for line in f:
            row=json.loads(line);margin=str(row['margin_ms']);arm=row['arm'];key=(row['dataset_id'],row['window_start_ns'])
            if arm not in ('F0','F0_F1') or margin not in ('50','100','200'):raise ValueError('unregistered arm')
            d=datasets[key[0]]
            if d['index']!=3 or row['source_role']!='train' or row['development_partition']!='internal_run03':raise ValueError('role leak')
            if row['actual']!=labels[key] or row['raw_predictions']!=row['final_predictions']:raise ValueError('label/output differs')
            if row['online_time_eligibility'] or row['features_available_upper_ns']>row['feature_deadline_ns']:
                raise ValueError('availability assertion differs')
            if row['feature_deadline_ns']!=key[1]-30_000_000-int(margin)*1_000_000:raise ValueError('cutoff differs')
            if any(not math.isfinite(v) or v<0 for v in row['actual']+row['raw_predictions']):raise ValueError('invalid number')
            if arm=='F0':
                if key in frozen:compare(dict(enumerate(row['raw_predictions'])),dict(enumerate(frozen[key])))
                frozen[key]=row['raw_predictions']
            keys[(margin,arm)].append(key)
            w=row['workload']
            if w!=d['workload']:raise ValueError('workload differs')
            for i,h in enumerate(hosts):groups[(margin,arm,w,h)].append((row['raw_predictions'][i],row['actual'][i]))
            groups[(margin,arm,w,'summed_node_cluster')].append((sum(row['raw_predictions'][:8]),row['actual'][8]))
    if len(keys)!=6 or any(v!=expected for v in keys.values()):raise ValueError('sample set/order differs')
    for (margin,arm,w,h),pairs in groups.items():compare(metrics(pairs),report['margins'][margin]['arms'][arm][w][h])
    # Additional fixed diagnostic bands report ordinary-window MAE as well as
    # MAPE contribution. Band definitions were fixed before interpreting output.
    for margin in ('50','100','200'):
        out={}
        for arm in ('F0','F0_F1'):
            ps=[pair for (m,a,w,h),v in groups.items() if m==margin and a==arm and h in hosts[:8] for pair in v]
            out[arm]={name:dict(n=len(v),mae_iops=sum(abs(p-y) for p,y in v)/len(v) if v else None,
                               signed_bias_iops=sum(p-y for p,y in v)/len(v) if v else None)
                      for name,v in [('actual_zero',[(p,y) for p,y in ps if y==0]),
                                     ('positive_le_100',[(p,y) for p,y in ps if 0<y<=100]),
                                     ('positive_above_100',[(p,y) for p,y in ps if y>100])]}
        diagnostics[margin]=out
    for name,meta in report['models'].items():
        if file_sha(root/(name+'.pkl'))!=meta['sha256']:raise ValueError('model hash differs '+name)
    result=dict(status='verified',scope='independent metric/label/hash replay, not live accuracy',
                code_commit=report['code_commit'],report_sha256=file_sha(root/'report.json'),
                predictions_sha256=report['predictions_sha256'],sample_ids_sha256=file_sha(a.samples),
                groups_recomputed=len(groups),prediction_rows=sum(map(len,keys.values())),
                original_node_labels_checked=len(labels)*8,models_sha256_checked=len(report['models']),
                diagnostic_bands=diagnostics,formal_acceptance=False,new_external_validation_evaluations=0)
    Path(a.output).write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result,indent=2))


if __name__=='__main__':main()
