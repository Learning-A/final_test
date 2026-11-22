import gzip,hashlib,json,statistics,sys,zipfile
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'idfs_forecast_lab'))
import argparse
p=argparse.ArgumentParser(description='Read-only Graph R0 evidence audit; no fitting or remote operations')
for name in ('directory','archive','train','validation','index','topology','sample-ids','output'):
    p.add_argument('--'+name,required=True)
p.add_argument('--plot')
args=p.parse_args()
from idfs_lab.fine_inputs import index_fine,build_fine_input,SECOND,GRID_NS
from idfs_lab.fine_receipt import file_sha
from idfs_lab.fine_paired_dataset import paired_capture
from idfs_lab.dataset import build_windows,build_buckets
from idfs_lab.fine_export import FineAccumulator
root=Path(args.directory);m=json.loads((root/'export_manifest.json').read_text())
did='dev_20261122_graph_zipf_train_03_a01';hosts=['idfs'+str(i) for i in (44,45,46,47,64,65,66,67)]
expected='9f25bff3f4870ccb9f2b7e96e82d79d1d544dc4b627101061d881564a78d6752'
expected_inputs={'train': '889d303141802a9c826497eb3cf5f31ddac7c7a5d4db63c48698c5894d1d4425', 'validation': '0c7bcbf682db0b895569211966eed82cf37bd77663c126ab29269aae8cb05ecf', 'index': '52e0bf86ff62079e6104a78b58126f59d0f63210ecb405d86b1614d98e92fd2d', 'topology': '81ddb8350790e2a1e165757e01c9c3d6cdcd328d956179778f1b7298230eb98b', 'sample_ids': '0458ae7eb0f67db694954034291a2d450a8b6ca2e6f53596b179cc5b3ae0c558'}
for key,sha in expected_inputs.items():
    if file_sha(getattr(args,key))!=sha:raise ValueError('pinned '+key+' checksum mismatch')
assert file_sha(args.archive)==expected
for item in m['files']:
    path=root/item['name'];assert path.stat().st_size==item['bytes'] and file_sha(path)==item['sha256']
with zipfile.ZipFile(args.train) as z:
    manifest=z.read(did+'/handoff_manifest.json');assert hashlib.sha256(manifest).hexdigest()==m['input_manifest_sha256']
    old=json.loads(manifest)
    assert old['role']=='train' and old['source_run_id']==m['source_run_id'] and not old['fault_affected']
    quality=json.loads(z.read(did+'/processed/quality.json'))
    assert hashlib.sha256(z.read(did+'/processed/quality.json')).hexdigest()==m['quality_sha256']
    assert sorted(quality['raw_files_sha256'].values())==sorted(m['source_raw_files_sha256'])
    originals=[json.loads(line) for line in z.read(did+'/processed/windows.jsonl').splitlines()]
    t0=json.loads(z.read(did+'/recording_start.json'))['epoch_ns']
with gzip.open(root/'fine_context.jsonl.gz','rt',encoding='utf-8') as f:fine=[json.loads(line) for line in f if line.strip()]
ix=index_fine(fine);replay={(r['scope'],r['entity'],r['window_start_ns']):r for r in build_windows(fine)}
compared=0;incomplete=0
for r in originals:
    if r['scope']!='host_oss' and not (r['scope']=='client_nfs_rpc' and r['entity']=='client7414'):continue
    p=replay.get((r['scope'],r['entity'],r['window_start_ns']))
    assert p is not None and r['complete']==p['complete']
    if r['complete']:
        assert all(p[c+'_iops']==r[c+'_iops'] for c in ('read','write','total'))
        compared+=1
    else:incomplete+=1
assert compared==m['original_complete_windows_recomputed']==5921
sample_ids=json.loads(Path(args.sample_ids).read_text())
starts=[t for dataset,t in sample_ids['internal_dev'] if dataset==did]
paired=paired_capture(originals,fine,capture_id=did,run_id=m['source_run_id'],source_role='train',run_index=3,
                       workload='graph_zipf',hosts=hosts,recording_start_ns=t0,expected_starts=starts)
events=m['diagnostic_event_start_ns'];evidence=[]
for t in events:
    contexts=[build_fine_input(ix,t,h,did,((t0+SECOND-1)//SECOND)*SECOND) for h in hosts]
    series=[]
    for lag in range(20,0,-1):
        rows=[ix.get(('host_oss',h,t-lag*GRID_NS)) for h in hosts]
        legal=all(r is not None and r['sample_ok'] and r['bucket_end_ns']<=t-30_000_000 and r['available_epoch_ns']<=t-30_000_000 for r in rows)
        series.append({'start_offset_ms':-lag*100,'cluster_operations':sum(r['total_count'] for r in rows) if legal else None,'source_time_eligible':legal})
    evidence.append({'target_start_ns':t,'target_cluster_actual':next(r['actual'][8] for r in paired['samples'] if r['window_start_ns']==t),
                     'pre_target_100ms':series,'node_observed_buckets':[c['observed_buckets'] for c in contexts],
                     'online_ready':False,'clock_bound':'unknown','processing_budget_ns':0})
audit=json.loads((root/'graph_event_audit.json').read_text());audit_verified=[]
index=json.loads(Path(args.index).read_text())
for role,archive in [('train','train.zip'),('validation','validation.zip')]:
    ds=next(d for d in index['datasets'] if d['run_id']=='graph_zipf_'+('train_03' if role=='train' else 'validation_01'))
    with zipfile.ZipFile(args.train if role=='train' else args.validation) as z:
        raw=z.read(ds['dataset_id']+'/processed/windows.jsonl');ref={(r['scope'],str(r['entity']),r['window_start_ns']):r for r in map(json.loads,raw.splitlines())}
    for event in [e for e in audit if e['role']==role]:
        t=event['target_start_ns'];oss=[ref.get(('oss',str(i),t)) for i in range(96)]
        assert all(r and r['complete'] for r in oss)
        assert sum(r['total_iops'] for r in oss)==event['oss_total_iops']
        assert ref[('client_nfs_rpc','client7414',t)]['complete']
        assert ref[('client_nfs_rpc','client7414',t)]['total_iops']==event['client7414_iops']
        assert all(ref[('host_oss',h,t)]['complete'] for h in hosts)
        audit_verified.append({'run_id':event['run'],'role':role,'target_start_ns':t,'oss_complete':96,'actual_cluster':event['oss_total_iops'],'client7414_actual':event['client7414_iops'],
                              'A_supplied_nearby_vdbench_min_iops':min(v['vdbench_iops'] for v in event['nearby_vdbench'].values()),'original_vdbench_log_independently_received':False})
with gzip.open(root/'raw_event_context.jsonl.gz','rt',encoding='utf-8') as f:raw=[json.loads(l) for l in f if l.strip()]
acc=FineAccumulator(json.loads(Path(args.topology).read_text()))
for bucket in build_buckets(raw,list(range(96))):acc.add(bucket)
raw_fine=index_fine(acc.rows());raw_compared=0
for t in events:
    for i in range(10):
        for scope,entity in [('host_oss',h) for h in hosts]+[('client_nfs_rpc','client7414')]:
            key=(scope,entity,t+i*GRID_NS);a=raw_fine[key];b=ix[key]
            assert a['sample_ok']==b['sample_ok']
            assert all(a[c+'_count']==b[c+'_count'] for c in ('read','write','total'))
            raw_compared+=1
result={'audit_script_sha256':file_sha(__file__),'audit_dependency_sha256':{name:file_sha(Path(__file__).resolve().parents[1]/'idfs_forecast_lab/idfs_lab'/name) for name in ('fine_inputs.py','fine_paired_dataset.py','dataset.py','counters.py','fine_export.py','masked_inputs.py')},'request_id':'B19-FINE-SCHEMA-SAMPLE-20261122-01','status':'verified','scope':'delivered_graph_train03_core_and_original_label_replay; validation_audit_labels_only',
        'archive_sha256':expected,'export_manifest_sha256':file_sha(root/'export_manifest.json'),'dataset_id':did,'source_role':'train',
        'raw_cumulative_event_buckets_independently_rebuilt':raw_compared,'pinned_input_hashes':expected_inputs,'fine_rows':len(fine),'raw_event_rows':m['raw_event_rows'],'all_original_complete_node_client_windows_recomputed':compared,
        'incomplete_window_flags_matched':incomplete,'source_raw_file_hashes_matched_original_quality':len(m['source_raw_files_sha256']),
        'pinned_internal_dev_seconds':len(starts),'paired_samples_identical_to_prior_baseline':paired['accepted_seconds']==len(starts),
        'event_audit_label_checks':audit_verified,'source_time_only_event_context':evidence,
        'limitations':['Only graph train03 has full fine core; no train01/02 fit range or other workloads.',
                       'Validation three events have audit summaries, not raw fine neighborhood package.',
                       'Online reception and cross-host clock bound unknown; no online eligibility.',
                       'VDbench intervals from A audit only; raw Vdbench/GC logs and auxiliary units not independently verified.',
                       'Export manifest identifies exporter by short upstream commit; exact modified sanitizer source still needed.'],
        'fitting_performed':False,'new_external_validation_model_evaluations':0,'formal_acceptance':False}
Path(args.output).write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps({k:v for k,v in result.items() if k!='source_time_only_event_context'},indent=2))
print('EVENT_NEAR_HISTORY',json.dumps(evidence))

if args.plot:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(2,2,figsize=(11,6),sharex=True,sharey=True)
    for ax,e in zip(axes.flat,result['source_time_only_event_context']):
        points=[v for v in e['pre_target_100ms'] if v['source_time_eligible']]
        ax.plot([(v['start_offset_ms']+100)/1000 for v in points],[v['cluster_operations'] for v in points],marker='o',markersize=3)
        ax.axvspan(-.1,0,color='gray',alpha=.18)
        ax.axvline(0,color='black',linestyle='--',linewidth=.8)
        ax.set_title('Target actual = %s IOPS | %s' % (e['target_cluster_actual'],e['target_start_ns']),fontsize=10)
        ax.grid(alpha=.2)
    fig.suptitle('Graph train03: pre-target context only (source clocks; online arrival unverified)',fontsize=12)
    fig.supylabel('Cluster operations / nominal 100 ms')
    fig.supxlabel('Bucket end relative to target start (s); gray boundary bucket excluded')
    fig.tight_layout(rect=(.02,.03,1,.94))
    fig.savefig(args.plot,dpi=150)
