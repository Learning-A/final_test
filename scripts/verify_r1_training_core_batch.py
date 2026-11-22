"""Read-only B receipt: pinned 15-capture core, label replay and A coverage audit."""
import argparse,gzip,hashlib,json,sys,zipfile
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'idfs_forecast_lab'))
from idfs_lab.dataset import build_windows
from idfs_lab.fine_inputs import index_fine,SECOND,GRID_NS
from idfs_lab.fine_paired_dataset import paired_capture
from idfs_lab.fine_receipt import file_sha


PINS={
    'index':'b4efd63c33aa44dd124c3553233857a5d9d896e6f4c53e42b2ed8f59747381ce',
    'delivery_index':'975dc0e18d2fa4ee14c08d59a2269f83d602a2e16b5bfb35d389ea1b9257f447',
    'archive':'d6fc9b0e6227d4ea67fbc53c914702c278cf1d4ec2b8d68ec189a538eb9bd0f9',
    'exporter':'962e906415f47a4bd3772f794cdc48db884f2f4968475e6b4ef3e00ab8ad6aaf',
    'original_index':'52e0bf86ff62079e6104a78b58126f59d0f63210ecb405d86b1614d98e92fd2d',
    'train':'889d303141802a9c826497eb3cf5f31ddac7c7a5d4db63c48698c5894d1d4425',
    'topology':'81ddb8350790e2a1e165757e01c9c3d6cdcd328d956179778f1b7298230eb98b',
    'sample_ids':'0458ae7eb0f67db694954034291a2d450a8b6ca2e6f53596b179cc5b3ae0c558',
    'reference_archive':'9f25bff3f4870ccb9f2b7e96e82d79d1d544dc4b627101061d881564a78d6752'}
REFERENCE='dev_20261122_graph_zipf_train_03_a01'
MARGINS=(0,25,50,75,100,200,500)


def require(condition,message):
    if not condition:raise ValueError(message)


def coverage(originals,fine,t0):
    tally={str(m):{'eligible_node_windows':0,'usable_slots':0,'no_usable_slots':0,
                    'at_least_one_last_500ms_slot':0} for m in MARGINS}
    for row in originals:
        t=row['window_start_ns']
        if row['scope']!='host_oss' or not row['complete'] or t<t0+16*SECOND or row['window_end_ns']>t0+600*SECOND:continue
        slacks=[]
        for lag in range(1,21):
            r=fine.get(('host_oss',row['entity'],t-lag*GRID_NS))
            slacks.append(t-30_000_000-max(r['bucket_end_ns'],r['available_epoch_ns']) if r and r['sample_ok'] else None)
        for margin in MARGINS:
            usable=[s is not None and s>=margin*1_000_000 for s in slacks]
            out=tally[str(margin)];out['eligible_node_windows']+=1;out['usable_slots']+=sum(usable)
            out['no_usable_slots']+=int(not any(usable));out['at_least_one_last_500ms_slot']+=int(any(usable[:5]))
    for out in tally.values():out['mean_usable_slots']=out['usable_slots']/out['eligible_node_windows']
    return tally


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in (*PINS,'core','reference','a_coverage','output'):
        p.add_argument('--'+name.replace('_','-'),required=True)
    a=p.parse_args()
    for key,expected in PINS.items():require(file_sha(getattr(a,key))==expected,'pinned '+key+' checksum differs')
    read=lambda path:json.loads(Path(path).read_text(encoding='utf-8'))
    idx=read(a.index);delivery=read(a.delivery_index);original_idx=read(a.original_index)
    require(idx['schema_version']=='r1_training_core_batch_v1' and idx['state']=='delivered','index not delivered v1')
    require(idx['request_id']==delivery['request_id']=='B19-R1-TRAIN-CORE-20261122-01','request identity differs')
    require(idx['source_dataset_index_sha256']==PINS['original_index'] and idx['source_train_zip_sha256']==PINS['train'] and idx['topology_sha256']==PINS['topology'],'source pins differ')
    require(delivery['export_script_sha256']==PINS['exporter'],'executed exporter pin differs')
    with zipfile.ZipFile(a.archive) as z:require(z.read('dataset_index.json')==Path(a.delivery_index).read_bytes(),'embedded delivery index differs')
    old={d['dataset_id']:d for d in original_idx['datasets'] if d['role']=='train'}
    captures=idx['captures'];require(len(captures)==15 and len({c['dataset_id'] for c in captures})==15 and {c['dataset_id'] for c in captures}==set(old),'not exact fifteen train captures')
    new={d['dataset_id']:d for d in delivery['datasets']};require(len(new)==14 and set(new)==set(old)-{REFERENCE},'not exact fourteen new captures')
    a_report=read(a.a_coverage);require(set(a_report['captures'])==set(old),'A timing report scope differs')
    samples=read(a.sample_ids);hosts=[n['entity'] for n in read(a.topology)['storage_nodes']]
    result=dict(request_id=idx['request_id'],status='running',scope='15 original train captures; core/labels/source-clock coverage only',
                pins=PINS,a_coverage_sha256=file_sha(a.a_coverage),audit_script_sha256=file_sha(__file__),
                captures={},totals={'new_complete_windows_recomputed':0,'new_incomplete_flags_matched':0,'fit_seconds':0,'internal_dev_seconds':0},
                coverage={str(m):{'eligible_node_windows':0,'usable_slots':0,'no_usable_slots':0,'at_least_one_last_500ms_slot':0} for m in MARGINS},
                online_time_eligibility=False,clock_bounds_known=False,fitting_performed=False,new_external_validation_model_evaluations=0,formal_acceptance=False)
    with zipfile.ZipFile(a.train) as z:
        for c in captures:
            did=c['dataset_id'];d=old[did];ref=did==REFERENCE
            require((c['source_run_id'],c['source_role'],c['index'],c['workload'])==(d['run_id'],'train',d['index'],d['workload']),'capture role/identity differs: '+did)
            require(c['development_partition']==('internal_run03' if d['index']==3 else 'fit_run01_02'),'partition differs: '+did)
            require(c['clock_uncertainty_ns'] is None and c['online_receive_proof'] is False,'timing assertion requires separate evidence: '+did)
            root=Path(a.reference).resolve() if ref else (Path(a.core)/c['delivery']['directory']).resolve()
            require(ref or Path(a.core).resolve() in root.parents,'core directory escapes root')
            require(c['delivery']['kind']==('existing_release' if ref else 'archive_member'),'delivery mapping differs')
            require(c['delivery']['archive_sha256']==PINS['reference_archive' if ref else 'archive'],'capture archive pin differs')
            mp=root/'export_manifest.json';require(file_sha(mp)==c['export_manifest_sha256'],'export manifest differs: '+did)
            m=read(mp)
            if not ref:require({k:v for k,v in new[did].items() if k!='export_manifest_sha256'}==m,'supplemental index differs from export manifest')
            require((m['dataset_id'],m['source_run_id'],m['source_role'])==(did,d['run_id'],'train'),'manifest identity differs')
            require(all(m[k] is False for k in ('fault_affected','formal_acceptance','collection_started','fitting_performed','original_labels_modified')),'source status differs')
            raw=z.read(did+'/handoff_manifest.json');mh=hashlib.sha256(raw).hexdigest()
            require(mh==d['manifest_sha256']==c['original_handoff_manifest_sha256']==m['input_manifest_sha256'],'original handoff differs')
            original_manifest=json.loads(raw);qraw=z.read(did+'/processed/quality.json');q=json.loads(qraw)
            wraw=z.read(did+'/processed/windows.jsonl')
            declared={r['path']:r['sha256'] for r in original_manifest['files']}
            require(hashlib.sha256(qraw).hexdigest()==m['quality_sha256']==declared['processed/quality.json'],'quality differs')
            require(hashlib.sha256(wraw).hexdigest()==declared['processed/windows.jsonl']==q['windows_sha256'],'original windows differ')
            if 'windows_sha256' in m:require(m['windows_sha256']==q['windows_sha256'],'export windows pin differs')
            hashes=m['source_raw_files_sha256'];hashes=hashes.values() if isinstance(hashes,dict) else hashes
            require(sorted(hashes)==sorted(q['raw_files_sha256'].values()),'source raw hash list differs')
            require(m['topology_sha256']==PINS['topology'] and m['availability_evidence']==c['availability_evidence']==q['availability_evidence'],'topology or time evidence differs')
            for f in m['files']:
                require(Path(f['name']).name==f['name'],'unsafe member name')
                path=(root/f['name']).resolve();require(path.parent==root and path.stat().st_size==f['bytes'] and file_sha(path)==f['sha256'],'member hash/size differs')
            with gzip.open(root/'fine_context.jsonl.gz','rt',encoding='utf-8') as f:fine=index_fine(json.loads(line) for line in f if line.strip())
            require(len(fine)==m['fine_rows'],'fine row count differs')
            if 'valid_fine_rows' in m:require(sum(r['sample_ok'] for r in fine.values())==m['valid_fine_rows'],'valid fine count differs')
            require({h for scope,h,_ in fine if scope=='host_oss'}==set(hosts),'fine topology differs')
            originals=[json.loads(line) for line in wraw.splitlines()];complete=incomplete=0
            if not ref:
                replay={(r['scope'],r['entity'],r['window_start_ns']):r for r in build_windows(fine.values())}
                for r in originals:
                    if r['scope']!='host_oss' and not(r['scope']=='client_nfs_rpc' and r['entity']=='client7414'):continue
                    other=replay.get((r['scope'],r['entity'],r['window_start_ns']))
                    require(other is not None and r['complete']==other['complete'],'completeness differs')
                    if r['complete']:
                        require(all(r[k]==other[k] for k in ('read_iops','write_iops','total_iops','available_epoch_ns')),'count or availability replay differs')
                        complete+=1
                    else:incomplete+=1
                require(complete==m['original_complete_windows_recomputed'] and incomplete==m['original_incomplete_windows_matched'],'reported replay totals differ')
                result['totals']['new_complete_windows_recomputed']+=complete;result['totals']['new_incomplete_flags_matched']+=incomplete
            t0=json.loads(z.read(did+'/recording_start.json'))['epoch_ns'];cv=coverage(originals,fine,t0)
            ac=a_report['captures'][did]
            for margin,counts in cv.items():
                require(counts['eligible_node_windows']==ac['eligible_node_windows'],'coverage label scope differs')
                for k in ('usable_slots','no_usable_slots','at_least_one_last_500ms_slot'):
                    require(counts[k]==ac['margins'][margin][k],'A coverage counter differs: '+k)
                    result['coverage'][margin][k]+=counts[k]
                result['coverage'][margin]['eligible_node_windows']+=counts['eligible_node_windows']
            partition='internal_dev' if d['index']==3 else 'fit'
            expected=[t for dataset,t in samples[partition] if dataset==did]
            paired=paired_capture(originals,list(fine.values()),capture_id=did,run_id=d['run_id'],source_role='train',run_index=d['index'],
                                  workload=d['workload'],hosts=hosts,recording_start_ns=t0,expected_starts=expected)
            result['totals']['internal_dev_seconds' if d['index']==3 else 'fit_seconds']+=paired['accepted_seconds']
            result['captures'][did]=dict(source_role='train',index=d['index'],workload=d['workload'],fine_rows=len(fine),original_manifest_sha256=mh,
                export_manifest_sha256=c['export_manifest_sha256'],complete_windows_recomputed=complete if not ref else None,
                incomplete_flags_matched=incomplete if not ref else None,previously_verified_graph_reused=ref,
                paired_seconds=paired['accepted_seconds'],pinned_sample_set_identical=True,coverage_matches_A=True,coverage=cv,
                raw_hash_scope='original quality hash list matched; raw files not independently delivered')
            print(json.dumps({'verified_capture':did,'paired_seconds':paired['accepted_seconds'],'coverage_matches_A':True}),flush=True)
    for counts in result['coverage'].values():
        n=counts['eligible_node_windows'];counts['mean_usable_slots']=counts['usable_slots']/n
        counts['last_500ms_any_fraction']=counts['at_least_one_last_500ms_slot']/n
    require(result['totals']=={'new_complete_windows_recomputed':82881,'new_incomplete_flags_matched':508,'fit_seconds':5328,'internal_dev_seconds':2661},'batch totals differ')
    require(result['coverage']['0']['eligible_node_windows']==69820,'coverage denominator differs')
    result['status']='verified'
    result['limitations']=['Source-clock margins are sensitivity assumptions, not measured clock/arrival bounds.',
        'Raw collector workstation receive time is not model-service receive time.',
        'Graph full-label receipt reused from prior R0; coverage/sample mapping independently checked here.',
        'No fitting, accuracy comparison, external validation selection or formal/site acceptance.']
    Path(a.output).write_text(json.dumps(result,indent=2)+'\n',encoding='utf-8')
    print(json.dumps({'status':'verified','totals':result['totals'],'coverage':result['coverage']},indent=2))


if __name__=='__main__':main()
