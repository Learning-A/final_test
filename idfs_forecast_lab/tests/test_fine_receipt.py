import gzip,json,tempfile,unittest
from pathlib import Path
from idfs_lab.fine_receipt import verify_delivery,file_sha,SECOND,GRID_NS


class FineReceiptTests(unittest.TestCase):
    def fixture(self,root):
        hosts=['node'+str(i) for i in range(8)];t=100*SECOND
        fine=[dict(scope='host_oss',entity=h,bucket_start_ns=t+i*GRID_NS,bucket_end_ns=t+(i+1)*GRID_NS,
                   sample_ok=True,read_count=1,write_count=0,total_count=1,expected_oss=12,observed_oss=12,
                   unit='operations per nominal100ms bucket',available_epoch_ns=t+(i+1)*GRID_NS) for h in hosts for i in range(10)]
        with gzip.open(root/'fine_context.jsonl.gz','wt',encoding='utf-8') as f:
            for r in fine:f.write(json.dumps(r)+'\n')
        with gzip.open(root/'raw_event_context.jsonl.gz','wt',encoding='utf-8') as f:f.write('')
        windows=root/'reference.jsonl'
        windows.write_text(''.join(json.dumps(dict(scope='host_oss',entity=h,window_start_ns=t,complete=True,n_ticks=10,read_iops=10,write_iops=0,total_iops=10))+'\n' for h in hosts),encoding='utf-8')
        m=dict(dataset_id='dataset',source_run_id='run',source_role='validation',formal_acceptance=False,
               fault_affected=False,collection_started=False,fitting_performed=False,original_labels_modified=False,
               fine_rows=80,raw_event_rows=0,diagnostic_event_start_ns=[t],availability_evidence='source_read_end_only_not_online_reception_proof',files=[])
        for name in ('fine_context.jsonl.gz','raw_event_context.jsonl.gz'):
            f=root/name;m['files'].append(dict(name=name,bytes=f.stat().st_size,sha256=file_sha(f)))
        manifest=root/'export_manifest.json';manifest.write_text(json.dumps(m),encoding='utf-8')
        args=[root,file_sha(manifest),windows,file_sha(windows),'dataset','run','validation',hosts,[t]]
        return m,manifest,args

    def test_exact_replay_is_verified_with_online_limitations(self):
        with tempfile.TemporaryDirectory() as tmp:
            m,p,a=self.fixture(Path(tmp));r=verify_delivery(*a)
            self.assertEqual(r['node_event_labels_recomputed'],8)
            self.assertEqual(next(iter(r['cluster_actual'].values())),80)
            self.assertFalse(r['online_reception_clock_and_auxiliary_units_verified'])
            self.assertFalse(r['training_authorized_by_receipt'])

    def test_role_tampering_and_formal_rejected(self):
        for patch in ({'source_role':'train'},{'formal_acceptance':True},{'fault_affected':True}):
            with self.subTest(patch=patch),tempfile.TemporaryDirectory() as tmp:
                m,p,a=self.fixture(Path(tmp));m.update(patch);p.write_text(json.dumps(m));a[1]=file_sha(p)
                with self.assertRaises(ValueError):verify_delivery(*a)

    def test_hash_and_original_label_mismatch_fail(self):
        with tempfile.TemporaryDirectory() as tmp:
            m,p,a=self.fixture(Path(tmp));a[1]='0'*64
            with self.assertRaises(ValueError):verify_delivery(*a)
        with tempfile.TemporaryDirectory() as tmp:
            m,p,a=self.fixture(Path(tmp));s=a[2].read_text().replace('"total_iops": 10','"total_iops": 9');a[2].write_text(s);a[3]=file_sha(a[2])
            with self.assertRaises(ValueError):verify_delivery(*a)

    def test_manifest_path_escape_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            m,p,a=self.fixture(Path(tmp));m['files'][0]['name']='../outside.gz';p.write_text(json.dumps(m));a[1]=file_sha(p)
            with self.assertRaises(ValueError):verify_delivery(*a)


if __name__=='__main__':unittest.main()
