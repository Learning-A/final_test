import gzip,hashlib,json,subprocess,sys,tempfile,unittest
from pathlib import Path
from idfs_lab.fine_export import FineAccumulator,GRID_NS
from idfs_lab.dataset import build_windows


def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()


class CoreExportCLITests(unittest.TestCase):
    def fixture(self,root):
        raw=root/'raw';raw.mkdir();(raw/'collection_manifest.json').write_text('{"kind":"counter_collection"}')
        (raw/'capture.jsonl').write_text('{"source_id":"DO_NOT_EXPORT_SOURCE_CONFIG"}\n')
        topo={'storage_nodes':[{'entity':'node'+str(i),'oss_ids':list(range(i*12,(i+1)*12))} for i in range(8)]}
        (root/'topology.json').write_text(json.dumps(topo))
        buckets=[dict(scope='oss',entity=str(oss),bucket_start_ns=i*GRID_NS,bucket_end_ns=(i+1)*GRID_NS,
                      read_count=1,write_count=0,total_count=1,sample_ok=True,available_epoch_ns=(i+1)*GRID_NS,
                      observed_interval_ns=GRID_NS) for oss in range(96) for i in range(10)]
        (root/'buckets.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in buckets))
        a=FineAccumulator(topo)
        for r in buckets:a.add(r)
        (root/'windows.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in build_windows(a.rows())))
        q=dict(source_kind='counter_collection',source_manifest_sha256=sha(raw/'collection_manifest.json'),
               windows_sha256=sha(root/'windows.jsonl'),raw_files_sha256={'capture.jsonl':sha(raw/'capture.jsonl')},
               buckets=len(buckets),availability_evidence='source_read_end_only_not_online_reception_proof')
        (root/'quality.json').write_text(json.dumps(q))
        m=dict(dataset_id='synthetic_train03',source_run_id='run03',role='train',fault_affected=False,formal_acceptance=False,
               files=[dict(path='processed/'+name,sha256=sha(root/name)) for name in ('windows.jsonl','quality.json')])
        (root/'handoff_manifest.json').write_text(json.dumps(m))
        script=Path(__file__).resolve().parents[2]/'scripts/export_existing_fine_context.py'
        args=[sys.executable,str(script),'--handoff-manifest',str(root/'handoff_manifest.json'),'--quality',str(root/'quality.json'),
              '--windows',str(root/'windows.jsonl'),'--raw',str(raw),'--buckets',str(root/'buckets.jsonl'),
              '--topology',str(root/'topology.json'),'--output',str(root/'output')]
        return args

    def test_core_only_replays_labels_and_does_not_export_raw_context(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);args=self.fixture(root)
            original={p:sha(p) for p in root.rglob('*') if p.is_file()}
            result=subprocess.run(args+['--core-only'],capture_output=True,text=True,timeout=30)
            self.assertEqual(result.returncode,0,result.stderr)
            m=json.loads((root/'output/export_manifest.json').read_text())
            self.assertEqual(m['export_scope'],'full_capture_core_only')
            self.assertEqual(m['raw_event_rows'],0)
            self.assertEqual(m['diagnostic_event_start_ns'],[])
            self.assertEqual(m['original_complete_windows_recomputed'],8)
            self.assertEqual(m['fine_rows'],90)
            self.assertEqual(len(m['exporter_script_sha256']),64)
            self.assertFalse(m['collection_started']);self.assertFalse(m['fitting_performed'])
            with gzip.open(root/'output/raw_event_context.jsonl.gz','rt') as f:self.assertEqual(f.read(),'')
            self.assertNotIn('DO_NOT_EXPORT_SOURCE_CONFIG',json.dumps(m))
            self.assertEqual(original,{p:sha(p) for p in original})
            second=subprocess.run(args+['--core-only'],capture_output=True,text=True,timeout=30)
            self.assertNotEqual(second.returncode,0)

    def test_conflicting_or_missing_mode_stops_before_output(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);args=self.fixture(root)
            for extra in ([],['--core-only','--event-start-ns','0']):
                result=subprocess.run(args+extra,capture_output=True,text=True,timeout=30)
                self.assertNotEqual(result.returncode,0)
                self.assertFalse((root/'output').exists())


if __name__=='__main__':unittest.main()
