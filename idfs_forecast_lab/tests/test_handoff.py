import importlib.util
import json
from pathlib import Path
import tempfile
import unittest

SCRIPT = Path(__file__).resolve().parents[2]/'scripts/export_handoff.py'
spec = importlib.util.spec_from_file_location('handoff', SCRIPT)
handoff = importlib.util.module_from_spec(spec)
spec.loader.exec_module(handoff)


class HandoffTests(unittest.TestCase):
    def make_run(self, base):
        run = base/'source'
        (run/'processed').mkdir(parents=True)
        for name, content in {'processed/windows.jsonl':'{}\n', 'processed/quality.json':'{}',
                              'recording_start.json':'{}', 'workload_manifest.json':'{}',
                              'workload_exit.json':'{"exit":0}',
                              'excluded_from_normal_training.json':'{}'}.items():
            (run/name).write_text(content, encoding='utf8')
        (run/'collection_manifest.json').write_text('{"credentials":"DO_NOT_COPY"}')
        return run

    def test_fault_requires_explicit_diagnostic_and_does_not_export_connection_info(self):
        with tempfile.TemporaryDirectory() as tmp:
            base=Path(tmp);run=self.make_run(base)
            with self.assertRaisesRegex(ValueError,'fault-affected'):
                handoff.export(run,base/'out')
            result=handoff.export(run,base/'out',True)
            self.assertEqual(result['role'],'diagnostic')
            self.assertFalse((base/'out/collection_manifest.json').exists())
            self.assertFalse(result['normal_training_certified'])
            self.assertEqual(handoff.verify(base/'out')['dataset_id'],'out')

    def test_checksum_tampering_is_detected(self):
        with tempfile.TemporaryDirectory() as tmp:
            base=Path(tmp);run=self.make_run(base)
            handoff.export(run,base/'out',True)
            (base/'out/processed/windows.jsonl').write_text('tampered')
            with self.assertRaisesRegex(ValueError,'mismatch'):
                handoff.verify(base/'out')

    def test_manifest_cannot_escape_package(self):
        with tempfile.TemporaryDirectory() as tmp:
            base=Path(tmp);run=self.make_run(base)
            handoff.export(run,base/'out',True)
            p=base/'out/handoff_manifest.json';data=json.loads(p.read_text())
            data['files'][0]['path']='../../outside'
            p.write_text(json.dumps(data))
            with self.assertRaisesRegex(ValueError,'invalid handoff'):
                handoff.verify(base/'out')
