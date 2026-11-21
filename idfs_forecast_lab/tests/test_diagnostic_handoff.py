import importlib.util
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

SCRIPTS=Path(__file__).resolve().parents[2]/'scripts'
sys.path.insert(0,str(SCRIPTS))
spec=importlib.util.spec_from_file_location('diagnostic_handoff',SCRIPTS/'build_diagnostic_handoff.py')
module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)


class DiagnosticHandoffTests(unittest.TestCase):
    def test_package_keeps_fault_and_exploratory_labels_and_all_checksums(self):
        with tempfile.TemporaryDirectory() as tmp:
            base=Path(tmp);run=base/'source';(run/'processed').mkdir(parents=True)
            (run/'processed/windows.jsonl').write_text('{}\n',encoding='utf8')
            contents={'processed/quality.json':{'windows_sha256':module.digest(run/'processed/windows.jsonl')},
                      'recording_start.json':{'epoch_ns':0},'workload_manifest.json':{'id':'ai_training_ses'},
                      'workload_exit.json':{'exit':0},'excluded_from_normal_training.json':{'reason':'fault'}}
            for name,obj in contents.items():(run/name).write_text(json.dumps(obj),encoding='utf8')
            site=base/'site.json';site.write_text(json.dumps({'storage_nodes':[{'hostname':'idfs44'}]}))
            row={'window_start_ns':10**9,'window_end_ns':2*10**9,'features':[0]*415,'role':'diagnostic'}
            with patch.object(module,'feature_rows',return_value=iter([row])):
                out=module.build(run,base/'package',site)
            manifest=module.verify(base/'package')
            self.assertTrue(manifest['fault_affected'])
            self.assertTrue(manifest['exploratory_development_allowed'])
            self.assertFalse(manifest['normal_training_certified'])
            self.assertFalse(manifest['independent_validation'])
            self.assertEqual(out['examples_by_node'],{'idfs44':1})
            self.assertEqual(module.digest(Path(out['archive'])),out['sha256'])
            exported=json.loads((base/'package/features/idfs44.jsonl').read_text())
            self.assertTrue(exported['fault_affected'])
            self.assertFalse(exported['normal_training_allowed'])
