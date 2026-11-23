import hashlib
import json
from pathlib import Path
import tempfile
import unittest
import zipfile
from idfs_lab.first_round_gate import WORKLOADS,check_packages


def package(root,workload):
    contents={'processed/windows.jsonl':b'', 'processed/quality.json':b'{}',
      'recording_start.json':json.dumps({'event':'recording_start','epoch_ns':0}).encode(),
      'workload_manifest.json':json.dumps({'id':workload}).encode(),
      'workload_exit.json':b'{"exit":0}', 'workload_quality.json':b'{}'}
    m=dict(dataset_id=workload,role='train',fault_affected=False,normal_training_certified=False,
           source_run_id='original-run',source_batch='batch',source_attempt=1,
           files=[dict(path=p,bytes=len(b),sha256=hashlib.sha256(b).hexdigest()) for p,b in contents.items()])
    raw=json.dumps(m).encode();path=root/(workload+'.zip')
    with zipfile.ZipFile(path,'w') as z:
        z.writestr('package/handoff_manifest.json',raw)
        for p,b in contents.items():z.writestr('package/'+p,b)
    return dict(workload=workload,dataset_id=workload,archive=str(path),
                archive_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),manifest_sha256=hashlib.sha256(raw).hexdigest(),
                time_boundary_reviewed=True,quality_approved_for_development=True,review_evidence_reference='synthetic test QA')


class FirstRoundGateTests(unittest.TestCase):
    def test_no_handoff_is_not_a_capture_failure(self):
        r=check_packages([])
        self.assertFalse(r['ready']);self.assertEqual(r['state'],'awaiting_handoff')
        self.assertEqual(len(r['workloads']),5)

    def test_all_packages_and_explicit_development_review_needed(self):
        with tempfile.TemporaryDirectory() as tmp:
            entries=[package(Path(tmp),w) for w in WORKLOADS]
            r=check_packages(entries)
            self.assertTrue(r['ready']);self.assertFalse(r['formal_acceptance'])
            entries[-1]['time_boundary_reviewed']=False
            self.assertEqual(check_packages(entries)['state'],'pending_quality_review')
            entries[-1]['time_boundary_reviewed']=True;entries[-1]['review_evidence_reference']=''
            self.assertFalse(check_packages(entries)['ready'])

    def test_partial_and_duplicate_workload(self):
        with tempfile.TemporaryDirectory() as tmp:
            e=package(Path(tmp),WORKLOADS[0])
            self.assertEqual(check_packages([e])['state'],'awaiting_handoff')
            with self.assertRaises(ValueError):check_packages([e,e])

    def test_hash_and_role_identity_cannot_be_bypassed_by_review_flags(self):
        with tempfile.TemporaryDirectory() as tmp:
            e=package(Path(tmp),WORKLOADS[0]);e['dataset_id']='other'
            with self.assertRaises(ValueError):check_packages([e])
            e['dataset_id']=WORKLOADS[0];e['archive_sha256']='bad'
            with self.assertRaises(ValueError):check_packages([e])


if __name__=='__main__':unittest.main()
