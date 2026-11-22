import copy
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
from idfs_lab.io import load, save, append_event
from idfs_lab.joint import TARGET, causal_features, feature_rows, host_windows, index_windows, topology
from idfs_lab.joint_training import read_split, train_joint
from idfs_lab.readiness import report
from idfs_lab.workflow import job_plan
from idfs_lab.joint_prepare import export
from idfs_lab.online import LiveState,serve

ROOT = Path(__file__).resolve().parent.parent
NS = 1000000000


def fixture():
    site = load(ROOT/"config/site.json")
    windows = []
    for second in range(8):
        for scope, entities in (("oss", map(str, range(96))),
                                ("client_nfs_rpc", [c["hostname"] for c in site["clients"]])):
            for entity in entities:
                windows.append({"scope": scope, "entity": entity, "window_start_ns": second*NS,
                    "window_end_ns": (second+1)*NS, "available_epoch_ns": (second+1)*NS+1000000,
                    "complete": True, "n_ticks": 10, "read_iops": 10.0+second,
                    "write_iops": 2.0, "total_iops": 12.0+second})
    return site, windows+list(host_windows(windows,site))


class JointTests(unittest.TestCase):
    def test_smoke_model_cannot_be_frozen_as_formal(self):
        from idfs_lab.selection import freeze_selection
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp)/'model_manifest.json'
            save(path,{'experiment_kind':'smoke_not_acceptance'})
            with self.assertRaisesRegex(ValueError,'smoke model'):
                freeze_selection(path,'background','x',{},Path(tmp)/'selection.json')

    def test_shared_service_warms_eight_models_and_closes_on_eof(self):
        import io,json
        from contextlib import redirect_stdout
        from idfs_lab.multi_online import serve as multi_serve
        site,_=fixture();entries=[]
        with tempfile.TemporaryDirectory() as tmp:
            base=Path(tmp)
            for node in site['storage_nodes']:
                model=base/node['hostname'];model.mkdir()
                save(model/'cpu_model.json',{'kind':'last_value','feature_count':415})
                save(model/'site.json',site)
                save(model/'parity_vectors.json',[{'features':[1.]*415,'expected':1.}])
                save(model/'deployment.json',{'files':{},'equivalence':{'passed':True},'source_model_sha256':'x',
                     'target':{'scope':'host_oss','entity':node['hostname'],'component':'total'}})
                entries.append({'host':node['hostname'],'model':str(model)})
            captured=io.StringIO()
            with redirect_stdout(captured):
                multi_serve(entries,load(ROOT/'config/protocol_smoke.json'),base/'logs',source=io.StringIO(''))
            self.assertEqual(len(json.loads(captured.getvalue())['models']),8)
            self.assertFalse(load(base/'logs/service_finished.json')['training_performed'])

    def test_live_host_sum_matches_offline_and_requires_all_twelve(self):
        site,_=fixture()
        state=LiveState(range(96),site=site)
        from test_pipeline import sample
        for tick in range(11):
            for oss in range(12):
                if oss==11 and tick==10:continue
                state.ingest(sample(tick,tick,entity=str(oss)),tick*100000000+2000000)
        self.assertNotIn(("host_oss","idfs44",0),state.windows)
        fresh=state.ingest(sample(10,10,entity="11"),1002000000)
        host=next(r for r in fresh if r['scope']=='host_oss')
        self.assertTrue(host['complete'])
        self.assertEqual(host['total_iops'],12*10*3)

    def test_smoke_anchor_is_distinct_from_tool_first_io(self):
        import io
        with tempfile.TemporaryDirectory() as tmp:
            base=Path(tmp);model=base/'model';model.mkdir()
            save(model/'cpu_model.json',{'kind':'last_value'})
            save(model/'deployment.json',{'files':{},'equivalence':{'passed':True},'target':TARGET,'source_model_sha256':'x'})
            protocol=load(ROOT/'config/protocol_smoke.json')
            save(base/'protocol.json',protocol)
            serve(model,base/'protocol.json',base/'logs',source=io.StringIO('{"type":"observation_start","epoch_ns":9999999999999999999}\n'))
            self.assertTrue((base/'logs/observation_start.json').exists())
            self.assertFalse((base/'logs/first_io.json').exists())

    def test_future_target_does_not_change_features(self):
        site, windows = fixture()
        before = causal_features(index_windows(windows), 7*NS, site)
        for row in windows:
            if row["window_start_ns"] >= 6*NS:
                row.update(read_iops=99990., write_iops=10., total_iops=100000.)
        after = causal_features(index_windows(windows), 7*NS, site)
        self.assertEqual(before["features"], after["features"])
        self.assertEqual(len(before["features"]), 415)
        self.assertEqual(before["attribution"], "storage_context_only")

    def test_late_client_history_blocks_example(self):
        site, windows = fixture()
        for r in windows:
            if r["entity"] == TARGET["entity"] and r["window_start_ns"] == 5*NS:
                r["available_epoch_ns"] = 7*NS
        self.assertIsNone(causal_features(index_windows(windows), 7*NS, site))

    def test_missing_oss_is_null_and_masked_not_complete_host(self):
        site, windows = fixture()
        windows = [r for r in windows if not (r["scope"] == "oss" and r["entity"] == "12")]
        result = causal_features(index_windows(windows), 7*NS, site)
        self.assertEqual(result["observed_oss"], 95)
        self.assertEqual(result["spatial_observations"][12], [None, None, None, 0.])
        hosts = list(host_windows(windows, site))
        self.assertTrue(all(not r["complete"] for r in hosts if r["entity"] == "idfs45"))
        self.assertTrue(all(r["complete"] for r in hosts if r["entity"] == "idfs44"))

    def test_low_storage_coverage_rejected(self):
        site, windows = fixture()
        windows = [r for r in windows if not (r["scope"] == "oss" and int(r["entity"]) < 5)]
        self.assertIsNone(causal_features(index_windows(windows), 7*NS, site))

    def test_physical_graph_is_not_routing(self):
        site, _ = fixture(); graph = topology(site)
        self.assertAlmostEqual(sum(graph[0]), 1.)
        self.assertEqual(graph[0][12], 0.)
        self.assertGreater(graph[0][11], 0.)
        site["storage_nodes"][0]["oss_ids"].append(12)
        with self.assertRaises(ValueError): topology(site)

    def test_confirmed_study_has_held_out_formal_rounds(self):
        result = report(ROOT)
        self.assertEqual(result["roles"], {"train": 15, "validation": 5, "formal": 15})
        self.assertFalse(result["collection_ready"])
        self.assertFalse(result["formal_ready"])
        plan = job_plan(load(ROOT/"config/training_host.json"), load(ROOT/"config/gpu_job_moe.example.json"), ROOT/"config/protocol.json")
        self.assertEqual(plan["job"]["model"], "joint_moe")
        self.assertFalse(plan["execute_ready"])
        self.assertEqual(len(plan["missing_inputs"]), 20)

    def test_joint_fit_is_forbidden_on_windows(self):
        # Make the non-training-host condition explicit on Linux test runners.
        with patch('idfs_lab.placement.socket.gethostname', return_value='windows-test'), \
                self.assertRaisesRegex(RuntimeError, "restricted"):
            train_joint([], [], "unused", {}, {}, {})

    def test_preprocessing_export_requires_provenance_and_refuses_formal(self):
        import hashlib
        site, windows=fixture()
        study=load(ROOT/"config/study_plan.json")
        train_id=next(r["run_id"] for r in study["rounds"] if r["role"]=="train")
        formal_id=next(r["run_id"] for r in study["rounds"] if r["role"]=="formal")
        with tempfile.TemporaryDirectory() as tmp:
            base=Path(tmp); source=base/"windows.jsonl"
            for row in windows:append_event(source,row)
            save(base/"quality.json",{"source_kind":"counter_collection","windows_sha256":hashlib.sha256(source.read_bytes()).hexdigest()})
            result=export(source,ROOT/"config/study_plan.json",ROOT/"config/site.json",train_id,base/"prepared")
            self.assertGreater(result["examples"],0)
            self.assertFalse(result["normalization_fitted"])
            with self.assertRaisesRegex(ValueError,"held-out"):
                export(source,ROOT/"config/study_plan.json",ROOT/"config/site.json",formal_id,base/"formal")
            source.write_text("{}\n")
            with self.assertRaisesRegex(ValueError,"provenance"):
                export(source,ROOT/"config/study_plan.json",ROOT/"config/site.json",train_id,base/"tampered")

    def test_split_refuses_formal_role_and_duplicate_windows(self):
        site, windows = fixture()
        row = list(feature_rows(windows, site, "tr", "train"))[-1]
        row["workload"] = "x"
        val = dict(row, run_id="va", role="validation")
        study = {"rounds": [{"run_id": "tr", "role": "train", "workload": "x"},
                            {"run_id": "va", "role": "validation", "workload": "x"}]}
        with tempfile.TemporaryDirectory() as tmp:
            tr, va = Path(tmp)/"tr.jsonl", Path(tmp)/"va.jsonl"
            append_event(tr, row); append_event(va, val)
            self.assertEqual(len(read_split([tr], [va], study, site)[0]), 1)
            append_event(tr, row)
            with self.assertRaisesRegex(ValueError, "duplicate"):
                read_split([tr], [va], study, site)
            tr.write_text("")
            append_event(tr, dict(row, role="formal"))
            with self.assertRaisesRegex(ValueError, "formal"):
                read_split([tr], [va], study, site)


class ArchitectureTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        try:
            import torch
        except ImportError:
            raise unittest.SkipTest("optional torch unavailable; architecture not tested")
        cls.torch = torch
        torch.set_num_threads(1)

    def test_true_experts_gates_masking_and_cpu_serialization_without_fit(self):
        from idfs_lab.moe import JointMoE, training_scaler
        from idfs_lab.runtime import Predictor
        torch = self.torch
        site, windows = fixture()
        features = causal_features(index_windows(windows), 7*NS, site)["features"]
        x = torch.tensor([features, features], dtype=torch.float32)
        mean, std = training_scaler(x)
        self.assertTrue(bool((std>=0.25).all()))
        model = JointMoE(topology(site), mean, std).eval()
        with torch.inference_mode():
            detail = model.details(x)
        self.assertEqual(tuple(detail.shape), (2, 6))
        self.assertTrue(torch.allclose(detail[:, 0], detail[:, 1]+detail[:, 2]))
        self.assertTrue(torch.allclose(detail[:, 3:6].sum(dim=1), torch.ones(2)))
        self.assertTrue(bool((detail[:, 3:6] > 0).all()))
        # Arbitrary values under an absent-node mask must not affect predictions.
        x[:, 18] = 0
        a = model(x)
        x[:, 15:18] = 999999
        self.assertTrue(torch.allclose(a, model(x)))
        # CPU forward/serialization only: no optimizer, backward or model fitting.
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp)
            model=model.double()  # runtime casts float32 input to exported precision
            torch.jit.script(model).save(str(path/"weights.ts"))
            save(path/"cpu_model.json", {"kind": "torchscript", "artifact": "weights.ts", "feature_count": 415,
                                       "diagnostics": "moe_read_write_gates"})
            predictor = Predictor(path/"cpu_model.json")
            actual = predictor.predict(features)
            combined, diagnostics = predictor.predict_with_diagnostics(features)
            self.assertAlmostEqual(combined, actual, places=6)
            self.assertAlmostEqual(diagnostics['total_iops'], actual, places=6)
            self.assertAlmostEqual(actual, float(model(torch.tensor([features]))[0].detach()), places=5)
            self.assertAlmostEqual(sum(predictor.diagnostics(features)["expert_weights"].values()), 1., places=6)
            with self.assertRaises(ValueError): predictor.predict([1.]*6)
            with self.assertRaises(ValueError): predictor.predict_with_diagnostics([1.]*6)


if __name__ == "__main__":
    unittest.main()
