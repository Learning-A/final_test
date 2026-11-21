import hashlib
import io
import json
from pathlib import Path
import stat
import tarfile
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from idfs_lab.artifacts import bundle_inference,export_cpu,lightgbm_tree,xgboost_tree
from idfs_lab.gpu import socks_socket
from idfs_lab.io import load,save
from idfs_lab.offline_job import choose_idle_gpu
from idfs_lab.online import LiveState,serve
from idfs_lab.placement import require_training_host
from idfs_lab.runtime import Predictor,check_runtime
from idfs_lab.workflow import job_plan,owned_directory,scoped_path,submit_job,fetch_results
from test_pipeline import sample

ROOT=Path(__file__).resolve().parent.parent


class RuntimeTests(unittest.TestCase):
    def test_portable_linear_and_tree_predictions(self):
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp)/"model.json"
            save(path,{"kind":"linear","coefficients":[2,0,0,0,0,0],"intercept":3})
            self.assertEqual(Predictor(path).predict([4,0,0,0,0,0]),11)
            tree=lightgbm_tree({"split_feature":0,"threshold":5,"decision_type":"<=",
                 "left_child":{"leaf_value":2},"right_child":{"leaf_value":7}})
            save(path,{"kind":"tree_sum","base":1,"trees":[tree]})
            self.assertEqual(Predictor(path).predict([5,0,0,0,0,0]),3)
            self.assertEqual(Predictor(path).predict([6,0,0,0,0,0]),8)

    def test_xgboost_strict_boundary_preserved(self):
        tree=xgboost_tree({"nodeid":0,"split":"f0","split_condition":5,"yes":1,"no":2,
                          "children":[{"nodeid":1,"leaf":2},{"nodeid":2,"leaf":7}]})
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp)/"model.json";save(path,{"kind":"tree_sum","base":0,"trees":[tree]})
            self.assertEqual(Predictor(path).predict([5,0,0,0,0,0]),7)

    def make_export(self,root):
        model=root/"trained_fixture";model.mkdir()
        save(model/"model.json",{"model":"last_value"})
        metadata={"model":"last_value","artifact":"model.json","sha256":hashlib.sha256((model/"model.json").read_bytes()).hexdigest(),
                  "target":{"scope":"oss","entity":"0","component":"total"},"validation_runs":["val"],
                  "training_runs":["train"],"validation":{"accuracy_percent":"synthetic_test_only"}}
        save(model/"model_manifest.json",metadata)
        csv=root/"val.csv"
        csv.write_text("run_id,scope,entity,component,feature_end_ns,features_available_ns,window_start_ns,lag1,lag2,lag3,lag4,lag5,rolling_mean,target_iops\nval,oss,0,total,1,2,1000000000,10,9,8,7,6,8,12\n",encoding="utf-8")
        export_cpu(model,[str(csv)],root/"export")
        return root/"export"

    def test_export_runtime_gate_and_inference_only_archive(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);export=self.make_export(root)
            self.assertTrue(check_runtime(export)["passed"])
            result=bundle_inference(export,ROOT/"config/protocol.json",root/"inference.tar.gz")
            self.assertFalse(result["contains_training_code"])
            with tarfile.open(root/"inference.tar.gz") as f:
                names=f.getnames()
                self.assertIn("idfs_lab/online.py",names)
                self.assertNotIn("idfs_lab/training.py",names)
                self.assertNotIn("idfs_lab/offline_job.py",names)
                self.assertNotIn("idfs_lab/models_torch.py",names)

    def test_inference_ends_cleanly_without_training_on_eof(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);export=self.make_export(root)
            serve(export,ROOT/"config/protocol.json",root/"service",source=io.StringIO(""))
            self.assertFalse(load(root/"service/service_finished.json")["fit_performed"])
            self.assertEqual(load(root/"service/service_finished.json")["predictions"],0)

    def test_live_features_match_offline_availability_policy(self):
        state=LiveState(["0"])
        for tick in range(101):
            row=sample(tick,tick)
            state.ingest(row,row["read_end_epoch_ns"]+1000000)
        feat=state.feature({"scope":"oss","entity":"0","component":"total"},8*1000000000)
        self.assertEqual(feat["lags"],[30,30,30,30,30])
        self.assertLess(feat["features_available_ns"],8*1000000000)


class IsolationTests(unittest.TestCase):
    def test_non_gpu_host_cannot_train(self):
        with patch("idfs_lab.placement.getpass.getuser",return_value="hust"):
            with self.assertRaisesRegex(RuntimeError,"restricted"):
                require_training_host()

    def test_paths_cannot_escape_lmj_workspace(self):
        for path in ["/home/other/job","/home/lmj/idfs_forecast_lab/../other", "relative/job"]:
            with self.assertRaises(ValueError):scoped_path(path,"/home/lmj","/home/lmj/idfs_forecast_lab")

    def test_wrong_owner_or_symlink_is_never_modified(self):
        class SFTP:
            def lstat(self,path):return SimpleNamespace(st_mode=stat.S_IFLNK|0o777,st_uid=1010)
            def mkdir(self,*unused,**kw):raise AssertionError("must not create anything")
        with self.assertRaises(ValueError):owned_directory(SFTP(),"/home/lmj/idfs_forecast_lab/jobs/a","/home/lmj","/home/lmj/idfs_forecast_lab",1010)

    def test_busy_gpu_is_not_selected_or_evicted(self):
        replies=["GPU-a, 2000, 20\nGPU-b, 0, 0\n","GPU-a\n"]
        with patch("idfs_lab.offline_job.subprocess.check_output",side_effect=replies):
            self.assertEqual(choose_idle_gpu(),"GPU-b")
        with patch("idfs_lab.offline_job.subprocess.check_output",side_effect=["GPU-a, 0, 0\n","GPU-a\n"]):
            with self.assertRaisesRegex(RuntimeError,"no idle"):choose_idle_gpu()

    def test_job_plan_missing_data_is_plan_only(self):
        config=load(ROOT/"config/training_host.json")
        spec=load(ROOT/"config/gpu_job.example.json")
        plan=job_plan(config,spec,ROOT/"config/protocol.json")
        self.assertFalse(plan["execute_ready"])
        self.assertEqual(plan["user"],"lmj")
        self.assertFalse(plan["changes_other_users"])
        self.assertFalse(plan["starts_training"])

    def test_plan_commands_do_not_connect_or_create_remote_jobs(self):
        config=load(ROOT/"config/training_host.json");spec=load(ROOT/"config/gpu_job.example.json")
        with tempfile.TemporaryDirectory() as tmp,patch("idfs_lab.workflow.gpu_connection",side_effect=AssertionError("must not connect")):
            output=Path(tmp)/"new_job"
            plan=submit_job(config,spec,ROOT/"config/protocol.json",output,execute=False)
            self.assertFalse(output.exists())
            self.assertFalse(plan["starts_training"])
            self.assertEqual(fetch_results(config,"test_job",output,execute=False)["state"],"plan_only")

    def test_socks5_handshake_uses_fixed_destination(self):
        class FakeSocket:
            def __init__(self):self.data=b"\x05\x00\x05\x00\x00\x01\x7f\x00\x00\x01\x00\x16";self.sent=[]
            def sendall(self,v):self.sent.append(v)
            def recv(self,n):v=self.data[:n];self.data=self.data[n:];return v
            def close(self):pass
        sock=FakeSocket()
        with patch("idfs_lab.gpu.socket.create_connection",return_value=sock):
            result=socks_socket("proxy",1080,"192.168.1.13",22)
        self.assertIs(result,sock)
        self.assertTrue(sock.sent[-1].endswith(b"\xc0\xa8\x01\x0d\x00\x16"))


if __name__=="__main__":unittest.main()
