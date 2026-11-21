import copy
import hashlib
import json
from pathlib import Path
import tempfile
import unittest

from idfs_lab.counters import aggregate_cluster, counter_delta, parse_gnfs_client_io, parse_mountstats
from idfs_lab.dataset import build_windows, feature_rows, process
from idfs_lab.evaluation import score, summarize_rounds, verify_files, verify_round
from idfs_lab.io import append_event, canonical_hash, load
from idfs_lab.journal import PredictionJournal
from idfs_lab.remote_agent import decode_loose
from idfs_lab.training import train

ROOT=Path(__file__).resolve().parent.parent
NS=1000000000


def sample(tick, count, **extra):
    row=dict(scope="oss",entity="0",host="idfs44",source_id="socket0",boot_id="boot0",
             instance_id="daemon0",sample_interval_ns=100000000,scheduled_epoch_ns=tick*100000000,
             read_end_epoch_ns=tick*100000000+1000000,read_end_mono_ns=tick*100000000+1000000,
             read_count=count,write_count=count*2,sample_ok=True)
    row.update(extra)
    return row


def valid_round(accuracy=92, index=1):
    protocol=load(ROOT/"config/protocol.json");ph=canonical_hash(protocol)
    target=protocol["formal_target"]
    manifest={"round_id":"round%d"%index,"round_index":index,"workload":"hpc_wrf",
              "target":target,"model_sha256":"a"*64,"protocol_hash":ph,
              "selection_hash":"b"*64,"selection_frozen_before_formal_test":True,
              "t0_epoch_ns":0,"t0_source":"tool_first_io","data_kind":"measured",
              "workload_commit":"fb83711","clock_alignment":{"verified":True,
                  "max_relative_offset_ms":1,"uncertainty_ms":2,"evidence_sha256":"c"*64}}
    predictions=[];actuals=[]
    for i in range(180,600):
        start=i*NS
        predictions.append(dict(target=target,window_start_ns=start,window_end_ns=start+NS,
            issue_epoch_ns=start-30000000,compute_start_ns=start-31000000,
            feature_end_ns=start-NS,features_available_ns=start-NS+1000000,
            prediction_iops=100+(100-accuracy),model_sha256="a"*64,protocol_hash=ph))
        predictions.append(dict(event="prediction_commit",target=target,window_start_ns=start,
                                committed_epoch_ns=start-29000000))
        actuals.append(dict(scope=target["scope"],entity=target["entity"],window_start_ns=start,
            window_end_ns=start+NS,complete=True,n_ticks=10,total_iops=100,
            available_epoch_ns=start+NS+1000000))
    return manifest,predictions,actuals,protocol


class CounterTests(unittest.TestCase):
    def test_mount_selection_and_rpc_semantics(self):
        text="device x:/other mounted on /other with fstype nfs\n READ: 999 0 0 0 0 0 0 0\n"
        text+="device vip:/sys_ns mounted on /mnt/baize-idfs with fstype nfs\n READ: 12 13 0 100 3000 4 5 6 0\n WRITE: 7 7 0 5000 200 1 2 3 0\n"
        v=parse_mountstats(text,"/mnt/baize-idfs")
        self.assertEqual((v["read_count"],v["write_count"]),(12,7))
        self.assertEqual(v["rpc_fields"]["READ"][1],13)  # transmissions are not operations

    def test_gnfs_counts_not_display_iops(self):
        v=parse_gnfs_client_io("READ : 1000 2 4096\nWRITE: 80 0 1024\n")
        self.assertEqual(v["read_count"],1000)
        self.assertEqual(v["write_bytes"],1024)

    def test_nonfinite_normalization_preserves_strings(self):
        obj,n=decode_loose(b'{"oss":{"op_r":2,"op_w":3,"avg":-nan},"name":"nan inf"}')
        self.assertEqual(n,1);self.assertIsNone(obj["oss"]["avg"])
        self.assertEqual(obj["name"],"nan inf")

    def test_rate_uses_actual_interval(self):
        row=counter_delta(sample(1,10),sample(2,14))
        self.assertTrue(row["sample_ok"])
        self.assertEqual(row["total_count"],12)
        self.assertEqual(row["read_iops"],40)

    def test_gap_is_not_interpolated(self):
        self.assertEqual(counter_delta(sample(1,10),sample(3,20))["error"],"nonadjacent_grid")

    def test_reset_is_not_zero_or_current_count(self):
        self.assertEqual(counter_delta(sample(1,10),sample(2,3))["error"],"counter_reset:read_count")

    def test_socket_restart_rebuilds_baseline(self):
        self.assertEqual(counter_delta(sample(1,10),sample(2,20,instance_id="new"))["error"],"identity_changed:instance_id")

    def test_slow_sample_invalid(self):
        self.assertEqual(counter_delta(sample(1,10),sample(2,20,read_end_epoch_ns=280000000))["error"],"sampling_jitter")

    def test_full_cluster_requires_all_96_and_ignores_clients(self):
        rows=[]
        for i in range(96):
            d=counter_delta(sample(1,10),sample(2,11));d["entity"]=str(i);rows.append(d)
        client=dict(rows[0],scope="client_nfs_rpc",total_count=1000000)
        full=aggregate_cluster(rows+[client],map(str,range(96)))[0]
        self.assertTrue(full["sample_ok"]);self.assertEqual(full["total_count"],288)
        self.assertFalse(aggregate_cluster(rows[:-1],map(str,range(96)))[0]["sample_ok"])

    def test_incomplete_second_is_not_zero(self):
        rows=[counter_delta(sample(i,10*i),sample(i+1,10*(i+1))) for i in range(1,9)]
        self.assertFalse(build_windows(rows)[0]["complete"])


class TimingAndEvaluationTests(unittest.TestCase):
    def test_issue_time_is_after_model_finishes(self):
        with tempfile.TemporaryDirectory() as tmp:
            times=iter([90,110,111]);journal=PredictionJournal(Path(tmp)/"events.jsonl",lambda:next(times))
            event=journal.issue(lambda x:10,{}, {},100,200,80,80,"a","b")
            self.assertEqual(event["issue_epoch_ns"],110)
            self.assertTrue(event["late"])

    def test_every_positive_actual_is_included_only_zero_excluded(self):
        protocol=load(ROOT/"config/protocol.json")
        result=score([(20,10),(1000,1000),(100,0)],protocol["mask"])
        self.assertEqual(result["accuracy_percent"],"50.0")
        self.assertEqual(result["excluded"],1)
        self.assertEqual(result["n_eval"],2)

    def test_exactly_90_is_not_a_pass(self):
        m,p,a,protocol=valid_round(90)
        r=verify_round(m,p,a,protocol)
        self.assertEqual(r["accuracy_percent"],"90.0")
        self.assertFalse(r["passed"])

    def test_complete_round_recomputes_420_windows(self):
        m,p,a,protocol=valid_round(92)
        r=verify_round(m,p,a,protocol)
        self.assertTrue(r["passed"],r["errors"])
        self.assertEqual(r["n_eval"],420)

    def test_accurate_aux_target_cannot_replace_formal_client(self):
        m,p,a,protocol=valid_round(100)
        other={"scope":"cluster_oss","entity":"all","component":"total"}
        m["target"]=other
        for item in p:item["target"]=other
        for item in a:item.update(scope="cluster_oss",entity="all")
        self.assertIn("formal_target_mismatch",verify_round(m,p,a,protocol)["errors"])

    def test_missing_actual_and_late_prediction_fail(self):
        m,p,a,protocol=valid_round()
        p[2]["issue_epoch_ns"]=p[2]["window_start_ns"]
        r=verify_round(m,p,a[:-1],protocol)
        self.assertFalse(r["passed"])
        self.assertTrue(any("late" in e for e in r["errors"]))
        self.assertTrue(any("incomplete_actual" in e for e in r["errors"]))

    def test_unverified_clock_cannot_pass(self):
        m,p,a,protocol=valid_round();m["clock_alignment"]["verified"]=False
        self.assertIn("clock_alignment_unverified",verify_round(m,p,a,protocol)["errors"])

    def test_recording_start_accepted_without_calling_it_first_io(self):
        m,p,a,protocol=valid_round(92)
        m['t0_source']='recording_start'
        self.assertTrue(verify_round(m,p,a,protocol)['passed'])
        m['t0_source']='unknown'
        self.assertIn('t0_source_missing',verify_round(m,p,a,protocol)['errors'])

    def test_fsync_delay_cannot_be_backdated(self):
        m,p,a,protocol=valid_round()
        p[1]["committed_epoch_ns"]=p[1]["window_start_ns"]+1
        r=verify_round(m,p,a,protocol)
        self.assertFalse(r["passed"])
        self.assertTrue(any("late_durable" in e for e in r["errors"]))

    def test_three_rounds_all_required_not_the_mean(self):
        reports=[]
        for i,accuracy in enumerate([92,94,89],1):
            m,p,a,protocol=valid_round(accuracy,i);reports.append(verify_round(m,p,a,protocol))
        self.assertFalse(summarize_rounds(reports,protocol)["passed"])
        self.assertGreater(sum([92,94,89])/3,90)

    def test_target_cannot_switch_between_formal_rounds(self):
        reports=[]
        for i in range(1,4):
            m,p,a,protocol=valid_round(92,i);reports.append(verify_round(m,p,a,protocol))
        reports[-1]["target"]={"scope":"oss","entity":"0","component":"total"}
        self.assertFalse(summarize_rounds(reports,protocol)["passed"])

    def test_missing_frozen_selection_fails_closed(self):
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp);m,_,_,protocol=valid_round()
            (path/"first_io.json").write_text(json.dumps({"event":"tool_first_io","epoch_ns":m["t0_epoch_ns"]}),encoding="utf-8")
            m["first_io_evidence"]={"file":"first_io.json","sha256":hashlib.sha256((path/"first_io.json").read_bytes()).hexdigest()}
            (path/"round_manifest.json").write_text(json.dumps(m),encoding="utf-8")
            (path/"predictions.jsonl").write_text("{}\n",encoding="utf-8")
            with self.assertRaisesRegex(FileNotFoundError,"selection.json"):
                verify_files(path,protocol)

    def test_sealed_files_recompute_and_detect_tampering(self):
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp);m,p,a,protocol=valid_round()
            (path/"first_io.json").write_text(json.dumps({"event":"tool_first_io","epoch_ns":m["t0_epoch_ns"]}),encoding="utf-8")
            m["first_io_evidence"]={"file":"first_io.json","sha256":hashlib.sha256((path/"first_io.json").read_bytes()).hexdigest()}
            selection={k:m[k] for k in ["target","model_sha256","workload","workload_commit","protocol_hash"]}
            selection.update(frozen_epoch_ns=-1,training_runs=["trainA"],validation_runs=["valA"])
            (path/"selection.json").write_text(json.dumps(selection),encoding="utf-8")
            m["selection_hash"]=canonical_hash(selection)
            clock={k:m["clock_alignment"][k] for k in ["verified","max_relative_offset_ms","uncertainty_ms"]}
            (path/"clock_alignment.json").write_text(json.dumps(clock),encoding="utf-8")
            m["clock_alignment"]["evidence_sha256"]=hashlib.sha256((path/"clock_alignment.json").read_bytes()).hexdigest()
            for r in p:append_event(path/"predictions.jsonl",r)
            for r in a:append_event(path/"windows.jsonl",r)
            m["log_sha256"]={name:hashlib.sha256((path/name).read_bytes()).hexdigest() for name in ["predictions.jsonl","windows.jsonl"]}
            (path/"round_manifest.json").write_text(json.dumps(m),encoding="utf-8")
            self.assertTrue(verify_files(path,protocol)["passed"])
            with (path/"windows.jsonl").open("a",encoding="utf-8") as f:f.write("{}\n")
            with self.assertRaisesRegex(ValueError,"sealed log"):
                verify_files(path,protocol)

    def test_measured_counter_processing_all_three_scopes(self):
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp);raw=path/"raw";raw.mkdir()
            (raw/"collection_manifest.json").write_text("{}")
            for tick in range(21):
                for entity in range(96):
                    r=sample(tick,tick,entity=str(entity));append_event(raw/"oss.jsonl",r)
                r=sample(tick,tick*100,scope="client_nfs_rpc",entity="clientA")
                append_event(raw/"client.jsonl",r)
            q=process(raw,path/"processed",load(ROOT/"config/protocol.json"))
            self.assertEqual(q["complete_windows"],212) # 2*(96 OSS + 8 hosts + cluster + client)
            windows=[json.loads(x) for x in (path/"processed/windows.jsonl").read_text(encoding="utf-8").splitlines()]
            cluster=[r for r in windows if r["scope"]=="cluster_oss"]
            self.assertTrue(all(r["total_iops"]==2880 for r in cluster))

    def test_causal_features_do_not_use_target(self):
        rows=[dict(scope="oss",entity="0",window_start_ns=i*NS,window_end_ns=(i+1)*NS,
                   available_epoch_ns=(i+1)*NS+1000000,complete=True,n_ticks=10,
                   read_iops=i,write_iops=2*i,total_iops=3*i) for i in range(12)]
        features=list(feature_rows(rows));before=[r for r in features if r["window_start_ns"]==10*NS and r["component"]=="total"][0]
        rows[10]["total_iops"]=999999
        after=[r for r in feature_rows(rows) if r["window_start_ns"]==10*NS and r["component"]=="total"][0]
        self.assertEqual(before["lags"],after["lags"])
        self.assertLess(after["features_available_ns"],after["window_start_ns"])

    def test_probe_snapshots_are_not_training_data(self):
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp);(path/"probe.json").write_text("{}")
            with self.assertRaisesRegex(ValueError,"not experimental"):
                process(path,path/"out",load(ROOT/"config/protocol.json"))

    def test_train_validation_run_overlap_rejected(self):
        import csv
        with tempfile.TemporaryDirectory() as tmp:
            p=Path(tmp)/"features.csv"
            fields=["run_id","scope","entity","component","feature_end_ns","features_available_ns","window_start_ns"]
            with p.open("w",newline="") as f:
                w=csv.DictWriter(f,fieldnames=fields);w.writeheader()
                w.writerow(dict(run_id="same",scope="oss",entity="0",component="total",feature_end_ns=0,features_available_ns=1,window_start_ns=NS))
            with self.assertRaisesRegex(ValueError,"leakage"):
                train([str(p)],[str(p)],{"scope":"oss","entity":"0","component":"total"},"last_value",Path(tmp)/"model",load(ROOT/"config/protocol.json"))


if __name__=="__main__":
    unittest.main()
