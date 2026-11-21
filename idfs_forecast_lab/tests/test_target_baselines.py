import copy
import unittest

from idfs_lab.target_baselines import SECOND, paired_baselines, metrics

HOSTS = ["idfs"+str(i) for i in (44,45,46,47,64,65,66,67)]


def fixture():
    return [{"scope": "host_oss", "entity": h, "window_start_ns": t*SECOND,
             "window_end_ns": (t+1)*SECOND, "available_epoch_ns": (t+1)*SECOND,
             "complete": True, "n_ticks": 10, "total_iops": (i+1)*t}
            for t in range(8) for i,h in enumerate(HOSTS)]


class TargetBaselineTests(unittest.TestCase):
    def test_linear_aggregation_and_known_values(self):
        result = paired_baselines(fixture(), HOSTS)
        self.assertEqual(result["paired_windows"], 2)
        p = result["pairs"][0]
        self.assertEqual(p["node_last_available"], [4*i for i in range(1,9)])
        self.assertEqual(p["cluster_last_available"], 144)
        self.assertEqual(p["cluster_mean_five"], 72)
        self.assertEqual(p["cluster_actual"], 216)
        self.assertEqual(sum(result["rejected"].values())+2, result["candidate_windows"])

    def test_future_actual_changes_labels_only(self):
        original = paired_baselines(fixture(), HOSTS)["pairs"][0]
        rows = fixture()
        for row in rows:
            if row["window_start_ns"] >= 6*SECOND:
                row["total_iops"] += 1000
        changed = paired_baselines(rows, HOSTS)["pairs"][0]
        for key in ("node_last_available", "node_mean_five", "features_available_ns"):
            self.assertEqual(original[key], changed[key])
        self.assertNotEqual(original["cluster_actual"], changed["cluster_actual"])

    def test_missing_node_never_zero_filled(self):
        rows = [r for r in fixture() if not (r["entity"] == HOSTS[0] and r["window_start_ns"] == 6*SECOND)]
        result = paired_baselines(rows, HOSTS)
        self.assertEqual(result["paired_windows"], 1)
        self.assertEqual(result["rejected"]["incomplete_label"], 1)

    def test_late_history_and_margin_boundary(self):
        rows = fixture()
        row = next(r for r in rows if r["entity"] == HOSTS[0] and r["window_start_ns"] == 4*SECOND)
        row["available_epoch_ns"] = 6*SECOND-30_000_000
        self.assertEqual(paired_baselines(rows, HOSTS)["paired_windows"], 2)
        row["available_epoch_ns"] += 1
        self.assertEqual(paired_baselines(rows, HOSTS)["rejected"]["late_history"], 1)

    def test_duplicates_invalid_values_and_short_history(self):
        rows = fixture()
        with self.assertRaises(ValueError): paired_baselines(rows+[copy.deepcopy(rows[0])], HOSTS)
        rows[0]["total_iops"] = float("nan")
        with self.assertRaises(ValueError): paired_baselines(rows, HOSTS)
        self.assertEqual(paired_baselines(fixture()[:8], HOSTS)["paired_windows"], 0)

    def test_metrics_zero_policy_and_error_cancellation(self):
        self.assertEqual(metrics([(1,0),(4,2)])["mape_percent"], 100)
        self.assertIsNone(metrics([])["mape_percent"])
        self.assertIsNone(metrics([(0,0)])["wape_percent"])
        self.assertEqual(metrics([(9,10),(11,10)])["mae_iops"], 1)
        self.assertEqual(metrics([(20,20)])["mae_iops"], 0)


if __name__ == "__main__":
    unittest.main()
