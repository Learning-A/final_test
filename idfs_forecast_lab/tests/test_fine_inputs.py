import unittest
from idfs_lab.fine_inputs import index_fine, build_fine_input, build_long_input, SECOND, GRID_NS
from idfs_lab.joint import index_windows
from test_multiscale_inputs import rows, HOSTS

T = 62*SECOND


def fine_rows():
    return [dict(scope='host_oss', entity=HOSTS[0], bucket_start_ns=t, bucket_end_ns=t+GRID_NS,
                 sample_ok=True, read_count=0, write_count=0, total_count=0,
                 expected_oss=12, observed_oss=12, unit='operations per nominal100ms bucket',
                 available_epoch_ns=t+GRID_NS) for t in range(T-2*SECOND, T+SECOND, GRID_NS)]


def make(data, **kwargs):
    return build_fine_input(index_fine(data), T, HOSTS[0], 'capture', 0, **kwargs)


class FineInputsTests(unittest.TestCase):
    def test_excludes_boundary_bucket_but_uses_recent_available_history(self):
        out=make(fine_rows());x=dict(zip(out['feature_names'],out['features']))
        self.assertEqual(out['observed_buckets'],19)
        self.assertEqual(out['feature_end_ns'],T-GRID_NS)
        self.assertEqual(x['fine_lag1_observed'],0)
        self.assertEqual(x['fine_lag2_observed'],1)
        self.assertFalse(out['online_ready'])

    def test_future_and_boundary_values_cannot_change_features(self):
        data=fine_rows();a=make(data)
        for row in data:
            if row['bucket_end_ns']>T-30_000_000:
                row.update(read_count=999,write_count=999,total_count=1998)
        self.assertEqual(a,make(data))

    def test_late_and_clock_uncertainty_are_masked(self):
        data=fine_rows();data[18]['available_epoch_ns']=T
        out=make(data)
        self.assertEqual(out['observed_buckets'],18)
        out=make(fine_rows(),clock_uncertainty_ns=100_000_000)
        self.assertEqual(out['observed_buckets'],18)

    def test_real_zero_differs_from_missing_and_invalid(self):
        data=fine_rows();original=make(data)
        data[18]['sample_ok']=False
        missing=make(data)
        self.assertNotEqual(original['features'],missing['features'])
        self.assertEqual(missing,make([r for i,r in enumerate(data) if i!=18]))

    def test_duplicate_partial_nodes_bad_units_and_values_rejected(self):
        data=fine_rows()
        with self.assertRaises(ValueError):index_fine(data+[data[0]])
        for patch in [{'observed_oss':11},{'unit':'IOPS'},{'read_count':float('nan')},{'total_count':1}]:
            bad=fine_rows();bad[0].update(patch)
            with self.assertRaises(ValueError):index_fine(bad)

    def test_online_claim_requires_reception_and_clock_bound(self):
        with self.assertRaises(ValueError):make(fine_rows(),require_online=True)
        with self.assertRaises(ValueError):make(fine_rows(),require_online=True,availability_evidence='inference_host_reception')
        with self.assertRaises(ValueError):make(fine_rows(),require_online=True,availability_evidence='inference_host_reception',clock_uncertainty_ns=0)
        self.assertTrue(make(fine_rows(),require_online=True,availability_evidence='inference_host_reception',clock_uncertainty_ns=0,processing_budget_ns=1_000_000)['online_ready'])

    def test_processing_budget_is_reserved_before_publication_deadline(self):
        out=make(fine_rows(),processing_budget_ns=100_000_000)
        self.assertEqual(out['observed_buckets'],18)
        self.assertEqual(out['feature_deadline_ns'],T-130_000_000)
        self.assertLessEqual(out['features_available_ns'],out['feature_deadline_ns'])
        for value in (-1,True,SECOND):
            with self.assertRaises(ValueError):make(fine_rows(),processing_budget_ns=value)

    def test_history_floor_and_all_missing_keep_fixed_schema(self):
        a=make([])
        b=build_fine_input(index_fine(fine_rows()),T,HOSTS[0],'capture',T)
        self.assertEqual(a['features'],b['features'])
        self.assertEqual(a['feature_count'],114)
        self.assertIsNone(a['features_available_ns'])

    def test_optional_long_context_does_not_require_sixty_seconds(self):
        ix=index_windows(rows());a=build_long_input(ix,T,HOSTS[0],'capture',56*SECOND)
        self.assertEqual(a['observed_seconds'],5)
        self.assertEqual(a['feature_count'],321)
        x=dict(zip(a['feature_names'],a['features']))
        self.assertEqual(x['long_lag6_observed'],0)
        for r in rows():
            if r['window_start_ns']<56*SECOND or r['window_start_ns']>=61*SECOND:
                ix[(r['scope'],r['entity'],r['window_start_ns'])].update(read_iops=999,write_iops=999,total_iops=1998)
        self.assertEqual(a,build_long_input(ix,T,HOSTS[0],'capture',56*SECOND))


if __name__=='__main__':unittest.main()
