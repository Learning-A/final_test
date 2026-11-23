import unittest
from idfs_lab.fine_paired_dataset import paired_capture,vector,SECOND
from idfs_lab.masked_inputs import build_masked_input
from idfs_lab.joint import index_windows
from test_multiscale_inputs import rows,HOSTS
from test_fine_inputs import fine_rows,T


def make(windows=None,fine=None,**overrides):
    args=dict(capture_id='capture',run_id='run03',source_role='train',run_index=3,
              workload='graph_zipf',hosts=HOSTS,recording_start_ns=0)
    args.update(overrides)
    return paired_capture(rows() if windows is None else windows,[] if fine is None else fine,**args)


class FinePairedDatasetTests(unittest.TestCase):
    def test_missing_fine_keeps_identical_labels_and_baseline_vectors(self):
        a=make();b=make(fine=fine_rows())
        self.assertEqual(a['accepted_seconds'],b['accepted_seconds'])
        for x,y in zip(a['samples'],b['samples']):
            self.assertEqual(x['actual'],y['actual'])
            self.assertEqual(vector(x,0,'F0'),vector(y,0,'F0'))
        s=next(r for r in b['samples'] if r['window_start_ns']==T)
        expected=build_masked_input(index_windows(rows()),T,HOSTS[0],HOSTS,'capture',0)
        self.assertEqual(s['base'][0],expected['features'])
        self.assertEqual(s['fine_observed'][0],19)
        self.assertEqual(s['development_partition'],'internal_run03')
        self.assertEqual(len(vector(s,0,'F0')),144)
        self.assertEqual(len(vector(s,0,'F0_F1')),258)
        self.assertEqual(len(vector(s,8,'F0')),928)
        self.assertEqual(len(vector(s,8,'F0_F1')),1840)
        self.assertFalse(s['formal_acceptance'])

    def test_target_label_change_never_changes_current_input(self):
        a=next(r for r in make()['samples'] if r['window_start_ns']==T)
        data=rows()
        for r in data:
            if r['window_start_ns']>=T:
                r.update(read_iops=1,write_iops=2,total_iops=3)
        b=next(r for r in make(windows=data)['samples'] if r['window_start_ns']==T)
        self.assertEqual(a['base'],b['base'])
        self.assertEqual(a['fine'],b['fine'])
        self.assertNotEqual(a['actual'],b['actual'])

    def test_complete_zero_label_is_kept_incomplete_label_not_filled(self):
        data=rows()
        for r in data:
            if r['window_start_ns']==T:
                r.update(read_iops=0,write_iops=0,total_iops=0)
        s=next(r for r in make(windows=data)['samples'] if r['window_start_ns']==T)
        self.assertEqual(s['actual'],[0]*9)
        next(r for r in data if r['window_start_ns']==T)['complete']=False
        self.assertNotIn(T,[r['window_start_ns'] for r in make(windows=data)['samples']])

    def test_pinned_baseline_mismatch_and_roles_fail_closed(self):
        starts=[r['window_start_ns'] for r in make()['samples']]
        self.assertEqual(make(expected_starts=starts)['accepted_seconds'],len(starts))
        with self.assertRaises(ValueError):make(expected_starts=starts[:-1])
        for patch in ({'source_role':'validation'},{'run_index':True},{'capture_id':''}):
            with self.assertRaises(ValueError):make(**patch)

    def test_processing_budget_and_clock_bounds_apply_to_both_arms(self):
        out=make(fine=fine_rows(),processing_budget_ns=100_000_000,clock_uncertainty_ns=10_000_000)
        for s in out['samples']:
            self.assertLessEqual(s['features_available_upper_ns'],s['feature_deadline_ns'])
            self.assertEqual(s['publication_deadline_ns']-s['feature_deadline_ns'],100_000_000)
        self.assertFalse(out['fitting_performed'])


if __name__=='__main__':unittest.main()
