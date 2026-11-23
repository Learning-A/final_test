import unittest
from idfs_lab.joint import index_windows
from idfs_lab.masked_inputs import build_masked_input, SECOND
from test_multiscale_inputs import rows, HOSTS


def make(data, floor=0):
    return build_masked_input(index_windows(data), 62*SECOND, HOSTS[0], HOSTS, 'masked-test', floor)


class MaskedInputsTests(unittest.TestCase):
    def test_missing_old_history_is_masked_not_silently_zero(self):
        data=rows()
        for r in data:
            if r['entity']==HOSTS[0] and r['window_start_ns']==50*SECOND:r['complete']=False
        out=make(data);x=dict(zip(out['feature_names'],out['features']))
        self.assertEqual(out['feature_count'],144)
        self.assertEqual(out['long_observed_count'],14)
        self.assertEqual(x['optional_lag11_observed'],0)
        self.assertEqual(x['optional_lag11_total'],0)
        self.assertEqual(out['feature_groups'],{'short':[0,35],'long':[35,112],'spatial':[112,144]})
        for r in data:
            if r['entity']==HOSTS[0] and r['window_start_ns']==50*SECOND:r.update(complete=True,read_iops=0,write_iops=0,total_iops=0)
        x=dict(zip(make(data)['feature_names'],make(data)['features']))
        self.assertEqual(x['optional_lag11_observed'],1)
        self.assertEqual(x['optional_lag11_total'],0)

    def test_late_optional_is_unobserved_but_late_short_rejects(self):
        data=rows()
        for r in data:
            if r['entity']==HOSTS[0] and r['window_start_ns']==50*SECOND:r['available_epoch_ns']=62*SECOND
        self.assertEqual(make(data)['long_observed_count'],14)
        for r in data:
            if r['entity']==HOSTS[0] and r['window_start_ns']==60*SECOND:r['available_epoch_ns']=62*SECOND
        self.assertIsNone(make(data))

    def test_future_labels_and_prior_partition_cannot_change_input(self):
        data=rows();before=make(data,56*SECOND)
        self.assertEqual(before['long_observed_count'],5)
        for r in data:
            if r['window_start_ns']<56*SECOND or r['window_start_ns']>=61*SECOND:
                r.update(read_iops=999,write_iops=999,total_iops=1998)
        self.assertEqual(before,make(data,56*SECOND))
        self.assertIsNone(make(data,57*SECOND))

    def test_consumed_corrupt_complete_history_fails(self):
        data=rows()
        for r in data:
            if r['entity']==HOSTS[0] and r['window_start_ns']==50*SECOND:r['total_iops']=float('nan')
        with self.assertRaises(ValueError):make(data)


if __name__=='__main__':unittest.main()
