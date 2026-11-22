import unittest
from idfs_lab.causal_io_context import build_io_context
from idfs_lab.joint import index_windows
from idfs_lab.multiscale_inputs import SECOND


def rows():
    return [dict(scope=s,entity=str(e),window_start_ns=t*SECOND,window_end_ns=(t+1)*SECOND,n_ticks=10,complete=True,available_epoch_ns=(t+1)*SECOND,read_iops=1,write_iops=2,total_iops=3) for s,es in [('oss',range(12)),('client_nfs_rpc',['client7414'])] for e in es for t in range(20)]


class IOContextTests(unittest.TestCase):
    def test_topology_statistics_and_future_invariance(self):
        data=rows();a=build_io_context(index_windows(data),15*SECOND,list(range(12)))
        self.assertEqual(a['feature_count'],38)
        x=dict(zip(a['feature_names'],a['features']));self.assertAlmostEqual(x['oss_total_concentration'],1/12)
        for r in data:
            if r['window_start_ns']>=14*SECOND:r.update(read_iops=999,write_iops=999,total_iops=1998)
        self.assertEqual(a,build_io_context(index_windows(data),15*SECOND,list(range(12))))

    def test_missing_late_and_real_zero_have_masks(self):
        data=rows()
        for r in data:
            if r['scope']=='client_nfs_rpc' and r['window_start_ns']==13*SECOND:r.update(read_iops=0,write_iops=0,total_iops=0)
        a=build_io_context(index_windows(data),15*SECOND,list(range(12)));x=dict(zip(a['feature_names'],a['features']))
        self.assertEqual(x['client_lag1_observed'],1)
        for r in data:
            if r['scope']=='client_nfs_rpc' and r['window_start_ns']==13*SECOND:r['available_epoch_ns']=15*SECOND
        a=build_io_context(index_windows(data),15*SECOND,list(range(12)));x=dict(zip(a['feature_names'],a['features']))
        self.assertEqual(x['client_lag1_observed'],0)


if __name__=='__main__':unittest.main()
