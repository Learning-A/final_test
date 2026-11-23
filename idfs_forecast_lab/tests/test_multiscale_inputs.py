import unittest
from idfs_lab.joint import index_windows
from idfs_lab.multiscale_inputs import build_input,SECOND

HOSTS=['idfs'+str(i) for i in (44,45,46,47,64,65,66,67)]

def rows():
    return [dict(scope='host_oss',entity=h,window_start_ns=t*SECOND,window_end_ns=(t+1)*SECOND,
                 available_epoch_ns=(t+1)*SECOND,complete=True,n_ticks=10,
                 read_iops=t,write_iops=2*t,total_iops=3*t) for h in HOSTS for t in range(65)]

def make(data): return build_input(index_windows(data),62*SECOND,HOSTS[0],HOSTS,'capture-test')

class MultiscaleTests(unittest.TestCase):
    def test_known_history_statistics_and_signed_changes(self):
        out=make(rows()); x=dict(zip(out['feature_names'],out['features']))
        self.assertEqual(x['lag1_total'],180)
        self.assertEqual(x['total_5s_mean'],174)
        self.assertEqual(x['total_60s_median'],91.5)
        self.assertEqual(x['total_difference'],3)
        self.assertEqual(out['feature_count'],97)
        self.assertEqual(out['feature_end_ns'],61*SECOND)
        self.assertEqual(out['feature_groups']['spatial_context'],[65,97])

    def test_target_and_future_changes_cannot_change_input(self):
        data=rows();original=make(data)
        for r in data:
            if r['window_start_ns']>=61*SECOND:
                r.update(read_iops=999,write_iops=999,total_iops=1998)
        self.assertEqual(original,make(data))

    def test_missing_and_late_required_history_reject_without_fill(self):
        data=rows(); r=next(r for r in data if r['entity']==HOSTS[0] and r['window_start_ns']==30*SECOND)
        r['complete']=False
        self.assertIsNone(make(data))
        r['complete']=True;r['available_epoch_ns']=62*SECOND-30_000_000
        self.assertIsNotNone(make(data))
        r['available_epoch_ns']+=1
        self.assertIsNone(make(data))

    def test_neighbor_missing_and_true_zero_have_different_masks(self):
        data=rows()
        for r in data:
            if r['window_start_ns']==60*SECOND and r['entity']==HOSTS[1]:r['complete']=False
            if r['window_start_ns']==60*SECOND and r['entity']==HOSTS[2]:r.update(read_iops=0,write_iops=0,total_iops=0)
        out=make(data);x=dict(zip(out['feature_names'],out['features']))
        self.assertEqual(x['context_'+HOSTS[1]+'_observed'],0)
        self.assertEqual(x['context_'+HOSTS[2]+'_observed'],1)
        self.assertEqual(x['context_'+HOSTS[2]+'_total'],0)

    def test_corrupt_complete_row_and_duplicate_are_not_silently_dropped(self):
        data=rows();data[0]['total_iops']=float('nan')
        # t0 is outside required history; corrupt a consumed historical row.
        data[1]['total_iops']=float('nan')
        with self.assertRaises(ValueError):make(data)
        data=rows()
        with self.assertRaises(ValueError):make(data+[data[0]])


if __name__=='__main__':unittest.main()
