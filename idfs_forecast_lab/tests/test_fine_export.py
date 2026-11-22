import unittest
from idfs_lab.fine_export import FineAccumulator,GRID_NS


class FineExportTests(unittest.TestCase):
    def test_real_zero_complete_node_and_missing_node_not_zero_label(self):
        topo={'storage_nodes':[{'entity':'node'+str(i),'oss_ids':list(range(i*12,(i+1)*12))} for i in range(8)]}
        a=FineAccumulator(topo)
        for i in range(12):a.add({'scope':'oss','entity':str(i),'bucket_start_ns':0,'bucket_end_ns':GRID_NS,'sample_ok':True,'read_count':0,'write_count':0,'available_epoch_ns':GRID_NS,'observed_interval_ns':GRID_NS})
        rows=list(a.rows());self.assertTrue(rows[0]['sample_ok']);self.assertEqual(rows[0]['total_count'],0)
        self.assertFalse(rows[1]['sample_ok']);self.assertNotIn('total_count',rows[1])
        with self.assertRaises(ValueError):a.add({'scope':'oss','entity':'0','bucket_start_ns':0,'bucket_end_ns':GRID_NS,'sample_ok':False})


if __name__=='__main__':unittest.main()
