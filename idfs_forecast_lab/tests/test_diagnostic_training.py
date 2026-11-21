import unittest
from unittest.mock import patch
from idfs_lab.diagnostic_split import HOSTS
from idfs_lab.diagnostic_training import examples, fit


class DiagnosticTrainingTests(unittest.TestCase):
    def rows(self):
        return [dict(window_start_ns=100, target={'entity':h}, features=list(range(415)),
                     target_iops=i+1,feature_end_ns=90,features_available_ns=91)
                for i,h in enumerate(HOSTS)]

    def test_temporal_order_and_cluster_sum(self):
        rows=self.rows()
        node=examples(rows,HOSTS[0])[0]
        cluster=examples(rows,'cluster')[0]
        self.assertEqual(node['features'],[14,11,8,5,2,8])
        self.assertEqual(cluster['features'],[112,88,64,40,16,64])
        self.assertEqual(cluster['target_iops'],36)

    def test_missing_node_blocks_all_targets_and_duplicates_rejected(self):
        for target in (HOSTS[0],'cluster'):
            with self.assertRaises(ValueError): examples(self.rows()[:-1],target)
        with self.assertRaises(ValueError): examples(self.rows()+[self.rows()[0]],'cluster')

    def test_host_rejection_precedes_any_data_read_or_fit(self):
        with patch('idfs_lab.placement.require_training_host',side_effect=RuntimeError('wrong host')):
            with self.assertRaisesRegex(RuntimeError,'wrong host'):
                fit('missing.zip','must-not-be-created','cluster')


if __name__=='__main__': unittest.main()
