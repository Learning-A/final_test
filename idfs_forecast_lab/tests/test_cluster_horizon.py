import unittest
import numpy as np
import torch
from idfs_lab.cluster_horizon import (ClusterHorizonMLP,collect_future_counts,seconds_from_counts,window_means,cluster_features)


class ClusterHorizonTests(unittest.TestCase):
    def test_counts_and_rates_units(self):
        counts=np.full((2,100),100.)
        seconds=seconds_from_counts(counts)
        np.testing.assert_array_equal(seconds,np.full((2,10),1000.))
        np.testing.assert_array_equal(window_means(seconds),np.full((2,4),1000.))
        self.assertEqual(window_means(np.arange(1.,11.)).tolist(),[1.,2.,3.,5.5])

    def test_missing_future_never_becomes_zero(self):
        self.assertEqual(collect_future_counts({},20_000_000_000,['h'],0),(None,'future_target_incomplete'))
        self.assertEqual(collect_future_counts({},599_000_000_000,['h'],0),(None,'target_beyond_recording_workload_range'))

    def test_equal_capacity_and_same_initial_output(self):
        fine=ClusterHorizonMLP(2000,100);direct=ClusterHorizonMLP(2000,10)
        self.assertEqual(sum(p.numel() for p in fine.parameters()),sum(p.numel() for p in direct.parameters()))
        for (_,p),(_,q) in zip(fine.named_parameters(),direct.named_parameters()):
            self.assertTrue(torch.equal(p,q))
        with torch.inference_mode():
            for model,steps in [(fine,100),(direct,10)]:
                result=model(torch.zeros(2,1840));self.assertEqual(tuple(result.shape),(2,steps))
                np.testing.assert_allclose(result.numpy(),2000.,rtol=1e-6)

    def test_cluster_features_keeps_each_nodes_short_and_fine_context(self):
        x=np.arange(2*8*258).reshape(2,8,258)
        out=cluster_features(x)
        self.assertEqual(out.shape,(2,1840))
        np.testing.assert_array_equal(out[:,:896],x[:,:,:112].reshape(2,-1))
        np.testing.assert_array_equal(out[:,896:928],x[:,0,112:144])
