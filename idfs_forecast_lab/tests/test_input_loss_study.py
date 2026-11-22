import unittest
from idfs_lab.input_loss_study import select_features, relative_weights


class InputLossStudyTests(unittest.TestCase):
    def test_node_and_cluster_use_only_declared_groups(self):
        self.assertEqual(select_features(list(range(82)),0,'total5'),[2,5,8,11,14])
        x=select_features(list(range(432)),8,'total5')
        self.assertEqual(len(x),40)
        self.assertEqual(x[-5:],[352,355,358,361,364])
        self.assertEqual(select_features(list(range(432)),8,'self50'),list(range(400)))
        self.assertEqual(len(select_features(list(range(82)),7,'spatial')),82)

    def test_training_floor_does_not_drop_zero_or_low_positive(self):
        self.assertEqual(relative_weights([0,.5,1,2,100]),[1,1,1,.5,.01])
        with self.assertRaises(ValueError):relative_weights([float('nan')])
        with self.assertRaises(ValueError):select_features([0]*82,8,'spatial')


if __name__=='__main__':unittest.main()
