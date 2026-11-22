import unittest
from idfs_lab.low_state import quiet_state,cluster_input


class LowStateTests(unittest.TestCase):
    def test_fixed_auxiliary_definition_and_no_future_feature_access(self):
        self.assertEqual(quiet_state(0,0),1)
        self.assertEqual(quiet_state(11,100),0)
        self.assertEqual(quiet_state(100,1000),1)
        row={'x':[[1.]*144 for _ in range(8)],'extra':[[2.]*38 for _ in range(8)],'y':[3.]*9}
        before=cluster_input(row);row['y']=[99999.]*9
        self.assertEqual(before,cluster_input(row));self.assertEqual(len(before),1092)
        with self.assertRaises(ValueError):quiet_state(float('nan'),100)


if __name__=='__main__':unittest.main()
