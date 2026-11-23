import unittest
from idfs_lab.v2_auxiliary_inputs import index_auxiliary,auxiliary_features


class AuxiliaryTests(unittest.TestCase):
    def row(self,end=1_900_000_000,**updates):
        row=dict(dataset_id='run',scope='host_oss',entity='idfs44',
            bucket_start_ns=end-100_000_000,bucket_end_ns=end,sample_ok=True,
            expected_oss=12,observed_oss=12,available_epoch_ns=end+1_000_000,
            read_bytes=0,write_bytes=128,read_count=0,write_count=2,
            read_bytes_valid=True,write_bytes_valid=True,
            op_wip_valid=True,op_wip_sum=4,op_wip_max=2)
        row.update(updates)
        return row

    def build(self,rows):
        return auxiliary_features(index_auxiliary(rows,'run',['idfs44']),2_000_000_000,'idfs44',0)

    def test_true_zero_ratio_missing_and_dimensions(self):
        got=self.build([self.row()])
        b=dict(zip(got['B_names'],got['B']))
        self.assertEqual(len(b),180)
        self.assertEqual(len(got['P']),80)
        self.assertEqual(b['aux_lag2_read_bytes_observed'],1)
        self.assertEqual(b['aux_lag2_read_bytes_per_operation_observed'],0)
        self.assertEqual(b['aux_lag2_write_bytes_per_operation'],64)
        self.assertAlmostEqual(b['aux_lag2_byte_age_seconds'],.07)

    def test_cutoff_and_future_bucket_do_not_leak(self):
        good=self.row(available_epoch_ns=1_920_000_000)
        self.assertEqual(self.build([good])['source_available_upper_ns'],1_920_000_000)
        late=self.row(available_epoch_ns=1_920_000_001)
        self.assertIsNone(self.build([late,self.row(end=2_000_000_000)])['source_available_upper_ns'])

    def test_missing_fields_keep_keys_and_other_direction(self):
        row=self.row();row.update(read_bytes=None,read_bytes_valid=False,
                                  op_wip_valid=False,op_wip_sum=None,op_wip_max=None)
        got=self.build([row]);b=dict(zip(got['B_names'],got['B']))
        self.assertEqual(b['aux_lag2_read_bytes_observed'],0)
        self.assertEqual(b['aux_lag2_write_bytes_per_operation'],64)
        self.assertTrue(all(v==0 for v in got['P']))

    def test_duplicate_and_invalid_field_masks_fail(self):
        with self.assertRaises(ValueError): self.build([self.row(),self.row()])
        row=self.row();row['read_bytes_valid']=False
        with self.assertRaises(ValueError): self.build([row])
        row=self.row();row['op_wip_max']=5
        with self.assertRaises(ValueError): self.build([row])


if __name__=='__main__':
    unittest.main()
