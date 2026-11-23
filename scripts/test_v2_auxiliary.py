import unittest
from export_v2_auxiliary import aggregate, byte_delta


class AuxiliaryContract(unittest.TestCase):
    def member(self, **updates):
        return dict(core_ok=True,source_end=101,read_count=0,write_count=2,
                    read_bytes=0,write_bytes=128,op_wip=3,**updates)

    def test_zero_ops_and_gauge_are_not_fabricated_rates(self):
        row=aggregate('run','node',100,{'0':self.member(),'1':self.member()},['0','1'])
        self.assertTrue(row['sample_ok'])
        self.assertEqual(row['read_bytes'],0)
        self.assertIsNone(row['read_bytes_per_operation'])
        self.assertEqual(row['write_bytes_per_operation'],64)
        self.assertEqual(row['op_wip_sum'],6)
        self.assertEqual(row['op_wip_max'],3)

    def test_missing_member_is_not_zero(self):
        row=aggregate('run','node',100,{'0':self.member()},['0','1'])
        self.assertFalse(row['sample_ok'])
        self.assertIsNone(row['read_bytes'])
        self.assertFalse(row['op_wip_valid'])

    def test_byte_reset_invalidates_field_without_changing_core(self):
        self.assertIsNone(byte_delta({'v':8},{'v':2},'v',True))
        self.assertIsNone(byte_delta({'v':8},{'v':9},'v',False))
        self.assertEqual(byte_delta({'v':8},{'v':9},'v',True),1)
        a=self.member();a['read_bytes']=None
        row=aggregate('run','node',100,{'0':a},['0'])
        self.assertTrue(row['sample_ok'])
        self.assertFalse(row['read_bytes_valid'])
        self.assertTrue(row['write_bytes_valid'])


if __name__=='__main__':
    unittest.main()
