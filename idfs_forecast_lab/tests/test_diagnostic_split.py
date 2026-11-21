import unittest
from idfs_lab.diagnostic_split import partition_rows, SECOND


def row(t):
    x=[0.0]*415
    for i in range(18,399,4): x[i]=1
    return dict(role='diagnostic',fault_affected=True,normal_training_allowed=False,
                exploratory_development_allowed=True,run_id='source',source_windows_sha256='hash',
                target=dict(scope='host_oss',entity='idfs44',component='total'),
                window_start_ns=t*SECOND,window_end_ns=(t+1)*SECOND,
                feature_end_ns=(t-1)*SECOND,features_available_ns=(t-1)*SECOND,
                features=x,target_read_write=[1,2],target_iops=3,attribution='storage_context_only')


class DiagnosticSplitTests(unittest.TestCase):
    def test_partitions_preserve_roles_and_original_run(self):
        rows=[row(t) for t in (5,6,359,360,420,425,426,599,600)]
        split, dropped=partition_rows(rows,0,'source','hash')
        self.assertEqual([r['window_start_ns']//SECOND for r in split['train']], [6,359])
        self.assertEqual([r['window_start_ns']//SECOND for r in split['validation']], [426,599])
        self.assertEqual(dropped['boundary_or_embargo'],5)
        self.assertTrue(all(r['role']=='diagnostic' and r['run_id']=='source'
                            for rows in split.values() for r in rows))
        self.assertNotIn('diagnostic_partition',rows[1])

    def test_no_relabeling_or_foreign_provenance(self):
        for key,value in [('role','train'),('fault_affected',False),('run_id','other'),
                          ('normal_training_allowed',True),('source_windows_sha256','other')]:
            a=row(10); a[key]=value
            with self.assertRaises(ValueError): partition_rows([a,row(430)],0,'source','hash')

    def test_duplicates_lateness_bad_masks_and_empty_split(self):
        for change in ('duplicate','late','mask','empty'):
            a,b=row(10),row(430)
            rows=[a,b]
            if change=='duplicate': rows.append(row(10))
            elif change=='late': a['features_available_ns']=10*SECOND
            elif change=='mask': a['features'][18]=0.5
            else: rows=[a]
            with self.assertRaises(ValueError): partition_rows(rows,0,'source','hash')


if __name__=='__main__': unittest.main()
