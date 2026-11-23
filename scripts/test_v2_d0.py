"""Boundary checks for the event/probe contract (synthetic counters only)."""
import unittest
from analyze_v2_d0 import GRID, SECOND, label_states, probe, score, segments


class D0Contract(unittest.TestCase):
    def test_no_fallback_or_late_bucket(self):
        start=20*SECOND
        buckets={start-i*GRID:dict(sample_ok=True,bucket_end_ns=start-i*GRID,
                    available_epoch_ns=start-i*GRID+1,total_count=100) for i in range(1,11)}
        buckets[start-GRID]['total_count']=0
        self.assertTrue(probe(buckets,start)['alert'])
        buckets[start-GRID]['available_epoch_ns']=start-79_999_999
        self.assertFalse(probe(buckets,start)['observable'])
        del buckets[start-GRID]
        self.assertFalse(probe(buckets,start)['observable'])

    def test_cutoff_and_historical_reference(self):
        cluster={i*SECOND:dict(total=1000 if i<4 else 1,end=(i+1)*SECOND,
                              available=(i+1)*SECOND+1) for i in range(7)}
        states=label_states(cluster)
        self.assertIsNone(states[SECOND]['quiet'])
        self.assertEqual(states[4*SECOND]['phase'],'known_onset')
        self.assertEqual(states[5*SECOND]['phase'],'quiet_continuation')

    def test_gaps_censor_and_unknown_alert_is_not_negative(self):
        rows={0:dict(quiet=None,phase='unknown'),SECOND:dict(quiet=True,phase='quiet_left_censored_start'),
              2*SECOND:dict(quiet=True,phase='quiet_continuation'),4*SECOND:dict(quiet=False,phase='ordinary')}
        events=segments(rows)
        self.assertEqual(len(events),1)
        self.assertTrue(events[0]['left_censored'] and events[0]['right_censored'])
        scores=score(rows,{s:dict(alert=True,observable=True,reason='eligible') for s in rows})
        self.assertEqual(scores['counts']['alert_unknown_label'],1)
        self.assertEqual(scores['counts']['known_alerts'],3)
        self.assertAlmostEqual(scores['precision'],2/3)


if __name__=='__main__':
    unittest.main()
