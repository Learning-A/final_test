"""Hand-computed safeguards for frozen-error attribution."""
import importlib.util
from pathlib import Path
import unittest

SRC = Path(__file__).resolve().parents[2] / 'scripts/analyze_v2_d0_frozen.py'
SPEC = importlib.util.spec_from_file_location('d0_frozen', SRC)
d0 = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(d0)


class AttributionTests(unittest.TestCase):
    def test_zero_is_retained_in_mae_but_excluded_from_mape(self):
        s=d0.new_stats()
        d0.add(s, 0, 4)
        d0.add(s, 2, 4)
        got=d0.finish(s, 1)
        self.assertEqual(got['mape_percent'],100)
        self.assertEqual(got['mae'],3)
        self.assertEqual(got['zero_count'],1)
        self.assertEqual(got['wape_percent'],300)
        self.assertEqual(got['macro_mape_contribution_pp'],2.5)

    def test_additive_contribution_uses_full_group_denominator(self):
        low,high=d0.new_stats(),d0.new_stats()
        d0.add(low,2,4)
        d0.add(high,100,110)
        a,b=d0.finish(low,2),d0.finish(high,2)
        self.assertEqual(a['mape_percent'],100)
        self.assertEqual(b['mape_percent'],10)
        self.assertAlmostEqual(a['macro_mape_contribution_pp']+b['macro_mape_contribution_pp'],55/40)

    def test_keys_must_be_complete_unique_and_integer(self):
        keys={('a',1),('a',2)}
        row={'dataset_id':'a','window_start_ns':1}
        for rows in ([row],[row,row],[dict(row,window_start_ns=1.0)]):
            with self.subTest(rows=rows), self.assertRaises(ValueError):
                d0.keyed(rows,keys)

    def test_nonfinite_negative_and_zero_denominators(self):
        for y,p in ((-1,0),(1,-1),(1,float('nan'))):
            with self.assertRaises(ValueError):
                d0.add(d0.new_stats(),y,p)
        empty=d0.finish(d0.new_stats())
        self.assertIsNone(empty['mape_percent'])
        self.assertIsNone(empty['wape_percent'])


if __name__ == '__main__':
    unittest.main()
