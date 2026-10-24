import importlib.util
from pathlib import Path
import unittest
import numpy as np
spec=importlib.util.spec_from_file_location('metrics',Path(__file__).parents[1]/'scripts/analyze_pilot_metrics.py')
m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)

class MetricTests(unittest.TestCase):
    def test_correlations_known_values(self):
        x=np.arange(20,dtype=float)
        self.assertAlmostEqual(float(m.correlation(x,x)),1.)
        self.assertAlmostEqual(float(m.correlation(x,-x)),-1.)

    def test_question_level_support(self):
        self.assertEqual(m.statistics([1,2,3],[0,0,1])['status'],'insufficient_support')
        self.assertEqual(m.statistics(np.ones(20),[0,1]*10)['status'],'insufficient_support')

    def test_adjustment_removes_length_confound(self):
        rng=np.random.default_rng(18);c=rng.normal(size=160);x=c+rng.normal(scale=.03,size=160);y=(c>0).astype(float)
        a=m.statistics(x,y,bootstrap=100);b=m.statistics(x,y,c[:,None],bootstrap=100)
        self.assertGreater(a['r'],.7);self.assertLess(abs(b['r']),.2)

    def test_cap_all_zero_bootstrap_is_supported(self):
        x=np.arange(24);y=np.array([0,1]*12)
        result=m.statistics(x,y,np.c_[np.log(x+1),np.zeros(24)],bootstrap=30)
        self.assertEqual(result['valid_bootstraps'],30)

    def test_shuffled_difference_and_missing_are_distinct(self):
        row={'overlap_lag16':.7,'overlap_shuffled_lag16':.2,'full_jsd':.1,'full_shuffled_jsd':.04}
        v=dict(zip(m.METRICS,m.metric_values(row)))
        self.assertAlmostEqual(v['overlap_excess_lag16'],.5)
        self.assertAlmostEqual(v['full_jsd_excess'],.06)
        self.assertTrue(np.isnan(v['mixture_jsd_excess']))

    def test_averages_do_not_turn_missing_into_zero(self):
        d={};m.add(d,'x',np.array([1.,np.nan]));m.add(d,'x',np.array([3.,4.]))
        np.testing.assert_allclose(m.mean(d['x']),[2.,4.])

if __name__=='__main__': unittest.main()
