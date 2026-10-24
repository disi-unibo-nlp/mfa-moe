import sys,unittest
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
from measurement_study.production_supplement import temporal_features,class_features
from measurement_study.production_report import macro_stats
from measurement_study.contract import DATASETS
class SupplementTests(unittest.TestCase):
    def test_unknown_endpoints_do_not_enter_class_contrast(self):
        ss=[{'label':'Analyze','token_indices':[0]},{'label':'Verify','token_indices':[1]},{'label':None,'token_indices':[2]}]
        windows=[dict(layer=3,window=64,end_token=i+1,selection_entropy=v) for i,v in enumerate((2.,4.,100.))]
        features,detail=temporal_features(windows,ss,[3])
        self.assertEqual(features['class_routing_contrast/Analyze/W64/selection_entropy'],-2.)
        self.assertEqual(detail['Analyze/L3/W64/selection_entropy']['other_labeled_windows'],1)
    def test_absent_patterns_zero_only_with_observed_denominator(self):
        empty={'motifs':[],'transitions':[]}
        self.assertNotIn('motif_2/Analyze/Verify',class_features([],empty))
        f=class_features([],{'motifs':[{'pattern':['Analyze','Verify']}],'transitions':[]})
        self.assertEqual(f['motif_2/Analyze/Verify'],1)
        self.assertEqual(f['motif_2/Verify/Analyze'],0)
        self.assertNotIn('motif_3/Analyze/Verify/Plan',f)
    def test_macro_weights_benchmarks_equally(self):
        groups={d:[dict(question_id=str(i),is_correct=(j==0),completion_tokens=10,capped=False) for i in range(1 if j==0 else 20)] for j,d in enumerate(DATASETS)}
        r=macro_stats(groups,bootstrap=10)
        self.assertAlmostEqual(r['metrics']['accuracy']['value'],1/6)
        groups.pop(DATASETS[0]);self.assertEqual(macro_stats(groups)['status'],'unavailable_incomplete_benchmarks')
if __name__=='__main__':unittest.main()
