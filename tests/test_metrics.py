from pathlib import Path
import tempfile
import unittest
from app import evaluate,load_cases,main,valid_prediction

CASES=[{'id':'1','input':'a','expected':{'category':'compras','needs_review':False}},
       {'id':'2','input':'b','expected':{'category':'financeiro','needs_review':False}}]

class MetricsTests(unittest.TestCase):
    def test_perfect_predictions(self):
        rows=[{'id':r['id'],'prediction':r['expected']} for r in CASES]
        result=evaluate(CASES,rows)
        self.assertEqual(result['exact_accuracy'],1);self.assertEqual(result['macro_f1'],1)
    def test_missing_prediction_counts_as_failure(self):
        result=evaluate(CASES,[{'id':'1','prediction':CASES[0]['expected']}])
        self.assertEqual(result['exact_accuracy'],.5);self.assertEqual(result['format_valid_rate'],.5)
    def test_review_flag_is_part_of_exact_match(self):
        result=evaluate(CASES,[{'id':'1','prediction':{'category':'compras','needs_review':True}}])
        self.assertEqual(result['exact_accuracy'],0);self.assertEqual(result['category_accuracy'],.5)
    def test_duplicate_predictions_are_rejected(self):
        with self.assertRaises(ValueError):evaluate(CASES,[{'id':'1'},{'id':'1'}])
    def test_unknown_case_is_rejected(self):
        with self.assertRaises(ValueError):evaluate(CASES,[{'id':'x'}])
    def test_extra_fields_and_string_booleans_are_invalid(self):
        self.assertFalse(valid_prediction({'category':'compras','needs_review':'false'}))
        self.assertFalse(valid_prediction({'category':'compras','needs_review':False,'extra':1}))
    def test_macro_f1_matches_hand_calculated_example(self):
        rows=[{'id':'1','prediction':{'category':'compras','needs_review':False}},
              {'id':'2','prediction':{'category':'compras','needs_review':False}}]
        self.assertAlmostEqual(evaluate(CASES,rows)['macro_f1'],1/3)
    def test_demo_gate_fails_and_writes_report(self):
        with tempfile.TemporaryDirectory(dir=Path(__file__).resolve().parent) as temp:
            code=main(['replay','--output',temp])
            self.assertEqual(code,1)
            report=(Path(temp)/'report.md').read_text(encoding='utf-8')
            self.assertIn('REPROVADO',report);self.assertIn('Não é benchmark',report)

if __name__=='__main__':unittest.main()
