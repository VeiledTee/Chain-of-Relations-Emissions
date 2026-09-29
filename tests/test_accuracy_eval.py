"""The canonical effectiveness evaluator (chain_of_relations/eval/accuracy.py).

Regression tests for answer-only extraction, normalized exact matching and
set-based Hits@1 / Precision / Recall / F1, plus the dataset-specific gold
handling it is called with (WebQSP parses, CWQ answer list).
"""

import json
import os
import sys
import tempfile
import unittest
from unittest import mock

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "measurement"))

from chain_of_relations.eval import accuracy, webqsp_canonical  # noqa: E402
import export_outcomes  # noqa: E402


def results(*names):
	return [{"id": "", "name": n} for n in names]


def parse(names, parse_id="P0"):
	return {"parse_id": parse_id, "question_quality": "Good", "parse_quality": "Complete",
	        "answers": [{"answer_type": "Entity", "answer_argument": f"m.{i}", "entity_name": n}
	                    for i, n in enumerate(names)]}


class AnswerExtraction(unittest.TestCase):
	def test_rationale_text_is_ignored(self):
		answers, status = accuracy.extract_answers(results(
			'"rationale": "The capital might be Paris, as in France",\n  "answer": ["Lyon"]'))
		self.assertEqual((answers, status), (["Lyon"], accuracy.STATUS_JSON))
		self.assertEqual(accuracy.score(answers, [["Paris"]])["hit1"], 0)

	def test_rationale_without_answer_field_gets_no_credit(self):
		answers, status = accuracy.extract_answers(results('"rationale": "Dirk Nowitzki\'s wife is Jessica."'))
		self.assertEqual((answers, status), ([], accuracy.STATUS_MALFORMED))
		scored = accuracy.score(answers, [["Jessica Olsson"]])
		self.assertEqual((scored["hit1"], scored["recall"], scored["f1"]), (0, 0.0, 0.0))

	def test_valid_json_answer_field(self):
		answers, _ = accuracy.extract_answers(results('"rationale": "x",\n  "answer": ["Bill Roach", "John Altman"]'))
		self.assertEqual(answers, ["Bill Roach", "John Altman"])

	def test_fragmented_cor_json_is_reconstructed(self):
		answers, status = accuracy.extract_answers(results(
			'"rationale": "The candidates are languages"', '"answer": ["Jamaican English"',
			'"Jamaican Creole English Language"]'))
		self.assertEqual(status, accuracy.STATUS_JSON)
		self.assertEqual(answers, ["Jamaican English", "Jamaican Creole English Language"])
		scored = accuracy.score(answers, [["Jamaican English", "Jamaican Creole English Language"]])
		self.assertEqual((scored["precision"], scored["recall"], scored["f1"]), (1.0, 1.0, 1.0))

	def test_plain_answer_items_are_used_verbatim(self):
		self.assertEqual(accuracy.extract_answers(results("Mobile"))[0], ["Mobile"])
		self.assertEqual(accuracy.extract_answers(results("Jamaican Creole English", "Jamaican English"))[0],
		                 ["Jamaican Creole English", "Jamaican English"])

	def test_none_and_missing_results_are_empty(self):
		self.assertEqual(accuracy.extract_answers([]), ([], accuracy.STATUS_EMPTY))
		self.assertEqual(accuracy.extract_answers(None), ([], accuracy.STATUS_EMPTY))
		self.assertEqual(accuracy.extract_answers(results("None"))[0], [])


class Matching(unittest.TestCase):
	def test_substring_is_not_a_match(self):
		self.assertEqual(accuracy.hit(["Islam"], ["Shia Islam"]), 0)
		self.assertEqual(accuracy.hit(["Russia"], ["US"]), 0)

	def test_explanatory_text_cannot_create_a_hit(self):
		verbose, _ = accuracy.extract_answers(results("He was born in Mobile, Alabama."))
		self.assertEqual(accuracy.score(verbose, [["Mobile"]])["hit1"], 0)
		self.assertEqual(accuracy.score(["Mobile"], [["Mobile"]])["hit1"], 1)

	def test_normalization(self):
		self.assertEqual(accuracy.hit(["the Beatles!"], ["Beatles"]), 1)
		self.assertEqual(accuracy.normalize("  The  U.S.  Open "), "us open")


class SetMetrics(unittest.TestCase):
	def test_multiple_correct_answers(self):
		p, r, f1 = accuracy.prf(["Oslo", "Bergen", "Tromso"], ["Oslo", "Bergen", "Bodo", "Alta"])
		self.assertEqual((p, r), (2 / 3, 0.5))
		self.assertAlmostEqual(f1, 2 * (2 / 3) * 0.5 / (2 / 3 + 0.5))

	def test_duplicate_predictions_do_not_distort_precision(self):
		self.assertEqual(accuracy.prf(["Oslo", "oslo", "Oslo."], ["Oslo"]), (1.0, 1.0, 1.0))
		self.assertEqual(accuracy.prf(["Oslo", "Oslo", "Bergen"], ["Oslo"])[0], 0.5)

	def test_empty_set_rules(self):
		self.assertEqual(accuracy.prf([], ["Oslo"]), (1.0, 0.0, 0.0))
		self.assertEqual(accuracy.prf(["Oslo"], []), (0.0, 1.0, 0.0))
		self.assertEqual(accuracy.prf([], []), (1.0, 1.0, 1.0))


class WebQSPGold(unittest.TestCase):
	def test_best_parse_supplies_prf_and_union_supplies_hit(self):
		scored = webqsp_canonical.score_prediction(["Bergen"], [parse(["Oslo"], "P0"), parse(["Bergen", "Alta"], "P1")])
		self.assertEqual((scored["best_parse_id"], scored["hit"], scored["recall"]), ("P1", 1, 0.5))

	def test_ties_keep_the_earliest_parse(self):
		scored = webqsp_canonical.score_prediction(["Oslo"], [parse(["Oslo"], "P0"), parse(["Oslo"], "P1")])
		self.assertEqual(scored["best_parse_id"], "P0")

	def test_empty_gold(self):
		scored = webqsp_canonical.score_prediction(["Oslo"], [parse([])])
		self.assertEqual((scored["precision"], scored["recall"], scored["f1"], scored["hit"], scored["empty_gold"]),
		                 (0.0, 1.0, 0.0, 0, True))


class CWQGold(unittest.TestCase):
	def test_cwq_answer_list(self):
		scored = accuracy.score(["Denmark"], [["Denmark"]])
		self.assertEqual((scored["hit1"], scored["f1"], scored["empty_gold"]), (1, 1.0, False))
		self.assertEqual(accuracy.score(["Shia Islam"], [["Islam"]])["hit1"], 0)


class SameEvaluatorForEverySystem(unittest.TestCase):
	RECORD = {"id": "q1", "action": "stop", "gold_answer": [{"id": "m.1", "name": "Oslo"}],
	          "results": results('"rationale": "Oslo or Bergen", "answer": ["Oslo"]')}

	def test_pog_tog_cor_rows_are_scored_identically(self):
		with tempfile.TemporaryDirectory() as tmp:
			path = os.path.join(tmp, "predict.jsonl")
			with open(path, "w") as f:
				f.write(json.dumps(self.RECORD) + "\n")
			with mock.patch.object(accuracy, "score", wraps=accuracy.score) as spy:
				rows = {s: export_outcomes.score(path, "cwq", s)[0] for s in ("PoG", "ToG", "CoR")}
			self.assertEqual(spy.call_count, 3)
			metrics = {s: tuple(r[k] for k in ("hit1", "precision", "recall", "f1")) for s, r in rows.items()}
			self.assertEqual(len(set(metrics.values())), 1)
			self.assertEqual(metrics["PoG"], (1, 1.0, 1.0, 1.0))

	def test_webqsp_rows_share_the_parse_scorer(self):
		gold = {"q1": {"parses": [parse(["Oslo"])]}}
		record = dict(self.RECORD)
		with tempfile.TemporaryDirectory() as tmp:
			path = os.path.join(tmp, "predict.jsonl")
			with open(path, "w") as f:
				f.write(json.dumps(record) + "\n")
			with mock.patch.object(webqsp_canonical, "load_gold", return_value=gold), \
			     mock.patch.object(accuracy, "score", wraps=accuracy.score) as spy:
				rows = [export_outcomes.score(path, "webqsp", s)[0] for s in ("PoG", "ToG", "CoR")]
			self.assertEqual(spy.call_count, 3)
			self.assertTrue(all((r["hit1"], r["f1"]) == (1, 1.0) for r in rows))


if __name__ == "__main__":
	unittest.main()
