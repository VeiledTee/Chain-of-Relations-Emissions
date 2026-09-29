"""Correct vs incorrect energy analysis: category boundaries, partition checks, arithmetic."""

import os
import sys
import unittest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "measurement"))

import correctness_energy as ce  # noqa: E402


def row(qid, f1, energy, hit1=None, **kw):
	base = {"system": "T", "question_id": qid, "f1": f1,
	        "hit1": (1 if f1 > 0 else 0) if hit1 is None else hit1, "empty_gold": False,
	        "binary_category": ce.binary_category(f1),
	        "threeway_category": ce.threeway_category(f1),
	        "trajectory_gpu_energy_j": energy, "input_tokens": 100, "output_tokens": 10,
	        "llm_calls": 2, "kg_calls": 3, "max_traversal_depth": 1, "max_iteration": 1,
	        "wall_s": 5.0, "fallback": False}
	base.update(kw)
	return base


class CategoryTest(unittest.TestCase):
	def test_binary_boundary(self):
		self.assertEqual(ce.binary_category(1.0), ce.CORRECT)
		for f1 in (0.0, 0.5, 0.999999):
			self.assertEqual(ce.binary_category(f1), ce.INCORRECT)

	def test_threeway_boundaries(self):
		self.assertEqual(ce.threeway_category(1.0), ce.FULLY)
		self.assertEqual(ce.threeway_category(0.4), ce.PARTIAL)
		self.assertEqual(ce.threeway_category(0.0), ce.WRONG)

	def test_missing_f1_is_an_error_not_incorrect(self):
		with self.assertRaises(ValueError):
			ce.binary_category(None)
		with self.assertRaises(ValueError):
			ce.threeway_category(None)

	def test_gold_found_is_not_correct(self):
		# Hit@1 = 1 with a partial answer set must land in Incorrect.
		r = row("q", 0.5, 10.0, hit1=1)
		self.assertEqual(r["binary_category"], ce.INCORRECT)
		self.assertEqual(r["threeway_category"], ce.PARTIAL)


class PartitionTest(unittest.TestCase):
	def test_counts_cover_every_question(self):
		rows = [row("a", 1.0, 1), row("b", 0.5, 1), row("c", 0.0, 1), row("d", 0.0, 1)]
		binary, three = ce.validate_partition("T", rows, n_expected=4)
		self.assertEqual(binary, {ce.CORRECT: 1, ce.INCORRECT: 3})
		self.assertEqual(three, {ce.FULLY: 1, ce.PARTIAL: 1, ce.WRONG: 2})

	def test_wrong_denominator_fails(self):
		with self.assertRaises(SystemExit):
			ce.validate_partition("T", [row("a", 1.0, 1)], n_expected=1639)

	def test_duplicate_question_fails(self):
		with self.assertRaises(SystemExit):
			ce.validate_partition("T", [row("a", 1.0, 1), row("a", 0.0, 1)], n_expected=2)

	def test_real_denominator_constant(self):
		self.assertEqual(ce.N_QUESTIONS, 1639)


class ArithmeticTest(unittest.TestCase):
	def setUp(self):
		self.rows = [row("a", 1.0, 100.0), row("b", 1.0, 300.0),
		             row("c", 0.5, 400.0, fallback=True, llm_calls=4),
		             row("d", 0.0, 200.0), row("e", 0.0, 600.0, fallback=True)]

	def test_binary_row(self):
		b = ce.binary_row("T", self.rows)
		self.assertEqual((b["correct_n"], b["incorrect_n"]), (2, 3))
		self.assertEqual(b["mean_gpu_j_correct"], 200.0)
		self.assertEqual(b["median_gpu_j_correct"], 200.0)
		self.assertEqual(b["mean_gpu_j_incorrect"], 400.0)
		self.assertEqual(b["median_gpu_j_incorrect"], 400.0)
		self.assertEqual(b["incorrect_correct_energy_ratio"], 2.0)
		self.assertEqual(b["percent_difference"], 100.0)

	def test_threeway_rows(self):
		t = {r["category"]: r for r in ce.threeway_rows("T", self.rows)}
		self.assertEqual(sum(r["n"] for r in t.values()), 5)
		partial = t[ce.THREEWAY_LABELS[ce.PARTIAL]]
		self.assertEqual((partial["n"], partial["mean_gpu_j"], partial["mean_llm_calls"],
		                  partial["fallback_rate_pct"]), (1, 400.0, 4, 100.0))
		wrong = t[ce.THREEWAY_LABELS[ce.WRONG]]
		self.assertEqual((wrong["mean_gpu_j"], wrong["fallback_rate_pct"]), (400.0, 50.0))

	def test_crosstab(self):
		cross = {r["category"]: r for r in ce.hit_crosstab_rows("T", self.rows)}
		self.assertEqual(cross[ce.THREEWAY_LABELS[ce.PARTIAL]]["gold_found"], 1)
		self.assertEqual(sum(r["total"] for r in cross.values()), 5)


if __name__ == "__main__":
	unittest.main()
