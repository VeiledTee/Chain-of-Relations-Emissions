"""Supervisor summary statistics, computed from fixtures with known answers.

Every number here is hand-checkable from the tiny fixture below, so a change in
the aggregation shows up as a failing arithmetic assertion rather than a plot
that looks slightly different. The script must stay dataset-agnostic: nothing in
these fixtures is WebQSP-shaped.
"""

import csv
import json
import os
import sys
import tempfile
import unittest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "measurement"))

import summarize_comparable_runs as scr  # noqa: E402


def write_run(directory, questions):
	"""questions: qid -> {energy, depth, events}. events: list of event dicts."""
	os.makedirs(directory, exist_ok=True)
	with open(os.path.join(directory, "trajectory_summary.csv"), "w", newline="") as f:
		writer = csv.writer(f, lineterminator="\n")
		writer.writerow(["question_id", "trajectory_gpu_energy_j", "max_traversal_depth"])
		for qid, spec in questions.items():
			writer.writerow([qid, spec["energy"], spec["depth"]])
	with open(os.path.join(directory, "events_attributed.jsonl"), "w", newline="") as f:
		for qid, spec in questions.items():
			for index, event in enumerate(spec["events"]):
				row = dict(event)
				row.setdefault("question_id", qid)
				row.setdefault("step_index", index)
				f.write(json.dumps(row) + "\n")
	return directory


def write_outcomes(path, rows):
	"""rows: qid -> (outcome, f1, hit1) or (outcome, f1, hit1, empty_gold)."""
	fields = ("question_id", "run", "outcome", "hit1", "f1", "empty_gold")
	with open(path, "w", newline="") as f:
		writer = csv.DictWriter(f, fieldnames=fields, lineterminator="\n")
		writer.writeheader()
		for qid, values in rows.items():
			outcome, f1, hit1 = values[:3]
			empty = values[3] if len(values) > 3 else False
			writer.writerow({"question_id": qid, "run": "T", "outcome": outcome,
			                 "hit1": hit1, "f1": f1, "empty_gold": empty})
	return path


def llm(gpu, **kw):
	return dict({"operation_type": "llm", "operation_label": "llm:reason",
	             "gpu_energy_j": gpu, "input_tokens": 100, "output_tokens": 10}, **kw)


def kg(gpu):
	return {"operation_type": "kg", "operation_label": "kg:id2name", "gpu_energy_j": gpu}


def fallback_event(gpu, reason="no_graph_answer"):
	return {"operation_type": "llm", "operation_label": "llm:direct_answer",
	        "gpu_energy_j": gpu, "input_tokens": 50, "output_tokens": 5,
	        "meta": {"fallback": True, "fallback_reason": reason}}


#: Ten questions: energies 100..1000, two of them fallback trajectories.
FIXTURE = {}
for i in range(1, 11):
	qid = f"q{i}"
	events = [kg(1.0), llm(9.0)]
	if i in (9, 10):                       # the two most expensive are fallbacks
		events = [kg(1.0), llm(9.0), fallback_event(10.0,
		          "depth_exhausted" if i == 10 else "no_graph_answer")]
	FIXTURE[qid] = {"energy": 100.0 * i, "depth": (i % 3), "events": events}

#: q1..q6 found, q7..q10 not found.
OUTCOMES = {f"q{i}": (scr.HIT if i <= 6 else scr.MISS, 1.0 if i <= 6 else 0.0,
                      1 if i <= 6 else 0) for i in range(1, 11)}


class SummaryFixture(unittest.TestCase):

	def setUp(self):
		self.tmp = tempfile.TemporaryDirectory()
		self.addCleanup(self.tmp.cleanup)
		self.run_dir = write_run(os.path.join(self.tmp.name, "run"), FIXTURE)
		self.outcomes = write_outcomes(os.path.join(self.tmp.name, "out.csv"), OUTCOMES)
		self.summary = scr.summarize("CoR", self.run_dir, self.outcomes)


class TestCoreStatistics(SummaryFixture):

	def test_question_counts_and_outcomes(self):
		self.assertEqual(self.summary["questions_run"], 10)
		self.assertEqual(self.summary["gold_found"], 6)
		self.assertEqual(self.summary["gold_not_found"], 4)
		self.assertAlmostEqual(self.summary["gold_found_rate_pct"], 60.0)

	def test_energy_distribution(self):
		"""Energies are 100..1000: mean 550, median 550, total 5500, p90 900."""
		self.assertAlmostEqual(self.summary["mean_gpu_j"], 550.0)
		self.assertAlmostEqual(self.summary["median_gpu_j"], 550.0)
		self.assertAlmostEqual(self.summary["total_gpu_j"], 5500.0)
		self.assertAlmostEqual(self.summary["p90_gpu_j"], 900.0)

	def test_top_ten_percent_share(self):
		"""ceil(0.1 * 10) = 1 question, the 1000 J one: 1000/5500 = 18.18%."""
		self.assertAlmostEqual(self.summary["top10pct_energy_share_pct"],
		                       100.0 * 1000.0 / 5500.0, places=6)

	def test_outcome_conditioned_energy(self):
		"""found = 100..600 -> 350; not found = 700..1000 -> 850."""
		self.assertAlmostEqual(self.summary["mean_gpu_j_gold_found"], 350.0)
		self.assertAlmostEqual(self.summary["mean_gpu_j_gold_not_found"], 850.0)

	def test_workload_means(self):
		"""Eight questions have 1 LLM event, two have 2 -> 1.2 mean calls."""
		self.assertAlmostEqual(self.summary["mean_llm_calls"], 1.2)
		self.assertAlmostEqual(self.summary["mean_output_tokens"], (8 * 10 + 2 * 15) / 10)
		self.assertAlmostEqual(self.summary["mean_input_tokens"], (8 * 100 + 2 * 150) / 10)

	def test_effectiveness_comes_from_the_outcome_csv(self):
		"""Six questions at F1 1.0 out of ten -> 60%, not a recomputation."""
		self.assertAlmostEqual(self.summary["f1_pct"], 60.0)
		self.assertAlmostEqual(self.summary["hit1_pct"], 60.0)

	def test_effectiveness_override_wins(self):
		override = scr.summarize("CoR", self.run_dir, self.outcomes,
		                         {"f1": 43.54, "hit": 62.48})
		self.assertAlmostEqual(override["f1_pct"], 43.54)
		self.assertAlmostEqual(override["hit1_pct"], 62.48)


class TestDepthTable(SummaryFixture):

	def test_depth_rows_cover_every_depth(self):
		rows = {r["max_traversal_depth"]: r for r in self.summary["_depth_rows"]}
		self.assertEqual(sorted(rows), [0, 1, 2])
		self.assertEqual(sum(r["total"] for r in rows.values()), 10)

	def test_depth_counts_are_right(self):
		"""depth = i % 3 over q1..q10; found is q1..q6."""
		rows = {r["max_traversal_depth"]: r for r in self.summary["_depth_rows"]}
		# depth 0: q3, q6, q9 -> found q3, q6; not found q9
		self.assertEqual((rows[0]["gold_found"], rows[0]["gold_not_found"]), (2, 1))
		# depth 1: q1, q4, q7, q10 -> found q1, q4
		self.assertEqual((rows[1]["gold_found"], rows[1]["gold_not_found"]), (2, 2))
		# depth 2: q2, q5, q8 -> found q2, q5
		self.assertEqual((rows[2]["gold_found"], rows[2]["gold_not_found"]), (2, 1))
		self.assertAlmostEqual(rows[1]["gold_found_rate_pct"], 50.0)


class TestFallback(SummaryFixture):

	def test_fallback_counts_and_rates(self):
		"""q9 and q10 are fallbacks; both are misses."""
		self.assertEqual(self.summary["fallback_questions"], 2)
		self.assertAlmostEqual(self.summary["fallback_rate_pct"], 20.0)
		self.assertEqual(self.summary["fallback_gold_found"], 0)
		self.assertEqual(self.summary["fallback_gold_not_found"], 2)
		self.assertAlmostEqual(self.summary["fallback_success_rate_pct"], 0.0)
		# non-fallback: q1..q8, six found -> 75%
		self.assertAlmostEqual(self.summary["non_fallback_success_rate_pct"], 75.0)

	def test_fallback_conditioned_energy(self):
		"""fallback = 900, 1000 -> 950; non-fallback = 100..800 -> 450."""
		self.assertAlmostEqual(self.summary["mean_gpu_j_fallback"], 950.0)
		self.assertAlmostEqual(self.summary["mean_gpu_j_non_fallback"], 450.0)

	def test_phase_split_uses_attributed_event_energy(self):
		"""Each fallback question: 10 J before, 10 J in the fallback call, 0 after."""
		self.assertAlmostEqual(self.summary["mean_gpu_j_before_fallback"], 10.0)
		self.assertAlmostEqual(self.summary["mean_gpu_j_in_fallback"], 10.0)
		self.assertAlmostEqual(self.summary["mean_gpu_j_after_fallback"], 0.0)
		self.assertAlmostEqual(self.summary["pct_fallback_energy_before"], 50.0)
		self.assertAlmostEqual(self.summary["pct_fallback_energy_in"], 50.0)

	def test_fallback_reason_breakdown(self):
		rows = {r["fallback_reason"]: r for r in self.summary["_reason_rows"]}
		self.assertEqual(sorted(rows), ["depth_exhausted", "no_graph_answer"])
		self.assertEqual(rows["no_graph_answer"]["n"], 1)
		self.assertAlmostEqual(rows["no_graph_answer"]["pct_of_fallback"], 50.0)
		self.assertAlmostEqual(rows["depth_exhausted"]["mean_gpu_j"], 1000.0)
		self.assertAlmostEqual(rows["depth_exhausted"]["gold_found_rate_pct"], 0.0)

	def test_a_run_without_fallback_reports_none_not_zero(self):
		plain = {"q1": {"energy": 10.0, "depth": 1, "events": [llm(1.0)]}}
		run_dir = write_run(os.path.join(self.tmp.name, "plain"), plain)
		outcomes = write_outcomes(os.path.join(self.tmp.name, "plain.csv"),
		                          {"q1": (scr.HIT, 1.0, 1)})
		summary = scr.summarize("PoG", run_dir, outcomes)
		self.assertEqual(summary["fallback_questions"], 0)
		self.assertIsNone(summary["pct_fallback_energy_before"])
		self.assertIsNone(summary["mean_gpu_j_fallback"])
		self.assertEqual(summary["_reason_rows"], [])


class TestJoinAndOrdering(SummaryFixture):

	def test_questions_without_an_outcome_are_excluded_and_counted(self):
		partial = dict(OUTCOMES)
		del partial["q10"]
		outcomes = write_outcomes(os.path.join(self.tmp.name, "partial.csv"), partial)
		summary = scr.summarize("ToG", self.run_dir, outcomes)
		self.assertEqual(summary["questions_run"], 9)
		self.assertEqual(summary["questions_missing_outcome"], 1)

	def test_empty_gold_questions_are_counted_not_dropped(self):
		rows = dict(OUTCOMES)
		rows["q7"] = (scr.MISS, 0.0, 0, True)
		outcomes = write_outcomes(os.path.join(self.tmp.name, "empty.csv"), rows)
		summary = scr.summarize("ToG", self.run_dir, outcomes)
		self.assertEqual(summary["questions_run"], 10)
		self.assertEqual(summary["empty_gold_questions"], 1)

	def test_system_display_order_is_pog_tog_cor(self):
		self.assertEqual(scr.order_systems({"CoR", "PoG", "ToG"}), ["PoG", "ToG", "CoR"])
		self.assertEqual(scr.order_systems({"CoR", "ToG"}), ["ToG", "CoR"])

	def test_unknown_systems_sort_after_the_known_ones(self):
		self.assertEqual(scr.order_systems({"CoR", "SubgraphRAG", "PoG"}),
		                 ["PoG", "CoR", "SubgraphRAG"])


class TestOutputs(SummaryFixture):

	def test_cli_writes_every_artifact(self):
		out = os.path.join(self.tmp.name, "tables")
		rc = scr.main(["--run", f"CoR={self.run_dir}",
		               "--outcomes", f"CoR={self.outcomes}",
		               "--dataset", "cwq", "--out", out])
		self.assertEqual(rc, 0)
		for name in ("summary_overview.csv", "summary_fallback.csv",
		             "summary_depth_outcome.csv", "summary_fallback_reasons.csv",
		             "SUMMARY.md"):
			self.assertTrue(os.path.exists(os.path.join(out, name)), name)

	def test_overview_csv_has_systems_as_columns(self):
		out = os.path.join(self.tmp.name, "tables2")
		scr.main(["--run", f"CoR={self.run_dir}", "--run", f"PoG={self.run_dir}",
		          "--outcomes", f"CoR={self.outcomes}", "--outcomes", f"PoG={self.outcomes}",
		          "--out", out])
		with open(os.path.join(out, "summary_overview.csv")) as f:
			header = next(csv.reader(f))
		self.assertEqual(header, ["metric", "PoG", "CoR"])

	def test_markdown_reports_the_dataset_label(self):
		out = os.path.join(self.tmp.name, "tables3")
		scr.main(["--run", f"CoR={self.run_dir}", "--outcomes", f"CoR={self.outcomes}",
		          "--dataset", "cwq", "--out", out])
		with open(os.path.join(out, "SUMMARY.md")) as f:
			text = f.read()
		self.assertIn("`cwq`", text)
		self.assertIn("Fallback reasons", text)

	def test_missing_outcome_file_fails_loudly(self):
		with self.assertRaises(SystemExit):
			scr.summarize("CoR", self.run_dir, os.path.join(self.tmp.name, "nope.csv"))

	def test_unattributed_run_fails_loudly(self):
		empty = os.path.join(self.tmp.name, "empty_run")
		os.makedirs(empty, exist_ok=True)
		with self.assertRaises(SystemExit):
			scr.summarize("CoR", empty, self.outcomes)


if __name__ == "__main__":
	unittest.main()
