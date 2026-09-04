"""agent_energy_profiler.aggregate: reconciliation and honesty guarantees.

The aggregate layer is where a measurement becomes a figure, so it is where a
missing domain most easily turns into a confident zero slice, a partial sum
into a whole, and a negative residual into a tidy 100%. These tests pin the
behaviour that stops that happening.
"""

import ast
import json
import os
import sys
import tempfile
import unittest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from agent_energy_profiler import aggregate, trajectory  # noqa: E402

RUNS = os.path.join(ROOT, "measurement", "runs")


def event(**overrides):
	"""A schema-v1-shaped attributed event with sane defaults."""
	base = {
		"schema_version": 1,
		"event_id": "e",
		"run_id": "run-1",
		"question_id": "q1",
		"dataset": "webqsp",
		"paradigm": "cor",
		"iteration": 0,
		"traversal_depth": 0,
		"step_index": 0,
		"operation_type": "llm",
		"operation_label": "llm:reason",
		"start_timestamp": 100.0,
		"end_timestamp": 101.0,
		"duration_s": 1.0,
		"status": "ok",
		"meta": {},
		"gpu_energy_j": 10.0,
		"cpu_package_energy_j": None,
		"dram_energy_j": None,
		"measured_energy_j": None,
		"measurement_complete": False,
	}
	base.update(overrides)
	return base


def window(energies):
	"""A window_energy callable returning fixed trajectory-level energy."""
	def measure(t0, t1):
		return dict(energies)
	return measure


# --------------------------------------------------------------------------
# 1. additivity and reconciliation
# --------------------------------------------------------------------------

class TestReconciliation(unittest.TestCase):

	def build(self, events, trajectory_gpu, group_by=("operation_label",)):
		trajectories = trajectory.accumulate(
			events, window_energy=window({
				"gpu_energy_j": trajectory_gpu,
				"cpu_package_energy_j": None,
				"dram_energy_j": None,
				"measured_energy_j": None,
			}))
		rows = aggregate.aggregate(events, trajectories, group_by=group_by,
		                           domains=("gpu_energy_j",))
		return rows, trajectories

	def test_named_groups_plus_residual_equal_the_trajectory_reference(self):
		events = [
			event(operation_label="llm:reason", gpu_energy_j=60.0),
			event(operation_label="kg:sparql", operation_type="kg",
			      start_timestamp=101.0, end_timestamp=102.0, gpu_energy_j=15.0),
		]
		rows, _ = self.build(events, trajectory_gpu=100.0)
		named = sum(r["energy_j"] for r in rows if r["row_kind"] == "measured")
		residual = sum(r["energy_j"] for r in rows if r["row_kind"] == "residual")
		self.assertAlmostEqual(named, 75.0)
		self.assertAlmostEqual(residual, 25.0)
		self.assertAlmostEqual(named + residual, 100.0)

	def test_reconciliation_report_difference_is_zero(self):
		events = [
			event(operation_label="llm:reason", gpu_energy_j=60.0),
			event(operation_label="kg:sparql", start_timestamp=101.0,
			      end_timestamp=102.0, gpu_energy_j=15.0),
		]
		rows, _ = self.build(events, trajectory_gpu=100.0)
		report = aggregate.reconciliation(rows, ("operation_label",))
		self.assertEqual(len(report), 1)
		self.assertAlmostEqual(report[0]["difference_j"], 0.0, places=9)
		self.assertAlmostEqual(report[0]["trajectory_energy_j"], 100.0)

	def test_shares_of_named_groups_and_residual_sum_to_one(self):
		events = [
			event(operation_label="llm:reason", gpu_energy_j=60.0),
			event(operation_label="kg:sparql", start_timestamp=101.0,
			      end_timestamp=102.0, gpu_energy_j=15.0),
		]
		rows, _ = self.build(events, trajectory_gpu=100.0)
		self.assertAlmostEqual(sum(r["share_of_trajectory"] for r in rows), 1.0)

	def test_reconciliation_holds_per_question_when_grouped_by_question(self):
		events = [
			event(question_id="q1", gpu_energy_j=40.0),
			event(question_id="q2", start_timestamp=200.0, end_timestamp=201.0,
			      gpu_energy_j=30.0),
		]
		trajectories = trajectory.accumulate(
			events, window_energy=lambda t0, t1: {
				"gpu_energy_j": 50.0 if t0 < 150 else 35.0,
				"cpu_package_energy_j": None, "dram_energy_j": None,
				"measured_energy_j": None,
			})
		rows = aggregate.aggregate(events, trajectories,
		                           group_by=("question_id", "operation_label"),
		                           domains=("gpu_energy_j",))
		report = aggregate.reconciliation(
			rows, ("question_id", "operation_label"))
		self.assertEqual(len(report), 2)
		for entry in report:
			self.assertAlmostEqual(entry["difference_j"], 0.0, places=9)

	def test_a_residual_row_exists_per_trajectory_scope(self):
		events = [
			event(question_id="q1", gpu_energy_j=40.0),
			event(question_id="q2", start_timestamp=200.0, end_timestamp=201.0,
			      gpu_energy_j=30.0),
		]
		trajectories = trajectory.accumulate(
			events, window_energy=lambda t0, t1: {
				"gpu_energy_j": 50.0, "cpu_package_energy_j": None,
				"dram_energy_j": None, "measured_energy_j": None})
		rows = aggregate.aggregate(events, trajectories,
		                           group_by=("question_id",),
		                           domains=("gpu_energy_j",))
		residuals = [r for r in rows if r["row_kind"] == "residual"]
		self.assertEqual({r["question_id"] for r in residuals}, {"q1", "q2"})


# --------------------------------------------------------------------------
# 2. negative residuals survive
# --------------------------------------------------------------------------

class TestNegativeResidual(unittest.TestCase):

	def test_a_negative_residual_is_reported_not_clamped(self):
		"""Events summing above the window is a real counter-boundary effect."""
		events = [event(gpu_energy_j=120.0)]
		trajectories = trajectory.accumulate(
			events, window_energy=window({
				"gpu_energy_j": 100.0, "cpu_package_energy_j": None,
				"dram_energy_j": None, "measured_energy_j": None}))
		rows = aggregate.aggregate(events, trajectories,
		                           domains=("gpu_energy_j",))
		residual, = [r for r in rows if r["row_kind"] == "residual"]
		self.assertAlmostEqual(residual["energy_j"], -20.0)
		self.assertLess(residual["share_of_trajectory"], 0.0)

	def test_reconciliation_still_exact_with_a_negative_residual(self):
		events = [event(gpu_energy_j=120.0)]
		trajectories = trajectory.accumulate(
			events, window_energy=window({
				"gpu_energy_j": 100.0, "cpu_package_energy_j": None,
				"dram_energy_j": None, "measured_energy_j": None}))
		rows = aggregate.aggregate(events, trajectories,
		                           domains=("gpu_energy_j",))
		report = aggregate.reconciliation(rows, ("operation_label",))
		self.assertAlmostEqual(report[0]["difference_j"], 0.0, places=9)


# --------------------------------------------------------------------------
# 3. null preservation
# --------------------------------------------------------------------------

class TestNullPreservation(unittest.TestCase):

	def rows_for(self, events, **kwargs):
		return aggregate.aggregate(events, [], **kwargs)

	def test_an_unmeasured_domain_aggregates_to_null_not_zero(self):
		rows = self.rows_for([event()], domains=("dram_energy_j",))
		row, = rows
		self.assertIsNone(row["energy_j"],
		                  "a domain the hardware never reported must not be 0.0")
		self.assertEqual(row["n_events_missing_energy"], 1)
		self.assertFalse(row["energy_complete"])

	def test_a_partial_sum_is_flagged_incomplete_and_not_reconciled(self):
		events = [event(gpu_energy_j=10.0), event(gpu_energy_j=None)]
		trajectories = trajectory.accumulate(
			events, window_energy=window({
				"gpu_energy_j": 30.0, "cpu_package_energy_j": None,
				"dram_energy_j": None, "measured_energy_j": None}))
		rows = aggregate.aggregate(events, trajectories,
		                           domains=("gpu_energy_j",))
		measured = [r for r in rows if r["row_kind"] == "measured"]
		row, = measured
		self.assertEqual(row["energy_j"], 10.0)
		self.assertEqual(row["n_events_missing_energy"], 1)
		self.assertFalse(row["energy_complete"])
		self.assertFalse(row["reconciles"])
		self.assertIsNone(row["share_of_trajectory"],
		                  "a partial sum must not be presented as a share")

	def test_csv_writes_an_empty_cell_for_null_never_zero(self):
		rows = self.rows_for([event()], domains=("dram_energy_j",))
		with tempfile.TemporaryDirectory() as tmp:
			path = os.path.join(tmp, "agg.csv")
			aggregate.write_csv(rows, path)
			with open(path) as f:
				text = f.read()
		body = text.splitlines()[1]
		self.assertNotIn(",0.0,", body,
		                 "a null domain must not be serialised as 0.0")
		self.assertIn(",,", body)


# --------------------------------------------------------------------------
# 4. shares only against a valid reference
# --------------------------------------------------------------------------

class TestShareGuards(unittest.TestCase):

	def test_no_trajectory_summary_means_no_share_and_no_residual(self):
		rows = aggregate.aggregate([event()], [], domains=("gpu_energy_j",))
		row, = rows
		self.assertIsNone(row["share_of_trajectory"])
		self.assertIsNone(row["trajectory_energy_j"])
		self.assertFalse(row["trajectory_reference_valid"])
		self.assertFalse(any(r["row_kind"] == "residual" for r in rows))

	def test_a_dependent_coverage_reference_is_refused(self):
		"""Without an independent window, coverage is circular: no share."""
		events = [event(gpu_energy_j=10.0)]
		trajectories = trajectory.accumulate(events)  # no window_energy
		self.assertFalse(trajectories[0]["coverage_is_independent"])
		rows = aggregate.aggregate(events, trajectories,
		                           domains=("gpu_energy_j",))
		row, = [r for r in rows if r["row_kind"] == "measured"]
		self.assertIsNone(row["share_of_trajectory"])
		self.assertFalse(row["trajectory_reference_valid"])

	def test_overlapping_events_invalidate_the_reference(self):
		"""Overlap makes plain summation double-count; shares must not render."""
		events = [
			event(gpu_energy_j=10.0, start_timestamp=100.0, end_timestamp=105.0),
			event(gpu_energy_j=10.0, start_timestamp=102.0, end_timestamp=107.0),
		]
		trajectories = trajectory.accumulate(
			events, window_energy=window({
				"gpu_energy_j": 30.0, "cpu_package_energy_j": None,
				"dram_energy_j": None, "measured_energy_j": None}))
		self.assertTrue(trajectories[0]["events_overlap"])
		self.assertFalse(trajectories[0]["coverage_valid"])
		rows = aggregate.aggregate(events, trajectories,
		                           domains=("gpu_energy_j",))
		for row in rows:
			self.assertIsNone(row["share_of_trajectory"])
		self.assertFalse(any(r["row_kind"] == "residual" for r in rows))

	def test_diagnostic_domains_are_never_given_a_share_or_residual(self):
		"""cpu_core is inside cpu_package; a share would double-count."""
		events = [event(cpu_core_energy_j=5.0)]
		trajectories = trajectory.accumulate(
			events, window_energy=window({
				"gpu_energy_j": 30.0, "cpu_package_energy_j": None,
				"dram_energy_j": None, "measured_energy_j": None}))
		rows = aggregate.aggregate(events, trajectories,
		                           domains=("cpu_core_energy_j",))
		row, = rows
		self.assertEqual(row["domain_role"], "diagnostic")
		self.assertEqual(row["energy_j"], 5.0)
		self.assertIsNone(row["share_of_trajectory"])
		self.assertFalse(row["reconciles"])


# --------------------------------------------------------------------------
# 5. arbitrary groupings
# --------------------------------------------------------------------------

class TestArbitraryGrouping(unittest.TestCase):

	def events(self):
		return [
			event(operation_label="llm:reason", traversal_depth=0,
			      iteration=0, gpu_energy_j=10.0),
			event(operation_label="llm:reason", traversal_depth=1,
			      iteration=1, gpu_energy_j=20.0),
			event(operation_label="kg:sparql", operation_type="kg",
			      traversal_depth=1, iteration=1, gpu_energy_j=5.0,
			      meta={"query_kind": "head"}),
		]

	def test_every_standard_dimension_is_groupable(self):
		for dimension in aggregate.STANDARD_DIMENSIONS:
			rows = aggregate.aggregate(self.events(), [], group_by=(dimension,),
			                           domains=("gpu_energy_j",))
			self.assertTrue(rows, f"grouping by {dimension} produced nothing")
			for row in rows:
				self.assertIn(dimension, row)

	def test_multi_dimensional_grouping_splits_correctly(self):
		rows = aggregate.aggregate(
			self.events(), [], group_by=("operation_label", "traversal_depth"),
			domains=("gpu_energy_j",))
		found = {(r["operation_label"], r["traversal_depth"]): r["energy_j"]
		         for r in rows}
		self.assertEqual(found[("llm:reason", 0)], 10.0)
		self.assertEqual(found[("llm:reason", 1)], 20.0)
		self.assertEqual(found[("kg:sparql", 1)], 5.0)

	def test_grouping_by_a_metadata_key_works(self):
		rows = aggregate.aggregate(self.events(), [],
		                           group_by=("meta.query_kind",),
		                           domains=("gpu_energy_j",))
		found = {r["meta.query_kind"]: r["energy_j"] for r in rows}
		self.assertEqual(found["head"], 5.0)
		self.assertEqual(found[None], 30.0)

	def test_grouping_is_not_hardcoded_to_any_host_vocabulary(self):
		"""A non-KGQA host's labels aggregate identically."""
		events = [event(operation_label="robotics:grasp", operation_type="robotics",
		                gpu_energy_j=7.0)]
		rows = aggregate.aggregate(events, [], domains=("gpu_energy_j",))
		row, = rows
		self.assertEqual(row["operation_label"], "robotics:grasp")
		self.assertEqual(row["energy_j"], 7.0)

	def test_available_dimensions_reports_top_level_and_meta_keys(self):
		dimensions = aggregate.available_dimensions(self.events())
		self.assertIn("operation_label", dimensions)
		self.assertIn("traversal_depth", dimensions)
		self.assertIn("meta.query_kind", dimensions)

	def test_grouping_with_no_trajectory_dimension_scopes_residual_globally(self):
		events = [
			event(question_id="q1", gpu_energy_j=10.0),
			event(question_id="q2", start_timestamp=200.0, end_timestamp=201.0,
			      gpu_energy_j=20.0),
		]
		trajectories = trajectory.accumulate(
			events, window_energy=window({
				"gpu_energy_j": 25.0, "cpu_package_energy_j": None,
				"dram_energy_j": None, "measured_energy_j": None}))
		rows = aggregate.aggregate(events, trajectories,
		                           group_by=("operation_label",),
		                           domains=("gpu_energy_j",))
		residuals = [r for r in rows if r["row_kind"] == "residual"]
		self.assertEqual(len(residuals), 1)
		self.assertAlmostEqual(residuals[0]["energy_j"], 50.0 - 30.0)


# --------------------------------------------------------------------------
# 6. quantization is surfaced, not corrected
# --------------------------------------------------------------------------

class TestQuantizationVisibility(unittest.TestCase):

	def test_zero_delta_events_are_counted_so_they_cannot_read_as_true_zero(self):
		events = [event(operation_label="kg:sparql", gpu_energy_j=0.0)
		          for _ in range(5)]
		events.append(event(operation_label="kg:sparql", gpu_energy_j=2.0))
		rows = aggregate.aggregate(events, [], domains=("gpu_energy_j",))
		row, = rows
		self.assertEqual(row["n_zero_energy_events"], 5)
		self.assertAlmostEqual(row["zero_energy_fraction"], 5 / 6)
		self.assertEqual(row["energy_j"], 2.0,
		                 "the sum must stay as measured; no quantization fix")

	def test_an_all_zero_group_keeps_its_zero_count_visible(self):
		events = [event(operation_label="kg:sparql", gpu_energy_j=0.0)
		          for _ in range(4)]
		rows = aggregate.aggregate(events, [], domains=("gpu_energy_j",))
		row, = rows
		self.assertEqual(row["energy_j"], 0.0)
		self.assertEqual(row["n_zero_energy_events"], 4)
		self.assertEqual(row["zero_energy_fraction"], 1.0)


# --------------------------------------------------------------------------
# 7. completeness is carried through
# --------------------------------------------------------------------------

class TestCompletenessCarried(unittest.TestCase):

	def test_incomplete_boundary_is_visible_on_every_row(self):
		rows = aggregate.aggregate([event(measurement_complete=False)], [],
		                           domains=("gpu_energy_j",))
		row, = rows
		self.assertFalse(row["all_events_measurement_complete"])
		self.assertEqual(row["n_events_measurement_complete"], 0)

	def test_complete_boundary_is_reported_when_every_event_has_it(self):
		events = [event(measurement_complete=True, cpu_package_energy_j=1.0,
		                dram_energy_j=1.0, measured_energy_j=12.0)]
		rows = aggregate.aggregate(events, [], domains=("measured_energy_j",))
		row, = rows
		self.assertTrue(row["all_events_measurement_complete"])

	def test_failed_events_are_counted_per_group(self):
		events = [event(status="error"), event(status="ok")]
		rows = aggregate.aggregate(events, [], domains=("gpu_energy_j",))
		row, = rows
		self.assertEqual(row["n_failed_events"], 1)


# --------------------------------------------------------------------------
# 8. no idle-baseline subtraction, no plotting, no host imports
# --------------------------------------------------------------------------

class TestScopeDiscipline(unittest.TestCase):

	MODULE = os.path.join(ROOT, "agent_energy_profiler", "aggregate.py")

	def test_gross_energy_is_reported_with_no_idle_subtraction(self):
		"""Aggregation must not silently change what the number means."""
		events = [event(gpu_energy_j=100.0, duration_s=10.0)]
		rows = aggregate.aggregate(events, [], domains=("gpu_energy_j",))
		self.assertEqual(rows[0]["energy_j"], 100.0)

		with open(self.MODULE, encoding="utf-8") as f:
			text = f.read()
		for needle in ("idle_w", "idle_power", "baseline_w", "- idle"):
			self.assertNotIn(needle, text)

	def test_the_module_imports_no_host_or_plotting_code(self):
		with open(self.MODULE, encoding="utf-8") as f:
			tree = ast.parse(f.read(), filename=self.MODULE)
		roots = set()
		for node in ast.walk(tree):
			if isinstance(node, ast.Import):
				for alias in node.names:
					roots.add(alias.name.split(".")[0])
			elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
				roots.add(node.module.split(".")[0])
		forbidden = {"chain_of_relations", "measurement", "matplotlib",
		             "seaborn", "pandas", "numpy", "plotly", "codecarbon"}
		self.assertEqual(roots & forbidden, set(),
		                 f"aggregate.py must stay generic and plot-free: {roots}")

	def test_no_plotting_symbols_leak_into_the_package(self):
		with open(self.MODULE, encoding="utf-8") as f:
			text = f.read()
		for needle in ("plt.", "savefig", "pyplot"):
			self.assertNotIn(needle, text)


# --------------------------------------------------------------------------
# 9. against the real validation artifacts
# --------------------------------------------------------------------------

class TestAgainstRecordedRuns(unittest.TestCase):
	"""Runs only where the recorded Gemma validation artifacts are present."""

	RUN_DIRS = ("gemma4b_v2_webqsp_3", "gemma4b_v2_cwq_3")

	def load(self, tag):
		run_dir = os.path.join(RUNS, tag)
		events_path = os.path.join(run_dir, "events_attributed.jsonl")
		traj_path = os.path.join(run_dir, "trajectory_summary.json")
		if not (os.path.exists(events_path) and os.path.exists(traj_path)):
			self.skipTest(f"recorded artifacts for {tag} not present")
		return (aggregate.load_events(events_path),
		        aggregate.load_trajectories(traj_path))

	def test_recorded_runs_reconcile_exactly_by_label(self):
		for tag in self.RUN_DIRS:
			with self.subTest(run=tag):
				events, trajectories = self.load(tag)
				rows = aggregate.aggregate(events, trajectories,
				                           group_by=("operation_label",),
				                           domains=("gpu_energy_j",))
				report = aggregate.reconciliation(rows, ("operation_label",))
				self.assertTrue(report)
				for entry in report:
					self.assertAlmostEqual(entry["difference_j"], 0.0, places=6)

	def test_recorded_runs_reconcile_per_question(self):
		for tag in self.RUN_DIRS:
			with self.subTest(run=tag):
				events, trajectories = self.load(tag)
				group_by = ("question_id", "operation_label")
				rows = aggregate.aggregate(events, trajectories,
				                           group_by=group_by,
				                           domains=("gpu_energy_j",))
				report = aggregate.reconciliation(rows, group_by)
				self.assertEqual(len(report), len(trajectories))
				for entry in report:
					self.assertAlmostEqual(entry["difference_j"], 0.0, places=6)

	def test_label_by_traversal_depth_is_available_on_recorded_runs(self):
		events, trajectories = self.load("gemma4b_v2_webqsp_3")
		rows = aggregate.aggregate(
			events, trajectories,
			group_by=("operation_label", "traversal_depth"),
			domains=("gpu_energy_j",))
		measured = [r for r in rows if r["row_kind"] == "measured"]
		self.assertGreater(len(measured), 8)
		self.assertIn(None, {r["traversal_depth"] for r in measured},
		              "llm:direct_answer sits outside the traversal")

	def test_unmeasured_cpu_and_dram_stay_null_on_recorded_runs(self):
		events, trajectories = self.load("gemma4b_v2_webqsp_3")
		rows = aggregate.aggregate(
			events, trajectories, group_by=("operation_type",),
			domains=("cpu_package_energy_j", "dram_energy_j",
			         "measured_energy_j"))
		for row in rows:
			self.assertIsNone(row["energy_j"],
			                  f"{row['energy_domain']} was not measured on this "
			                  "host and must aggregate to null")
			self.assertIsNone(row["share_of_trajectory"])

	def test_kg_zero_delta_events_are_flagged_on_recorded_runs(self):
		events, trajectories = self.load("gemma4b_v2_webqsp_3")
		rows = aggregate.aggregate(events, trajectories,
		                           group_by=("operation_label",),
		                           domains=("gpu_energy_j",))
		kg = [r for r in rows if str(r["operation_label"]).startswith("kg:")]
		self.assertTrue(kg)
		self.assertTrue(any(r["zero_energy_fraction"] > 0.9 for r in kg),
		                "the sub-resolution KG events must be visible, not "
		                "silently aggregated into a confident total")


class TestMixedParadigmAggregation(unittest.TestCase):
	"""One aggregation over CoR, ToG and PoG events at once.

	The acceptance criterion for the ToG/PoG instrumentation slice: the
	generic aggregator groups by (paradigm, operation_label) with no
	paradigm-specific branch anywhere in it. Semantics come from the labels
	the host agents attached; the aggregator only sums.
	"""

	MIXED = [
		("cor", "llm:relation_rank", 12.0),
		("cor", "llm:reason", 8.0),
		("cor", "kg:relation_search", 0.5),
		("tog", "llm:relation_rank", 9.0),
		("tog", "llm:entity_prune", 15.0),
		("tog", "llm:reason", 7.0),
		("tog", "kg:entity_search", 0.4),
		("pog", "llm:subquestion_decompose", 4.0),
		("pog", "llm:entity_prune", 11.0),
		("pog", "llm:memory_update", 6.0),
		("pog", "llm:reverse_retrieval_decision", 3.0),
		("pog", "embedding:prune", 1.5),
	]

	def events(self):
		rows = []
		for index, (paradigm, label, energy) in enumerate(self.MIXED):
			rows.append(event(
				paradigm=paradigm,
				question_id=f"q-{paradigm}",
				operation_label=label,
				operation_type=label.split(":", 1)[0],
				gpu_energy_j=energy,
				step_index=index,
				start_timestamp=100.0 + index,
				end_timestamp=101.0 + index,
			))
		return rows

	def rows(self):
		return [r for r in aggregate.aggregate(
			self.events(), group_by=("paradigm", "operation_label"),
			domains=("gpu_energy_j",))
			if r["row_kind"] == "measured"]

	def test_every_paradigm_stage_gets_its_own_row(self):
		got = {(r["paradigm"], r["operation_label"]): r["energy_j"]
		       for r in self.rows()}
		self.assertEqual(got, {(p, l): e for p, l, e in self.MIXED})

	def test_a_shared_label_stays_separable_by_paradigm(self):
		rows = {(r["paradigm"], r["operation_label"]): r["energy_j"]
		        for r in self.rows()}
		self.assertEqual(rows[("cor", "llm:relation_rank")], 12.0)
		self.assertEqual(rows[("tog", "llm:relation_rank")], 9.0)
		self.assertEqual(rows[("tog", "llm:entity_prune")], 15.0)
		self.assertEqual(rows[("pog", "llm:entity_prune")], 11.0)

	def test_llm_versus_kg_split_is_available_per_paradigm(self):
		rows = [r for r in aggregate.aggregate(
			self.events(), group_by=("paradigm", "operation_type"),
			domains=("gpu_energy_j",)) if r["row_kind"] == "measured"]
		got = {(r["paradigm"], r["operation_type"]): r["energy_j"] for r in rows}
		self.assertEqual(got[("tog", "llm")], 31.0)
		self.assertEqual(got[("tog", "kg")], 0.4)
		self.assertEqual(got[("pog", "embedding")], 1.5)

	def test_the_aggregator_needs_no_paradigm_specific_branch(self):
		"""No paradigm name may appear in the aggregation layer's source."""
		import inspect
		source = inspect.getsource(aggregate)
		for paradigm in ("cor", "tog", "pog", "chain_of_relations"):
			self.assertNotIn(f'"{paradigm}"', source)
			self.assertNotIn(f"'{paradigm}'", source)

	def test_no_share_is_claimed_without_a_trajectory_reference(self):
		for row in self.rows():
			self.assertIsNone(row["share_of_trajectory"])
			self.assertFalse(row["trajectory_reference_valid"])



if __name__ == "__main__":
	unittest.main()
