"""agent_energy_profiler.visualize: generic figures that cannot misstate energy.

Small synthetic runs only. The fixtures deliberately use operation labels,
operation types and paradigm names that no host in this repository emits, so
passing proves the visualizer reads semantics from the data.
"""

import ast
import contextlib
import csv
import hashlib
import io
import json
import os
import re
import subprocess
import sys
import tempfile
import unittest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from agent_energy_profiler import trajectory, visualize  # noqa: E402

try:
	import matplotlib  # noqa: F401
	HAVE_MPL = True
except ImportError:
	HAVE_MPL = False

RESIDUAL_J = 2.0   # trajectory window energy beyond the summed events


def event(q, step, label, gpu, paradigm="alpha", start=None, **overrides):
	start = (100.0 * int(re.sub(r"\D", "", q) or 0) + step) if start is None else start
	base = {
		"schema_version": 1, "event_id": f"{paradigm}-{q}-{step}", "run_id": f"run-{paradigm}",
		"question_id": q, "dataset": "synthetic", "paradigm": paradigm,
		"iteration": step, "traversal_depth": min(step, 2), "step_index": step,
		"operation_type": label.split(":")[0], "operation_label": label,
		"start_timestamp": start, "end_timestamp": start + 0.5, "duration_s": 0.5,
		"status": "ok", "meta": {}, "input_tokens": None, "output_tokens": None,
		"gpu_energy_j": gpu, "cpu_package_energy_j": None, "dram_energy_j": None,
		"measured_energy_j": None, "measurement_complete": False,
	}
	base.update(overrides)
	return base


def fallback(q, step, label, gpu, **kw):
	return event(q, step, label, gpu, meta={"fallback": True, "fallback_reason": "no_path"}, **kw)


def alpha_events():
	"""Two answered questions and two fallback questions."""
	return [
		event("q1", 0, "llm:plan", 10.0, input_tokens=100, output_tokens=10),
		event("q1", 1, "kg:lookup", 0.0),
		event("q1", 2, "llm:answer", 20.0, input_tokens=50, output_tokens=5),
		event("q2", 0, "llm:plan", 12.0, input_tokens=120, output_tokens=12),
		event("q2", 1, "kg:lookup", 1.0),
		event("q2", 2, "llm:answer", 30.0, input_tokens=60, output_tokens=9),
		event("q3", 0, "llm:plan", 8.0, input_tokens=80, output_tokens=8),
		event("q3", 1, "kg:lookup", 0.0),
		fallback("q3", 2, "llm:closed_reply", 5.0, input_tokens=20, output_tokens=4),
		event("q4", 0, "llm:plan", 15.0, input_tokens=150, output_tokens=20),
		event("q4", 1, "kg:lookup", 2.0),
		fallback("q4", 2, "llm:closed_reply", 6.0, input_tokens=20, output_tokens=6),
	]


def beta_events():
	"""No fallback metadata; operation types no other run has."""
	return [
		event("q1", 0, "llm:plan", 9.0, paradigm="beta", input_tokens=90, output_tokens=9),
		event("q1", 1, "robot:grasp", 4.0, paradigm="beta"),
		event("q2", 0, "llm:plan", 11.0, paradigm="beta", input_tokens=110, output_tokens=11),
		event("q2", 1, "robot:grasp", 3.0, paradigm="beta"),
		event("q2", 2, "reranker:score", 1.0, paradigm="beta"),
		event("q3", 0, "llm:plan", 7.0, paradigm="beta", input_tokens=70, output_tokens=7),
	]


def write_run(directory, events, with_trajectory=True, independent=True):
	os.makedirs(directory, exist_ok=True)
	with open(os.path.join(directory, visualize.EVENTS_FILE), "w") as f:
		for e in events:
			f.write(json.dumps(e) + "\n")
	if with_trajectory:
		def window(t0, t1):
			inside = [e for e in events if t0 <= e["start_timestamp"] and e["end_timestamp"] <= t1]
			return {"gpu_energy_j": sum(e["gpu_energy_j"] or 0.0 for e in inside) + RESIDUAL_J,
			        "cpu_package_energy_j": None, "dram_energy_j": None,
			        "measured_energy_j": None}
		rows = trajectory.accumulate(events, window_energy=window if independent else None)
		with open(os.path.join(directory, visualize.TRAJECTORY_FILE), "w") as f:
			json.dump({"summary": {}, "trajectories": rows}, f)
	return directory


def read_csv(path):
	with open(path, newline="") as f:
		return list(csv.DictReader(f))


def snapshot(directory):
	digests = {}
	for name in sorted(os.listdir(directory)):
		with open(os.path.join(directory, name), "rb") as f:
			digests[name] = hashlib.sha256(f.read()).hexdigest()
	return digests


@unittest.skipUnless(HAVE_MPL, "matplotlib not installed")
class VisualizeCase(unittest.TestCase):

	def setUp(self):
		self._tmp = tempfile.TemporaryDirectory()
		self.tmp = self._tmp.name
		self.alpha = write_run(os.path.join(self.tmp, "run_a"), alpha_events())
		self.beta = write_run(os.path.join(self.tmp, "run_b"), beta_events(),
		                      with_trajectory=False)
		self.out = os.path.join(self.tmp, "figures")

	def tearDown(self):
		self._tmp.cleanup()

	def cli(self, *argv):
		out, err = io.StringIO(), io.StringIO()
		with contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
			code = visualize.main(list(argv))
		return code, out.getvalue(), err.getvalue()

	def plot(self, plot, *runs, extra=()):
		argv = ["--plot", plot, "--out", self.out]
		for run in runs:
			argv += ["--run", run]
		return self.cli(*argv, *extra)

	def outputs(self):
		return sorted(os.listdir(self.out)) if os.path.isdir(self.out) else []

	def csv_for(self, suffix=".csv"):
		matches = [n for n in self.outputs() if n.endswith(suffix)
		           and not n.endswith("_stats.csv") and not n.endswith("_per_question.csv")]
		self.assertEqual(len(matches), 1, self.outputs())
		return read_csv(os.path.join(self.out, matches[0]))

	def load(self, *specs):
		return visualize.load_runs(list(specs))


# --------------------------------------------------------------------------
# operation energy
# --------------------------------------------------------------------------

class TestOperationEnergy(VisualizeCase):

	def test_single_run_writes_png_pdf_csv_with_expected_energy(self):
		code, out, err = self.plot("operation-energy", self.alpha)
		self.assertEqual(code, 0, err)
		stem = "operation-energy_j_gpu_energy_j_alpha"
		for ext in (".png", ".pdf", ".csv"):
			self.assertIn(stem + ext, self.outputs())
		rows = {r["operation_label"]: r for r in self.csv_for() if r["row_kind"] == "operation"}
		self.assertEqual(float(rows["llm:plan"]["energy_j"]), 45.0)
		self.assertEqual(float(rows["llm:answer"]["energy_j"]), 50.0)
		self.assertEqual(float(rows["llm:closed_reply"]["energy_j"]), 11.0)
		self.assertEqual(float(rows["kg:lookup"]["energy_j"]), 3.0)
		self.assertAlmostEqual(sum(float(r["pct_attributed_energy"]) for r in rows.values()), 100.0)
		self.assertTrue(all(r["energy_basis"] == "attributed_events" for r in rows.values()))

	def test_csv_matches_drawn_bars(self):
		plt = visualize._pyplot()
		runs = self.load(self.alpha, self.beta)
		rows, _ = visualize.prepare_operation_energy(runs, "gpu_energy_j")
		fig = visualize.draw_operation_energy(plt, runs, rows, "gpu_energy_j", "j")
		for ax, run in zip(fig.axes, runs):
			drawn = {t.get_text(): p.get_height()
			         for t, p in zip(ax.get_xticklabels(), ax.patches)}
			expected = {r["operation_label"]: r["energy_j"] for r in rows
			            if r["run"] == run.label and r["plotted"]}
			self.assertEqual(drawn, expected)
		plt.close(fig)

	def test_label_present_in_one_run_only_gets_no_fabricated_zero(self):
		code, _, err = self.plot("operation-energy", self.alpha, self.beta)
		self.assertEqual(code, 0, err)
		rows = self.csv_for()
		alpha_labels = {r["operation_label"] for r in rows if r["run"] == "alpha"}
		beta_labels = {r["operation_label"] for r in rows if r["run"] == "beta"}
		self.assertNotIn("robot:grasp", alpha_labels)
		self.assertNotIn("llm:closed_reply", beta_labels)
		self.assertIn("robot:grasp", beta_labels)

	def test_unattributed_is_not_an_operation(self):
		code, out, err = self.plot("operation-energy", self.alpha)
		self.assertEqual(code, 0, err)
		residual = [r for r in self.csv_for() if r["row_kind"] == "unattributed"]
		self.assertEqual(len(residual), 1)
		self.assertEqual(residual[0]["plotted"], "False")
		self.assertEqual(residual[0]["pct_attributed_energy"], "")
		self.assertAlmostEqual(float(residual[0]["energy_j"]), 4 * RESIDUAL_J)
		self.assertIn("unattributed", out)

		plt = visualize._pyplot()
		runs = self.load(self.alpha)
		rows, _ = visualize.prepare_operation_energy(runs, "gpu_energy_j")
		fig = visualize.draw_operation_energy(plt, runs, rows, "gpu_energy_j", "j")
		self.assertNotIn("<unattributed>", [t.get_text() for t in fig.axes[0].get_xticklabels()])
		plt.close(fig)
		rows, _ = visualize.prepare_operation_energy(runs, "gpu_energy_j", include_unattributed=True)
		fig = visualize.draw_operation_energy(plt, runs, rows, "gpu_energy_j", "j")
		self.assertIn("<unattributed>", [t.get_text() for t in fig.axes[0].get_xticklabels()])
		plt.close(fig)

	def test_percent_with_unattributed_is_refused(self):
		code, _, err = self.plot("operation-energy", self.alpha,
		                         extra=("--unit", "percent", "--include-unattributed"))
		self.assertEqual(code, 2)
		self.assertIn("no share", err)

	def test_type_filter_keeps_shares_of_the_full_attributed_total(self):
		code, _, err = self.plot("operation-energy", self.alpha,
		                         extra=("--unit", "percent", "--operation-type", "kg"))
		self.assertEqual(code, 0, err)
		rows = [r for r in self.csv_for() if r["row_kind"] == "operation"]
		self.assertEqual([r["operation_label"] for r in rows], ["kg:lookup"])
		self.assertAlmostEqual(float(rows[0]["pct_attributed_energy"]), 3.0 / 109.0 * 100)

	def test_trajectory_basis_is_refused_for_operation_plots(self):
		code, _, err = self.plot("operation-energy", self.alpha, extra=("--basis", "trajectory"))
		self.assertEqual(code, 2)
		self.assertIn("attributed event energy only", err)


# --------------------------------------------------------------------------
# domains and titles
# --------------------------------------------------------------------------

class TestDomains(VisualizeCase):

	def test_unavailable_domain_fails_clearly_and_writes_nothing(self):
		code, _, err = self.plot("operation-energy", self.alpha,
		                         extra=("--domain", "cpu_package_energy_j"))
		self.assertEqual(code, 2)
		self.assertIn("not measured", err)
		self.assertEqual(self.outputs(), [])

	def test_selected_domain_drives_title_and_axis(self):
		events = [dict(e, cpu_package_energy_j=e["gpu_energy_j"] * 2) for e in alpha_events()]
		run = write_run(os.path.join(self.tmp, "run_cpu"), events)
		plt = visualize._pyplot()
		runs = self.load(run)
		rows, _ = visualize.prepare_operation_energy(runs, "cpu_package_energy_j")
		fig = visualize.draw_operation_energy(plt, runs, rows, "cpu_package_energy_j", "j")
		self.assertEqual(fig._suptitle.get_text(), "alpha CPU Package Energy by Operation")
		self.assertIn("CPU Package energy", fig.axes[-1].get_ylabel())
		self.assertIn("(J)", fig.axes[-1].get_ylabel())
		self.assertNotIn("GPU", fig._suptitle.get_text())
		plt.close(fig)

	def test_domain_names_are_centralised_and_extend_to_new_domains(self):
		self.assertEqual(visualize.plot_title("operation-energy", "gpu_energy_j"),
		                 "GPU Energy by Operation")
		self.assertEqual(visualize.plot_title("question-energy", "dram_energy_j"),
		                 "Question-Level DRAM Energy")
		self.assertEqual(visualize.domain_name("npu_board_energy_j"), "Npu Board")


# --------------------------------------------------------------------------
# question energy and bases
# --------------------------------------------------------------------------

class TestQuestionEnergy(VisualizeCase):

	def test_multi_run_ecdf_has_one_row_per_question_and_run(self):
		code, _, err = self.plot("question-energy", self.alpha, self.alpha.replace("run_a", "run_c"))
		# the second directory does not exist: must fail by artifact, not by name
		self.assertEqual(code, 2)
		other = write_run(os.path.join(self.tmp, "run_c"),
		                  [dict(e, paradigm="gamma") for e in alpha_events()])
		code, _, err = self.plot("question-energy", self.alpha, other)
		self.assertEqual(code, 0, err)
		rows = self.csv_for()
		self.assertEqual(len(rows), 8)
		for run in ("alpha", "gamma"):
			ecdf = [float(r["ecdf"]) for r in rows if r["run"] == run]
			self.assertEqual(ecdf[-1], 1.0)

	def test_ecdf_line_matches_csv(self):
		plt = visualize._pyplot()
		runs = self.load(self.alpha)
		rows, stats, _ = visualize.prepare_question_energy(runs, "gpu_energy_j", "trajectory")
		fig = visualize.draw_question_energy(plt, runs, rows, stats, "gpu_energy_j",
		                                     "trajectory", [], False)
		line = fig.axes[0].get_lines()[0]
		# energy on y, cumulative fraction of questions on x
		self.assertEqual(list(line.get_xdata()), [r["ecdf"] for r in rows])
		self.assertEqual(list(line.get_ydata()), [r["energy_j"] for r in rows])
		plt.close(fig)

	def test_trajectory_and_attributed_bases_give_their_own_values(self):
		runs = self.load(self.alpha)
		traj, _, _ = visualize.prepare_question_energy(runs, "gpu_energy_j", "trajectory")
		attr, _, _ = visualize.prepare_question_energy(runs, "gpu_energy_j", "attributed_events")
		traj = {r["question_id"]: r["energy_j"] for r in traj}
		attr = {r["question_id"]: r["energy_j"] for r in attr}
		self.assertEqual(attr["q1"], 30.0)
		self.assertEqual(traj["q1"], 30.0 + RESIDUAL_J)

	def test_trajectory_basis_refuses_a_non_independent_window(self):
		run = write_run(os.path.join(self.tmp, "run_dep"), alpha_events(), independent=False)
		code, _, err = self.plot("question-energy", run)
		self.assertEqual(code, 2)
		self.assertIn("refusing to substitute", err)

	def test_missing_trajectory_only_blocks_plots_that_need_it(self):
		code, _, err = self.plot("question-energy", self.beta)
		self.assertEqual(code, 3)
		self.assertIn("--basis attributed", err)
		code, _, err = self.plot("question-energy", self.beta, extra=("--basis", "attributed"))
		self.assertEqual(code, 0, err)
		code, out, err = self.plot("operation-energy", self.beta)
		self.assertEqual(code, 0, err)
		self.assertIn("unattributed energy unavailable", out)


# --------------------------------------------------------------------------
# energy vs workload
# --------------------------------------------------------------------------

class TestEnergyVs(VisualizeCase):

	def test_arbitrary_operation_types_generate_call_counts(self):
		variables = visualize.workload_variables(self.load(self.beta)[0])
		self.assertIn("robot_calls", variables)
		self.assertIn("reranker_calls", variables)
		code, _, err = self.plot("energy-vs", self.beta,
		                         extra=("--x", "robot_calls", "--basis", "attributed"))
		self.assertEqual(code, 0, err)
		rows = self.csv_for()
		self.assertEqual({r["question_id"]: r["x"] for r in rows}, {"q1": "1", "q2": "1", "q3": "0"})

	def test_pairs_and_stats_are_written(self):
		code, _, err = self.plot("energy-vs", self.alpha, extra=("--x", "output_tokens"))
		self.assertEqual(code, 0, err)
		rows = self.csv_for()
		self.assertEqual(len(rows), 4)
		self.assertEqual({r["question_id"]: float(r["x"]) for r in rows},
		                 {"q1": 15, "q2": 21, "q3": 12, "q4": 26})
		stats = read_csv(os.path.join(self.out, [n for n in self.outputs()
		                                         if n.endswith("_stats.csv")][0]))
		self.assertEqual(stats[0]["status"], "ok")
		self.assertTrue(-1.0 <= float(stats[0]["pearson_r"]) <= 1.0)

	def annotations_file(self, mapping, column="outcome", run=None):
		path = os.path.join(self.tmp, f"annotations_{column}_{run or 'all'}.csv")
		with open(path, "w", newline="") as f:
			writer = csv.writer(f, lineterminator="\n")
			writer.writerow(["question_id", "run", column])
			for question, value in mapping.items():
				writer.writerow([question, run or "", value])
		return path

	def test_points_are_coloured_by_an_opaque_annotation(self):
		path = self.annotations_file({"q1": "alpha-hit", "q2": "alpha-hit", "q3": "beta-miss"})
		code, out, err = self.plot("energy-vs", self.alpha,
		                           extra=("--x", "output_tokens", "--annotations", path,
		                                  "--color-by", "outcome"))
		self.assertEqual(code, 0, err)
		rows = self.csv_for()
		self.assertEqual({r["question_id"]: r["category"] for r in rows},
		                 {"q1": "alpha-hit", "q2": "alpha-hit", "q3": "beta-miss", "q4": ""})
		self.assertIn("no annotation", out)
		stats = read_csv(os.path.join(self.out, [n for n in self.outputs()
		                                         if n.endswith("_stats.csv")][0]))
		self.assertEqual({s["category"] for s in stats},
		                 {"(all)", "alpha-hit", "beta-miss", "unannotated"})

	def test_categories_drive_colour_marker_and_legend(self):
		plt = visualize._pyplot()
		runs = self.load(self.alpha)
		annotations = {"q1": "keep", "q2": "keep", "q3": "drop"}
		rows, stats, _ = visualize.prepare_energy_vs(runs, "gpu_energy_j", "trajectory",
		                                             "output_tokens", annotations, "outcome")
		style = visualize.category_styles(["keep", "drop"], {"keep": "#0ca30c", "drop": "#d03b3b"},
		                                  {"keep": "o", "drop": "^"},
		                                  {"keep": "Kept", "drop": "Dropped"})
		fig = visualize.draw_energy_vs(plt, runs, rows, stats, "gpu_energy_j", "trajectory",
		                               "output_tokens", False, category_order=["keep", "drop"],
		                               category_style=style)
		labels = [t.get_text() for t in fig.axes[0].get_legend().get_texts()]
		self.assertEqual(labels[0], "Kept (n = 2)")
		self.assertEqual(labels[1], "Dropped (n = 1)")
		self.assertTrue(labels[2].startswith("unannotated"))   # q4 is never a category
		plt.close(fig)

	def test_annotation_file_problems_are_refused(self):
		path = self.annotations_file({"q1": "hit"})
		code, _, err = self.plot("energy-vs", self.alpha,
		                         extra=("--x", "output_tokens", "--annotations", path))
		self.assertEqual(code, 2)
		self.assertIn("used together", err)
		code, _, err = self.plot("energy-vs", self.alpha,
		                         extra=("--x", "output_tokens", "--annotations", path,
		                                "--color-by", "missing_column"))
		self.assertEqual(code, 2)
		self.assertIn("no column", err)
		code, _, err = self.plot("energy-vs", self.alpha, self.beta,
		                         extra=("--x", "output_tokens", "--basis", "attributed",
		                                "--annotations", path, "--color-by", "outcome"))
		self.assertEqual(code, 2)
		self.assertIn("one run per figure", err)
		code, _, err = self.plot("question-energy", self.alpha,
		                         extra=("--annotations", path, "--color-by", "outcome"))
		self.assertEqual(code, 2)
		self.assertIn("energy-vs only", err)

	def test_constant_variable_reports_correlation_unavailable(self):
		n, r, rho, status = visualize.correlation([(1, 2), (1, 3), (1, 4)])
		self.assertIsNone(r)
		self.assertIsNone(rho)
		self.assertTrue(status.startswith("unavailable"))
		n, r, rho, status = visualize.correlation([(1, 2), (2, 3)])
		self.assertTrue(status.startswith("unavailable"))

	def test_spearman_handles_ties(self):
		_, r, rho, status = visualize.correlation([(1, 1), (2, 2), (2, 2), (3, 3)])
		self.assertEqual(status, "ok")
		self.assertAlmostEqual(r, 1.0)
		self.assertAlmostEqual(rho, 1.0)


# --------------------------------------------------------------------------
# fallback
# --------------------------------------------------------------------------

class TestFallback(VisualizeCase):

	def test_outcome_energy_groups_by_schema_marker(self):
		code, _, err = self.plot("outcome-energy", self.alpha)
		self.assertEqual(code, 0, err)
		rows = {r["question_id"]: r for r in self.csv_for()}
		self.assertEqual(rows["q3"]["outcome"], "fallback")
		self.assertEqual(rows["q3"]["fallback_reason"], "no_path")
		self.assertEqual(rows["q1"]["outcome"], "no_fallback")
		self.assertEqual(float(rows["q3"]["energy_j"]), 13.0 + RESIDUAL_J)

	def test_fallback_split_uses_the_marked_event(self):
		code, _, err = self.plot("fallback-split", self.alpha)
		self.assertEqual(code, 0, err)
		means = {r["component"]: r for r in self.csv_for()}
		self.assertAlmostEqual(float(means["before_fallback_operation"]["mean_energy_j"]), 12.5)
		self.assertAlmostEqual(float(means["fallback_operation"]["mean_energy_j"]), 5.5)
		# no event follows the fallback operation: the component is absent, not zero
		self.assertNotIn("after_fallback_operation", means)
		self.assertEqual(means["fallback_operation"]["fallback_operation_labels"], "llm:closed_reply")
		self.assertEqual(means["fallback_operation"]["energy_basis"], "attributed_events")

	def test_fallback_absent_is_not_applicable(self):
		for plot in ("outcome-energy", "fallback-split"):
			code, _, err = self.plot(plot, self.beta, extra=("--basis", "attributed")
			                         if plot == "outcome-energy" else ())
			self.assertEqual(code, 3, err)
			self.assertIn("does not apply", err)

	def test_multiple_fallback_events_fail_rather_than_guess(self):
		events = alpha_events() + [fallback("q4", 3, "llm:closed_reply", 1.0)]
		run = write_run(os.path.join(self.tmp, "run_two"), events)
		code, _, err = self.plot("fallback-split", run)
		self.assertEqual(code, 2)
		self.assertIn("q4", err)
		self.assertIn("will not guess", err)


# --------------------------------------------------------------------------
# safety, listing, determinism, genericity
# --------------------------------------------------------------------------

class TestSafetyAndGenericity(VisualizeCase):

	def test_source_run_files_are_never_modified(self):
		before = {d: snapshot(d) for d in (self.alpha, self.beta)}
		for plot, extra in (("operation-energy", ()), ("question-energy", ("--basis", "attributed")),
		                    ("energy-vs", ("--x", "total_tokens", "--basis", "attributed"))):
			code, _, err = self.plot(plot, self.alpha, self.beta, extra=extra)
			self.assertEqual(code, 0, err)
		self.plot("outcome-energy", self.alpha)
		self.plot("fallback-split", self.alpha)
		code, _, err = self.plot("semantic-flow", self.alpha)
		self.assertEqual(code, 0, err)
		self.assertEqual({d: snapshot(d) for d in (self.alpha, self.beta)}, before)

	def test_output_inside_a_run_directory_is_refused(self):
		code, _, err = self.cli("--plot", "operation-energy", "--run", self.alpha,
		                        "--out", os.path.join(self.alpha, "figs"))
		self.assertEqual(code, 2)
		self.assertIn("inside run directory", err)
		self.assertFalse(os.path.exists(os.path.join(self.alpha, "figs")))

	def test_list_reports_what_a_user_needs_before_plotting(self):
		code, out, err = self.cli("--list", "--run", self.alpha, "--run", self.beta)
		self.assertEqual(code, 0, err)
		for needle in ("run: alpha", "paradigm: alpha", "questions: 4", "energy domains: gpu_energy_j",
		               "operation types:", "operation labels:", "energy-vs variables:",
		               "fallback metadata: yes", "trajectory artifact: yes",
		               "run: beta", "trajectory artifact: no", "fallback metadata: no",
		               "robot_calls"):
			self.assertIn(needle, out)

	def test_outputs_are_deterministic(self):
		self.plot("question-energy", self.alpha)
		first = snapshot(self.out)
		self.plot("question-energy", self.alpha)
		self.assertEqual(snapshot(self.out), first)

	def test_visualizer_contains_no_paradigm_specific_logic(self):
		path = os.path.join(ROOT, "agent_energy_profiler", "visualize.py")
		with open(path, encoding="utf-8") as f:
			tree = ast.parse(f.read())
		offenders = []
		for node in ast.walk(tree):
			if isinstance(node, ast.Constant) and isinstance(node.value, str):
				text = node.value.lower()
				if re.search(r"\b(cor|tog|pog)\b", text):
					offenders.append(node.value[:60])
				if re.search(r"\b(llm|kg|embedding):", text) or "direct_answer" in text:
					offenders.append(node.value[:60])
		self.assertEqual(offenders, [], "paradigm names or host labels hard-coded")

	def test_module_help_runs_without_drawing(self):
		out = subprocess.run([sys.executable, "-m", "agent_energy_profiler.visualize", "--help"],
		                     cwd=ROOT, capture_output=True, text=True, timeout=120)
		self.assertEqual(out.returncode, 0, out.stderr)
		self.assertIn("100%", out.stdout)
		self.assertNotIn("RuntimeWarning", out.stderr)


# --------------------------------------------------------------------------
# energy-on-y convention and shared energy scales
# --------------------------------------------------------------------------

class TestEnergyOnYAxis(VisualizeCase):

	def figures(self, ylim=None):
		plt = visualize._pyplot()
		runs, d, basis = self.load(self.alpha), "gpu_energy_j", "trajectory"
		rows, _ = visualize.prepare_operation_energy(runs, d)
		yield visualize.draw_operation_energy(plt, runs, rows, d, "j", ylim)
		rows, _ = visualize.prepare_operation_energy(runs, d)
		yield visualize.draw_operation_energy(plt, runs, rows, d, "percent")
		rows, stats, _ = visualize.prepare_question_energy(runs, d, basis)
		yield visualize.draw_question_energy(plt, runs, rows, stats, d, basis, ["median"], False, ylim)
		rows, stats, _ = visualize.prepare_energy_vs(runs, d, basis, "output_tokens")
		yield visualize.draw_energy_vs(plt, runs, rows, stats, d, basis, "output_tokens", False, ylim)
		used, rows, _, _ = visualize.prepare_outcome_energy(runs, d, basis)
		yield visualize.draw_outcome_energy(plt, used, rows, d, basis, False, ylim)
		used, means, _, _ = visualize.prepare_fallback_split(runs, d)
		yield visualize.draw_fallback_split(plt, used, means, d, ylim)

	def test_every_axis_plot_puts_energy_on_y(self):
		plt = visualize._pyplot()
		for fig in self.figures():
			for ax in fig.axes:
				self.assertIn("energy", ax.get_ylabel().lower(), fig._suptitle or ax.get_title())
				self.assertNotIn("energy", ax.get_xlabel().lower())
			plt.close(fig)

	def test_y_limits_fix_the_energy_axis(self):
		plt = visualize._pyplot()
		for fig in self.figures(ylim=(0.5, 400.0)):
			ax = fig.axes[0]
			if "Share of" not in ax.get_ylabel():
				self.assertEqual(tuple(ax.get_ylim()), (0.5, 400.0))
			plt.close(fig)
		code, _, err = self.plot("question-energy", self.alpha, extra=("--y-limits", "0.5,400"))
		self.assertEqual(code, 0, err)

	def test_plots_can_share_one_composite_figure(self):
		plt = visualize._pyplot()
		runs, d, basis = self.load(self.alpha), "gpu_energy_j", "trajectory"
		used, rows, _, _ = visualize.prepare_outcome_energy(runs, d, basis)
		_, means, _, _ = visualize.prepare_fallback_split(runs, d)
		fig, (left, right) = plt.subplots(1, 2, figsize=(9.0, 4.6))
		self.assertIs(visualize.draw_outcome_energy(plt, used, rows, d, basis, False, ax=left), fig)
		self.assertIs(visualize.draw_fallback_split(plt, used, means, d, ax=right), fig)
		self.assertEqual(len(fig.axes), 2)
		for ax in fig.axes:
			self.assertIn("energy", ax.get_ylabel().lower())
		plt.close(fig)

	def test_invalid_y_limits_are_refused(self):
		for bad in ("400,1", "abc", "1,2,3"):
			code, _, err = self.plot("question-energy", self.alpha, extra=("--y-limits", bad))
			self.assertEqual(code, 2, bad)
		code, _, err = self.plot("semantic-flow", self.alpha, extra=("--y-limits", "0,10"))
		self.assertEqual(code, 2)
		self.assertIn("no energy axis", err)
		self.assertEqual(visualize.parse_limits(""), None)


# --------------------------------------------------------------------------
# semantic flow
# --------------------------------------------------------------------------

class TestSemanticFlow(VisualizeCase):

	def flow(self, events=None, run=None):
		if run is None:
			run = write_run(os.path.join(self.tmp, "run_flow"), events)
		runs = self.load(run)
		rows, notes = visualize.prepare_semantic_flow(runs[0], "gpu_energy_j")
		return runs, rows, notes

	@staticmethod
	def edges(rows, level):
		return [(r["source"], r["target"], r["energy_j"]) for r in rows
		        if r["row_kind"] == "flow" and r["source_level"] == level]

	def test_hierarchy_reconciles_and_orders_by_energy(self):
		_, rows, notes = self.flow(run=self.alpha)
		self.assertEqual(self.edges(rows, "root"), [("total", "llm", 106.0), ("total", "kg", 3.0)])
		self.assertEqual(self.edges(rows, "operation_type"),
		                 [("llm", "llm:answer", 50.0), ("llm", "llm:plan", 45.0),
		                  ("llm", "llm:closed_reply", 11.0), ("kg", "kg:lookup", 3.0)])
		type_pct = [r["pct_total_attributed"] for r in rows if r["source_level"] == "root"
		            and r["row_kind"] == "flow"]
		self.assertAlmostEqual(sum(type_pct), 100.0)
		leaf_pct = sum(r["pct_total_attributed"] for r in rows if r["source"] == "llm")
		self.assertAlmostEqual(leaf_pct, type_pct[0])
		self.assertTrue(any("sum of 3 labels" in n for n in notes))

	def test_future_operation_types_become_their_own_branches(self):
		events = alpha_events() + [event("q1", 3, "embedding:prune", 4.0)]
		_, rows, _ = self.flow(events)
		self.assertIn(("total", "embedding", 4.0), self.edges(rows, "root"))
		self.assertIn(("embedding", "embedding:prune", 4.0), self.edges(rows, "operation_type"))
		_, rows, _ = self.flow(run=self.beta)
		self.assertIn(("robot", "robot:grasp", 7.0), self.edges(rows, "operation_type"))
		self.assertIn(("reranker", "reranker:score", 1.0), self.edges(rows, "operation_type"))

	def test_parent_is_the_recorded_type_never_the_label_prefix(self):
		events = alpha_events() + [event("q1", 3, "planner:think", 2.0, operation_type="llm")]
		_, rows, _ = self.flow(events)
		self.assertNotIn("planner", [t for _, t, _ in self.edges(rows, "root")])
		self.assertEqual([s for s, t, _ in self.edges(rows, "operation_type")
		                  if t == "planner:think"], ["llm"])

	def test_label_recorded_under_two_types_keeps_both_pairs(self):
		events = alpha_events() + [event("q2", 3, "kg:lookup", 7.0, operation_type="llm")]
		runs, rows, notes = self.flow(events)
		leaves = self.edges(rows, "operation_type")
		self.assertIn(("llm", "kg:lookup", 7.0), leaves)
		self.assertIn(("kg", "kg:lookup", 3.0), leaves)
		self.assertTrue(any("recorded under 2 operation types" in n for n in notes))
		plt = visualize._pyplot()
		fig = visualize.draw_semantic_flow(plt, runs[0], rows, "gpu_energy_j")
		gids = {p.get_gid() for p in fig.axes[0].patches}
		self.assertIn("node:operation_label:llm|kg:lookup", gids)
		self.assertIn("node:operation_label:kg|kg:lookup", gids)
		plt.close(fig)

	def test_flow_widths_equal_the_csv_values(self):
		code, _, err = self.plot("semantic-flow", self.alpha)
		self.assertEqual(code, 0, err)
		self.assertIn("semantic-flow_gpu_energy_j_alpha.png", self.outputs())
		self.assertIn("semantic-flow_gpu_energy_j_alpha.pdf", self.outputs())
		csv_rows = [r for r in self.csv_for() if r["row_kind"] == "flow"]
		runs = self.load(self.alpha)
		rows, _ = visualize.prepare_semantic_flow(runs[0], "gpu_energy_j")
		plt = visualize._pyplot()
		fig = visualize.draw_semantic_flow(plt, runs[0], rows, "gpu_energy_j")
		patches = {p.get_gid(): p for p in fig.axes[0].patches}
		total = sum(float(r["energy_j"]) for r in csv_rows if r["source_level"] == "root")
		scale = patches["node:root:total"].get_height() / total
		for r in csv_rows:
			v = patches[f"flow:{r['source']}->{r['target']}"].get_path().vertices
			self.assertAlmostEqual(v[0][1] - v[7][1], float(r["energy_j"]) * scale, places=9)
			self.assertAlmostEqual(v[3][1] - v[4][1], float(r["energy_j"]) * scale, places=9)
		plt.close(fig)

	def test_unattributed_is_reported_but_never_drawn(self):
		runs, rows, notes = self.flow(run=self.alpha)
		residual = [r for r in rows if r["row_kind"] == "unattributed"]
		self.assertEqual(len(residual), 1)
		self.assertFalse(residual[0]["plotted"])
		self.assertIsNone(residual[0]["pct_total_attributed"])
		self.assertAlmostEqual(residual[0]["energy_j"], 4 * RESIDUAL_J)
		self.assertTrue(any("unattributed" in n for n in notes))
		self.assertFalse(any(r["target"] == "<unattributed>" for r in rows if r["row_kind"] == "flow"))
		plt = visualize._pyplot()
		fig = visualize.draw_semantic_flow(plt, runs[0], rows, "gpu_energy_j")
		self.assertFalse(any("<unattributed>" in str(p.get_gid()) for p in fig.axes[0].patches))
		self.assertFalse(any("<unattributed>" in t.get_text() for t in fig.axes[0].texts))
		plt.close(fig)
		code, _, err = self.plot("semantic-flow", self.alpha, extra=("--include-unattributed",))
		self.assertEqual(code, 2)
		self.assertIn("operation-energy only", err)

	def test_type_colours_never_change_when_other_types_appear(self):
		base = visualize.type_colours(["llm", "kg"])
		self.assertEqual(visualize.type_colours(["kg", "llm"]), base)
		for i in range(64):   # whatever else is present, including colliding types
			grown = visualize.type_colours(["llm", "kg", f"future{i}"])
			self.assertEqual(grown["llm"], base["llm"])
			self.assertEqual(grown["kg"], base["kg"])

	def test_colour_collisions_are_reported_not_resolved(self):
		names = [f"t{i}" for i in range(200)]
		twin = next(n for n in names[1:] if visualize._type_slot(n) == visualize._type_slot(names[0]))
		colours = visualize.type_colours([names[0], twin])
		self.assertEqual(colours[names[0]], colours[twin])
		self.assertEqual(visualize.type_colours([names[0]])[names[0]], colours[names[0]])
		self.assertEqual(visualize.colour_collisions([names[0], twin]), [sorted([names[0], twin])])
		self.assertEqual(visualize.colour_collisions([names[0]]), [])

	def test_type_label_sits_above_its_node(self):
		runs, rows, _ = self.flow(run=self.alpha)
		plt = visualize._pyplot()
		fig = visualize.draw_semantic_flow(plt, runs[0], rows, "gpu_energy_j")
		ax = fig.axes[0]
		patches = {p.get_gid(): p for p in ax.patches}
		for kind in ("llm", "kg"):
			node = patches[f"node:operation_type:{kind}"]
			label = next(t for t in ax.texts if t.get_text().startswith(kind.upper()))
			self.assertGreater(label.get_position()[1], node.get_y() + node.get_height())
			self.assertIsNone(label.get_bbox_patch())
		plt.close(fig)

	def test_unavailable_or_partial_domain_is_refused(self):
		code, _, err = self.plot("semantic-flow", self.alpha, extra=("--domain", "dram_energy_j"))
		self.assertEqual(code, 2)
		self.assertIn("not measured", err)
		events = alpha_events()
		events[0] = dict(events[0], gpu_energy_j=None)
		run = write_run(os.path.join(self.tmp, "run_partial"), events)
		code, _, err = self.plot("semantic-flow", run)
		self.assertEqual(code, 2)
		self.assertIn("partial sum", err)

	def test_one_run_per_figure_and_attributed_basis_only(self):
		code, _, err = self.plot("semantic-flow", self.alpha, self.beta)
		self.assertEqual(code, 2)
		self.assertIn("one run per figure", err)
		code, _, err = self.plot("semantic-flow", self.alpha, extra=("--basis", "trajectory"))
		self.assertEqual(code, 2)

	def test_title_follows_domain_and_units_scale(self):
		self.assertEqual(visualize.plot_title("semantic-flow", "cpu_package_energy_j"),
		                 "CPU Package Energy by Semantic Stage")
		self.assertEqual(visualize.plot_title("semantic-flow", "dram_energy_j"),
		                 "DRAM Energy by Semantic Stage")
		self.assertEqual(visualize.energy_unit(950.0), (1.0, "J"))
		self.assertEqual(visualize.energy_unit(2.27e6), (1e3, "kJ"))
		self.assertEqual(visualize.energy_unit(5.0e7), (1e6, "MJ"))


if __name__ == "__main__":
	unittest.main()
