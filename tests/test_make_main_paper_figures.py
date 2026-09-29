"""Structural tests for the main-paper figure pipeline: correlation-table
loading (Figure 2's data source) and the orchestrator's argument wiring.
These do not render pixels or touch real run data -- they check that the
expected inputs are loadable, that all expected (dataset, system) cells are
required, and that the orchestrator calls both underlying scripts with the
arguments it promises to forward.
"""

import csv
import os
import sys
import tempfile
import unittest
from unittest import mock

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "measurement"))

import thesis_figures as tf  # noqa: E402
import make_main_paper_figures as mmpf  # noqa: E402
import table1_effectiveness_energy as t1e  # noqa: E402

CORR_FIELDS = ("run", "paradigm", "x_variable", "y_variable", "energy_basis", "scale",
              "category_column", "category", "n_pairs", "pearson_r", "spearman_rho", "status")


def _write_correlation_csv(path, rho_by_variable):
	with open(path, "w", newline="") as f:
		w = csv.DictWriter(f, fieldnames=CORR_FIELDS, lineterminator="\n")
		w.writeheader()
		for variable, rho in rho_by_variable.items():
			w.writerow({"run": "T", "paradigm": "t", "x_variable": variable,
			           "y_variable": "gpu_energy_j", "energy_basis": "trajectory",
			           "scale": "raw values", "category_column": "outcome",
			           "category": "(drawn)", "n_pairs": 10, "pearson_r": rho + 0.01,
			           "spearman_rho": rho, "status": "ok"})


class LoadTrajectoryCorrelationsTest(unittest.TestCase):
	def test_loads_all_expected_cells(self):
		fake_rho = {"output_tokens": 0.9, "input_tokens": 0.8,
		           "llm_calls": 0.95, "max_traversal_depth": 0.6}
		with tempfile.TemporaryDirectory() as tmp:
			with mock.patch.object(tf, "ROOT", tmp):
				for ds in tf.DATASET_ORDER:
					for sys_name in tf.SYSTEM_ORDER:
						d = os.path.join(tmp, "results", "tables", ds, sys_name.lower())
						os.makedirs(d, exist_ok=True)
						_write_correlation_csv(
							os.path.join(d, f"fig3_{sys_name.lower()}_trajectory_properties_correlations.csv"),
							fake_rho)
				rho_by_key = tf.load_trajectory_correlations(tf.DATASET_ORDER, tf.SYSTEM_ORDER)
		self.assertEqual(len(rho_by_key), len(tf.DATASET_ORDER) * len(tf.SYSTEM_ORDER))
		for key, row in rho_by_key.items():
			for variable in tf.HEATMAP_VARIABLES:
				self.assertAlmostEqual(row[variable], fake_rho[variable])

	def test_missing_file_raises(self):
		with tempfile.TemporaryDirectory() as tmp:
			with mock.patch.object(tf, "ROOT", tmp):
				with self.assertRaises(SystemExit):
					tf.load_trajectory_correlations(tf.DATASET_ORDER, tf.SYSTEM_ORDER)

	def test_missing_variable_row_raises(self):
		with tempfile.TemporaryDirectory() as tmp:
			with mock.patch.object(tf, "ROOT", tmp):
				for ds in tf.DATASET_ORDER:
					for sys_name in tf.SYSTEM_ORDER:
						d = os.path.join(tmp, "results", "tables", ds, sys_name.lower())
						os.makedirs(d, exist_ok=True)
						incomplete = {"output_tokens": 0.9}  # missing the other 3 rows
						_write_correlation_csv(
							os.path.join(d, f"fig3_{sys_name.lower()}_trajectory_properties_correlations.csv"),
							incomplete)
				with self.assertRaises(SystemExit):
					tf.load_trajectory_correlations(tf.DATASET_ORDER, tf.SYSTEM_ORDER)


class DrawHeatmapTest(unittest.TestCase):
	def test_draws_without_error(self):
		plt = __import__("matplotlib.pyplot", fromlist=["pyplot"])
		rho_by_key = {(ds, sys_name): {v: 0.75 for v in tf.HEATMAP_VARIABLES}
		             for ds in tf.DATASET_ORDER for sys_name in tf.SYSTEM_ORDER}
		fig = tf.draw_trajectory_correlation_heatmap(plt, rho_by_key, tf.DATASET_ORDER,
		                                             tf.DATASET_LABELS, tf.SYSTEM_ORDER)
		self.assertEqual(len(fig.axes), len(tf.DATASET_ORDER) + 1)  # + colorbar axis
		plt.close(fig)


class OrchestratorWiringTest(unittest.TestCase):
	def test_forwards_dataset_and_backbone_to_both_scripts(self):
		dataset_args = ["webqsp=/fake/webqsp", "cwq=/fake/cwq"]
		with mock.patch.object(mmpf.thesis_figures, "main", return_value=0) as tf_main, \
		    mock.patch.object(mmpf.table1_effectiveness_energy, "main", return_value=0) as t1_main:
			rc = mmpf.main(["--dataset", dataset_args[0], "--dataset", dataset_args[1],
			               "--backbone", "Gemma-3-4B"])
		self.assertEqual(rc, 0)
		tf_main.assert_called_once()
		t1_main.assert_called_once()
		tf_argv = tf_main.call_args[0][0]
		t1_argv = t1_main.call_args[0][0]
		for argv in (tf_argv, t1_argv):
			self.assertIn("--dataset", argv)
			self.assertIn(dataset_args[0], argv)
			self.assertIn(dataset_args[1], argv)
		self.assertIn("Gemma-3-4B", t1_argv)

	def test_stops_if_figure_1_2_script_fails(self):
		with mock.patch.object(mmpf.thesis_figures, "main", return_value=1), \
		    mock.patch.object(mmpf.table1_effectiveness_energy, "main") as t1_main:
			rc = mmpf.main(["--dataset", "webqsp=/fake/webqsp", "--dataset", "cwq=/fake/cwq"])
		self.assertEqual(rc, 1)
		t1_main.assert_not_called()

	def test_forwards_out_figures_to_figure_3_script(self):
		"""Regression test: table1_effectiveness_energy.py used to hardcode its
		figure directory and ignore --out-figures entirely (it wrote straight into
		the live results/figures/cross_dataset regardless of what the orchestrator
		was told). The orchestrator must forward its --out-figures value through."""
		with mock.patch.object(mmpf.thesis_figures, "main", return_value=0), \
		    mock.patch.object(mmpf.table1_effectiveness_energy, "main", return_value=0) as t1_main:
			rc = mmpf.main(["--dataset", "webqsp=/fake/webqsp", "--dataset", "cwq=/fake/cwq",
			               "--out-figures", "/fake/custom_figures"])
		self.assertEqual(rc, 0)
		t1_argv = t1_main.call_args[0][0]
		self.assertIn("--out-figures", t1_argv)
		self.assertEqual(t1_argv[t1_argv.index("--out-figures") + 1], "/fake/custom_figures")


class Table1OutFiguresDefaultTest(unittest.TestCase):
	"""The --out-figures default must preserve the pre-fix behaviour exactly
	(results/figures/cross_dataset), checked without running the expensive body
	of main() or touching any real directory."""

	def test_default_is_the_historical_path(self):
		captured = {}
		real_parse_args = t1e.argparse.ArgumentParser.parse_args

		def spy(self, argv=None, namespace=None):
			ns = real_parse_args(self, argv, namespace)
			captured["args"] = ns
			raise RuntimeError("stop before any file I/O")

		with mock.patch.object(t1e.argparse.ArgumentParser, "parse_args", spy):
			with self.assertRaises(RuntimeError):
				t1e.main(["--dataset", "webqsp=/fake", "--dataset", "cwq=/fake"])
		self.assertEqual(captured["args"].out_figures, "results/figures/cross_dataset")


class Table1FigureOutputPathTest(unittest.TestCase):
	"""Regression test for the results/figures/cross_dataset hardcode: running
	table1_effectiveness_energy.main() end-to-end (with cache/scoring mocked so
	no real run data is needed) must write its figures under --out-figures and
	must not create or modify anything under the live results/ tree."""

	@staticmethod
	def _fake_load_cache(dataset, out_dir):
		systems = {name: {"label": name} for name in t1e.SYSTEM_ORDER}
		rows = [{"question_id": f"{dataset}_{name}_{i}", "energy_trajectory_j": 100.0 + 10.0 * i,
		        "fallback": i == 2, "run": name}
		       for name in t1e.SYSTEM_ORDER for i in range(3)]
		return rows, systems

	@staticmethod
	def _fake_score(pred_path, dataset, name):
		return [{"question_id": f"{dataset}_{name}_{i}", "hit1": 1.0 if i == 0 else 0.0,
		        "precision": 0.8, "recall": 0.7, "f1": 0.75,
		        "n_prediction_items": 2, "n_gold_answers": 2} for i in range(3)]

	def test_figures_land_in_out_figures_and_results_tree_is_untouched(self):
		real_figures_dir = os.path.join(t1e.ROOT, "results", "figures", "cross_dataset")
		before = (dict.fromkeys(os.listdir(real_figures_dir)) if os.path.isdir(real_figures_dir)
		         else None)
		before_mtimes = ({f: os.path.getmtime(os.path.join(real_figures_dir, f)) for f in before}
		                if before is not None else None)

		with tempfile.TemporaryDirectory() as out_tables, \
		    tempfile.TemporaryDirectory() as out_figures:
			with mock.patch.object(t1e, "_load_cache", side_effect=self._fake_load_cache), \
			    mock.patch.object(t1e.export_outcomes, "score", side_effect=self._fake_score):
				rc = t1e.main(["--dataset", "webqsp=/fake/webqsp", "--dataset", "cwq=/fake/cwq",
				              "--out", out_tables, "--out-figures", out_figures])
			self.assertEqual(rc, 0)
			written = set(os.listdir(out_figures))
			for stem in ("fig_effectiveness_energy_tradeoffs", "fig_f1_energy_frontier_main"):
				self.assertIn(f"{stem}.pdf", written)
				self.assertNotIn(f"{stem}.png", written)  # PDF-only per Generated Artifact Hygiene
			# Precision-vs-energy is no longer a canonical output (precision already
			# lives in the results tables) -- confirm it is not (re)generated here.
			self.assertNotIn("fig_precision_energy.pdf", written)
			self.assertNotIn("fig_precision_energy_v2.pdf", written)

		after = (dict.fromkeys(os.listdir(real_figures_dir)) if os.path.isdir(real_figures_dir)
		        else None)
		after_mtimes = ({f: os.path.getmtime(os.path.join(real_figures_dir, f)) for f in after}
		               if after is not None else None)
		self.assertEqual(before, after, "results/figures/cross_dataset file list changed")
		self.assertEqual(before_mtimes, after_mtimes,
		                 "a file under results/figures/cross_dataset was rewritten")


if __name__ == "__main__":
	unittest.main()
