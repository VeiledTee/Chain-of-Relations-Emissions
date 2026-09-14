"""The runner's --output_dir override, which keeps throwaway runs out of results/.

Without it every run of the same method/dataset/model shares one
`results/<method>/<dataset>/<model>/predict.jsonl`: a one-question smoke run
appends to (or resumes from) a finished experiment's predictions. These tests
pin the flag and the unchanged default.
"""

import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from chain_of_relations import run as run_module  # noqa: E402


class TestOutputDirFlag(unittest.TestCase):

	def test_parser_accepts_output_dir(self):
		args = run_module.build_parser().parse_args(
			["--method", "cor", "--dataset", "webqsp", "--output_dir", "/tmp/smoke"])
		self.assertEqual(args.output_dir, "/tmp/smoke")

	def test_output_dir_defaults_to_empty(self):
		"""Empty default means the historical results/ path is used unchanged."""
		args = run_module.build_parser().parse_args(["--method", "cor", "--dataset", "webqsp"])
		self.assertEqual(args.output_dir, "")

	def test_default_results_path_is_unchanged(self):
		"""The documented layout: results/<method>/<dataset>/<model dirname>."""
		expected = os.path.join(
			str(run_module.PROJECT_ROOT), "results", "cor", "webqsp", "gemma-3-4b-it")
		self.assertEqual(
			str(run_module.PROJECT_ROOT / "results" / "cor" / "webqsp"
			    / run_module.model_name_to_dirname("google/gemma-3-4b-it", "")),
			expected)


if __name__ == "__main__":
	unittest.main()
