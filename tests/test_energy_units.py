"""Figures draw energy in Wh by default; j_to_wh is the one J -> Wh conversion."""

import math
import os
import sys
import unittest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "measurement"))

from agent_energy_profiler import visualize  # noqa: E402


class JToWh(unittest.TestCase):
	def test_scalar(self):
		self.assertEqual(visualize.j_to_wh(7200.0), 2.0)
		self.assertEqual(visualize.j_to_wh(3600), 1.0)

	def test_none_stays_none(self):
		self.assertIsNone(visualize.j_to_wh(None))
		self.assertEqual(visualize.j_to_wh([3600.0, None, 1800.0]), [1.0, None, 0.5])

	def test_container_types(self):
		self.assertEqual(visualize.j_to_wh((3600.0, 7200.0)), (1.0, 2.0))
		self.assertEqual(visualize.j_to_wh(x for x in (3600.0,)), [1.0])

	def test_numpy_and_pandas_keep_nan(self):
		import numpy as np
		arr = visualize.j_to_wh(np.array([3600.0, np.nan]))
		self.assertEqual(arr[0], 1.0)
		self.assertTrue(math.isnan(arr[1]))
		try:
			import pandas as pd
		except ImportError:  # pragma: no cover
			return
		series = visualize.j_to_wh(pd.Series([7200.0, None], name="energy_j"))
		self.assertIsInstance(series, pd.Series)
		self.assertEqual(series.iloc[0], 2.0)
		self.assertTrue(math.isnan(series.iloc[1]))

	def test_same_bits_as_division(self):
		# CSVs derived alongside figures must not change: same arithmetic as e / 3600.
		for e in (0.1, 123456.789, 2209.1614773367123):
			self.assertEqual(visualize.j_to_wh(e), e / 3600.0)


class Defaults(unittest.TestCase):
	def test_display_default_is_wh(self):
		self.assertEqual(visualize.DEFAULT_ENERGY_UNIT, "Wh")
		self.assertEqual(visualize.display_energy_unit(), "Wh")
		self.assertEqual(visualize.to_display_energy(3600.0), 1.0)
		with visualize.energy_display("J"):
			self.assertEqual(visualize.to_display_energy(3600.0), 3600.0)
		self.assertEqual(visualize.display_energy_unit(), "Wh")

	def test_figure_clis_default_to_wh(self):
		import correctness_energy
		import correctness_vs_energy
		import make_comparable_figures
		for module in (correctness_energy, correctness_vs_energy, make_comparable_figures):
			src = open(module.__file__, encoding="utf-8").read()
			self.assertIn('"--energy-unit", default=v', src.replace("visualize.DEFAULT", "v.DEFAULT")
			              .replace("default=v.DEFAULT_ENERGY_UNIT", "default=v"), module.__name__)


if __name__ == "__main__":
	unittest.main()
