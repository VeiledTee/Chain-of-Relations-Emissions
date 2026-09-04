"""Tests for the hardware-validation layer and trajectory accounting.

Kept separate from tests/test_energy_measurement.py so the frozen schema-v1
tests stay untouched. Pure unit tests: no GPU, no RAPL, no endpoints.

Run:
  python -m unittest discover -s tests -p "test_*.py" -v
"""

import importlib
import os
import sys
import unittest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "measurement"))

import attribute
import trajectory
import validate_hardware as hw


# --------------------------------------------------------------------------
# RAPL zone-name classification, against names real hosts actually produce
# --------------------------------------------------------------------------

class TestRaplNameClassification(unittest.TestCase):
	"""Names below are the real sysfs `name` values, not invented ones."""

	def test_intel_client_zone_names(self):
		cases = {
			"package-0": "cpu_package",
			"core": "cpu_core",
			"uncore": "uncore",
			"dram": "dram",
			"psys": "psys",
		}
		for name, expected in cases.items():
			self.assertEqual(hw.classify_zone_name(name), expected, name)

	def test_intel_dual_socket_zone_names(self):
		for name in ("package-0", "package-1"):
			self.assertEqual(hw.classify_zone_name(name), "cpu_package", name)

	def test_amd_socket_is_the_package_equivalent(self):
		self.assertEqual(hw.classify_zone_name("Esocket0"), "cpu_package")
		self.assertEqual(hw.classify_zone_name("socket-0"), "cpu_package")

	def test_amd_core_labels_are_diagnostic(self):
		self.assertEqual(hw.classify_zone_name("Ecore000"), "cpu_core")

	def test_uncore_is_not_mistaken_for_core(self):
		"""'uncore' contains the substring 'core'; order must not confuse them."""
		self.assertEqual(hw.classify_zone_name("uncore"), "uncore")
		self.assertNotEqual(hw.classify_zone_name("uncore"), "cpu_core")

	def test_unrecognized_name_is_unknown_not_silently_additive(self):
		self.assertEqual(hw.classify_zone_name("mmio-0"), "unknown")
		self.assertEqual(hw.classify_zone_name(""), "unknown")

	def test_diagnostic_and_excluded_domains_are_declared(self):
		self.assertIn("cpu_core", hw.DIAGNOSTIC_DOMAINS)
		self.assertIn("uncore", hw.DIAGNOSTIC_DOMAINS)
		self.assertIn("psys", hw.EXCLUDED_DOMAINS)
		for domain in hw.DIAGNOSTIC_DOMAINS + hw.EXCLUDED_DOMAINS:
			self.assertNotIn(domain, hw.BOUNDARY_DOMAINS)


class TestAttributeClassifyRapl(unittest.TestCase):
	"""attribute.classify_rapl must agree with the capability probe."""

	def _cols(self, *names):
		return {name: [] for name in names}

	def test_intel_full_set_separates_correctly(self):
		package, core, dram = attribute.classify_rapl(self._cols(
			"rapl_package-0_intel-rapl:0_uj",
			"rapl_core_intel-rapl:0:0_uj",
			"rapl_dram_intel-rapl:0:1_uj",
			"rapl_psys_intel-rapl:1_uj",
		))
		self.assertEqual(list(package), ["rapl_package-0_intel-rapl:0_uj"])
		self.assertEqual(list(core), ["rapl_core_intel-rapl:0:0_uj"])
		self.assertEqual(list(dram), ["rapl_dram_intel-rapl:0:1_uj"])

	def test_uncore_is_excluded_from_every_bucket(self):
		package, core, dram = attribute.classify_rapl(self._cols(
			"rapl_package-0_intel-rapl:0_uj",
			"rapl_uncore_intel-rapl:0:2_uj",
		))
		self.assertEqual(list(package), ["rapl_package-0_intel-rapl:0_uj"])
		self.assertEqual(core, {}, "uncore must not be filed as the core diagnostic")
		self.assertEqual(dram, {})

	def test_psys_never_enters_package(self):
		package, _, _ = attribute.classify_rapl(self._cols(
			"rapl_psys_intel-rapl:1_uj", "rapl_package-0_intel-rapl:0_uj"))
		self.assertEqual(list(package), ["rapl_package-0_intel-rapl:0_uj"])

	def test_amd_socket_counts_as_package(self):
		package, core, dram = attribute.classify_rapl(self._cols(
			"rapl_Esocket0_hwmon3_uj", "rapl_Ecore000_hwmon3_uj"))
		self.assertEqual(list(package), ["rapl_Esocket0_hwmon3_uj"])
		self.assertEqual(list(core), ["rapl_Ecore000_hwmon3_uj"])
		self.assertEqual(dram, {}, "AMD exposes no DRAM domain")

	def test_dual_socket_package_zones_both_counted(self):
		package, _, _ = attribute.classify_rapl(self._cols(
			"rapl_package-0_intel-rapl:0_uj", "rapl_package-1_intel-rapl:1_uj"))
		self.assertEqual(len(package), 2)

	def test_probe_and_attribute_agree_on_every_real_name(self):
		"""The capability probe must not promise a domain attribution drops."""
		for name in ("package-0", "core", "uncore", "dram", "psys",
		             "Esocket0", "Ecore000"):
			column = f"rapl_{name}_node_uj"
			package, core, dram = attribute.classify_rapl({column: []})
			probe = hw.classify_zone_name(name)
			in_package, in_core, in_dram = bool(package), bool(core), bool(dram)
			self.assertEqual(in_package, probe == "cpu_package", name)
			self.assertEqual(in_dram, probe == "dram", name)
			self.assertEqual(in_core, probe == "cpu_core", name)


class TestMissingDomainPolicy(unittest.TestCase):

	def test_no_rapl_yields_null_not_zero(self):
		event = {"start_timestamp": 0.0, "end_timestamp": 2.0, "duration_s": 2.0,
		         "gpu_energy_j": 3.0, "operation_label": "llm:reason",
		         "operation_type": "llm"}
		rows, _ = attribute.attribute_events([event], [0.0, 2.0], {}, {})
		row, = rows
		self.assertIsNone(row["cpu_package_energy_j"])
		self.assertIsNone(row["dram_energy_j"])
		self.assertIsNone(row["measured_energy_j"])
		self.assertFalse(row["measurement_complete"])
		self.assertEqual(row["available_energy_domains"], ["gpu"])

	def test_complete_boundary_sets_measurement_complete(self):
		ts = [0.0, 1.0, 2.0]
		rapls = {"rapl_package-0_intel-rapl:0_uj": [0.0, 10e6, 20e6],
		         "rapl_dram_intel-rapl:0:1_uj": [0.0, 2e6, 4e6]}
		event = {"start_timestamp": 0.0, "end_timestamp": 2.0, "duration_s": 2.0,
		         "gpu_energy_j": 1.0, "operation_label": "llm:reason",
		         "operation_type": "llm"}
		rows, _ = attribute.attribute_events([event], ts, {}, rapls)
		row, = rows
		self.assertTrue(row["measurement_complete"])
		self.assertAlmostEqual(row["measured_energy_j"], 25.0)

	def test_a_genuine_zero_is_not_confused_with_missing(self):
		"""0.0 J measured is a value; None is absence. They must differ."""
		ts = [0.0, 1.0, 2.0]
		rapls = {"rapl_package-0_intel-rapl:0_uj": [0.0, 0.0, 0.0],
		         "rapl_dram_intel-rapl:0:1_uj": [0.0, 0.0, 0.0]}
		event = {"start_timestamp": 0.0, "end_timestamp": 2.0, "duration_s": 2.0,
		         "gpu_energy_j": 0.0, "operation_label": "kg:id2name",
		         "operation_type": "kg"}
		rows, _ = attribute.attribute_events([event], ts, {}, rapls)
		row, = rows
		self.assertEqual(row["cpu_package_energy_j"], 0.0)
		self.assertIsNotNone(row["cpu_package_energy_j"])
		self.assertTrue(row["measurement_complete"])
		self.assertEqual(row["measured_energy_j"], 0.0)
		self.assertEqual(sorted(row["available_energy_domains"]),
		                 ["cpu_package", "dram", "gpu"])


class TestBoundaryVerdict(unittest.TestCase):

	def test_gpu_only_host_cannot_complete_the_boundary(self):
		nvml = {"status": hw.SUPPORTED, "cumulative_energy_status": hw.SUPPORTED}
		verdict = hw.boundary_verdict(nvml, [])
		self.assertEqual(verdict["available_energy_domains"], ["gpu"])
		self.assertEqual(sorted(verdict["missing_domains"]), ["cpu_package", "dram"])
		self.assertFalse(verdict["measurement_complete_possible"])

	def test_full_host_can_complete_the_boundary(self):
		nvml = {"status": hw.SUPPORTED, "cumulative_energy_status": hw.SUPPORTED}
		zones = [
			{"name": "package-0", "domain": "cpu_package", "status": hw.SUPPORTED},
			{"name": "dram", "domain": "dram", "status": hw.SUPPORTED},
			{"name": "core", "domain": "cpu_core", "status": hw.SUPPORTED},
		]
		verdict = hw.boundary_verdict(nvml, zones)
		self.assertTrue(verdict["measurement_complete_possible"])
		self.assertEqual(verdict["missing_domains"], [])

	def test_core_alone_does_not_satisfy_the_package_domain(self):
		"""core is diagnostic; a host with only core cannot claim package."""
		nvml = {"status": hw.SUPPORTED, "cumulative_energy_status": hw.SUPPORTED}
		zones = [{"name": "core", "domain": "cpu_core", "status": hw.SUPPORTED}]
		verdict = hw.boundary_verdict(nvml, zones)
		self.assertIn("cpu_package", verdict["missing_domains"])

	def test_unreadable_zone_is_not_treated_as_supported(self):
		nvml = {"status": hw.SUPPORTED, "cumulative_energy_status": hw.SUPPORTED}
		zones = [{"name": "package-0", "domain": "cpu_package",
		          "status": hw.UNREADABLE}]
		verdict = hw.boundary_verdict(nvml, zones)
		self.assertIn("cpu_package", verdict["missing_domains"])
		self.assertEqual(verdict["domains"]["cpu_package"]["status"], hw.UNREADABLE)


class TestCounterConsistency(unittest.TestCase):
	"""Recorded hardware cases, not synthetic ones.

	Both figures below came from the same 10 s probe run on real machines:
	cumulative NVML energy differenced across the window, against the
	trapezoidal integral of nvmlDeviceGetPowerUsage sampled at 20 Hz.
	"""

	# RTX 4090, WSL: counter and integrated power agree to ~0.1%.
	GOOD = dict(counter_j=1005.0, sampled_j=1004.0, elapsed_s=10.0, n_samples=200)
	# RTX 3050 Ti Laptop, bare-metal Ubuntu: counter advances steadily at a rate
	# no reading of board power can account for.
	BAD = dict(counter_j=954.652, sampled_j=67.106, elapsed_s=10.0, n_samples=200)

	def test_known_good_counter_is_supported(self):
		r = hw.assess_counter_consistency(**self.GOOD)
		self.assertEqual(r["verdict"], hw.SUPPORTED, r["detail"])
		self.assertLess(r["relative_difference_pct"], 1.0)

	def test_known_bad_counter_is_inconsistent(self):
		r = hw.assess_counter_consistency(**self.BAD)
		self.assertEqual(r["verdict"], hw.INCONSISTENT, r["detail"])
		self.assertAlmostEqual(r["ratio"], 14.226, places=2)
		self.assertAlmostEqual(r["relative_difference_pct"], 1322.6, places=0)

	def test_inconsistent_is_not_supported_and_not_merely_unknown(self):
		"""Advancing-but-wrong must be its own verdict: not SUPPORTED, and not
		silently folded into 'we could not tell'."""
		verdict = hw.assess_counter_consistency(**self.BAD)["verdict"]
		self.assertNotEqual(verdict, hw.SUPPORTED)
		self.assertNotEqual(verdict, hw.UNKNOWN)

	def test_tolerance_is_not_tuned_to_these_two_machines(self):
		"""The band must be a general one: comfortably wider than sampling
		noise, far narrower than the observed failure."""
		self.assertGreaterEqual(hw.CONSISTENCY_REL_TOL, 0.25)
		self.assertLessEqual(hw.CONSISTENCY_REL_TOL, 1.0)
		good = hw.assess_counter_consistency(**self.GOOD)
		bad = hw.assess_counter_consistency(**self.BAD)
		self.assertLess(good["difference_j"], good["tolerance_j"] / 10.0)
		self.assertGreater(bad["difference_j"], bad["tolerance_j"] * 10.0)

	def test_ordinary_sampling_noise_still_passes(self):
		"""A 20% disagreement is normal for 20 Hz sampling of a bursty load."""
		for sampled in (833.0, 1200.0):  # counter 20% above and below sampled
			r = hw.assess_counter_consistency(counter_j=1000.0, sampled_j=sampled,
			                                  elapsed_s=10.0, n_samples=200)
			self.assertEqual(r["verdict"], hw.SUPPORTED, r["detail"])

	def test_counter_off_by_a_factor_of_two_fails(self):
		r = hw.assess_counter_consistency(counter_j=2000.0, sampled_j=1000.0,
		                                  elapsed_s=10.0, n_samples=200)
		self.assertEqual(r["verdict"], hw.INCONSISTENT)

	def test_undercounting_is_caught_as_well_as_overcounting(self):
		r = hw.assess_counter_consistency(counter_j=100.0, sampled_j=1000.0,
		                                  elapsed_s=10.0, n_samples=200)
		self.assertEqual(r["verdict"], hw.INCONSISTENT)

	def test_short_window_yields_no_verdict(self):
		r = hw.assess_counter_consistency(counter_j=954.0, sampled_j=67.0,
		                                  elapsed_s=0.5, n_samples=10)
		self.assertEqual(r["verdict"], hw.UNKNOWN)

	def test_idle_window_with_too_little_energy_yields_no_verdict(self):
		r = hw.assess_counter_consistency(counter_j=0.4, sampled_j=0.1,
		                                  elapsed_s=10.0, n_samples=200)
		self.assertEqual(r["verdict"], hw.UNKNOWN)

	def test_missing_signal_yields_no_verdict_not_a_failure(self):
		self.assertEqual(
			hw.assess_counter_consistency(None, 10.0, 10.0, 200)["verdict"],
			hw.UNKNOWN)
		self.assertEqual(
			hw.assess_counter_consistency(10.0, None, 10.0, 200)["verdict"],
			hw.UNKNOWN)

	def test_edge_allowance_covers_a_bursty_mostly_idle_window(self):
		"""Window edges are not sample-aligned. On an idle window with bursts,
		the misalignment is worth burst power, so peak power sets the floor of
		the tolerance and such a window is not condemned."""
		bursty = dict(counter_j=40.0, sampled_j=20.0, elapsed_s=10.0, n_samples=200)
		self.assertEqual(
			hw.assess_counter_consistency(**bursty)["verdict"], hw.INCONSISTENT)
		r = hw.assess_counter_consistency(peak_power_w=250.0, **bursty)
		self.assertEqual(r["verdict"], hw.SUPPORTED, r["detail"])

	def test_peak_power_allowance_cannot_rescue_the_known_bad_counter(self):
		"""The edge allowance is bounded by real board power: a 14x counter
		stays rejected even when the GPU peaked at its power limit."""
		r = hw.assess_counter_consistency(peak_power_w=80.0, **self.BAD)
		self.assertEqual(r["verdict"], hw.INCONSISTENT, r["detail"])

	def test_integrate_power_is_a_trapezoidal_integral(self):
		self.assertAlmostEqual(hw.integrate_power([(0.0, 10.0), (1.0, 20.0)]), 15.0)
		self.assertAlmostEqual(hw.integrate_power([(0.0, 5.0)]), 0.0)
		self.assertAlmostEqual(hw.integrate_power([]), 0.0)


class TestInconsistentCounterLeavesTheBoundary(unittest.TestCase):

	def _nvml(self, verdict, **extra):
		nvml = {"status": hw.SUPPORTED, "cumulative_energy_status": hw.SUPPORTED,
		        "counter_consistency": {"verdict": verdict}}
		nvml.update(extra)
		return nvml

	def test_inconsistent_gpu_counter_is_not_an_available_domain(self):
		verdict = hw.boundary_verdict(self._nvml(hw.INCONSISTENT), [])
		self.assertNotIn("gpu", verdict["available_energy_domains"])
		self.assertIn("gpu", verdict["missing_domains"])
		self.assertEqual(verdict["domains"]["gpu"]["status"], hw.INCONSISTENT)
		self.assertFalse(verdict["measurement_complete_possible"])

	def test_inconsistent_gpu_blocks_an_otherwise_complete_host(self):
		zones = [{"name": "package-0", "domain": "cpu_package", "status": hw.SUPPORTED},
		         {"name": "dram", "domain": "dram", "status": hw.SUPPORTED}]
		verdict = hw.boundary_verdict(self._nvml(hw.INCONSISTENT), zones)
		self.assertFalse(verdict["measurement_complete_possible"])
		self.assertEqual(verdict["missing_domains"], ["gpu"])

	def test_consistent_gpu_counter_remains_supported(self):
		verdict = hw.boundary_verdict(self._nvml(hw.SUPPORTED), [])
		self.assertEqual(verdict["available_energy_domains"], ["gpu"])

	def test_unknown_verdict_does_not_revoke_a_readable_counter(self):
		"""No claim either way must not be treated as evidence of failure."""
		verdict = hw.boundary_verdict(self._nvml(hw.UNKNOWN), [])
		self.assertEqual(verdict["available_energy_domains"], ["gpu"])

	def test_host_without_the_cross_check_behaves_as_before(self):
		nvml = {"status": hw.SUPPORTED, "cumulative_energy_status": hw.SUPPORTED}
		self.assertEqual(hw.boundary_verdict(nvml, [])["available_energy_domains"],
		                 ["gpu"])

	def test_sampled_power_never_becomes_the_gpu_source(self):
		"""Even where sampled power is the only credible figure, it is not a
		measurement: the domain goes missing rather than being filled in."""
		verdict = hw.boundary_verdict(self._nvml(hw.INCONSISTENT), [])
		source = verdict["domains"]["gpu"]["source"] or ""
		self.assertNotIn("sampled", source.lower())
		self.assertIn("gpu", verdict["missing_domains"])


# --------------------------------------------------------------------------
# trajectory accounting
# --------------------------------------------------------------------------

def _event(q, t0, t1, gpu=None, **extra):
	e = {"question_id": q, "start_timestamp": t0, "end_timestamp": t1,
	     "duration_s": t1 - t0, "gpu_energy_j": gpu,
	     "cpu_package_energy_j": None, "dram_energy_j": None,
	     "measured_energy_j": None, "status": "ok"}
	e.update(extra)
	return e


class TestOverlapDetection(unittest.TestCase):

	def test_sequential_events_have_no_overlap(self):
		union, busy, overlap, depth = trajectory.overlap_stats(
			[(0.0, 1.0), (1.0, 2.0), (2.0, 3.0)])
		self.assertAlmostEqual(union, 3.0)
		self.assertAlmostEqual(busy, 3.0)
		self.assertAlmostEqual(overlap, 0.0)
		self.assertEqual(depth, 1)

	def test_nested_event_is_detected(self):
		union, busy, overlap, depth = trajectory.overlap_stats(
			[(0.0, 10.0), (2.0, 4.0)])
		self.assertAlmostEqual(union, 10.0)
		self.assertAlmostEqual(busy, 12.0)
		self.assertAlmostEqual(overlap, 2.0)
		self.assertEqual(depth, 2)

	def test_partial_overlap_is_detected(self):
		union, busy, overlap, depth = trajectory.overlap_stats(
			[(0.0, 2.0), (1.0, 3.0)])
		self.assertAlmostEqual(union, 3.0)
		self.assertAlmostEqual(overlap, 1.0)
		self.assertEqual(depth, 2)

	def test_gaps_are_excluded_from_union(self):
		union, _, overlap, _ = trajectory.overlap_stats([(0.0, 1.0), (5.0, 6.0)])
		self.assertAlmostEqual(union, 2.0)
		self.assertAlmostEqual(overlap, 0.0)

	def test_zero_duration_events_are_handled(self):
		union, busy, overlap, depth = trajectory.overlap_stats([(1.0, 1.0)])
		self.assertAlmostEqual(union, 0.0)
		self.assertAlmostEqual(busy, 0.0)
		self.assertAlmostEqual(overlap, 0.0)


class TestTrajectoryAccounting(unittest.TestCase):

	def test_accounting_identity_holds(self):
		"""unattributed == trajectory - attributed, exactly."""
		events = [_event("q1", 0.0, 1.0, gpu=10.0), _event("q1", 1.0, 2.0, gpu=20.0)]
		rows = trajectory.accumulate(
			events, window_energy=lambda t0, t1: {"gpu_energy_j": 50.0,
			                                      "cpu_package_energy_j": None,
			                                      "dram_energy_j": None,
			                                      "measured_energy_j": None})
		row, = rows
		self.assertAlmostEqual(row["sum_attributed_gpu_energy_j"], 30.0)
		self.assertAlmostEqual(row["trajectory_gpu_energy_j"], 50.0)
		self.assertAlmostEqual(row["unattributed_gpu_energy_j"], 20.0)
		self.assertAlmostEqual(
			row["trajectory_gpu_energy_j"] - row["sum_attributed_gpu_energy_j"],
			row["unattributed_gpu_energy_j"])

	def test_coverage_is_the_attributed_fraction(self):
		events = [_event("q1", 0.0, 1.0, gpu=25.0)]
		rows = trajectory.accumulate(
			events, window_energy=lambda t0, t1: {"gpu_energy_j": 100.0,
			                                      "cpu_package_energy_j": None,
			                                      "dram_energy_j": None,
			                                      "measured_energy_j": None})
		self.assertAlmostEqual(rows[0]["coverage_gpu_energy_j"], 0.25)

	def test_coverage_is_not_forced_to_one(self):
		events = [_event("q1", 0.0, 1.0, gpu=1.0)]
		rows = trajectory.accumulate(
			events, window_energy=lambda t0, t1: {"gpu_energy_j": 100.0,
			                                      "cpu_package_energy_j": None,
			                                      "dram_energy_j": None,
			                                      "measured_energy_j": None})
		self.assertLess(rows[0]["coverage_gpu_energy_j"], 1.0)
		self.assertGreater(rows[0]["unattributed_gpu_energy_j"], 0.0)

	def test_no_independent_window_means_no_coverage_claim(self):
		"""Without an independent total, coverage would be 1.0 by construction."""
		events = [_event("q1", 0.0, 1.0, gpu=10.0)]
		rows = trajectory.accumulate(events, window_energy=None)
		row, = rows
		self.assertIsNone(row["coverage_gpu_energy_j"])
		self.assertIsNone(row["unattributed_gpu_energy_j"])
		self.assertFalse(row["coverage_is_independent"])
		self.assertFalse(row["coverage_valid"])

	def test_overlap_invalidates_simple_summation(self):
		events = [_event("q1", 0.0, 10.0, gpu=100.0), _event("q1", 2.0, 4.0, gpu=20.0)]
		rows = trajectory.accumulate(
			events, window_energy=lambda t0, t1: {"gpu_energy_j": 100.0,
			                                      "cpu_package_energy_j": None,
			                                      "dram_energy_j": None,
			                                      "measured_energy_j": None})
		row, = rows
		self.assertTrue(row["events_overlap"])
		self.assertEqual(row["max_concurrency"], 2)
		self.assertGreater(row["coverage_gpu_energy_j"], 1.0)
		self.assertFalse(row["coverage_valid"],
		                 "coverage must be flagged invalid when events overlap")

	def test_missing_event_energy_makes_the_trajectory_total_null(self):
		events = [_event("q1", 0.0, 1.0, gpu=10.0), _event("q1", 1.0, 2.0, gpu=None)]
		rows = trajectory.accumulate(events)
		self.assertIsNone(rows[0]["sum_attributed_gpu_energy_j"])

	def test_inter_event_gap_is_reported(self):
		events = [_event("q1", 0.0, 1.0, gpu=1.0), _event("q1", 5.0, 6.0, gpu=1.0)]
		rows = trajectory.accumulate(events)
		row, = rows
		self.assertAlmostEqual(row["trajectory_wall_s"], 6.0)
		self.assertAlmostEqual(row["event_union_s"], 2.0)
		self.assertAlmostEqual(row["inter_event_gap_s"], 4.0)

	def test_trajectories_are_separated_by_question(self):
		events = [_event("q1", 0.0, 1.0, gpu=1.0), _event("q2", 2.0, 3.0, gpu=2.0)]
		rows = trajectory.accumulate(events)
		self.assertEqual(len(rows), 2)
		self.assertEqual({r["question_id"] for r in rows}, {"q1", "q2"})

	def test_zero_energy_trajectory_does_not_divide_by_zero(self):
		events = [_event("q1", 0.0, 1.0, gpu=0.0)]
		rows = trajectory.accumulate(
			events, window_energy=lambda t0, t1: {"gpu_energy_j": 0.0,
			                                      "cpu_package_energy_j": None,
			                                      "dram_energy_j": None,
			                                      "measured_energy_j": None})
		self.assertIsNone(rows[0]["coverage_gpu_energy_j"])

	def test_failed_events_are_counted_not_dropped(self):
		events = [_event("q1", 0.0, 1.0, gpu=1.0, status="error"),
		          _event("q1", 1.0, 2.0, gpu=1.0)]
		rows = trajectory.accumulate(events)
		self.assertEqual(rows[0]["n_events"], 2)
		self.assertEqual(rows[0]["n_failed_events"], 1)

	def test_schema_v1_context_is_carried_into_trajectory_rows(self):
		events = [_event("q1", 0.0, 1.0, gpu=1.0, iteration=0, traversal_depth=0),
		          _event("q1", 1.0, 2.0, gpu=1.0, iteration=3, traversal_depth=1)]
		rows = trajectory.accumulate(events)
		self.assertEqual(rows[0]["max_iteration"], 3)
		self.assertEqual(rows[0]["max_traversal_depth"], 1)

	def test_null_traversal_depth_does_not_break_the_max(self):
		events = [_event("q1", 0.0, 1.0, gpu=1.0, iteration=0, traversal_depth=None)]
		rows = trajectory.accumulate(events)
		self.assertIsNone(rows[0]["max_traversal_depth"])

	def test_summary_flags_coverage_above_one(self):
		events = [_event("q1", 0.0, 10.0, gpu=100.0), _event("q1", 2.0, 4.0, gpu=20.0)]
		rows = trajectory.accumulate(
			events, window_energy=lambda t0, t1: {"gpu_energy_j": 100.0,
			                                      "cpu_package_energy_j": None,
			                                      "dram_energy_j": None,
			                                      "measured_energy_j": None})
		summary = trajectory.summarize(rows)
		self.assertEqual(summary["n_coverage_above_one"], 1)
		self.assertEqual(summary["n_trajectories_with_overlap"], 1)

	def test_empty_input_summarizes_without_error(self):
		self.assertEqual(trajectory.summarize([])["n_trajectories"], 0)


class TestCounterBasedTrajectoryEnergy(unittest.TestCase):
	"""The trajectory reference must be the cumulative counter, not sampled power."""

	def _power_csv(self, tmpdir, header, rows):
		path = os.path.join(tmpdir, "power.csv")
		with open(path, "w") as f:
			f.write(",".join(header) + "\n")
			for r in rows:
				f.write(",".join("" if v is None else str(v) for v in r) + "\n")
		return path

	def test_load_power_separates_the_three_column_families(self):
		import tempfile
		with tempfile.TemporaryDirectory() as d:
			p = self._power_csv(
				d,
				["t", "gpu0_w", "gpu0_energy_mj", "rapl_package-0_intel-rapl:0_uj"],
				[[0.0, 100.0, 1000, 0], [1.0, 200.0, 2000, 5e6]])
			ts, gpu_power, rapl, gpu_energy = attribute.load_power(p)
		self.assertEqual(list(gpu_power), ["gpu0_w"])
		self.assertEqual(list(gpu_energy), ["gpu0_energy_mj"])
		self.assertEqual(list(rapl), ["rapl_package-0_intel-rapl:0_uj"])
		self.assertNotIn("gpu0_energy_mj", gpu_power,
		                 "counter column must not be treated as watts")

	def test_legacy_power_csv_yields_no_gpu_counter(self):
		"""A log predating the counter column reports unavailable, not zero."""
		import tempfile
		with tempfile.TemporaryDirectory() as d:
			p = self._power_csv(d, ["t", "gpu0_w"], [[0.0, 50.0], [1.0, 50.0]])
			ts, gpu_power, rapl, gpu_energy = attribute.load_power(p)
		self.assertEqual(gpu_energy, {})
		self.assertEqual(list(gpu_power), ["gpu0_w"])

	def test_counter_delta_converts_millijoules_to_joules(self):
		ts = [0.0, 1.0, 2.0]
		vals = [0.0, 1000.0, 3000.0]  # mJ
		self.assertAlmostEqual(
			attribute.counter_delta(ts, vals, 0.0, 2.0, 1e3), 3.0)

	def test_counter_delta_interpolates_within_a_sample_interval(self):
		ts = [0.0, 1.0]
		vals = [0.0, 1000.0]
		self.assertAlmostEqual(
			attribute.counter_delta(ts, vals, 0.0, 0.5, 1e3), 0.5)

	def test_counter_delta_is_none_without_enough_samples(self):
		self.assertIsNone(attribute.counter_delta([0.0], [1.0], 0.0, 1.0, 1e3))
		self.assertIsNone(attribute.counter_delta([], [], 0.0, 1.0, 1e3))

	def test_sum_counter_domain_is_none_when_group_empty(self):
		self.assertIsNone(attribute.sum_counter_domain([0.0, 1.0], {}, 0.0, 1.0, 1e3))

	def test_counter_and_sampled_fields_have_distinct_names(self):
		"""The diagnostic must never share a name with the reference."""
		self.assertIn("sampled_gpu_energy_estimate_j", trajectory.DIAGNOSTIC_FIELDS)
		self.assertIn("gpu_energy_j", trajectory.ENERGY_FIELDS)
		self.assertNotIn("sampled_gpu_energy_estimate_j", trajectory.ENERGY_FIELDS)
		for f in trajectory.DIAGNOSTIC_FIELDS:
			self.assertNotIn(f, trajectory.ENERGY_FIELDS, f)

	def test_coverage_uses_the_counter_not_the_sampled_estimate(self):
		"""Given both, coverage must divide by the counter."""
		events = [_event("q1", 0.0, 1.0, gpu=90.0)]
		window = {
			"gpu_energy_j": 100.0,               # counter  -> coverage 0.90
			"sampled_gpu_energy_estimate_j": 50.0,  # sampled -> would give 1.80
			"cpu_package_energy_j": None, "dram_energy_j": None,
			"measured_energy_j": None,
			"gpu_energy_source": "nvml_cumulative_counter",
		}
		row, = trajectory.accumulate(events, window_energy=lambda a, b: window)
		self.assertAlmostEqual(row["coverage_gpu_energy_j"], 0.90)
		self.assertAlmostEqual(row["trajectory_gpu_energy_j"], 100.0)
		self.assertAlmostEqual(row["sampled_gpu_energy_estimate_j"], 50.0)
		self.assertNotAlmostEqual(row["coverage_gpu_energy_j"], 1.80)

	def test_sampled_estimate_is_carried_but_never_becomes_the_reference(self):
		events = [_event("q1", 0.0, 1.0, gpu=10.0)]
		window = {
			"gpu_energy_j": None,                   # counter unavailable
			"sampled_gpu_energy_estimate_j": 42.0,  # must NOT be substituted
			"cpu_package_energy_j": None, "dram_energy_j": None,
			"measured_energy_j": None,
			"gpu_energy_source": "unavailable",
		}
		row, = trajectory.accumulate(events, window_energy=lambda a, b: window)
		self.assertIsNone(row["trajectory_gpu_energy_j"])
		self.assertIsNone(row["coverage_gpu_energy_j"])
		self.assertEqual(row["gpu_energy_source"], "unavailable")
		self.assertAlmostEqual(row["sampled_gpu_energy_estimate_j"], 42.0)

	def test_sampled_vs_counter_ratio_is_recorded(self):
		events = [_event("q1", 0.0, 1.0, gpu=50.0)]
		window = {
			"gpu_energy_j": 100.0,
			"sampled_gpu_energy_estimate_j": 80.0,
			"cpu_package_energy_j": None, "dram_energy_j": None,
			"measured_energy_j": None,
			"gpu_energy_source": "nvml_cumulative_counter",
		}
		row, = trajectory.accumulate(events, window_energy=lambda a, b: window)
		self.assertAlmostEqual(row["sampled_vs_counter_ratio"], 0.80)


class TestResidualNotClamped(unittest.TestCase):

	def _window(self, counter):
		return lambda a, b: {
			"gpu_energy_j": counter,
			"cpu_package_energy_j": None, "dram_energy_j": None,
			"measured_energy_j": None,
			"gpu_energy_source": "nvml_cumulative_counter",
			"sampled_gpu_energy_estimate_j": None,
		}

	def test_negative_residual_is_preserved(self):
		"""Events summing above the window must yield a negative residual."""
		events = [_event("q1", 0.0, 1.0, gpu=105.0)]
		row, = trajectory.accumulate(events, window_energy=self._window(100.0))
		self.assertAlmostEqual(row["unattributed_gpu_energy_j"], -5.0)
		self.assertLess(row["unattributed_gpu_energy_j"], 0.0)
		self.assertGreater(row["coverage_gpu_energy_j"], 1.0)

	def test_negative_residual_survives_the_identity(self):
		events = [_event("q1", 0.0, 1.0, gpu=105.0)]
		row, = trajectory.accumulate(events, window_energy=self._window(100.0))
		self.assertAlmostEqual(
			row["trajectory_gpu_energy_j"] - row["sum_attributed_gpu_energy_j"],
			row["unattributed_gpu_energy_j"])

	def test_summary_counts_and_reports_negative_residuals(self):
		events = [_event("q1", 0.0, 1.0, gpu=105.0),
		          _event("q2", 2.0, 3.0, gpu=80.0)]
		rows = trajectory.accumulate(events, window_energy=self._window(100.0))
		summary = trajectory.summarize(rows)
		self.assertEqual(summary["n_negative_gpu_residual"], 1)
		self.assertAlmostEqual(summary["gpu_residual_min_j"], -5.0)
		self.assertAlmostEqual(summary["max_abs_negative_gpu_residual_j"], 5.0)

	def test_zero_negative_residuals_reported_as_zero_not_missing(self):
		events = [_event("q1", 0.0, 1.0, gpu=80.0)]
		rows = trajectory.accumulate(events, window_energy=self._window(100.0))
		summary = trajectory.summarize(rows)
		self.assertEqual(summary["n_negative_gpu_residual"], 0)
		self.assertAlmostEqual(summary["max_abs_negative_gpu_residual_j"], 0.0)


class TestFrozenSchemaPreserved(unittest.TestCase):
	"""This slice must not have altered the frozen schema-v1 contract."""

	def test_carried_fields_still_include_the_frozen_context(self):
		for field in ("schema_version", "event_id", "run_id", "question_id",
		              "dataset", "paradigm", "iteration", "traversal_depth",
		              "step_index", "operation_type", "operation_label",
		              "input_tokens", "output_tokens", "status", "meta"):
			self.assertIn(field, attribute.CARRIED_FIELDS, field)

	def test_taxonomy_labels_unchanged(self):
		from chain_of_relations import energy_taxonomy as tax
		self.assertEqual(tax.SCHEMA_VERSION, 1)
		self.assertEqual(
			{l.value for l in tax.COR_LLM_LABELS},
			{"llm:relation_rank", "llm:reason", "llm:answer_filter",
			 "llm:direct_answer"})

	def test_measured_total_still_excludes_core(self):
		self.assertAlmostEqual(attribute.measured_total(1.0, 2.0, 3.0), 6.0)
		self.assertIsNone(attribute.measured_total(1.0, None, 3.0))


if __name__ == "__main__":
	unittest.main(verbosity=2)
