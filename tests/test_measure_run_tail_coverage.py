"""The measured run must keep sampling past its own last event.

The run process writes an event's `end_timestamp` as that event finishes, so
the final event of a run can end a few milliseconds after the power sampler's
last tick. `agent_energy_profiler.attribution` then refuses the whole run —
correctly, because it will not fabricate energy for an event the hardware
timeline does not cover. Stopping the sampler the instant the child exited
therefore made every short measured run unattributable:

    ERROR: 1 events (14.3%) fall outside hardware measurement coverage
      hardware timeline : ... -> 1789426074.621
      uncovered events  : ... -> 1789426074.640

Observed on real 1-question CoR (19 ms late) and ToG (24 ms late) smoke runs.
These tests pin the settle that fixes it, at the unit level and against the
real sampler subprocess.
"""

import csv
import os
import signal
import subprocess
import sys
import tempfile
import time
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

MEASUREMENT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                           "measurement")
sys.path.insert(0, MEASUREMENT)

import measure_run  # noqa: E402


class TestSamplerSettle(unittest.TestCase):
	"""The settle is at least two sample periods, never zero."""

	def test_covers_at_least_two_sample_periods(self):
		for hz in (0.5, 1.0, 2.0, 10.0, 50.0):
			with self.subTest(hz=hz):
				self.assertGreaterEqual(measure_run.sampler_settle_seconds(hz), 2.0 / hz)

	def test_slow_sampler_dominates_the_floor(self):
		"""At 0.5 Hz a period is 2 s, so the settle must exceed the 1 s floor."""
		self.assertEqual(measure_run.sampler_settle_seconds(0.5), 4.0)

	def test_fast_sampler_still_gets_the_floor(self):
		"""Two periods at 100 Hz is 20 ms — far too tight against process teardown."""
		self.assertEqual(measure_run.sampler_settle_seconds(100.0),
		                 measure_run.MIN_SAMPLER_SETTLE_S)

	def test_never_zero_or_negative_on_bad_input(self):
		for hz in (0, None, "", "abc"):
			with self.subTest(hz=hz):
				self.assertGreaterEqual(measure_run.sampler_settle_seconds(hz),
				                        measure_run.MIN_SAMPLER_SETTLE_S)

	def test_teardown_uses_the_settle_helper(self):
		"""Pin the call site: the fix must not be silently reverted to an instant stop."""
		with open(os.path.join(MEASUREMENT, "measure_run.py"), encoding="utf-8") as f:
			source = f.read()
		settle = source.index("sampler_settle_seconds(args.hz)")
		sigterm = source.index("logger.send_signal(signal.SIGTERM)")
		self.assertLess(settle, sigterm,
		                "the sampler must settle BEFORE it is signalled")


class TestRealSamplerTailCoverage(unittest.TestCase):
	"""End to end against the real sampler subprocess, no agent required.

	Reproduces the shape of the bug: a child process whose final event ends at
	the moment it exits, then the teardown sequence. The last power sample must
	land at or after that event's end timestamp.
	"""

	HZ = 10.0

	def _run_sampler(self, settle_seconds):
		"""Start the sampler, end an 'event' at child-exit, settle, stop.

		Returns (event_end_timestamp, last_sample_timestamp) or (None, None)
		when the host produced no samples at all.
		"""
		with tempfile.TemporaryDirectory() as tmp:
			power_csv = os.path.join(tmp, "power.csv")
			logger = subprocess.Popen(
				[sys.executable, os.path.join(MEASUREMENT, "power_logger.py"),
				 "--out", power_csv, "--hz", str(self.HZ)],
				stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
			try:
				time.sleep(1.0)                      # baseline, as measure_run does
				time.sleep(2.5 / self.HZ)            # stand in for the run itself
				event_end = time.time()              # the child's last event ends here
				time.sleep(settle_seconds)
				logger.send_signal(signal.SIGTERM)
				logger.wait(timeout=10)
			finally:
				if logger.poll() is None:            # pragma: no cover - safety net
					logger.kill()
					logger.wait(timeout=10)

			if not os.path.exists(power_csv):
				return None, None
			with open(power_csv) as f:
				stamps = [float(row["t"]) for row in csv.DictReader(f) if row.get("t")]
			return (event_end, stamps[-1]) if stamps else (None, None)

	def test_settle_keeps_the_final_event_inside_the_timeline(self):
		event_end, last_sample = self._run_sampler(
			measure_run.sampler_settle_seconds(self.HZ))
		if event_end is None:
			self.skipTest("no power samples on this host (no NVML/RAPL counters)")
		self.assertGreaterEqual(
			last_sample, event_end,
			f"last sample {last_sample:.3f} precedes the final event end "
			f"{event_end:.3f}: attribution would refuse this run")

	def test_without_the_settle_the_tail_is_lost(self):
		"""Characterises the bug: stopping immediately leaves the event uncovered.

		Not asserted as always-failing — the sampler may happen to tick in the
		gap — but the regression above must hold whatever this one does.
		"""
		event_end, last_sample = self._run_sampler(0.0)
		if event_end is None:
			self.skipTest("no power samples on this host (no NVML/RAPL counters)")
		self.assertIsInstance(last_sample, float)


if __name__ == "__main__":
	unittest.main()
