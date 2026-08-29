"""Whole-machine power sampler. Run on the HOST (not in a container).

Samples at ~10 Hz:
  - GPU cumulative energy (mJ) via NVML        -> the GPU measurement instrument
  - GPU power (W) via NVML, per GPU            -> diagnostic only
  - CPU package + DRAM energy (uJ) via RAPL    -> bare-metal Linux only
    (silently absent on WSL2 / locked HPC nodes; GPU columns still logged)

Output CSV columns:
  t, gpu<i>_w ..., gpu<i>_energy_mj ..., rapl_<domain>_uj ...

Every energy column is a CUMULATIVE counter that attribution.py differences
across a window. gpu<i>_w is instantaneous power, retained for diagnostics
(mean/peak power, ramp analysis, time-series plots) and NOT used as the
primary trajectory energy reference: integrating ~10 Hz point samples resolves
fast inference power transients poorly and errs in both directions, with error
growing as windows shorten. That previously produced attribution coverage > 1
on short trajectories.

Usage:
  python -m agent_energy_profiler.sampling --out power.csv   # Ctrl-C to stop
  python -m agent_energy_profiler.sampling --out power.csv --hz 10
"""

import argparse
import csv
import signal
import sys
import time

from .hardware import nvml, rapl

nvml.init()
_RAPL = rapl.discover_zones()


def build_header():
	"""CSV header for the columns this host can actually produce."""
	n = nvml.device_count()
	# Two GPU column families, deliberately distinct:
	#   gpu<i>_w          instantaneous power (W)  -- DIAGNOSTIC only
	#   gpu<i>_energy_mj  cumulative energy (mJ)   -- the measurement instrument
	# The cumulative counter is the same hardware register the in-band event
	# log reads, so trajectory totals and per-event energy come from one
	# instrument. Integrated sampled power is never the primary reference.
	return (["t"]
	        + [f"gpu{i}_w" for i in range(n)]
	        + [f"gpu{i}_energy_mj" for i in range(n)]
	        + rapl.column_names(_RAPL))


def sample_row(handles, rapl_paths):
	"""One CSV row. An unreadable counter writes an empty cell, never a zero."""
	row = [f"{time.time():.3f}"]
	for h in handles:
		watts = nvml.device_power_w(h)
		row.append(watts if watts is not None else "")
	for h in handles:
		energy = nvml.device_energy_mj(h)
		row.append(energy if energy is not None else "")
	for path in rapl_paths:
		value = rapl.read_zone(path)
		row.append(value if value is not None else "")
	return row


def warn_about_gaps():
	if nvml.device_count() == 0:
		print("WARNING: no NVML GPUs visible", file=sys.stderr)
	elif not nvml.energy_counter_supported():
		print("WARNING: NVML present but cumulative GPU energy unavailable -"
		      " trajectory GPU energy will be reported as unavailable, not"
		      " silently substituted with integrated power", file=sys.stderr)
	if not _RAPL:
		print("WARNING: no RAPL domains readable (WSL2/shared node/permissions?)"
		      " - GPU-only logging", file=sys.stderr)


def main():
	ap = argparse.ArgumentParser()
	ap.add_argument("--out", required=True)
	ap.add_argument("--hz", type=float, default=10.0)
	args = ap.parse_args()

	period = 1.0 / args.hz
	stop = {"flag": False}
	signal.signal(signal.SIGINT, lambda *_: stop.update(flag=True))
	signal.signal(signal.SIGTERM, lambda *_: stop.update(flag=True))

	handles = nvml.handles()
	warn_about_gaps()

	with open(args.out, "w", newline="") as f:
		w = csv.writer(f)
		w.writerow(build_header())
		while not stop["flag"]:
			w.writerow(sample_row(handles, _RAPL))
			f.flush()
			time.sleep(period)
	print(f"power log written: {args.out}")


if __name__ == "__main__":
	main()
