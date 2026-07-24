"""Measured run orchestrator. Run on the HOST, machine otherwise idle.

Wraps one chain_of_relations.run invocation with:
  1. the power logger (GPU NVML + CPU/DRAM RAPL, 10 Hz)
  2. optional CodeCarbon machine-mode tracker (whole-run kWh + gCO2e)
  3. the energy event log (via ENERGY_EVENTS_FILE)
then joins events to power and writes the per-category/label/question summary.

Usage (passthrough after --):
  python measurement/measure_run.py --tag cor_webqsp_qwen7b -- \
      --method cor --dataset webqsp --relation_width 3 --entity_width 3 \
      --depth 3 --temperature_exploration 0.01 --temperature_reasoning 0.01 \
      --run_size 5 --save_detail true
Outputs in measurement/runs/<tag>/: events.jsonl, power.csv,
energy_summary.csv, codecarbon/ (if installed), run.log
"""

import argparse
import os
import signal
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)


def main():
	ap = argparse.ArgumentParser()
	ap.add_argument("--tag", required=True)
	ap.add_argument("--hz", type=float, default=10.0)
	ap.add_argument("run_args", nargs=argparse.REMAINDER,
	                help="args after -- passed to chain_of_relations.run")
	args = ap.parse_args()
	run_args = [a for a in args.run_args if a != "--"]

	outdir = os.path.join(HERE, "runs", args.tag)
	os.makedirs(outdir, exist_ok=True)
	events_f = os.path.join(outdir, "events.jsonl")
	power_f = os.path.join(outdir, "power.csv")
	log_f = os.path.join(outdir, "run.log")

	# 1. power logger
	logger = subprocess.Popen(
		[sys.executable, os.path.join(HERE, "power_logger.py"),
		 "--out", power_f, "--hz", str(args.hz)])
	time.sleep(1.0)  # let it establish baseline samples

	# 2. optional CodeCarbon (machine mode; needs RAPL for CPU/RAM, NVML for GPU)
	tracker = None
	os.makedirs(os.path.join(outdir, "codecarbon"), exist_ok=True)
	try:
		from codecarbon import OfflineEmissionsTracker
		tracker = OfflineEmissionsTracker(
			country_iso_code=os.getenv("CC_COUNTRY", "CAN"),
			output_dir=os.path.join(outdir, "codecarbon"),
			measure_power_secs=5, tracking_mode="machine",
			log_level="warning")
		tracker.start()
	except Exception as e:
		print(f"CodeCarbon unavailable ({e}) - continuing with raw power log only")

	# 3. the run, with the event log enabled
	env = dict(os.environ, ENERGY_EVENTS_FILE=events_f)
	t0 = time.time()
	with open(log_f, "w") as lf:
		rc = subprocess.call(
			[sys.executable, "-m", "chain_of_relations.run"] + run_args,
			cwd=ROOT, env=env, stdout=lf, stderr=subprocess.STDOUT)
	wall = time.time() - t0

	# teardown
	if tracker is not None:
		try:
			tracker.stop()
		except Exception:
			pass
	logger.send_signal(signal.SIGTERM)
	logger.wait(timeout=10)

	print(f"run exit={rc}, wall={wall:.0f}s; attributing...")
	subprocess.call(
		[sys.executable, os.path.join(HERE, "attribute.py"),
		 "--events", events_f, "--power", power_f,
		 "--out", os.path.join(outdir, "energy_summary.csv")])
	print(f"artifacts: {outdir}")


if __name__ == "__main__":
	main()
