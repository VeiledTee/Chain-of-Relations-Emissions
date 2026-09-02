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
events_attributed.jsonl, energy_summary.csv, codecarbon/ (if installed),
run.log
"""

import argparse
import os
import signal
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)

#: Artifacts whose presence proves a measured run already used this directory.
#: Reusing such a directory silently corrupts the measurement: the event writer
#: APPENDS to events.jsonl while the power logger OVERWRITES power.csv, so the
#: older events survive with no hardware timeline covering them and reconcile
#: to zero trajectory energy.
RUN_ARTIFACTS = (
	"events.jsonl",
	"power.csv",
	"events_attributed.jsonl",
	"energy_summary.csv",
	"trajectory_summary.csv",
	"trajectory_summary.json",
	"run.log",
)


def existing_artifacts(outdir):
	"""Artifact filenames already present in outdir, in RUN_ARTIFACTS order.

	An empty directory (or one holding only empty subdirectories, such as a
	pre-created codecarbon/) yields an empty list and is safe to reuse: there
	is no prior measurement to contaminate.
	"""
	if not os.path.isdir(outdir):
		return []
	return [n for n in RUN_ARTIFACTS if os.path.exists(os.path.join(outdir, n))]


def refuse_tag_reuse(outdir, tag):
	"""Exit non-zero if outdir already holds a measured run.

	There is deliberately no --resume. Resuming a measured run would require
	appending to the hardware timeline with explicit gap handling, a run_id
	stable across invocations, and trajectory accounting over several disjoint
	measurement windows. None of those exist, so a partial run cannot be
	continued without contaminating the measurement: start a new tag.

	Note this is unrelated to CoR's own id-based resume, which skips questions
	already present in results/.../predict.jsonl. That still works, and pairs
	correctly with a FRESH measurement tag.
	"""
	found = existing_artifacts(outdir)
	if not found:
		return
	print("ERROR: refusing to reuse an existing measured-run directory.",
	      file=sys.stderr)
	print(f"  tag       : {tag}", file=sys.stderr)
	print(f"  directory : {outdir}", file=sys.stderr)
	print(f"  artifacts : {', '.join(found)}", file=sys.stderr)
	print("", file=sys.stderr)
	print("Reusing a tag appends new events to the old events.jsonl while",
	      file=sys.stderr)
	print("overwriting power.csv, leaving the earlier events with no hardware",
	      file=sys.stderr)
	print("timeline. Measured runs cannot be resumed; choose a fresh --tag",
	      file=sys.stderr)
	print("(or move/delete the existing directory if it is not needed).",
	      file=sys.stderr)
	sys.exit(2)


def main():
	ap = argparse.ArgumentParser()
	ap.add_argument("--tag", required=True)
	ap.add_argument("--hz", type=float, default=10.0)
	ap.add_argument("run_args", nargs=argparse.REMAINDER,
	                help="args after -- passed to chain_of_relations.run")
	args = ap.parse_args()
	run_args = [a for a in args.run_args if a != "--"]

	outdir = os.path.join(HERE, "runs", args.tag)
	refuse_tag_reuse(outdir, args.tag)
	os.makedirs(outdir, exist_ok=True)
	events_f = os.path.join(outdir, "events.jsonl")
	power_f = os.path.join(outdir, "power.csv")
	log_f = os.path.join(outdir, "run.log")
	attributed_f = os.path.join(outdir, "events_attributed.jsonl")
	t_run_id = time.time()

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

	# 3. the run, with the event log enabled. run_id is fixed here so every
	# artifact in this directory shares one identifier; the run process fills
	# in the rest of the provenance (git commit, hardware, model).
	env = dict(os.environ,
	           ENERGY_EVENTS_FILE=events_f,
	           ENERGY_RUN_ID=os.environ.get("ENERGY_RUN_ID") or f"{args.tag}-{int(t_run_id)}")
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
	arc = subprocess.call(
		[sys.executable, os.path.join(HERE, "attribute.py"),
		 "--events", events_f, "--power", power_f,
		 "--out", os.path.join(outdir, "energy_summary.csv"),
		 "--out-events", attributed_f])
	print(f"artifacts: {outdir}")
	if arc != 0:
		# Attribution rejected the run (e.g. events outside hardware coverage).
		# Fail the pipeline rather than leaving artifacts that look complete.
		print(f"ERROR: attribution failed (exit {arc}); this run is NOT a valid "
		      f"measurement.", file=sys.stderr)
		sys.exit(arc)
	if rc != 0:
		print(f"WARNING: the measured workload itself exited {rc}.",
		      file=sys.stderr)
		sys.exit(rc)


if __name__ == "__main__":
	main()
