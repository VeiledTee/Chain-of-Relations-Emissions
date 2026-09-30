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
import json
import os
import signal
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)
sys.path.insert(0, HERE)

import run_preflight  # noqa: E402
from agent_energy_profiler import attribution  # noqa: E402
from agent_energy_profiler import events as profiler_events  # noqa: E402

#: Run-level provenance record: code/model/serving/GPU preflight + post-run.
PROVENANCE_FILE = "run_provenance.json"

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
	"run_provenance.json",
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


#: Floor on the post-run settle, so even a fast sampler covers the tail.
MIN_SAMPLER_SETTLE_S = 1.0


def sampler_settle_seconds(hz):
	"""How long to keep sampling after the run process exits.

	The run process writes an event's `end_timestamp` as that event finishes,
	so the last event of a run can end a few milliseconds AFTER the sampler's
	final tick. Attribution refuses a run with an event outside hardware
	coverage — correctly, since it will not fabricate energy — so stopping the
	sampler the instant the child exits made every short measured run
	unattributable. Waiting at least two sample periods guarantees a sample
	after the final event, and mirrors the baseline settle before the run.
	"""
	try:
		period = 1.0 / float(hz)
	except (TypeError, ValueError, ZeroDivisionError):
		period = 0.0
	return max(2.0 * period, MIN_SAMPLER_SETTLE_S)


def apply_policy(record, mode, max_util):
	"""Stamp mode/citability onto a preflight record; return the violations.

	full  -> any violation must abort the run (the caller exits before the
	         benchmark, sampler or run directory exist).
	smoke -> violations become citable_blockers; the run is never citable.
	"""
	problems = run_preflight.evaluate(record, max_util=max_util)
	record["mode"] = mode
	record["citable"] = mode == "full" and not problems
	record["citable_blockers"] = (["smoke/development mode"] + problems
	                              if mode == "smoke" else list(problems))
	record["warnings"] = run_preflight.warnings_for(record)
	record["policy"] = {
		"max_idle_gpu_util_pct": max_util,
		"full_mode_requires": [
			"clean git working tree", "non-empty MODEL_REVISION",
			"MODEL_REVISION equal to the live server's --revision when visible",
			"endpoint serves MODEL_NAME", "no running/waiting vLLM requests",
			"median GPU utilization <= max_idle_gpu_util_pct",
			"no visible non-vLLM GPU compute process"],
		"claim": "no competing GPU workload was detected by the available preflight "
		         "checks (device-level; Windows-side processes are not visible under WSL2)",
	}
	return problems


def energy_domains(power_f):
	"""Which energy instruments the hardware timeline actually carried.

	Read from the power.csv header with the attribution module's own RAPL
	classification, so this reports exactly what attribution could use.
	"""
	try:
		with open(power_f) as f:
			header = f.readline().strip().split(",")
	except OSError:
		return None
	rapl = {h: [] for h in header if h.startswith("rapl_")}
	pkg, core, dram = attribution.classify_rapl(rapl)
	return {"gpu_counter": any(h.startswith("gpu") and h.endswith("_energy_mj") for h in header),
	        "cpu_package": bool(pkg), "dram": bool(dram), "cpu_core_diagnostic": bool(core),
	        "measured_total_possible": bool(pkg) and bool(dram)}


def main():
	ap = argparse.ArgumentParser()
	ap.add_argument("--tag", required=True)
	ap.add_argument("--hz", type=float, default=10.0)
	ap.add_argument("--mode", choices=run_preflight.MODES, default="full",
	                help="full (default): citable measurement, any preflight violation "
	                     "aborts. smoke: development run, violations recorded and the "
	                     "run marked citable=false.")
	ap.add_argument("--max-idle-gpu-util", type=float,
	                default=float(os.getenv("PREFLIGHT_MAX_GPU_UTIL",
	                                        run_preflight.DEFAULT_MAX_IDLE_GPU_UTIL_PCT)),
	                help="median GPU utilization %% above which the device is "
	                     "considered busy (see run_preflight.py for the rationale)")
	ap.add_argument("run_args", nargs=argparse.REMAINDER,
	                help="args after -- passed to chain_of_relations.run")
	args = ap.parse_args()
	run_args = [a for a in args.run_args if a != "--"]

	outdir = os.path.join(HERE, "runs", args.tag)
	refuse_tag_reuse(outdir, args.tag)
	t_run_id = time.time()
	run_id = os.environ.get("ENERGY_RUN_ID") or f"{args.tag}-{int(t_run_id)}"

	# 0. preflight: provenance + GPU exclusivity, BEFORE anything is created.
	base_url = os.getenv("OPENAI_BASE_URL", "")
	record = run_preflight.collect(
		base_url=base_url, model_name=os.getenv("MODEL_NAME", ""),
		model_revision=os.getenv("MODEL_REVISION", ""), repo_root=ROOT,
		git_commit_fn=lambda: profiler_events.detect_git_commit(ROOT),
		git_dirty_fn=lambda: profiler_events.detect_git_dirty(ROOT))
	problems = apply_policy(record, args.mode, args.max_idle_gpu_util)
	if problems and args.mode == "full":
		print("ERROR: measured-run preflight failed; the benchmark was NOT started.",
		      file=sys.stderr)
		for problem in problems:
			print(f"  - {problem}", file=sys.stderr)
		print("Fix the above, or use --mode smoke for a non-citable development run.",
		      file=sys.stderr)
		sys.exit(3)
	for note in record["warnings"]:
		print(f"NOTE: {note}", file=sys.stderr)
	if args.mode == "smoke":
		print(f"SMOKE MODE: this run is NOT citable ({len(problems)} preflight "
		      f"violation(s) recorded).", file=sys.stderr)

	os.makedirs(outdir, exist_ok=True)
	events_f = os.path.join(outdir, "events.jsonl")
	power_f = os.path.join(outdir, "power.csv")
	log_f = os.path.join(outdir, "run.log")
	attributed_f = os.path.join(outdir, "events_attributed.jsonl")
	provenance_f = os.path.join(outdir, PROVENANCE_FILE)
	record.update({"schema": "run_provenance/1", "tag": args.tag, "run_id": run_id,
	               "created_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
	               "run_args": run_args, "sampler_hz": args.hz})
	run_preflight.write_json(provenance_f, record)

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
	# The preflight's code state is handed down so events, param.json and
	# run_provenance.json all carry the same commit/dirty values.
	code = record["code"]
	env = dict(os.environ,
	           ENERGY_EVENTS_FILE=events_f,
	           ENERGY_RUN_ID=run_id,
	           ENERGY_GIT_COMMIT=code["git_commit"] or "",
	           ENERGY_GIT_DIRTY={True: "true", False: "false"}.get(code["git_dirty"], "unknown"),
	           ENERGY_RUN_MODE=args.mode,
	           ENERGY_RUN_CITABLE="true" if record["citable"] else "false",
	           ENERGY_RUN_PROVENANCE_FILE=provenance_f)
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
	# Keep sampling past the run's last event: see sampler_settle_seconds.
	time.sleep(sampler_settle_seconds(args.hz))
	logger.send_signal(signal.SIGTERM)
	logger.wait(timeout=10)

	print(f"run exit={rc}, wall={wall:.0f}s; attributing...")
	arc = subprocess.call(
		[sys.executable, os.path.join(HERE, "attribute.py"),
		 "--events", events_f, "--power", power_f,
		 "--out", os.path.join(outdir, "energy_summary.csv"),
		 "--out-events", attributed_f])
	print(f"artifacts: {outdir}")

	# post-run provenance: GPU state after, clock anomalies, energy domains
	post = {"finished_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"), "wall_s": wall,
	        "run_exit_code": rc, "attribution_exit_code": arc,
	        "energy_domains": energy_domains(power_f), "clock": None}
	try:
		post["gpu_after"] = run_preflight.gpu_snapshot(base_url, samples=3)
	except Exception as exc:  # noqa: BLE001 - never fail a finished run on this
		post["gpu_after"] = {"error": f"{type(exc).__name__}: {exc}"}
	try:
		with open(os.path.join(outdir, "trajectory_summary.json")) as f:
			summary = json.load(f).get("summary", {})
		post["clock"] = {k: summary.get(k) for k in (
			"clock_anomaly", "power_backward_steps", "n_reversed_events",
			"n_clock_anomaly_trajectories")}
	except (OSError, ValueError):
		pass
	record["post_run"] = post
	run_preflight.write_json(provenance_f, record)

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
