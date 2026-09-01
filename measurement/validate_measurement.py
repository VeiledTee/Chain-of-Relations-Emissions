"""Controlled validation workloads for the measurement instrument.

These are VALIDATION workloads, not benchmark experiments. Each mode answers a
specific question about whether the instrument measures what it claims to.

  --mode idle       baseline draw and measurement noise with no intended work
  --mode cpu        repeated fixed SPARQL against Freebase, no LLM inference
  --mode gpu        deterministic GPU load, to prove the counter tracks work
  --mode repeated   N identical operations, for repeatability statistics
  --mode cor-smoke  print the tiny real-CoR command; does not launch it

Every mode reports each energy domain separately and never substitutes zero
for a domain the hardware does not expose.

Usage:
  python measurement/validate_measurement.py --mode idle --seconds 60
  python measurement/validate_measurement.py --mode cpu --reps 15
  python measurement/validate_measurement.py --mode gpu --seconds 20
  python measurement/validate_measurement.py --mode repeated --reps 15
  python measurement/validate_measurement.py --mode idle --json out.json
"""

import argparse
import json
import os
import statistics
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)
sys.path.insert(0, HERE)

import validate_hardware as hw


# --------------------------------------------------------------------------
# domain sampling
# --------------------------------------------------------------------------

class DomainCounters:
	"""Reads every available cumulative energy counter at an instant.

	Domains the host does not expose are absent from the reading rather than
	present as zero, so a missing domain can never be mistaken for no energy.
	"""

	def __init__(self):
		self.gpu_handles = []
		self.gpu_ok = False
		self.pynvml = None
		try:
			import pynvml
			pynvml.nvmlInit()
			self.pynvml = pynvml
			self.gpu_handles = [pynvml.nvmlDeviceGetHandleByIndex(i)
			                    for i in range(pynvml.nvmlDeviceGetCount())]
			if self.gpu_handles:
				pynvml.nvmlDeviceGetTotalEnergyConsumption(self.gpu_handles[0])
				self.gpu_ok = True
		except Exception:
			self.gpu_ok = False

		self.zones = []
		for zone in hw.discover_powercap_zones() + hw.discover_amd_hwmon_zones():
			domain = hw.classify_zone_name(zone["name"])
			if domain in ("cpu_package", "dram", "cpu_core"):
				try:
					open(zone["energy_uj_path"]).read()
				except Exception:
					continue
				self.zones.append((domain, zone))

	def available_domains(self):
		domains = ["gpu"] if self.gpu_ok else []
		domains.extend(sorted({d for d, _ in self.zones}))
		return domains

	def read(self):
		reading = {"t": time.time()}
		if self.gpu_ok:
			try:
				reading["gpu_mj"] = sum(
					self.pynvml.nvmlDeviceGetTotalEnergyConsumption(h)
					for h in self.gpu_handles)
			except Exception:
				pass
			try:
				reading["gpu_w"] = sum(
					self.pynvml.nvmlDeviceGetPowerUsage(h) / 1000.0
					for h in self.gpu_handles)
			except Exception:
				pass
		for domain, zone in self.zones:
			try:
				value = int(open(zone["energy_uj_path"]).read().strip())
			except Exception:
				continue
			reading.setdefault(f"{domain}_uj", 0)
			reading[f"{domain}_uj"] += value
		return reading

	def delta(self, first, second):
		"""Energy in Joules per domain between two readings, or None."""
		out = {"duration_s": second["t"] - first["t"]}
		if "gpu_mj" in first and "gpu_mj" in second:
			out["gpu_energy_j"] = (second["gpu_mj"] - first["gpu_mj"]) / 1000.0
		else:
			out["gpu_energy_j"] = None
		for domain, key in (("cpu_package", "cpu_package_uj"),
		                    ("dram", "dram_uj"),
		                    ("cpu_core", "cpu_core_uj")):
			field = "cpu_package_energy_j" if domain == "cpu_package" else (
				"dram_energy_j" if domain == "dram" else "cpu_core_energy_j")
			if key in first and key in second:
				out[field] = (second[key] - first[key]) / 1e6
			else:
				out[field] = None
		gpu, pkg, dram = (out["gpu_energy_j"], out["cpu_package_energy_j"],
		                  out["dram_energy_j"])
		out["measured_energy_j"] = (None if None in (gpu, pkg, dram)
		                            else gpu + pkg + dram)
		out["measurement_complete"] = out["measured_energy_j"] is not None
		return out


def measure(counters, fn):
	"""Run fn between two counter reads; return (result, per-domain energy)."""
	before = counters.read()
	result = fn()
	after = counters.read()
	return result, counters.delta(before, after)


# --------------------------------------------------------------------------
# statistics
# --------------------------------------------------------------------------

def describe(values, label):
	"""mean/median/sd/min/max/CV, or an explicit unavailable marker."""
	present = [v for v in values if v is not None]
	if not present:
		return {"field": label, "status": "UNAVAILABLE", "n": 0,
		        "detail": "no sample supplied this domain"}
	row = {
		"field": label, "status": "SUPPORTED", "n": len(present),
		"mean": statistics.fmean(present),
		"median": statistics.median(present),
		"min": min(present), "max": max(present),
	}
	row["stdev"] = statistics.stdev(present) if len(present) > 1 else None
	row["cv"] = (row["stdev"] / row["mean"]
	             if row["stdev"] is not None and row["mean"] else None)
	return row


def render_stats(rows, title, n_note=""):
	L = [f"--- {title} ---"]
	if n_note:
		L.append(f"    {n_note}")
	L.append(f"    {'field':<26} {'n':>3} {'mean':>12} {'median':>12} "
	         f"{'sd':>10} {'min':>12} {'max':>12} {'CV':>7}")
	for r in rows:
		if r["status"] != "SUPPORTED":
			L.append(f"    {r['field']:<26} {'-':>3} {r['status']:>12}"
			         f"   {r.get('detail', '')}")
			continue
		sd = "-" if r["stdev"] is None else f"{r['stdev']:.4f}"
		cv = "-" if r["cv"] is None else f"{r['cv']:.3f}"
		L.append(f"    {r['field']:<26} {r['n']:>3} {r['mean']:>12.4f} "
		         f"{r['median']:>12.4f} {sd:>10} {r['min']:>12.4f} "
		         f"{r['max']:>12.4f} {cv:>7}")
	return "\n".join(L)


# --------------------------------------------------------------------------
# workloads
# --------------------------------------------------------------------------

FIXED_SPARQL = (
	"PREFIX ns: <http://rdf.freebase.com/ns/>\n"
	"SELECT DISTINCT ?relation WHERE {\n"
	"  ns:m.0f2y0 ?relation ?x .\n"
	"} LIMIT 100"
)


def mode_idle(counters, args):
	"""Baseline draw and noise with no intended workload.

	Measured in fixed slices so both the total and the slice-to-slice noise
	are visible. Idle energy is characterized here, NOT subtracted from
	research measurements.
	"""
	slice_s = args.slice_seconds
	n_slices = max(1, int(args.seconds / slice_s))
	print(f"idle: {n_slices} slices x {slice_s}s "
	      f"(~{n_slices * slice_s:.0f}s total), machine should be otherwise quiet")

	slices = []
	for i in range(n_slices):
		_, energy = measure(counters, lambda: time.sleep(slice_s))
		slices.append(energy)
		if not args.quiet:
			gpu = energy["gpu_energy_j"]
			print(f"  slice {i + 1:>3}/{n_slices}  {energy['duration_s']:.2f}s  "
			      f"gpu={'-' if gpu is None else f'{gpu:.2f} J'}")

	fields = ["gpu_energy_j", "cpu_package_energy_j", "dram_energy_j",
	          "cpu_core_energy_j", "measured_energy_j"]
	stats = [describe([s[f] for s in slices], f) for f in fields]
	power = [s["gpu_energy_j"] / s["duration_s"]
	         for s in slices if s["gpu_energy_j"] is not None and s["duration_s"]]
	stats.append(describe(power, "idle_gpu_power_w"))

	print()
	print(render_stats(stats, f"idle baseline over {n_slices} x {slice_s}s slices"))
	return {"mode": "idle", "slices": slices, "stats": stats,
	        "slice_seconds": slice_s, "n_slices": n_slices}


def _sparql_call():
	from SPARQLWrapper import SPARQLWrapper, JSON
	endpoint = os.getenv("FREEBASE_SPARQL_ENDPOINT", "http://127.0.0.1:8890/sparql")
	sparql = SPARQLWrapper(endpoint)
	sparql.setQuery(FIXED_SPARQL)
	sparql.setReturnFormat(JSON)
	sparql.setTimeout(30)
	result = sparql.query().convert()
	return len(result["results"]["bindings"])


def mode_cpu(counters, args):
	"""Repeated fixed SPARQL, no LLM inference.

	Verifies CPU package energy rises (where available), DRAM rises (where
	available), and GPU stays near idle during pure KG work.
	"""
	print(f"cpu/kg: {args.reps} repetitions of one fixed SPARQL query")
	try:
		rows_returned = _sparql_call()
		print(f"  endpoint reachable, query returns {rows_returned} rows")
	except Exception as e:
		print(f"  SPARQL endpoint UNREACHABLE: {type(e).__name__}: {e}")
		return {"mode": "cpu", "status": "UNAVAILABLE", "detail": str(e)}

	reps = []
	for i in range(args.reps):
		count, energy = measure(counters, _sparql_call)
		energy["rows"] = count
		reps.append(energy)
		if not args.quiet:
			gpu = energy["gpu_energy_j"]
			print(f"  rep {i + 1:>3}/{args.reps}  {energy['duration_s'] * 1000:>8.1f} ms  "
			      f"rows={count}  gpu={'-' if gpu is None else f'{gpu:.3f} J'}")

	fields = ["duration_s", "gpu_energy_j", "cpu_package_energy_j",
	          "dram_energy_j", "measured_energy_j"]
	stats = [describe([r[f] for r in reps], f) for f in fields]
	print()
	print(render_stats(stats, f"fixed SPARQL x{args.reps}"))
	return {"mode": "cpu", "status": "SUPPORTED", "reps": reps, "stats": stats}


def mode_gpu(counters, args):
	"""Deterministic GPU load.

	No LLM is required: the question here is whether the NVML counter tracks
	real GPU work and by how much it separates from idle. Token-level checks
	need a real model endpoint and are out of scope for this mode.
	"""
	try:
		import torch
	except Exception as e:
		print(f"gpu: torch UNAVAILABLE ({e}) - cannot run GPU workload")
		return {"mode": "gpu", "status": "UNAVAILABLE", "detail": str(e)}
	if not torch.cuda.is_available():
		print("gpu: torch present but CUDA UNAVAILABLE - cannot run GPU workload")
		return {"mode": "gpu", "status": "UNAVAILABLE", "detail": "cuda unavailable"}

	size = args.matrix
	print(f"gpu: deterministic {size}x{size} matmul loop for ~{args.seconds}s "
	      f"on {torch.cuda.get_device_name(0)}")
	torch.manual_seed(0)
	a = torch.randn(size, size, device="cuda", dtype=torch.float32)
	b = torch.randn(size, size, device="cuda", dtype=torch.float32)
	torch.cuda.synchronize()

	iterations = {"n": 0}

	def workload():
		deadline = time.time() + args.seconds
		while time.time() < deadline:
			for _ in range(10):
				a.matmul(b)
				iterations["n"] += 1
			# GPU work is asynchronous: without this the counter would be read
			# before the queued kernels have actually executed, attributing
			# their energy to whatever event happens to be open next.
			torch.cuda.synchronize()

	_, energy = measure(counters, workload)
	energy["iterations"] = iterations["n"]
	energy["matrix"] = size
	if energy["gpu_energy_j"] is not None and energy["duration_s"]:
		energy["gpu_power_w"] = energy["gpu_energy_j"] / energy["duration_s"]

	def _j(value):
		return "-" if value is None else f"{value:.1f} J"

	print(f"  iterations           {iterations['n']}")
	print(f"  duration             {energy['duration_s']:.2f} s")
	print(f"  gpu energy           {_j(energy['gpu_energy_j'])}")
	power = energy.get("gpu_power_w")
	print(f"  mean gpu power       "
	      f"{'-' if power is None else f'{power:.1f} W'}")
	print(f"  cpu package          {_j(energy['cpu_package_energy_j'])}")
	print(f"  dram                 {_j(energy['dram_energy_j'])}")
	print(f"  measured total       {_j(energy['measured_energy_j'])}"
	      f"   complete={energy['measurement_complete']}")
	return {"mode": "gpu", "status": "SUPPORTED", "result": energy}


def mode_repeated(counters, args):
	"""N identical operations for repeatability/noise characterization."""
	print(f"repeated: {args.reps} identical operations "
	      f"(workload={args.repeat_workload})")
	if args.repeat_workload == "sparql":
		try:
			_sparql_call()
		except Exception as e:
			print(f"  SPARQL UNREACHABLE: {e}")
			return {"mode": "repeated", "status": "UNAVAILABLE", "detail": str(e)}
		operation = _sparql_call
	elif args.repeat_workload == "gpu":
		try:
			import torch
			assert torch.cuda.is_available()
		except Exception as e:
			print(f"  GPU workload UNAVAILABLE: {e}")
			return {"mode": "repeated", "status": "UNAVAILABLE", "detail": str(e)}
		torch.manual_seed(0)
		a = torch.randn(args.matrix, args.matrix, device="cuda")
		b = torch.randn(args.matrix, args.matrix, device="cuda")
		torch.cuda.synchronize()

		def operation():
			deadline = time.time() + args.slice_seconds
			n = 0
			while time.time() < deadline:
				for _ in range(10):
					a.matmul(b)
					n += 1
				torch.cuda.synchronize()
			return n
	else:
		def operation():
			time.sleep(args.slice_seconds)
			return 0

	reps = []
	for i in range(args.reps):
		_, energy = measure(counters, operation)
		reps.append(energy)
		if not args.quiet:
			gpu = energy["gpu_energy_j"]
			print(f"  rep {i + 1:>3}/{args.reps}  {energy['duration_s'] * 1000:>8.1f} ms  "
			      f"gpu={'-' if gpu is None else f'{gpu:.3f} J'}")

	fields = ["duration_s", "gpu_energy_j", "cpu_package_energy_j",
	          "dram_energy_j", "measured_energy_j"]
	stats = [describe([r[f] for r in reps], f) for f in fields]
	print()
	print(render_stats(
		stats, f"repeatability over {args.reps} identical operations",
		"CV on a small sample is indicative only; do not over-interpret."))
	return {"mode": "repeated", "status": "SUPPORTED", "reps": reps, "stats": stats}


def mode_cor_smoke(counters, args):
	"""Print the tiny real-CoR validation command. Does not launch it."""
	cmd = (
		"python measurement/measure_run.py --tag validate_cor_smoke -- \\\n"
		"    --method cor --dataset webqsp --kb freebase \\\n"
		f"    --run_size {args.reps} --depth 2 --relation_width 2 --save_detail true"
	)
	print("cor-smoke: requires BOTH a real LLM endpoint and Freebase.")
	print()
	print("  required environment:")
	print("    OPENAI_BASE_URL, OPENAI_API_KEY, MODEL_NAME, MODEL_REVISION")
	print("    FREEBASE_SPARQL_ENDPOINT")
	print()
	print("  command:")
	for line in cmd.splitlines():
		print(f"    {line}")
	print()
	print("  NOT LAUNCHED by this tool: launching it without a real model would")
	print("  produce LLM events whose GPU energy is idle draw, which must never")
	print("  be reported as inference energy.")
	return {"mode": "cor-smoke", "status": "PREPARED", "command": cmd}


MODES = {
	"idle": mode_idle,
	"cpu": mode_cpu,
	"gpu": mode_gpu,
	"repeated": mode_repeated,
	"cor-smoke": mode_cor_smoke,
}


def main():
	ap = argparse.ArgumentParser(description=__doc__,
	                             formatter_class=argparse.RawDescriptionHelpFormatter)
	ap.add_argument("--mode", required=True, choices=sorted(MODES))
	ap.add_argument("--seconds", type=float, default=60.0,
	                help="total duration for idle/gpu modes")
	ap.add_argument("--slice-seconds", type=float, default=5.0,
	                help="slice length for idle mode")
	ap.add_argument("--reps", type=int, default=15,
	                help="repetitions for cpu/repeated modes")
	ap.add_argument("--repeat-workload", default="sparql",
	                choices=["sparql", "gpu", "sleep"])
	ap.add_argument("--matrix", type=int, default=4096,
	                help="matmul size for gpu mode")
	ap.add_argument("--json", default="", help="write results as JSON here")
	ap.add_argument("--quiet", action="store_true")
	args = ap.parse_args()

	counters = DomainCounters()
	available = counters.available_domains()
	print("=" * 72)
	print(f"MEASUREMENT VALIDATION -- mode={args.mode}")
	print("=" * 72)
	print(f"available energy domains: {available or 'NONE'}")
	missing = [d for d in ("gpu", "cpu_package", "dram") if d not in available]
	if missing:
		print(f"UNAVAILABLE domains: {missing}  "
		      f"-> these stay null; they are never reported as zero")
	print()

	result = MODES[args.mode](counters, args)
	result["available_energy_domains"] = available
	result["missing_domains"] = missing
	result["measurement_complete_possible"] = not missing

	if args.json:
		with open(args.json, "w") as f:
			json.dump(result, f, indent=2, default=str)
		print(f"\nJSON -> {args.json}")
	return 0


if __name__ == "__main__":
	sys.exit(main())
