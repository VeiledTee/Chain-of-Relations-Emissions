"""Join semantic energy events to the hardware power timeline.

Inputs:
  events.jsonl  (written by agent_energy_profiler.events during the run)
  power.csv     (written by agent_energy_profiler.sampling on the same clock)

Outputs:
  events_attributed.jsonl  every source event, unchanged, plus attributed
                           energy fields, the list of domains actually
                           measured (available_energy_domains) and whether
                           the boundary is complete (measurement_complete).
                           This is the canonical per-event artifact for C2.
  energy_summary.csv       derived aggregate (by operation_label, by
                           operation_type, by question).

Method:
  GPU:  prefer the in-band NVML counter delta the event already carries;
        otherwise integrate sampled GPU power (trapezoid) inside the event's
        window. Events shorter than the sampling period get nearest-sample
        power x duration (an estimate, flagged in EMISSIONS.md).
  RAPL: cumulative counters -> energy delta over the window, interpolated
        between samples; handles counter wraparound.

Energy accounting boundary (thesis definition):

    measured_energy_j = gpu_energy_j + cpu_package_energy_j + dram_energy_j

The RAPL `core` domain is DIAGNOSTIC ONLY: it is a subdomain contained within
`package`, so adding both double-counts CPU energy. cpu_core_energy_j is
reported for diagnostics and never enters the total.

Missing counters stay null. A domain the hardware did not report is not zero:
on WSL2 there is no /sys/class/powercap at all, so cpu/dram come back null and
measured_energy_j is null too. Reporting 0.0 there would silently understate
the measurement boundary.

Honest units:
  Per-QUESTION energy is the robust primitive (long windows, many samples).
  Per-EVENT energy is a timestamp-attributed estimate; sub-100ms SPARQL events
  are order-of-magnitude, not precision, figures. Idle machine assumed:
  concurrent activity is attributed to whatever event overlaps it.

Usage:
  python -m agent_energy_profiler.attribution --events events.jsonl --power power.csv \
      --out energy_summary.csv [--out-events events_attributed.jsonl]
"""

import argparse
import bisect
import csv
import json
import os
from collections import defaultdict

from . import trajectory

RAPL_MAX = 2**32 * 1e0  # wraparound guard; actual max_energy_range varies

#: Semantic fields carried through from the raw event to the attributed one.
#: Attribution must never lose question id, iteration, traversal depth, step
#: index, operation label, tokens, status or provenance.
CARRIED_FIELDS = (
	"schema_version", "event_id",
	"run_id", "question_id", "dataset", "paradigm",
	"iteration", "traversal_depth", "step_index",
	"operation_type", "operation_label",
	"start_timestamp", "end_timestamp", "duration_s",
	"input_tokens", "output_tokens",
	"status",
	"model_name", "model_revision", "git_commit", "hardware_id",
	"meta",
)


def load_power(path):
	"""Read the hardware timeline.

	Returns (ts, gpu_power_cols, rapl_cols, gpu_energy_cols).

	Three distinct column families, kept separate because they are different
	instruments and must never be mixed:

	  gpu<i>_w           instantaneous power (W)   -> DIAGNOSTIC
	  gpu<i>_energy_mj   cumulative energy (mJ)    -> GPU measurement instrument
	  rapl_<domain>_uj   cumulative energy (uJ)    -> CPU/DRAM instrument

	A power.csv written before the GPU counter column existed simply yields an
	empty gpu_energy_cols, which is reported as unavailable rather than being
	silently replaced by integrated power.
	"""
	ts, gpu_power_cols, rapl_cols, gpu_energy_cols = [], {}, {}, {}
	with open(path) as f:
		r = csv.reader(f)
		header = next(r)
		gpu_pw_idx = [(i, h) for i, h in enumerate(header)
		              if h.startswith("gpu") and h.endswith("_w")]
		gpu_en_idx = [(i, h) for i, h in enumerate(header)
		              if h.startswith("gpu") and h.endswith("_energy_mj")]
		rapl_idx = [(i, h) for i, h in enumerate(header) if h.startswith("rapl_")]
		for _, h in gpu_pw_idx:
			gpu_power_cols[h] = []
		for _, h in gpu_en_idx:
			gpu_energy_cols[h] = []
		for _, h in rapl_idx:
			rapl_cols[h] = []
		for row in r:
			if not row or not row[0]:
				continue
			ts.append(float(row[0]))
			for i, h in gpu_pw_idx:
				gpu_power_cols[h].append(float(row[i]) if row[i] else 0.0)
			for i, h in gpu_en_idx:
				gpu_energy_cols[h].append(float(row[i]) if row[i] else None)
			for i, h in rapl_idx:
				rapl_cols[h].append(float(row[i]) if row[i] else None)
	return ts, gpu_power_cols, rapl_cols, gpu_energy_cols


def classify_rapl(rapl_cols):
	"""Split RAPL columns into package / core / dram.

	Column names look like rapl_<domain>_<sysfs-node>_uj, e.g.
	rapl_package-0_intel-rapl:0_uj, rapl_core_intel-rapl:0:0_uj,
	rapl_dram_intel-rapl:0:1_uj.

	`core` is matched separately and deliberately excluded from the CPU total:
	it is contained within `package`. `psys` is dropped for the same reason in
	the other direction -- it is a platform-wide domain that *contains*
	package, so counting both would double-count as well.

	Match order matters:
	  * `uncore` is tested before `core`, because the string "uncore" contains
	    "core" and would otherwise be filed as the core domain, silently
	    inflating the cpu_core diagnostic. uncore is dropped: it is neither
	    additive nor the core diagnostic.
	  * `socket` is accepted as a package synonym, because some drivers label
	    the package-equivalent domain that way. Domains are classified by the
	    name the host reports; no CPU vendor is assumed. A host that exposes
	    no DRAM domain simply cannot complete the additive boundary, and
	    measured_energy_j stays null there.
	"""
	package, core, dram = {}, {}, {}
	for name, values in rapl_cols.items():
		lowered = name.lower()
		if "psys" in lowered:
			continue  # platform superset of package; would double-count
		if "dram" in lowered:
			dram[name] = values
		elif "package" in lowered or "socket" in lowered:
			package[name] = values
		elif "uncore" in lowered:
			continue  # not additive, and not the core diagnostic
		elif "core" in lowered:
			core[name] = values
	return package, core, dram


def integrate_gpu(ts, watts, t0, t1):
	"""Trapezoidal integral of power over [t0, t1] -> Joules."""
	if not ts or t1 <= t0:
		return 0.0
	lo = bisect.bisect_left(ts, t0)
	hi = bisect.bisect_right(ts, t1)
	if lo >= len(ts):
		return watts[-1] * (t1 - t0)
	if hi <= 0:
		return watts[0] * (t1 - t0)
	if hi - lo < 2:  # window shorter than sampling period
		k = min(max(lo, 0), len(ts) - 1)
		return watts[k] * (t1 - t0)
	j = 0.0
	for k in range(lo, hi - 1):
		dt = ts[k + 1] - ts[k]
		j += 0.5 * (watts[k] + watts[k + 1]) * dt
	# edge slivers
	j += watts[lo] * max(0.0, ts[lo] - t0)
	j += watts[hi - 1] * max(0.0, t1 - ts[hi - 1])
	return j


def rapl_delta(ts, uj, t0, t1):
	"""Energy delta (J) from a cumulative uJ counter over [t0, t1].

	Returns None when the counter has too few usable samples to difference:
	an absent counter is missing data, not zero energy.
	"""
	pts = [(t, v) for t, v in zip(ts, uj) if v is not None]
	if len(pts) < 2 or t1 <= t0:
		return None

	def value_at(t):
		tt = [p[0] for p in pts]
		i = bisect.bisect_left(tt, t)
		if i <= 0:
			return pts[0][1]
		if i >= len(pts):
			return pts[-1][1]
		(ta, va), (tb, vb) = pts[i - 1], pts[i]
		if vb < va:  # wraparound between samples
			vb += RAPL_MAX
		return va + (vb - va) * (t - ta) / (tb - ta)

	d = value_at(t1) - value_at(t0)
	if d < 0:
		d += RAPL_MAX
	return d / 1e6  # uJ -> J


def counter_delta(ts, values, t0, t1, scale):
	"""Energy (J) from a cumulative counter over [t0, t1].

	`scale` converts the counter's native unit to Joules (1e3 for mJ, 1e6 for
	uJ). Interpolates linearly between the bracketing samples, which is well
	behaved for a monotone accumulator at locally-constant power.

	Returns None when fewer than two usable samples exist: an absent counter is
	missing data, never zero energy.
	"""
	pts = [(t, v) for t, v in zip(ts, values) if v is not None]
	if len(pts) < 2 or t1 <= t0:
		return None
	tt = [p[0] for p in pts]

	def value_at(t):
		i = bisect.bisect_left(tt, t)
		if i <= 0:
			return pts[0][1]
		if i >= len(pts):
			return pts[-1][1]
		(ta, va), (tb, vb) = pts[i - 1], pts[i]
		if tb == ta:
			return vb
		return va + (vb - va) * (t - ta) / (tb - ta)

	return (value_at(t1) - value_at(t0)) / scale


def sum_counter_domain(ts, cols, t0, t1, scale):
	"""Total energy (J) across a counter column group, or None if unavailable."""
	if not cols:
		return None
	parts = [counter_delta(ts, values, t0, t1, scale) for values in cols.values()]
	present = [p for p in parts if p is not None]
	if not present or len(present) != len(parts):
		return None if not present else sum(present)
	return sum(present)


def sum_domain(ts, cols, t0, t1):
	"""Total energy (J) across one RAPL domain group, or None if unavailable."""
	if not cols:
		return None
	parts = [rapl_delta(ts, values, t0, t1) for values in cols.values()]
	present = [p for p in parts if p is not None]
	if not present:
		return None
	return sum(present)


def measured_total(gpu_j, cpu_package_j, dram_j):
	"""gpu + cpu_package + dram, or None if any component is unmeasured.

	`core` is never a term here. A partial sum would misrepresent the
	measurement boundary, so an unmeasured component makes the total null.
	"""
	parts = (gpu_j, cpu_package_j, dram_j)
	if any(p is None for p in parts):
		return None
	return sum(parts)


def attribute_events(events, ts, gpus, rapls):
	pkg_cols, core_cols, dram_cols = classify_rapl(rapls)
	gpu_src = {"counter": 0, "integrated": 0}
	rows = []

	for e in events:
		# schema-v1 keys, falling back to the pre-v1 names so an old
		# events.jsonl still attributes rather than crashing.
		t0 = e.get("start_timestamp", e.get("t_start"))
		t1 = e.get("end_timestamp", e.get("t_end"))
		if t0 is None or t1 is None:
			continue

		measured = e.get("gpu_energy_j")
		if measured is not None:
			gpu_j = measured
			gpu_source = "nvml_counter"
			gpu_src["counter"] += 1
		elif gpus:
			gpu_j = sum(integrate_gpu(ts, w, t0, t1) for w in gpus.values())
			gpu_source = "power_integration"
			gpu_src["integrated"] += 1
		else:
			gpu_j = None
			gpu_source = None

		cpu_package_j = sum_domain(ts, pkg_cols, t0, t1)
		cpu_core_j = sum_domain(ts, core_cols, t0, t1)  # diagnostic only
		dram_j = sum_domain(ts, dram_cols, t0, t1)

		row = {key: e.get(key) for key in CARRIED_FIELDS}
		# Legacy readers (measurement/analyze.py, audit_runs.py) still look for
		# these; keep them until every consumer is on schema v1.
		row["category"] = e.get("category")
		row["label"] = e.get("label", e.get("operation_label"))

		total = measured_total(gpu_j, cpu_package_j, dram_j)
		# Which domains of the measurement boundary this event actually has.
		# core is excluded: it is diagnostic, not part of the boundary.
		domains = [name for name, value in (
			("gpu", gpu_j), ("cpu_package", cpu_package_j), ("dram", dram_j),
		) if value is not None]

		row.update({
			"gpu_energy_j": gpu_j,
			"gpu_energy_source": gpu_source,
			"cpu_package_energy_j": cpu_package_j,
			"cpu_core_energy_j": cpu_core_j,
			"dram_energy_j": dram_j,
			"measured_energy_j": total,
			"available_energy_domains": domains,
			"measurement_complete": total is not None,
		})
		rows.append(row)

	return rows, gpu_src


def _acc(table, key, row):
	bucket = table[key]
	bucket["n"] += 1
	bucket["duration_s"] += row.get("duration_s") or 0.0
	for field in ("gpu_energy_j", "cpu_package_energy_j", "cpu_core_energy_j",
	              "dram_energy_j", "measured_energy_j"):
		value = row.get(field)
		if value is None:
			bucket[f"{field}__missing"] += 1
		else:
			bucket[field] += value


def _fmt(bucket, field):
	"""Empty cell when no event in the bucket had the measurement."""
	if bucket[f"{field}__missing"] and bucket["n"] == bucket[f"{field}__missing"]:
		return ""
	return f"{bucket[field]:.3f}"


def main():
	ap = argparse.ArgumentParser()
	ap.add_argument("--events", required=True)
	ap.add_argument("--power", required=True)
	ap.add_argument("--out", default="energy_summary.csv")
	ap.add_argument("--out-events", default="",
	                help="per-event artifact (default: events_attributed.jsonl "
	                     "beside --events)")
	args = ap.parse_args()

	out_events = args.out_events or os.path.join(
		os.path.dirname(os.path.abspath(args.events)), "events_attributed.jsonl")

	ts, gpus, rapls, gpu_energy_cols = load_power(args.power)
	events = []
	for line in open(args.events):
		line = line.strip()
		if not line:
			continue
		try:
			events.append(json.loads(line))
		except json.JSONDecodeError:
			pass  # torn line from a crash; skip, don't die

	rows, gpu_src = attribute_events(events, ts, gpus, rapls)

	with open(out_events, "w") as f:
		for row in rows:
			f.write(json.dumps(row) + "\n")

	pkg_cols, core_cols, dram_cols = classify_rapl(rapls)
	print(f"GPU energy source: {gpu_src['counter']} from NVML counter, "
	      f"{gpu_src['integrated']} integrated from power curve")
	print(f"RAPL domains found: package={len(pkg_cols)} core={len(core_cols)} "
	      f"dram={len(dram_cols)}")
	if not pkg_cols and not dram_cols:
		print("NOTE: no RAPL package/dram counters in power.csv -> "
		      "cpu_package_energy_j, dram_energy_j and measured_energy_j are "
		      "null (not zero). GPU-only measurement for this run.")
	if core_cols:
		print("NOTE: RAPL core is reported as a diagnostic only and is NOT "
		      "added to measured_energy_j (it is contained within package).")

	by_label = defaultdict(lambda: defaultdict(float))
	by_type = defaultdict(lambda: defaultdict(float))
	by_q = defaultdict(lambda: defaultdict(float))
	for row in rows:
		_acc(by_label, row.get("operation_label") or row.get("label") or "", row)
		_acc(by_type, row.get("operation_type") or "", row)
		_acc(by_q, row.get("question_id") or "", row)

	fields = ("gpu_energy_j", "cpu_package_energy_j", "cpu_core_energy_j",
	          "dram_energy_j", "measured_energy_j")
	with open(args.out, "w", newline="") as f:
		w = csv.writer(f)
		w.writerow(["scope", "key", "n_events", "duration_s"] + list(fields))
		for scope, table in (("operation_type", by_type),
		                     ("operation_label", by_label),
		                     ("question", by_q)):
			for k, v in sorted(table.items()):
				w.writerow([scope, k, int(v["n"]), f"{v['duration_s']:.3f}"]
				           + [_fmt(v, field) for field in fields])

	# --- trajectory accounting ------------------------------------------
	# LIKE-FOR-LIKE INSTRUMENTS. Every trajectory-level energy reference is a
	# cumulative hardware counter differenced across the window, matching how
	# per-event energy is produced:
	#
	#   GPU          NVML gpu<i>_energy_mj      (same register the event log reads)
	#   CPU package  RAPL package cumulative uJ
	#   DRAM         RAPL dram cumulative uJ
	#
	# Integrated sampled power is NOT the reconciliation denominator. At ~10 Hz
	# it resolves fast inference power transients poorly and errs in both
	# directions, which inflated coverage above 1 on short trajectories. It is
	# retained only as the separately named diagnostic
	# sampled_gpu_energy_estimate_j, plus mean/peak power.
	#
	# When the GPU counter is absent, gpu_energy_j is reported as None and the
	# availability is stated explicitly -- never silently back-filled with the
	# sampled estimate.
	def window_energy(t0, t1):
		if t0 is None or t1 is None:
			return None
		gpu_counter = sum_counter_domain(ts, gpu_energy_cols, t0, t1, 1e3)
		pkg_c = sum_domain(ts, pkg_cols, t0, t1)
		dram_c = sum_domain(ts, dram_cols, t0, t1)

		# diagnostics from the sampled power curve
		sampled = (sum(integrate_gpu(ts, w, t0, t1) for w in gpus.values())
		           if gpus else None)
		mean_w = peak_w = None
		if gpus:
			lo = bisect.bisect_left(ts, t0)
			hi = bisect.bisect_right(ts, t1)
			window = [sum(vals[k] for vals in gpus.values())
			          for k in range(lo, max(lo, hi))]
			if window:
				mean_w = sum(window) / len(window)
				peak_w = max(window)

		return {
			"gpu_energy_j": gpu_counter,
			"cpu_package_energy_j": pkg_c,
			"dram_energy_j": dram_c,
			"measured_energy_j": measured_total(gpu_counter, pkg_c, dram_c),
			# --- diagnostics, never the reference ---
			"gpu_energy_source": ("nvml_cumulative_counter" if gpu_counter is not None
			                      else "unavailable"),
			"sampled_gpu_energy_estimate_j": sampled,
			"mean_gpu_power_w": mean_w,
			"peak_gpu_power_w": peak_w,
		}

	if ts and not gpu_energy_cols:
		print("NOTE: power.csv has no gpu*_energy_mj column -> trajectory GPU "
		      "energy is UNAVAILABLE (this log predates the cumulative GPU "
		      "counter). Integrated sampled power is reported only as the "
		      "diagnostic sampled_gpu_energy_estimate_j and is NOT substituted.")

	traj_rows = trajectory.accumulate(rows, window_energy=window_energy if ts else None)
	traj_summary = trajectory.summarize(traj_rows)
	traj_csv = os.path.join(os.path.dirname(os.path.abspath(out_events)),
	                        "trajectory_summary.csv")
	traj_json = os.path.join(os.path.dirname(os.path.abspath(out_events)),
	                         "trajectory_summary.json")
	trajectory.write_artifacts(traj_rows, traj_summary, traj_csv, traj_json)

	print(f"\n{len(rows)} events attributed")
	print(f"  per-event artifact -> {out_events}")
	print(f"  aggregate summary  -> {args.out}")
	print(f"  trajectory summary -> {traj_csv}")

	print("\n=== energy by operation_type ===")
	for k, v in sorted(by_type.items()):
		print(f"  {k:<10} n={int(v['n']):>6}  gpu={_fmt(v, 'gpu_energy_j'):>12}  "
		      f"pkg={_fmt(v, 'cpu_package_energy_j'):>10}  "
		      f"dram={_fmt(v, 'dram_energy_j'):>10}  "
		      f"measured={_fmt(v, 'measured_energy_j'):>12}")
	print("\n=== energy by operation_label ===")
	for k, v in sorted(by_label.items()):
		print(f"  {k:<22} n={int(v['n']):>6}  gpu={_fmt(v, 'gpu_energy_j'):>12}  "
		      f"pkg={_fmt(v, 'cpu_package_energy_j'):>10}  "
		      f"dram={_fmt(v, 'dram_energy_j'):>10}")

	print()
	print(trajectory.render(traj_rows, traj_summary))


if __name__ == "__main__":
	main()
