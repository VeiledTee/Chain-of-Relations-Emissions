"""Join energy events to the power timeseries -> per-event/label/category/question energy.

Inputs:
  events.jsonl  (written by chain_of_relations.energy_events during the run)
  power.csv     (written by measurement/power_logger.py on the same host clock)

Method:
  GPU:  integrate sampled GPU power (trapezoid over samples) inside each
        event's [t_start, t_end]; events shorter than the sampling period get
        nearest-sample power x duration (estimate, flagged in EMISSIONS.md).
  RAPL: cumulative counters -> energy delta over the window, interpolated
        between samples; handles counter wraparound.

Honest units:
  Per-QUESTION energy is the robust primitive (long windows, many samples).
  Per-EVENT energy is a timestamp-attributed estimate; sub-100ms SPARQL events
  are order-of-magnitude, not precision, figures. Idle machine assumed:
  concurrent activity is attributed to whatever event overlaps it.

Usage:
  python measurement/attribute.py --events events.jsonl --power power.csv \
      --out summary.csv
"""

import argparse
import bisect
import csv
import json
from collections import defaultdict

RAPL_MAX = 2**32 * 1e0  # wraparound guard; actual max_energy_range varies


def load_power(path):
	ts, gpu_cols, rapl_cols = [], {}, {}
	with open(path) as f:
		r = csv.reader(f)
		header = next(r)
		gpu_idx = [(i, h) for i, h in enumerate(header) if h.startswith("gpu")]
		rapl_idx = [(i, h) for i, h in enumerate(header) if h.startswith("rapl_")]
		for h in gpu_idx:
			gpu_cols[h[1]] = []
		for h in rapl_idx:
			rapl_cols[h[1]] = []
		for row in r:
			if not row or not row[0]:
				continue
			ts.append(float(row[0]))
			for i, h in gpu_idx:
				gpu_cols[h].append(float(row[i]) if row[i] else 0.0)
			for i, h in rapl_idx:
				rapl_cols[h].append(float(row[i]) if row[i] else None)
	return ts, gpu_cols, rapl_cols


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
	"""Energy delta (J) from a cumulative uJ counter over [t0, t1]."""
	pts = [(t, v) for t, v in zip(ts, uj) if v is not None]
	if len(pts) < 2 or t1 <= t0:
		return 0.0

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


def main():
	ap = argparse.ArgumentParser()
	ap.add_argument("--events", required=True)
	ap.add_argument("--power", required=True)
	ap.add_argument("--out", default="energy_summary.csv")
	args = ap.parse_args()

	ts, gpus, rapls = load_power(args.power)
	events = [json.loads(l) for l in open(args.events) if l.strip()]

	rows = []
	for e in events:
		t0, t1 = e["t_start"], e["t_end"]
		gpu_j = sum(integrate_gpu(ts, w, t0, t1) for w in gpus.values())
		cpu_j = sum(rapl_delta(ts, v, t0, t1) for name, v in rapls.items()
		            if "package" in name or "core" in name)
		dram_j = sum(rapl_delta(ts, v, t0, t1) for name, v in rapls.items()
		             if "dram" in name)
		rows.append({
			"question_id": e.get("question_id", ""),
			"category": e["category"],
			"label": e["label"],
			"duration_s": e["duration_s"],
			"gpu_j": gpu_j, "cpu_j": cpu_j, "dram_j": dram_j,
		})

	# aggregations
	def agg(keyfn):
		out = defaultdict(lambda: defaultdict(float))
		for r in rows:
			k = keyfn(r)
			out[k]["n"] += 1
			for c in ("duration_s", "gpu_j", "cpu_j", "dram_j"):
				out[k][c] += r[c]
		return out

	by_label = agg(lambda r: r["label"])
	by_cat = agg(lambda r: r["category"])
	by_q = agg(lambda r: r["question_id"])

	with open(args.out, "w", newline="") as f:
		w = csv.writer(f)
		w.writerow(["scope", "key", "n_events", "duration_s", "gpu_j", "cpu_j", "dram_j"])
		for scope, table in (("category", by_cat), ("label", by_label), ("question", by_q)):
			for k, v in sorted(table.items()):
				w.writerow([scope, k, int(v["n"]),
				            f"{v['duration_s']:.3f}", f"{v['gpu_j']:.2f}",
				            f"{v['cpu_j']:.2f}", f"{v['dram_j']:.2f}"])

	print(f"{len(rows)} events attributed -> {args.out}")
	print("\n=== energy by category (J) ===")
	for k, v in sorted(by_cat.items()):
		print(f"  {k:<10} n={int(v['n']):>6}  gpu={v['gpu_j']:>10.1f}  "
		      f"cpu={v['cpu_j']:>9.1f}  dram={v['dram_j']:>8.1f}  "
		      f"({v['duration_s']:.1f}s)")
	print("\n=== energy by label (J) ===")
	for k, v in sorted(by_label.items()):
		print(f"  {k:<22} n={int(v['n']):>6}  gpu={v['gpu_j']:>10.1f}  "
		      f"cpu={v['cpu_j']:>9.1f}  dram={v['dram_j']:>8.1f}")


if __name__ == "__main__":
	main()
