"""Compare per-question energy across systems on the intersection of question IDs.

Reads the `question`-scope rows of each run's energy_summary.csv (schema:
scope,key,n_events,duration_s,gpu_j,cpu_j,dram_j) and reports, per system:
coverage, per-question GPU energy (mean/median/p90), events per question,
duration per question, and pairwise ratios on the shared-question set.

Usage:
  python compare_runs.py \
      --run sgrag=path/to/sgr_webqsp_qwen7b/energy_summary.csv \
      --run cor=path/to/cor_webqsp_qwen7b_full/energy_summary.csv \
      [--acc sgrag=83.35 --acc cor=71.2]        # optional Hit/accuracy, any metric
      [--csv out.csv]                            # per-question joined rows
"""

import argparse
import csv
import statistics as st


def load_summary(path):
	q = {}
	for r in csv.DictReader(open(path)):
		if r["scope"] != "question" or not r["key"]:
			continue
		q[r["key"]] = {
			"n_events": int(r["n_events"]),
			"duration_s": float(r["duration_s"]),
			"gpu_j": float(r["gpu_j"]),
			"cpu_j": float(r["cpu_j"]),
			"dram_j": float(r["dram_j"]),
		}
	return q


def describe(vals):
	v = sorted(vals)
	return (st.mean(v), st.median(v), v[int(0.9 * (len(v) - 1))])


def main():
	ap = argparse.ArgumentParser()
	ap.add_argument("--run", action="append", required=True,
	                metavar="NAME=PATH", help="system name = energy_summary.csv")
	ap.add_argument("--acc", action="append", default=[],
	                metavar="NAME=VALUE", help="optional accuracy per system")
	ap.add_argument("--csv", default=None, help="write joined per-question rows")
	args = ap.parse_args()

	runs = {}
	for spec in args.run:
		name, path = spec.split("=", 1)
		runs[name] = load_summary(path)
	acc = dict(s.split("=", 1) for s in args.acc)

	ids = None
	for name, q in runs.items():
		print(f"{name}: {len(q)} questions with energy data")
		ids = set(q) if ids is None else ids & set(q)
	print(f"intersection: {len(ids)} questions\n")
	ids = sorted(ids)

	hdr = (f"{'system':<10} {'acc':>6} {'J/q mean':>9} {'J/q med':>8} "
	       f"{'J/q p90':>8} {'Wh/q':>7} {'ev/q':>7} {'s/q':>6}")
	print(hdr)
	print("-" * len(hdr))
	stats = {}
	for name, q in runs.items():
		gj = [q[i]["gpu_j"] for i in ids]
		ev = [q[i]["n_events"] for i in ids]
		du = [q[i]["duration_s"] for i in ids]
		m, med, p90 = describe(gj)
		stats[name] = m
		print(f"{name:<10} {acc.get(name,'-'):>6} {m:>9.0f} {med:>8.0f} "
		      f"{p90:>8.0f} {m/3600:>7.3f} {st.mean(ev):>7.2f} {st.mean(du):>6.2f}")

	names = list(runs)
	if len(names) >= 2:
		print("\npairwise mean-energy ratios (row / col), intersection only:")
		for a in names:
			for b in names:
				if a != b:
					print(f"  {a}/{b} = {stats[a]/stats[b]:.2f}x")

	if args.csv:
		with open(args.csv, "w", newline="") as f:
			w = csv.writer(f)
			w.writerow(["question_id"] + [f"{n}_{c}" for n in names
			           for c in ("gpu_j", "n_events", "duration_s")])
			for i in ids:
				w.writerow([i] + [runs[n][i][c] for n in names
				           for c in ("gpu_j", "n_events", "duration_s")])
		print(f"\nper-question rows -> {args.csv}")


if __name__ == "__main__":
	main()
