"""Build a datasets x systems results grid from energy summaries + accuracies.

Each --entry is  dataset:system:accuracy:path_to_energy_summary.csv
Energy = mean GPU J/question over that run's question rows (no intersection
logic here; use compare_runs.py for pairwise intersection stats).

Usage:
  python results_grid.py \
    --entry webqsp:query-only:XX.X:runs/io_webqsp_qwen7b/energy_summary.csv \
    --entry webqsp:sgrag:83.35:...  --entry webqsp:cor:74.88:... \
    --entry cwq:query-only:XX.X:... --entry cwq:sgrag:50.67:... \
    --entry cwq:cor:37.36:... \
    [--latex grid.tex]
"""

import argparse
import csv
import statistics as st
from collections import defaultdict


def mean_j_per_q(path):
	vals = [float(r["gpu_j"]) for r in csv.DictReader(open(path))
	        if r["scope"] == "question" and r["key"]]
	return st.mean(vals), len(vals)


def main():
	ap = argparse.ArgumentParser()
	ap.add_argument("--entry", action="append", required=True,
	                metavar="DATASET:SYSTEM:ACC:PATH")
	ap.add_argument("--latex", default=None)
	args = ap.parse_args()

	grid = defaultdict(dict)   # dataset -> system -> (acc, jq, n)
	sys_order, ds_order = [], []
	for e in args.entry:
		ds, system, acc, path = e.split(":", 3)
		jq, n = mean_j_per_q(path)
		grid[ds][system] = (float(acc), jq, n)
		if system not in sys_order:
			sys_order.append(system)
		if ds not in ds_order:
			ds_order.append(ds)

	for ds in ds_order:
		base = min(v[1] for v in grid[ds].values())
		print(f"\n=== {ds} ===")
		hdr = (f"{'system':<12} {'Hit@1':>7} {'J/q':>8} {'Wh/q':>7} "
		       f"{'x cheapest':>10} {'n':>6}")
		print(hdr)
		print("-" * len(hdr))
		for s in sys_order:
			if s not in grid[ds]:
				continue
			acc, jq, n = grid[ds][s]
			print(f"{s:<12} {acc:>7.2f} {jq:>8.0f} {jq/3600:>7.3f} "
			      f"{jq/base:>9.2f}x {n:>6}")

	if args.latex:
		with open(args.latex, "w") as f:
			f.write("\\begin{tabular}{ll" + "rrr" + "}\n\\toprule\n")
			f.write("Dataset & System & Hit@1 & Wh/q & $\\times$cheapest "
			        "\\\\\n\\midrule\n")
			for ds in ds_order:
				base = min(v[1] for v in grid[ds].values())
				for s in sys_order:
					if s not in grid[ds]:
						continue
					acc, jq, _ = grid[ds][s]
					f.write(f"{ds} & {s} & {acc:.2f} & {jq/3600:.3f} & "
					        f"{jq/base:.2f}$\\times$ \\\\\n")
				f.write("\\midrule\n")
			f.write("\\bottomrule\n\\end{tabular}\n")
		print(f"\nLaTeX -> {args.latex}")


if __name__ == "__main__":
	main()
