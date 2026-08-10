"""Pareto scatter: accuracy (y) vs mean GPU Wh/question (x, log scale),
one panel per dataset, Pareto frontier highlighted.

Same --entry format as results_grid.py:
  dataset:system:accuracy:path_to_energy_summary.csv

Usage:
  python pareto_plot.py --entry ... --entry ... [-o pareto.png]
"""

import argparse
import csv
import statistics as st
from collections import defaultdict

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def mean_whq(path):
	vals = [float(r["gpu_j"]) for r in csv.DictReader(open(path))
	        if r["scope"] == "question" and r["key"]]
	return st.mean(vals) / 3600.0


def pareto_front(points):
	"""points: list of (wh, acc). Frontier = not dominated (lower wh, higher acc)."""
	front = []
	for wh, acc in points:
		if not any(w2 <= wh and a2 >= acc and (w2, a2) != (wh, acc)
		           for w2, a2 in points):
			front.append((wh, acc))
	return sorted(front)


def main():
	ap = argparse.ArgumentParser()
	ap.add_argument("--entry", action="append", required=True)
	ap.add_argument("-o", "--out", default="pareto.png")
	args = ap.parse_args()

	data = defaultdict(list)   # dataset -> [(system, acc, whq)]
	for e in args.entry:
		ds, system, acc, path = e.split(":", 3)
		data[ds].append((system, float(acc), mean_whq(path)))

	n = len(data)
	fig, axes = plt.subplots(1, n, figsize=(5.2 * n, 4.2), squeeze=False)
	for ax, (ds, rows) in zip(axes[0], data.items()):
		pts = [(wh, acc) for _, acc, wh in rows]
		front = pareto_front(pts)
		if len(front) > 1:
			ax.plot(*zip(*front), "--", color="0.6", lw=1.2, zorder=1,
			        label="Pareto frontier")
		for system, acc, wh in rows:
			on_front = (wh, acc) in front
			ax.scatter(wh, acc, s=90 if on_front else 60,
			           zorder=3, edgecolors="black" if on_front else "none",
			           linewidths=1.2)
			ax.annotate(system, (wh, acc), textcoords="offset points",
			            xytext=(7, 4), fontsize=9)
		ax.set_xscale("log")
		ax.set_xlabel("GPU energy (Wh / question, log)")
		ax.set_ylabel("Hit@1 (%)")
		ax.set_title(ds)
		ax.grid(True, which="both", alpha=0.25)
	fig.suptitle("Effectiveness vs. energy, Qwen2.5-7B-Instruct / vLLM / RTX 4090",
	             fontsize=11)
	fig.tight_layout(rect=[0, 0, 1, 0.94])
	fig.savefig(args.out, dpi=200)
	print(f"wrote {args.out}")


if __name__ == "__main__":
	main()
