"""Why do fully correct and not-fully-correct questions differ in workload,
per system and dataset? One compact table, one figure, from the cache only.

Reads each dataset's already-built per-question cache
(measurement/make_comparable_figures.py --dataset <name> [--rebuild-cache]);
never opens events_attributed.jsonl or predict.jsonl itself.

    python measurement/workload_by_correctness.py \
        --dataset webqsp=~/webqsp_supervisor_summary/cross_dataset/webqsp/wh \
        --dataset cwq=~/cwq_supervisor_summary/wh \
        --out ~/cwq_supervisor_summary/cross_dataset/wh

Correctness is F1-based: fully correct means F1 == 1, not fully correct
means F1 < 1 (includes partial credit, never called "incorrect").

Metrics, all already in the per-question cache:
    total GPU energy            energy_trajectory_j (whole-question, joules)
    total LLM calls              llm_calls (from visualize.workload())
    relation_rank / entity_prune / reason call counts
                                 count_llm:<label> (pivoted per-question counts)
    max traversal depth          max_traversal_depth
    fallback rate                fallback (bool)
    max iteration                max_iteration -- NOT a backtrack count. No
                                 event carries a forward/backtrack/stop/filter
                                 decision field; that only appears as text in
                                 run.log, which this script does not parse (a
                                 log-parsing step is exactly the "new
                                 complicated pipeline" this stays out of).
                                 max_iteration is reported instead as a proxy:
                                 CoR's control-loop iteration count advances on
                                 every processed state, forward or backtrack,
                                 while PoG/ToG's iteration count tracks depth
                                 almost exactly (established separately in
                                 this project's analysis) -- so for CoR only,
                                 iteration running ahead of depth is a real
                                 signal of extra search churn; for PoG/ToG it
                                 is redundant with depth and not informative.

Writes only into --out: fig8_workload_by_correctness_<dataset>.{png,pdf},
fig8_workload_by_correctness_summary.csv.
"""

import argparse
import os
import statistics
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)
sys.path.insert(0, HERE)

import make_comparable_figures as mcf  # noqa: E402
from agent_energy_profiler import visualize as v  # noqa: E402

DATASET_ORDER = ("webqsp", "cwq")
DATASET_LABELS = {"webqsp": "WebQSP", "cwq": "CWQ"}
SYSTEM_ORDER = ("PoG", "ToG", "CoR")
FULLY, NOT_FULLY = "fully_correct", "not_fully_correct"
BIN_ORDER = (FULLY, NOT_FULLY)
BIN_LABELS = {FULLY: "Fully correct\n(F1 = 1)", NOT_FULLY: "Not fully correct\n(F1 < 1)"}
BIN_COLOURS_SLOT = {FULLY: 0, NOT_FULLY: 1}
STEM = "fig8_workload_by_correctness"

#: (key, label, extractor, exclude_none). exclude_none=False treats a missing
#: value as 0 -- correct for a call count (the operation simply never fired
#: for that question, e.g. CoR has no entity_prune at all) but wrong for a
#: measurement that can be genuinely absent (energy, depth, iteration).
METRICS = (
	("energy_j", "Total GPU energy (J)",
	 lambda r: r.get("energy_trajectory_j"), True),
	("llm_calls", "Total LLM calls", lambda r: r.get("llm_calls"), False),
	("relation_rank_calls", "relation_rank calls",
	 lambda r: r.get("count_llm:relation_rank"), False),
	("entity_prune_calls", "entity_prune calls",
	 lambda r: r.get("count_llm:entity_prune"), False),
	("reason_calls", "reason calls", lambda r: r.get("count_llm:reason"), False),
	("max_traversal_depth", "Max traversal depth",
	 lambda r: r.get("max_traversal_depth"), True),
	("max_iteration", "Max iteration (search-churn proxy, not a backtrack count)",
	 lambda r: r.get("max_iteration"), True),
	("fallback_rate", "Fallback rate", lambda r: 1.0 if r.get("fallback") else 0.0, False),
)
#: The one metric drawn in the figure grid: LLM calls is comparable across all
#: three systems (unlike relation_rank/entity_prune, which are not symmetric
#: -- CoR has no entity_prune at all) and is the direct driver of GPU energy.
FIGURE_METRIC = "llm_calls"
SUMMARY_FIELDS = ("dataset", "paradigm", "metric", "n_full", "n_not_full",
                  "mean_full", "mean_not_full", "mean_diff",
                  "median_full", "median_not_full", "median_diff")


def _bin_of(f1):
	return FULLY if f1 == 1.0 else NOT_FULLY


def _load_dataset(name, out_dir):
	cache_dir = os.path.join(os.path.expanduser(out_dir), mcf.CACHE_DIRNAME)
	meta = mcf._cache_manifest(cache_dir)
	if meta is None:
		raise SystemExit(f"{name}: no cache at {cache_dir}. Build it first, e.g.\n"
		                 f"  python measurement/make_comparable_figures.py --dataset {name} "
		                 f"--rebuild-cache --out {out_dir} --run ...")
	if meta.get("cache_version", 0) < 3:
		raise SystemExit(f"{name}: cache at {cache_dir} predates per-question F1 "
		                 f"(cache_version {meta.get('cache_version')} < 3). Rebuild it:\n"
		                 f"  python measurement/make_comparable_figures.py --dataset {name} "
		                 f"--rebuild-cache --out {out_dir} --run ...")
	rows = mcf._read_question_table(os.path.join(cache_dir, mcf.CACHE_FILENAME))
	return rows, meta["systems"]


def _grouped_rows(rows, systems):
	"""{(system, bin): [row, ...]} over questions with an F1 value."""
	label_to_system = {info["label"]: name for name, info in systems.items()}
	groups = {(name, b): [] for name in systems for b in BIN_ORDER}
	for r in rows:
		name = label_to_system.get(r.get("run"))
		f1 = r.get("f1")
		if name is None or f1 is None:
			continue
		groups[(name, _bin_of(f1))].append(r)
	return groups


def _values(rows, extractor, exclude_none):
	vals = [extractor(r) for r in rows]
	if exclude_none:
		vals = [v for v in vals if v is not None]
	else:
		vals = [(v or 0) for v in vals]
	return vals


def _summary_rows(dataset, groups, order):
	rows = []
	for name in order:
		full = groups[(name, FULLY)]
		not_full = groups[(name, NOT_FULLY)]
		for key, label, extractor, exclude_none in METRICS:
			vf = _values(full, extractor, exclude_none)
			vn = _values(not_full, extractor, exclude_none)
			mean_f = statistics.mean(vf) if vf else None
			mean_n = statistics.mean(vn) if vn else None
			med_f = statistics.median(vf) if vf else None
			med_n = statistics.median(vn) if vn else None
			rows.append({
				"dataset": DATASET_LABELS[dataset], "paradigm": name, "metric": label,
				"n_full": len(vf), "n_not_full": len(vn),
				"mean_full": mean_f, "mean_not_full": mean_n,
				"mean_diff": (mean_n - mean_f) if None not in (mean_f, mean_n) else None,
				"median_full": med_f, "median_not_full": med_n,
				"median_diff": (med_n - med_f) if None not in (med_f, med_n) else None,
			})
	return rows


def _panel(plt, ax, groups, name):
	key, label, extractor, exclude_none = next(m for m in METRICS if m[0] == FIGURE_METRIC)
	data = [_values(groups[(name, b)], extractor, exclude_none) for b in BIN_ORDER]
	positions = [1, 2]
	box = ax.boxplot(data, positions=positions, widths=0.55, patch_artist=True,
	                 showmeans=True, showfliers=True,
	                 medianprops={"color": v.INK, "linewidth": 1.4},
	                 meanprops={"marker": "D", "markerfacecolor": v.INK,
	                            "markeredgecolor": v.SURFACE, "markersize": 5},
	                 flierprops={"marker": "o", "markersize": 4, "alpha": 0.6,
	                             "markeredgewidth": 0},
	                 whiskerprops={"color": v.INK2}, capprops={"color": v.INK2})
	for patch, b in zip(box["boxes"], BIN_ORDER):
		patch.set_facecolor(v.PALETTE[BIN_COLOURS_SLOT[b]])
		patch.set_alpha(0.55)
		patch.set_edgecolor(v.INK2)
	for flier, b in zip(box["fliers"], BIN_ORDER):
		flier.set_markerfacecolor(v.PALETTE[BIN_COLOURS_SLOT[b]])
	ticks = []
	for b, vals in zip(BIN_ORDER, data):
		if vals:
			m, md = statistics.mean(vals), statistics.median(vals)
			ticks.append(f"{BIN_LABELS[b]}\nn = {len(vals):,}\nmean {m:,.1f}\nmed {md:,.1f}")
		else:
			ticks.append(f"{BIN_LABELS[b]}\nn = 0")
	ax.set_xticks(positions, ticks, fontsize=6.3)
	ax.set_xlim(0.4, 2.6)
	ax.grid(axis="x", visible=False)
	ax.set_title(name, fontsize=9.5, fontweight="bold")


def _panel_v2(plt, ax, groups, name):
	"""Cleanup pass over _panel for review as fig8_workload_by_correctness_v2:
	log y-axis so outliers stop compressing the boxes, and the tick labels
	keep only n (mean/median stay available in fig8_workload_by_correctness_
	summary.csv, not duplicated as on-figure text)."""
	key, label, extractor, exclude_none = next(m for m in METRICS if m[0] == FIGURE_METRIC)
	data = [_values(groups[(name, b)], extractor, exclude_none) for b in BIN_ORDER]
	positions = [1, 2]
	box = ax.boxplot(data, positions=positions, widths=0.55, patch_artist=True,
	                 showmeans=True, showfliers=True,
	                 medianprops={"color": v.INK, "linewidth": 1.4},
	                 meanprops={"marker": "D", "markerfacecolor": v.INK,
	                            "markeredgecolor": v.SURFACE, "markersize": 5},
	                 flierprops={"marker": "o", "markersize": 4, "alpha": 0.6,
	                             "markeredgewidth": 0},
	                 whiskerprops={"color": v.INK2}, capprops={"color": v.INK2})
	for patch, b in zip(box["boxes"], BIN_ORDER):
		patch.set_facecolor(v.PALETTE[BIN_COLOURS_SLOT[b]])
		patch.set_alpha(0.55)
		patch.set_edgecolor(v.INK2)
	for flier, b in zip(box["fliers"], BIN_ORDER):
		flier.set_markerfacecolor(v.PALETTE[BIN_COLOURS_SLOT[b]])
	ax.set_yscale("log")
	ticks = []
	for b, vals in zip(BIN_ORDER, data):
		ticks.append(f"{BIN_LABELS[b]}\nn = {len(vals):,}")
	ax.set_xticks(positions, ticks, fontsize=8.5)
	ax.set_xlim(0.4, 2.6)
	ax.grid(axis="x", visible=False)
	ax.set_title(name, fontsize=11, fontweight="bold")


def main(argv=None):
	ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
	ap.add_argument("--dataset", action="append", required=True, metavar="NAME=OUTDIR")
	ap.add_argument("--out", required=True)
	args = ap.parse_args(argv)

	out_dirs = mcf.parse_pairs(args.dataset, {})
	missing = [d for d in DATASET_ORDER if d not in out_dirs]
	if missing:
		raise SystemExit(f"--dataset required for {missing}")

	per_dataset, orders = {}, {}
	for ds in DATASET_ORDER:
		rows, systems = _load_dataset(ds, out_dirs[ds])
		order = [s for s in SYSTEM_ORDER if s in systems]
		per_dataset[ds] = _grouped_rows(rows, systems)
		orders[ds] = order

	summary = []
	for ds in DATASET_ORDER:
		summary.extend(_summary_rows(ds, per_dataset[ds], orders[ds]))

	out = os.path.abspath(os.path.expanduser(args.out))
	os.makedirs(out, exist_ok=True)

	from matplotlib.lines import Line2D
	legend_handles = [Line2D([], [], color=v.INK, linewidth=1.4, label="median (box line)"),
	                  Line2D([], [], marker="D", linestyle="", markerfacecolor=v.INK,
	                         markeredgecolor=v.SURFACE, label="mean")]

	written = []
	plt = v._pyplot()
	with plt.rc_context(v.STYLE):
		fig, grid = plt.subplots(len(DATASET_ORDER), len(SYSTEM_ORDER), figsize=(11.4, 7.8),
		                         sharey=True)
		for row_i, ds in enumerate(DATASET_ORDER):
			for col_i, name in enumerate(SYSTEM_ORDER):
				ax = grid[row_i, col_i]
				if name in orders[ds]:
					_panel(plt, ax, per_dataset[ds], name)
				if col_i == 0:
					ax.set_ylabel(f"{DATASET_LABELS[ds]}\nTotal LLM calls", fontsize=9)
		fig.suptitle("Total LLM Calls: Fully Correct vs. Not Fully Correct",
		            fontsize=12, fontweight="bold")
		fig.legend(handles=legend_handles, loc="lower center", ncol=2, frameon=False,
		          fontsize=8, bbox_to_anchor=(0.5, -0.01))
		fig.tight_layout(rect=(0, 0.03, 1, 0.94))
		written += v.save_figure(fig, out, STEM)
		plt.close(fig)

	# ---- v2 (review only, does not touch fig8_workload_by_correctness.pdf/.png):
	# log y-axis, trimmed tick text, shorter non-claiming title, PDF only
	# (this figure is moving to supplementary, so it drops the PNG duplicate
	# per CLAUDE.md's "final thesis figures: PDF only" hygiene rule).
	with plt.rc_context(v.STYLE):
		fig, grid = plt.subplots(len(DATASET_ORDER), len(SYSTEM_ORDER), figsize=(11.4, 7.8),
		                         sharey=True)
		for row_i, ds in enumerate(DATASET_ORDER):
			for col_i, name in enumerate(SYSTEM_ORDER):
				ax = grid[row_i, col_i]
				if name in orders[ds]:
					_panel_v2(plt, ax, per_dataset[ds], name)
				if col_i == 0:
					ax.set_ylabel(f"{DATASET_LABELS[ds]}\nTotal LLM calls (log)", fontsize=9)
		fig.suptitle("LLM calls by answer correctness", fontsize=12, fontweight="bold")
		fig.legend(handles=legend_handles, loc="lower center", ncol=2, frameon=False,
		          fontsize=8, bbox_to_anchor=(0.5, -0.01))
		fig.tight_layout(rect=(0, 0.03, 1, 0.94))
		written += v.save_figure(fig, out, f"{STEM}_v2", png=False)
		plt.close(fig)

	written.append(v.write_rows(os.path.join(out, f"{STEM}_summary.csv"), summary,
	                            SUMMARY_FIELDS))
	for path in written:
		print(f"  -> {path}")
	return 0


if __name__ == "__main__":
	sys.exit(main())
