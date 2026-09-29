"""Candidate C2 figures: two alternative visualizations of "where energy goes
inside each agentic system", built on the same per-question cache as
measurement/thesis_figures.py -- no experiment is rerun, no cache is rebuilt,
no existing figure is modified.

    python measurement/candidate_c2_figures.py \
        --out-figures results/figures/cross_dataset

Defaults to the same two cache directories thesis_figures.py documents
(WebQSP under the cwq_supervisor_summary/cross_dataset tree, CWQ under
cwq_supervisor_summary/wh); pass --dataset NAME=CACHE_OUTDIR to override.

Candidate A (candidate_c2_energy_by_operation.pdf): 1x2 (WebQSP, CWQ), one
stacked bar per system (ToG, PoG, CoR as laid out), stack = mean ATTRIBUTED
energy per question by canonical operation label (same taxonomy order/colours
as thesis_figures.py's Figure 3). Stacked attributed-operation bars only (no
trajectory-energy reference); the axis/caption say "Attributed energy", not
"Total energy".

Candidate B (candidate_c2_operation_frequency_vs_cost.pdf): 2x3 grid (rows =
dataset, columns = ToG/PoG/CoR as laid out), one point per canonical operation
that system actually emits: x = mean calls/question, y = mean attributed
energy per execution (mWh -- the one unit that keeps every operation's value
readable across five orders of magnitude), point colour = operation (same
mapping as Candidate A), point area ~ mean attributed Wh/question. Both axes
are log-scaled (labelled as such) and share identical limits across all six
panels; no operation is dropped and no outlier is trimmed.

Candidate C (operation cost components) is not produced: Candidate B's
per-panel point count (7-12 operations) stays legible at this size, so the
"only if Figure B is genuinely unreadable" condition is not met.

Prints validation and top-3-by-metric summaries to stdout; writes no new CSV.
"""

import argparse
import csv
import math
import os
import statistics
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)
sys.path.insert(0, HERE)

import thesis_figures as tf  # noqa: E402
from agent_energy_profiler import visualize as v  # noqa: E402

#: System order for these candidate figures: matches thesis_figures.SYSTEM_ORDER
#: (PoG, ToG, CoR), the order/colour convention already used by every other
#: canonical figure in this project (fig1-fig8, fig_effectiveness_energy_
#: tradeoffs.pdf, fig_operation_energy_shares_main.pdf).
CANDIDATE_SYSTEM_ORDER = ("PoG", "ToG", "CoR")

DEFAULT_CACHE_DIRS = {
	"webqsp": "~/cwq_supervisor_summary/cross_dataset/webqsp/wh",
	"cwq": "~/cwq_supervisor_summary/wh",
}

MWH_PER_WH = 1000.0  # Wh -> mWh (J -> Wh is visualize.j_to_wh)


def _op_stats(qrows):
	"""Per (dataset, system) operation stats from the cached per-question rows:
	{label: {"energy_j": total, "count": total, "wh_per_q": mean, "calls_per_q":
	mean, "mwh_per_exec": mean}} plus n and the mean/median trajectory Wh/question
	(same coverage numbers thesis_figures._op_energy_means already reports)."""
	n = len(qrows)
	energy_sum, count_sum = {}, {}
	for r in qrows:
		for k, val in r.items():
			if val is None:
				continue
			if k.startswith("energy_") and k != "energy_trajectory_j":
				energy_sum[k[len("energy_"):]] = energy_sum.get(k[len("energy_"):], 0.0) + val
			elif k.startswith("count_"):
				count_sum[k[len("count_"):]] = count_sum.get(k[len("count_"):], 0) + int(val)

	labels = set(energy_sum) | set(count_sum)
	stats = {}
	for label in labels:
		e_j = energy_sum.get(label, 0.0)
		c = count_sum.get(label, 0)
		stats[label] = {
			"energy_j": e_j,
			"count": c,
			"wh_per_q": v.j_to_wh(e_j) / n if n else None,
			"calls_per_q": c / n if n else None,
			"mwh_per_exec": v.j_to_wh(e_j) * MWH_PER_WH / c if c else None,
		}

	traj = [r["energy_trajectory_j"] for r in qrows if r.get("energy_trajectory_j") is not None]
	n_attr = sum(1 for r in qrows
	            if any(k.startswith("energy_") and k != "energy_trajectory_j" and v_ is not None
	                  for k, v_ in r.items()))
	coverage = {
		"n": n,
		"n_trajectory": len(traj),
		"n_operation_attributed": n_attr,
		"n_categories": sum(1 for label in labels if count_sum.get(label, 0) > 0),
		"n_events": sum(count_sum.values()),
		"total_attributed_wh": v.j_to_wh(sum(s["energy_j"] for s in stats.values())),
		"mean_attributed_wh": (v.j_to_wh(sum(s["energy_j"] for s in stats.values())) / n) if n else None,
		"mean_trajectory_wh": v.j_to_wh(statistics.mean(traj)) if traj else None,
		"median_trajectory_wh": v.j_to_wh(statistics.median(traj)) if traj else None,
	}
	return stats, coverage


def _validate_and_load(dataset_order, cache_dirs):
	loaded, panel_stats, panel_coverage = {}, {}, {}
	print("=" * 78)
	print("DATA VALIDATION")
	print("=" * 78)
	for ds in dataset_order:
		order, qrows_by_system = tf._load_dataset(ds, cache_dirs[ds])
		order = [s for s in CANDIDATE_SYSTEM_ORDER if s in order]
		loaded[ds] = (order, qrows_by_system)
		for name in order:
			qrows = qrows_by_system[name]
			ids = [r["question_id"] for r in qrows]
			dup = len(ids) - len(set(ids))
			stats, coverage = _op_stats(qrows)
			panel_stats[(ds, name)] = stats
			panel_coverage[(ds, name)] = coverage
			frac = (coverage["n_operation_attributed"] / coverage["n"]) if coverage["n"] else 0.0
			traj_frac = (coverage["n_trajectory"] / coverage["n"]) if coverage["n"] else 0.0
			print(f"{tf.DATASET_LABELS[ds]:7s} {name:4s}  n={coverage['n']:,}  "
			     f"duplicate_ids={dup}  "
			     f"trajectory_coverage={coverage['n_trajectory']:,}/{coverage['n']:,} "
			     f"({traj_frac:.1%})  "
			     f"operation_attribution_coverage={coverage['n_operation_attributed']:,}/"
			     f"{coverage['n']:,} ({frac:.1%})  "
			     f"n_attributed_events={coverage['n_events']:,}  "
			     f"n_categories={coverage['n_categories']}  "
			     f"total_attributed_wh={coverage['total_attributed_wh']:.2f}  "
			     f"mean_attributed_wh/q={coverage['mean_attributed_wh']:.5f}  "
			     f"mean_trajectory_wh/q="
			     + (f"{coverage['mean_trajectory_wh']:.5f}" if coverage['mean_trajectory_wh']
			        is not None else "n/a"))
	expected = {"webqsp": 1639, "cwq": 3520}
	for ds in dataset_order:
		for name in loaded[ds][0]:
			n = panel_coverage[(ds, name)]["n"]
			if ds in expected and n != expected[ds]:
				print(f"  ** NOTE: {tf.DATASET_LABELS[ds]} {name} has n={n:,}, "
				     f"expected {expected[ds]:,} **")
	return loaded, panel_stats, panel_coverage


# ---------------------------------------------------------------------------
# Candidate A: absolute attributed energy by operation, stacked, per system
# ---------------------------------------------------------------------------

def draw_candidate_a(plt, dataset_order, loaded_order, panel_stats, panel_coverage,
                     label_order, colour_of):
	fig, axes = plt.subplots(1, len(dataset_order), figsize=(6.2 * len(dataset_order), 5.8))
	axes = [axes] if len(dataset_order) == 1 else list(axes)
	handles_by_label = {}
	for ax, ds in zip(axes, dataset_order):
		systems = loaded_order[ds]
		x = list(range(len(systems)))
		bottom = [0.0] * len(systems)
		for label in label_order:
			vals = [panel_stats[(ds, s)].get(label, {}).get("wh_per_q", 0.0) or 0.0
			       for s in systems]
			if all(val == 0.0 for val in vals):
				continue
			bars = ax.bar(x, vals, bottom=bottom, color=colour_of[label], width=0.6,
			             edgecolor=v.SURFACE, linewidth=0.8, label=tf._display_label(label))
			handles_by_label.setdefault(label, bars)
			bottom = [b + val for b, val in zip(bottom, vals)]
		# Stacked attributed-operation bars only. Attribution coverage is
		# reported once, in the caption, not annotated per bar.
		ax.set_xticks(x)
		ax.set_xticklabels(systems, fontsize=12.5, fontweight="bold")
		ax.set_title(tf.DATASET_LABELS[ds], fontsize=14, fontweight="bold")
		ax.tick_params(axis="y", labelsize=10)
		if ax is axes[0]:
			ax.set_ylabel("Mean attributed energy per question (Wh)", fontsize=11.5)
	handles = list(handles_by_label.values())
	labels = [tf._display_label(lbl) for lbl in handles_by_label]
	ncol = min(4, len(labels))
	handles, labels = tf._legend_row_major(handles, labels, ncol)
	fig.legend(handles, labels, loc="lower center", ncol=ncol, frameon=False, fontsize=9,
	          bbox_to_anchor=(0.5, -0.12))
	fig.suptitle("Attributed energy by operation", fontsize=13, fontweight="bold", y=1.02)
	fig.tight_layout(rect=(0, 0.14, 1, 0.92))
	return fig


# ---------------------------------------------------------------------------
# Candidate B: operation frequency vs. per-execution cost
# ---------------------------------------------------------------------------

#: Uniform marker size for every scatter point (and legend marker) in
#: Candidate B: no variable encodes into point size.
POINT_SIZE = 65.0
#: Log-space margin (decades) added to each axis beyond the observed min/max,
#: so no point or its marker edge is clipped or pushed against the frame.
LOG_PAD_DECADES = 0.18


def draw_candidate_b(plt, dataset_order, loaded_order, panel_stats, label_order, colour_of):
	all_x = [s["calls_per_q"] for stats in panel_stats.values() for s in stats.values()
	        if s["calls_per_q"]]
	all_y = [s["mwh_per_exec"] for stats in panel_stats.values() for s in stats.values()
	        if s["mwh_per_exec"]]
	x_lo = 10 ** (math.log10(min(all_x)) - LOG_PAD_DECADES)
	x_hi = 10 ** (math.log10(max(all_x)) + LOG_PAD_DECADES)
	y_lo = 10 ** (math.log10(min(all_y)) - LOG_PAD_DECADES)
	y_hi = 10 ** (math.log10(max(all_y)) + LOG_PAD_DECADES)

	n_rows, n_cols = len(dataset_order), 3
	fig, axes = plt.subplots(n_rows, n_cols, figsize=(4.3 * n_cols, 4.0 * n_rows))
	handles_by_label = {}
	for ri, ds in enumerate(dataset_order):
		systems = loaded_order[ds]
		for ci, sys_name in enumerate(CANDIDATE_SYSTEM_ORDER):
			ax = axes[ri][ci] if n_rows > 1 else axes[ci]
			if sys_name not in systems:
				ax.axis("off")
				continue
			stats = panel_stats[(ds, sys_name)]
			for label in label_order:
				s = stats.get(label)
				if not s or not s["calls_per_q"] or not s["mwh_per_exec"]:
					continue
				pt = ax.scatter([s["calls_per_q"]], [s["mwh_per_exec"]], s=POINT_SIZE,
				                color=colour_of[label], edgecolor=v.INK, linewidth=0.4,
				                alpha=0.88, zorder=3)
				handles_by_label.setdefault(label, pt)
			ax.set_xscale("log")
			ax.set_yscale("log")
			ax.set_xlim(x_lo, x_hi)
			ax.set_ylim(y_lo, y_hi)
			ax.tick_params(labelsize=8.5)
			if ri == 0:
				ax.set_title(sys_name, fontsize=12.5, fontweight="bold")
			if ci == 0:
				ax.annotate(tf.DATASET_LABELS[ds], xy=(-0.34, 0.5), xycoords="axes fraction",
				           fontsize=11.5, fontweight="bold", ha="right", va="center",
				           rotation=90)
	fig.supxlabel("Mean calls per question (log scale)", fontsize=11, y=0.14)
	fig.supylabel("Mean attributed energy per execution (mWh, log scale)", fontsize=11)

	handles = list(handles_by_label.values())
	labels = [tf._display_label(lbl) for lbl in handles_by_label]
	ncol = min(5, len(labels))
	handles, labels = tf._legend_row_major(handles, labels, ncol)
	legend = fig.legend(handles, labels, loc="lower center", ncol=ncol, frameon=False,
	                    fontsize=10, bbox_to_anchor=(0.5, -0.03), title="Operation",
	                    markerscale=1.8)
	legend.get_title().set_fontsize(10.5)

	fig.suptitle("Operation frequency vs. per-execution energy cost", fontsize=13.5,
	            fontweight="bold", y=0.98)
	fig.tight_layout(rect=(0.045, 0.19, 1, 0.92))
	return fig


# ---------------------------------------------------------------------------


def _print_top3(dataset_order, loaded_order, panel_stats):
	print()
	print("=" * 78)
	print("TOP 3 OPERATIONS PER (dataset, system)")
	print("=" * 78)
	for ds in dataset_order:
		for name in loaded_order[ds]:
			stats = panel_stats[(ds, name)]
			print(f"\n{tf.DATASET_LABELS[ds]} {name}:")
			for metric, key, unit, fmt in (
				("attributed Wh/question", "wh_per_q", "Wh", "{:.5f}"),
				("calls/question", "calls_per_q", "calls", "{:.3f}"),
				("energy/execution", "mwh_per_exec", "mWh", "{:.4f}"),
			):
				ranked = sorted(((label, s[key]) for label, s in stats.items()
				                if s.get(key) is not None), key=lambda kv: -kv[1])[:3]
				parts = ", ".join(f"{tf._display_label(lbl)}={fmt.format(val)} {unit}"
				                 for lbl, val in ranked)
				print(f"  top 3 by {metric:24s}: {parts}")


def main(argv=None):
	ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
	ap.add_argument("--dataset", action="append", metavar="NAME=CACHE_OUTDIR",
	                help="webqsp=DIR and cwq=DIR, each an already-built "
	                     "make_comparable_figures.py --out directory; repeatable "
	                     "(default: the two documented thesis_figures.py cache dirs)")
	ap.add_argument("--out-figures", default="results/figures/cross_dataset")
	args = ap.parse_args(argv)

	cache_dirs = tf.mcf.parse_pairs(args.dataset, DEFAULT_CACHE_DIRS)
	missing = [d for d in tf.DATASET_ORDER if d not in cache_dirs]
	if missing:
		raise SystemExit(f"--dataset required for {missing}")

	figures_out = os.path.abspath(os.path.expanduser(args.out_figures))
	os.makedirs(figures_out, exist_ok=True)

	loaded, panel_stats, panel_coverage = _validate_and_load(tf.DATASET_ORDER, cache_dirs)
	loaded_order = {ds: order for ds, (order, _qrows) in loaded.items()}

	labels_present = {label for stats in panel_stats.values() for label in stats}
	label_order, colour_of = tf._label_colours(labels_present)

	plt = v._pyplot()
	with plt.rc_context(v.STYLE):
		fig_a = draw_candidate_a(plt, tf.DATASET_ORDER, loaded_order, panel_stats,
		                         panel_coverage, label_order, colour_of)
		paths_a = v.save_figure(fig_a, figures_out, "candidate_c2_energy_by_operation", png=False)
		plt.close(fig_a)

		fig_b = draw_candidate_b(plt, tf.DATASET_ORDER, loaded_order, panel_stats,
		                         label_order, colour_of)
		paths_b = v.save_figure(fig_b, figures_out, "candidate_c2_operation_frequency_vs_cost",
		                        png=False)
		plt.close(fig_b)

	print()
	print(f"Candidate B readability check: {len(label_order)} canonical operation labels "
	     f"across all panels; largest single panel carries "
	     f"{max(len(panel_stats[(ds, s)]) for ds in tf.DATASET_ORDER for s in loaded_order[ds])} "
	     "points. Candidate C (calls/question + Wh/question side-by-side panels) is NOT "
	     "produced: this stays legible without it.")

	_print_top3(tf.DATASET_ORDER, loaded_order, panel_stats)

	print()
	print("Canonical operation categories used, by system (taxonomy order):")
	for name in CANDIDATE_SYSTEM_ORDER:
		labels_for_system = sorted(
			{label for ds in tf.DATASET_ORDER if name in loaded_order[ds]
			 for label in panel_stats[(ds, name)]},
			key=lambda lbl: label_order.index(lbl) if lbl in label_order else 999)
		print(f"  {name}: {', '.join(labels_for_system)}")

	print()
	for p in paths_a + paths_b:
		print(f"  -> {p}")
	return 0


if __name__ == "__main__":
	sys.exit(main())
