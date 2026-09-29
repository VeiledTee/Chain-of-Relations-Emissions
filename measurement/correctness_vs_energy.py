"""Fully correct vs not-fully-correct per-question GPU energy: one figure
per dataset, one panel per system.

Reads each dataset's already-built per-question cache
(measurement/make_comparable_figures.py --dataset <name> [--rebuild-cache]);
never opens events_attributed.jsonl or predict.jsonl itself. If a cache is
missing the per-question "f1" column (cache_version < 3), this script stops
and says which dataset needs `--rebuild-cache` rather than re-deriving F1
on its own -- correctness scoring stays owned by export_outcomes.py.

    python measurement/correctness_vs_energy.py \
        --dataset webqsp=~/webqsp_supervisor_summary/cross_dataset/webqsp/wh \
        --dataset cwq=~/cwq_supervisor_summary/wh \
        --out ~/cwq_supervisor_summary/cross_dataset/wh

Correctness is F1-based, matching the rest of this project: fully correct
means F1 == 1, not fully correct means F1 < 1 (which includes partial
credit and is therefore never called "incorrect"). The measured quantity is
GPU **energy** (Wh by default) -- it is not "emissions": no carbon-intensity
conversion is applied anywhere in this project, so the word is never used
here. Energy is the whole-question trajectory-basis GPU energy already in
the cache, in the figure's display unit; the summary CSV stays in joules.

Writes only into --out: fig7_correctness_vs_energy_<dataset>.{png,pdf} and
fig7_correctness_vs_energy_summary.csv.
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
#: Never "Incorrect": the F1 < 1 group includes partially correct answers.
BIN_LABELS = {FULLY: "Fully correct\n(F1 = 1)", NOT_FULLY: "Not fully correct\n(F1 < 1)"}
BIN_COLOURS_SLOT = {FULLY: 0, NOT_FULLY: 1}
STEM = "fig7_correctness_vs_energy"
SUMMARY_FIELDS = ("dataset", "paradigm", "correctness", "n", "mean_energy_j",
                  "median_energy_j", "std_energy_j", "min_energy_j", "max_energy_j")


def _bin_of(f1):
	return FULLY if f1 == 1.0 else NOT_FULLY


def _load_dataset(name, out_dir):
	"""(question_rows, systems) from an already-built cache, or a clear error
	naming exactly which rebuild command fixes it."""
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


def _grouped_energies(rows, systems):
	"""{(system, bin): [energy_j, ...]} over questions with both an F1 and a
	trajectory energy value."""
	label_to_system = {info["label"]: name for name, info in systems.items()}
	groups = {(name, b): [] for name in systems for b in BIN_ORDER}
	for r in rows:
		name = label_to_system.get(r.get("run"))
		f1, energy = r.get("f1"), r.get("energy_trajectory_j")
		if name is None or f1 is None or energy is None:
			continue
		groups[(name, _bin_of(f1))].append(energy)
	return groups


def _summary_rows(dataset, groups, order):
	rows = []
	for name in order:
		for b in BIN_ORDER:
			vals = groups[(name, b)]
			rows.append({
				"dataset": DATASET_LABELS[dataset], "paradigm": name,
				"correctness": BIN_LABELS[b].replace("\n", " "), "n": len(vals),
				"mean_energy_j": statistics.mean(vals) if vals else None,
				"median_energy_j": statistics.median(vals) if vals else None,
				"std_energy_j": statistics.stdev(vals) if len(vals) > 1 else None,
				"min_energy_j": min(vals) if vals else None,
				"max_energy_j": max(vals) if vals else None,
			})
	return rows


def _panel(plt, ax, groups, name, title):
	data = [[v.to_display_energy(e) for e in groups[(name, b)]] for b in BIN_ORDER]
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
	# Points beyond the whiskers, coloured to match their own box so a panel
	# with a wide-spread group is still readable rather than a grey smear.
	for flier, b in zip(box["fliers"], BIN_ORDER):
		flier.set_markerfacecolor(v.PALETTE[BIN_COLOURS_SLOT[b]])
	# n / mean / median live in the tick label: with three narrow panels there
	# is no room to place text beside each box without it colliding with the
	# next panel, and every group needs a label regardless of whether it has data.
	ticks = []
	for b, vals in zip(BIN_ORDER, data):
		if vals:
			m, md = statistics.mean(vals), statistics.median(vals)
			ticks.append(f"{BIN_LABELS[b]}\nn = {len(vals):,}\n"
			             f"mean {v.format_display_energy(m)}\nmed {v.format_display_energy(md)}")
		else:
			ticks.append(f"{BIN_LABELS[b]}\nn = 0")
	ax.set_xticks(positions, ticks, fontsize=6.3)
	ax.set_xlim(0.4, 2.6)
	ax.grid(axis="x", visible=False)
	ax.set_title(title, fontsize=9.5, fontweight="bold")


def main(argv=None):
	ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
	ap.add_argument("--dataset", action="append", required=True, metavar="NAME=OUTDIR",
	                help="webqsp=DIR and cwq=DIR, each an already-built "
	                     "make_comparable_figures.py --out directory; repeatable")
	ap.add_argument("--out", required=True)
	ap.add_argument("--energy-unit", default=v.DEFAULT_ENERGY_UNIT, choices=v.ENERGY_UNIT_CHOICES)
	args = ap.parse_args(argv)

	out_dirs = mcf.parse_pairs(args.dataset, {})
	missing = [d for d in DATASET_ORDER if d not in out_dirs]
	if missing:
		raise SystemExit(f"--dataset required for {missing}")

	per_dataset, orders = {}, {}
	for ds in DATASET_ORDER:
		rows, systems = _load_dataset(ds, out_dirs[ds])
		order = [s for s in SYSTEM_ORDER if s in systems]
		per_dataset[ds] = _grouped_energies(rows, systems)
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
	with plt.rc_context(v.STYLE), v.energy_display(args.energy_unit):
		for ds in DATASET_ORDER:
			order = orders[ds]
			energies = [e for name in order for vals in per_dataset[ds].values() for e in vals]
			ylim_j = (min(energies) * 0.85, max(energies) * 1.18)

			# Same data, same layout, same colours: only the y-scale differs.
			for linear, sfx in mcf.SCALES:
				fig, axes = plt.subplots(1, len(order), figsize=(3.8 * len(order), 4.6),
				                         sharey=True)
				axes = [axes] if len(order) == 1 else list(axes)
				for ax, name in zip(axes, order):
					_panel(plt, ax, per_dataset[ds], name, name)
				axes[0].set_ylabel(f"GPU energy per question ({v.display_energy_unit()})")
				if not linear:
					axes[0].set_yscale("log")
				v._apply_ylim(axes[0], ylim_j)
				axes[0].yaxis.set_major_formatter(v._energy_formatter())
				fig.legend(handles=legend_handles, loc="lower center", ncol=2, frameon=False,
				          fontsize=8, bbox_to_anchor=(0.5, -0.04))
				fig.suptitle(f"{DATASET_LABELS[ds]}: Fully Correct vs. Not Fully Correct Energy",
				            fontsize=12, fontweight="bold")
				fig.tight_layout(rect=(0, 0.04, 1, 0.94))
				written += v.save_figure(fig, out, f"{STEM}_{ds}{sfx}")
				plt.close(fig)

	written.append(v.write_rows(os.path.join(out, f"{STEM}_summary.csv"), summary,
	                            SUMMARY_FIELDS))

	for ds in DATASET_ORDER:
		for name in orders[ds]:
			full = per_dataset[ds][(name, FULLY)]
			not_full = per_dataset[ds][(name, NOT_FULLY)]
			mean_f = statistics.mean(full) if full else None
			mean_n = statistics.mean(not_full) if not_full else None
			ratio = (mean_n / mean_f) if mean_f else None
			print(f"{DATASET_LABELS[ds]:7s} {name:4s} n_full={len(full):5,} "
			     f"n_not_full={len(not_full):5,}"
			     + (f"  mean ratio not_full/full={ratio:.3f}" if ratio is not None else ""))
	for path in written:
		print(f"  -> {path}")
	return 0


if __name__ == "__main__":
	sys.exit(main())
