"""Comparable per-system supervisor figures, GPU energy on the y-axis.

Everything needed lives in this repository: the figures are drawn by
agent_energy_profiler.visualize, and the per-question answer outcomes that
colour Figure 3 come from measurement/export_outcomes.py (this repository's
evaluator). Only the run artifacts and the prediction files are read.

    fig1_{sys}_operation_energy      energy by semantic operation (vertical bars)
    fig2_{sys}_energy_ecdf           question distribution (cumulative share on x)
    fig3_{sys}_trajectory_properties workload vs energy, coloured by answer outcome
    fig4_{sys}_outcome_and_fallback  one image: energy by outcome + fallback split
    fig5_{sys}_semantic_flow         total -> operation type -> operation label

Usage (defaults are the WebQSP Gemma-3-4B runs of 2026-09-05 in measurement/runs):

    python measurement/make_comparable_figures.py --out ~/webqsp_supervisor_summary
    python measurement/make_comparable_figures.py --out DIR \
        --run CoR=measurement/runs/<run> --predictions CoR=results/cor/<...>/predict.jsonl

Writes only into --out, and never modifies a run artifact.
"""

import argparse
import hashlib
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)
sys.path.insert(0, HERE)

import export_outcomes  # noqa: E402
from agent_energy_profiler import visualize  # noqa: E402

DOMAIN, BASIS = "gpu_energy_j", "trajectory"
#: Documented defaults: the three complete WebQSP Gemma-3-4B runs in this repository.
DEFAULT_RUNS = {"CoR": "measurement/runs/cor_webqsp_gemma4b_0905_0856",
                "ToG": "measurement/runs/tog_webqsp_gemma4b_0905_0856",
                "PoG": "measurement/runs/pog_webqsp_gemma4b_0905_0856"}
DEFAULT_PREDICTIONS = {"CoR": "results/cor/webqsp/gemma-3-4b-it/predict.jsonl",
                       "ToG": "results/tog/webqsp/gemma-3-4b-it/predict.jsonl",
                       "PoG": "results/pog/webqsp/gemma-3-4b-it/predict.jsonl"}
#: Figure 3: workload variables on x, energy on y. The last one is discrete.
VS_VARIABLES = ("output_tokens", "input_tokens", "llm_calls", "max_traversal_depth")
DISCRETE = "max_traversal_depth"
FOUND, NOT_FOUND = "Gold answer found", "Gold answer not found"
OUTCOME_STYLE = {export_outcomes.HIT: {"label": FOUND, "colour": "#0ca30c", "marker": "o"},
                 export_outcomes.MISS: {"label": NOT_FOUND, "colour": "#d03b3b", "marker": "^"}}
OUTCOME_ORDER = (export_outcomes.HIT, export_outcomes.MISS)
FIG3_SIZE, JITTER, SEED = (9.2, 7.0), 0.17, 20260911


def parse_pairs(values, default):
	out = dict(default)
	for item in values or ():
		if "=" not in item:
			raise SystemExit(f"expected NAME=VALUE, got {item!r}")
		name, value = item.split("=", 1)
		out[name.strip()] = value.strip()
	return out


def main(argv=None):
	ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
	ap.add_argument("--out", default=os.path.expanduser("~/webqsp_supervisor_summary"),
	                help="output directory (default ~/webqsp_supervisor_summary)")
	ap.add_argument("--run", action="append", metavar="NAME=DIR",
	                help="run directory per system; repeatable (default: the three WebQSP runs)")
	ap.add_argument("--predictions", action="append", metavar="NAME=FILE",
	                help="predict.jsonl per system; repeatable")
	ap.add_argument("--dataset", default="webqsp", help="dataset name for the evaluator")
	args = ap.parse_args(argv)

	runs = parse_pairs(args.run, {} if args.run else DEFAULT_RUNS)
	predictions = parse_pairs(args.predictions, {} if args.predictions else DEFAULT_PREDICTIONS)
	missing = sorted(set(runs) - set(predictions))
	if missing:
		raise SystemExit(f"--predictions missing for {missing}")
	out = os.path.abspath(os.path.expanduser(args.out))
	os.makedirs(out, exist_ok=True)

	def resolve(path):
		return path if os.path.isabs(path) else os.path.join(ROOT, path)

	checks, placed, figure_ylabel = [], {}, {}

	def check(name, ok, detail=""):
		checks.append(f"{'PASS' if ok else 'FAIL'}  {name}  {detail}".rstrip())
		if not ok:
			open(os.path.join(out, "comparable_figures_checks.txt"), "w").write("\n".join(checks) + "\n")
			raise SystemExit(f"CHECK FAILED: {name} {detail}")

	def tree(directory):
		return sorted(os.path.join(r, f) for r, _, fs in os.walk(directory) for f in fs)

	def digest(paths):
		h = hashlib.sha256()
		for path in paths:
			h.update(path.encode())
			with open(path, "rb") as f:
				h.update(f.read())
		return h.hexdigest()

	def close(a, b, rel=1e-9):
		return abs(a - b) <= rel * max(1.0, abs(b))

	run_files = [f for directory in runs.values() for f in tree(resolve(directory))]
	before = digest(run_files)

	plt = visualize._pyplot()
	import numpy as np
	from matplotlib.lines import Line2D

	# ---- gather everything once per run, from the artifacts themselves
	data = {}
	for name, directory in runs.items():
		run = visualize.load_runs([f"{name}={resolve(directory)}"])[0]
		outcomes = export_outcomes.score(resolve(predictions[name]), args.dataset, name)
		export_outcomes.write_csv(outcomes, os.path.join(out, f"outcomes_{name.lower()}.csv"))
		annotations = {r["question_id"]: r["outcome"] for r in outcomes
		               if r["outcome"] in OUTCOME_ORDER}
		op_rows, _ = visualize.prepare_operation_energy([run], DOMAIN)
		q_rows, q_stats, _ = visualize.prepare_question_energy([run], DOMAIN, BASIS)
		flow_rows, _ = visualize.prepare_semantic_flow(run, DOMAIN)
		used_outcome, outcome_rows, outcome_stats, _ = visualize.prepare_outcome_energy(
			[run], DOMAIN, BASIS)
		used_split, means, per_question, _ = visualize.prepare_fallback_split([run], DOMAIN)
		vs = {}
		for variable in VS_VARIABLES:
			rows, stats, _ = visualize.prepare_energy_vs([run], DOMAIN, BASIS, variable,
			                                             annotations, "outcome")
			vs[variable] = ([r for r in rows if r["category"]], stats)
		data[name] = {"run": run, "outcomes": outcomes, "op": op_rows, "q": q_rows,
		              "q_stats": q_stats, "flow": flow_rows, "outcome_rows": outcome_rows,
		              "outcome_stats": outcome_stats, "used_outcome": used_outcome,
		              "used_split": used_split, "means": means, "per_question": per_question,
		              "vs": vs}

	# ---- shared axis limits, computed across systems
	op_max = max(r["energy_j"] for d in data.values() for r in d["op"]
	             if r["row_kind"] == visualize.ROW_OPERATION and r["energy_j"] is not None)
	q_values = [r["energy_j"] for d in data.values() for r in d["q"]]
	split_max = max(sum(m["mean_energy_j"] for m in d["means"]) for d in data.values())
	OP_YLIM = (0.0, op_max * 1.12)
	Q_YLIM = (min(q_values) * 0.85, max(q_values) * 1.18)
	SPLIT_YLIM = (0.0, split_max * 1.12)
	XLIM = {}
	for variable in VS_VARIABLES:
		values = [r["x"] for d in data.values() for r in d["vs"][variable][0]]
		XLIM[variable] = ((min(v for v in values if v > 0) * 0.85, max(values) * 1.18)
		                  if variable != DISCRETE else (min(values) - 0.6, max(values) + 0.6))

	def save(fig, stem, tables=()):
		for path in visualize.save_figure(fig, out, stem):
			placed[os.path.basename(path)] = fig
		for suffix, rows, fields in tables:
			visualize.write_rows(os.path.join(out, f"{stem}{suffix}.csv"), rows, fields)
			placed[f"{stem}{suffix}.csv"] = None
		plt.close(fig)

	rng = np.random.default_rng(SEED)
	with plt.rc_context(visualize.STYLE):
		for name, d in data.items():
			key = name.lower()
			fig = visualize.draw_operation_energy(plt, [d["run"]], d["op"], DOMAIN, "j", OP_YLIM)
			save(fig, f"fig1_{key}_operation_energy",
			     [("", d["op"], visualize.OPERATION_FIELDS)])
			fig = visualize.draw_question_energy(plt, [d["run"]], d["q"], d["q_stats"], DOMAIN,
			                                     BASIS, ["median", "mean", "p90", "p95"], False,
			                                     Q_YLIM)
			save(fig, f"fig2_{key}_energy_ecdf",
			     [("", d["q"], visualize.QUESTION_FIELDS),
			      ("_stats", d["q_stats"], ("run", "paradigm", "domain", "energy_basis")
			       + visualize.STAT_FIELDS)])

			# Figure 3: one panel per workload variable, coloured by answer outcome.
			fig, axes = plt.subplots(2, 2, figsize=FIG3_SIZE, sharey=True)
			for ax, variable in zip(axes.flat, VS_VARIABLES):
				rows, _ = d["vs"][variable]
				if variable == DISCRETE:
					for row in rows:
						offset = -JITTER if row["category"] == export_outcomes.HIT else JITTER
						row["x_plotted"] = row["x"] + offset + float(rng.uniform(-0.1, 0.1))
					drawn = [dict(r, x=r["x_plotted"]) for r in rows]
				else:
					drawn = rows
				visualize.draw_energy_vs(plt, [d["run"]], drawn, d["vs"][variable][1], DOMAIN,
				                         BASIS, variable, False, Q_YLIM, ax=ax,
				                         category_order=OUTCOME_ORDER,
				                         category_style=OUTCOME_STYLE)
				ax.set_xlim(*XLIM[variable])
				ax.set_title("")
				if variable == DISCRETE:
					ax.set_xscale("linear")
					ax.set_xticks(sorted({int(r["x"]) for r in rows}))
					ax.grid(axis="x", visible=False)
					ax.set_xlabel(visualize.variable_axis_label(variable) + " (black bar = median)")
					for category in OUTCOME_ORDER:
						offset = -JITTER if category == export_outcomes.HIT else JITTER
						for depth in sorted({r["x"] for r in rows}):
							group = [r["energy_j"] for r in rows
							         if r["category"] == category and r["x"] == depth]
							if group:
								ax.plot([depth + offset - 0.13, depth + offset + 0.13],
								        [float(np.median(group))] * 2, color=visualize.INK,
								        linewidth=1.6)
				if ax is not axes.flat[0]:
					ax.get_legend().remove()
			# Shared y: one figure-level label, so the long per-panel labels cannot collide.
			shared_label = axes.flat[0].get_ylabel()
			for ax in axes.flat:
				ax.set_ylabel("")
			fig.supylabel(shared_label, fontsize=9)
			figure_ylabel[fig] = shared_label
			fig.suptitle(f"{name} Energy and Trajectory Properties", fontsize=11, fontweight="bold")
			fig.tight_layout(rect=(0, 0, 1, 0.96))
			by_question, stats3 = {}, []
			for variable in VS_VARIABLES:
				rows, _ = d["vs"][variable]
				# Correlations over exactly the points drawn (raw x, never the jitter),
				# so the companion CSV matches the figure.
				for category in ("(drawn)",) + OUTCOME_ORDER:
					subset = [r for r in rows if category == "(drawn)" or r["category"] == category]
					n, pearson, spearman, status = visualize.correlation(
						[(r["x"], r["energy_j"]) for r in subset])
					stats3.append({"run": name, "paradigm": d["run"].paradigm,
					               "x_variable": variable, "y_variable": DOMAIN,
					               "energy_basis": BASIS, "scale": "raw values",
					               "category_column": "outcome", "category": category,
					               "n_pairs": n, "pearson_r": pearson, "spearman_rho": spearman,
					               "status": status})
				for row in rows:
					entry = by_question.setdefault(row["question_id"], {
						"question_id": row["question_id"], "run": name,
						"outcome": OUTCOME_STYLE[row["category"]]["label"],
						"gpu_energy_j": row["energy_j"], "energy_basis": BASIS})
					entry[variable] = row["x"]
					if variable == DISCRETE:
						entry["max_traversal_depth_plotted"] = row["x_plotted"]
			save(fig, f"fig3_{key}_trajectory_properties",
			     [("", list(by_question.values()),
			       ("question_id", "run", "outcome", "gpu_energy_j", "energy_basis")
			       + VS_VARIABLES + ("max_traversal_depth_plotted",)),
			      ("_correlations", stats3, visualize.CORRELATION_FIELDS)])

			# Figure 4: both halves in one image.
			fig, (left, right) = plt.subplots(1, 2, figsize=(9.0, 4.6),
			                                  gridspec_kw={"width_ratios": [1.2, 1.0]})
			visualize.draw_outcome_energy(plt, d["used_outcome"], d["outcome_rows"], DOMAIN,
			                              BASIS, False, Q_YLIM, ax=left)
			visualize.draw_fallback_split(plt, d["used_split"], d["means"], DOMAIN, SPLIT_YLIM,
			                              ax=right)
			left.set_title("Question energy by outcome", fontsize=10, fontweight="normal")
			right.set_title("Fallback energy split", fontsize=10, fontweight="normal")
			fig.suptitle(visualize.plot_title(visualize.PLOT_OUTCOME, DOMAIN, name),
			             fontsize=11, fontweight="bold")
			fig.tight_layout(rect=(0, 0, 1, 0.95))
			save(fig, f"fig4_{key}_outcome_and_fallback",
			     [("_outcome", d["outcome_rows"], visualize.OUTCOME_FIELDS),
			      ("_outcome_stats", d["outcome_stats"],
			       ("run", "paradigm", "outcome", "domain", "energy_basis", "n_questions")
			       + visualize.STAT_FIELDS[2:]),
			      ("_fallback_split", d["means"], visualize.SPLIT_FIELDS),
			      ("_fallback_split_per_question", d["per_question"],
			       visualize.SPLIT_QUESTION_FIELDS)])

			fig = visualize.draw_semantic_flow(plt, d["run"], d["flow"], DOMAIN)
			save(fig, f"fig5_{key}_semantic_flow", [("", d["flow"], visualize.FLOW_FIELDS)])

	# ---- validation, against the run artifacts and the evaluator only
	for name, d in data.items():
		run, key = d["run"], name.lower()
		attributed = sum(e[DOMAIN] for e in run.events)
		drawn = [r for r in d["op"] if r["row_kind"] == visualize.ROW_OPERATION and r["plotted"]]
		check(f"{name} fig1 operation energy sums to the attributed event energy",
		      close(sum(r["energy_j"] for r in drawn), attributed),
		      f"{sum(r['energy_j'] for r in drawn):,.3f} J")
		windows = sum(r[f"trajectory_{DOMAIN}"] for r in run.trajectories
		              if r.get("coverage_is_independent"))
		check(f"{name} fig2 has every question and sums to the question-window energy",
		      len(d["q"]) == len(visualize.question_ids(run.events))
		      and close(sum(r["energy_j"] for r in d["q"]), windows),
		      f"{len(d['q']):,} questions, {sum(r['energy_j'] for r in d['q']):,.3f} J")
		counts = {k: sum(1 for r in d["outcomes"] if r["outcome"] == k)
		          for k in (export_outcomes.HIT, export_outcomes.MISS, export_outcomes.UNSCORED)}
		scored = [r for r in d["vs"][VS_VARIABLES[0]][0]]
		check(f"{name} fig3 draws the scored questions only, coloured by outcome",
		      len(scored) == counts[export_outcomes.HIT] + counts[export_outcomes.MISS]
		      and sum(1 for r in scored if r["category"] == export_outcomes.HIT) == counts[export_outcomes.HIT],
		      f"hit={counts[export_outcomes.HIT]:,} miss={counts[export_outcomes.MISS]:,} "
		      f"unscored={counts[export_outcomes.UNSCORED]:,} (excluded)")
		fallbacks = len(visualize.fallback_events(run))
		check(f"{name} fig4 left panel has every question, {fallbacks:,} fallbacks",
		      len(d["outcome_rows"]) == len(d["q"])
		      and sum(1 for r in d["outcome_rows"] if r["outcome"] == visualize.FALLBACK) == fallbacks)
		check(f"{name} fig4 right panel covers every fallback question",
		      all(m["n_fallback_questions"] == fallbacks for m in d["means"]))
		roots = [r for r in d["flow"] if r["row_kind"] == visualize.ROW_FLOW
		         and r["source_level"] == visualize.LEVEL_ROOT]
		check(f"{name} fig5 operation types sum to the attributed event energy",
		      close(sum(r["energy_j"] for r in roots), attributed))

	families = {}
	for filename, fig in sorted(placed.items()):
		if not filename.endswith(".png"):
			continue
		axes = [ax for ax in fig.axes if ax.axison]
		family = filename.split("_")[0]
		if family == "fig5":
			check(f"{filename}: flow diagram, exempt from the energy-on-y and size rules",
			      axes == [], "no visible axes")
			continue
		check(f"{filename}: no axis puts energy on x",
		      all("energy" not in ax.get_xlabel().lower() for ax in axes))
		labelled = [ax.get_ylabel() for ax in axes if ax.get_ylabel()]
		if not labelled and fig in figure_ylabel:
			# A shared-y grid names the axis once for the whole figure instead.
			labelled = [figure_ylabel[fig]]
		check(f"{filename}: energy on y",
		      bool(labelled) and all("energy" in label.lower() for label in labelled),
		      str(labelled))
		if family == "fig4":
			check(f"{filename}: both halves in one image", len(axes) == 2)
		families.setdefault(family, []).append(
			(tuple(round(v, 3) for v in fig.get_size_inches()),
			 sorted({tuple(round(float(v), 6) for v in ax.get_ylim()) for ax in axes})))
	for family, entries in families.items():
		check(f"{family}: identical y-limits across systems",
		      len({str(e[1]) for e in entries}) == 1, str(entries[0][1]))
		check(f"{family}: identical figure size across systems",
		      len({e[0] for e in entries}) == 1, str(entries[0][0]))
	check("run artifacts unchanged", digest(run_files) == before)

	with open(os.path.join(out, "COMPARABLE_FIGURES.md"), "w") as f:
		f.write("# Comparable per-system figures\n\nRebuild with "
		        "`python measurement/make_comparable_figures.py --out <dir>` (all code is in the "
		        "repository). GPU energy is on the y-axis of every figure that has axes.\n\n"
		        "| Figure | Files | Energy basis | Shared y-limits (J) |\n|---|---|---|---|\n"
		        f"| 1 operations | `fig1_{{sys}}_operation_energy.*` | attributed events | "
		        f"{OP_YLIM[0]:,.0f} to {OP_YLIM[1]:,.0f} (linear) |\n"
		        f"| 2 distribution | `fig2_{{sys}}_energy_ecdf.*` | question window | "
		        f"{Q_YLIM[0]:,.1f} to {Q_YLIM[1]:,.1f} (log) |\n"
		        f"| 3 trajectory | `fig3_{{sys}}_trajectory_properties.*` | question window | "
		        f"{Q_YLIM[0]:,.1f} to {Q_YLIM[1]:,.1f} (log) |\n"
		        f"| 4 outcome + fallback | `fig4_{{sys}}_outcome_and_fallback.*` | left: question "
		        f"window; right: attributed events | left {Q_YLIM[0]:,.1f} to {Q_YLIM[1]:,.1f} "
		        f"(log); right {SPLIT_YLIM[0]:,.0f} to {SPLIT_YLIM[1]:,.0f} (linear) |\n"
		        f"| 5 semantic flow | `fig5_{{sys}}_semantic_flow.*` | attributed events | "
		        "no axes (energy is ribbon thickness) |\n\n"
		        "Figure 3 colours each question by the outcome in `outcomes_{sys}.csv`, written by "
		        "`measurement/export_outcomes.py`; questions with no gold answer are unscored and "
		        "are excluded from it.\n")
	open(os.path.join(out, "comparable_figures_checks.txt"), "w").write("\n".join(checks) + "\n")
	print("\n".join(checks))
	print(f"\nshared y-limits (J): fig1 {OP_YLIM}, fig2/3/4-left {Q_YLIM}, fig4-right {SPLIT_YLIM}")
	print(f"\n{len(placed)} files in {out}")
	return 0


if __name__ == "__main__":
	sys.exit(main())
