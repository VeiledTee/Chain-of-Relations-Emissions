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
    fig6_webqsp_answerability_outcomes  WebQSP only: one square per question, gold
                                     answer found / not found, by answerability
                                     group (rows) and system (columns)

Usage (defaults are the WebQSP Gemma-3-4B runs of 2026-09-05 in measurement/runs):

    python measurement/make_comparable_figures.py --out ~/webqsp_supervisor_summary
    python measurement/make_comparable_figures.py --out DIR \
        --run CoR=measurement/runs/<run> --predictions CoR=results/cor/<...>/predict.jsonl

Writes only into --out, and never modifies a run artifact.
"""

import argparse
import csv
import hashlib
import json
import math
import os
import sys
from collections import defaultdict

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)
sys.path.insert(0, HERE)

import export_outcomes  # noqa: E402
import webqsp_answerability  # noqa: E402
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
#: Question-level GPU energy on a log y-axis also gets a linear companion.
#: fig1 is attributed-event energy on a linear axis and fig5 has no axes, so
#: neither qualifies for it; correctness_vs_energy.py reuses this constant so
#: every question-level-energy figure in the project offers the same pair.
SCALES = ((False, ""), (True, "_linear"))
FIG6_STEM = "fig6_webqsp_answerability_outcomes"

# --------------------------------------------------------------------------
# Analysis cache: everything expensive (parsing events_attributed.jsonl,
# re-scoring predictions) happens once and is consolidated into ONE table,
# one row per (system, question_id) -- energy by operation label is pivoted
# into `energy_<label>` / `count_<label>` columns rather than a separate
# table. A small meta.json alongside it records the arguments the cache was
# built from and each system's label/paradigm.
#
# Every figure family (operation totals, ECDF, workload-vs-energy, outcome
# and fallback, semantic flow) is a cheap groupby/aggregate over this one
# table, done fresh on every render. Axis limits, subplot layout, colours
# and display units stay entirely in the drawing code and are recomputed on
# every render, cached or not -- a look-and-feel change never needs
# --rebuild-cache.
# --------------------------------------------------------------------------
CACHE_VERSION = 3  # v3 adds per-question "f1" (needed for correctness-vs-energy figures)
CACHE_DIRNAME = "_cache"
CACHE_FILENAME = "questions.csv"


class RunProxy:
	"""Stand-in for visualize.Run once only derived rows are cached.

	Every draw_* function reads at most `.label` and `.paradigm` off a run --
	never `.events` or `.trajectories`, which is exactly what caching removes.
	"""

	def __init__(self, label, paradigm):
		self.label = label
		self.paradigm = paradigm


def _cast_cell(value):
	"""Best-effort inverse of str(value): int, float, bool, None, else text.

	Safe here because every cached field is a number, a flag, or an id/label
	string that never looks numeric (question ids, operation labels, run
	names). Values are written with plain str(), so this only has to undo
	that, not parse arbitrary CSV.
	"""
	if value == "":
		return None
	if value == "True":
		return True
	if value == "False":
		return False
	try:
		return int(value)
	except ValueError:
		pass
	try:
		return float(value)
	except ValueError:
		return value


def _write_question_table(path, rows):
	"""One CSV, one row per (system, question_id). Column set is the union of
	keys across rows, so a run with an operation label another run never used
	still round-trips (missing cells become empty, i.e. None on read)."""
	fields = []
	for row in rows:
		for key in row:
			if key not in fields:
				fields.append(key)
	with open(path, "w", newline="") as f:
		writer = csv.DictWriter(f, fieldnames=fields, lineterminator="\n")
		writer.writeheader()
		for row in rows:
			writer.writerow({k: ("" if row.get(k) is None else str(row.get(k))) for k in fields})


def _read_question_table(path):
	if not os.path.exists(path):
		return []
	with open(path, newline="") as f:
		return [{k: _cast_cell(v) for k, v in row.items()} for row in csv.DictReader(f)]


def _cache_manifest(cache_dir):
	path = os.path.join(cache_dir, "meta.json")
	if not os.path.exists(path):
		return None
	with open(path) as f:
		return json.load(f)


def _cache_matches(cache_dir, manifest):
	"""True only when a complete, version-matching cache exists for exactly
	these runs/predictions/dataset. This compares the CLI arguments the
	cache was built from, not file contents -- confirming file contents are
	unchanged is exactly the expensive check --rebuild-cache exists to skip."""
	existing = _cache_manifest(cache_dir)
	if existing is None or "systems" not in existing:
		return False
	return {k: existing.get(k) for k in manifest} == manifest


def make_fig6(plt, out, dataset, outcome_csvs, groups_path=None):
	"""WebQSP answerability figure plus its tables, or None for any other dataset.

	outcome_csvs: {system: canonical outcomes CSV}. Reads only those CSVs and the
	committed group file; writes only into out. Returns (written paths, rows, systems).
	"""
	if str(dataset or "").strip().lower() != "webqsp":
		return None
	wa = webqsp_answerability
	groups = wa.load_groups(groups_path or wa.DEFAULT_GROUPS_PATH)
	found = {name: wa.load_outcomes(path, groups) for name, path in outcome_csvs.items()}
	rows, systems = wa.outcome_table(groups, found)
	fields = wa.table_fields(systems)
	compact = wa.compact_table(rows, systems)
	written = []
	fig = wa.draw_waffle(plt, groups, found)
	written += visualize.save_figure(fig, out, FIG6_STEM)
	plt.close(fig)
	visualize.write_rows(os.path.join(out, f"{FIG6_STEM}.csv"), rows, fields)
	visualize.write_rows(os.path.join(out, f"{FIG6_STEM}_compact.csv"), compact,
	                     ["Dataset group"] + systems)
	visualize.write_rows(os.path.join(out, f"{FIG6_STEM}_questions.csv"),
	                     [dict({"question_id": q, "group": groups[q]},
	                           **{s: wa.FOUND_LABEL if found[s][q] else wa.NOT_FOUND_LABEL
	                              for s in systems})
	                      for g in wa.GROUP_ORDER for q in wa.members(groups, g)],
	                     ["question_id", "group"] + systems)
	with open(os.path.join(out, f"{FIG6_STEM}.md"), "w") as f:
		f.write("# WebQSP Gold-Answer Outcomes by Dataset Group\n\n"
		        "Gold answer found = canonical Hit@1 from `measurement/export_outcomes.py` "
		        "(`outcomes_{sys}.csv`). Groups: `datasets/webqsp/webqsp_answerability_groups.csv`."
		        "\n\n" + wa.markdown(compact, ["Dataset group"] + systems) + "\n"
		        + wa.markdown(rows, fields))
	written += [os.path.join(out, f"{FIG6_STEM}{s}") for s in
	            (".csv", "_compact.csv", "_questions.csv", ".md")]
	return written, rows, systems


def draw_trajectory_properties(plt, np, name, d, xlim, ylim, rng, linear, figure_ylabel):
	"""Figure 3 for one system at one y-scale: energy against four workload variables.

	Called once per scale. The depth jitter is computed on the first call and
	kept on the row, so the linear companion draws points in identical places.
	"""
	fig, axes = plt.subplots(2, 2, figsize=FIG3_SIZE, sharey=True)
	for ax, variable in zip(axes.flat, VS_VARIABLES):
		rows, _ = d["vs"][variable]
		if variable == DISCRETE:
			for row in rows:
				if "x_plotted" not in row:
					offset = -JITTER if row["category"] == export_outcomes.HIT else JITTER
					row["x_plotted"] = row["x"] + offset + float(rng.uniform(-0.1, 0.1))
			drawn = [dict(r, x=r["x_plotted"]) for r in rows]
		else:
			drawn = rows
		visualize.draw_energy_vs(plt, [d["run"]], drawn, d["vs"][variable][1], DOMAIN,
		                         BASIS, variable, linear, ylim, ax=ax,
		                         category_order=OUTCOME_ORDER,
		                         category_style=OUTCOME_STYLE)
		ax.set_xlim(*xlim[variable])
		ax.set_title("")
		if variable == DISCRETE:
			ax.set_xscale("linear")
			ax.set_xticks(sorted({int(r["x"]) for r in rows}))
			ax.grid(axis="x", visible=False)
			ax.set_xlabel(visualize.variable_axis_label(variable) + " (black bar = median)")
			for category in OUTCOME_ORDER:
				offset = -JITTER if category == export_outcomes.HIT else JITTER
				for depth in sorted({r["x"] for r in rows}):
					group = [visualize.to_display_energy(r["energy_j"]) for r in rows
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
	return fig


#: Combined RQ2 main figure (fig_trajectory_cost_drivers_main): the single
#: variable the main paper leads with -- LLM call count -- vs whole-question
#: trajectory GPU energy. Output tokens, input tokens and traversal depth
#: stay in the untouched per-dataset fig3_*_trajectory_properties
#: supplementary figures; this is a narrower, display-only subset of that
#: same cached data, not a new measurement.
MAIN2_X_LABEL = "LLM calls per question"
MAIN2_Y_LABEL = "GPU energy per question (Wh)"


def draw_trajectory_cost_drivers_main(plt, rows_by_key, dataset_order, dataset_labels,
                                      system_order):
	"""Main RQ2 figure: 2x3 grid (rows = datasets, columns = systems), x =
	LLM calls per question, y = whole-question trajectory GPU energy.
	Single neutral point style, no outcome colour/marker split, no fitted
	line -- this figure makes one visual point (more LLM calls, more
	energy, across every tested architecture and dataset) and leaves the
	rest of the RQ2 evidence (tokens, depth) to the supplementary figures
	and prose/table. Reuses the same per-question cache rows the per-dataset
	fig3 figures already load; no new computation, no new metric.
	"""
	n_rows, n_cols = len(dataset_order), len(system_order)
	fig, axes = plt.subplots(n_rows, n_cols, figsize=(3.4 * n_cols, 3.0 * n_rows),
	                         sharex=False, sharey=False)
	for ri, ds in enumerate(dataset_order):
		row_ys = []
		for sys_name in system_order:
			for r in rows_by_key.get((ds, sys_name), []):
				e = r.get("energy_trajectory_j")
				if e is not None:
					row_ys.append(visualize.j_to_wh(e))
		y_lo = max(min(row_ys) * 0.85, 1e-4) if row_ys else 1e-4
		y_hi = max(row_ys) * 1.1 if row_ys else 1.0
		for ci, sys_name in enumerate(system_order):
			ax = axes[ri][ci]
			rows = rows_by_key.get((ds, sys_name), [])
			xs, ys = [], []
			for r in rows:
				x, e = r.get("llm_calls"), r.get("energy_trajectory_j")
				if x is None or e is None:
					continue
				xs.append(x)
				ys.append(visualize.j_to_wh(e))
			ax.scatter(xs, ys, s=7, alpha=0.22, color=visualize.NEUTRAL, linewidths=0)
			ax.set_xscale("log")
			ax.set_yscale("log")
			ax.set_ylim(y_lo, y_hi)
			if ri == 0:
				ax.set_title(sys_name, fontsize=12, fontweight="bold")
			if ci == 0:
				ax.annotate(dataset_labels[ds], xy=(-0.42, 0.5), xycoords="axes fraction",
				           fontsize=11.5, fontweight="bold", ha="right", va="center",
				           rotation=90)
			ax.tick_params(labelsize=8.5)
	fig.supxlabel(MAIN2_X_LABEL, fontsize=10.5)
	fig.supylabel(MAIN2_Y_LABEL, fontsize=10.5)
	fig.suptitle("LLM calls and trajectory GPU energy", fontsize=13, fontweight="bold")
	fig.tight_layout(rect=(0.04, 0.04, 1, 0.95))
	return fig


def _sum_or_none(values):
	"""Same rule as the original fallback-split preparation: None (component
	does not apply) if any member of the group lacks the domain."""
	if any(v is None for v in values):
		return None
	return float(sum(values))


def _operation_type(label):
	"""Operation type is the label's own namespace prefix (`llm:x` -> `llm`),
	the taxonomy convention every operation label in this repository follows.
	Reading it back out avoids a second cached column that would only ever
	repeat what the label already says."""
	text = str(label)
	return text.split(":", 1)[0] if ":" in text else text


def _build_question_table(runs, predictions, dataset, resolve, out, check):
	"""Raw path: parse events_attributed.jsonl, re-score predictions, and
	consolidate everything a figure needs into ONE row per (system,
	question_id). Energy by operation label is pivoted into `energy_<label>`
	/ `count_<label>` columns on that row rather than a separate table.

	Also runs the raw-data validation checks, since this is the only place
	run.events/run.trajectories are actually loaded; a cached render has
	nothing to check them against and skips this entirely.

	Returns (rows, systems, run_files, before) -- the last two only make
	sense here, since a cached render never reads run/prediction files.
	"""
	def tree(directory):
		return sorted(os.path.join(r, f) for r, _, fs in os.walk(directory) for f in fs)

	def digest(paths):
		h = hashlib.sha256()
		for path in paths:
			h.update(path.encode())
			with open(path, "rb") as f:
				h.update(f.read())
		return h.hexdigest()

	run_files = [f for directory in runs.values() for f in tree(resolve(directory))]
	run_files += sorted(resolve(path) for path in predictions.values())
	before = digest(run_files)

	rows, systems = [], {}
	for name, directory in runs.items():
		run = visualize.load_runs([f"{name}={resolve(directory)}"])[0]
		systems[name] = {"label": run.label, "paradigm": run.paradigm}
		outcomes = export_outcomes.score(resolve(predictions[name]), dataset, name)
		export_outcomes.write_csv(outcomes, os.path.join(out, f"outcomes_{name.lower()}.csv"))
		outcome_by_q = {r["question_id"]: r for r in outcomes}

		work = visualize.workload(run)
		values, _missing = visualize.question_energy(run, DOMAIN, BASIS)
		marked = visualize.fallback_events(run)
		by_question_events = defaultdict(list)
		op_energy = defaultdict(lambda: defaultdict(float))
		op_count = defaultdict(lambda: defaultdict(int))
		unplaced = 0
		for e in run.events:
			q = e.get("question_id")
			by_question_events[q].append(e)
			label = e.get("operation_label")
			if q is None or label is None:
				unplaced += 1
				continue
			val = e.get(DOMAIN)
			if val is not None:
				op_energy[q][label] += val
			op_count[q][label] += 1
		check(f"{name}: every event has a question_id and an operation_label",
		      unplaced == 0, f"{unplaced} unplaced events")

		n_with_energy = 0
		for q in visualize.question_ids(run.events):
			o = outcome_by_q.get(q, {})
			row = {"dataset": dataset, "run": name, "paradigm": run.paradigm, "question_id": q,
			       "outcome": o.get("outcome"), "hit1": o.get("hit1"), "f1": o.get("f1"),
			       "energy_trajectory_j": values.get(q)}
			if row["energy_trajectory_j"] is not None:
				n_with_energy += 1
			row.update(work.get(q, {}))
			for label, energy in op_energy.get(q, {}).items():
				row[f"energy_{label}"] = energy
			for label, count in op_count.get(q, {}).items():
				row[f"count_{label}"] = count
			marks = marked.get(q, [])
			row["fallback"] = bool(marks)
			if marks:
				reasons = {str((e.get("meta") or {}).get("fallback_reason"))
				          for e in marks if (e.get("meta") or {}).get("fallback_reason") is not None}
				row["fallback_reason"] = "|".join(sorted(reasons))
			if len(marks) == 1:
				mark = marks[0]
				group = by_question_events[q]
				if (mark.get("step_index") is not None
				        and all(e.get("step_index") is not None for e in group)):
					step = mark["step_index"]
					before_vals = [e.get(DOMAIN) for e in group if e["step_index"] < step]
					later = [e.get(DOMAIN) for e in group if e["step_index"] > step]
					before_e = _sum_or_none(before_vals)
					operation_e = mark.get(DOMAIN)
					after_e = _sum_or_none(later) if later else None
					if before_e is not None and operation_e is not None and not (later and after_e is None):
						row["fallback_before_j"] = before_e
						row["fallback_operation_j"] = float(operation_e)
						row["fallback_has_after"] = bool(later)
						row["fallback_after_j"] = after_e if later else None
						row["fallback_operation_label"] = mark.get("operation_label")
			rows.append(row)
		check(f"{name}: every question has a trajectory energy value, "
		     f"{n_with_energy:,} of {len(visualize.question_ids(run.events)):,}",
		     n_with_energy == len(visualize.question_ids(run.events)))

	# Confirms parsing/scoring never wrote into its own inputs. The figure-
	# drawing phase after this only ever writes into --out, so this is
	# checked here rather than a second time at the very end of the render.
	check("run artifacts and prediction files unchanged", digest(run_files) == before)
	return rows, systems


def _assemble_data(question_rows, systems, order):
	"""Everything draw_*/save() need, derived fresh from the one per-question
	table -- cheap groupby/sum/summary_stats over already-materialized rows,
	whether that table just came off disk or was just built. This is the
	single place fig1/2/3/4/5's shapes are reconstructed, so cached and
	freshly-built renders are guaranteed to agree.
	"""
	by_run = defaultdict(list)
	for row in question_rows:
		by_run[row["run"]].append(row)

	data = {}
	for name in order:
		info = systems[name]
		run = RunProxy(info["label"], info["paradigm"])
		qrows = by_run.get(info["label"], [])

		# ---- fig1: energy by operation label, summed across all questions.
		label_energy, label_count = defaultdict(float), defaultdict(int)
		for r in qrows:
			for k, v in r.items():
				if v is None:
					continue
				if k.startswith("energy_") and k != "energy_trajectory_j":
					label_energy[k[len("energy_"):]] += v
				elif k.startswith("count_"):
					label_count[k[len("count_"):]] += v
		total_attributed = sum(label_energy.values())
		op_rows = [{
			"run": info["label"], "paradigm": info["paradigm"], "domain": DOMAIN,
			"energy_basis": visualize.BASIS_ATTRIBUTED, "row_kind": visualize.ROW_OPERATION,
			"operation_type": _operation_type(label), "operation_label": label,
			"event_count": label_count.get(label, 0), "energy_j": energy,
			"pct_attributed_energy": (energy / total_attributed * 100.0
			                          if total_attributed else None),
			"energy_complete": True, "n_events_missing_energy": 0, "n_zero_energy_events": 0,
			"plotted": True,
		} for label, energy in sorted(label_energy.items())]
		# No <unattributed> residual row: computing it correctly requires the
		# same "is there an independent trajectory reference to trust" check
		# aggregate.aggregate() makes (some runs -- CoR's CWQ run among them --
		# have questions where that reference is not trustworthy, and the
		# original code omits the row rather than report a misleading number).
		# This script never plots that row anyway (include_unattributed is
		# never passed), so it is left out entirely rather than approximated.

		# ---- fig5: total -> operation type -> operation label, same totals.
		by_type = defaultdict(float)
		for label, energy in label_energy.items():
			by_type[_operation_type(label)] += energy
		flow_rows = []
		ordered_types = sorted(by_type.items(), key=lambda kv: (-kv[1], str(kv[0])))
		for kind, type_energy in ordered_types:
			type_count = sum(c for label, c in label_count.items()
			                 if _operation_type(label) == kind)
			flow_rows.append({
				"run": info["label"], "paradigm": info["paradigm"], "domain": DOMAIN,
				"energy_basis": visualize.BASIS_ATTRIBUTED, "row_kind": visualize.ROW_FLOW,
				"source_level": visualize.LEVEL_ROOT, "source": visualize.ROOT_NODE,
				"target_level": visualize.LEVEL_TYPE, "target": kind, "energy_j": type_energy,
				"pct_total_attributed": (type_energy / total_attributed * 100.0
				                         if total_attributed else None),
				"event_count": type_count, "plotted": True,
			})
		for kind, _type_energy in ordered_types:
			labels = sorted((label for label in label_energy if _operation_type(label) == kind),
			                key=lambda label: (-label_energy[label], str(label)))
			for label in labels:
				flow_rows.append({
					"run": info["label"], "paradigm": info["paradigm"], "domain": DOMAIN,
					"energy_basis": visualize.BASIS_ATTRIBUTED, "row_kind": visualize.ROW_FLOW,
					"source_level": visualize.LEVEL_TYPE, "source": kind,
					"target_level": visualize.LEVEL_LABEL, "target": label,
					"energy_j": label_energy[label],
					"pct_total_attributed": (label_energy[label] / total_attributed * 100.0
					                         if total_attributed else None),
					"event_count": label_count.get(label, 0), "plotted": True,
				})

		# ---- fig2: whole-question (trajectory-basis) energy, ECDF-ranked.
		values = {r["question_id"]: r["energy_trajectory_j"] for r in qrows
		         if r["energy_trajectory_j"] is not None}
		missing = [r["question_id"] for r in qrows if r["energy_trajectory_j"] is None]
		ordered = sorted(values.items(), key=lambda kv: (kv[1], str(kv[0])))
		n = len(ordered)
		q_rows = [{"run": info["label"], "paradigm": info["paradigm"], "question_id": q,
		          "domain": DOMAIN, "energy_basis": BASIS, "energy_j": v, "ecdf": i / n}
		         for i, (q, v) in enumerate(ordered, 1)]
		q_stats = [{"run": info["label"], "paradigm": info["paradigm"], "domain": DOMAIN,
		           "energy_basis": BASIS, "n_questions": n,
		           "n_questions_without_value": len(missing),
		           **visualize.summary_stats(values.values())}]

		# ---- fig4 left: question energy split by fallback / non-fallback.
		has_fallback = any(r["fallback"] for r in qrows)
		used = [run] if has_fallback else []
		outcome_rows, outcome_groups = [], defaultdict(list)
		for r in qrows:
			if r["energy_trajectory_j"] is None:
				continue
			outcome = visualize.FALLBACK if r["fallback"] else visualize.NO_FALLBACK
			outcome_groups[outcome].append(r["energy_trajectory_j"])
			outcome_rows.append({
				"run": info["label"], "paradigm": info["paradigm"], "question_id": r["question_id"],
				"outcome": outcome, "fallback_reason": r.get("fallback_reason") or "",
				"domain": DOMAIN, "energy_basis": BASIS, "energy_j": r["energy_trajectory_j"],
			})
		outcome_stats = [{
			"run": info["label"], "paradigm": info["paradigm"], "outcome": outcome,
			"domain": DOMAIN, "energy_basis": BASIS, "n_questions": len(outcome_groups[outcome]),
			**visualize.summary_stats(outcome_groups[outcome]),
		} for outcome in (visualize.NO_FALLBACK, visualize.FALLBACK)]

		# ---- fig4 right: mean energy before / in / after the fallback call.
		fb_rows = [r for r in qrows if r.get("fallback_operation_label") is not None]
		per_question = [{
			"run": info["label"], "paradigm": info["paradigm"], "question_id": r["question_id"],
			"fallback_reason": r.get("fallback_reason") or "",
			"fallback_operation_label": r["fallback_operation_label"], "domain": DOMAIN,
			"energy_basis": visualize.BASIS_ATTRIBUTED,
			"before_fallback_operation_j": r["fallback_before_j"],
			"fallback_operation_j": r["fallback_operation_j"],
			"after_fallback_operation_j": r.get("fallback_after_j"),
		} for r in fb_rows]
		means, n_fb = [], len(fb_rows)
		if n_fb:
			any_after = any(r.get("fallback_has_after") for r in fb_rows)
			components = {
				visualize.COMPONENT_BEFORE: [r["fallback_before_j"] for r in fb_rows],
				visualize.COMPONENT_OPERATION: [r["fallback_operation_j"] for r in fb_rows],
			}
			if any_after:
				components[visualize.COMPONENT_AFTER] = [r.get("fallback_after_j") or 0.0
				                                         for r in fb_rows]
			total = sum(sum(v) for v in components.values()) / n_fb
			labels = sorted({r["fallback_operation_label"] for r in fb_rows})
			reasons = sorted({r.get("fallback_reason") for r in fb_rows if r.get("fallback_reason")})
			for component, vals in components.items():
				mean = sum(vals) / n_fb
				means.append({
					"run": info["label"], "paradigm": info["paradigm"], "domain": DOMAIN,
					"energy_basis": visualize.BASIS_ATTRIBUTED, "component": component,
					"n_fallback_questions": n_fb, "mean_energy_j": mean,
					"pct_of_mean_fallback_question_energy": mean / total * 100.0 if total else None,
					"fallback_operation_labels": "|".join(labels),
					"fallback_reasons": "|".join(reasons),
				})

		# ---- fig3: per-question (workload variable, energy) pairs, outcome
		# coloured; the same output-token-vs-energy sanity check as before.
		vs = {}
		for variable in VS_VARIABLES:
			var_rows, pairs, by_category = [], [], defaultdict(list)
			for r in qrows:
				xv, yv = r.get(variable), r["energy_trajectory_j"]
				if xv is None or yv is None:
					continue
				category = r.get("outcome") if r.get("outcome") in OUTCOME_ORDER else ""
				var_rows.append({"run": info["label"], "paradigm": info["paradigm"],
				                 "question_id": r["question_id"], "x_variable": variable,
				                 "x": xv, "domain": DOMAIN, "energy_basis": BASIS,
				                 "energy_j": yv, "category": category})
				pairs.append((xv, yv))
				by_category[category].append((xv, yv))
			stats = []
			groups = [("(all)", pairs)] + [(c or UNANNOTATED, p)
			                               for c, p in sorted(by_category.items())]
			for category, group in groups:
				count, r_val, rho, status = visualize.correlation(group)
				stats.append({"run": info["label"], "paradigm": info["paradigm"],
				              "x_variable": variable, "y_variable": DOMAIN,
				              "energy_basis": BASIS, "scale": "raw values",
				              "category_column": "outcome", "category": category,
				              "n_pairs": count, "pearson_r": r_val, "spearman_rho": rho,
				              "status": status})
			drawn = [r for r in var_rows if r["category"]]
			vs[variable] = (drawn, stats)

		data[name] = {"run": run, "outcomes": [{"question_id": r["question_id"],
		                                        "outcome": r.get("outcome"),
		                                        "hit1": r.get("hit1")} for r in qrows],
		              "op": op_rows, "q": q_rows, "q_stats": q_stats, "flow": flow_rows,
		              "outcome_rows": outcome_rows, "outcome_stats": outcome_stats,
		              "used_outcome": used, "used_split": used, "means": means,
		              "per_question": per_question, "vs": vs}
	return data


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
	ap.add_argument("--energy-unit", default=visualize.DEFAULT_ENERGY_UNIT,
	                choices=visualize.ENERGY_UNIT_CHOICES,
	                help="unit for plotted energy; companion CSVs stay in joules "
	                     f"(default {visualize.DEFAULT_ENERGY_UNIT})")
	ap.add_argument("--rebuild-cache", action="store_true",
	                help="reprocess events_attributed.jsonl and predictions even if a "
	                     "matching cache exists, and overwrite it")
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

	plt = visualize._pyplot()
	import numpy as np
	from matplotlib.lines import Line2D

	# ---- cached per-question table, or the expensive raw path that builds it.
	# A cached render never opens events_attributed.jsonl or predict.jsonl, so
	# it also never hashes the run directories: that check only proves
	# something about files this render path does not read. outcomes_{sys}.csv
	# already exists in --out from whichever run built the cache, so a cached
	# render leaves it alone rather than reconstructing it from a narrower
	# cached row.
	cache_dir = os.path.join(out, CACHE_DIRNAME)
	cache_path = os.path.join(cache_dir, CACHE_FILENAME)
	manifest = {"cache_version": CACHE_VERSION, "dataset": args.dataset,
	            "runs": {n: resolve(d) for n, d in runs.items()},
	            "predictions": {n: resolve(p) for n, p in predictions.items()}}
	cached = ((not args.rebuild_cache) and os.path.exists(cache_path)
	         and _cache_matches(cache_dir, manifest))
	if cached:
		meta = _cache_manifest(cache_dir)
		question_rows, systems = _read_question_table(cache_path), meta["systems"]
		checks.append(f"cached: loaded {len(question_rows):,} question rows from {cache_path} "
		             f"(raw event logs and predictions were not re-read; "
		             f"pass --rebuild-cache to reprocess them)")
	else:
		if args.rebuild_cache:
			checks.append(f"rebuilding cache at {cache_dir} (--rebuild-cache)")
		elif _cache_manifest(cache_dir) is not None:
			checks.append(f"cache at {cache_dir} does not match this --run/--predictions/"
			             f"--dataset; rebuilding")
		else:
			checks.append(f"no cache at {cache_dir}; building it")
		question_rows, systems = _build_question_table(
			runs, predictions, args.dataset, resolve, out, check)
		os.makedirs(cache_dir, exist_ok=True)
		_write_question_table(cache_path, question_rows)
		with open(os.path.join(cache_dir, "meta.json"), "w") as f:
			json.dump(dict(manifest, systems=systems), f, indent=1)

	data = _assemble_data(question_rows, systems, list(systems))

	# ---- shared axis limits, computed across systems. Cheap either way (a
	# handful of min/max passes over already-materialized rows), so this
	# always runs fresh rather than being cached -- a look-and-feel change
	# such as a new axis margin never requires --rebuild-cache.
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
	with plt.rc_context(visualize.STYLE), visualize.energy_display(args.energy_unit):
		for name, d in data.items():
			key = name.lower()
			fig = visualize.draw_operation_energy(plt, [d["run"]], d["op"], DOMAIN, "j", OP_YLIM)
			save(fig, f"fig1_{key}_operation_energy",
			     [("", d["op"], visualize.OPERATION_FIELDS)])
			for linear, sfx in SCALES:
				fig = visualize.draw_question_energy(
					plt, [d["run"]], d["q"], d["q_stats"], DOMAIN, BASIS,
					["median", "mean", "p90", "p95"], linear, Q_YLIM)
				# The companion CSVs describe the data, not the scale, so they
				# are written once, with the log (default) version.
				save(fig, f"fig2_{key}_energy_ecdf{sfx}",
				     [] if linear else
				     [("", d["q"], visualize.QUESTION_FIELDS),
				      ("_stats", d["q_stats"], ("run", "paradigm", "domain", "energy_basis")
				       + visualize.STAT_FIELDS)])

			# Figure 3: one panel per workload variable, coloured by answer outcome.
			fig = draw_trajectory_properties(plt, np, name, d, XLIM, Q_YLIM, rng, False,
			                                 figure_ylabel)
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
			save(draw_trajectory_properties(plt, np, name, d, XLIM, Q_YLIM, rng, True,
			                                figure_ylabel),
			     f"fig3_{key}_trajectory_properties_linear")

			# Figure 4: both halves in one image. Only the left panel carries
			# question-level energy, so only its scale changes in the companion.
			for linear, sfx in SCALES:
				fig, (left, right) = plt.subplots(1, 2, figsize=(9.0, 4.6),
				                                  gridspec_kw={"width_ratios": [1.2, 1.0]})
				visualize.draw_outcome_energy(plt, d["used_outcome"], d["outcome_rows"], DOMAIN,
				                              BASIS, linear, Q_YLIM, ax=left)
				visualize.draw_fallback_split(plt, d["used_split"], d["means"], DOMAIN,
				                              SPLIT_YLIM, ax=right)
				left.set_title("Question energy by outcome", fontsize=10, fontweight="normal")
				right.set_title("Fallback energy split", fontsize=10, fontweight="normal")
				fig.suptitle(visualize.plot_title(visualize.PLOT_OUTCOME, DOMAIN, name),
				             fontsize=11, fontweight="bold")
				fig.tight_layout(rect=(0, 0, 1, 0.95))
				save(fig, f"fig4_{key}_outcome_and_fallback{sfx}",
				     [] if linear else
				     [("_outcome", d["outcome_rows"], visualize.OUTCOME_FIELDS),
				      ("_outcome_stats", d["outcome_stats"],
				       ("run", "paradigm", "outcome", "domain", "energy_basis", "n_questions")
				       + visualize.STAT_FIELDS[2:]),
				      ("_fallback_split", d["means"], visualize.SPLIT_FIELDS),
				      ("_fallback_split_per_question", d["per_question"],
				       visualize.SPLIT_QUESTION_FIELDS)])

			fig = visualize.draw_semantic_flow(plt, d["run"], d["flow"], DOMAIN)
			save(fig, f"fig5_{key}_semantic_flow", [("", d["flow"], visualize.FLOW_FIELDS)])

		# ---- all-system companions of figures 1-4, drawn from the same prepared
		# rows concatenated across runs. The draw functions already take a list of
		# runs, so these use the identical code path and the same shared y-limits.
		order = webqsp_answerability.ordered_systems(data)
		runs_all = [data[n]["run"] for n in order]
		cat = lambda key: [r for n in order for r in data[n][key]]  # noqa: E731

		fig = visualize.draw_operation_energy(plt, runs_all, cat("op"), DOMAIN, "j", OP_YLIM)
		save(fig, "fig1_all_systems_operation_energy")

		for linear, sfx in SCALES:
			fig = visualize.draw_question_energy(
				plt, runs_all, cat("q"), cat("q_stats"), DOMAIN, BASIS,
				["median", "mean", "p90", "p95"], linear, Q_YLIM)
			save(fig, f"fig2_all_systems_energy_ecdf{sfx}")

		# Figure 3 across systems: one COLUMN per system, left to right
		# (PoG, ToG, CoR), one ROW per workload variable. Each panel is a
		# single system's own outcome-coloured points; overlaying all three
		# systems on shared axes was too crowded to read.
		for linear, sfx in SCALES:
			fig, grid = plt.subplots(len(VS_VARIABLES), len(order), sharey=True,
			                         figsize=(3.6 * len(order), 2.6 * len(VS_VARIABLES)))
			for row_i, variable in enumerate(VS_VARIABLES):
				for col_i, name in enumerate(order):
					ax = grid[row_i, col_i]
					rows = data[name]["vs"][variable][0]
					stats = data[name]["vs"][variable][1]
					# max_traversal_depth rows already carry the jittered
					# x_plotted set by this system's own per-system figure,
					# drawn earlier in this loop.
					drawn = ([dict(r, x=r.get("x_plotted", r["x"])) for r in rows]
					         if variable == DISCRETE else rows)
					visualize.draw_energy_vs(plt, [data[name]["run"]], drawn, stats, DOMAIN,
					                         BASIS, variable, linear, Q_YLIM, ax=ax,
					                         category_order=OUTCOME_ORDER,
					                         category_style=OUTCOME_STYLE)
					ax.set_xlim(*XLIM[variable])
					ax.set_title(name if row_i == 0 else "", fontsize=10, fontweight="bold")
					if variable == DISCRETE:
						ax.set_xscale("linear")
						ax.set_xticks(sorted({int(r["x"]) for r in rows}))
						ax.grid(axis="x", visible=False)
					if not (row_i == 0 and col_i == 0):
						legend = ax.get_legend()
						if legend is not None:
							legend.remove()
			shared_label = grid[0, 0].get_ylabel()
			for ax in grid.flat:
				ax.set_ylabel("")
			fig.supylabel(shared_label, fontsize=9)
			figure_ylabel[fig] = shared_label
			fig.suptitle("Energy and Trajectory Properties", fontsize=11, fontweight="bold")
			fig.tight_layout(rect=(0, 0, 1, 0.96))
			save(fig, f"fig3_all_systems_trajectory_properties{sfx}")

		# Figure 4 across systems: one COLUMN per system, left to right
		# (PoG, ToG, CoR); top row = outcome boxplot, bottom row = fallback
		# split, each column drawn from that system's own data only.
		for linear, sfx in SCALES:
			fig, grid = plt.subplots(2, len(order), figsize=(3.9 * len(order), 8.6))
			for col_i, name in enumerate(order):
				d = data[name]
				top, bottom = grid[0, col_i], grid[1, col_i]
				visualize.draw_outcome_energy(plt, [d["run"]], d["outcome_rows"], DOMAIN, BASIS,
				                              linear, Q_YLIM, ax=top)
				visualize.draw_fallback_split(plt, [d["run"]], d["means"], DOMAIN, SPLIT_YLIM,
				                              ax=bottom)
				top.set_title(name, fontsize=10, fontweight="bold")
				# The two-line outcome labels ("Gold answer not found\n(n = ...)")
				# are wide relative to one of three columns; shrink and let each
				# column keep clear air from its neighbour instead of overlapping.
				top.tick_params(axis="x", labelsize=7.5)
				if col_i > 0:
					top.set_ylabel("")
					bottom.set_ylabel("")
				# Both rows' legends say the same thing in every column (median/
				# mean; fallback-split components); one copy, on CoR, is enough.
				if name != "CoR":
					for legend_ax in (top, bottom):
						legend = legend_ax.get_legend()
						if legend is not None:
							legend.remove()
			fig.suptitle(visualize.plot_title(visualize.PLOT_OUTCOME, DOMAIN),
			             fontsize=11, fontweight="bold")
			fig.tight_layout(rect=(0, 0, 1, 0.95))
			fig.subplots_adjust(wspace=0.5)
			save(fig, f"fig4_all_systems_outcome_and_fallback{sfx}")

		# Figure 6: WebQSP only, from the outcome CSVs just written.
		outcome_csvs = {name: os.path.join(out, f"outcomes_{name.lower()}.csv") for name in data}
		fig6 = make_fig6(plt, out, args.dataset, outcome_csvs)

	# Raw-data validation (fig1/2/4/5 sums against run.events/trajectories) has
	# already run inside _build_cache_data, the only place those are loaded; a
	# cached render has no raw events to re-check against. The figure-only
	# checks below (axis contents, panel visibility, family consistency, fig6)
	# still run on every render regardless of data source.
	families = {}
	for filename, fig in sorted(placed.items()):
		if not filename.endswith(".png"):
			continue
		axes = [ax for ax in fig.axes if ax.axison]
		family = filename.split("_")[0]
		# An all-system figure stacks or widens per run, so its size and limits are
		# compared against its own kind rather than against the per-system panels.
		if "_all_systems" in filename:
			family += "_all_systems"
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
		if family.startswith("fig3"):
			# After saving: each panel must actually carry points inside its own
			# y-limits, counted per scatter series so the paradigms stay separate.
			for index, ax in enumerate(axes, start=1):
				lo, hi = ax.get_ylim()
				series = []
				for collection in ax.collections:
					offsets = np.asarray(collection.get_offsets())
					if not len(offsets):
						continue
					y = offsets[:, 1]
					series.append(int(((y >= lo) & (y <= hi)).sum()))
				check(f"{filename}: panel {index} draws visible points in every series",
				      bool(series) and all(n > 0 for n in series),
				      f"{len(series)} series, visible per series {series}, "
				      f"total {sum(series):,}")
		if family == "fig4":
			check(f"{filename}: both halves in one image", len(axes) == 2)
		if family == "fig4_all_systems":
			check(f"{filename}: outcome + fallback panel for every system",
			      len(axes) == 2 * len(order), f"{len(axes)} axes, expected {2 * len(order)}")
		families.setdefault(family, []).append(
			(tuple(round(v, 3) for v in fig.get_size_inches()),
			 sorted({tuple(round(float(v), 6) for v in ax.get_ylim()) for ax in axes})))
	for family, entries in families.items():
		check(f"{family}: identical y-limits across systems",
		      len({str(e[1]) for e in entries}) == 1, str(entries[0][1]))
		check(f"{family}: identical figure size across systems",
		      len({e[0] for e in entries}) == 1, str(entries[0][0]))
	if fig6 is None:
		checks.append(f"SKIP  {FIG6_STEM}: WebQSP-specific, dataset is {args.dataset}")
	else:
		written, rows6, systems6 = fig6
		wa = webqsp_answerability
		check(f"{FIG6_STEM}: systems ordered PoG -> ToG -> CoR",
		      systems6 == wa.ordered_systems(data), str(systems6))
		check(f"{FIG6_STEM}: groups are 11 / 11 / 1,617 in the documented order",
		      [(r["group"], r["n"]) for r in rows6]
		      == [(g, wa.EXPECTED_COUNTS[g]) for g in wa.GROUP_ORDER])
		for name in systems6:
			hits = {r["question_id"]: r["outcome"] == export_outcomes.HIT for r in data[name]["outcomes"]}
			check(f"{FIG6_STEM}: {name} found + not found = n, and found matches outcomes_{name.lower()}.csv",
			      all(r[f"{name} found"] + r[f"{name} not found"] == r["n"] for r in rows6)
			      and sum(r[f"{name} found"] for r in rows6) == sum(hits.values()),
			      " ".join(f"{r[f'{name} found']}/{r['n']}" for r in rows6))
		for path in written:
			placed[os.path.basename(path)] = None
	if cached:
		checks.append("SKIP  run artifacts and prediction files unchanged  "
		              "(cached render never opened them; checked when the cache was built)")

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
		        "no axes (energy is ribbon thickness) |\n"
		        + ("| 6 WebQSP answerability | `fig6_webqsp_answerability_outcomes.*` | none "
		           "(gold answer found / not found) | no axes (one square per question) |\n"
		           if fig6 is not None else "") + "\n"
		        "Figure 3 colours each question by the outcome in `outcomes_{sys}.csv`, written by "
		        "`measurement/export_outcomes.py` (the repository's one evaluator: final answer "
		        "only, normalized exact match); empty-gold questions are scored as misses.\n"
		        + ("\nFigure 6 (WebQSP only) splits the 1,639 questions into the answerability "
		           "groups of `datasets/webqsp/webqsp_answerability_groups.csv` (11 official "
		           "empty-gold, 11 gold-query mismatch, 1,617 expected/reproducible) and draws one "
		           "square per question: green = gold answer found, red = gold answer not found, "
		           "from the same `outcomes_{sys}.csv`. Counts: "
		           "`fig6_webqsp_answerability_outcomes.csv`, `_compact.csv`, `.md`; per question: "
		           "`_questions.csv`.\n" if fig6 is not None else ""))
	open(os.path.join(out, "comparable_figures_checks.txt"), "w").write("\n".join(checks) + "\n")
	print("\n".join(checks))
	print(f"\nshared y-limits (J): fig1 {OP_YLIM}, fig2/3/4-left {Q_YLIM}, fig4-right {SPLIT_YLIM}")
	print(f"\n{len(placed)} files in {out}")
	return 0


if __name__ == "__main__":
	sys.exit(main())
