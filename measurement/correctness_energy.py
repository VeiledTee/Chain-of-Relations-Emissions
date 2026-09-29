"""GPU energy of fully correct vs not-fully-correct answers, by answer-set F1.

Works for any dataset this repository can score. Scoring is always the one
evaluator, chain_of_relations.eval.accuracy, called through
`export_outcomes.score`: the final answer only (never rationale text),
normalized exact matching, set-based precision / recall / F1. Only the gold
differs by dataset:

    webqsp   every official parse, best F1 over parses (chain_of_relations.eval.webqsp_canonical)
    cwq      the answer list shipped with the dataset

The official-id whitelist is WebQSP-specific. For any other dataset the question
set comes from the predictions themselves and every system must present exactly
the same ids, which is checked.


"Gold answer found" (Hit@1) and "Correct" are different outcomes and this script
keeps them apart:

    Gold answer found  at least one official answer was present in the final
                       prediction (any parse; `hit1` from export_outcomes.py)
    Correct            the complete predicted answer set achieved canonical
                       best-over-parses F1 = 1

Binary categories (the main analysis):

    correct      F1 == 1.0
    incorrect    F1 <  1.0

Three-way categories (secondary):

    fully_correct      F1 == 1.0
    partially_correct  0 < F1 < 1.0
    incorrect          F1 == 0.0

F1 comes from chain_of_relations.eval.webqsp_canonical via export_outcomes.score,
re-scored here from each predict.jsonl, so no stale outcome CSV can leak in. The
11 official empty-gold questions stay in the 1,639 denominator and take whatever
F1 the canonical evaluator gives them (0.0 for any non-empty prediction).

Energy is the whole-question `trajectory_gpu_energy_j` from trajectory_summary.csv,
never summed attributed-event energy. Tokens, LLM calls and fallback come from
events_attributed.jsonl through summarize_comparable_runs.read_events, so they
match the existing supervisor tables.

    python measurement/correctness_energy.py \
        --out ~/webqsp_supervisor_summary/correctness \
        --run PoG=measurement/runs/pog_webqsp_gemma4b_0905_0856 \
        --run ToG=measurement/runs/tog_webqsp_gemma4b_0905_0856 \
        --run CoR=measurement/runs/cor_webqsp_gemma4b_0905_0856 \
        --predictions PoG=results/pog/webqsp/gemma-3-4b-it/predict.jsonl \
        --predictions ToG=results/tog/webqsp/gemma-3-4b-it/predict.jsonl \
        --predictions CoR=results/cor/webqsp/gemma-3-4b-it/predict.jsonl

Writes into --out only. Run artifacts and predictions are hashed before and after
and the script fails if any byte changed. It also fails unless every system's
categories partition exactly the 1,639 official question ids, each with energy.
"""

import argparse
import csv
import hashlib
import json
import os
import random
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)
sys.path.insert(0, HERE)

import export_outcomes  # noqa: E402
import summarize_comparable_runs as scr  # noqa: E402
import webqsp_answerability  # noqa: E402
from agent_energy_profiler import visualize  # noqa: E402

SYSTEM_ORDER = ("PoG", "ToG", "CoR")
N_QUESTIONS = webqsp_answerability.N_QUESTIONS

CORRECT, INCORRECT = "correct", "incorrect"
FULLY, PARTIAL, WRONG = "fully_correct", "partially_correct", "incorrect"
BINARY_ORDER = (CORRECT, INCORRECT)
THREEWAY_ORDER = (FULLY, PARTIAL, WRONG)
#: The second binary group holds partially correct answers too, so it is never
#: labelled "incorrect" in anything a reader sees.
BINARY_LABELS = {CORRECT: "Fully correct (F1 = 1)",
                 INCORRECT: "Not fully correct (F1 < 1)"}
BOOTSTRAP_REPS = 2000
BOOTSTRAP_SEED = 20260920
THREEWAY_LABELS = {FULLY: "Fully correct (F1 = 1)", PARTIAL: "Partially correct (0 < F1 < 1)",
                   WRONG: "Incorrect (F1 = 0)"}
BINARY_COLOURS = {CORRECT: visualize.PALETTE[0], INCORRECT: visualize.PALETTE[1]}
THREEWAY_COLOURS = {FULLY: visualize.PALETTE[0], PARTIAL: visualize.PALETTE[3],
                    WRONG: visualize.PALETTE[1]}

DEFINITION = ("Gold answer found means at least one official answer was present in the final "
              "prediction; Correct means the complete predicted answer set achieved canonical "
              "F1 = 1.")


def binary_category(f1):
	if f1 is None:
		raise ValueError("question has no F1; every WebQSP question must be scored")
	return CORRECT if f1 == 1.0 else INCORRECT


def threeway_category(f1):
	if f1 is None:
		raise ValueError("question has no F1; every WebQSP question must be scored")
	if f1 == 1.0:
		return FULLY
	return WRONG if f1 == 0.0 else PARTIAL


def build_questions(name, run_dir, predictions_path, dataset, expected_ids, id_source):
	"""One row per question, validated against the expected id set."""
	outcomes = export_outcomes.score(predictions_path, dataset, name)
	traj = scr.read_trajectories(run_dir)
	events = scr.read_events(run_dir)

	scored_ids = [r["question_id"] for r in outcomes]
	if len(scored_ids) != len(set(scored_ids)):
		raise SystemExit(f"{name}: duplicate question ids in {predictions_path}")
	for label, ids in (("predictions", set(scored_ids)), ("trajectory_summary.csv", set(traj))):
		if ids != expected_ids:
			raise SystemExit(f"{name}: {label} has {len(ids):,} ids; expected exactly the "
			                 f"{len(expected_ids):,} {id_source} ids (missing "
			                 f"{sorted(expected_ids - ids)[:5]}, extra {sorted(ids - expected_ids)[:5]})")

	rows = []
	for outcome in outcomes:
		qid = outcome["question_id"]
		energy = traj[qid]["gpu_energy_j"]
		if energy is None:
			raise SystemExit(f"{name}: {qid} has no trajectory_gpu_energy_j")
		ev = events.get(qid, {})
		f1 = float(outcome["f1"])
		if not 0.0 <= f1 <= 1.0:
			raise SystemExit(f"{name}: {qid} F1 {f1} outside [0, 1]")
		if 1.0 - 1e-9 < f1 < 1.0 or 0.0 < f1 < 1e-9:
			raise SystemExit(f"{name}: {qid} F1 {f1!r} is within float noise of a boundary")
		rows.append({
			"system": name, "question_id": qid, "f1": f1, "hit1": outcome["hit1"],
			"precision": outcome["precision"], "recall": outcome["recall"],
			"empty_gold": outcome["empty_gold"],
			"binary_category": binary_category(f1), "threeway_category": threeway_category(f1),
			"trajectory_gpu_energy_j": energy,
			"input_tokens": ev.get("input_tokens"), "output_tokens": ev.get("output_tokens"),
			"llm_calls": ev.get("llm_calls"), "kg_calls": ev.get("kg_calls"),
			"max_traversal_depth": traj[qid]["max_traversal_depth"],
			"max_iteration": traj[qid]["max_iteration"],
			"wall_s": traj[qid]["wall_s"],
			"failed_events": ev.get("failed_events"),
			"fallback": bool(ev.get("fallback_reason")),
			"fallback_reason": ev.get("fallback_reason") or "",
			# Kept out of the CSV: only the per-operation aggregation reads it.
			"_operation_energy_j": ev.get("operation_energy_j") or {},
		})
	return rows


def validate_partition(name, rows, n_expected=N_QUESTIONS):
	"""Counts per category; raises unless both schemes cover every question once."""
	n = len(rows)
	binary = {c: sum(1 for r in rows if r["binary_category"] == c) for c in BINARY_ORDER}
	three = {c: sum(1 for r in rows if r["threeway_category"] == c) for c in THREEWAY_ORDER}
	problems = []
	if n != n_expected:
		problems.append(f"{n} questions, expected {n_expected}")
	if len({r["question_id"] for r in rows}) != n:
		problems.append("duplicate question ids")
	if sum(binary.values()) != n:
		problems.append(f"binary counts {binary} sum to {sum(binary.values())}")
	if sum(three.values()) != n:
		problems.append(f"three-way counts {three} sum to {sum(three.values())}")
	if binary[CORRECT] != three[FULLY]:
		problems.append("correct != fully correct")
	if binary[INCORRECT] != three[PARTIAL] + three[WRONG]:
		problems.append("incorrect != partially correct + incorrect")
	if problems:
		raise SystemExit(f"{name}: partition check failed: " + "; ".join(problems))
	return binary, three


def energies(rows):
	return [r["trajectory_gpu_energy_j"] for r in rows]


def bootstrap_difference_ci(a, b, stat, reps=BOOTSTRAP_REPS, seed=BOOTSTRAP_SEED):
	"""95% percentile bootstrap CI for stat(b) - stat(a), resampling each group.

	Both groups are resampled independently with replacement, which is the right
	scheme here: a question belongs to exactly one outcome group and the groups
	are not paired.
	"""
	if not a or not b:
		return None, None
	rng = random.Random(seed)
	draws = []
	for _ in range(reps):
		ra = [a[rng.randrange(len(a))] for _ in range(len(a))]
		rb = [b[rng.randrange(len(b))] for _ in range(len(b))]
		draws.append(stat(rb) - stat(ra))
	draws.sort()
	return draws[int(0.025 * reps)], draws[int(0.975 * reps) - 1]


def binary_row(name, rows):
	"""One row: fully correct vs not fully correct, energy and workload."""
	correct = [r for r in rows if r["binary_category"] == CORRECT]
	incorrect = [r for r in rows if r["binary_category"] == INCORRECT]
	ec, ei = energies(correct), energies(incorrect)
	mean_c, mean_i = scr.mean(ec), scr.mean(ei)
	median_c, median_i = scr.median(ec), scr.median(ei)
	ratio = mean_i / mean_c if mean_c else None
	median_ratio = median_i / median_c if median_c else None
	mean_lo, mean_hi = bootstrap_difference_ci(ec, ei, scr.mean)
	med_lo, med_hi = bootstrap_difference_ci(ec, ei, scr.median)
	row = {
		"system": name, "correct_n": len(correct), "incorrect_n": len(incorrect),
		"mean_gpu_j_correct": mean_c, "median_gpu_j_correct": median_c,
		"p90_gpu_j_correct": scr.percentile(ec, 0.90),
		"mean_gpu_j_incorrect": mean_i, "median_gpu_j_incorrect": median_i,
		"p90_gpu_j_incorrect": scr.percentile(ei, 0.90),
		"incorrect_correct_energy_ratio": ratio,
		"percent_difference": (ratio - 1.0) * 100.0 if ratio is not None else None,
		"median_ratio": median_ratio,
		"median_percent_difference": ((median_ratio - 1.0) * 100.0
		                              if median_ratio is not None else None),
		"mean_difference_j": (mean_i - mean_c) if None not in (mean_i, mean_c) else None,
		"mean_difference_ci_low": mean_lo, "mean_difference_ci_high": mean_hi,
		"median_difference_j": (median_i - median_c) if None not in (median_i, median_c) else None,
		"median_difference_ci_low": med_lo, "median_difference_ci_high": med_hi,
	}
	for key, suffix in (("input_tokens", "input_tokens"), ("output_tokens", "output_tokens"),
	                    ("llm_calls", "llm_calls"), ("kg_calls", "kg_calls"),
	                    ("max_traversal_depth", "depth"), ("max_iteration", "iteration"),
	                    ("wall_s", "wall_s")):
		row[f"mean_{suffix}_correct"] = scr.mean([r[key] for r in correct])
		row[f"mean_{suffix}_incorrect"] = scr.mean([r[key] for r in incorrect])
	row["fallback_rate_pct_correct"] = scr.pct(sum(1 for r in correct if r["fallback"]),
	                                           len(correct))
	row["fallback_rate_pct_incorrect"] = scr.pct(sum(1 for r in incorrect if r["fallback"]),
	                                             len(incorrect))
	return row


def threeway_rows(name, rows):
	out = []
	for category in THREEWAY_ORDER:
		group = [r for r in rows if r["threeway_category"] == category]
		out.append({
			"system": name, "category": THREEWAY_LABELS[category], "n": len(group),
			"mean_gpu_j": scr.mean(energies(group)), "median_gpu_j": scr.median(energies(group)),
			"p90_gpu_j": scr.percentile(energies(group), 0.90),
			"mean_input_tokens": scr.mean([r["input_tokens"] for r in group]),
			"mean_output_tokens": scr.mean([r["output_tokens"] for r in group]),
			"mean_llm_calls": scr.mean([r["llm_calls"] for r in group]),
			"mean_kg_calls": scr.mean([r["kg_calls"] for r in group]),
			"mean_max_traversal_depth": scr.mean([r["max_traversal_depth"] for r in group]),
			"mean_max_iteration": scr.mean([r["max_iteration"] for r in group]),
			"mean_wall_s": scr.mean([r["wall_s"] for r in group]),
			"fallback_rate_pct": scr.pct(sum(1 for r in group if r["fallback"]), len(group)),
			"gpu_j_per_output_token": (
				scr.mean(energies(group)) / scr.mean([r["output_tokens"] for r in group])
				if scr.mean([r["output_tokens"] for r in group]) else None),
		})
	return out


def operation_energy_rows(name, rows):
	"""Mean per-question GPU energy by operation label, within each outcome group.

	Attributed event energy, which is the only basis that can be split by
	operation. It is therefore not the whole-question trajectory counter used
	everywhere else, and the two are not mixed in one column.
	"""
	labels = sorted({label for r in rows for label in r["_operation_energy_j"]})
	out = []
	for category in THREEWAY_ORDER:
		group = [r for r in rows if r["threeway_category"] == category]
		if not group:
			continue
		total = sum(sum(r["_operation_energy_j"].values()) for r in group)
		for label in labels:
			energy = sum(r["_operation_energy_j"].get(label, 0.0) for r in group)
			out.append({"system": name, "category": THREEWAY_LABELS[category],
			            "operation_label": label, "n_questions": len(group),
			            "total_attributed_gpu_j": energy,
			            "mean_attributed_gpu_j_per_question": energy / len(group),
			            "share_of_group_attributed_pct": scr.pct(energy, total)})
	return out


def depth_rows(name, rows):
	"""Traversal depth x correctness category: does deeper search pay off?"""
	out = []
	depths = sorted({r["max_traversal_depth"] for r in rows
	                 if r["max_traversal_depth"] is not None})
	for depth in depths:
		group = [r for r in rows if r["max_traversal_depth"] == depth]
		counts = {c: sum(1 for r in group if r["threeway_category"] == c)
		          for c in THREEWAY_ORDER}
		out.append({
			"system": name, "max_traversal_depth": depth, "n": len(group),
			"fully_correct": counts[FULLY], "partially_correct": counts[PARTIAL],
			"incorrect": counts[WRONG],
			"fully_correct_pct": scr.pct(counts[FULLY], len(group)),
			"mean_gpu_j": scr.mean(energies(group)),
			"median_gpu_j": scr.median(energies(group)),
			"mean_llm_calls": scr.mean([r["llm_calls"] for r in group]),
		})
	return out


def heavy_tail_rows(name, rows, share=0.10):
	"""What the most expensive questions are: the top `share` vs the rest."""
	ranked = sorted(rows, key=lambda r: -r["trajectory_gpu_energy_j"])
	cut = max(1, int(round(share * len(ranked))))
	total = sum(energies(rows))
	out = []
	for label, group in ((f"top {share:.0%} by energy", ranked[:cut]),
	                     ("remaining questions", ranked[cut:])):
		counts = {c: sum(1 for r in group if r["threeway_category"] == c)
		          for c in THREEWAY_ORDER}
		out.append({
			"system": name, "bucket": label, "n": len(group),
			"share_of_total_gpu_energy_pct": scr.pct(sum(energies(group)), total),
			"mean_gpu_j": scr.mean(energies(group)),
			"fully_correct": counts[FULLY], "partially_correct": counts[PARTIAL],
			"incorrect": counts[WRONG],
			"fully_correct_pct": scr.pct(counts[FULLY], len(group)),
			"fallback_rate_pct": scr.pct(sum(1 for r in group if r["fallback"]), len(group)),
			"mean_max_traversal_depth": scr.mean([r["max_traversal_depth"] for r in group]),
			"mean_llm_calls": scr.mean([r["llm_calls"] for r in group]),
		})
	return out


def hit_crosstab_rows(name, rows):
	"""Gold found (Hit@1) vs three-way correctness: shows the two are not the same."""
	out = []
	for category in THREEWAY_ORDER:
		group = [r for r in rows if r["threeway_category"] == category]
		found = sum(1 for r in group if r["hit1"] == 1)
		out.append({"system": name, "category": THREEWAY_LABELS[category],
		            "gold_found": found, "gold_not_found": len(group) - found,
		            "total": len(group)})
	return out


#: The headline table: n, mean, median, p90, ratio and percent difference per group.
BINARY_FIELDS = ("system", "correct_n", "mean_gpu_j_correct", "median_gpu_j_correct",
                 "p90_gpu_j_correct", "incorrect_n", "mean_gpu_j_incorrect",
                 "median_gpu_j_incorrect", "p90_gpu_j_incorrect",
                 "incorrect_correct_energy_ratio", "percent_difference",
                 "median_ratio", "median_percent_difference",
                 "mean_difference_j", "mean_difference_ci_low", "mean_difference_ci_high",
                 "median_difference_j", "median_difference_ci_low", "median_difference_ci_high")
BINARY_HEADERS = ("System", "Fully correct n", "Fully correct mean J",
                  "Fully correct median J", "Fully correct p90 J",
                  "Not fully correct n", "Not fully correct mean J",
                  "Not fully correct median J", "Not fully correct p90 J",
                  "Mean ratio", "Mean % difference", "Median ratio", "Median % difference",
                  "Mean diff J", "Mean diff CI low", "Mean diff CI high",
                  "Median diff J", "Median diff CI low", "Median diff CI high")
BINARY_DECIMALS = (None, 0, 0, 0, 0, 0, 0, 0, 0, 3, 1, 3, 1, 0, 0, 0, 0, 0, 0)
#: Why the groups differ: the workload each one performed.
WORKLOAD_FIELDS = ("system", "mean_input_tokens_correct", "mean_input_tokens_incorrect",
                   "mean_output_tokens_correct", "mean_output_tokens_incorrect",
                   "mean_llm_calls_correct", "mean_llm_calls_incorrect",
                   "mean_kg_calls_correct", "mean_kg_calls_incorrect",
                   "mean_depth_correct", "mean_depth_incorrect",
                   "mean_iteration_correct", "mean_iteration_incorrect",
                   "mean_wall_s_correct", "mean_wall_s_incorrect",
                   "fallback_rate_pct_correct", "fallback_rate_pct_incorrect")
WORKLOAD_HEADERS = ("System", "Input tok — full", "Input tok — not full",
                    "Output tok — full", "Output tok — not full",
                    "LLM calls — full", "LLM calls — not full",
                    "KG calls — full", "KG calls — not full",
                    "Depth — full", "Depth — not full",
                    "Iterations — full", "Iterations — not full",
                    "Wall s — full", "Wall s — not full",
                    "Fallback % — full", "Fallback % — not full")
WORKLOAD_DECIMALS = (None, 0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2, 1, 1, 1, 1)
THREEWAY_FIELDS = ("system", "category", "n", "mean_gpu_j", "median_gpu_j", "p90_gpu_j",
                   "mean_input_tokens", "mean_output_tokens", "mean_llm_calls",
                   "mean_kg_calls", "mean_max_traversal_depth", "mean_max_iteration",
                   "mean_wall_s", "fallback_rate_pct", "gpu_j_per_output_token")
THREEWAY_HEADERS = ("System", "Category", "n", "Mean GPU J", "Median GPU J", "p90 GPU J",
                    "Mean input tokens", "Mean output tokens", "Mean LLM calls",
                    "Mean KG calls", "Mean max traversal depth", "Mean max iteration",
                    "Mean wall s", "Fallback rate (%)", "GPU J / output token")
THREEWAY_DECIMALS = (None, None, 0, 0, 0, 0, 0, 0, 1, 1, 2, 2, 1, 1, 2)
DEPTH_FIELDS = ("system", "max_traversal_depth", "n", "fully_correct", "partially_correct",
                "incorrect", "fully_correct_pct", "mean_gpu_j", "median_gpu_j",
                "mean_llm_calls")
HEAVY_TAIL_FIELDS = ("system", "bucket", "n", "share_of_total_gpu_energy_pct", "mean_gpu_j",
                     "fully_correct", "partially_correct", "incorrect", "fully_correct_pct",
                     "fallback_rate_pct", "mean_max_traversal_depth", "mean_llm_calls")
OPERATION_FIELDS = ("system", "category", "operation_label", "n_questions",
                    "total_attributed_gpu_j", "mean_attributed_gpu_j_per_question",
                    "share_of_group_attributed_pct")
QUESTION_FIELDS = ("system", "question_id", "f1", "hit1", "precision", "recall", "empty_gold",
                   "binary_category", "threeway_category", "trajectory_gpu_energy_j",
                   "input_tokens", "output_tokens", "llm_calls", "kg_calls",
                   "max_traversal_depth", "max_iteration", "wall_s", "failed_events",
                   "fallback", "fallback_reason")


def cell(value, decimals):
	if value is None:
		return ""
	if decimals is None or isinstance(value, str):
		return str(value)
	if decimals == 0:
		return f"{value:,.0f}"
	return f"{value:,.{decimals}f}"


def write_csv(path, rows, fields):
	"""Full-precision CSV: formatting belongs to the Markdown view."""
	with open(path, "w", newline="") as f:
		writer = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore", lineterminator="\n")
		writer.writeheader()
		for row in rows:
			writer.writerow({k: ("" if row.get(k) is None else row[k]) for k in fields})
	return path


def markdown_table(rows, fields, headers, decimals):
	lines = ["| " + " | ".join(headers) + " |", "|" + "---|" * len(headers)]
	for row in rows:
		lines.append("| " + " | ".join(cell(row.get(k), d) for k, d in zip(fields, decimals)) + " |")
	return lines


def _box_panel(ax, groups, order, labels, colours, linear=False):
	"""Boxplot (whiskers 1.5 IQR, outliers as faint points) with mean and median marked."""
	data = [[visualize.to_display_energy(v) for v in energies(groups[c])] for c in order]
	positions = list(range(1, len(order) + 1))
	box = ax.boxplot(data, positions=positions, widths=0.5, patch_artist=True,
	                 showfliers=True, medianprops={"color": visualize.INK, "linewidth": 1.8},
	                 flierprops={"marker": "o", "markersize": 2.2, "alpha": 0.25,
	                             "markeredgewidth": 0},
	                 whiskerprops={"color": visualize.INK2}, capprops={"color": visualize.INK2})
	for patch, category in zip(box["boxes"], order):
		patch.set_facecolor(colours[category])
		patch.set_alpha(0.55)
		patch.set_edgecolor(visualize.INK2)
	for flier, category in zip(box["fliers"], order):
		flier.set_markerfacecolor(colours[category])
	for pos, values in zip(positions, data):
		if not values:
			continue
		m, md = scr.mean(values), scr.median(values)
		ax.scatter([pos], [m], marker="D", s=34, color="white", edgecolor=visualize.INK,
		           linewidth=1.2, zorder=5)
		# `values` is already in the display unit, so the annotation formats it
		# with the precision that unit needs rather than always whole joules.
		fmt = visualize.format_display_energy
		ax.annotate(f"mean {fmt(m)}\nmedian {fmt(md)}", (pos + 0.28, md), fontsize=6.5,
		            color=visualize.INK2, va="center", ha="left")
	ax.set_xticks(positions)
	ax.set_xticklabels([f"{labels[c].replace(' (', chr(10) + '(')}\nn = {len(groups[c]):,}"
	                    for c in order], fontsize=7.5)
	ax.set_xlim(0.4, len(order) + 1.15)
	if not linear:
		ax.set_yscale("log")
	ax.yaxis.set_major_formatter(visualize._energy_formatter())
	ax.grid(axis="x", visible=False)


def draw_figure(plt, systems, per_system, scheme, title, caption, linear=False):
	from matplotlib.lines import Line2D
	order, labels, colours, key = ((BINARY_ORDER, BINARY_LABELS, BINARY_COLOURS, "binary_category")
	                               if scheme == "binary" else
	                               (THREEWAY_ORDER, THREEWAY_LABELS, THREEWAY_COLOURS,
	                                "threeway_category"))
	width = 4.3 if scheme == "binary" else 6.2
	fig, axes = plt.subplots(1, len(systems), figsize=(width * len(systems), 4.6), sharey=True)
	axes = [axes] if len(systems) == 1 else list(axes)
	for ax, name in zip(axes, systems):
		rows = per_system[name]
		groups = {c: [r for r in rows if r[key] == c] for c in order}
		_box_panel(ax, groups, order, labels, colours, linear)
		ax.set_title(f"{name}  (n = {len(rows):,})", fontsize=10, fontweight="bold")
	axes[0].set_ylabel(f"GPU energy per question ({visualize.display_energy_unit()}, "
	                   f"{'linear' if linear else 'log'} scale)")
	handles = [Line2D([], [], color=visualize.INK, linewidth=1.8, label="Median (box line)"),
	           Line2D([], [], marker="D", linestyle="", markerfacecolor="white",
	                  markeredgecolor=visualize.INK, label="Mean")]
	fig.legend(handles=handles, loc="lower center", ncol=2, frameon=False, fontsize=8,
	           bbox_to_anchor=(0.5, -0.06))
	fig.suptitle(title, fontsize=11, fontweight="bold")
	# No caption under the figure: the provenance lives in CORRECTNESS.md and the
	# companion CSVs, and the figure is read on its own.
	fig.tight_layout()
	return fig


def tree_digest(paths):
	h = hashlib.sha256()
	for path in paths:
		files = ([os.path.join(d, f) for d, _, fs in os.walk(path) for f in fs]
		         if os.path.isdir(path) else [path])
		for file in sorted(files):
			h.update(file.encode())
			with open(file, "rb") as f:
				h.update(f.read())
	return h.hexdigest()


def main(argv=None):
	ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
	ap.add_argument("--run", action="append", metavar="NAME=DIR", required=True)
	ap.add_argument("--predictions", action="append", metavar="NAME=FILE", required=True)
	ap.add_argument("--dataset", default="webqsp")
	ap.add_argument("--expect-questions", type=int, default=None,
	                help="required question count per system; default 1,639 for WebQSP, "
	                     "otherwise whatever the predictions hold (still checked identical "
	                     "across systems)")
	ap.add_argument("--scorer-label", default="",
	                help="how the scorer is named in CORRECTNESS.md, e.g. "
	                     "'exact-match answer-set evaluation'")
	ap.add_argument("--energy-unit", default=visualize.DEFAULT_ENERGY_UNIT,
	                choices=visualize.ENERGY_UNIT_CHOICES,
	                help="unit for plotted energy; tables stay in joules "
	                     f"(default {visualize.DEFAULT_ENERGY_UNIT})")
	ap.add_argument("--out", required=True)
	args = ap.parse_args(argv)

	dataset = args.dataset.strip().lower()
	runs = scr.parse_pairs(args.run, "--run")
	predictions = scr.parse_pairs(args.predictions, "--predictions")
	if set(runs) != set(predictions):
		raise SystemExit("--run and --predictions must name the same systems")
	systems = scr.order_systems(runs)
	out = os.path.expanduser(args.out)
	for path in list(runs.values()) + list(predictions.values()):
		if os.path.realpath(out).startswith(os.path.realpath(path)):
			raise SystemExit(f"--out must not be inside an input: {path}")
	os.makedirs(out, exist_ok=True)

	inputs = [runs[s] for s in systems] + [predictions[s] for s in systems]
	before = tree_digest(inputs)

	if dataset == "webqsp":
		expected_ids = set(webqsp_answerability.official_question_ids())
		id_source = "official"
		n_expected = args.expect_questions or N_QUESTIONS
		if len(expected_ids) != N_QUESTIONS:
			raise SystemExit(f"official gold has {len(expected_ids)} ids, "
			                 f"expected {N_QUESTIONS}")
	else:
		# No official id whitelist for this dataset: take the id set from the first
		# system's predictions and require every other system to match it exactly.
		id_source = f"{systems[0]} prediction"
		by_system = {}
		for name in systems:
			with open(predictions[name]) as f:
				by_system[name] = {json.loads(line)["id"] for line in f if line.strip()}
		expected_ids = by_system[systems[0]]
		for name in systems[1:]:
			if by_system[name] != expected_ids:
				raise SystemExit(
					f"{name}: question-id set differs from {systems[0]} "
					f"(missing {sorted(expected_ids - by_system[name])[:5]}, "
					f"extra {sorted(by_system[name] - expected_ids)[:5]})")
		n_expected = args.expect_questions or len(expected_ids)
	if len(expected_ids) != n_expected:
		raise SystemExit(f"{len(expected_ids):,} {id_source} ids, "
		                 f"but --expect-questions is {n_expected:,}")

	per_system, binary, three, cross, operations, checks = {}, [], [], [], [], []
	depths, heavy = [], []
	for name in systems:
		rows = build_questions(name, runs[name], predictions[name], args.dataset,
		                       expected_ids, id_source)
		b_counts, t_counts = validate_partition(name, rows, n_expected)
		per_system[name] = rows
		binary.append(binary_row(name, rows))
		three.extend(threeway_rows(name, rows))
		cross.extend(hit_crosstab_rows(name, rows))
		operations.extend(operation_energy_rows(name, rows))
		depths.extend(depth_rows(name, rows))
		heavy.extend(heavy_tail_rows(name, rows))
		empty = [r for r in rows if r["empty_gold"]]
		checks.append(
			f"{name}: n={len(rows):,} ids=={id_source}  "
			f"binary {b_counts[CORRECT]:,}+{b_counts[INCORRECT]:,}={sum(b_counts.values()):,}  "
			f"three-way {t_counts[FULLY]:,}+{t_counts[PARTIAL]:,}+{t_counts[WRONG]:,}="
			f"{sum(t_counts.values()):,}  empty_gold={len(empty)} "
			f"(categories: {sorted({r['threeway_category'] for r in empty})})  "
			f"mean F1={100 * scr.mean([r['f1'] for r in rows]):.2f}%  "
			f"Hit@1={100 * scr.mean([r['hit1'] for r in rows]):.2f}%")

	written = [
		write_csv(os.path.join(out, "correctness_binary.csv"), binary, BINARY_FIELDS),
		write_csv(os.path.join(out, "correctness_workload.csv"), binary, WORKLOAD_FIELDS),
		write_csv(os.path.join(out, "correctness_threeway.csv"), three, THREEWAY_FIELDS),
		write_csv(os.path.join(out, "correctness_operation_energy.csv"), operations,
		          OPERATION_FIELDS),
		write_csv(os.path.join(out, "correctness_depth.csv"), depths, DEPTH_FIELDS),
		write_csv(os.path.join(out, "correctness_heavy_tail.csv"), heavy, HEAVY_TAIL_FIELDS),
		write_csv(os.path.join(out, "correctness_vs_gold_found.csv"), cross,
		          ("system", "category", "gold_found", "gold_not_found", "total")),
		write_csv(os.path.join(out, "correctness_questions.csv"),
		          [r for s in systems for r in per_system[s]], QUESTION_FIELDS),
	]

	scorer = args.scorer_label or (
		"canonical WebQSP best-over-parses F1" if dataset == "webqsp"
		else f"the repository's `{dataset}` answer-set evaluator")
	caption = (f"Whole-question trajectory GPU energy; {scorer}; all {n_expected:,} "
	           f"{dataset} questions per system. Box = IQR, whiskers = 1.5 IQR.")
	plt = visualize._pyplot()
	with plt.rc_context(visualize.STYLE), visualize.energy_display(args.energy_unit):
		for stem, scheme, title in (
				("fig_correct_vs_incorrect_gpu_energy", "binary",
				 "Fully correct vs Not fully correct GPU energy per question"),
				("fig_fully_partial_incorrect_gpu_energy", "threeway",
				 "Fully correct vs Partially correct vs Incorrect GPU energy per question")):
			# Same data, same layout, same colours: only the y-scale differs.
			for linear, suffix in ((False, ""), (True, "_linear")):
				fig = draw_figure(plt, systems, per_system, scheme, title, caption, linear)
				written += visualize.save_figure(fig, out, stem + suffix)
				plt.close(fig)

	after = tree_digest(inputs)
	if before != after:
		raise SystemExit("input run artifacts or predictions changed during the analysis")
	checks.append(f"inputs unchanged: sha256 {before}")

	lines = [f"# {dataset} fully correct vs not fully correct GPU energy", "", DEFINITION, "",
	         f"Scorer: {scorer}. Fully correct = F1 == 1.0; Not fully correct = F1 < 1.0, "
	         "which includes partially correct answers and is therefore never called "
	         "\"incorrect\". Energy = whole-question `trajectory_gpu_energy_j`, never summed "
	         f"event energy. All {n_expected:,} questions per system.", "",
	         "## Binary", ""]
	lines += markdown_table(binary, BINARY_FIELDS, BINARY_HEADERS, BINARY_DECIMALS)
	lines += ["", "Ratio = not-fully-correct / fully correct; percent difference = "
	          "(ratio - 1) x 100. Difference CIs are 95% percentile bootstrap "
	          f"({BOOTSTRAP_REPS:,} resamples, seed {BOOTSTRAP_SEED}), each group resampled "
	          "independently; an interval excluding 0 is a difference the resampling supports.",
	          "", "## Workload by outcome group", ""]
	lines += markdown_table(binary, WORKLOAD_FIELDS, WORKLOAD_HEADERS, WORKLOAD_DECIMALS)
	lines += ["", "Observed workload differences, not a causal claim.",
	          "", "## Three-way", ""]
	lines += markdown_table(three, THREEWAY_FIELDS, THREEWAY_HEADERS, THREEWAY_DECIMALS)
	lines += ["", "## Traversal depth vs correctness", ""]
	lines += markdown_table(depths, DEPTH_FIELDS,
	                        ("System", "Depth", "n", "Fully", "Partial", "Incorrect",
	                         "Fully correct (%)", "Mean GPU J", "Median GPU J",
	                         "Mean LLM calls"),
	                        (None, 0, 0, 0, 0, 0, 1, 0, 0, 1))
	lines += ["", "## Heavy tail: the most expensive 10% of questions", ""]
	lines += markdown_table(heavy, HEAVY_TAIL_FIELDS,
	                        ("System", "Bucket", "n", "Share of total GPU energy (%)",
	                         "Mean GPU J", "Fully", "Partial", "Incorrect",
	                         "Fully correct (%)", "Fallback (%)", "Mean depth",
	                         "Mean LLM calls"),
	                        (None, None, 0, 1, 0, 0, 0, 0, 1, 1, 2, 1))
	lines += ["", "## Diagnostic: gold answer found (Hit@1) within each correctness category", ""]
	lines += markdown_table(cross, ("system", "category", "gold_found", "gold_not_found", "total"),
	                        ("System", "Category", "Gold found", "Gold not found", "Total"),
	                        (None, None, 0, 0, 0))
	lines += ["", "## Validation", ""] + [f"- {c}" for c in checks] + [""]
	lines += ["## Inputs", ""] + [f"- {s}: `{runs[s]}`, `{predictions[s]}`" for s in systems] + [""]
	md = os.path.join(out, "CORRECTNESS.md")
	with open(md, "w", newline="") as f:
		f.write("\n".join(lines))
	written.append(md)

	for c in checks:
		print(c)
	for path in written:
		print(f"  -> {path}")
	return 0


if __name__ == "__main__":
	sys.exit(main())
