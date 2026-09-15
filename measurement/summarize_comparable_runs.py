"""Supervisor-level summary statistics for comparable runs, any dataset.

Reads what other tools already produced and aggregates it. Nothing here
re-measures energy and nothing here re-scores an answer:

    run directory      per-question energy, depth, tokens, LLM calls, fallback
                       (trajectory_summary.csv + events_attributed.jsonl)
    outcome CSV        per-question hit/miss and F1, written by
                       measurement/export_outcomes.py, which calls THIS
                       repository's evaluator for the dataset in question

Effectiveness is therefore the dataset's own evaluator: canonical best-F1-over-
parses for WebQSP, the plain answer-set evaluator for CWQ. This script only
averages the per-question values that evaluator produced. Use --effectiveness
to paste headline numbers from `python -m chain_of_relations.eval.eval` instead
when you want the evaluator's own aggregate verbatim.

    python measurement/summarize_comparable_runs.py \
        --out ~/webqsp_summary_tables \
        --run PoG=measurement/runs/pog_webqsp_gemma4b_0905_0856 \
        --run ToG=measurement/runs/tog_webqsp_gemma4b_0905_0856 \
        --run CoR=measurement/runs/cor_webqsp_gemma4b_0905_0856 \
        --outcomes PoG=~/webqsp_supervisor_summary/outcomes_pog.csv \
        --outcomes ToG=~/webqsp_supervisor_summary/outcomes_tog.csv \
        --outcomes CoR=~/webqsp_supervisor_summary/outcomes_cor.csv

Writes into --out only, and never touches a run artifact or a prediction file:

    summary_overview.csv          one row per metric, one column per system
    summary_depth_outcome.csv     depth x outcome, one row per (system, depth)
    summary_fallback.csv          one row per metric, one column per system
    summary_fallback_reasons.csv  one row per (system, fallback reason)
    SUMMARY.md                    the same tables in Markdown

Energy is the whole-question trajectory counter value, not summed event energy,
except where a metric is explicitly about attributed events (the fallback phase
split). A domain the hardware did not report stays absent rather than zero.
"""

import argparse
import collections
import csv
import json
import math
import os
import statistics
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)

#: Display order. Systems are rows in the per-row tables and columns elsewhere.
SYSTEM_ORDER = ("PoG", "ToG", "CoR")
#: Fraction of questions in the "most expensive" bucket.
TOP_SHARE = 0.10
HIT, MISS = "hit", "miss"


def parse_pairs(values, flag):
	out = {}
	for item in values or ():
		if "=" not in item:
			raise SystemExit(f"expected NAME=VALUE for {flag}, got {item!r}")
		name, value = item.split("=", 1)
		out[name.strip()] = os.path.expanduser(value.strip())
	return out


def order_systems(names):
	"""SYSTEM_ORDER first, then anything else alphabetically."""
	known = [n for n in SYSTEM_ORDER if n in names]
	return known + sorted(n for n in names if n not in SYSTEM_ORDER)


def pct(part, whole):
	return 100.0 * part / whole if whole else None


def mean(values):
	values = [v for v in values if v is not None]
	return statistics.mean(values) if values else None


def median(values):
	values = [v for v in values if v is not None]
	return statistics.median(values) if values else None


def percentile(values, q):
	values = sorted(v for v in values if v is not None)
	if not values:
		return None
	index = min(len(values) - 1, max(0, int(round(q * (len(values) - 1)))))
	return values[index]


def read_trajectories(run_dir):
	"""question_id -> {gpu_energy_j, max_traversal_depth} from trajectory_summary.csv."""
	path = os.path.join(run_dir, "trajectory_summary.csv")
	if not os.path.exists(path):
		raise SystemExit(f"{run_dir}: no trajectory_summary.csv (not an attributed run?)")
	out = {}
	with open(path) as f:
		for row in csv.DictReader(f):
			energy = row.get("trajectory_gpu_energy_j")
			depth = row.get("max_traversal_depth")
			out[row["question_id"]] = {
				"gpu_energy_j": float(energy) if energy not in (None, "") else None,
				"max_traversal_depth": int(depth) if depth not in (None, "") else None,
			}
	return out


def read_events(run_dir):
	"""Per-question workload and fallback structure from events_attributed.jsonl.

	Returns question_id -> dict with llm_calls, input/output tokens, the fallback
	reason (None when the question has no fallback event) and the attributed GPU
	energy split into before / in / after the fallback event.
	"""
	path = os.path.join(run_dir, "events_attributed.jsonl")
	if not os.path.exists(path):
		raise SystemExit(f"{run_dir}: no events_attributed.jsonl (run not attributed?)")
	by_question = collections.defaultdict(list)
	with open(path) as f:
		for line in f:
			line = line.strip()
			if not line:
				continue
			event = json.loads(line)
			by_question[event.get("question_id")].append(event)

	out = {}
	for question_id, events in by_question.items():
		events.sort(key=lambda e: (e.get("step_index")
		                           if e.get("step_index") is not None else -1))
		llm_calls = input_tokens = output_tokens = 0
		for event in events:
			if event.get("operation_type") == "llm":
				llm_calls += 1
				input_tokens += event.get("input_tokens") or 0
				output_tokens += event.get("output_tokens") or 0
		marks = [i for i, e in enumerate(events) if (e.get("meta") or {}).get("fallback") is True]
		energy = lambda items: sum(e.get("gpu_energy_j") or 0.0 for e in items)  # noqa: E731
		record = {"llm_calls": llm_calls, "input_tokens": input_tokens,
		          "output_tokens": output_tokens, "fallback_reason": None,
		          "energy_before": None, "energy_in": None, "energy_after": None}
		if marks:
			first, last = marks[0], marks[-1]
			record["fallback_reason"] = ((events[first].get("meta") or {})
			                             .get("fallback_reason") or "(unlabelled)")
			record["energy_before"] = energy(events[:first])
			record["energy_in"] = energy(events[i] for i in marks)
			record["energy_after"] = energy(events[last + 1:])
		out[question_id] = record
	return out


def read_outcomes(path):
	"""question_id -> {outcome, f1, empty_gold} from an export_outcomes.py CSV."""
	path = os.path.expanduser(path)
	if not os.path.exists(path):
		raise SystemExit(f"missing outcome CSV: {path}")
	out = {}
	with open(path) as f:
		reader = csv.DictReader(f)
		for column in ("question_id", "outcome"):
			if column not in (reader.fieldnames or ()):
				raise SystemExit(f"{path}: outcome CSV has no {column!r} column")
		for row in reader:
			f1 = row.get("f1")
			out[row["question_id"]] = {
				"outcome": row["outcome"],
				"f1": float(f1) if f1 not in (None, "") else None,
				"hit1": float(row["hit1"]) if row.get("hit1") not in (None, "") else None,
				"empty_gold": str(row.get("empty_gold", "")).strip().lower() == "true",
			}
	return out


def summarize(name, run_dir, outcomes_path, effectiveness=None):
	"""Every statistic for one system."""
	traj = read_trajectories(run_dir)
	events = read_events(run_dir)
	outcomes = read_outcomes(outcomes_path)

	questions = sorted(set(traj) & set(outcomes))
	missing_energy = sorted(set(outcomes) - set(traj))
	missing_outcome = sorted(set(traj) - set(outcomes))

	energy_of = lambda q: traj[q]["gpu_energy_j"]  # noqa: E731
	energies = [energy_of(q) for q in questions]
	total_energy = sum(e for e in energies if e is not None)

	hits = [q for q in questions if outcomes[q]["outcome"] == HIT]
	misses = [q for q in questions if outcomes[q]["outcome"] == MISS]

	ranked = sorted((e for e in energies if e is not None), reverse=True)
	top_n = math.ceil(TOP_SHARE * len(ranked)) if ranked else 0
	top_share = pct(sum(ranked[:top_n]), total_energy) if ranked else None

	fallback = [q for q in questions if events.get(q, {}).get("fallback_reason")]
	non_fallback = [q for q in questions if q not in set(fallback)]
	fb_hits = [q for q in fallback if outcomes[q]["outcome"] == HIT]
	nf_hits = [q for q in non_fallback if outcomes[q]["outcome"] == HIT]

	before = [events[q]["energy_before"] for q in fallback]
	inside = [events[q]["energy_in"] for q in fallback]
	after = [events[q]["energy_after"] for q in fallback]
	attributed = sum(before) + sum(inside) + sum(after) if fallback else 0.0

	f1_values = [outcomes[q]["f1"] for q in questions if outcomes[q]["f1"] is not None]
	hit_values = [outcomes[q]["hit1"] for q in questions if outcomes[q]["hit1"] is not None]
	f1_mean = effectiveness["f1"] if effectiveness else (
		100.0 * mean(f1_values) if f1_values else None)
	hit_mean = effectiveness["hit"] if effectiveness else (
		100.0 * mean(hit_values) if hit_values else None)

	depth_rows = []
	by_depth = collections.defaultdict(collections.Counter)
	for question in questions:
		by_depth[traj[question]["max_traversal_depth"]][outcomes[question]["outcome"]] += 1
	for depth in sorted(d for d in by_depth if d is not None):
		found, not_found = by_depth[depth][HIT], by_depth[depth][MISS]
		depth_rows.append({"system": name, "max_traversal_depth": depth,
		                   "gold_found": found, "gold_not_found": not_found,
		                   "total": found + not_found,
		                   "gold_found_rate_pct": pct(found, found + not_found)})

	reason_rows = []
	for reason, count in collections.Counter(
			events[q]["fallback_reason"] for q in fallback).most_common():
		group = [q for q in fallback if events[q]["fallback_reason"] == reason]
		group_hits = sum(1 for q in group if outcomes[q]["outcome"] == HIT)
		reason_rows.append({"system": name, "fallback_reason": reason, "n": len(group),
		                    "pct_of_fallback": pct(len(group), len(fallback)),
		                    "gold_found_rate_pct": pct(group_hits, len(group)),
		                    "mean_gpu_j": mean([energy_of(q) for q in group])})

	return {
		"system": name, "run_dir": run_dir, "outcomes": outcomes_path,
		"questions_run": len(questions),
		"questions_missing_energy": len(missing_energy),
		"questions_missing_outcome": len(missing_outcome),
		"empty_gold_questions": sum(1 for q in questions if outcomes[q]["empty_gold"]),
		"f1_pct": f1_mean,
		"gold_found": len(hits),
		"gold_not_found": len(misses),
		"gold_found_rate_pct": pct(len(hits), len(questions)),
		"hit1_pct": hit_mean,
		"mean_gpu_j": mean(energies),
		"median_gpu_j": median(energies),
		"p90_gpu_j": percentile(energies, 0.90),
		"total_gpu_j": total_energy,
		"mean_input_tokens": mean([events.get(q, {}).get("input_tokens") for q in questions]),
		"mean_output_tokens": mean([events.get(q, {}).get("output_tokens") for q in questions]),
		"mean_llm_calls": mean([events.get(q, {}).get("llm_calls") for q in questions]),
		"top10pct_energy_share_pct": top_share,
		"mean_gpu_j_gold_found": mean([energy_of(q) for q in hits]),
		"mean_gpu_j_gold_not_found": mean([energy_of(q) for q in misses]),
		"fallback_questions": len(fallback),
		"fallback_rate_pct": pct(len(fallback), len(questions)),
		"fallback_gold_found": len(fb_hits),
		"fallback_gold_not_found": len(fallback) - len(fb_hits),
		"fallback_success_rate_pct": pct(len(fb_hits), len(fallback)),
		"non_fallback_success_rate_pct": pct(len(nf_hits), len(non_fallback)),
		"mean_gpu_j_fallback": mean([energy_of(q) for q in fallback]),
		"mean_gpu_j_non_fallback": mean([energy_of(q) for q in non_fallback]),
		"mean_gpu_j_before_fallback": mean(before),
		"mean_gpu_j_in_fallback": mean(inside),
		"mean_gpu_j_after_fallback": mean(after),
		"pct_fallback_energy_before": pct(sum(before), attributed) if fallback else None,
		"pct_fallback_energy_in": pct(sum(inside), attributed) if fallback else None,
		"pct_fallback_energy_after": pct(sum(after), attributed) if fallback else None,
		"_depth_rows": depth_rows,
		"_reason_rows": reason_rows,
	}


#: (key, label, decimals) for the overview table, in display order.
OVERVIEW_ROWS = (
	("questions_run", "Questions run", 0),
	("f1_pct", "F1 (%)", 2),
	("gold_found", "Gold answer found", 0),
	("gold_not_found", "Gold answer not found", 0),
	("gold_found_rate_pct", "Gold answer found (%)", 2),
	("mean_gpu_j", "Mean GPU energy / question (J)", 0),
	("median_gpu_j", "Median GPU energy / question (J)", 0),
	("p90_gpu_j", "p90 GPU energy / question (J)", 0),
	("total_gpu_j", "Total GPU energy (J)", 0),
	("mean_input_tokens", "Mean input tokens / question", 0),
	("mean_output_tokens", "Mean output tokens / question", 0),
	("mean_llm_calls", "Mean LLM calls / question", 1),
	("top10pct_energy_share_pct", "Top 10% of questions: share of total GPU energy (%)", 1),
	("mean_gpu_j_gold_found", "Mean GPU energy when gold found (J)", 0),
	("mean_gpu_j_gold_not_found", "Mean GPU energy when gold not found (J)", 0),
)

FALLBACK_ROWS = (
	("fallback_questions", "Fallback questions", 0),
	("fallback_rate_pct", "Fallback rate (%)", 1),
	("fallback_gold_found", "Fallback gold found", 0),
	("fallback_gold_not_found", "Fallback gold not found", 0),
	("fallback_success_rate_pct", "Fallback success rate (%)", 1),
	("non_fallback_success_rate_pct", "Non-fallback success rate (%)", 1),
	("mean_gpu_j_fallback", "Mean GPU J, fallback", 0),
	("mean_gpu_j_non_fallback", "Mean GPU J, non-fallback", 0),
	("mean_gpu_j_before_fallback", "Mean GPU J before fallback operation", 0),
	("mean_gpu_j_in_fallback", "Mean GPU J in fallback operation", 0),
	("pct_fallback_energy_before", "% fallback energy spent before fallback operation", 1),
	("pct_fallback_energy_in", "% fallback energy spent in fallback operation", 1),
)


def fmt(value, decimals):
	if value is None:
		return ""
	if decimals == 0:
		return f"{value:,.0f}"
	return f"{value:,.{decimals}f}"


def write_matrix_csv(path, rows, systems, summaries):
	with open(path, "w", newline="") as f:
		writer = csv.writer(f, lineterminator="\n")
		writer.writerow(["metric"] + list(systems))
		for key, label, decimals in rows:
			writer.writerow([label] + [fmt(summaries[s][key], decimals) for s in systems])
	return path


def write_rows_csv(path, rows, fields):
	with open(path, "w", newline="") as f:
		writer = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore",
		                        lineterminator="\n")
		writer.writeheader()
		for row in rows:
			writer.writerow({k: ("" if row.get(k) is None else
			                     (f"{row[k]:.2f}" if isinstance(row[k], float) else row[k]))
			                 for k in fields})
	return path


def markdown_matrix(title, rows, systems, summaries):
	lines = [f"## {title}", "", "| Metric | " + " | ".join(systems) + " |",
	         "|---" * (len(systems) + 1) + "|"]
	for key, label, decimals in rows:
		lines.append(f"| {label} | "
		             + " | ".join(fmt(summaries[s][key], decimals) for s in systems) + " |")
	lines.append("")
	return lines


def main(argv=None):
	ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
	ap.add_argument("--run", action="append", metavar="NAME=DIR", required=True,
	                help="measured run directory per system; repeatable")
	ap.add_argument("--outcomes", action="append", metavar="NAME=FILE", required=True,
	                help="export_outcomes.py CSV per system; repeatable")
	ap.add_argument("--effectiveness", action="append", metavar="NAME=F1,HIT", default=[],
	                help="override F1 and Hit@1 percentages with the evaluator's own "
	                     "aggregates, e.g. CoR=43.54,62.48")
	ap.add_argument("--out", required=True, help="output directory (created if absent)")
	ap.add_argument("--dataset", default="", help="dataset label recorded in SUMMARY.md")
	ap.add_argument("--title", default="", help="heading for SUMMARY.md")
	args = ap.parse_args(argv)

	runs = parse_pairs(args.run, "--run")
	outcomes = parse_pairs(args.outcomes, "--outcomes")
	missing = sorted(set(runs) - set(outcomes))
	if missing:
		raise SystemExit(f"no --outcomes for: {', '.join(missing)}")

	effectiveness = {}
	for name, value in parse_pairs(args.effectiveness, "--effectiveness").items():
		parts = [p.strip() for p in value.split(",")]
		if len(parts) != 2:
			raise SystemExit(f"--effectiveness {name}: expected F1,HIT, got {value!r}")
		effectiveness[name] = {"f1": float(parts[0]), "hit": float(parts[1])}

	systems = order_systems(runs)
	summaries = {name: summarize(name, runs[name], outcomes[name],
	                             effectiveness.get(name))
	             for name in systems}

	out = os.path.expanduser(args.out)
	os.makedirs(out, exist_ok=True)
	written = [
		write_matrix_csv(os.path.join(out, "summary_overview.csv"),
		                 OVERVIEW_ROWS, systems, summaries),
		write_matrix_csv(os.path.join(out, "summary_fallback.csv"),
		                 FALLBACK_ROWS, systems, summaries),
		write_rows_csv(os.path.join(out, "summary_depth_outcome.csv"),
		               [r for s in systems for r in summaries[s]["_depth_rows"]],
		               ("system", "max_traversal_depth", "gold_found", "gold_not_found",
		                "total", "gold_found_rate_pct")),
		write_rows_csv(os.path.join(out, "summary_fallback_reasons.csv"),
		               [r for s in systems for r in summaries[s]["_reason_rows"]],
		               ("system", "fallback_reason", "n", "pct_of_fallback",
		                "gold_found_rate_pct", "mean_gpu_j")),
	]

	title = args.title or (f"{args.dataset} summary" if args.dataset else "Run summary")
	lines = [f"# {title}", ""]
	if args.dataset:
		lines += [f"Dataset: `{args.dataset}`", ""]
	lines += ["| System | Run directory | Outcomes |", "|---|---|---|"]
	for name in systems:
		lines.append(f"| {name} | `{runs[name]}` | `{outcomes[name]}` |")
	lines += ["", "GPU energy only unless the run recorded more; per-question energy is the",
	          "whole-question trajectory counter value. Effectiveness comes from the",
	          "dataset's own evaluator via the outcome CSV.", ""]
	lines += markdown_matrix("Overview", OVERVIEW_ROWS, systems, summaries)
	lines += markdown_matrix("Fallback", FALLBACK_ROWS, systems, summaries)

	lines += ["## Depth vs outcome", "",
	          "| System | Max traversal depth | Gold found | Gold not found | Total | % found |",
	          "|---|---|---|---|---|---|"]
	for name in systems:
		for row in summaries[name]["_depth_rows"]:
			lines.append(f"| {name} | {row['max_traversal_depth']} | {row['gold_found']:,} | "
			             f"{row['gold_not_found']:,} | {row['total']:,} | "
			             f"{fmt(row['gold_found_rate_pct'], 1)} |")
	lines.append("")

	lines += ["## Fallback reasons", "",
	          "| System | Fallback reason | N | % fallback | Gold found (%) | Mean GPU J |",
	          "|---|---|---|---|---|---|"]
	for name in systems:
		for row in summaries[name]["_reason_rows"]:
			lines.append(f"| {name} | {row['fallback_reason']} | {row['n']:,} | "
			             f"{fmt(row['pct_of_fallback'], 1)} | "
			             f"{fmt(row['gold_found_rate_pct'], 1)} | "
			             f"{fmt(row['mean_gpu_j'], 0)} |")
	lines.append("")

	notes = []
	for name in systems:
		s = summaries[name]
		if s["questions_missing_outcome"]:
			notes.append(f"- {name}: {s['questions_missing_outcome']:,} question(s) in the run "
			             f"have no outcome row and are excluded.")
		if s["questions_missing_energy"]:
			notes.append(f"- {name}: {s['questions_missing_energy']:,} scored question(s) have "
			             f"no trajectory energy and are excluded.")
		if s["empty_gold_questions"]:
			notes.append(f"- {name}: {s['empty_gold_questions']:,} question(s) carry empty gold "
			             f"and count as not found, as the evaluator scores them.")
	if notes:
		lines += ["## Notes", ""] + sorted(set(notes)) + [""]

	md = os.path.join(out, "SUMMARY.md")
	with open(md, "w", newline="") as f:
		f.write("\n".join(lines))
	written.append(md)

	for path in written:
		print(f"  -> {path}")
	print(f"\n{len(systems)} systems: " + ", ".join(
		f"{n} ({summaries[n]['questions_run']:,} questions)" for n in systems))
	return 0


if __name__ == "__main__":
	sys.exit(main())
