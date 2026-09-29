"""Audit the non-ok events of a measured run, and what they did to the results.

`measurement/audit_runs.py` audits the *hardware* record of a run: counters
against the power curve, split sessions, impossible power. This script audits
the *semantic* record instead: every event whose `status` is not `ok`, grouped
by error type and operation label, and the questions those events belong to.

    python measurement/failed_event_audit.py \
        --out ~/cwq_supervisor_summary/failed_events \
        --run PoG=measurement/runs/gemma4b_full_pog_cwq_0919_1646 \
        --predictions PoG=results/pog/cwq/gemma-3-4b-it/predict.jsonl \
        --questions ~/cwq_supervisor_summary/correctness/correctness_questions.csv

It answers, per system:

    error type x operation label   how many events, how many distinct questions
    still predicted                did the affected questions reach predict.jsonl
    malformed-identifier share     how many failures carry a raw `/en/...` MID,
                                   the already-observed malformed Freebase/SPARQL
                                   behaviour
    sensitivity                    the fully-correct / not-fully-correct energy
                                   comparison recomputed with affected questions
                                   flagged, so the reader can see whether the
                                   conclusion depends on them

Affected questions are never dropped from the main analysis. The sensitivity
table is a diagnostic: it reports the same comparison on the affected and
unaffected subsets side by side, and nothing here rewrites the headline numbers.

Writes into --out only; the run artifacts and predictions are read-only.
"""

import argparse
import collections
import csv
import json
import os
import re
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)
sys.path.insert(0, HERE)

import summarize_comparable_runs as scr  # noqa: E402

#: The malformed Freebase identifier already observed in these runs: a raw
#: `/en/...` name where a `m.`/`g.` MID belongs.
MALFORMED_MID = re.compile(r"/en/[A-Za-z0-9_.-]+")
OK = "ok"
EVENT_FIELDS = ("system", "status", "error_type", "operation_type", "operation_label",
                "n_events", "n_questions", "n_questions_with_prediction",
                "n_malformed_identifier_events", "mean_duration_s", "example_question_id",
                "example_detail")
QUESTION_FIELDS = ("system", "question_id", "n_failed_events", "statuses", "error_types",
                   "operation_labels", "malformed_identifier", "has_prediction",
                   "f1", "binary_category", "trajectory_gpu_energy_j")
SENSITIVITY_FIELDS = ("system", "subset", "n", "fully_correct_n", "fully_correct_mean_gpu_j",
                      "not_fully_correct_n", "not_fully_correct_mean_gpu_j", "mean_ratio",
                      "percent_difference", "fully_correct_median_gpu_j",
                      "not_fully_correct_median_gpu_j", "median_ratio")


def error_type(event):
	"""The failure's class, from whatever the logger actually recorded.

	Never guessed from prompt text: only `meta` keys the instrumentation wrote
	and, failing those, the status itself.
	"""
	meta = event.get("meta") or {}
	if event.get("status") == "timeout" or meta.get("timed_out") is True:
		return "timeout"
	for key in ("error_type", "error", "exception", "exception_type", "reason",
	            "failure_reason"):
		value = meta.get(key)
		if isinstance(value, dict) and value.get("type"):
			return str(value["type"])[:80]
		if not (isinstance(value, str) and value.strip()):
			continue
		text = value.strip()
		# An OpenAI-style error carries the real class inside the payload; an
		# exception repr starts with its own class name.
		payload = re.search(r"'type':\s*'([A-Za-z_]+)'", text)
		if payload:
			if "maximum context length" in text:
				return f"{payload.group(1)} (context length exceeded)"
			return payload.group(1)
		head = text.split(":")[0][:80]
		return head if head and not head.lower().startswith("error code") else text[:80]
	return f"(unlabelled {event.get('status')})"


def detail(event):
	meta = event.get("meta") or {}
	for key in ("error", "message", "detail", "query", "sparql"):
		value = meta.get(key)
		if isinstance(value, str) and value.strip():
			return value.strip()[:300]
	return json.dumps({k: v for k, v in meta.items() if k != "attempts"})[:300]


def read_failed(run_dir):
	"""Every non-ok event of a run, as (event, error_type, detail)."""
	path = os.path.join(run_dir, "events_attributed.jsonl")
	if not os.path.exists(path):
		path = os.path.join(run_dir, "events.jsonl")
	if not os.path.exists(path):
		raise SystemExit(f"{run_dir}: no events_attributed.jsonl or events.jsonl")
	out, total = [], 0
	with open(path) as f:
		for line in f:
			line = line.strip()
			if not line:
				continue
			total += 1
			event = json.loads(line)
			if event.get("status") in (None, OK):
				continue
			out.append((event, error_type(event), detail(event)))
	return out, total


def read_predictions(path):
	ids = set()
	with open(path) as f:
		for line in f:
			line = line.strip()
			if line:
				ids.add(json.loads(line)["id"])
	return ids


def read_questions(path):
	"""question rows from correctness_energy.py, keyed (system, question_id)."""
	if not path:
		return {}
	out = {}
	with open(os.path.expanduser(path), newline="") as f:
		for row in csv.DictReader(f):
			out[(row["system"], row["question_id"])] = row
	return out


def event_rows(name, failed, predicted):
	groups = collections.defaultdict(list)
	for event, etype, det in failed:
		groups[(event.get("status"), etype, event.get("operation_type"),
		        event.get("operation_label"))].append((event, det))
	rows = []
	for (status, etype, otype, label), items in sorted(
			groups.items(), key=lambda kv: -len(kv[1])):
		questions = {e.get("question_id") for e, _ in items}
		malformed = sum(1 for e, d in items
		                if MALFORMED_MID.search(d or "")
		                or MALFORMED_MID.search(json.dumps(e.get("meta") or {})))
		durations = [e.get("duration_s") for e, _ in items if e.get("duration_s") is not None]
		rows.append({
			"system": name, "status": status, "error_type": etype,
			"operation_type": otype, "operation_label": label,
			"n_events": len(items), "n_questions": len(questions),
			"n_questions_with_prediction": len(questions & predicted),
			"n_malformed_identifier_events": malformed,
			"mean_duration_s": scr.mean(durations),
			"example_question_id": items[0][0].get("question_id"),
			"example_detail": items[0][1],
		})
	return rows


def question_rows(name, failed, predicted, questions):
	by_question = collections.defaultdict(list)
	for event, etype, det in failed:
		by_question[event.get("question_id")].append((event, etype, det))
	rows = []
	for qid, items in sorted(by_question.items(), key=lambda kv: -len(kv[1])):
		q = questions.get((name, qid), {})
		rows.append({
			"system": name, "question_id": qid, "n_failed_events": len(items),
			"statuses": "|".join(sorted({e.get("status") or "" for e, _, _ in items})),
			"error_types": "|".join(sorted({t for _, t, _ in items})),
			"operation_labels": "|".join(sorted({e.get("operation_label") or ""
			                                     for e, _, _ in items})),
			"malformed_identifier": any(
				MALFORMED_MID.search(d or "")
				or MALFORMED_MID.search(json.dumps(e.get("meta") or {}))
				for e, _, d in items),
			"has_prediction": qid in predicted,
			"f1": q.get("f1", ""), "binary_category": q.get("binary_category", ""),
			"trajectory_gpu_energy_j": q.get("trajectory_gpu_energy_j", ""),
		})
	return rows


def sensitivity_rows(name, questions, affected):
	"""The binary energy comparison on all / affected / unaffected questions."""
	rows = []
	mine = [r for (s, _), r in questions.items() if s == name]
	if not mine:
		return rows
	subsets = (("all questions", mine),
	           ("affected by a failed event",
	            [r for r in mine if r["question_id"] in affected]),
	           ("not affected", [r for r in mine if r["question_id"] not in affected]))
	for label, group in subsets:
		full = [float(r["trajectory_gpu_energy_j"]) for r in group
		        if r["binary_category"] == "correct"]
		notfull = [float(r["trajectory_gpu_energy_j"]) for r in group
		           if r["binary_category"] == "incorrect"]
		mean_f, mean_n = scr.mean(full), scr.mean(notfull)
		med_f, med_n = scr.median(full), scr.median(notfull)
		ratio = mean_n / mean_f if mean_f else None
		rows.append({
			"system": name, "subset": label, "n": len(group),
			"fully_correct_n": len(full), "fully_correct_mean_gpu_j": mean_f,
			"not_fully_correct_n": len(notfull), "not_fully_correct_mean_gpu_j": mean_n,
			"mean_ratio": ratio,
			"percent_difference": (ratio - 1.0) * 100.0 if ratio is not None else None,
			"fully_correct_median_gpu_j": med_f,
			"not_fully_correct_median_gpu_j": med_n,
			"median_ratio": med_n / med_f if med_f else None,
		})
	return rows


def write_csv(path, rows, fields):
	with open(path, "w", newline="") as f:
		writer = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore", lineterminator="\n")
		writer.writeheader()
		for row in rows:
			writer.writerow({k: ("" if row.get(k) is None else row[k]) for k in fields})
	return path


def fmt(value, decimals=2):
	if value in (None, ""):
		return ""
	if isinstance(value, str):
		return value
	return f"{value:,.0f}" if decimals == 0 else f"{value:,.{decimals}f}"


def main(argv=None):
	ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
	ap.add_argument("--run", action="append", metavar="NAME=DIR", required=True)
	ap.add_argument("--predictions", action="append", metavar="NAME=FILE", required=True)
	ap.add_argument("--questions", default="",
	                help="correctness_questions.csv, for the sensitivity table")
	ap.add_argument("--out", required=True)
	args = ap.parse_args(argv)

	runs = scr.parse_pairs(args.run, "--run")
	predictions = scr.parse_pairs(args.predictions, "--predictions")
	if set(runs) != set(predictions):
		raise SystemExit("--run and --predictions must name the same systems")
	systems = scr.order_systems(runs)
	out = os.path.expanduser(args.out)
	os.makedirs(out, exist_ok=True)
	questions = read_questions(args.questions)

	events, per_question, sensitivity, headline = [], [], [], []
	for name in systems:
		failed, total = read_failed(runs[name])
		predicted = read_predictions(predictions[name])
		events.extend(event_rows(name, failed, predicted))
		qrows = question_rows(name, failed, predicted, questions)
		per_question.extend(qrows)
		affected = {r["question_id"] for r in qrows}
		sensitivity.extend(sensitivity_rows(name, questions, affected))
		malformed_events = sum(1 for r in events if r["system"] == name
		                       for _ in range(r["n_malformed_identifier_events"]))
		headline.append(
			f"{name}: {len(failed):,} failed of {total:,} events "
			f"({100.0 * len(failed) / total:.3f}%)  "
			f"{len(affected):,} affected questions  "
			f"{sum(1 for r in qrows if r['has_prediction']):,} of them still produced a "
			f"prediction  {malformed_events:,} events carry a raw /en/ identifier")

	written = [
		write_csv(os.path.join(out, "failed_events_by_type.csv"), events, EVENT_FIELDS),
		write_csv(os.path.join(out, "failed_events_by_question.csv"), per_question,
		          QUESTION_FIELDS),
	]
	if sensitivity:
		written.append(write_csv(os.path.join(out, "failed_events_sensitivity.csv"),
		                         sensitivity, SENSITIVITY_FIELDS))

	lines = ["# Failed-event audit", "",
	         "Every event whose `status` is not `ok`, grouped by error type and operation "
	         "label. Affected questions are flagged, never excluded.", ""]
	lines += [f"- {h}" for h in headline] + ["", "## Failed events by type", "",
	          "| System | Status | Error type | Operation | Events | Questions | "
	          "Still predicted | `/en/` events | Mean s |",
	          "|---|---|---|---|---|---|---|---|---|"]
	for r in events:
		lines.append(f"| {r['system']} | {r['status']} | {r['error_type']} | "
		             f"{r['operation_label']} | {r['n_events']:,} | {r['n_questions']:,} | "
		             f"{r['n_questions_with_prediction']:,} | "
		             f"{r['n_malformed_identifier_events']:,} | "
		             f"{fmt(r['mean_duration_s'], 3)} |")
	if sensitivity:
		lines += ["", "## Sensitivity: does flagging affected questions change the "
		          "fully-correct vs not-fully-correct comparison?", "",
		          "| System | Subset | n | Full n | Full mean J | Not-full n | "
		          "Not-full mean J | Mean ratio | % diff | Median ratio |",
		          "|---|---|---|---|---|---|---|---|---|---|"]
		for r in sensitivity:
			lines.append(f"| {r['system']} | {r['subset']} | {r['n']:,} | "
			             f"{r['fully_correct_n']:,} | {fmt(r['fully_correct_mean_gpu_j'], 0)} | "
			             f"{r['not_fully_correct_n']:,} | "
			             f"{fmt(r['not_fully_correct_mean_gpu_j'], 0)} | "
			             f"{fmt(r['mean_ratio'], 3)} | {fmt(r['percent_difference'], 1)} | "
			             f"{fmt(r['median_ratio'], 3)} |")
	lines += ["", "## Inputs", ""] + [f"- {s}: `{runs[s]}`, `{predictions[s]}`"
	                                  for s in systems] + [""]
	md = os.path.join(out, "FAILED_EVENTS.md")
	with open(md, "w", newline="") as f:
		f.write("\n".join(lines))
	written.append(md)

	for h in headline:
		print(h)
	for path in written:
		print(f"  -> {path}")
	return 0


if __name__ == "__main__":
	sys.exit(main())
