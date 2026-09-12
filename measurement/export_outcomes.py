"""Per-question answer outcomes for a benchmark run, as a visualizer annotation file.

This is the dataset-specific half of outcome-coloured figures: it runs THIS
repository's evaluator over a run's predict.jsonl and writes one row per
question. agent_energy_profiler never sees it; the profiler only reads back
opaque category strings through --annotations / --color-by.

    python measurement/export_outcomes.py \
        --predictions results/cor/webqsp/gemma-3-4b-it/predict.jsonl \
        --dataset webqsp --out outcomes_cor.csv [--run CoR]

Columns: question_id, run, outcome, hit1, f1, precision, recall, accuracy,
n_gold_answers, n_prediction_items, action.

outcome is `hit` when the evaluator's Hit@1 is 1, `miss` when it is 0, and
`unscored` when the question carries no gold answer (the evaluator skips those,
so they are never counted as wrong).
"""

import argparse
import csv
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from chain_of_relations.eval.accuracy import eval_acc, eval_f1, eval_hit  # noqa: E402
from chain_of_relations.eval.eval import (  # noqa: E402
	extract_answer_from_braces, postprocess_prediction_for_dataset)

HIT, MISS, UNSCORED = "hit", "miss", "unscored"
FIELDS = ("question_id", "run", "outcome", "hit1", "f1", "precision", "recall", "accuracy",
          "n_gold_answers", "n_prediction_items", "action")


def prediction_items(record, dataset, gold):
	"""The evaluator's own prediction list for one record."""
	results = record.get("results", [])
	if not results or not isinstance(results, list):
		return ["None"]
	items = []
	for raw in [r["name"] for r in results if isinstance(r, dict) and "name" in r]:
		extracted = extract_answer_from_braces(raw)
		items.extend(extracted) if extracted else items.append(raw)
	return postprocess_prediction_for_dataset(dataset, items or ["None"], gold)


def score(predictions_path, dataset, run_label=""):
	"""One row per question, scored exactly as chain_of_relations.eval.eval does."""
	rows = []
	with open(predictions_path) as f:
		for line in f:
			line = line.strip()
			if not line:
				continue
			record = json.loads(line)
			gold = [a["name"] for a in record.get("gold_answer", []) if "name" in a]
			items = prediction_items(record, dataset, gold)
			joined = " ".join(items)
			try:
				f1, precision, recall = eval_f1(items, gold)
				accuracy, hit = eval_acc(joined, gold), eval_hit(joined, gold)
				outcome = HIT if hit == 1 else MISS
			except BaseException:
				# No gold answers: the evaluator divides by zero and skips the question.
				f1 = precision = recall = accuracy = hit = None
				outcome = UNSCORED
			rows.append({"question_id": record["id"], "run": run_label, "outcome": outcome,
			             "hit1": hit, "f1": f1, "precision": precision, "recall": recall,
			             "accuracy": accuracy, "n_gold_answers": len(gold),
			             "n_prediction_items": len(items), "action": record.get("action", "")})
	return rows


def write_csv(rows, path):
	with open(path, "w", newline="") as f:
		writer = csv.DictWriter(f, fieldnames=FIELDS, extrasaction="ignore", lineterminator="\n")
		writer.writeheader()
		for row in rows:
			writer.writerow({k: ("" if row.get(k) is None else row[k]) for k in FIELDS})
	return path


def main(argv=None):
	ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
	ap.add_argument("--predictions", required=True, help="path to predict.jsonl")
	ap.add_argument("--dataset", default="webqsp", help="dataset name (evaluator post-processing)")
	ap.add_argument("--run", default="", help="run label written into the run column")
	ap.add_argument("--out", required=True, help="output CSV")
	args = ap.parse_args(argv)
	rows = score(args.predictions, args.dataset, args.run)
	write_csv(rows, args.out)
	counts = {k: sum(1 for r in rows if r["outcome"] == k) for k in (HIT, MISS, UNSCORED)}
	print(f"{args.out}: {len(rows):,} questions  " +
	      "  ".join(f"{k}={v:,}" for k, v in counts.items()))
	return 0


if __name__ == "__main__":
	sys.exit(main())
