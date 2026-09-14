"""Per-question answer outcomes for a benchmark run, as a visualizer annotation file.

This is the dataset-specific half of outcome-coloured figures: it runs THIS
repository's evaluator over a run's predict.jsonl and writes one row per
question. agent_energy_profiler never sees it; the profiler only reads back
opaque category strings through --annotations / --color-by.

    python measurement/export_outcomes.py \
        --predictions results/cor/webqsp/gemma-3-4b-it/predict.jsonl \
        --dataset webqsp --out outcomes_cor.csv [--run CoR]

Columns: question_id, run, outcome, hit1, f1, precision, recall, accuracy,
n_gold_answers, n_prediction_items, action, empty_gold, n_parses, best_parse_id.

WebQSP is scored canonically: best F1 over the official parses and Hit@1 against
any parse, via chain_of_relations.eval.webqsp_canonical. The gold set therefore
comes from datasets/webqsp/webqsp_official_gold.json, NOT from the `gold_answer`
field copied into predict.jsonl, which holds only `Parses[0]` and understates F1
on multi-parse questions.

`outcome` is `hit` when Hit@1 is 1 and `miss` when it is 0, for all 1,639 WebQSP
questions -- the 11 official empty-gold questions included, which is how the
official evaluator and ToG/PoG count them. `empty_gold` marks those 11 so an
analysis that needs usable gold can select the 1,628 subset and say it did.
`unscored` survives only for non-WebQSP datasets whose evaluator divides by zero
on a question with no gold answer.
"""

import argparse
import csv
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from chain_of_relations.eval import webqsp_canonical  # noqa: E402
from chain_of_relations.eval.accuracy import eval_acc, eval_f1, eval_hit  # noqa: E402
from chain_of_relations.eval.eval import (  # noqa: E402
	extract_answer_from_braces, postprocess_prediction_for_dataset)

HIT, MISS, UNSCORED = "hit", "miss", "unscored"
FIELDS = ("question_id", "run", "outcome", "hit1", "f1", "precision", "recall", "accuracy",
          "n_gold_answers", "n_prediction_items", "action", "empty_gold", "n_parses",
          "best_parse_id")
#: Datasets scored with the canonical multi-parse rule (defined in the evaluator).
CANONICAL_DATASETS = webqsp_canonical.CANONICAL_DATASETS


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


def _canonical_row(record, dataset, run_label, gold_index):
	"""One row scored against every official parse (best F1, any-parse Hit@1)."""
	question_id = record["id"]
	parses = webqsp_canonical.parses_for(question_id, gold_index)
	# Post-processing is gold-dependent for numeric datasets only; the local
	# gold list is the right shape for it and never reaches the score itself.
	local_gold = [a["name"] for a in record.get("gold_answer", []) if "name" in a]
	items = prediction_items(record, dataset, local_gold)
	scored = webqsp_canonical.score_prediction(items, parses)
	best_names = (webqsp_canonical.parse_answer_names(parses[scored["best_parse_index"]])
	              if scored["best_parse_index"] is not None else [])
	return {
		"question_id": question_id, "run": run_label,
		"outcome": HIT if scored["hit"] == 1 else MISS,
		"hit1": scored["hit"], "f1": scored["f1"],
		"precision": scored["precision"], "recall": scored["recall"],
		"accuracy": eval_acc(" ".join(items), best_names) if best_names else 0.0,
		"n_gold_answers": len(best_names), "n_prediction_items": len(items),
		"action": record.get("action", ""), "empty_gold": scored["empty_gold"],
		"n_parses": scored["n_parses"], "best_parse_id": scored["best_parse_id"],
	}


def _legacy_row(record, dataset, run_label):
	"""Pre-canonical scoring, still used by datasets with no parse-level gold."""
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
	return {"question_id": record["id"], "run": run_label, "outcome": outcome,
	        "hit1": hit, "f1": f1, "precision": precision, "recall": recall,
	        "accuracy": accuracy, "n_gold_answers": len(gold),
	        "n_prediction_items": len(items), "action": record.get("action", ""),
	        "empty_gold": not gold, "n_parses": "", "best_parse_id": ""}


def score(predictions_path, dataset, run_label=""):
	"""One row per question.

	WebQSP uses the canonical multi-parse rule; every other dataset keeps the
	single-gold-list scoring it had.
	"""
	canonical = str(dataset or "").strip().lower() in CANONICAL_DATASETS
	gold_index = webqsp_canonical.load_gold() if canonical else None
	rows = []
	with open(predictions_path) as f:
		for line in f:
			line = line.strip()
			if not line:
				continue
			record = json.loads(line)
			rows.append(_canonical_row(record, dataset, run_label, gold_index) if canonical
			            else _legacy_row(record, dataset, run_label))
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
	empty = sum(1 for r in rows if r.get("empty_gold"))
	print(f"{args.out}: {len(rows):,} questions  " +
	      "  ".join(f"{k}={v:,}" for k, v in counts.items()) +
	      f"  empty_gold={empty:,}")
	return 0


if __name__ == "__main__":
	sys.exit(main())
