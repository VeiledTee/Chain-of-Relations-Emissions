"""Per-question answer outcomes for a benchmark run, as a visualizer annotation file.

This is the dataset-specific half of outcome-coloured figures: it runs THIS
repository's evaluator over a run's predict.jsonl and writes one row per
question. agent_energy_profiler never sees it; the profiler only reads back
opaque category strings through --annotations / --color-by.

    python measurement/export_outcomes.py \
        --predictions results/cor/webqsp/gemma-3-4b-it/predict.jsonl \
        --dataset webqsp --out outcomes_cor.csv [--run CoR]

Columns: question_id, run, outcome, hit1, f1, precision, recall, n_gold_answers,
n_prediction_items, answer_extraction, action, empty_gold, n_parses, best_parse_id.

Scoring is `chain_of_relations.eval.accuracy`, the repository's one evaluator:
the final answer only (a JSON "answer" field or plain answer items; rationale
text is never scored), normalized exact matching, set-based precision / recall
/ F1. `n_prediction_items` is the number of distinct normalized predicted
answers -- the precision denominator. `answer_extraction` records how the answer
was obtained (`eval.accuracy.extract_answers`).

WebQSP is scored against every official parse (best F1 over parses, Hit@1
against any parse) via chain_of_relations.eval.webqsp_canonical. Its gold set
comes from datasets/webqsp/webqsp_official_gold.json, NOT from the `gold_answer`
field copied into predict.jsonl, which holds only `Parses[0]`. Every other
dataset is scored against predict.jsonl's `gold_answer` list.

`outcome` is `hit` when Hit@1 is 1 and `miss` when it is 0, for every question
-- the 11 official WebQSP empty-gold questions included, which is how the
official evaluator and ToG/PoG count them. `empty_gold` marks them so an
analysis that needs usable gold can select the 1,628 subset and say it did.
"""

import argparse
import csv
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from chain_of_relations.eval import accuracy, webqsp_canonical  # noqa: E402
from chain_of_relations.eval.eval import postprocess_prediction_for_dataset  # noqa: E402

HIT, MISS = "hit", "miss"
FIELDS = ("question_id", "run", "outcome", "hit1", "f1", "precision", "recall",
          "n_gold_answers", "n_prediction_items", "answer_extraction", "action",
          "empty_gold", "n_parses", "best_parse_id")
#: Datasets scored with the canonical multi-parse rule (defined in the evaluator).
CANONICAL_DATASETS = webqsp_canonical.CANONICAL_DATASETS


def prediction_answers(record, dataset, gold):
	"""(answers, extraction status) the evaluator scores for one record."""
	answers, status = accuracy.extract_answers(record.get("results"))
	return postprocess_prediction_for_dataset(dataset, answers, gold), status


def _row(record, dataset, run_label, gold_index):
	question_id = record["id"]
	local_gold = [a["name"] for a in record.get("gold_answer", []) if "name" in a]
	answers, status = prediction_answers(record, dataset, local_gold)
	if gold_index is not None:
		parses = webqsp_canonical.parses_for(question_id, gold_index)
		scored = webqsp_canonical.score_prediction(answers, parses)
		index = scored["best_parse_index"]
		best_gold = webqsp_canonical.parse_answer_names(parses[index]) if index is not None else []
		hit, n_parses, best_parse_id = scored["hit"], scored["n_parses"], scored["best_parse_id"]
	else:
		scored = accuracy.score(answers, [local_gold])
		best_gold, hit, n_parses, best_parse_id = local_gold, scored["hit1"], "", ""
	return {
		"question_id": question_id, "run": run_label,
		"outcome": HIT if hit == 1 else MISS,
		"hit1": hit, "f1": scored["f1"],
		"precision": scored["precision"], "recall": scored["recall"],
		"n_gold_answers": len(accuracy.answer_set(best_gold)),
		"n_prediction_items": len(accuracy.answer_set(answers)),
		"answer_extraction": status, "action": record.get("action", ""),
		"empty_gold": scored["empty_gold"], "n_parses": n_parses, "best_parse_id": best_parse_id,
	}


def score(predictions_path, dataset, run_label=""):
	"""One row per question, every system through the same evaluator."""
	canonical = str(dataset or "").strip().lower() in CANONICAL_DATASETS
	gold_index = webqsp_canonical.load_gold() if canonical else None
	rows = []
	with open(predictions_path) as f:
		for line in f:
			line = line.strip()
			if not line:
				continue
			rows.append(_row(json.loads(line), dataset, run_label, gold_index))
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
	ap.add_argument("--dataset", default="webqsp", help="dataset name (selects the gold handling)")
	ap.add_argument("--run", default="", help="run label written into the run column")
	ap.add_argument("--out", required=True, help="output CSV")
	args = ap.parse_args(argv)
	rows = score(args.predictions, args.dataset, args.run)
	write_csv(rows, args.out)
	counts = {k: sum(1 for r in rows if r["outcome"] == k) for k in (HIT, MISS)}
	empty = sum(1 for r in rows if r.get("empty_gold"))
	print(f"{args.out}: {len(rows):,} questions  " +
	      "  ".join(f"{k}={v:,}" for k, v in counts.items()) +
	      f"  empty_gold={empty:,}")
	return 0


if __name__ == "__main__":
	sys.exit(main())
