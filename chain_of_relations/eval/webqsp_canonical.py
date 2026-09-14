"""Canonical WebQSP scoring: best F1 over official parses, Hit@1 over any parse.

Why this module exists
----------------------
A WebQSP question may carry several semantic parses, each with its own answer
set. The official evaluator (`WebQSP/eval/eval.py` in the Microsoft release)
scores a prediction against *every* parse and keeps the **maximum F1**; it never
unions the parses and never fixes on the first one. `datasets/webqsp/webqsp.json`
stores only the answers of `Parses[0]`, so scoring against it understates F1 on
the 38 test questions whose later parses contribute answers `Parses[0]` lacks.

That dataset file stays byte-for-byte as upstream CoR shipped it. The parse-level
gold lives beside it in `webqsp_official_gold.json`, distilled from the official
release by `scripts/build_webqsp_official_gold.py`.

The rules implemented here
--------------------------
* **F1** — compute precision/recall/F1 against each parse, keep the best F1.
  Ties go to the earliest parse, exactly as the official `if f1 > bestf1` loop.
* **Hit@1 / gold answer found** — 1 when the prediction matches an answer of
  *any* parse. This is identical to the union rule ToG and PoG use in their
  `eval/utils.py`, so our Hit@1 is directly comparable to their Exact Match.
* **Empty gold** — the 11 official empty-answer questions are scored, not
  dropped, following the official `CalculatePRF1`: empty gold with an empty
  prediction is correct (1.0); empty gold with any prediction scores 0.0. Every
  paradigm here always emits at least one item, so in practice these 11 score 0,
  which is also how ToG and PoG count them.
* **Question skipping** — the official evaluator skips a question when no parse
  is `QuestionQuality == "Good"` and `ParseQuality == "Complete"`. No question in
  the 1,639-question test set is skipped by that rule; `evaluable()` reports it
  so a future release cannot change underfoot unnoticed.

Matching is this repository's own name matching (`eval.accuracy.match`), applied
per parse. The official script compares MIDs because its reference predictions
are MIDs; ours are generated surface names, so MID equality is not available.
Literal-valued answers (dates, numbers, currency codes) are used exactly as the
official `AnswerArgument` gives them -- no identifier is invented for them.

Denominators
------------
The headline WebQSP denominator is the full **1,639** questions, matching the
official evaluator and ToG/PoG. An analysis conditioned on usable gold may
additionally report the **1,628** subset that excludes the 11 empty-gold
questions, but it must say so; `empty_gold` on every result row is what makes
that subset selectable.
"""

import json
import os
from typing import Any, Dict, List, Optional, Sequence

from chain_of_relations.eval.accuracy import eval_f1, eval_hit

__all__ = [
	"CANONICAL_DATASETS", "GOLD_PATH", "GOLD_SCHEMA_VERSION", "load_gold", "parses_for",
	"parse_answer_names", "evaluable", "score_prediction", "ScoreResult",
]

#: Datasets scored with the canonical multi-parse rule. One definition, so the
#: evaluator, the outcome export and the distribution plots cannot diverge.
CANONICAL_DATASETS = frozenset({"webqsp"})

GOLD_SCHEMA_VERSION = 1
GOLD_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(
	os.path.abspath(__file__)))), "datasets", "webqsp", "webqsp_official_gold.json")

_CACHE: Dict[str, Dict[str, Any]] = {}


def load_gold(path: Optional[str] = None) -> Dict[str, Any]:
	"""Parse-level gold, keyed by QuestionId. Cached per path."""
	resolved = os.path.abspath(path or GOLD_PATH)
	if resolved not in _CACHE:
		with open(resolved, encoding="utf-8") as f:
			gold = json.load(f)
		version = gold.get("gold_schema_version")
		if version != GOLD_SCHEMA_VERSION:
			raise ValueError(
				f"{resolved}: gold_schema_version {version!r}, expected {GOLD_SCHEMA_VERSION}")
		_CACHE[resolved] = gold["questions"]
	return _CACHE[resolved]


def parses_for(question_id: str, gold: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
	"""Official parses for a question; empty list when the id is unknown."""
	entry = (gold if gold is not None else load_gold()).get(question_id)
	return list(entry.get("parses") or []) if entry else []


def parse_answer_names(parse: Dict[str, Any]) -> List[str]:
	"""Answer strings of one parse.

	`entity_name` for entities; the official `answer_argument` verbatim for
	literals, whose `entity_name` is null.

	A blank `entity_name` also falls back to `answer_argument`. 30 answers
	across 15 test questions carry `""` as their EntityName, and an empty gold
	string matches every prediction under substring matching, which would score
	those questions correct no matter what the system answered.
	"""
	names = []
	for answer in parse.get("answers") or []:
		name = answer.get("entity_name")
		if name is None or not str(name).strip():
			name = answer.get("answer_argument")
		names.append(name)
	return [n for n in names if n is not None and str(n).strip()]


def evaluable(parses: Sequence[Dict[str, Any]]) -> bool:
	"""The official skip rule: at least one Good + Complete parse."""
	return any(p.get("question_quality") == "Good" and p.get("parse_quality") == "Complete"
	           for p in parses)


class ScoreResult(dict):
	"""Scoring outcome for one question. A dict, so it serializes as-is."""

	@property
	def f1(self) -> float:
		return self["f1"]

	@property
	def hit(self) -> int:
		return self["hit"]


def _score_against(prediction: Sequence[str], gold_names: Sequence[str]):
	"""precision, recall, f1 for one parse, following the official empty cases."""
	if not gold_names:
		# Official CalculatePRF1: no labeled answer.
		return (1.0, 1.0, 1.0) if not prediction else (0.0, 1.0, 0.0)
	if not prediction:
		return 1.0, 0.0, 0.0
	f1, precision, recall = eval_f1(list(prediction), list(gold_names))
	return precision, recall, f1


def score_prediction(prediction: Sequence[str], parses: Sequence[Dict[str, Any]]) -> ScoreResult:
	"""Canonical score for one question.

	`prediction` is the already post-processed item list this repository's
	evaluator builds. Returns best-F1-over-parses plus any-parse Hit@1.
	"""
	prediction = [p for p in prediction if p is not None]
	answer_sets = [parse_answer_names(p) for p in parses]
	empty_gold = not any(answer_sets)

	best = None
	for index, (parse, gold_names) in enumerate(zip(parses, answer_sets)):
		precision, recall, f1 = _score_against(prediction, gold_names)
		# Strictly greater, so ties keep the earliest parse: the official loop.
		if best is None or f1 > best["f1"]:
			best = {"f1": f1, "precision": precision, "recall": recall,
			        "best_parse_index": index, "best_parse_id": parse.get("parse_id")}
	if best is None:
		# No parses at all: treat as empty gold, which is how the official
		# CalculatePRF1 would score it.
		precision, recall, f1 = _score_against(prediction, [])
		best = {"f1": f1, "precision": precision, "recall": recall,
		        "best_parse_index": None, "best_parse_id": None}

	# Hit@1 against the union of every parse, which is ToG's and PoG's rule.
	hit = 0
	if not empty_gold and prediction:
		joined = " ".join(prediction)
		hit = 1 if any(eval_hit(joined, names) == 1 for names in answer_sets if names) else 0

	return ScoreResult(dict(best, hit=hit, empty_gold=empty_gold,
	                        n_parses=len(parses), evaluable=evaluable(parses)))
