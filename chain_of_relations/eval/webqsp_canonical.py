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
This module owns only the WebQSP *gold* handling; every metric is computed by
`eval.accuracy.score`, the one scorer shared with CWQ and every system.

* **F1** — precision/recall/F1 against each parse, keep the best F1. Ties go to
  the earliest parse, exactly as the official `if f1 > bestf1` loop.
* **Hit@1 / gold answer found** — 1 when a predicted answer equals an answer of
  *any* parse (the union rule ToG and PoG use in their `eval/utils.py`).
* **Empty gold** — the 11 official empty-answer questions are scored, not
  dropped, following the official `CalculatePRF1`: empty gold with an empty
  prediction scores (1, 1, 1); empty gold with any prediction scores P=0, R=1,
  F1=0, and Hit@1 is 0.
* **Question skipping** — the official evaluator skips a question when no parse
  is `QuestionQuality == "Good"` and `ParseQuality == "Complete"`. No question in
  the 1,639-question test set is skipped by that rule; `evaluable()` reports it
  so a future release cannot change underfoot unnoticed.

Matching is `eval.accuracy`'s normalized exact name match, applied per parse.
The official script compares MIDs because its reference predictions are MIDs;
ours are generated surface names, so MID equality is not available to every
system. Literal-valued answers (dates, numbers, currency codes) are used exactly
as the official `AnswerArgument` gives them -- no identifier is invented for them.

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

from chain_of_relations.eval import accuracy

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

	A blank `entity_name` also falls back to `answer_argument`: 30 answers
	across 15 test questions carry `""` as their EntityName, and a blank gold
	name can never be matched, so falling back keeps those answers scorable.
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


def score_prediction(prediction: Sequence[str], parses: Sequence[Dict[str, Any]]) -> ScoreResult:
	"""Canonical score for one question.

	`prediction` is the answer list `eval.accuracy.extract_answers` produced.
	Returns best-F1-over-parses plus any-parse Hit@1; no parses at all is
	scored as empty gold, as the official CalculatePRF1 would.
	"""
	prediction = [p for p in prediction if p is not None]
	scored = accuracy.score(prediction, [parse_answer_names(p) for p in parses])
	index = scored["best_index"]
	return ScoreResult({
		"f1": scored["f1"], "precision": scored["precision"], "recall": scored["recall"],
		"best_parse_index": index,
		"best_parse_id": parses[index].get("parse_id") if index is not None else None,
		"hit": scored["hit1"], "empty_gold": scored["empty_gold"],
		"n_parses": len(parses), "evaluable": evaluable(parses)})
