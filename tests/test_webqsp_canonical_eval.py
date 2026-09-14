"""Canonical WebQSP scoring: best F1 over parses, Hit@1 over any parse.

The rules under test come from the official release's own evaluator
(WebQSP/eval/eval.py): score against every parse, keep the maximum F1, and
score -- never silently drop -- a question whose gold answer list is empty.
`datasets/webqsp/webqsp.json` keeps only Parses[0], so these tests also pin the
cases where scoring against that file alone would understate the result.
"""

import json
import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from chain_of_relations.eval import webqsp_canonical  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
LOCAL_DATASET = os.path.join(ROOT, "datasets", "webqsp", "webqsp.json")


def parse(answers, parse_id="P0", quality=("Good", "Complete")):
	"""One synthetic parse. `answers` is a list of (entity_name, answer_argument)."""
	return {"parse_id": parse_id, "question_quality": quality[0], "parse_quality": quality[1],
	        "answers": [{"answer_type": "Entity" if name else "Value",
	                     "answer_argument": argument, "entity_name": name}
	                    for name, argument in answers]}


# --------------------------------------------------------------------------
# 1. single-parse question
# --------------------------------------------------------------------------

def test_single_parse_exact_match_scores_one():
	parses = [parse([("Jamaican English", "m.01428y")])]
	scored = webqsp_canonical.score_prediction(["Jamaican English"], parses)
	assert scored["f1"] == pytest.approx(1.0)
	assert scored["hit"] == 1
	assert scored["best_parse_id"] == "P0"
	assert scored["empty_gold"] is False
	assert scored["n_parses"] == 1


def test_single_parse_partial_answer_gives_partial_f1():
	parses = [parse([("Asia", "m.0j0k"), ("Africa", "m.0dg3n1")])]
	scored = webqsp_canonical.score_prediction(["Asia"], parses)
	assert scored["hit"] == 1
	assert 0.0 < scored["f1"] < 1.0
	assert scored["recall"] == pytest.approx(0.5)


# --------------------------------------------------------------------------
# 2. multi-parse question where Parses[0] is the incomplete one
# --------------------------------------------------------------------------

def test_best_f1_comes_from_a_later_parse():
	"""Parses[0] holds one answer; a later parse holds the full set.

	Scoring against Parses[0] alone -- what datasets/webqsp/webqsp.json
	supports -- would score this prediction below 1.0.
	"""
	parses = [parse([("Egypt", "m.02k54")], parse_id="P0"),
	          parse([("Egypt", "m.02k54"), ("Sudan", "m.06tw8")], parse_id="P1")]
	prediction = ["Egypt", "Sudan"]

	first_only = webqsp_canonical.score_prediction(prediction, parses[:1])
	both = webqsp_canonical.score_prediction(prediction, parses)

	assert first_only["f1"] < 1.0
	assert both["f1"] == pytest.approx(1.0)
	assert both["best_parse_id"] == "P1"
	assert both["best_parse_index"] == 1


def test_ties_keep_the_earliest_parse():
	"""The official loop advances only on a strictly greater F1."""
	parses = [parse([("Egypt", "m.02k54")], parse_id="P0"),
	          parse([("Egypt", "m.02k54")], parse_id="P1")]
	scored = webqsp_canonical.score_prediction(["Egypt"], parses)
	assert scored["f1"] == pytest.approx(1.0)
	assert scored["best_parse_id"] == "P0"


def test_hit_is_the_union_rule_across_parses():
	"""Hit@1 must fire on a match in ANY parse: ToG's and PoG's gold rule."""
	parses = [parse([("Egypt", "m.02k54")], parse_id="P0"),
	          parse([("Sudan", "m.06tw8")], parse_id="P1")]
	scored = webqsp_canonical.score_prediction(["Sudan"], parses)
	assert scored["hit"] == 1
	assert scored["best_parse_id"] == "P1"

	missed = webqsp_canonical.score_prediction(["Chad"], parses)
	assert missed["hit"] == 0


# --------------------------------------------------------------------------
# 3. literal-valued answer
# --------------------------------------------------------------------------

def test_literal_answer_uses_answer_argument_verbatim():
	"""Literals carry entity_name = null; the official AnswerArgument is the answer.

	No identifier is invented for them: the answer string is the literal itself.
	"""
	parses = [parse([(None, "1841-03-04")])]
	assert webqsp_canonical.parse_answer_names(parses[0]) == ["1841-03-04"]
	scored = webqsp_canonical.score_prediction(["1841-03-04"], parses)
	assert scored["f1"] == pytest.approx(1.0)
	assert scored["hit"] == 1


def test_blank_entity_name_falls_back_to_the_answer_argument():
	"""A blank EntityName would substring-match anything, so it is never gold.

	30 answers across 15 test questions carry "" as their EntityName; scoring
	against the empty string would mark those questions correct for any output.
	"""
	parses = [parse([("", "m.02k54")])]
	assert webqsp_canonical.parse_answer_names(parses[0]) == ["m.02k54"]
	assert webqsp_canonical.score_prediction(["anything at all"], parses)["hit"] == 0


def test_literal_answers_are_present_in_the_shipped_gold():
	"""The real artifact keeps literal answers, not blanks."""
	gold = webqsp_canonical.load_gold()
	parses = webqsp_canonical.parses_for("WebQTest-46", gold)
	names = [n for p in parses for n in webqsp_canonical.parse_answer_names(p)]
	assert "1841-03-04" in names


# --------------------------------------------------------------------------
# 4. empty-gold question
# --------------------------------------------------------------------------

def test_empty_gold_with_a_prediction_scores_zero():
	"""Official CalculatePRF1: no labeled answer, some predicted answer -> F1 0."""
	parses = [parse([])]
	scored = webqsp_canonical.score_prediction(["David Beckham"], parses)
	assert scored["empty_gold"] is True
	assert scored["f1"] == pytest.approx(0.0)
	assert scored["recall"] == pytest.approx(1.0)
	assert scored["precision"] == pytest.approx(0.0)
	assert scored["hit"] == 0


def test_empty_gold_with_no_prediction_scores_one():
	"""Official CalculatePRF1: no labeled answer and no prediction -> correct."""
	scored = webqsp_canonical.score_prediction([], [parse([])])
	assert scored["f1"] == pytest.approx(1.0)
	assert scored["hit"] == 0


def test_the_eleven_empty_gold_questions_are_present_and_empty():
	"""They are scored, not dropped: the official denominator stays 1,639."""
	gold = webqsp_canonical.load_gold()
	assert len(gold) == 1639
	empty = [qid for qid in gold
	         if not any(webqsp_canonical.parse_answer_names(p)
	                    for p in webqsp_canonical.parses_for(qid, gold))]
	assert len(empty) == 11
	assert "WebQTest-1703" in empty


# --------------------------------------------------------------------------
# 5. best-F1-over-parses differs from union-F1
# --------------------------------------------------------------------------

def test_best_parse_f1_differs_from_union_f1():
	"""Union inflates the gold set; the canonical rule scores one parse at a time.

	Two parses of two answers each, prediction exactly equal to parse P1:
	best-parse F1 is 1.0, while scoring against the 4-answer union is not.
	"""
	parses = [parse([("Asia", "m.0j0k"), ("Africa", "m.0dg3n1")], parse_id="P0"),
	          parse([("Europe", "m.02j9z"), ("Oceania", "m.05nrg")], parse_id="P1")]
	prediction = ["Europe", "Oceania"]

	best = webqsp_canonical.score_prediction(prediction, parses)
	union = webqsp_canonical.score_prediction(
		prediction, [parse([("Asia", "m.0j0k"), ("Africa", "m.0dg3n1"),
		                    ("Europe", "m.02j9z"), ("Oceania", "m.05nrg")])])

	assert best["f1"] == pytest.approx(1.0)
	assert union["f1"] < best["f1"]
	assert best["hit"] == union["hit"] == 1


# --------------------------------------------------------------------------
# gold artifact integrity and the local-dataset relationship
# --------------------------------------------------------------------------

def test_gold_covers_every_local_question():
	gold = webqsp_canonical.load_gold()
	with open(LOCAL_DATASET, encoding="utf-8") as f:
		local = json.load(f)
	assert len(local) == 1639
	assert all(record["id"] in gold for record in local)


def test_local_dataset_is_a_subset_of_the_official_parses():
	"""Pins the defect the canonical rule fixes.

	Every local answer list matches SOME parse or the union; on 38 questions the
	official parses carry answers Parses[0] lacks, so the local file alone
	cannot reproduce the canonical F1.
	"""
	gold = webqsp_canonical.load_gold()
	with open(LOCAL_DATASET, encoding="utf-8") as f:
		local = json.load(f)
	short = 0
	for record in local:
		local_names = {a.get("name") for a in record.get("answer") or []}
		union = set()
		for p in webqsp_canonical.parses_for(record["id"], gold):
			union |= set(webqsp_canonical.parse_answer_names(p))
		if local_names and local_names < union:
			short += 1
	assert short == 38


def test_no_test_question_is_skipped_by_the_official_quality_rule():
	gold = webqsp_canonical.load_gold()
	skipped = [qid for qid in gold
	           if not webqsp_canonical.evaluable(webqsp_canonical.parses_for(qid, gold))]
	assert skipped == []
