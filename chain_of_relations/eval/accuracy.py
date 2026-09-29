#!/usr/bin/env python3
# -*- encoding: utf-8 -*-
"""The one effectiveness scorer: Hits@1, Precision, Recall and F1.

Every result-generation path (the `eval.eval` CLI, `measurement/export_outcomes.py`,
`eval.webqsp_canonical`, `graph.py`) scores through this module, so PoG, ToG and
CoR are treated identically within a dataset. Only the gold answer sets differ by
dataset (WebQSP: every official parse; CWQ: the shipped answer list).

The rules follow the CoR paper's protocol (ACL 2026 Findings, Sec. 5.4-5.5:
strict exact match, Hits@1 = any predicted answer matches a gold answer) and
score the model's final answer only:

Answer extraction (`extract_answers`)
  1. The result-item names are joined with ", ". CoR's agent splits one JSON
     reply on "," / ";" into several items; joining puts it back together.
  2. If the text carries a JSON `"answer"` key, ONLY that list is scored.
     Rationale text never reaches the scorer.
  3. Text with a `"rationale"` key but no `"answer"` key is malformed: no answer.
  4. Otherwise each result item is one final answer, used verbatim.
  5. Blank and "None" items are dropped; nothing left is an empty prediction.
Matching: `normalize` (lowercase, strip punctuation, drop a/an/the, collapse
  whitespace), then string equality. Never substring.
Per question, over de-duplicated normalized sets:
  Precision = |pred & gold| / |pred|,  Recall = |pred & gold| / |gold|,
  F1 = harmonic mean.  Empty sets follow the official WebQSP `CalculatePRF1`:
  empty gold -> (1, 1, 1) if the prediction is empty too, else (0, 1, 0);
  empty prediction with gold -> (1, 0, 0).
  Hits@1 = 1 if any predicted answer equals any gold answer (0 for empty gold).
Several gold sets (WebQSP parses): the best-F1 set supplies P/R/F1 (ties keep
  the earliest, as the official loop does); Hits@1 is taken over the union.
"""

import json
import re
import string
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

_PUNCT = set(string.punctuation)
_ANSWER_KEY = re.compile(r'"answer"\s*:\s*\[')
_JSON_STRING = re.compile(r'"((?:[^"\\]|\\.)*)"')

#: How `extract_answers` obtained a prediction; recorded per question.
STATUS_JSON = "json_answer"
STATUS_JSON_FALLBACK = "json_answer_fallback_parse"
STATUS_PLAIN = "plain_answer_items"
STATUS_MALFORMED = "malformed_rationale_without_answer"
STATUS_EMPTY = "empty"


def normalize(s: str) -> str:
    """Lower text and remove punctuation, articles and extra whitespace."""
    s = str(s).lower()
    s = "".join(char for char in s if char not in _PUNCT)
    s = re.sub(r"\b(a|an|the)\b", " ", s)
    # remove <pad> token:
    s = re.sub(r"\b(<pad>)\b", " ", s)
    return " ".join(s.split())


def _answer_field(text: str) -> Tuple[Optional[List[str]], str]:
    """The JSON "answer" list inside `text`, or (None, "") when there is none."""
    m = _ANSWER_KEY.search(text)
    if not m:
        return None, ""
    start = m.end() - 1  # the '['
    end = text.find("]", start)
    if end != -1:
        try:
            parsed = json.loads(text[start:end + 1])
            if isinstance(parsed, list):
                return [str(x) for x in parsed if isinstance(x, (str, int, float))], STATUS_JSON
        except ValueError:
            pass
    segment = text[start:end + 1] if end != -1 else text[start:]
    # Narrowest fallback: the quoted strings inside the answer list, nothing else.
    return [_json_unescape(s) for s in _JSON_STRING.findall(segment)], STATUS_JSON_FALLBACK


def _json_unescape(s: str) -> str:
    try:
        return json.loads(f'"{s}"')
    except ValueError:
        return s


def extract_answers(results: Any) -> Tuple[List[str], str]:
    """Final answer strings from a record's `results` list, and how they were found."""
    items = [r["name"] for r in (results if isinstance(results, list) else [])
             if isinstance(r, dict) and isinstance(r.get("name"), str)]
    if not items:
        return [], STATUS_EMPTY
    joined = ", ".join(items)
    answers, status = _answer_field(joined)
    if answers is None:
        if '"rationale"' in joined:
            return [], STATUS_MALFORMED
        answers, status = items, STATUS_PLAIN
    answers = [a.strip() for a in answers if a and a.strip() and a.strip().lower() != "none"]
    return answers, (status if answers else STATUS_EMPTY)


def answer_set(values: Iterable[str]) -> set:
    """Distinct normalized, non-empty answers."""
    return {n for n in (normalize(v) for v in values) if n}


def prf(prediction: Sequence[str], gold: Sequence[str]) -> Tuple[float, float, float]:
    """(precision, recall, f1) for one gold set."""
    pred, gold_set = answer_set(prediction), answer_set(gold)
    if not gold_set:
        return (1.0, 1.0, 1.0) if not pred else (0.0, 1.0, 0.0)
    if not pred:
        return 1.0, 0.0, 0.0
    tp = len(pred & gold_set)
    precision, recall = tp / len(pred), tp / len(gold_set)
    return precision, recall, (0.0 if tp == 0 else 2 * precision * recall / (precision + recall))


def hit(prediction: Sequence[str], gold: Sequence[str]) -> int:
    """1 if any predicted answer equals any gold answer."""
    return int(bool(answer_set(prediction) & answer_set(gold)))


def score(prediction: Sequence[str], gold_sets: Sequence[Sequence[str]]) -> Dict[str, Any]:
    """Score one question against one (CWQ) or several (WebQSP parses) gold sets.

    Returns precision, recall, f1 (from the best-F1 gold set), best_index,
    hit1 (any gold set) and empty_gold (no gold set has an answer).
    """
    best = None
    for index, gold in enumerate(gold_sets):
        precision, recall, f1 = prf(prediction, gold)
        if best is None or f1 > best["f1"]:
            best = {"precision": precision, "recall": recall, "f1": f1, "best_index": index}
    if best is None:
        precision, recall, f1 = prf(prediction, [])
        best = {"precision": precision, "recall": recall, "f1": f1, "best_index": None}
    usable = [g for g in gold_sets if answer_set(g)]
    best["hit1"] = int(any(hit(prediction, g) for g in usable))
    best["empty_gold"] = not usable
    return best
