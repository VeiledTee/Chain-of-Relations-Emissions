#!/usr/bin/env python3
"""Derive the WebQSP answerability groups from the official gold and our Freebase.

A one-time, read-only build step. It writes
`datasets/webqsp/webqsp_answerability_groups.csv` (`question_id,group`), which
the figure pipeline reads instead of re-querying the graph. Every one of the
1,639 official WebQSP test questions lands in exactly one group:

    empty_gold      `Answers: []` on every parse in the official release
                    (datasets/webqsp/webqsp_official_gold.json)
    query_mismatch  non-empty gold, but the stored gold query executed unchanged
                    against our Freebase returns none of the gold answers
    expected        non-empty gold, and the stored gold query returns all of it

"Stored gold query" is the `sparql` field of `datasets/webqsp/webqsp.json`
(the query of `Parses[0]`), compared against that parse's `answer` list: MIDs
by identity, literals by the evaluator's `normalize` (or a prefix match, so a
typed date literal matches its official year string). A question that fits no
group -- the query errors or times out, or returns only part of the gold -- is
not silently bucketed: the script stops and names it.

Usage (needs the local Virtuoso endpoint; ~5 s):

    python scripts/build_webqsp_answerability_groups.py \
        --out datasets/webqsp/webqsp_answerability_groups.csv

    FREEBASE_SPARQL_ENDPOINT overrides http://127.0.0.1:8890/sparql
"""

import argparse
import csv
import json
import os
import sys
import urllib.parse
import urllib.request

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "measurement"))

from chain_of_relations.eval.accuracy import normalize  # noqa: E402
import webqsp_answerability as wa  # noqa: E402

NS = "http://rdf.freebase.com/ns/"
TIMEOUT_S = 120


def sparql_values(endpoint, query):
	"""Values of the first SELECT variable; raises on any failure or partial result."""
	data = urllib.parse.urlencode({"query": query, "format": "application/sparql-results+json",
	                               "timeout": str(TIMEOUT_S * 1000)}).encode()
	request = urllib.request.Request(endpoint, data=data,
	                                 headers={"Accept": "application/sparql-results+json"})
	with urllib.request.urlopen(request, timeout=TIMEOUT_S + 15) as response:
		state = response.headers.get("X-SQL-State", "")
		if state and state != "00000":
			raise RuntimeError(f"partial result (X-SQL-State {state})")
		payload = json.load(response)
	var = (payload.get("head", {}).get("vars") or [None])[0]
	values = []
	for binding in payload["results"]["bindings"]:
		term = binding.get(var)
		if term:
			value = term.get("value", "")
			values.append(value[len(NS):] if term.get("type") == "uri" and value.startswith(NS)
			              else value)
	return values


def classify(question, official_empty, values):
	"""Group for one question, or None when it fits none of the three."""
	if official_empty:
		return wa.EMPTY_GOLD
	gold = question["answer"]
	returned = set(values)

	def returned_by_query(answer):
		if answer["id"]:
			return answer["id"] in returned
		return any(normalize(v) == normalize(answer["name"]) or str(v).startswith(answer["name"])
		           for v in returned)

	got = sum(1 for a in gold if returned_by_query(a))
	if got == len(gold):
		return wa.EXPECTED
	return wa.QUERY_MISMATCH if got == 0 else None


def main(argv=None):
	ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
	ap.add_argument("--dataset", default=os.path.join(ROOT, "datasets/webqsp/webqsp.json"))
	ap.add_argument("--out", default=wa.DEFAULT_GROUPS_PATH)
	ap.add_argument("--endpoint", default=os.environ.get("FREEBASE_SPARQL_ENDPOINT",
	                                                     "http://127.0.0.1:8890/sparql"))
	args = ap.parse_args(argv)

	official = wa.official_question_ids()
	empty = wa.official_empty_gold_ids()
	questions = json.load(open(args.dataset))
	if {q["id"] for q in questions} != set(official):
		raise SystemExit("dataset ids differ from the official gold ids")

	groups, unplaced = {}, []
	for q in questions:
		if q["id"] not in empty and not q["answer"]:
			unplaced.append((q["id"], "Parses[0] empty but another official parse is not"))
			continue
		try:
			values = [] if q["id"] in empty else sparql_values(args.endpoint, q.get("sparql") or "")
		except Exception as exc:
			unplaced.append((q["id"], f"gold query failed: {exc}"))
			continue
		group = classify(q, q["id"] in empty, values)
		if group is None:
			unplaced.append((q["id"], "gold query returns only part of the gold"))
			continue
		groups[q["id"]] = group
		if group == wa.QUERY_MISMATCH:
			gold = "|".join(a["id"] or a["name"] for a in q["answer"])
			print(f"  query_mismatch {q['id']}: gold {gold}; query returns "
			      f"{'|'.join(values[:3]) or '(no rows)'}")
	if unplaced:
		for qid, why in unplaced:
			print(f"UNPLACED {qid}: {why}", file=sys.stderr)
		raise SystemExit(f"{len(unplaced)} questions fit no group; nothing written")

	with open(args.out, "w", newline="") as f:
		writer = csv.writer(f, lineterminator="\n")
		writer.writerow(["question_id", "group"])
		for qid in sorted(groups, key=wa.question_sort_key):
			writer.writerow([qid, groups[qid]])
	wa.load_groups(args.out)  # the same validation the figure pipeline applies
	counts = {g: sum(1 for v in groups.values() if v == g) for g in wa.GROUP_ORDER}
	print(f"{args.out}: {len(groups):,} questions  " +
	      "  ".join(f"{g}={n:,}" for g, n in counts.items()))
	return 0


if __name__ == "__main__":
	sys.exit(main())
