#!/usr/bin/env python3
"""Derive the parse-level WebQSP gold artifact from the official release.

The canonical WebQSP evaluator scores a prediction against *every* semantic
parse of a question and keeps the best F1 (WebQSP/eval/eval.py in the official
release). Our `datasets/webqsp/webqsp.json` carries only the answers of
`Parses[0]`, so it cannot express that rule. This script distils the official
`WebQSP.test.json` into the minimum needed to evaluate canonically, leaving the
upstream CoR dataset file untouched.

Official release (accessed 2026-09-13):

    https://download.microsoft.com/download/F/5/0/F5012144-A4FB-4084-897F-CFDA99C60BDF/WebQSP.zip
    WebQSP/data/WebQSP.test.json
    sha256 856f50eb1ede43b799b10ddd0a980e72c88475b1a2259491af57e5db410ded52

Usage:

    python scripts/build_webqsp_official_gold.py \
        --official /path/to/WebQSP.test.json \
        --out datasets/webqsp/webqsp_official_gold.json

Answer values are copied verbatim: `answer_argument` keeps the official
AnswerArgument (an MID for entities, the literal itself for values) and
`entity_name` keeps the official EntityName, which is null for literals. No
identifier is invented for a literal-valued answer.
"""

import argparse
import datetime
import hashlib
import json
import os

SOURCE_URL = ("https://download.microsoft.com/download/F/5/0/"
              "F5012144-A4FB-4084-897F-CFDA99C60BDF/WebQSP.zip")
SOURCE_MEMBER = "WebQSP/data/WebQSP.test.json"
GOLD_SCHEMA_VERSION = 1


def sha256(path):
	digest = hashlib.sha256()
	with open(path, "rb") as f:
		for block in iter(lambda: f.read(1 << 20), b""):
			digest.update(block)
	return digest.hexdigest()


def build(official_path):
	with open(official_path, encoding="utf-8") as f:
		official = json.load(f)
	questions = {}
	for entry in official["Questions"]:
		parses = []
		for parse in entry.get("Parses") or []:
			comment = parse.get("AnnotatorComment") or {}
			parses.append({
				"parse_id": parse.get("ParseId"),
				"question_quality": comment.get("QuestionQuality"),
				"parse_quality": comment.get("ParseQuality"),
				"answers": [{"answer_type": a.get("AnswerType"),
				             "answer_argument": a.get("AnswerArgument"),
				             "entity_name": a.get("EntityName")}
				            for a in (parse.get("Answers") or [])],
			})
		questions[entry["QuestionId"]] = {
			"question": entry.get("ProcessedQuestion"),
			"parses": parses,
		}
	return {
		"gold_schema_version": GOLD_SCHEMA_VERSION,
		"source": {
			"url": SOURCE_URL,
			"member": SOURCE_MEMBER,
			"sha256": sha256(official_path),
			"generated_utc": datetime.datetime.now(datetime.timezone.utc)
			                 .replace(microsecond=0).isoformat(),
		},
		"questions": questions,
	}


def main(argv=None):
	ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
	ap.add_argument("--official", required=True, help="path to the official WebQSP.test.json")
	ap.add_argument("--out", required=True, help="output path for the derived gold artifact")
	args = ap.parse_args(argv)
	gold = build(args.official)
	os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
	with open(args.out, "w", encoding="utf-8") as f:
		json.dump(gold, f, ensure_ascii=False, indent=1, sort_keys=True)
		f.write("\n")
	n_parses = sum(len(q["parses"]) for q in gold["questions"].values())
	n_answers = sum(len(p["answers"]) for q in gold["questions"].values() for p in q["parses"])
	print(f"{args.out}: {len(gold['questions']):,} questions  {n_parses:,} parses  "
	      f"{n_answers:,} answers  (source sha256 {gold['source']['sha256'][:16]}…)")
	return 0


if __name__ == "__main__":
	raise SystemExit(main())
