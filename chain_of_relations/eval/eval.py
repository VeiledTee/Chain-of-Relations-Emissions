#!/usr/bin/env python3
# -*- encoding: utf-8 -*-
"""
@File    :   eval.py
@Time    :   2025/03/27 20:25:26
@Author  :   liuchenhui
@Contact :   liuchh9@mail2.sysu.edu.cn
@Desc    :   None
"""

import re
import json
import argparse
import os

from chain_of_relations.eval import webqsp_canonical
from chain_of_relations.eval.accuracy import eval_acc, eval_f1, eval_hit
from chain_of_relations.eval.faithfulness import sorted_knowledge_distribution, update_knowledge_distribution
from chain_of_relations.eval.efficiency import analyze_detailed_results


#: Re-exported so callers can ask "is this dataset scored canonically?" without
#: reaching past the evaluator. Defined once in webqsp_canonical.
CANONICAL_DATASETS = webqsp_canonical.CANONICAL_DATASETS


def load_predictions(output_file):
    """Load prediction results from jsonl file."""
    output_datas = []
    with open(output_file, 'r') as f:
        for line in f:
            try:
                data_row = json.loads(line)
            except BaseException:
                print(line)
                exit(0)
            output_datas.append(data_row)
    return output_datas

def extract_answer_from_braces(text):
    """Extract answer from curly braces in ToG-style results."""
    if not isinstance(text, str):
        return None

    # Find content within curly braces {}
    import re
    pattern = r'\{([^}]+)\}'
    matches = re.findall(pattern, text)

    if matches:
        # Take the last match (usually the final answer)
        answer_text = matches[-1].strip()

        # Handle multiple answers separated by comma or "and"
        # Split by comma or "and" and clean up
        if ',' in answer_text:
            answers = [a.strip() for a in answer_text.split(',')]
        elif ' and ' in answer_text.lower():
            answers = [a.strip() for a in re.split(r'\s+and\s+', answer_text, flags=re.IGNORECASE)]
        else:
            answers = [answer_text]

        return answers

    return None


def is_integer_text(text: str) -> bool:
    return bool(re.fullmatch(r"[-+]?\d+", str(text or "").strip()))


def postprocess_prediction_for_dataset(dataset: str, prediction: list, gold_answer: list) -> list:
    dataset_key = str(dataset or "").strip().lower()
    if dataset_key not in {"quad2", "quad_2"}:
        return prediction

    if not prediction:
        return prediction

    normalized = [str(item).strip() for item in prediction]

    for item in normalized:
        if is_integer_text(item):
            return [item]

    gold_non_empty = [str(item).strip() for item in gold_answer if str(item).strip() and str(item).strip().lower() != "none"]
    gold_is_numeric = bool(gold_non_empty) and all(is_integer_text(item) for item in gold_non_empty)
    if not gold_is_numeric:
        return normalized

    non_empty = [str(item).strip() for item in normalized if str(item).strip() and str(item).strip().lower() != "none"]
    return [str(len(non_empty))]

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Evaluate QA model results')
    parser.add_argument("--dataset", type=str, required=True,
                        help="Dataset name (e.g., cwq, webqsp)")
    parser.add_argument("--output_dir", type=str, required=True,
                        help="Output directory containing predict.jsonl and detailed JSON files")
    parser.add_argument("--run_size", type=int, default=-1,
                        help="Number of prediction rows to evaluate (-1 for all)")
    args = parser.parse_args()

    # Construct file paths
    args.output_file = os.path.join(args.output_dir, "predict.jsonl")
    result_dir = args.output_dir

    output_datas = load_predictions(args.output_file)
    if args.run_size > 0:
        output_datas = output_datas[:args.run_size]

    total_data = len(output_datas)
    if total_data == 0:
        print("No samples to evaluate.")
        exit(0)
    print("total data: ", total_data)

    # Analyze detailed results for LLM statistics
    llm_stats = analyze_detailed_results(result_dir)

    # WebQSP is scored canonically, against every official parse. Every other
    # dataset keeps the single-gold-list scoring it already had.
    canonical_gold = (webqsp_canonical.load_gold()
                      if str(args.dataset or "").strip().lower() in CANONICAL_DATASETS else None)
    empty_gold_ids = []
    skipped_ids = []
    missing_gold_ids = []
    usable_f1_list = []
    usable_hit_list = []

    acc_list = []
    hit_list = []
    f1_list = []
    precission_list = []
    recall_list = []
    knowledge_distribution = {}

    for pred_data in output_datas:
        # Extract gold answers (list of names)
        gold_answers = pred_data.get("gold_answer", [])
        answer = [ans["name"] for ans in gold_answers if "name" in ans]

        # Extract predictions (list of names)
        pred_results = pred_data.get("results", [])
        if pred_results is None or len(pred_results) == 0:
            prediction = ["None"]
        elif isinstance(pred_results, list):
            # Extract names from result objects
            raw_predictions = [res["name"] for res in pred_results if isinstance(res, dict) and "name" in res]

            # Check if predictions contain answers in curly braces (ToG format)
            prediction = []
            for raw_pred in raw_predictions:
                extracted = extract_answer_from_braces(raw_pred)
                if extracted:
                    prediction.extend(extracted)
                else:
                    # If no braces found, use the raw prediction
                    prediction.append(raw_pred)

            if len(prediction) == 0:
                prediction = ["None"]
        else:
            prediction = ["None"]

        # Dataset-specific post-processing (quad2 numeric-answer samples)
        prediction = postprocess_prediction_for_dataset(args.dataset, prediction, answer)

        prediction_str = " ".join(prediction)

        # knowledge distribution
        predict_type = pred_data.get("action", "unknown")
        update_knowledge_distribution(knowledge_distribution, predict_type)

        if canonical_gold is not None:
            # Canonical WebQSP: best F1 over the official parses, Hit@1 against
            # any parse. `answer` above holds only Parses[0] and is not scored.
            parses = webqsp_canonical.parses_for(pred_data["id"], canonical_gold)
            if not parses:
                missing_gold_ids.append(pred_data["id"])
            elif not webqsp_canonical.evaluable(parses):
                # The official evaluator skips a question with no Good+Complete
                # parse. No such question exists in the 1,639-question test set;
                # if a future gold release introduces one, skip it and say so.
                skipped_ids.append(pred_data["id"])
                continue
            scored = webqsp_canonical.score_prediction(prediction, parses)
            f1_score = scored["f1"]
            precision_score, recall_score = scored["precision"], scored["recall"]
            hit = scored["hit"]
            best_names = (webqsp_canonical.parse_answer_names(parses[scored["best_parse_index"]])
                          if scored["best_parse_index"] is not None else [])
            acc = eval_acc(prediction_str, best_names) if best_names else 0.0
            if scored["empty_gold"]:
                empty_gold_ids.append(pred_data["id"])
            else:
                usable_f1_list.append(f1_score)
                usable_hit_list.append(hit)
        else:
            try:
                f1_score, precision_score, recall_score = eval_f1(prediction, answer)
            except BaseException as e:
                print(f"Error: {e}, {pred_data['id']}")
                continue
            acc = eval_acc(prediction_str, answer)
            hit = eval_hit(prediction_str, answer)

        f1_list.append(f1_score)
        precission_list.append(precision_score)
        recall_list.append(recall_score)
        acc_list.append(acc)
        hit_list.append(hit)

    result_str = "Accuracy: " + str(sum(acc_list) * 100 / len(acc_list)) + " Hit: " + str(sum(hit_list) * 100 / len(hit_list)) + " F1: " + str(sum(f1_list) * 100 / len(f1_list)) + " Precision: " + str(sum(precission_list) * 100 / len(precission_list)) + " Recall: " + str(sum(recall_list) * 100 / len(recall_list))

    # Extract method name from output_dir for display
    method_name = os.path.basename(args.output_dir)

    print("\n" + "="*50)
    print(f"Evaluation Results for {args.dataset} - {method_name}")
    print("="*50)
    print(result_str)
    print(f"\nHit@1: {sum(hit_list) * 100 / len(hit_list):.2f}%")
    print(f"F1: {sum(f1_list) * 100 / len(f1_list):.2f}%")
    print(f"Precision: {sum(precission_list) * 100 / len(precission_list):.2f}%")
    print(f"Recall: {sum(recall_list) * 100 / len(recall_list):.2f}%")
    print(f"Accuracy: {sum(acc_list) * 100 / len(acc_list):.2f}%")
    if canonical_gold is not None:
        print(f"\nScoring: canonical WebQSP - best F1 over the official parses, "
              f"Hit@1 over any parse, {len(f1_list)} questions in the denominator.")
        if missing_gold_ids:
            print(f"  WARNING: {len(missing_gold_ids)} question(s) absent from the parse-level "
                  f"gold, scored as empty gold: {missing_gold_ids[:5]}")
        if skipped_ids:
            print(f"  Skipped, no Good+Complete parse: {len(skipped_ids)} -> {skipped_ids[:5]}")
        if usable_f1_list:
            print(f"  Non-empty-gold subset: {len(usable_f1_list)} questions, excludes "
                  f"{len(empty_gold_ids)} official empty-gold questions -> "
                  f"Hit@1 {sum(usable_hit_list) * 100 / len(usable_hit_list):.2f}%  "
                  f"F1 {sum(usable_f1_list) * 100 / len(usable_f1_list):.2f}%")
            print("  Headline WebQSP numbers are the full-denominator ones above.")
    print("\nAction Distribution:")
    for action, count in sorted_knowledge_distribution(knowledge_distribution):
        print(f"  {action}: {count} ({count*100/total_data:.2f}%)")

    # Print LLM statistics if available
    if llm_stats:
        print("\n" + "-"*50)
        print("LLM Usage Statistics")
        print("-"*50)
        print(f"Total Queries Analyzed: {llm_stats['total_queries']}")
        print(f"Average LLM Calls per Query: {llm_stats['avg_llm_calls']:.2f}")
        print(f"Average Input Tokens per Query: {llm_stats['avg_input_tokens']:.2f}")
        print(f"Average Output Tokens per Query: {llm_stats['avg_output_tokens']:.2f}")
        print(f"Average Total Tokens per Query: {llm_stats['avg_total_tokens']:.2f}")

        # Calculate average tokens per LLM call
        if llm_stats['total_llm_calls'] > 0:
            avg_tokens_per_call = (llm_stats['total_input_tokens'] + llm_stats['total_output_tokens']) / llm_stats['total_llm_calls']
            avg_input_per_call = llm_stats['total_input_tokens'] / llm_stats['total_llm_calls']
            avg_output_per_call = llm_stats['total_output_tokens'] / llm_stats['total_llm_calls']
            print(f"\nAverage Tokens per LLM Call:")
            print(f"  Input: {avg_input_per_call:.2f}")
            print(f"  Output: {avg_output_per_call:.2f}")
            print(f"  Total: {avg_tokens_per_call:.2f}")

        print(f"\nTotal Statistics:")
        print(f"  Total LLM Calls: {llm_stats['total_llm_calls']:,}")
        print(f"  Total Input Tokens: {llm_stats['total_input_tokens']:,}")
        print(f"  Total Output Tokens: {llm_stats['total_output_tokens']:,}")
        print(f"  Total Tokens: {llm_stats['total_input_tokens'] + llm_stats['total_output_tokens']:,}")

    print("="*50)
