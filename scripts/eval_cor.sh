#!/usr/bin/env bash
# Evaluate a finished CoR run. Override with environment variables:
#   DATASET=cwq MODEL_DIR=Qwen2.5-7B-Instruct scripts/eval_cor.sh
# MODEL_DIR is the directory name under results/<method>/<dataset>/, which is
# the served model id with the organisation prefix stripped.
set -eu

DATASET="${DATASET:-webqsp}"
MODEL_DIR="${MODEL_DIR:-gemma-3-4b-it}"
METHOD="${METHOD:-cor}"

python -m chain_of_relations.eval.eval \
	--dataset "$DATASET" \
	--output_dir "results/${METHOD}/${DATASET}/${MODEL_DIR}"
