#!/usr/bin/env bash
# Sequential measured runs for one or more paradigms, one at a time.
#
# One paradigm at a time on purpose: a measured run needs the machine to
# itself, and each results directory has a single writer.
#
# Everything is configurable; nothing is hard-coded to a particular machine.
# The repository root is derived from this script's own location.
#
#   scripts/run_experiments.sh --check              # verify services only, no run
#   PRED_DIR=/tmp/smoke scripts/run_experiments.sh  # smoke: 1 question, CoR ToG PoG
#   RUN_SIZE=-1 scripts/run_experiments.sh          # full dataset (WebQSP: 1,639 questions)
#   METHODS="cor" RUN_SIZE=25 scripts/run_experiments.sh
#   DATASET=cwq DEPTH=4 RUN_SIZE=-1 scripts/run_experiments.sh
#
# Configuration (environment variables, all with defaults):
#   METHODS       space-separated paradigms to run        (default "cor tog pog")
#   DATASET       webqsp | cwq | qald10_en                (default webqsp)
#   RUN_SIZE      questions per paradigm, -1 = all        (default 1)
#   DEPTH         max search depth                        (default 3)
#   RELATION_WIDTH / ENTITY_WIDTH                         (default 3 / 3)
#   MODEL_NAME    served model id, must match vLLM        (default google/gemma-3-4b-it)
#   MODEL_REVISION  model commit; required for a citable measurement
#   OPENAI_BASE_URL / OPENAI_API_KEY / OPENAI_TIMEOUT
#   FREEBASE_SPARQL_ENDPOINT
#   LLM_HEALTH_TIMEOUT  seconds for the test generation            (default 60)
#   FREEBASE_GRAPH    graph counted by EXPECTED_TRIPLES    (default http://freebase.com)
#   EXPECTED_TRIPLES  assert this many triples in FREEBASE_GRAPH; empty (default)
#                 = only check the endpoint answers. The DEFAULT-graph total is
#                 not a stable invariant: it also counts Virtuoso's own system
#                 graphs, which differ between images, so count the data graph.
#   TRIPLE_COUNT_TIMEOUT  seconds for that count            (default 600)
#   RUNS_DIR      measurement run root                    (default <repo>/measurement/runs)
#   PRED_DIR      predictions root; each method writes PRED_DIR/<method>.
#                 Empty = results/<method>/<dataset>/<model>, which RESUMES past
#                 questions already answered there. Set it for smoke runs, or a
#                 finished experiment's predictions make them answer nothing.
#   TAG_PREFIX    extra prefix on each run tag            (default empty)
#   VENV          virtualenv to activate, empty = use the active interpreter
#
# Exits non-zero if a preflight check fails or any paradigm fails.

set -uo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

METHODS="${METHODS:-cor tog pog}"
DATASET="${DATASET:-webqsp}"
RUN_SIZE="${RUN_SIZE:-1}"
DEPTH="${DEPTH:-3}"
RELATION_WIDTH="${RELATION_WIDTH:-3}"
ENTITY_WIDTH="${ENTITY_WIDTH:-3}"
TEMPERATURE_EXPLORATION="${TEMPERATURE_EXPLORATION:-0.01}"
TEMPERATURE_REASONING="${TEMPERATURE_REASONING:-0.01}"

export MODEL_NAME="${MODEL_NAME:-google/gemma-3-4b-it}"
export MODEL_REVISION="${MODEL_REVISION:-}"
export OPENAI_BASE_URL="${OPENAI_BASE_URL:-http://localhost:8000/v1}"
export OPENAI_API_KEY="${OPENAI_API_KEY:-dummy}"
export OPENAI_TIMEOUT="${OPENAI_TIMEOUT:-300}"
export FREEBASE_SPARQL_ENDPOINT="${FREEBASE_SPARQL_ENDPOINT:-http://127.0.0.1:8890/sparql}"

RUNS_DIR="${RUNS_DIR:-$REPO_ROOT/measurement/runs}"
PRED_DIR="${PRED_DIR:-}"
LLM_HEALTH_TIMEOUT="${LLM_HEALTH_TIMEOUT:-60}"
FREEBASE_GRAPH="${FREEBASE_GRAPH:-http://freebase.com}"
TRIPLE_COUNT_TIMEOUT="${TRIPLE_COUNT_TIMEOUT:-600}"
EXPECTED_TRIPLES="${EXPECTED_TRIPLES:-}"
VENV="${VENV:-$REPO_ROOT/.venv}"
STAMP="$(date +%m%d_%H%M)"

cd "$REPO_ROOT" || exit 1
if [ -n "$VENV" ] && [ -f "$VENV/bin/activate" ]; then
	# shellcheck disable=SC1091
	source "$VENV/bin/activate"
fi

fail() { echo "ERROR: $*" >&2; exit 1; }

preflight() {
	local models reply probe triples

	models="$(curl -sf --max-time 10 "${OPENAI_BASE_URL%/}/models" || true)"
	[ -n "$models" ] || fail "no LLM endpoint at ${OPENAI_BASE_URL%/}/models — is vLLM running?"
	grep -q "$MODEL_NAME" <<<"$models" || fail \
		"LLM endpoint is up but does not serve '$MODEL_NAME'. Served: $(grep -o '"id":"[^"]*"' <<<"$models" | tr '\n' ' ')"

	# A served model list is NOT proof the server can generate. A wedged vLLM
	# kept answering /v1/models while every completion hung: a 5-token request
	# timed out at 120 s with the GPU pinned at 100%, and the experiment then
	# burned an hour on retries. So complete one tiny generation before
	# starting anything.
	reply="$(curl -sf --max-time "$LLM_HEALTH_TIMEOUT" \
		"${OPENAI_BASE_URL%/}/chat/completions" \
		-H 'Content-Type: application/json' \
		-H "Authorization: Bearer $OPENAI_API_KEY" \
		-d "{\"model\":\"$MODEL_NAME\",\"messages\":[{\"role\":\"user\",\"content\":\"Say OK\"}],\"max_tokens\":5,\"temperature\":0}" \
		|| true)"
	[ -n "$reply" ] || fail \
		"LLM endpoint lists '$MODEL_NAME' but did not complete a ${LLM_HEALTH_TIMEOUT}s test generation. The server is up but not serving (wedged engine, model still loading, or a stuck request). Restart it before running an experiment."
	grep -q '"content"' <<<"$reply" || fail \
		"LLM test generation returned no content: $(head -c 300 <<<"$reply")"

	# Freebase: availability by default. The exact triple total is NOT a
	# reliable invariant (see EXPECTED_TRIPLES above), and a full COUNT(*) over
	# 3.1B triples costs minutes, so only run it when explicitly asked.
	probe="$(curl -sf --max-time 30 "$FREEBASE_SPARQL_ENDPOINT" \
		--data-urlencode 'query=SELECT ?s WHERE { ?s ?p ?o } LIMIT 1' \
		--data-urlencode 'format=application/json' || true)"
	grep -q '"bindings"' <<<"$probe" || fail \
		"Freebase SPARQL endpoint $FREEBASE_SPARQL_ENDPOINT did not answer a trivial query"

	triples="(not counted)"
	if [ -n "$EXPECTED_TRIPLES" ]; then
		triples="$(curl -sf --max-time "$TRIPLE_COUNT_TIMEOUT" "$FREEBASE_SPARQL_ENDPOINT" \
			--data-urlencode "query=SELECT (COUNT(*) AS ?c) WHERE { GRAPH <$FREEBASE_GRAPH> {?s ?p ?o} }" \
			--data-urlencode 'format=application/json' \
			| grep -o '"value": "[0-9]*"' | head -1 | grep -o '[0-9]*' || true)"
		[ -n "$triples" ] || fail \
			"could not count triples in <$FREEBASE_GRAPH> within ${TRIPLE_COUNT_TIMEOUT}s"
		[ "$triples" = "$EXPECTED_TRIPLES" ] || fail \
			"graph <$FREEBASE_GRAPH> holds $triples triples, expected $EXPECTED_TRIPLES"
	fi

	[ -n "$MODEL_REVISION" ] || echo \
		"WARNING: MODEL_REVISION is empty — these runs are validation runs, not citable measurements." >&2
	echo "preflight OK: model=$MODEL_NAME generation=ok kg=reachable triples=$triples"
}

if [ "$RUN_SIZE" = "-1" ]; then
	echo "RUN_SIZE=-1: FULL $DATASET run per paradigm (WebQSP is 1,639 questions; expect hours each)."
else
	echo "RUN_SIZE=$RUN_SIZE question(s) per paradigm."
fi
echo "=== start $(date -Is) | methods='$METHODS' dataset=$DATASET depth=$DEPTH runs_dir=$RUNS_DIR ==="

preflight || exit 1

# --check / PREFLIGHT_ONLY=1: verify the services and stop. Use it before a long
# run, or to prove an endpoint can actually generate.
if [ "${1:-}" = "--check" ] || [ -n "${PREFLIGHT_ONLY:-}" ]; then
	echo "preflight only: no experiment started."
	exit 0
fi

status=0
for METHOD in $METHODS; do
	# TAG_PREFIX prefixes the per-method tag rather than replacing it: a shared
	# tag across methods would collide, and measure_run refuses tag reuse.
	TAG="${TAG_PREFIX:+${TAG_PREFIX}_}${METHOD}_${DATASET}_${STAMP}"
	OUTDIR="$RUNS_DIR/$TAG"
	echo
	echo "=== $METHOD -> $OUTDIR | $(date -Is) ==="
	# PRED_DIR sends predictions somewhere other than results/<method>/<dataset>/
	# <model>. Without it a repeat run resumes from the finished predict.jsonl
	# there and answers nothing, which is what you want for a real experiment
	# and never what you want for a smoke run.
	pred_args=()
	[ -n "$PRED_DIR" ] && pred_args=(--output_dir "$PRED_DIR/$METHOD")

	python measurement/measure_run.py --tag "$TAG" -- \
		--method "$METHOD" --dataset "$DATASET" --kb freebase \
		--run_size "$RUN_SIZE" "${pred_args[@]}" \
		--relation_width "$RELATION_WIDTH" --entity_width "$ENTITY_WIDTH" \
		--depth "$DEPTH" \
		--temperature_exploration "$TEMPERATURE_EXPLORATION" \
		--temperature_reasoning "$TEMPERATURE_REASONING" \
		--save_detail true
	rc=$?
	echo "=== $METHOD exit=$rc | $(date -Is) ==="
	[ "$rc" -eq 0 ] || status=$rc
done

echo
echo "=== operation labels per run | $(date -Is) ==="
for d in "$RUNS_DIR"/*_"$STAMP"; do
	[ -d "$d" ] || continue
	echo "--- $d"
	python - "$d/events.jsonl" <<'PY'
import collections, json, sys
counts = collections.Counter()
try:
	with open(sys.argv[1]) as f:
		for line in f:
			counts[json.loads(line)["operation_label"]] += 1
except FileNotFoundError:
	print("   no events.jsonl"); raise SystemExit
for label, n in sorted(counts.items()):
	print(f"   {label:<34} {n}")
PY
done

exit "$status"
