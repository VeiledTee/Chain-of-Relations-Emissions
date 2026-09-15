# Chain-of-Relations Emissions

An **energy-instrumented fork of [Chain-of-Relations (CoR)](#upstream-chain-of-relations)**,
used as a unified experimental harness for **hardware-grounded, operation-level
energy measurement of agentic KGQA**.

> **Status:** the measurement instrument is built and validated in software and
> against real small-model inference. **No final experimental results have been
> produced.** See [docs/ROADMAP.md](docs/ROADMAP.md).

This README is the reproducibility entry point: clone → environment → services →
run → evaluate → figures. Measurement *semantics* live in
[MEASUREMENT_SPEC.md](MEASUREMENT_SPEC.md), operational detail in
[docs/RUNBOOK.md](docs/RUNBOOK.md).

---

## Project overview

The question is where energy is actually spent when inference is embedded in a
multi-step agentic loop over a knowledge graph — energy attributed to *what the
agent was doing*, not merely that a model was called.

Three KGQA paradigms are compared over one shared graph, model-serving stack and
measurement procedure:

| Paradigm | What it does | Semantic LLM stages emitted |
|---|---|---|
| **CoR** | relation-centric DFS over the graph | `llm:relation_rank`, `llm:reason`, `llm:answer_filter`, `llm:direct_answer` |
| **ToG** | beam search over entity/relation frontiers | `llm:relation_rank`, `llm:entity_prune`, `llm:reason`, `llm:direct_answer` |
| **PoG** | plan-then-traverse with working memory and reverse retrieval | `llm:subquestion_decompose`, `llm:relation_rank`, `llm:entity_prune`, `llm:memory_update`, `llm:reason`, `llm:reverse_retrieval_decision`, `llm:reverse_entity_select`, `llm:direct_answer`, `embedding:prune` |

All three also emit graph operations: `kg:relation_search`, `kg:entity_search`,
`kg:id2name`, `kg:sparql`. Labels are assigned by the caller that knows *why* an
operation is happening and are never inferred from prompt text; the canonical
list is [`chain_of_relations/energy_taxonomy.py`](chain_of_relations/energy_taxonomy.py).

**Datasets.** `webqsp` (1,639 test questions) and `cwq` over a self-hosted
Freebase; `qald10_en` over Wikidata. **The Wikidata backend emits no KG events**,
so on `qald10_en` graph energy is unmeasured, not zero.

**What the profiler measures.** Per semantic operation: wall-clock window, GPU
energy from the NVML cumulative counter, CPU-package and DRAM energy from RAPL
where readable, token counts, status, and full run provenance. The measurement
boundary is

```
measured_energy_j = gpu_energy_j + cpu_package_energy_j + dram_energy_j
```

CPU `core` is diagnostic only and never added (it is inside `package`). A domain
the hardware did not report stays `null`, never `0`.

---

## Repository structure

| Path | Contents |
|---|---|
| `agent_energy_profiler/` | the measurement instrument: a standalone, paradigm-agnostic package (events, sampling, attribution, trajectory accounting, aggregation, figures) |
| `chain_of_relations/` | the agents (`methods/cor`, `methods/tog`, `methods/pog`), KG backends, LLM API wrapper, runner (`run.py`), evaluators (`eval/`) |
| `measurement/` | experiment-side entry points: measured-run orchestrator, hardware probe, outcome export, figure packages, run audits |
| `measurement/runs/` | measured run artifacts (git-ignored) |
| `results/` | predictions and per-question detail JSONs (git-ignored) |
| `datasets/` | question sets plus `webqsp_official_gold.json`, the parse-level WebQSP gold |
| `scripts/` | `run_experiments.sh` (portable run driver), gold builder, git hooks |
| `tests/` | profiler, instrumentation, boundary, evaluation and figure tests |
| `docs/` | SETUP, RUNBOOK, ROADMAP |

The dependency direction is one-way and test-enforced
(`tests/test_package_boundary.py`): a host may import the profiler, never the
reverse.

---

## Requirements

**Python / software**

- Python 3.9+ (3.12 used for the current runs); `python3-venv`, `git`, `curl`
- Runtime dependencies in `requirements.txt` (openai, PyYAML, tqdm,
  SPARQLWrapper, sentence-transformers, matplotlib, numpy)
- `pytest` for the test suite (`pip install -e ".[dev]"`)

**LLM serving** — an OpenAI-compatible endpoint. vLLM is what the runs use; any
server that answers `/v1/models` and `/v1/chat/completions` works. The model id
in `MODEL_NAME` must match the served id exactly. `--max-model-len 32768` is
required; 8192 caused silent agentic degradation.

**Knowledge graph** — Freebase in Virtuoso, reachable over SPARQL
(the loaded dump is **3,124,791,155** triples in `<http://freebase.com>`), or
Wikidata for `qald10_en`.
Build instructions: [docs/SETUP.md](docs/SETUP.md).

**GPU / NVML** — an NVIDIA GPU plus `nvidia-ml-py` (`pip install -e ".[gpu]"`)
for GPU energy. Without it GPU energy is reported unavailable, never estimated.

**CPU / DRAM RAPL** — readable Intel/AMD RAPL counters
(`/sys/class/powercap/...`) for `cpu_package_energy_j` and `dram_energy_j`.

> **Development hosts usually produce GPU-only measurements.** WSL2 exposes no
> RAPL, so `measured_energy_j` is `null` there and the run is a *validation run*,
> not a citable measurement. A complete measurement requires every boundary
> domain to be readable — check with the hardware probe below, and see
> [MEASUREMENT_SPEC.md](MEASUREMENT_SPEC.md) §12.

---

## Installation

```bash
git clone -b energy-instrumentation https://github.com/VeiledTee/Chain-of-Relations-Emissions.git
cd Chain-of-Relations-Emissions

python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt

pip install -e ".[gpu]"      # NVML GPU counters
pip install -e ".[dev]"      # pytest, for the test suite
```

Check the harness imports and the host's measurement capability:

```bash
python -c "import chain_of_relations, agent_energy_profiler; print('harness OK')"
python measurement/validate_hardware.py --json measurement/hardware_report.json
```

Read the **MEASUREMENT BOUNDARY** block in that output. All three additive
domains must be `SUPPORTED` and `measurement_complete possible on this host`
must be `True` before a run can be cited.

---

## External services

Both services must be running before any experiment. Nothing in the harness
starts them for you.

**Environment variables**

```bash
export OPENAI_BASE_URL="http://localhost:8000/v1"
export OPENAI_API_KEY="dummy"                 # non-empty; value ignored by vLLM
export MODEL_NAME="google/gemma-3-4b-it"      # must match the served id EXACTLY
export MODEL_REVISION="093f9f38..."           # model commit; required for a citable run
export FREEBASE_SPARQL_ENDPOINT="http://127.0.0.1:8890/sparql"
export OPENAI_TIMEOUT=300                     # 30s default causes double-billed retries
export FLASHINFER_DISABLE_VERSION_CHECK=1     # REQUIRED to start vLLM here; see below
```

**`FLASHINFER_DISABLE_VERSION_CHECK=1` is required in the validated vLLM
environment.** Without it the engine aborts at startup:

```
RuntimeError: flashinfer-cubin version (0.6.13) does not match
              flashinfer version (0.5.2).
```

`vllm==0.11.1` pins `flashinfer 0.5.2`, while `flashinfer-cubin` resolves to a
newer release, and FlashInfer refuses the pair. The kernels themselves work; only
the version assertion fails. Export it in the shell that starts vLLM **and** in
the shell that runs experiments. The cleaner long-term fix is to pin
`flashinfer-cubin==0.5.2` to match, which would remove the need for the flag.

`.env.example` lists the same variables. Optional: `OPENAI_MAX_RETRIES`,
`OPENAI_RETRY_INTERVAL`, `SPARQL_TIMEOUT`, `WIKIDATA_SPARQL_ENDPOINT`,
`POG_SENTENCE_MODEL`, and the provenance overrides `ENERGY_RUN_ID`,
`ENERGY_GIT_COMMIT`, `ENERGY_HARDWARE_ID`. `ENERGY_EVENTS_FILE` turns event
logging on; `measure_run.py` sets it for you, and without it a run produces
answers but no events.

**Health checks** — the one command that covers all of them:

```bash
scripts/run_experiments.sh --check
```

It verifies the endpoint serves `MODEL_NAME`, **completes a real test
generation**, and that Freebase answers a query; it exits non-zero otherwise and
starts nothing. Takes about 3 seconds.

By hand:

```bash
# LLM: served model id and context length
curl -s "$OPENAI_BASE_URL/models" | grep -o '"id":"[^"]*"'
curl -s "$OPENAI_BASE_URL/models" | grep -o '"max_model_len":[0-9]*'

# LLM: it can actually GENERATE -- a listed model is not enough
curl -s --max-time 60 "$OPENAI_BASE_URL/chat/completions" \
  -H 'Content-Type: application/json' \
  -d "{\"model\":\"$MODEL_NAME\",\"messages\":[{\"role\":\"user\",\"content\":\"Say OK\"}],\"max_tokens\":5}" \
  | grep -o '"content":"[^"]*"'

# Freebase: answers a trivial query
curl -s "$FREEBASE_SPARQL_ENDPOINT" \
  --data-urlencode 'query=SELECT ?s WHERE { ?s ?p ?o } LIMIT 1' \
  --data-urlencode 'format=application/json' | grep -o '"bindings"'
```

**Listing the model is not a health check.** A vLLM server here kept answering
`/v1/models` while every completion hung: a 5-token request timed out after
120 s with the GPU pinned at 100%. A run against it burned an hour on 300-second
timeouts and retries. Only a completed generation proves the server is serving.

If the LLM endpoint is down or wedged, a run still "succeeds" with exit 0: every
LLM call is recorded `status: "error"`, the agent falls through to
`llm:direct_answer`, and predictions come out empty. **Always preflight.**

**On the Freebase triple count.** Older notes asserted the store must report
exactly `3124793702`. It currently reports `3124793701`, and that one-triple
difference is *not* in the data: `<http://freebase.com>` holds
**3,124,791,155** triples, and the remainder of the default-graph total is
Virtuoso's own system graphs (`virtrdf#` 2,527, `localuriqaserver/sparql` 14,
`ldp#` 3, `activitystreams-owl:map` 2), whose contents vary with the image
version. The deployment uses the unpinned `openlink/virtuoso-opensource-7:latest`
tag, so the total is not a stable invariant and is not asserted by default.
To assert the data graph instead:

```bash
EXPECTED_TRIPLES=3124791155 scripts/run_experiments.sh --check
```

---

## Running experiments

### One-question smoke run

```bash
python measurement/measure_run.py --tag smoke_cor -- \
  --method cor --dataset webqsp --kb freebase --run_size 1 \
  --relation_width 3 --entity_width 3 --depth 3 \
  --save_detail true --output_dir /tmp/smoke_cor
```

`--output_dir` keeps a smoke run's predictions out of `results/`; without it
every run of the same method/dataset/model shares one `predict.jsonl` and a
throwaway run appends to a finished experiment.

Or drive all three paradigms with the portable script, which preflights both
services and stops if either is unavailable:

```bash
scripts/run_experiments.sh                        # 1 question, cor tog pog
METHODS="cor" RUN_SIZE=25 scripts/run_experiments.sh
DATASET=cwq DEPTH=4 RUN_SIZE=-1 scripts/run_experiments.sh
```

### Per paradigm

```bash
# CoR
python measurement/measure_run.py --tag cor_webqsp_$(date +%m%d_%H%M) -- \
  --method cor --dataset webqsp --kb freebase --run_size 25 \
  --relation_width 3 --entity_width 3 --depth 3 \
  --temperature_exploration 0.01 --temperature_reasoning 0.01 --save_detail true

# ToG  — same, with --method tog
# PoG  — same, with --method pog
```

Without measurement (answers only, any host):

```bash
python -m chain_of_relations.run --method cor --dataset webqsp --kb freebase --run_size 25
```

### Common options

| Want to change | Flag / variable |
|---|---|
| dataset | `--dataset webqsp \| cwq \| qald10_en` |
| number of questions | `--run_size N` (`-1` = the whole dataset) |
| one specific question | `--question_id WebQTest-0` |
| model | `export MODEL_NAME=...` (must match the served id) |
| prediction output directory | `--output_dir DIR` |
| measurement run directory | `--tag TAG` → `measurement/runs/TAG` |
| search shape | `--depth`, `--relation_width`, `--entity_width` |
| sampling rate | `measure_run.py --hz 10` |

### Full runs

```bash
python measurement/measure_run.py --tag cor_webqsp_full -- \
  --method cor --dataset webqsp --kb freebase --run_size -1 \
  --relation_width 3 --entity_width 3 --depth 3 \
  --temperature_exploration 0.01 --temperature_reasoning 0.01 --save_detail true
```

> **A full WebQSP run is 1,639 questions.** On an RTX 4090 with Gemma-3-4B that
> was 2h48m (CoR), 5h00m (ToG) and 1h55m (PoG). CWQ is 3,531 questions. Runs
> resume by question id, so an interrupted run continues where it stopped —
> which also means a new run with the same method/dataset/model and no
> `--output_dir` will skip everything already answered.

`measure_run.py` refuses to reuse a tag that already holds a measured run:
resuming would append events to a timeline the power log no longer covers.

---

## Output structure

**Measurement artifacts** — `measurement/runs/<tag>/`:

| File | Layer | Contents |
|---|---|---|
| `events.jsonl` | semantic events | one line per operation: schema v1, operation label, question/iteration/depth/step, tokens, status, provenance |
| `power.csv` | hardware timeline | out-of-band NVML + RAPL samples (`t`, `gpu0_w`, `gpu0_energy_mj`, RAPL columns where available) |
| `events_attributed.jsonl` | attribution | the join of the two — the canonical per-event research artifact |
| `trajectory_summary.csv` / `.json` | trajectory accounting | per-question reconciliation, residual, coverage |
| `energy_summary.csv` | aggregation | rollups by operation type, label and question |
| `run.log` | provenance | full stdout/stderr of the wrapped run |
| `codecarbon/` | optional | machine-mode kWh/gCO2e, when CodeCarbon is installed |

**Answer artifacts** — `results/<method>/<dataset>/<model>/` (or `--output_dir`):

| File | Contents |
|---|---|
| `predict.jsonl` | one line per question: `id`, `question`, `results`, `gold_answer`, `action` |
| `param.json` | resolved run parameters, written once per directory |
| `<question_id>.json` | per-question detail with `step_history` (only with `--save_detail true`) |

Both trees are git-ignored: a fresh clone has no runs and no predictions.

---

## Evaluation

WebQSP is scored with the **canonical rule from the official evaluator**
(`chain_of_relations/eval/webqsp_canonical.py`):

- **Full 1,639-question denominator.**
- **F1** = precision/recall/F1 against *every* official semantic parse, keeping
  the **maximum F1** for the question. Ties keep the earliest parse.
- **Hit@1 / gold answer found** = the prediction matches an answer of **any**
  parse — identical to the union rule ToG and PoG use, so the number is directly
  comparable to their Exact Match.
- **The 11 official empty-gold questions stay in the evaluation** and score 0,
  exactly as ToG and PoG count them. They are empty in the official Microsoft
  release, not lost here.
- The **1,628 non-empty-gold subset** may be reported as well, but **must be
  labelled as such**. The evaluator prints it on its own line; `empty_gold` in
  the outcome CSV selects it.

Gold comes from `datasets/webqsp/webqsp_official_gold.json`, distilled from the
official release by `scripts/build_webqsp_official_gold.py`. The upstream
`datasets/webqsp/webqsp.json` is unchanged and carries only `Parses[0]`.

Evaluation is separate from generation — run it any time over a predictions
directory:

```bash
python -m chain_of_relations.eval.eval --dataset webqsp \
  --output_dir results/cor/webqsp/gemma-3-4b-it
```

Current WebQSP / Gemma-3-4B results, full 1,639 denominator:

| System | F1 | Gold answer found (Hit@1) |
|---|---|---|
| CoR | 43.54% | 62.48% |
| ToG | 42.10% | 57.23% |
| PoG | 45.79% | 57.90% |

---

## Creating CSVs and figures

**Per-question outcomes** (the annotation file that colours figures):

```bash
python measurement/export_outcomes.py \
  --predictions results/cor/webqsp/gemma-3-4b-it/predict.jsonl \
  --dataset webqsp --run CoR --out outcomes_cor.csv
```

**Individual figures** — any run directory with an `events_attributed.jsonl`:

```bash
python measurement/visualize.py --list --run measurement/runs/<run>
python measurement/visualize.py --plot operation-energy --run measurement/runs/<run> --out <dir>
python measurement/visualize.py --plot question-energy \
  --run CoR=<cor-run> --run ToG=<tog-run> --run PoG=<pog-run> --out <dir>
python measurement/visualize.py --plot energy-vs --x output_tokens \
  --run CoR=<run> --annotations outcomes_cor.csv --color-by outcome --out <dir>
```

Each figure is written as PNG, PDF and a CSV of exactly the values drawn.
`--out` must lie outside every run directory. `--domain` selects the energy
domain (default `gpu_energy_j`); an unmeasured domain is refused, not drawn as
zero. Full options: [docs/RUNBOOK.md](docs/RUNBOOK.md#figures).

**The full comparable package** — Figures 1–5 for all three systems on shared
axes, plus every plotted-data CSV and a checks log:

```bash
python measurement/make_comparable_figures.py --out "$HOME/webqsp_supervisor_summary"
```

| Family | Shows |
|---|---|
| `fig1_{sys}_operation_energy` | energy by semantic operation label |
| `fig2_{sys}_energy_ecdf` | per-question energy distribution (cumulative share) |
| `fig3_{sys}_trajectory_properties` | workload vs energy (output/input tokens, LLM calls, max depth), coloured by answer outcome |
| `fig4_{sys}_outcome_and_fallback` | energy by outcome + the fallback/non-fallback split |
| `fig5_{sys}_semantic_flow` | total → operation type → operation label energy flow |
| `outcomes_{sys}.csv` | per-question outcomes from the dataset's own evaluator |

**Summary tables** — the supervisor-level statistics, from the run directories
plus those outcome CSVs. Dataset-agnostic; `--dataset` is only a label:

```bash
python measurement/summarize_comparable_runs.py \
  --dataset cwq --out "$HOME/cwq_summary_tables" \
  --run PoG=$POG_RUN --run ToG=$TOG_RUN --run CoR=$COR_RUN \
  --outcomes PoG=outcomes_pog_cwq.csv \
  --outcomes ToG=outcomes_tog_cwq.csv \
  --outcomes CoR=outcomes_cor_cwq.csv
```

Writes `summary_overview.csv`, `summary_fallback.csv`,
`summary_depth_outcome.csv`, `summary_fallback_reasons.csv` and `SUMMARY.md`:
gold-found counts and rate, mean/median/p90/total GPU energy, tokens and LLM
calls per question, top-10% energy share, energy by outcome, depth vs outcome,
fallback rates and success, the before/in-fallback energy split, and the
fallback-reason breakdown. Effectiveness is read from the outcome CSVs (the
dataset's own evaluator), never recomputed; `--effectiveness CoR=43.54,62.48`
pastes the evaluator's own aggregates instead.

Custom runs, predictions and output directory:

```bash
python measurement/make_comparable_figures.py \
  --out "$HOME/analysis_out" \
  --run CoR=measurement/runs/cor_webqsp_gemma4b_0905_0856 \
  --run ToG=measurement/runs/tog_webqsp_gemma4b_0905_0856 \
  --run PoG=measurement/runs/pog_webqsp_gemma4b_0905_0856 \
  --predictions CoR=results/cor/webqsp/gemma-3-4b-it/predict.jsonl \
  --predictions ToG=results/tog/webqsp/gemma-3-4b-it/predict.jsonl \
  --predictions PoG=results/pog/webqsp/gemma-3-4b-it/predict.jsonl
```

The script writes only into `--out`, verifies that no run artifact changed, and
prints a `PASS run artifacts unchanged` line.

---

## Reproducing the current WebQSP analysis

With the three WebQSP Gemma-3-4B runs present in `measurement/runs/` (naming
convention `<method>_<dataset>_<model>_<MMDD>_<HHMM>`):

```bash
cd "$(git rev-parse --show-toplevel)"
source .venv/bin/activate

for m in cor tog pog; do
  python -m chain_of_relations.eval.eval --dataset webqsp \
    --output_dir "results/$m/webqsp/gemma-3-4b-it"
done

python measurement/make_comparable_figures.py --out "$HOME/webqsp_supervisor_summary"
python measurement/audit_runs.py measurement/runs

python measurement/summarize_comparable_runs.py \
  --dataset webqsp --out "$HOME/webqsp_summary_tables" \
  --run PoG=measurement/runs/pog_webqsp_gemma4b_0905_0856 \
  --run ToG=measurement/runs/tog_webqsp_gemma4b_0905_0856 \
  --run CoR=measurement/runs/cor_webqsp_gemma4b_0905_0856 \
  --outcomes PoG="$HOME/webqsp_supervisor_summary/outcomes_pog.csv" \
  --outcomes ToG="$HOME/webqsp_supervisor_summary/outcomes_tog.csv" \
  --outcomes CoR="$HOME/webqsp_supervisor_summary/outcomes_cor.csv"
```

Expect `CoR 43.54 / 62.48`, `ToG 42.10 / 57.23`, `PoG 45.79 / 57.90`
(F1 / Hit@1), 60 files in the figure directory, and every check passing.

---

## Measurement caveats

- **Boundary.** `measured_energy_j = gpu + cpu_package + dram`. CPU `core` is
  diagnostic only and never added; `psys` and `uncore` are excluded.
- **Incomplete measurements.** A run is complete only when every boundary domain
  was readable. Attributed events carry `available_energy_domains` and
  `measurement_complete`; an incomplete run is not a citable measurement.
- **Development hosts.** The WSL2 development machine exposes no RAPL, so its
  runs are GPU-only and `measured_energy_j` is `null`. The existing WebQSP runs
  in this repository are of that kind — real-model validation, not study results.
- **Missing is not zero.** An unreadable domain stays `null` and any total
  containing it is `null` too. Nothing is estimated to fill a gap, and the
  visualizer refuses an unmeasured domain rather than drawing zeros.
- **Residual is visible.** `unattributed_energy_j = trajectory − Σ attributed`;
  coverage is never forced to 1.0.
- **Counter resolution.** The RTX 4090 NVML energy counter updates every ~106 ms;
  per-event GPU energy for much shorter operations is biased low.
- **Energy vs carbon.** Joules are the measured product. CO2e is derived
  separately (`agent_energy_profiler/carbon.py`) and needs a caller-supplied grid
  intensity with provenance; nothing here estimates one.

Full protocol: [MEASUREMENT_SPEC.md](MEASUREMENT_SPEC.md).

---

## Testing

```bash
source .venv/bin/activate

pytest tests/ -q                                   # full suite (343 tests)
pytest tests/test_energy_measurement.py -q         # profiler: events, schema, attribution
pytest tests/test_package_boundary.py -q           # one-way dependency rule
pytest tests/test_webqsp_canonical_eval.py -q      # canonical WebQSP evaluation
pytest tests/test_tog_pog_instrumentation.py -q    # ToG/PoG semantic labels
pytest tests/test_visualize.py -q                  # figures match their CSVs
```

Use `pytest`, not `python -m unittest discover`: the unittest runner silently
skips the pytest-style test modules.

End-to-end validation, in increasing cost:

```bash
python measurement/validate_hardware.py                  # host capability probe
python measurement/validate_measurement.py --mode idle --seconds 60
python measurement/audit_runs.py measurement/runs        # internal consistency of runs
python audit_steps.py results                            # step histories in detail JSONs
scripts/run_experiments.sh                               # 1-question smoke of all three
```

---

## Further documentation

| Document | Contents |
|---|---|
| [docs/SETUP.md](docs/SETUP.md) | Freebase/Virtuoso, vLLM and harness installation from scratch |
| [docs/RUNBOOK.md](docs/RUNBOOK.md) | running experiments, verification, eval, figures, validation protocol |
| [MEASUREMENT_SPEC.md](MEASUREMENT_SPEC.md) | authoritative protocol: boundary, schema v1, taxonomy, missing-domain policy, trajectory accounting, host capability gate, analysis contract |
| [docs/ROADMAP.md](docs/ROADMAP.md) | status, development-host findings, phase plan |
| [CLAUDE.md](CLAUDE.md) | project contract and schema-v1 field definitions |
| [EMISSIONS.md](EMISSIONS.md) | **superseded**; pre-schema-v1, retained for history |

---

## Using the profiler on its own

`agent_energy_profiler` is a standalone package. A host supplies thin semantic
span annotations; the profiler records and attributes hardware energy:

```python
from agent_energy_profiler import profiler

with profiler.span("llm:reason") as s:
    ...
    s["output_tokens"] = n
```

Console entry points: `aep-validate` (host capability probe), `aep-sample`
(power timeline), `aep-attribute` (join events to power), `aep-aggregate`
(rollups), `aep-visualize` (figures). The `measurement/*.py` paths documented
here are thin compatibility shims over these modules.

---

## Upstream Chain-of-Relations

This repository is a **fork**. Upstream CoR is a knowledge-graph reasoning
method; everything in the measurement layer is added here.

*Chain-of-Relations: Faithful and Efficient LLM Reasoning over Knowledge Graphs
via Relation-Centric Exploration* — accepted to Findings of ACL 2026.

![Chain-of-Relations Method](imgs/cor_method.png)

Unmodified upstream CoR runs via [scripts/run_cor.sh](scripts/run_cor.sh).
Knowledge-base deployment follows the ToG guides for
[Freebase](https://github.com/DataArcTech/ToG/tree/main/Freebase) and
[Wikidata](https://github.com/DataArcTech/ToG/tree/main/Wikidata); Wikidata can
alternatively use the official
[Query Service](https://www.wikidata.org/wiki/Wikidata:SPARQL_query_service).

```bibtex
@inproceedings{liu2026cor,
  title={Chain-of-Relations: Faithful and Efficient LLM Reasoning over Knowledge Graphs via Relation-Centric Exploration},
  author={Liu, Chenhui and Zhou, Jianpeng and Wang, Jiahai},
  booktitle={Findings of the Association for Computational Linguistics: ACL 2026},
  year={2026}
}
```
