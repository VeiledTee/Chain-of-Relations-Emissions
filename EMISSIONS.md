# Emissions Measurement — How It Works

For future-me, six months from now. Two layers, joined by wall-clock time.

## The one-paragraph version

Every unit of work the system performs (an LLM call, a SPARQL query, an
embedding encode) writes one line to `events.jsonl` with its start/end time,
a **category**, and a **label**. A separate host-side process samples GPU
power (NVML) and CPU/DRAM energy (RAPL) at 10 Hz into `power.csv`. After the
run, `measurement/attribute.py` integrates power over each event's time
window and aggregates: energy per event → per label → per category → per
question. Carbon (gCO₂e) comes from CodeCarbon running in machine mode over
the whole run.

## The taxonomy (grouping contract)

| category    | meaning                              | labels                                   |
|-------------|--------------------------------------|------------------------------------------|
| `inference` | all LLM text generation              | `llm:generate`                            |
| `tool`      | everything else the system does      | `kg:relation_search`, `kg:entity_search`, `kg:id2name`, `kg:sparql` (unclassified), `embedding:prune` (PoG's SentenceTransformer) |

Group by `category` for the inference-vs-tool headline; group by `label` for
the per-tool breakdown. Both groupings come from the same events — the label
always determines the category.

SPARQL labels are classified from the query text in
`energy_events._classify_sparql()` (relation queries contain `?relation`,
name lookups contain `type.object.name`, entity queries contain
`?targetEntity`/`?x`). If you add new query templates, extend that classifier.

## Where the hooks live (Layer 1 — in-band, always cheap)

| file | what's bracketed | notes |
|------|------------------|-------|
| `chain_of_relations/llm_api.py` → `generate()` | the whole call **including retries** | one event per logical call; `meta.attempts` > 1 means retried — retried calls double-compute on the server, so any measured run should show `attempts == 1` everywhere |
| `chain_of_relations/kg_backend/freebase/db_func.py` → `execurte_sparql_with_meta()` and `id2entity_name_or_type()` | each SPARQL round-trip incl. retries | failure events carry `meta.failed=true` |
| `chain_of_relations/methods/pog/tools/entity_condition_prune.py` | the two `model.encode()` calls | PoG's hidden second model — without this its GPU draw would be smeared into neighboring steps |
| `chain_of_relations/run.py` | `energy_events.set_question(qid)` per question | tags every event with its question |

All hooks are no-ops unless `ENERGY_EVENTS_FILE` is set — normal runs are
completely unaffected. Hooks never raise (measurement must not break runs).

## Layer 2 — out-of-band, host-side

- `measurement/power_logger.py` — 10 Hz sampler: GPU watts per device (NVML),
  RAPL cumulative energy counters (`/sys/class/powercap/intel-rapl*`).
- `measurement/attribute.py` — offline join: trapezoidal GPU-power integral
  per event window; RAPL counter deltas (interpolated, wraparound-handled).
- `measurement/measure_run.py` — orchestrator: starts logger (+ CodeCarbon if
  installed), runs `chain_of_relations.run` with the event log enabled,
  stops everything, runs attribution.

## Setup

```bash
pip install pynvml codecarbon        # zeus-ml optional, not required
# RAPL read permission (bare-metal Linux; resets on reboot):
sudo chmod -R a+r /sys/class/powercap/intel-rapl
# verify:
python -c "import pynvml; pynvml.nvmlInit(); print('NVML OK')"
cat /sys/class/powercap/intel-rapl:0/energy_uj   # a number = RAPL OK
```

Platform truth table:
- **Bare-metal Linux**: GPU ✓ CPU ✓ DRAM ✓ — the only tier where all numbers count.
- **WSL2**: GPU ✓ (NVML works), RAPL ✗ — pipeline rehearsal only.
- **Shared HPC (Compute Canada)**: GPU usually ✓, RAPL usually ✗ unless
  exclusive whole-node allocation *and* powercap readable — verify before trusting.

## Running a measured experiment

```bash
# machine otherwise idle; vLLM + Virtuoso already up; single writer
python measurement/measure_run.py --tag cor_webqsp_qwen7b -- \
  --method cor --dataset webqsp --relation_width 3 --entity_width 3 \
  --depth 3 --temperature_exploration 0.01 --temperature_reasoning 0.01 \
  --save_detail true
# outputs: measurement/runs/cor_webqsp_qwen7b/{events.jsonl,power.csv,
#          energy_summary.csv,codecarbon/,run.log}
```

`energy_summary.csv` rows: `scope` ∈ {category, label, question} × columns
`n_events, duration_s, gpu_j, cpu_j, dram_j`.

## What the numbers mean (read before quoting anything)

1. **Whole-machine attribution.** vLLM and Virtuoso are separate processes;
   power counters see the whole box. Valid only if the machine runs nothing
   else. Never run two experiments at once.
2. **Per-question energy is the robust unit** (long windows, many samples).
   **Per-event energy is a timestamp-attributed estimate** — sub-100ms SPARQL
   events fall inside one 10 Hz sample, so their figures are
   order-of-magnitude. Inference events (seconds) are well-resolved.
3. **Gaps between events** (harness bookkeeping, idle) are *not* attributed
   to any event: sum(events) < whole-run energy. The difference is
   idle/overhead — report it, don't hide it. CodeCarbon's whole-run number
   is the envelope; the event sum is the attributed portion.
4. **Retries**: `meta.attempts > 1` on any inference event invalidates the
   token↔energy correspondence for that call (server computed more than the
   final usage reports). Measured runs require `OPENAI_TIMEOUT=300` and zero
   retries — grep events for `"attempts": 2` before trusting a run.
5. **Idle draw**: characterize once per machine (run the power logger 5 min
   with everything up but no questions) — that baseline is what "the system
   existing" costs, versus the marginal cost of questions.
