# Emissions Measurement

---

This extention of CoR measures the energy of an agentic KGQA system (CoR / ToG / PoG over Freebase) decomposed into **inference** (LLM generation) vs **tool** (KG retrieval + embedding) work, per step and per question. It combines three instruments: (1) an **in-band event log** that timestamps every LLM call, SPARQL query, and embedding encode and reads the GPU's cumulative energy counter at each event's boundaries; (2) an **out-of-band whole-machine power sampler** (NVML GPU power + RAPL CPU/DRAM energy at 10 Hz); (3) **CodeCarbon** for the whole-run energy→carbon (gCO₂e) conversion and an independent cross-check. Per-step energy is attributed offline by joining the event log to the hardware counters on a shared wall clock. All measurement is **whole-machine** and assumes an otherwise-idle, single-user host running one request at a time.

## Measurement taxonomy

Every recorded event carries a `category` and a `label`:

| category    | meaning                                | labels                                                                 |
|-------------|----------------------------------------|------------------------------------------------------------------------|
| `inference` | LLM text generation                    | `llm:generate`                                                          |
| `tool`      | everything else the system does        | `kg:relation_search`, `kg:entity_search`, `kg:id2name`, `kg:sparql` (unclassified), `embedding:prune` (PoG's SentenceTransformer) |

Group by `category` for the headline inference-vs-tool split; group by `label` for the per-tool breakdown. The label always determines the category. SPARQL labels are inferred from query text in `energy_events._classify_sparql()`; extend that function if you add query templates.

## Measurement Instruments

### Instrument 1 — in-band event log (`chain_of_relations/energy_events.py`)
The only component that knows about individual steps. Enabled by the `ENERGY_EVENTS_FILE` env var (unset = complete no-op; normal runs unaffected).

- `mark()` captures a tuple `(wall_time, cumulative_gpu_energy_mj)` at an instant. The GPU energy comes from **NVML's `nvmlDeviceGetTotalEnergyConsumption`** — a monotonic hardware energy counter (Volta+), read directly via `pynvml`. This is **read-only / observer-only**: no power capping, no DVFS, no intervention (it is the same counter Zeus's `ZeusMonitor` reads, without Zeus's optimizer components).
- Each instrumented function calls `mark()` at start and end and passes both to `record()`, which writes one JSON line to `events.jsonl` with `t_start`, `t_end`, duration_s`, `category`, `label`, per-call `meta` (token counts, SPARQL row counts, retry `attempts`), and **`gpu_energy_j`** = counter delta (mJ→J) when the counter is available, else `null`.
- Why a boundary-read counter and not just the 10 Hz power sampler: sub-sample events (a 2 ms SPARQL round-trip) fall *between* 10 Hz samples, so integrating the power curve over them is only order-of-magnitude. Reading a monotonic energy counter **at the exact event boundaries** captures every Joule regardless of event brevity. This is the accuracy fix for the tool side, where many events are milliseconds long.

Hook locations (all no-ops unless `ENERGY_EVENTS_FILE` is set; none can raise):

| file | function | brackets | notes |
|------|----------|----------|-------|
| `chain_of_relations/llm_api.py` | `generate()` | the whole LLM call **including retries** | `meta.attempts` > 1 flags a retried call — retries recompute on the server while reporting one usage object, so measured runs must show `attempts == 1` everywhere |
| `chain_of_relations/kg_backend/freebase/db_func.py` | `execurte_sparql_with_meta()`, `id2entity_name_or_type()` | each SPARQL round-trip incl. retries | failures carry `meta.failed`; entity-name lookups (`kg:id2name`) dominate call count |
| `chain_of_relations/methods/pog/tools/entity_condition_prune.py` | the two `model.encode()` calls | PoG's SentenceTransformer | a second neural model whose GPU draw would otherwise be smeared into neighboring steps |
| `chain_of_relations/run.py` | main loop | `energy_events.set_question(qid)` | tags every subsequent event with its question |

### Instrument 2 — out-of-band power sampler (`measurement/power_logger.py`)
A separate host process sampling the **whole machine** at 10 Hz into `power.csv`:
- **GPU power (W)** per device via **NVML** (`pynvml.nvmlDeviceGetPowerUsage`).
- **CPU + DRAM energy (µJ, cumulative)** by reading **RAPL** sysfs directly (`/sys/class/powercap/intel-rapl:*/energy_uj`) — no library, raw file reads. Provides: the continuous power curve (for events without a counter reading, and for whole-run context) and the **only** source of CPU/DRAM energy. RAPL is absent on WSL2 and often on shared HPC nodes → those columns are blank there.

### Instrument 3 — CodeCarbon (`measurement/measure_run.py`)
Wraps the *entire* run (`tracker.start()` before the harness, `tracker.stop()` after). Samples every 5 s and emits **one whole-run figure**: kWh and gCO₂e, using NVML for GPU, RAPL-or-TDP-estimate for CPU, a heuristic for RAM, and a grid carbon-intensity factor (set via `country_iso_code`, default `CAN`). It is **not** in the per-step path. Its role is (a) the energy→carbon conversion the raw counters don't provide, and (b) an independent whole-run cross-check: the sum of attributed event energy plus unattributed overhead should reconcile with CodeCarbon's total. CodeCarbon knows nothing about steps or the inference/tool split.

### Attribution (`measurement/attribute.py`) — offline join
After the run, joins `events.jsonl` to `power.csv` on wall-clock time:
- **GPU energy per event**: use the in-band counter delta (`gpu_energy_j`) when present (accurate for short events); otherwise fall back to trapezoidal integration of the 10 Hz power curve over the event window. The tool prints how many events used each path (`N from counter, M integrated`).
- **CPU / DRAM energy per event**: difference the cumulative RAPL counters over the event window (interpolated, wraparound-handled). Aggregates to `energy_summary.csv` at three scopes — per `category`, per `label`, per `question` — each with `n_events, duration_s, gpu_j, cpu_j, dram_j`.

## Data-flow diagram
```
harness process                          host processes
---------------                          -------------
llm_api.generate ─┐                      power_logger.py ── 10Hz ──> power.csv
db_func SPARQL   ─┼─ mark()/record() ─>  events.jsonl        (NVML W + RAPL µJ)
pog encode       ─┘   (NVML energy                │
run.py set_question    counter @ bounds)          │
                                                  ▼
CodeCarbon (wraps whole run) ─> codecarbon/emissions.csv (kWh, gCO₂e)
                                                  │
                        attribute.py: join events × power ─> energy_summary.csv
                          GPU: counter delta, else integrate; CPU/DRAM: RAPL delta
```

## Setup

```bash
pip install pynvml codecarbon        # Zeus not required; NVML read directly
# bare-metal Linux only, for CPU/DRAM (resets on reboot):
sudo chmod -R a+r /sys/class/powercap/intel-rapl
python -c "import pynvml; pynvml.nvmlInit(); print('NVML OK')"
cat /sys/class/powercap/intel-rapl:0/energy_uj   # a number = RAPL OK
```

Platform truth table:

| platform | GPU power | GPU energy counter | CPU/DRAM (RAPL) | verdict |
|----------|-----------|--------------------|-----------------|---------|
| bare-metal Linux | ✓ | ✓ | ✓ | authoritative — the only measurement tier |
| WSL2 | ✓ | maybe (NVML may gate the counter) | ✗ | pipeline rehearsal only |
| shared HPC (e.g. Compute Canada) | usually ✓ | usually ✓ | usually ✗ (unless exclusive whole-node + powercap readable) | verify before trusting |

## Running a measured experiment

```bash
# machine otherwise idle; vLLM + Virtuoso already up; single writer; max-num-seqs 1
python measurement/measure_run.py --tag cor_webqsp_qwen7b -- \
  --method cor --dataset webqsp --relation_width 3 --entity_width 3 \
  --depth 3 --temperature_exploration 0.01 --temperature_reasoning 0.01 \
  --save_detail true
# -> measurement/runs/cor_webqsp_qwen7b/{events.jsonl, power.csv,
#     energy_summary.csv, codecarbon/emissions.csv, run.log}
```

## What the numbers mean — read before quoting

1. **Whole-machine attribution.** vLLM and Virtuoso are separate processes; every instrument reads the whole box, not a process. Valid only on an idle, single-user machine running one request at a time (`--max-num-seqs 1`, single writer). Never run two experiments concurrently.
2. **Per-question energy is the robust primitive** (long windows, many samples, counter deltas over seconds). **Per-event GPU energy is accurate when the NVML energy counter is available** (boundary reads), and an order-of-magnitude estimate only when it falls back to power integration — the `attribute.py` "counter vs integrated" line tells you which regime a run is in.
3. **GPU-only vs full.** Where RAPL is absent (WSL2), the inference/tool split is **GPU-only** and understates tool cost — SPARQL and embedding work is CPU/RAM-bound and largely invisible to GPU counters. The true split is less lopsided than a GPU-only run suggests; bare-metal RAPL is what closes the gap. Report GPU-only numbers as a lower bound on tool cost.
4. **Attributed < total.** Sum(event energy) < CodeCarbon whole-run total; the difference is unattributed time (Python/harness overhead, idle between steps). Report it as overhead — CodeCarbon's number is the envelope, the event sum is the attributed portion.
5. **Retries invalidate a call's energy↔token correspondence.** Any event with `meta.attempts > 1` means the server computed more than the final usage reports. Measured runs require `OPENAI_TIMEOUT=300` and should show zero retries: `grep '"attempts": [2-9]' events.jsonl` must return nothing.
6. **Measurement overhead is bounded and CPU-localized.** The 10 Hz sampler and the per-event NVML/clock reads add microsecond-scale CPU work against a
   GPU-bound workload. Characterize it with an **idle baseline** (run the sampler ~5 min with vLLM + Virtuoso up but no questions) and report marginal
   (question-driven) energy above that baseline. Optionally quantify the observer effect by running the same sample with and without
   `ENERGY_EVENTS_FILE` and comparing CodeCarbon totals.
7. **Carbon intensity** is a grid factor, not a measurement — set `country_iso_code` (or a finer region) to your actual grid; New Brunswick's intensity differs substantially from, e.g., hydro-dominant grids.
