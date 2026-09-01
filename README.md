# Chain-of-Relations Emissions

An **energy-instrumented fork of [Chain-of-Relations (CoR)](#upstream-chain-of-relations)**,
used as a unified experimental harness for **hardware-grounded, operation-level
energy measurement of agentic KGQA**.

The question is where energy is actually spent when inference is embedded in a
multi-step agentic loop over a knowledge graph — energy attributed to *what the
agent was doing* (relation ranking, reasoning, answer filtering, graph
retrieval), not merely that a model was called. **CoR, ToG and PoG** are the
paradigms being compared, over one shared graph, model-serving stack and
measurement procedure.

> **Status:** the measurement instrument is built and validated in software and
> against real small-model inference. **No final experimental results have
> been produced.** See [docs/ROADMAP.md](docs/ROADMAP.md).

---

## `agent_energy_profiler`

The measurement instrument is a **standalone, reusable package**, deliberately
separate from the KGQA implementations. Host systems supply thin **semantic
span annotations**; the profiler records and attributes hardware energy.

```python
from agent_energy_profiler import profiler

with profiler.span("llm:reason") as s:
    ...
    s["output_tokens"] = n
```

The dependency direction is one-way and test-enforced
(`tests/test_package_boundary.py`): a host may import the profiler, never the
reverse. Nothing in the package imports a benchmark, dataset or graph, and no
label is ever inferred from prompt text, function names or the call stack.

**Measurement boundary** — GPU, CPU-package and DRAM, where hardware counters
are available:

```
measured_energy_j = gpu_energy_j + cpu_package_energy_j + dram_energy_j
```

- CPU **`core`** is **diagnostic only** and never added — it is contained
  within `package`, so summing both double-counts. `psys` is excluded for the
  same reason; `uncore` is excluded from both buckets.
- A domain the hardware did not report stays **`null`**, never `0`. Any total
  with a missing term is `null` too, and the measurement is **incomplete**.
- Energy in Joules is the measured product. **CO2e is derived separately**
  (`agent_energy_profiler/carbon.py`) and requires a caller-supplied grid
  intensity with provenance; nothing here estimates one.

---

## Architecture at a glance

```text
CoR ──┐
ToG ──┼──agent_energy_profiler
PoG ──┘
          ↓
    semantic events
          +
    hardware counters
          ↓
      attribution
          ↓
   trajectory accounting
          ↓
      aggregation
```

---

## Quick start

```bash
pip install -r requirements.txt
pip install -e ".[gpu]"        # pynvml; without it GPU energy is unavailable,
                               # never estimated
```

Freebase (Virtuoso) and vLLM must be running — see
**[docs/SETUP.md](docs/SETUP.md)**.

```bash
# 1. is this host eligible to produce citable measurements?
python measurement/validate_hardware.py --json measurement/hardware_report.json

# 2. smallest measured run
python measurement/measure_run.py --tag smoke -- \
    --method cor --dataset webqsp --kb freebase \
    --run_size 3 --depth 3 --relation_width 3 --save_detail true

# 3. aggregate the result
aep-aggregate --events measurement/runs/smoke/events_attributed.jsonl \
              --trajectories measurement/runs/smoke/trajectory_summary.json \
              --group-by operation_label --out rollup.csv
```

Instrument characterisation workloads:
`validate_measurement.py --mode idle|cpu|gpu|repeated|cor-smoke`.
Tests: `python -m unittest discover -s tests -p "test_*.py"`.

---

## Profiler outputs

Written to `measurement/runs/<tag>/`:

| Artifact | Layer | Contents |
|---|---|---|
| `events.jsonl` | semantic events | one line per operation: schema v1, operation label, trajectory position, tokens, status, provenance |
| `power.csv` | hardware timeline | out-of-band NVML + RAPL counter samples |
| `events_attributed.jsonl` | attribution | the join of the two — the canonical per-event research artifact |
| `trajectory_summary.{csv,json}` | trajectory accounting | per-question reconciliation, residual and coverage |
| `energy_summary.csv` | aggregation | derived rollups by operation type and label |

The four layers are kept separate on purpose and are not collapsed into one
logger. Aggregation is derived: it never re-measures, and refuses to report a
share, residual or total it cannot defend from the layers below.

---

## Measurement validity

- **Complete vs incomplete.** A measurement is complete only when every
  boundary domain was readable. Attributed events carry
  `available_energy_domains` and `measurement_complete`; an incomplete run is
  not a citable measurement whatever else it produces.
- **Residual energy is visible, not absorbed.** Per question,
  `unattributed_energy_j = trajectory_energy_j − Σ attributed operation
  energy`, and `attribution_coverage` is **never forced to 1.0** — inter-event,
  orchestration and background energy stay explicit.
- **Overlap is checked**, not assumed away: overlapping spans within a
  trajectory invalidate the accounting.
- **Short operations may fall below counter resolution.** The development
  host's NVML energy counter updates at ≈100 ms; per-event GPU energy for
  operations substantially shorter than that is biased low rather than
  symmetrically noisy. CPU-package and DRAM temporal resolution has **not**
  been characterised, because those domains are unavailable on that host.
- **Development-host runs are validation-only.** The WSL2 development machine
  exposes no RAPL, so `measured_energy_j` is always `null` there.

Full protocol: **[MEASUREMENT_SPEC.md](MEASUREMENT_SPEC.md)**.

---

## Profiler validation

Before any citable measurement, and after any change to the profiler or a
paradigm's instrumentation, a fixed protocol validates the **instrument and its
integration**:

- a fixed **25-question WebQSP** subset and a fixed **25-question CWQ** subset;
- the **same questions** for CoR, ToG and PoG;
- the same profiler version and measurement configuration throughout;
- a standardized **warm-up** before every measured window;
- verification of schema, semantic labels, provenance, hardware domains,
  overlap, trajectory reconciliation, residuals, aggregation and measurement
  completeness;
- a **rerun of a small fixed question subset** to characterize run-to-run and
  trajectory variability.

Real `google/gemma-3-4b-it` runs through vLLM already exist in
`measurement/runs/gemma4b_*`. They are **real-model profiler and integration
validation on the incomplete development host** — CPU-package and DRAM
unavailable, `measured_energy_j` null — and are **explicitly not study findings**.

> **These runs validate the instrument, not the science.** The final bare-metal
> experiment matrix has not been run.

Details: [docs/RUNBOOK.md](docs/RUNBOOK.md#profiler-validation-protocol).

---

## Reproducing the experiments

| Paradigm | Instrumentation status |
|---|---|
| **CoR** | on frozen **schema v1** |
| **ToG** | implemented; **not instrumented** |
| **PoG** | implemented; **limited legacy-format instrumentation**, **not migrated to schema v1** |

Datasets are WebQSP and CWQ over a self-hosted Freebase (≈3.12B triples,
verified as `3124793702`). The planned model ladder is Gemma 3 at
1B / 4B / 12B / 27B; that matrix begins only once a bare-metal host passes the
capability probe with `measurement_complete_possible == true`.

This section is not the runbook. Build the environment with
[docs/SETUP.md](docs/SETUP.md), run and verify experiments with
[docs/RUNBOOK.md](docs/RUNBOOK.md), and read
[MEASUREMENT_SPEC.md](MEASUREMENT_SPEC.md) before quoting any number.

---

## Documentation

| Document | Contents |
|---|---|
| [MEASUREMENT_SPEC.md](MEASUREMENT_SPEC.md) | authoritative protocol: boundary, schema v1, taxonomy, event lifecycle, missing-domain policy, trajectory accounting, host capability requirement, limitations, analysis contract |
| [docs/SETUP.md](docs/SETUP.md) | Freebase/Virtuoso, vLLM and harness installation |
| [docs/RUNBOOK.md](docs/RUNBOOK.md) | running experiments, verification, eval, run-directory tiers, validation protocol |
| [docs/ROADMAP.md](docs/ROADMAP.md) | detailed status, development-host findings, phase plan |
| [CLAUDE.md](CLAUDE.md) | project contract and schema-v1 field definitions |
| [EMISSIONS.md](EMISSIONS.md) | **superseded**; pre-schema-v1, retained for history |

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
