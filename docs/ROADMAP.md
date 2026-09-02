# Project Status and Roadmap

Detailed status, development-host validation findings, current bottleneck and
the phase plan. The README carries the summary; this is the
long form.

Related: [../README.md](../README.md) ·
[../MEASUREMENT_SPEC.md](../MEASUREMENT_SPEC.md) ·
[SETUP.md](SETUP.md) · [RUNBOOK.md](RUNBOOK.md).

---

## Research goal

**Research question — When inference is embedded in a multi-step agentic loop
over a knowledge graph, where is energy actually spent?**

- **Operation-level decomposition** — How does per-step, hardware-grounded
  energy consumption decompose across reasoning, tool invocation, and graph
  retrieval calls, and how does the total energy of a trajectory scale with the
  number of iterations?
- **Cross-paradigm energy–effectiveness comparison** — Among competing agentic
  KGQA paradigms answering the same questions over the same graph, does any
  paradigm Pareto-dominate the rest on the F1–energy frontier?

Answering these requires energy attributed to *what the agent was actually
doing* — relation ranking, reasoning, answer filtering, graph retrieval — not
merely "an LLM was called". That is what the measurement layer in this fork
provides.

---

## Experimental scope

This codebase is the common experimental harness for three agentic KGQA
paradigms, all running against the same self-hosted Freebase instance
(≈ **3.12 billion triples**, verified as `3124793702`), one model-serving
stack, one measurement procedure and one hardware platform — so an observed
difference is a property of the paradigm, not of the setup.

| Paradigm | Instrumentation status |
|---|---|
| **Chain-of-Relations (CoR)** | on frozen schema v1 |
| **Think-on-Graph (ToG)** | implemented; **not instrumented** |
| **Plan-on-Graph (PoG)** | implemented; **limited legacy-format instrumentation** (a single `embedding:prune` event in `methods/pog/tools/entity_condition_prune.py`, using the pre-schema-v1 call form); **not migrated to schema v1** |

Datasets: **WebQSP** and **ComplexWebQuestions (CWQ)**.

The planned model ladder is **Gemma 3** at **1B / 4B / 12B / 27B**. The full
matrix has not been run; see [Phase 7](#phase-7--gemma-3-matrix-runs).

### Planned paradigm × data matrix

| Paradigm | WebQSP | CWQ |
|---|---|---|
| CoR | full | full |
| ToG | full | full |
| PoG | full | **`PoG_sub`** (stratified subset) |

`PoG_sub` is a stratified subset of CWQ that preserves the relevant
distribution by stratifying on **reasoning hop count** and **answer-set
cardinality**. It exists because PoG's rehearsal execution grows steeply enough
with traversal depth that running full deep CWQ across the complete model
ladder is infeasible. (This is a practical infeasibility claim — *not* a claim
that PoG runtime is formally exponential.)

**All cross-paradigm comparisons use common questions.** Where PoG is involved,
CoR and ToG are evaluated on the same `PoG_sub` question set so the comparison
is paired rather than across differing question distributions.

---

## Current Status

Three tiers of result exist in this repository and **must not be conflated**:

| Tier | What it is | Citable as a study finding? |
|---|---|---|
| **Rehearsal** | Qwen2.5-7B CoR runs; engineering shakedown | ❌ No |
| **Validation** | schema tests, NVML experiments, synthetic GPU load, stub-LLM smoke runs | ❌ No |
| **Final experiments** | Gemma 3 matrix on a validated bare-metal node | **Not yet run** |

### ✅ Completed — rehearsal and harness setup

Self-hosted Freebase is operational, and full CoR runs have been performed with
**Qwen2.5-7B-Instruct** over WebQSP and CWQ. These are **rehearsal /
engineering results**, not measurements — different model family, and a host
that cannot complete the measurement boundary.

Rehearsal exposed four operational behaviours that shaped the design:

1. **Context growth and truncation at depth.** Deep CWQ trajectories grow the
   prompt until it overflows; 6 questions exceeded native 32K at depth 4.
   `--max-model-len 32768` is required — 8192 caused silent agentic degradation.
2. **Graph-grounding vs closed-book fallback.** When DFS finds no answer, CoR
   falls back to closed-book generation: 12.4% of WebQSP and 30.0% of CWQ
   questions. Fallback trajectories are a different kind of work and are now
   labelled distinctly (`llm:direct_answer`, tagged `fallback=true`).
3. **Highly variable trajectory workload.** Rehearsal traces showed substantial
   question-to-question variation in trajectory length and operation count —
   the rehearsal tool-call count ranged over roughly **2000×** between the
   cheapest and most expensive WebQSP question — motivating distributional
   rather than mean-only analysis. The magnitude of the corresponding **energy**
   spread will be established on the validated bare-metal experimental
   platform.
4. **Very high `kg:id2name` call volume.** 112,302 of 142,473 WebQSP events
   (**78.8%**) were entity-name resolutions.

> **Call volume is not energy — and neither is settled here.** The number of
> times an operation runs and what it costs are different quantities, and
> reporting either alone is misleading. Establishing the actual relationship
> between them is the object of the operation-level decomposition, not
> something the rehearsal answers.
>
> **Validation-only observation.** The rehearsal trace attributed nearly all
> measured GPU energy to LLM inference, but **this split is not interpreted as
> a study finding**. Most KG operations were substantially shorter than
> the development host's NVML counter-update interval, while CPU-package and
> DRAM energy were unavailable entirely. The final operation-level energy
> decomposition will therefore be determined only on the validated bare-metal
> experimental platform.

### ✅ Completed — CoR schema-v1 instrumentation

CoR now emits a **frozen, operation-level semantic event schema**. Every unit of
work is one JSON line identifying *which stage of the algorithm* was running:

| LLM operations | KG operations |
|---|---|
| `llm:relation_rank` | `kg:id2name` |
| `llm:reason` | `kg:relation_search` |
| `llm:answer_filter` | `kg:entity_search` |
| `llm:direct_answer` | `kg:sparql` |

Labels are assigned by the caller that knows *why* the model is being invoked —
never inferred from prompt text.

Three trajectory-position fields are tracked, and they are **distinct concepts**:

| Field | Meaning |
|---|---|
| `iteration` | monotonically increasing CoR control-loop iteration; does not decrease |
| `traversal_depth` | DFS/search depth; **may increase or decrease** through backtracking, `null` outside the DFS |
| `step_index` | chronological measured-event index within the question |

Events also carry top-level **token counts**, explicit **status** (`ok` /
`error` / `timeout` / `cancelled`), **retry accounting** that separates
API-level retries (`meta.attempts`, collapsed into one event) from higher-level
tool attempts (`meta.tool_attempt`, separate events), **batch/entity-count
metadata** for name resolution, and full **run provenance**.

The authoritative schema definition lives in
**[MEASUREMENT_SPEC.md](../MEASUREMENT_SPEC.md)** §4–§5 and the project
contract in [CLAUDE.md](../CLAUDE.md); it is not duplicated here.

### ✅ Completed — energy accounting corrections

- The non-overlapping boundary `gpu + cpu_package + dram` is enforced.
- **RAPL `package` + `core` double-counting was found and fixed.** `core` is
  retained as a diagnostic and never summed. `psys` is excluded as a superset
  of `package`; `uncore` is excluded from both buckets.
- **Missing domains stay `null`.** A previous version reported `cpu_j=0.00`,
  `dram_j=0.00` for a host with no RAPL at all — a fabricated zero. Attributed
  events now carry `available_energy_domains` and `measurement_complete`.

### ✅ Completed — development-host validation framework

A reusable validation layer now exists:

| Tool | Purpose |
|---|---|
| [`measurement/validate_hardware.py`](../measurement/validate_hardware.py) | hardware capability probe; per-domain `SUPPORTED`/`UNREADABLE`/`UNAVAILABLE`/`UNKNOWN`; `--json` report |
| [`measurement/validate_measurement.py`](../measurement/validate_measurement.py) | validation workloads: `--mode idle\|cpu\|gpu\|repeated\|cor-smoke` |
| [`measurement/trajectory.py`](../measurement/trajectory.py) | trajectory accounting, overlap detection, attribution coverage |
| [`MEASUREMENT_SPEC.md`](../MEASUREMENT_SPEC.md) | the measurement protocol contract |

#### Development-host findings — **VALIDATION ONLY**

The development machine is **WSL2 with an RTX 4090**. These are instrument
characterisations, **not** experimental results:

- ✅ NVML cumulative GPU energy works and tracks real load (446.8 W under
  sustained load vs 29.5 W idle).
- ✅ Sustained GPU measurements are repeatable — **CV ≈ 1.7%** over 12 identical
  3 s workloads.
- ⚠️ The RTX 4090 NVML energy counter has a **median update interval of
  106 ms** (106.1–107.1 ms across six `validate_hardware.py` runs, ~80 counter
  updates observed per run). The median update **step** was 138–158 mJ across
  the same runs; the step is not a hardware constant, since it is the product
  of instantaneous power and the interval, and so tracks idle-power
  fluctuation. **Per-event GPU energy for operations substantially shorter than
  the interval is not reliably resolved**, and is biased low rather than
  symmetrically noisy.
  - Observed directly in `validate_measurement.py --mode cpu`: five fixed
    SPARQL queries of 4.0–11.5 ms returned `gpu = 2.388, 0, 0, 0, 0 J`. Four
    windows closed inside one counter interval and differenced to zero; one
    straddled an update and absorbed a whole step. The resulting `CV 2.24` on
    `gpu_energy_j` is quantization, not physical variance.
- ✅ Freebase is functional (3.12B triples).
- ❌ **CPU-package and DRAM counters are unavailable under WSL2** — no
  `/sys/class/powercap`, no MSR interface.
- ❌ Therefore **`measurement_complete` cannot become true on this host**, and
  `measured_energy_j` is always `null` here.

> **Scope of the 106 ms result.** It applies to the **GPU / NVML counter
> only**. The temporal resolution of the **CPU-package and DRAM** counters has
> **not** been measured, because those domains are unavailable on this host.
> RAPL is a separate mechanism with its own update characteristics. Since KG
> operations are primarily CPU and memory work, **whether short KG operations
> are resolvable in their primary domains remains an open question**, to be
> answered on the bare-metal node in Phase 3.


### ✅ Completed — real-model profiler validation (development host)

Real `google/gemma-3-4b-it` CoR runs through vLLM exist in
`measurement/runs/` (`gemma4b_val_*`, `gemma4b_v2_*`, `gemma4b_v3_*`,
`gemma4b_v4_*`), each with full run provenance — `model_revision`,
`git_commit`, `hardware_id` — and a complete artifact set
(`events.jsonl`, `power.csv`, `events_attributed.jsonl`,
`energy_summary.csv`, `trajectory_summary.{csv,json}`).

These are **profiler and integration validation runs, not study findings.** They
were executed on the WSL2 development host at `run_size 3`, where CPU-package
and DRAM counters are unavailable; `measurement_complete` is therefore false
and `measured_energy_j` is `null` throughout. What they establish is that the
instrument, the CoR adapter and real vLLM inference work together end to end
and produce schema-v1 events with token counts — not where energy is spent.

**The final bare-metal experiment matrix has not been run.**

---

## Current Bottleneck

**The immediate blocker is access to / provisioning of the dedicated bare-metal
experimental node.**

The final experimental host must **pass the capability probe before any citable
run is performed**:

```bash
python measurement/validate_hardware.py --json measurement/hardware_report.json
```

Required capabilities:

- readable **NVML GPU cumulative energy**;
- readable **CPU-package energy**;
- readable **DRAM energy**;
- **`measurement_complete_possible == true`** under the project's measurement
  boundary.

> This requirement is **capability-based, not vendor-based.** The methodology
> requires a host that *exposes and permits reading* these energy domains. No
> particular CPU vendor, microarchitecture or sysfs driver is assumed — the
> probe discovers whatever domains the host provides and classifies them by the
> names the host reports.

The current WSL2 development host **remains useful for software validation** —
unit tests, schema checks, KG plumbing, GPU counter behaviour — but is **not
eligible for final measurements**.

Additionally, **real local LLM / vLLM inference has been exercised only on the
development host.** Real `google/gemma-3-4b-it` runs through vLLM exist
(`measurement/runs/gemma4b_*`), which confirms token-field attribution and LLM
event boundaries under genuine inference — but on a host where CPU-package and
DRAM are unavailable, so those runs are integration validation, not measurement.
Real inference must still be validated on the experimental node before the
Gemma pilot.

---

## Roadmap

`[x]` complete `[~]` in progress / next `[ ]` planned

### Phase 0 — Rehearsal / baseline
- [x] Self-host Freebase
- [x] Reproduce CoR workflow with Qwen2.5-7B
- [x] Characterize initial operational issues

### Phase 1 — Formalize CoR measurement
- [x] Schema-v1 semantic event instrumentation
- [x] Stable CoR operation taxonomy
- [x] Correct non-overlapping energy accounting
- [x] Preserve missing domains as null
- [x] Attributed per-event artifact
- [x] Software / unit validation

### Phase 2 — Measurement validation framework
- [x] Hardware capability probe
- [x] Idle / noise validation tooling
- [x] CPU / KG validation tooling
- [x] GPU validation tooling
- [x] Repeatability statistics
- [x] Trajectory accounting and overlap detection
- [x] `MEASUREMENT_SPEC.md`
- [x] Development-host NVML validation

### ⬅️ Phase 3 — Dedicated bare-metal dry run — **CURRENT NEXT PHASE**
- [ ] Provision dedicated bare-metal node
- [ ] Run `validate_hardware.py`
- [ ] Verify GPU, CPU-package and DRAM domains
- [ ] Verify `measurement_complete_possible`
- [ ] Characterize idle baseline
- [ ] Characterize **CPU-package** temporal resolution
- [ ] Characterize **DRAM** temporal resolution
- [ ] Characterize GPU / NVML behaviour on that host
- [ ] Check independent BMC / IPMI / Redfish node telemetry if available
- [ ] Validate real vLLM inference
- [ ] Confirm repeated-run variance
- [ ] Validate whole-trajectory reconciliation
- [ ] Confirm the host is eligible for citable measurements

### Phase 4 — Small real-model pilot
- [ ] Freeze machine-readable experiment configuration
- [ ] Configure pinned Gemma / vLLM environment
- [ ] Run one representative Gemma model with CoR
- [ ] Use a small WebQSP/CWQ sample, **not** the full matrix
- [ ] Inspect real per-operation energy profiles
- [ ] Confirm token / event attribution under real inference
- [ ] Confirm trajectory coverage and measurement quality
- [ ] Establish practical run-time / compute-cost estimates

This pilot determines whether anything needs correcting before scaling.

### Phase 5 — Extend unified measurement to ToG and PoG
- [ ] Audit ToG semantic stages
- [ ] Audit PoG semantic stages
- [ ] Map both into the same canonical schema
- [ ] Validate each paradigm with small smoke runs
- [ ] Ensure cross-paradigm operation categories remain analytically comparable

> **Instrumentation status.** Only CoR is on schema v1. **ToG has no
> instrumentation.** **PoG has limited legacy-format instrumentation** — a
> single `embedding:prune` event recorded through the pre-schema-v1 call form
> in `chain_of_relations/methods/pog/tools/entity_condition_prune.py` — and is
> **not migrated to schema v1**. Neither paradigm can currently produce a
> schema-v1 trajectory.

### Phase 6 — Finalize `PoG_sub`
- [ ] Quantify PoG runtime / depth behaviour
- [ ] Construct documented stratification over CWQ using reasoning hop count
      and answer-set cardinality
- [ ] Freeze `PoG_sub` before the final matrix
- [ ] Ensure ToG and CoR are evaluated on the same `PoG_sub` questions for
      direct PoG comparisons

### Phase 7 — Gemma 3 matrix runs
- [ ] Gemma 3 1B / 4B / 12B / 27B
- [ ] CoR × full WebQSP, CoR × full CWQ
- [ ] ToG × full WebQSP, ToG × full CWQ
- [ ] PoG × full WebQSP, PoG × `PoG_sub` on CWQ

Every matrix run must hold constant and record: hardware platform, model
revision, precision/quantization policy, vLLM configuration, decoding settings,
benchmark settings, measurement boundary, and run provenance. Exact final
settings are **not yet frozen** and will be fixed in Phase 4.

### Phase 8 — Operation-level decomposition
> *"Where is energy actually spent?"*

- [ ] Decompose hardware-grounded energy by semantic operation type; LLM vs KG
      work; `iteration`; `traversal_depth`; trajectory length; question;
      dataset; model size; paradigm
- [ ] Report **both total and distributional** behaviour, with robust summary
      statistics rather than means alone — rehearsal trajectories varied
      substantially in length and operation count, so the energy distribution
      is expected to be wide and should not be summarised by a mean
- [ ] Compare measured energy against proxies: token counts, call counts,
      latency — to identify where proxies **agree with or diverge from**
      hardware-grounded energy

### Phase 9 — Cross-paradigm energy–effectiveness comparison
> *"Does any paradigm Pareto-dominate on the F1–energy frontier?"*

- [ ] For **common benchmark questions**, compare ToG / CoR / PoG on F1, Joules
      per trajectory, derived CO2e reported alongside energy, and trajectory
      characteristics
- [ ] Locate configurations on the **F1–energy frontier** and test whether any
      paradigm Pareto-dominates another

Energy is the **primary physical metric**; CO2e is **derived** from it using a
declared location-specific grid carbon-intensity factor, and is always reported
alongside the energy it came from.

### Phase 10 — Final output and handoff to follow-on work
- [ ] Produce final operation-level energy profiles
- [ ] Produce model / paradigm F1–energy comparisons
- [ ] Report fallback / truncation trajectories separately where required
- [ ] Document the dominant and redundant operation classes identified here
- [ ] Hand those measurements to follow-on intervention work as empirical input

> **Scope boundary.** This work **measures where energy is spent**. Using that
> diagnosis to **redesign agent execution** is follow-on work; intervention
> design is out of scope for this repository.

---
