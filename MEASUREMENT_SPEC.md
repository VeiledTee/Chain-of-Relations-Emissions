# Measurement Specification

**Status:** measurement protocol contract for this study.
**Schema:** event schema v1 (frozen — see §4).
**Scope:** Chain-of-Relations (CoR) over Freebase. ToG, PoG and SubgraphRAG are
out of scope for this document.

This is a research-method contract, not developer documentation. It states what
is measured, what is explicitly *not* measured, and what the instrument's
validated limits are. It contains no experimental conclusions.

---

## 1. Measurement objective

This study asks:

> When inference is embedded in a multi-step agentic loop over a knowledge
> graph, where is energy actually spent?

Answering that requires hardware-grounded operational energy at two levels:

1. **trajectory level** — energy consumed answering one question end to end;
2. **semantic-operation level** — energy attributed to the specific reasoning
   or retrieval operation being performed (relation ranking, reasoning, answer
   filtering, direct fallback generation, KG search, name resolution).

The instrument must therefore identify *what operation was running*, not merely
that an LLM or a tool was called, and must attribute measured hardware energy to
those operations without fabricating precision it does not have.

---

## 2. Measurement boundary

**Included (additive):**

| Domain | Source |
|---|---|
| GPU | NVML `nvmlDeviceGetTotalEnergyConsumption` (cumulative counter) |
| CPU package | RAPL package-level domain, whatever the host names it (`package-*`, `socket*`) |
| DRAM | RAPL `dram` |

**Measured but NOT additive (diagnostic only):**

| Domain | Why excluded from the total |
|---|---|
| CPU `core` | A subdomain *contained within* `package`. Adding both double-counts. |
| `uncore` | Also within `package`, and not the `core` diagnostic. Dropped entirely. |

**Excluded from `measured_energy_j`:**

- `psys` — a platform domain that is a *superset* of `package`; counting both
  double-counts in the opposite direction;
- storage, networking, fans/cooling, PSU conversion loss;
- facility overhead (PUE), building HVAC;
- any component the host does not instrument.

> This is **measured operational hardware energy within a defined boundary**.
> It is **not** total machine energy and **not** facility energy. Any figure
> derived from it must be described that way.

---

## 3. Accounting equation

```
E_measured = E_GPU + E_CPU-package + E_DRAM
```

`core` is **never** a term. RAPL `core` reports the energy of the CPU core
complex, which is already included inside the `package` domain reading; summing
them counts the same joules twice. `core` is retained only as a diagnostic
signal (e.g. to see how much of package energy is core versus uncore/IO).

If **any** additive component is unavailable, `measured_energy_j` is `null` —
never a partial sum. A partial sum would silently redefine the boundary and
understate energy while appearing precise.

---

## 4. Event schema (v1 — frozen)

Every event is one JSON line in `events.jsonl`.

```
schema_version      always 1
event_id            unique per event

run_id              provenance, §12
question_id
dataset
paradigm

iteration           see below
traversal_depth     see below
step_index          see below

operation_type      llm | kg | embedding | system
operation_label     §5; type is always derivable from the label prefix

start_timestamp
end_timestamp
duration_s

gpu_energy_j        NVML counter delta, or null
cpu_package_energy_j  always null at write time; filled by attribution
dram_energy_j         always null at write time; filled by attribution

input_tokens        null when the API reported no usage
output_tokens

status              ok | error | timeout | cancelled

model_name
model_revision
git_commit
hardware_id

meta                operation-specific: attempts, tool_attempt, entity_count,
                    fallback, fallback_reason, rows, error, ...
```

### The three ordering fields

These measure different things and must not be conflated:

| Field | Definition |
|---|---|
| `iteration` | Monotonically increasing CoR control-loop iteration within a question. Advances once per processed control-loop/search state and **does not decrease**. |
| `traversal_depth` | Current DFS/search depth of the state being expanded. May increase **or decrease** through backtracking. `null` for operations outside the DFS, such as `llm:direct_answer`. |
| `step_index` | Monotonically increasing measured-event index within the question. Orders **measurements**, not reasoning decisions. |

A backtracking question looks like:

```
iteration        0  1  2  3  4  5  6  7
traversal_depth  0  1  2  2  1  2  2  null
```

with `step_index` climbing straight through, typically several steps per
iteration.

**Schema v1 is frozen.** It changes only if hardware validation exposes an
actual correctness defect, and only with that defect documented first.

---

## 5. Operation taxonomy (frozen)

Defined in `chain_of_relations/energy_taxonomy.py`. Labels are assigned by the
caller that knows *why* an operation is happening, and are **never** inferred
from prompt text.

| Label | Real CoR call site | Semantic stage |
|---|---|---|
| `llm:relation_rank` | `CoRAgent.relation_prune` → `tools/relation_prune.py` | LLM selects top-k relations from head/tail candidates |
| `llm:reason` | `CoRAgent.reasoning` → `tools/reasoning.py` | LLM emits `Stop`/`Forward`/`Backtrack`/`Constraint` |
| `llm:answer_filter` | `CoRAgent.validate` → `methods/cor/tools/filter.py` | LLM selects valid answers from TargetEntity candidates (KG-grounded answering) |
| `llm:direct_answer` | `CoRAgent.generate_directly` → `tools/generate_directly.py` | Closed-book fallback when DFS produces no answer. Carries `meta.fallback=true`, `meta.fallback_reason` |
| `kg:relation_search` | `tools/relation_search.py` | SPARQL for head/tail relations |
| `kg:entity_search` | `tools/entity_search.py` | SPARQL for target entities |
| `kg:id2name` | `db_func.id2entity_name_or_type` / `id2entity_names` | Entity id → label resolution |
| `kg:sparql` | any other SPARQL execution | unclassified graph query |

`llm:entity_prune`, `embedding:prune` and `system:orchestration` exist in the
taxonomy for other paradigms but are **not** part of the CoR taxonomy: CoR has
no embedding stage and does not use `tools/entity_prune.py`.

`llm:generate` is legacy and must never appear in a CoR run.

---

## 6. Event lifecycle

**Timing.** `energy_events.mark()` captures `(wall_clock, cumulative_gpu_mJ)` at
the instant an operation starts and again when it ends. Both boundaries are
taken in-band, in the same process that performs the work.

**GPU energy.** Computed as the NVML counter delta between the two marks,
converted mJ → J. Where the counter is unavailable, `gpu_energy_j` is `null`
and attribution falls back to integrating the sampled power curve, recording
`gpu_energy_source` so the two are never confused.

**CPU/DRAM.** Not knowable in-band. `measurement/power_logger.py` samples
cumulative RAPL counters out of band on the same wall clock;
`measurement/attribute.py` differences them across each event's window.

**Status.** Controlled vocabulary `ok | error | timeout | cancelled`. Terminal
failures are recorded explicitly and never dropped — their energy was really
spent.

**Retries.** Two distinct levels, deliberately separate fields:

- `meta.attempts` — retries *inside* one `LLMAPI.generate` call, collapsed into
  a single event;
- `meta.tool_attempt` — the shared tool's own retry loop, where each pass is a
  separate `llm_generate` call and therefore a **separate event**.

`attempts=3, tool_attempt=2` means the second tool-level pass, whose single
event covers three API attempts.

**Asynchronous GPU work.** CUDA kernels are queued asynchronously. An event
boundary taken without synchronisation can be reached before the queued work
executes, attributing its energy to whichever event is open next. Any workload
that submits GPU work directly must synchronise before closing an event. Work
executed inside a separate model-serving process (vLLM) cannot be synchronised
from the client and is discussed in §14.

---

## 7. Raw versus attributed artifacts

Four layers, deliberately not collapsed:

| Artifact | Written by | Contents |
|---|---|---|
| `events.jsonl` | `chain_of_relations/energy_events.py` | Semantic event timeline; GPU counter delta where available |
| `power.csv` | `measurement/power_logger.py` | Hardware timeline: GPU watts + cumulative RAPL counters, ~10 Hz |
| `events_attributed.jsonl` | `measurement/attribute.py` | Every raw event, unchanged, plus attributed energy fields. **Canonical per-event artifact for this study.** |
| `energy_summary.csv` | `measurement/attribute.py` | Derived aggregate by operation label / type / question |
| `trajectory_summary.csv` / `.json` | `measurement/trajectory.py` | Per-question trajectory accounting (§9) |
| `codecarbon/emissions.csv` | CodeCarbon | Whole-run CO2e only (§13) |

Attribution must preserve every semantic field: question id, iteration,
traversal depth, step index, operation label, tokens, status and all
provenance. Attributed events add:

```
gpu_energy_j, gpu_energy_source
cpu_package_energy_j
cpu_core_energy_j            # diagnostic, never additive
dram_energy_j
measured_energy_j
available_energy_domains     # which additive domains this event actually has
measurement_complete         # true iff all three additive domains present
```

---

## 8. Missing-domain policy

> **A missing measurement is not zero.**

- A domain the host does not expose is `null` in every artifact.
- `measured_energy_j` is `null` whenever any additive component is missing.
- `available_energy_domains` lists the additive domains actually obtained for
  that event, e.g. `["gpu"]` on a GPU-only host.
- `measurement_complete` is `true` only when GPU, CPU package and DRAM are all
  present.
- `0.0` is a legitimate *measured* value and is distinct from `null`. Analysis
  must not conflate them.

The capability probe (`measurement/validate_hardware.py`) reports every domain
as `SUPPORTED` / `UNREADABLE` / `UNAVAILABLE` / `UNKNOWN`, and states up front
whether `measurement_complete` can *ever* be true on the host.

---

## 9. Trajectory accounting

Implemented in `measurement/trajectory.py`, per question:

```
trajectory_energy_j       energy over the whole question window, taken from
                          the hardware timeline in one shot
sum_attributed_j          sum of per-event attributed energies
unattributed_energy_j  =  trajectory_energy_j - sum_attributed_j
attribution_coverage   =  sum_attributed_j / trajectory_energy_j
```

**Coverage is never forced to 1.0.** The residual is real and must stay
visible: Python orchestration, inter-event gaps, model-server idle draw and
background services all consume energy inside the trajectory window that
belongs to no named operation. Forcing it into named operations would overstate
operation-level attribution — precisely the claim this study is trying to
establish honestly.

### Like-for-like instruments (required)

Both sides of the reconciliation must come from the **same physical
instrument**, or the residual measures instrument disagreement rather than
unattributed work. For every domain the trajectory reference is a **cumulative
hardware counter differenced across the window**:

```
trajectory_gpu_energy_j          = nvml_cumulative_end - nvml_cumulative_start
trajectory_cpu_package_energy_j  = rapl_package_end    - rapl_package_start
trajectory_dram_energy_j         = rapl_dram_end       - rapl_dram_start
```

which matches how per-event energy is produced. The whole-window figure is
still derived independently of the event log — the sampler process reads the
counter, the harness reads it in-band — so the check remains genuine rather
than circular, while removing the instrument mismatch.

**Integrated sampled power is NOT a reference.** At ~10 Hz it resolves fast
inference power transients poorly, and the resulting window estimate is
unreliable in **both directions** — measured `sampled_vs_counter_ratio` on real
Gemma 3 4B trajectories spans roughly 0.91–1.20, i.e. the estimate errs low on
some windows and high on others rather than carrying a fixed bias that could be
corrected. Its error grows as trajectories shorten and sample counts fall,
which is what previously drove attribution coverage above 1 on short
trajectories despite zero event overlap. It survives only as the separately
named diagnostic `sampled_gpu_energy_estimate_j`, alongside `mean_gpu_power_w`
and `peak_gpu_power_w`, with `sampled_vs_counter_ratio` characterising the
disagreement.

A domain whose counter is unavailable is reported as unavailable
(`gpu_energy_source: "unavailable"`, `trajectory_gpu_energy_j: null`,
coverage `null`). It is **never** back-filled from the sampled estimate.

### Residual sign

```
unattributed_energy_j = trajectory_energy_j - sum_attributed_j
```

Small **negative** residuals are expected and are **not clamped to zero**. The
event log reads the counter in-band at exact event boundaries, while the
trajectory reference interpolates the sampler's reads to the trajectory edges;
counter update granularity and boundary timing can place slightly more energy
inside events than the interpolated window shows. Clamping would conceal the
very instrument behaviour this reconciliation exists to expose. The summary
reports `n_negative_gpu_residual`, `gpu_residual_min_j` and
`max_abs_negative_gpu_residual_j` so the magnitude can be characterised.

### Overlap rule

Plain summation of per-event energy is valid **only when events do not overlap
in time**. Where events nest or overlap, the overlapped interval is counted once
per overlapping event and the sum can exceed the trajectory total, giving
coverage > 1.0.

The instrument therefore **measures overlap rather than silently correcting for
it**. Every trajectory row reports `overlap_s`, `max_concurrency`,
`events_overlap` and `coverage_valid`. The accounting rule is:

> **Plain summation, applied only to runs proven overlap-free.** That condition
> is checked per trajectory, not assumed. `coverage_valid` is true only when an
> independent window energy exists *and* no events overlap. Coverage above 1.0
> is reported as-is and flagged, never clipped.

CoR's control loop is strictly sequential, so overlap is expected to be zero;
`max_concurrency > 1` in a CoR run indicates an instrumentation defect and must
be investigated, not corrected away.

### Hard cases

| Case | Handling |
|---|---|
| Nested events | Detected via `max_concurrency`; invalidates plain summation for that trajectory |
| Overlapping events | Same; `overlap_s` quantifies the double-counted interval |
| GPU asynchronous execution | Requires synchronisation before closing an event (§6); unsynchronisable for out-of-process serving (§14) |
| Extremely short events | Below the **GPU** counter resolution floor (§10), GPU energy is systematically under-measured and the deficit surfaces as unattributed energy. CPU/DRAM resolution at this scale is untested (§10) |
| Background vLLM activity | Model-server idle draw is inside the trajectory window but belongs to no operation; appears as unattributed |
| Background Virtuoso activity | Same; Virtuoso CPU work during a `kg:*` event *is* attributable, its idle draw is not |
| Retry events | Separate events with distinct `tool_attempt`; all retained, none dropped |
| Inter-event energy | Reported explicitly as `inter_event_gap_s` and unattributed energy |
| Model-serving idle energy | Unattributed by construction; characterise via §10 idle baseline |
| Event energy exceeding trajectory energy | Possible only under overlap; flagged by `coverage_valid=false`, never clipped |

---

## 10. Resolution floor, idle and background energy

### GPU counter resolution floor

**Scope of this finding.** What follows concerns the **NVML GPU energy counter
only**. It establishes that **per-event GPU energy for operations substantially
shorter than the NVML update interval is not reliably resolvable**. On the only
host measured to date (RTX 4090, driver 591.86) that interval is **106 ms**
— median 106.1–107.1 ms across six `validate_hardware.py` runs, ~80 counter
updates observed per run. This says nothing about the CPU-package or DRAM
domains, and the interval must be re-measured on every host.

The median update **step** on that host was 138–158 mJ across the same runs.
The step is not a hardware constant: it is instantaneous power multiplied by
the update interval, so it moves with load. Only the interval bounds which
operations the counter can resolve.

The NVML cumulative energy counter does not update continuously. Its update
interval must be measured per host (`validate_hardware.py` reports it) because
it bounds which operations its counter can resolve.

A window shorter than the update interval reads either **exactly 0 J** or one
whole update quantum. This is **not** symmetric noise: such windows are
**systematically biased low**, because the zero case dominates.

The validation procedure therefore records, per host:

- median counter update interval and median update step;
- the fraction of windows reading exactly 0 J as a function of window length;
- the window length at which implied power converges to the known true power.

**Per-operation GPU energy must only be reported for operations above the
measured GPU floor.** For shorter operations the per-event GPU figure is
quantization, not measurement, and must be reported as such — the energy is not
lost from the trajectory total, it simply cannot be assigned to a named
operation from the GPU counter and appears in `unattributed_energy_j`.

### CPU-package and DRAM temporal resolution: NOT YET TESTED

The equivalent question for the CPU-package and DRAM domains — how short an
operation their counters can resolve — **has not been tested**, because those
domains were unavailable on the host used for validation to date (a virtualised
host exposing no powercap/MSR interface). RAPL counters are a different
mechanism with their own, independent update characteristics; the GPU figure
above must **not** be assumed to transfer to them.

This matters directly for KG operations. `kg:*` work is primarily CPU and
memory work: the CPU-package and DRAM domains are its **primary** domains, and
the GPU is largely a bystander during it. Whether short KG operations are
resolvable **in the domains that actually carry their cost therefore remains an
open question**, to be answered by running the §11 procedure on the target
bare-metal host once CPU-package and DRAM are available. Until then:

- do **not** claim that short KG operations are energy-unresolvable in general;
- do claim, and only claim, that their **per-event GPU** energy is not reliably
  resolvable below the measured GPU floor;
- treat the CPU/DRAM resolution characterisation as a required, outstanding
  validation step before any operation-level energy claim about KG work.

### Idle and background

Idle draw is **characterised, not subtracted**. `validate_measurement.py --mode
idle` measures baseline draw and its slice-to-slice variation per domain. No
idle subtraction is applied to research measurements, and none may be applied
without a separate, documented justification: subtracting a baseline from
whole-device counters would silently convert a measured quantity into a modelled
one.

---

## 11. Validation procedure

Run in this order. Each step has a pass condition; failures are findings, not
tool errors.

1. **Hardware capability check** — `measurement/validate_hardware.py`.
   Records environment (bare metal / VM / WSL / container), CPU, GPU, NVML
   status, every RAPL zone with its exact name and classified domain,
   readability and permissions, counter resolution, and whether
   `measurement_complete` is possible at all.
2. **Idle** — `--mode idle`. Baseline and noise per domain. Verifies counters
   advance sensibly and reveals background contamination.
3. **CPU / KG** — `--mode cpu`. Repeated fixed SPARQL, no inference. Verifies
   CPU package and DRAM rise where available, GPU stays near idle, and `kg:*`
   event boundaries behave.
4. **GPU** — `--mode gpu`. Deterministic load. Verifies the GPU counter tracks
   real work and quantifies separation from idle. With a real model endpoint
   this additionally validates token fields and LLM event boundaries.
5. **Repeated** — `--mode repeated`. 10–20 identical operations; report mean,
   median, sd, min, max and CV. Small-sample CV is indicative only.
6. **CoR smoke** — 3–10 questions with real inference and real Freebase.
   Validates schema v1 under real conditions, semantic label distribution,
   per-event attribution and `measurement_complete`.
7. **Trajectory reconciliation** — §9. Report coverage and its decomposition
   into inter-event gap versus sub-floor quantization loss.
8. **Independent node telemetry** — §5 below.

### Independent validation

Independent validation requires a power source that does **not** share a
mechanism with NVML or RAPL: BMC/IPMI/Redfish node power, or an external
wall meter. Where such a source exists, aggregate counter-derived energy is
compared against it over a long controlled workload and the difference reported.

**CodeCarbon is not independent validation** and must never be used as such
(§13).

Where no independent source exists, that must be stated explicitly as a
limitation rather than substituted with a correlated source.

---

## 12. Host capability requirement and reproducibility metadata

### Host capability requirement

The experimental host is specified by **capability, not by vendor, model or
product line**. A host is eligible only if it:

1. exposes a **readable CPU-package energy domain**;
2. exposes a **readable DRAM energy domain**;
3. exposes **NVML GPU energy** (`nvmlDeviceGetTotalEnergyConsumption`);
4. **passes `measurement/validate_hardware.py`** — that is, the probe reports
   `measurement_complete_possible: true`, with every additive domain
   `SUPPORTED` and its counter observed to advance.

`validate_hardware.py` **must be run and its report retained before any run
intended as a citable measurement.** A host that fails it may still be used for
software validation, but its runs are validation runs, not measurements.

Two further conditions apply for a complete boundary in practice:

- the CPU energy interface must be reachable, which in general requires a
  non-virtualised OS installation;
- the additive domains must remain readable by the measuring user for the
  duration of the run.

No assumption is made about CPU vendor, microarchitecture or the sysfs driver
providing the counters. The instrument discovers whatever energy domains the
host exposes, classifies them by their reported domain names, and reports
capability accordingly.

### Reproducibility metadata

Every event carries run-scoped provenance, configured once per run:

| Field | Source |
|---|---|
| `run_id` | `ENERGY_RUN_ID`, set by `measure_run.py` so all artifacts share it |
| `dataset` | run argument |
| `paradigm` | run argument (`cor`) |
| `model_name` | `MODEL_NAME` |
| `model_revision` | `MODEL_REVISION` — **must be set** for a citable run |
| `git_commit` | auto-detected working-tree revision |
| `hardware_id` | hostname + NVML GPU model(s) |

### TODO — serving-stack readiness before citable runs

Two settings used during validation on the WSL2 development host are
**validation-only** and must not be carried into the final experiment manifest:

- [ ] **`FLASHINFER_DISABLE_VERSION_CHECK=1` must be removed.** The development
      environment currently has mutually incompatible FlashInfer packages
      (`flashinfer-cubin` 0.6.13 against `flashinfer-python` 0.5.2), and vLLM
      refuses to start without the bypass. Suppressing a version check is
      acceptable to prove an integration path; it is **not** acceptable for a
      citable measurement, because the attention backend actually exercised is
      then unverified. Final runs must pin **mutually compatible** serving
      dependencies and start with **no version-check bypass**.
- [ ] **`--gpu-memory-utilization 0.78` must be re-derived, not copied.** That
      value is specific to this WSL2/Windows-shared RTX 4090, where roughly
      3.9 GB of the 24 GB device is held by the Windows host and invisible to
      WSL, so vLLM's fraction-of-*total* accounting over-commits at the usual
      0.90. On a dedicated bare-metal node the free/total ratio differs and the
      value must be recomputed from that host's actual free VRAM.

Both settings must be resolved and the resolution recorded before any run is
treated as a measurement.

A publishable run additionally requires, recorded alongside the artifacts:

- software versions (driver, CUDA, Python, torch/vLLM, SPARQL endpoint build);
- model serving configuration (engine, quantisation, `max-num-seqs`,
  tensor parallelism, KV cache settings);
- decoding configuration (temperature, `max_tokens`, seeds where applicable);
- host state (bare metal vs virtualised, otherwise-idle, single writer).

Runs missing `model_revision` or serving configuration are **validation runs,
not citable measurements**.

---

## 13. CO2e conversion

CO2e is a **derived environmental metric**, not a measurement:

```
measured Joules -> declared grid carbon intensity -> gCO2e
```

The carbon-intensity factor must be stated explicitly (region, source, year)
wherever a CO2e figure appears. CodeCarbon's role is **only** this conversion.

CodeCarbon reads NVML for the GPU and RAPL-or-a-TDP-estimate for the CPU — the
same mechanisms this instrument reads directly, plus an estimate where RAPL is
absent. Comparing our counter sums against CodeCarbon compares an instrument
against itself and must not be presented as validation.

---

## 14. Known limitations

1. **Whole-device, not per-process.** NVML reports energy for the entire GPU.
   On a shared or multi-tenant GPU, attribution is invalid. All measurements
   assume an otherwise-idle, single-user, single-writer host.
2. **GPU counter resolution floor.** Operations substantially shorter than the
   NVML update interval cannot have their **GPU** energy resolved by counter
   differencing and are systematically biased low (§10). Since `kg:*`
   operations are milliseconds long and `llm:*` operations are seconds, any
   claim about the inference/retrieval split that rests on GPU energy must
   state this.
3. **CPU-package and DRAM temporal resolution is uncharacterised.** Those are
   the primary domains for KG work, and their resolvability at millisecond
   scale has not been tested because the domains were unavailable on the
   validation host (§10). No claim about whether short KG operations are
   resolvable in their primary domains can be made until the §11 procedure has
   been run on a host that exposes them.
4. **Asynchronous execution.** GPU work queued by a separate serving process
   cannot be synchronised from the client, so an LLM event's boundaries bound
   the *request*, not precisely the kernels. Longer events make this
   proportionally smaller.
5. **Uninstrumented hardware.** Storage, network, fans, PSU loss and any
   platform domain outside RAPL are excluded from the boundary (§2).
6. **Background services.** Virtuoso and the model server draw power inside the
   trajectory window irrespective of the current operation; this appears as
   unattributed energy and is not redistributed.
7. **Domain availability is host-dependent.** Not every host exposes every
   additive domain: some platforms expose no DRAM energy domain, and
   virtualised hosts typically expose no CPU energy interface at all. Where any
   additive domain is absent, `measured_energy_j` stays `null` regardless of
   privileges. The host capability requirement is stated in §12 and is
   capability-based, not vendor-based.
8. **No independent node telemetry** where BMC/IPMI/Redfish is absent; the
   counters are then unvalidated against an external reference.
9. **Small-sample statistics.** Repeatability figures from 10–20 repetitions
   are indicative; CV should not be quoted as a precision claim.

---

## 15. Analysis contract

Which fields and aggregations support the planned analyses. No conclusions here.

### Operation-level decomposition — per operation, iteration and depth

Source: `events_attributed.jsonl`.

- **By operation:** group by `operation_label`; report event count, summed
  energy per available domain, and mean energy per call. Per-domain resolution
  gating applies (§10): **GPU** energy is reported only for operations above
  the measured GPU floor, and below it is marked unresolved rather than
  reported as a value. CPU-package and DRAM energy is gated the same way once
  their floors have been measured on the experimental host; until that
  characterisation exists, no resolution claim is made about those domains in
  either direction.
- **By control-loop progress:** group by `iteration` to show how cost
  accumulates as the agent makes more decisions.
- **By search depth:** group by `traversal_depth`, with `null` (outside-DFS)
  bucketed separately from depth 0, to show how cost varies with how deep the
  search goes and how backtracking redistributes it.
- **Fallback cost:** isolate `meta.fallback=true` events to separate
  closed-book fallback cost from KG-grounded answering.
- **Retry cost:** `meta.attempts` and `meta.tool_attempt` quantify wasted work.
- **Coverage context:** every decomposition is reported alongside
  `attribution_coverage` from `trajectory_summary`, so the share of trajectory
  energy the decomposition actually explains is always visible.

### Cross-paradigm energy–effectiveness comparison — question-paired, on an F1–energy frontier

Source: `trajectory_summary.csv` joined to answer scoring on `question_id`.

- **Pairing:** comparisons are paired by `question_id` across paradigms, never
  by dataset means, so per-question difficulty variation does not confound.
- **Energy axis:** `trajectory_measured_energy_j` where
  `measurement_complete`; otherwise the explicitly-named available-domain
  subset, with the domain set reported next to every figure.
- **Quality axis:** existing answer scoring, unmodified by this instrument.
- **Dispersion:** per-question energy distributions (median, p90, p99, tail
  ratios), since the agentic fan-out story is about variance, not just means.
- **Validity gate:** trajectories with `coverage_valid=false` are excluded from
  frontier claims and reported separately with the reason.

---

## Appendix — tooling

| Tool | Purpose |
|---|---|
| `measurement/validate_hardware.py` | Capability probe; `--json` for a machine-readable report |
| `measurement/validate_measurement.py` | Validation workloads: `--mode idle\|cpu\|gpu\|repeated\|cor-smoke` |
| `measurement/power_logger.py` | Out-of-band hardware timeline |
| `measurement/attribute.py` | Joins timelines; writes attributed events + summaries |
| `measurement/trajectory.py` | Trajectory accounting and overlap detection |
| `measurement/analyze.py` | Derived tables and figures |
| `measurement/measure_run.py` | Measured-run orchestrator |
| `tests/test_energy_measurement.py` | Frozen schema-v1 contract tests |
| `tests/test_hardware_validation.py` | Capability, classification and accounting tests |
| `tests/test_package_boundary.py` | One-way dependency rule and profiler public API |

### Where the instrument lives

The measurement instrument is the standalone package `agent_energy_profiler`.
It knows nothing about CoR, Freebase, KGQA or any benchmark, and the dependency
direction is one-way and test-enforced:

```
chain_of_relations / measurement  ->  agent_energy_profiler     allowed
agent_energy_profiler             ->  chain_of_relations        FORBIDDEN
```

| Contract | Implementation | Compatibility entry point |
|---|---|---|
| Semantic event timeline | `agent_energy_profiler/events.py` | `chain_of_relations/energy_events.py` (CoR adapter) |
| Hardware timeline | `agent_energy_profiler/sampling.py` | `measurement/power_logger.py` |
| Attribution | `agent_energy_profiler/attribution.py` | `measurement/attribute.py` |
| Trajectory accounting | `agent_energy_profiler/trajectory.py` | `measurement/trajectory.py` |
| Long-format rollups | `agent_energy_profiler/aggregate.py` | — |
| Capability probe | `agent_energy_profiler/validation.py` | `measurement/validate_hardware.py` |
| Counter access | `agent_energy_profiler/hardware/{nvml,rapl,discovery}.py` | — |
| Joules -> CO2e | `agent_energy_profiler/carbon.py` | — |

Every command line documented in this spec still works: the paths in the table
above are thin re-export shims, not reimplementations. The move was verified
behaviour-preserving by re-attributing three existing runs and confirming
`events_attributed.jsonl`, `energy_summary.csv` and `trajectory_summary.{csv,json}`
byte-identical before and after.

CoR's *semantics* stay with CoR: the frozen operation taxonomy
(`chain_of_relations/energy_taxonomy.py`) and the one structural SPARQL
classifier live in the adapter. The profiler accepts whatever label the caller
supplies and never infers one from prompt text, function names or the call
stack.

Carbon conversion is a separate, explicit step. `carbon.py` multiplies measured
Joules by a grid intensity the **caller supplies**, and records its source,
region, units, method (average vs marginal) and timestamps. It performs no
lookup and has no default factor: without a supplied intensity there is no CO2e
figure. Joules remain the measured product; CO2e is a derived claim about a
grid, and is only as defensible as the factor behind it.
