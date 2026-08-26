# Chain-of-Relations Emissions

An **energy-instrumented fork of Chain-of-Relations (CoR)**, used as the unified
experimental harness for **Contribution 2 (C2)** of a PhD thesis on
hardware-grounded energy measurement of agentic KGQA.

Upstream CoR is a knowledge-graph reasoning method (ACL 2026 Findings — see
[How to Cite](#how-to-cite)). This fork adds an operation-level energy
measurement layer on top of it and treats CoR, ToG and PoG as comparable
paradigms running over one shared graph, model-serving stack and measurement
procedure.

> **Status in one line:** the measurement instrument is built and
> software-validated; **no final C2 experimental results have been produced
> yet**. See [Current Status](#current-status) and [Roadmap](#roadmap).

---

## Contribution 2 Research Goal

**RQ2 — When inference is embedded in a multi-step agentic loop over a
knowledge graph, where is energy actually spent?**

- **RQ2a** — How does per-step, hardware-grounded energy consumption decompose
  across reasoning, tool invocation, and graph retrieval calls, and how does the
  total energy of a trajectory scale with the number of iterations?
- **RQ2b** — Among competing agentic KGQA paradigms answering the same
  questions over the same graph, does any paradigm Pareto-dominate the rest on
  the F1–energy frontier?

Answering these requires energy attributed to *what the agent was actually
doing* — relation ranking, reasoning, answer filtering, graph retrieval — not
merely "an LLM was called". That is what the measurement layer in this fork
provides.

---

## Experimental Scope

### Unified harness

This codebase is the **common experimental harness** for three agentic KGQA
paradigms:

| Paradigm | Status in this repo |
|---|---|
| **Chain-of-Relations (CoR)** | instrumented to frozen schema v1 |
| **Think-on-Graph (ToG)** | implemented; **not yet** migrated to the schema |
| **Plan-on-Graph (PoG)** | implemented; **not yet** migrated to the schema |

All paradigms run against the **same self-hosted Freebase instance**
(≈ **3.12 billion triples**, verified as `3124793702`). The point of a unified
harness is that paradigms become comparable under one knowledge graph, one
model-serving stack, one measurement procedure and one hardware platform —
so an observed difference is a property of the paradigm, not of the setup.

### Datasets

- **WebQSP**
- **ComplexWebQuestions (CWQ)**

### Model family (planned)

The planned model ladder is **Gemma 3** at **1B / 4B / 12B / 27B**.

> These runs **have not been performed**. The Gemma matrix begins only once the
> measurement system passes real bare-metal validation on the dedicated
> experimental node (Phase 3).

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

### Measurement boundary

```
measured_energy_j = gpu_energy_j + cpu_package_energy_j + dram_energy_j
```

- CPU **core** is **diagnostic only** and is never added — it overlaps with
  CPU-package energy, and summing both double-counts.
- A hardware domain the host does not expose stays **`null`**, never `0`.
- Energy is the primary physical metric; **CO2e is derived** from it using a
  declared location-specific grid carbon-intensity factor.

Full definition: **[C2_MEASUREMENT_SPEC.md](C2_MEASUREMENT_SPEC.md)**.

---

## Current Status

Three tiers of result exist in this repository and **must not be conflated**:

| Tier | What it is | Citable as a thesis finding? |
|---|---|---|
| **Rehearsal** | Qwen2.5-7B CoR runs; engineering shakedown | ❌ No |
| **Validation** | schema tests, NVML experiments, synthetic GPU load, stub-LLM smoke runs | ❌ No |
| **Final C2 experiments** | Gemma 3 matrix on a validated bare-metal node | **Not yet run** |

### ✅ Completed — rehearsal and harness setup

Self-hosted Freebase is operational, and full CoR runs have been performed with
**Qwen2.5-7B-Instruct** over WebQSP and CWQ. These are **rehearsal /
engineering results**, not C2 measurements — different model family, and a host
that cannot complete the measurement boundary.

Rehearsal exposed four operational behaviours that shaped the design:

1. **Context growth and truncation at depth.** Deep CWQ trajectories grow the
   prompt until it overflows; 6 questions exceeded native 32K at depth 4.
   `--max-model-len 32768` is required — 8192 caused silent agentic degradation.
2. **Graph-grounding vs closed-book fallback.** When DFS finds no answer, CoR
   falls back to closed-book generation: 12.4% of WebQSP and 30.0% of CWQ
   questions. Fallback trajectories are a different kind of work and are now
   labelled distinctly (`llm:direct_answer`, tagged `fallback=true`).
3. **Strongly skewed question-level workload.** Per-question GPU energy spread
   **75×** min→max; tool-call count spread **2007×**. Means alone will not
   describe this distribution.
4. **Very high `kg:id2name` call volume.** 112,302 of 142,473 WebQSP events
   (**78.8%**) were entity-name resolutions.

> **Call volume is not energy — and neither is settled here.** The number of
> times an operation runs and what it costs are different quantities, and
> reporting either alone is misleading. Establishing the actual relationship
> between them is the object of RQ2a, not something the rehearsal answers.
>
> **Validation-only observation.** The rehearsal trace attributed nearly all
> measured GPU energy to LLM inference, but **this split is not interpreted as
> a Contribution 2 result**. Most KG operations were substantially shorter than
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
**[C2_MEASUREMENT_SPEC.md](C2_MEASUREMENT_SPEC.md)** §4–§5 and the project
contract in [CLAUDE.md](CLAUDE.md); it is not duplicated here.

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
| [`measurement/validate_hardware.py`](measurement/validate_hardware.py) | hardware capability probe; per-domain `SUPPORTED`/`UNREADABLE`/`UNAVAILABLE`/`UNKNOWN`; `--json` report |
| [`measurement/validate_measurement.py`](measurement/validate_measurement.py) | validation workloads: `--mode idle\|cpu\|gpu\|repeated\|cor-smoke` |
| [`measurement/trajectory.py`](measurement/trajectory.py) | trajectory accounting, overlap detection, attribution coverage |
| [`C2_MEASUREMENT_SPEC.md`](C2_MEASUREMENT_SPEC.md) | the measurement protocol contract |

#### Development-host findings — **VALIDATION ONLY**

The development machine is **WSL2 with an RTX 4090**. These are instrument
characterisations, **not** C2 experimental results:

- ✅ NVML cumulative GPU energy works and tracks real load (446.8 W under
  sustained load vs 29.5 W idle).
- ✅ Sustained GPU measurements are repeatable — **CV ≈ 1.7%** over 12 identical
  3 s workloads.
- ⚠️ The RTX 4090 NVML energy counter has an **≈100 ms update interval**.
  **Per-event GPU energy for operations substantially shorter than that is not
  reliably resolved**, and is biased low rather than symmetrically noisy.
- ✅ Freebase is functional (3.12B triples).
- ❌ **CPU-package and DRAM counters are unavailable under WSL2** — no
  `/sys/class/powercap`, no MSR interface.
- ❌ Therefore **`measurement_complete` cannot become true on this host**, and
  `measured_energy_j` is always `null` here.

> **Scope of the ≈100 ms result.** It applies to the **GPU / NVML counter
> only**. The temporal resolution of the **CPU-package and DRAM** counters has
> **not** been measured, because those domains are unavailable on this host.
> RAPL is a separate mechanism with its own update characteristics. Since KG
> operations are primarily CPU and memory work, **whether short KG operations
> are resolvable in their primary domains remains an open question**, to be
> answered on the bare-metal node in Phase 3.

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
eligible for final C2 measurements**.

Additionally, **real local LLM / vLLM inference still needs to be validated on
the experimental node** before the Gemma pilot. All measurement work to date
used either the Qwen2.5-7B rehearsal stack or a stub LLM; token-field
attribution and LLM event boundaries under real vLLM inference are unverified.

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
- [x] `C2_MEASUREMENT_SPEC.md`
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
- [ ] Confirm the host is eligible for citable C2 measurements

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

> ToG and PoG are **not yet instrumented**. Only CoR is on schema v1 today.

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

### Phase 8 — RQ2a trajectory analysis
> *"Where is energy actually spent?"*

- [ ] Decompose hardware-grounded energy by semantic operation type; LLM vs KG
      work; `iteration`; `traversal_depth`; trajectory length; question;
      dataset; model size; paradigm
- [ ] Report **both total and distributional** behaviour — rehearsal showed
      question costs are heavy-tailed (75× energy spread, 2007× tool-call
      spread), so means alone will not be sufficient
- [ ] Compare measured energy against proxies: token counts, call counts,
      latency — to identify where proxies **agree with or diverge from**
      hardware-grounded energy

### Phase 9 — RQ2b cross-paradigm analysis
> *"Does any paradigm Pareto-dominate on the F1–energy frontier?"*

- [ ] For **common benchmark questions**, compare ToG / CoR / PoG on F1, Joules
      per trajectory, derived CO2e reported alongside energy, and trajectory
      characteristics
- [ ] Locate configurations on the **F1–energy frontier** and test whether any
      paradigm Pareto-dominates another

Energy is the **primary physical metric**; CO2e is **derived** from it using a
declared location-specific grid carbon-intensity factor, and is always reported
alongside the energy it came from.

### Phase 10 — C2 output / handoff to C3
- [ ] Produce final operation-level energy profiles
- [ ] Produce model / paradigm F1–energy comparisons
- [ ] Report fallback / truncation trajectories separately where required
- [ ] Document the dominant and redundant operation classes identified by C2
- [ ] Use those measurements as the empirical input to Contribution 3

> **C2 / C3 boundary.** C2 **measures where energy is spent**. C3 **uses that
> diagnosis to redesign agent execution**. Intervention design is out of scope
> for this repository's current work.

---

## Measurement Architecture

Four layers, deliberately not collapsed into one logger:

```
events.jsonl              semantic event timeline (in-band, per operation)
power.csv                 hardware timeline (out-of-band NVML + RAPL, ~10 Hz)
        │
        ▼  measurement/attribute.py  joins the two on a shared wall clock
events_attributed.jsonl   canonical per-event research artifact for C2
energy_summary.csv        derived aggregate
trajectory_summary.csv    per-question trajectory accounting
```

Trajectory accounting reports, per question:

```
unattributed_energy_j = trajectory_energy_j - sum_attributed_operation_energy_j
attribution_coverage  = sum_attributed_operation_energy_j / trajectory_energy_j
```

**Coverage is never forced to 1.0.** Inter-event, orchestration and background
energy stay explicitly visible rather than being pushed into named operations.

| Document | Contents |
|---|---|
| **[C2_MEASUREMENT_SPEC.md](C2_MEASUREMENT_SPEC.md)** | authoritative measurement protocol: boundary, schema v1, taxonomy, event lifecycle, missing-domain policy, trajectory accounting, validation procedure, host capability requirement, limitations, RQ2 analysis contract |
| **[EMISSIONS.md](EMISSIONS.md)** | instrument-level description of the event log, power sampler and attribution join |
| **[CLAUDE.md](CLAUDE.md)** | project contract and schema-v1 field definitions |

### Validation commands

```bash
# 1. is this host eligible to produce citable measurements?
python measurement/validate_hardware.py --json measurement/hardware_report.json

# 2. characterize baseline draw and noise
python measurement/validate_measurement.py --mode idle --seconds 60

# 3. KG workload, no inference
python measurement/validate_measurement.py --mode cpu --reps 15

# 4. GPU counter behaviour under deterministic load
python measurement/validate_measurement.py --mode gpu --seconds 20

# 5. repeatability statistics
python measurement/validate_measurement.py --mode repeated --reps 15

# 6. print (do not launch) the tiny real-CoR validation command
python measurement/validate_measurement.py --mode cor-smoke

# tests
python -m unittest discover -s tests -p "test_*.py"
```

### Running a measured experiment

```bash
python measurement/measure_run.py --tag <run_tag> -- \
    --method cor --dataset webqsp --kb freebase \
    --run_size 3 --depth 3 --relation_width 3 --save_detail true
```

Writes `events.jsonl`, `power.csv`, `events_attributed.jsonl`,
`energy_summary.csv`, `trajectory_summary.csv` and `run.log` into
`measurement/runs/<run_tag>/`.

---

## Reproducibility / Research Integrity

- **Pinned versions.** vLLM, model revision, driver, CUDA, engine flags and the
  Virtuoso triple count are fingerprinted into every results directory.
- **Provenance on every event.** `run_id`, `dataset`, `paradigm`, `model_name`,
  `model_revision`, `git_commit`, `hardware_id`. A run missing `model_revision`
  or serving configuration is a **validation run, not a citable measurement**.
- **Missing ≠ zero.** An unavailable hardware domain is `null`. `0.0` is a
  measured value and is distinct from absence.
- **No `package` + `core` double counting.** `core` is diagnostic only.
- **Validation runs vs citable runs.** A host must pass
  `validate_hardware.py` with `measurement_complete_possible == true` before its
  runs count as measurements.
- **Common-question comparisons.** Cross-paradigm claims are paired by
  `question_id`, never compared across differing question distributions.
- **Report limitations over unsupported precision.** Operations below a
  domain's measured resolution floor are reported as unresolved, not as a
  number.

---

## Upstream Chain-of-Relations

*Chain-of-Relations: Faithful and Efficient LLM Reasoning over Knowledge Graphs via Relation-Centric Exploration*

🥳 This paper has been accepted to Findings of ACL 2026.

## Knowledge Base Setup

### Freebase

For Freebase deployment and configuration, please refer to the [DataArcTech/ToG Freebase Setup guide](https://github.com/DataArcTech/ToG/tree/main/Freebase).

### Wikidata

For Wikidata, you can either follow the [DataArcTech/ToG Wikidata Setup guide](https://github.com/DataArcTech/ToG/tree/main/Wikidata) for self-hosted deployment or use the official [Wikidata Query Service](https://www.wikidata.org/wiki/Wikidata:SPARQL_query_service).

## Chain-of-Relations

![Chain-of-Relations Method](imgs/cor_method.png)

### How to Run?

You can run Chain-of-Relations with the example script [scripts/run_cor.sh](scripts/run_cor.sh):

```bash
sh scripts/run_cor.sh
```


## How to Cite?

If you use CoR or any code from this repository in your research, please cite the following paper.

```
@inproceedings{liu2026cor,
  title={Chain-of-Relations: Faithful and Efficient LLM Reasoning over Knowledge Graphs via Relation-Centric Exploration},
  author={Liu, Chenhui and Zhou, Jianpeng and Wang, Jiahai},
  booktitle={Findings of the Association for Computational Linguistics: ACL 2026},
  year={2026}
}
```
---
# CoR Reproduction Runbook — From-Scratch local Setup

Target: fresh Ubuntu 22.04/24.04 bare-metal machine, NVIDIA GPU (24GB+), 200GB+ disk, 64GB+ RAM.
Three services: Virtuoso (Freebase), vLLM (inference), CoR harness (experiments).

> **For C2 measurement runs, the host must additionally expose readable
> CPU-package and DRAM energy domains alongside NVML GPU energy, and must pass
> [`measurement/validate_hardware.py`](measurement/validate_hardware.py) with
> `measurement_complete_possible == true`.** See
> [Current Bottleneck](#current-bottleneck). A host that fails the probe can
> still run this runbook — its runs are validation runs, not citable
> measurements. Run the probe *before* the full dry run
> ([step below](#measurement-capability-check-before-any-citable-run)).

---

## System prep (once per machine)

```bash
sudo apt update && sudo apt install -y git curl tmux unzip python3-venv python3-pip
nvidia-smi                                    # curl -fsSL https://get.docker.com | sudo sh
sudo usermod -aG docker $USER && newgrp docker
docker run --rm hello-world                   # ```

Record for provenance: `nvidia-smi --query-gpu=name,driver_version --format=csv`,
`lsb_release -a`, `nproc`, `free -g`, `df -h`.

## Freebase (Virtuoso) — one-time ~1h + 53GB download

```bash
mkdir -p ~/freebase && cd ~/freebase
wget -O virtuoso_db.zip "https://www.dropbox.com/s/q38g0fwx1a3lz8q/virtuoso_db.zip?dl=1"
# Final size ~52.5GB

unzip virtuoso_db.zip                          # do NOT use sudo (ownership lesson)
sudo chown -R $USER:$USER ~/freebase           # zip carries UID 65532; reclaim everything
chmod -R u+rw virtuoso_db
rm -f virtuoso_db/virtuoso.lck
```

Write the config (paths are the container's `/database`, buffers sized for ~64GB host —
scale NumberOfBuffers at ~8KB/buffer if RAM differs):

```bash
cat > ~/freebase/virtuoso_db/virtuoso.ini << 'EOF'
[Database]
DatabaseFile = /database/virtuoso.db
ErrorLogFile = /database/virtuoso.log
LockFile = /database/virtuoso.lck
TransactionFile = /database/virtuoso.trx
xa_persistent_file = /database/virtuoso.pxa
ErrorLogLevel = 7
FileExtend = 200
MaxCheckpointRemap = 2000
Striping = 0
TempStorage = TempDatabase

[TempDatabase]
DatabaseFile = /database/virtuoso-temp.db
TransactionFile = /database/virtuoso-temp.trx
MaxCheckpointRemap = 2000
Striping = 0

[Parameters]
ServerPort = 1111
LiteMode = 0
DisableUnixSocket = 1
DisableTcpSocket = 0
ServerThreads = 100
CheckpointInterval = 60
O_DIRECT = 1
CaseMode = 2
MaxStaticCursorRows = 100000
CheckpointAuditTrail = 0
AllowOSCalls = 0
SchedulerInterval = 10
DirsAllowed = ., /database
ThreadThreshold = 10
ResourcesCleanupInterval = 0
FreeTextBatchSize = 100000
NumberOfBuffers = 4000000
MaxDirtyBuffers = 3000000

[HTTPServer]
ServerPort = 8890
ServerThreads = 20
MaxKeepAlives = 10
DavRoot = DAV

[SPARQL]
ResultSetMaxRows = 100000
MaxQueryExecutionTime = 6000
MaxQueryCostEstimationTime = 6000
EOF
```

Launch (no `--user` flag — container entrypoint needs its own user; files were chmod'd):

```bash
docker run --name freebase-virtuoso -d \
  --publish 8890:8890 --publish 1111:1111 \
  --volume ~/freebase/virtuoso_db:/database \
  --env DBA_PASSWORD=dba \
  openlink/virtuoso-opensource-7:latest
docker update --restart unless-stopped freebase-virtuoso
docker logs -f freebase-virtuoso                           # wait for server online message
```
```bash
curl -s "http://localhost:8890/sparql" \
  --data-urlencode "query=SELECT (COUNT(*) AS ?c) WHERE { ?s ?p ?o }" \
  --data-urlencode "format=application/json" | grep -o '"value": "[0-9]*"'
# MUST return: "value": "3124793702"  — same artifact as rehearsal tier.
```

## vLLM server (Terminal 1 / tmux session `vllm`)

```bash
tmux new -s vllm
python3 -m venv ~/.venv-vllm && source ~/.venv-vllm/bin/activate
pip install --upgrade pip && pip install "vllm==0.11.1"     # pinned: ML.ENERGY lineage
pip install "flashinfer-cubin==0.5.2" || export FLASHINFER_DISABLE_VERSION_CHECK=1

vllm serve Qwen/Qwen2.5-7B-Instruct --host 0.0.0.0 --port 8000 \
  --dtype bfloat16 --max-model-len 32768 --max-num-seqs 1 \
  --gpu-memory-utilization 0.90
# --max-model-len 32768 is REQUIRED (8192 caused silent agentic degradation).
# Wait for "Application startup complete." then detach: Ctrl-B D
```

```bash
curl -s http://localhost:8000/v1/models | grep -o '"id":"[^"]*"'   # exact model name
curl -s http://localhost:8000/v1/models | grep -o '"max_model_len":[0-9]*'  # 32768
```

## Harness (Terminal 2 / tmux session `run`)

```bash
tmux new -s run
cd ~ && git clone -b energy-instrumentation https://github.com/VeiledTee/Chain-of-Relations-Emissions.git
cd Chain-of-Relations-Emissions
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# Environment — put these in ~/.bashrc as well:
export OPENAI_BASE_URL="http://localhost:8000/v1"
export OPENAI_API_KEY="dummy"                    # non-empty; value ignored
export MODEL_NAME="Qwen/Qwen2.5-7B-Instruct"     # must match served name EXACTLY
export FREEBASE_SPARQL_ENDPOINT="http://127.0.0.1:8890/sparql"
export OPENAI_TIMEOUT=300                        # 30s default causes double-billed retries
export FLASHINFER_DISABLE_VERSION_CHECK=1
```

```bash
grep -q 'done_ids' chain_of_relations/run.py && echo "resume patch: OK" || echo "resume patch: MISSING - STOP"
grep -q 'startswith("http' chain_of_relations/kg_backend/freebase_backend.py && echo "URI guard: OK" || echo "URI guard: MISSING - STOP"
```

```bash
curl -s http://localhost:8000/v1/models | grep -o '"id":"[^"]*"'
curl -s "http://localhost:8890/sparql" --data-urlencode "query=SELECT (COUNT(*) AS ?c) WHERE {?s ?p ?o}" --data-urlencode "format=application/json" | grep -o '"value": "[0-9]*"'
python3 -c "import chain_of_relations; print('harness OK')"
```

```bash
python -m chain_of_relations.run --method io_prompt --dataset webqsp --run_size 3 --save_detail true
python -m chain_of_relations.run --method cor --dataset webqsp \
  --relation_width 3 --entity_width 3 --depth 3 \
  --temperature_exploration 0.01 --temperature_reasoning 0.01 \
  --run_size 3 --save_detail true
# First lines MUST show "timeout: 300.0s". Detail JSONs must show non-(-1) tokens
# and interleaved sparql/llm steps with status "ok".
```

## Measurement capability check (before any citable run)

```bash
python measurement/validate_hardware.py --json measurement/hardware_report.json
```

Read the **MEASUREMENT BOUNDARY** block. All three additive domains
(`gpu`, `cpu_package`, `dram`) must be `SUPPORTED` and
`measurement_complete possible on this host` must be `True`. If not, this host
produces validation runs only — record the report and proceed knowing
`measured_energy_j` will be `null`.

Then characterize the instrument on this host:

```bash
python measurement/validate_measurement.py --mode idle --seconds 60
python measurement/validate_measurement.py --mode cpu  --reps 15
python measurement/validate_measurement.py --mode gpu  --seconds 20
python measurement/validate_measurement.py --mode repeated --reps 15
```

## Full dry run

The fork's id-based resume makes plain sequential commands interruption-safe:
```bash
python -m chain_of_relations.run --method cor --dataset webqsp \
  --relation_width 3 --entity_width 3 --depth 3 \
  --temperature_exploration 0.01 --temperature_reasoning 0.01 \
  --save_detail true 2>&1 | tee -a ~/cor_full_webqsp.log
python -m chain_of_relations.run --method cor --dataset cwq \
  --relation_width 3 --entity_width 3 --depth 4 \
  --temperature_exploration 0.01 --temperature_reasoning 0.01 \
  --save_detail true 2>&1 | tee -a ~/cor_full_cwq.log
```

Fallback ONLY if running unpatched upstream:
```bash
python3 - << 'EOF'
import json
for ds in ("webqsp","cwq"):
    data = json.load(open(f"datasets/{ds}/{ds}.json"))
    try:
        done = {json.loads(l)["id"] for l in open(f"results/cor/{ds}/Qwen2.5-7B-Instruct/predict.jsonl")}
    except FileNotFoundError:
        done = set()
    todo = [d["id"] for d in data if d["id"] not in done]
    open(f"todo_{ds}.txt","w").write("\n".join(todo) + "\n")   # trailing newline REQUIRED
    print(ds, "todo:", len(todo))
EOF
while read Q || [ -n "$Q" ]; do
  python -m chain_of_relations.run --method cor --dataset webqsp \
    --relation_width 3 --entity_width 3 --depth 3 \
    --temperature_exploration 0.01 --temperature_reasoning 0.01 \
    --question_id "$Q" --save_detail true 2>&1 | tee -a ~/cor_webqsp.log
done < todo_webqsp.txt
# (repeat for cwq with --depth 4)
```

## Verification & eval

```bash
# completeness: lines == unique == expected
python3 -c "
import json
for ds,total in (('webqsp',1639),('cwq',3520)):
    ids=[json.loads(l)['id'] for l in open(f'results/cor/{ds}/Qwen2.5-7B-Instruct/predict.jsonl')]
    assert len(ids)==len(set(ids))==total, (ds,len(ids),len(set(ids)))
    print(ds,'complete')
"
grep -o "maximum context length is [0-9]*" ~/cor_*.log | sort | uniq -c
grep -c "Retrying request" ~/cor_*.log
python3 audit_steps.py results
python -m chain_of_relations.eval.eval --dataset webqsp --output_dir results/cor/webqsp/Qwen2.5-7B-Instruct
python -m chain_of_relations.eval.eval --dataset cwq    --output_dir results/cor/cwq/Qwen2.5-7B-Instruct
```

**Rehearsal-tier** reference numbers (Qwen2.5-7B, 32K, this exact stack) —
engineering baselines for reproducing the harness, **not C2 results**:
WebQSP F1 62.00 / Hit 73.83, fallback 12.4%. CWQ F1 32.97 / Hit 39.20, fallback 30.0%,
6 questions overflow native 32K at depth 4. Paper (gpt-4.1-mini): 74.9 / 52.1.
11 WebQSP qids have empty gold lists (division-by-zero) — dataset rot, document.

The corresponding rehearsal energy artifacts are in
`measurement/runs/cor_{webqsp,cwq}_qwen7b/`. They predate schema v1 (events
carry the legacy `category`/`label` keys) and were captured on a host without
RAPL, so their CPU and DRAM figures are absent, not zero. See
[Current Status](#current-status) for what may and may not be concluded from
them.

## Measured runs

### Reproduction runs vs citable C2 energy experiments

The two are different activities, and the commands above are the former:

**Ordinary CoR / reproduction runs remain useful** — and are the right tool —
for functional validation, debugging, harness shakedown, and reproducing
answer-quality scores (F1 / Hit). Nothing about the measurement layer
supersedes them. Run them freely on any host.

**A run is not a citable C2 energy experiment unless it is executed through the
validated measurement workflow** and produces the required measurement and
provenance artifacts: `events.jsonl`, `power.csv`, `events_attributed.jsonl`,
`energy_summary.csv`, `trajectory_summary.csv`, and complete run provenance
(`run_id`, `dataset`, `paradigm`, `model_name`, `model_revision`, `git_commit`,
`hardware_id`).

**Final C2 runs additionally require a host that passes the hardware capability
gate** defined in [`C2_MEASUREMENT_SPEC.md`](C2_MEASUREMENT_SPEC.md) §12 —
readable NVML GPU, CPU-package and DRAM energy, with
`measurement_complete_possible == true`. A run on a host that fails that gate is
a validation run whatever else it produces.

### Producing measurement artifacts

To produce measurement artifacts rather than answers alone, wrap the harness:

```bash
python measurement/measure_run.py --tag cor_webqsp_<model> -- \
  --method cor --dataset webqsp --kb freebase \
  --relation_width 3 --depth 3 \
  --temperature_exploration 0.01 --temperature_reasoning 0.01 \
  --save_detail true
```

Set `MODEL_REVISION` alongside `MODEL_NAME` — a run without it is a validation
run, not a citable measurement. Artifacts land in `measurement/runs/<tag>/`;
see [Measurement Architecture](#measurement-architecture).

---

## Operational rules
1. **Shutdown:** Ctrl-C in a multi-command paste STARTS THE NEXT COMMAND.
   After any stop: `tmux ls` + `ps aux | grep "[c]hain_of_relations.run"` until empty.
   Two concurrent writers produced 1,909 duplicate lines.
2. **Single writer per results directory. Single config per results directory.**
3. **All accounting by unique IDs in structured artifacts** (predict.jsonl, detail JSONs).
   Never line counts (resume bug), never console-log text mining (grep produced a
   70-qid false list). Upstream resume skips by position — the id patch or the driver.
4. **Session-scoped exports die with the shell.** OPENAI_TIMEOUT, FLASHINFER_* →
   ~/.bashrc. Verify via the logged init line, not memory.
5. **Files written by todo generators need trailing newlines**, or use
   `while read Q || [ -n "$Q" ]`.
6. **Env fingerprints into every results dir:** vllm version, model revision, engine
   flags, virtuoso triple count, git commit of the fork, `param.json`.
7. **Watch strings:** "maximum context length", "Retrying request",
   "Connection error", "status": "timeout".
