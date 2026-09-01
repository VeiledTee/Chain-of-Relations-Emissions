# CLAUDE.md

## Project

Repository: `Chain-of-Relations-Emissions`  
Local path: `/home/penguins/Chain-of-Relations-Emissions`  
WSL path from Windows: `\\wsl.localhost\Ubuntu\home\penguins\Chain-of-Relations-Emissions`

This repository is an energy-instrumented fork of Chain-of-Relations (CoR) for a study on hardware-grounded energy measurement of agentic KGQA.

The immediate task is **only to formalize the CoR measurement layer**. Do not modify ToG, PoG, SubgraphRAG, or unrelated experiment logic in this slice.

---

## Research context

This study asks:

> When inference is embedded in a multi-step agentic loop over a knowledge graph, where is energy actually spent?

The measurement layer must support operation-level attribution across a CoR trajectory. A useful event must identify **what operation was being performed**, not merely that an LLM or tool was called.

The instrumentation should make it possible to distinguish operations such as:

- LLM relation ranking / relation selection
- LLM entity pruning / entity selection
- intermediate reasoning
- final answer generation
- KG identifier-to-name resolution
- KG relation search
- KG entity search
- SPARQL / graph query execution
- embedding-based pruning, if present
- orchestration / other system work where explicitly measured

Do **not** blindly use these exact labels if the CoR code has different semantic stages. Inspect the actual CoR call sites first and define the taxonomy from the algorithm implemented in this repository.

The taxonomy must be stable and explicit enough to answer the research question later.

---

## Current instrumentation architecture

The branch already contains:

- `chain_of_relations/energy_events.py`
- an instrumented LLM API wrapper
- KG-side event logging
- `measurement/measure_run.py`
- `measurement/attribute.py`
- `measurement/analyze.py`
- `tests/`
- `audit_steps.py`

Preserve the existing architecture where possible.

The desired separation is:

1. `events.jsonl` = semantic event timeline
2. sampled/counter hardware data = hardware timeline
3. attribution step = joins hardware measurements to events
4. attributed per-event artifact = canonical input for analysis
5. aggregate summaries = derived artifacts

Do not collapse these layers into a single logger.

---

## Scope of this implementation slice

### In scope

Formalize the **CoR** event schema and semantic operation taxonomy.

Specifically:

1. inspect current CoR execution and all LLM/KG instrumentation call sites;
2. define a canonical operation taxonomy in code;
3. version the event schema;
4. attach stable run/question/iteration/traversal-depth/step context;
5. replace generic `llm:generate` logging with semantic CoR operation labels;
6. make status explicit;
7. promote token counts to top-level event fields;
8. attach useful provenance metadata;
9. correct CPU package/core accounting in attribution code;
10. preserve raw per-event information in an attributed per-event artifact;
11. add tests and a small smoke/audit path.

### Out of scope

Do not:

- modify ToG instrumentation;
- modify PoG instrumentation;
- modify SubgraphRAG;
- switch models to Gemma yet;
- redesign CoR reasoning behaviour;
- optimize prompts;
- change benchmark semantics;
- change answer scoring;
- implement follow-on intervention work;
- add experimental results;
- perform a broad refactor unrelated to measurement;
- commit or push unless explicitly asked.

---

## Required raw event schema

Every formal schema-v1 event should expose these fields where applicable:

```text
schema_version
event_id

run_id
question_id
dataset
paradigm

iteration
traversal_depth
step_index

operation_type
operation_label

start_timestamp
end_timestamp
duration_s

gpu_energy_j
cpu_package_energy_j
dram_energy_j

input_tokens
output_tokens

status

model_name
model_revision
git_commit
hardware_id

meta
```

Notes:

- `schema_version` should start at `1`.
- `event_id` should be unique for the event.
- `paradigm` for this slice should resolve to CoR.
- `iteration`, `traversal_depth` and `step_index` are three distinct concepts and
  must not be conflated:
  - `iteration` = monotonically increasing CoR control-loop iteration within a
    question; advances once per processed control-loop/search state and does not
    decrease.
  - `traversal_depth` = current DFS/search depth of the state being expanded; may
    increase or decrease through backtracking, and is null for operations outside
    DFS such as `llm:direct_answer`.
  - `step_index` = monotonically increasing measured-event index within the
    question; orders measurements rather than reasoning decisions.
- A backtracking question therefore looks like:

  ```text
  iteration        0  1  2  3  4
  traversal_depth  0  1  2  1  2
  ```

  with `step_index` climbing straight through, typically several steps per
  iteration.
- unavailable measurements may be `null`; do not fabricate zeroes.
- token counts should be top-level fields, not only buried in `meta`.
- retain `meta` for operation-specific data such as attempts, query type, etc.
- preserving legacy keys for backward compatibility is acceptable temporarily, but schema-v1 analysis must use the new canonical keys.

---

## Operation taxonomy

Create a canonical taxonomy module, preferably:

```text
chain_of_relations/energy_taxonomy.py
```

Use enums/constants rather than free-form strings.

At minimum define operation types:

```text
llm
kg
embedding
system
```

Then define operation labels by **inspecting the actual CoR algorithm**.

Candidate labels may include:

```text
llm:relation_rank
llm:entity_prune
llm:reason
llm:direct_answer   # closed-book fallback when graph search produces no answer

kg:id2name
kg:relation_search
kg:entity_search
kg:sparql

embedding:prune
system:orchestration
```

However, do not invent a label for a stage that does not exist.

Rules:

1. semantic labels are assigned by the **caller that knows why an operation is happening**;
2. never infer the operation by parsing prompt text;
3. do not leave all LLM calls as `llm:generate`;
4. avoid synonymous label variants;
5. taxonomy changes after schema freeze should be deliberate and documented.

---

## Run and trajectory context

Reuse the existing `contextvars` pattern in `energy_events.py` where appropriate.

The measurement layer should have a reliable way to attach:

```text
question_id
iteration
traversal_depth
step_index
```

to every relevant event without manually threading all fields through every function call.

Run-scoped provenance should also be available:

```text
run_id
dataset
paradigm
model_name
model_revision
git_commit
hardware_id
```

Prefer configuring run-level context once from the run wrapper / environment rather than repeatedly at every event.

Do not let missing optional provenance break a run; use `null`/empty values where necessary and test the behavior.

---

## Event timing and status

Keep the current lightweight event timing mechanism unless inspection shows a correctness issue.

Each event should have:

```text
start_timestamp
end_timestamp
duration_s
status
```

Use a controlled status vocabulary, for example:

```text
ok
error
timeout
cancelled
```

Retries should normally be captured as metadata, for example:

```json
{
  "status": "ok",
  "meta": {
    "attempts": 2
  }
}
```

A terminal failure should be explicitly marked as failure.

Instrumentation must remain best-effort and must not crash the benchmark workload if event writing itself fails, unless an existing test or research-integrity constraint requires fail-fast behavior.

---

## Energy accounting rule

For the measurement boundary, the non-overlapping total is:

```text
measured_energy_j =
    gpu_energy_j
    + cpu_package_energy_j
    + dram_energy_j
```

The RAPL `core` domain is **diagnostic only** because it is contained within the CPU `package` domain.

Therefore:

- keep/log `core` if useful for diagnostics;
- never sum `package + core`;
- treat DRAM separately;
- do not silently substitute an estimate for a missing hardware counter in the raw hardware-grounded total.

Inspect `measurement/attribute.py` carefully for existing `package + core` double counting and correct it.

---

## Raw versus attributed events

The raw semantic event logger may not know CPU-package or DRAM energy at event-write time. That is acceptable.

Preferred design:

```text
events.jsonl
    -> raw semantic events, timestamps, GPU counter delta if directly available

power/counter timeline
    -> sampled/cumulative CPU/RAPL/DRAM/etc.

measurement/attribute.py
    -> joins timelines

events_attributed.jsonl
    -> enriched per-event research artifact

energy_summary.csv
    -> aggregate derived summary
```

The attributed artifact should preserve all raw event context and add fields such as:

```text
gpu_energy_j
cpu_package_energy_j
cpu_core_energy_j          # diagnostic only, if available
dram_energy_j
measured_energy_j
```

Do not discard question ID, iteration, traversal depth, step index, operation label, token counts, status, or provenance when attributing.

---

## Implementation approach

Before editing:

1. inspect `git status`;
2. inspect current branch and recent relevant history;
3. read:
   - `chain_of_relations/energy_events.py`
   - LLM wrapper/API implementation
   - CoR algorithm/control-flow files
   - KG backend instrumentation
   - `measurement/measure_run.py`
   - `measurement/attribute.py`
   - `measurement/analyze.py`
   - existing tests;
4. locate every current event call;
5. locate every CoR LLM generation call;
6. map each call site to its semantic role.

Then present a concise implementation plan before making changes.

Do not assume filenames or semantics from this document if the repository differs. Inspect first.

---

## Testing requirements

Add or update tests for at least:

### Schema

A schema-v1 event contains required keys and valid values.

### Taxonomy

- operation labels are from the canonical taxonomy;
- no CoR LLM event is emitted as a generic `llm:generate` after migration;
- operation type and label agree.

### Context

Events correctly inherit:

- question ID;
- iteration;
- traversal depth;
- step index;
- run metadata.

Tests must also demonstrate that iteration and traversal depth are genuinely
separate fields: a backtracking sequence where traversal depth decreases while
iteration continues to increase.

### LLM events

Successful and failed/retried LLM calls correctly expose:

- semantic operation label;
- input tokens;
- output tokens;
- attempts;
- status.

### Energy accounting

Verify:

```text
measured_energy_j =
gpu + package + dram
```

and that `core` is not added to the total.

### Attribution preservation

An attributed event preserves the semantic fields of the source event.

### Backward/disabled behavior

If instrumentation is disabled or no event path is configured, the normal CoR benchmark still runs.

---

## Smoke-run acceptance criteria

After tests pass, use the smallest safe CoR smoke run supported by the repo (for example `run_size=3`) and inspect the output.

The result is acceptable when:

1. the normal CoR answers/results are still generated;
2. every event has schema version and run/question context;
3. LLM events use multiple semantic labels where the CoR algorithm has multiple semantic LLM stages;
4. KG events remain semantically labeled;
5. token counts are present for LLM events when returned by the API;
6. statuses are explicit;
7. events have valid ordered timing;
8. attributed events preserve semantic context;
9. CPU total accounting does not double-count `core`;
10. existing tests plus new tests pass.

Useful audit output should show counts by operation label, for example:

```text
kg:id2name
kg:relation_search
kg:entity_search
llm:relation_rank
llm:entity_prune
llm:reason
llm:direct_answer
```

The exact labels depend on what inspection of the CoR code proves exists.

---

## Research-integrity constraints

This instrumentation will support published results. Favor correctness and traceability over cleverness.

Never:

- silently drop failed events;
- silently coerce missing energy to zero;
- mix incompatible schema versions without marking them;
- infer semantic operation labels from prompt strings;
- double-count overlapping RAPL domains;
- change CoR algorithm behavior merely to make measurement easier;
- remove existing provenance or resume safeguards;
- report a measurement field that the hardware/logger did not actually provide.

When uncertain about a measurement semantic, stop and explain the ambiguity before implementing a guess.

---

## Deliverable for this slice

At completion, provide:

1. summary of files changed;
2. final operation taxonomy and why each label maps to a real CoR call site;
3. schema-v1 example event;
4. explanation of iteration/traversal-depth/step tracking;
5. explanation of raw vs attributed energy fields;
6. confirmation that CPU package/core double counting is fixed;
7. tests added/updated and results;
8. smoke-run result if environment permits;
9. remaining limitations or questions;
10. suggested commit message.

Do not proceed to ToG/PoG or Gemma-model experiments as part of this task.
