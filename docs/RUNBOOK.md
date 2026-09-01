# Runbook — Running and Verifying Experiments

Operational procedures for running CoR, verifying output completeness, and
producing measurement artifacts. Assumes the environment from
[SETUP.md](SETUP.md) is up.

The authoritative measurement protocol — boundary, schema v1, taxonomy,
missing-domain policy, trajectory accounting, host capability requirement — is
[../MEASUREMENT_SPEC.md](../MEASUREMENT_SPEC.md). This document is how to
operate the harness, not what the measurements mean.

Related: [../README.md](../README.md) · [SETUP.md](SETUP.md) ·
[ROADMAP.md](ROADMAP.md).

---

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
engineering baselines for reproducing the harness, **not study results**:
WebQSP F1 62.00 / Hit 73.83, fallback 12.4%. CWQ F1 32.97 / Hit 39.20, fallback 30.0%,
6 questions overflow native 32K at depth 4. Paper (gpt-4.1-mini): 74.9 / 52.1.
11 WebQSP qids have empty gold lists (division-by-zero) — dataset rot, document.

The corresponding rehearsal energy artifacts are in
`measurement/runs/cor_{webqsp,cwq}_qwen7b/`. They predate schema v1 (events
carry the legacy `category`/`label` keys) and were captured on a host without
RAPL, so their CPU and DRAM figures are absent, not zero. See
[ROADMAP.md](ROADMAP.md) for what may and may not be concluded from
them.

## Measured runs

### Reproduction runs vs citable energy experiments

The two are different activities, and the commands above are the former:

**Ordinary CoR / reproduction runs remain useful** — and are the right tool —
for functional validation, debugging, harness shakedown, and reproducing
answer-quality scores (F1 / Hit). Nothing about the measurement layer
supersedes them. Run them freely on any host.

**A run is not a citable energy experiment unless it is executed through the
validated measurement workflow** and produces the required measurement and
provenance artifacts: `events.jsonl`, `power.csv`, `events_attributed.jsonl`,
`energy_summary.csv`, `trajectory_summary.csv`, and complete run provenance
(`run_id`, `dataset`, `paradigm`, `model_name`, `model_revision`, `git_commit`,
`hardware_id`).

**Final measured runs additionally require a host that passes the hardware capability
gate** defined in [`MEASUREMENT_SPEC.md`](../MEASUREMENT_SPEC.md) §12 —
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
see [Profiler outputs](../README.md#profiler-outputs).

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

---

## Run directory naming

`measurement/runs/<tag>/` holds every measured run ever produced, across all
three result tiers. The tag is the only thing distinguishing them, so read it
before quoting any number:

| Tag pattern | Tier | Citable as a study finding? |
|---|---|---|
| `cor_{webqsp,cwq}_qwen7b`, `cot_*`, `io_*` | Rehearsal — Qwen2.5-7B engineering shakedown, pre-schema-v1 (legacy `category`/`label` keys), captured without RAPL | No |
| `gemma4b_val_*`, `gemma4b_v2_*`, `gemma4b_v3_*`, `gemma4b_v4_*` | Profiler / integration validation — real `google/gemma-3-4b-it` inference on the WSL2 development host, `run_size 3`, `measurement_complete == false` | No |
| (not yet produced) | Final experiments — Gemma 3 matrix on a bare-metal host that passes the capability probe | Yes, once produced |

Anything whose `hardware_id` names the development host is a validation run by
construction: CPU-package and DRAM are unavailable there, so
`measured_energy_j` is `null` and the measurement boundary is incomplete.

---

## Profiler validation protocol

Run before the first citable measurement on a new host, and again after any
change to the profiler or to a paradigm's instrumentation. This validates the
**instrument and its integration**; it produces no study findings.

Fixed inputs, held constant across the whole protocol:

- a fixed 25-question WebQSP subset;
- a fixed 25-question CWQ subset;
- the **same questions** for CoR, ToG and PoG;
- the same profiler version and measurement configuration;
- a standardized warm-up before any measured window.

Then verify:

1. **Schema** — every event is schema v1 with the required fields populated.
2. **Semantic labels** — every event carries a canonical taxonomy label; no
   `llm:generate` survives, and no label is inferred from prompt text.
3. **Provenance** — `run_id`, `dataset`, `paradigm`, `model_name`,
   `model_revision`, `git_commit`, `hardware_id` present on every event.
4. **Hardware domains** — the capability probe's per-domain classification
   matches what the attributed events actually contain.
5. **Overlap** — no overlapping measured spans within a trajectory.
6. **Trajectory reconciliation** — per-question operation energy reconciles
   against the cumulative counter window.
7. **Residuals** — `unattributed_energy_j` and `attribution_coverage` are
   reported and never forced to 1.0.
8. **Aggregation** — rollups reproduce the per-event totals they summarize.
9. **Measurement completeness** — `measurement_complete` and
   `available_energy_domains` agree with the host's real capability.

Finally, **rerun a small fixed subset of questions** to characterize
run-to-run and trajectory-level variability on that host.

None of the above is an experimental result. These runs establish that the
instrument measures what it claims to measure on this host.

---

## Reproducibility / research integrity checklist

- **Pinned versions.** vLLM, model revision, driver, CUDA, engine flags and the
  Virtuoso triple count are fingerprinted into every results directory.
- **Provenance on every event.** `run_id`, `dataset`, `paradigm`, `model_name`,
  `model_revision`, `git_commit`, `hardware_id`. A run missing `model_revision`
  or serving configuration is a **validation run, not a citable measurement**.
- **Missing ≠ zero.** An unavailable hardware domain is `null`. `0.0` is a
  measured value and is distinct from absence.
- **No `package` + `core` double counting.** `core` is diagnostic only.
- **Validation runs vs citable runs.** A host must pass `validate_hardware.py`
  with `measurement_complete_possible == true` before its runs count as
  measurements.
- **Common-question comparisons.** Cross-paradigm claims are paired by
  `question_id`, never compared across differing question distributions.
- **Report limitations over unsupported precision.** Operations below a
  domain's measured resolution floor are reported as unresolved, not as a
  number.
