# Setup — From-Scratch Local Environment

Environment build for this repository: Virtuoso (Freebase), vLLM (inference)
and the CoR harness. Target: fresh Ubuntu 22.04/24.04, NVIDIA GPU (24GB+),
200GB+ disk, 64GB+ RAM.

This document builds a **working** host. It does not make a host **eligible for
citable measurements** — that gate is the hardware capability probe, covered
in [RUNBOOK.md](RUNBOOK.md) and specified in
[../MEASUREMENT_SPEC.md](../MEASUREMENT_SPEC.md) §12. A host that fails
the probe can still run everything below; its runs are validation runs.

Related: [../README.md](../README.md) (overview) ·
[RUNBOOK.md](RUNBOOK.md) (running experiments) ·
[ROADMAP.md](ROADMAP.md) (project status and history).

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

---

## Profiler install

The measurement instrument is the standalone `agent_energy_profiler` package.
It has no required runtime dependencies; GPU energy needs the optional extra.

```bash
pip install -e ".[gpu]"     # installs pynvml; without it GPU energy is
                            # reported unavailable, never estimated
```

Console entry points installed alongside it:

| Command | Equivalent module |
|---|---|
| `aep-validate` | `python -m agent_energy_profiler.validation` |
| `aep-sample` | `python -m agent_energy_profiler.sampling` |
| `aep-attribute` | `python -m agent_energy_profiler.attribution` |
| `aep-aggregate` | `python -m agent_energy_profiler.aggregate` |

The `measurement/*.py` paths documented elsewhere are thin compatibility
shims over these modules, not separate implementations.

Next: [RUNBOOK.md](RUNBOOK.md).
