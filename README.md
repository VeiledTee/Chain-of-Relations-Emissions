# Chain-of-Relations

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

Rehearsal-tier reference numbers (Qwen2.5-7B, 32K, this exact stack):
WebQSP F1 62.00 / Hit 73.83, fallback 12.4%. CWQ F1 32.97 / Hit 39.20, fallback 30.0%,
6 questions overflow native 32K at depth 4. Paper (gpt-4.1-mini): 74.9 / 52.1.
11 WebQSP qids have empty gold lists (division-by-zero) — dataset rot, document.

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
