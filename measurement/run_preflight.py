"""Measured-run preflight: provenance and GPU-exclusivity evidence.

Called by measure_run.py before the benchmark starts (and again after it ends,
for the GPU snapshot). It records, and in full mode enforces:

  1. code provenance   git commit + whether the working tree is dirty
  2. model provenance  MODEL_REVISION declared, and checked against the live
                       vLLM server's command line where that is visible
  3. serving config    parsed from the vLLM process command line (never
                       invented: an unrecoverable setting is null)
  4. GPU exclusivity   nvidia-smi device state, visible compute processes and
                       the vLLM request queue

Modes
-----
  full   (default) a citable measurement. Any policy violation aborts the run
         before the benchmark starts.
  smoke  development/plumbing runs. Violations are recorded as
         citable_blockers and the run is marked citable=false, but it runs.

What this can and cannot establish
----------------------------------
NVML measures the whole device. From inside WSL2, Windows-side GPU processes
(games, browsers, compositors) are NOT listed by nvidia-smi; only their load
shows up in device utilization and power. The defensible claim for a passing
run is therefore: "no competing GPU workload was detected by the available
preflight checks" -- not that no other process used the GPU.

Idle-utilization threshold
--------------------------
DEFAULT_MAX_IDLE_GPU_UTIL_PCT = 30. Rationale, from this project's host (RTX
4090 under WSL2, vLLM serving gemma-3-4b-it with the model resident): an idle
loaded server measured 5-7 % device utilization across several checks (the
Windows desktop shares the device), while a game running alongside drove it to
100 %. 30 % sits far from both. The median of several samples is compared, so
a single transient does not fail a run. Power is recorded for audit only: a
loaded model draws power at idle, and no calibrated idle-power threshold
exists. Override with --max-idle-gpu-util or PREFLIGHT_MAX_GPU_UTIL.
"""

import json
import os
import platform
import re
import statistics
import subprocess
import time
import urllib.request
from urllib.parse import urlparse

MODES = ("full", "smoke")
DEFAULT_MAX_IDLE_GPU_UTIL_PCT = 30.0
GPU_UTIL_SAMPLES = 5
GPU_UTIL_INTERVAL_S = 0.5

GPU_QUERY = ("index,name,uuid,utilization.gpu,power.draw,memory.used,memory.total")

#: vLLM serve flags recorded as serving configuration (value-taking).
VLLM_VALUE_FLAGS = {
	"--revision": "revision",
	"--served-model-name": "served_model_name",
	"--host": "host",
	"--port": "port",
	"--dtype": "dtype",
	"--max-model-len": "max_model_len",
	"--max-num-seqs": "max_num_seqs",
	"--gpu-memory-utilization": "gpu_memory_utilization",
	"--quantization": "quantization",
	"--tensor-parallel-size": "tensor_parallel_size",
	"--kv-cache-dtype": "kv_cache_dtype",
	"--seed": "seed",
	"--tokenizer-revision": "tokenizer_revision",
	"--max-num-batched-tokens": "max_num_batched_tokens",
}
#: Boolean vLLM flags; absent means "server default", recorded as None.
VLLM_BOOL_FLAGS = {
	"--enable-prefix-caching": ("enable_prefix_caching", True),
	"--no-enable-prefix-caching": ("enable_prefix_caching", False),
	"--enforce-eager": ("enforce_eager", True),
}
VLLM_DEFAULT_PORT = 8000

REVISION_VERIFIED = "verified_against_process_cmdline"
REVISION_DECLARED_ONLY = "declared_not_verifiable"
REVISION_MISMATCH = "mismatch"
REVISION_MISSING = "missing"


# --- small parsers (pure; unit-tested with fixtures) -----------------------

def _num(text):
	text = str(text).strip()
	if not text or text.startswith("[") or text.upper() in ("N/A", "NA"):
		return None
	try:
		return float(text)
	except ValueError:
		return None


def parse_gpu_csv(text):
	"""nvidia-smi --query-gpu=GPU_QUERY --format=csv,noheader,nounits."""
	gpus = []
	for line in (text or "").splitlines():
		parts = [p.strip() for p in line.split(",")]
		if len(parts) < 7:
			continue
		gpus.append({
			"index": int(_num(parts[0])) if _num(parts[0]) is not None else None,
			"name": parts[1],
			"uuid": parts[2],
			"utilization_gpu_pct": _num(parts[3]),
			"power_draw_w": _num(parts[4]),
			"memory_used_mib": _num(parts[5]),
			"memory_total_mib": _num(parts[6]),
		})
	return gpus


def parse_compute_apps(text):
	"""nvidia-smi --query-compute-apps=pid,process_name,used_memory (noheader,nounits)."""
	apps = []
	for line in (text or "").splitlines():
		parts = [p.strip() for p in line.split(",")]
		if len(parts) < 2 or not parts[0].isdigit():
			continue
		apps.append({"pid": int(parts[0]), "process_name": parts[1],
		             "used_memory_mib": _num(parts[2]) if len(parts) > 2 else None})
	return apps


def parse_vllm_metrics(text):
	"""Request-queue and cache gauges from a vLLM /metrics page (Prometheus text).

	Values are summed across label sets (engines/models). A metric absent from
	the page is None -- unknown, not zero.
	"""
	wanted = {
		"vllm:num_requests_running": "num_requests_running",
		"vllm:num_requests_waiting": "num_requests_waiting",
		"vllm:kv_cache_usage_perc": "kv_cache_usage_perc",
		"vllm:prefix_cache_queries_total": "prefix_cache_queries_total",
		"vllm:prefix_cache_hits_total": "prefix_cache_hits_total",
	}
	out = {v: None for v in wanted.values()}
	for line in (text or "").splitlines():
		if not line or line.startswith("#"):
			continue
		m = re.match(r"^([a-zA-Z_:][\w:]*)(\{[^}]*\})?\s+(\S+)", line)
		if not m or m.group(1) not in wanted:
			continue
		value = _num(m.group(3))
		if value is None:
			continue
		key = wanted[m.group(1)]
		out[key] = (out[key] or 0.0) + value
	return out


def parse_vllm_argv(argv):
	"""Serving configuration from a `vllm serve ...` argv. Unset flags -> None."""
	cfg = {name: None for name in VLLM_VALUE_FLAGS.values()}
	cfg.update({name: None for name, _ in VLLM_BOOL_FLAGS.values()})
	cfg["model"] = None
	args = list(argv)
	if "serve" in args:
		rest = args[args.index("serve") + 1:]
		if rest and not rest[0].startswith("-"):
			cfg["model"] = rest[0]
	i = 0
	while i < len(args):
		token = args[i]
		flag, _, inline = token.partition("=")
		if flag in VLLM_VALUE_FLAGS:
			value = inline if inline else (args[i + 1] if i + 1 < len(args) else None)
			cfg[VLLM_VALUE_FLAGS[flag]] = value
			i += 1 if inline else 2
			continue
		if flag in VLLM_BOOL_FLAGS:
			name, value = VLLM_BOOL_FLAGS[flag]
			cfg[name] = value
		if flag == "--model" and not cfg["model"]:
			cfg["model"] = inline or (args[i + 1] if i + 1 < len(args) else None)
		i += 1
	return cfg


def endpoint_port(base_url):
	parsed = urlparse(base_url or "")
	if parsed.port:
		return parsed.port
	return VLLM_DEFAULT_PORT if parsed.scheme in ("http", "") else 443


def verify_revision(declared, server_cfg):
	"""(status, observed). Only an exact match against the live command line
	counts as verified; anything less is reported as declared-only."""
	if not declared:
		return REVISION_MISSING, (server_cfg or {}).get("revision")
	observed = (server_cfg or {}).get("revision")
	if observed is None:
		return REVISION_DECLARED_ONLY, None
	return (REVISION_VERIFIED if observed == declared else REVISION_MISMATCH), observed


# --- live probes (best-effort; failures are recorded, never raised) --------

def _run(cmd, timeout=15):
	try:
		out = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
		return out.returncode, out.stdout
	except Exception as exc:  # noqa: BLE001 - absence of a tool is data
		return None, f"{type(exc).__name__}: {exc}"


def _http_get(url, timeout=5):
	try:
		with urllib.request.urlopen(url, timeout=timeout) as resp:
			return resp.read().decode("utf-8", "replace")
	except Exception:
		return None


def _proc_cmdline(pid, proc_root="/proc"):
	try:
		with open(os.path.join(proc_root, str(pid), "cmdline"), "rb") as f:
			return [a.decode("utf-8", "replace") for a in f.read().split(b"\0") if a]
	except OSError:
		return None


def _proc_comm(pid, proc_root="/proc"):
	try:
		with open(os.path.join(proc_root, str(pid), "comm")) as f:
			return f.read().strip()
	except OSError:
		return None


def _proc_ppid(pid, proc_root="/proc"):
	try:
		with open(os.path.join(proc_root, str(pid), "stat")) as f:
			return int(f.read().rsplit(")", 1)[1].split()[1])
	except (OSError, ValueError, IndexError):
		return None


def _looks_vllm(pid, proc_root="/proc"):
	text = " ".join(_proc_cmdline(pid, proc_root) or []) + " " + (_proc_comm(pid, proc_root) or "")
	return "vllm" in text.lower()


def classify_process(pid, proc_root="/proc"):
	"""'vllm' (server or a descendant), 'other', or 'unidentified' (not in /proc)."""
	if _proc_cmdline(pid, proc_root) is None and _proc_comm(pid, proc_root) is None:
		return "unidentified"
	seen, current = set(), pid
	while current and current not in seen and current > 1:
		seen.add(current)
		if _looks_vllm(current, proc_root):
			return "vllm"
		current = _proc_ppid(current, proc_root)
	return "other"


def find_vllm_servers(proc_root="/proc"):
	"""[{pid, argv, config}] for every visible `vllm serve` process."""
	servers = []
	try:
		pids = [int(n) for n in os.listdir(proc_root) if n.isdigit()]
	except OSError:
		return servers
	for pid in pids:
		argv = _proc_cmdline(pid, proc_root)
		if not argv:
			continue
		cli = ("serve" in argv and any(os.path.basename(a) == "vllm"
		                               for a in argv[:argv.index("serve")]))
		module = any(a.startswith("vllm.entrypoints") for a in argv)
		if cli or module:
			servers.append({"pid": pid, "argv": argv, "config": parse_vllm_argv(argv)})
	return servers


def select_server(servers, base_url):
	"""The server whose port matches the endpoint; None if absent or ambiguous."""
	port = endpoint_port(base_url)
	matches = [s for s in servers
	           if int(s["config"].get("port") or VLLM_DEFAULT_PORT) == port]
	return matches[0] if len(matches) == 1 else None


def gpu_snapshot(base_url, samples=GPU_UTIL_SAMPLES, interval_s=GPU_UTIL_INTERVAL_S,
                 proc_root="/proc", runner=_run, http_get=_http_get):
	"""Device state, visible compute processes and the vLLM queue, now."""
	snap = {"timestamp": time.time(), "nvidia_smi_available": True,
	        "gpu_samples": [], "gpus": [], "compute_apps": [],
	        "vllm_metrics": None, "note": (
	            "Device-level NVML view. Under WSL2, Windows-side GPU processes are "
	            "not listed; their load appears only in utilization/power.")}
	for k in range(max(1, samples)):
		rc, out = runner(["nvidia-smi", f"--query-gpu={GPU_QUERY}",
		                  "--format=csv,noheader,nounits"])
		if rc != 0:
			snap["nvidia_smi_available"] = False
			snap["nvidia_smi_error"] = out
			break
		snap["gpu_samples"].append(parse_gpu_csv(out))
		if k + 1 < samples:
			time.sleep(interval_s)
	if snap["gpu_samples"]:
		snap["gpus"] = snap["gpu_samples"][-1]
		per_gpu = {}
		for sample in snap["gpu_samples"]:
			for g in sample:
				if g["utilization_gpu_pct"] is not None:
					per_gpu.setdefault(g["index"], []).append(g["utilization_gpu_pct"])
		snap["utilization_median_pct"] = {str(i): statistics.median(v) for i, v in per_gpu.items()}
		rc, out = runner(["nvidia-smi", "--query-compute-apps=pid,process_name,used_memory",
		                  "--format=csv,noheader,nounits"])
		if rc == 0:
			for app in parse_compute_apps(out):
				app["classification"] = classify_process(app["pid"], proc_root)
				app["cmdline"] = " ".join(_proc_cmdline(app["pid"], proc_root) or [])[:300] or None
				snap["compute_apps"].append(app)
	root = (base_url or "").rstrip("/")
	root = root[:-3] if root.endswith("/v1") else root
	text = http_get(root + "/metrics") if root else None
	snap["vllm_metrics"] = parse_vllm_metrics(text) if text else None
	return snap


def served_models(base_url, http_get=_http_get):
	text = http_get((base_url or "").rstrip("/") + "/models") if base_url else None
	if not text:
		return None
	try:
		data = json.loads(text).get("data", [])
	except (ValueError, AttributeError):
		return None
	return [{"id": m.get("id"), "root": m.get("root"),
	         "max_model_len": m.get("max_model_len")} for m in data]


# --- policy ----------------------------------------------------------------

def evaluate(record, max_util=DEFAULT_MAX_IDLE_GPU_UTIL_PCT):
	"""Policy violations for a preflight record. Same checks in both modes;
	full mode aborts on any, smoke mode records them as citable_blockers."""
	problems = []
	code = record["code"]
	if code["git_dirty"] is True:
		problems.append("working tree is dirty (uncommitted or untracked changes): "
		                "the recorded commit does not identify the code that would run")
	elif code["git_dirty"] is None:
		problems.append("git working-tree state could not be determined")
	if not code["git_commit"]:
		problems.append("git commit could not be determined")

	model = record["model"]
	if not model["declared_revision"]:
		problems.append("MODEL_REVISION is empty: the served weights are not identified")
	if model["revision_status"] == REVISION_MISMATCH:
		problems.append(f"MODEL_REVISION {model['declared_revision']} does not match the "
		                f"live vLLM server's --revision {model['observed_revision']}")
	served = model["served_models"]
	if served is None:
		problems.append(f"model endpoint {record['serving']['endpoint']} did not answer /models")
	elif model["name"] not in [m["id"] for m in served]:
		problems.append(f"endpoint does not serve MODEL_NAME={model['name']!r} "
		                f"(serves {[m['id'] for m in served]})")

	snap = record["gpu_preflight"]["before"]
	metrics = snap.get("vllm_metrics")
	if metrics:
		for key in ("num_requests_running", "num_requests_waiting"):
			if metrics.get(key):
				problems.append(f"vLLM has {metrics[key]:g} {key.split('_')[-1]} request(s) "
				                f"before the run: another client is using the server")
	for index, util in (snap.get("utilization_median_pct") or {}).items():
		if util > max_util:
			problems.append(f"GPU {index} median utilization {util:g}% exceeds the idle "
			                f"threshold {max_util:g}%: a competing workload is likely active")
	for app in snap.get("compute_apps") or []:
		if app["classification"] == "other":
			problems.append(f"visible non-vLLM GPU compute process pid={app['pid']} "
			                f"({app.get('cmdline') or app['process_name']})")
	return problems


def warnings_for(record):
	"""Limitations worth recording that do not block a run."""
	notes = []
	snap = record["gpu_preflight"]["before"]
	if not snap.get("nvidia_smi_available"):
		notes.append("nvidia-smi unavailable: GPU exclusivity could not be checked")
	if snap.get("vllm_metrics") is None:
		notes.append("vLLM /metrics unavailable: request queue could not be checked")
	for app in snap.get("compute_apps") or []:
		if app["classification"] == "unidentified":
			notes.append(f"GPU compute process pid={app['pid']} is not visible in /proc "
			             f"(other namespace or host); it could not be classified")
	if record["model"]["revision_status"] == REVISION_DECLARED_ONLY:
		notes.append("model revision is declared (MODEL_REVISION) but could not be "
		             "verified: no visible vLLM process command line carries --revision")
	if record["serving"]["config"] is None:
		notes.append("vLLM command line not visible: serving configuration is unrecorded (null)")
	return notes


def collect(base_url, model_name, model_revision, repo_root, git_commit_fn, git_dirty_fn,
            proc_root="/proc", runner=_run, http_get=_http_get, samples=GPU_UTIL_SAMPLES):
	"""Build the pre-run provenance record (no policy applied)."""
	servers = find_vllm_servers(proc_root)
	server = select_server(servers, base_url)
	status, observed = verify_revision(model_revision, server["config"] if server else None)
	snap = gpu_snapshot(base_url, samples=samples, proc_root=proc_root,
	                    runner=runner, http_get=http_get)
	gpu_names = ",".join(g["name"] for g in snap.get("gpus") or [])
	return {
		"code": {"git_commit": git_commit_fn() or None, "git_dirty": git_dirty_fn(),
		         "repo_root": repo_root},
		"model": {"name": model_name, "declared_revision": model_revision or None,
		          "observed_revision": observed, "revision_status": status,
		          "served_models": served_models(base_url, http_get=http_get)},
		"serving": {"endpoint": base_url, "server_pid": server["pid"] if server else None,
		            "command_line": server["argv"] if server else None,
		            "config": server["config"] if server else None,
		            "visible_vllm_servers": len(servers)},
		"hardware": {"host": platform.node() or None, "gpus": gpu_names or None,
		             "hardware_id": f"{platform.node()}|{gpu_names}" if gpu_names else None},
		"gpu_preflight": {"before": snap},
	}


def write_json(path, obj):
	tmp = path + ".tmp"
	with open(tmp, "w") as f:
		json.dump(obj, f, indent=2, default=str)
	os.replace(tmp, path)
