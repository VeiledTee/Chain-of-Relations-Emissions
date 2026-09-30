"""Profiler hardening before the paired runs: provenance, clock safety, GPU
preflight, model revision, and a CPU/DRAM regression guard.

Every probe of the outside world (git, /proc, nvidia-smi, the vLLM HTTP
endpoints) is replaced by a fixture, so these tests never touch the real GPU,
server or repository state.
"""

import argparse
import contextlib
import io
import json
import os
import shutil
import subprocess
import sys
import tempfile
import unittest
from unittest import mock

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "measurement"))

import measure_run  # noqa: E402
import run_preflight as rp  # noqa: E402
from agent_energy_profiler import attribution, trajectory  # noqa: E402
from agent_energy_profiler import events as profiler_events  # noqa: E402
from chain_of_relations import run as run_module  # noqa: E402

REVISION = "093f9f388b31de276ce2de164bdc2081324b9767"
GPU_IDLE = "0, NVIDIA GeForce RTX 4090, GPU-24a1, 6, 28.75, 23565, 24564\n"
GPU_BUSY = "0, NVIDIA GeForce RTX 4090, GPU-24a1, 100, 310.2, 23565, 24564\n"
APPS_VLLM_ONLY = "101, [Not Found], [N/A]\n"
METRICS_IDLE = (
	'# HELP vllm:num_requests_running Number of requests in model execution batches.\n'
	'vllm:num_requests_running{engine="0",model_name="google/gemma-3-4b-it"} 0.0\n'
	'vllm:num_requests_waiting{engine="0",model_name="google/gemma-3-4b-it"} 0.0\n'
	'vllm:kv_cache_usage_perc{engine="0",model_name="google/gemma-3-4b-it"} 0.0\n'
	'vllm:prefix_cache_queries_total{engine="0",model_name="google/gemma-3-4b-it"} 123362.0\n'
	'vllm:prefix_cache_hits_total{engine="0",model_name="google/gemma-3-4b-it"} 61216.0\n')
MODELS = json.dumps({"object": "list", "data": [
	{"id": "google/gemma-3-4b-it", "root": "google/gemma-3-4b-it", "max_model_len": 32768}]})
VLLM_ARGV = ["/venv/bin/python3", "/venv/bin/vllm", "serve", "google/gemma-3-4b-it",
             "--revision", REVISION, "--served-model-name", "google/gemma-3-4b-it",
             "--host", "0.0.0.0", "--port", "8000", "--dtype", "bfloat16",
             "--max-model-len", "32768", "--max-num-seqs", "1",
             "--gpu-memory-utilization", "0.90"]


def make_proc(root, pid, argv, comm, ppid):
	d = os.path.join(root, str(pid))
	os.makedirs(d)
	with open(os.path.join(d, "cmdline"), "wb") as f:
		f.write(b"\0".join(a.encode() for a in argv) + b"\0")
	with open(os.path.join(d, "comm"), "w") as f:
		f.write(comm + "\n")
	with open(os.path.join(d, "stat"), "w") as f:
		f.write(f"{pid} ({comm}) S {ppid} 1 1 0 -1\n")


def fake_proc(root, argv=VLLM_ARGV, extra=()):
	make_proc(root, 100, argv, "vllm", 1)
	make_proc(root, 101, ["VLLM::EngineCore"], "VLLM::EngineCor", 100)
	make_proc(root, 102, ["/venv/bin/python3", "-c", "from multiprocessing import x"], "python3", 100)
	for pid, pargv, comm in extra:
		make_proc(root, pid, pargv, comm, 1)


def runner_for(gpu_csv=GPU_IDLE, apps=APPS_VLLM_ONLY):
	def run(cmd, timeout=15):
		if any(c.startswith("--query-gpu") for c in cmd):
			return 0, gpu_csv
		if any(c.startswith("--query-compute-apps") for c in cmd):
			return 0, apps
		return 1, "unexpected"
	return run


def http_for(metrics=METRICS_IDLE, models=MODELS):
	def get(url, timeout=5):
		if url.endswith("/metrics"):
			return metrics
		if url.endswith("/models"):
			return models
		return None
	return get


def collect(proc_root, revision=REVISION, dirty=False, gpu_csv=GPU_IDLE, apps=APPS_VLLM_ONLY,
            metrics=METRICS_IDLE, models=MODELS, commit="abc123"):
	return rp.collect(
		base_url="http://localhost:8000/v1", model_name="google/gemma-3-4b-it",
		model_revision=revision, repo_root="/repo",
		git_commit_fn=lambda: commit, git_dirty_fn=lambda: dirty,
		proc_root=proc_root, runner=runner_for(gpu_csv, apps),
		http_get=http_for(metrics, models), samples=1)


class TempDirCase(unittest.TestCase):
	def setUp(self):
		self.tmp = tempfile.mkdtemp()
		self.proc = os.path.join(self.tmp, "proc")
		os.makedirs(self.proc)

	def tearDown(self):
		shutil.rmtree(self.tmp, ignore_errors=True)


# ==========================================================================
# 1. code provenance
# ==========================================================================

class GitProvenance(TempDirCase):

	def git(self, *args):
		subprocess.run(["git", "-c", "user.name=t", "-c", "user.email=t@t", *args],
		               cwd=self.repo, check=True, capture_output=True)

	def setUp(self):
		super().setUp()
		self.repo = os.path.join(self.tmp, "repo")
		os.makedirs(self.repo)
		self.git("init", "-q")
		with open(os.path.join(self.repo, "a.txt"), "w") as f:
			f.write("a\n")
		self.git("add", "a.txt")
		self.git("commit", "-q", "-m", "init")

	def test_clean_tree_is_not_dirty(self):
		self.assertIs(profiler_events.detect_git_dirty(self.repo), False)
		self.assertTrue(profiler_events.detect_git_commit(self.repo))

	def test_modified_tracked_file_is_dirty(self):
		with open(os.path.join(self.repo, "a.txt"), "a") as f:
			f.write("b\n")
		self.assertIs(profiler_events.detect_git_dirty(self.repo), True)

	def test_untracked_file_is_dirty(self):
		open(os.path.join(self.repo, "new.py"), "w").close()
		self.assertIs(profiler_events.detect_git_dirty(self.repo), True)

	def test_ignored_file_is_not_dirty(self):
		with open(os.path.join(self.repo, ".gitignore"), "w") as f:
			f.write("results/\n")
		self.git("add", ".gitignore")
		self.git("commit", "-q", "-m", "ignore")
		os.makedirs(os.path.join(self.repo, "results"))
		open(os.path.join(self.repo, "results", "x.json"), "w").close()
		self.assertIs(profiler_events.detect_git_dirty(self.repo), False)

	def test_not_a_repository_is_unknown_not_clean(self):
		self.assertIsNone(profiler_events.detect_git_dirty(self.tmp + "/proc"))


class CleanTreePolicy(TempDirCase):

	def setUp(self):
		super().setUp()
		fake_proc(self.proc)

	def test_clean_full_run_is_citable(self):
		record = collect(self.proc)
		self.assertEqual(measure_run.apply_policy(record, "full", 30.0), [])
		self.assertTrue(record["citable"])
		self.assertEqual(record["citable_blockers"], [])

	def test_full_run_rejects_dirty_tree(self):
		record = collect(self.proc, dirty=True)
		problems = measure_run.apply_policy(record, "full", 30.0)
		self.assertTrue(any("dirty" in p for p in problems))
		self.assertFalse(record["citable"])

	def test_unknown_tree_state_is_not_treated_as_clean(self):
		record = collect(self.proc, dirty=None)
		self.assertTrue(any("could not be determined" in p
		                    for p in measure_run.apply_policy(record, "full", 30.0)))

	def test_smoke_mode_allows_dirty_but_marks_non_citable(self):
		record = collect(self.proc, dirty=True)
		problems = measure_run.apply_policy(record, "smoke", 30.0)
		self.assertTrue(problems)
		self.assertFalse(record["citable"])
		self.assertEqual(record["mode"], "smoke")
		self.assertEqual(record["citable_blockers"][0], "smoke/development mode")
		self.assertTrue(any("dirty" in b for b in record["citable_blockers"]))

	def test_clean_smoke_run_is_still_not_citable(self):
		record = collect(self.proc)
		measure_run.apply_policy(record, "smoke", 30.0)
		self.assertFalse(record["citable"])
		self.assertEqual(record["citable_blockers"], ["smoke/development mode"])


class MeasureRunGate(TempDirCase):
	"""main(): a failing full-mode preflight stops before anything starts;
	a passing one persists run_provenance.json before the benchmark."""

	def setUp(self):
		super().setUp()
		fake_proc(self.proc)
		os.makedirs(os.path.join(self.tmp, "runs"))

	def run_main(self, record, mode="full"):
		argv = ["measure_run.py", "--tag", "t1", "--mode", mode, "--", "--method", "cor"]
		launched = mock.Mock(side_effect=RuntimeError("benchmark would start here"))
		err = io.StringIO()
		with mock.patch.object(measure_run, "HERE", self.tmp), \
		     mock.patch.object(measure_run.run_preflight, "collect", return_value=record), \
		     mock.patch.object(measure_run.subprocess, "Popen", launched), \
		     mock.patch.object(measure_run.subprocess, "call", launched), \
		     mock.patch.object(sys, "argv", argv), contextlib.redirect_stderr(err):
			try:
				measure_run.main()
				code = 0
			except SystemExit as e:
				code = e.code
			except RuntimeError:
				code = "launched"
		return code, launched, err.getvalue()

	def test_dirty_full_run_exits_before_run_dir_sampler_or_benchmark(self):
		code, launched, err = self.run_main(collect(self.proc, dirty=True))
		self.assertEqual(code, 3)
		launched.assert_not_called()
		self.assertFalse(os.path.exists(os.path.join(self.tmp, "runs", "t1")))
		self.assertIn("benchmark was NOT started", err)

	def test_passing_preflight_persists_provenance_before_launch(self):
		code, launched, _ = self.run_main(collect(self.proc))
		self.assertEqual(code, "launched")
		path = os.path.join(self.tmp, "runs", "t1", "run_provenance.json")
		with open(path) as f:
			saved = json.load(f)
		self.assertEqual(saved["schema"], "run_provenance/1")
		self.assertTrue(saved["citable"])
		self.assertEqual(saved["code"], {"git_commit": "abc123", "git_dirty": False, "repo_root": "/repo"})
		self.assertEqual(saved["model"]["revision_status"], rp.REVISION_VERIFIED)
		self.assertEqual(saved["serving"]["config"]["max_num_seqs"], "1")
		self.assertIn("before", saved["gpu_preflight"])
		self.assertEqual(saved["run_args"], ["--method", "cor"])

	def test_smoke_dirty_run_proceeds_and_records_non_citable(self):
		code, launched, err = self.run_main(collect(self.proc, dirty=True), mode="smoke")
		self.assertEqual(code, "launched")
		with open(os.path.join(self.tmp, "runs", "t1", "run_provenance.json")) as f:
			saved = json.load(f)
		self.assertFalse(saved["citable"])
		self.assertIn("NOT citable", err)

	def test_provenance_file_blocks_tag_reuse(self):
		self.assertIn("run_provenance.json", measure_run.RUN_ARTIFACTS)


# ==========================================================================
# 2. clock safety
# ==========================================================================

class ClockSafety(unittest.TestCase):
	TS = [0.0, 1.0, 2.0, 3.0, 4.0]
	W = [100.0, 100.0, 100.0, 100.0, 100.0]

	def event(self, t0, t1, gpu=None, q="q1", label="llm:reason"):
		return {"question_id": q, "operation_label": label, "operation_type": label.split(":")[0],
		        "start_timestamp": t0, "end_timestamp": t1, "duration_s": t1 - t0,
		        "gpu_energy_j": gpu, "status": "ok"}

	def test_reversed_and_empty_windows_integrate_to_null_not_zero(self):
		self.assertIsNone(attribution.integrate_gpu(self.TS, self.W, 2.5, 1.5))
		self.assertIsNone(attribution.integrate_gpu(self.TS, self.W, 2.0, 2.0))
		self.assertIsNone(attribution.integrate_gpu([], [], 1.0, 2.0))
		self.assertIsNone(attribution.integrate_gpu(self.TS, [None] * 5, 1.0, 2.0))

	def test_normal_window_integrates_as_before(self):
		self.assertAlmostEqual(attribution.integrate_gpu(self.TS, self.W, 0.5, 3.5), 300.0)
		# a missing sample is skipped, not read as 0 W
		self.assertAlmostEqual(
			attribution.integrate_gpu(self.TS, [100.0, None, 100.0, 100.0, 100.0], 0.5, 3.5), 300.0)

	def test_reversed_event_without_counter_is_null_and_flagged(self):
		rows, src = attribution.attribute_events([self.event(2.5, 1.5)], self.TS, {"gpu0_w": self.W}, {})
		self.assertIsNone(rows[0]["gpu_energy_j"])
		self.assertIsNone(rows[0]["gpu_energy_source"])
		self.assertTrue(rows[0]["clock_anomaly"])
		self.assertEqual(src["unavailable"], 1)

	def test_reversed_event_keeps_valid_counter_energy(self):
		rows, _ = attribution.attribute_events([self.event(2.5, 1.5, gpu=42.0)], self.TS,
		                                       {"gpu0_w": self.W}, {})
		self.assertEqual(rows[0]["gpu_energy_j"], 42.0)
		self.assertEqual(rows[0]["gpu_energy_source"], "nvml_counter")
		self.assertTrue(rows[0]["clock_anomaly"])

	def test_normal_event_behaviour_unchanged(self):
		rows, src = attribution.attribute_events([self.event(0.5, 3.5)], self.TS, {"gpu0_w": self.W}, {})
		self.assertAlmostEqual(rows[0]["gpu_energy_j"], 300.0)
		self.assertEqual(rows[0]["gpu_energy_source"], "power_integration")
		self.assertFalse(rows[0]["clock_anomaly"])
		self.assertEqual(src, {"counter": 0, "integrated": 1, "unavailable": 0})

	def test_backward_power_steps_detected_not_repaired(self):
		ts = [0.0, 1.0, 0.6, 2.0]
		self.assertEqual(attribution.backward_steps(ts), [(1, 1.0, 0.6)])
		self.assertEqual(ts, [0.0, 1.0, 0.6, 2.0])
		self.assertEqual(attribution.backward_steps(self.TS), [])

	def test_trajectory_flags_backstep_inside_window_only(self):
		events = [self.event(0.0, 1.0, gpu=1.0, q="a"), self.event(5.0, 6.0, gpu=1.0, q="b")]
		rows = {r["question_id"]: r for r in trajectory.accumulate(events, clock_backsteps=[(0.8, 0.5)])}
		self.assertTrue(rows["a"]["clock_anomaly"])
		self.assertEqual(rows["a"]["power_backsteps_in_window"], 1)
		self.assertFalse(rows["b"]["clock_anomaly"])
		summary = trajectory.summarize(list(rows.values()), power_backward_steps=1)
		self.assertTrue(summary["clock_anomaly"])
		self.assertEqual(summary["n_clock_anomaly_trajectories"], 1)

	def test_reversed_event_is_not_reported_as_overlap(self):
		events = [self.event(0.0, 2.0, gpu=1.0), self.event(1.5, 1.2, gpu=1.0)]
		row, = trajectory.accumulate(events)
		self.assertFalse(row["events_overlap"])
		self.assertEqual(row["max_concurrency"], 1)
		self.assertEqual(row["n_reversed_events"], 1)
		self.assertTrue(row["clock_anomaly"])
		self.assertTrue(row["coverage_valid"] is False)  # no independent window here

	def test_real_overlap_still_detected(self):
		events = [self.event(0.0, 2.0, gpu=1.0), self.event(1.0, 3.0, gpu=1.0)]
		row, = trajectory.accumulate(events)
		self.assertTrue(row["events_overlap"])
		self.assertEqual(row["max_concurrency"], 2)
		self.assertFalse(row["clock_anomaly"])

	def test_monotonic_run_has_no_anomaly(self):
		events = [self.event(0.0, 1.0, gpu=1.0), self.event(1.0, 2.0, gpu=2.0)]
		row, = trajectory.accumulate(events, clock_backsteps=[])
		self.assertFalse(row["clock_anomaly"])
		self.assertEqual(row["sum_attributed_gpu_energy_j"], 3.0)
		self.assertFalse(trajectory.summarize([row], power_backward_steps=0)["clock_anomaly"])

	def test_empty_power_cell_loads_as_missing(self):
		with tempfile.TemporaryDirectory() as tmp:
			p = os.path.join(tmp, "power.csv")
			with open(p, "w") as f:
				f.write("t,gpu0_w,gpu0_energy_mj\n0.0,,100\n1.0,50.0,150\n")
			ts, gpus, _, energy = attribution.load_power(p)
		self.assertEqual(gpus["gpu0_w"], [None, 50.0])
		self.assertEqual(energy["gpu0_energy_mj"], [100.0, 150.0])

	def test_attribution_main_warns_and_writes_run_level_flag(self):
		with tempfile.TemporaryDirectory() as tmp:
			power = os.path.join(tmp, "power.csv")
			with open(power, "w") as f:
				f.write("t,gpu0_w,gpu0_energy_mj\n")
				for t, e in ((0.0, 0), (1.0, 100), (0.7, 200), (2.0, 300), (3.0, 400)):
					f.write(f"{t},100.0,{e}\n")
			events = os.path.join(tmp, "events.jsonl")
			with open(events, "w") as f:
				f.write(json.dumps(self.event(0.2, 1.9, gpu=5.0)) + "\n")
			err, out = io.StringIO(), io.StringIO()
			argv = ["attribution", "--events", events, "--power", power,
			        "--out", os.path.join(tmp, "s.csv"), "--out-events", os.path.join(tmp, "a.jsonl")]
			with mock.patch.object(sys, "argv", argv), contextlib.redirect_stderr(err), \
			     contextlib.redirect_stdout(out):
				self.assertEqual(attribution.main(), 0)
			with open(os.path.join(tmp, "trajectory_summary.json")) as f:
				saved = json.load(f)
			with open(os.path.join(tmp, "a.jsonl")) as f:
				row = json.loads(f.readline())
		self.assertIn("wall-clock anomaly: 1 backward step", err.getvalue())
		self.assertTrue(saved["summary"]["clock_anomaly"])
		self.assertEqual(saved["summary"]["power_backward_steps"], 1)
		self.assertTrue(saved["trajectories"][0]["clock_anomaly"])
		self.assertEqual(row["gpu_energy_j"], 5.0)  # counter energy untouched


# ==========================================================================
# 3. GPU preflight
# ==========================================================================

class GpuPreflight(TempDirCase):

	def setUp(self):
		super().setUp()
		fake_proc(self.proc, extra=[(200, ["python", "train.py"], "python")])

	def test_parse_nvidia_smi_and_metrics(self):
		gpu, = rp.parse_gpu_csv(GPU_IDLE)
		self.assertEqual((gpu["utilization_gpu_pct"], gpu["power_draw_w"], gpu["memory_used_mib"],
		                  gpu["memory_total_mib"]), (6.0, 28.75, 23565.0, 24564.0))
		app, = rp.parse_compute_apps(APPS_VLLM_ONLY)
		self.assertEqual((app["pid"], app["process_name"], app["used_memory_mib"]), (101, "[Not Found]", None))
		metrics = rp.parse_vllm_metrics(METRICS_IDLE)
		self.assertEqual(metrics["num_requests_running"], 0.0)
		self.assertEqual(metrics["prefix_cache_hits_total"], 61216.0)
		self.assertIsNone(rp.parse_vllm_metrics("")["num_requests_waiting"])

	def test_process_classification(self):
		self.assertEqual(rp.classify_process(101, self.proc), "vllm")   # EngineCore
		self.assertEqual(rp.classify_process(102, self.proc), "vllm")   # child of vllm serve
		self.assertEqual(rp.classify_process(200, self.proc), "other")
		self.assertEqual(rp.classify_process(999, self.proc), "unidentified")

	def test_idle_loaded_server_passes_despite_vram_and_power(self):
		record = collect(self.proc)
		self.assertEqual(record["gpu_preflight"]["before"]["gpus"][0]["memory_used_mib"], 23565.0)
		self.assertEqual(rp.evaluate(record, 30.0), [])

	def test_busy_gpu_fails(self):
		problems = rp.evaluate(collect(self.proc, gpu_csv=GPU_BUSY), 30.0)
		self.assertTrue(any("utilization 100%" in p for p in problems))

	def test_threshold_is_configurable(self):
		record = collect(self.proc, gpu_csv=GPU_IDLE.replace(", 6,", ", 40,"))
		self.assertTrue(rp.evaluate(record, 30.0))
		self.assertEqual(rp.evaluate(record, 50.0), [])

	def test_running_or_waiting_vllm_request_fails(self):
		for key in ("running", "waiting"):
			metrics = METRICS_IDLE.replace(f'num_requests_{key}{{engine="0",model_name="google/gemma-3-4b-it"}} 0.0',
			                               f'num_requests_{key}{{engine="0",model_name="google/gemma-3-4b-it"}} 2.0')
			with self.subTest(key=key):
				problems = rp.evaluate(collect(self.proc, metrics=metrics), 30.0)
				self.assertTrue(any(key in p for p in problems))

	def test_visible_competing_process_fails_unidentified_only_warns(self):
		record = collect(self.proc, apps=APPS_VLLM_ONLY + "200, python, 2048\n")
		self.assertTrue(any("pid=200" in p for p in rp.evaluate(record, 30.0)))
		record = collect(self.proc, apps=APPS_VLLM_ONLY + "999, [Not Found], [N/A]\n")
		self.assertEqual(rp.evaluate(record, 30.0), [])
		self.assertTrue(any("pid=999" in w for w in rp.warnings_for(record)))

	def test_snapshot_uses_median_of_samples(self):
		seq = iter([GPU_BUSY, GPU_IDLE, GPU_IDLE])

		def runner(cmd, timeout=15):
			if any(c.startswith("--query-gpu") for c in cmd):
				return 0, next(seq)
			return 0, ""
		snap = rp.gpu_snapshot("http://localhost:8000/v1", samples=3, interval_s=0,
		                       proc_root=self.proc, runner=runner, http_get=http_for())
		self.assertEqual(snap["utilization_median_pct"], {"0": 6.0})
		self.assertEqual(len(snap["gpu_samples"]), 3)

	def test_missing_tools_are_recorded_not_invented(self):
		snap = rp.gpu_snapshot("http://localhost:8000/v1", samples=1, proc_root=self.proc,
		                       runner=lambda cmd, timeout=15: (None, "FileNotFoundError"),
		                       http_get=lambda url, timeout=5: None)
		self.assertFalse(snap["nvidia_smi_available"])
		self.assertIsNone(snap["vllm_metrics"])


# ==========================================================================
# 4. model revision and serving configuration
# ==========================================================================

class ModelRevision(TempDirCase):

	def test_revision_required(self):
		fake_proc(self.proc)
		record = collect(self.proc, revision="")
		self.assertEqual(record["model"]["revision_status"], rp.REVISION_MISSING)
		self.assertTrue(any("MODEL_REVISION is empty" in p for p in rp.evaluate(record, 30.0)))

	def test_revision_verified_against_live_command_line(self):
		fake_proc(self.proc)
		record = collect(self.proc)
		self.assertEqual(record["model"]["revision_status"], rp.REVISION_VERIFIED)
		self.assertEqual(record["model"]["observed_revision"], REVISION)

	def test_mismatch_fails(self):
		fake_proc(self.proc)
		record = collect(self.proc, revision="deadbeef")
		self.assertEqual(record["model"]["revision_status"], rp.REVISION_MISMATCH)
		self.assertTrue(any("does not match" in p for p in rp.evaluate(record, 30.0)))

	def test_unverifiable_revision_is_recorded_honestly(self):
		record = collect(self.proc)  # no vLLM process visible
		self.assertEqual(record["model"]["revision_status"], rp.REVISION_DECLARED_ONLY)
		self.assertIsNone(record["serving"]["config"])
		self.assertEqual(rp.evaluate(record, 30.0), [])
		notes = rp.warnings_for(record)
		self.assertTrue(any("declared" in n and "could not be verified" in n for n in notes))
		self.assertTrue(any("serving configuration is unrecorded" in n for n in notes))

	def test_endpoint_must_serve_the_model(self):
		fake_proc(self.proc)
		record = collect(self.proc, models=json.dumps({"data": [{"id": "other/model"}]}))
		self.assertTrue(any("does not serve" in p for p in rp.evaluate(record, 30.0)))

	def test_serving_config_parsed_without_inventing_values(self):
		cfg = rp.parse_vllm_argv(VLLM_ARGV)
		self.assertEqual((cfg["model"], cfg["revision"], cfg["dtype"], cfg["max_model_len"],
		                  cfg["max_num_seqs"], cfg["gpu_memory_utilization"], cfg["port"]),
		                 ("google/gemma-3-4b-it", REVISION, "bfloat16", "32768", "1", "0.90", "8000"))
		self.assertIsNone(cfg["quantization"])
		self.assertIsNone(cfg["enable_prefix_caching"])  # server default, not asserted
		cfg = rp.parse_vllm_argv(["vllm", "serve", "m", "--dtype=float16", "--no-enable-prefix-caching"])
		self.assertEqual(cfg["dtype"], "float16")
		self.assertIs(cfg["enable_prefix_caching"], False)

	def test_server_selected_by_endpoint_port(self):
		fake_proc(self.proc)
		servers = rp.find_vllm_servers(self.proc)
		self.assertEqual([s["pid"] for s in servers], [100])
		self.assertIsNotNone(rp.select_server(servers, "http://localhost:8000/v1"))
		self.assertIsNone(rp.select_server(servers, "http://localhost:8001/v1"))

	def test_run_experiments_fails_on_empty_revision_in_full_mode(self):
		with open(os.path.join(ROOT, "scripts", "run_experiments.sh")) as f:
			text = f.read()
		self.assertIn('RUN_MODE="${RUN_MODE:-full}"', text)
		self.assertIn('--mode "$RUN_MODE"', text)
		full_branch = text.split("full)", 1)[1].split(";;", 1)[0]
		self.assertIn("fail", full_branch)
		self.assertIn("MODEL_REVISION", full_branch)


class ParamJsonSafety(unittest.TestCase):

	def setUp(self):
		self.tmp = tempfile.mkdtemp()
		self.args = run_module.build_parser().parse_args(
			["--method", "cor", "--dataset", "cwq", "--depth", "4", "--run_size", "3"])
		self.env = mock.patch.dict(os.environ, {"MODEL_NAME": "google/gemma-3-4b-it",
		                                        "MODEL_REVISION": REVISION})
		self.env.start()

	def tearDown(self):
		self.env.stop()
		shutil.rmtree(self.tmp, ignore_errors=True)

	def payload(self, args=None, commit="abc", dirty=False):
		return run_module.build_param_payload(args or self.args,
		                                      {"git_commit": commit, "git_dirty": dirty})

	def test_fresh_directory_records_full_provenance(self):
		path, created = run_module.save_or_check_param_json(self.tmp, self.payload())
		self.assertTrue(created)
		with open(path) as f:
			saved = json.load(f)
		self.assertEqual(saved["param_schema"], run_module.PARAM_SCHEMA)
		self.assertEqual(saved["environment"]["MODEL_REVISION"], REVISION)
		self.assertEqual(saved["provenance"], {"git_commit": "abc", "git_dirty": False})
		self.assertEqual(saved["argparse"]["depth"], 4)

	def test_compatible_resume_keeps_file_untouched(self):
		path, _ = run_module.save_or_check_param_json(self.tmp, self.payload())
		before = open(path).read()
		more = argparse.Namespace(**dict(vars(self.args), run_size=100))
		_, created = run_module.save_or_check_param_json(self.tmp, self.payload(more))
		self.assertFalse(created)
		self.assertEqual(open(path).read(), before)

	def test_incompatible_config_rejected(self):
		run_module.save_or_check_param_json(self.tmp, self.payload())
		deeper = argparse.Namespace(**dict(vars(self.args), depth=3))
		with self.assertRaisesRegex(run_module.ParamConflictError, "depth: 4 -> 3"):
			run_module.save_or_check_param_json(self.tmp, self.payload(deeper))
		with self.assertRaisesRegex(run_module.ParamConflictError, "git_commit"):
			run_module.save_or_check_param_json(self.tmp, self.payload(commit="other"))
		with mock.patch.dict(os.environ, {"MODEL_REVISION": "deadbeef"}):
			with self.assertRaisesRegex(run_module.ParamConflictError, "MODEL_REVISION"):
				run_module.save_or_check_param_json(self.tmp, self.payload())

	def test_dirty_tree_cannot_resume(self):
		run_module.save_or_check_param_json(self.tmp, self.payload(dirty=True))
		with self.assertRaisesRegex(run_module.ParamConflictError, "dirty"):
			run_module.save_or_check_param_json(self.tmp, self.payload(dirty=True))

	def test_legacy_param_json_is_rejected_not_overwritten(self):
		legacy = {"argparse": vars(self.args), "environment": {"MODEL_NAME": "google/gemma-3-4b-it"}}
		path = os.path.join(self.tmp, "param.json")
		with open(path, "w") as f:
			json.dump(legacy, f)
		with self.assertRaisesRegex(run_module.ParamConflictError, "predates provenance"):
			run_module.save_or_check_param_json(self.tmp, self.payload())
		with open(path) as f:
			self.assertEqual(json.load(f), legacy)

	def test_code_provenance_prefers_measure_run_handoff(self):
		with mock.patch.dict(os.environ, {"ENERGY_GIT_COMMIT": "c0ffee", "ENERGY_GIT_DIRTY": "true"}):
			self.assertEqual(run_module.code_provenance(), {"git_commit": "c0ffee", "git_dirty": True})
		with mock.patch.dict(os.environ, {"ENERGY_GIT_COMMIT": "c0ffee", "ENERGY_GIT_DIRTY": "unknown"}):
			self.assertIsNone(run_module.code_provenance()["git_dirty"])


# ==========================================================================
# 5. CPU / DRAM regression: hardening must not make the profiler GPU-only
# ==========================================================================

class CpuDramPreserved(unittest.TestCase):
	TS = [0.0, 1.0, 2.0, 3.0]
	RAPL = {
		"rapl_package-0_intel-rapl:0_uj": [0.0, 10e6, 20e6, 30e6],     # 10 J/s
		"rapl_core_intel-rapl:0:0_uj": [0.0, 6e6, 12e6, 18e6],         # diagnostic
		"rapl_dram_intel-rapl:0:1_uj": [0.0, 2e6, 4e6, 6e6],           # 2 J/s
	}

	def event(self, gpu=7.0):
		return {"question_id": "q", "operation_label": "llm:reason", "operation_type": "llm",
		        "start_timestamp": 0.5, "end_timestamp": 2.5, "duration_s": 2.0,
		        "gpu_energy_j": gpu, "status": "ok"}

	def test_mocked_rapl_domains_are_measured_and_summed(self):
		row, = attribution.attribute_events([self.event()], self.TS, {}, self.RAPL)[0]
		self.assertAlmostEqual(row["cpu_package_energy_j"], 20.0)
		self.assertAlmostEqual(row["dram_energy_j"], 4.0)
		self.assertAlmostEqual(row["cpu_core_energy_j"], 12.0)
		self.assertAlmostEqual(row["measured_energy_j"], 7.0 + 20.0 + 4.0)  # core not added
		self.assertEqual(row["available_energy_domains"], ["gpu", "cpu_package", "dram"])
		self.assertTrue(row["measurement_complete"])

	def test_unavailable_rapl_stays_null(self):
		row, = attribution.attribute_events([self.event()], self.TS, {}, {})[0]
		for key in ("cpu_package_energy_j", "dram_energy_j", "cpu_core_energy_j", "measured_energy_j"):
			self.assertIsNone(row[key], key)
		self.assertEqual(row["available_energy_domains"], ["gpu"])
		self.assertFalse(row["measurement_complete"])

	def test_rapl_trajectory_window_still_measured(self):
		rows, _ = attribution.attribute_events([self.event()], self.TS, {}, self.RAPL)
		pkg, core, dram = attribution.classify_rapl(self.RAPL)
		window = {"gpu_energy_j": 7.0, "cpu_package_energy_j": attribution.sum_domain(self.TS, pkg, 0.5, 2.5),
		          "dram_energy_j": attribution.sum_domain(self.TS, dram, 0.5, 2.5)}
		window["measured_energy_j"] = attribution.measured_total(*window.values())
		row, = trajectory.accumulate(rows, window_energy=lambda t0, t1: window)
		self.assertAlmostEqual(row["trajectory_measured_energy_j"], 31.0)
		self.assertAlmostEqual(row["coverage_measured_energy_j"], 1.0)
		self.assertFalse(row["clock_anomaly"])

	def test_provenance_reports_cpu_dram_domains_when_present(self):
		with tempfile.TemporaryDirectory() as tmp:
			with_rapl = os.path.join(tmp, "a.csv")
			with open(with_rapl, "w") as f:
				f.write("t,gpu0_w,gpu0_energy_mj," + ",".join(self.RAPL) + "\n")
			gpu_only = os.path.join(tmp, "b.csv")
			with open(gpu_only, "w") as f:
				f.write("t,gpu0_w,gpu0_energy_mj\n")
			self.assertEqual(measure_run.energy_domains(with_rapl), {
				"gpu_counter": True, "cpu_package": True, "dram": True,
				"cpu_core_diagnostic": True, "measured_total_possible": True})
			self.assertEqual(measure_run.energy_domains(gpu_only), {
				"gpu_counter": True, "cpu_package": False, "dram": False,
				"cpu_core_diagnostic": False, "measured_total_possible": False})

	def test_preflight_policy_has_no_domain_requirement(self):
		# The GPU checks gate exclusivity only; nothing in the policy assumes
		# or requires a GPU-only boundary.
		with tempfile.TemporaryDirectory() as tmp:
			fake_proc(tmp)
			record = collect(tmp)
		measure_run.apply_policy(record, "full", 30.0)
		self.assertNotIn("domain", json.dumps(record["policy"]).lower())


if __name__ == "__main__":
	unittest.main()
