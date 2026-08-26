"""Side-channel semantic energy event log (schema v1).

Every unit of work the system performs (LLM call, SPARQL query, embedding
encode) is recorded here as one JSON line: a semantic operation label, the
run/question/trajectory context it happened in, wall-clock start/end times, an
explicit status, and whatever energy the process could read in-band.

Trajectory position is three separate fields -- iteration, traversal_depth and
step_index -- which measure different things; see the contextvar block below.

This file is layer 1 of four. It deliberately does NOT know CPU-package or DRAM
energy: those come from the out-of-band sampler.

  1. events.jsonl            <- this module (semantic timeline)
  2. power.csv               <- measurement/power_logger.py (hardware timeline)
  3. measurement/attribute.py   joins 1 and 2
  4. events_attributed.jsonl    per-event research artifact for C2
     energy_summary.csv         derived aggregate

Semantic labels come from chain_of_relations.energy_taxonomy and are supplied
by the caller that knows why an operation is happening -- never inferred from
prompt text. The CoR agent declares them with the operation() context manager;
LLMAPI.generate reads the innermost declared label.

Instrumentation is best-effort: record() never raises, so a measurement bug can
never break the benchmark workload. Disable entirely by leaving
ENERGY_EVENTS_FILE unset.
"""

import contextlib
import contextvars
import json
import os
import subprocess
import threading
import time
import uuid

from chain_of_relations import energy_taxonomy as tax
from chain_of_relations.energy_taxonomy import OperationLabel, Status

_EVENTS_FILE = os.getenv("ENERGY_EVENTS_FILE", "")
_lock = threading.Lock()
_fh = None

# --- trajectory context ----------------------------------------------------
# Three distinct orderings, all reset per question. They must not be conflated:
#
#   iteration        monotonically increasing CoR reasoning/control-loop
#                    iteration. One per pass of the agent's control loop.
#                    Never decreases within a question.
#   traversal_depth  the DFS/search depth of the state being expanded. Rises
#                    and falls as the search descends and backtracks. Null
#                    for work that happens outside the traversal.
#   step_index       monotonically increasing measured-event index. One per
#                    recorded event, so it counts measurements, not decisions.
#
# A backtracking question therefore looks like
#   iteration       0  1  2  3  4
#   traversal_depth 0  1  2  1  2
# with step_index climbing straight through, several steps per iteration.
_current_question = contextvars.ContextVar("energy_current_question", default="")
_current_iteration = contextvars.ContextVar("energy_current_iteration", default=None)
_current_traversal_depth = contextvars.ContextVar(
	"energy_current_traversal_depth", default=None)
_current_operation = contextvars.ContextVar("energy_current_operation", default=None)
_current_event_meta = contextvars.ContextVar("energy_current_event_meta", default={})
_step_counter = 0
_iteration_counter = -1

# --- run-scoped provenance, configured once by the run wrapper -------------
_run_context = {
	"run_id": "",
	"dataset": "",
	"paradigm": "",
	"model_name": "",
	"model_revision": "",
	"git_commit": "",
	"hardware_id": "",
}

# --- NVML cumulative energy counter (observer-only, optional) --------------
# Reads nvmlDeviceGetTotalEnergyConsumption at each event boundary. Unlike the
# 10 Hz power sampler, this captures a monotonic hardware energy counter at the
# exact instants an event starts and ends, so even sub-millisecond events get a
# precise Joule figure (counter delta). No power capping / no intervention:
# this is a read-only counter, the same one Zeus's ZeusMonitor uses.
# Falls back to None where unsupported (pre-Volta GPUs, some WSL2 configs);
# attribute.py then reverts to power-curve integration for that event.
_pynvml = None
_nvml_handles = []
_nvml_ok = False
_nvml_tried = False


def _init_nvml() -> None:
	global _pynvml, _nvml_handles, _nvml_ok, _nvml_tried
	if _nvml_tried or not _EVENTS_FILE:
		return
	_nvml_tried = True
	try:
		import pynvml
		pynvml.nvmlInit()
		n = pynvml.nvmlDeviceGetCount()
		_pynvml = pynvml
		_nvml_handles = [pynvml.nvmlDeviceGetHandleByIndex(i) for i in range(n)]
		_nvml_ok = True
	except Exception:
		_nvml_ok = False


def gpu_energy_mj():
	"""Cumulative device energy (millijoules), summed across GPUs, or None."""
	_init_nvml()
	if not _nvml_ok:
		return None
	try:
		return sum(_pynvml.nvmlDeviceGetTotalEnergyConsumption(h)
		           for h in _nvml_handles)
	except Exception:
		return None


def mark():
	"""Capture (wall_time, cumulative_gpu_energy_mj) at this instant.

	Pass the returned marks as the start/end args to record()/record_sparql();
	they compute the GPU energy delta directly from the counter.
	"""
	return (time.time(), gpu_energy_mj())


def _split(m):
	"""Accept either a float wall-time (legacy) or a (t, energy_mj) mark."""
	if isinstance(m, tuple):
		return m[0], m[1]
	return m, None


def enabled() -> bool:
	return bool(_EVENTS_FILE)


def now() -> float:
	return time.time()


# --- configuration ---------------------------------------------------------

def configure(events_file=None, **run_context) -> None:
	"""Set the event sink and/or run-scoped provenance. Best-effort.

	Called once per run (from chain_of_relations.run). Unknown keys are
	ignored; missing provenance stays an empty string rather than breaking
	the run -- attribute.py emits null for empty provenance.
	"""
	global _EVENTS_FILE, _fh
	try:
		if events_file is not None and events_file != _EVENTS_FILE:
			with _lock:
				if _fh is not None:
					try:
						_fh.close()
					except Exception:
						pass
				_fh = None
			_EVENTS_FILE = events_file
		for key in _run_context:
			if key in run_context and run_context[key] is not None:
				_run_context[key] = str(run_context[key])
	except Exception:
		pass


def close() -> None:
	"""Flush and close the event sink. Best-effort; safe to call twice."""
	global _fh
	try:
		with _lock:
			if _fh is not None:
				_fh.close()
				_fh = None
	except Exception:
		pass


def run_context() -> dict:
	return dict(_run_context)


def detect_git_commit() -> str:
	"""Best-effort short-circuiting git revision of the working tree."""
	try:
		out = subprocess.run(
			["git", "rev-parse", "HEAD"],
			cwd=os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
			capture_output=True, text=True, timeout=5,
		)
		if out.returncode == 0:
			return out.stdout.strip()
	except Exception:
		pass
	return ""


def detect_hardware_id() -> str:
	"""Stable-ish host descriptor: hostname plus GPU model(s)."""
	try:
		import platform
		host = platform.node() or ""
	except Exception:
		host = ""
	gpus = ""
	try:
		_init_nvml()
		if _nvml_ok:
			names = []
			for h in _nvml_handles:
				name = _pynvml.nvmlDeviceGetName(h)
				if isinstance(name, bytes):
					name = name.decode("utf-8", "replace")
				names.append(str(name))
			gpus = ",".join(names)
	except Exception:
		gpus = ""
	return f"{host}|{gpus}" if (host or gpus) else ""


# --- trajectory context ----------------------------------------------------

def set_question(question_id) -> None:
	"""Called once per question by run.py; tags all subsequent events.

	Resets step_index and iteration so both are ordered *within* the question
	trajectory, and clears traversal depth.
	"""
	global _step_counter, _iteration_counter
	_current_question.set(str(question_id))
	with _lock:
		_step_counter = 0
		_iteration_counter = -1
	_current_iteration.set(None)
	_current_traversal_depth.set(None)


def current_question() -> str:
	return _current_question.get()


def begin_iteration() -> int:
	"""Advance to the next control-loop iteration and return its index.

	Called once per pass of the CoR control loop. Iteration counts decisions
	the agent makes, so it never decreases within a question even when the
	search backtracks to a shallower depth.
	"""
	global _iteration_counter
	with _lock:
		_iteration_counter += 1
		value = _iteration_counter
	_current_iteration.set(value)
	return value


def set_iteration(iteration) -> None:
	"""Set the control-loop iteration explicitly (tests, non-DFS callers)."""
	_current_iteration.set(None if iteration is None else int(iteration))


def current_iteration():
	return _current_iteration.get()


@contextlib.contextmanager
def iteration(value):
	"""Scope the control-loop iteration for the events emitted inside."""
	token = _current_iteration.set(None if value is None else int(value))
	try:
		yield
	finally:
		_current_iteration.reset(token)


def set_traversal_depth(depth) -> None:
	"""Set the DFS/search depth of the state currently being expanded.

	Null means the work happens outside the traversal, as the closed-book
	llm:direct_answer fallback does.
	"""
	_current_traversal_depth.set(None if depth is None else int(depth))


def current_traversal_depth():
	return _current_traversal_depth.get()


@contextlib.contextmanager
def traversal_depth(value):
	"""Scope the DFS/search depth for the events emitted inside."""
	token = _current_traversal_depth.set(None if value is None else int(value))
	try:
		yield
	finally:
		_current_traversal_depth.reset(token)


@contextlib.contextmanager
def operation(label):
	"""Declare the semantic operation for events emitted inside this scope.

	This is how a caller that knows *why* the model is being invoked hands the
	label to the generic LLM wrapper, without threading it through every
	shared tool signature. Nesting is allowed; the innermost scope wins.
	"""
	token = _current_operation.set(str(label))
	try:
		yield
	finally:
		_current_operation.reset(token)


def current_operation():
	return _current_operation.get()


@contextlib.contextmanager
def event_meta(**meta):
	"""Attach metadata to every event recorded inside this scope.

	Used where the fact being recorded is known by an enclosing caller rather
	than at the record() site: the CoR fallback flag, and the entity/batch
	counts for a KG lookup that resolves many ids in one round trip.

	Scopes nest and merge, inner keys winning; explicit record() kwargs
	outrank both. Note this applies to *all* events in the scope, so keep the
	scope tight around the call it describes.
	"""
	merged = dict(_current_event_meta.get())
	merged.update({k: v for k, v in meta.items() if v is not None})
	token = _current_event_meta.set(merged)
	try:
		yield
	finally:
		_current_event_meta.reset(token)


def current_event_meta() -> dict:
	return dict(_current_event_meta.get())


def _next_step_index() -> int:
	global _step_counter
	with _lock:
		value = _step_counter
		_step_counter += 1
	return value


# --- the event record ------------------------------------------------------

def record(label, start, end, status=Status.OK, input_tokens=None,
           output_tokens=None, **meta) -> None:
	"""Append one schema-v1 event. Never raises: measurement must not break
	the run.

	label   -- an energy_taxonomy.OperationLabel (or its string value). The
	           operation_type is derived from it, so the two can never disagree.
	start/end -- floats (wall time) or marks from mark(). When both marks carry
	           a GPU energy reading, the event stores a measured gpu_energy_j
	           (counter delta); otherwise gpu_energy_j is null and
	           attribute.py integrates the power curve for that event instead.
	status  -- an energy_taxonomy.Status. Terminal failures must be recorded
	           as such; retries belong in meta["attempts"].

	cpu_package_energy_j / dram_energy_j are always null here: this process
	cannot read RAPL in-band. attribute.py fills them from power.csv, and
	leaves them null when no counter existed. They are never fabricated as 0.
	"""
	if not _EVENTS_FILE:
		return
	global _fh
	try:
		# Backward compatibility with the pre-schema-v1 signature
		#   record(category, label, start, end, **meta)
		# still used by the non-CoR paradigms, which this slice must not touch.
		if isinstance(start, str) and str(label) in (
			tax.LEGACY_CATEGORY_INFERENCE, tax.LEGACY_CATEGORY_TOOL
		):
			label, start, end, status = start, end, status, Status.OK
			if meta.pop("failed", False):
				status = Status.ERROR

		text_label = str(label)
		try:
			operation_type = tax.type_of(text_label)
		except ValueError:
			# An unknown label is a bug, but dropping the event silently would
			# be worse for research integrity than recording it flagged.
			operation_type = text_label.split(":", 1)[0] if ":" in text_label else ""
			meta = dict(meta, unknown_label=True)

		# Scoped metadata from an enclosing caller; explicit kwargs outrank it.
		scoped_meta = _current_event_meta.get()
		if scoped_meta:
			merged_meta = dict(scoped_meta)
			merged_meta.update(meta)
			meta = merged_meta

		text_status = str(status)
		if not tax.is_known_status(text_status):
			meta = dict(meta, unknown_status=text_status)
			text_status = Status.ERROR.value

		t0, e0 = _split(start)
		t1, e1 = _split(end)
		gpu_energy_j = None
		if e0 is not None and e1 is not None and e1 >= e0:
			gpu_energy_j = (e1 - e0) / 1000.0  # mJ -> J

		event = {
			"schema_version": tax.SCHEMA_VERSION,
			"event_id": uuid.uuid4().hex,

			"run_id": _run_context["run_id"] or None,
			"question_id": _current_question.get() or None,
			"dataset": _run_context["dataset"] or None,
			"paradigm": _run_context["paradigm"] or None,

			"iteration": _current_iteration.get(),
			"traversal_depth": _current_traversal_depth.get(),
			"step_index": _next_step_index(),

			"operation_type": operation_type,
			"operation_label": text_label,

			"start_timestamp": t0,
			"end_timestamp": t1,
			"duration_s": t1 - t0,

			"gpu_energy_j": gpu_energy_j,
			"cpu_package_energy_j": None,
			"dram_energy_j": None,

			"input_tokens": input_tokens,
			"output_tokens": output_tokens,

			"status": text_status,

			"model_name": _run_context["model_name"] or None,
			"model_revision": _run_context["model_revision"] or None,
			"git_commit": _run_context["git_commit"] or None,
			"hardware_id": _run_context["hardware_id"] or None,

			"meta": dict(meta),

			# Legacy keys, pre-schema-v1. Kept so existing readers
			# (measurement/analyze.py, audit_runs.py) keep working during the
			# migration. Schema-v1 analysis must use the canonical keys above.
			"category": tax.LEGACY_CATEGORY_INFERENCE
			            if operation_type == "llm" else tax.LEGACY_CATEGORY_TOOL,
			"label": text_label,
			"t_start": t0,
			"t_end": t1,
		}
		with _lock:
			if _fh is None:
				_fh = open(_EVENTS_FILE, "a", buffering=1)
			_fh.write(json.dumps(event) + "\n")
	except Exception:
		pass


def _classify_sparql(sparql_txt: str):
	"""Structural classification of a SPARQL query into a KG label.

	This inspects the *query structure* the caller built, not any natural
	language. Call sites whose semantics are unambiguous regardless of query
	shape (id2name) pin their label explicitly instead of relying on this.
	"""
	t = sparql_txt or ""
	if "type.object.name" in t:
		return OperationLabel.KG_ID2NAME
	if "?relation" in t:
		return OperationLabel.KG_RELATION_SEARCH
	if "?targetEntity" in t or "?tailEntity" in t or "?x" in t:
		return OperationLabel.KG_ENTITY_SEARCH
	return OperationLabel.KG_SPARQL


def record_sparql(sparql_txt: str, start, end, label=None,
                  status=Status.OK, **meta) -> None:
	"""Record a KG event. Pass label to pin the semantics explicitly."""
	record(label or _classify_sparql(sparql_txt), start, end,
	       status=status, **meta)
