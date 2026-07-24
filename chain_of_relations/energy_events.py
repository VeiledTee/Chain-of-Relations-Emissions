"""Side-channel energy event log.

Every unit of work the system performs (LLM call, SPARQL query, embedding
encode) is recorded here as one JSON line with wall-clock start/end times,
a category, and a label. The out-of-band power logger (measurement/power_logger.py)
samples hardware power on the same wall clock; measurement/attribute.py joins
the two offline to attribute energy per event, per label, per category, and
per question.

Taxonomy (grouping contract):
  category = "inference"  -> all LLM text generation (the model thinking)
  category = "tool"       -> everything else the system does
  label    = subcategory, always "<family>:<detail>", e.g.
      inference -> "llm:generate"
      tool      -> "kg:relation_search", "kg:entity_search", "kg:id2name",
                   "kg:sparql" (unclassified), "embedding:prune"

Events are appended immediately (line-buffered) so a crash loses at most the
in-flight event. Disable entirely by not setting ENERGY_EVENTS_FILE.
"""

import contextvars
import json
import os
import threading
import time

_EVENTS_FILE = os.getenv("ENERGY_EVENTS_FILE", "")
_lock = threading.Lock()
_fh = None

_current_question = contextvars.ContextVar("energy_current_question", default="")

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


def set_question(question_id) -> None:
	"""Called once per question by run.py; tags all subsequent events."""
	_current_question.set(str(question_id))


def _classify_sparql(sparql_txt: str) -> str:
	t = sparql_txt or ""
	if "type.object.name" in t:
		return "kg:id2name"
	if "?relation" in t:
		return "kg:relation_search"
	if "?targetEntity" in t or "?tailEntity" in t or "?x" in t:
		return "kg:entity_search"
	return "kg:sparql"


def record(category: str, label: str, start, end, **meta) -> None:
	"""Append one event. Never raises: measurement must not break the run.

	start/end may be floats (wall time) or marks from mark(). When both marks
	carry a GPU energy reading, the event stores a measured gpu_energy_j
	(counter delta); otherwise gpu_energy_j is null and attribute.py integrates
	the power curve for that event instead.
	"""
	if not _EVENTS_FILE:
		return
	global _fh
	try:
		t0, e0 = _split(start)
		t1, e1 = _split(end)
		gpu_energy_j = None
		if e0 is not None and e1 is not None and e1 >= e0:
			gpu_energy_j = (e1 - e0) / 1000.0  # mJ -> J
		event = {
			"question_id": _current_question.get(),
			"category": category,
			"label": label,
			"t_start": t0,
			"t_end": t1,
			"duration_s": t1 - t0,
			"gpu_energy_j": gpu_energy_j,
		}
		if meta:
			event["meta"] = meta
		with _lock:
			if _fh is None:
				_fh = open(_EVENTS_FILE, "a", buffering=1)
			_fh.write(json.dumps(event) + "\n")
	except Exception:
		pass


def record_sparql(sparql_txt: str, start, end, **meta) -> None:
	record("tool", _classify_sparql(sparql_txt), start, end, **meta)


def now() -> float:
	return time.time()
