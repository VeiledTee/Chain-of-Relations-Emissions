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


def record(category: str, label: str, t_start: float, t_end: float, **meta) -> None:
	"""Append one event. Never raises: measurement must not break the run."""
	if not _EVENTS_FILE:
		return
	global _fh
	try:
		event = {
			"question_id": _current_question.get(),
			"category": category,
			"label": label,
			"t_start": t_start,
			"t_end": t_end,
			"duration_s": t_end - t_start,
		}
		if meta:
			event["meta"] = meta
		with _lock:
			if _fh is None:
				_fh = open(_EVENTS_FILE, "a", buffering=1)
			_fh.write(json.dumps(event) + "\n")
	except Exception:
		pass


def record_sparql(sparql_txt: str, t_start: float, t_end: float, **meta) -> None:
	record("tool", _classify_sparql(sparql_txt), t_start, t_end, **meta)


def now() -> float:
	return time.time()
