"""CoR adapter for the generic energy profiler.

This module is the whole of what Chain-of-Relations knows about measurement.
It does three things and nothing else:

  1. registers the frozen CoR vocabulary (energy_taxonomy) with the profiler,
     so a label outside it is flagged rather than silently accepted;
  2. owns the one piece of KG-specific label logic -- classifying a SPARQL
     query by the *structure the caller built*, for the call sites that do not
     pin their label explicitly;
  3. re-exports the profiler's recording API under the name the CoR call sites
     already import, so the ~35 instrumentation lines inside the algorithm did
     not have to change.

Dependency direction is one-way and enforced by tests/test_package_boundary.py:

    chain_of_relations -> agent_energy_profiler        allowed
    agent_energy_profiler -> chain_of_relations        forbidden

The profiler measures spans and knows nothing about relation ranking, DFS
traversal or Freebase. Meaning lives here.

Schema, field semantics and the operation taxonomy are unchanged and frozen;
see agent_energy_profiler/schema.py and chain_of_relations/energy_taxonomy.py.
"""

import os

from agent_energy_profiler import events as _events
from agent_energy_profiler import labels as _labels
from chain_of_relations import energy_taxonomy as tax
from chain_of_relations.energy_taxonomy import OperationLabel, Status

# --- 1. declare the CoR vocabulary ----------------------------------------
_labels.register(
	labels=[item.value for item in OperationLabel],
	types=[item.value for item in tax.OperationType],
)

# Re-read ENERGY_EVENTS_FILE and reset per-run counters at (re)import, which is
# how this module behaved when the recorder lived inside it.
_events.reset_from_env()

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# --- 3. re-export the profiler API ----------------------------------------
gpu_energy_mj = _events.gpu_energy_mj
mark = _events.mark
enabled = _events.enabled
now = _events.now
configure = _events.configure
close = _events.close
reset_from_env = _events.reset_from_env
run_context = _events.run_context
detect_hardware_id = _events.detect_hardware_id

set_question = _events.set_question
current_question = _events.current_question
begin_iteration = _events.begin_iteration
set_iteration = _events.set_iteration
current_iteration = _events.current_iteration
iteration = _events.iteration
set_traversal_depth = _events.set_traversal_depth
current_traversal_depth = _events.current_traversal_depth
traversal_depth = _events.traversal_depth
operation = _events.operation
current_operation = _events.current_operation
event_meta = _events.event_meta
current_event_meta = _events.current_event_meta
span = _events.span
record = _events.record


def detect_git_commit() -> str:
	"""Git revision of THIS repository, whatever the process CWD is."""
	return _events.detect_git_commit(_REPO_ROOT)


# --- 2. the one KG-specific label rule ------------------------------------

def _classify_sparql(sparql_txt: str):
	"""Structural classification of a SPARQL query into a KG label.

	This inspects the *query structure* the caller built -- variable names and
	predicates chosen by CoR's own query templates -- not any natural language,
	prompt text or user input. Call sites whose semantics are unambiguous
	regardless of query shape (id2name) pin their label explicitly instead of
	relying on this.

	It lives in the adapter, not the profiler: the profiler has no business
	knowing what a Freebase predicate is.
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
