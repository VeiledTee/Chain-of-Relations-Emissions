"""Canonical operation taxonomy for the energy measurement layer (schema v1).

Semantic labels are assigned by the caller that knows *why* an operation is
happening. They are never inferred from prompt text. Every label is
"<type>:<detail>" where <type> is an OperationType member, so the type is
always recoverable from the label (see type_of).

CoR (Chain-of-Relations) is the only paradigm formalized in this slice. Each
CoR label below maps to exactly one real call site:

  llm:relation_rank  CoRAgent.relation_prune      -> tools/relation_prune.py
  llm:reason         CoRAgent.reasoning           -> tools/reasoning.py
  llm:answer_filter  CoRAgent.validate            -> methods/cor/tools/filter.py
  llm:direct_answer  CoRAgent.generate_directly   -> tools/generate_directly.py

llm:direct_answer is the closed-book fallback reached only when the DFS ends
without a graph-grounded answer. Its events carry meta.fallback = true and
meta.fallback_reason, so fallback cost is separable from the KG-grounded
answering done by llm:answer_filter.

  kg:relation_search tools/relation_search.py     -> SPARQL head/tail relations
  kg:entity_search   tools/entity_search.py       -> SPARQL target entities
  kg:id2name         db_func.id2entity_name*      -> entity id -> label
  kg:sparql          any other SPARQL execution

CoR has no embedding stage and does not use tools/entity_prune.py, so
EMBEDDING_PRUNE / LLM_ENTITY_PRUNE are NOT part of the CoR taxonomy; they are
declared here only because other paradigms in this repository emit them.

Taxonomy changes after schema freeze must be deliberate and documented.
"""

from agent_energy_profiler.schema import SCHEMA_VERSION, Status, _Str

__all__ = [
	"SCHEMA_VERSION", "Status", "OperationType", "OperationLabel", "Paradigm",
	"COR_LABELS", "COR_LLM_LABELS", "FALLBACK_NO_GRAPH_ANSWER",
	"LEGACY_CATEGORY_INFERENCE", "LEGACY_CATEGORY_TOOL",
	"is_known_label", "is_known_status", "type_of", "legacy_category",
]

# SCHEMA_VERSION and Status are the generic event schema's, re-exported so
# there is exactly one definition of each. This module owns only the CoR
# *semantics*: which operations exist and what they mean.


class OperationType(_Str):
	LLM = "llm"
	KG = "kg"
	EMBEDDING = "embedding"
	SYSTEM = "system"


class OperationLabel(_Str):
	# --- LLM: the four semantic stages the CoR algorithm actually has ---
	LLM_RELATION_RANK = "llm:relation_rank"
	LLM_REASON = "llm:reason"
	LLM_ANSWER_FILTER = "llm:answer_filter"
	LLM_DIRECT_ANSWER = "llm:direct_answer"

	# --- KG ---
	KG_ID2NAME = "kg:id2name"
	KG_RELATION_SEARCH = "kg:relation_search"
	KG_ENTITY_SEARCH = "kg:entity_search"
	KG_SPARQL = "kg:sparql"

	# --- Not emitted by CoR; present for other paradigms in this repo ---
	LLM_ENTITY_PRUNE = "llm:entity_prune"
	EMBEDDING_PRUNE = "embedding:prune"
	SYSTEM_ORCHESTRATION = "system:orchestration"

	# --- Legacy, pre-schema-v1. Retained so old artifacts stay readable and
	# so non-CoR paradigms keep running unchanged. A formalized CoR run must
	# never emit it: see tests/test_energy_taxonomy.py.
	LLM_GENERATE = "llm:generate"


class Paradigm(_Str):
	COR = "cor"
	TOG = "tog"
	POG = "pog"
	COT_PROMPT = "cot_prompt"
	IO_PROMPT = "io_prompt"


#: Labels a formalized CoR run is allowed to emit.
COR_LABELS = frozenset({
	OperationLabel.LLM_RELATION_RANK,
	OperationLabel.LLM_REASON,
	OperationLabel.LLM_ANSWER_FILTER,
	OperationLabel.LLM_DIRECT_ANSWER,
	OperationLabel.KG_ID2NAME,
	OperationLabel.KG_RELATION_SEARCH,
	OperationLabel.KG_ENTITY_SEARCH,
	OperationLabel.KG_SPARQL,
	OperationLabel.SYSTEM_ORCHESTRATION,
})

#: The CoR LLM stages specifically.
COR_LLM_LABELS = frozenset({
	OperationLabel.LLM_RELATION_RANK,
	OperationLabel.LLM_REASON,
	OperationLabel.LLM_ANSWER_FILTER,
	OperationLabel.LLM_DIRECT_ANSWER,
})

_LABEL_VALUES = frozenset(item.value for item in OperationLabel)
_TYPE_VALUES = frozenset(item.value for item in OperationType)
_STATUS_VALUES = frozenset(item.value for item in Status)

#: Controlled meta.fallback_reason values for llm:direct_answer.
FALLBACK_NO_GRAPH_ANSWER = "no_graph_answer"

#: Pre-schema-v1 category values, kept as a legacy field on every event.
LEGACY_CATEGORY_INFERENCE = "inference"
LEGACY_CATEGORY_TOOL = "tool"


def is_known_label(label) -> bool:
	return str(label) in _LABEL_VALUES


def is_known_status(status) -> bool:
	return str(status) in _STATUS_VALUES


def type_of(label) -> str:
	"""Operation type implied by a label. Raises on an unknown label."""
	text = str(label)
	if text not in _LABEL_VALUES:
		raise ValueError(f"unknown operation label: {text!r}")
	prefix = text.split(":", 1)[0]
	if prefix not in _TYPE_VALUES:
		raise ValueError(f"label {text!r} has no valid operation type prefix")
	return prefix


def legacy_category(label) -> str:
	"""The pre-schema-v1 category this label used to be filed under."""
	return (LEGACY_CATEGORY_INFERENCE
	        if type_of(label) == OperationType.LLM.value
	        else LEGACY_CATEGORY_TOOL)
