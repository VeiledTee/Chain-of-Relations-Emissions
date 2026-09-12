"""Canonical operation taxonomy for the energy measurement layer (schema v1).

Semantic labels are assigned by the caller that knows *why* an operation is
happening. They are never inferred from prompt text. Every label is
"<type>:<detail>" where <type> is an OperationType member, so the type is
always recoverable from the label (see type_of).

Three paradigms in this repository are formalized against schema v1: CoR, ToG
and PoG. A label is reused across paradigms only where the semantic operation
is genuinely the same, and a new label is introduced only where a paradigm does
something the others do not. No label exists to make the paradigms look
symmetrical.

Each CoR label below maps to exactly one real call site:

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
declared here only because ToG and PoG emit them.

ToG (methods/tog/agent.py) reuses the CoR vocabulary wherever the operation is
the same shared tool doing the same job, and adds nothing of its own:

  llm:relation_rank   relation_prune per frontier entity  -> tools/relation_prune.py
  llm:entity_prune    entity_prune, one LLM call per branch -> tools/entity_prune.py
  llm:reason          _run_reasoning_yes_no, the stop/continue call
  llm:direct_answer   _generate_directly, the closed-book fallback

ToG has no answer-filter stage and no embedding stage, so it emits neither.

PoG (methods/pog/agent.py) reuses four labels and adds four of its own, for the
stages no other paradigm here has:

  llm:subquestion_decompose      decomposes the question into sub-objectives
  llm:memory_update              rewrites the working-memory JSON from new triples
  llm:reverse_retrieval_decision decides whether to reverse-retrieve
  llm:reverse_entity_select      picks which earlier entities re-enter the frontier

PoG entity selection goes through entity_condition_prune, which filters by a
subquestion condition rather than scoring. That is the same semantic operation
as ToG entity scoring -- deciding which candidate entities survive the hop --
so both emit llm:entity_prune and the difference is recorded in
meta.prune_strategy ("condition" vs "score"). PoG is also the only paradigm
with an embedding stage: embedding:prune, the SentenceTransformer cut applied
before the LLM sees a large candidate group.

Structural context, for every paradigm:

  iteration        the outer search/reasoning round (CoR: one DFS state pop;
                   ToG/PoG: one pass of the depth loop). Never decreases.
  traversal_depth  the actual graph traversal depth. The comparable structural
                   field across paradigms.
  step_index       globally monotonic event ordering within the question.

Repeated work inside one round is metadata, not a new iteration:
frontier_index, branch_index, group_index, reverse_round.

KG energy is instrumented once, at the shared Freebase backend
(kg_backend/freebase/db_func.py), so all three paradigms get kg:* events with
no agent-side spans. The Wikidata backend is NOT instrumented: see
WIKIDATA_KG_UNINSTRUMENTED below.

Taxonomy changes after schema freeze must be deliberate and documented.
"""

from agent_energy_profiler.schema import SCHEMA_VERSION, Status, _Str

__all__ = [
	"SCHEMA_VERSION", "Status", "OperationType", "OperationLabel", "Paradigm",
	"COR_LABELS", "COR_LLM_LABELS", "TOG_LABELS", "TOG_LLM_LABELS",
	"POG_LABELS", "POG_LLM_LABELS", "LABELS_BY_PARADIGM",
	"FALLBACK_NO_GRAPH_ANSWER", "FALLBACK_REASONS",
	"PRUNE_STRATEGY_SCORE", "PRUNE_STRATEGY_CONDITION",
	"WIKIDATA_KG_UNINSTRUMENTED",
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

	# --- PoG-specific stages. Each maps to exactly one real call site; no
	# other paradigm in this repository performs these operations. ---
	LLM_SUBQUESTION_DECOMPOSE = "llm:subquestion_decompose"
	LLM_MEMORY_UPDATE = "llm:memory_update"
	LLM_REVERSE_RETRIEVAL_DECISION = "llm:reverse_retrieval_decision"
	LLM_REVERSE_ENTITY_SELECT = "llm:reverse_entity_select"

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

#: Labels a formalized ToG run is allowed to emit. ToG adds no label of its
#: own: every LLM stage it has is one CoR also has.
TOG_LLM_LABELS = frozenset({
	OperationLabel.LLM_RELATION_RANK,
	OperationLabel.LLM_ENTITY_PRUNE,
	OperationLabel.LLM_REASON,
	OperationLabel.LLM_DIRECT_ANSWER,
})

TOG_LABELS = frozenset(TOG_LLM_LABELS | {
	OperationLabel.KG_ID2NAME,
	OperationLabel.KG_RELATION_SEARCH,
	OperationLabel.KG_ENTITY_SEARCH,
	OperationLabel.KG_SPARQL,
	OperationLabel.SYSTEM_ORCHESTRATION,
})

#: The PoG LLM stages: four shared with other paradigms, four its own.
POG_LLM_LABELS = frozenset({
	OperationLabel.LLM_SUBQUESTION_DECOMPOSE,
	OperationLabel.LLM_RELATION_RANK,
	OperationLabel.LLM_ENTITY_PRUNE,
	OperationLabel.LLM_MEMORY_UPDATE,
	OperationLabel.LLM_REASON,
	OperationLabel.LLM_REVERSE_RETRIEVAL_DECISION,
	OperationLabel.LLM_REVERSE_ENTITY_SELECT,
	OperationLabel.LLM_DIRECT_ANSWER,
})

POG_LABELS = frozenset(POG_LLM_LABELS | {
	OperationLabel.EMBEDDING_PRUNE,
	OperationLabel.KG_ID2NAME,
	OperationLabel.KG_RELATION_SEARCH,
	OperationLabel.KG_ENTITY_SEARCH,
	OperationLabel.KG_SPARQL,
	OperationLabel.SYSTEM_ORCHESTRATION,
})

#: Which vocabulary each formalized paradigm may emit. Analysis groups by
#: (paradigm, operation_label); this is what makes such a grouping checkable.
LABELS_BY_PARADIGM = {
	Paradigm.COR.value: COR_LABELS,
	Paradigm.TOG.value: TOG_LABELS,
	Paradigm.POG.value: POG_LABELS,
}

_LABEL_VALUES = frozenset(item.value for item in OperationLabel)
_TYPE_VALUES = frozenset(item.value for item in OperationType)
_STATUS_VALUES = frozenset(item.value for item in Status)

#: Controlled meta.fallback_reason values for llm:direct_answer. Every closed-
#: book fallback names why the graph search produced no answer, so fallback
#: cost is separable and attributable to the exit that caused it. The event
#: keeps the iteration and traversal depth at which the fallback fired; depth
#: is null only where the fallback genuinely runs outside the traversal.
FALLBACK_NO_GRAPH_ANSWER = "no_graph_answer"          # CoR: DFS ended, no answer
FALLBACK_NO_TOPIC_ENTITIES = "no_topic_entities"      # ToG/PoG: pre-loop, depth null
FALLBACK_NO_BRANCHES = "no_branches"                  # ToG: nothing to expand
FALLBACK_ENTITY_PRUNE_FAILED = "entity_prune_failed"  # ToG: pruning produced nothing
FALLBACK_EMPTY_FRONTIER = "empty_frontier"            # ToG: no entities survive
FALLBACK_NO_CANDIDATE_GROUPS = "no_candidate_groups"  # PoG: retrieval found nothing
FALLBACK_PRUNE_EMPTY = "prune_empty"                  # PoG: pruning selected nothing
FALLBACK_STOP_WITHOUT_RESULTS = "stop_without_results"  # PoG: stop, no usable answer
FALLBACK_DEPTH_EXHAUSTED = "depth_exhausted"          # ToG/PoG: loop ran out

FALLBACK_REASONS = frozenset({
	FALLBACK_NO_GRAPH_ANSWER,
	FALLBACK_NO_TOPIC_ENTITIES,
	FALLBACK_NO_BRANCHES,
	FALLBACK_ENTITY_PRUNE_FAILED,
	FALLBACK_EMPTY_FRONTIER,
	FALLBACK_NO_CANDIDATE_GROUPS,
	FALLBACK_PRUNE_EMPTY,
	FALLBACK_STOP_WITHOUT_RESULTS,
	FALLBACK_DEPTH_EXHAUSTED,
})

#: Controlled meta.prune_strategy values for llm:entity_prune. The label is the
#: shared semantic operation (which candidates survive the hop); this records
#: how the paradigm decides it, so ToG and PoG stay comparable without
#: pretending their mechanisms are identical.
PRUNE_STRATEGY_SCORE = "score"          # ToG: LLM scores candidates, top-k win
PRUNE_STRATEGY_CONDITION = "condition"  # PoG: LLM filters by a subquestion condition

#: KG energy instrumentation lives in the Freebase backend only. A Wikidata run
#: (dataset qald10_en) therefore emits LLM and embedding events but NO kg:*
#: events: its KG energy is unmeasured, not zero. Analysis must not read the
#: absence of kg:* rows on such a run as "the KG cost nothing".
WIKIDATA_KG_UNINSTRUMENTED = (
	"kg_backend/wikidata/db_func.py emits no energy events: KG operations on "
	"the Wikidata backend (dataset qald10_en) are unmeasured, not zero"
)

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
