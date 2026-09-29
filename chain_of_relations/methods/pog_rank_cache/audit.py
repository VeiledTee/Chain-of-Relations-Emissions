"""Analysis-side helpers for the pog-rank-cache ablation (read-only).

Why this exists
---------------
A relation_rank cache hit keeps its saved step record (step_history, with
`rank_cache.status == "hit"` and 0 input/output tokens) but makes no model
request, so it has NO llm:relation_rank energy event. Saved steps and energy
events share no identifier (events carry event_id/step_index; steps carry the
prompt-history trace_id), so any step<->event join is positional within a
question. After the first hit, pairing every relation_prune step with every
llm:relation_rank event by order is off by one.

Use these helpers instead of positional pairing on raw step lists. The C2
baseline pipeline is untouched: every energy/effectiveness script in
measurement/ reads events.jsonl or predict.jsonl only and is unaffected.

Scripts that count LLM calls from step_history (chain_of_relations/eval/
efficiency.py via `python -m chain_of_relations.eval.eval`, and
chain_of_relations/graph.py) count a hit as a call; use event-based counts
(events_attributed.jsonl) for pog-rank-cache instead.
"""

from typing import Any, Dict, Iterable, List, Sequence, Tuple

RELATION_RANK_LABEL = "llm:relation_rank"
RELATION_RANK_STEP = "relation_prune"


def is_cache_hit(step: Dict[str, Any]) -> bool:
	"""True for a saved step that reused a cached reply (no inference ran)."""
	audit = step.get("rank_cache")
	return isinstance(audit, dict) and audit.get("status") == "hit"


def inference_steps(steps: Iterable[Dict[str, Any]]) -> List[Dict[str, Any]]:
	"""Saved LLM steps that correspond to a real model request (hits removed)."""
	return [s for s in steps if s.get("step_type", "llm") == "llm" and not is_cache_hit(s)]


def pair_rank_steps_with_events(steps: Sequence[Dict[str, Any]],
                                events: Sequence[Dict[str, Any]]) -> List[Tuple[Dict[str, Any], Dict[str, Any]]]:
	"""Pair one question's inferred relation_prune steps with its llm:relation_rank events.

	`steps` is that question's saved step_history; `events` its schema-v1
	events (any labels). Events are ordered by step_index. Raises ValueError
	on any count mismatch instead of silently misaligning.
	"""
	rank_steps = [s for s in inference_steps(steps) if s.get("operation_type") == RELATION_RANK_STEP]
	rank_events = sorted((e for e in events if e.get("operation_label") == RELATION_RANK_LABEL),
	                     key=lambda e: e.get("step_index", 0))
	if len(rank_steps) != len(rank_events):
		raise ValueError(f"{len(rank_steps)} inferred relation_prune steps vs "
		                 f"{len(rank_events)} llm:relation_rank events; refusing to pair")
	return list(zip(rank_steps, rank_events))
