"""cor-rank-compact: baseline CoR with a <=10-word relation_rank rationale.

C2 preliminary intervention, not a C3 redesign. The single independent
variable is the rationale text in CoR's relation_pruning prompt (see prompt.yml
next to this file, which differs from methods/cor/prompt.yml in exactly four
lines: the length instruction, the format hint, and the two few-shot example
rationales, all <=10 words; example selections are unchanged). Everything else
is inherited from CoRAgent unchanged: the
same shared tool (tools/relation_prune.py) builds the prompt, calls the model
with the same temperature and max_tokens, parses the reply and validates the
selection; the same DFS, reasoning, filter and fallback stages follow.

Brevity is prompt-driven only. The reply is never truncated or rewritten, so
any energy difference comes from the model generating fewer tokens, measured
by the unchanged llm:relation_rank event.

The ranking rationale is not consumed downstream in baseline CoR:
CoRAgent.relation_prune returns only the parsed selected relations, and the
raw reply is kept solely in prompt_history/step_history for the record.

Audit metadata: each relation_prune record gains a namespaced `rank_compact`
dict (rationale word/char counts, selected-relation count, trace_id). Token
counts are not duplicated here; they already live on the step and the event.
"""

import copy
import json
from pathlib import Path
from typing import Any, Dict, List, Optional

from chain_of_relations.methods.cor.agent import CoRAgent
from chain_of_relations.schema import Entity, Relation

METHOD_ID = "cor-rank-compact"
RANK_PROMPT_KEY = "relation_pruning"
COMPACT_PROMPT_FILE = str(Path(__file__).resolve().parent / "prompt.yml")


def _extract_rationale(response: Optional[str]) -> Optional[str]:
	"""The rationale string of a relation_rank reply, or None if unreadable.

	Locates the JSON object the same way tools/relation_prune._clean_relations
	does. The prompt's key is "rational" (sic); "rationale" is accepted too.
	Analysis-only: nothing here feeds back into the search.
	"""
	payload = (response or "").strip()
	if not payload:
		return None
	if not payload.startswith("{"):
		start, end = payload.find("{"), payload.rfind("}")
		if start == -1 or end <= start:
			return None
		payload = payload[start : end + 1]
	try:
		obj = json.loads(payload)
	except Exception:
		return None
	if not isinstance(obj, dict):
		return None
	for key in ("rational", "rationale"):
		if isinstance(obj.get(key), str):
			return obj[key]
	return None


def rationale_stats(response: Optional[str]) -> Dict[str, Any]:
	"""Rationale length of one reply; unknown values are None, never 0."""
	text = _extract_rationale(response)
	if text is None:
		return {"rationale_parsed": False, "rationale_words": None, "rationale_chars": None}
	return {"rationale_parsed": True, "rationale_words": len(text.split()), "rationale_chars": len(text)}


class CoRRankCompactAgent(CoRAgent):
	def __init__(self, *args, **kwargs):
		super().__init__(*args, **kwargs)
		self._use_compact_rank_prompt()

	def _use_compact_rank_prompt(self) -> None:
		"""Swap in the compact relation_pruning block, on this instance only.

		The other stages keep the baseline file's blocks (deep-copied, so no
		state is shared with any CoRAgent). self.prompt_file stays the baseline
		path, which _get_prompt_block only uses in error messages.
		"""
		compact = self._load_prompts(COMPACT_PROMPT_FILE)
		prompts = copy.deepcopy(self.prompts)
		prompts[RANK_PROMPT_KEY] = compact[RANK_PROMPT_KEY]
		self.prompts = prompts
		self.rank_prompt_file = COMPACT_PROMPT_FILE

	def relation_prune(
		self,
		question: str,
		topic_entity: Entity,
		relation_chain: List[Relation],
		head_relations: List[str],
		tail_relations: List[str],
		prompt_history: List[Dict[str, Any]],
	) -> List[Relation]:
		before = len(prompt_history)
		selected = super().relation_prune(
			question=question,
			topic_entity=topic_entity,
			relation_chain=relation_chain,
			head_relations=head_relations,
			tail_relations=tail_relations,
			prompt_history=prompt_history,
		)
		for record in prompt_history[before:]:
			if record.get("type") == "relation_prune":
				record["rank_compact"] = dict(
					rationale_stats(record.get("response")),
					selected_relation_count=len(record.get("parsed_result") or []),
					trace_id=record.get("trace_id"),
				)
		return selected

	@staticmethod
	def _llm_event_to_step(item: Dict[str, Any]) -> Dict[str, Any]:
		step = CoRAgent._llm_event_to_step(item)
		if "rank_compact" in item:
			step["rank_compact"] = item["rank_compact"]
		return step
