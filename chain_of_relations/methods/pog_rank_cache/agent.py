"""pog-rank-cache: baseline PoG + exact per-question relation_rank memoization.

C2 preliminary mechanism-validation ablation (not a C3 redesign). Frozen C2
baseline: tag c2-analysis-complete-2026-09-29. Baseline PoG
(chain_of_relations/methods/pog) is inherited unchanged; this subclass alters
exactly one thing:

    When a relation_rank model request is byte-for-byte identical to one
    already completed earlier in the SAME question, the first request's model
    reply is reused instead of calling the LLM again.

How, and why it is exact
------------------------
PoG issues relation_rank from one call site, through
`self._llm_generate_for(OperationLabel.LLM_RELATION_RANK, ...)`, and the shared
tool `tools/relation_prune.py` calls the returned function with the fully
rendered prompt. That function is the last point before the model request, so
the cache wraps it:

* Key = the exact request LLMAPI.generate would send: model, messages
  (system prompt if present, then the rendered user prompt), temperature,
  max_tokens, and the request constants LLMAPI always adds. The key is the
  canonical JSON of that request itself (not a hash of a subset), so equal keys
  mean an identical model request. Candidate relations and their order, the
  question, subquestions, and anything else the prompt renders are therefore
  covered without reconstructing them.
* On a HIT the wrapped function returns the cached reply text without calling
  the LLM, so no llm:relation_rank inference event is emitted. The unchanged
  tool then parses that reply exactly as baseline parses a live reply, against
  the CURRENT call's head/tail relations. (PoG renders relations in "unified"
  mode, which omits the topic entity, so two frontier entities with the same
  relation list send identical requests; re-parsing per call keeps each call's
  own relation directions, exactly as baseline would with the same reply.)
* A reply is stored only after relation_prune has turned it into a successful
  ranking (a non-empty selected-relation list). Empty replies, exceptions and
  unparseable replies are never cached.

Scope: one question. The cache is created empty at the start of answer() and
discarded when answer() returns or raises. Nothing is shared across questions,
runs, datasets or processes, and nothing touches disk.

The cached value is the reply string (immutable) plus the originating trace
id; every hit builds fresh result objects through the tool, so no caller can
mutate the cached value.

Audit metadata (namespaced "rank_cache"): per question lookups / hits / misses
/ hit_rate on the answer() result, and per relation_prune record in
prompt_history a {"status", "request_sha256", ...} entry. A hit's record
reports 0 input/output tokens because no inference ran; its "source_trace_id"
points at the record whose reply it reused.
"""

import hashlib
import json
import os
from typing import Any, Dict, List, Optional

from chain_of_relations.energy_taxonomy import OperationLabel
from chain_of_relations.methods.pog.agent import PoGAgent

#: User-facing method identifier (CLI --method, results/<method>/..., paradigm).
METHOD_ID = "pog-rank-cache"

#: Constant fields LLMAPI.generate adds to every chat request (llm_api.py).
#: `timeout` is transport-only and cannot change the reply, so it is excluded.
_REQUEST_CONSTANTS = {"frequency_penalty": 0, "presence_penalty": 0, "stream": False}


class PoGRankCacheAgent(PoGAgent):
	method_id = METHOD_ID

	# ------------------------------------------------------------------ key
	def _rank_request_key(self, user_prompt: str, temperature: Any, max_tokens: Any,
	                      system_prompt: Optional[str]) -> str:
		"""The exact chat request LLMAPI.generate would send, as canonical JSON."""
		messages = []
		if system_prompt is not None:
			messages.append({"role": "system", "content": system_prompt})
		messages.append({"role": "user", "content": user_prompt})
		model_name = self.llm_api.model_name
		request = {"model": model_name, "messages": messages,
		           "temperature": temperature, "max_tokens": max_tokens, **_REQUEST_CONSTANTS}
		if str(model_name).startswith("gpt-5"):
			request["reasoning_effort"] = os.getenv("OPENAI_REASONING_EFFORT", "low")
		# sort_keys orders dict keys only; list order (messages) and every
		# character of every prompt are kept exactly.
		return json.dumps(request, sort_keys=True, ensure_ascii=False)

	@staticmethod
	def _key_sha256(key: str) -> str:
		return hashlib.sha256(key.encode("utf-8")).hexdigest()

	# ------------------------------------------------------ question scope
	def _rank_cache_begin(self) -> None:
		self._rank_cache: Optional[Dict[str, Dict[str, Any]]] = {}
		self._rank_cache_stats: Optional[Dict[str, int]] = {"lookups": 0, "hits": 0, "misses": 0}
		self._rank_cache_call: Optional[Dict[str, Any]] = None
		self._rank_cache_pending: Optional[Dict[str, Any]] = None

	def _rank_cache_end(self) -> Dict[str, Any]:
		stats = dict(getattr(self, "_rank_cache_stats", None) or {"lookups": 0, "hits": 0, "misses": 0})
		self._rank_cache = None
		self._rank_cache_stats = None
		self._rank_cache_call = None
		self._rank_cache_pending = None
		stats["hit_rate"] = (stats["hits"] / stats["lookups"]) if stats["lookups"] else None
		return stats

	def answer(self, question: str, topic_entities: List[Any], **kwargs) -> Dict[str, Any]:
		self._rank_cache_begin()
		try:
			result = super().answer(question, topic_entities, **kwargs)
		finally:
			stats = self._rank_cache_end()
		totals = getattr(self, "rank_cache_run_totals", None)
		if totals is None:
			totals = self.rank_cache_run_totals = {"questions": 0, "lookups": 0, "hits": 0, "misses": 0}
		totals["questions"] += 1
		for k in ("lookups", "hits", "misses"):
			totals[k] += stats[k]
		result = dict(result)
		result["rank_cache"] = {"method": METHOD_ID, "scope": "question", **stats}
		return result

	# ------------------------------------------------- the one intervention
	def _llm_generate_for(self, operation_label, **extra_meta):
		inner = super()._llm_generate_for(operation_label, **extra_meta)
		if operation_label != OperationLabel.LLM_RELATION_RANK:
			return inner

		def generate(user_prompt, temperature, max_tokens, system_prompt=None):
			cache = getattr(self, "_rank_cache", None)
			if cache is None:  # outside a question: behave exactly like baseline
				return inner(user_prompt, temperature=temperature, max_tokens=max_tokens,
				             system_prompt=system_prompt)
			key = self._rank_request_key(user_prompt, temperature, max_tokens, system_prompt)
			stats = self._rank_cache_stats
			stats["lookups"] += 1
			cached = cache.get(key)
			if cached is not None:
				stats["hits"] += 1
				self._rank_cache_call = {"status": "hit", "key": key,
				                         "source_trace_id": cached["trace_id"]}
				self._rank_cache_pending = None
				# No model request is made, so no inference and no energy event.
				return cached["response"], {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0}
			stats["misses"] += 1
			self._rank_cache_call = {"status": "miss", "key": key}
			self._rank_cache_pending = None
			response, usage = inner(user_prompt, temperature=temperature, max_tokens=max_tokens,
			                        system_prompt=system_prompt)
			if response:
				# Candidate only: committed once relation_prune reports success.
				self._rank_cache_pending = {"key": key, "user_prompt": user_prompt,
				                            "response": response}
			return response, usage

		return generate

	def _record_prompt_history(
		self,
		prompt_history: List[Dict[str, Any]],
		type_name: str,
		prompt: str,
		response: Optional[str],
		parsed_result: Any,
		usage: Dict[str, Any],
		candidate_size: int = -1,
		filtered_candidate_size: int = -1,
		warnings: Optional[List[str]] = None,
	) -> None:
		super()._record_prompt_history(
			prompt_history=prompt_history, type_name=type_name, prompt=prompt,
			response=response, parsed_result=parsed_result, usage=usage,
			candidate_size=candidate_size, filtered_candidate_size=filtered_candidate_size,
			warnings=warnings)
		if type_name != "relation_prune" or getattr(self, "_rank_cache", None) is None:
			return
		call, pending = self._rank_cache_call, self._rank_cache_pending
		self._rank_cache_call = self._rank_cache_pending = None
		if call is None:
			return
		record = prompt_history[-1]
		audit = {"status": call["status"], "request_sha256": self._key_sha256(call["key"])}
		if call["status"] == "hit":
			audit["source_trace_id"] = call["source_trace_id"]
		# relation_prune returns a non-empty selected list exactly when it succeeds.
		succeeded = bool(parsed_result)
		if (call["status"] == "miss" and succeeded and pending is not None
		        and pending["key"] == call["key"] and pending["user_prompt"] == prompt):
			self._rank_cache[pending["key"]] = {"response": pending["response"],
			                                    "trace_id": record.get("trace_id")}
			audit["stored"] = True
		record["rank_cache"] = audit
