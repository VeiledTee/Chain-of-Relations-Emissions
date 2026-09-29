"""pog-rank-cache: exact per-question relation_rank memoization (C2 ablation).

Drives the real, unchanged shared tool (tools/relation_prune.py) and the real
PoG control flow with the fake LLM / fake KG of test_tog_pog_instrumentation,
so a "call" here is a real request reaching the fake chat-completions client.
"""

import copy
import os
import sys
import unittest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "tests"))
sys.path.insert(0, os.path.join(ROOT, "measurement"))

import export_outcomes  # noqa: E402

from test_tog_pog_instrumentation import (  # noqa: E402
	EventsHarness, FakeResponse, FakeUsage, StageAwareCompletions, make_pog_agent)

from chain_of_relations import run as run_module  # noqa: E402
from chain_of_relations.energy_taxonomy import OperationLabel  # noqa: E402
from chain_of_relations.methods.pog.agent import PoGAgent  # noqa: E402
from chain_of_relations.methods.pog_rank_cache import audit  # noqa: E402
from chain_of_relations.methods.pog_rank_cache.agent import (  # noqa: E402
	METHOD_ID, PoGRankCacheAgent)
from chain_of_relations.schema import Entity, Relation  # noqa: E402
from chain_of_relations.tools.relation_prune import RelationPruneInput, relation_prune  # noqa: E402

RANK_REPLY = '{"rational": "r", "relation_by_relevance": ["r.a", "r.b"], "relevance_score": [0.9, 0.5]}'


class CountingCompletions:
	"""Fake chat-completions client: scripted replies, every request recorded."""

	def __init__(self, replies=None):
		self.replies = list(replies or [RANK_REPLY])
		self.requests = []

	def create(self, **kwargs):
		self.requests.append(kwargs)
		item = self.replies[min(len(self.requests) - 1, len(self.replies) - 1)]
		if isinstance(item, Exception):
			raise item
		return FakeResponse(item, FakeUsage(40, 12))


def make_rank_cache_agent(ee, completions=None, **overrides):
	base = make_pog_agent(ee, completions, **overrides)
	agent = PoGRankCacheAgent.__new__(PoGRankCacheAgent)   # __init__ would load a model
	agent.__dict__.update(base.__dict__)
	return agent


def rank(agent, prompt_history, question="who knows Alice?", head=("r.a", "r.b"), tail=(),
         render_mode="unified", relation_chain=(), temperature=None, max_tokens=None,
         system_prompt="__agent__", entity=("m.alice", "Alice")):
	"""One relation_rank exactly as PoG's call site issues it, then its record."""
	out = relation_prune(
		RelationPruneInput(
			question=question,
			topic_entity=Entity(id=entity[0], name=entity[1]),
			relation_chain=list(relation_chain),
			head_relations=list(head),
			tail_relations=list(tail),
			relation_display_names={r: r for r in list(head) + list(tail)},
			user_prompt=agent._get_user_prompt("relation_pruning"),
			top_k=agent.relation_width,
			sample_relation_threshold=agent.sample_relation_threshold,
			system_prompt=(agent._get_system_prompt("relation_pruning")
			               if system_prompt == "__agent__" else system_prompt),
			render_mode=render_mode,
			temperature=agent.temperature_exploration if temperature is None else temperature,
			max_tokens=agent.max_token if max_tokens is None else max_tokens,
		),
		llm_generate=agent._llm_generate_for(OperationLabel.LLM_RELATION_RANK, frontier_index=0),
	)
	agent._record_prompt_history(
		prompt_history=prompt_history, type_name="relation_prune", prompt=out.prompt,
		response=out.response,
		parsed_result=[{"id": r.id, "left": r.left, "score": r.score} for r in out.selected_relations],
		usage=out.usage, candidate_size=out.candidate_size,
		filtered_candidate_size=out.pruned_candidate_size, warnings=out.warnings)
	return out


def picked(out):
	return [(r.id, r.left, r.score) for r in out.selected_relations]


class RankCacheUnit(EventsHarness):

	def setUp(self):
		super().setUp()
		self.completions = CountingCompletions()
		self.agent = make_rank_cache_agent(self.ee, self.completions)
		self.agent._rank_cache_begin()
		self.history = []

	def stats(self):
		return dict(self.agent._rank_cache_stats)

	def rank_events(self):
		return [e for e in self.llm_events() if e["operation_label"] == "llm:relation_rank"]

	# A
	def test_first_request_is_a_miss_and_runs_once(self):
		out = rank(self.agent, self.history)
		self.assertTrue(out.success)
		self.assertEqual(picked(out), [("r.a", True, 0.9), ("r.b", True, 0.5)])
		self.assertEqual(len(self.completions.requests), 1)
		self.assertEqual(self.stats(), {"lookups": 1, "hits": 0, "misses": 1})
		self.assertEqual(self.history[-1]["rank_cache"]["status"], "miss")
		self.assertTrue(self.history[-1]["rank_cache"]["stored"])

	# B
	def test_exact_repeat_is_a_hit_without_inference(self):
		first = rank(self.agent, self.history)
		second = rank(self.agent, self.history)
		self.assertEqual(len(self.completions.requests), 1)
		self.assertEqual(picked(second), picked(first))
		self.assertEqual(second.response, first.response)
		self.assertEqual(self.stats(), {"lookups": 2, "hits": 1, "misses": 1})
		hit = self.history[-1]["rank_cache"]
		self.assertEqual(hit["status"], "hit")
		self.assertEqual(hit["request_sha256"], self.history[0]["rank_cache"]["request_sha256"])
		self.assertEqual(hit["source_trace_id"], self.history[0]["trace_id"])
		# no inference -> exactly one relation_rank energy event, and the hit
		# record reports zero tokens rather than copying the first call's
		self.assertEqual(len(self.rank_events()), 1)
		self.assertEqual((self.history[-1]["input_tokens"], self.history[-1]["output_tokens"]), (0, 0))

	# C
	def test_candidate_order_matters(self):
		rank(self.agent, self.history, head=("r.a", "r.b"))
		rank(self.agent, self.history, head=("r.b", "r.a"))
		self.assertEqual(len(self.completions.requests), 2)
		self.assertEqual(self.stats()["hits"], 0)

	# D
	def test_question_and_subquestion_context_matter(self):
		rank(self.agent, self.history, question="who knows Alice?")
		rank(self.agent, self.history, question="who knows Alice?\nSubobjectives: ['a']")
		rank(self.agent, self.history, question="who knows Alice?\nSubobjectives: ['b']")
		self.assertEqual(len(self.completions.requests), 3)
		self.assertEqual(self.stats()["hits"], 0)

	# E
	def test_path_and_direction_matter_when_they_change_the_request(self):
		rank(self.agent, self.history, render_mode="directional", head=("r.a", "r.b"))
		rank(self.agent, self.history, render_mode="directional", head=("r.a",), tail=("r.b",))
		rank(self.agent, self.history, render_mode="directional", head=("r.a", "r.b"),
		     relation_chain=[Relation(id="r.prev", left=True)])
		self.assertEqual(len(self.completions.requests), 3)
		self.assertEqual(self.stats()["hits"], 0)

	def test_every_generation_parameter_is_part_of_the_key(self):
		rank(self.agent, self.history)
		rank(self.agent, self.history, temperature=0.7)
		rank(self.agent, self.history, max_tokens=99)
		rank(self.agent, self.history, system_prompt=None)
		self.agent.llm_api.model_name = "another-model"
		rank(self.agent, self.history)
		self.assertEqual(len(self.completions.requests), 5)
		self.assertEqual(self.stats()["hits"], 0)

	def test_unified_hit_is_reparsed_against_the_current_directions(self):
		# PoG's unified prompt omits direction, so these two requests are
		# identical (a legitimate hit); the reply is re-parsed for the second
		# call's own head/tail split, exactly as baseline would parse it.
		rank(self.agent, self.history, head=("r.a", "r.b"), tail=())
		second = rank(self.agent, self.history, head=("r.a",), tail=("r.b",))
		self.assertEqual(len(self.completions.requests), 1)
		self.assertEqual(picked(second), [("r.a", True, 0.9), ("r.b", False, 0.5)])

	# F
	def test_no_reuse_across_questions(self):
		rank(self.agent, self.history)
		first_stats = self.agent._rank_cache_end()
		self.agent._rank_cache_begin()
		rank(self.agent, self.history)
		self.assertEqual(len(self.completions.requests), 2)
		self.assertEqual((first_stats["hits"], self.stats()["hits"]), (0, 0))

	def test_no_cache_outside_a_question(self):
		self.agent._rank_cache_end()
		rank(self.agent, self.history)
		rank(self.agent, self.history)
		self.assertEqual(len(self.completions.requests), 2)
		self.assertNotIn("rank_cache", self.history[-1])

	# G
	def test_mutating_results_cannot_corrupt_the_cache(self):
		first = rank(self.agent, self.history)
		expected = picked(first)
		first.selected_relations[0].score = -1.0
		first.selected_relations.clear()
		hit = rank(self.agent, self.history)
		self.assertEqual(picked(hit), expected)
		hit.selected_relations[0].id = "tampered"
		again = rank(self.agent, self.history)
		self.assertEqual(picked(again), expected)
		self.assertEqual(len(self.completions.requests), 1)

	# H
	def test_unparseable_reply_is_not_cached(self):
		self.completions.replies = ['{"nothing": "useful"}', RANK_REPLY]
		bad = rank(self.agent, self.history)
		self.assertFalse(bad.success)
		self.assertNotIn("stored", self.history[-1]["rank_cache"])
		good = rank(self.agent, self.history)            # same request: must run again
		self.assertTrue(good.success)
		self.assertEqual(len(self.completions.requests), 2)
		self.assertEqual(self.stats(), {"lookups": 2, "hits": 0, "misses": 2})

	def test_empty_or_failed_replies_are_not_cached(self):
		self.agent.llm_api.max_retries = 1
		self.completions.replies = [None, None, None, RANK_REPLY]
		failed = rank(self.agent, self.history)
		self.assertFalse(failed.success)
		self.assertEqual(self.agent._rank_cache, {})
		self.completions.replies = [RuntimeError("down")] * 3 + [RANK_REPLY]
		self.completions.requests.clear()
		raised = rank(self.agent, self.history)
		self.assertFalse(raised.success)
		self.assertEqual(self.agent._rank_cache, {})


class BaselineAndIdentity(EventsHarness):

	# I
	def test_baseline_pog_has_no_cache(self):
		completions = StageAwareCompletions(self.ee, "pog")
		agent = make_pog_agent(self.ee, completions)
		self.ee.set_question("q1")
		result = agent.answer("who knows Alice?", [Entity(id="m.alice", name="Alice")])
		records = [r for r in result["prompt_history"] if r["type"] == "relation_prune"]
		self.assertNotIn("rank_cache", result)
		self.assertTrue(all("rank_cache" not in r for r in records))
		self.assertEqual(completions.calls_by_label.get("llm:relation_rank", 0), len(records))
		self.assertNotIsInstance(agent, PoGRankCacheAgent)
		self.assertEqual(type(agent)._llm_generate_for.__qualname__, "PoGAgent._llm_generate_for")

	# J
	def test_method_identity_is_separate(self):
		self.assertEqual(METHOD_ID, "pog-rank-cache")
		args = run_module.build_parser().parse_args(["--method", "pog-rank-cache"])
		self.assertEqual(args.method, "pog-rank-cache")
		self.assertIs(run_module.load_agent_class("pog-rank-cache"), PoGRankCacheAgent)
		self.assertIs(run_module.load_agent_class("pog"), PoGAgent)
		self.assertTrue(issubclass(PoGRankCacheAgent, PoGAgent))

	def test_step_history_keeps_audit_only_for_the_variant(self):
		base = {"type": "relation_prune", "prompt": "p", "trace_id": 3}
		baseline_steps = run_module.build_step_history([dict(base)], [])
		self.assertNotIn("rank_cache", baseline_steps[0])
		self.assertNotIn("trace_id", baseline_steps[0])
		variant_steps = run_module.build_step_history(
			[dict(base, rank_cache={"status": "hit", "source_trace_id": 1})], [])
		self.assertEqual(variant_steps[0]["rank_cache"]["status"], "hit")
		self.assertEqual(variant_steps[0]["trace_id"], 3)


class DeterministicEquivalence(EventsHarness):
	"""Item 16: same deterministic replies -> identical PoG behaviour."""

	QUESTIONS = (("q1", "who knows Alice?"), ("q2", "who knows Alice?"))

	def run_questions(self, agent):
		outs = []
		for qid, question in self.QUESTIONS:
			self.ee.set_question(qid)
			outs.append(agent.answer(question, [Entity(id="m.alice", name="Alice")]))
		return outs

	@staticmethod
	def behaviour(result):
		steps = [(r["type"], r["prompt"], r["response"], r["parsed_result"])
		         for r in result["prompt_history"]]
		return (result["action"], result["results"], result["reasoning_chains"], steps)

	def test_same_search_and_answers_with_fewer_rank_inferences(self):
		base_llm = StageAwareCompletions(self.ee, "pog")
		base = self.run_questions(make_pog_agent(self.ee, base_llm))
		cache_llm = StageAwareCompletions(self.ee, "pog")
		cached = self.run_questions(make_rank_cache_agent(self.ee, cache_llm))

		for b, c in zip(base, cached):
			self.assertEqual(self.behaviour(b), self.behaviour(c))

		base_rank = base_llm.calls_by_label["llm:relation_rank"]
		cache_rank = cache_llm.calls_by_label["llm:relation_rank"]
		hits = sum(r["rank_cache"]["hits"] for r in cached)
		self.assertGreater(hits, 0, "the harness must actually produce exact duplicates")
		self.assertEqual(cache_rank, base_rank - hits)
		# every other stage makes exactly the same model calls
		other = {k: v for k, v in base_llm.calls_by_label.items() if k != "llm:relation_rank"}
		self.assertEqual(other, {k: v for k, v in cache_llm.calls_by_label.items()
		                         if k != "llm:relation_rank"})
		for r in cached:
			self.assertEqual(r["rank_cache"]["lookups"], r["rank_cache"]["hits"] + r["rank_cache"]["misses"])


class AnalysisAlignment(EventsHarness):
	"""Saved steps vs energy events: hits have a step but no event."""

	def run_cached(self):
		agent = make_rank_cache_agent(self.ee, StageAwareCompletions(self.ee, "pog"))
		per_question = []
		for qid, question in DeterministicEquivalence.QUESTIONS:
			self.ee.set_question(qid)
			result = agent.answer(question, [Entity(id="m.alice", name="Alice")])
			steps = run_module.build_step_history(result["prompt_history"], result.get("sparql_history", []))
			per_question.append((qid, steps))
		return per_question

	def test_hit_filtered_steps_pair_one_to_one_with_rank_events(self):
		per_question = self.run_cached()
		events = self.read_events()
		total_hits = 0
		for qid, steps in per_question:
			q_events = [e for e in events if e["question_id"] == qid]
			rank_steps = [s for s in steps if s.get("operation_type") == "relation_prune"]
			rank_events = [e for e in q_events if e["operation_label"] == "llm:relation_rank"]
			hits = sum(audit.is_cache_hit(s) for s in rank_steps)
			total_hits += hits
			# naive positional pairing is misaligned exactly when a hit occurred
			self.assertEqual(len(rank_steps) - hits, len(rank_events))
			pairs = audit.pair_rank_steps_with_events(steps, q_events)
			self.assertEqual(len(pairs), len(rank_events))
			for step, event in pairs:
				self.assertNotEqual(step["rank_cache"]["status"], "hit")
				self.assertEqual(step["input_tokens"], event["input_tokens"])
				self.assertEqual(step["output_tokens"], event["output_tokens"])
		self.assertGreater(total_hits, 0, "the harness must actually produce hits")

	def test_pairing_refuses_count_mismatch(self):
		step = {"step_type": "llm", "operation_type": "relation_prune", "rank_cache": {"status": "miss"}}
		with self.assertRaises(ValueError):
			audit.pair_rank_steps_with_events([step], [])

	def test_inference_steps_drops_only_hits_and_non_llm(self):
		steps = [{"step_type": "sparql"},
		         {"step_type": "llm", "operation_type": "relation_prune", "rank_cache": {"status": "hit"}},
		         {"step_type": "llm", "operation_type": "relation_prune", "rank_cache": {"status": "miss"}},
		         {"step_type": "llm", "operation_type": "entity_prune"}]
		self.assertEqual(audit.inference_steps(steps), steps[2:])


class OutcomeExportCompatibility(unittest.TestCase):
	"""The canonical evaluator ignores the ablation's extra predict.jsonl key."""

	def test_rank_cache_key_does_not_change_scores(self):
		import json
		import tempfile
		record = {"id": "q1", "results": [{"id": "", "name": "Paris"}], "action": "answer",
		          "gold_answer": [{"name": "Paris"}]}
		rows = {}
		with tempfile.TemporaryDirectory() as tmp:
			for name, rec in (("plain", record),
			                  ("cached", dict(record, rank_cache={"hits": 2, "misses": 3}))):
				path = os.path.join(tmp, f"{name}.jsonl")
				with open(path, "w") as f:
					f.write(json.dumps(rec) + "\n")
				rows[name] = export_outcomes.score(path, "cwq", "PoG")
		self.assertEqual(rows["plain"], rows["cached"])
		self.assertEqual(rows["cached"][0]["hit1"], 1)


if __name__ == "__main__":
	unittest.main()
