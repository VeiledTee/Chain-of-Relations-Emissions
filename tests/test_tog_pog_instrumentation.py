"""Tests for ToG and PoG semantic energy instrumentation (schema v1).

The CoR tests in tests/test_energy_measurement.py are the reference; these
mirror them for the two paradigms migrated afterwards. Pure unit tests: no LLM
endpoint, no SPARQL endpoint, no GPU, no sentence-transformers.

Both agents are driven through their real answer() control flow with a fake
LLM client and a fake KG backend that emits KG events exactly where the real
Freebase backend does, so "did we double-count the KG" is a question these
tests can actually answer.

Run:
  python -m pytest -q tests/test_tog_pog_instrumentation.py
  python -m unittest discover -s tests -p "test_*.py" -v
"""

import importlib
import json
import os
import sys
import tempfile
import types
import unittest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from chain_of_relations import energy_taxonomy as tax
from chain_of_relations.energy_taxonomy import OperationLabel, Status
from chain_of_relations.schema import Entity

REQUIRED_KEYS = (
	"schema_version", "event_id",
	"run_id", "question_id", "dataset", "paradigm",
	"iteration", "traversal_depth", "step_index",
	"operation_type", "operation_label",
	"start_timestamp", "end_timestamp", "duration_s",
	"gpu_energy_j", "cpu_package_energy_j", "dram_energy_j",
	"input_tokens", "output_tokens",
	"status",
	"model_name", "model_revision", "git_commit", "hardware_id",
	"meta",
)


# --------------------------------------------------------------------------
# fakes
# --------------------------------------------------------------------------

class FakeUsage:
	def __init__(self, prompt_tokens, completion_tokens):
		self.prompt_tokens = prompt_tokens
		self.completion_tokens = completion_tokens
		self.total_tokens = prompt_tokens + completion_tokens


class FakeMessage:
	def __init__(self, content):
		self.content = content


class FakeChoice:
	def __init__(self, content):
		self.message = FakeMessage(content)


class FakeResponse:
	def __init__(self, content, usage=None):
		self.choices = [FakeChoice(content)] if content is not None else []
		self.usage = usage


class StageAwareCompletions:
	"""Answers according to the semantic stage the agent declared.

	The fake reads energy_events.current_operation() -- the label the agent
	itself put in scope -- and replies with something that stage's parser
	accepts. That is a test convenience, and the opposite of what production
	does: no shipped code ever derives behaviour from a label, and nothing
	anywhere derives a label from prompt text.
	"""

	def __init__(self, ee, paradigm, overrides=None):
		self.ee = ee
		self.paradigm = paradigm
		self.overrides = overrides or {}
		self.calls = 0
		self.calls_by_label = {}
		self.prompts = []

	def create(self, **kwargs):
		self.calls += 1
		label = self.ee.current_operation() or "?"
		index = self.calls_by_label.get(label, 0)
		self.calls_by_label[label] = index + 1
		self.prompts.append(kwargs.get("messages", [{}])[-1].get("content", ""))

		scripted = self.overrides.get(label)
		if scripted is not None:
			item = scripted[min(index, len(scripted) - 1)]
			if isinstance(item, Exception):
				raise item
			return FakeResponse(item, FakeUsage(30 + index, 5 + index))
		return FakeResponse(self._reply(label, index), FakeUsage(30 + index, 5 + index))

	def _reply(self, label, index):
		meta = self.ee.current_event_meta()
		if label == OperationLabel.LLM_RELATION_RANK.value:
			return '{"relation_by_relevance": ["r.knows"], "relevance_score": [0.9]}'
		if label == OperationLabel.LLM_ENTITY_PRUNE.value:
			if meta.get("prune_strategy") == tax.PRUNE_STRATEGY_CONDITION:
				return '["Alice"]'
			return '{"entity_by_relevance": ["Alice", "Bob"], "entity_score": [0.7, 0.3]}'
		if label == OperationLabel.LLM_REASON.value:
			if self.paradigm == "pog":
				# Sufficient only on the second round, so a realistic PoG
				# trajectory runs its reverse-retrieval stages first.
				if index == 0:
					return '{"A": {"Sufficient": "no", "Answer": "null"}, "R": "keep going"}'
				return '{"A": {"Sufficient": "yes", "Answer": "Alice"}, "R": "found"}'
			if index == 0:
				return '{"decision": "no", "answer": []}'
			return '{"decision": "yes", "answer": ["Alice"]}'
		if label == OperationLabel.LLM_SUBQUESTION_DECOMPOSE.value:
			return '["who knows Alice", "when"]'
		if label == OperationLabel.LLM_MEMORY_UPDATE.value:
			return '{"known": ["Alice"]}'
		if label == OperationLabel.LLM_REVERSE_RETRIEVAL_DECISION.value:
			return ('{"need_reverse": true, "reverse_entity_names": ["Bob"], '
			        '"reason": "one more hop is needed here"}')
		if label == OperationLabel.LLM_REVERSE_ENTITY_SELECT.value:
			return '["Bob"]'
		if label == OperationLabel.LLM_DIRECT_ANSWER.value:
			return '{"answer": ["closed book"]}'
		return '{"answer": ["unlabelled"]}'


class FakeClient:
	def __init__(self, completions):
		self.chat = type("chat", (), {})()
		self.chat.completions = completions


def make_llm_api(completions):
	os.environ.setdefault("OPENAI_API_KEY", "test-key")
	from chain_of_relations.llm_api import LLMAPI
	api = LLMAPI.__new__(LLMAPI)
	api.model_name = "test-model"
	api.api_key = "test-key"
	api.base_url = ""
	api.timeout = 1.0
	api.max_retries = 3
	api.retry_interval = 0.0
	api.client = FakeClient(completions)
	return api


class FakeBackend:
	"""A KG backend instrumented where the real Freebase backend is.

	Every SPARQL execution records exactly one event through
	energy_events.record_sparql, and id2entity_names records one per batch --
	the same boundary db_func uses. Nothing else in the stack may add a second
	event for the same physical call.
	"""

	name = "fake"
	default_entity_prefix = "m."

	def __init__(self, ee, entities=("m.alice", "m.bob")):
		self.ee = ee
		self.entities = list(entities)
		self.queries = []
		self.name_lookups = 0

	# --- query construction ---
	def build_sparql_relations(self, query):
		return "SELECT ?relation WHERE { " + query + " }"

	def build_sparql_entities(self, query):
		return "SELECT ?targetEntity WHERE { " + query + " }"

	def format_entity_node(self, entity_id):
		return "ns:" + str(entity_id)

	def format_relation_node(self, relation_id):
		return "ns:" + str(relation_id)

	# --- execution: the one instrumented boundary ---
	def execute_sparql_with_meta(self, sparql_txt):
		start = self.ee.mark()
		self.queries.append(sparql_txt)
		if "?relation" in sparql_txt:
			rows = [{"relation": {"value": "r.knows"}}]
		else:
			rows = [{"targetEntity": {"value": entity}} for entity in self.entities]
		self.ee.record_sparql(sparql_txt, start, self.ee.mark(), rows=len(rows))
		return {"rows": rows, "status": "ok", "timed_out": False}

	def execute_sparql(self, sparql_txt):
		return self.execute_sparql_with_meta(sparql_txt)["rows"]

	# --- parsing ---
	def parse_relations(self, rows):
		return [row["relation"]["value"] for row in rows]

	def parse_entities(self, rows):
		return [], [row["targetEntity"]["value"] for row in rows]

	def is_unnecessary_relation(self, relation):
		return False

	def normalize_uri(self, value):
		return str(value)

	def is_entity_id(self, value):
		return str(value).startswith("m.")

	def relation_id2label(self, relation_id):
		return str(relation_id)

	def relation_ids2labels(self, relation_ids):
		return {str(r): str(r) for r in relation_ids}

	# --- name resolution: instrumented, one event per round trip ---
	def id2entity_names(self, entity_ids):
		start = self.ee.mark()
		self.name_lookups += 1
		mapping = {entity_id: self._name(entity_id) for entity_id in entity_ids}
		self.ee.record_sparql("SELECT ?targetEntity type.object.name", start,
		                      self.ee.mark(),
		                      label=OperationLabel.KG_ID2NAME,
		                      entity_count=len(entity_ids))
		return mapping

	def id2entity_name_or_type(self, entity_id):
		return self.id2entity_names([entity_id])[entity_id]

	@staticmethod
	def _name(entity_id):
		return {"m.alice": "Alice", "m.bob": "Bob"}.get(str(entity_id), "Carol")


def make_tog_agent(ee, completions=None, **overrides):
	from chain_of_relations.methods.tog.agent import ToGAgent
	agent = ToGAgent.__new__(ToGAgent)
	agent.model_name = "test-model"
	agent.relation_width = 2
	agent.entity_width = 2
	agent.width = 2
	agent.depth = 2
	agent.sample_relation_threshold = 500
	agent.sample_entity_threshold = 500
	agent.remove_unnecessary_rel = True
	agent.temperature_exploration = 0.3
	agent.temperature_reasoning = 0.1
	agent.max_token = 256
	agent.kg_backend = FakeBackend(ee)
	agent.llm_api = make_llm_api(
		completions or StageAwareCompletions(ee, "tog"))
	agent.prompt_file = os.path.join(
		ROOT, "chain_of_relations", "methods", "tog", "prompt.yml")
	agent.prompts = agent._load_prompts(agent.prompt_file)
	agent._trace_seq = 0
	for key, value in overrides.items():
		setattr(agent, key, value)
	return agent


def make_pog_agent(ee, completions=None, **overrides):
	from chain_of_relations.methods.pog.agent import PoGAgent
	agent = PoGAgent.__new__(PoGAgent)          # __init__ would load a model
	agent.model_name = "test-model"
	agent.relation_width = 2
	agent.entity_width = 2
	agent.width = 2
	agent.depth = 2
	agent.sample_relation_threshold = 500
	agent.sample_entity_threshold = 500
	agent.remove_unnecessary_rel = True
	agent.temperature_exploration = 0.3
	agent.temperature_reasoning = 0.3
	agent.max_token = 256
	agent.max_reverse_rounds = 2
	agent.kg_backend = FakeBackend(ee)
	agent.llm_api = make_llm_api(
		completions or StageAwareCompletions(ee, "pog"))
	agent.prompt_file = os.path.join(
		ROOT, "chain_of_relations", "methods", "pog", "prompt.yml")
	agent.prompts = agent._load_prompts(agent.prompt_file)
	agent._trace_seq = 0
	for key, value in overrides.items():
		setattr(agent, key, value)
	return agent


class EventsHarness(unittest.TestCase):
	"""Each test gets a fresh energy_events module writing to a temp file."""

	def setUp(self):
		self._tmp = tempfile.TemporaryDirectory()
		self.events_path = os.path.join(self._tmp.name, "events.jsonl")
		os.environ["ENERGY_EVENTS_FILE"] = self.events_path
		import chain_of_relations.energy_events as ee
		self.ee = importlib.reload(ee)
		self.ee.configure(events_file=self.events_path, run_id="run-test",
		                  dataset="webqsp", model_name="test-model")

	def tearDown(self):
		self.ee.close()
		os.environ.pop("ENERGY_EVENTS_FILE", None)
		self._tmp.cleanup()

	def read_events(self):
		if not os.path.exists(self.events_path):
			return []
		with open(self.events_path) as f:
			return [json.loads(line) for line in f if line.strip()]

	def llm_events(self):
		return [e for e in self.read_events() if e["operation_type"] == "llm"]

	def labels(self):
		return [e["operation_label"] for e in self.read_events()]

	def run_tog(self, agent=None, entities=(("m.alice", "Alice"),), **kwargs):
		agent = agent or make_tog_agent(self.ee)
		self.ee.set_question("q-tog-1")
		topics = [Entity(id=i, name=n) for i, n in entities]
		result = agent.answer("who knows Alice?", topics, **kwargs)
		return agent, result

	def run_pog(self, agent=None, entities=(("m.alice", "Alice"),), **kwargs):
		agent = agent or make_pog_agent(self.ee)
		self.ee.set_question("q-pog-1")
		topics = [Entity(id=i, name=n) for i, n in entities]
		result = agent.answer("who knows Alice?", topics, **kwargs)
		return agent, result


# --------------------------------------------------------------------------
# ToG
# --------------------------------------------------------------------------

class TestToGSemanticLabels(EventsHarness):

	def test_tog_emits_its_semantic_stages_and_never_llm_generate(self):
		self.run_tog()
		labels = {e["operation_label"] for e in self.llm_events()}
		self.assertNotIn("llm:generate", labels,
		                 "ToG LLM work collapsed to the generic label")
		self.assertEqual(labels, {"llm:relation_rank", "llm:entity_prune",
		                          "llm:reason"})

	def test_every_tog_label_is_in_the_declared_tog_vocabulary(self):
		self.run_tog()
		for event in self.read_events():
			self.assertIn(event["operation_label"],
			              {l.value for l in tax.TOG_LABELS},
			              event["operation_label"])

	def test_tog_events_are_schema_v1_and_carry_run_context(self):
		self.run_tog()
		events = self.read_events()
		self.assertTrue(events)
		for event in events:
			for key in REQUIRED_KEYS:
				self.assertIn(key, event, key)
			self.assertEqual(event["schema_version"], tax.SCHEMA_VERSION)
			self.assertEqual(event["question_id"], "q-tog-1")
			self.assertEqual(event["run_id"], "run-test")
			self.assertEqual(event["dataset"], "webqsp")
			self.assertEqual(event["status"], Status.OK.value)
			self.assertNotIn("unknown_label", event["meta"])

	def test_operation_type_agrees_with_label(self):
		self.run_tog()
		for event in self.read_events():
			self.assertEqual(event["operation_type"],
			                 tax.type_of(event["operation_label"]))


class TestToGTrajectoryContext(EventsHarness):

	def test_iteration_is_the_search_round_and_depth_is_the_graph_depth(self):
		self.run_tog()
		events = [e for e in self.read_events() if e["iteration"] is not None]
		iterations = [e["iteration"] for e in events]
		self.assertEqual(iterations, sorted(iterations),
		                 "iteration must never decrease within a question")
		# Two depth rounds ran: reasoning said "no" first, "yes" second.
		self.assertEqual(sorted(set(iterations)), [0, 1])
		for event in events:
			self.assertEqual(event["traversal_depth"], event["iteration"] + 1,
			                 "ToG depth round n is graph depth n")

	def test_step_index_is_globally_monotonic_within_the_question(self):
		self.run_tog()
		steps = [e["step_index"] for e in self.read_events()]
		self.assertEqual(steps, list(range(len(steps))))

	def test_repeated_work_inside_a_round_is_metadata_not_a_new_iteration(self):
		agent, _ = self.run_tog(
			entities=(("m.alice", "Alice"), ("m.bob", "Bob")))
		round_zero = [e for e in self.llm_events() if e["iteration"] == 0]
		ranks = [e for e in round_zero
		         if e["operation_label"] == "llm:relation_rank"]
		self.assertEqual(len(ranks), 2, "one ranking call per frontier entity")
		self.assertEqual([e["meta"]["frontier_index"] for e in ranks], [0, 1])
		self.assertEqual({e["iteration"] for e in ranks}, {0},
		                 "frontier expansion must not advance the iteration")

	def test_branch_index_is_attached_to_entity_pruning_events(self):
		self.run_tog()
		prunes = [e for e in self.llm_events()
		          if e["operation_label"] == "llm:entity_prune"]
		self.assertTrue(prunes)
		for event in prunes:
			self.assertIn("branch_index", event["meta"])
			self.assertEqual(event["meta"]["prune_strategy"],
			                 tax.PRUNE_STRATEGY_SCORE)

	def test_tokens_and_attempts_are_recorded_on_tog_llm_events(self):
		self.run_tog()
		for event in self.llm_events():
			self.assertIsInstance(event["input_tokens"], int)
			self.assertIsInstance(event["output_tokens"], int)
			self.assertEqual(event["meta"]["attempts"], 1)
			self.assertIn("tool_attempt", event["meta"])


class TestToGFallbacks(EventsHarness):

	def _fallback_events(self):
		return [e for e in self.read_events()
		        if e["operation_label"] == "llm:direct_answer"]

	def test_no_topic_entities_is_a_pre_loop_fallback_with_null_depth(self):
		agent = make_tog_agent(self.ee)
		self.ee.set_question("q-tog-empty")
		agent.answer("q?", [])
		event, = self._fallback_events()
		self.assertEqual(event["meta"]["fallback_reason"],
		                 tax.FALLBACK_NO_TOPIC_ENTITIES)
		self.assertTrue(event["meta"]["fallback"])
		self.assertIsNone(event["traversal_depth"])
		self.assertEqual(event["iteration"], 0)

	def test_no_branches_keeps_the_depth_where_the_search_gave_up(self):
		agent = make_tog_agent(self.ee)
		agent.kg_backend.entities = []          # entity_search finds nothing
		self.run_tog(agent=agent)
		event, = self._fallback_events()
		self.assertEqual(event["meta"]["fallback_reason"],
		                 tax.FALLBACK_NO_BRANCHES)
		self.assertEqual(event["traversal_depth"], 1)
		self.assertEqual(event["iteration"], 0)

	def test_depth_exhausted_keeps_the_final_depth(self):
		completions = StageAwareCompletions(
			self.ee, "tog",
			overrides={"llm:reason": ['{"decision": "no", "answer": []}']})
		agent = make_tog_agent(self.ee, completions=completions)
		self.run_tog(agent=agent)
		event, = self._fallback_events()
		self.assertEqual(event["meta"]["fallback_reason"],
		                 tax.FALLBACK_DEPTH_EXHAUSTED)
		self.assertEqual(event["traversal_depth"], agent.depth)

	def test_every_fallback_reason_is_from_the_controlled_vocabulary(self):
		agent = make_tog_agent(self.ee)
		agent.kg_backend.entities = []
		self.run_tog(agent=agent)
		for event in self._fallback_events():
			self.assertIn(event["meta"]["fallback_reason"], tax.FALLBACK_REASONS)


# --------------------------------------------------------------------------
# PoG
# --------------------------------------------------------------------------

class TestPoGSemanticLabels(EventsHarness):

	def test_pog_emits_its_semantic_stages_and_never_llm_generate(self):
		self.run_pog()
		labels = {e["operation_label"] for e in self.llm_events()}
		self.assertNotIn("llm:generate", labels,
		                 "PoG LLM work collapsed to the generic label")
		self.assertEqual(labels, {
			"llm:subquestion_decompose", "llm:relation_rank",
			"llm:entity_prune", "llm:memory_update", "llm:reason",
			"llm:reverse_retrieval_decision", "llm:reverse_entity_select"})

	def test_every_pog_label_is_in_the_declared_pog_vocabulary(self):
		self.run_pog()
		for event in self.read_events():
			self.assertIn(event["operation_label"],
			              {l.value for l in tax.POG_LABELS},
			              event["operation_label"])

	def test_pog_events_are_schema_v1_and_carry_run_context(self):
		self.run_pog()
		events = self.read_events()
		self.assertTrue(events)
		for event in events:
			for key in REQUIRED_KEYS:
				self.assertIn(key, event, key)
			self.assertEqual(event["question_id"], "q-pog-1")
			self.assertEqual(event["status"], Status.OK.value)
			self.assertNotIn("unknown_label", event["meta"])
			self.assertEqual(event["operation_type"],
			                 tax.type_of(event["operation_label"]))

	def test_condition_pruning_reuses_the_shared_entity_prune_label(self):
		self.run_pog()
		prunes = [e for e in self.llm_events()
		          if e["operation_label"] == "llm:entity_prune"]
		self.assertTrue(prunes)
		for event in prunes:
			self.assertEqual(event["meta"]["prune_strategy"],
			                 tax.PRUNE_STRATEGY_CONDITION)
			self.assertIn("group_index", event["meta"])

	def test_reverse_stages_carry_the_reverse_round(self):
		self.run_pog()
		reverse = [e for e in self.llm_events()
		           if e["operation_label"].startswith("llm:reverse")]
		self.assertTrue(reverse)
		for event in reverse:
			self.assertIn("reverse_round", event["meta"])

	def _multi_cycle_agent(self):
		"""A PoG agent that never finds the answer, so it keeps reversing."""
		completions = StageAwareCompletions(
			self.ee, "pog",
			overrides={"llm:reason": [
				'{"A": {"Sufficient": "no", "Answer": "null"}, "R": "more"}']})
		return make_pog_agent(self.ee, completions=completions, depth=3,
		                      max_reverse_rounds=3)

	def test_decision_and_selection_share_one_reverse_cycle_id(self):
		"""reverse_round identifies the cycle, not the call.

		A decision to reverse-retrieve and the selection that answers it are
		two halves of one cycle, so grouping by reverse_round must give the
		cost of the whole cycle. The counter advances only once both halves
		have run, and the cycle is 0-based.
		"""
		self.run_pog(agent=self._multi_cycle_agent())
		decisions = [e for e in self.llm_events()
		             if e["operation_label"] == "llm:reverse_retrieval_decision"]
		selections = [e for e in self.llm_events()
		              if e["operation_label"] == "llm:reverse_entity_select"]
		self.assertGreaterEqual(len(decisions), 2,
		                        "need at least two reverse cycles to test this")
		self.assertEqual(len(decisions), len(selections))

		rounds = [e["meta"]["reverse_round"] for e in decisions]
		self.assertEqual(rounds, list(range(len(decisions))),
		                 "cycles are numbered 0, 1, 2 ...")
		for decision, selection in zip(decisions, selections):
			self.assertEqual(selection["meta"]["reverse_round"],
			                 decision["meta"]["reverse_round"],
			                 "both halves of one cycle share its identifier")
			self.assertLess(decision["step_index"], selection["step_index"])

	def test_reverse_round_groups_the_whole_cycle(self):
		"""The point of the shared id: one group per cycle, both halves in it."""
		self.run_pog(agent=self._multi_cycle_agent())
		by_round = {}
		for event in self.llm_events():
			if not event["operation_label"].startswith("llm:reverse"):
				continue
			by_round.setdefault(event["meta"]["reverse_round"], []).append(
				event["operation_label"])
		self.assertTrue(by_round)
		self.assertEqual(sorted(by_round), [0, 1, 2])
		for cycle, labels in by_round.items():
			self.assertEqual(sorted(labels),
			                 ["llm:reverse_entity_select",
			                  "llm:reverse_retrieval_decision"],
			                 f"cycle {cycle} must hold exactly its two halves")

	def test_a_cycle_capped_by_max_rounds_records_only_its_decision(self):
		"""When the round budget is spent the selection never runs.

		The decision still cost energy and is still recorded, under the cycle
		it belongs to. Its missing other half is a fact about the run, not a
		gap in the instrumentation.
		"""
		completions = StageAwareCompletions(
			self.ee, "pog",
			overrides={"llm:reason": [
				'{"A": {"Sufficient": "no", "Answer": "null"}, "R": "more"}']})
		agent = make_pog_agent(self.ee, completions=completions, depth=3,
		                       max_reverse_rounds=1)
		self.run_pog(agent=agent)
		decisions = [e["meta"]["reverse_round"] for e in self.llm_events()
		             if e["operation_label"] == "llm:reverse_retrieval_decision"]
		selections = [e["meta"]["reverse_round"] for e in self.llm_events()
		              if e["operation_label"] == "llm:reverse_entity_select"]
		self.assertEqual(selections, [0], "only one cycle was affordable")
		self.assertEqual(decisions[:1], [0])
		self.assertTrue(all(r >= 1 for r in decisions[1:]),
		                "later decisions keep advancing past the capped cycle")


class TestPoGTrajectoryContext(EventsHarness):

	def test_planning_runs_before_the_traversal_with_null_depth(self):
		self.run_pog()
		event, = [e for e in self.llm_events()
		          if e["operation_label"] == "llm:subquestion_decompose"]
		self.assertEqual(event["iteration"], 0)
		self.assertIsNone(event["traversal_depth"],
		                  "decomposition happens before any graph traversal")

	def test_depth_rounds_advance_iteration_and_depth_together(self):
		self.run_pog()
		traversal = [e for e in self.read_events()
		             if e["traversal_depth"] is not None]
		for event in traversal:
			self.assertEqual(event["traversal_depth"], event["iteration"],
			                 "PoG round n follows the iteration-0 planning step")
		iterations = [e["iteration"] for e in self.read_events()]
		self.assertEqual(iterations, sorted(iterations))
		self.assertGreaterEqual(max(iterations), 1)

	def test_step_index_is_globally_monotonic_within_the_question(self):
		self.run_pog()
		steps = [e["step_index"] for e in self.read_events()]
		self.assertEqual(steps, list(range(len(steps))))

	def test_frontier_index_marks_repeated_ranking_inside_one_round(self):
		self.run_pog(entities=(("m.alice", "Alice"), ("m.bob", "Bob")))
		first_round = [e for e in self.llm_events()
		               if e["operation_label"] == "llm:relation_rank"
		               and e["iteration"] == 1]
		self.assertEqual([e["meta"]["frontier_index"] for e in first_round],
		                 [0, 1])


class TestPoGFallbacks(EventsHarness):

	def _fallback_events(self):
		return [e for e in self.read_events()
		        if e["operation_label"] == "llm:direct_answer"]

	def test_no_topic_entities_is_pre_loop_with_null_depth(self):
		agent = make_pog_agent(self.ee)
		self.ee.set_question("q-pog-empty")
		agent.answer("q?", [])
		event, = self._fallback_events()
		self.assertEqual(event["meta"]["fallback_reason"],
		                 tax.FALLBACK_NO_TOPIC_ENTITIES)
		self.assertIsNone(event["traversal_depth"])

	def test_no_candidate_groups_keeps_the_current_depth(self):
		agent = make_pog_agent(self.ee)
		agent.kg_backend.entities = []
		self.run_pog(agent=agent)
		event, = self._fallback_events()
		self.assertEqual(event["meta"]["fallback_reason"],
		                 tax.FALLBACK_NO_CANDIDATE_GROUPS)
		self.assertEqual(event["traversal_depth"], 1)

	def test_depth_exhausted_keeps_the_final_depth(self):
		completions = StageAwareCompletions(
			self.ee, "pog",
			overrides={"llm:reason": [
				'{"A": {"Sufficient": "no", "Answer": "null"}, "R": "more"}']})
		agent = make_pog_agent(self.ee, completions=completions)
		self.run_pog(agent=agent)
		event, = self._fallback_events()
		self.assertEqual(event["meta"]["fallback_reason"],
		                 tax.FALLBACK_DEPTH_EXHAUSTED)
		self.assertEqual(event["traversal_depth"], agent.depth)


# --------------------------------------------------------------------------
# no double counting
# --------------------------------------------------------------------------

class TestNoDoubleCounting(EventsHarness):

	def test_tog_kg_events_match_the_backend_round_trips_exactly(self):
		agent, _ = self.run_tog()
		kg_events = [e for e in self.read_events()
		             if e["operation_type"] == "kg"]
		physical = len(agent.kg_backend.queries) + agent.kg_backend.name_lookups
		self.assertEqual(len(kg_events), physical,
		                 "an agent-side span would duplicate a KG event")

	def test_pog_kg_events_match_the_backend_round_trips_exactly(self):
		agent, _ = self.run_pog()
		kg_events = [e for e in self.read_events()
		             if e["operation_type"] == "kg"]
		physical = len(agent.kg_backend.queries) + agent.kg_backend.name_lookups
		self.assertEqual(len(kg_events), physical)

	def test_one_llm_event_per_actual_api_call(self):
		agent, _ = self.run_tog()
		self.assertEqual(len(self.llm_events()),
		                 agent.llm_api.client.chat.completions.calls,
		                 "no aggregate wrapper events around per-branch calls")

	def test_pog_one_llm_event_per_actual_api_call(self):
		agent, _ = self.run_pog()
		self.assertEqual(len(self.llm_events()),
		                 agent.llm_api.client.chat.completions.calls)

	def test_semantic_pruning_emits_exactly_one_embedding_event(self):
		"""The SentenceTransformer cut is one physical operation."""
		from chain_of_relations.methods.pog.tools import entity_condition_prune as ecp

		class FakeModel:
			def encode(self, value):
				return [0.1, 0.2]

		fake_st = types.ModuleType("sentence_transformers")
		fake_st.util = types.SimpleNamespace(
			dot_score=lambda a, b: [types.SimpleNamespace(
				cpu=lambda: types.SimpleNamespace(
					tolist=lambda: [float(i) for i in range(80)]))])
		sys.modules["sentence_transformers"] = fake_st
		original = ecp._get_semantic_model
		ecp._get_semantic_model = lambda: FakeModel()
		try:
			self.ee.set_question("q-embed")
			entities = [Entity(id="m.%d" % i, name="E%d" % i) for i in range(80)]
			kept = ecp._semantic_topn("q?", entities, 70)
		finally:
			ecp._get_semantic_model = original
			sys.modules.pop("sentence_transformers", None)

		self.assertEqual(len(kept), 70)
		events = self.read_events()
		self.assertEqual(len(events), 1)
		self.assertEqual(events[0]["operation_label"], "embedding:prune")
		self.assertEqual(events[0]["operation_type"], "embedding")
		self.assertEqual(events[0]["meta"]["n_entities"], 80)
		self.assertEqual(events[0]["question_id"], "q-embed")

	def test_embedding_event_no_longer_uses_the_legacy_signature(self):
		"""The old call recorded label 'tool' positionally; that is migrated."""
		import inspect
		from chain_of_relations.methods.pog.tools import entity_condition_prune as ecp
		source = inspect.getsource(ecp._semantic_topn)
		self.assertNotIn('record("tool"', source)
		self.assertIn("OperationLabel.EMBEDDING_PRUNE", source)


# --------------------------------------------------------------------------
# labels through the shared LLMAPI
# --------------------------------------------------------------------------

class TestLabelsThroughSharedLLMAPI(EventsHarness):

	def test_declared_scope_labels_a_tog_stage(self):
		api = make_llm_api(StageAwareCompletions(self.ee, "tog"))
		with self.ee.operation(OperationLabel.LLM_ENTITY_PRUNE):
			api.generate("prompt")
		event, = self.read_events()
		self.assertEqual(event["operation_label"], "llm:entity_prune")

	def test_explicit_argument_overrides_the_scope(self):
		api = make_llm_api(StageAwareCompletions(self.ee, "pog"))
		with self.ee.operation(OperationLabel.LLM_REASON):
			api.generate("prompt",
			             operation_label=OperationLabel.LLM_MEMORY_UPDATE)
		event, = self.read_events()
		self.assertEqual(event["operation_label"], "llm:memory_update")

	def test_retried_call_collapses_into_one_event_with_attempts(self):
		completions = StageAwareCompletions(
			self.ee, "pog",
			overrides={"llm:memory_update": [
				RuntimeError("boom"), '{"known": []}']})
		api = make_llm_api(completions)
		with self.ee.operation(OperationLabel.LLM_MEMORY_UPDATE):
			api.generate("prompt")
		event, = self.read_events()
		self.assertEqual(event["status"], Status.OK.value)
		self.assertEqual(event["meta"]["attempts"], 2)

	def test_terminal_failure_is_recorded_not_dropped(self):
		completions = StageAwareCompletions(
			self.ee, "pog",
			overrides={"llm:reverse_entity_select": [RuntimeError("boom")]})
		api = make_llm_api(completions)
		with self.ee.operation(OperationLabel.LLM_REVERSE_ENTITY_SELECT):
			result, _usage = api.generate("prompt")
		self.assertIsNone(result)
		event, = self.read_events()
		self.assertEqual(event["operation_label"], "llm:reverse_entity_select")
		self.assertEqual(event["status"], Status.ERROR.value)
		self.assertEqual(event["meta"]["attempts"], 3)


# --------------------------------------------------------------------------
# vocabulary and boundaries
# --------------------------------------------------------------------------

class TestVocabulary(unittest.TestCase):

	def test_new_labels_have_valid_types(self):
		for label in tax.TOG_LABELS | tax.POG_LABELS:
			self.assertEqual(tax.type_of(label), str(label).split(":", 1)[0])

	def test_tog_adds_no_label_of_its_own(self):
		self.assertTrue(tax.TOG_LLM_LABELS.issubset(
			tax.COR_LLM_LABELS | {OperationLabel.LLM_ENTITY_PRUNE}))

	def test_pog_specific_labels_are_not_claimed_by_the_others(self):
		pog_only = {OperationLabel.LLM_SUBQUESTION_DECOMPOSE,
		            OperationLabel.LLM_MEMORY_UPDATE,
		            OperationLabel.LLM_REVERSE_RETRIEVAL_DECISION,
		            OperationLabel.LLM_REVERSE_ENTITY_SELECT,
		            OperationLabel.EMBEDDING_PRUNE}
		self.assertTrue(pog_only.issubset(tax.POG_LABELS))
		self.assertFalse(pog_only & tax.COR_LABELS)
		self.assertFalse(pog_only & tax.TOG_LABELS)

	def test_cor_vocabulary_is_unchanged_by_this_slice(self):
		self.assertEqual(
			{l.value for l in tax.COR_LLM_LABELS},
			{"llm:relation_rank", "llm:reason", "llm:answer_filter",
			 "llm:direct_answer"})
		self.assertNotIn(OperationLabel.LLM_ENTITY_PRUNE, tax.COR_LABELS)
		self.assertNotIn(OperationLabel.EMBEDDING_PRUNE, tax.COR_LABELS)

	def test_legacy_llm_generate_is_in_no_paradigm_vocabulary(self):
		for labels in tax.LABELS_BY_PARADIGM.values():
			self.assertNotIn(OperationLabel.LLM_GENERATE, labels)

	def test_wikidata_kg_gap_is_declared_not_silently_absent(self):
		self.assertIn("unmeasured", tax.WIKIDATA_KG_UNINSTRUMENTED)
		import chain_of_relations.kg_backend.wikidata.db_func as wikidata
		import inspect
		self.assertNotIn("energy_events", inspect.getsource(wikidata),
		                 "if Wikidata gained instrumentation, update the note")

	def test_profiler_imports_no_paradigm_module(self):
		"""Adding ToG/PoG semantics must not have leaked into the profiler.

		Checked on the parsed import statements, not the file text: the
		profiler's prose may name its hosts, it just may not import them.
		"""
		import ast
		import inspect
		import agent_energy_profiler
		from agent_energy_profiler import (aggregate, attribution, events,
		                                   labels, schema, trajectory)
		paradigm_roots = {"chain_of_relations"}
		for module in (agent_energy_profiler, aggregate, attribution, events,
		               labels, schema, trajectory):
			tree = ast.parse(inspect.getsource(module))
			for node in ast.walk(tree):
				if isinstance(node, ast.Import):
					names = [alias.name for alias in node.names]
				elif isinstance(node, ast.ImportFrom):
					names = [node.module or ""]
				else:
					continue
				for name in names:
					self.assertNotIn(name.split(".")[0], paradigm_roots,
					                 f"{module.__name__} imports {name}")


class TestInstrumentationDisabled(unittest.TestCase):
	"""With no events file configured, both agents must still answer."""

	def setUp(self):
		os.environ.pop("ENERGY_EVENTS_FILE", None)
		import chain_of_relations.energy_events as ee
		self.ee = importlib.reload(ee)
		self.ee.configure(events_file="")

	def test_tog_runs_with_instrumentation_off(self):
		agent = make_tog_agent(self.ee)
		result = agent.answer("who knows Alice?",
		                      [Entity(id="m.alice", name="Alice")])
		self.assertIn("results", result)
		self.assertFalse(self.ee.enabled())

	def test_pog_runs_with_instrumentation_off(self):
		agent = make_pog_agent(self.ee)
		result = agent.answer("who knows Alice?",
		                      [Entity(id="m.alice", name="Alice")])
		self.assertIn("results", result)
		self.assertFalse(self.ee.enabled())


if __name__ == "__main__":
	unittest.main(verbosity=2)
