"""Tests for the CoR energy measurement layer (schema v1).

Run:
  python -m unittest discover -s tests -p "test_*.py" -v

These are pure unit tests: no LLM endpoint, no SPARQL endpoint, no GPU.
"""

import importlib
import json
import os
import sys
import tempfile
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from chain_of_relations import energy_taxonomy as tax
from chain_of_relations.energy_taxonomy import OperationLabel, OperationType, Status

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


class EventsHarness(unittest.TestCase):
	"""Gives each test a fresh energy_events module writing to a temp file."""

	def setUp(self):
		self._tmp = tempfile.TemporaryDirectory()
		self.events_path = os.path.join(self._tmp.name, "events.jsonl")
		os.environ["ENERGY_EVENTS_FILE"] = self.events_path
		import chain_of_relations.energy_events as ee
		self.ee = importlib.reload(ee)

	def tearDown(self):
		self.ee.close()
		os.environ.pop("ENERGY_EVENTS_FILE", None)
		self._tmp.cleanup()

	def read_events(self):
		if not os.path.exists(self.events_path):
			return []
		with open(self.events_path) as f:
			return [json.loads(line) for line in f if line.strip()]


# --------------------------------------------------------------------------
# 1. schema
# --------------------------------------------------------------------------

class TestSchemaV1(EventsHarness):

	def test_event_has_all_required_keys(self):
		self.ee.record(OperationLabel.KG_ENTITY_SEARCH, 100.0, 100.5)
		event, = self.read_events()
		for key in REQUIRED_KEYS:
			self.assertIn(key, event, f"schema-v1 event missing {key!r}")

	def test_schema_version_is_1(self):
		self.ee.record(OperationLabel.KG_SPARQL, 1.0, 2.0)
		event, = self.read_events()
		self.assertEqual(event["schema_version"], 1)
		self.assertEqual(tax.SCHEMA_VERSION, 1)

	def test_event_ids_are_unique(self):
		for _ in range(50):
			self.ee.record(OperationLabel.KG_SPARQL, 1.0, 2.0)
		ids = [e["event_id"] for e in self.read_events()]
		self.assertEqual(len(ids), len(set(ids)))

	def test_timing_is_valid_and_ordered(self):
		self.ee.record(OperationLabel.KG_ID2NAME, 10.0, 12.5)
		event, = self.read_events()
		self.assertLessEqual(event["start_timestamp"], event["end_timestamp"])
		self.assertAlmostEqual(
			event["duration_s"],
			event["end_timestamp"] - event["start_timestamp"])

	def test_unavailable_measurements_are_null_not_zero(self):
		# No NVML marks and no in-band RAPL: these must be null, never 0.0.
		self.ee.record(OperationLabel.KG_SPARQL, 1.0, 2.0)
		event, = self.read_events()
		for field in ("gpu_energy_j", "cpu_package_energy_j", "dram_energy_j"):
			self.assertIsNone(event[field], f"{field} fabricated a value")

	def test_gpu_counter_delta_is_converted_from_millijoules(self):
		self.ee.record(OperationLabel.LLM_REASON, (1.0, 1000), (2.0, 3500))
		event, = self.read_events()
		self.assertAlmostEqual(event["gpu_energy_j"], 2.5)

	def test_legacy_keys_are_preserved_for_unmigrated_readers(self):
		self.ee.record(OperationLabel.LLM_DIRECT_ANSWER, 1.0, 2.0)
		event, = self.read_events()
		self.assertEqual(event["category"], "inference")
		self.assertEqual(event["label"], "llm:direct_answer")
		self.assertEqual(event["t_start"], event["start_timestamp"])
		self.assertEqual(event["t_end"], event["end_timestamp"])


# --------------------------------------------------------------------------
# 2 + 3. taxonomy validity and type/label consistency
# --------------------------------------------------------------------------

class TestTaxonomy(unittest.TestCase):

	def test_every_label_has_a_valid_type_prefix(self):
		for label in OperationLabel:
			prefix = tax.type_of(label)
			self.assertIn(prefix, {t.value for t in OperationType})
			self.assertTrue(label.value.startswith(prefix + ":"))

	def test_unknown_label_is_rejected(self):
		with self.assertRaises(ValueError):
			tax.type_of("llm:not_a_real_stage")

	def test_no_synonymous_label_variants(self):
		values = [label.value for label in OperationLabel]
		self.assertEqual(len(values), len(set(values)))

	def test_cor_taxonomy_excludes_stages_cor_does_not_have(self):
		# CoR has no embedding stage and does not use tools/entity_prune.py.
		self.assertNotIn(OperationLabel.EMBEDDING_PRUNE, tax.COR_LABELS)
		self.assertNotIn(OperationLabel.LLM_ENTITY_PRUNE, tax.COR_LABELS)
		self.assertNotIn(OperationLabel.LLM_GENERATE, tax.COR_LABELS)

	def test_cor_llm_labels_are_the_four_real_stages(self):
		self.assertEqual(
			{label.value for label in tax.COR_LLM_LABELS},
			{"llm:relation_rank", "llm:reason", "llm:answer_filter", "llm:direct_answer"})

	def test_legacy_category_mapping(self):
		self.assertEqual(tax.legacy_category(OperationLabel.LLM_REASON), "inference")
		self.assertEqual(tax.legacy_category(OperationLabel.KG_SPARQL), "tool")

	def test_status_vocabulary_is_controlled(self):
		self.assertTrue(tax.is_known_status("ok"))
		self.assertTrue(tax.is_known_status(Status.TIMEOUT))
		self.assertFalse(tax.is_known_status("failed"))


class TestEventTypeLabelAgreement(EventsHarness):

	def test_operation_type_always_matches_label_prefix(self):
		for label in tax.COR_LABELS:
			self.ee.record(label, 1.0, 2.0)
		for event in self.read_events():
			self.assertEqual(event["operation_type"],
			                 event["operation_label"].split(":")[0])
			self.assertTrue(tax.is_known_label(event["operation_label"]))

	def test_unknown_label_is_flagged_not_dropped(self):
		self.ee.record("llm:bogus_stage", 1.0, 2.0)
		event, = self.read_events()
		self.assertTrue(event["meta"].get("unknown_label"))

	def test_unknown_status_is_flagged_not_silently_accepted(self):
		self.ee.record(OperationLabel.KG_SPARQL, 1.0, 2.0, status="exploded")
		event, = self.read_events()
		self.assertEqual(event["status"], "error")
		self.assertEqual(event["meta"].get("unknown_status"), "exploded")


# --------------------------------------------------------------------------
# 4. CoR LLM calls no longer collapse to a generic label
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


class FakeCompletions:
	"""Replays a scripted sequence of responses/exceptions."""

	def __init__(self, script):
		self.script = list(script)
		self.calls = 0

	def create(self, **kwargs):
		self.calls += 1
		item = self.script.pop(0) if self.script else self.script_default()
		if isinstance(item, Exception):
			raise item
		return item

	def script_default(self):
		return FakeResponse("ok", FakeUsage(1, 1))


class FakeClient:
	def __init__(self, script):
		self.chat = type("chat", (), {})()
		self.chat.completions = FakeCompletions(script)


def make_llm_api(script):
	os.environ.setdefault("OPENAI_API_KEY", "test-key")
	from chain_of_relations.llm_api import LLMAPI
	api = LLMAPI.__new__(LLMAPI)
	api.model_name = "test-model"
	api.api_key = "test-key"
	api.base_url = ""
	api.timeout = 1.0
	api.max_retries = 3
	api.retry_interval = 0.0
	api.client = FakeClient(script)
	return api


class TestSemanticLLMInstrumentation(EventsHarness):

	def test_declared_operation_scope_labels_the_event(self):
		api = make_llm_api([FakeResponse("answer", FakeUsage(120, 7))])
		with self.ee.operation(OperationLabel.LLM_RELATION_RANK):
			api.generate("prompt")
		event, = self.read_events()
		self.assertEqual(event["operation_label"], "llm:relation_rank")
		self.assertEqual(event["operation_type"], "llm")

	def test_explicit_operation_label_argument_wins(self):
		api = make_llm_api([FakeResponse("answer", FakeUsage(1, 1))])
		api.generate("prompt", operation_label=OperationLabel.LLM_ANSWER_FILTER)
		event, = self.read_events()
		self.assertEqual(event["operation_label"], "llm:answer_filter")

	def test_unmigrated_caller_keeps_the_legacy_label(self):
		# ToG/PoG declare nothing and must be unaffected by this slice.
		api = make_llm_api([FakeResponse("answer", FakeUsage(1, 1))])
		api.generate("prompt")
		event, = self.read_events()
		self.assertEqual(event["operation_label"], "llm:generate")

	def test_cor_agent_emits_four_distinct_labels_and_never_llm_generate(self):
		"""Drive the real CoRAgent stage wiring with a fake LLM and KG."""
		agent = make_cor_agent(self.ee)
		api = agent.llm_api

		api.client.chat.completions.script = [
			FakeResponse('{"relations": []}', FakeUsage(10, 2)),
			FakeResponse('{"action": "Stop"}', FakeUsage(11, 3)),
			FakeResponse('{"answer": ["x"]}', FakeUsage(12, 4)),
			FakeResponse('{"answer": ["y"]}', FakeUsage(13, 5)),
		]

		from chain_of_relations.schema import Entity
		topic = Entity(id="m.01", name="Topic")

		agent.relation_prune(question="q", topic_entity=topic, relation_chain=[],
		                     head_relations=["r.a"], tail_relations=[],
		                     prompt_history=[])
		agent.reasoning(question="q", topic_entity=topic, relation_chain=[],
		                target_entity_candidates=["c"], prompt_history=[])
		agent.validate(question="q", topic_entity=topic, relation_chain=[],
		               target_entity_names=["c"], prompt_history=[])
		agent.generate_directly(question="q", prompt_history=[])

		labels = [e["operation_label"] for e in self.read_events()
		          if e["operation_type"] == "llm"]
		self.assertNotIn("llm:generate", labels,
		                 "CoR LLM work collapsed to the generic label")
		self.assertEqual(set(labels), {
			"llm:relation_rank", "llm:reason", "llm:answer_filter", "llm:direct_answer"})
		for label in labels:
			self.assertIn(label, {l.value for l in tax.COR_LLM_LABELS})


class TestFallbackLabelling(EventsHarness):

	def test_direct_answer_carries_fallback_metadata(self):
		agent = make_cor_agent(self.ee)
		agent.llm_api.client.chat.completions.script = [
			FakeResponse('{"answer": ["y"]}', FakeUsage(9, 3))]
		agent.generate_directly(question="q", prompt_history=[])

		event, = self.read_events()
		self.assertEqual(event["operation_label"], "llm:direct_answer")
		self.assertTrue(event["meta"]["fallback"])
		self.assertEqual(event["meta"]["fallback_reason"], "no_graph_answer")
		self.assertEqual(tax.FALLBACK_NO_GRAPH_ANSWER, "no_graph_answer")

	def test_kg_grounded_answer_stage_is_not_marked_fallback(self):
		agent = make_cor_agent(self.ee)
		agent.llm_api.client.chat.completions.script = [
			FakeResponse('{"answer": ["y"]}', FakeUsage(9, 3))]
		from chain_of_relations.schema import Entity
		agent.validate(question="q", topic_entity=Entity(id="m.01", name="T"),
		               relation_chain=[], target_entity_names=["c"],
		               prompt_history=[])

		event, = self.read_events()
		self.assertEqual(event["operation_label"], "llm:answer_filter")
		self.assertNotIn("fallback", event["meta"])

	def test_old_llm_answer_label_is_gone(self):
		self.assertFalse(tax.is_known_label("llm:answer"))


class TestScopedEventMeta(EventsHarness):

	def test_scoped_meta_is_merged_into_events(self):
		with self.ee.event_meta(entity_count=64, batched=True):
			self.ee.record(OperationLabel.KG_ID2NAME, 1.0, 2.0, attempts=1)
		event, = self.read_events()
		self.assertEqual(event["meta"]["entity_count"], 64)
		self.assertTrue(event["meta"]["batched"])
		self.assertEqual(event["meta"]["attempts"], 1)

	def test_explicit_kwargs_outrank_scoped_meta(self):
		with self.ee.event_meta(entity_count=64):
			self.ee.record(OperationLabel.KG_ID2NAME, 1.0, 2.0, entity_count=1)
		event, = self.read_events()
		self.assertEqual(event["meta"]["entity_count"], 1)

	def test_scope_is_restored_and_does_not_leak(self):
		with self.ee.event_meta(batched=True):
			self.ee.record(OperationLabel.KG_ID2NAME, 1.0, 2.0)
		self.ee.record(OperationLabel.KG_ID2NAME, 2.0, 3.0)
		first, second = self.read_events()
		self.assertTrue(first["meta"]["batched"])
		self.assertNotIn("batched", second["meta"])

	def test_nested_scopes_merge_with_inner_winning(self):
		with self.ee.event_meta(entity_count=64, batched=True):
			with self.ee.event_meta(entity_count=8):
				self.ee.record(OperationLabel.KG_ID2NAME, 1.0, 2.0)
		event, = self.read_events()
		self.assertEqual(event["meta"]["entity_count"], 8)
		self.assertTrue(event["meta"]["batched"])


class TestRetryAccounting(EventsHarness):
	"""attempts (API-level, one event) vs tool_attempt (separate events)."""

	def test_api_retries_collapse_into_one_event_with_attempts(self):
		api = make_llm_api([
			RuntimeError("transient"),
			RuntimeError("transient"),
			FakeResponse("answer", FakeUsage(10, 2)),
		])
		with self.ee.operation(OperationLabel.LLM_REASON):
			api.generate("prompt")
		event, = self.read_events()
		self.assertEqual(event["meta"]["attempts"], 3)

	def test_tool_level_retries_create_separate_events_numbered_by_tool_attempt(self):
		"""The shared tool retries llm_generate; each pass is its own event.

		Nine scripted failures = three tool passes, each exhausting three API
		attempts inside one event.
		"""
		agent = make_cor_agent(self.ee)
		agent.llm_api.client.chat.completions.script = [RuntimeError("boom")] * 9
		agent.generate_directly(question="q", prompt_history=[])

		events = self.read_events()
		self.assertEqual(len(events), 3, "expected one event per tool-level pass")
		self.assertEqual([e["meta"]["tool_attempt"] for e in events], [1, 2, 3])
		for event in events:
			self.assertEqual(event["meta"]["attempts"], 3)
			self.assertEqual(event["status"], "error")
			self.assertEqual(event["operation_label"], "llm:direct_answer")

	def test_tool_attempt_restarts_per_stage_invocation(self):
		agent = make_cor_agent(self.ee)
		agent.llm_api.client.chat.completions.script = [
			FakeResponse('{"answer": ["a"]}', FakeUsage(1, 1)),
			FakeResponse('{"answer": ["b"]}', FakeUsage(1, 1)),
		]
		agent.generate_directly(question="q1", prompt_history=[])
		agent.generate_directly(question="q2", prompt_history=[])
		self.assertEqual(
			[e["meta"]["tool_attempt"] for e in self.read_events()], [1, 1])

	def test_the_two_counters_are_independent_fields(self):
		agent = make_cor_agent(self.ee)
		agent.llm_api.client.chat.completions.script = [
			RuntimeError("boom"), RuntimeError("boom"), RuntimeError("boom"),
			FakeResponse('{"answer": ["y"]}', FakeUsage(5, 1)),
		]
		agent.generate_directly(question="q", prompt_history=[])
		first, second = self.read_events()
		# pass 1 burned three API attempts and failed; pass 2 succeeded first try
		self.assertEqual((first["meta"]["tool_attempt"], first["meta"]["attempts"]),
		                 (1, 3))
		self.assertEqual((second["meta"]["tool_attempt"], second["meta"]["attempts"]),
		                 (2, 1))


class TestLLMEventStatusAndTokens(EventsHarness):

	def test_successful_call_exposes_tokens_at_top_level(self):
		api = make_llm_api([FakeResponse("answer", FakeUsage(4172, 73))])
		with self.ee.operation(OperationLabel.LLM_REASON):
			api.generate("prompt")
		event, = self.read_events()
		self.assertEqual(event["input_tokens"], 4172)
		self.assertEqual(event["output_tokens"], 73)
		self.assertEqual(event["status"], "ok")
		self.assertEqual(event["meta"]["attempts"], 1)

	def test_retried_then_successful_call_records_attempts_and_ok(self):
		api = make_llm_api([
			RuntimeError("transient"),
			FakeResponse("answer", FakeUsage(10, 2)),
		])
		with self.ee.operation(OperationLabel.LLM_REASON):
			result, _ = api.generate("prompt")
		self.assertEqual(result, "answer")
		event, = self.read_events()
		self.assertEqual(event["status"], "ok")
		self.assertEqual(event["meta"]["attempts"], 2)

	def test_terminal_failure_is_recorded_as_error_not_dropped(self):
		api = make_llm_api([RuntimeError("boom")] * 3)
		with self.ee.operation(OperationLabel.LLM_DIRECT_ANSWER):
			result, _ = api.generate("prompt")
		self.assertIsNone(result)
		event, = self.read_events()
		self.assertEqual(event["status"], "error")
		self.assertEqual(event["operation_label"], "llm:direct_answer")
		self.assertEqual(event["meta"]["attempts"], 3)

	def test_timeout_failure_gets_the_timeout_status(self):
		api = make_llm_api([RuntimeError("Request timed out")] * 3)
		with self.ee.operation(OperationLabel.LLM_REASON):
			api.generate("prompt")
		event, = self.read_events()
		self.assertEqual(event["status"], "timeout")

	def test_tokens_are_null_when_the_api_reported_no_usage(self):
		api = make_llm_api([FakeResponse("answer", usage=None)])
		with self.ee.operation(OperationLabel.LLM_REASON):
			api.generate("prompt")
		event, = self.read_events()
		self.assertIsNone(event["input_tokens"])
		self.assertIsNone(event["output_tokens"])


# --------------------------------------------------------------------------
# 5. run / question / iteration / step context propagation
# --------------------------------------------------------------------------

class TestContextPropagation(EventsHarness):

	def test_question_id_is_attached(self):
		self.ee.set_question("WebQTest-42")
		self.ee.record(OperationLabel.KG_SPARQL, 1.0, 2.0)
		event, = self.read_events()
		self.assertEqual(event["question_id"], "WebQTest-42")

	def test_step_index_is_ordered_and_resets_per_question(self):
		self.ee.set_question("q1")
		for _ in range(3):
			self.ee.record(OperationLabel.KG_SPARQL, 1.0, 2.0)
		self.ee.set_question("q2")
		for _ in range(2):
			self.ee.record(OperationLabel.KG_SPARQL, 1.0, 2.0)

		events = self.read_events()
		q1 = [e["step_index"] for e in events if e["question_id"] == "q1"]
		q2 = [e["step_index"] for e in events if e["question_id"] == "q2"]
		self.assertEqual(q1, [0, 1, 2])
		self.assertEqual(q2, [0, 1])

	def test_iteration_and_step_index_are_distinct(self):
		self.ee.set_question("q1")
		with self.ee.iteration(2):
			self.ee.record(OperationLabel.KG_SPARQL, 1.0, 2.0)
			self.ee.record(OperationLabel.KG_SPARQL, 2.0, 3.0)
		events = self.read_events()
		self.assertEqual([e["iteration"] for e in events], [2, 2])
		self.assertEqual([e["step_index"] for e in events], [0, 1])

	def test_iteration_scope_is_restored(self):
		self.ee.set_question("q1")
		with self.ee.iteration(1):
			with self.ee.iteration(2):
				self.ee.record(OperationLabel.KG_SPARQL, 1.0, 2.0)
			self.ee.record(OperationLabel.KG_SPARQL, 2.0, 3.0)
		self.assertEqual([e["iteration"] for e in self.read_events()], [2, 1])

	def test_begin_iteration_counts_up_from_zero_and_resets_per_question(self):
		self.ee.set_question("q1")
		self.assertEqual([self.ee.begin_iteration() for _ in range(3)], [0, 1, 2])
		self.ee.set_question("q2")
		self.assertEqual(self.ee.begin_iteration(), 0)

	def test_traversal_depth_scope_is_restored(self):
		self.ee.set_question("q1")
		with self.ee.traversal_depth(1):
			with self.ee.traversal_depth(2):
				self.ee.record(OperationLabel.KG_SPARQL, 1.0, 2.0)
			self.ee.record(OperationLabel.KG_SPARQL, 2.0, 3.0)
		self.assertEqual([e["traversal_depth"] for e in self.read_events()], [2, 1])

	def test_traversal_depth_is_null_for_work_outside_the_dfs(self):
		self.ee.set_question("q1")
		self.ee.begin_iteration()
		self.ee.set_traversal_depth(None)
		self.ee.record(OperationLabel.LLM_DIRECT_ANSWER, 1.0, 2.0)
		event, = self.read_events()
		self.assertIsNone(event["traversal_depth"])
		self.assertEqual(event["iteration"], 0,
		                 "iteration must keep advancing outside the DFS")

	def test_backtracking_depth_falls_while_iteration_keeps_rising(self):
		"""The critical distinction, on the required sequence.

		A DFS that descends to depth 2, backtracks to 1, then descends again
		produces depths [0, 1, 2, 1, 2] while iterations run [0, 1, 2, 3, 4].
		Each control-loop pass emits two events, so step_index runs 0..9
		independently of both.
		"""
		self.ee.set_question("q1")
		for depth in (0, 1, 2, 1, 2):
			self.ee.begin_iteration()
			self.ee.set_traversal_depth(depth)
			self.ee.record(OperationLabel.KG_RELATION_SEARCH, 1.0, 2.0)
			self.ee.record(OperationLabel.LLM_RELATION_RANK, 2.0, 3.0)

		events = self.read_events()
		# one iteration/depth pair per control-loop pass, two events each
		self.assertEqual([e["traversal_depth"] for e in events],
		                 [0, 0, 1, 1, 2, 2, 1, 1, 2, 2])
		self.assertEqual([e["iteration"] for e in events],
		                 [0, 0, 1, 1, 2, 2, 3, 3, 4, 4])
		self.assertEqual([e["step_index"] for e in events], list(range(10)))

		# the defining properties
		iterations = [e["iteration"] for e in events]
		depths = [e["traversal_depth"] for e in events]
		self.assertEqual(iterations, sorted(iterations),
		                 "iteration must never decrease within a question")
		self.assertTrue(any(b < a for a, b in zip(depths, depths[1:])),
		                "traversal_depth must be able to decrease")
		self.assertNotEqual(iterations, depths,
		                    "iteration and traversal_depth must not be the same field")

	def test_iteration_and_depth_survive_into_the_attributed_artifact(self):
		self.ee.set_question("q1")
		self.ee.begin_iteration()
		self.ee.set_traversal_depth(2)
		self.ee.record(OperationLabel.LLM_REASON, 1.0, 2.0)
		raw, = self.read_events()

		sys.path.insert(0, os.path.join(
			os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "measurement"))
		import attribute
		rows, _ = attribute.attribute_events([raw], [1.0, 2.0], {}, {})
		row, = rows
		self.assertEqual(row["iteration"], 0)
		self.assertEqual(row["traversal_depth"], 2)
		self.assertEqual(row["step_index"], raw["step_index"])

	def test_run_provenance_is_configured_once_and_attached_to_every_event(self):
		self.ee.configure(
			run_id="run-abc", dataset="webqsp", paradigm="cor",
			model_name="qwen2.5-7b", model_revision="rev1",
			git_commit="deadbeef", hardware_id="host|RTX 4090")
		self.ee.set_question("q1")
		self.ee.record(OperationLabel.KG_SPARQL, 1.0, 2.0)
		self.ee.record(OperationLabel.LLM_REASON, 2.0, 3.0)
		for event in self.read_events():
			self.assertEqual(event["run_id"], "run-abc")
			self.assertEqual(event["dataset"], "webqsp")
			self.assertEqual(event["paradigm"], "cor")
			self.assertEqual(event["model_name"], "qwen2.5-7b")
			self.assertEqual(event["model_revision"], "rev1")
			self.assertEqual(event["git_commit"], "deadbeef")
			self.assertEqual(event["hardware_id"], "host|RTX 4090")

	def test_missing_optional_provenance_is_null_and_does_not_break(self):
		self.ee.configure(run_id="run-abc", dataset="webqsp", paradigm="cor")
		self.ee.record(OperationLabel.KG_SPARQL, 1.0, 2.0)
		event, = self.read_events()
		self.assertEqual(event["run_id"], "run-abc")
		self.assertIsNone(event["model_revision"])
		self.assertIsNone(event["git_commit"])
		self.assertIsNone(event["hardware_id"])


# --------------------------------------------------------------------------
# KG labelling
# --------------------------------------------------------------------------

class TestKGLabelling(EventsHarness):

	def test_sparql_classification_covers_the_real_cor_query_shapes(self):
		cases = [
			("SELECT ?targetEntity WHERE { ?e ns:type.object.name ?targetEntity }",
			 "kg:id2name"),
			("SELECT ?relation WHERE { ns:m.01 ?relation ?x }", "kg:relation_search"),
			("SELECT ?targetEntity WHERE { ns:m.01 ns:r ?targetEntity }",
			 "kg:entity_search"),
			("ASK { ?s ?p ?o }", "kg:sparql"),
		]
		for query, expected in cases:
			self.assertEqual(str(self.ee._classify_sparql(query)), expected, query)

	def test_explicit_label_overrides_classification(self):
		self.ee.record_sparql("ASK { ?s ?p ?o }", 1.0, 2.0,
		                      label=OperationLabel.KG_ID2NAME)
		event, = self.read_events()
		self.assertEqual(event["operation_label"], "kg:id2name")

	def test_batched_id2name_is_one_event_carrying_the_entity_count(self):
		"""Event boundaries follow real round trips, not entity counts."""
		with self.ee.event_meta(entity_count=64, batch_index=1, total_batches=2,
		                        batched=True):
			self.ee.record_sparql(
				"SELECT ?entity ?targetEntity WHERE { ?entity ns:type.object.name ?targetEntity }",
				1.0, 2.0, attempts=1)
		events = self.read_events()
		self.assertEqual(len(events), 1, "must not synthesize per-entity events")
		event, = events
		self.assertEqual(event["operation_label"], "kg:id2name")
		self.assertEqual(event["meta"]["entity_count"], 64)
		self.assertTrue(event["meta"]["batched"])

	def test_single_and_batched_id2name_share_the_normalizing_field(self):
		self.ee.record_sparql("ASK {}", 1.0, 2.0,
		                      label=OperationLabel.KG_ID2NAME,
		                      entity_count=1, batched=False)
		with self.ee.event_meta(entity_count=32, batched=True):
			self.ee.record_sparql("ASK {}", 2.0, 3.0,
			                      label=OperationLabel.KG_ID2NAME)
		single, batched = self.read_events()
		self.assertEqual(single["meta"]["entity_count"], 1)
		self.assertFalse(single["meta"]["batched"])
		self.assertEqual(batched["meta"]["entity_count"], 32)
		# energy per entity is derivable for both shapes
		for event in (single, batched):
			self.assertGreater(event["meta"]["entity_count"], 0)

	def test_legacy_record_signature_still_works_for_other_paradigms(self):
		# PoG calls record("tool", "embedding:prune", t0, t1, ...) and must
		# not be broken by this slice.
		self.ee.record("tool", "embedding:prune", 1.0, 2.0, n=5)
		event, = self.read_events()
		self.assertEqual(event["operation_label"], "embedding:prune")
		self.assertEqual(event["operation_type"], "embedding")
		self.assertEqual(event["status"], "ok")
		self.assertEqual(event["meta"]["n"], 5)


# --------------------------------------------------------------------------
# 8 + 9. energy accounting and attribution preservation
# --------------------------------------------------------------------------

class TestEnergyAccounting(unittest.TestCase):

	def setUp(self):
		sys.path.insert(0, os.path.join(
			os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "measurement"))
		import attribute
		self.attribute = importlib.reload(attribute)

	def test_core_is_classified_separately_from_package(self):
		cols = {
			"rapl_package-0_intel-rapl:0_uj": [],
			"rapl_core_intel-rapl:0:0_uj": [],
			"rapl_dram_intel-rapl:0:1_uj": [],
		}
		package, core, dram = self.attribute.classify_rapl(cols)
		self.assertEqual(list(package), ["rapl_package-0_intel-rapl:0_uj"])
		self.assertEqual(list(core), ["rapl_core_intel-rapl:0:0_uj"])
		self.assertEqual(list(dram), ["rapl_dram_intel-rapl:0:1_uj"])

	def test_psys_is_excluded_because_it_contains_package(self):
		package, core, dram = self.attribute.classify_rapl(
			{"rapl_psys_intel-rapl:1_uj": [], "rapl_package-0_intel-rapl:0_uj": []})
		self.assertEqual(list(package), ["rapl_package-0_intel-rapl:0_uj"])

	def test_measured_total_is_gpu_plus_package_plus_dram(self):
		self.assertAlmostEqual(self.attribute.measured_total(10.0, 5.0, 2.0), 17.0)

	def test_core_is_never_added_to_the_total(self):
		"""End-to-end: a run with package=20J, core=15J, dram=4J, gpu=1J."""
		ts = [0.0, 1.0, 2.0]
		rapls = {
			# uJ cumulative counters
			"rapl_package-0_intel-rapl:0_uj": [0.0, 10e6, 20e6],
			"rapl_core_intel-rapl:0:0_uj": [0.0, 7.5e6, 15e6],
			"rapl_dram_intel-rapl:0:1_uj": [0.0, 2e6, 4e6],
		}
		events = [{
			"start_timestamp": 0.0, "end_timestamp": 2.0, "duration_s": 2.0,
			"gpu_energy_j": 1.0, "operation_label": "llm:reason",
			"operation_type": "llm",
		}]
		rows, _ = self.attribute.attribute_events(events, ts, {}, rapls)
		row, = rows
		self.assertAlmostEqual(row["cpu_package_energy_j"], 20.0)
		self.assertAlmostEqual(row["cpu_core_energy_j"], 15.0)
		self.assertAlmostEqual(row["dram_energy_j"], 4.0)
		# 1 + 20 + 4 == 25. If core were summed in, it would be 40.
		self.assertAlmostEqual(row["measured_energy_j"], 25.0)
		self.assertNotAlmostEqual(row["measured_energy_j"], 40.0)

	def test_missing_rapl_yields_null_not_zero(self):
		events = [{
			"start_timestamp": 0.0, "end_timestamp": 2.0, "duration_s": 2.0,
			"gpu_energy_j": 3.0, "operation_label": "llm:reason",
			"operation_type": "llm",
		}]
		rows, _ = self.attribute.attribute_events(events, [0.0, 2.0], {}, {})
		row, = rows
		self.assertIsNone(row["cpu_package_energy_j"])
		self.assertIsNone(row["dram_energy_j"])
		self.assertIsNone(row["measured_energy_j"],
		                  "an unmeasured component must not produce a partial total")

	def test_available_domains_and_completeness_are_reported(self):
		ts = [0.0, 1.0, 2.0]
		rapls = {
			"rapl_package-0_intel-rapl:0_uj": [0.0, 10e6, 20e6],
			"rapl_dram_intel-rapl:0:1_uj": [0.0, 2e6, 4e6],
		}
		event = {"start_timestamp": 0.0, "end_timestamp": 2.0, "duration_s": 2.0,
		         "gpu_energy_j": 1.0, "operation_label": "llm:reason",
		         "operation_type": "llm"}
		row, = self.attribute.attribute_events([event], ts, {}, rapls)[0]
		self.assertEqual(row["available_energy_domains"],
		                 ["gpu", "cpu_package", "dram"])
		self.assertTrue(row["measurement_complete"])

	def test_incomplete_measurement_is_reported_as_incomplete(self):
		event = {"start_timestamp": 0.0, "end_timestamp": 2.0, "duration_s": 2.0,
		         "gpu_energy_j": 3.0, "operation_label": "llm:reason",
		         "operation_type": "llm"}
		rows, _ = self.attribute.attribute_events([event], [0.0, 2.0], {}, {})
		row, = rows
		self.assertEqual(row["available_energy_domains"], ["gpu"])
		self.assertFalse(row["measurement_complete"])
		self.assertIsNone(row["measured_energy_j"])

	def test_core_is_not_listed_as_a_boundary_domain(self):
		ts = [0.0, 1.0, 2.0]
		rapls = {"rapl_core_intel-rapl:0:0_uj": [0.0, 7.5e6, 15e6]}
		event = {"start_timestamp": 0.0, "end_timestamp": 2.0, "duration_s": 2.0,
		         "gpu_energy_j": 1.0, "operation_label": "llm:reason",
		         "operation_type": "llm"}
		rows, _ = self.attribute.attribute_events([event], ts, {}, rapls)
		row, = rows
		self.assertAlmostEqual(row["cpu_core_energy_j"], 15.0)
		self.assertNotIn("cpu_core", row["available_energy_domains"])
		self.assertFalse(row["measurement_complete"])

	def test_attribution_preserves_every_semantic_field(self):
		event = {
			"schema_version": 1, "event_id": "abc123",
			"run_id": "run-1", "question_id": "WebQTest-7",
			"dataset": "webqsp", "paradigm": "cor",
			"iteration": 7, "traversal_depth": 2, "step_index": 11,
			"operation_type": "llm", "operation_label": "llm:relation_rank",
			"start_timestamp": 0.0, "end_timestamp": 1.0, "duration_s": 1.0,
			"gpu_energy_j": 5.0,
			"input_tokens": 4172, "output_tokens": 73,
			"status": "ok",
			"model_name": "qwen2.5-7b", "model_revision": "rev1",
			"git_commit": "deadbeef", "hardware_id": "host|RTX 4090",
			"meta": {"attempts": 2},
		}
		rows, _ = self.attribute.attribute_events([event], [0.0, 1.0], {}, {})
		row, = rows
		for key in ("run_id", "question_id", "dataset", "paradigm", "iteration",
		            "traversal_depth", "step_index", "operation_type", "operation_label",
		            "input_tokens", "output_tokens", "status", "model_name",
		            "model_revision", "git_commit", "hardware_id", "meta",
		            "event_id", "schema_version"):
			self.assertEqual(row[key], event[key], f"attribution lost {key!r}")
		self.assertIn("measured_energy_j", row)
		self.assertIn("cpu_core_energy_j", row)


# --------------------------------------------------------------------------
# 10. instrumentation disabled
# --------------------------------------------------------------------------

class TestInstrumentationDisabled(unittest.TestCase):

	def setUp(self):
		os.environ.pop("ENERGY_EVENTS_FILE", None)
		import chain_of_relations.energy_events as ee
		self.ee = importlib.reload(ee)

	def test_disabled_reports_disabled(self):
		self.assertFalse(self.ee.enabled())

	def test_record_is_a_no_op_and_never_raises(self):
		self.ee.set_question("q1")
		with self.ee.iteration(3):
			self.ee.record(OperationLabel.LLM_REASON, 1.0, 2.0, input_tokens=5)
		self.ee.record_sparql("ASK {}", 1.0, 2.0)

	def test_llm_generate_still_works_with_instrumentation_off(self):
		api = make_llm_api([FakeResponse("answer", FakeUsage(3, 1))])
		result, usage = api.generate("prompt")
		self.assertEqual(result, "answer")
		self.assertEqual(usage["input_tokens"], 3)

	def test_cor_agent_runs_its_llm_stages_with_instrumentation_off(self):
		agent = make_cor_agent(self.ee)
		agent.llm_api.client.chat.completions.script = [
			FakeResponse('{"answer": ["y"]}', FakeUsage(1, 1))]
		self.assertIsInstance(agent.generate_directly(question="q", prompt_history=[]), str)

	def test_record_never_raises_on_a_bad_sink(self):
		self.ee.configure(events_file="/nonexistent-dir/nope/events.jsonl")
		try:
			self.ee.record(OperationLabel.LLM_REASON, 1.0, 2.0)
		finally:
			self.ee.configure(events_file="")


# --------------------------------------------------------------------------
# helpers that need the CoR agent
# --------------------------------------------------------------------------

class FakeBackend:
	name = "fake"
	default_entity_prefix = "m."

	def relation_ids2labels(self, relation_ids):
		return {str(r): str(r) for r in relation_ids}

	def relation_id2label(self, relation_id):
		return str(relation_id)


def make_cor_agent(ee):
	"""A CoRAgent with a fake LLM client and fake KG, no network."""
	from chain_of_relations.methods.cor.agent import CoRAgent
	agent = CoRAgent.__new__(CoRAgent)
	agent.model_name = "test-model"
	agent.relation_width = 3
	agent.depth = 3
	agent.sample_relation_threshold = 500
	agent.remove_unnecessary_rel = True
	agent.temperature_exploration = 0.3
	agent.temperature_reasoning = 0.1
	agent.max_token = 512
	agent.kg_backend = FakeBackend()
	agent.llm_api = make_llm_api([])
	root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
	agent.prompt_file = os.path.join(
		root, "chain_of_relations", "methods", "cor", "prompt.yml")
	agent.prompts = agent._load_prompts(agent.prompt_file)
	agent._trace_seq = 0
	return agent


if __name__ == "__main__":
	unittest.main(verbosity=2)
