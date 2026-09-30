"""cor-rank-compact: CoR with a <=10-word relation_rank rationale (C2 intervention).

The variant must differ from baseline CoR in one thing only: the rationale
length instruction in the relation_pruning prompt. These tests drive the real
CoRAgent control flow, the real shared relation_prune tool/parser and the real
reasoning tool, with a fake chat-completions client (routed by stage and by the
path in the prompt) and a deterministic fake KG at the agent's search methods.
"""

import copy
import difflib
import json
import os
import sys
import tempfile
import unittest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "tests"))
sys.path.insert(0, os.path.join(ROOT, "measurement"))

import export_outcomes  # noqa: E402
from test_energy_measurement import FakeBackend  # noqa: E402
from test_tog_pog_instrumentation import (  # noqa: E402
	EventsHarness, FakeResponse, FakeUsage, make_llm_api)

from chain_of_relations import run as run_module  # noqa: E402
from chain_of_relations.methods.cor.agent import CoRAgent  # noqa: E402
from chain_of_relations.methods.cor_rank_compact import agent as compact_module  # noqa: E402
from chain_of_relations.methods.cor_rank_compact.agent import (  # noqa: E402
	COMPACT_PROMPT_FILE, METHOD_ID, CoRRankCompactAgent, rationale_stats)
from chain_of_relations.schema import Entity, Relation  # noqa: E402

BASELINE_PROMPT_FILE = os.path.join(ROOT, "chain_of_relations", "methods", "cor", "prompt.yml")

#: The complete intended prompt difference, as (baseline line, variant line)
#: of the loaded system prompt (YAML strips the block's indentation).
INTENDED_DIFF = [
	("6. Keep rationale short and specific.",
	 "6. Keep rationale to no more than 10 words."),
	('  "rational": "brief reasoning for selection",',
	 '  "rational": "reasoning in no more than 10 words",'),
	('  "rational": "The question asks for locations connected to where Saki lived, so prioritize '
	 'relations that expose concrete place signals tied to the person and avoid non-location metadata.",',
	 '  "rational": "Location question: prefer relations linking Saki to places.",'),
	('  "rational": "The target answer type is character, so first choose the direct '
	 'performance-to-character edge, then relations that enforce the Lord of the Rings film constraint '
	 'or provide robust cast linkage for recovery.",',
	 '  "rational": "Character answer: performance-character edge, then film constraint relations.",'),
]

LONG_RATIONALE = ("The question asks which relation links the topic entity to the answer type, so "
                  "prioritize the direct edge and then the relation that best enforces the constraint.")
SHORT_RATIONALE = "Best matches the question."

ALICE = Entity(id="m.alice", name="Alice")


def rank_reply(relations, rationale, style="plain"):
	"""A relation_rank reply in CoR's real schema ("rational" sic)."""
	body = json.dumps({"rational": rationale, "relation_by_relevance": list(relations)})
	if style == "fenced":
		return "```json\n" + body + "\n```"
	if style == "spaced":
		return json.dumps({"rational": rationale, "relation_by_relevance": list(relations)}, indent=4)
	return body


def _new_agent(cls, ee):
	agent = cls.__new__(cls)
	agent.model_name = "test-model"
	agent.relation_width = 3
	agent.depth = 3
	agent.sample_relation_threshold = 500
	agent.remove_unnecessary_rel = True
	agent.temperature_exploration = 0.01
	agent.temperature_reasoning = 0.01
	agent.max_token = 512
	agent.kg_backend = FakeBackend()
	agent.llm_api = make_llm_api(None)
	agent.prompt_file = BASELINE_PROMPT_FILE
	agent.prompts = agent._load_prompts(agent.prompt_file)
	agent._trace_seq = 0
	if cls is CoRRankCompactAgent:
		agent._use_compact_rank_prompt()
	return agent


def make_baseline(ee):
	return _new_agent(CoRAgent, ee)


def make_compact(ee):
	return _new_agent(CoRRankCompactAgent, ee)


class Script:
	"""Replays scripted replies, one per chat-completions request."""

	def __init__(self, replies):
		self.replies = list(replies)
		self.requests = []

	def create(self, **kwargs):
		self.requests.append(copy.deepcopy(kwargs))
		item = self.replies.pop(0)
		if isinstance(item, Exception):
			raise item
		return item


def use(agent, completions):
	agent.llm_api.client.chat.completions = completions
	return completions


def system_of(request):
	return next((m["content"] for m in request["messages"] if m["role"] == "system"), None)


def user_of(request):
	return request["messages"][-1]["content"]


def rel_tuples(relations):
	return [(r.id, bool(r.left), r.score) for r in relations]


# --------------------------------------------------------------------------
# A deterministic CoR world: fake KG keyed by relation chain, fake LLM keyed
# by stage and by the path rendered into the prompt.
# --------------------------------------------------------------------------

def sig(chain):
	return tuple((r.id, bool(r.left)) for r in chain)


#: relation_search(chain) -> (head, tail)
RELATIONS = {
	(): (["r.a", "r.b"], ["r.c"]),
	(("r.a", True),): (["r.d"], ["r.a"]),                  # r.a reverse is acyclic-filtered
	(("r.a", True), ("r.d", True)): (["r.f"], []),
	(("r.c", False),): (["r.g"], ["r.h"]),
}
#: entity_search(chain) -> (ids, names, literals)
ENTITIES = {
	(("r.a", True),): (["m.bob"], ["Bob"], []),
	(("r.a", True), ("r.d", True)): (["m.carol"], ["Carol"], []),
	(("r.a", True), ("r.d", True), ("r.f", True)): ([], [], []),   # depth 3, empty -> Backtrack
	(("r.c", False),): (["m.dave"], ["Dave"], []),
	(("r.c", False), ("r.h", False)): (["m.erin"], ["Erin"], []),
}
#: first "# " snippet line of a relation_rank prompt -> selected relations
RANKS = {
	"TopicEntity(Alice) ?relation ?x": ["r.a", "r.c"],                                   # C: several, B: r.c backward
	"TopicEntity(Alice) r.a e1 . e1 ?relation ?x": ["r.d", "r.a"],                       # D: previous hop
	"TopicEntity(Alice) r.a e1 . e1 r.d e2 . e2 ?relation ?x": ["r.f"],                  # E: deeper state
	"e1 r.c TopicEntity(Alice) . e1 ?relation ?x": ["r.h", "r.g"],                       # F: after backtrack
}
#: reasoning path (without the candidate line) -> decision/answer
REASONS = {
	"TopicEntity(Alice) r.a e1 .": ("forward", []),
	"TopicEntity(Alice) r.a e1 .\ne1 r.d e2 .": ("forward", []),
	"e1 r.c TopicEntity(Alice) .": ("forward", []),
	"e1 r.c TopicEntity(Alice) .\ne2 r.h e1 .": ("stop", ["Erin"]),
}


class WorldLLM:
	"""Answers every CoR stage deterministically; only the rank rationale varies."""

	def __init__(self, rationale):
		self.rationale = rationale
		self.requests = []

	def create(self, **kwargs):
		self.requests.append(copy.deepcopy(kwargs))
		system, user = system_of(kwargs), user_of(kwargs)
		if "relation-pruning expert" in system:
			snippet = next(line[2:] for line in user.splitlines() if line.startswith("# "))
			content = rank_reply(RANKS[snippet], self.rationale, style="fenced")
			return FakeResponse(content, FakeUsage(100, 20 + len(self.rationale.split())))
		if "reasoning-path validator" in system:
			lines = user.split("Reasoning Chain (SPARQL-style):\n", 1)[1].split("\nAnswer:")[0].splitlines()
			path = "\n".join(line for line in lines if "candidate.target" not in line)
			decision, answer = REASONS[path]
			content = json.dumps({"rationale": "Path check.", "decision": decision, "answer": answer})
			return FakeResponse(content, FakeUsage(80, 15))
		raise AssertionError("unexpected stage in the controlled world: " + system[:80])


def wire_world(agent, rationale):
	llm = use(agent, WorldLLM(rationale))
	agent.relation_search = lambda topic_entity, relation_chain, sparql_history: \
		tuple(list(x) for x in RELATIONS[sig(relation_chain)])
	agent.entity_search = lambda topic_entity, relation_chain, sparql_history: \
		tuple(list(x) for x in ENTITIES[sig(relation_chain)])
	return llm


# --------------------------------------------------------------------------
# 1-6, 17: identity, registration, prompt isolation
# --------------------------------------------------------------------------

class MethodIdentity(unittest.TestCase):

	def test_separate_method_registration(self):
		self.assertEqual(METHOD_ID, "cor-rank-compact")
		args = run_module.build_parser().parse_args(["--method", "cor-rank-compact"])
		self.assertEqual(args.method, "cor-rank-compact")
		self.assertIs(run_module.load_agent_class("cor-rank-compact"), CoRRankCompactAgent)
		self.assertIs(run_module.load_agent_class("cor"), CoRAgent)
		self.assertTrue(issubclass(CoRRankCompactAgent, CoRAgent))

	def test_output_identity_is_separate_from_baseline(self):
		# run.py derives the results dir and the energy paradigm from args.method;
		# run_experiments.sh derives the measurement tag from the method name.
		source = open(os.path.join(ROOT, "chain_of_relations", "run.py")).read()
		self.assertIn('PROJECT_ROOT / "results" / args.method / args.dataset / model_dirname', source)
		self.assertIn("paradigm=args.method", source)
		shell = open(os.path.join(ROOT, "scripts", "run_experiments.sh")).read()
		self.assertIn('TAG="${TAG_PREFIX:+${TAG_PREFIX}_}${METHOD}_${DATASET}_${STAMP}"', shell)
		baseline_dir = os.path.join("results", "cor", "cwq")
		variant_dir = os.path.join("results", METHOD_ID, "cwq")
		self.assertNotEqual(os.path.commonpath([baseline_dir, variant_dir]), baseline_dir)
		self.assertFalse(f"{METHOD_ID}_cwq_0101_0000".startswith("cor_cwq_"))

	def test_real_constructor_uses_compact_rank_prompt(self):
		os.environ.setdefault("OPENAI_API_KEY", "test-key")
		agent = CoRRankCompactAgent(model_name="test-model", backend=FakeBackend())
		self.assertIn("no more than 10 words", agent._get_system_prompt("relation_pruning"))
		self.assertEqual(agent.rank_prompt_file, COMPACT_PROMPT_FILE)
		self.assertEqual(agent.max_token, CoRAgent(model_name="test-model", backend=FakeBackend()).max_token)


class PromptIsolation(unittest.TestCase):

	def setUp(self):
		self.base = make_baseline(None).prompts
		self.compact = make_compact(None).prompts

	def test_only_the_rationale_length_lines_differ(self):
		base_sys = self.base["relation_pruning"]["system_prompt"].splitlines()
		comp_sys = self.compact["relation_pruning"]["system_prompt"].splitlines()
		self.assertEqual(len(base_sys), len(comp_sys))
		changed = [(b, c) for b, c in zip(base_sys, comp_sys) if b != c]
		self.assertEqual(changed, INTENDED_DIFF)
		# the unified diff agrees: nothing inserted, deleted or reordered
		diff = [l for l in difflib.ndiff(base_sys, comp_sys) if l[:1] in "+-"]
		self.assertEqual(len(diff), 2 * len(INTENDED_DIFF))

	def test_user_template_and_every_other_stage_identical(self):
		self.assertEqual(self.base["relation_pruning"]["user_prompt"],
		                 self.compact["relation_pruning"]["user_prompt"])
		for stage in set(self.base) | set(self.compact):
			if stage != "relation_pruning":
				self.assertEqual(self.base[stage], self.compact[stage], stage)

	def test_ranking_semantics_preserved_in_compact_prompt(self):
		text = self.compact["relation_pruning"]["system_prompt"]
		for needle in ("select the most useful top-3 relations",
		               "Select no more than top-3 relations",
		               "Use the SPARQL snippets as the source of truth for direction",
		               "TopicEntity(Tom) father.of ?x => ?x is Tom's child.",
		               '"relation_by_relevance": ["relation_1", "relation_2"]',
		               '"rational": ',
		               "must be ordered from most relevant to less relevant",
		               "Include no more than 3 relations.",
		               "Return exactly one JSON object."):
			self.assertIn(needle, text)
		# few-shot examples: same questions, candidates and selections, in order
		def examples(prompt):
			lines = prompt.splitlines()
			return [l for l in lines[lines.index("# For Examples"):] if '"rational"' not in l]
		self.assertEqual(examples(text), examples(self.base["relation_pruning"]["system_prompt"]))
		self.assertIn('"relation_by_relevance": ["people.person.place_of_birth", '
		              '"people.deceased_person.place_of_death", "people.person.nationality"]', text)
		self.assertIn('"relation_by_relevance": ["film.performance.character", '
		              '"film.performance.film", "film.film.starring"]', text)

	def test_every_rationale_shown_to_the_model_is_at_most_10_words(self):
		import re
		for name, prompts, limit_ok in (("compact", self.compact, True), ("baseline", self.base, False)):
			shown = re.findall(r'"rational": "([^"]*)"', prompts["relation_pruning"]["system_prompt"])
			self.assertEqual(len(shown), 3, name)  # format hint + two examples
			self.assertEqual(all(len(s.split()) <= 10 for s in shown), limit_ok, name)

	def test_compact_prompt_has_brevity_and_baseline_does_not(self):
		self.assertIn("no more than 10 words", self.compact["relation_pruning"]["system_prompt"])
		base_text = open(BASELINE_PROMPT_FILE).read()
		self.assertNotIn("10 words", base_text)
		self.assertIn("6. Keep rationale short and specific.", base_text)

	def test_variant_file_only_defines_the_ranking_stage(self):
		import yaml
		with open(COMPACT_PROMPT_FILE) as f:
			self.assertEqual(set(yaml.safe_load(f)), {"relation_pruning"})


class Coexistence(unittest.TestCase):
	"""18: both in one process, in either construction order, no leakage."""

	def test_no_prompt_state_leaks_between_instances(self):
		base_before = make_baseline(None)
		compact = make_compact(None)
		base_after = make_baseline(None)
		for base in (base_before, base_after):
			self.assertIn("Keep rationale short and specific.",
			              base._get_system_prompt("relation_pruning"))
			self.assertNotIn("10 words", base._get_system_prompt("relation_pruning"))
		self.assertIn("no more than 10 words", compact._get_system_prompt("relation_pruning"))
		self.assertIsNot(compact.prompts, base_before.prompts)
		self.assertIsNot(compact.prompts["reasoning"], base_before.prompts["reasoning"])
		self.assertIs(type(base_after).relation_prune, CoRAgent.relation_prune)
		self.assertIs(type(base_after)._llm_event_to_step, CoRAgent._llm_event_to_step)


# --------------------------------------------------------------------------
# 7-12, 16, 17: single ranking calls through the real tool, both agents
# --------------------------------------------------------------------------

class RankingCalls(EventsHarness):

	def rank_both(self, chain, head, tail, relations, compact_rationale=SHORT_RATIONALE,
	              compact_style="plain"):
		out = {}
		for name, make, reply in (
				("base", make_baseline, rank_reply(relations, LONG_RATIONALE)),
				("compact", make_compact, rank_reply(relations, compact_rationale, compact_style))):
			agent = make(self.ee)
			script = use(agent, Script([FakeResponse(reply, FakeUsage(100, 30))]))
			history = []
			selected = agent.relation_prune(question="q?", topic_entity=ALICE, relation_chain=chain,
			                                head_relations=head, tail_relations=tail,
			                                prompt_history=history)
			out[name] = (rel_tuples(selected), history, script.requests)
		base_req, comp_req = out["base"][2][0], out["compact"][2][0]
		self.assertEqual(user_of(base_req), user_of(comp_req))
		for key in ("temperature", "max_tokens", "model"):
			self.assertEqual(base_req[key], comp_req[key], key)
		self.assertEqual(out["base"][0], out["compact"][0])
		return out

	def test_A_forward_ranking(self):
		out = self.rank_both([], ["r.a", "r.b"], [], ["r.b", "r.a"])
		self.assertEqual(out["compact"][0], [("r.b", True, 0.0), ("r.a", True, 0.0)])

	def test_B_backward_direction_preserved(self):
		out = self.rank_both([], ["r.a"], ["r.c"], ["r.c", "r.a"])
		self.assertEqual(out["compact"][0], [("r.c", False, 0.0), ("r.a", True, 0.0)])

	def test_C_multiple_candidates_capped_at_width(self):
		out = self.rank_both([], ["r.a", "r.b", "r.d"], ["r.c"], ["r.d", "r.c", "r.b", "r.a"])
		self.assertEqual([r[0] for r in out["compact"][0]], ["r.d", "r.c", "r.b"])

	def test_D_previous_hop_path_and_acyclic_filter(self):
		chain = [Relation(id="r.a", left=True)]
		out = self.rank_both(chain, ["r.d"], ["r.a"], ["r.a", "r.d"])
		self.assertEqual(out["compact"][0], [("r.d", True, 0.0)])
		self.assertIn("TopicEntity(Alice) r.a e1 . e1 ?relation ?x", user_of(out["compact"][2][0]))

	def test_E_deeper_state(self):
		chain = [Relation(id="r.a", left=True), Relation(id="r.d", left=False)]
		out = self.rank_both(chain, ["r.f"], ["r.g"], ["r.g", "r.f"])
		self.assertEqual(out["compact"][0], [("r.g", False, 0.0), ("r.f", True, 0.0)])
		self.assertIn("TopicEntity(Alice) r.a e1 . e2 r.d e1 . ?x ?relation e2", user_of(out["compact"][2][0]))

	def test_short_rationales_parse_to_the_same_selection(self):
		for rationale, style in (
				("Birthplace relation directly answers where the person was born today.", "plain"),  # 10 words
				("Direct answer.", "plain"),
				("Best.", "plain"),
				("Most relevant relation.", "fenced"),
				('Answer type: place; "born" -> birthplace {direct}!', "plain"),
				("Matches the question, (clearly).", "spaced")):
			with self.subTest(rationale=rationale, style=style):
				self.rank_both([], ["r.a", "r.b"], ["r.c"], ["r.c", "r.b"],
				               compact_rationale=rationale, compact_style=style)

	def test_rank_compact_metadata(self):
		out = self.rank_both([], ["r.a", "r.b"], [], ["r.b", "r.a"])
		record = out["compact"][1][0]
		self.assertEqual(record["rank_compact"], {
			"rationale_parsed": True, "rationale_words": 4, "rationale_chars": len(SHORT_RATIONALE),
			"selected_relation_count": 2, "trace_id": record["trace_id"]})
		self.assertNotIn("rank_compact", out["base"][1][0])
		step = CoRRankCompactAgent._llm_event_to_step(record)
		self.assertEqual(step["rank_compact"], record["rank_compact"])
		self.assertEqual(step["output_tokens"], 30)  # token count stays where it was, not duplicated

	def test_rationale_stats_never_invents_zero(self):
		self.assertEqual(rationale_stats("not json"),
		                 {"rationale_parsed": False, "rationale_words": None, "rationale_chars": None})
		self.assertEqual(rationale_stats(None)["rationale_words"], None)
		self.assertEqual(rationale_stats('{"relation_by_relevance": ["r.a"]}')["rationale_words"], None)
		self.assertEqual(rationale_stats(rank_reply(["r.a"], "", "fenced"))["rationale_words"], 0)

	def test_energy_label_and_paradigm(self):
		self.ee.configure(paradigm=METHOD_ID)
		agent = make_compact(self.ee)
		use(agent, Script([FakeResponse(rank_reply(["r.a"], SHORT_RATIONALE), FakeUsage(100, 12))]))
		agent.relation_prune(question="q?", topic_entity=ALICE, relation_chain=[],
		                     head_relations=["r.a"], tail_relations=[], prompt_history=[])
		event, = self.read_events()
		self.assertEqual(event["operation_label"], "llm:relation_rank")
		self.assertEqual(event["operation_type"], "llm")
		self.assertEqual(event["paradigm"], "cor-rank-compact")
		self.assertEqual((event["input_tokens"], event["output_tokens"]), (100, 12))
		self.assertEqual(event["status"], "ok")


# --------------------------------------------------------------------------
# 13-14: failure and retry behaviour identical to baseline
# --------------------------------------------------------------------------

class FailureAndRetry(EventsHarness):

	def outcome(self, make, replies, head=("r.a",), tail=()):
		agent = make(self.ee)
		script = use(agent, Script(replies))
		history = []
		selected = agent.relation_prune(question="q?", topic_entity=ALICE, relation_chain=[],
		                                head_relations=list(head), tail_relations=list(tail),
		                                prompt_history=history)
		events = [(e["operation_label"], e["status"], e["meta"].get("tool_attempt"))
		          for e in self.read_events()]
		self.ee.close()
		open(self.events_path, "w").close()
		record = {k: v for k, v in history[0].items() if k != "rank_compact"}
		return rel_tuples(selected), record["parsed_result"], record["warnings"], len(script.requests), events

	def assert_same(self, replies, **kw):
		base = self.outcome(make_baseline, copy.deepcopy(replies), **kw)
		compact = self.outcome(make_compact, copy.deepcopy(replies), **kw)
		self.assertEqual(base, compact)
		return compact

	def test_malformed_json(self):
		selected, _, warnings, calls, _ = self.assert_same([FakeResponse('{"rational": "x", ', FakeUsage(9, 9))])
		self.assertEqual(selected, [])
		self.assertIn("Failed to parse relation prune result: invalid json output", warnings)
		self.assertEqual(calls, 1)  # a parse failure is not retried, in either method

	def test_missing_relation_field(self):
		selected, _, warnings, _, _ = self.assert_same([FakeResponse('{"rational": "Short."}', FakeUsage(9, 3))])
		self.assertEqual(selected, [])
		self.assertIn("Failed to parse relation prune result: no relations found", warnings)

	def test_relation_not_in_candidates(self):
		selected, _, _, _, _ = self.assert_same([FakeResponse(rank_reply(["r.zzz"], "Short."), FakeUsage(9, 5))])
		self.assertEqual(selected, [])

	def test_empty_selection(self):
		selected, _, _, _, _ = self.assert_same([FakeResponse(rank_reply([], "Short."), FakeUsage(9, 5))])
		self.assertEqual(selected, [])

	def test_retry_on_empty_reply_then_success(self):
		replies = [FakeResponse("", FakeUsage(9, 0)), FakeResponse("", FakeUsage(9, 0)),
		           FakeResponse("", FakeUsage(9, 0)),  # LLMAPI's own retries for pass 1
		           FakeResponse(rank_reply(["r.a"], "Short."), FakeUsage(9, 5))]
		selected, _, _, calls, events = self.assert_same(replies)
		self.assertEqual(selected, [("r.a", True, 0.0)])
		self.assertEqual(calls, 4)
		self.assertTrue(all(label == "llm:relation_rank" for label, _, _ in events))


# --------------------------------------------------------------------------
# 12, 19: full DFS in the controlled world, baseline vs compact
# --------------------------------------------------------------------------

class DeterministicEquivalence(EventsHarness):

	def run_world(self, make, rationale):
		agent = make(self.ee)
		agent.depth = 3
		llm = wire_world(agent, rationale)
		self.ee.set_question("q-world")
		result = agent.answer("who is it?", [ALICE])
		events = self.read_events()
		self.ee.close()
		open(self.events_path, "w").close()
		return result, llm.requests, events

	def test_same_search_same_answer_only_rank_system_prompt_and_rationale_differ(self):
		base, base_req, base_ev = self.run_world(make_baseline, LONG_RATIONALE)
		comp, comp_req, comp_ev = self.run_world(make_compact, SHORT_RATIONALE)

		# final output
		for key in ("action", "results", "reasoning_chains"):
			self.assertEqual(base[key], comp[key], key)
		self.assertEqual(comp["action"], "Stop")
		self.assertEqual([r["name"] for r in comp["results"]], ["Erin"])

		# every request: same user prompt and generation parameters; system
		# prompts identical except the ranking stage's two intended lines
		self.assertEqual(len(base_req), len(comp_req))
		n_rank = 0
		for b, c in zip(base_req, comp_req):
			self.assertEqual(user_of(b), user_of(c))
			for key in ("temperature", "max_tokens", "model"):
				self.assertEqual(b[key], c[key])
			if "relation-pruning expert" in system_of(b):
				n_rank += 1
				changed = [(x, y) for x, y in zip(system_of(b).splitlines(), system_of(c).splitlines()) if x != y]
				self.assertEqual(changed, INTENDED_DIFF)
			else:
				self.assertEqual(system_of(b), system_of(c))
				self.assertNotIn(LONG_RATIONALE, user_of(b))  # rationale never reaches later prompts
		self.assertEqual(n_rank, 4)

		# same selections, same DFS decisions, same event trajectory
		def llm_trace(result):
			return [(s["operation_type"], s["parsed_result"]) for s in result["step_history"] if s["step_type"] == "llm"]
		self.assertEqual(llm_trace(base), llm_trace(comp))
		self.assertEqual([p for t, p in llm_trace(comp) if t == "reasoning"], ["Forward", "Forward", "Forward", "Stop"])

		def ev_trace(events):
			return [(e["operation_label"], e["iteration"], e["traversal_depth"], e["step_index"], e["status"])
			        for e in events]
		self.assertEqual(ev_trace(base_ev), ev_trace(comp_ev))

		# F: a ranking call happens after the depth-3 backtrack, at a shallower depth
		depths = [(e["operation_label"], e["traversal_depth"]) for e in comp_ev]
		rank_depths = [d for label, d in depths if label == "llm:relation_rank"]
		self.assertEqual(rank_depths, [0, 1, 2, 1])

		# only the rationale text and its output-token count differ in the saved steps
		b_rank = [s for s in base["step_history"] if s.get("operation_type") == "relation_prune"]
		c_rank = [s for s in comp["step_history"] if s.get("operation_type") == "relation_prune"]
		for b, c in zip(b_rank, c_rank):
			self.assertLess(c["output_tokens"], b["output_tokens"])
			self.assertEqual(c["rank_compact"]["rationale_words"], 4)
			self.assertNotIn("rank_compact", b)


# --------------------------------------------------------------------------
# 15, 20: evaluator and output isolation
# --------------------------------------------------------------------------

class EvaluatorCompatibility(unittest.TestCase):

	def test_scores_unaffected_by_experimental_metadata(self):
		record = {"id": "q1", "action": "Stop", "gold_answer": [{"name": "Erin"}],
		          "results": [{"id": "m.erin", "name": "Erin"}]}
		rows = {}
		with tempfile.TemporaryDirectory() as tmp:
			for name, rec in (("plain", record),
			                  ("meta", dict(record, rank_compact={"rationale_words": 4}))):
				path = os.path.join(tmp, f"{name}.jsonl")
				with open(path, "w") as f:
					f.write(json.dumps(rec) + "\n")
				rows[name] = export_outcomes.score(path, "cwq", METHOD_ID)
		self.assertEqual(rows["plain"], rows["meta"])
		self.assertEqual((rows["meta"][0]["hit1"], rows["meta"][0]["f1"]), (1, 1.0))

	def test_module_does_not_touch_baseline_prompt_file(self):
		self.assertNotEqual(os.path.realpath(COMPACT_PROMPT_FILE), os.path.realpath(BASELINE_PROMPT_FILE))
		self.assertEqual(compact_module.RANK_PROMPT_KEY, "relation_pruning")


if __name__ == "__main__":
	unittest.main()
