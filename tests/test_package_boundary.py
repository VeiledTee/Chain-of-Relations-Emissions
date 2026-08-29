"""The one-way dependency rule, and the profiler's public API.

    host agent (chain_of_relations, measurement/, adapters)
        |
        v
    agent_energy_profiler          <- never imports back up

These tests fail the build if that direction is ever reversed, if the profiler
starts inferring semantic labels instead of being told them, or if the
measurement contract in the package docstrings drifts from the code.

They deliberately do NOT re-test schema v1 semantics: tests/test_energy_measurement.py
and tests/test_hardware_validation.py own that, and both still run unchanged
against the moved implementation.
"""

import ast
import json
import os
import subprocess
import sys
import tempfile
import unittest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PACKAGE = os.path.join(ROOT, "agent_energy_profiler")
sys.path.insert(0, ROOT)

from agent_energy_profiler import carbon, labels, profiler, schema  # noqa: E402
from agent_energy_profiler import events as aep_events  # noqa: E402

#: Top-level modules the profiler must never depend on. Hosts, benchmark and
#: experiment code, dataset code, and the KG client library.
FORBIDDEN_IMPORTS = frozenset({
	"chain_of_relations",
	"measurement",
	"adapters",
	"datasets",
	"results",
	"SPARQLWrapper",
	"codecarbon",
	"audit_steps",
})


def package_files():
	"""Every .py file in the profiler package."""
	found = []
	for dirpath, dirnames, filenames in os.walk(PACKAGE):
		dirnames[:] = [d for d in dirnames if d != "__pycache__"]
		for name in sorted(filenames):
			if name.endswith(".py"):
				found.append(os.path.join(dirpath, name))
	return sorted(found)


def imported_roots(path):
	"""Top-level module name of every absolute import in one file."""
	with open(path, encoding="utf-8") as f:
		tree = ast.parse(f.read(), filename=path)
	roots = set()
	for node in ast.walk(tree):
		if isinstance(node, ast.Import):
			for alias in node.names:
				roots.add(alias.name.split(".")[0])
		elif isinstance(node, ast.ImportFrom):
			# node.level > 0 is a relative import: stays inside the package.
			if node.level == 0 and node.module:
				roots.add(node.module.split(".")[0])
	return roots


# --------------------------------------------------------------------------
# 1. the dependency rule
# --------------------------------------------------------------------------

class TestOneWayDependency(unittest.TestCase):

	def test_package_has_files(self):
		self.assertTrue(package_files(), "profiler package appears to be empty")

	def test_no_profiler_module_imports_a_host(self):
		offenders = []
		for path in package_files():
			bad = imported_roots(path) & FORBIDDEN_IMPORTS
			if bad:
				offenders.append((os.path.relpath(path, ROOT), sorted(bad)))
		self.assertEqual(offenders, [],
		                 "agent_energy_profiler must not import host, benchmark "
		                 "or experiment code: " + repr(offenders))

	def test_no_relative_import_escapes_the_package(self):
		"""A `from ...x import y` could climb out of the package silently."""
		offenders = []
		for path in package_files():
			depth = os.path.relpath(path, PACKAGE).count(os.sep) + 1
			with open(path, encoding="utf-8") as f:
				tree = ast.parse(f.read(), filename=path)
			for node in ast.walk(tree):
				if isinstance(node, ast.ImportFrom) and node.level > depth:
					offenders.append((os.path.relpath(path, ROOT), node.level))
		self.assertEqual(offenders, [], f"relative import escapes package: {offenders}")

	def test_importing_the_profiler_does_not_import_a_host(self):
		"""Checked in a clean interpreter, not this already-loaded one."""
		code = (
			"import sys; sys.path.insert(0, %r);"
			"import agent_energy_profiler;"
			"import json; print(json.dumps(sorted("
			"m for m in sys.modules if m.split('.')[0] in %r)))"
			% (ROOT, sorted(FORBIDDEN_IMPORTS))
		)
		out = subprocess.run([sys.executable, "-c", code], cwd=ROOT,
		                     capture_output=True, text=True, timeout=120)
		self.assertEqual(out.returncode, 0, out.stderr)
		leaked = json.loads(out.stdout.strip().splitlines()[-1])
		self.assertEqual(leaked, [],
		                 f"importing the profiler pulled in host code: {leaked}")

	def test_host_adapter_imports_the_profiler_not_the_reverse(self):
		adapter = os.path.join(ROOT, "chain_of_relations", "energy_events.py")
		roots = imported_roots(adapter)
		self.assertIn("agent_energy_profiler", roots)

	def test_profiler_is_importable_without_the_repo_root_on_the_path(self):
		"""Extraction check: the package must not rely on its siblings."""
		code = ("import sys; sys.path.insert(0, %r);"
		        "import agent_energy_profiler.attribution;"
		        "import agent_energy_profiler.trajectory;"
		        "import agent_energy_profiler.validation;"
		        "import agent_energy_profiler.carbon;"
		        "print('ok')" % ROOT)
		with tempfile.TemporaryDirectory() as elsewhere:
			out = subprocess.run([sys.executable, "-c", code], cwd=elsewhere,
			                     capture_output=True, text=True, timeout=120)
		self.assertEqual(out.returncode, 0, out.stderr)
		self.assertIn("ok", out.stdout)


# --------------------------------------------------------------------------
# 2. the profiler never infers semantics
# --------------------------------------------------------------------------

class TestNoSemanticInference(unittest.TestCase):

	FORBIDDEN_CALLS = ("inspect.stack", "sys._getframe", "extract_stack",
	                   "currentframe", "co_name")

	def test_package_does_not_inspect_the_call_stack(self):
		offenders = []
		for path in package_files():
			with open(path, encoding="utf-8") as f:
				text = f.read()
			for needle in self.FORBIDDEN_CALLS:
				if needle in text:
					offenders.append((os.path.relpath(path, ROOT), needle))
		self.assertEqual(offenders, [],
		                 "the profiler must be told what an operation is, not "
		                 f"guess it from the stack: {offenders}")

	def test_query_and_prompt_heuristics_live_in_the_host_adapter(self):
		"""The one structural query classifier belongs to CoR, not the profiler."""
		for path in package_files():
			with open(path, encoding="utf-8") as f:
				text = f.read()
			self.assertNotIn("type.object.name", text, path)
			self.assertNotIn("targetEntity", text, path)
			self.assertNotIn("_classify_sparql", text, path)

		adapter = os.path.join(ROOT, "chain_of_relations", "energy_events.py")
		with open(adapter, encoding="utf-8") as f:
			self.assertIn("_classify_sparql", f.read())

	def test_profiler_accepts_an_arbitrary_caller_supplied_label(self):
		"""With no vocabulary registered, any label is recorded as given."""
		saved = labels.REGISTRY.known_labels()
		try:
			labels.clear()
			operation_type, unknown = labels.REGISTRY.resolve("robotics:grasp")
			self.assertEqual(operation_type, "robotics")
			self.assertFalse(unknown)
		finally:
			labels.register(saved)

	def test_operation_type_is_the_prefix_the_caller_supplied(self):
		self.assertEqual(schema.type_of_label("llm:reason"), "llm")
		self.assertEqual(schema.type_of_label("anything:at:all"), "anything")
		self.assertEqual(schema.type_of_label("nocolon"), "")

	def test_a_label_outside_a_registered_vocabulary_is_flagged_not_dropped(self):
		saved = labels.REGISTRY.known_labels()
		try:
			labels.register(["llm:reason"], types=["llm"])
			self.assertEqual(labels.REGISTRY.resolve("llm:reason"), ("llm", False))
			self.assertEqual(labels.REGISTRY.resolve("llm:mystery"), ("llm", True))
		finally:
			labels.register(saved)


# --------------------------------------------------------------------------
# 3. public API
# --------------------------------------------------------------------------

class TestPublicAPI(unittest.TestCase):

	REQUIRED = ("span", "configure", "close", "record", "mark", "enabled",
	            "set_question", "begin_iteration", "set_traversal_depth",
	            "operation", "event_meta", "run_context")

	def test_profiler_exposes_the_documented_surface(self):
		for name in self.REQUIRED:
			self.assertTrue(hasattr(profiler, name), f"profiler.{name} missing")

	def test_schema_version_is_frozen_at_one(self):
		self.assertEqual(schema.SCHEMA_VERSION, 1)

	def test_status_vocabulary_is_controlled(self):
		self.assertEqual(
			sorted(item.value for item in schema.Status),
			["cancelled", "error", "ok", "timeout"])

	def test_taxonomy_and_profiler_share_one_schema_definition(self):
		from chain_of_relations import energy_taxonomy as tax
		self.assertIs(tax.Status, schema.Status)
		self.assertEqual(tax.SCHEMA_VERSION, schema.SCHEMA_VERSION)


class TestLazyPackageImport(unittest.TestCase):
	"""Submodules load on first access, and importing the package is inert.

	Two properties, one mechanism: `python -m agent_energy_profiler.<module>`
	must not warn that runpy is re-executing an already-imported module, and
	importing the package must not open NVML or enumerate RAPL zones. A
	measurement tool that touches the hardware merely by being imported cannot
	be trusted to have left the machine alone.
	"""

	CLI_MODULES = ("aggregate", "attribution", "validation")

	def run_python(self, *args):
		return subprocess.run([sys.executable, *args], cwd=ROOT,
		                      capture_output=True, text=True, timeout=120)

	def test_module_entry_points_run_clean_with_no_runtimewarning(self):
		for module in self.CLI_MODULES:
			with self.subTest(module=module):
				out = self.run_python("-m", f"agent_energy_profiler.{module}",
				                      "--help")
				self.assertEqual(out.returncode, 0, out.stderr)
				self.assertNotIn("RuntimeWarning", out.stderr)
				self.assertNotIn("found in sys.modules", out.stderr)

	def test_every_public_submodule_is_still_importable_by_name(self):
		import agent_energy_profiler as aep
		for name in ("aggregate", "attribution", "carbon", "events", "labels",
		             "profiler", "schema", "trajectory"):
			with self.subTest(name=name):
				attribute = getattr(aep, name)
				direct = __import__(f"agent_energy_profiler.{name}",
				                    fromlist=[name])
				self.assertIs(attribute, direct)

	def test_from_import_form_still_works(self):
		code = ("import sys; sys.path.insert(0, %r);"
		        "from agent_energy_profiler import aggregate, attribution, "
		        "carbon, events, labels, profiler, schema, trajectory;"
		        "from agent_energy_profiler import span, SCHEMA_VERSION, Status;"
		        "assert SCHEMA_VERSION == 1;"
		        "assert Status.OK.value == 'ok';"
		        "assert callable(span);"
		        "print('ok')" % ROOT)
		out = self.run_python("-c", code)
		self.assertEqual(out.returncode, 0, out.stderr)
		self.assertIn("ok", out.stdout)

	def test_unknown_attribute_still_raises_attributeerror(self):
		import agent_energy_profiler as aep
		with self.assertRaises(AttributeError):
			aep.no_such_module

	def test_dir_advertises_the_full_public_surface(self):
		import agent_energy_profiler as aep
		listed = set(dir(aep))
		for name in aep.__all__:
			self.assertIn(name, listed)

	def test_importing_the_package_does_not_touch_the_hardware(self):
		"""No NVML init, no RAPL enumeration, no sampler, on a bare import."""
		code = ("import sys; sys.path.insert(0, %r);"
		        "import agent_energy_profiler;"
		        "import json;"
		        "print(json.dumps({"
		        "'pynvml': 'pynvml' in sys.modules,"
		        "'sampling': 'agent_energy_profiler.sampling' in sys.modules,"
		        "'hardware': 'agent_energy_profiler.hardware' in sys.modules,"
		        "}))" % ROOT)
		out = self.run_python("-c", code)
		self.assertEqual(out.returncode, 0, out.stderr)
		state = json.loads(out.stdout.strip().splitlines()[-1])
		self.assertFalse(state["pynvml"], "importing the package opened NVML")
		self.assertFalse(state["sampling"],
		                 "importing the package loaded the sampler, which "
		                 "enumerates RAPL zones and initialises NVML at import")
		self.assertFalse(state["hardware"])

	def test_nvml_stays_uninitialised_until_something_asks_for_energy(self):
		code = ("import sys; sys.path.insert(0, %r);"
		        "import agent_energy_profiler as aep;"
		        "aep.events;"  # loads the recorder, which imports hardware.nvml
		        "from agent_energy_profiler.hardware import nvml;"
		        "print(nvml.available(), nvml.device_count())" % ROOT)
		out = self.run_python("-c", code)
		self.assertEqual(out.returncode, 0, out.stderr)
		self.assertIn("False 0", out.stdout)


class TestSpanAPI(unittest.TestCase):
	"""span() is the host-facing recording primitive."""

	def setUp(self):
		self._tmp = tempfile.TemporaryDirectory()
		self.path = os.path.join(self._tmp.name, "events.jsonl")
		os.environ["ENERGY_EVENTS_FILE"] = self.path
		aep_events.reset_from_env()

	def tearDown(self):
		aep_events.close()
		os.environ.pop("ENERGY_EVENTS_FILE", None)
		aep_events.reset_from_env()
		self._tmp.cleanup()

	def read(self):
		if not os.path.exists(self.path):
			return []
		with open(self.path) as f:
			return [json.loads(line) for line in f if line.strip()]

	def test_span_records_the_label_the_caller_supplied(self):
		with profiler.span("llm:reason"):
			pass
		event, = self.read()
		self.assertEqual(event["operation_label"], "llm:reason")
		self.assertEqual(event["operation_type"], "llm")
		self.assertEqual(event["status"], "ok")
		self.assertGreaterEqual(event["duration_s"], 0.0)

	def test_span_carries_attributes_and_token_counts(self):
		with profiler.span("llm:reason", attributes={"tool_attempt": 2}) as s:
			s["input_tokens"] = 11
			s["output_tokens"] = 3
		event, = self.read()
		self.assertEqual(event["input_tokens"], 11)
		self.assertEqual(event["output_tokens"], 3)
		self.assertEqual(event["meta"]["tool_attempt"], 2)

	def test_span_records_a_failure_as_error_and_reraises(self):
		with self.assertRaises(ValueError):
			with profiler.span("kg:sparql"):
				raise ValueError("query failed")
		event, = self.read()
		self.assertEqual(event["status"], "error")
		self.assertEqual(event["meta"]["exception_type"], "ValueError")

	def test_span_declares_the_operation_for_nested_records(self):
		with profiler.span("llm:reason"):
			self.assertEqual(aep_events.current_operation(), "llm:reason")

	def test_span_is_a_no_op_when_measurement_is_disabled(self):
		os.environ.pop("ENERGY_EVENTS_FILE", None)
		aep_events.reset_from_env()
		with profiler.span("llm:reason") as s:
			s["input_tokens"] = 1
		self.assertEqual(self.read(), [])


# --------------------------------------------------------------------------
# 4. the host adapter stays thin
# --------------------------------------------------------------------------

class TestHostAdapterIsThin(unittest.TestCase):

	ALGORITHM_FILES = (
		"chain_of_relations/methods/cor/agent.py",
		"chain_of_relations/llm_api.py",
		"chain_of_relations/run.py",
		"chain_of_relations/kg_backend/freebase/db_func.py",
	)

	#: Guard against instrumentation creeping back into the algorithm. This is
	#: a ceiling, not a target: measurement belongs in the profiler and the
	#: adapter, and the algorithm should only ever say what it is doing.
	MAX_INSTRUMENTATION_LINES = 60

	def instrumentation_lines(self, relative):
		with open(os.path.join(ROOT, relative), encoding="utf-8") as f:
			return [line for line in f
			        if "energy_events" in line or "energy_taxonomy" in line
			        or "agent_energy_profiler" in line]

	def test_instrumentation_inside_the_algorithm_stays_small(self):
		total = sum(len(self.instrumentation_lines(p)) for p in self.ALGORITHM_FILES)
		self.assertLessEqual(
			total, self.MAX_INSTRUMENTATION_LINES,
			f"{total} instrumentation lines inside CoR algorithm code; the "
			"measurement layer should absorb this, not the algorithm")

	def test_algorithm_files_do_not_touch_hardware_counters(self):
		"""CoR declares semantics. It must not read NVML or RAPL itself."""
		for relative in self.ALGORITHM_FILES:
			with open(os.path.join(ROOT, relative), encoding="utf-8") as f:
				text = f.read()
			for needle in ("pynvml", "powercap", "energy_uj", "nvmlDevice"):
				self.assertNotIn(needle, text, f"{relative} reads hardware directly")

	def test_adapter_holds_no_energy_arithmetic(self):
		adapter = os.path.join(ROOT, "chain_of_relations", "energy_events.py")
		with open(adapter, encoding="utf-8") as f:
			text = f.read()
		for needle in ("measured_energy_j", "cpu_package_energy_j", "1e6", "1e3"):
			self.assertNotIn(needle, text,
			                 "energy accounting belongs in the profiler")


# --------------------------------------------------------------------------
# 5. carbon conversion is explicit and never estimated
# --------------------------------------------------------------------------

class TestCarbonConversion(unittest.TestCase):

	def intensity(self):
		return carbon.CarbonIntensity(
			value_g_co2e_per_kwh=200.0, region="GB",
			source="test fixture", reference_timestamp="2026-01-01T00:00:00Z",
			retrieved_timestamp="2026-01-02T00:00:00Z", method="average")

	def test_conversion_is_energy_times_supplied_intensity(self):
		# 3.6e6 J = 1 kWh, at 200 gCO2e/kWh
		self.assertAlmostEqual(
			carbon.to_co2e_g(3.6e6, self.intensity()), 200.0, places=6)

	def test_missing_energy_stays_null_and_never_becomes_zero(self):
		self.assertIsNone(carbon.to_co2e_g(None, self.intensity()))
		self.assertIsNone(carbon.joules_to_kwh(None))

	def test_there_is_no_default_grid_intensity(self):
		"""A caller must supply one; the module must not invent a factor."""
		with self.assertRaises(TypeError):
			carbon.to_co2e_g(1.0)

	def test_provenance_is_retained_on_every_conversion(self):
		record = carbon.convert(3.6e6, self.intensity())
		provenance = record["carbon_intensity"]
		for key in ("value_g_co2e_per_kwh", "region", "source", "units",
		            "method", "reference_timestamp", "retrieved_timestamp"):
			self.assertIn(key, provenance)
		self.assertEqual(provenance["units"], "gCO2e/kWh")

	def test_converting_events_preserves_every_semantic_field(self):
		source = [{
			"question_id": "q1", "operation_label": "llm:reason",
			"iteration": 2, "traversal_depth": 1, "step_index": 7,
			"status": "ok", "measured_energy_j": 3.6e6,
		}, {
			"question_id": "q1", "operation_label": "kg:sparql",
			"iteration": 3, "traversal_depth": None, "step_index": 8,
			"status": "ok", "measured_energy_j": None,
		}]
		rows, provenance = carbon.convert_events(source, self.intensity())
		self.assertEqual(len(rows), 2)
		for original, row in zip(source, rows):
			for key, value in original.items():
				self.assertEqual(row[key], value)
		self.assertAlmostEqual(rows[0]["co2e_g"], 200.0, places=6)
		self.assertIsNone(rows[1]["co2e_g"],
		                  "an unmeasured domain must not become zero emissions")
		self.assertEqual(provenance["region"], "GB")


if __name__ == "__main__":
	unittest.main()
