"""agent_energy_profiler -- hardware-grounded, operation-level energy profiling
for agentic systems.

The profiler measures spans of work against hardware energy counters. The HOST
agent says what each span means. That split is the whole design:

    host agent  ->  agent_energy_profiler        allowed
    agent_energy_profiler  ->  host agent        FORBIDDEN

Nothing in this package imports a host, a benchmark, a dataset or a knowledge
graph, and nothing in it infers a semantic label from prompt text, function
names or the call stack. A label arrives from the caller that knows why the
work is happening, or it does not exist. (tests/test_package_boundary.py fails
the build if either rule is broken.)

Four layers, kept separate on purpose:

    1. events.py       semantic event timeline        -> events.jsonl
    2. sampling.py     hardware counter timeline      -> power.csv
    3. attribution.py  joins 1 and 2                  -> events_attributed.jsonl
    4. trajectory.py   reconciliation and coverage    -> trajectory_summary.*
    5. aggregate.py    long-format rollups            -> plotting-ready tables

Layer 5 is derived: it never re-measures anything, and refuses to report a
share, a residual or a total it cannot defend from layers 1-4. Plotting lives
outside this package.

Measurement boundary, enforced identically at every layer:

    measured_energy_j = gpu_energy_j + cpu_package_energy_j + dram_energy_j

CPU `core`/`uncore` are diagnostic and never added (core is contained within
package); `psys` is excluded (it contains package). A domain the hardware did
not report stays null -- never a fabricated zero -- and any total with a
missing term is null too.

Energy in Joules is the measured product. Carbon is a separate, explicit
conversion step (`carbon.py`) that requires the caller to supply a grid
intensity with provenance; nothing here estimates one.

Quick start:

    from agent_energy_profiler import profiler

    with profiler.span("llm:reason") as s:
        ...
        s["output_tokens"] = n
"""

import importlib

__version__ = "0.1.0"

#: Submodules reachable as attributes of the package. Resolved on first access
#: rather than at import (PEP 562), for two reasons:
#:
#:   * `python -m agent_energy_profiler.<module>` warns when the package
#:     __init__ has already imported the module runpy is about to execute;
#:   * importing the package should cost nothing and touch no hardware. The
#:     sampler in particular opens NVML and enumerates RAPL zones at import,
#:     so it is deliberately NOT listed here and stays an explicit import.
#:
#: The set matches the previously eager one exactly: this changes when a
#: submodule is loaded, never which names exist.
_SUBMODULES = frozenset({
	"aggregate", "attribution", "carbon", "events", "labels", "profiler",
	"schema", "trajectory",
})

#: Values re-exported from submodules, resolved the same way.
_REEXPORTS = {
	"span": ("profiler", "span"),
	"SCHEMA_VERSION": ("schema", "SCHEMA_VERSION"),
	"Status": ("schema", "Status"),
}

__all__ = [
	"aggregate", "attribution", "carbon", "events", "labels", "profiler",
	"schema", "trajectory", "span", "SCHEMA_VERSION", "Status", "__version__",
]


def __getattr__(name):
	if name in _SUBMODULES:
		module = importlib.import_module(f".{name}", __name__)
		globals()[name] = module
		return module
	if name in _REEXPORTS:
		module_name, attribute = _REEXPORTS[name]
		value = getattr(importlib.import_module(f".{module_name}", __name__),
		                attribute)
		globals()[name] = value
		return value
	raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__():
	return sorted(set(globals()) | _SUBMODULES | set(_REEXPORTS))
