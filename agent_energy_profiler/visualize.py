"""Standard figures over attributed events and trajectory summaries. Layer 6.

Draws a small set of paradigm-agnostic figures from a run's own artifacts:

    events_attributed.jsonl   attributed events (layer 3), required
    trajectory_summary.json   per-question references (layer 4), optional

Nothing here knows a host's stage names. Operation labels, operation types,
paradigm names and fallback reasons are read from the events, so a label or an
operation type a future host emits is drawn exactly like one that exists today.

ENERGY BASES are never interchanged, and every figure and CSV names its basis:

  attributed_events  per-event counter deltas (aggregate.py sums them). Used by
                     operation-energy, semantic-flow, fallback-split, and
                     question-level plots run with --basis attributed.
  trajectory         the whole-question counter reference from trajectory.py,
                     inter-event energy included. Used by question-level plots
                     run with --basis trajectory. Only an independently
                     measured window qualifies: a summary whose trajectory
                     total fell back to summed events is refused, not
                     relabelled.

A domain the hardware did not report is unavailable: the figure is refused,
never drawn from zeros. The residual aggregate.py emits as "<unattributed>" is
not an operation. It is written to the CSV and the terminal, drawn only on
request, and never given a share of attributed energy.

FALLBACK is read from the schema convention alone: an event whose meta carries
fallback=true or a fallback_reason marks its trajectory as a fallback
trajectory, and is that trajectory's fallback operation. Which operation label
it carries is data, not an assumption.

ENERGY IS ALWAYS ON THE Y-AXIS. Operation bars are vertical, the question
distribution is drawn as cumulative fraction of questions (x) against energy
(y), and every workload or outcome variable goes on x. --y-limits fixes the
energy axis so figures of different runs share a scale. semantic-flow has no
axes; there energy is ribbon thickness.

Every figure is written as PNG, PDF and a CSV of exactly the values drawn
(plus a _stats.csv or _per_question.csv where summary values are involved).
Titles are short; methodology goes to the terminal and the CSVs, never into
the image. matplotlib is imported only when a figure is drawn.
"""

import argparse
import csv
import hashlib
import math
import os
import re
import sys
from collections import Counter, defaultdict

from . import aggregate, trajectory

EVENTS_FILE = "events_attributed.jsonl"
TRAJECTORY_FILE = "trajectory_summary.json"

PLOT_OPERATION = "operation-energy"
PLOT_QUESTION = "question-energy"
PLOT_VS = "energy-vs"
PLOT_OUTCOME = "outcome-energy"
PLOT_SPLIT = "fallback-split"
PLOT_FLOW = "semantic-flow"
PLOTS = (PLOT_OPERATION, PLOT_FLOW, PLOT_QUESTION, PLOT_VS, PLOT_OUTCOME, PLOT_SPLIT)

BASIS_ATTRIBUTED = "attributed_events"
BASIS_TRAJECTORY = "trajectory"
BASIS_CHOICES = {"attributed": BASIS_ATTRIBUTED, "trajectory": BASIS_TRAJECTORY}
BASIS_PHRASE = {BASIS_ATTRIBUTED: "attributed events",
                BASIS_TRAJECTORY: "whole-question counter"}

UNIT_JOULES = "j"
UNIT_PERCENT = "percent"

ROW_OPERATION = "operation"
ROW_UNATTRIBUTED = "unattributed"

NO_FALLBACK = "no_fallback"
FALLBACK = "fallback"
OUTCOME_NAMES = {NO_FALLBACK: "No fallback", FALLBACK: "Fallback"}

COMPONENT_BEFORE = "before_fallback_operation"
COMPONENT_OPERATION = "fallback_operation"
COMPONENT_AFTER = "after_fallback_operation"
COMPONENT_NAMES = {COMPONENT_BEFORE: "Before fallback operation",
                   COMPONENT_OPERATION: "Fallback operation",
                   COMPONENT_AFTER: "After fallback operation"}

EXIT_OK, EXIT_USAGE, EXIT_NOT_APPLICABLE = 0, 2, 3

#: Correlations need at least this many valid (x, y) pairs; fewer is reported
#: as unavailable rather than as a coefficient computed from almost nothing.
MIN_CORRELATION_PAIRS = 3

#: Display names for energy domains: the single source for every title and
#: axis label. A domain missing here is named from its field, so a future
#: domain needs no code.
DOMAIN_NAMES = {
	"gpu_energy_j": "GPU",
	"cpu_package_energy_j": "CPU Package",
	"dram_energy_j": "DRAM",
	"measured_energy_j": "Measured",
	"cpu_core_energy_j": "CPU Core",
}

#: Title templates, filled from DOMAIN_NAMES and the x variable.
TITLES = {
	PLOT_OPERATION: "{domain} Energy by Operation",
	PLOT_QUESTION: "Question-Level {domain} Energy",
	PLOT_VS: "{domain} Energy vs {x}",
	PLOT_OUTCOME: "{domain} Energy by Outcome",
	PLOT_SPLIT: "Fallback {domain} Energy",
	PLOT_FLOW: "{domain} Energy by Semantic Stage",
}

#: semantic-flow hierarchy levels and layout (inches; one data unit = one inch).
ROW_FLOW = "flow"
LEVEL_ROOT, LEVEL_TYPE, LEVEL_LABEL = "root", "operation_type", "operation_label"
ROOT_NODE = "total"
FLOW_TOLERANCE_J = 1e-6
FLOW_TOTAL_IN = 4.2         # height of the total-energy bar
FLOW_TYPE_LABEL_IN = 0.3    # room above a type bar for its one-line label
FLOW_LEAF_SLOT_IN = 0.62    # minimum room for a 4-line leaf label
FLOW_GAP_IN = 0.08
FLOW_GROUP_GAP_IN = 0.2
FLOW_BAR_IN = 0.14
FLOW_X = (1.6, 4.6, 7.6)    # root, type and leaf bar positions
FLOW_WIDTH_IN = 10.2
OPERATION_FIG_IN = (8.4, 4.8)   # width, height per panel of operation-energy

#: Named workload variables. Any "<operation_type>_calls" is also accepted and
#: generated from the operation types present in the data.
VARIABLES = {
	"input_tokens": ("Input Tokens", "Input tokens per question"),
	"output_tokens": ("Output Tokens", "Output tokens per question"),
	"total_tokens": ("Total Tokens", "Total tokens per question"),
	"events": ("Events", "Events per question"),
	"max_traversal_depth": ("Max Traversal Depth", "Maximum traversal depth reached"),
	"max_iteration": ("Max Iteration", "Maximum control-loop iteration"),
	"event_duration_s": ("Event Duration", "Summed event duration per question (s)"),
	"trajectory_wall_s": ("Wall Time", "Question wall time (s)"),
}

#: Fixed categorical order of the validated palette. A run keeps the slot of
#: its --run position; overlapping-mark plots are validated for 3 series.
PALETTE = ("#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300",
           "#4a3aa7", "#e34948")
ALL_PAIRS_SAFE_SERIES = 3
NEUTRAL = "#898781"
INK, INK2, GRID, AXIS, SURFACE = "#0b0b0b", "#52514e", "#e1e0d9", "#c3c2b7", "#ffffff"
MARK_SHAPES = {"median": "o", "mean": "D", "p90": "^", "p95": "v"}
#: Marker shapes for categorical colouring: a second channel beside colour.
MARKER_CYCLE = ("o", "^", "s", "D", "v", "P", "X", "*")
UNANNOTATED = "unannotated"

STYLE = {
	"font.family": "DejaVu Sans", "font.size": 9, "axes.titlesize": 10,
	"figure.titlesize": 11, "figure.titleweight": "bold",
	"axes.edgecolor": AXIS, "axes.linewidth": 0.8, "axes.labelcolor": INK,
	"text.color": INK, "xtick.color": INK2, "ytick.color": INK2,
	"axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6,
	"grid.linestyle": "-", "axes.axisbelow": True,
	"axes.spines.top": False, "axes.spines.right": False,
	"figure.facecolor": SURFACE, "axes.facecolor": SURFACE,
	"savefig.facecolor": SURFACE, "pdf.fonttype": 42, "legend.frameon": False,
	"svg.hashsalt": "agent-energy-profiler",
}


class VisualizeError(Exception):
	"""A request the data cannot honestly answer. Exit code 2."""
	exit_code = EXIT_USAGE


class NotApplicable(VisualizeError):
	"""The plot does not apply to this data (e.g. no fallback metadata). Exit 3."""
	exit_code = EXIT_NOT_APPLICABLE


# --------------------------------------------------------------------------
# names
# --------------------------------------------------------------------------

def domain_name(domain):
	if domain in DOMAIN_NAMES:
		return DOMAIN_NAMES[domain]
	stem = domain[:-len("_energy_j")] if domain.endswith("_energy_j") else domain
	return stem.replace("_", " ").title()


def plot_title(plot, domain, prefix="", x=""):
	text = TITLES[plot].format(domain=domain_name(domain), x=x)
	return f"{prefix} {text}" if prefix else text


def energy_axis_label(domain, per="", basis=None, unit="J", lead=""):
	label = f"{lead}{domain_name(domain)} energy"
	if per:
		label += f" {per}"
	if basis:
		label += f", {BASIS_PHRASE[basis]}"
	return f"{label} ({unit})"


def calls_type(variable):
	return variable[:-len("_calls")] if variable.endswith("_calls") else None


def type_display(kind):
	"""Presentation of an operation-type value: short ones read as acronyms."""
	kind = str(kind)
	return kind.upper() if len(kind) <= 3 else kind.title()


def _type_slot(kind):
	"""Palette slot from a sha256 of the type string: stable across processes and runs."""
	return int(hashlib.sha256(str(kind).encode("utf-8")).hexdigest(), 16) % len(PALETTE)


def type_colours(kinds):
	"""{operation_type: colour}, identical in every run and figure.

	A type's colour depends only on its own string, never on which other types
	are present, so adding a type can never recolour an existing one. With a
	fixed palette two types can share a colour; that is reported by
	colour_collisions rather than resolved by moving one of them.
	"""
	return {str(k): PALETTE[_type_slot(k)] for k in kinds}


def colour_collisions(kinds):
	"""Groups of types in one figure that share a colour."""
	by_slot = defaultdict(set)
	for kind in kinds:
		by_slot[_type_slot(kind)].add(str(kind))
	return [sorted(group) for _, group in sorted(by_slot.items()) if len(group) > 1]


def energy_unit(total_j):
	"""(divisor, unit) keeping the total below 10,000 of the unit: J, kJ or MJ."""
	if abs(total_j) < 1e4:
		return 1.0, "J"
	if abs(total_j) < 1e7:
		return 1e3, "kJ"
	return 1e6, "MJ"


def variable_title(variable):
	if variable in VARIABLES:
		return VARIABLES[variable][0]
	kind = calls_type(variable)
	if kind:
		return f"{type_display(kind)} Calls"
	return variable.replace("_", " ").title()


def variable_axis_label(variable):
	if variable in VARIABLES:
		return VARIABLES[variable][1]
	kind = calls_type(variable)
	if kind:
		return f"{type_display(kind)} operations per question"
	return variable


# --------------------------------------------------------------------------
# runs
# --------------------------------------------------------------------------

class Run:
	"""One measured run, recognised by its artifacts, not its directory name."""

	def __init__(self, label, directory, events, trajectories):
		self.label = label
		self.directory = directory
		self.events = events
		self.trajectories = trajectories   # None when the artifact is absent

	@property
	def paradigm(self):
		return "|".join(distinct(self.events, "paradigm"))


def distinct(events, field):
	return sorted({str(e.get(field)) for e in events if e.get(field) is not None})


def question_ids(events):
	seen = {}
	for e in events:
		seen.setdefault(e.get("question_id"), None)
	return list(seen)


def parse_run_spec(spec):
	"""[NAME=]DIR. A directory whose own path contains '=' is taken whole."""
	if "=" in spec and not os.path.isdir(spec):
		name, directory = spec.split("=", 1)
		return name.strip() or None, directory
	return None, spec


def load_run(spec):
	name, directory = parse_run_spec(spec)
	events_path = os.path.join(directory, EVENTS_FILE)
	if not os.path.isfile(events_path):
		raise VisualizeError(
			f"{directory}: no {EVENTS_FILE}. A run is recognised by its "
			f"attributed-event artifact (produce it with measurement/attribute.py).")
	events = aggregate.load_events(events_path)
	if not events:
		raise VisualizeError(f"{events_path}: contains no events")
	trajectory_path = os.path.join(directory, TRAJECTORY_FILE)
	trajectories = (aggregate.load_trajectories(trajectory_path)
	                if os.path.isfile(trajectory_path) else None)
	paradigms = distinct(events, "paradigm")
	label = name or ("+".join(paradigms) if paradigms
	                 else os.path.basename(os.path.normpath(directory)))
	return Run(label, directory, events, trajectories)


def load_runs(specs):
	runs = [load_run(spec) for spec in specs]
	counts = Counter(run.label for run in runs)
	duplicated = sorted(label for label, n in counts.items() if n > 1)
	if duplicated:
		raise VisualizeError(
			f"several runs are labelled {', '.join(duplicated)}; name them "
			f"explicitly with --run NAME=DIR")
	if len(runs) > len(PALETTE):
		raise VisualizeError(f"at most {len(PALETTE)} runs per figure")
	return runs


def select_paradigms(runs, wanted):
	kept = []
	for run in runs:
		events = [e for e in run.events if str(e.get("paradigm")) in wanted]
		if not events:
			continue
		ids = {e.get("question_id") for e in events}
		trajectories = (None if run.trajectories is None else
		                [r for r in run.trajectories if r.get("question_id") in ids])
		kept.append(Run(run.label, run.directory, events, trajectories))
	if not kept:
		raise VisualizeError(f"no events with paradigm in {sorted(wanted)}")
	return kept


# --------------------------------------------------------------------------
# domains and energy
# --------------------------------------------------------------------------

def candidate_domains(events):
	found = list(trajectory.ENERGY_FIELDS) + ["cpu_core_energy_j"]
	for e in events:
		for key in e:
			if key.endswith("_energy_j") and key not in found:
				found.append(key)
	return found


def available_domains(events):
	return [d for d in candidate_domains(events)
	        if any(e.get(d) is not None for e in events)]


def require_event_domain(run, domain):
	if not any(e.get(domain) is not None for e in run.events):
		raise VisualizeError(
			f"run {run.label}: {domain} was not measured (null on every event). "
			f"Available: {', '.join(available_domains(run.events)) or 'none'}.")


def trajectory_index(run):
	if run.trajectories is None:
		raise NotApplicable(
			f"run {run.label}: no {TRAJECTORY_FILE}, so no whole-question counter "
			f"reference exists. Use --basis attributed for summed event energy.")
	return {r.get("question_id"): r for r in run.trajectories}


def question_energy(run, domain, basis):
	"""{question_id: energy_j} on one basis, plus the questions without a value.

	trajectory basis: trajectory_<domain> from rows whose window was measured
	independently; a row that fell back to summed events has no value here.
	attributed basis: aggregate.py's per-question sum; a partial sum (some
	events missing the domain) has no value.
	"""
	values, missing = {}, []
	if basis == BASIS_TRAJECTORY:
		index = trajectory_index(run)
		for q in question_ids(run.events):
			row = index.get(q)
			value = row.get(f"trajectory_{domain}") if row else None
			if value is None or not row.get("coverage_is_independent"):
				missing.append(q)
			else:
				values[q] = float(value)
		if not values:
			if not any(r.get(f"trajectory_{domain}") is not None for r in run.trajectories):
				raise VisualizeError(
					f"run {run.label}: {domain} has no whole-question reference "
					f"(not measured). Available: " + (", ".join(
						d for d in trajectory.ENERGY_FIELDS
						if any(r.get(f"trajectory_{d}") is not None for r in run.trajectories))
						or "none") + ".")
			raise VisualizeError(
				f"run {run.label}: no question has an independently measured "
				f"whole-question {domain} reference; refusing to substitute "
				f"summed event energy (use --basis attributed for that).")
		return values, missing

	require_event_domain(run, domain)
	for row in aggregate.aggregate(run.events, None, group_by=("question_id",),
	                               domains=(domain,)):
		if row["row_kind"] != aggregate.MEASURED:
			continue
		if row["energy_complete"]:
			values[row["question_id"]] = row["energy_j"]
		else:
			missing.append(row["question_id"])
	return values, missing


# --------------------------------------------------------------------------
# workload (counts only; no energy arithmetic)
# --------------------------------------------------------------------------

def workload(run):
	"""Per-question workload variables read from the events.

	<operation_type>_calls is generated for every operation type in the data.
	Token totals sum the events that report tokens. max_iteration and
	max_traversal_depth are each paradigm's own counters and are not
	comparable across paradigms.
	"""
	types = distinct(run.events, "operation_type")
	raw = {}
	for e in run.events:
		q = e.get("question_id")
		w = raw.get(q)
		if w is None:
			w = raw[q] = {"events": 0, "event_duration_s": 0.0, "in": [], "out": [],
			              "depth": [], "iteration": []}
			for kind in types:
				w[f"{kind}_calls"] = 0
		w["events"] += 1
		if e.get("operation_type") is not None:
			w[f"{e['operation_type']}_calls"] += 1
		w["event_duration_s"] += e.get("duration_s") or 0.0
		for key, field in (("in", "input_tokens"), ("out", "output_tokens"),
		                   ("depth", "traversal_depth"), ("iteration", "iteration")):
			if e.get(field) is not None:
				w[key].append(e[field])

	walls = ({r.get("question_id"): r.get("trajectory_wall_s") for r in run.trajectories}
	         if run.trajectories is not None else {})
	out = {}
	for q, w in raw.items():
		row = {k: v for k, v in w.items() if k not in ("in", "out", "depth", "iteration")}
		row["input_tokens"] = sum(w["in"]) if w["in"] else None
		row["output_tokens"] = sum(w["out"]) if w["out"] else None
		row["total_tokens"] = (row["input_tokens"] + row["output_tokens"]
		                       if w["in"] and w["out"] else None)
		row["max_traversal_depth"] = max(w["depth"]) if w["depth"] else None
		row["max_iteration"] = max(w["iteration"]) if w["iteration"] else None
		if run.trajectories is not None:
			row["trajectory_wall_s"] = walls.get(q)
		out[q] = row
	return out


def workload_variables(run):
	rows = workload(run).values()
	names = set()
	for row in rows:
		names.update(k for k, v in row.items() if v is not None)
	return sorted(names)


# --------------------------------------------------------------------------
# fallback (schema convention only)
# --------------------------------------------------------------------------

def is_fallback_event(event):
	meta = event.get("meta") or {}
	return meta.get("fallback") is True or meta.get("fallback_reason") is not None


def has_fallback_metadata(run):
	return any("fallback" in (e.get("meta") or {}) or
	           "fallback_reason" in (e.get("meta") or {}) for e in run.events)


def fallback_events(run):
	marked = defaultdict(list)
	for e in run.events:
		if is_fallback_event(e):
			marked[e.get("question_id")].append(e)
	return marked


def _reason(events):
	return "|".join(sorted({str((e.get("meta") or {}).get("fallback_reason"))
	                        for e in events
	                        if (e.get("meta") or {}).get("fallback_reason") is not None}))


def _sum_or_none(values):
	"""Energy of a set of events; None if any member lacks the domain."""
	if any(v is None for v in values):
		return None
	return float(sum(values))


# --------------------------------------------------------------------------
# statistics (dependency-free)
# --------------------------------------------------------------------------

def percentile(sorted_values, q):
	"""Linear interpolation between closest ranks."""
	if not sorted_values:
		return None
	position = (len(sorted_values) - 1) * q
	low, high = math.floor(position), math.ceil(position)
	return sorted_values[low] + (sorted_values[high] - sorted_values[low]) * (position - low)


def summary_stats(values):
	v = sorted(values)
	if not v:
		return {k: None for k in ("min", "p25", "median", "mean", "p75", "p90",
		                          "p95", "p99", "max")}
	return {"min": v[0], "p25": percentile(v, .25), "median": percentile(v, .5),
	        "mean": sum(v) / len(v), "p75": percentile(v, .75),
	        "p90": percentile(v, .9), "p95": percentile(v, .95),
	        "p99": percentile(v, .99), "max": v[-1]}


def _ranks(values):
	order = sorted(range(len(values)), key=values.__getitem__)
	ranks = [0.0] * len(values)
	i = 0
	while i < len(order):
		j = i
		while j + 1 < len(order) and values[order[j + 1]] == values[order[i]]:
			j += 1
		for k in range(i, j + 1):
			ranks[order[k]] = (i + j) / 2.0 + 1.0
		i = j + 1
	return ranks


def _pearson(xs, ys):
	n = len(xs)
	mx, my = sum(xs) / n, sum(ys) / n
	sxx = sum((x - mx) ** 2 for x in xs)
	syy = sum((y - my) ** 2 for y in ys)
	if sxx == 0 or syy == 0:
		return None
	return sum((x - mx) * (y - my) for x, y in zip(xs, ys)) / math.sqrt(sxx * syy)


def correlation(pairs):
	"""(n_pairs, pearson_r, spearman_rho, status). Unavailable, never NaN."""
	valid = [(float(x), float(y)) for x, y in pairs
	         if x is not None and y is not None
	         and math.isfinite(float(x)) and math.isfinite(float(y))]
	n = len(valid)
	if n < MIN_CORRELATION_PAIRS:
		return n, None, None, f"unavailable: {n} valid pairs (< {MIN_CORRELATION_PAIRS})"
	xs, ys = [p[0] for p in valid], [p[1] for p in valid]
	if len(set(xs)) < 2:
		return n, None, None, "unavailable: x is constant"
	if len(set(ys)) < 2:
		return n, None, None, "unavailable: y is constant"
	return n, _pearson(xs, ys), _pearson(_ranks(xs), _ranks(ys)), "ok"


# --------------------------------------------------------------------------
# data preparation: one function per plot, returning the rows it draws
# --------------------------------------------------------------------------

def prepare_operation_energy(runs, domain, operation_types=(), include_unattributed=False):
	"""Bars = aggregate.py MEASURED rows by (operation_type, operation_label).

	pct_attributed_energy is a share of the run's total attributed event
	energy over all operations, so all operation shares sum to 100%; a
	type filter shows a subset of those shares without renormalising. The
	<unattributed> residual gets no share.
	"""
	rows, notes = [], []
	for run in runs:
		require_event_domain(run, domain)
		result = aggregate.aggregate(run.events, run.trajectories or [],
		                             group_by=("operation_type", "operation_label"),
		                             domains=(domain,))
		measured = [r for r in result if r["row_kind"] == aggregate.MEASURED]
		total = sum(r["energy_j"] for r in measured if r["energy_j"] is not None)
		for r in measured:
			if operation_types and r["operation_type"] not in operation_types:
				continue
			energy = r["energy_j"]
			rows.append({
				"run": run.label, "paradigm": run.paradigm, "domain": domain,
				"energy_basis": BASIS_ATTRIBUTED, "row_kind": ROW_OPERATION,
				"operation_type": r["operation_type"], "operation_label": r["operation_label"],
				"event_count": r["n_events"], "energy_j": energy,
				"pct_attributed_energy": (energy / total * 100.0
				                          if energy is not None and total else None),
				"energy_complete": r["energy_complete"],
				"n_events_missing_energy": r["n_events_missing_energy"],
				"n_zero_energy_events": r["n_zero_energy_events"],
				"plotted": energy is not None,
			})
			if energy is None:
				notes.append(f"{run.label}: {r['operation_label']} has no {domain} "
				             f"on any event; not drawn")
			elif not r["energy_complete"]:
				notes.append(f"{run.label}: {r['operation_label']} is a partial sum "
				             f"({r['n_events_missing_energy']} events lack {domain})")
		residual = [r for r in result if r["row_kind"] == aggregate.RESIDUAL]
		if residual:
			r = residual[0]
			rows.append({
				"run": run.label, "paradigm": run.paradigm, "domain": domain,
				"energy_basis": "trajectory_minus_attributed",
				"row_kind": ROW_UNATTRIBUTED, "operation_type": aggregate.RESIDUAL_LABEL,
				"operation_label": aggregate.RESIDUAL_LABEL, "event_count": 0,
				"energy_j": r["energy_j"], "pct_attributed_energy": None,
				"energy_complete": True, "n_events_missing_energy": 0,
				"n_zero_energy_events": 0, "plotted": bool(include_unattributed),
			})
			notes.append(f"{run.label}: unattributed (trajectory - attributed) = "
			             f"{r['energy_j']:,.3f} J; attributed total = {total:,.3f} J")
		else:
			notes.append(f"{run.label}: unattributed energy unavailable (no valid "
			             f"independent trajectory reference)")
			if include_unattributed:
				raise VisualizeError(f"run {run.label}: --include-unattributed needs "
				                     f"a valid {TRAJECTORY_FILE}")
		if not any(row["run"] == run.label and row["plotted"] for row in rows):
			raise VisualizeError(f"run {run.label}: nothing to draw after filtering")
	return rows, notes


def _reconciles(a, b):
	return math.isclose(a, b, rel_tol=1e-9, abs_tol=FLOW_TOLERANCE_J)


def _flow_row(run, domain, kind, source_level, source, target_level, target, energy,
              total, events, plotted=True):
	return {"run": run.label, "paradigm": run.paradigm, "domain": domain,
	        "energy_basis": BASIS_ATTRIBUTED, "row_kind": kind,
	        "source_level": source_level, "source": source,
	        "target_level": target_level, "target": target, "energy_j": energy,
	        "pct_total_attributed": energy / total * 100.0 if kind == ROW_FLOW else None,
	        "event_count": events, "plotted": plotted}


def prepare_semantic_flow(run, domain):
	"""Rows of the hierarchy total -> operation_type -> operation_label.

	Parentage is the (operation_type, operation_label) pair recorded on each
	event; a label's text is never parsed for its type. A label recorded under
	several types becomes one leaf per observed pair. Energies come from three
	separate aggregate.py passes (whole run, per type, per type and label)
	that must reconcile before anything is drawn. pct_total_attributed is a
	share of the run's total attributed event energy at every level. Rows are
	returned in drawing order and are the only input the renderer uses.
	"""
	require_event_domain(run, domain)
	unplaced = sum(1 for e in run.events
	               if e.get("operation_type") is None or e.get("operation_label") is None)
	if unplaced:
		raise VisualizeError(f"run {run.label}: {unplaced} events lack operation_type or "
		                     f"operation_label and cannot be placed in the hierarchy")

	def measured(rows):
		return [r for r in rows if r["row_kind"] == aggregate.MEASURED]

	whole = aggregate.aggregate(run.events, run.trajectories or [], group_by=(),
	                            domains=(domain,))
	root = measured(whole)[0]
	if not root["energy_complete"]:
		raise VisualizeError(
			f"run {run.label}: {root['n_events_missing_energy']} of {root['n_events']} "
			f"events lack {domain}; a partial sum cannot be split into a hierarchy")
	total = root["energy_j"]
	if not total or total <= 0:
		raise VisualizeError(f"run {run.label}: total attributed {domain} is {total}; "
		                     f"flow widths need a positive total")
	types = measured(aggregate.aggregate(run.events, None, group_by=("operation_type",),
	                                     domains=(domain,)))
	leaves = measured(aggregate.aggregate(
		run.events, None, group_by=("operation_type", "operation_label"), domains=(domain,)))

	observed = {(e["operation_type"], e["operation_label"]) for e in run.events}
	if {(r["operation_type"], r["operation_label"]) for r in leaves} != observed:
		raise VisualizeError(f"run {run.label}: hierarchy pairs differ from the pairs "
		                     f"recorded on events")
	parents = defaultdict(set)
	for kind, label in observed:
		parents[label].add(kind)

	checks, type_sum = [], 0.0
	for t in types:
		children = [r for r in leaves if r["operation_type"] == t["operation_type"]]
		child_sum = sum(r["energy_j"] for r in children)
		if not _reconciles(child_sum, t["energy_j"]):
			raise VisualizeError(
				f"run {run.label}: labels under {t['operation_type']} sum to {child_sum!r} J "
				f"but the type totals {t['energy_j']!r} J")
		checks.append(f"{t['operation_type']}: {t['energy_j']:,.3f} J = sum of "
		              f"{len(children)} labels {child_sum:,.3f} J "
		              f"(diff {child_sum - t['energy_j']:+.2e})")
		type_sum += t["energy_j"]
	if not _reconciles(type_sum, total):
		raise VisualizeError(f"run {run.label}: operation types sum to {type_sum!r} J but "
		                     f"the total attributed energy is {total!r} J")
	checks.insert(0, f"total attributed: {total:,.3f} J over {root['n_events']:,} events; "
	                 f"sum of types {type_sum:,.3f} J (diff {type_sum - total:+.2e})")
	checks += [f"label {label} is recorded under {len(kinds)} operation types "
	           f"({', '.join(sorted(map(str, kinds)))}); each observed pair is its own leaf"
	           for label, kinds in sorted(parents.items(), key=lambda kv: str(kv[0]))
	           if len(kinds) > 1]

	ordered = sorted(types, key=lambda r: (-r["energy_j"], str(r["operation_type"])))
	rows = [_flow_row(run, domain, ROW_FLOW, LEVEL_ROOT, ROOT_NODE, LEVEL_TYPE,
	                  t["operation_type"], t["energy_j"], total, t["n_events"])
	        for t in ordered]
	for t in ordered:
		children = sorted((r for r in leaves if r["operation_type"] == t["operation_type"]),
		                  key=lambda r: (-r["energy_j"], str(r["operation_label"])))
		rows += [_flow_row(run, domain, ROW_FLOW, LEVEL_TYPE, t["operation_type"],
		                   LEVEL_LABEL, r["operation_label"], r["energy_j"], total,
		                   r["n_events"]) for r in children]

	# The flow is rooted at attributed energy, so the residual (trajectory minus
	# attributed) is outside it: reported, never drawn.
	residual = [r for r in whole if r["row_kind"] == aggregate.RESIDUAL]
	if residual:
		value = residual[0]["energy_j"]
		rows.append(_flow_row(run, domain, ROW_UNATTRIBUTED, LEVEL_ROOT,
		                      aggregate.RESIDUAL_LABEL, "", "", value, total, 0,
		                      plotted=False))
		checks.append(f"unattributed (trajectory - attributed): {value:,.3f} J; outside "
		              f"the attributed root, reported only")
	else:
		checks.append("unattributed energy unavailable (no valid independent "
		              "trajectory reference)")
	return rows, checks


def prepare_question_energy(runs, domain, basis):
	rows, stats, notes = [], [], []
	for run in runs:
		values, missing = question_energy(run, domain, basis)
		ordered = sorted(values.items(), key=lambda kv: (kv[1], str(kv[0])))
		n = len(ordered)
		for i, (q, v) in enumerate(ordered, 1):
			rows.append({"run": run.label, "paradigm": run.paradigm, "question_id": q,
			             "domain": domain, "energy_basis": basis, "energy_j": v,
			             "ecdf": i / n})
		stats.append({"run": run.label, "paradigm": run.paradigm, "domain": domain,
		              "energy_basis": basis, "n_questions": n,
		              "n_questions_without_value": len(missing),
		              **summary_stats(values.values())})
		if missing:
			notes.append(f"{run.label}: {len(missing)} questions have no {basis} "
			             f"{domain} value and are not drawn")
	return rows, stats, notes


def load_annotations(path, column, run_label=None):
	"""{question_id: category} from a CSV with a question_id column plus `column`.

	An optional `run` column scopes rows to one run label. The values are opaque
	strings: this module never interprets what a category means.
	"""
	found = {}
	with open(path, newline="") as f:
		reader = csv.DictReader(f)
		if reader.fieldnames is None or "question_id" not in reader.fieldnames:
			raise VisualizeError(f"{path}: needs a question_id column")
		if column not in (reader.fieldnames or ()):
			raise VisualizeError(f"{path}: no column {column!r}; has "
			                     f"{', '.join(reader.fieldnames)}")
		for row in reader:
			if run_label and (row.get("run") or run_label) != run_label:
				continue
			question = row["question_id"]
			if question in found:
				raise VisualizeError(f"{path}: duplicate question_id {question!r}")
			found[question] = (row.get(column) or "").strip()
	if not found:
		raise VisualizeError(f"{path}: no rows for run {run_label!r}")
	return found


def prepare_energy_vs(runs, domain, basis, x, annotations=None, color_by=""):
	"""Rows of (x, energy) per question; `annotations` adds an opaque category."""
	rows, stats, notes = [], [], []
	for run in runs:
		values, missing = question_energy(run, domain, basis)
		work = workload(run)
		available = workload_variables(run)
		if x not in available:
			raise VisualizeError(f"run {run.label}: no variable {x!r}. "
			                     f"Available: {', '.join(available)}")
		pairs, dropped, by_category = [], 0, defaultdict(list)
		for q in question_ids(run.events):
			xv, yv = work.get(q, {}).get(x), values.get(q)
			if xv is None or yv is None:
				dropped += 1
				continue
			category = (annotations or {}).get(q, "")
			pairs.append((xv, yv))
			by_category[category].append((xv, yv))
			rows.append({"run": run.label, "paradigm": run.paradigm, "question_id": q,
			             "x_variable": x, "x": xv, "domain": domain,
			             "energy_basis": basis, "energy_j": yv, "category": category})
		groups = [("(all)", pairs)] + ([(c or UNANNOTATED, p) for c, p in sorted(by_category.items())]
		                               if annotations else [])
		for category, group in groups:
			n, r, rho, status = correlation(group)
			stats.append({"run": run.label, "paradigm": run.paradigm, "x_variable": x,
			              "y_variable": domain, "energy_basis": basis, "scale": "raw values",
			              "category_column": color_by, "category": category, "n_pairs": n,
			              "pearson_r": r, "spearman_rho": rho, "status": status})
		if annotations:
			unannotated = sum(1 for r in rows if r["run"] == run.label and not r["category"])
			if unannotated:
				notes.append(f"{run.label}: {unannotated} questions have no annotation; "
				             f"drawn as {UNANNOTATED}, never assigned to a category")
		if dropped:
			notes.append(f"{run.label}: {dropped} questions lack {x} or energy; not drawn")
	return rows, stats, notes


def prepare_outcome_energy(runs, domain, basis):
	rows, stats, notes, used = [], [], [], []
	for run in runs:
		marked = fallback_events(run)
		if not marked:
			notes.append(f"{run.label}: no fallback-marked events; not applicable, skipped")
			continue
		used.append(run)
		values, missing = question_energy(run, domain, basis)
		groups = defaultdict(list)
		for q in question_ids(run.events):
			if q not in values:
				continue
			outcome = FALLBACK if q in marked else NO_FALLBACK
			groups[outcome].append(values[q])
			rows.append({"run": run.label, "paradigm": run.paradigm, "question_id": q,
			             "outcome": outcome, "fallback_reason": _reason(marked.get(q, [])),
			             "domain": domain, "energy_basis": basis, "energy_j": values[q]})
		for outcome in (NO_FALLBACK, FALLBACK):
			stats.append({"run": run.label, "paradigm": run.paradigm, "outcome": outcome,
			              "domain": domain, "energy_basis": basis,
			              "n_questions": len(groups[outcome]),
			              **summary_stats(groups[outcome])})
		if missing:
			notes.append(f"{run.label}: {len(missing)} questions without a value not drawn")
	if not used:
		raise NotApplicable("no run carries fallback metadata (meta.fallback / "
		                    "meta.fallback_reason); outcome-energy does not apply")
	return used, rows, stats, notes


def prepare_fallback_split(runs, domain):
	per_question, means, notes, used = [], [], [], []
	for run in runs:
		marked = fallback_events(run)
		if not marked:
			notes.append(f"{run.label}: no fallback-marked events; not applicable, skipped")
			continue
		require_event_domain(run, domain)
		by_question = defaultdict(list)
		for e in run.events:
			by_question[e.get("question_id")].append(e)
		components = {COMPONENT_BEFORE: [], COMPONENT_OPERATION: [], COMPONENT_AFTER: []}
		labels, reasons, skipped, any_after = set(), set(), 0, False
		for q, marks in marked.items():
			if len(marks) != 1:
				raise VisualizeError(
					f"run {run.label}, question {q}: {len(marks)} events carry a "
					f"fallback marker; fallback-split needs exactly one fallback "
					f"operation per trajectory and will not guess which")
			mark = marks[0]
			group = by_question[q]
			if mark.get("step_index") is None or any(e.get("step_index") is None for e in group):
				raise VisualizeError(f"run {run.label}, question {q}: events lack "
				                     f"step_index, so before/after cannot be ordered")
			step = mark["step_index"]
			before = _sum_or_none([e.get(domain) for e in group if e["step_index"] < step])
			operation = mark.get(domain)
			later = [e.get(domain) for e in group if e["step_index"] > step]
			# No event after the marker means the component does not exist for
			# this trajectory (None), which is different from events that sum to 0.
			after = _sum_or_none(later) if later else None
			reason = _reason([mark])
			if before is None or operation is None or (later and after is None):
				skipped += 1
				continue
			any_after = any_after or bool(later)
			labels.add(str(mark.get("operation_label")))
			reasons.add(reason)
			components[COMPONENT_BEFORE].append(before)
			components[COMPONENT_OPERATION].append(float(operation))
			components[COMPONENT_AFTER].append(after or 0.0)
			per_question.append({
				"run": run.label, "paradigm": run.paradigm, "question_id": q,
				"fallback_reason": reason, "fallback_operation_label": mark.get("operation_label"),
				"domain": domain, "energy_basis": BASIS_ATTRIBUTED,
				"before_fallback_operation_j": before, "fallback_operation_j": float(operation),
				"after_fallback_operation_j": after})
		n = len(components[COMPONENT_OPERATION])
		if not n:
			raise VisualizeError(f"run {run.label}: no fallback question has {domain} "
			                     f"on every event")
		if not any_after:
			del components[COMPONENT_AFTER]   # no trajectory continues past its fallback
		total = sum(sum(v) for v in components.values()) / n
		for component, values in components.items():
			mean = sum(values) / n
			means.append({
				"run": run.label, "paradigm": run.paradigm, "domain": domain,
				"energy_basis": BASIS_ATTRIBUTED, "component": component,
				"n_fallback_questions": n, "mean_energy_j": mean,
				"pct_of_mean_fallback_question_energy": mean / total * 100.0 if total else None,
				"fallback_operation_labels": "|".join(sorted(labels)),
				"fallback_reasons": "|".join(sorted(r for r in reasons if r))})
		used.append(run)
		if skipped:
			notes.append(f"{run.label}: {skipped} fallback questions with events lacking "
			             f"{domain} not drawn")
	if not used:
		raise NotApplicable("no run carries fallback metadata (meta.fallback / "
		                    "meta.fallback_reason); fallback-split does not apply")
	return used, means, per_question, notes


# --------------------------------------------------------------------------
# drawing
# --------------------------------------------------------------------------

def _pyplot():
	try:
		import matplotlib
		matplotlib.use("Agg")
		import matplotlib.pyplot as plt
	except ImportError as exc:
		raise VisualizeError("drawing figures needs matplotlib "
		                     "(pip install 'agent-energy-profiler[plot]')") from exc
	return plt


def _number_formatter():
	from matplotlib.ticker import FuncFormatter
	return FuncFormatter(lambda v, _: f"{v:,.0f}" if abs(v) >= 1 else f"{v:g}")


def _use_log(values, linear):
	values = [v for v in values if v is not None]
	return (not linear and values and min(values) > 0
	        and max(values) / min(values) >= 10)


def _fmt(value, unit):
	if unit == UNIT_PERCENT:
		return f"{value:.1f}%"
	return f"{value:,.0f}" if abs(value) >= 10 else f"{value:.2f}"


def _prefix(runs):
	return runs[0].label if len(runs) == 1 else ""


def _legend_opaque(legend):
	for handle in getattr(legend, "legend_handles", None) or getattr(legend, "legendHandles", []):
		try:
			handle.set_alpha(1.0)
		except AttributeError:
			pass


def parse_limits(text):
	"""'LOW,HIGH' -> (low, high) energy-axis limits, or None."""
	if not text:
		return None
	parts = [p.strip() for p in str(text).split(",")]
	try:
		low, high = (float(p) for p in parts)
	except ValueError:
		raise VisualizeError(f"--y-limits needs two numbers LOW,HIGH; got {text!r}")
	if not low < high:
		raise VisualizeError(f"--y-limits needs LOW < HIGH; got {text!r}")
	return low, high


def _figure_or_axes(plt, ax, figsize):
	"""(figure, axes, owned). With ax given, draw into the caller's axes -- so a
	composite figure can place several plots side by side -- and leave its layout
	to the caller; otherwise make a figure of our own."""
	if ax is not None:
		return ax.get_figure(), ax, False
	figure, axes = plt.subplots(figsize=figsize)
	return figure, axes, True


def _apply_ylim(ax, ylim):
	"""Caller-fixed energy-axis limits, e.g. to compare figures of different runs."""
	if not ylim:
		return
	if ax.get_yscale() == "log" and ylim[0] <= 0:
		raise VisualizeError("--y-limits must be positive on a log energy axis (or add --linear)")
	ax.set_ylim(*ylim)


def draw_operation_energy(plt, runs, rows, domain, unit, ylim=None):
	"""Vertical bars: operation label on x, energy on y."""
	from matplotlib.patches import Patch
	key = "pct_attributed_energy" if unit == UNIT_PERCENT else "energy_j"
	drawn = [r for r in rows if r["plotted"]]
	# One series, one colour: the operation type is already the label prefix, and
	# a per-type colour would change meaning between figures with different types.
	panels = []
	for run in runs:
		sub = [r for r in drawn if r["run"] == run.label]
		sub.sort(key=lambda r: (r["row_kind"] != ROW_OPERATION, -(r[key] or 0.0),
		                        r["operation_label"]))
		panels.append((run, sub))
	# A fixed size per panel keeps single-run figures of different runs comparable.
	fig, axes = plt.subplots(len(panels), 1, sharey=True, squeeze=False,
	                         figsize=(OPERATION_FIG_IN[0], OPERATION_FIG_IN[1] * len(panels)))
	values = [r[key] for _, sub in panels for r in sub]
	hi, lo = max([0.0] + values), min([0.0] + values)
	span = (hi - lo) or 1.0
	mixed = any(r["row_kind"] == ROW_UNATTRIBUTED for r in drawn)
	for ax, (run, sub) in zip(axes[:, 0], panels):
		vals = [r[key] for r in sub]
		colours = [PALETTE[0] if r["row_kind"] == ROW_OPERATION else NEUTRAL for r in sub]
		ax.bar(range(len(sub)), vals, color=colours, width=0.64,
		       edgecolor=SURFACE, linewidth=1.0)
		ax.set_xticks(range(len(sub)), [r["operation_label"] for r in sub],
		              rotation=35, ha="right", rotation_mode="anchor")
		ax.set_xlim(-0.6, max(len(sub), 1) - 0.4)
		ax.grid(axis="x", visible=False)
		ax.axhline(0, color=AXIS, linewidth=0.8)
		for i, v in enumerate(vals):
			ax.text(i, v + span * 0.012 * (1 if v >= 0 else -1), _fmt(v, unit),
			        ha="center", va="bottom" if v >= 0 else "top", fontsize=7.5, color=INK2)
		if len(panels) > 1:
			ax.set_title(run.label, loc="left")
		if unit == UNIT_PERCENT:
			ax.set_ylabel(f"Share of {domain_name(domain)} energy, "
			              f"{BASIS_PHRASE[BASIS_ATTRIBUTED]} (%)")
		else:
			# The residual bar is not attributed energy, so the axis names no basis
			# when it is drawn; the legend separates operations from the residual.
			ax.set_ylabel(energy_axis_label(domain, basis=None if mixed else BASIS_ATTRIBUTED))
			ax.yaxis.set_major_formatter(_number_formatter())
	axes[0, 0].set_ylim(lo - (span * 0.08 if lo < 0 else 0.0), hi + span * 0.12)
	_apply_ylim(axes[0, 0], ylim)
	if mixed:
		axes[0, 0].legend(handles=[Patch(color=PALETTE[0], label="operation"),
		                           Patch(color=NEUTRAL, label="not an operation")],
		                  loc="upper right", fontsize=8)
	fig.suptitle(plot_title(PLOT_OPERATION, domain, _prefix(runs)))
	fig.tight_layout()
	return fig


def _flow_layout(heights, min_slot, gap, groups=None, group_gap=0.0):
	"""Top-down (bar_top, bar_bottom) offsets from the column top, and column height.

	A bar is exactly its energy height; only the slot around it grows to fit the
	label, so flow widths stay proportional to energy.
	"""
	spans, y = [], 0.0
	for i, h in enumerate(heights):
		if i:
			y -= gap + (group_gap if groups and groups[i] != groups[i - 1] else 0.0)
		slot = max(h, min_slot)
		top = y - (slot - h) / 2.0
		spans.append((top, top - h))
		y -= slot
	return spans, -y


def draw_semantic_flow(plt, run, rows, domain):
	"""Left-to-right flow drawn only from the prepare_semantic_flow rows.

	Rooted at total attributed event energy; every ribbon ends at the edge of
	the node it feeds. Rows that are not flows (the residual) are not drawn.
	"""
	from matplotlib.patches import PathPatch, Rectangle
	from matplotlib.path import Path
	flows = [r for r in rows if r["row_kind"] == ROW_FLOW]
	to_type = [r for r in flows if r["source_level"] == LEVEL_ROOT]
	to_leaf = [r for r in flows if r["source_level"] == LEVEL_TYPE]
	total = sum(r["energy_j"] for r in to_type)
	divisor, unit = energy_unit(total)
	scale = FLOW_TOTAL_IN / total
	colour = type_colours(r["target"] for r in to_type)

	root_spans, root_h = _flow_layout([total * scale], 0.0, 0.0)
	# Each type slot reserves room above its bar for a one-line label.
	slots, type_h = _flow_layout([r["energy_j"] * scale + FLOW_TYPE_LABEL_IN for r in to_type],
	                             0.0, FLOW_GROUP_GAP_IN)
	type_spans = [(top - FLOW_TYPE_LABEL_IN, bottom) for top, bottom in slots]
	leaf_spans, leaf_h = _flow_layout([r["energy_j"] * scale for r in to_leaf],
	                                  FLOW_LEAF_SLOT_IN, FLOW_GAP_IN,
	                                  [str(r["source"]) for r in to_leaf], FLOW_GROUP_GAP_IN)
	body = max(root_h, type_h, leaf_h)
	bottom, top_margin = 0.25, 0.8
	height = body + bottom + top_margin
	middle = bottom + body / 2.0

	def place(spans, column_h):
		top = middle + column_h / 2.0
		return [(top + a, top + b) for a, b in spans]

	root_spans = place(root_spans, root_h)
	type_spans = place(type_spans, type_h)
	leaf_spans = place(leaf_spans, leaf_h)

	fig = plt.figure(figsize=(FLOW_WIDTH_IN, height))
	ax = fig.add_axes([0, 0, 1, 1])
	ax.set_xlim(0, FLOW_WIDTH_IN)
	ax.set_ylim(0, height)
	ax.axis("off")
	x_root, x_type, x_leaf = FLOW_X

	def bar(x, span, fill, gid):
		ax.add_patch(Rectangle((x, span[1]), FLOW_BAR_IN, span[0] - span[1],
		                       facecolor=fill, edgecolor="none", gid=gid))

	def band(x0, start, x1, end, fill, gid):
		xm = (x0 + x1) / 2.0
		verts = [(x0, start[0]), (xm, start[0]), (xm, end[0]), (x1, end[0]),
		         (x1, end[1]), (xm, end[1]), (xm, start[1]), (x0, start[1]),
		         (x0, start[0])]
		codes = ([Path.MOVETO] + [Path.CURVE4] * 3 + [Path.LINETO] + [Path.CURVE4] * 3
		         + [Path.CLOSEPOLY])
		ax.add_patch(PathPatch(Path(verts, codes), facecolor=fill, edgecolor="none",
		                       alpha=0.32, gid=gid))

	def amount(joules):
		return f"{joules / divisor:,.1f} {unit}"

	bar(x_root, root_spans[0], INK2, f"node:{LEVEL_ROOT}:{ROOT_NODE}")
	ax.text(x_root - 0.1, sum(root_spans[0]) / 2.0, f"Total attributed\n{amount(total)}",
	        ha="right", va="center", fontsize=9)

	cursor, type_tops = root_spans[0][0], {}
	for r, span in zip(to_type, type_spans):
		kind, h = str(r["target"]), r["energy_j"] * scale
		band(x_root + FLOW_BAR_IN, (cursor, cursor - h), x_type, span, colour[kind],
		     f"flow:{r['source']}->{kind}")
		cursor -= h
		bar(x_type, span, colour[kind], f"node:{LEVEL_TYPE}:{kind}")
		type_tops[kind] = span[0]
		ax.text(x_type + FLOW_BAR_IN / 2.0, span[0] + 0.05,
		        f"{type_display(kind)}   {r['pct_total_attributed']:.1f}%   "
		        f"{amount(r['energy_j'])}", ha="center", va="bottom", fontsize=9)
	for r, span in zip(to_leaf, leaf_spans):
		kind, label, h = str(r["source"]), str(r["target"]), r["energy_j"] * scale
		top = type_tops[kind]
		band(x_type + FLOW_BAR_IN, (top, top - h), x_leaf, span, colour[kind],
		     f"flow:{kind}->{label}")
		type_tops[kind] = top - h
		bar(x_leaf, span, colour[kind], f"node:{LEVEL_LABEL}:{kind}|{label}")
		ax.text(x_leaf + FLOW_BAR_IN + 0.1, sum(span) / 2.0,
		        f"{label}\n{r['pct_total_attributed']:.1f}%\n{amount(r['energy_j'])}\n"
		        f"{r['event_count']:,} events", ha="left", va="center", fontsize=8)
	ax.text(FLOW_WIDTH_IN / 2.0, height - 0.35, plot_title(PLOT_FLOW, domain, run.label),
	        ha="center", va="center", fontsize=11, fontweight="bold")
	return fig


def draw_question_energy(plt, runs, rows, stats, domain, basis, marks, linear, ylim=None):
	"""Cumulative fraction of questions on x, energy per question on y."""
	from matplotlib.lines import Line2D
	fig, ax = plt.subplots(figsize=(6.4, 4.2))
	for i, run in enumerate(runs):
		sub = [r for r in rows if r["run"] == run.label]
		energies = [r["energy_j"] for r in sub]
		ax.step([r["ecdf"] for r in sub], energies, where="pre", color=PALETTE[i],
		        linewidth=1.8, label=run.label)
		stat = next(s for s in stats if s["run"] == run.label)
		for mark in marks:
			value = stat[mark]
			share = sum(1 for e in energies if e <= value) / len(energies)
			ax.plot([share], [value], MARK_SHAPES[mark], color=PALETTE[i], markersize=6,
			        markeredgecolor=SURFACE, markeredgewidth=1.2)
	if _use_log([r["energy_j"] for r in rows], linear):
		ax.set_yscale("log")
	ax.yaxis.set_major_formatter(_number_formatter())
	ax.set_xlim(0, 1.0)
	ax.set_xlabel("Cumulative fraction of questions")
	ax.set_ylabel(energy_axis_label(domain, "per question", basis))
	_apply_ylim(ax, ylim)
	handles = []
	if len(runs) > 1:
		handles += [Line2D([], [], color=PALETTE[i], linewidth=1.8, label=run.label)
		            for i, run in enumerate(runs)]
	handles += [Line2D([], [], marker=MARK_SHAPES[m], linestyle="", color=INK2, label=m)
	            for m in marks]
	if handles:
		ax.legend(handles=handles, loc="upper left", fontsize=8)
	ax.set_title(plot_title(PLOT_QUESTION, domain, _prefix(runs)), fontweight="bold",
	             fontsize=11)
	fig.tight_layout()
	return fig


def category_styles(categories, colours=None, markers=None, labels=None):
	"""{category: {label, colour, marker}} for categorical colouring."""
	styles = {}
	for i, category in enumerate(categories):
		styles[category] = {
			"label": (labels or {}).get(category, category),
			"colour": (colours or {}).get(category, PALETTE[i % len(PALETTE)]),
			"marker": (markers or {}).get(category, MARKER_CYCLE[i % len(MARKER_CYCLE)])}
	styles.setdefault(UNANNOTATED, {"label": UNANNOTATED, "colour": NEUTRAL, "marker": "o"})
	return styles


def draw_energy_vs(plt, runs, rows, stats, domain, basis, x, linear, ylim=None, ax=None,
                   category_order=(), category_style=None):
	fig, ax, owned = _figure_or_axes(plt, ax, (6.4, 4.2))
	present = {r.get("category") or UNANNOTATED for r in rows if r.get("category") is not None}
	categories = [c for c in (category_order or sorted(present)) if c in present]
	if categories and UNANNOTATED in present and UNANNOTATED not in categories:
		categories.append(UNANNOTATED)
	if categories:
		styles = category_style or category_styles(categories)
		# Colour encodes the annotation, so the run is named in the title instead.
		for category in categories:
			sub = [r for r in rows if (r.get("category") or UNANNOTATED) == category]
			spec = styles.get(category, {})
			ax.scatter([r["x"] for r in sub], [r["energy_j"] for r in sub], s=11, alpha=0.35,
			           color=spec.get("colour", NEUTRAL), marker=spec.get("marker", "o"),
			           linewidths=0, label=f"{spec.get('label', category)} (n = {len(sub):,})")
	else:
		for i, run in enumerate(runs):
			sub = [r for r in rows if r["run"] == run.label]
			stat = next(s for s in stats if s["run"] == run.label)
			if stat["status"] == "ok":
				label = (f"{run.label} (r = {stat['pearson_r']:.2f}, "
				         f"ρ = {stat['spearman_rho']:.2f}, n = {stat['n_pairs']:,})")
			else:
				label = f"{run.label} (n = {stat['n_pairs']:,}; correlation unavailable)"
			ax.scatter([r["x"] for r in sub], [r["energy_j"] for r in sub], s=10, alpha=0.45,
			           color=PALETTE[i], linewidths=0, label=label)
	if _use_log([r["x"] for r in rows], linear):
		ax.set_xscale("log")
	if _use_log([r["energy_j"] for r in rows], linear):
		ax.set_yscale("log")
	ax.xaxis.set_major_formatter(_number_formatter())
	ax.yaxis.set_major_formatter(_number_formatter())
	ax.set_xlabel(variable_axis_label(x))
	ax.set_ylabel(energy_axis_label(domain, "per question", basis))
	_apply_ylim(ax, ylim)
	_legend_opaque(ax.legend(loc="upper left", fontsize=8))
	ax.set_title(plot_title(PLOT_VS, domain, _prefix(runs), variable_title(x)),
	             fontweight="bold", fontsize=11)
	if owned:
		fig.tight_layout()
	return fig


def draw_outcome_energy(plt, runs, rows, domain, basis, linear, ylim=None, ax=None):
	from matplotlib.lines import Line2D
	fig, ax, owned = _figure_or_axes(plt, ax, (max(4.2, 2.2 + 1.9 * len(runs)), 4.2))
	data, positions, colours, alphas, ticks = [], [], [], [], []
	position = 0.0
	for i, run in enumerate(runs):
		for outcome in (NO_FALLBACK, FALLBACK):
			values = [r["energy_j"] for r in rows
			          if r["run"] == run.label and r["outcome"] == outcome]
			if not values:
				continue
			position += 1.0
			data.append(values)
			positions.append(position)
			colours.append(PALETTE[i])
			alphas.append(0.35 if outcome == NO_FALLBACK else 0.8)
			name = f"{OUTCOME_NAMES[outcome]}\n(n = {len(values):,})"
			ticks.append(name if len(runs) == 1 else f"{run.label}\n{name}")
		position += 0.6
	box = ax.boxplot(data, positions=positions, widths=0.55, patch_artist=True,
	                 showmeans=True, showfliers=True,
	                 medianprops={"color": INK, "linewidth": 1.4},
	                 meanprops={"marker": "D", "markerfacecolor": INK,
	                            "markeredgecolor": SURFACE, "markersize": 5},
	                 flierprops={"marker": "o", "markersize": 2.5, "markerfacecolor": INK2,
	                             "markeredgecolor": "none", "alpha": 0.5},
	                 whiskerprops={"color": INK2}, capprops={"color": INK2})
	for patch, colour, alpha in zip(box["boxes"], colours, alphas):
		patch.set_facecolor(colour)
		patch.set_alpha(alpha)
		patch.set_edgecolor(colour)
	ax.set_xticks(positions, ticks)
	ax.grid(axis="x", visible=False)
	if _use_log([v for d in data for v in d], linear):
		ax.set_yscale("log")
	ax.yaxis.set_major_formatter(_number_formatter())
	ax.set_ylabel(energy_axis_label(domain, "per question", basis))
	_apply_ylim(ax, ylim)
	ax.legend(handles=[Line2D([], [], color=INK, linewidth=1.4, label="median"),
	                   Line2D([], [], marker="D", linestyle="", color=INK, label="mean")],
	          loc="upper left", fontsize=8)
	ax.set_title(plot_title(PLOT_OUTCOME, domain, _prefix(runs)), fontweight="bold",
	             fontsize=11)
	if owned:
		fig.tight_layout()
	return fig


def draw_fallback_split(plt, runs, means, domain, ylim=None, ax=None):
	"""Vertical stacked bars: run on x, mean energy per fallback question on y."""
	components = [c for c in (COMPONENT_BEFORE, COMPONENT_OPERATION, COMPONENT_AFTER)
	              if any(m["component"] == c for m in means)]
	fig, ax, owned = _figure_or_axes(plt, ax, (max(4.2, 2.4 + 1.2 * len(runs)), 4.2))
	labels, legend_done = [], set()
	for i, run in enumerate(runs):
		bottom = 0.0
		for j, component in enumerate(components):
			m = next((r for r in means if r["run"] == run.label
			          and r["component"] == component), None)
			if m is None:
				continue   # this run has no such component; nothing is drawn for it
			ax.bar(i, m["mean_energy_j"], bottom=bottom, color=PALETTE[j], width=0.55,
			       edgecolor=SURFACE, linewidth=1.0,
			       label=None if component in legend_done else COMPONENT_NAMES[component])
			legend_done.add(component)
			bottom += m["mean_energy_j"]
		n = next(r["n_fallback_questions"] for r in means if r["run"] == run.label)
		labels.append(f"{run.label}\n(n = {n:,})")
	ax.set_xticks(range(len(runs)), labels)
	ax.set_xlim(-0.6, len(runs) - 0.4)
	ax.grid(axis="x", visible=False)
	ax.yaxis.set_major_formatter(_number_formatter())
	ax.set_ylabel(energy_axis_label(domain, "per fallback question", BASIS_ATTRIBUTED,
	                                lead="Mean "))
	_apply_ylim(ax, ylim)
	ax.legend(loc="upper left", bbox_to_anchor=(1.02, 1.0), fontsize=8)
	ax.set_title(plot_title(PLOT_SPLIT, domain, _prefix(runs)), fontweight="bold",
	             fontsize=11)
	if owned:
		fig.tight_layout()
	return fig


# --------------------------------------------------------------------------
# output
# --------------------------------------------------------------------------

def _pairs_option(text):
	"""'a=1,b=2' -> {'a': '1', 'b': '2'}."""
	out = {}
	for item in (text or "").split(","):
		if not item.strip():
			continue
		if "=" not in item:
			raise VisualizeError(f"expected KEY=VALUE pairs, got {item!r}")
		key, value = item.split("=", 1)
		out[key.strip()] = value.strip()
	return out


def file_stem(plot, domain, runs, *parts):
	pieces = [plot] + [p for p in parts if p] + [domain, "-".join(r.label for r in runs)]
	return re.sub(r"[^A-Za-z0-9._+-]", "-", "_".join(pieces))


def write_rows(path, rows, fields):
	with open(path, "w", newline="") as f:
		writer = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore",
		                        lineterminator="\n")
		writer.writeheader()
		for row in rows:
			writer.writerow({k: ("" if row.get(k) is None else row.get(k)) for k in fields})
	return path


def save_figure(fig, out_dir, stem):
	png = os.path.join(out_dir, f"{stem}.png")
	pdf = os.path.join(out_dir, f"{stem}.pdf")
	fig.savefig(png, dpi=200, bbox_inches="tight")
	fig.savefig(pdf, bbox_inches="tight", metadata={"CreationDate": None, "ModDate": None})
	return [png, pdf]


def check_out_dir(out, runs):
	if not out:
		raise VisualizeError("--out is required for --plot")
	real_out = os.path.realpath(out)
	for run in runs:
		real_run = os.path.realpath(run.directory)
		if os.path.commonpath([real_out, real_run]) == real_run:
			raise VisualizeError(f"--out {out} is inside run directory {run.directory}; "
			                     f"figures are never written beside source artifacts")
	os.makedirs(out, exist_ok=True)


OPERATION_FIELDS = ("run", "paradigm", "domain", "energy_basis", "row_kind", "operation_type",
                    "operation_label", "event_count", "energy_j", "pct_attributed_energy",
                    "energy_complete", "n_events_missing_energy", "n_zero_energy_events",
                    "plotted")
FLOW_FIELDS = ("run", "paradigm", "domain", "energy_basis", "row_kind", "source_level", "source",
               "target_level", "target", "energy_j", "pct_total_attributed", "event_count",
               "plotted")
QUESTION_FIELDS = ("run", "paradigm", "question_id", "domain", "energy_basis", "energy_j", "ecdf")
STAT_FIELDS = ("n_questions", "n_questions_without_value", "min", "p25", "median", "mean",
               "p75", "p90", "p95", "p99", "max")
VS_FIELDS = ("run", "paradigm", "question_id", "x_variable", "x", "domain", "energy_basis",
             "energy_j", "category")
CORRELATION_FIELDS = ("run", "paradigm", "x_variable", "y_variable", "energy_basis", "scale",
                      "category_column", "category", "n_pairs", "pearson_r", "spearman_rho",
                      "status")
OUTCOME_FIELDS = ("run", "paradigm", "question_id", "outcome", "fallback_reason", "domain",
                  "energy_basis", "energy_j")
SPLIT_FIELDS = ("run", "paradigm", "domain", "energy_basis", "component", "n_fallback_questions",
                "mean_energy_j", "pct_of_mean_fallback_question_energy",
                "fallback_operation_labels", "fallback_reasons")
SPLIT_QUESTION_FIELDS = ("run", "paradigm", "question_id", "fallback_reason",
                         "fallback_operation_label", "domain", "energy_basis",
                         "before_fallback_operation_j", "fallback_operation_j",
                         "after_fallback_operation_j")


def render(args, runs):
	"""Draw one plot; returns (paths written, notes)."""
	plt = _pyplot()
	domain, out = args.domain, args.out
	basis = BASIS_CHOICES[args.basis] if args.basis else BASIS_TRAJECTORY
	ylim = parse_limits(args.y_limits)
	written, notes = [], []
	with plt.rc_context(STYLE):
		if args.plot == PLOT_OPERATION:
			types = tuple(t for t in args.operation_type.split(",") if t)
			rows, notes = prepare_operation_energy(runs, domain, types, args.include_unattributed)
			stem = file_stem(PLOT_OPERATION, domain, runs, args.unit,
			                 "with-unattributed" if args.include_unattributed else "",
			                 "types-" + "+".join(types) if types else "")
			fig = draw_operation_energy(plt, runs, rows, domain, args.unit, ylim)
			written += save_figure(fig, out, stem)
			written.append(write_rows(os.path.join(out, f"{stem}.csv"), rows, OPERATION_FIELDS))
			notes.insert(0, "basis: attributed event energy; pct_attributed_energy = share of "
			                "the run's total attributed event energy (all operations)")
		elif args.plot == PLOT_FLOW:
			rows, notes = prepare_semantic_flow(runs[0], domain)
			kinds = [r["target"] for r in rows
			         if r["row_kind"] == ROW_FLOW and r["source_level"] == LEVEL_ROOT]
			notes += [f"operation types {', '.join(group)} share a colour (stable per-type "
			          f"colours; distinguish them by position and label)"
			          for group in colour_collisions(kinds)]
			stem = file_stem(PLOT_FLOW, domain, runs)
			fig = draw_semantic_flow(plt, runs[0], rows, domain)
			written += save_figure(fig, out, stem)
			written.append(write_rows(os.path.join(out, f"{stem}.csv"), rows, FLOW_FIELDS))
			notes.insert(0, "basis: attributed event energy; parent of each label = the "
			                "operation_type recorded on its events; pct_total_attributed = "
			                "share of the run's total attributed event energy")
		elif args.plot == PLOT_QUESTION:
			rows, stats, notes = prepare_question_energy(runs, domain, basis)
			stem = file_stem(PLOT_QUESTION, domain, runs, basis)
			marks = [m for m in args.mark.split(",") if m]
			fig = draw_question_energy(plt, runs, rows, stats, domain, basis, marks, args.linear,
			                           ylim)
			written += save_figure(fig, out, stem)
			written.append(write_rows(os.path.join(out, f"{stem}.csv"), rows, QUESTION_FIELDS))
			written.append(write_rows(os.path.join(out, f"{stem}_stats.csv"), stats,
			                          ("run", "paradigm", "domain", "energy_basis") + STAT_FIELDS))
			notes.insert(0, f"basis: {basis} ({BASIS_PHRASE[basis]})")
		elif args.plot == PLOT_VS:
			annotations = (load_annotations(args.annotations, args.color_by, runs[0].label)
			               if args.color_by else None)
			rows, stats, notes = prepare_energy_vs(runs, domain, basis, args.x, annotations,
			                                       args.color_by)
			order = [c for c in args.category_order.split(",") if c]
			styles = (category_styles(order or sorted({r["category"] for r in rows if r["category"]}),
			                          _pairs_option(args.category_colors),
			                          _pairs_option(args.category_markers),
			                          _pairs_option(args.category_labels)) if annotations else None)
			stem = file_stem(PLOT_VS, domain, runs, args.x, basis,
			                 f"by-{args.color_by}" if args.color_by else "")
			fig = draw_energy_vs(plt, runs, rows, stats, domain, basis, args.x, args.linear, ylim,
			                     category_order=order, category_style=styles)
			written += save_figure(fig, out, stem)
			written.append(write_rows(os.path.join(out, f"{stem}.csv"), rows, VS_FIELDS))
			written.append(write_rows(os.path.join(out, f"{stem}_stats.csv"), stats,
			                          CORRELATION_FIELDS))
			notes.insert(0, f"basis: {basis} ({BASIS_PHRASE[basis]}); correlations on raw "
			                f"values; association only")
		elif args.plot == PLOT_OUTCOME:
			used, rows, stats, notes = prepare_outcome_energy(runs, domain, basis)
			stem = file_stem(PLOT_OUTCOME, domain, used, basis)
			fig = draw_outcome_energy(plt, used, rows, domain, basis, args.linear, ylim)
			written += save_figure(fig, out, stem)
			written.append(write_rows(os.path.join(out, f"{stem}.csv"), rows, OUTCOME_FIELDS))
			written.append(write_rows(os.path.join(out, f"{stem}_stats.csv"), stats,
			                          ("run", "paradigm", "outcome", "domain", "energy_basis",
			                           "n_questions") + STAT_FIELDS[2:]))
			notes.insert(0, f"basis: {basis} ({BASIS_PHRASE[basis]}); fallback = trajectory "
			                f"with an event whose meta has fallback=true or a fallback_reason")
		elif args.plot == PLOT_SPLIT:
			used, means, per_question, notes = prepare_fallback_split(runs, domain)
			stem = file_stem(PLOT_SPLIT, domain, used)
			fig = draw_fallback_split(plt, used, means, domain, ylim)
			written += save_figure(fig, out, stem)
			written.append(write_rows(os.path.join(out, f"{stem}.csv"), means, SPLIT_FIELDS))
			written.append(write_rows(os.path.join(out, f"{stem}_per_question.csv"),
			                          per_question, SPLIT_QUESTION_FIELDS))
			notes.insert(0, "basis: attributed event energy, split at the single "
			                "fallback-marked event of each fallback trajectory (bars = means)")
		plt.close(fig)
	return written, notes


def list_runs(runs):
	for run in runs:
		events = run.events
		print(f"run: {run.label}  ({run.directory})")
		print(f"  paradigm: {run.paradigm or '(not recorded)'}   dataset: "
		      f"{'|'.join(distinct(events, 'dataset')) or '(not recorded)'}   model: "
		      f"{'|'.join(distinct(events, 'model_name')) or '(not recorded)'}")
		print(f"  questions: {len(question_ids(events)):,}   events: {len(events):,}")
		if run.trajectories is None:
			print(f"  trajectory artifact: no ({TRAJECTORY_FILE} absent; only attributed-"
			      f"basis plots apply)")
		else:
			independent = sum(1 for r in run.trajectories if r.get("coverage_is_independent"))
			print(f"  trajectory artifact: yes ({independent:,}/{len(run.trajectories):,} "
			      f"questions with an independent window)")
		present = available_domains(events)
		absent = [d for d in candidate_domains(events) if d not in present]
		print(f"  energy domains: {', '.join(present) or 'none'}"
		      + (f"   (unavailable: {', '.join(absent)})" if absent else ""))
		if run.trajectories is not None:
			reference = [d for d in trajectory.ENERGY_FIELDS
			             if any(r.get(f"trajectory_{d}") is not None
			                    and r.get("coverage_is_independent") for r in run.trajectories)]
			print(f"  trajectory-reference domains: {', '.join(reference) or 'none'}")
		types = Counter(e.get("operation_type") for e in events)
		labels = Counter(e.get("operation_label") for e in events)
		print("  operation types: " + ", ".join(f"{k} ({v:,})" for k, v in sorted(
			types.items(), key=lambda kv: -kv[1])))
		print("  operation labels: " + ", ".join(f"{k} ({v:,})" for k, v in sorted(
			labels.items(), key=lambda kv: -kv[1])))
		print(f"  energy-vs variables: {', '.join(workload_variables(run))}")
		marked = fallback_events(run)
		if has_fallback_metadata(run):
			reasons = Counter(_reason(v) for v in marked.values())
			print(f"  fallback metadata: yes ({len(marked):,} fallback trajectories; "
			      + ", ".join(f"{k or '(no reason)'} {v:,}" for k, v in sorted(reasons.items()))
			      + ")")
		else:
			print("  fallback metadata: no (outcome-energy and fallback-split do not apply)")
		print()


def build_parser():
	ap = argparse.ArgumentParser(
		description=__doc__.splitlines()[0],
		epilog="Exit codes: 0 ok, 2 invalid request or unavailable energy domain, "
		       "3 plot not applicable to this data.")
	ap.add_argument("--run", action="append", required=True, metavar="[NAME=]DIR",
	                help="run directory holding events_attributed.jsonl (and optionally "
	                     "trajectory_summary.json); repeat for several runs. NAME defaults "
	                     "to the paradigm recorded on the events.")
	ap.add_argument("--plot", choices=PLOTS)
	ap.add_argument("--list", action="store_true",
	                help="describe the runs (domains, labels, variables, fallback) and exit")
	ap.add_argument("--domain", default="gpu_energy_j",
	                help="energy domain to plot (default gpu_energy_j); refused if unmeasured")
	ap.add_argument("--out", default="", help="output directory (outside every run directory)")
	ap.add_argument("--paradigm", default="",
	                help="comma-separated paradigm values to keep (as recorded on events)")
	ap.add_argument("--unit", choices=(UNIT_JOULES, UNIT_PERCENT), default=UNIT_JOULES,
	                help="operation-energy only. j: attributed Joules. percent: share of the "
	                     "run's total attributed event energy over all operations, so all "
	                     "operation shares sum to 100%%; the <unattributed> residual never "
	                     "gets a share.")
	ap.add_argument("--operation-type", default="",
	                help="operation-energy only: comma-separated operation types to draw")
	ap.add_argument("--include-unattributed", action="store_true",
	                help="operation-energy only (Joules): also draw the trajectory-minus-"
	                     "attributed residual as a separate, non-operation bar")
	ap.add_argument("--basis", choices=sorted(BASIS_CHOICES),
	                help="question-level plots only: trajectory (whole-question counter, "
	                     "default) or attributed (summed event energy)")
	ap.add_argument("--x", default="",
	                help="energy-vs only: input_tokens, output_tokens, total_tokens, events, "
	                     "<operation_type>_calls, max_traversal_depth, max_iteration, "
	                     "event_duration_s, trajectory_wall_s (see --list)")
	ap.add_argument("--mark", default="",
	                help="question-energy only: comma-separated median,mean,p90,p95 markers")
	ap.add_argument("--linear", action="store_true", help="never use log axes")
	ap.add_argument("--y-limits", default="", metavar="LOW,HIGH",
	                help="fix the energy (y) axis, in the plotted unit, so figures of "
	                     "different runs share a scale; not for semantic-flow")
	ap.add_argument("--annotations", default="", metavar="FILE",
	                help="energy-vs only: CSV with a question_id column, optionally a run "
	                     "column, and one or more categorical columns")
	ap.add_argument("--color-by", default="", metavar="COLUMN",
	                help="energy-vs only: colour points by this annotation column; the "
	                     "values are opaque categories. One run per figure.")
	ap.add_argument("--category-order", default="", help="comma-separated category values")
	ap.add_argument("--category-labels", default="", metavar="VALUE=LABEL,...")
	ap.add_argument("--category-colors", default="", metavar="VALUE=#RRGGBB,...")
	ap.add_argument("--category-markers", default="", metavar="VALUE=MARKER,...")
	return ap


def _validate(args, runs):
	plot = args.plot
	if plot is None:
		raise VisualizeError("choose --plot (or --list)")
	if parse_limits(args.y_limits) and plot == PLOT_FLOW:
		raise VisualizeError("semantic-flow has no energy axis (energy is ribbon thickness); "
		                     "--y-limits does not apply")
	if plot != PLOT_OPERATION and (args.unit != UNIT_JOULES or args.operation_type):
		raise VisualizeError("--unit and --operation-type apply to operation-energy only")
	if args.include_unattributed and plot != PLOT_OPERATION:
		raise VisualizeError("--include-unattributed applies to operation-energy only; "
		                     "semantic-flow is rooted at attributed energy and reports "
		                     "the residual in its CSV")
	if plot == PLOT_FLOW and len(runs) != 1:
		raise VisualizeError("semantic-flow draws one run per figure; run it once per --run")
	if plot in (PLOT_OPERATION, PLOT_FLOW, PLOT_SPLIT) and args.basis == "trajectory":
		raise VisualizeError(f"{plot} uses attributed event energy only; --basis "
		                     f"trajectory does not apply")
	if plot == PLOT_OPERATION and args.unit == UNIT_PERCENT and args.include_unattributed:
		raise VisualizeError("the <unattributed> residual has no share of attributed energy; "
		                     "use --unit j with --include-unattributed")
	if plot == PLOT_OPERATION and args.operation_type:
		wanted = {t for t in args.operation_type.split(",") if t}
		present = {e.get("operation_type") for run in runs for e in run.events}
		unknown = sorted(wanted - present)
		if unknown:
			raise VisualizeError(f"unknown operation type(s) {unknown}; present: "
			                     f"{sorted(t for t in present if t)}")
	if (plot == PLOT_VS) != bool(args.x):
		raise VisualizeError("--x is required for energy-vs and applies to it only")
	categorical = (args.annotations, args.color_by, args.category_order, args.category_labels,
	               args.category_colors, args.category_markers)
	if any(categorical) and plot != PLOT_VS:
		raise VisualizeError("--annotations and the --category-* options apply to energy-vs only")
	if bool(args.annotations) != bool(args.color_by):
		raise VisualizeError("--annotations and --color-by are used together")
	if args.color_by and len(runs) != 1:
		raise VisualizeError("--color-by uses colour for the annotation, so it draws one run "
		                     "per figure")
	marks = [m for m in args.mark.split(",") if m]
	if marks and plot != PLOT_QUESTION:
		raise VisualizeError("--mark applies to question-energy only")
	unknown = sorted(set(marks) - set(MARK_SHAPES))
	if unknown:
		raise VisualizeError(f"unknown --mark {unknown}; choose from {sorted(MARK_SHAPES)}")
	if plot in (PLOT_QUESTION, PLOT_VS) and len(runs) > ALL_PAIRS_SAFE_SERIES:
		print(f"note: {len(runs)} overlapping series; colour separation is validated for "
		      f"{ALL_PAIRS_SAFE_SERIES}", file=sys.stderr)


def main(argv=None):
	args = build_parser().parse_args(argv)
	try:
		runs = load_runs(args.run)
		if args.paradigm:
			runs = select_paradigms(runs, {p for p in args.paradigm.split(",") if p})
		if args.list:
			list_runs(runs)
			return EXIT_OK
		_validate(args, runs)
		check_out_dir(args.out, runs)
		written, notes = render(args, runs)
	except VisualizeError as exc:
		print(f"visualize: {exc}", file=sys.stderr)
		return exc.exit_code
	print(f"{args.plot}: domain={args.domain}  runs={', '.join(r.label for r in runs)}")
	for note in notes:
		print(f"  {note}")
	for path in written:
		print(f"  -> {path}")
	return EXIT_OK


if __name__ == "__main__":
	sys.exit(main())
