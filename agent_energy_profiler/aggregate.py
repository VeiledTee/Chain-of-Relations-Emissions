"""Long-format aggregation over attributed events. Layer 5, derived only.

Turns the canonical per-event artifact into a tidy table that plotting code can
consume without re-deriving accounting rules:

    attributed events + trajectory summaries  ->  one row per (group, domain)

Grouping is arbitrary. Any field present on an attributed event is a valid
dimension, including nested metadata addressed as "meta.<key>", so this module
hard-codes no host's stage names. `operation_type`, `operation_label`,
`question_id`, `iteration`, `traversal_depth`, `status`, `dataset` and
`paradigm` are simply the dimensions schema v1 guarantees.

What this module refuses to do, because a figure would inherit the lie:

  * fabricate zero for a domain the hardware did not measure;
  * report a share against a trajectory reference that is missing or invalid;
  * clamp a negative residual to make coverage look tidy;
  * correct counter quantization, or let a run of exact-zero deltas read as
    proven zero energy (see zero_energy_events);
  * subtract an idle baseline. Gross measured energy stays canonical; an
    idle-adjusted view is a separate sensitivity analysis, not an aggregate.

RECONCILIATION. Named groups plus the synthetic residual row sum to the
trajectory reference:

    sum(energy_j over row_kind="measured")
      + energy_j of row_kind="residual"
      == trajectory_energy_j

This holds per (scope, domain) whenever `reconciles` is true, and is exact
rather than approximate. It is false, and every share is null, when the domain
was not measured for every event in scope or the trajectory reference is
unavailable -- the same strictness trajectory.py applies, for the same reason:
a partial sum misstates the boundary.

NULL SEMANTICS. `energy_j` is null when no event in the group carried the
domain at all. When only *some* did, the partial sum is reported alongside
`n_events_missing_energy` and `energy_complete = False`; the group is then
excluded from reconciliation. Missing is never zero, and a partial sum is
never silently presented as a whole.
"""

import argparse
import csv
import json
from collections import defaultdict

from . import trajectory

#: Additive domains of the measurement boundary, plus the total. Mirrors
#: trajectory.ENERGY_FIELDS deliberately: aggregation must not invent a
#: different boundary from the one attribution and reconciliation use.
DEFAULT_DOMAINS = trajectory.ENERGY_FIELDS

#: What each domain is for. A "diagnostic" domain is never reconciled and never
#: given a share: cpu_core is contained within cpu_package, so treating it as a
#: slice of a total would double-count.
DOMAIN_ROLES = {
	"gpu_energy_j": "boundary",
	"cpu_package_energy_j": "boundary",
	"dram_energy_j": "boundary",
	"measured_energy_j": "total",
	"cpu_core_energy_j": "diagnostic",
}

#: Dimensions that identify a trajectory rather than a step within one. A
#: residual is only defined per trajectory, so these decide the scope a
#: residual row is emitted at.
TRAJECTORY_DIMENSIONS = ("run_id", "dataset", "paradigm", "question_id")

#: Dimensions schema v1 guarantees exist on an attributed event.
STANDARD_DIMENSIONS = (
	"operation_type", "operation_label", "question_id", "iteration",
	"traversal_depth", "status", "dataset", "paradigm", "run_id",
	"model_name", "hardware_id",
)

#: Marker used in dimension columns of a synthetic residual row. Deliberately
#: not a taxonomy-shaped string: it must never be mistaken for an operation
#: label a host emitted.
RESIDUAL_LABEL = "<unattributed>"

MEASURED = "measured"
RESIDUAL = "residual"


# --------------------------------------------------------------------------
# dimension access
# --------------------------------------------------------------------------

def dimension_value(event, dimension):
	"""Value of one grouping dimension, supporting "meta.<key>" addressing."""
	if dimension.startswith("meta."):
		meta = event.get("meta") or {}
		return meta.get(dimension[5:])
	return event.get(dimension)


def available_dimensions(events):
	"""Every dimension these events can be grouped by, meta included."""
	top, meta = set(), set()
	for event in events:
		for key, value in event.items():
			if key == "meta":
				for meta_key in (value or {}):
					meta.add(f"meta.{meta_key}")
			elif not isinstance(value, (dict, list)):
				top.add(key)
	return sorted(top) + sorted(meta)


# --------------------------------------------------------------------------
# trajectory references
# --------------------------------------------------------------------------

def _scope_dimensions(group_by):
	"""The trajectory-identifying subset of the requested grouping."""
	return tuple(d for d in group_by if d in TRAJECTORY_DIMENSIONS)


def _trajectory_index(trajectories):
	return {row.get("question_id"): row for row in trajectories}


def scope_reference(events, trajectories, scope_dimensions, domain):
	"""Trajectory-level reference for one scope and domain.

	Returns (reference_energy_j, attributed_energy_j, valid). `valid` is true
	only when every trajectory in scope measured this domain for every event
	AND its coverage is independently checkable (an independently measured
	window, no overlapping events). Anything less makes shares and residuals
	null rather than approximate.
	"""
	index = _trajectory_index(trajectories)
	role = DOMAIN_ROLES.get(domain, "diagnostic")
	if role == "diagnostic":
		return None, None, False

	question_ids = {e.get("question_id") for e in events}
	rows = [index[q] for q in question_ids if q in index]
	if not rows or len(rows) != len(question_ids):
		return None, None, False

	references, attributed = [], []
	for row in rows:
		if not row.get("coverage_valid"):
			return None, None, False
		reference = row.get(f"trajectory_{domain}")
		summed = row.get(f"sum_attributed_{domain}")
		if reference is None or summed is None:
			return None, None, False
		references.append(reference)
		attributed.append(summed)

	return sum(references), sum(attributed), True


# --------------------------------------------------------------------------
# aggregation
# --------------------------------------------------------------------------

def _blank(domain):
	return {
		"n_events": 0,
		"duration_s": 0.0,
		"energy_sum": 0.0,
		"n_with_energy": 0,
		"n_missing_energy": 0,
		"n_zero_energy": 0,
		"n_measurement_complete": 0,
		"n_failed_events": 0,
		"_domain": domain,
	}


def aggregate(events, trajectories=None, group_by=("operation_label",),
              domains=DEFAULT_DOMAINS):
	"""Long-format aggregate rows, one per (group, energy_domain).

	events        canonical attributed events (attribution.attribute_events)
	trajectories  trajectory summary rows (trajectory.accumulate). Without
	              them there is no reference, so no shares and no residual
	              rows are emitted -- never a fabricated denominator.
	group_by      any dimensions present on the events, "meta.<key>" included
	domains       energy domains to aggregate; diagnostic domains are reported
	              but never reconciled or given a share
	"""
	group_by = tuple(group_by)
	trajectories = list(trajectories or [])
	scope_dimensions = _scope_dimensions(group_by)

	buckets = defaultdict(lambda: None)
	scoped_events = defaultdict(list)

	for event in events:
		key = tuple(dimension_value(event, d) for d in group_by)
		scope = tuple(dimension_value(event, d) for d in scope_dimensions)
		scoped_events[scope].append(event)
		for domain in domains:
			bucket_key = (key, domain)
			bucket = buckets[bucket_key]
			if bucket is None:
				bucket = buckets[bucket_key] = _blank(domain)
			bucket["n_events"] += 1
			bucket["duration_s"] += event.get("duration_s") or 0.0
			if event.get("measurement_complete"):
				bucket["n_measurement_complete"] += 1
			if event.get("status") not in (None, "ok"):
				bucket["n_failed_events"] += 1

			value = event.get(domain)
			if value is None:
				bucket["n_missing_energy"] += 1
			else:
				bucket["n_with_energy"] += 1
				bucket["energy_sum"] += value
				if value == 0.0:
					bucket["n_zero_energy"] += 1

	# Reference per (scope, domain), computed once.
	references = {}
	for scope, scope_group in scoped_events.items():
		for domain in domains:
			references[(scope, domain)] = scope_reference(
				scope_group, trajectories, scope_dimensions, domain)

	rows = []
	for (key, domain), bucket in buckets.items():
		scope = tuple(
			key[group_by.index(d)] for d in scope_dimensions) if scope_dimensions else ()
		reference, attributed, valid = references.get((scope, domain), (None, None, False))

		energy_complete = bucket["n_missing_energy"] == 0 and bucket["n_with_energy"] > 0
		energy_j = bucket["energy_sum"] if bucket["n_with_energy"] else None

		# A share is only defensible against a valid reference, over a group
		# whose own energy is complete, with a non-zero denominator.
		share = None
		if valid and energy_complete and energy_j is not None and reference:
			share = energy_j / reference

		rows.append(_row(
			row_kind=MEASURED, group_by=group_by, key=key, domain=domain,
			n_events=bucket["n_events"], duration_s=bucket["duration_s"],
			energy_j=energy_j,
			n_with_energy=bucket["n_with_energy"],
			n_missing_energy=bucket["n_missing_energy"],
			energy_complete=energy_complete,
			n_zero_energy=bucket["n_zero_energy"],
			n_measurement_complete=bucket["n_measurement_complete"],
			n_failed_events=bucket["n_failed_events"],
			reference=reference if valid else None,
			share=share,
			reference_valid=valid,
			reconciles=bool(valid and energy_complete),
		))

	rows.extend(_residual_rows(scoped_events, trajectories, group_by,
	                           scope_dimensions, domains, references))
	rows.sort(key=_sort_key)
	return rows


def _residual_rows(scoped_events, trajectories, group_by, scope_dimensions,
                   domains, references):
	"""One synthetic row per (scope, domain) for energy no event accounts for.

	This is the difference between the independently measured trajectory
	window and the sum of its events: inter-event gaps, and counter boundary
	effects. It is emitted so that named categories plus residual reconcile to
	the trajectory total instead of leaving an unexplained gap in a figure.

	Negative residuals are reported as measured. A small negative value is a
	real consequence of counter resolution and interpolation at window
	boundaries; clamping it to zero would hide exactly the effect a reader
	needs to judge the measurement by.
	"""
	index = _trajectory_index(trajectories)
	rows = []
	for scope, scope_group in scoped_events.items():
		gap_s = 0.0
		for question_id in {e.get("question_id") for e in scope_group}:
			row = index.get(question_id)
			if row:
				gap_s += row.get("inter_event_gap_s") or 0.0

		for domain in domains:
			reference, attributed, valid = references.get(
				(scope, domain), (None, None, False))
			if not valid:
				continue  # no reference => no residual, never a guessed one
			residual = reference - attributed
			rows.append(_row(
				row_kind=RESIDUAL, group_by=group_by,
				key=tuple(scope[scope_dimensions.index(d)]
				          if d in scope_dimensions else RESIDUAL_LABEL
				          for d in group_by),
				domain=domain,
				n_events=0, duration_s=gap_s,
				energy_j=residual,
				n_with_energy=0, n_missing_energy=0, energy_complete=True,
				n_zero_energy=0, n_measurement_complete=0, n_failed_events=0,
				reference=reference,
				share=(residual / reference) if reference else None,
				reference_valid=True, reconciles=True,
			))
	return rows


def _row(row_kind, group_by, key, domain, n_events, duration_s, energy_j,
         n_with_energy, n_missing_energy, energy_complete, n_zero_energy,
         n_measurement_complete, n_failed_events, reference, share,
         reference_valid, reconciles):
	row = {"row_kind": row_kind}
	for dimension, value in zip(group_by, key):
		row[dimension] = value
	row.update({
		"energy_domain": domain,
		"domain_role": DOMAIN_ROLES.get(domain, "diagnostic"),
		"n_events": n_events,
		"duration_s": duration_s,
		"energy_j": energy_j,
		"n_events_with_energy": n_with_energy,
		"n_events_missing_energy": n_missing_energy,
		"energy_complete": energy_complete,
		# Counter quantization is surfaced, never corrected. A group where
		# most events read exactly 0.000 J has not been shown to consume no
		# energy; its spans were shorter than the counter's update interval.
		# Downstream code must not read energy_j as a measured floor there.
		"n_zero_energy_events": n_zero_energy,
		"zero_energy_fraction": (n_zero_energy / n_with_energy
		                         if n_with_energy else None),
		"n_events_measurement_complete": n_measurement_complete,
		"all_events_measurement_complete": bool(
			n_events and n_measurement_complete == n_events),
		"n_failed_events": n_failed_events,
		"trajectory_energy_j": reference,
		"share_of_trajectory": share,
		"trajectory_reference_valid": reference_valid,
		"reconciles": reconciles,
	})
	return row


def _sort_key(row):
	return (row.get("energy_domain") or "",
	        row.get("row_kind") == RESIDUAL,
	        -(row.get("energy_j") or 0.0))


# --------------------------------------------------------------------------
# reconciliation check
# --------------------------------------------------------------------------

def reconciliation(rows, group_by):
	"""Per (scope, domain): do named groups plus residual meet the reference?

	Returns one dict per reconcilable (scope, domain) with the trajectory
	reference, the summed rows, and the signed difference. A caller can assert
	on `difference_j`; a figure can refuse to render a slice set that does not
	reconcile.
	"""
	scope_dimensions = _scope_dimensions(tuple(group_by))
	totals = defaultdict(lambda: {"named_j": 0.0, "residual_j": 0.0,
	                              "reference_j": None, "complete": True})
	for row in rows:
		if not row.get("reconciles"):
			if row["row_kind"] == MEASURED:
				scope = tuple(row.get(d) for d in scope_dimensions)
				totals[(scope, row["energy_domain"])]["complete"] = False
			continue
		scope = tuple(row.get(d) for d in scope_dimensions)
		bucket = totals[(scope, row["energy_domain"])]
		bucket["reference_j"] = row["trajectory_energy_j"]
		if row["row_kind"] == RESIDUAL:
			bucket["residual_j"] += row["energy_j"] or 0.0
		else:
			bucket["named_j"] += row["energy_j"] or 0.0

	report = []
	for (scope, domain), bucket in sorted(totals.items(), key=lambda kv: str(kv[0])):
		if bucket["reference_j"] is None or not bucket["complete"]:
			continue
		accounted = bucket["named_j"] + bucket["residual_j"]
		report.append({
			"scope": dict(zip(scope_dimensions, scope)),
			"energy_domain": domain,
			"named_j": bucket["named_j"],
			"residual_j": bucket["residual_j"],
			"accounted_j": accounted,
			"trajectory_energy_j": bucket["reference_j"],
			"difference_j": accounted - bucket["reference_j"],
		})
	return report


# --------------------------------------------------------------------------
# output
# --------------------------------------------------------------------------

def field_order(rows):
	"""Stable column order: dimensions first, then measures."""
	if not rows:
		return []
	measures = ("energy_domain", "domain_role", "n_events", "duration_s",
	            "energy_j", "n_events_with_energy", "n_events_missing_energy",
	            "energy_complete", "n_zero_energy_events",
	            "zero_energy_fraction", "n_events_measurement_complete",
	            "all_events_measurement_complete", "n_failed_events",
	            "trajectory_energy_j", "share_of_trajectory",
	            "trajectory_reference_valid", "reconciles")
	dimensions = [k for k in rows[0] if k not in measures and k != "row_kind"]
	return ["row_kind"] + dimensions + list(measures)


def write_csv(rows, path):
	"""Long-format CSV. A null stays an empty cell, never 0."""
	fields = field_order(rows)
	with open(path, "w", newline="") as f:
		writer = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
		writer.writeheader()
		for row in rows:
			writer.writerow({k: ("" if row.get(k) is None else row.get(k))
			                 for k in fields})


def write_json(rows, path, report=None):
	with open(path, "w") as f:
		json.dump({"rows": rows, "reconciliation": report or []}, f, indent=2)


def load_events(path):
	events = []
	with open(path) as f:
		for line in f:
			line = line.strip()
			if not line:
				continue
			try:
				events.append(json.loads(line))
			except json.JSONDecodeError:
				pass  # torn line from a crash; skip, don't die
	return events


def load_trajectories(path):
	"""Accepts trajectory_summary.json (dict) or a bare list of rows."""
	with open(path) as f:
		data = json.load(f)
	if isinstance(data, dict):
		return data.get("trajectories", [])
	return data


def main():
	ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
	ap.add_argument("--events", required=True,
	                help="events_attributed.jsonl")
	ap.add_argument("--trajectories", default="",
	                help="trajectory_summary.json; without it no shares or "
	                     "residual rows are produced")
	ap.add_argument("--group-by", default="operation_label",
	                help="comma-separated dimensions, e.g. "
	                     "operation_label,traversal_depth")
	ap.add_argument("--domains", default=",".join(DEFAULT_DOMAINS))
	ap.add_argument("--out", default="", help="CSV output path")
	ap.add_argument("--out-json", default="", help="JSON output path")
	ap.add_argument("--list-dimensions", action="store_true")
	args = ap.parse_args()

	events = load_events(args.events)
	if args.list_dimensions:
		for dimension in available_dimensions(events):
			print(dimension)
		return

	trajectories = load_trajectories(args.trajectories) if args.trajectories else []
	group_by = [d.strip() for d in args.group_by.split(",") if d.strip()]
	domains = [d.strip() for d in args.domains.split(",") if d.strip()]

	rows = aggregate(events, trajectories, group_by=group_by, domains=domains)
	report = reconciliation(rows, group_by)

	if args.out:
		write_csv(rows, args.out)
		print(f"aggregate -> {args.out}")
	if args.out_json:
		write_json(rows, args.out_json, report)
		print(f"aggregate -> {args.out_json}")

	if not trajectories:
		print("NOTE: no trajectory summary given -> no trajectory reference, "
		      "so share_of_trajectory is null and no residual rows exist. "
		      "Shares are never computed against a fabricated denominator.")

	_render(rows, group_by, report)


def _render(rows, group_by, report):
	def cell(value, width, spec=".3f"):
		if value is None:
			return f"{'':>{width}}"
		if isinstance(value, float):
			return f"{value:>{width}{spec}}"
		return f"{str(value):>{width}}"

	widths = {d: max([len(d)] + [len(str(r.get(d))) for r in rows]) + 2
	          for d in group_by}
	for domain in dict.fromkeys(r["energy_domain"] for r in rows):
		subset = [r for r in rows if r["energy_domain"] == domain]
		if not any(r["energy_j"] is not None for r in subset):
			print(f"\n=== {domain}: not measured on this host (null, not zero) ===")
			continue
		print(f"\n=== {domain} ===")
		head = "".join(f"{d:<{widths[d]}}" for d in group_by)
		print(f"{head}{'n':>6}{'dur_s':>10}{'energy_J':>14}"
		      f"{'share':>9}{'zero_ev':>9}{'complete':>10}")
		for row in subset:
			dims = "".join(f"{str(row.get(d)):<{widths[d]}}" for d in group_by)
			print(f"{dims}{row['n_events']:>6}{cell(row['duration_s'], 10)}"
			      f"{cell(row['energy_j'], 14)}"
			      f"{cell(row['share_of_trajectory'], 9, '.4f')}"
			      f"{row['n_zero_energy_events']:>9}"
			      f"{str(row['all_events_measurement_complete']):>10}")

	if report:
		print("\n=== reconciliation (named + residual vs trajectory) ===")
		for entry in report:
			print(f"  {entry['energy_domain']:<22} "
			      f"scope={entry['scope'] or '{all}'}  "
			      f"named={entry['named_j']:.3f}  "
			      f"residual={entry['residual_j']:.3f}  "
			      f"reference={entry['trajectory_energy_j']:.3f}  "
			      f"diff={entry['difference_j']:+.9f}")


if __name__ == "__main__":
	main()
