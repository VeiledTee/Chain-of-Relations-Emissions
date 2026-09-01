"""Trajectory-level energy accounting.

The claim is not merely that hardware energy can be measured, but that it
can be *attributed to operations inside an agent trajectory*. That requires
reconciling two independently-derived quantities per question:

    trajectory_energy_j        energy over the whole question window, taken
                               from the hardware timeline in one shot
    sum_attributed_j           sum of the per-event attributed energies
    unattributed_energy_j      trajectory_energy_j - sum_attributed_j
    attribution_coverage       sum_attributed_j / trajectory_energy_j

Coverage is NOT forced to 1.0. The residual is real: Python orchestration,
inter-event gaps, idle model-server draw and background services all consume
energy inside the trajectory window that belongs to no named operation.
Making that residual explicit is the honest result; hiding it inside named
operations would overstate operation-level attribution.

Like-for-like instruments
-------------------------
Both sides of the reconciliation must come from the SAME physical instrument,
otherwise the residual measures instrument disagreement rather than
unattributed work. For every domain the trajectory reference is a cumulative
hardware counter differenced across the window:

    trajectory_gpu_energy_j          = nvml_cumulative_end - nvml_cumulative_start
    trajectory_cpu_package_energy_j  = rapl_package_end   - rapl_package_start
    trajectory_dram_energy_j         = rapl_dram_end      - rapl_dram_start

matching how per-event energy is produced. Integrated ~10 Hz sampled power is
NOT a reference; it resolves fast inference transients poorly and errs in both
directions (measured ratio to the counter spans roughly 0.91-1.20 on real
trajectories), with error growing as windows shorten. That previously drove
coverage above 1 on short trajectories. It survives only as the separately
named diagnostic `sampled_gpu_energy_estimate_j`, alongside mean/peak power.

A domain whose counter is unavailable is reported as unavailable. It is never
back-filled from the sampled estimate.

Residual sign
-------------
    unattributed_energy_j = trajectory_energy_j - sum_attributed_j

Small NEGATIVE residuals are expected and are NOT clamped to zero. The event
log reads the counter in-band at exact event boundaries, while the trajectory
reference interpolates the sampler's reads to the trajectory edges; counter
update granularity and boundary timing can therefore place slightly more
energy inside the events than the interpolated window shows. Clamping would
hide exactly the instrument behaviour this reconciliation exists to expose.
Magnitude is reported so it can be characterised.

Overlap caveat
--------------
Summing per-event energy is only valid when events do not overlap in time.
Where events nest or overlap, the overlapped interval is counted once per
overlapping event and the sum can exceed the trajectory total, giving
coverage > 1.0. This module therefore MEASURES overlap rather than silently
correcting for it: every trajectory row reports overlap_seconds and
max_concurrency, and coverage above 1.0 is reported as-is and flagged. The
smallest defensible accounting rule, applied only when a run is proven
overlap-free, is plain summation; that condition is checked, not assumed.
"""

import json
from collections import defaultdict

#: Energy fields carried per trajectory, mirroring the event boundary.
ENERGY_FIELDS = ("gpu_energy_j", "cpu_package_energy_j", "dram_energy_j",
                 "measured_energy_j")

#: Diagnostic fields copied verbatim from the window measurement. These are
#: NOT energy references and must never be used as a coverage denominator.
#: sampled_gpu_energy_estimate_j in particular is integrated ~10 Hz power and
#: is kept only for comparison against the cumulative counter.
DIAGNOSTIC_FIELDS = ("gpu_energy_source", "sampled_gpu_energy_estimate_j",
                     "mean_gpu_power_w", "peak_gpu_power_w")


def event_window(event):
	"""(start, end) of an event, tolerating pre-schema-v1 key names."""
	start = event.get("start_timestamp", event.get("t_start"))
	end = event.get("end_timestamp", event.get("t_end"))
	return start, end


def merge_intervals(intervals):
	"""Union of [start, end) intervals, as a sorted non-overlapping list."""
	ordered = sorted(i for i in intervals if i[0] is not None and i[1] is not None)
	merged = []
	for start, end in ordered:
		if end < start:
			start, end = end, start
		if merged and start <= merged[-1][1]:
			merged[-1] = (merged[-1][0], max(merged[-1][1], end))
		else:
			merged.append((start, end))
	return merged


def overlap_stats(intervals):
	"""How much wall time is covered by more than one event, and how deeply.

	Returns (union_seconds, busy_seconds, overlap_seconds, max_concurrency)
	where busy_seconds is the naive sum of durations. overlap_seconds is
	busy - union: the time double-counted by plain summation.
	"""
	windows = [(s, e) for s, e in intervals if s is not None and e is not None]
	if not windows:
		return 0.0, 0.0, 0.0, 0

	busy = sum(max(0.0, e - s) for s, e in windows)
	union = sum(e - s for s, e in merge_intervals(windows))

	edges = []
	for start, end in windows:
		if end < start:
			start, end = end, start
		edges.append((start, 1))
		edges.append((end, -1))
	edges.sort()
	depth = peak = 0
	for _, delta in edges:
		depth += delta
		peak = max(peak, depth)

	return union, busy, max(0.0, busy - union), peak


def _sum_strict(values):
	"""Sum, or None if any component is missing.

	A partial sum over a field some events lack would misstate the boundary,
	so any missing component makes the trajectory total null.
	"""
	if not values:
		return None
	if any(v is None for v in values):
		return None
	return sum(values)


def trajectory_windows(events):
	"""First start and last end per question_id."""
	windows = {}
	for event in events:
		q = event.get("question_id")
		start, end = event_window(event)
		if q is None or start is None or end is None:
			continue
		if q not in windows:
			windows[q] = [start, end]
		else:
			windows[q][0] = min(windows[q][0], start)
			windows[q][1] = max(windows[q][1], end)
	return {q: tuple(v) for q, v in windows.items()}


def accumulate(events, window_energy=None):
	"""Per-question trajectory accounting rows.

	events        attributed events (attribution.py output)
	window_energy optional callable (t0, t1) -> dict of ENERGY_FIELDS for the
	              whole window, computed independently from the hardware
	              timeline. When omitted, the trajectory total falls back to
	              the summed event energy and coverage is reported as None,
	              because there is then no independent quantity to reconcile
	              against and a coverage of exactly 1.0 would be circular.
	"""
	by_question = defaultdict(list)
	for event in events:
		by_question[event.get("question_id")].append(event)

	rows = []
	for question_id, group in by_question.items():
		windows = [event_window(e) for e in group]
		union_s, busy_s, overlap_s, concurrency = overlap_stats(windows)
		starts = [w[0] for w in windows if w[0] is not None]
		ends = [w[1] for w in windows if w[1] is not None]
		t0 = min(starts) if starts else None
		t1 = max(ends) if ends else None

		row = {
			"question_id": question_id,
			"n_events": len(group),
			"trajectory_start": t0,
			"trajectory_end": t1,
			"trajectory_wall_s": (t1 - t0) if (t0 is not None and t1 is not None) else None,
			"event_busy_s": busy_s,
			"event_union_s": union_s,
			"overlap_s": overlap_s,
			"max_concurrency": concurrency,
			"events_overlap": overlap_s > 1e-9,
			"n_failed_events": sum(1 for e in group
			                       if e.get("status") not in (None, "ok")),
			"max_iteration": _max_or_none(e.get("iteration") for e in group),
			"max_traversal_depth": _max_or_none(e.get("traversal_depth") for e in group),
		}

		# Inter-event gap: wall time inside the trajectory that no event covers.
		if row["trajectory_wall_s"] is not None:
			row["inter_event_gap_s"] = max(0.0, row["trajectory_wall_s"] - union_s)
		else:
			row["inter_event_gap_s"] = None

		measured_window = window_energy(t0, t1) if (window_energy and t0 is not None) else None

		for field in ENERGY_FIELDS:
			attributed = _sum_strict([e.get(field) for e in group])
			row[f"sum_attributed_{field}"] = attributed

			if measured_window is not None:
				total = measured_window.get(field)
			else:
				total = attributed
			row[f"trajectory_{field}"] = total

			if total is None or attributed is None:
				row[f"unattributed_{field}"] = None
				row[f"coverage_{field}"] = None
			else:
				row[f"unattributed_{field}"] = total - attributed
				row[f"coverage_{field}"] = (attributed / total) if total else None

		# Diagnostics travel alongside the references but are never one.
		if measured_window is not None:
			for field in DIAGNOSTIC_FIELDS:
				row[field] = measured_window.get(field)
			sampled = measured_window.get("sampled_gpu_energy_estimate_j")
			counter = row.get("trajectory_gpu_energy_j")
			# How far the ~10 Hz integrated estimate sits from the counter.
			# Recorded to characterise the sampler, not to correct anything.
			row["sampled_vs_counter_ratio"] = (
				sampled / counter if (sampled is not None and counter) else None)

		# Coverage is only meaningful against an independently measured window.
		row["coverage_is_independent"] = measured_window is not None
		if not row["coverage_is_independent"]:
			for field in ENERGY_FIELDS:
				row[f"coverage_{field}"] = None
				row[f"unattributed_{field}"] = None

		row["coverage_valid"] = bool(
			row["coverage_is_independent"] and not row["events_overlap"])
		rows.append(row)

	rows.sort(key=lambda r: (r["trajectory_start"] is None, r["trajectory_start"]))
	return rows


def _max_or_none(values):
	present = [v for v in values if v is not None]
	return max(present) if present else None


def summarize(rows):
	"""Run-level roll-up of the trajectory rows."""
	if not rows:
		return {"n_trajectories": 0}

	overlapping = [r for r in rows if r["events_overlap"]]
	coverages = [r["coverage_gpu_energy_j"] for r in rows
	             if r.get("coverage_gpu_energy_j") is not None]
	summary = {
		"n_trajectories": len(rows),
		"n_events": sum(r["n_events"] for r in rows),
		"n_trajectories_with_overlap": len(overlapping),
		"max_concurrency": max(r["max_concurrency"] for r in rows),
		"total_overlap_s": sum(r["overlap_s"] for r in rows),
		"total_inter_event_gap_s": sum(r["inter_event_gap_s"] or 0.0 for r in rows),
		"coverage_is_independent": any(r["coverage_is_independent"] for r in rows),
		"n_failed_events": sum(r["n_failed_events"] for r in rows),
	}
	if coverages:
		ordered = sorted(coverages)
		summary["gpu_coverage_min"] = ordered[0]
		summary["gpu_coverage_median"] = ordered[len(ordered) // 2]
		summary["gpu_coverage_max"] = ordered[-1]
		summary["n_coverage_above_one"] = sum(1 for c in coverages if c > 1.0)

	# Residual characterisation. Negative residuals are preserved, counted and
	# reported -- never clamped -- so counter-boundary behaviour stays visible.
	residuals = [r["unattributed_gpu_energy_j"] for r in rows
	             if r.get("unattributed_gpu_energy_j") is not None]
	if residuals:
		ordered = sorted(residuals)
		summary["gpu_residual_min_j"] = ordered[0]
		summary["gpu_residual_median_j"] = ordered[len(ordered) // 2]
		summary["gpu_residual_max_j"] = ordered[-1]
		summary["n_negative_gpu_residual"] = sum(1 for v in residuals if v < 0)
		summary["max_abs_negative_gpu_residual_j"] = (
			abs(min(residuals)) if min(residuals) < 0 else 0.0)

	sources = {r.get("gpu_energy_source") for r in rows if r.get("gpu_energy_source")}
	if sources:
		summary["gpu_energy_sources"] = sorted(sources)
	ratios = [r["sampled_vs_counter_ratio"] for r in rows
	          if r.get("sampled_vs_counter_ratio") is not None]
	if ratios:
		ordered = sorted(ratios)
		summary["sampled_vs_counter_ratio_median"] = ordered[len(ordered) // 2]
		summary["sampled_vs_counter_ratio_min"] = ordered[0]
		summary["sampled_vs_counter_ratio_max"] = ordered[-1]
	return summary


def render(rows, summary):
	L = []
	L.append("=== trajectory accounting ===")
	L.append(f"trajectories={summary['n_trajectories']}  "
	         f"events={summary.get('n_events', 0)}  "
	         f"failed_events={summary.get('n_failed_events', 0)}")
	L.append(f"overlapping trajectories: {summary['n_trajectories_with_overlap']}"
	         f"  max_concurrency={summary['max_concurrency']}"
	         f"  total_overlap={summary['total_overlap_s']:.3f}s")
	L.append(f"inter-event gap (unattributed wall time): "
	         f"{summary['total_inter_event_gap_s']:.3f}s")
	if not summary.get("coverage_is_independent"):
		L.append("coverage: NOT COMPUTED -- no independent whole-window energy "
		         "available, so a coverage figure would be circular (1.0 by "
		         "construction). Reporting wall-time attribution only.")
	if summary.get("gpu_energy_sources"):
		L.append(f"GPU trajectory reference: {summary['gpu_energy_sources']}")
	if "n_negative_gpu_residual" in summary:
		L.append(f"GPU residual (traj - sum_events): "
		         f"min={summary['gpu_residual_min_j']:.3f}J "
		         f"median={summary['gpu_residual_median_j']:.3f}J "
		         f"max={summary['gpu_residual_max_j']:.3f}J   "
		         f"negative: {summary['n_negative_gpu_residual']}"
		         f"/{summary['n_trajectories']} (preserved, not clamped)")
	if "sampled_vs_counter_ratio_median" in summary:
		L.append(f"DIAGNOSTIC sampled/counter ratio: "
		         f"median={summary['sampled_vs_counter_ratio_median']:.3f} "
		         f"min={summary['sampled_vs_counter_ratio_min']:.3f} "
		         f"max={summary['sampled_vs_counter_ratio_max']:.3f}  "
		         f"(integrated ~10Hz power vs cumulative counter)")
	L.append("")
	L.append(f"{'question':<22} {'ev':>4} {'wall_s':>8} {'sum_ev_J':>10} "
	         f"{'traj_J':>10} {'resid_J':>9} {'cov':>7} {'sampl_J':>10}")
	for r in rows[:40]:
		cov = r.get("coverage_gpu_energy_j")
		L.append(f"{str(r['question_id']):<22} {r['n_events']:>4} "
		         f"{_f(r['trajectory_wall_s']):>8} "
		         f"{_f(r.get('sum_attributed_gpu_energy_j')):>10} "
		         f"{_f(r.get('trajectory_gpu_energy_j')):>10} "
		         f"{_f(r.get('unattributed_gpu_energy_j')):>9} "
		         f"{(f'{cov:.3f}' if cov is not None else '-'):>7} "
		         f"{_f(r.get('sampled_gpu_energy_estimate_j')):>10}")
	if len(rows) > 40:
		L.append(f"... {len(rows) - 40} more trajectories")
	return "\n".join(L)


def _f(value, places=3):
	return "-" if value is None else f"{value:.{places}f}"


def write_artifacts(rows, summary, csv_path=None, json_path=None):
	if json_path:
		with open(json_path, "w") as f:
			json.dump({"summary": summary, "trajectories": rows}, f, indent=2)
	if csv_path and rows:
		import csv as _csv
		fields = list(rows[0].keys())
		with open(csv_path, "w", newline="") as f:
			w = _csv.DictWriter(f, fieldnames=fields)
			w.writeheader()
			w.writerows(rows)
