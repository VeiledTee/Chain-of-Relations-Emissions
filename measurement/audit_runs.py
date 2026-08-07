"""Audit measurement runs: internal-consistency checks for events.jsonl,
power.csv, and CodeCarbon output in each run directory.

Checks per run:
  1. events-sum vs power-log span  (busy time cannot exceed wall)  [resume/append bug]
  2. multi-hour gaps between consecutive events                    [split sessions]
  3. duplicate question_ids                                         [double-appended runs]
  4. events outside the power-log window                            [stale events]
  5. energy triangle: counter sum vs power-curve integral vs CodeCarbon GPU
  6. implied mean power sanity (<= plausible board power)

Usage:
  python measurement/audit_runs.py                      # audits measurement/runs/*
  python measurement/audit_runs.py path/to/runs_dir     # explicit
"""

import csv
import glob
import json
import os
import sys

MAX_PLAUSIBLE_W = 460.0   # RTX 4090 board cap + margin
GAP_S = 300.0             # gap between events that suggests a split run


def load_events(path):
	ev = [json.loads(l) for l in open(path) if l.strip()]
	ev.sort(key=lambda e: e["t_start"])
	return ev


def load_power(path):
	rows = list(csv.DictReader(open(path)))
	t = [float(r["t"]) for r in rows]
	pcols = [c for c in rows[0] if c != "t"]
	p = [sum(float(r[c]) for c in pcols) for r in rows]
	return t, p


def trapz(t, p, lo=None, hi=None):
	E = 0.0
	for i in range(len(t) - 1):
		if lo is not None and t[i + 1] < lo:
			continue
		if hi is not None and t[i] > hi:
			break
		E += (p[i] + p[i + 1]) / 2.0 * (t[i + 1] - t[i])
	return E


def codecarbon_gpu_kwh(run_dir):
	for f in glob.glob(os.path.join(run_dir, "*emissions*.csv")):
		try:
			rows = list(csv.DictReader(open(f)))
			if rows and "gpu_energy" in rows[0]:
				return float(rows[-1]["gpu_energy"])  # kWh
		except Exception:
			pass
	return None


def audit(run_dir):
	name = os.path.basename(run_dir.rstrip("/"))
	ef = os.path.join(run_dir, "events.jsonl")
	pf = os.path.join(run_dir, "power.csv")
	problems, notes = [], []

	if not os.path.exists(ef):
		print(f"[{name}] SKIP: no events.jsonl")
		return
	ev = load_events(ef)
	busy = sum(e["t_end"] - e["t_start"] for e in ev)
	qids = [e["question_id"] for e in ev]
	span_ev = ev[-1]["t_end"] - ev[0]["t_start"]

	# duplicates (same qid + same label appearing more times than dc-retry allows
	# is fuzzy; flag exact duplicate (qid, label, t_start) instead)
	seen, dups = set(), 0
	for e in ev:
		k = (e["question_id"], e["label"], round(e["t_start"], 3))
		dups += k in seen
		seen.add(k)
	if dups:
		problems.append(f"{dups} exact duplicate events")

	# split-session gaps
	gaps = [ev[i + 1]["t_start"] - ev[i]["t_end"] for i in range(len(ev) - 1)]
	big = [g for g in gaps if g > GAP_S]
	if big:
		problems.append(f"{len(big)} gap(s) > {GAP_S:.0f}s between events "
		                f"(max {max(big)/3600:.2f}h) - split/resumed run?")

	counter_j = sum(e.get("gpu_energy_j") or 0 for e in ev)
	n_counter = sum(1 for e in ev if e.get("gpu_energy_j") is not None)

	line2 = ""
	if os.path.exists(pf):
		t, p = load_power(pf)
		wall = t[-1] - t[0]
		if busy > wall * 1.02:
			problems.append(f"busy {busy:.0f}s > power-log wall {wall:.0f}s "
			                f"(x{busy/wall:.2f}) - INVARIANT VIOLATION")
		outside = sum(1 for e in ev
		              if e["t_end"] < t[0] or e["t_start"] > t[-1])
		if outside:
			problems.append(f"{outside} events outside power window - stale/appended")
		integral_j = trapz(t, p)
		mean_w = integral_j / wall if wall else 0
		if counter_j and wall:
			implied_w = counter_j / max(busy, 1e-9)
			if implied_w > MAX_PLAUSIBLE_W:
				problems.append(f"implied busy power {implied_w:.0f}W > "
				                f"{MAX_PLAUSIBLE_W:.0f}W - impossible")
			ratio = counter_j / integral_j if integral_j else float("nan")
			notes.append(f"counter/integral = {ratio:.3f} "
			             f"(expect ~0.90-1.00; counter excludes idle/load)")
		cc = codecarbon_gpu_kwh(run_dir)
		cc_j = cc * 3.6e6 if cc is not None else None
		line2 = (f"    power: wall={wall:.0f}s mean={mean_w:.0f}W "
		         f"integral={integral_j/1000:.1f}kJ"
		         + (f"  codecarbon_gpu={cc_j/1000:.1f}kJ" if cc_j else
		            "  codecarbon_gpu=n/a"))
		if cc_j and integral_j and not 0.85 <= cc_j / integral_j <= 1.15:
			problems.append(f"CodeCarbon GPU {cc_j/1000:.0f}kJ vs integral "
			                f"{integral_j/1000:.0f}kJ disagree >15%")
	else:
		notes.append("no power.csv - counter-only audit")

	status = "FAIL" if problems else "OK  "
	print(f"[{status}] {name}: n={len(ev)} uniq_q={len(set(qids))} "
	      f"busy={busy:.0f}s span={span_ev:.0f}s "
	      f"counter={counter_j/1000:.1f}kJ ({n_counter}/{len(ev)} measured)")
	if line2:
		print(line2)
	for p_ in problems:
		print(f"    !! {p_}")
	for n_ in notes:
		print(f"    -- {n_}")


def main():
	base = sys.argv[1] if len(sys.argv) > 1 else os.path.join(
		os.path.dirname(os.path.abspath(__file__)), "runs")
	dirs = sorted(d for d in glob.glob(os.path.join(base, "*")) if os.path.isdir(d))
	if not dirs:
		print(f"no run dirs under {base}")
		return
	for d in dirs:
		audit(d)


if __name__ == "__main__":
	main()
