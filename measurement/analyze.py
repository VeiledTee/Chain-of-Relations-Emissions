"""Analyze one measured run's events.jsonl into the figures/tables C2 needs.

Reads events.jsonl directly (richer than energy_summary.csv: keeps per-event
meta and per-question structure). Produces, for a run directory:

  1. per-question energy distribution      (hist + summary stats)
  2. inference-vs-tool split               (energy share vs call-count share)
  3. per-label breakdown                   (energy, count, mean per call)
  4. token vs energy relationship          (prefill-dominated check)
  5. cost variance across questions        (the fanout story)
  6. a plaintext tables.txt for the thesis + PNGs

Usage:
  python measurement/analyze.py measurement/runs/smoke_cor_webqsp
  python measurement/analyze.py measurement/runs/<tag> --no-plots   # tables only

Honest-units note (mirrors EMISSIONS.md): per-question energy is robust;
per-event GPU energy is trustworthy where gpu_energy_j came from the NVML
counter; CPU/DRAM are only meaningful when RAPL was present (bare metal).
This script reports whatever is in the data and flags what's missing.
"""

import argparse
import json
import os
import sys
from collections import defaultdict


def load_events(run_dir):
	p = os.path.join(run_dir, "events.jsonl")
	if not os.path.isfile(p):
		sys.exit(f"no events.jsonl in {run_dir}")
	events = []
	for line in open(p):
		line = line.strip()
		if not line:
			continue
		try:
			events.append(json.loads(line))
		except json.JSONDecodeError:
			pass  # torn line from a crash; skip, don't die
	if not events:
		sys.exit("events.jsonl is empty")
	return events


def summarize(events):
	"""Return nested dicts of aggregates plus per-question rollups."""
	by_cat = defaultdict(lambda: defaultdict(float))
	by_label = defaultdict(lambda: defaultdict(float))
	by_q = defaultdict(lambda: defaultdict(float))
	retries = 0
	gpu_measured = gpu_missing = 0
	tok_in = tok_out = 0

	for e in events:
		cat, lab, q = e["category"], e["label"], e.get("question_id", "")
		gpu = e.get("gpu_energy_j")
		meta = e.get("meta", {})
		dur = e.get("duration_s", 0.0)

		if gpu is None:
			gpu_missing += 1
			gpu = 0.0
		else:
			gpu_measured += 1

		for tbl, key in ((by_cat, cat), (by_label, lab), (by_q, q)):
			tbl[key]["n"] += 1
			tbl[key]["gpu_j"] += gpu
			tbl[key]["dur_s"] += dur

		if meta.get("attempts", 1) and meta.get("attempts", 1) > 1:
			retries += 1
		tok_in += meta.get("input_tokens", 0) or 0
		tok_out += meta.get("output_tokens", 0) or 0
		# also record per-question LLM-call and tool-call counts
		if cat == "inference":
			by_q[q]["llm_calls"] += 1
		else:
			by_q[q]["tool_calls"] += 1

	return {
		"by_cat": by_cat, "by_label": by_label, "by_q": by_q,
		"retries": retries, "gpu_measured": gpu_measured,
		"gpu_missing": gpu_missing, "tok_in": tok_in, "tok_out": tok_out,
		"n_events": len(events), "n_questions": len(by_q),
	}


def _pct(x, total):
	return 100.0 * x / total if total else 0.0


def write_tables(s, run_dir):
	out = os.path.join(run_dir, "tables.txt")
	L = []
	L.append(f"# Analysis of {run_dir}")
	L.append(f"events={s['n_events']}  questions={s['n_questions']}  "
	         f"retries={s['retries']}  "
	         f"gpu_counter={s['gpu_measured']}  gpu_missing={s['gpu_missing']}")
	L.append("")

	# category split
	tot_gpu = sum(v["gpu_j"] for v in s["by_cat"].values())
	tot_n = sum(v["n"] for v in s["by_cat"].values())
	L.append("## inference vs tool")
	L.append(f"{'category':<10} {'events':>7} {'ev%':>6} {'gpu_J':>12} {'gpu%':>6} {'dur_s':>8}")
	for k in sorted(s["by_cat"]):
		v = s["by_cat"][k]
		L.append(f"{k:<10} {int(v['n']):>7} {_pct(v['n'], tot_n):>5.1f}% "
		         f"{v['gpu_j']:>12.1f} {_pct(v['gpu_j'], tot_gpu):>5.1f}% {v['dur_s']:>8.1f}")
	L.append("")

	# per-label
	L.append("## per-label")
	L.append(f"{'label':<22} {'events':>7} {'gpu_J':>12} {'J/call':>9} {'dur_s':>8}")
	for k in sorted(s["by_label"], key=lambda x: -s["by_label"][x]["gpu_j"]):
		v = s["by_label"][k]
		perc = v["gpu_j"] / v["n"] if v["n"] else 0
		L.append(f"{k:<22} {int(v['n']):>7} {v['gpu_j']:>12.1f} {perc:>9.3f} {v['dur_s']:>8.1f}")
	L.append("")

	# tokens
	L.append("## tokens")
	tt = s["tok_in"] + s["tok_out"]
	L.append(f"input={s['tok_in']:,}  output={s['tok_out']:,}  "
	         f"input%={_pct(s['tok_in'], tt):.1f}%  (prefill-dominated if >>50%)")
	L.append("")

	# per-question distribution
	qs = s["by_q"]
	gpus = sorted(v["gpu_j"] for v in qs.values())
	llmc = sorted(v.get("llm_calls", 0) for v in qs.values())
	toolc = sorted(v.get("tool_calls", 0) for v in qs.values())

	def stats(a):
		if not a:
			return (0, 0, 0, 0, 0)
		n = len(a)
		mean = sum(a) / n
		med = a[n // 2]
		return (min(a), med, mean, max(a), a[-1] / a[0] if a[0] else float('inf'))

	L.append("## per-question distributions (min / median / mean / max / max:min ratio)")
	for name, arr in (("gpu_J", gpus), ("llm_calls", llmc), ("tool_calls", toolc)):
		lo, md, mn, hi, ratio = stats(arr)
		L.append(f"{name:<12} {lo:>8.1f} {md:>8.1f} {mn:>8.1f} {hi:>8.1f}  "
		         f"spread {ratio:>6.1f}x")
	L.append("")
	L.append("Interpretation cue: a large tool_calls max:min ratio is the "
	         "graph-fanout cost-variance story; input%>>50 is prefill dominance; "
	         "gpu% concentrated in inference with events% in tool is the "
	         "energy-vs-frequency inversion.")

	text = "\n".join(L)
	open(out, "w").write(text + "\n")
	print(text)
	print(f"\n-> {out}")
	return qs, gpus


def make_plots(s, qs, gpus, run_dir):
	try:
		import matplotlib
		matplotlib.use("Agg")
		import matplotlib.pyplot as plt
	except Exception as e:
		print(f"(plots skipped: {e})")
		return

	tag = os.path.basename(run_dir.rstrip("/"))

	# 1. per-question energy histogram
	fig, ax = plt.subplots(figsize=(7, 4))
	ax.hist(gpus, bins=30, color="#4C72B0", edgecolor="white")
	ax.set_xlabel("GPU energy per question (J)")
	ax.set_ylabel("questions")
	ax.set_title(f"Per-question GPU energy — {tag}")
	fig.tight_layout()
	fig.savefig(os.path.join(run_dir, "fig_q_energy_hist.png"), dpi=130)
	plt.close(fig)

	# 2. inference-vs-tool: energy share vs count share
	cats = sorted(s["by_cat"])
	tot_gpu = sum(s["by_cat"][c]["gpu_j"] for c in cats) or 1
	tot_n = sum(s["by_cat"][c]["n"] for c in cats) or 1
	eshare = [s["by_cat"][c]["gpu_j"] / tot_gpu * 100 for c in cats]
	nshare = [s["by_cat"][c]["n"] / tot_n * 100 for c in cats]
	fig, ax = plt.subplots(figsize=(6, 4))
	x = range(len(cats))
	ax.bar([i - 0.2 for i in x], eshare, 0.4, label="energy %", color="#C44E52")
	ax.bar([i + 0.2 for i in x], nshare, 0.4, label="call-count %", color="#55A868")
	ax.set_xticks(list(x))
	ax.set_xticklabels(cats)
	ax.set_ylabel("% of run")
	ax.set_title(f"Energy vs frequency inversion — {tag}")
	ax.legend()
	fig.tight_layout()
	fig.savefig(os.path.join(run_dir, "fig_energy_vs_count.png"), dpi=130)
	plt.close(fig)

	# 3. per-label energy bar
	labs = sorted(s["by_label"], key=lambda x: -s["by_label"][x]["gpu_j"])
	vals = [s["by_label"][l]["gpu_j"] for l in labs]
	fig, ax = plt.subplots(figsize=(7, 4))
	ax.barh(labs, vals, color="#8172B3")
	ax.set_xlabel("GPU energy (J)")
	ax.set_title(f"Energy by label — {tag}")
	ax.invert_yaxis()
	fig.tight_layout()
	fig.savefig(os.path.join(run_dir, "fig_label_energy.png"), dpi=130)
	plt.close(fig)

	print(f"-> 3 PNGs in {run_dir}")


def main():
	ap = argparse.ArgumentParser()
	ap.add_argument("run_dir")
	ap.add_argument("--no-plots", action="store_true")
	args = ap.parse_args()

	events = load_events(args.run_dir)
	s = summarize(events)
	qs, gpus = write_tables(s, args.run_dir)
	if not args.no_plots:
		make_plots(s, qs, gpus, args.run_dir)


if __name__ == "__main__":
	main()