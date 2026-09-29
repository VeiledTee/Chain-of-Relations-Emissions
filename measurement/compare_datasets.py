"""Side-by-side comparison of two analysed datasets, from their output directories.

Reads only the CSVs that `correctness_energy.py` and `summarize_comparable_runs.py`
already wrote for each dataset. Nothing is re-scored, re-measured or pooled: a
dataset is a condition, and every number stays labelled with the dataset it came
from. The point is to see whether the same qualitative findings hold in both,
not to produce a combined statistic.

    python measurement/compare_datasets.py \
        --dataset WebQSP=~/webqsp_supervisor_summary \
        --dataset CWQ=~/cwq_supervisor_summary \
        --out ~/cwq_supervisor_summary/cross_dataset

Each directory must contain `correctness/correctness_binary.csv`,
`correctness/correctness_threeway.csv` and `summary/summary_overview.csv`;
`correctness/correctness_operation_energy.csv` and `summary/summary_fallback.csv`
are used when present.

Both datasets are scored by the same evaluator, but their gold structures differ
(best F1 over official parses for WebQSP, one answer list for CWQ) and so do
their difficulty and depth limits, so effectiveness levels are not comparable
across the two columns. What is comparable is the *direction* of
each within-dataset contrast, which is what the output tables show.
"""

import argparse
import csv
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)
sys.path.insert(0, HERE)

import summarize_comparable_runs as scr  # noqa: E402

SYSTEM_ORDER = scr.SYSTEM_ORDER
#: (question, how each dataset answers it) -- the findings to check for agreement.
ENERGY_FIELDS = ("dataset", "system", "n", "fully_correct_n", "fully_correct_mean_gpu_j",
                 "not_fully_correct_n", "not_fully_correct_mean_gpu_j", "mean_ratio",
                 "percent_difference", "median_ratio", "mean_difference_ci_low",
                 "mean_difference_ci_high", "direction")
OVERVIEW_FIELDS = ("dataset", "metric", "PoG", "ToG", "CoR")
OPERATION_FIELDS = ("dataset", "system", "operation_label", "total_attributed_gpu_j",
                    "share_of_system_attributed_pct", "rank")


def read_rows(path):
	if not os.path.exists(path):
		return None
	with open(path, newline="") as f:
		return list(csv.DictReader(f))


def number(value):
	if value in (None, ""):
		return None
	try:
		return float(value)
	except ValueError:
		return None


def direction(ratio, lo, hi):
	"""How this system answers "does not-fully-correct cost more?"."""
	if ratio is None:
		return "no data"
	if lo is not None and hi is not None:
		if lo > 0:
			return "costs more"
		if hi < 0:
			return "costs less"
		return "no difference resolved"
	return "costs more" if ratio > 1.0 else "costs less"


def energy_rows(dataset, directory):
	rows = read_rows(os.path.join(directory, "correctness", "correctness_binary.csv"))
	if rows is None:
		raise SystemExit(f"{dataset}: no correctness/correctness_binary.csv in {directory}")
	out = []
	for row in rows:
		ratio = number(row.get("incorrect_correct_energy_ratio"))
		lo = number(row.get("mean_difference_ci_low"))
		hi = number(row.get("mean_difference_ci_high"))
		correct_n = number(row.get("correct_n")) or 0
		incorrect_n = number(row.get("incorrect_n")) or 0
		out.append({
			"dataset": dataset, "system": row["system"],
			"n": int(correct_n + incorrect_n),
			"fully_correct_n": int(correct_n),
			"fully_correct_mean_gpu_j": number(row.get("mean_gpu_j_correct")),
			"not_fully_correct_n": int(incorrect_n),
			"not_fully_correct_mean_gpu_j": number(row.get("mean_gpu_j_incorrect")),
			"mean_ratio": ratio,
			"percent_difference": number(row.get("percent_difference")),
			"median_ratio": number(row.get("median_ratio")),
			"mean_difference_ci_low": lo, "mean_difference_ci_high": hi,
			"direction": direction(ratio, lo, hi),
		})
	return sorted(out, key=lambda r: scr.order_systems([r["system"]] ) and
	              (SYSTEM_ORDER.index(r["system"]) if r["system"] in SYSTEM_ORDER else 99))


def overview_rows(dataset, directory):
	rows = read_rows(os.path.join(directory, "summary", "summary_overview.csv"))
	if rows is None:
		raise SystemExit(f"{dataset}: no summary/summary_overview.csv in {directory}")
	fallback = read_rows(os.path.join(directory, "summary", "summary_fallback.csv")) or []
	out = []
	for row in rows + fallback:
		entry = {"dataset": dataset, "metric": row["metric"]}
		for name in ("PoG", "ToG", "CoR"):
			entry[name] = row.get(name, "")
		out.append(entry)
	return out


def operation_rows(dataset, directory, top=4):
	"""The operation labels that dominate each system's attributed GPU energy."""
	rows = read_rows(os.path.join(directory, "correctness",
	                              "correctness_operation_energy.csv"))
	if rows is None:
		return []
	totals = {}
	for row in rows:
		key = (row["system"], row["operation_label"])
		totals[key] = totals.get(key, 0.0) + (number(row["total_attributed_gpu_j"]) or 0.0)
	out = []
	for system in {k[0] for k in totals}:
		system_total = sum(v for k, v in totals.items() if k[0] == system)
		ranked = sorted(((label, v) for (s, label), v in totals.items() if s == system),
		                key=lambda kv: -kv[1])
		for rank, (label, value) in enumerate(ranked[:top], start=1):
			out.append({"dataset": dataset, "system": system, "operation_label": label,
			            "total_attributed_gpu_j": value,
			            "share_of_system_attributed_pct": scr.pct(value, system_total),
			            "rank": rank})
	return sorted(out, key=lambda r: (SYSTEM_ORDER.index(r["system"])
	                                  if r["system"] in SYSTEM_ORDER else 99, r["rank"]))


def write_csv(path, rows, fields):
	with open(path, "w", newline="") as f:
		writer = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore", lineterminator="\n")
		writer.writeheader()
		for row in rows:
			writer.writerow({k: ("" if row.get(k) is None else row[k]) for k in fields})
	return path


def fmt(value, decimals=2):
	if value in (None, ""):
		return ""
	if isinstance(value, str):
		return value
	return f"{value:,.0f}" if decimals == 0 else f"{value:,.{decimals}f}"


def main(argv=None):
	ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
	ap.add_argument("--dataset", action="append", metavar="NAME=DIR", required=True,
	                help="analysed output directory per dataset; repeatable")
	ap.add_argument("--out", required=True)
	args = ap.parse_args(argv)

	directories = scr.parse_pairs(args.dataset, "--dataset")
	if len(directories) < 2:
		raise SystemExit("give at least two --dataset NAME=DIR to compare")
	names = list(directories)
	out = os.path.expanduser(args.out)
	os.makedirs(out, exist_ok=True)

	energy, overview, operations = [], [], []
	for name in names:
		energy.extend(energy_rows(name, directories[name]))
		overview.extend(overview_rows(name, directories[name]))
		operations.extend(operation_rows(name, directories[name]))

	written = [
		write_csv(os.path.join(out, "cross_dataset_correctness_energy.csv"), energy,
		          ENERGY_FIELDS),
		write_csv(os.path.join(out, "cross_dataset_overview.csv"), overview, OVERVIEW_FIELDS),
	]
	if operations:
		written.append(write_csv(os.path.join(out, "cross_dataset_top_operations.csv"),
		                         operations, OPERATION_FIELDS))

	lines = ["# " + " vs ".join(names) + ": do the same findings hold?", "",
	         "Each dataset is a separate condition. Nothing is pooled, and the scorers "
	         "differ between datasets, so effectiveness levels are not comparable across "
	         "columns -- only the direction of each within-dataset contrast is.", "",
	         "## Does not-fully-correct cost more than fully correct?", "",
	         "| Dataset | System | Full n | Full mean J | Not-full n | Not-full mean J | "
	         "Mean ratio | % diff | Median ratio | 95% CI of mean difference | Reading |",
	         "|---|---|---|---|---|---|---|---|---|---|---|"]
	for r in energy:
		ci = (f"[{fmt(r['mean_difference_ci_low'], 0)}, {fmt(r['mean_difference_ci_high'], 0)}]"
		      if r["mean_difference_ci_low"] is not None else "")
		lines.append(f"| {r['dataset']} | {r['system']} | {r['fully_correct_n']:,} | "
		             f"{fmt(r['fully_correct_mean_gpu_j'], 0)} | {r['not_fully_correct_n']:,} | "
		             f"{fmt(r['not_fully_correct_mean_gpu_j'], 0)} | {fmt(r['mean_ratio'], 3)} | "
		             f"{fmt(r['percent_difference'], 1)} | {fmt(r['median_ratio'], 3)} | {ci} | "
		             f"{r['direction']} |")

	metrics = ("Mean GPU energy / question (J)", "Median GPU energy / question (J)",
	           "p90 GPU energy / question (J)", "Total GPU energy (J)",
	           "Mean LLM calls / question", "Mean input tokens / question",
	           "Mean output tokens / question",
	           "Top 10% of questions: share of total GPU energy (%)",
	           "Fallback rate (%)", "Mean GPU J, fallback", "Mean GPU J, non-fallback")
	lines += ["", "## Energy and workload per dataset", "",
	          "| Metric | Dataset | PoG | ToG | CoR |", "|---|---|---|---|---|"]
	for metric in metrics:
		for name in names:
			row = next((r for r in overview if r["dataset"] == name and r["metric"] == metric),
			           None)
			if row:
				lines.append(f"| {metric} | {name} | {row['PoG']} | {row['ToG']} | "
				             f"{row['CoR']} |")

	if operations:
		lines += ["", "## Which operations dominate attributed GPU energy", "",
		          "| Dataset | System | Rank | Operation | Share of system (%) |",
		          "|---|---|---|---|---|"]
		for r in operations:
			lines.append(f"| {r['dataset']} | {r['system']} | {r['rank']} | "
			             f"{r['operation_label']} | "
			             f"{fmt(r['share_of_system_attributed_pct'], 1)} |")

	lines += ["", "## Inputs", ""] + [f"- {n}: `{directories[n]}`" for n in names] + [""]
	md = os.path.join(out, "CROSS_DATASET.md")
	with open(md, "w", newline="") as f:
		f.write("\n".join(lines))
	written.append(md)

	for r in energy:
		print(f"{r['dataset']:8s} {r['system']:4s} ratio {fmt(r['mean_ratio'], 3):>6s}  "
		      f"median ratio {fmt(r['median_ratio'], 3):>6s}  {r['direction']}")
	for path in written:
		print(f"  -> {path}")
	return 0


if __name__ == "__main__":
	sys.exit(main())
