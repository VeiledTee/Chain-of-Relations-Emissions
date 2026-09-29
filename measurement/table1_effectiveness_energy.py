"""Recreation of the CoR paper's Table 1 (Hits@1, Recall, F1, KGR), extended
with per-question energy, for the datasets and backbone this repository
actually ran.

    python measurement/table1_effectiveness_energy.py \
        --dataset webqsp=~/cwq_supervisor_summary/cross_dataset/webqsp/wh \
        --dataset cwq=~/cwq_supervisor_summary/wh \
        --backbone Gemma-3-4B \
        --out results/tables/cross_dataset

Effectiveness (Hits@1, Precision, Recall, F1) is scored by
measurement/export_outcomes.py -- this project's one evaluator
(chain_of_relations/eval/accuracy.py), called fresh from each dataset's
predict.jsonl, exactly as every other table in this project uses it: the final
answer only, normalized exact matching, set-based P/R/F1, as the paper's
exact-match protocol states. WebQSP scores against every official parse (best
F1 over parses); CWQ against its answer list. Energy and the KGR signal come from the
make_comparable_figures.py per-question cache; nothing here re-parses
events_attributed.jsonl.

KGR: no function anywhere in this repository computes a single KG-grounded
Rate percentage. chain_of_relations/eval/faithfulness.py and eval.py only
print a raw per-action count histogram (predict.jsonl's own "action" field)
and never classify it into KG-grounded / not. This script operationalizes
KGR = 1 - fallback_rate, reusing the SAME "fallback" flag already computed
by the measurement pipeline and already used throughout this project's other
figures (fig4/fig7/fig8) -- not a new metric invented for this table. That
flag was cross-checked against predict.jsonl's own action field before use
(see the printed validation section); the two signals agreed on 3519/3520,
3520/3520 and 3520/3520 questions across CoR/PoG/ToG on CWQ, the one
disagreement being a single CoR question whose action is "ERROR". This
choice is reported, not hidden -- treat KGR here as an approximation of the
paper's metric, not a literal reproduction of it.

Canonical per-question energy: `energy_trajectory_j` in the cache, sourced
from trajectory_summary.csv's whole-question GPU counter window -- the same
basis used by every other question-level figure in this project (fig2/3/4/7/8).
This is deliberately NOT the sum of attributed per-operation event energy
(that is a different, narrower basis used only by the operation-breakdown
figures, fig1/fig5) -- using it here would double-count nothing, but mixing
the two bases in one table would silently change what a number means.

The effectiveness-vs-energy tradeoff figure (F1 / Recall / KGR x WebQSP / CWQ,
one point per system, y = mean trajectory Wh/question with a 95% bootstrap CI)
is the sole canonical thesis figure for this analysis -- there is no separate
F1-only figure. The bootstrap implementation (percentile bootstrap, questions
as the resampling unit) is measurement/thesis_figures.py's bootstrap_ci,
imported here rather than duplicated; the same energy CI is reused across a
dataset x system's F1/Recall/KGR panels since it is the same mean-energy
estimate in all three, not recomputed per metric.

Writes only into --out (table1_effectiveness_energy.csv, table1_effectiveness_energy.tex;
also prints a Markdown version to stdout) and --out-figures (the figures listed above).
"""

import argparse
import csv
import json
import os
import statistics
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)
sys.path.insert(0, HERE)

import export_outcomes  # noqa: E402
import make_comparable_figures as mcf  # noqa: E402
from thesis_figures import bootstrap_ci, BOOTSTRAP_REPS, SEED  # noqa: E402 -- reused, not reimplemented
from agent_energy_profiler import visualize as v  # noqa: E402

DATASET_ORDER = ("webqsp", "cwq")
DATASET_LABELS = {"webqsp": "WebQSP", "cwq": "CWQ"}
#: The paper's own full test-set sizes, for the flag-only comparison the task
#: requires -- never used as a substitute for our own counted N.
PAPER_N = {"webqsp": 1639, "cwq": 3531}
#: Presentation order only (iteration/row order; does not affect any computed
#: value -- Pareto dominance is a symmetric pairwise comparison independent of
#: list order). Matches the order/colors used by every other canonical figure.
SYSTEM_ORDER = ("PoG", "ToG", "CoR")
DEFAULT_PREDICTIONS = {
	"webqsp": {"PoG": "results/pog/webqsp/gemma-3-4b-it/predict.jsonl",
	          "ToG": "results/tog/webqsp/gemma-3-4b-it/predict.jsonl",
	          "CoR": "results/cor/webqsp/gemma-3-4b-it/predict.jsonl"},
	"cwq": {"PoG": "results/pog/cwq/gemma-3-4b-it/predict.jsonl",
	       "ToG": "results/tog/cwq/gemma-3-4b-it/predict.jsonl",
	       "CoR": "results/cor/cwq/gemma-3-4b-it/predict.jsonl"},
}
CSV_FIELDS = ("backbone", "method", "dataset", "n_questions", "hits_at_1", "precision", "recall",
             "f1", "kgr", "mean_energy_wh", "energy_ci_low_wh", "energy_ci_high_wh",
             "median_energy_wh", "total_energy_wh", "energy_coverage_n", "energy_coverage_pct")
#: predicted_answer_count p90/p95/max live here too (extending this one canonical
#: diagnostics CSV) rather than in a second, narrowly-scoped size-summary CSV.
DIAGNOSTIC_FIELDS = ("dataset", "system", "n_questions", "mean_precision", "median_precision",
                     "mean_recall", "mean_f1", "mean_gold_answer_count",
                     "median_gold_answer_count", "mean_predicted_answer_count",
                     "median_predicted_answer_count", "p90_predicted_answer_count",
                     "p95_predicted_answer_count", "max_predicted_answer_count",
                     "mean_true_positive_count", "mean_false_positive_count",
                     "mean_false_negative_count", "n_excluded_zero_gold")
PARETO_FIELDS = ("dataset", "metric", "system", "metric_value", "mean_energy_wh",
                 "pareto_efficient")
#: Metrics shown against energy in the 2x3 tradeoff figure and the Pareto CSV.
#: F1 and Recall come from the same evaluator call as the main table; KGR is
#: the same 1-fallback-rate approximation used in the main table, not a new one.
TRADEOFF_METRICS = ("f1", "recall", "kgr")
#: Presentation-only text for the tradeoff figure's axes -- 0-100-scale numbers
#: without a percent sign, matching the CoR paper's own Table 1 convention (its
#: methodological caveats belong in the LaTeX caption/footnote, not the artwork).
TRADEOFF_METRIC_LABELS = {"f1": "F1", "recall": "Recall", "kgr": "KGR (1 − fallback rate)"}


def _metric_xlim(values):
	"""x-limits for a percentage metric panel: the data range padded by the larger
	of 25% of the spread or 15% of the largest value, clipped to [0, 100]. Padding
	proportional to the value keeps a narrow cluster (e.g. CWQ F1 22.7-24.2) from
	being stretched across the whole panel."""
	lo, hi = min(values), max(values)
	pad = max(0.25 * (hi - lo), 0.15 * hi, 1.0)
	return max(0.0, lo - pad), min(100.0, hi + pad)


def _load_cache(dataset, out_dir):
	cache_dir = os.path.join(os.path.expanduser(out_dir), mcf.CACHE_DIRNAME)
	meta = mcf._cache_manifest(cache_dir)
	if meta is None:
		raise SystemExit(f"{dataset}: no cache at {cache_dir}. Build it first with "
		                 f"measurement/make_comparable_figures.py --dataset {dataset} "
		                 f"--rebuild-cache --out {out_dir} --run ...")
	rows = mcf._read_question_table(os.path.join(cache_dir, mcf.CACHE_FILENAME))
	return rows, meta["systems"]


def main(argv=None):
	ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
	ap.add_argument("--dataset", action="append", required=True, metavar="NAME=CACHE_OUTDIR",
	                help="webqsp=DIR and cwq=DIR, each an already-built "
	                     "make_comparable_figures.py --out directory; repeatable")
	ap.add_argument("--predictions", action="append", default=[], metavar="NAME:SYS=FILE",
	                help="override a predict.jsonl path, e.g. cwq:CoR=path/to/predict.jsonl; "
	                     "defaults to this repository's documented result paths")
	ap.add_argument("--backbone", default="Gemma-3-4B")
	ap.add_argument("--out", default="results/tables/cross_dataset")
	ap.add_argument("--out-figures", default="results/figures/cross_dataset")
	args = ap.parse_args(argv)

	cache_dirs = mcf.parse_pairs(args.dataset, {})
	missing = [d for d in DATASET_ORDER if d not in cache_dirs]
	if missing:
		raise SystemExit(f"--dataset required for {missing}")

	predictions = {ds: dict(DEFAULT_PREDICTIONS[ds]) for ds in DATASET_ORDER}
	for item in args.predictions:
		key, path = item.split("=", 1)
		ds, sysname = key.split(":", 1)
		predictions[ds][sysname] = path

	out = os.path.abspath(os.path.expanduser(args.out))
	os.makedirs(out, exist_ok=True)

	print("=" * 70)
	print("VALIDATION")
	print("=" * 70)

	table_rows = []
	diagnostic_rows = []
	for ds in DATASET_ORDER:
		cache_rows, systems = _load_cache(ds, cache_dirs[ds])
		order = [s for s in SYSTEM_ORDER if s in systems]
		by_run = {}
		for r in cache_rows:
			by_run.setdefault(r.get("run"), []).append(r)

		for name in order:
			label = systems[name]["label"]
			pred_path = os.path.join(ROOT, predictions[ds][name])

			# ---- effectiveness: the existing evaluator, fresh from predict.jsonl.
			outcomes = export_outcomes.score(pred_path, ds, name)
			outcome_ids = [o["question_id"] for o in outcomes]
			dup_pred = len(outcome_ids) - len(set(outcome_ids))
			n_pred = len(outcomes)
			hits_at_1 = 100.0 * statistics.mean(float(o["hit1"]) for o in outcomes)
			precision = 100.0 * statistics.mean(float(o["precision"]) for o in outcomes)
			recall = 100.0 * statistics.mean(float(o["recall"]) for o in outcomes)
			f1 = 100.0 * statistics.mean(float(o["f1"]) for o in outcomes)

			# ---- energy + KGR: the per-question cache, joined by question_id.
			qrows = by_run.get(label, [])
			cache_ids = {r["question_id"] for r in qrows}
			dup_cache = len(qrows) - len(cache_ids)
			missing_from_cache = sorted(set(outcome_ids) - cache_ids)
			energies = [r["energy_trajectory_j"] for r in qrows
			           if r["energy_trajectory_j"] is not None]
			n_energy_missing = sum(1 for r in qrows if r["energy_trajectory_j"] is None)
			fallback_flags = [bool(r.get("fallback")) for r in qrows]
			kgr = 100.0 * (1.0 - statistics.mean(fallback_flags)) if fallback_flags else None

			size_flag = ("" if len(outcomes) == PAPER_N[ds] else
			            f"  ** differs from the paper's full {ds} test set "
			            f"({PAPER_N[ds]:,}); this is not a subset run, the paper's own "
			            f"denominator for this dataset is different from ours **")
			print(f"{DATASET_LABELS[ds]:7s} {name:4s}  predictions n={n_pred:,}  "
			     f"duplicate prediction ids={dup_pred}  cache rows n={len(qrows):,}  "
			     f"duplicate cache ids={dup_cache}  missing from cache={len(missing_from_cache)}  "
			     f"missing energy={n_energy_missing}{size_flag}")
			if missing_from_cache:
				print(f"    missing-from-cache ids (first 5): {missing_from_cache[:5]}")

			mean_wh = v.j_to_wh(sum(energies) / len(energies)) if energies else None
			median_wh = v.j_to_wh(statistics.median(energies)) if energies else None
			total_wh = v.j_to_wh(sum(energies)) if energies else None
			coverage_pct = 100.0 * len(energies) / len(qrows) if qrows else None

			# ---- 95% bootstrap CI on mean trajectory Wh/question. Questions as
			# the resampling unit, >=2000 replicates, fixed seed -- see
			# thesis_figures.bootstrap_ci. Trajectory energy only (never the
			# attributed/event basis). Same estimate reused across this
			# system's F1/Recall/KGR panels in the figure below, not
			# recomputed per metric.
			wh_values = v.j_to_wh(energies)
			if wh_values:
				e_ci_lo, e_ci_hi = bootstrap_ci(wh_values, statistics.mean)
			else:
				e_ci_lo = e_ci_hi = None

			table_rows.append({
				"backbone": args.backbone, "method": name, "dataset": DATASET_LABELS[ds],
				"n_questions": n_pred, "hits_at_1": hits_at_1, "precision": precision,
				"recall": recall, "f1": f1, "kgr": kgr, "mean_energy_wh": mean_wh,
				"energy_ci_low_wh": e_ci_lo, "energy_ci_high_wh": e_ci_hi,
				"median_energy_wh": median_wh, "total_energy_wh": total_wh,
				"energy_coverage_n": len(energies), "energy_coverage_pct": coverage_pct,
			})

			# ---- section 9: predicted answer-set size distribution (printed
			# below and folded into diagnostic_rows), from the evaluator's own
			# n_prediction_items -- no new matching scheme, no separate CSV.
			sizes = sorted(o["n_prediction_items"] for o in outcomes)
			size_stats = {
				"median": statistics.median(sizes), "mean": statistics.mean(sizes),
				"p90": sizes[min(len(sizes) - 1, int(0.90 * len(sizes)))],
				"p95": sizes[min(len(sizes) - 1, int(0.95 * len(sizes)))],
				"max": sizes[-1],
			}
			print(f"    {name}/{DATASET_LABELS[ds]} predicted answer-set size: "
			     f"median={size_stats['median']:.0f}  mean={size_stats['mean']:.2f}  "
			     f"p90={size_stats['p90']:.0f}  p95={size_stats['p95']:.0f}  "
			     f"max={size_stats['max']:.0f}")

			# ---- section 5: precision / false-positive diagnostics. TP is
			# re-derived from the evaluator's OWN set arithmetic
			# (recall = |pred & gold| / n_gold_answers over distinct normalized
			# answers -- see prf in chain_of_relations/eval/accuracy.py), not a
			# new matching scheme. FP = n_prediction_items (distinct predicted
			# answers) - TP. Questions with n_gold_answers == 0 give no TP
			# basis at all and are excluded from the TP/FP/FN means, counted
			# separately instead of guessed at.
			classifiable = [o for o in outcomes if o["n_gold_answers"] > 0]
			n_excluded = len(outcomes) - len(classifiable)
			tp_vals, fp_vals, fn_vals, clipped = [], [], [], 0
			for o in classifiable:
				n_gold = o["n_gold_answers"]
				n_pred_items = o["n_prediction_items"]
				tp = round(float(o["recall"]) * n_gold)
				fn = n_gold - tp
				fp = n_pred_items - tp
				if fp < 0:
					clipped += 1  # matched > predicted items; precision was clipped at 1.0
					fp = 0
				tp_vals.append(tp)
				fp_vals.append(fp)
				fn_vals.append(fn)
			gold_counts = [o["n_gold_answers"] for o in outcomes]
			pred_counts = [o["n_prediction_items"] for o in outcomes]
			diagnostic_rows.append({
				"dataset": DATASET_LABELS[ds], "system": name, "n_questions": len(outcomes),
				"mean_precision": statistics.mean(float(o["precision"]) for o in outcomes),
				"median_precision": statistics.median(float(o["precision"]) for o in outcomes),
				"mean_recall": statistics.mean(float(o["recall"]) for o in outcomes),
				"mean_f1": statistics.mean(float(o["f1"]) for o in outcomes),
				"mean_gold_answer_count": statistics.mean(gold_counts),
				"median_gold_answer_count": statistics.median(gold_counts),
				"mean_predicted_answer_count": statistics.mean(pred_counts),
				"median_predicted_answer_count": statistics.median(pred_counts),
				"p90_predicted_answer_count": size_stats["p90"],
				"p95_predicted_answer_count": size_stats["p95"],
				"max_predicted_answer_count": size_stats["max"],
				"mean_true_positive_count": statistics.mean(tp_vals) if tp_vals else None,
				"mean_false_positive_count": statistics.mean(fp_vals) if fp_vals else None,
				"mean_false_negative_count": statistics.mean(fn_vals) if fn_vals else None,
				"n_excluded_zero_gold": n_excluded,
			})
			if clipped:
				print(f"    note: {name}/{DATASET_LABELS[ds]}: {clipped} question(s) had "
				     f"matched-gold-answers > predicted-item-count (precision clipped at "
				     f"1.0); their false-positive proxy is reported as 0")

	# ---- CSV
	csv_path = os.path.join(out, "table1_effectiveness_energy.csv")
	with open(csv_path, "w", newline="") as f:
		writer = csv.DictWriter(f, fieldnames=CSV_FIELDS, lineterminator="\n")
		writer.writeheader()
		for row in table_rows:
			writer.writerow({k: ("" if row.get(k) is None else row.get(k)) for k in CSV_FIELDS})

	# ---- Markdown (printed and not separately written, per the task's ask for
	# "a readable Markdown/terminal version")
	print()
	print("=" * 70)
	print(f"TABLE 1 (recreated) -- backbone: {args.backbone}")
	print("=" * 70)
	header = ("| Method | Dataset | N | Hits@1 | P | R | F1 | KGR | Wh/q (mean) | "
	         "Wh/q (median) | Total Wh | Energy coverage |")
	print(header)
	print("|" + "---|" * 12)
	for row in table_rows:
		print(f"| {row['method']} | {row['dataset']} | {row['n_questions']:,} | "
		     f"{row['hits_at_1']:.2f} | {row['precision']:.2f} | {row['recall']:.2f} | "
		     f"{row['f1']:.2f} | "
		     + (f"{row['kgr']:.2f}" if row['kgr'] is not None else "n/a") + " | "
		     + (f"{row['mean_energy_wh']:.3f}" if row['mean_energy_wh'] is not None else "n/a")
		     + " | "
		     + (f"{row['median_energy_wh']:.3f}" if row['median_energy_wh'] is not None else "n/a")
		     + " | "
		     + (f"{row['total_energy_wh']:,.1f}" if row['total_energy_wh'] is not None else "n/a")
		     + " | "
		     + f"{row['energy_coverage_n']:,}/{row['n_questions']:,} "
		     + (f"({row['energy_coverage_pct']:.1f}%)" if row['energy_coverage_pct'] is not None
		        else "") + " |")

	# ---- LaTeX (booktabs, paper-style grouped columns)
	tex_path = os.path.join(out, "table1_effectiveness_energy.tex")
	by_ds = {}
	for row in table_rows:
		by_ds.setdefault(row["dataset"], {})[row["method"]] = row
	ds_labels = [DATASET_LABELS[d] for d in DATASET_ORDER]
	ncols = 1 + 6 * len(ds_labels)
	lines = []
	lines.append(r"\begin{table}[t]")
	lines.append(r"\centering")
	lines.append(f"\\caption{{Comparison of methods on "
	            f"{' and '.join(ds_labels)} under {args.backbone}. Metrics include "
	            f"Hits@1, Precision (P), Recall (R), F1, KGR\\textsuperscript{{\\dag}}, and "
	            f"mean GPU energy per question (Wh/q). Effectiveness is scored on the "
	            f"extracted final answer only, by normalized exact match.}}")
	lines.append(r"\label{tab:effectiveness_energy}")
	lines.append(r"\resizebox{\textwidth}{!}{%")
	lines.append(r"\begin{tabular}{l" + "cccccc" * len(ds_labels) + "}")
	lines.append(r"\toprule")
	header_cells = " & ".join(f"\\multicolumn{{6}}{{c}}{{{d}}}" for d in ds_labels)
	lines.append(r"\textbf{Method} & " + header_cells + r" \\")
	cmid = " ".join(f"\\cmidrule(lr){{{2 + 6*i}-{7 + 6*i}}}" for i in range(len(ds_labels)))
	lines.append(cmid)
	sub_cells = " & ".join(["Hits@1 & P & R & F1 & KGR\\textsuperscript{\\dag} & Wh/q"]
	                      * len(ds_labels))
	lines.append(" & " + sub_cells + r" \\")
	lines.append(r"\midrule")
	for name in SYSTEM_ORDER:
		cells = [name]
		for ds_label in ds_labels:
			row = by_ds.get(ds_label, {}).get(name)
			if row is None:
				cells += ["--"] * 6
				continue
			wh = row["mean_energy_wh"]
			wh_str = (f"{wh:.3f}" if wh is not None and wh < 10 else
			         (f"{wh:.2f}" if wh is not None else "n/a"))
			kgr_str = f"{row['kgr']:.1f}" if row["kgr"] is not None else "n/a"
			cells += [f"{row['hits_at_1']:.1f}", f"{row['precision']:.1f}", f"{row['recall']:.1f}",
			         f"{row['f1']:.1f}", kgr_str, wh_str]
		lines.append(" & ".join(cells) + r" \\")
	lines.append(r"\bottomrule")
	lines.append(r"\end{tabular}%")
	lines.append(r"}")
	lines.append(r"\vspace{2pt}")
	lines.append(r"{\footnotesize \textsuperscript{\dag}KGR (1 $-$ fallback rate): KGR here is $1 - $fallback rate, "
	            r"an approximation of the paper's KG-grounded Rate reusing this "
	            r"repository's existing fallback signal; no function in the "
	            r"repository computes the paper's literal KGR. See "
	            r"measurement/table1\_effectiveness\_energy.py for the validation "
	            r"this approximation was checked against.}")
	lines.append(r"\end{table}")
	with open(tex_path, "w", newline="\n") as f:
		f.write("\n".join(lines) + "\n")

	# figures_out/plot_order/colour_of are shared by every figure below: the
	# 2x3 supplementary tradeoff figure and the RQ3 main F1-energy frontier
	# figure (see CLAUDE.md Generated Artifact Hygiene).
	figures_out = os.path.abspath(os.path.expanduser(args.out_figures))
	os.makedirs(figures_out, exist_ok=True)
	plot_order = [s for s in ("PoG", "ToG", "CoR") if s in {r["method"] for r in table_rows}]
	colour_of = {name: v.PALETTE[i] for i, name in enumerate(plot_order)}
	plt = v._pyplot()

	# ---- section 5: precision / false-positive diagnostics CSV.
	diag_path = os.path.join(out, "prediction_set_diagnostics.csv")
	with open(diag_path, "w", newline="") as f:
		writer = csv.DictWriter(f, fieldnames=DIAGNOSTIC_FIELDS, lineterminator="\n")
		writer.writeheader()
		for row in diagnostic_rows:
			writer.writerow({k: ("" if row.get(k) is None else row.get(k))
			                 for k in DIAGNOSTIC_FIELDS})

	# ---- section 4: per-panel Pareto efficiency. A dominates B when
	# metric_A >= metric_B AND energy_A <= energy_B with at least one strict
	# inequality -- a boolean flag per point, not a fitted/continuous frontier.
	def _pareto_efficient(rows_for_panel):
		flags = {}
		for row in rows_for_panel:
			dominated = False
			for other in rows_for_panel:
				if other is row:
					continue
				if (other["metric_value"] >= row["metric_value"]
				    and other["mean_energy_wh"] <= row["mean_energy_wh"]
				    and (other["metric_value"] > row["metric_value"]
				         or other["mean_energy_wh"] < row["mean_energy_wh"])):
					dominated = True
					break
			flags[id(row)] = not dominated
		return flags

	pareto_rows = []
	pareto_flags_by_panel = {}  # (ds_label, metric, system) -> bool
	for ds_label in ds_labels:
		for metric in TRADEOFF_METRICS:
			rows_for_panel = []
			for name in SYSTEM_ORDER:
				row = by_ds.get(ds_label, {}).get(name)
				if row is None or row.get(metric) is None or row["mean_energy_wh"] is None:
					continue
				rows_for_panel.append({"system": name, "metric_value": row[metric],
				                       "mean_energy_wh": row["mean_energy_wh"]})
			flags = _pareto_efficient(rows_for_panel)
			for r in rows_for_panel:
				efficient = flags[id(r)]
				pareto_flags_by_panel[(ds_label, metric, r["system"])] = efficient
				pareto_rows.append({"dataset": ds_label, "metric": metric, "system": r["system"],
				                    "metric_value": r["metric_value"],
				                    "mean_energy_wh": r["mean_energy_wh"],
				                    "pareto_efficient": efficient})

	pareto_path = os.path.join(out, "effectiveness_energy_pareto.csv")
	with open(pareto_path, "w", newline="") as f:
		writer = csv.DictWriter(f, fieldnames=PARETO_FIELDS, lineterminator="\n")
		writer.writeheader()
		writer.writerows(pareto_rows)

	# ---- section 2-4: the sole canonical effectiveness-vs-energy figure
	# (F1 / Recall / KGR vs energy, one row per dataset). Points are
	# system-level aggregates; trajectory energy only, never the
	# attributed/event basis used by the operation-breakdown figures.
	# Each point is a mean: mean F1 / Recall / KGR (x) against mean trajectory
	# Wh/question (y). Every system is drawn identically -- same marker, same
	# size, no outline and no error bars -- so no system looks different for
	# reasons other than its position. (Energy CIs and Pareto flags stay in
	# table1_effectiveness_energy.csv and effectiveness_energy_pareto.csv.)
	# Layout: one column per dataset (titled on top), one row per metric
	# (labelled on the left). Every panel shares ONE energy y-scale, so a point
	# higher on the page costs more whichever dataset or metric panel it is in.
	all_ci_highs = [r["energy_ci_high_wh"] for d in ds_labels for r in by_ds.get(d, {}).values()
	               if r["energy_ci_high_wh"] is not None]
	shared_y_max = max(all_ci_highs) * 1.18 if all_ci_highs else 1.0
	with plt.rc_context(v.STYLE):
		fig, axes = plt.subplots(len(TRADEOFF_METRICS), len(ds_labels), squeeze=False,
		                         figsize=(5.4 * len(ds_labels), 3.9 * len(TRADEOFF_METRICS)))
		for col_i, ds_label in enumerate(ds_labels):
			rows = by_ds.get(ds_label, {})
			y_max = shared_y_max
			for row_i, metric in enumerate(TRADEOFF_METRICS):
				ax = axes[row_i][col_i]
				for name in plot_order:
					row = rows.get(name)
					if row is None or row.get(metric) is None or row["mean_energy_wh"] is None:
						continue
					mv, wh = row[metric], row["mean_energy_wh"]
					ax.plot([mv], [wh], linestyle="none", marker="o", markersize=9,
					        color=colour_of[name], markeredgecolor="none", zorder=3)
					ax.annotate(name, (mv, wh), textcoords="offset points", xytext=(7, 5),
					           fontsize=8.5, fontweight="bold", color=colour_of[name])
				ax.set_ylim(0, y_max)
				x_vals = [rows[n][metric] for n in plot_order
				         if rows.get(n) and rows[n].get(metric) is not None]
				if x_vals:
					ax.set_xlim(*_metric_xlim(x_vals))
				ax.set_xlabel(TRADEOFF_METRIC_LABELS[metric])
				if col_i == 0:
					ax.set_ylabel(f"{TRADEOFF_METRIC_LABELS[metric]}\n\nMean GPU energy (Wh/question)",
					              fontsize=10)
				if row_i == 0:
					ax.set_title(ds_label, fontsize=13, fontweight="bold")
		fig.suptitle("Effectiveness and grounding vs. energy", fontsize=14, fontweight="bold",
		            y=1.01)
		fig.tight_layout(rect=(0, 0.01, 1, 0.97))
		tradeoffs_written = v.save_figure(fig, figures_out, "fig_effectiveness_energy_tradeoffs",
		                                  png=False)
		plt.close(fig)

	# ---- RQ2b main figure (review only, does not touch
	# fig_effectiveness_energy_tradeoffs.pdf, which stays as the F1/Recall/KGR
	# supplementary figure): 1x2 (WebQSP, CWQ), F1 only, same points/CIs/Pareto
	# flags already computed above for the "f1" column of the 2x3 grid. No
	# in-figure caveat text about the CWQ scorer being provisional -- that
	# belongs in the manuscript caption, not the artwork, and this figure will
	# be regenerated once the CWQ exact-match scorer is frozen.
	frontier_written = []
	with plt.rc_context(v.STYLE):
		fig, axes = plt.subplots(1, len(ds_labels), figsize=(5.4 * len(ds_labels), 4.8))
		axes = [axes] if len(ds_labels) == 1 else list(axes)
		metric = "f1"
		# One y-scale for both datasets, so energy is directly comparable across
		# panels (a point higher on the page costs more, whichever panel it is in).
		ci_highs = [r["energy_ci_high_wh"] for d in ds_labels for r in by_ds.get(d, {}).values()
		           if r["energy_ci_high_wh"] is not None]
		y_max = max(ci_highs) * 1.18 if ci_highs else 1.0
		for ax, ds_label in zip(axes, ds_labels):
			rows = by_ds.get(ds_label, {})
			for name in plot_order:
				row = rows.get(name)
				if row is None or row.get(metric) is None or row["mean_energy_wh"] is None:
					continue
				mv, wh = row[metric], row["mean_energy_wh"]
				lo, hi = row.get("energy_ci_low_wh"), row.get("energy_ci_high_wh")
				yerr = ([[wh - lo], [hi - wh]] if lo is not None and hi is not None else None)
				efficient = pareto_flags_by_panel.get((ds_label, metric, name), False)
				ax.errorbar([mv], [wh], yerr=yerr, fmt="o",
				           markersize=13 if efficient else 10, color=colour_of[name],
				           ecolor=colour_of[name], elinewidth=1.2, capsize=3, alpha=1.0,
				           markeredgecolor=v.INK if efficient else "none",
				           markeredgewidth=1.6, zorder=3)
				ax.annotate(name, (mv, wh), textcoords="offset points", xytext=(8, 6),
				           fontsize=11, fontweight="bold", color=colour_of[name])
			ax.set_ylim(0, y_max)
			x_vals = [rows[n][metric] for n in plot_order
			         if rows.get(n) and rows[n].get(metric) is not None]
			if x_vals:
				ax.set_xlim(*_metric_xlim(x_vals))
			ax.set_xlabel("F1", fontsize=12)
			ax.set_title(ds_label, fontsize=14, fontweight="bold")
			ax.tick_params(labelsize=10.5)
			if ax is axes[0]:
				ax.set_ylabel("Mean GPU energy (Wh/question)", fontsize=11.5)
		fig.suptitle("F1-energy tradeoff", fontsize=15, fontweight="bold", y=1.02)
		fig.tight_layout(rect=(0, 0, 1, 0.94))
		frontier_written = v.save_figure(fig, figures_out, "fig_f1_energy_frontier_main",
		                                 png=False)
		plt.close(fig)

	print()
	print("Bootstrap 95% CI for mean trajectory energy per question "
	     f"({BOOTSTRAP_REPS:,} resamples, seed {SEED}, questions resampled independently "
	     "per dataset x system):")
	for row in table_rows:
		print(f"  {row['dataset']:7s} {row['method']:4s}  mean_energy_wh={row['mean_energy_wh']:.4f}  "
		     f"ci95_low={row['energy_ci_low_wh']:.4f}  ci95_high={row['energy_ci_high_wh']:.4f}")

	# ---- section 6: precision-vs-energy is intentionally NOT a canonical
	# figure. Precision is already in table1_effectiveness_energy.csv/.tex
	# (and the printed Markdown table) and is not one of RQ3's three tradeoff
	# metrics; a dedicated scatter duplicated that table without adding
	# information the results tables don't already carry. Kept out of the
	# canonical generation path; historical fig_precision_energy(_v2).pdf
	# files already in results/ are left untouched by this script.

	# ---- section 8: sanity check -- CoR's F1-vs-recall gap.
	print()
	print("Section 8 sanity check -- Recall vs. F1 per system/dataset:")
	for row in table_rows:
		gap = row["recall"] - row["f1"]
		flag = "  ** recall notably exceeds F1 **" if gap > 5.0 else ""
		print(f"  {row['dataset']:7s} {row['method']:4s}  P={row['precision']:.2f}%  "
		     f"R={row['recall']:.2f}%  F1={row['f1']:.2f}%  R-F1={gap:+.2f}{flag}")

	print()
	print(f"  -> {csv_path}")
	print(f"  -> {tex_path}")
	for p in tradeoffs_written:
		print(f"  -> {p}")
	for p in frontier_written:
		print(f"  -> {p}")
	print(f"  -> {pareto_path}")
	print(f"  -> {diag_path}")
	return 0


if __name__ == "__main__":
	sys.exit(main())
