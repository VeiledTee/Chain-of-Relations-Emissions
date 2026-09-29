"""Two cross-dataset thesis figures built on top of the existing per-question
cache (measurement/make_comparable_figures.py) -- nothing here re-parses
events_attributed.jsonl or re-runs any experiment.

    python measurement/thesis_figures.py \
        --dataset webqsp=~/cwq_supervisor_summary/cross_dataset/webqsp/wh \
        --dataset cwq=~/cwq_supervisor_summary/wh \
        --out-figures results/figures/cross_dataset \
        --out-tables results/tables/cross_dataset

The aggregate effectiveness-vs-energy tradeoff figure (F1/Recall/KGR x
WebQSP/CWQ, with bootstrap energy CIs and Pareto flags) lives in
measurement/table1_effectiveness_energy.py, not here -- it shares that
script's Table 1 data and evaluator calls, so it is built alongside the table
rather than duplicated. This script owns the two question-level diagnostics:

Figure 2 (fig_question_f1_vs_energy): 2x3 grid (dataset x system), per-question
F1 (continuous, unbinned, real values, 0-100%) vs whole-question trajectory
energy (Wh), hexbin density + a local-median trend line (statsmodels is not
installed in this environment, so this stands in for LOWESS; it is descriptive
only, not a fitted model) + Spearman rho per panel.

Figure 3 (fig_operation_energy_breakdown): 2x2 grid (absolute / normalized x
dataset), mean ATTRIBUTED per-operation energy per question, stacked by the
canonical operation label (chain_of_relations/energy_taxonomy.py declaration
order -- not invented, not magnitude-sorted). This is the attributed/event
energy basis (same as fig1/fig5 in make_comparable_figures.py), which is
narrower than and NOT summed with the trajectory energy used in the
effectiveness-vs-energy figure or Figure 2; operation_energy_coverage.csv
reports the gap between the two bases per (dataset, system) instead of
silently reconciling it.

Figure 4 (energy distribution violins) is deliberately not produced: the
effectiveness-vs-energy figure already reports the aggregate mean + 95% CI per
system, and Figure 2 already shows the full per-question distribution via
hexbin. A marginal violin/box of the same trajectory-energy column would
repeat information already in those two figures without adding a new
question-level or operation-level view.

Writes only into --out-figures / --out-tables:
    fig_question_f1_vs_energy.pdf
    fig_question_f1_vs_energy_v2.pdf   (review-only cleanup pass; see draw_fig2_v2)
    question_f1_energy_correlations.csv
    fig_operation_energy_breakdown.pdf
    fig_operation_energy_shares_main.pdf  (RQ1 main figure; see draw_fig3_main)
    operation_energy_coverage.csv
    fig_trajectory_cost_drivers_main.pdf  (RQ2 supplementary figure: combined
                                            WebQSP/CWQ LLM-calls vs energy; see
                                            make_comparable_figures.draw_trajectory_cost_drivers_main)
    fig_trajectory_correlation_heatmap_main.pdf  (RQ2 main figure: Spearman
                                            rho of output tokens/input tokens/
                                            LLM calls/traversal depth vs energy,
                                            by system and dataset; read
                                            verbatim from the existing per-
                                            system correlation tables, no
                                            recomputation; see
                                            draw_trajectory_correlation_heatmap)
"""

import argparse
import csv
import os
import random
import statistics
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)
sys.path.insert(0, HERE)

import make_comparable_figures as mcf  # noqa: E402
from agent_energy_profiler import visualize as v  # noqa: E402
from chain_of_relations import energy_taxonomy as tax  # noqa: E402

try:
	from scipy import stats as _scipy_stats
except ImportError:  # pragma: no cover -- scipy is present in this environment
	_scipy_stats = None

DATASET_ORDER = ("webqsp", "cwq")
DATASET_LABELS = {"webqsp": "WebQSP", "cwq": "CWQ"}
SYSTEM_ORDER = ("PoG", "ToG", "CoR")
BOOTSTRAP_REPS = 2000
SEED = 20260922

#: Canonical operation-label order for Figure 3: the taxonomy's own
#: declaration order (llm stages first, then kg, then embedding/system),
#: filtered down at draw time to whatever labels the data actually has.
#: Same order, same colour, in every panel.
LABEL_ORDER = [member.value for member in tax.OperationLabel]


def _load_dataset(ds, out_dir):
	cache_dir = os.path.join(os.path.expanduser(out_dir), mcf.CACHE_DIRNAME)
	meta = mcf._cache_manifest(cache_dir)
	if meta is None:
		raise SystemExit(f"{ds}: no cache at {cache_dir}. Build it first with "
		                 f"make_comparable_figures.py --dataset {ds} --rebuild-cache "
		                 f"--out {out_dir} --run ...")
	rows = mcf._read_question_table(os.path.join(cache_dir, mcf.CACHE_FILENAME))
	systems = meta["systems"]
	by_run = {}
	for r in rows:
		by_run.setdefault(r.get("run"), []).append(r)
	order = [s for s in SYSTEM_ORDER if s in systems]
	qrows_by_system = {name: by_run.get(systems[name]["label"], []) for name in order}
	return order, qrows_by_system


def bootstrap_ci(values, stat, reps=BOOTSTRAP_REPS, seed=SEED):
	"""Generic 95% percentile bootstrap CI, questions as the resampling unit.
	Reused (imported) by measurement/table1_effectiveness_energy.py for the
	canonical effectiveness-vs-energy figure's energy CIs -- kept here as the
	one definition rather than duplicated."""
	rng = random.Random(seed)
	n = len(values)
	draws = sorted(stat([values[rng.randrange(n)] for _ in range(n)]) for _ in range(reps))
	return draws[int(0.025 * reps)], draws[int(0.975 * reps) - 1]


# ---------------------------------------------------------------------------
# Figure 2: question-level F1 vs energy
# ---------------------------------------------------------------------------

FIG2_FIELDS = ("dataset", "system", "n", "spearman_rho", "p_value", "mean_f1",
              "mean_energy_wh", "median_energy_wh")


def _spearman(f1, energy):
	if _scipy_stats is not None:
		rho, p = _scipy_stats.spearmanr(f1, energy)
		return float(rho), float(p)
	# Fallback: rank correlation without scipy (not exercised in this
	# environment, kept only so the script degrades rather than crashes).
	def rank(xs):
		order = sorted(range(len(xs)), key=lambda i: xs[i])
		ranks = [0.0] * len(xs)
		for pos, i in enumerate(order):
			ranks[i] = pos
		return ranks
	rf, re = rank(f1), rank(energy)
	n = len(f1)
	mrf, mre = statistics.mean(rf), statistics.mean(re)
	cov = sum((a - mrf) * (b - mre) for a, b in zip(rf, re))
	sdf = (sum((a - mrf) ** 2 for a in rf)) ** 0.5
	sde = (sum((b - mre) ** 2 for b in re)) ** 0.5
	rho = cov / (sdf * sde) if sdf and sde else 0.0
	return rho, float("nan")


def _local_median_trend(x, y, n_bins=20):
	"""Descriptive local-median trend line, standing in for LOWESS (statsmodels
	is not installed here). Bins are equal-width in x; a bin is only plotted
	with >=5 points so sparse tails don't produce a noisy trend."""
	if not x:
		return [], []
	lo, hi = min(x), max(x)
	if hi <= lo:
		return [], []
	width = (hi - lo) / n_bins
	buckets = [[] for _ in range(n_bins)]
	for xv, yv in zip(x, y):
		idx = min(n_bins - 1, int((xv - lo) / width))
		buckets[idx].append(yv)
	bx, by = [], []
	for i, ys in enumerate(buckets):
		if len(ys) >= 5:
			bx.append(lo + width * (i + 0.5))
			by.append(statistics.median(ys))
	return bx, by


def draw_fig2_v2(plt, per_panel, corr_rows_by_key, use_log):
	"""Cleanup pass over draw_fig2 for review as fig_question_f1_vs_energy_v2:
	drop the descriptive local-median trend line (visually implied structure
	that is not present for several system/dataset pairs), boost hexbin
	density contrast, and use larger fonts. Y-limits are still shared only
	WITHIN a dataset row (across its 3 systems) -- WebQSP and CWQ keep their
	own ranges, as requested, so WebQSP is not needlessly compressed to CWQ's
	wider spread."""
	fig, axes = plt.subplots(len(DATASET_ORDER), len(SYSTEM_ORDER),
	                         figsize=(4.6 * len(SYSTEM_ORDER), 4.2 * len(DATASET_ORDER)))
	for row, ds in enumerate(DATASET_ORDER):
		order = [s for s in SYSTEM_ORDER if (ds, s) in per_panel]
		all_e = [e for s in order for e in per_panel[(ds, s)][1]]
		y_lo = max(min(all_e) * 0.9, 1e-3) if use_log else 0.0
		y_hi = max(all_e) * 1.08
		for col, name in enumerate(SYSTEM_ORDER):
			ax = axes[row][col] if len(DATASET_ORDER) > 1 else axes[col]
			if (ds, name) not in per_panel:
				ax.axis("off")
				continue
			f1, energy = per_panel[(ds, name)]
			ax.hexbin(f1, energy, gridsize=32, cmap="viridis", mincnt=1, bins="log",
			         yscale="log" if use_log else "linear", xscale="linear", linewidths=0.15)
			corr = corr_rows_by_key[(ds, name)]
			ax.annotate(f"ρ = {corr['spearman_rho']:.2f}\nn = {corr['n']:,}",
			            xy=(0.03, 0.95), xycoords="axes fraction", fontsize=10.5,
			            fontweight="bold", va="top", ha="left", color=v.INK,
			            bbox=dict(boxstyle="round,pad=0.28", facecolor=v.SURFACE,
			                     edgecolor=v.GRID, linewidth=0.6))
			ax.set_xlim(0, 100)
			ax.set_ylim(y_lo, y_hi)
			if use_log:
				ax.set_yscale("log")
			if row == 0:
				ax.set_title(name, fontsize=12.5, fontweight="bold")
			if col == 0:
				ax.set_ylabel(f"{DATASET_LABELS[ds]}\nEnergy per question (Wh)"
				             + (" [log]" if use_log else ""), fontsize=10.5)
			if row == len(DATASET_ORDER) - 1:
				ax.set_xlabel("Per-question F1 (%)", fontsize=10.5)
			ax.tick_params(labelsize=9.5)
	fig.suptitle("Per-question F1 vs. trajectory energy", fontsize=13, fontweight="bold", y=1.02)
	fig.tight_layout(rect=(0, 0, 1, 0.96))
	return fig


def draw_fig2(plt, per_panel, corr_rows_by_key, use_log):
	fig, axes = plt.subplots(len(DATASET_ORDER), len(SYSTEM_ORDER),
	                         figsize=(4.4 * len(SYSTEM_ORDER), 4.0 * len(DATASET_ORDER)))
	for row, ds in enumerate(DATASET_ORDER):
		order = [s for s in SYSTEM_ORDER if (ds, s) in per_panel]
		# shared y-limits within the dataset row
		all_e = [e for s in order for e in per_panel[(ds, s)][1]]
		y_lo = max(min(all_e) * 0.9, 1e-3) if use_log else 0.0
		y_hi = max(all_e) * 1.08
		for col, name in enumerate(SYSTEM_ORDER):
			ax = axes[row][col] if len(DATASET_ORDER) > 1 else axes[col]
			if (ds, name) not in per_panel:
				ax.axis("off")
				continue
			f1, energy = per_panel[(ds, name)]
			hb = ax.hexbin(f1, energy, gridsize=32, cmap="Blues", mincnt=1,
			               yscale="log" if use_log else "linear",
			               xscale="linear", linewidths=0.15)
			bx, by = _local_median_trend(f1, energy)
			if bx:
				ax.plot(bx, by, color=v.PALETTE[1], linewidth=1.8, zorder=4,
				        label="Local median trend")
			corr = corr_rows_by_key[(ds, name)]
			ax.annotate(f"ρ = {corr['spearman_rho']:.2f}\nn = {corr['n']:,}",
			            xy=(0.03, 0.95), xycoords="axes fraction", fontsize=8.5,
			            va="top", ha="left", color=v.INK,
			            bbox=dict(boxstyle="round,pad=0.25", facecolor=v.SURFACE,
			                     edgecolor=v.GRID, linewidth=0.6))
			ax.set_xlim(0, 100)
			ax.set_ylim(y_lo, y_hi)
			if use_log:
				ax.set_yscale("log")
			if row == 0:
				ax.set_title(name, fontsize=10.5, fontweight="bold")
			if col == 0:
				ax.set_ylabel(f"{DATASET_LABELS[ds]}\nEnergy per question (Wh)"
				             + (" [log]" if use_log else ""), fontsize=9)
			if row == len(DATASET_ORDER) - 1:
				ax.set_xlabel("Per-question F1 (%)")
	fig.suptitle("Question-level F1 vs. whole-question trajectory energy "
	            "(hexbin = point density; trend is a descriptive local median, "
	            "not a fitted model)", fontsize=10.5, fontweight="bold", y=1.02)
	fig.tight_layout(rect=(0, 0, 1, 0.96))
	return fig


# ---------------------------------------------------------------------------
# Figure 3: operation energy breakdown (attributed basis)
# ---------------------------------------------------------------------------

COVERAGE_FIELDS = ("dataset", "system", "n", "mean_trajectory_wh", "mean_attributed_wh",
                   "mean_attribution_fraction", "median_attribution_fraction")


def _op_energy_means(qrows):
	"""{label: mean attributed Wh/question} + coverage stats for one (ds, system)."""
	label_cols = None
	sums = {}
	counts_present = {}
	for r in qrows:
		for k, val in r.items():
			if not k.startswith("energy_") or k == "energy_trajectory_j":
				continue
			label = k[len("energy_"):]
			if val is None:
				continue
			sums[label] = sums.get(label, 0.0) + val
	n = len(qrows)
	means_wh = {label: v.j_to_wh(total / n) for label, total in sums.items()}

	fractions = []
	attributed_wh_per_q = []
	trajectory_wh_per_q = []
	for r in qrows:
		traj = r.get("energy_trajectory_j")
		attributed = sum(v for k, v in r.items()
		                 if k.startswith("energy_") and k != "energy_trajectory_j"
		                 and v is not None)
		if traj is not None:
			trajectory_wh_per_q.append(v.j_to_wh(traj))
			attributed_wh_per_q.append(v.j_to_wh(attributed))
			if traj > 0:
				fractions.append(attributed / traj)
	coverage = {
		"n": n,
		"mean_trajectory_wh": statistics.mean(trajectory_wh_per_q) if trajectory_wh_per_q else None,
		"mean_attributed_wh": statistics.mean(attributed_wh_per_q) if attributed_wh_per_q else None,
		"mean_attribution_fraction": statistics.mean(fractions) if fractions else None,
		"median_attribution_fraction": statistics.median(fractions) if fractions else None,
	}
	return means_wh, coverage


#: Legend text for the main (RQ2a) figure -- canonical taxonomy label -> short
#: display name, no "llm:"/"kg:"/"embedding:" prefix. Any label present in the
#: data but missing here (e.g. legacy llm:generate) falls back to a generic
#: strip-prefix/underscore transform in _display_label, not a raised error.
SIMPLE_LABELS = {
	"llm:relation_rank": "Relation ranking",
	"llm:reason": "Reasoning",
	"llm:answer_filter": "Answer filter",
	"llm:direct_answer": "Direct answer",
	"llm:entity_prune": "Entity prune",
	"llm:memory_update": "Memory update",
	"llm:subquestion_decompose": "Subquestion decomposition",
	"llm:reverse_retrieval_decision": "Reverse retrieval",
	"llm:reverse_entity_select": "Reverse entity select",
	"embedding:prune": "Embedding prune",
	"kg:id2name": "KG id→name",
	"kg:relation_search": "KG relation search",
	"kg:entity_search": "KG entity search",
	"kg:sparql": "KG SPARQL",
	"system:orchestration": "Orchestration",
}


def _display_label(label):
	if label in SIMPLE_LABELS:
		return SIMPLE_LABELS[label]
	return label.split(":", 1)[-1].replace("_", " ").capitalize()


#: Fixed threshold for the main operation-shares figure's "Other" buckets: a
#: label is kept on its own if it reaches this share in at least ONE of the
#: (dataset, system) panels; the rule is evaluated globally across all six
#: panels, not chosen per-panel, so the same set of kept labels appears (or is
#: legitimately absent/zero) in every panel. Below-threshold labels are split
#: into two buckets by operation type (not merged into one generic "Other"),
#: so the figure keeps the RQ1 LLM-vs-KG distinction even in the residual:
#: llm:* -> "Other LLM", kg:*/embedding:* -> "KG / embedding".
OTHER_THRESHOLD_PCT = 5.0
OTHER_LLM_LABEL = "Other LLM"
OTHER_KG_LABEL = "KG / embedding"
OTHER_LLM_COLOUR = "#b7b3a8"
OTHER_KG_COLOUR = "#6e6a5f"


def _other_bucket(label):
	"""llm:* below threshold -> Other LLM; kg:*/embedding:* -> KG / embedding."""
	return OTHER_LLM_LABEL if label.split(":", 1)[0] == "llm" else OTHER_KG_LABEL


def _classify_for_main(means_by_key):
	"""Return (kept_labels_in_taxonomy_order, other_llm_labels, other_kg_labels,
	max_share_by_label) using the >=5%-in-any-panel rule, computed once over
	every (dataset, system) panel in means_by_key -- never decided separately
	per panel."""
	max_share = {}
	for means in means_by_key.values():
		total = sum(means.values()) or 1.0
		for label, wh in means.items():
			share = 100.0 * wh / total
			if share > max_share.get(label, 0.0):
				max_share[label] = share
	kept = {lbl for lbl, s in max_share.items() if s >= OTHER_THRESHOLD_PCT}
	other_llm = {lbl for lbl in max_share if lbl not in kept and _other_bucket(lbl) == OTHER_LLM_LABEL}
	other_kg = {lbl for lbl in max_share if lbl not in kept and _other_bucket(lbl) == OTHER_KG_LABEL}
	kept_order = [lbl for lbl in LABEL_ORDER if lbl in kept]
	kept_order += sorted(kept - set(kept_order))  # unexpected labels, kept visible
	return kept_order, other_llm, other_kg, max_share


def draw_fig3_main(plt, means_by_key, kept_order, other_llm_labels, other_kg_labels, colour_of):
	"""Main RQ1 figure: 1x2 (WebQSP, CWQ), one 100% stacked bar per system,
	share of attributed GPU energy only (no absolute-Wh panel -- that stays in
	the untouched supplementary fig_operation_energy_breakdown.pdf). Residual
	below-threshold labels are kept as two buckets, not merged into one
	generic "Other", so the LLM-vs-KG distinction survives into the residual:
	"Other LLM" and "KG / embedding" are always the last two stack segments
	and legend entries, in that order, whenever either is non-empty."""
	fig, axes = plt.subplots(1, len(DATASET_ORDER), figsize=(5.8 * len(DATASET_ORDER), 5.6))
	axes = [axes] if len(DATASET_ORDER) == 1 else list(axes)
	other_specs = []
	if other_llm_labels:
		other_specs.append((OTHER_LLM_LABEL, other_llm_labels, OTHER_LLM_COLOUR))
	if other_kg_labels:
		other_specs.append((OTHER_KG_LABEL, other_kg_labels, OTHER_KG_COLOUR))
	legend_order = list(kept_order) + [spec[0] for spec in other_specs]
	other_members = {spec[0]: spec[1] for spec in other_specs}
	other_colours = {spec[0]: spec[2] for spec in other_specs}
	handles_by_label = {}
	for ax, ds in zip(axes, DATASET_ORDER):
		systems = [s for s in SYSTEM_ORDER if (ds, s) in means_by_key]
		x = list(range(len(systems)))
		bottom = [0.0] * len(systems)
		for label in legend_order:
			if label in other_members:
				members = other_members[label]
				vals = [100.0 * sum(means_by_key[(ds, s)].get(l, 0.0) for l in members)
				       / (sum(means_by_key[(ds, s)].values()) or 1.0) for s in systems]
				colour = other_colours[label]
			else:
				vals = [100.0 * means_by_key[(ds, s)].get(label, 0.0)
				       / (sum(means_by_key[(ds, s)].values()) or 1.0) for s in systems]
				colour = colour_of.get(label, OTHER_LLM_COLOUR)
			bars = ax.bar(x, vals, bottom=bottom, color=colour, width=0.6,
			             edgecolor=v.SURFACE, linewidth=0.8,
			             label=label if label in other_members else _display_label(label))
			handles_by_label.setdefault(label, bars)
			bottom = [b + val for b, val in zip(bottom, vals)]
		ax.set_xticks(x)
		ax.set_xticklabels(systems, fontsize=13, fontweight="bold")
		ax.set_ylim(0, 100)
		ax.set_title(DATASET_LABELS[ds], fontsize=14, fontweight="bold")
		ax.tick_params(axis="y", labelsize=10.5)
		if ax is axes[0]:
			ax.set_ylabel("Share of attributed GPU energy (%)", fontsize=11.5)
	handles = [handles_by_label[lbl] for lbl in legend_order if lbl in handles_by_label]
	labels = [lbl if lbl in other_members else _display_label(lbl)
	         for lbl in legend_order if lbl in handles_by_label]
	ncol = min(5, len(labels))
	handles, labels = _legend_row_major(handles, labels, ncol)
	fig.legend(handles, labels, loc="lower center", ncol=ncol, frameon=False,
	          fontsize=10, bbox_to_anchor=(0.5, -0.1))
	fig.suptitle("Energy distribution by operation", fontsize=15, fontweight="bold", y=1.02)
	fig.tight_layout(rect=(0, 0.16, 1, 0.94))
	return fig


def _legend_row_major(handles, labels, ncol):
	"""matplotlib's multi-column legend fills column-major (down each column,
	then across); this transposes a row-major-ordered (handles, labels) input
	so the RENDERED legend reads left-to-right/top-to-bottom in that same
	order -- the standard reshape/pad/transpose fix, not a new layout."""
	n = len(handles)
	nrows = -(-n // ncol)  # ceil(n / ncol)
	pad = nrows * ncol - n
	padded_h = list(handles) + [None] * pad
	padded_l = list(labels) + [None] * pad
	grid_h = [padded_h[i * ncol:(i + 1) * ncol] for i in range(nrows)]
	grid_l = [padded_l[i * ncol:(i + 1) * ncol] for i in range(nrows)]
	new_h, new_l = [], []
	for c in range(ncol):
		for r in range(nrows):
			if grid_h[r][c] is not None:
				new_h.append(grid_h[r][c])
				new_l.append(grid_l[r][c])
	return new_h, new_l


# ---------------------------------------------------------------------------
# Main RQ2 figure: trajectory-characteristic correlation heatmap
# ---------------------------------------------------------------------------

#: The four VS_VARIABLES correlation rows already computed and persisted by
#: make_comparable_figures.py (results/tables/<ds>/<sys>/fig3_<sys>_trajectory_
#: properties_correlations.csv). KG calls is deliberately excluded: no script
#: in this repository has ever computed a kg_calls-vs-energy correlation, and
#: this figure must not silently introduce a new computation to fill that row
#: (see project notes -- add it as a separate, explicitly-approved task if
#: needed later).
HEATMAP_VARIABLES = ("output_tokens", "input_tokens", "llm_calls", "max_traversal_depth")
HEATMAP_VARIABLE_LABELS = {
	"output_tokens": "Output tokens",
	"input_tokens": "Input tokens",
	"llm_calls": "LLM calls",
	"max_traversal_depth": "Traversal depth",
}
HEATMAP_VMIN, HEATMAP_VMAX = 0.0, 1.0


def _correlation_csv_path(ds, sys_name):
	"""Project-root-relative path to the already-computed, already-persisted
	per-system correlation CSV -- never a machine-specific absolute path."""
	sys_lower = sys_name.lower()
	return os.path.join(ROOT, "results", "tables", ds, sys_lower,
	                    f"fig3_{sys_lower}_trajectory_properties_correlations.csv")


def load_trajectory_correlations(dataset_order, system_order):
	"""{(ds, sys): {variable: spearman_rho or None}} read straight from the
	existing per-system correlation CSVs -- no recomputation. Only the
	all-questions row (category == "(drawn)") is used, matching the
	all-outcomes case everywhere else in this figure set. Raises with a clear
	message if an expected file or row is missing, rather than silently
	plotting a gap as data."""
	rho_by_key = {}
	for ds in dataset_order:
		for sys_name in system_order:
			path = _correlation_csv_path(ds, sys_name)
			if not os.path.exists(path):
				raise SystemExit(f"missing correlation table for {ds}/{sys_name}: {path}\n"
				                 f"(expected output of measurement/make_comparable_figures.py)")
			rows = {}
			with open(path, newline="") as f:
				for row in csv.DictReader(f):
					if row.get("category") == "(drawn)" and row.get("x_variable") in HEATMAP_VARIABLES:
						rows[row["x_variable"]] = row
			missing = [v for v in HEATMAP_VARIABLES if v not in rows]
			if missing:
				raise SystemExit(f"{path}: missing correlation row(s) for {missing}")
			rho_by_key[(ds, sys_name)] = {
				variable: (float(rows[variable]["spearman_rho"])
				          if rows[variable]["status"] == "ok" else None)
				for variable in HEATMAP_VARIABLES
			}
	return rho_by_key


def draw_trajectory_correlation_heatmap(plt, rho_by_key, dataset_order, dataset_labels,
                                        system_order):
	"""Main RQ2 figure: 1x2 (WebQSP, CWQ), rows = trajectory characteristics,
	columns = systems, cell = Spearman rho (already computed, read verbatim
	from the existing per-system correlation tables -- see
	load_trajectory_correlations). Spearman is used throughout, not Pearson,
	because several of these relationships are monotonic but not linear (most
	notably CoR's traversal-depth relationship); one consistent statistic
	across all rows avoids implying a linear-fit claim this figure does not
	make. Non-causal framing only: this figure shows association, not a
	causal or predictive claim."""
	fig, axes = plt.subplots(1, len(dataset_order),
	                         figsize=(3.9 * len(system_order), 1.1 * len(HEATMAP_VARIABLES) + 1.6))
	axes = [axes] if len(dataset_order) == 1 else list(axes)
	im = None
	for panel_i, (ax, ds) in enumerate(zip(axes, dataset_order)):
		matrix = [[rho_by_key[(ds, sys_name)][variable] for sys_name in system_order]
		         for variable in HEATMAP_VARIABLES]
		im = ax.imshow(matrix, vmin=HEATMAP_VMIN, vmax=HEATMAP_VMAX, cmap="viridis",
		               aspect="auto")
		for i, variable in enumerate(HEATMAP_VARIABLES):
			for j, sys_name in enumerate(system_order):
				val = rho_by_key[(ds, sys_name)][variable]
				text = f"{val:.2f}" if val is not None else "n/a"
				colour = "white" if (val is None or val < 0.55) else "black"
				ax.text(j, i, text, ha="center", va="center", fontsize=10.5, color=colour)
		ax.set_xticks(range(len(system_order)))
		ax.set_xticklabels(system_order, fontsize=11, fontweight="bold")
		ax.set_yticks(range(len(HEATMAP_VARIABLES)))
		if panel_i == 0:
			ax.set_yticklabels([HEATMAP_VARIABLE_LABELS[var] for var in HEATMAP_VARIABLES],
			                   fontsize=10)
		else:
			ax.set_yticklabels([])
		ax.set_title(dataset_labels[ds], fontsize=13, fontweight="bold")
		ax.set_xticks([x - 0.5 for x in range(1, len(system_order))], minor=True)
		ax.set_yticks([y - 0.5 for y in range(1, len(HEATMAP_VARIABLES))], minor=True)
		ax.grid(which="minor", color=v.SURFACE, linewidth=1.5)
		ax.tick_params(which="minor", length=0)
	fig.subplots_adjust(wspace=0.06)
	cbar = fig.colorbar(im, ax=axes, shrink=0.82, pad=0.03)
	cbar.set_label("Spearman rho (association with GPU energy)", fontsize=9.5)
	fig.suptitle("Trajectory characteristics and energy",
	            fontsize=13.5, fontweight="bold")
	return fig


def _label_colours(labels_present):
	cmap = __import__("matplotlib").colormaps["tab20"]
	ordered = [lbl for lbl in LABEL_ORDER if lbl in labels_present]
	extra = sorted(labels_present - set(ordered))
	if extra:
		ordered += extra  # unexpected label (e.g. legacy llm:generate); keep visible, don't drop
	return ordered, {lbl: cmap(i / max(1, len(ordered) - 1)) for i, lbl in enumerate(ordered)}


def draw_fig3(plt, means_by_key, order_labels, colour_of):
	fig, axes = plt.subplots(2, len(DATASET_ORDER), figsize=(5.2 * len(DATASET_ORDER), 8.4))
	for col, ds in enumerate(DATASET_ORDER):
		systems = [s for s in SYSTEM_ORDER if (ds, s) in means_by_key]
		abs_ax = axes[0][col]
		norm_ax = axes[1][col]
		x = range(len(systems))
		bottom_abs = [0.0] * len(systems)
		bottom_norm = [0.0] * len(systems)
		for label in order_labels:
			abs_vals = [means_by_key[(ds, s)].get(label, 0.0) for s in systems]
			totals = [sum(means_by_key[(ds, s)].values()) or 1.0 for s in systems]
			norm_vals = [100.0 * a / t for a, t in zip(abs_vals, totals)]
			if all(a == 0 for a in abs_vals):
				continue
			abs_ax.bar(x, abs_vals, bottom=bottom_abs, color=colour_of[label], width=0.62,
			          label=label)
			norm_ax.bar(x, norm_vals, bottom=bottom_norm, color=colour_of[label], width=0.62)
			bottom_abs = [b + a for b, a in zip(bottom_abs, abs_vals)]
			bottom_norm = [b + a for b, a in zip(bottom_norm, norm_vals)]
		for ax in (abs_ax, norm_ax):
			ax.set_xticks(list(x))
			ax.set_xticklabels(systems)
		abs_ax.set_title(DATASET_LABELS[ds], fontsize=11, fontweight="bold")
		abs_ax.set_ylabel("Mean attributed energy\nper question (Wh)")
		norm_ax.set_ylabel("Share of attributed\nenergy (%)")
		norm_ax.set_ylim(0, 100)
	fig.suptitle("Attributed per-operation energy (narrower than the whole-question "
	            "trajectory energy used in Figures 1-2; see "
	            "operation_energy_coverage.csv for the gap)", fontsize=10.5,
	            fontweight="bold", y=1.02)
	handles, labels = axes[0][0].get_legend_handles_labels()
	# de-duplicate while preserving order across the two dataset columns
	seen = set()
	dedup_h, dedup_l = [], []
	for h, l in zip(handles, labels):
		if l not in seen:
			seen.add(l)
			dedup_h.append(h)
			dedup_l.append(l)
	fig.legend(dedup_h, dedup_l, loc="lower center", ncol=min(4, len(dedup_l)), frameon=False,
	          fontsize=8, bbox_to_anchor=(0.5, -0.05))
	fig.tight_layout(rect=(0, 0.1, 1, 0.96))
	return fig


# ---------------------------------------------------------------------------


def main(argv=None):
	ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
	ap.add_argument("--dataset", action="append", required=True, metavar="NAME=CACHE_OUTDIR",
	                help="webqsp=DIR and cwq=DIR, each an already-built "
	                     "make_comparable_figures.py --out directory; repeatable")
	ap.add_argument("--out-figures", default="results/figures/cross_dataset")
	ap.add_argument("--out-tables", default="results/tables/cross_dataset")
	ap.add_argument("--fig2-scale", default="log", choices=("log", "linear"),
	                help="y-axis scale for Figure 2 (default log: per-question energy "
	                     "spans a 7-39x range within each system)")
	args = ap.parse_args(argv)

	cache_dirs = mcf.parse_pairs(args.dataset, {})
	missing = [d for d in DATASET_ORDER if d not in cache_dirs]
	if missing:
		raise SystemExit(f"--dataset required for {missing}")

	figures_out = os.path.abspath(os.path.expanduser(args.out_figures))
	tables_out = os.path.abspath(os.path.expanduser(args.out_tables))
	os.makedirs(figures_out, exist_ok=True)
	os.makedirs(tables_out, exist_ok=True)

	print("=" * 70)
	print("VALIDATION")
	print("=" * 70)

	loaded = {}
	for ds in DATASET_ORDER:
		order, qrows_by_system = _load_dataset(ds, cache_dirs[ds])
		loaded[ds] = (order, qrows_by_system)
		for name in order:
			qrows = qrows_by_system[name]
			ids = [r["question_id"] for r in qrows]
			dup = len(ids) - len(set(ids))
			n_f1 = sum(1 for r in qrows if r.get("f1") is not None)
			n_traj = sum(1 for r in qrows if r.get("energy_trajectory_j") is not None)
			n_attr = sum(1 for r in qrows
			            if any(k.startswith("energy_") and k != "energy_trajectory_j"
			                  and v is not None for k, v in r.items()))
			mean_f1 = 100.0 * statistics.mean(r["f1"] for r in qrows if r.get("f1") is not None)
			traj_vals = v.j_to_wh([r["energy_trajectory_j"] for r in qrows
			                       if r.get("energy_trajectory_j") is not None])
			mean_wh = statistics.mean(traj_vals) if traj_vals else None
			med_wh = statistics.median(traj_vals) if traj_vals else None
			legacy = sum(1 for r in qrows if r.get("energy_llm:generate") is not None)
			print(f"{DATASET_LABELS[ds]:7s} {name:4s}  n={len(qrows):,}  duplicate_ids={dup}  "
			     f"n_with_f1={n_f1:,}  n_with_trajectory_energy={n_traj:,}  "
			     f"n_with_operation_energy={n_attr:,}  mean_f1={mean_f1:.2f}%  "
			     f"mean_wh={mean_wh:.4f}  median_wh={med_wh:.4f}"
			     + (f"  ** {legacy} rows carry legacy llm:generate energy **" if legacy else ""))
			if n_f1 != n_traj:
				print(f"    note: F1 and trajectory-energy populations differ by "
				     f"{abs(n_f1 - n_traj)} question(s) for {name}/{ds}")

	plt = v._pyplot()

	# ---- Figure 2 -----------------------------------------------------
	per_panel = {}
	corr_rows = []
	corr_rows_by_key = {}
	for ds in DATASET_ORDER:
		order, qrows_by_system = loaded[ds]
		for name in order:
			qrows = qrows_by_system[name]
			f1, energy = [], []
			for r in qrows:
				if r.get("f1") is None or r.get("energy_trajectory_j") is None:
					continue
				f1.append(100.0 * r["f1"])
				energy.append(v.j_to_wh(r["energy_trajectory_j"]))
			per_panel[(ds, name)] = (f1, energy)
			rho, p = _spearman(f1, energy) if len(f1) > 1 else (float("nan"), float("nan"))
			row = {"dataset": DATASET_LABELS[ds], "system": name, "n": len(f1),
			      "spearman_rho": rho, "p_value": p,
			      "mean_f1": statistics.mean(f1) if f1 else None,
			      "mean_energy_wh": statistics.mean(energy) if energy else None,
			      "median_energy_wh": statistics.median(energy) if energy else None}
			corr_rows.append(row)
			corr_rows_by_key[(ds, name)] = row

	corr_path = os.path.join(tables_out, "question_f1_energy_correlations.csv")
	with open(corr_path, "w", newline="") as f:
		w = csv.DictWriter(f, fieldnames=FIG2_FIELDS, lineterminator="\n")
		w.writeheader()
		w.writerows(corr_rows)

	use_log = (args.fig2_scale == "log")
	with plt.rc_context(v.STYLE):
		fig = draw_fig2(plt, per_panel, corr_rows_by_key, use_log)
		fig2_paths = v.save_figure(fig, figures_out, "fig_question_f1_vs_energy", png=False)
		plt.close(fig)

	# ---- Figure 2 (v2, for review only -- does not touch fig_question_f1_vs_energy.pdf)
	with plt.rc_context(v.STYLE):
		fig = draw_fig2_v2(plt, per_panel, corr_rows_by_key, use_log)
		fig2v2_paths = v.save_figure(fig, figures_out, "fig_question_f1_vs_energy_v2", png=False)
		plt.close(fig)

	print()
	print("Figure 2 -- Spearman F1-vs-energy correlation (n, rho, p):")
	for row in corr_rows:
		p_str = f"{row['p_value']:.2e}" if row["p_value"] == row["p_value"] else "n/a"
		print(f"  {row['dataset']:7s} {row['system']:4s}  n={row['n']:,}  "
		     f"rho={row['spearman_rho']:+.3f}  p={p_str}")

	# ---- Figure 3 -------------------------------------------------------
	means_by_key = {}
	coverage_rows = []
	labels_present = set()
	for ds in DATASET_ORDER:
		order, qrows_by_system = loaded[ds]
		for name in order:
			means_wh, coverage = _op_energy_means(qrows_by_system[name])
			means_by_key[(ds, name)] = means_wh
			labels_present |= set(means_wh)
			coverage_rows.append({"dataset": DATASET_LABELS[ds], "system": name, **coverage})

	coverage_path = os.path.join(tables_out, "operation_energy_coverage.csv")
	with open(coverage_path, "w", newline="") as f:
		w = csv.DictWriter(f, fieldnames=COVERAGE_FIELDS, lineterminator="\n")
		w.writeheader()
		for row in coverage_rows:
			w.writerow({k: ("" if row.get(k) is None else row.get(k)) for k in COVERAGE_FIELDS})

	order_labels, colour_of_label = _label_colours(labels_present)
	with plt.rc_context(v.STYLE):
		fig = draw_fig3(plt, means_by_key, order_labels, colour_of_label)
		fig3_paths = v.save_figure(fig, figures_out, "fig_operation_energy_breakdown", png=False)
		plt.close(fig)

	# ---- Figure 3 main (RQ2a, for review only -- does not touch
	# fig_operation_energy_breakdown.pdf, which stays as the full-operation
	# supplementary figure). "Other" bucket: a label is kept on its own iff it
	# reaches >=2% of a panel's attributed energy in AT LEAST ONE of the six
	# (dataset, system) panels; this is evaluated once, globally, over all six
	# panels -- never chosen separately per panel/system.
	kept_order, other_llm_labels, other_kg_labels, max_share = _classify_for_main(means_by_key)
	with plt.rc_context(v.STYLE):
		fig = draw_fig3_main(plt, means_by_key, kept_order, other_llm_labels, other_kg_labels,
		                     colour_of_label)
		fig3main_paths = v.save_figure(fig, figures_out, "fig_operation_energy_shares_main",
		                               png=False)
		plt.close(fig)

	print()
	print(f"Figure 3 main -- residual-bucket rule: a label is kept on its own iff max share "
	     f"across all 6 (dataset, system) panels >= {OTHER_THRESHOLD_PCT:.0f}%; below-threshold "
	     f"labels split into 'Other LLM' vs 'KG / embedding' by operation type")
	print("  kept labels (taxonomy order, max share across panels):")
	for lbl in kept_order:
		print(f"    {lbl:32s} {_display_label(lbl):28s} max_share={max_share[lbl]:5.2f}%")
	if other_llm_labels:
		print("  -> Other LLM (max share across panels, all below threshold):")
		for lbl in sorted(other_llm_labels, key=lambda l: -max_share[l]):
			print(f"    {lbl:32s} max_share={max_share[lbl]:5.2f}%")
	else:
		print("  -> Other LLM: empty")
	if other_kg_labels:
		print("  -> KG / embedding (max share across panels, all below threshold):")
		for lbl in sorted(other_kg_labels, key=lambda l: -max_share[l]):
			print(f"    {lbl:32s} max_share={max_share[lbl]:5.2f}%")
	else:
		print("  -> KG / embedding: empty")

	print()
	print("Figure 3 -- mean attributed energy vs. mean trajectory energy "
	     "(attribution coverage):")
	for row in coverage_rows:
		mtw = row["mean_trajectory_wh"]
		maw = row["mean_attributed_wh"]
		frac = row["mean_attribution_fraction"]
		print(f"  {row['dataset']:7s} {row['system']:4s}  trajectory={mtw:.4f} Wh  "
		     f"attributed={maw:.4f} Wh  mean_attribution_fraction={frac:.3f}" if mtw is not None
		     else f"  {row['dataset']:7s} {row['system']:4s}  no trajectory energy")

	# ---- Figure 5 (RQ2 main): combined trajectory cost drivers, both
	# datasets in one file -- output tokens / LLM calls / traversal depth vs
	# whole-question trajectory GPU energy, no outcome colouring. Reuses the
	# same per-question rows already loaded above; does not touch the
	# untouched per-dataset fig3_*_trajectory_properties supplementary files.
	rows_by_key = {}
	for ds in DATASET_ORDER:
		order, qrows_by_system = loaded[ds]
		for name in order:
			rows_by_key[(ds, name)] = qrows_by_system[name]
	with plt.rc_context(v.STYLE):
		fig = mcf.draw_trajectory_cost_drivers_main(plt, rows_by_key, DATASET_ORDER,
		                                            DATASET_LABELS, SYSTEM_ORDER)
		fig_costdrivers_paths = v.save_figure(fig, figures_out, "fig_trajectory_cost_drivers_main",
		                                      png=False)
		plt.close(fig)

	# ---- Main RQ2 figure: trajectory-characteristic correlation heatmap.
	# Reads the already-computed, already-persisted per-system correlation
	# tables (results/tables/<ds>/<sys>/fig3_<sys>_trajectory_properties_
	# correlations.csv) -- no correlation is recomputed here. KG calls is
	# deliberately excluded (see HEATMAP_VARIABLES comment); wall time is out
	# of scope for this figure entirely.
	rho_by_key = load_trajectory_correlations(DATASET_ORDER, SYSTEM_ORDER)
	with plt.rc_context(v.STYLE):
		fig = draw_trajectory_correlation_heatmap(plt, rho_by_key, DATASET_ORDER, DATASET_LABELS,
		                                          SYSTEM_ORDER)
		heatmap_paths = v.save_figure(fig, figures_out, "fig_trajectory_correlation_heatmap_main",
		                              png=False)
		plt.close(fig)

	print()
	print("Figure 4 (energy distribution violins): not produced -- the "
	     "effectiveness-vs-energy figure (table1_effectiveness_energy.py) already "
	     "gives the aggregate mean+CI per system and Figure 2 already shows the full "
	     "per-question distribution; a marginal violin of the same trajectory-energy "
	     "column would not add a new view.")

	print()
	print("Trajectory correlation heatmap (main RQ2 figure) -- Spearman rho by "
	     "(dataset, system, variable), read verbatim from existing per-system "
	     "correlation tables (no recomputation):")
	for ds in DATASET_ORDER:
		for name in SYSTEM_ORDER:
			row = rho_by_key[(ds, name)]
			print(f"  {DATASET_LABELS[ds]:7s} {name:4s}  "
			     + "  ".join(f"{HEATMAP_VARIABLE_LABELS[var]}={row[var]:.3f}"
			                 if row[var] is not None else f"{HEATMAP_VARIABLE_LABELS[var]}=n/a"
			                 for var in HEATMAP_VARIABLES))

	print()
	for p in fig2_paths + fig2v2_paths + [corr_path]:
		print(f"  -> {p}")
	for p in fig3_paths + fig3main_paths + [coverage_path]:
		print(f"  -> {p}")
	for p in fig_costdrivers_paths:
		print(f"  -> {p}")
	for p in heatmap_paths:
		print(f"  -> {p}")
	return 0


if __name__ == "__main__":
	sys.exit(main())
