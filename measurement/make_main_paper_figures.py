"""Single entry point for the three main C2 paper figures.

This script does not compute anything itself: it calls the two existing,
already-tracked figure scripts with a consistent --dataset argument, so the
three main figures can always be regenerated with one command from data that
already exists. No metric, correlation, or score is computed here or
duplicated from those scripts.

    python -m measurement.make_main_paper_figures \\
        --dataset webqsp=~/cwq_supervisor_summary/cross_dataset/webqsp/wh \\
        --dataset cwq=~/cwq_supervisor_summary/wh \\
        --backbone Gemma-3-4B

Produces, in results/figures/cross_dataset/ by default (override with
--out-figures, forwarded to both underlying scripts):

    fig_operation_energy_shares_main.pdf           Figure 1 (RQ1) -- via
                                                    measurement.thesis_figures
    fig_trajectory_correlation_heatmap_main.pdf    Figure 2 (RQ2) -- via
                                                    measurement.thesis_figures
    fig_f1_energy_frontier_main.pdf                Figure 3 (current F1-energy
                                                    comparison; RQ3 energy is
                                                    robust, the F1 side uses
                                                    the current, still-
                                                    provisional evaluation
                                                    pipeline) -- via
                                                    measurement.table1_effectiveness_energy

Each figure's own script remains independently runnable and produces its own
additional supplementary outputs (see each script's docstring); this wrapper
only sequences the two calls needed to produce all three main figures with
one command and one --dataset argument.
"""

import argparse
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)
sys.path.insert(0, HERE)

import thesis_figures  # noqa: E402
import table1_effectiveness_energy  # noqa: E402


def main(argv=None):
	ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
	ap.add_argument("--dataset", action="append", required=True, metavar="NAME=CACHE_OUTDIR",
	                help="webqsp=DIR and cwq=DIR, each an already-built "
	                     "make_comparable_figures.py --out directory; repeatable; "
	                     "forwarded verbatim to both underlying scripts")
	ap.add_argument("--backbone", default="Gemma-3-4B",
	                help="forwarded to table1_effectiveness_energy.py (Figure 3)")
	ap.add_argument("--out-figures", default="results/figures/cross_dataset",
	                help="forwarded to thesis_figures.py (Figures 1 and 2)")
	ap.add_argument("--out-tables", default="results/tables/cross_dataset",
	                help="forwarded to thesis_figures.py (Figures 1 and 2) and, as --out, to "
	                     "table1_effectiveness_energy.py (Figure 3's table)")
	args = ap.parse_args(argv)

	dataset_args = []
	for item in args.dataset:
		dataset_args += ["--dataset", item]

	print("=" * 70)
	print("FIGURE 1 + FIGURE 2  (measurement.thesis_figures)")
	print("=" * 70)
	rc = thesis_figures.main(dataset_args + ["--out-figures", args.out_figures,
	                                         "--out-tables", args.out_tables])
	if rc:
		return rc

	print()
	print("=" * 70)
	print("FIGURE 3  (measurement.table1_effectiveness_energy)")
	print("=" * 70)
	rc = table1_effectiveness_energy.main(dataset_args + ["--backbone", args.backbone,
	                                                      "--out", args.out_tables,
	                                                      "--out-figures", args.out_figures])
	if rc:
		return rc

	print()
	print("Main figures regenerated:")
	for stem in ("fig_operation_energy_shares_main", "fig_trajectory_correlation_heatmap_main",
	            "fig_f1_energy_frontier_main"):
		print(f"  -> {os.path.join(args.out_figures, stem)}.pdf")
	return 0


if __name__ == "__main__":
	sys.exit(main())
