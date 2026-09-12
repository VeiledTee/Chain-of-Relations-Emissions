"""Compatibility entry point: figures live in the profiler package.

Implementation: agent_energy_profiler/visualize.py (paradigm-agnostic; reads a
run's events_attributed.jsonl and trajectory_summary.json).

    python measurement/visualize.py --list --run measurement/runs/<run>
    python measurement/visualize.py --plot operation-energy \
        --run measurement/runs/<run> --out <dir>

Prefer the package form in new work:

    python -m agent_energy_profiler.visualize ...
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from agent_energy_profiler.visualize import main  # noqa: E402

if __name__ == "__main__":
	sys.exit(main())
