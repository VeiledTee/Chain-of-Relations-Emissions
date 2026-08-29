"""Compatibility entry point: the power sampler moved to the profiler.

Implementation: agent_energy_profiler/sampling.py. Keeps the documented
command line working:

    python measurement/power_logger.py --out power.csv --hz 10

Prefer the package form in new work:

    python -m agent_energy_profiler.sampling --out power.csv --hz 10
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from agent_energy_profiler.sampling import main  # noqa: E402,F401

if __name__ == "__main__":
	main()
