"""Compatibility entry point: trajectory accounting moved to the profiler.

Implementation: agent_energy_profiler/trajectory.py.
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from agent_energy_profiler.trajectory import *  # noqa: E402,F401,F403
from agent_energy_profiler.trajectory import (  # noqa: E402,F401
	DIAGNOSTIC_FIELDS,
	ENERGY_FIELDS,
	accumulate,
	overlap_stats,
	render,
	summarize,
	write_artifacts,
)
