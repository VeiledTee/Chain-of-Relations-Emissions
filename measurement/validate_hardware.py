"""Compatibility entry point: the hardware capability probe moved.

Implementation: agent_energy_profiler/validation.py (with zone discovery and
domain classification in agent_energy_profiler/hardware/discovery.py). Keeps
the documented command line working:

    python measurement/validate_hardware.py --json hardware_report.json

Prefer the package form in new work:

    python -m agent_energy_profiler.validation --json hardware_report.json
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from agent_energy_profiler.validation import *  # noqa: E402,F401,F403
from agent_energy_profiler.validation import (  # noqa: E402,F401
	BOUNDARY_DOMAINS,
	CONSISTENCY_REL_TOL,
	DIAGNOSTIC_DOMAINS,
	EXCLUDED_DOMAINS,
	INCONSISTENT,
	SUPPORTED,
	UNAVAILABLE,
	UNKNOWN,
	UNREADABLE,
	assess_counter_consistency,
	boundary_verdict,
	build_report,
	classify_zone_name,
	discover_amd_hwmon_zones,
	discover_powercap_zones,
	integrate_power,
	main,
	probe_gpu_counter_consistency,
	probe_nvml,
	probe_zone,
	render,
)

if __name__ == "__main__":
	main()
