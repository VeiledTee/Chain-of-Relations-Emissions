"""Compatibility entry point: attribution now lives in the profiler package.

The implementation moved to agent_energy_profiler/attribution.py, which knows
nothing about CoR, Freebase or KGQA. This shim keeps the documented command
line and the `measurement/attribute.py` import path working:

    python measurement/attribute.py --events events.jsonl --power power.csv

Prefer the package form in new work:

    python -m agent_energy_profiler.attribution ...
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from agent_energy_profiler.attribution import (  # noqa: E402,F401
	CARRIED_FIELDS,
	RAPL_MAX,
	attribute_events,
	classify_rapl,
	counter_delta,
	integrate_gpu,
	load_power,
	main,
	measured_total,
	rapl_delta,
	sum_counter_domain,
	sum_domain,
)

if __name__ == "__main__":
	main()
