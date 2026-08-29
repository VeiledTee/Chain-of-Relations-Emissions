"""CPU/DRAM energy counters, whatever the host exposes them through.

Two access paths are searched, because a host that has RAPL MSRs does not
necessarily expose them through powercap:

  /sys/class/powercap/*/energy_uj        powercap (intel-rapl, and others)
  /sys/class/hwmon/hwmon*/energy*_input  amd_energy hwmon driver

No CPU vendor is assumed. Globbing only intel-rapl would make a host with
readable counters behind a different driver look identical to a host with no
CPU energy counters at all, which would silently understate what the machine
could have measured.

Zones are labelled by the name the host itself reports; deciding which of those
names are additive is `discovery`'s job, not this module's. Every counter here
is CUMULATIVE microjoules -- a reading is differenced across a window, never
treated as a rate.

Absent or unreadable counters are reported as absent. They are never zero.
"""

import glob
import os


def discover_zones():
	"""Readable cumulative CPU/DRAM energy counter paths, sorted and stable."""
	paths = sorted(glob.glob("/sys/class/powercap/*/energy_uj"))
	for hwmon in sorted(glob.glob("/sys/class/hwmon/hwmon*")):
		try:
			if open(os.path.join(hwmon, "name")).read().strip() != "amd_energy":
				continue
		except Exception:
			continue
		paths.extend(sorted(glob.glob(os.path.join(hwmon, "energy*_input"))))
	return paths


def zone_name(path):
	"""Domain name a host reports for one counter path.

	powercap zones carry a sibling `name` file; amd_energy hwmon uses a
	per-input `_label`. Falls back to the containing directory name.
	"""
	directory = os.path.dirname(path)
	label_file = path.replace("_input", "_label")
	for candidate in (os.path.join(directory, "name"), label_file):
		if candidate == path:
			continue
		try:
			value = open(candidate).read().strip()
			if value and value != "amd_energy":
				return value
		except Exception:
			continue
	return os.path.basename(directory)


def column_names(paths):
	"""CSV column name per counter path: rapl_<domain>_<node>_uj."""
	return [f"rapl_{zone_name(path)}_{os.path.basename(os.path.dirname(path))}_uj"
	        for path in paths]


def read_zone(path):
	"""Raw cumulative counter value (uJ) as text, or None if unreadable."""
	try:
		return open(path).read().strip()
	except Exception:
		return None
