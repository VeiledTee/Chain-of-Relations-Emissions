"""What can this host actually measure, and which domains may be added up.

Capability, never vendor. A host qualifies for a measurement boundary by
exposing readable, advancing counters for the domains that boundary needs. No
CPU or GPU vendor is assumed anywhere in this module: domains are classified by
the names the host itself reports.

The additive rule this package enforces everywhere:

    measured_energy_j = gpu_energy_j + cpu_package_energy_j + dram_energy_j

`core` and `uncore` are DIAGNOSTIC: core is contained within package, so adding
both double-counts. `psys` is EXCLUDED in the other direction: it is a platform
domain that contains package, so counting both double-counts as well.

Discovery reports what is there. It never reports a domain the host does not
expose as zero.
"""

import glob
import os

SUPPORTED = "SUPPORTED"
UNREADABLE = "UNREADABLE"
#: Counter is present and advancing, but its values are not physically
#: credible (see validation.assess_counter_consistency). Never additive.
INCONSISTENT = "INCONSISTENT"
UNAVAILABLE = "UNAVAILABLE"
UNKNOWN = "UNKNOWN"

#: Domains that make up the additive measurement boundary.
BOUNDARY_DOMAINS = ("gpu", "cpu_package", "dram")
#: Domains that are measured but must never be added to the total.
DIAGNOSTIC_DOMAINS = ("cpu_core", "uncore")
#: Domains deliberately dropped because they overlap an additive domain.
EXCLUDED_DOMAINS = ("psys",)


def discover_powercap_zones(root="/sys/class/powercap"):
	"""Every powercap zone on the host, whatever the vendor prefix.

	Deliberately not hardcoded to intel-rapl: a host may expose amd-rapl or
	another prefix, and silently finding nothing would look identical to a
	host that genuinely has no CPU energy counters.
	"""
	zones = []
	if not os.path.isdir(root):
		return zones
	for path in sorted(glob.glob(os.path.join(root, "*"))):
		energy_file = os.path.join(path, "energy_uj")
		if not os.path.exists(energy_file):
			continue
		try:
			with open(os.path.join(path, "name")) as f:
				name = f.read().strip()
		except Exception:
			name = os.path.basename(path)
		zones.append({
			"node": os.path.basename(path),
			"name": name,
			"energy_uj_path": energy_file,
			"column": f"rapl_{name}_{os.path.basename(path)}_uj",
		})
	return zones


def discover_amd_hwmon_zones(root="/sys/class/hwmon"):
	"""Some hosts expose RAPL through the amd_energy hwmon driver, not powercap.

	Probed separately so such a host is never mistaken for one with no CPU
	energy counters at all. Whatever domains are found are classified by the
	names the host reports; eligibility is decided by capability (see
	validation.boundary_verdict), never by CPU vendor.
	"""
	zones = []
	if not os.path.isdir(root):
		return zones
	for hwmon in sorted(glob.glob(os.path.join(root, "hwmon*"))):
		try:
			with open(os.path.join(hwmon, "name")) as f:
				driver = f.read().strip()
		except Exception:
			continue
		if driver != "amd_energy":
			continue
		for energy_file in sorted(glob.glob(os.path.join(hwmon, "energy*_input"))):
			label_file = energy_file.replace("_input", "_label")
			try:
				with open(label_file) as f:
					label = f.read().strip()
			except Exception:
				label = os.path.basename(energy_file)
			zones.append({
				"node": os.path.basename(hwmon),
				"name": label,
				"energy_uj_path": energy_file,
				"column": f"rapl_{label}_{os.path.basename(hwmon)}_uj",
				"driver": "amd_energy",
			})
	return zones


def classify_zone_name(name):
	"""Map a RAPL zone name to a measurement-boundary domain.

	Mirrors attribution.classify_rapl. Order matters:

	  dram     additive
	  psys     EXCLUDED -- platform domain that *contains* package
	  package  additive ("socket" is the package equivalent on some drivers)
	  uncore   diagnostic, and checked BEFORE core because "uncore"
	           contains the substring "core"
	  core     diagnostic -- contained within package, never added
	"""
	lowered = str(name).lower()
	if "dram" in lowered:
		return "dram"
	if "psys" in lowered:
		return "psys"
	if "package" in lowered or "socket" in lowered:
		return "cpu_package"
	if "uncore" in lowered:
		return "uncore"
	if "core" in lowered:
		return "cpu_core"
	return "unknown"
