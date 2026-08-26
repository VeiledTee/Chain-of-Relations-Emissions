"""Hardware capability probe for the C2 measurement instrument.

Answers one question: which parts of the C2 measurement boundary can this
host actually measure?

    measured_energy_j = gpu_energy_j + cpu_package_energy_j + dram_energy_j

Every domain is reported with an explicit status. Nothing missing is ever
reported as zero.

    SUPPORTED    present and readable, values advance
    UNREADABLE   present but permissions/errors block reading
    UNAVAILABLE  the host does not expose this domain at all
    UNKNOWN      probe could not determine the state

Usage:
  python measurement/validate_hardware.py
  python measurement/validate_hardware.py --json measurement/hardware_report.json
  python measurement/validate_hardware.py --settle 2.0
"""

import argparse
import glob
import json
import os
import platform
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))

SUPPORTED = "SUPPORTED"
UNREADABLE = "UNREADABLE"
UNAVAILABLE = "UNAVAILABLE"
UNKNOWN = "UNKNOWN"

#: Domains that make up the additive measurement boundary.
BOUNDARY_DOMAINS = ("gpu", "cpu_package", "dram")


# --------------------------------------------------------------------------
# RAPL zone discovery and classification
# --------------------------------------------------------------------------

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
	boundary_verdict), never by CPU vendor.
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

	Mirrors measurement/attribute.py:classify_rapl. Order matters:

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


#: Domains that are measured but must never be added to the total.
DIAGNOSTIC_DOMAINS = ("cpu_core", "uncore")
#: Domains deliberately dropped because they overlap an additive domain.
EXCLUDED_DOMAINS = ("psys",)


def probe_zone(zone, settle=0.5):
	"""Read a zone twice to prove it is readable and advancing."""
	path = zone["energy_uj_path"]
	result = dict(zone)
	result["domain"] = classify_zone_name(zone["name"])
	try:
		with open(path) as f:
			first = int(f.read().strip())
	except PermissionError as e:
		result.update(status=UNREADABLE, detail=f"permission denied: {e}")
		return result
	except Exception as e:
		result.update(status=UNREADABLE, detail=f"{type(e).__name__}: {e}")
		return result

	time.sleep(settle)
	try:
		with open(path) as f:
			second = int(f.read().strip())
	except Exception as e:
		result.update(status=UNREADABLE, detail=f"second read failed: {e}")
		return result

	delta = second - first
	result.update(
		status=SUPPORTED,
		first_uj=first,
		second_uj=second,
		delta_uj=delta,
		advancing=delta > 0,
		detail=("counter advancing" if delta > 0
		        else "counter did not advance during probe window"),
	)
	try:
		result["mode"] = oct(os.stat(path).st_mode & 0o777)
		result["readable_by_current_user"] = os.access(path, os.R_OK)
	except Exception:
		pass
	return result


# --------------------------------------------------------------------------
# GPU / NVML
# --------------------------------------------------------------------------

def probe_nvml(settle=1.0):
	report = {"status": UNKNOWN, "devices": [], "detail": ""}
	try:
		import pynvml
	except Exception as e:
		report.update(status=UNAVAILABLE, detail=f"pynvml not importable: {e}")
		return report
	try:
		pynvml.nvmlInit()
	except Exception as e:
		report.update(status=UNAVAILABLE, detail=f"nvmlInit failed: {e}")
		return report

	try:
		report["driver_version"] = _decode(pynvml.nvmlSystemGetDriverVersion())
	except Exception:
		report["driver_version"] = None

	try:
		count = pynvml.nvmlDeviceGetCount()
	except Exception as e:
		report.update(status=UNREADABLE, detail=f"device count failed: {e}")
		return report

	if count == 0:
		report.update(status=UNAVAILABLE, detail="NVML initialised but no devices")
		return report

	handles = []
	for i in range(count):
		h = pynvml.nvmlDeviceGetHandleByIndex(i)
		handles.append(h)
		device = {"index": i, "name": _decode(pynvml.nvmlDeviceGetName(h))}
		try:
			device["power_w"] = pynvml.nvmlDeviceGetPowerUsage(h) / 1000.0
			device["power_status"] = SUPPORTED
		except Exception as e:
			device["power_status"] = UNAVAILABLE
			device["power_detail"] = str(e)
		try:
			device["energy_mj_first"] = pynvml.nvmlDeviceGetTotalEnergyConsumption(h)
			device["energy_status"] = SUPPORTED
		except Exception as e:
			device["energy_status"] = UNAVAILABLE
			device["energy_detail"] = str(e)
		report["devices"].append(device)

	time.sleep(settle)
	for device, h in zip(report["devices"], handles):
		if device.get("energy_status") != SUPPORTED:
			continue
		try:
			second = pynvml.nvmlDeviceGetTotalEnergyConsumption(h)
			device["energy_mj_second"] = second
			device["energy_delta_mj"] = second - device["energy_mj_first"]
			device["energy_advancing"] = second > device["energy_mj_first"]
		except Exception as e:
			device["energy_status"] = UNREADABLE
			device["energy_detail"] = str(e)

	energy_ok = [d for d in report["devices"] if d.get("energy_status") == SUPPORTED]
	report["status"] = SUPPORTED if energy_ok else UNREADABLE
	report["cumulative_energy_status"] = SUPPORTED if energy_ok else UNAVAILABLE
	report["detail"] = f"{len(energy_ok)}/{count} device(s) expose cumulative energy"
	return report


def probe_nvml_resolution(samples=3000, interval=0.001):
	"""Measure how often the NVML energy counter actually updates.

	Windows shorter than the update interval cannot have their GPU energy
	measured by counter differencing: the delta quantizes to 0 or to one whole
	update step. This bounds per-operation attribution of *GPU* energy only.
	RAPL package/DRAM counters are a separate mechanism with their own update
	characteristics, which this probe does not measure and to which this figure
	must not be assumed to transfer.
	"""
	result = {"status": UNKNOWN}
	try:
		import pynvml
		pynvml.nvmlInit()
		h = pynvml.nvmlDeviceGetHandleByIndex(0)
		read = pynvml.nvmlDeviceGetTotalEnergyConsumption
		prev = read(h)
	except Exception as e:
		result.update(status=UNAVAILABLE, detail=str(e))
		return result

	intervals, steps = [], []
	last = time.time()
	for _ in range(samples):
		try:
			value = read(h)
		except Exception:
			break
		if value != prev:
			now = time.time()
			intervals.append(now - last)
			steps.append(value - prev)
			last, prev = now, value
		time.sleep(interval)

	if len(intervals) < 3:
		result.update(status=UNKNOWN,
		              detail=f"counter changed {len(intervals)} times; too few to characterize")
		return result

	intervals, steps = intervals[1:], steps[1:]
	result.update(
		status=SUPPORTED,
		updates_observed=len(intervals),
		median_update_interval_s=_median(intervals),
		median_update_step_mj=_median(steps),
		detail=("events shorter than the update interval quantize to 0 or one "
		        "whole step and are not measurable by counter differencing"),
	)
	return result


# --------------------------------------------------------------------------
# Independent node telemetry
# --------------------------------------------------------------------------

def probe_node_telemetry():
	"""Look for a power source independent of NVML/RAPL.

	Read-only inspection. Nothing is installed or configured. CodeCarbon is
	deliberately NOT considered here: it reads NVML and RAPL itself, so it
	cannot serve as independent validation of NVML and RAPL.
	"""
	report = {"status": UNAVAILABLE, "sources": [], "detail": ""}
	found = []

	ipmi_devices = glob.glob("/dev/ipmi*")
	if ipmi_devices:
		found.append({"kind": "ipmi", "detail": f"devices: {ipmi_devices}"})

	for tool in ("ipmitool", "ipmi-dcmi", "redfishtool"):
		path = _which(tool)
		if path:
			found.append({"kind": tool, "detail": f"binary at {path}"})

	hwmon_power = []
	for hwmon in sorted(glob.glob("/sys/class/hwmon/hwmon*")):
		try:
			with open(os.path.join(hwmon, "name")) as f:
				driver = f.read().strip()
		except Exception:
			continue
		inputs = glob.glob(os.path.join(hwmon, "power*_input"))
		if inputs:
			hwmon_power.append({"driver": driver, "inputs": inputs})
	if hwmon_power:
		found.append({"kind": "hwmon_power", "detail": hwmon_power})

	report["sources"] = found
	if found:
		report["status"] = UNKNOWN
		report["detail"] = ("candidate telemetry present; independence and "
		                    "readability not yet confirmed")
	else:
		report["detail"] = ("no IPMI/BMC/Redfish/hwmon power source found; "
		                    "no independent whole-node validation possible")
	return report


# --------------------------------------------------------------------------
# Host description
# --------------------------------------------------------------------------

def probe_host():
	host = {
		"hostname": platform.node(),
		"system": platform.system(),
		"release": platform.release(),
		"machine": platform.machine(),
		"python": platform.python_version(),
	}
	release = host["release"].lower()
	if "microsoft" in release or "wsl" in release:
		host["environment"] = "wsl"
	elif os.path.exists("/.dockerenv"):
		host["environment"] = "container"
	else:
		host["environment"] = _detect_virt()
	host["cpu_model"] = _cpu_model()
	host["bare_metal"] = host["environment"] in ("none", "bare-metal")
	return host


def _detect_virt():
	try:
		out = subprocess.run(["systemd-detect-virt"], capture_output=True,
		                     text=True, timeout=5)
		value = out.stdout.strip()
		return value or "none"
	except Exception:
		return "unknown"


def _cpu_model():
	try:
		with open("/proc/cpuinfo") as f:
			for line in f:
				if line.lower().startswith("model name"):
					return line.split(":", 1)[1].strip()
	except Exception:
		pass
	return "unknown"


# --------------------------------------------------------------------------
# Boundary verdict
# --------------------------------------------------------------------------

def boundary_verdict(nvml, zones):
	"""Which additive domains this host can supply, and whether the boundary
	can ever be complete here."""
	domains = {d: {"status": UNAVAILABLE, "source": None} for d in BOUNDARY_DOMAINS}

	if nvml.get("cumulative_energy_status") == SUPPORTED:
		domains["gpu"] = {"status": SUPPORTED, "source": "nvml_total_energy_consumption"}
	elif nvml.get("status") == SUPPORTED:
		domains["gpu"] = {"status": UNREADABLE,
		                  "source": "nvml present, cumulative energy unavailable"}

	for zone in zones:
		domain = zone.get("domain")
		if domain == "cpu_package":
			key = "cpu_package"
		elif domain == "dram":
			key = "dram"
		else:
			continue
		if zone.get("status") == SUPPORTED:
			domains[key] = {"status": SUPPORTED, "source": zone["name"]}
		elif domains[key]["status"] == UNAVAILABLE:
			domains[key] = {"status": zone.get("status", UNKNOWN), "source": zone["name"]}

	available = [d for d in BOUNDARY_DOMAINS if domains[d]["status"] == SUPPORTED]
	missing = [d for d in BOUNDARY_DOMAINS if domains[d]["status"] != SUPPORTED]
	return {
		"domains": domains,
		"available_energy_domains": available,
		"missing_domains": missing,
		"measurement_complete_possible": not missing,
		"detail": ("all additive domains available: measured_energy_j can be non-null"
		           if not missing else
		           "measured_energy_j will be null on this host; missing: "
		           + ", ".join(missing)),
	}


def build_report(settle=0.5, resolution_samples=3000):
	host = probe_host()
	nvml = probe_nvml(settle=max(settle, 1.0))
	powercap = discover_powercap_zones()
	amd = discover_amd_hwmon_zones()
	raw_zones = powercap + amd
	zones = [probe_zone(z, settle=settle) for z in raw_zones]
	verdict = boundary_verdict(nvml, zones)
	resolution = (probe_nvml_resolution(samples=resolution_samples)
	              if nvml.get("cumulative_energy_status") == SUPPORTED
	              else {"status": UNAVAILABLE, "detail": "no GPU energy counter"})

	return {
		"schema": "c2-hardware-report/1",
		"generated_at": time.time(),
		"host": host,
		"gpu": nvml,
		"gpu_counter_resolution": resolution,
		"rapl_zones": zones,
		"rapl_zone_count": len(zones),
		"boundary": verdict,
		"node_telemetry": probe_node_telemetry(),
		"notes": [
			"RAPL core and uncore are diagnostic only and are never added to "
			"measured_energy_j; core is contained within package.",
			"psys is excluded because it is a platform domain that contains package.",
			"A domain reported UNAVAILABLE must remain null downstream, never zero.",
			"CodeCarbon is not independent validation: it reads NVML and RAPL itself.",
		],
	}


# --------------------------------------------------------------------------
# rendering
# --------------------------------------------------------------------------

def _decode(value):
	if isinstance(value, bytes):
		return value.decode("utf-8", "replace")
	return value


def _median(values):
	ordered = sorted(values)
	return ordered[len(ordered) // 2] if ordered else None


def _which(name):
	for directory in os.environ.get("PATH", "").split(os.pathsep):
		candidate = os.path.join(directory, name)
		if os.path.isfile(candidate) and os.access(candidate, os.X_OK):
			return candidate
	return None


def render(report):
	L = []
	host = report["host"]
	L.append("=" * 72)
	L.append("C2 MEASUREMENT HARDWARE CAPABILITY REPORT")
	L.append("=" * 72)
	L.append(f"host          {host['hostname']}  ({host['system']} {host['release']})")
	L.append(f"environment   {host['environment']}"
	         f"{'  [NOT bare metal]' if not host['bare_metal'] else ''}")
	L.append(f"cpu           {host['cpu_model']}")
	L.append("")

	gpu = report["gpu"]
	L.append(f"-- GPU / NVML: {gpu['status']} --")
	L.append(f"   driver {gpu.get('driver_version')}   {gpu.get('detail', '')}")
	for d in gpu.get("devices", []):
		L.append(f"   gpu{d['index']}  {d.get('name')}")
		L.append(f"      cumulative energy   {d.get('energy_status')}"
		         f"   delta={d.get('energy_delta_mj')} mJ"
		         f"   advancing={d.get('energy_advancing')}")
		L.append(f"      instantaneous power {d.get('power_status')}"
		         f"   {d.get('power_w')} W")
	res = report["gpu_counter_resolution"]
	if res.get("status") == SUPPORTED:
		L.append(f"   counter resolution  median update "
		         f"{res['median_update_interval_s'] * 1000:.1f} ms, "
		         f"median step {res['median_update_step_mj']} mJ")
		L.append(f"      -> {res['detail']}")
	else:
		L.append(f"   counter resolution  {res.get('status')}: {res.get('detail')}")
	L.append("")

	L.append(f"-- RAPL / CPU zones: {report['rapl_zone_count']} found --")
	if not report["rapl_zones"]:
		L.append("   UNAVAILABLE  no powercap zones and no amd_energy hwmon zones")
		L.append("                (expected under WSL2 / containers / locked nodes)")
	for z in report["rapl_zones"]:
		L.append(f"   {z['name']:<16} node={z['node']:<20} domain={z['domain']:<12}"
		         f" {z['status']}")
		L.append(f"      column={z['column']}")
		if z.get("delta_uj") is not None:
			L.append(f"      delta={z['delta_uj']} uJ  advancing={z.get('advancing')}"
			         f"  mode={z.get('mode')}")
		if z.get("detail"):
			L.append(f"      {z['detail']}")
	L.append("")

	b = report["boundary"]
	L.append("-- MEASUREMENT BOUNDARY  (measured = gpu + cpu_package + dram) --")
	for domain in BOUNDARY_DOMAINS:
		info = b["domains"][domain]
		L.append(f"   {domain:<14} {info['status']:<12} {info.get('source') or ''}")
	L.append("")
	L.append(f"   measurement_complete possible on this host: "
	         f"{b['measurement_complete_possible']}")
	L.append(f"   -> {b['detail']}")
	L.append("")

	tel = report["node_telemetry"]
	L.append(f"-- INDEPENDENT NODE TELEMETRY: {tel['status']} --")
	L.append(f"   {tel['detail']}")
	for source in tel.get("sources", []):
		L.append(f"   candidate: {source['kind']}  {source['detail']}")
	L.append("")
	L.append("-- NOTES --")
	for note in report["notes"]:
		L.append(f"   * {note}")
	return "\n".join(L)


def main():
	ap = argparse.ArgumentParser(description=__doc__,
	                             formatter_class=argparse.RawDescriptionHelpFormatter)
	ap.add_argument("--json", default="", help="also write the report as JSON here")
	ap.add_argument("--settle", type=float, default=0.5,
	                help="seconds between counter reads when probing")
	ap.add_argument("--resolution-samples", type=int, default=3000,
	                help="polls used to characterize NVML counter update rate")
	ap.add_argument("--quiet", action="store_true", help="suppress the text report")
	args = ap.parse_args()

	report = build_report(settle=args.settle,
	                      resolution_samples=args.resolution_samples)
	if not args.quiet:
		print(render(report))
	if args.json:
		os.makedirs(os.path.dirname(os.path.abspath(args.json)), exist_ok=True)
		with open(args.json, "w") as f:
			json.dump(report, f, indent=2)
		print(f"\nJSON report -> {args.json}")

	# Exit 0 always: an incomplete boundary is a finding, not a tool failure.
	return 0


if __name__ == "__main__":
	sys.exit(main())
