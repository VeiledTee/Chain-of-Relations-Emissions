"""Hardware capability probe for the measurement instrument.

Answers one question: which parts of the measurement boundary can this
host actually measure?

    measured_energy_j = gpu_energy_j + cpu_package_energy_j + dram_energy_j

Every domain is reported with an explicit status. Nothing missing is ever
reported as zero.

    SUPPORTED    present and readable, values advance, values are credible
    INCONSISTENT counter advances but disagrees grossly with independent
                 physical evidence; present but not usable as a measurement
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

# Zone discovery and domain classification live in hardware.discovery; they are
# re-exported here because they are part of this module's published surface.
from .hardware.discovery import (  # noqa: E402
	BOUNDARY_DOMAINS,
	INCONSISTENT,
	DIAGNOSTIC_DOMAINS,
	EXCLUDED_DOMAINS,
	SUPPORTED,
	UNAVAILABLE,
	UNKNOWN,
	UNREADABLE,
	classify_zone_name,
	discover_amd_hwmon_zones,
	discover_powercap_zones,
)


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


# --------------------------------------------------------------------------
# GPU cumulative-counter consistency
# --------------------------------------------------------------------------
#
# An advancing counter is not the same as a correct counter. Some drivers and
# some parts (observed on a laptop GPU under bare-metal Linux) return a
# nvmlDeviceGetTotalEnergyConsumption register that increments steadily but at
# a rate that no reading of board power can account for. Such a counter passes
# the "present and advancing" probe and would silently poison every joule in
# the study.
#
# The cross-check integrates the *diagnostic* instantaneous-power signal over
# the same window and asks whether the counter is of the right physical size.
# Sampled power is used only as a plausibility yardstick here: it never becomes
# a measurement, is never substituted for a missing counter, and a host that
# passes this check still reports energy exclusively from the counter.

#: Default cross-check window. Long enough that counter-update quantisation and
#: sampling phase are small next to the integral, short enough to run in a probe.
CONSISTENCY_WINDOW_S = 10.0
#: Power poll spacing. ~20 Hz: well inside NVML's own power refresh rate.
CONSISTENCY_SAMPLE_INTERVAL_S = 0.05

#: Relative band. Trapezoidal integration of a ~10 Hz-refreshed power reading
#: against a hardware integrator is a genuinely noisy comparison: transients
#: between polls are missed in both directions, the power reading is itself a
#: driver-side average over an unspecified window, and board power and the
#: energy register need not cover byte-identical rails. Half an order of
#: magnitude of disagreement is therefore tolerated, which still bounds the
#: counter to within a factor of 1.5. Grossly invalid counters observed in
#: practice are off by multiples, not by tens of percent.
CONSISTENCY_REL_TOL = 0.50
#: Absolute allowance for edge effects: the two counter reads do not coincide
#: with the first and last power samples, and the counter advances in discrete
#: steps, so a slice of real work can fall on one side of the comparison only.
#: Bounded by peak observed board power (falling back to the average) times this
#: slop -- peak, because on a mostly-idle window with bursts at the edges the
#: misalignment is worth burst power, not mean power. Always scaled by the
#: sampled signal, never by the counter, so an inflated counter cannot widen its
#: own tolerance.
CONSISTENCY_EDGE_SLOP_S = 0.25
#: Below these the comparison says nothing; report UNKNOWN rather than a verdict.
CONSISTENCY_MIN_WINDOW_S = 5.0
CONSISTENCY_MIN_SAMPLES = 20
CONSISTENCY_MIN_ENERGY_J = 1.0


def integrate_power(samples):
	"""Trapezoidal integral of (timestamp_s, power_w) pairs, in joules.

	Diagnostic only. The result is never reported as measured GPU energy.
	"""
	total = 0.0
	for (ta, pa), (tb, pb) in zip(samples, samples[1:]):
		total += (tb - ta) * (pa + pb) / 2.0
	return total


def assess_counter_consistency(counter_j, sampled_j, elapsed_s, n_samples,
                               sample_interval_s=CONSISTENCY_SAMPLE_INTERVAL_S,
                               peak_power_w=None):
	"""Is the cumulative energy counter physically credible on this host?

	Pure function so the verdict can be tested against recorded hardware cases
	without a GPU present.

	Verdicts:
	  SUPPORTED     counter and integrated power agree within tolerance
	  INCONSISTENT  counter advances but disagrees grossly: not usable
	  UNKNOWN       window/signal too small to judge; no claim either way
	"""
	result = {
		"counter_energy_j": counter_j,
		"sampled_energy_j": sampled_j,
		"elapsed_s": elapsed_s,
		"n_samples": n_samples,
	}
	if counter_j is None or sampled_j is None:
		result.update(verdict=UNKNOWN, detail="counter or sampled power unavailable")
		return result
	if elapsed_s < CONSISTENCY_MIN_WINDOW_S or n_samples < CONSISTENCY_MIN_SAMPLES:
		result.update(verdict=UNKNOWN,
		              detail=(f"window too short to judge: {elapsed_s:.2f}s / "
		                      f"{n_samples} samples"))
		return result
	if sampled_j < CONSISTENCY_MIN_ENERGY_J or counter_j < CONSISTENCY_MIN_ENERGY_J:
		result.update(verdict=UNKNOWN,
		              detail=(f"too little energy in the window to judge: "
		                      f"counter {counter_j:.3f} J, sampled {sampled_j:.3f} J"))
		return result

	sampled_avg_w = sampled_j / elapsed_s
	edge_slop_s = CONSISTENCY_EDGE_SLOP_S + 2.0 * sample_interval_s
	edge_power_w = max(peak_power_w or 0.0, sampled_avg_w)
	abs_tol_j = edge_power_w * edge_slop_s
	tolerance_j = max(CONSISTENCY_REL_TOL * sampled_j, abs_tol_j)
	difference_j = abs(counter_j - sampled_j)

	result.update(
		counter_avg_power_w=counter_j / elapsed_s,
		sampled_avg_power_w=sampled_avg_w,
		ratio=counter_j / sampled_j,
		relative_difference_pct=difference_j / sampled_j * 100.0,
		difference_j=difference_j,
		tolerance_j=tolerance_j,
		rel_tol=CONSISTENCY_REL_TOL,
		abs_tol_j=abs_tol_j,
		peak_power_w=peak_power_w,
	)
	if difference_j <= tolerance_j:
		result.update(
			verdict=SUPPORTED,
			detail=(f"counter {counter_j:.3f} J vs integrated power "
			        f"{sampled_j:.3f} J: {result['relative_difference_pct']:.1f}% "
			        f"apart, within tolerance {tolerance_j:.3f} J"))
	else:
		result.update(
			verdict=INCONSISTENT,
			detail=(f"counter {counter_j:.3f} J is {result['ratio']:.3f}x the "
			        f"integrated power {sampled_j:.3f} J "
			        f"({result['relative_difference_pct']:.1f}% apart, tolerance "
			        f"{tolerance_j:.3f} J): counter advances but is not "
			        f"physically credible; GPU energy is not usable on this host"))
	return result


def probe_gpu_counter_consistency(handle=None,
                                  window_s=CONSISTENCY_WINDOW_S,
                                  sample_interval_s=CONSISTENCY_SAMPLE_INTERVAL_S):
	"""Cross-check device 0's energy counter against integrated board power."""
	try:
		import pynvml
		pynvml.nvmlInit()
		h = handle if handle is not None else pynvml.nvmlDeviceGetHandleByIndex(0)
		name = _decode(pynvml.nvmlDeviceGetName(h))
	except Exception as e:
		return {"verdict": UNKNOWN, "detail": f"NVML unavailable: {e}"}

	samples = []
	try:
		start_mj = pynvml.nvmlDeviceGetTotalEnergyConsumption(h)
		t0 = time.monotonic()
		while time.monotonic() - t0 < window_s:
			samples.append((time.monotonic(),
			                pynvml.nvmlDeviceGetPowerUsage(h) / 1000.0))
			time.sleep(sample_interval_s)
		t1 = time.monotonic()
		end_mj = pynvml.nvmlDeviceGetTotalEnergyConsumption(h)
	except Exception as e:
		return {"verdict": UNKNOWN, "detail": f"probe read failed: {e}",
		        "device": name}

	result = assess_counter_consistency(
		counter_j=(end_mj - start_mj) / 1000.0,
		sampled_j=integrate_power(samples),
		elapsed_s=t1 - t0,
		n_samples=len(samples),
		sample_interval_s=sample_interval_s,
		peak_power_w=max((p for _, p in samples), default=None),
	)
	result["device"] = name
	result["method"] = "cumulative counter vs trapezoidal integral of sampled power"
	return result


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

	consistency = nvml.get("counter_consistency") or {}
	if consistency.get("verdict") == INCONSISTENT:
		# The counter is present and advancing but not physically credible.
		# Advancing is not the same as correct: this host may not contribute
		# GPU energy to the boundary.
		domains["gpu"] = {"status": INCONSISTENT,
		                  "source": "nvml cumulative counter failed the "
		                            "power-consistency cross-check"}
	elif nvml.get("cumulative_energy_status") == SUPPORTED:
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


def build_report(settle=0.5, resolution_samples=3000,
                 consistency_window_s=CONSISTENCY_WINDOW_S):
	host = probe_host()
	nvml = probe_nvml(settle=max(settle, 1.0))
	if nvml.get("cumulative_energy_status") == SUPPORTED and consistency_window_s > 0:
		consistency = probe_gpu_counter_consistency(window_s=consistency_window_s)
		nvml["counter_consistency"] = consistency
		if consistency.get("verdict") == INCONSISTENT:
			# Keep the raw fact (the counter reads and advances) and mark the
			# capability separately: never report an unusable counter as SUPPORTED.
			nvml["cumulative_energy_status"] = INCONSISTENT
			nvml["status"] = INCONSISTENT
	powercap = discover_powercap_zones()
	amd = discover_amd_hwmon_zones()
	raw_zones = powercap + amd
	zones = [probe_zone(z, settle=settle) for z in raw_zones]
	verdict = boundary_verdict(nvml, zones)
	resolution = (probe_nvml_resolution(samples=resolution_samples)
	              if nvml.get("cumulative_energy_status") in (SUPPORTED, INCONSISTENT)
	              else {"status": UNAVAILABLE, "detail": "no GPU energy counter"})

	# Legacy identifier: reports generated before the project framing was made
	# standalone carry "c2-hardware-report/1". Same format, same version --
	# only the name changed. Historical report artifacts are left untouched, so
	# a consumer reading old and new reports together should accept both.
	return {
		"schema": "hardware-report/1",
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
			"An advancing GPU energy counter is cross-checked against the integral "
			"of sampled board power; a counter that disagrees grossly is reported "
			"INCONSISTENT and contributes no GPU energy to the boundary.",
			"Sampled power is a diagnostic plausibility yardstick only. It is never "
			"reported as measured energy and never substituted for a counter.",
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
	L.append("MEASUREMENT HARDWARE CAPABILITY REPORT")
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
	cons = gpu.get("counter_consistency")
	if cons:
		L.append(f"   counter consistency  {cons.get('verdict')}")
		if cons.get("ratio") is not None:
			L.append(f"      counter {cons['counter_energy_j']:.3f} J vs sampled "
			         f"{cons['sampled_energy_j']:.3f} J over "
			         f"{cons['elapsed_s']:.2f} s"
			         f"   ratio={cons['ratio']:.3f}x"
			         f"   rel_diff={cons['relative_difference_pct']:.1f}%"
			         f"   tol={cons['tolerance_j']:.3f} J")
		L.append(f"      -> {cons.get('detail')}")
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
	ap.add_argument("--consistency-window", type=float, default=CONSISTENCY_WINDOW_S,
	                help="seconds spent cross-checking the GPU energy counter "
	                     "against integrated sampled power (0 disables)")
	ap.add_argument("--quiet", action="store_true", help="suppress the text report")
	args = ap.parse_args()

	report = build_report(settle=args.settle,
	                      resolution_samples=args.resolution_samples,
	                      consistency_window_s=args.consistency_window)
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
