"""Whole-machine power sampler. Run on the HOST (not in a container).

Samples at ~10 Hz:
  - GPU power (W) via NVML, per GPU            -> works on Linux and WSL2
  - CPU package + DRAM energy (uJ) via RAPL    -> bare-metal Linux only
    (silently absent on WSL2 / locked HPC nodes; GPU columns still logged)

Output CSV columns: t, gpu<i>_w ..., rapl_<domain>_uj ...
RAPL counters are cumulative energy; attribute.py differences them.

Usage:
  python measurement/power_logger.py --out power.csv          # Ctrl-C to stop
  python measurement/power_logger.py --out power.csv --hz 10
"""

import argparse
import csv
import glob
import os
import signal
import sys
import time

try:
	import pynvml
	pynvml.nvmlInit()
	_NGPU = pynvml.nvmlDeviceGetCount()
	_HANDLES = [pynvml.nvmlDeviceGetHandleByIndex(i) for i in range(_NGPU)]
except Exception:
	_NGPU, _HANDLES = 0, []

def _discover_rapl():
	"""Every readable CPU energy counter, whatever the vendor exposes.

	Not hardcoded to intel-rapl: some hosts expose their RAPL MSRs through the
	amd_energy *hwmon* driver rather than powercap, and globbing only
	intel-rapl would make such a host look identical to one with no CPU energy
	counters at all. Whatever is found is labelled by its own domain name;
	attribute.py decides which domains are additive. No CPU vendor is assumed.
	"""
	paths = sorted(glob.glob("/sys/class/powercap/*/energy_uj"))
	for hwmon in sorted(glob.glob("/sys/class/hwmon/hwmon*")):
		try:
			if open(os.path.join(hwmon, "name")).read().strip() != "amd_energy":
				continue
		except Exception:
			continue
		paths.extend(sorted(glob.glob(os.path.join(hwmon, "energy*_input"))))
	return paths


_RAPL = _discover_rapl()


def _zone_name(path):
	directory = os.path.dirname(path)
	# powercap zones carry a sibling `name`; amd_energy hwmon uses per-input labels
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


def _rapl_names():
	return [f"rapl_{_zone_name(path)}_{os.path.basename(os.path.dirname(path))}_uj"
	        for path in _RAPL]


def main():
	ap = argparse.ArgumentParser()
	ap.add_argument("--out", required=True)
	ap.add_argument("--hz", type=float, default=10.0)
	args = ap.parse_args()

	period = 1.0 / args.hz
	stop = {"flag": False}
	signal.signal(signal.SIGINT, lambda *_: stop.update(flag=True))
	signal.signal(signal.SIGTERM, lambda *_: stop.update(flag=True))

	header = ["t"] + [f"gpu{i}_w" for i in range(_NGPU)] + _rapl_names()
	if _NGPU == 0:
		print("WARNING: no NVML GPUs visible", file=sys.stderr)
	if not _RAPL:
		print("WARNING: no RAPL domains readable (WSL2/shared node/permissions?)"
		      " - GPU-only logging", file=sys.stderr)

	with open(args.out, "w", newline="") as f:
		w = csv.writer(f)
		w.writerow(header)
		while not stop["flag"]:
			row = [f"{time.time():.3f}"]
			for h in _HANDLES:
				try:
					row.append(pynvml.nvmlDeviceGetPowerUsage(h) / 1000.0)  # mW -> W
				except Exception:
					row.append("")
			for path in _RAPL:
				try:
					row.append(open(path).read().strip())
				except Exception:
					row.append("")
			w.writerow(row)
			f.flush()
			time.sleep(period)
	print(f"power log written: {args.out}")


if __name__ == "__main__":
	main()
