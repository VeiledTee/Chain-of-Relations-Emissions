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

_RAPL = sorted(glob.glob("/sys/class/powercap/intel-rapl:*/energy_uj")
               + glob.glob("/sys/class/powercap/intel-rapl:*:*/energy_uj"))


def _rapl_names():
	names = []
	for path in _RAPL:
		d = os.path.dirname(path)
		try:
			name = open(os.path.join(d, "name")).read().strip()
		except Exception:
			name = os.path.basename(d)
		names.append(f"rapl_{name}_{os.path.basename(d)}_uj")
	return names


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
