"""NVML GPU access: the cumulative energy counter, plus diagnostics.

Two readings, deliberately not interchangeable:

  nvmlDeviceGetTotalEnergyConsumption   cumulative energy (mJ)  -> INSTRUMENT
  nvmlDeviceGetPowerUsage               instantaneous power (mW) -> DIAGNOSTIC

The cumulative counter is a monotonic hardware register. Differencing it across
a window gives energy directly, so even a sub-millisecond span gets a real
Joule figure, and the in-band event log and the out-of-band sampler read the
same register -- which is what makes trajectory totals reconcilable against the
sum of their events.

Integrating sampled power is NOT equivalent. At ~10 Hz it resolves fast
inference transients poorly and errs in both directions, with error growing as
windows shorten.

Everything here is read-only observation. No power capping, no clock changes,
no intervention in the workload. Where the counter is unsupported (pre-Volta
parts, some WSL2 configurations) every accessor returns None; the caller must
report that as unavailable and never substitute an estimate.
"""

_pynvml = None
_handles = []
_ok = False
_tried = False


def init() -> bool:
	"""Initialise NVML once. Returns whether the library came up.

	Callers that must not touch the GPU when measurement is disabled should
	gate their own call to this; it is not called implicitly at import.
	"""
	global _pynvml, _handles, _ok, _tried
	if _tried:
		return _ok
	_tried = True
	try:
		import pynvml
		pynvml.nvmlInit()
		count = pynvml.nvmlDeviceGetCount()
		_pynvml = pynvml
		_handles = [pynvml.nvmlDeviceGetHandleByIndex(i) for i in range(count)]
		_ok = True
	except Exception:
		_ok = False
	return _ok


def reset() -> None:
	"""Forget the initialisation attempt (tests, re-probing)."""
	global _pynvml, _handles, _ok, _tried
	_pynvml, _handles, _ok, _tried = None, [], False, False


def available() -> bool:
	return _ok


def handles():
	return list(_handles)


def device_count() -> int:
	return len(_handles)


def cumulative_energy_mj():
	"""Cumulative device energy (mJ) summed across GPUs, or None.

	None means "this host does not provide the counter", never zero energy.
	"""
	if not _ok:
		return None
	try:
		return sum(_pynvml.nvmlDeviceGetTotalEnergyConsumption(h)
		           for h in _handles)
	except Exception:
		return None


def device_energy_mj(handle):
	"""Cumulative energy (mJ) for one device, or None."""
	if not _ok:
		return None
	try:
		return _pynvml.nvmlDeviceGetTotalEnergyConsumption(handle)
	except Exception:
		return None


def device_power_w(handle):
	"""Instantaneous board power (W) for one device, or None. Diagnostic only."""
	if not _ok:
		return None
	try:
		return _pynvml.nvmlDeviceGetPowerUsage(handle) / 1000.0  # mW -> W
	except Exception:
		return None


def energy_counter_supported() -> bool:
	"""Whether the cumulative energy register is actually readable here."""
	if not _ok:
		return False
	for h in _handles:
		try:
			_pynvml.nvmlDeviceGetTotalEnergyConsumption(h)
			return True
		except Exception:
			return False
	return False


def device_names():
	"""Model name of each visible GPU; empty list when NVML is unavailable."""
	if not _ok:
		return []
	names = []
	try:
		for h in _handles:
			name = _pynvml.nvmlDeviceGetName(h)
			if isinstance(name, bytes):
				name = name.decode("utf-8", "replace")
			names.append(str(name))
	except Exception:
		return names
	return names
