"""Joules -> CO2e, as an explicit, auditable conversion. No estimation.

The measurement chain this package implements is:

    hardware counters  ->  Joules  ->  carbon-intensity conversion  ->  CO2e

Only the first arrow is a measurement. This module implements the last one, and
it is deliberately dumb: it multiplies energy by a carbon intensity the CALLER
supplied, and records where that intensity came from.

What this module will not do, on purpose:

  * look up a grid factor over the network;
  * fall back to a regional or global average when none is given;
  * infer a region from the host, the timezone or an IP address.

A silently defaulted grid factor is the single easiest way to publish a
confident carbon number that means nothing. If the caller does not supply an
intensity, there is no CO2e -- the energy measurement stands on its own.

Nor is CodeCarbon (or any other estimator) treated as ground truth here. Where
CodeCarbon estimates CPU power from a TDP table it is not an independent
measurement of the same quantity, and must not be used to validate one.
"""

from dataclasses import asdict, dataclass
from typing import Optional

JOULES_PER_KWH = 3.6e6


@dataclass(frozen=True)
class CarbonIntensity:
	"""A grid carbon intensity together with the provenance to defend it.

	value_g_co2e_per_kwh  the factor itself
	region                what it applies to, in the source's own vocabulary
	                      (e.g. "GB", "US-CAL-CISO"); never inferred
	source                who published it (e.g. "NationalGrid ESO", a DOI,
	                      a filename)
	reference_timestamp   ISO-8601 instant the factor describes
	retrieved_timestamp   ISO-8601 instant it was obtained
	method                "average" or "marginal" -- these answer different
	                      questions and must not be mixed in one analysis
	notes                 anything a reader needs to reproduce the choice
	"""

	value_g_co2e_per_kwh: float
	region: str
	source: str
	reference_timestamp: Optional[str] = None
	retrieved_timestamp: Optional[str] = None
	method: str = "average"
	notes: str = ""
	units: str = "gCO2e/kWh"

	def as_provenance(self) -> dict:
		return asdict(self)


def joules_to_kwh(joules):
	"""Energy unit conversion. None propagates: missing energy is not zero."""
	if joules is None:
		return None
	return joules / JOULES_PER_KWH


def to_co2e_g(joules, intensity: CarbonIntensity):
	"""Grams CO2e for a measured energy, or None if the energy is unmeasured.

	Returns None rather than 0.0 when `joules` is None: an unmeasured domain
	must not become a confident zero-emission claim.
	"""
	kwh = joules_to_kwh(joules)
	if kwh is None:
		return None
	return kwh * float(intensity.value_g_co2e_per_kwh)


def convert(joules, intensity: CarbonIntensity) -> dict:
	"""One auditable conversion record: inputs, output and full provenance."""
	return {
		"energy_j": joules,
		"energy_kwh": joules_to_kwh(joules),
		"co2e_g": to_co2e_g(joules, intensity),
		"carbon_intensity": intensity.as_provenance(),
	}


def convert_events(attributed_events, intensity: CarbonIntensity,
                   energy_field="measured_energy_j"):
	"""Add co2e_g to attributed events, leaving every other field untouched.

	Events whose `energy_field` is null keep a null co2e_g. The carbon
	intensity's provenance is returned once alongside, not copied onto every
	row.
	"""
	rows = []
	for event in attributed_events:
		row = dict(event)
		row["co2e_g"] = to_co2e_g(event.get(energy_field), intensity)
		row["co2e_energy_field"] = energy_field
		rows.append(row)
	return rows, intensity.as_provenance()
