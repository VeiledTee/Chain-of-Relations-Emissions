"""Operation-label registry.

The profiler measures spans; the HOST decides what a span means. This module
is the whole of that contract.

A host may:

  * register its vocabulary, so that a label outside it is flagged
    (meta.unknown_label = true) instead of being silently accepted; or
  * register nothing, in which case any caller-supplied label is accepted.

Either way the profiler NEVER derives a label. It does not read prompts, does
not inspect function names, does not walk the stack, and does not pattern-match
on arguments. A label arrives from the caller that knows why the work is
happening, or the event is recorded with whatever the caller passed.

Registering a vocabulary is an integrity aid, not a gate: an unregistered label
still produces an event. Dropping the event would lose the measurement, which
is worse for research integrity than recording it flagged.
"""


class LabelRegistry:
	"""Known operation labels and types for one host vocabulary."""

	def __init__(self):
		self._labels = frozenset()
		self._types = frozenset()

	def register(self, labels=(), types=()) -> None:
		"""Declare the labels (and optionally the type prefixes) a host emits.

		Replaces any previous registration. Values may be plain strings or
		str-valued enum members.
		"""
		self._labels = frozenset(str(item) for item in labels)
		self._types = frozenset(str(item) for item in types)

	def clear(self) -> None:
		self._labels = frozenset()
		self._types = frozenset()

	@property
	def registered(self) -> bool:
		return bool(self._labels)

	def known_labels(self) -> frozenset:
		return self._labels

	def is_known(self, label) -> bool:
		"""True when no vocabulary is registered, or the label is in it."""
		return (not self._labels) or str(label) in self._labels

	def resolve(self, label):
		"""Return (operation_type, is_unknown) for a caller-supplied label.

		operation_type is the structural prefix before ":" (see
		schema.type_of_label) -- never a guess. is_unknown is True only when a
		vocabulary is registered and the label, or its type prefix, is not in
		it.
		"""
		from . import schema

		text = str(label)
		operation_type = schema.type_of_label(text)
		unknown = False
		if self._labels and text not in self._labels:
			unknown = True
		elif self._types and operation_type not in self._types:
			unknown = True
		return operation_type, unknown


#: Process-wide registry the recorder consults. A host installs its vocabulary
#: once, from its adapter module.
REGISTRY = LabelRegistry()


def register(labels=(), types=()) -> None:
	REGISTRY.register(labels, types)


def clear() -> None:
	REGISTRY.clear()


def is_known(label) -> bool:
	return REGISTRY.is_known(label)
