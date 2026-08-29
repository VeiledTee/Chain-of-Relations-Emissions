"""Event schema v1: the generic, host-independent part.

This module owns the *shape* of a measured event. It does not own the
*meaning* of any operation label -- that belongs to the host agent, which
registers its own vocabulary through `labels.LabelRegistry`.

The only structural rule the profiler imposes on a label is that it reads
"<type>:<detail>", so that operation_type is always recoverable from
operation_label and the two can never disagree. Anything before the colon is
accepted as a type; the profiler does not maintain a closed list of types,
because a host may measure work this package has never heard of.

Schema v1 is FROZEN. Field names, units and null semantics must not change:
existing measured artifacts are research data. In particular:

  * a domain the hardware did not report stays null, never 0.0;
  * measured_energy_j = gpu + cpu_package + dram, and is null if any term is;
  * cpu core/uncore are diagnostic and never enter a total.

Two field names in schema v1 carry vocabulary from the KGQA host that first
used it -- `question_id` (the trajectory grouping key) and `dataset` /
`paradigm` (run provenance). They are generic in role and are documented as
such rather than renamed, because renaming a frozen schema field would
invalidate existing artifacts.
"""

from enum import Enum

#: Schema version stamped onto every event. Frozen at 1.
SCHEMA_VERSION = 1


class _Str(str, Enum):
	"""str-valued enum that serializes and compares as its plain value."""

	def __str__(self) -> str:
		return self.value


class Status(_Str):
	"""Controlled terminal status vocabulary for a measured span.

	Retries collapsed inside one span belong in meta (e.g. meta["attempts"]),
	not here: this records how the span *ended*.
	"""

	OK = "ok"
	ERROR = "error"
	TIMEOUT = "timeout"
	CANCELLED = "cancelled"


_STATUS_VALUES = frozenset(item.value for item in Status)

#: Canonical schema-v1 field order. Every event carries all of these keys;
#: unmeasured values are null.
EVENT_FIELDS = (
	"schema_version", "event_id",
	"run_id", "question_id", "dataset", "paradigm",
	"iteration", "traversal_depth", "step_index",
	"operation_type", "operation_label",
	"start_timestamp", "end_timestamp", "duration_s",
	"gpu_energy_j", "cpu_package_energy_j", "dram_energy_j",
	"input_tokens", "output_tokens",
	"status",
	"model_name", "model_revision", "git_commit", "hardware_id",
	"meta",
)

#: Run-scoped provenance keys, configured once per run by the host.
RUN_CONTEXT_FIELDS = (
	"run_id", "dataset", "paradigm",
	"model_name", "model_revision", "git_commit", "hardware_id",
)

# --- pre-schema-v1 compatibility ------------------------------------------
# The only host-flavoured thing this generic module still emits. Events carry
# duplicate legacy keys (category/label/t_start/t_end) so readers written
# before schema v1 keep working against new artifacts. Set EMIT_LEGACY_FIELDS
# to False in a host that has no such readers; it changes nothing about the
# canonical schema-v1 keys.
EMIT_LEGACY_FIELDS = True
LEGACY_CATEGORY_INFERENCE = "inference"
LEGACY_CATEGORY_TOOL = "tool"


def is_known_status(status) -> bool:
	return str(status) in _STATUS_VALUES


def type_of_label(label) -> str:
	"""Operation type implied by a label, by the structural "type:detail" rule.

	Returns "" for a label with no type prefix. This is pure string structure:
	the profiler never infers meaning from prompt text, function names, stack
	frames or any other heuristic.
	"""
	text = str(label)
	return text.split(":", 1)[0] if ":" in text else ""


def legacy_category(operation_type: str) -> str:
	"""The pre-schema-v1 category an operation type used to be filed under."""
	return (LEGACY_CATEGORY_INFERENCE if operation_type == "llm"
	        else LEGACY_CATEGORY_TOOL)
