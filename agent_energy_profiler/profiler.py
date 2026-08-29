"""The measurement API a host agent uses. Thin facade over `events`.

    from agent_energy_profiler import profiler

    profiler.configure(
        events_file="events.jsonl",
        run_id="run-1", dataset="webqsp", paradigm="cor",
        model_name="gemma-3-4b-it", model_revision="<sha>",
        git_commit=..., hardware_id=...,
    )

    profiler.set_question("q-17")
    profiler.begin_iteration()
    profiler.set_traversal_depth(2)

    with profiler.span("llm:reason", attributes={"tool_attempt": 1}) as s:
        response = client.generate(prompt)
        s["input_tokens"] = response.usage.prompt_tokens
        s["output_tokens"] = response.usage.completion_tokens

`span` does not know what "llm:reason" means, and never tries to find out. The
label comes from the caller that knows why the work is happening. A host that
wants its vocabulary policed registers it once:

    from agent_energy_profiler import labels
    labels.register(labels=[...], types=["llm", "kg"])

after which a label outside the vocabulary is still recorded, but flagged with
meta.unknown_label so it shows up in analysis rather than disappearing.

Three trajectory orderings are tracked and must not be conflated:

    iteration        control-loop iteration; never decreases within a question
    traversal_depth  search depth being expanded; rises and falls on backtrack,
                     null for work outside the traversal
    step_index       measured-event index; counts measurements, not decisions

Measurement is best-effort by construction: recording never raises, so an
instrumentation bug cannot break the workload being measured.
"""

from .events import (
	begin_iteration,
	close,
	configure,
	current_event_meta,
	current_iteration,
	current_operation,
	current_question,
	current_traversal_depth,
	detect_git_commit,
	detect_hardware_id,
	enabled,
	event_meta,
	gpu_energy_mj,
	iteration,
	mark,
	now,
	operation,
	record,
	reset_from_env,
	run_context,
	set_iteration,
	set_question,
	set_traversal_depth,
	span,
	traversal_depth,
)

__all__ = [
	"begin_iteration", "close", "configure", "current_event_meta",
	"current_iteration", "current_operation", "current_question",
	"current_traversal_depth", "detect_git_commit", "detect_hardware_id",
	"enabled", "event_meta", "gpu_energy_mj", "iteration", "mark", "now",
	"operation", "record", "reset_from_env", "run_context", "set_iteration",
	"set_question", "set_traversal_depth", "span", "traversal_depth",
]
