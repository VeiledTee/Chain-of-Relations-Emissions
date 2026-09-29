"""WebQSP answerability groups and the gold-answer outcome table built on them.

The 1,639 official WebQSP test questions split into three exclusive groups,
read from the committed `datasets/webqsp/webqsp_answerability_groups.csv`
(built once by `scripts/build_webqsp_answerability_groups.py`; provenance in
`datasets/README.md`). Nothing here queries the graph.

    empty_gold      11    `Answers: []` in the official release
    query_mismatch  11    stored gold query returns a different result on our Freebase
    expected        1617  stored gold query reproduces the gold answer

`load_groups` refuses a file that is not exactly that partition. Outcomes come
from the canonical `outcomes_{sys}.csv` of `measurement/export_outcomes.py`:
`hit` is "Gold answer found", `miss` is "Gold answer not found".
"""

import csv
import os
import re
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)

from chain_of_relations.eval import webqsp_canonical  # noqa: E402

DEFAULT_GROUPS_PATH = os.path.join(ROOT, "datasets", "webqsp", "webqsp_answerability_groups.csv")
EMPTY_GOLD, QUERY_MISMATCH, EXPECTED = "empty_gold", "query_mismatch", "expected"
GROUP_ORDER = (EMPTY_GOLD, QUERY_MISMATCH, EXPECTED)
GROUP_LABELS = {EMPTY_GOLD: "Official empty-gold", QUERY_MISMATCH: "Gold-query mismatch",
                EXPECTED: "Expected/reproducible"}
EXPECTED_COUNTS = {EMPTY_GOLD: 11, QUERY_MISMATCH: 11, EXPECTED: 1617}
N_QUESTIONS = 1639
SYSTEM_ORDER = ("PoG", "ToG", "CoR")
FOUND_OUTCOME, NOT_FOUND_OUTCOME = "hit", "miss"
FOUND_LABEL, NOT_FOUND_LABEL = "Gold answer found", "Gold answer not found"
FOUND_COLOUR, NOT_FOUND_COLOUR = "#0ca30c", "#d03b3b"


def question_sort_key(question_id):
	"""WebQTest-2 before WebQTest-10: numeric suffix, then the id itself."""
	match = re.search(r"(\d+)$", question_id)
	return (int(match.group(1)) if match else -1, question_id)


def official_question_ids(gold=None):
	return sorted((gold or webqsp_canonical.load_gold()).keys(), key=question_sort_key)


def official_empty_gold_ids(gold=None):
	"""The evaluator's own empty-gold rule: no answer on any official parse."""
	gold = gold or webqsp_canonical.load_gold()
	return {qid for qid in gold
	        if not any(webqsp_canonical.parse_answer_names(p)
	                   for p in webqsp_canonical.parses_for(qid, gold))}


def load_groups(path=DEFAULT_GROUPS_PATH, gold=None):
	"""{question_id: group}, validated as the exact 11 / 11 / 1,617 partition."""
	with open(path, newline="") as f:
		rows = list(csv.DictReader(f))
	if not rows or set(rows[0]) != {"question_id", "group"}:
		raise ValueError(f"{path}: expected columns question_id,group")
	groups = {}
	for row in rows:
		qid, group = row["question_id"], row["group"]
		if qid in groups:
			raise ValueError(f"{path}: duplicate question_id {qid}")
		if group not in GROUP_ORDER:
			raise ValueError(f"{path}: unknown group {group!r} for {qid}")
		groups[qid] = group
	official = set(official_question_ids(gold))
	if len(groups) != N_QUESTIONS or set(groups) != official:
		missing, extra = sorted(official - set(groups)), sorted(set(groups) - official)
		raise ValueError(f"{path}: {len(groups)} ids, expected the {N_QUESTIONS} official ids "
		                 f"(missing {missing[:5]}, unknown {extra[:5]})")
	counts = {g: sum(1 for v in groups.values() if v == g) for g in GROUP_ORDER}
	if counts != EXPECTED_COUNTS:
		raise ValueError(f"{path}: group counts {counts}, expected {EXPECTED_COUNTS}")
	empty = {qid for qid, g in groups.items() if g == EMPTY_GOLD}
	if empty != official_empty_gold_ids(gold):
		raise ValueError(f"{path}: empty_gold ids disagree with the official gold file")
	return groups


def members(groups, group):
	return sorted((q for q, g in groups.items() if g == group), key=question_sort_key)


def load_outcomes(path, groups):
	"""{question_id: found?} from a canonical outcome CSV covering every grouped id once."""
	found = {}
	with open(path, newline="") as f:
		for row in csv.DictReader(f):
			qid, outcome = row["question_id"], row["outcome"]
			if qid in found:
				raise ValueError(f"{path}: duplicate question_id {qid}")
			if outcome not in (FOUND_OUTCOME, NOT_FOUND_OUTCOME):
				raise ValueError(f"{path}: {qid} has outcome {outcome!r}, expected hit or miss")
			found[qid] = outcome == FOUND_OUTCOME
			empty_flag = str(row.get("empty_gold", "")).strip().lower() == "true"
			if qid in groups and empty_flag != (groups[qid] == EMPTY_GOLD):
				raise ValueError(f"{path}: {qid} empty_gold={row.get('empty_gold')} disagrees "
				                 f"with group {groups[qid]}")
	if set(found) != set(groups):
		raise ValueError(f"{path}: covers {len(found)} questions, not the {len(groups)} grouped ids")
	return found


def ordered_systems(names):
	"""PoG, ToG, CoR first, in that order; anything else after, alphabetically."""
	return [s for s in SYSTEM_ORDER if s in names] + sorted(set(names) - set(SYSTEM_ORDER))


def percent(found, n):
	return round(100.0 * found / n, 1) if n else None


def outcome_table(groups, found_by_system):
	"""One row per group: n, then found / not found / found % per system."""
	systems = ordered_systems(found_by_system)
	rows = []
	for group in GROUP_ORDER:
		ids = members(groups, group)
		row = {"Dataset group": GROUP_LABELS[group], "group": group, "n": len(ids)}
		for system in systems:
			found = sum(1 for q in ids if found_by_system[system][q])
			row[f"{system} found"] = found
			row[f"{system} not found"] = len(ids) - found
			row[f"{system} found %"] = percent(found, len(ids))
		rows.append(row)
	return rows, systems


def table_fields(systems):
	return (["Dataset group", "n"] +
	        [f"{s} {k}" for s in systems for k in ("found", "not found", "found %")])


def compact_table(rows, systems):
	return [dict({"Dataset group": f"{r['Dataset group']} (n={r['n']:,})"},
	             **{s: f"{r[f'{s} found']:,} / {r['n']:,} ({r[f'{s} found %']:.1f}%)"
	                for s in systems}) for r in rows]


def markdown(rows, fields):
	def cell(value):
		return f"{value:.1f}" if isinstance(value, float) else (
			f"{value:,}" if isinstance(value, int) else str(value))
	lines = ["| " + " | ".join(fields) + " |", "|" + "---|" * len(fields)]
	lines += ["| " + " | ".join(cell(r[f]) for f in fields) + " |" for r in rows]
	return "\n".join(lines) + "\n"


# ------------------------------------------------------------------ figure
#: Squares per grid row in every cell, so a square is the same size everywhere.
#: 1,617 = 49 x 33 exactly.
COLUMNS = 49
UNIT_IN, SQUARE = 0.066, 0.8


def draw_waffle(plt, groups, found_by_system, title="WebQSP Gold-Answer Outcomes by Dataset Group"):
	"""Grid of one square per question: rows are groups, columns are systems.

	Axes are placed in absolute inches so every square has identical size and
	spacing in every cell. Returns the figure.
	"""
	from matplotlib.collections import PatchCollection
	from matplotlib.patches import Patch, Rectangle

	systems = ordered_systems(found_by_system)
	grid_rows = {g: -(-len(members(groups, g)) // COLUMNS) for g in GROUP_ORDER}
	label_w, cell_w, col_gap = 2.3, COLUMNS * UNIT_IN, 0.35
	top, header, row_gap, bottom = 0.55, 0.3, 0.4, 0.6
	width = 0.2 + label_w + len(systems) * (cell_w + col_gap)
	height = top + header + sum(n * UNIT_IN + row_gap for n in grid_rows.values()) + bottom
	fig = plt.figure(figsize=(width, height))
	fig.suptitle(title, fontsize=11, fontweight="bold", y=1 - 0.25 / height)

	def place(x, y, w, h):  # inches from the top-left corner
		return fig.add_axes([x / width, 1 - (y + h) / height, w / width, h / height])

	for j, system in enumerate(systems):
		x = 0.2 + label_w + j * (cell_w + col_gap)
		fig.text((x + cell_w / 2) / width, 1 - (top + header / 2) / height, system,
		         ha="center", va="center", fontsize=10, fontweight="bold")
	y = top + header
	for group in GROUP_ORDER:
		ids, n_rows = members(groups, group), grid_rows[group]
		h = n_rows * UNIT_IN
		y += row_gap
		fig.text(0.2 / width, 1 - (y + h / 2) / height, f"{GROUP_LABELS[group]} (n={len(ids):,})",
		         ha="left", va="center", fontsize=9)
		for j, system in enumerate(systems):
			x = 0.2 + label_w + j * (cell_w + col_gap)
			ax = place(x, y, cell_w, h)
			colours = [FOUND_COLOUR if found_by_system[system][q] else NOT_FOUND_COLOUR
			           for q in ids]
			squares = [Rectangle((k % COLUMNS + (1 - SQUARE) / 2, k // COLUMNS + (1 - SQUARE) / 2),
			                     SQUARE, SQUARE) for k in range(len(ids))]
			ax.add_collection(PatchCollection(squares, facecolors=colours, edgecolors="none"))
			ax.set_xlim(0, COLUMNS)
			ax.set_ylim(n_rows, 0)
			ax.set_axis_off()
			found = sum(1 for q in ids if found_by_system[system][q])
			ax.text(0, -0.06 / UNIT_IN,f"{found:,} / {len(ids):,} found "
			        f"({percent(found, len(ids)):.1f}%)", ha="left", va="bottom", fontsize=7.5,
			        color="#444444")
		y += h
	fig.legend(handles=[Patch(color=FOUND_COLOUR, label=FOUND_LABEL),
	                    Patch(color=NOT_FOUND_COLOUR, label=NOT_FOUND_LABEL)],
	           loc="lower center", ncol=2, frameon=False, fontsize=9,
	           bbox_to_anchor=(0.5, 0.08 / height))
	return fig
