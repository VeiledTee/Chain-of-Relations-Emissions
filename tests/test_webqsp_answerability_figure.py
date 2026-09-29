"""WebQSP answerability groups and Figure 6.

The group file is the committed one; outcome CSVs are synthetic but cover the
real 1,639 ids, so every count below is checkable against the fixture itself.
"""

import csv
import hashlib
import os
import sys
import tempfile
import unittest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "measurement"))

import make_comparable_figures as mcf  # noqa: E402
import webqsp_answerability as wa  # noqa: E402
from agent_energy_profiler import visualize  # noqa: E402

GROUPS = wa.load_groups()


def write_outcomes(path, found, groups=GROUPS):
	with open(path, "w", newline="") as f:
		writer = csv.DictWriter(f, fieldnames=("question_id", "run", "outcome", "hit1", "empty_gold"),
		                        lineterminator="\n")
		writer.writeheader()
		for qid in sorted(groups):
			writer.writerow({"question_id": qid, "run": "T",
			                 "outcome": "hit" if found(qid) else "miss",
			                 "hit1": int(found(qid)), "empty_gold": groups[qid] == wa.EMPTY_GOLD})
	return path


def number(qid):
	return wa.question_sort_key(qid)[0]


#: Deterministic, distinct patterns per system (empty gold is never found, as scored).
PATTERNS = {"CoR": lambda q: GROUPS[q] != wa.EMPTY_GOLD and number(q) % 3 != 0,
            "ToG": lambda q: GROUPS[q] != wa.EMPTY_GOLD and number(q) % 2 == 0,
            "PoG": lambda q: GROUPS[q] == wa.EXPECTED and number(q) % 5 != 1}


def sha(paths):
	h = hashlib.sha256()
	for p in sorted(paths):
		h.update(open(p, "rb").read())
	return h.hexdigest()


class GroupFileTest(unittest.TestCase):
	def test_partition_is_11_11_1617(self):
		counts = {g: sum(1 for v in GROUPS.values() if v == g) for g in wa.GROUP_ORDER}
		self.assertEqual(counts, {wa.EMPTY_GOLD: 11, wa.QUERY_MISMATCH: 11, wa.EXPECTED: 1617})

	def test_every_official_id_exactly_once(self):
		with open(wa.DEFAULT_GROUPS_PATH, newline="") as f:
			ids = [r["question_id"] for r in csv.DictReader(f)]
		self.assertEqual(len(ids), 1639)
		self.assertEqual(len(set(ids)), 1639)
		self.assertEqual(set(ids), set(wa.official_question_ids()))

	def test_empty_gold_matches_official_release(self):
		self.assertEqual({q for q, g in GROUPS.items() if g == wa.EMPTY_GOLD},
		                 wa.official_empty_gold_ids())

	def _broken(self, mutate):
		with open(wa.DEFAULT_GROUPS_PATH, newline="") as f:
			rows = list(csv.DictReader(f))
		rows = mutate(rows)
		with tempfile.TemporaryDirectory() as tmp:
			path = os.path.join(tmp, "g.csv")
			with open(path, "w", newline="") as f:
				writer = csv.DictWriter(f, fieldnames=("question_id", "group"), lineterminator="\n")
				writer.writeheader()
				writer.writerows(rows)
			with self.assertRaises(ValueError):
				wa.load_groups(path)

	def test_rejects_duplicate_missing_and_wrong_counts(self):
		self._broken(lambda rows: rows + rows[:1])
		self._broken(lambda rows: rows[1:])
		def relabel(rows):
			for r in rows:
				if r["group"] == wa.QUERY_MISMATCH:
					r["group"] = wa.EXPECTED
					break
			return rows
		self._broken(relabel)


class Fig6Test(unittest.TestCase):
	def setUp(self):
		self.tmp = tempfile.TemporaryDirectory()
		self.inputs = os.path.join(self.tmp.name, "inputs")
		self.out = os.path.join(self.tmp.name, "out")
		os.makedirs(self.inputs)
		os.makedirs(self.out)
		# Insertion order deliberately not PoG, ToG, CoR.
		self.csvs = {name: write_outcomes(os.path.join(self.inputs, f"outcomes_{name.lower()}.csv"),
		                                  PATTERNS[name]) for name in ("CoR", "ToG", "PoG")}
		self.plt = visualize._pyplot()

	def tearDown(self):
		self.tmp.cleanup()

	def test_skipped_cleanly_for_cwq(self):
		self.assertIsNone(mcf.make_fig6(self.plt, self.out, "cwq", self.csvs))
		self.assertEqual(os.listdir(self.out), [])

	def test_produced_for_webqsp_without_touching_inputs(self):
		before = sha(self.csvs.values())
		written, rows, systems = mcf.make_fig6(self.plt, self.out, "webqsp", self.csvs)
		self.assertEqual(sha(self.csvs.values()), before)
		names = {os.path.basename(p) for p in written}
		for suffix in (".png", ".pdf", ".csv", "_compact.csv", "_questions.csv", ".md"):
			self.assertIn(mcf.FIG6_STEM + suffix, names)
			self.assertGreater(os.path.getsize(os.path.join(self.out, mcf.FIG6_STEM + suffix)), 0)
		self.assertEqual(set(os.listdir(self.out)), names)

	def test_system_order_and_table_counts_match_outcome_csvs(self):
		_, rows, systems = mcf.make_fig6(self.plt, self.out, "webqsp", self.csvs)
		self.assertEqual(systems, ["PoG", "ToG", "CoR"])
		with open(os.path.join(self.out, mcf.FIG6_STEM + ".csv"), newline="") as f:
			header = next(csv.reader(f))
		self.assertEqual(header, ["Dataset group", "n"] + [f"{s} {k}" for s in systems
		                                                   for k in ("found", "not found", "found %")])
		self.assertEqual([r["group"] for r in rows], list(wa.GROUP_ORDER))
		for row in rows:
			ids = [q for q, g in GROUPS.items() if g == row["group"]]
			self.assertEqual(row["n"], len(ids))
			for system in systems:
				with open(self.csvs[system], newline="") as f:
					hits = {r["question_id"] for r in csv.DictReader(f) if r["outcome"] == "hit"}
				expected = sum(1 for q in ids if q in hits)
				self.assertEqual(row[f"{system} found"], expected)
				self.assertEqual(row[f"{system} found"] + row[f"{system} not found"], row["n"])
				self.assertEqual(row[f"{system} found %"], round(100 * expected / len(ids), 1))

	def test_same_membership_for_every_system(self):
		mcf.make_fig6(self.plt, self.out, "webqsp", self.csvs)
		with open(os.path.join(self.out, mcf.FIG6_STEM + "_questions.csv"), newline="") as f:
			rows = list(csv.DictReader(f))
		self.assertEqual(len(rows), 1639)
		self.assertEqual({r["question_id"]: r["group"] for r in rows}, GROUPS)
		for system in ("PoG", "ToG", "CoR"):
			self.assertEqual(sum(1 for r in rows if r[system]), 1639)
		# Deterministic order: grouped, then by numeric QuestionId.
		first = [r["question_id"] for r in rows if r["group"] == wa.EXPECTED][:3]
		self.assertEqual(first, sorted(first, key=wa.question_sort_key))

	def test_compact_format(self):
		_, rows, systems = mcf.make_fig6(self.plt, self.out, "webqsp", self.csvs)
		compact = wa.compact_table(rows, systems)
		mismatch = next(r for r in rows if r["group"] == wa.QUERY_MISMATCH)
		cell = compact[1]["ToG"]
		self.assertEqual(cell, f"{mismatch['ToG found']} / 11 "
		                       f"({100 * mismatch['ToG found'] / 11:.1f}%)")
		self.assertEqual(compact[2]["Dataset group"], "Expected/reproducible (n=1,617)")

	def test_one_square_per_question(self):
		found = {s: wa.load_outcomes(p, GROUPS) for s, p in self.csvs.items()}
		fig = wa.draw_waffle(self.plt, GROUPS, found)
		counts = sorted(len(c.get_paths()) for ax in fig.axes for c in ax.collections)
		self.assertEqual(counts, [11] * 6 + [1617] * 3)
		self.plt.close(fig)

	def test_outcomes_missing_a_question_are_refused(self):
		short = dict(list(GROUPS.items())[1:])
		path = write_outcomes(os.path.join(self.inputs, "short.csv"), PATTERNS["CoR"], short)
		with self.assertRaises(ValueError):
			wa.load_outcomes(path, GROUPS)


if __name__ == "__main__":
	unittest.main()
