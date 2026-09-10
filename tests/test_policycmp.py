"""Tests for the policy comparison command."""

from __future__ import annotations

import contextlib
import csv
import io
import os
import tempfile
import unittest

from cachesim.policies import POLICIES
from cachesim.policycmp import CSV_HEADER, compare_policies, format_table, main, write_csv
from cachesim.workloads import conflict_streams, matmul, write_trace


def as_accesses(pairs: object) -> list[tuple[int, bool]]:
    return [(addr, op == "W") for addr, op in pairs]  # type: ignore[attr-defined]


class TestComparePolicies(unittest.TestCase):
    def setUp(self) -> None:
        self.accesses = [(addr, op == "W") for addr, op in matmul(n=16)]

    def test_every_registered_policy_gets_a_row(self) -> None:
        results = compare_policies(self.accesses, 4096, 64, 4, hit_time=4, mem_time=100)
        self.assertEqual({r.policy for r in results}, set(POLICIES))
        self.assertIn("opt", {r.policy for r in results})

    def test_rows_are_sorted_best_first_and_opt_leads(self) -> None:
        results = compare_policies(self.accesses, 4096, 64, 4, hit_time=4, mem_time=100)
        self.assertEqual([r.misses for r in results], sorted(r.misses for r in results))
        # Belady inside each set is a lower bound for every other policy at
        # the same geometry, so no row can beat it, and it leads any tie.
        best = results[0]
        self.assertEqual(best.policy, "opt")
        self.assertEqual(best.misses, min(r.misses for r in results))
        self.assertEqual(best.gap_to_opt, 1.0)
        for r in results:
            self.assertGreaterEqual(r.gap_to_opt, 1.0)
            self.assertAlmostEqual(r.gap_to_opt, r.misses / best.misses)

    def test_derived_columns_follow_the_single_level_formulas(self) -> None:
        results = compare_policies(self.accesses, 4096, 64, 4, hit_time=4, mem_time=100)
        for r in results:
            self.assertEqual(r.accesses, len(self.accesses))
            self.assertAlmostEqual(r.miss_rate, r.misses / r.accesses)
            self.assertAlmostEqual(r.amat, 4 + r.miss_rate * 100)

    def test_a_subset_of_policies_can_be_compared(self) -> None:
        results = compare_policies(
            self.accesses, 4096, 64, 4, hit_time=4, mem_time=100, policies=["lru", "fifo"]
        )
        # OPT is always added: the gap column needs it.
        self.assertEqual({r.policy for r in results}, {"lru", "fifo", "opt"})

    def test_a_direct_mapped_cache_leaves_every_policy_identical(self) -> None:
        # One way means no choice to make: replacement cannot fix a mapping
        # problem, and the four aliasing streams miss 100% under all of them.
        accesses = as_accesses(conflict_streams(words_per_stream=512))
        results = compare_policies(accesses, 8192, 64, 1, hit_time=4, mem_time=100)
        self.assertEqual({r.misses for r in results}, {len(accesses)})
        self.assertEqual({r.gap_to_opt for r in results}, {1.0})


class TestFormatting(unittest.TestCase):
    def test_the_table_has_a_row_per_policy_and_the_bound(self) -> None:
        accesses = [(addr, op == "W") for addr, op in matmul(n=16)]
        results = compare_policies(accesses, 4096, 64, 4, hit_time=4, mem_time=100)
        table = format_table(results, fully_associative_bound=17, accesses=len(accesses))
        lines = table.splitlines()
        self.assertIn("policy", lines[0])
        self.assertEqual(len([ln for ln in lines if ln.strip()]), len(results) + 2)
        self.assertIn("fully associative Belady bound: 17 misses", table)

    def test_the_csv_round_trips(self) -> None:
        accesses = [(addr, op == "W") for addr, op in matmul(n=16)]
        results = compare_policies(accesses, 4096, 64, 4, hit_time=4, mem_time=100)
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "policies.csv")
            write_csv(path, results)
            with open(path) as f:
                rows = list(csv.reader(f))
        self.assertEqual(tuple(rows[0]), CSV_HEADER)
        self.assertEqual(len(rows), len(results) + 1)
        self.assertEqual(rows[1][0], results[0].policy)
        self.assertEqual(int(rows[1][2]), results[0].misses)


class TestPolicyCompareCLI(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.trace = os.path.join(self._tmp.name, "c.trace")
        write_trace(self.trace, conflict_streams(words_per_stream=256))

    def run_main(self, argv: list[str]) -> tuple[int, str, str]:
        out, err = io.StringIO(), io.StringIO()
        with contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
            try:
                code = main(argv)
            except SystemExit as exc:
                code = int(exc.code) if exc.code is not None else 0
        return code, out.getvalue(), err.getvalue()

    def test_happy_path_prints_the_table(self) -> None:
        code, out, _ = self.run_main([self.trace, "--size", "8192", "--assoc", "4"])
        self.assertEqual(code, 0)
        self.assertIn("policy", out)
        for name in POLICIES:
            self.assertIn(name, out)
        self.assertIn("fully associative Belady bound", out)

    def test_csv_flag_writes_a_file(self) -> None:
        path = os.path.join(self._tmp.name, "out.csv")
        code, out, _ = self.run_main([self.trace, "--size", "8192", "--csv", path])
        self.assertEqual(code, 0)
        self.assertTrue(os.path.exists(path))
        self.assertIn(f"wrote {path}", out)

    def test_an_unwritable_csv_is_reported_not_dumped(self) -> None:
        """A --csv under a plain file cannot be created; say so in one line."""
        blocker = os.path.join(self._tmp.name, "blocker")
        with open(blocker, "w") as handle:
            handle.write("not a directory\n")
        path = os.path.join(blocker, "out.csv")
        code, _, err = self.run_main([self.trace, "--size", "8192", "--csv", path])
        self.assertEqual(code, 2)
        self.assertIn("cannot write", err)
        self.assertNotIn("Traceback", err)

    def test_bad_flags_fail_with_a_message(self) -> None:
        for argv, fragment in [
            ([self.trace, "--size", "0"], "positive integer"),
            ([self.trace, "--assoc", "0"], "positive integer"),
            ([self.trace, "--hit-time", "-1"], "non-negative"),
            ([self.trace, "--size", "3000"], "not a multiple"),
        ]:
            with self.subTest(argv=argv):
                code, _, err = self.run_main(argv)
                self.assertEqual(code, 2)
                self.assertIn(fragment, err)

    def test_a_missing_trace_is_reported(self) -> None:
        code, _, err = self.run_main([os.path.join(self._tmp.name, "nope.trace")])
        self.assertEqual(code, 2)
        self.assertIn("trace not found", err)

    def test_an_empty_trace_is_rejected(self) -> None:
        path = os.path.join(self._tmp.name, "empty.trace")
        with open(path, "w") as f:
            f.write("# nothing here\n")
        code, _, err = self.run_main([path])
        self.assertEqual(code, 2)
        self.assertIn("no accesses", err)

    def test_the_subcommand_is_registered_in_the_cli(self) -> None:
        from cachesim.cli import build_parser

        parser = build_parser()
        with contextlib.redirect_stdout(io.StringIO()) as out, contextlib.suppress(SystemExit):
            parser.parse_args(["--help"])
        self.assertIn("policies", out.getvalue())


if __name__ == "__main__":
    unittest.main(verbosity=2)
