"""Tests for the parameter sweep command."""

from __future__ import annotations

import contextlib
import io
import os
import tempfile
import unittest
from collections.abc import Iterable

from cachesim.sweep import main, run_single_level, sweep_associativity, sweep_size
from cachesim.workloads import conflict_streams, write_trace


def as_accesses(gen: Iterable[tuple[int, str]]) -> list[tuple[int, bool]]:
    return [(addr, op == "W") for addr, op in gen]


class TestSweepFunctions(unittest.TestCase):
    def test_associativity_knee_on_conflict_streams(self) -> None:
        """Four aliasing streams: 100% miss below 4-way, one miss per block after."""
        trace = as_accesses(conflict_streams(words_per_stream=1024))
        with contextlib.redirect_stdout(io.StringIO()):
            results = sweep_associativity(trace, 8192, 64, "lru", 4, 100)
        rates = dict(results)
        self.assertEqual(rates[1], 1.0)
        self.assertEqual(rates[2], 1.0)
        self.assertEqual(rates[4], 0.125)  # 1 miss per 8 words of each block
        self.assertEqual(rates[8], 0.125)
        self.assertEqual(rates[16], 0.125)

    def test_size_sweep_returns_one_point_per_size(self) -> None:
        trace = as_accesses(conflict_streams(words_per_stream=256))
        with contextlib.redirect_stdout(io.StringIO()):
            results = sweep_size(trace, 4, 64, "lru", 4, 100)
        self.assertEqual([kb for kb, _ in results], [1, 2, 4, 8, 16, 32, 64])
        for _, rate in results:
            self.assertEqual(rate, 0.125)

    def test_run_single_level_amat(self) -> None:
        trace = [(0, False), (0, False), (64, False), (64, False)]
        miss_rate, amat, cache = run_single_level(trace, 1024, 64, 1, "lru", 4, 100)
        self.assertEqual(miss_rate, 0.5)
        self.assertEqual(amat, 4 + 0.5 * 100)
        self.assertEqual(cache.misses, 2)


class TestSweepCLI(unittest.TestCase):
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

    def test_bad_flags_fail_with_a_message(self) -> None:
        for argv, fragment in [
            (["--size", "0", self.trace], "positive integer"),
            (["--size", "-8192", self.trace], "positive integer"),
            (["--assoc", "0", self.trace], "positive integer"),
            (["--block-size", "0", self.trace], "positive integer"),
            (["--block-size", "48", self.trace], "power of two"),
            (["--hit-time", "-1", self.trace], "non-negative"),
            (["--size", "3000", self.trace], "not a multiple"),
            (["--assoc", "3", self.trace], "not a multiple"),
        ]:
            with self.subTest(argv=argv):
                code, _, err = self.run_main(argv)
                self.assertEqual(code, 2)
                self.assertIn(fragment, err)

    def test_missing_trace_is_reported(self) -> None:
        code, _, err = self.run_main([os.path.join(self._tmp.name, "nope.trace")])
        self.assertEqual(code, 2)
        self.assertIn("trace not found", err)

    def test_empty_trace_is_rejected(self) -> None:
        path = os.path.join(self._tmp.name, "empty.trace")
        with open(path, "w") as f:
            f.write("# nothing here\n")
        code, _, err = self.run_main([path])
        self.assertEqual(code, 2)
        self.assertIn("no accesses", err)

    def test_happy_path_prints_tables_and_plots(self) -> None:
        out_dir = os.path.join(self._tmp.name, "plots")
        code, out, _ = self.run_main(["--size", "8192", "--out-dir", out_dir, self.trace])
        self.assertEqual(code, 0)
        self.assertIn("associativity sweep", out)
        self.assertIn("size sweep", out)
        self.assertEqual(
            sorted(os.listdir(out_dir)),
            ["miss_rate_vs_associativity_c.png", "miss_rate_vs_size_c.png"],
        )


if __name__ == "__main__":
    unittest.main(verbosity=2)
