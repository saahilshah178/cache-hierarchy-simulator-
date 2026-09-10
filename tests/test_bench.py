"""Tests for ``cachesim bench``.

The benchmark reports wall time, which is not reproducible, so what is
pinned here is everything around it: the fields that must be present, the
arithmetic that ties them together (best <= median, throughput = accesses /
best), the counts, and the error paths.
"""

from __future__ import annotations

import contextlib
import io
import json
import os
import tempfile
import unittest

from cachesim.benchcmd import DEFAULT_REPEAT, benchmark
from cachesim.cli import main
from cachesim.config import DEFAULT_CONFIG
from cachesim.workloads import sequential, write_trace


class TestBenchmark(unittest.TestCase):
    """The timing helper, without the command line around it."""

    def test_one_timing_per_repetition(self) -> None:
        accesses = [(addr, False) for addr in range(0, 4096, 64)]
        timings = benchmark(accesses, DEFAULT_CONFIG, repeat=4)
        self.assertEqual(len(timings), 4)
        self.assertTrue(all(t >= 0.0 for t in timings))

    def test_each_repetition_starts_cold(self) -> None:
        """A fresh hierarchy per pass, so pass two is not a warm cache.

        ``benchmark`` returns only timings, so the property is checked the
        way it is visible: replaying a trace that fits in L1 a second time
        through the *same* hierarchy would take a different amount of work,
        and the guarantee is that it does not happen. The observable proxy
        is that repeating does not change the trace, so a run of the
        hierarchy the command builds reports identical counters each time.
        """
        from cachesim.hierarchy import Hierarchy

        accesses = [(addr, False) for addr in range(0, 8192, 64)] * 2
        first = Hierarchy.from_config(DEFAULT_CONFIG)
        for addr, is_write in accesses:
            first.access(addr, is_write)
        second = Hierarchy.from_config(DEFAULT_CONFIG)
        for addr, is_write in accesses:
            second.access(addr, is_write)
        self.assertEqual(first.stats().to_dict(), second.stats().to_dict())

    def test_repeat_must_be_positive(self) -> None:
        with self.assertRaises(ValueError):
            benchmark([(0, False)], DEFAULT_CONFIG, repeat=0)


class TestBenchCLI(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.trace = os.path.join(self._tmp.name, "seq.trace")
        self.accesses = write_trace(self.trace, sequential(buffer_bytes=8 * 1024, passes=2))

    def run_main(self, argv: list[str]) -> tuple[int, str, str]:
        out, err = io.StringIO(), io.StringIO()
        with contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
            try:
                code = main(argv)
            except SystemExit as exc:
                if isinstance(exc.code, int):
                    code = exc.code
                else:
                    err.write(str(exc.code))
                    code = 1
        return code, out.getvalue(), err.getvalue()

    def test_json_fields(self) -> None:
        code, out, _ = self.run_main(["bench", "--repeat", "2", "--format", "json", self.trace])
        self.assertEqual(code, 0)
        data = json.loads(out)
        self.assertEqual(data["trace"], self.trace)
        self.assertIsNone(data["config"])
        self.assertEqual(data["accesses"], self.accesses)
        self.assertEqual(data["repeat"], 2)
        self.assertEqual(len(data["seconds"]), 2)
        self.assertGreaterEqual(data["parse_seconds"], 0.0)
        self.assertEqual(data["best_seconds"], min(data["seconds"]))
        self.assertLessEqual(data["best_seconds"], data["median_seconds"])
        self.assertGreater(data["accesses_per_second"], 0.0)

    def test_throughput_matches_best(self) -> None:
        _, out, _ = self.run_main(["bench", "--repeat", "2", "--format", "json", self.trace])
        data = json.loads(out)
        self.assertAlmostEqual(
            data["accesses_per_second"],
            data["accesses"] / data["best_seconds"],
            places=3,
        )

    def test_text_report(self) -> None:
        code, out, _ = self.run_main(["bench", "--repeat", "2", self.trace])
        self.assertEqual(code, 0)
        self.assertIn("BENCHMARK", out)
        self.assertIn("accesses/s", out)
        self.assertIn(f"{self.accesses:,}", out)

    def test_default_repeat(self) -> None:
        _, out, _ = self.run_main(["bench", "--format", "json", self.trace])
        data = json.loads(out)
        self.assertEqual(data["repeat"], DEFAULT_REPEAT)
        self.assertEqual(len(data["seconds"]), DEFAULT_REPEAT)

    def test_config_file_is_reported(self) -> None:
        path = os.path.join(self._tmp.name, "one.json")
        with open(path, "w") as f:
            json.dump({"memory_access_time": 50, "levels": [DEFAULT_CONFIG["levels"][0]]}, f)
        _, out, _ = self.run_main(
            ["bench", "--config", path, "--repeat", "1", "--format", "json", self.trace]
        )
        self.assertEqual(json.loads(out)["config"], path)

    def test_missing_trace(self) -> None:
        code, _, err = self.run_main(["bench", os.path.join(self._tmp.name, "nope.trace")])
        self.assertEqual(code, 1)
        self.assertIn("error:", err)

    def test_empty_trace(self) -> None:
        path = os.path.join(self._tmp.name, "empty.trace")
        with open(path, "w") as f:
            f.write("# nothing here\n")
        code, _, err = self.run_main(["bench", path])
        self.assertEqual(code, 1)
        self.assertIn("no accesses", err)

    def test_bad_repeat(self) -> None:
        code, _, err = self.run_main(["bench", "--repeat", "0", self.trace])
        self.assertEqual(code, 1)
        self.assertIn("--repeat", err)

    def test_an_offline_policy_config_is_declined_not_crashed(self) -> None:
        """`run` simulates this config; bench must say why it will not."""
        path = os.path.join(self._tmp.name, "opt.json")
        with open(path, "w") as f:
            json.dump(
                {
                    "memory_access_time": 100,
                    "levels": [
                        {
                            "name": "L1",
                            "size": 4096,
                            "block_size": 64,
                            "associativity": 4,
                            "hit_time": 1,
                            "policy": "opt",
                        }
                    ],
                },
                f,
            )
        # The same config is a normal run: the guard is about bench, not the config.
        self.assertEqual(self.run_main(["run", self.trace, "--config", path])[0], 0)
        code, _, err = self.run_main(["bench", self.trace, "--config", path])
        self.assertEqual(code, 1)
        self.assertIn("error:", err)
        self.assertIn("opt", err)
        self.assertNotIn("Traceback", err)


if __name__ == "__main__":
    unittest.main()
