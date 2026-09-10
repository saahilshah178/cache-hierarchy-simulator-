"""Tests for the synthetic workload registry and its generators."""

from __future__ import annotations

import contextlib
import io
import os
import tempfile
import unittest
from dataclasses import FrozenInstanceError
from itertools import islice

from cachesim.cli import main
from cachesim.workloads import SAMPLE_NAMES, WORKLOADS, get_workload, write_traces

#: How many accesses of each workload the structural tests inspect. The
#: generators are lazy, so this bounds their cost regardless of defaults.
PREFIX = 2_000


class TestRegistry(unittest.TestCase):
    def test_keys_match_names(self) -> None:
        for key, workload in WORKLOADS.items():
            self.assertEqual(key, workload.name)

    def test_sample_names_are_registered_and_unique(self) -> None:
        self.assertEqual(len(set(SAMPLE_NAMES)), len(SAMPLE_NAMES))
        for name in SAMPLE_NAMES:
            self.assertIn(name, WORKLOADS)

    def test_every_workload_is_documented(self) -> None:
        for name, workload in WORKLOADS.items():
            with self.subTest(name=name):
                self.assertTrue(workload.description.endswith("."), workload.description)
                self.assertGreater(len(workload.expectation), 40)
                # partial() hides the function's docstring behind .func.
                generator = getattr(workload.generator, "func", workload.generator)
                self.assertTrue(generator.__doc__, f"{name}: generator has no docstring")

    def test_workloads_are_frozen(self) -> None:
        with self.assertRaises(FrozenInstanceError):
            WORKLOADS["sequential"].name = "other"  # type: ignore[misc]

    def test_generators_emit_well_formed_accesses(self) -> None:
        for name, workload in WORKLOADS.items():
            with self.subTest(name=name):
                accesses = list(islice(workload.generator(), PREFIX))
                self.assertTrue(accesses, "generator produced nothing")
                for addr, op in accesses:
                    self.assertIsInstance(addr, int)
                    self.assertGreaterEqual(addr, 0)
                    self.assertIn(op, ("R", "W"))

    def test_generators_are_deterministic(self) -> None:
        for name, workload in WORKLOADS.items():
            with self.subTest(name=name):
                first = list(islice(workload.generator(), PREFIX))
                second = list(islice(workload.generator(), PREFIX))
                self.assertEqual(first, second)

    def test_get_workload_names_the_alternatives(self) -> None:
        with self.assertRaises(ValueError) as ctx:
            get_workload("nope")
        self.assertIn("unknown workload 'nope'", str(ctx.exception))
        self.assertIn("sequential", str(ctx.exception))


class TestWriteTraces(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)

    def test_writes_the_named_workloads_only(self) -> None:
        counts = write_traces(self._tmp.name, ["conflict"], verbose=False)
        self.assertEqual(counts, {"conflict": 65_536})
        self.assertEqual(os.listdir(self._tmp.name), ["conflict.trace"])

    def test_unknown_name_is_rejected_before_writing(self) -> None:
        with self.assertRaises(ValueError):
            write_traces(self._tmp.name, ["nope"], verbose=False)
        self.assertEqual(os.listdir(self._tmp.name), [])

    def test_cli_list_prints_every_workload(self) -> None:
        out = io.StringIO()
        with contextlib.redirect_stdout(out):
            code = main(["gen-traces", "--list"])
        self.assertEqual(code, 0)
        printed = out.getvalue()
        for name in WORKLOADS:
            self.assertIn(f"\n{name}\n", "\n" + printed)
        self.assertEqual(printed.count("expectation:"), len(WORKLOADS))

    def test_cli_list_writes_nothing(self) -> None:
        with contextlib.redirect_stdout(io.StringIO()):
            main(["gen-traces", "--list", "--out-dir", self._tmp.name])
        self.assertEqual(os.listdir(self._tmp.name), [])


if __name__ == "__main__":
    unittest.main(verbosity=2)
