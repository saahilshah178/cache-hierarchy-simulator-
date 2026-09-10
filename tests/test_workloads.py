"""Tests for the synthetic workload registry and its generators."""

from __future__ import annotations

import contextlib
import io
import os
import tempfile
import unittest
from collections.abc import Iterable
from dataclasses import FrozenInstanceError
from itertools import islice

from cachesim.cache import Cache
from cachesim.cli import main
from cachesim.workloads import (
    SAMPLE_NAMES,
    WORKLOADS,
    binary_search,
    column_walk,
    cyclic,
    get_workload,
    hash_probe,
    stencil_2d,
    strided,
    transpose,
    write_traces,
)

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


class KnownAnswerCase(unittest.TestCase):
    """Base class: run a generator through one cache and read its counters."""

    def run_workload(
        self,
        accesses: Iterable[tuple[int, str]],
        size: int,
        associativity: int,
        block_size: int = 64,
    ) -> Cache:
        cache = Cache("T", size, block_size, associativity)
        for addr, op in accesses:
            cache.access(addr, op == "W")
        return cache

    def fully_associative(self, blocks: int) -> tuple[int, int]:
        """Geometry arguments for a fully-associative cache of ``blocks`` blocks."""
        return blocks * 64, blocks


class TestStrided(KnownAnswerCase):
    """strided: one block per access, so the footprint is count blocks."""

    def test_footprint_that_fits_misses_only_on_the_first_pass(self) -> None:
        size, ways = self.fully_associative(64)
        cache = self.run_workload(strided(stride_bytes=512, count=32, passes=3), size, ways)
        self.assertEqual(cache.accesses, 96)
        self.assertEqual(cache.misses, 32)  # one compulsory miss per block
        self.assertEqual(cache.hits, 64)  # passes 2 and 3 hit throughout

    def test_footprint_larger_than_the_cache_misses_every_access(self) -> None:
        size, ways = self.fully_associative(64)
        cache = self.run_workload(strided(stride_bytes=512, count=128, passes=3), size, ways)
        self.assertEqual(cache.accesses, 384)
        self.assertEqual(cache.misses, 384)  # LRU's worst case: a cycle of 128 in 64
        self.assertEqual(cache.hits, 0)

    def test_stride_below_the_block_size_shares_blocks(self) -> None:
        size, ways = self.fully_associative(64)
        # 8 accesses per 64-byte block, one pass: one miss per block.
        cache = self.run_workload(strided(stride_bytes=8, count=64, passes=1), size, ways)
        self.assertEqual((cache.accesses, cache.misses), (64, 8))


class TestColumnWalk(KnownAnswerCase):
    """column_walk: a 64x64 matrix of doubles on an 8 KB direct-mapped cache.

    512 B rows, 8 blocks apart, so a column reaches only 128/8 = 16 of the
    128 sets; padding by one block makes the row distance 9 blocks, which is
    coprime with 128, so the 64 rows of a column spread over 64 sets.
    """

    ACCESSES = 64 * 64
    FLOOR = 64 * 8  # rows x ceil(cols / 8): the blocks the walk must touch

    def test_power_of_two_stride_misses_every_access(self) -> None:
        cache = self.run_workload(column_walk(rows=64, cols=64), 8192, 1)
        self.assertEqual(cache.accesses, self.ACCESSES)
        self.assertEqual(cache.misses, self.ACCESSES)
        self.assertEqual(cache.compulsory_misses, self.FLOOR)

    def test_padding_the_row_stride_reaches_the_compulsory_floor(self) -> None:
        cache = self.run_workload(
            column_walk(rows=64, cols=64, row_stride_bytes=64 * 8 + 64), 8192, 1
        )
        self.assertEqual(cache.accesses, self.ACCESSES)
        self.assertEqual(cache.misses, self.FLOOR)
        self.assertEqual(cache.compulsory_misses, self.FLOOR)

    def test_associativity_does_not_fix_the_stride(self) -> None:
        # 4 ways over 32 sets still reach only 4 sets: 16 blocks of the 64 needed.
        cache = self.run_workload(column_walk(rows=64, cols=64), 8192, 4)
        self.assertEqual(cache.misses, self.ACCESSES)


class TestStencil2D(KnownAnswerCase):
    """stencil_2d: 6 accesses per interior point, 5 reads and 1 write."""

    def test_access_count_and_mix(self) -> None:
        accesses = list(stencil_2d(n=10, passes=1))
        self.assertEqual(len(accesses), 6 * 8 * 8)
        self.assertEqual(sum(1 for _, op in accesses if op == "W"), 8 * 8)

    def test_three_resident_rows_reach_the_compulsory_floor(self) -> None:
        # 8 blocks hold 4 rows of the 10x10 grid (80 B each) plus the out row.
        size, ways = self.fully_associative(8)
        cache = self.run_workload(stencil_2d(n=10, passes=1), size, ways)
        self.assertEqual(cache.accesses, 384)
        self.assertEqual(cache.misses, 24)  # 13 blocks of `in` + 11 of `out`
        self.assertEqual(cache.misses, cache.compulsory_misses)

    def test_miss_rate_approaches_two_per_eight_points(self) -> None:
        cache = self.run_workload(stencil_2d(n=96, passes=2), 32 * 1024, 4)
        self.assertEqual((cache.accesses, cache.misses), (106_032, 4560))
        points = 2 * 94 * 94
        self.assertAlmostEqual(cache.misses / points, 0.25, delta=0.02)  # 2 per 8 points


class TestTranspose(KnownAnswerCase):
    def test_both_matrices_resident_reach_the_compulsory_floor(self) -> None:
        # 16x16 doubles is 2 KB per matrix: 64 blocks in a 4 KB cache.
        cache = self.run_workload(transpose(n=16), 4096, 4)
        self.assertEqual(cache.accesses, 2 * 16 * 16)
        self.assertEqual(cache.misses, 64)
        self.assertEqual(cache.misses, cache.compulsory_misses)

    def test_column_major_writes_cost_one_miss_each(self) -> None:
        cache = self.run_workload(transpose(n=128), 32 * 1024, 4)
        elements = 128 * 128
        self.assertEqual(cache.accesses, 2 * elements)
        self.assertEqual(cache.misses, 18_432)
        self.assertEqual(cache.misses / elements, 1.125)  # 1 per write + 1/8 per read


class TestBinarySearch(KnownAnswerCase):
    def test_probe_count_follows_log2_of_the_array(self) -> None:
        accesses = list(binary_search(n_elements=4096, queries=200, seed=3))
        self.assertEqual(len(accesses), 2177)  # 10.9 probes per query, log2(4096) = 12
        self.assertTrue(all(op == "R" for _, op in accesses))

    def test_seeded_miss_count_is_exact(self) -> None:
        size, ways = self.fully_associative(64)  # 4 KB against a 32 KB array
        cache = self.run_workload(binary_search(n_elements=4096, queries=200, seed=3), size, ways)
        self.assertEqual((cache.accesses, cache.misses), (2177, 819))
        # log2(32 KB / 4 KB) = 3 probes per query beyond the cached prefix.
        self.assertLess(abs(cache.misses / 200 - 3), 1.5)

    def test_an_array_that_fits_reaches_the_compulsory_floor(self) -> None:
        size, ways = self.fully_associative(512)  # 32 KB: the whole array
        cache = self.run_workload(binary_search(n_elements=4096, queries=200, seed=3), size, ways)
        self.assertEqual(cache.misses, 349)  # the blocks any probe ever lands in
        self.assertEqual(cache.misses, cache.compulsory_misses)


class TestHashProbe(KnownAnswerCase):
    TABLE = 1024 * 1024
    CACHE = 64 * 1024

    def test_seeded_miss_count_is_exact(self) -> None:
        size, ways = self.fully_associative(self.CACHE // 64)
        cache = self.run_workload(
            hash_probe(table_bytes=self.TABLE, probes=40_000, seed=4), size, ways
        )
        self.assertEqual((cache.accesses, cache.misses), (40_000, 37_630))

    def test_steady_state_miss_rate_matches_one_minus_c_over_n(self) -> None:
        size, ways = self.fully_associative(self.CACHE // 64)
        cache = Cache("T", size, 64, ways)
        accesses = list(hash_probe(table_bytes=self.TABLE, probes=40_000, seed=4))
        for addr, _ in accesses[:20_000]:  # warm-up
            cache.access(addr)
        cache.reset_stats()
        for addr, _ in accesses[20_000:]:
            cache.access(addr)
        predicted = 1 - self.CACHE / self.TABLE
        self.assertEqual(cache.misses, 18_824)
        self.assertAlmostEqual(cache.miss_rate, predicted, delta=0.01)


class TestCyclic(KnownAnswerCase):
    def test_a_cycle_that_fits_misses_only_once_per_block(self) -> None:
        size, ways = self.fully_associative(64)
        cache = self.run_workload(cyclic(blocks=32, passes=4), size, ways)
        self.assertEqual((cache.accesses, cache.misses, cache.hits), (128, 32, 96))

    def test_a_cycle_exactly_the_size_of_the_cache_still_fits(self) -> None:
        size, ways = self.fully_associative(64)
        cache = self.run_workload(cyclic(blocks=64, passes=4), size, ways)
        self.assertEqual((cache.accesses, cache.misses, cache.hits), (256, 64, 192))

    def test_one_block_too_many_misses_every_access(self) -> None:
        size, ways = self.fully_associative(64)
        cache = self.run_workload(cyclic(blocks=65, passes=4), size, ways)
        self.assertEqual((cache.accesses, cache.misses, cache.hits), (260, 260, 0))


if __name__ == "__main__":
    unittest.main(verbosity=2)
