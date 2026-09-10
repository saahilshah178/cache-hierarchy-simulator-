"""Tests for the Mattson stack-distance profiler.

The load-bearing tests are the cross-checks: for every capacity, the miss
count derived from the stack-distance histogram must equal, exactly, the
misses of a simulated fully-associative LRU cache of that capacity, and the
per-set variant must equal a simulated set-associative LRU cache for every
associativity. Those two equalities validate the profiler and the simulator
against each other.
"""

from __future__ import annotations

import contextlib
import csv
import io
import itertools
import os
import random
import tempfile
import unittest
from collections.abc import Iterable, Iterator

from cachesim import plot
from cachesim.cache import Cache
from cachesim.stackdist import (
    INFINITE,
    ReuseBucket,
    block_stream,
    main,
    miss_ratio_curve,
    per_set_profile,
    power_of_two_capacities,
    stack_distance_profile,
    stack_distances,
)
from cachesim.workloads import (
    SAMPLE_NAMES,
    conflict_streams,
    matmul,
    pointer_chase,
    random_access,
    sequential,
    write_trace,
)

BLOCK = 64


def blocks_of(gen: Iterable[tuple[int, str]], block_size: int = BLOCK) -> list[int]:
    """The block-number stream of a workload generator."""
    return [addr // block_size for addr, _ in gen]


def simulate_fully_associative(blocks: Iterable[int], capacity: int) -> int:
    """Misses of a real ``Cache`` of ``capacity`` blocks, fully associative."""
    cache = Cache(
        "FA",
        size=capacity * BLOCK,
        block_size=BLOCK,
        associativity=capacity,
        policy="lru",
        track_3c=False,
    )
    for block in blocks:
        if not cache.probe(block):
            cache.allocate(block)
    return cache.misses


def simulate_set_associative(blocks: Iterable[int], num_sets: int, ways: int) -> int:
    """Misses of a real ``Cache`` with ``num_sets`` sets and ``ways`` ways."""
    cache = Cache(
        "SA",
        size=num_sets * ways * BLOCK,
        block_size=BLOCK,
        associativity=ways,
        policy="lru",
        track_3c=False,
    )
    for block in blocks:
        if not cache.probe(block):
            cache.allocate(block)
    return cache.misses


def small_samples() -> Iterator[tuple[str, list[int]]]:
    """Every sample workload, shrunk so the cross-checks stay quick."""
    yield "sequential", blocks_of(sequential(buffer_bytes=8 * 1024, passes=3))
    yield "random", blocks_of(random_access(region_bytes=64 * 1024, n=3000, seed=7))
    yield "matmul_naive", blocks_of(matmul(n=16))
    yield "matmul_blocked", blocks_of(matmul(n=16, tile=4))
    yield "conflict", blocks_of(conflict_streams(streams=4, words_per_stream=256))
    yield "pointer_chase", blocks_of(pointer_chase(nodes=256, hops=2000, seed=3))


class TestStackDistances(unittest.TestCase):
    def test_hand_checked_distances(self) -> None:
        """A, B, C, A, C, A: distances counted by hand."""
        blocks = [10, 11, 12, 10, 12, 10]
        self.assertEqual(
            list(stack_distances(blocks)),
            [
                INFINITE,  # 10: first touch
                INFINITE,  # 11: first touch
                INFINITE,  # 12: first touch
                2,  # 10: 11 and 12 seen since
                1,  # 12: only 10 seen since
                1,  # 10: only 12 seen since
            ],
        )

    def test_immediate_reuse_is_distance_zero(self) -> None:
        self.assertEqual(list(stack_distances([5, 5, 5])), [INFINITE, 0, 0])

    def test_repeated_block_counted_once(self) -> None:
        """B referenced twice between two references to A still costs 1."""
        self.assertEqual(list(stack_distances([1, 2, 2, 2, 1])), [INFINITE, INFINITE, 0, 0, 1])

    def test_empty_stream(self) -> None:
        self.assertEqual(list(stack_distances([])), [])
        profile = stack_distance_profile([])
        self.assertEqual(profile.references, 0)
        self.assertEqual(profile.infinite, 0)
        self.assertEqual(profile.misses(4), 0)
        self.assertEqual(profile.miss_ratio(4), 0.0)

    def test_distance_is_bounded_by_distinct_blocks(self) -> None:
        rng = random.Random(11)
        blocks = [rng.randrange(20) for _ in range(500)]
        for distance in stack_distances(blocks):
            self.assertLess(distance, 20)


class TestProfileShape(unittest.TestCase):
    def test_counts_add_up(self) -> None:
        blocks = blocks_of(sequential(buffer_bytes=4 * 1024, passes=3))
        profile = stack_distance_profile(blocks)
        self.assertEqual(profile.references, len(blocks))
        self.assertEqual(profile.infinite + sum(profile.histogram), len(blocks))
        self.assertEqual(profile.distinct_blocks, len(set(blocks)))

    def test_miss_curve_is_non_increasing_and_reaches_the_floor(self) -> None:
        blocks = blocks_of(matmul(n=16))
        profile = stack_distance_profile(blocks)
        capacities = power_of_two_capacities(profile.max_distance + 1)
        curve = profile.miss_curve(capacities)
        for a, b in itertools.pairwise(curve):
            self.assertGreaterEqual(a, b)
        self.assertEqual(curve[0], profile.misses(1))
        self.assertEqual(profile.misses(profile.max_distance + 1), profile.infinite)
        self.assertEqual(profile.misses(0), profile.references)

    def test_negative_capacity_is_rejected(self) -> None:
        with self.assertRaises(ValueError):
            stack_distance_profile([1, 2, 1]).misses(-1)


class TestMissRatioCurveMatchesSimulation(unittest.TestCase):
    """The inclusion property, checked against the simulator itself."""

    def check_stream(self, blocks: list[int], capacities: list[int]) -> None:
        derived = miss_ratio_curve(blocks, capacities)
        for capacity, misses in zip(capacities, derived, strict=True):
            self.assertEqual(
                misses,
                simulate_fully_associative(blocks, capacity),
                f"capacity {capacity} blocks",
            )

    def test_random_stream(self) -> None:
        rng = random.Random(2024)
        blocks = [rng.randrange(64) for _ in range(4000)]
        self.check_stream(blocks, [1, 2, 3, 4, 5, 7, 8, 11, 16, 32, 63, 64, 65, 128])

    def test_every_sample_workload(self) -> None:
        for name, blocks in small_samples():
            with self.subTest(workload=name):
                self.check_stream(blocks, [1, 2, 4, 8, 16, 32, 64, 128, 256, 512])

    def test_matches_shadow_misses_of_a_set_associative_cache(self) -> None:
        """The shadow cache is a fully-associative LRU twin of equal capacity."""
        blocks = blocks_of(conflict_streams(streams=4, words_per_stream=256))
        profile = stack_distance_profile(blocks)
        for num_sets, ways in [(64, 1), (32, 2), (16, 4), (8, 8), (4, 16)]:
            with self.subTest(sets=num_sets, ways=ways):
                cache = Cache(
                    "L1",
                    size=num_sets * ways * BLOCK,
                    block_size=BLOCK,
                    associativity=ways,
                    policy="lru",
                )
                for block in blocks:
                    if not cache.probe(block):
                        cache.allocate(block)
                self.assertEqual(cache.shadow_misses, profile.misses(num_sets * ways))
                self.assertEqual(cache.compulsory_misses, profile.infinite)


class TestPerSetProfile(unittest.TestCase):
    def test_one_set_reproduces_the_fully_associative_profile(self) -> None:
        blocks = blocks_of(matmul(n=16))
        flat = stack_distance_profile(blocks)
        per_set = per_set_profile(blocks, 1)
        self.assertEqual(per_set.profile.histogram, flat.histogram)
        self.assertEqual(per_set.infinite, flat.infinite)
        self.assertEqual(per_set.per_set_references, (len(blocks),))

    def test_all_associativities_match_simulation(self) -> None:
        for name, blocks in small_samples():
            for num_sets in (4, 16, 64):
                with self.subTest(workload=name, sets=num_sets):
                    profile = per_set_profile(blocks, num_sets)
                    for ways in range(1, 17):
                        self.assertEqual(
                            profile.misses(ways),
                            simulate_set_associative(blocks, num_sets, ways),
                            f"{name}: {num_sets} sets x {ways} ways",
                        )

    def test_per_set_counts_sum_to_the_stream(self) -> None:
        blocks = blocks_of(conflict_streams(streams=4, words_per_stream=256))
        profile = per_set_profile(blocks, 32)
        self.assertEqual(sum(profile.per_set_references), len(blocks))
        self.assertEqual(sum(profile.per_set_distinct), profile.infinite)
        self.assertEqual(len(profile.per_set_references), 32)

    def test_conflict_streams_need_four_ways(self) -> None:
        """Four streams alias to one set: 1 and 2 ways miss everything."""
        blocks = blocks_of(conflict_streams(streams=4, words_per_stream=256))
        profile = per_set_profile(blocks, 64)
        self.assertEqual(profile.misses(1), len(blocks))
        self.assertEqual(profile.misses(2), len(blocks))
        # From four ways on, only the first reference to each block misses.
        self.assertEqual(profile.misses(4), profile.infinite)
        self.assertEqual(profile.misses(8), profile.infinite)
        self.assertEqual(profile.infinite, len(blocks) // 8)  # 8 words per block

    def test_zero_sets_is_rejected(self) -> None:
        with self.assertRaises(ValueError):
            per_set_profile([1, 2, 3], 0)


class TestReuseHistogram(unittest.TestCase):
    def test_buckets_are_powers_of_two(self) -> None:
        """Block 1 is re-referenced at distances 0, 1, 2, 3 and 4 in turn."""
        blocks = [1, 1, 2, 1, 3, 4, 1, 5, 6, 7, 1, 8, 9, 10, 11, 1]
        profile = stack_distance_profile(blocks)
        self.assertEqual([d for d in stack_distances(blocks) if d != INFINITE], [0, 1, 2, 3, 4])
        histogram = profile.reuse_histogram()
        self.assertEqual(histogram.references, len(blocks))
        self.assertEqual(histogram.infinite, 11)  # 11 distinct blocks
        self.assertEqual(
            [(b.low, b.high) for b in histogram.buckets],
            [(0, 0), (1, 1), (2, 3), (4, 7)],
        )
        self.assertEqual(sum(b.count for b in histogram.buckets), len(blocks) - 11)
        self.assertEqual(histogram.buckets[0], ReuseBucket(0, 0, 1))  # distance 0
        self.assertEqual(histogram.buckets[1], ReuseBucket(1, 1, 1))  # distance 1
        self.assertEqual(histogram.buckets[2], ReuseBucket(2, 3, 2))  # distances 2 and 3
        self.assertEqual(histogram.buckets[3], ReuseBucket(4, 7, 1))  # distance 4

    def test_no_reuse_gives_no_buckets(self) -> None:
        histogram = stack_distance_profile([1, 2, 3]).reuse_histogram()
        self.assertEqual(histogram.buckets, ())
        self.assertEqual(histogram.infinite, 3)


class TestWorkingSetSize(unittest.TestCase):
    def test_exact_working_set_of_a_cyclic_stream(self) -> None:
        """Cycling over W blocks: nothing below W helps, W is exact."""
        working = 40
        blocks = [i % working for i in range(working * 20)]
        profile = stack_distance_profile(blocks)
        self.assertEqual(profile.working_set_size(tolerance=0.0), working)
        # Below the working set every reuse misses.
        self.assertEqual(profile.misses(working - 1), len(blocks))
        self.assertEqual(profile.misses(working), working)

    def test_tolerance_discards_a_thin_long_distance_tail(self) -> None:
        """990 references at distance 1, then 100 at distance 99.

        Only 100 of 1,190 references (8.4%) need more than two blocks, so a
        one-percentage-point tolerance still demands the full 100, while a
        ten-point tolerance is satisfied by two blocks.
        """
        blocks = [i % 2 for i in range(990)] + [100 + (i % 100) for i in range(200)]
        profile = stack_distance_profile(blocks)
        self.assertEqual(profile.references, 1190)
        self.assertEqual(profile.infinite, 102)
        self.assertEqual(profile.histogram[1], 988)
        self.assertEqual(profile.histogram[99], 100)
        self.assertEqual(profile.working_set_size(tolerance=0.0), 100)
        self.assertEqual(profile.working_set_size(tolerance=0.01), 100)
        self.assertEqual(profile.working_set_size(tolerance=0.10), 2)

    def test_no_reuse_returns_one_block(self) -> None:
        profile = stack_distance_profile([1, 2, 3, 4])
        self.assertEqual(profile.working_set_size(), 1)

    def test_bad_tolerance_is_rejected(self) -> None:
        profile = stack_distance_profile([1, 2, 1])
        for bad in (-0.1, 1.5):
            with self.subTest(tolerance=bad), self.assertRaises(ValueError):
                profile.working_set_size(tolerance=bad)


class TestBlockStream(unittest.TestCase):
    def test_addresses_collapse_to_blocks(self) -> None:
        accesses = [(0, False), (8, True), (63, False), (64, False), (128, True)]
        self.assertEqual(block_stream(accesses, 64), [0, 0, 0, 1, 2])

    def test_bad_block_size_is_rejected(self) -> None:
        with self.assertRaises(ValueError):
            block_stream([(0, False)], 0)

    def test_sample_traces_are_all_covered(self) -> None:
        """small_samples() must track SAMPLE_NAMES so no workload is skipped."""
        self.assertEqual({name for name, _ in small_samples()}, set(SAMPLE_NAMES))


class TestPowerOfTwoCapacities(unittest.TestCase):
    def test_covers_the_limit(self) -> None:
        self.assertEqual(power_of_two_capacities(1), [1])
        self.assertEqual(power_of_two_capacities(5), [1, 2, 4, 8])
        self.assertEqual(power_of_two_capacities(8), [1, 2, 4, 8])
        self.assertEqual(power_of_two_capacities(0), [1])


class TestMrcCli(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.trace = os.path.join(self._tmp.name, "c.trace")
        write_trace(self.trace, conflict_streams(streams=4, words_per_stream=256))
        self.blocks = blocks_of(conflict_streams(streams=4, words_per_stream=256))

    def run_main(self, argv: list[str]) -> tuple[int, str, str]:
        out, err = io.StringIO(), io.StringIO()
        with contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
            try:
                code = main(argv)
            except SystemExit as exc:
                code = int(exc.code) if exc.code is not None else 0
        return code, out.getvalue(), err.getvalue()

    def test_prints_the_curve_and_the_working_set(self) -> None:
        code, out, _ = self.run_main([self.trace])
        self.assertEqual(code, 0)
        self.assertIn("fully-associative LRU miss curve", out)
        self.assertIn("reuse-distance histogram", out)
        self.assertIn("working set:", out)
        # 1,024 references over 128 distinct blocks: the floor is 128 misses.
        self.assertIn("references: 1,024", out)
        self.assertIn("distinct 64 B blocks: 128", out)

    def test_sets_adds_the_associativity_curve(self) -> None:
        code, out, _ = self.run_main([self.trace, "--sets", "64"])
        self.assertEqual(code, 0)
        self.assertIn("set-associative LRU miss curve, 64 sets", out)
        self.assertIn("conflict", out)

    def test_csv_matches_the_profile(self) -> None:
        path = os.path.join(self._tmp.name, "mrc.csv")
        code, _, _ = self.run_main([self.trace, "--sets", "64", "--csv", path])
        self.assertEqual(code, 0)
        with open(path, newline="") as handle:
            rows = list(csv.DictReader(handle))
        profile = stack_distance_profile(self.blocks)
        per_set = per_set_profile(self.blocks, 64)
        self.assertTrue(rows)
        for row in rows:
            capacity = int(row["capacity_blocks"])
            expected = (
                profile.misses(capacity)
                if int(row["sets"]) == 1
                else per_set.misses(int(row["ways"]))
            )
            self.assertEqual(int(row["misses"]), expected, row)
            self.assertEqual(int(row["capacity_bytes"]), capacity * 64)

    def test_default_table_stops_at_the_compulsory_floor(self) -> None:
        """Three other streams sit between reuses, so no distance exceeds 3.

        The default curve therefore ends at 4 blocks, where every reuse hits
        and only the 128 first touches remain.
        """
        self.assertEqual(self.capacities_in_table([]), [1, 2, 4])
        self.assertEqual(stack_distance_profile(self.blocks).max_distance, 3)

    def test_max_capacity_truncates_the_table(self) -> None:
        self.assertEqual(self.capacities_in_table(["--max-capacity", "2"]), [1, 2])

    def capacities_in_table(self, extra: list[str]) -> list[int]:
        """The capacity column of the fully-associative table, in order."""
        code, out, _ = self.run_main([self.trace, *extra])
        self.assertEqual(code, 0)
        body = out.split("fully-associative LRU miss curve")[1].split("working set")[0]
        capacities = []
        for line in body.splitlines():
            fields = line.split()
            if fields and fields[0].replace(",", "").isdigit():
                capacities.append(int(fields[0].replace(",", "")))
        return capacities

    def test_missing_trace_is_reported(self) -> None:
        code, _, err = self.run_main([os.path.join(self._tmp.name, "nope.trace")])
        self.assertEqual(code, 2)
        self.assertIn("trace not found", err)

    def test_empty_trace_is_rejected(self) -> None:
        path = os.path.join(self._tmp.name, "empty.trace")
        with open(path, "w") as handle:
            handle.write("# nothing here\n")
        code, _, err = self.run_main([path])
        self.assertEqual(code, 2)
        self.assertIn("no accesses", err)

    def test_bad_flags_fail_with_a_message(self) -> None:
        for argv, fragment in [
            ([self.trace, "--block-size", "0"], "positive integer"),
            ([self.trace, "--sets", "0"], "positive integer"),
            ([self.trace, "--max-capacity", "-1"], "positive integer"),
        ]:
            with self.subTest(argv=argv):
                code, _, err = self.run_main(argv)
                self.assertEqual(code, 2)
                self.assertIn(fragment, err)


@unittest.skipUnless(plot.available(), "matplotlib is not installed")
class TestPlotMrc(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)

    def test_writes_a_non_empty_file(self) -> None:
        path = os.path.join(self._tmp.name, "mrc.png")
        written = plot.plot_mrc([1, 2, 4, 8], [1.0, 0.5, 0.25, 0.125], path)
        self.assertEqual(written, path)
        self.assertGreater(os.path.getsize(path), 0)

    def test_extra_series_and_nested_directory(self) -> None:
        path = os.path.join(self._tmp.name, "nested", "mrc.png")
        plot.plot_mrc(
            [1, 2, 4],
            [1.0, 0.5, 0.25],
            path,
            title="test",
            extra=[("64-set", [2, 4], [0.9, 0.4])],
        )
        self.assertGreater(os.path.getsize(path), 0)

    def test_ragged_or_empty_series_are_rejected(self) -> None:
        path = os.path.join(self._tmp.name, "bad.png")
        with self.assertRaises(ValueError):
            plot.plot_mrc([1, 2], [1.0], path)
        with self.assertRaises(ValueError):
            plot.plot_mrc([], [], path)
        with self.assertRaises(ValueError):
            plot.plot_mrc([1], [1.0], path, extra=[("x", [1, 2], [0.5])])

    def test_cli_writes_a_plot(self) -> None:
        trace = os.path.join(self._tmp.name, "c.trace")
        write_trace(trace, conflict_streams(streams=4, words_per_stream=128))
        path = os.path.join(self._tmp.name, "curve.png")
        with contextlib.redirect_stdout(io.StringIO()):
            self.assertEqual(main([trace, "--sets", "32", "--plot", path]), 0)
        self.assertGreater(os.path.getsize(path), 0)


class TestFormatBytes(unittest.TestCase):
    def test_units(self) -> None:
        self.assertEqual(plot.format_bytes(64), "64 B")
        self.assertEqual(plot.format_bytes(1024), "1 KB")
        self.assertEqual(plot.format_bytes(32768), "32 KB")
        self.assertEqual(plot.format_bytes(1536), "1.5 KB")
        self.assertEqual(plot.format_bytes(2 * 1024 * 1024), "2 MB")


if __name__ == "__main__":
    unittest.main(verbosity=2)
