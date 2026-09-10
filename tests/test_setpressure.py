"""Tests for the per-set diagnostics.

The load-bearing test is TestUniformHistogramHidesTheCrowding: it pins a
workload whose CUMULATIVE per-set miss histogram is flat to within 0.3%
while more than seventy per cent of its (window, set) pairs are
oversubscribed. That is the whole reason the windowed measurement exists,
so it is checked with exact counts rather than inequalities.
"""

from __future__ import annotations

import contextlib
import io
import json
import os
import tempfile
import unittest
from collections.abc import Iterable

from cachesim.cache import Cache
from cachesim.setpressure import (
    format_hot_sets,
    format_window_pressure,
    hot_sets,
    main,
    window_pressure,
)
from cachesim.workloads import conflict_streams, matmul, write_trace
from cachesim.workloads import matmul as blocked

BLOCK = 64


def as_accesses(gen: Iterable[tuple[int, str]]) -> list[tuple[int, bool]]:
    return [(addr, op == "W") for addr, op in gen]


class TestHotSetsKnownAnswers(unittest.TestCase):
    def test_all_traffic_in_one_set(self) -> None:
        """Two even-numbered blocks in a 2-set direct-mapped cache collide."""
        # Blocks 0 and 2 both map to set 0; alternating them misses every time.
        accesses = [(addr, False) for _ in range(2) for addr in (0, 128)]
        hot = hot_sets(accesses, size=2 * BLOCK, block_size=BLOCK, associativity=1)
        self.assertEqual(hot.num_sets, 2)
        self.assertEqual(hot.per_set_misses, (4, 0))
        self.assertEqual(hot.misses, 4)
        self.assertEqual(hot.accesses, 4)
        self.assertEqual(hot.max_misses, 4)
        self.assertEqual(hot.min_misses, 0)
        self.assertEqual(hot.mean_misses, 2.0)
        self.assertEqual(hot.spread, 2.0)
        self.assertEqual(hot.top_sets, 1)  # 5% of 2 sets rounds up to 1
        self.assertEqual(hot.top_share, 1.0)
        self.assertEqual(hot.uniform_share, 0.5)
        self.assertEqual(hot.hottest(2), [(0, 4), (1, 0)])

    def test_per_set_counts_sum_to_the_miss_count(self) -> None:
        accesses = as_accesses(matmul(n=16))
        hot = hot_sets(accesses, size=4096, block_size=BLOCK, associativity=4)
        self.assertEqual(sum(hot.per_set_misses), hot.misses)
        self.assertEqual(len(hot.per_set_misses), hot.num_sets)

    def test_miss_count_matches_a_plain_simulation(self) -> None:
        accesses = as_accesses(matmul(n=16))
        for size, ways in [(2048, 1), (4096, 4), (8192, 8)]:
            with self.subTest(size=size, ways=ways):
                cache = Cache("L1", size, BLOCK, ways, "lru", track_3c=False)
                for addr, is_write in accesses:
                    cache.access(addr, is_write)
                hot = hot_sets(accesses, size, BLOCK, ways)
                self.assertEqual(hot.misses, cache.misses)

    def test_hottest_breaks_ties_on_the_lower_index(self) -> None:
        accesses = [(0, False), (64, False)]
        hot = hot_sets(accesses, size=8 * BLOCK, block_size=BLOCK, associativity=1)
        self.assertEqual(hot.hottest(3), [(0, 1), (1, 1), (2, 0)])

    def test_bad_top_fraction_is_rejected(self) -> None:
        for bad in (0.0, -0.1, 1.5):
            with self.subTest(top_fraction=bad), self.assertRaises(ValueError):
                hot_sets([(0, False)], 1024, BLOCK, 1, top_fraction=bad)


class TestWindowPressureKnownAnswers(unittest.TestCase):
    def test_four_blocks_in_one_set_oversubscribe_below_four_ways(self) -> None:
        """Blocks 0, 2, 4, 6 all map to set 0 of a two-set cache."""
        accesses = [(addr, False) for addr in (0, 128, 256, 384)]
        for ways, expected in [(1, 1), (2, 1), (3, 1), (4, 0), (8, 0)]:
            with self.subTest(ways=ways):
                pressure = window_pressure(
                    accesses, num_sets=2, block_size=BLOCK, ways=ways, window=4
                )
                self.assertEqual(pressure.windows, 1)
                self.assertEqual(pressure.touched_pairs, 1)
                self.assertEqual(pressure.oversubscribed_pairs, expected)
                self.assertEqual(pressure.max_distinct, 4)
                self.assertEqual(pressure.mean_distinct, 4.0)

    def test_repeats_of_one_block_are_one_distinct_block(self) -> None:
        accesses = [(0, False)] * 100
        pressure = window_pressure(accesses, num_sets=4, block_size=BLOCK, ways=1, window=10)
        self.assertEqual(pressure.windows, 10)
        self.assertEqual(pressure.touched_pairs, 10)
        self.assertEqual(pressure.oversubscribed_pairs, 0)
        self.assertEqual(pressure.max_distinct, 1)

    def test_trailing_partial_window_is_kept(self) -> None:
        accesses = [(addr * BLOCK, False) for addr in range(6)]
        pressure = window_pressure(accesses, num_sets=1, block_size=BLOCK, ways=1, window=4)
        self.assertEqual(pressure.windows, 2)  # 4 accesses, then 2
        self.assertEqual(pressure.accesses, 6)
        self.assertEqual(pressure.touched_pairs, 2)
        self.assertEqual(pressure.max_distinct, 4)
        self.assertEqual(pressure.total_distinct, 6)

    def test_window_larger_than_the_trace_is_one_window(self) -> None:
        accesses = [(0, False), (64, False)]
        pressure = window_pressure(accesses, num_sets=2, block_size=BLOCK, ways=1, window=1000)
        self.assertEqual(pressure.windows, 1)
        self.assertEqual(pressure.touched_pairs, 2)  # blocks 0 and 1 land in different sets

    def test_empty_trace(self) -> None:
        pressure = window_pressure([], num_sets=4, block_size=BLOCK, ways=1)
        self.assertEqual(pressure.windows, 0)
        self.assertEqual(pressure.oversubscribed_fraction, 0.0)
        self.assertEqual(pressure.windows_oversubscribed_fraction, 0.0)
        self.assertEqual(pressure.mean_distinct, 0.0)

    def test_bad_geometry_is_rejected(self) -> None:
        for kwargs in (
            {"num_sets": 0},
            {"ways": 0},
            {"window": 0},
            {"block_size": 0},
        ):
            args = {"num_sets": 4, "block_size": BLOCK, "ways": 1, "window": 8, **kwargs}
            with self.subTest(**kwargs), self.assertRaises(ValueError):
                window_pressure([(0, False)], **args)


class TestConflictStreams(unittest.TestCase):
    """Four aliasing streams: uniform cumulatively, crowded in every window."""

    def setUp(self) -> None:
        self.accesses = as_accesses(conflict_streams(words_per_stream=1024))

    def test_cumulative_histogram_is_perfectly_uniform(self) -> None:
        hot = hot_sets(self.accesses, size=4096, block_size=BLOCK, associativity=1)
        self.assertEqual(hot.num_sets, 64)
        self.assertEqual(hot.misses, 4096)
        self.assertEqual(hot.min_misses, 64)
        self.assertEqual(hot.max_misses, 64)
        self.assertEqual(hot.mean_misses, 64.0)
        self.assertEqual(hot.spread, 1.0)
        self.assertEqual(hot.top_share, hot.uniform_share)

    def test_every_touched_pair_is_oversubscribed_at_one_way(self) -> None:
        pressure = window_pressure(self.accesses, num_sets=64, block_size=BLOCK, ways=1)
        self.assertEqual(pressure.windows, 8)
        self.assertEqual(pressure.touched_pairs, 128)
        self.assertEqual(pressure.oversubscribed_pairs, 128)
        self.assertEqual(pressure.oversubscribed_fraction, 1.0)
        self.assertEqual(pressure.windows_oversubscribed_fraction, 1.0)
        # Exactly the four streams, every time.
        self.assertEqual(pressure.mean_distinct, 4.0)
        self.assertEqual(pressure.max_distinct, 4)

    def test_four_ways_removes_every_oversubscription(self) -> None:
        pressure = window_pressure(self.accesses, num_sets=64, block_size=BLOCK, ways=4)
        self.assertEqual(pressure.oversubscribed_pairs, 0)
        self.assertEqual(pressure.oversubscribed_fraction, 0.0)
        self.assertEqual(pressure.windows_oversubscribed, 0)


class TestUniformHistogramHidesTheCrowding(unittest.TestCase):
    """A naive 32x32 matmul in 4 KB, 4-way: flat histogram, crowded windows.

    The cumulative per-set miss counts span 2,192 to 2,200 out of a mean of
    2,192.5 -- flat to a third of a percent -- and the hottest 5% of sets
    hold 6.27% of the misses where a perfectly even spread would hold
    6.25%. Nothing in that histogram suggests a problem. The windowed
    measurement does: 968 of 1,352 touched (window, set) pairs saw more
    distinct blocks than the cache has ways, in every one of the 132
    windows, averaging 6.43 blocks competing for 4 ways.
    """

    def setUp(self) -> None:
        self.accesses = as_accesses(matmul(n=32))
        self.hot = hot_sets(self.accesses, size=4096, block_size=BLOCK, associativity=4)
        self.pressure = window_pressure(self.accesses, num_sets=16, block_size=BLOCK, ways=4)

    def test_cumulative_histogram_looks_uniform(self) -> None:
        self.assertEqual(self.hot.num_sets, 16)
        self.assertEqual(self.hot.misses, 35080)
        self.assertEqual(self.hot.min_misses, 2192)
        self.assertEqual(self.hot.max_misses, 2200)
        self.assertAlmostEqual(self.hot.mean_misses, 2192.5)
        self.assertLess(self.hot.spread, 1.005)
        self.assertAlmostEqual(self.hot.top_share, 0.062714, places=6)
        self.assertEqual(self.hot.uniform_share, 0.0625)
        self.assertLess(self.hot.top_share - self.hot.uniform_share, 0.001)

    def test_windows_are_badly_oversubscribed(self) -> None:
        self.assertEqual(self.pressure.windows, 132)
        self.assertEqual(self.pressure.touched_pairs, 1352)
        self.assertEqual(self.pressure.oversubscribed_pairs, 968)
        self.assertAlmostEqual(self.pressure.oversubscribed_fraction, 968 / 1352)
        self.assertEqual(self.pressure.windows_oversubscribed, 132)
        self.assertEqual(self.pressure.windows_oversubscribed_fraction, 1.0)
        self.assertAlmostEqual(self.pressure.mean_distinct, 6.4260, places=4)
        self.assertEqual(self.pressure.max_distinct, 10)

    def test_the_miss_rate_confirms_the_windows_not_the_histogram(self) -> None:
        """51.9% of accesses miss, which the flat histogram never hinted at."""
        self.assertAlmostEqual(self.hot.misses / self.hot.accesses, 0.519, places=3)

    def test_tiling_relieves_the_pressure_while_the_histogram_gets_worse(self) -> None:
        """The 8x8-tiled version of the same multiplication, same geometry.

        Misses fall from 51.9% to 1.7% of accesses, and set pressure falls
        with them: oversubscribed pairs 71.6% -> 20.6%, windows affected
        100% -> 44.4%, mean distinct blocks per touched set 6.43 -> 2.79.
        The cumulative histogram moves the OTHER way -- max/mean rises from
        1.003 to 1.107 -- so on that measure the fast version looks like the
        less uniform one. The windowed measurement is the one that tracks
        the miss rate.
        """
        accesses = as_accesses(blocked(n=32, tile=8))
        hot = hot_sets(accesses, size=4096, block_size=BLOCK, associativity=4)
        pressure = window_pressure(accesses, num_sets=hot.num_sets, block_size=BLOCK, ways=4)

        self.assertEqual(hot.misses, 1272)
        self.assertAlmostEqual(hot.misses / hot.accesses, 0.0173, places=4)
        self.assertEqual(pressure.oversubscribed_pairs, 208)
        self.assertAlmostEqual(pressure.oversubscribed_fraction, 208 / 1008)
        self.assertAlmostEqual(pressure.windows_oversubscribed_fraction, 64 / 144)
        self.assertAlmostEqual(pressure.mean_distinct, 2.7937, places=4)

        # Pressure tracks the miss rate; the cumulative spread does not.
        self.assertLess(pressure.oversubscribed_fraction, self.pressure.oversubscribed_fraction)
        self.assertLess(pressure.mean_distinct, self.pressure.mean_distinct)
        self.assertGreater(hot.spread, self.hot.spread)


class TestFormatting(unittest.TestCase):
    def test_hot_set_summary_mentions_the_uniform_baseline(self) -> None:
        hot = hot_sets(as_accesses(matmul(n=16)), 4096, BLOCK, 4)
        text = format_hot_sets(hot, top=3)
        self.assertIn("cumulative misses per set", text)
        self.assertIn("uniform would be", text)
        self.assertIn("busiest sets", text)

    def test_top_zero_omits_the_busiest_list(self) -> None:
        hot = hot_sets(as_accesses(matmul(n=16)), 4096, BLOCK, 4)
        self.assertNotIn("busiest sets", format_hot_sets(hot, top=0))

    def test_one_way_is_singular(self) -> None:
        pressure = window_pressure([(0, False)], num_sets=1, block_size=BLOCK, ways=1)
        self.assertIn("> 1 way)", format_window_pressure(pressure))
        pressure4 = window_pressure([(0, False)], num_sets=1, block_size=BLOCK, ways=4)
        self.assertIn("> 4 ways)", format_window_pressure(pressure4))


class TestSetsCLI(unittest.TestCase):
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

    def test_text_report_has_both_sections(self) -> None:
        code, out, _ = self.run_main([self.trace, "--size", "4k", "--assoc", "1"])
        self.assertEqual(code, 0)
        self.assertIn("cumulative misses per set", out)
        self.assertIn("windowed set pressure", out)
        self.assertIn("64 sets", out)
        self.assertIn("100.00% of touched pairs", out)

    def test_json_report(self) -> None:
        code, out, _ = self.run_main(
            [self.trace, "--size", "4k", "--assoc", "1", "--format", "json"]
        )
        self.assertEqual(code, 0)
        payload = json.loads(out)
        self.assertEqual(payload["trace"], self.trace)
        self.assertEqual(payload["geometry"]["num_sets"], 64)
        self.assertEqual(payload["geometry"]["size"], 4096)
        self.assertEqual(payload["hot_sets"]["misses"], payload["hot_sets"]["accesses"])
        self.assertEqual(payload["window_pressure"]["oversubscribed_fraction"], 1.0)

    def test_window_flag_changes_the_window_count(self) -> None:
        code, out, _ = self.run_main(
            [self.trace, "--size", "4k", "--assoc", "1", "--window", "128", "--format", "json"]
        )
        self.assertEqual(code, 0)
        payload = json.loads(out)
        self.assertEqual(payload["window_pressure"]["window"], 128)
        self.assertEqual(payload["window_pressure"]["windows"], 8)  # 1,024 / 128

    def test_missing_trace_is_reported(self) -> None:
        code, _, err = self.run_main([os.path.join(self._tmp.name, "nope.trace")])
        self.assertEqual(code, 2)
        self.assertIn("trace not found", err)

    def test_empty_trace_is_rejected(self) -> None:
        path = os.path.join(self._tmp.name, "empty.trace")
        with open(path, "w") as handle:
            handle.write("# nothing\n")
        code, _, err = self.run_main([path])
        self.assertEqual(code, 2)
        self.assertIn("no accesses", err)

    def test_bad_flags_fail_with_a_message(self) -> None:
        for argv, fragment in [
            ([self.trace, "--size", "0"], "positive"),
            ([self.trace, "--assoc", "0"], "positive integer"),
            ([self.trace, "--window", "0"], "positive integer"),
            ([self.trace, "--top", "-1"], "non-negative"),
            ([self.trace, "--size", "junk"], "not a size"),
            ([self.trace, "--size", "3000"], "multiple of"),
        ]:
            with self.subTest(argv=argv):
                code, _, err = self.run_main(argv)
                self.assertEqual(code, 2)
                self.assertIn(fragment, err)


if __name__ == "__main__":
    unittest.main(verbosity=2)
