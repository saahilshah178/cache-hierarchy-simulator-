"""Tests for the invariant checker and for ``cachesim run --check``.

Two things have to be true of a checker: it stays quiet on a correct run,
and it speaks up when something is wrong. The first is covered by running
real workloads through several configurations; the second by corrupting
one counter, or one line of cache state, at a time and asserting both that
a violation is reported and that its message names the identity that
broke.
"""

from __future__ import annotations

import contextlib
import io
import os
import random
import tempfile
import unittest
from collections.abc import Callable, Iterator
from typing import Any

from cachesim import run_trace
from cachesim.cache import EMPTY, Cache
from cachesim.cli import main
from cachesim.config import DEFAULT_CONFIG, CacheSpec, ConstantMemorySpec, HierarchySpec
from cachesim.hierarchy import Hierarchy, Level
from cachesim.invariants import check_hierarchy, check_invariants
from cachesim.workloads import Access, conflict_streams, matmul, sequential, write_trace

SMALL = HierarchySpec(
    levels=(
        CacheSpec("L1", size=1024, block_size=64, associativity=2, hit_time=4),
        CacheSpec("L2", size=2048, block_size=64, associativity=4, hit_time=12),
        CacheSpec("L3", size=8192, block_size=64, associativity=8, hit_time=40),
    ),
    memory=ConstantMemorySpec(100),
)


def simulated(spec: HierarchySpec = SMALL, accesses: int = 4000, seed: int = 5) -> Hierarchy:
    """A hierarchy that has run a mixed read/write stream."""
    h = Hierarchy.from_spec(spec)
    rng = random.Random(seed)
    for _ in range(accesses):
        h.access(rng.randrange(0, 1 << 15), is_write=rng.random() < 0.3)
    return h


class TestCleanRuns(unittest.TestCase):
    def test_a_fresh_hierarchy_is_consistent(self) -> None:
        h = Hierarchy.from_config(DEFAULT_CONFIG)
        self.assertEqual(check_invariants(h), [])

    def test_sample_workloads_are_consistent(self) -> None:
        workloads: list[tuple[str, Callable[[], Iterator[Access]]]] = [
            ("sequential", lambda: sequential(buffer_bytes=32 * 1024, passes=2)),
            ("conflict", lambda: conflict_streams(words_per_stream=1024)),
            ("matmul", lambda: matmul(n=16)),
        ]
        for name, workload in workloads:
            with self.subTest(workload=name):
                h = Hierarchy.from_config(DEFAULT_CONFIG)
                for addr, op in workload():
                    h.access(addr, op == "W")
                self.assertEqual(check_invariants(h), [])

    def test_consistent_after_reset_and_after_flush(self) -> None:
        h = simulated()
        self.assertEqual(check_invariants(h), [])
        h.flush()
        self.assertEqual(check_invariants(h, after_flush=True), [])
        h.reset_stats()
        self.assertEqual(check_invariants(h), [])

    def test_every_policy_and_a_single_level(self) -> None:
        for policy in ("lru", "fifo", "random"):
            with self.subTest(policy=policy):
                spec = HierarchySpec(
                    levels=(CacheSpec("L1", 512, 64, 2, hit_time=4, policy=policy, rng_seed=3),),
                    memory=ConstantMemorySpec(50),
                )
                self.assertEqual(check_invariants(simulated(spec)), [])

    def test_the_sweep_actually_checks_something(self) -> None:
        report = check_hierarchy(simulated())
        self.assertTrue(report.ok)
        self.assertGreater(report.checked, 40)
        self.assertEqual(report.skipped, ())


class TestCorruptedCounters(unittest.TestCase):
    """Each case breaks exactly one identity and names it."""

    def assert_reports(self, h: Hierarchy, fragment: str, *, after_flush: bool = False) -> None:
        violations = check_invariants(h, after_flush=after_flush)
        self.assertTrue(violations, f"no violation reported; expected one mentioning {fragment!r}")
        self.assertTrue(
            any(fragment in v for v in violations),
            f"expected a violation mentioning {fragment!r}, got {violations}",
        )

    def test_a_stray_hit_breaks_the_counter_algebra(self) -> None:
        h = simulated()
        h.levels[0].cache.hits += 1
        self.assert_reports(h, "read_hits + write_hits != hits")

    def test_a_stray_miss_breaks_the_level_chain(self) -> None:
        h = simulated()
        h.levels[0].cache.misses += 1
        self.assert_reports(h, "accesses != L1.misses")

    def test_a_lost_compulsory_miss_breaks_the_three_c_sum(self) -> None:
        h = simulated()
        h.levels[1].cache.compulsory_misses -= 1
        self.assert_reports(h, "compulsory + capacity + conflict != misses")

    def test_a_stray_anti_conflict_hit_breaks_the_taxonomy_bridge(self) -> None:
        h = simulated()
        h.levels[0].cache.anti_conflict_hits += 1
        self.assert_reports(h, "capacity_aggregate != capacity + anti_conflict_hits")

    def test_a_lost_shadow_miss_is_detected(self) -> None:
        h = simulated()
        h.levels[2].cache.shadow_misses += 3
        self.assert_reports(h, "shadow_misses != compulsory + capacity + anti_conflict_hits")

    def test_a_writeback_that_nobody_received(self) -> None:
        h = simulated()
        h.levels[0].cache.writebacks += 1
        self.assert_reports(h, "writebacks_received != L1.writebacks")

    def test_a_dram_write_that_no_level_produced(self) -> None:
        h = simulated()
        h.dram_writes += 1
        self.assert_reports(h, "dram_writes != L3.writebacks")

    def test_a_wrong_cycle_count_is_detected(self) -> None:
        h = simulated()
        h.read_cycles += 1
        violations = check_invariants(h)
        self.assertEqual(len(violations), 2)  # the sum and the AMAT that follows from it
        self.assertIn("total_cycles !=", violations[0])
        self.assertIn("amat() != measured_amat()", violations[1])

    def test_an_uncounted_fill_is_detected(self) -> None:
        h = simulated()
        h.levels[1].cache.fills += 1
        self.assert_reports(h, "fills != misses + writeback_allocations")

    def test_a_dram_read_that_no_level_missed(self) -> None:
        h = simulated()
        h.dram_reads += 1
        self.assert_reports(h, "dram_reads != L3.misses")

    def test_reads_and_writes_must_partition_the_accesses(self) -> None:
        h = simulated()
        h.writes += 1
        self.assert_reports(h, "reads + writes != accesses")

    def test_an_invalidation_outside_the_hierarchy_is_explained(self) -> None:
        """Removing a dirty line with Cache.invalidate counts a write-back
        that no level below receives; the message says so."""
        h = Hierarchy.from_spec(SMALL)
        h.access(0x0, is_write=True)
        h.levels[0].cache.invalidate(0)
        violations = check_invariants(h)
        self.assertEqual(len(violations), 1)
        self.assertIn("writebacks_received != L1.writebacks", violations[0])
        self.assertIn("1 invalidation(s)", violations[0])


class TestCorruptedState(unittest.TestCase):
    def test_a_block_held_in_two_ways_is_detected(self) -> None:
        h = simulated()
        cache = h.levels[0].cache
        set_idx, way, block, _ = next(iter(cache.lines()))
        other = 1 if way == 0 else 0
        cache._blocks[set_idx][other] = block  # duplicate the block into a second way
        violations = check_invariants(h)
        self.assertTrue(any("more than one way" in v for v in violations))

    def test_a_line_in_the_wrong_set_is_detected(self) -> None:
        cache = Cache("L1", 512, 64, 1)  # 8 sets, direct-mapped
        h = Hierarchy([Level(cache, 4)], memory_access_time=100)
        h.access(0x0)
        cache._blocks[3][0] = 0  # block 0 belongs in set 0, not set 3
        violations = check_invariants(h)
        self.assertTrue(any("wrong set" in v for v in violations), violations)

    def test_a_dirty_line_after_a_flush_is_detected(self) -> None:
        h = simulated()
        h.flush()
        self.assertEqual(check_invariants(h, after_flush=True), [])
        cache = h.levels[0].cache
        set_idx, way, _, _ = next(iter(cache.lines()))
        cache._dirty[set_idx][way] = True
        self.assert_dirty_reported(h)

    def assert_dirty_reported(self, h: Hierarchy) -> None:
        self.assertEqual(check_invariants(h), [])  # only checked when asked
        violations = check_invariants(h, after_flush=True)
        self.assertTrue(any("dirty lines after a flush" in v for v in violations), violations)

    def test_an_empty_way_reported_as_valid_is_detected(self) -> None:
        """A line whose block number is the EMPTY sentinel would be invisible
        to contains(), so lines() and contains() must not disagree."""
        cache = Cache("L1", 512, 64, 2)
        h = Hierarchy([Level(cache, 4)], memory_access_time=100)
        for block in range(4):
            h.access(block * 64)
        set_idx, way, _, _ = next(iter(cache.lines()))
        cache._blocks[set_idx][way] = EMPTY - 1  # a block number no set maps to
        violations = check_invariants(h)
        self.assertTrue(any("wrong set" in v or "contains()" in v for v in violations), violations)


class TestOptionalFeatureGuards(unittest.TestCase):
    """A later feature must be able to relax an identity without editing it.

    The checker looks for optional attributes rather than assuming they are
    absent, so adding a transfer time or a prefetcher makes the affected
    identities step aside instead of firing spuriously.
    """

    def test_a_bus_width_on_a_level_skips_the_timing_identities(self) -> None:
        h = simulated()
        h.read_cycles += 1000  # would be a violation under the constant model
        self.assertTrue(check_invariants(h))
        h.levels[1].bus_width = 32
        report = check_hierarchy(h)
        self.assertTrue(report.ok, report.violations)
        self.assertEqual(report.skipped, ("timing identities (L2: bus_width is set)",))

    def test_a_non_constant_memory_model_skips_the_timing_identities(self) -> None:
        h = simulated()
        h.read_cycles -= 7
        self.assertTrue(check_invariants(h))
        h.memory_access_time = None  # what a row-buffer DRAM model reports
        report = check_hierarchy(h)
        self.assertTrue(report.ok, report.violations)
        self.assertEqual(
            report.skipped, ("timing identities (memory model is not a constant latency)",)
        )

    def test_a_prefetcher_skips_the_traffic_flow_identities(self) -> None:
        h = simulated()
        h.levels[1].cache.fills += 5  # a prefetcher would fill without a miss
        self.assertTrue(check_invariants(h))
        h.levels[1].prefetcher = object()  # type: ignore[assignment]
        report = check_hierarchy(h)
        self.assertTrue(report.ok, report.violations)
        self.assertEqual(len(report.skipped), 3)  # conservation, traffic flow, timing
        self.assertTrue(all("L2: prefetcher is" in reason for reason in report.skipped))

    def test_a_non_nine_inclusion_policy_skips_the_traffic_flow_identities(self) -> None:
        h = simulated()
        h.levels[2].cache.fills += 2  # an exclusive level fills on eviction
        self.assertTrue(check_invariants(h))
        h.levels[2].inclusion = "exclusive"
        report = check_hierarchy(h)
        self.assertTrue(report.ok, report.violations)
        self.assertTrue(all("L3: inclusion is 'exclusive'" in r for r in report.skipped))

    def test_a_write_policy_skips_the_traffic_flow_identities(self) -> None:
        h = simulated()
        h.levels[0].cache.fills += 1  # a no-write-allocate level declines a fill
        self.assertTrue(check_invariants(h))
        h.levels[0].write_allocate = False
        report = check_hierarchy(h)
        self.assertTrue(report.ok, report.violations)
        self.assertTrue(all("L1: write_allocate is False" in r for r in report.skipped))

    def test_default_feature_values_do_not_relax_anything(self) -> None:
        h = simulated()
        h.levels[0].bus_width = None
        h.levels[0].inclusion = "nine"
        h.levels[0].write_policy = "write-back"
        h.levels[0].write_allocate = True
        self.assertEqual(check_hierarchy(h).skipped, ())

    def test_a_run_with_no_accesses_skips_the_timing_identities(self) -> None:
        report = check_hierarchy(Hierarchy.from_spec(SMALL))
        self.assertTrue(report.ok)
        self.assertEqual(report.skipped, ("timing identities (no accesses were simulated)",))


class TestVictimBuffer(unittest.TestCase):
    def test_a_victim_buffer_raises_the_residency_bound(self) -> None:
        """Lines parked in the victim buffer are resident at the level, so a
        level with a 4-entry buffer may hold num_blocks + 4 lines."""
        spec = HierarchySpec(
            levels=(
                CacheSpec(
                    "L1", size=1024, block_size=64, associativity=1, hit_time=3, victim_cache=4
                ),
                CacheSpec("L2", size=8192, block_size=64, associativity=8, hit_time=12),
            ),
            memory=ConstantMemorySpec(100),
        )
        h = simulated(spec)
        l1 = h.levels[0].cache
        self.assertGreater(sum(1 for _ in l1.lines()), l1.num_blocks)
        report = check_hierarchy(h)
        self.assertEqual(report.violations, ())
        self.assertTrue(any("victim cache" in reason for reason in report.skipped))


class TestRunCheckFlag(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.trace = os.path.join(self._tmp.name, "seq.trace")
        write_trace(self.trace, sequential(buffer_bytes=8 * 1024, passes=2))

    def run_main(self, argv: list[str]) -> tuple[int, str, str]:
        out, err = io.StringIO(), io.StringIO()
        with contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
            try:
                code = main(argv)
            except SystemExit as exc:
                code = exc.code if isinstance(exc.code, int) else 1
        return code, out.getvalue(), err.getvalue()

    def test_check_passes_on_a_normal_run(self) -> None:
        code, out, err = self.run_main(["run", "--check", self.trace])
        self.assertEqual(code, 0)
        self.assertIn("CACHE HIERARCHY SIMULATION REPORT", out)
        self.assertRegex(err, r"invariants: \d+ checks passed")

    def test_check_keeps_json_output_parseable(self) -> None:
        import json

        code, out, err = self.run_main(["run", "--check", "--format", "json", self.trace])
        self.assertEqual(code, 0)
        self.assertEqual(json.loads(out)["accesses"], 2048)
        self.assertIn("checks passed", err)  # diagnostics went to stderr

    def test_without_the_flag_nothing_is_checked(self) -> None:
        code, _, err = self.run_main(["run", self.trace])
        self.assertEqual(code, 0)
        self.assertNotIn("invariants", err)

    def test_a_violation_exits_non_zero_and_lists_it(self) -> None:
        """The CLI path is exercised by corrupting the hierarchy that
        run_trace hands back."""
        from unittest import mock

        import cachesim.cli as cli_module

        def corrupting(*args: Any, **kwargs: Any) -> Hierarchy:
            hierarchy = run_trace(*args, **kwargs)
            hierarchy.read_cycles += 13
            hierarchy.levels[0].cache.hits += 1
            return hierarchy

        with mock.patch.object(cli_module, "run_trace", corrupting):
            code, _, err = self.run_main(["run", "--check", self.trace])
        self.assertEqual(code, 1)
        self.assertIn("error: invariant violated:", err)
        self.assertIn("read_hits + write_hits != hits", err)
        self.assertIn("total_cycles !=", err)


class TestPublicAPI(unittest.TestCase):
    """The checker is part of the package's public surface.

    Every other library-level module (cache, config, hierarchy, policies,
    report, stats, trace) is reachable as ``from cachesim import X``, so
    the checker is too. ``cachesim.reference`` deliberately stays out: it
    is a test oracle, not an API.
    """

    def test_the_checker_is_reachable_from_the_top_level_package(self) -> None:
        import cachesim
        import cachesim.invariants as module

        self.assertIs(cachesim.check_invariants, module.check_invariants)
        self.assertIs(cachesim.check_hierarchy, module.check_hierarchy)
        self.assertIs(cachesim.CheckReport, module.CheckReport)

    def test_the_checker_is_listed_in_dunder_all(self) -> None:
        import cachesim

        for name in ("CheckReport", "check_hierarchy", "check_invariants"):
            self.assertIn(name, cachesim.__all__)

    def test_every_exported_name_exists(self) -> None:
        import cachesim

        for name in cachesim.__all__:
            self.assertTrue(hasattr(cachesim, name), name)


if __name__ == "__main__":
    unittest.main(verbosity=2)
