"""Known-answer and cross-check tests for Belady's OPT."""

from __future__ import annotations

import random
import unittest
from collections.abc import Iterable, Sequence
from itertools import islice

from cachesim.cache import Cache
from cachesim.config import ConfigError, parse_config
from cachesim.hierarchy import Hierarchy
from cachesim.opt import (
    OPTPolicy,
    next_use_indices,
    opt_misses,
    run_opt,
    simulate_with_opt,
)
from cachesim.workloads import SAMPLE_NAMES, get_workload

# Silberschatz, Galvin and Gagne, "Operating System Concepts", 10th ed.,
# 2018, sec. 10.4: the reference string used for every page-replacement
# example in the chapter.
SILBERSCHATZ = [7, 0, 1, 2, 0, 3, 0, 4, 2, 3, 0, 3, 2, 1, 2, 0, 1, 7, 0, 1]

BLOCK = 16


def brute_force_opt(blocks: Sequence[int], capacity: int) -> int:
    """Belady's rule, implemented as directly as possible.

    Rescans the rest of the stream for every eviction, so it is O(N^2 * C)
    and obviously correct; used only to check ``opt_misses``.
    """
    total = len(blocks)
    resident: list[int] = []
    misses = 0
    for i, block in enumerate(blocks):
        if block in resident:
            continue
        misses += 1
        if len(resident) == capacity:
            victim = resident[0]
            farthest = -1
            for candidate in resident:
                when = total  # never referenced again
                for j in range(i + 1, total):
                    if blocks[j] == candidate:
                        when = j
                        break
                if when > farthest:
                    farthest = when
                    victim = candidate
            resident.remove(victim)
        resident.append(block)
    return misses


def fully_associative(policy: str, capacity: int) -> Cache:
    """A single-set cache of ``capacity`` blocks: no set mapping in the way."""
    return Cache("FA", capacity * BLOCK, BLOCK, capacity, policy=policy, track_3c=False)


def misses_of(policy: str, blocks: Sequence[int], capacity: int) -> int:
    cache = fully_associative(policy, capacity)
    for block in blocks:
        cache.access(block * BLOCK)
    return cache.misses


def sample_accesses(name: str, limit: int) -> list[tuple[int, bool]]:
    """The first ``limit`` accesses of a sample workload as (addr, is_write)."""
    pairs: Iterable[tuple[int, str]] = get_workload(name).generator()
    return [(addr, op == "W") for addr, op in islice(pairs, limit)]


class TestNextUseIndices(unittest.TestCase):
    def test_next_use_indices_point_at_the_following_reference(self) -> None:
        # "never again" is len(blocks), which sorts after every real index.
        self.assertEqual(next_use_indices([1, 2, 1, 3, 2]), [2, 4, 5, 5, 5])
        self.assertEqual(next_use_indices([]), [])


class TestOptMisses(unittest.TestCase):
    """The fully associative Belady bound."""

    def test_the_silberschatz_string_with_three_frames(self) -> None:
        self.assertEqual(opt_misses(SILBERSCHATZ, 3), 9)

    def test_opt_matches_the_brute_force_oracle_on_random_streams(self) -> None:
        rng = random.Random(20)
        for distinct in (4, 9, 25):
            for capacity in (1, 2, 3, 8):
                stream = [rng.randrange(distinct) for _ in range(300)]
                self.assertEqual(
                    opt_misses(stream, capacity),
                    brute_force_opt(stream, capacity),
                    f"distinct={distinct} capacity={capacity}",
                )

    def test_opt_is_a_lower_bound_for_lru_and_fifo(self) -> None:
        rng = random.Random(21)
        for _ in range(20):
            stream = [rng.randrange(12) for _ in range(400)]
            for capacity in (2, 4, 8):
                optimal = opt_misses(stream, capacity)
                self.assertLessEqual(optimal, misses_of("lru", stream, capacity))
                self.assertLessEqual(optimal, misses_of("fifo", stream, capacity))

    def test_a_cache_large_enough_only_takes_compulsory_misses(self) -> None:
        stream = [7, 0, 1, 2, 0, 3, 0, 4]
        self.assertEqual(len(set(stream)), 6)
        self.assertEqual(opt_misses(stream, 6), 6)  # one per distinct block
        self.assertEqual(opt_misses(stream, 50), 6)

    def test_an_empty_stream_has_no_misses(self) -> None:
        self.assertEqual(opt_misses([], 4), 0)

    def test_a_capacity_below_one_block_is_rejected(self) -> None:
        with self.assertRaises(ValueError):
            opt_misses([1, 2, 3], 0)


class TestOptPolicy(unittest.TestCase):
    """The set-associative policy driven by a preloaded reference stream."""

    def test_run_opt_reproduces_the_silberschatz_answer(self) -> None:
        accesses = [(block * BLOCK, False) for block in SILBERSCHATZ]
        cache = run_opt(accesses, size=3 * BLOCK, block_size=BLOCK, associativity=3)
        self.assertEqual(cache.misses, 9)
        self.assertEqual(cache.num_sets, 1)  # fully associative: Belady's setting

    def test_run_opt_equals_the_oracle_when_fully_associative(self) -> None:
        # With one set the policy is Belady's original rule, so it must
        # agree exactly with the independent heap implementation and with
        # the brute-force oracle.
        rng = random.Random(22)
        for capacity in (2, 4, 7):
            stream = [rng.randrange(15) for _ in range(500)]
            accesses = [(block * BLOCK, False) for block in stream]
            cache = run_opt(accesses, capacity * BLOCK, BLOCK, capacity)
            self.assertEqual(cache.misses, opt_misses(stream, capacity))
            self.assertEqual(cache.misses, brute_force_opt(stream, capacity))

    def test_a_set_associative_opt_is_worse_than_the_fully_associative_bound(self) -> None:
        # Belady inside each set cannot undo a bad set mapping: the four
        # conflict streams still collide in a direct-mapped cache.
        accesses = sample_accesses("conflict", 20_000)
        blocks = [addr // 64 for addr, _ in accesses]
        direct_mapped = run_opt(accesses, 8 * 1024, 64, 1, track_3c=False)
        self.assertEqual(direct_mapped.misses, 20_000)  # every access misses
        self.assertEqual(opt_misses(blocks, 128), 2500)  # 8 KB, fully associative

    def test_opt_beats_lru_and_fifo_on_every_sample_workload(self) -> None:
        # Small single level (4 KB, 4-way, 64 B blocks) so the policies have
        # something to decide; the traces are truncated to keep the suite
        # quick, which does not weaken the ordering.
        for name in SAMPLE_NAMES:
            with self.subTest(workload=name):
                accesses = sample_accesses(name, 30_000)
                counts = {}
                for policy in ("fifo", "lru"):
                    cache = Cache("L1", 4096, 64, 4, policy=policy, track_3c=False)
                    for addr, is_write in accesses:
                        cache.access(addr, is_write)
                    counts[policy] = cache.misses
                counts["opt"] = run_opt(accesses, 4096, 64, 4, track_3c=False).misses
                self.assertLessEqual(counts["opt"], counts["lru"])
                self.assertLessEqual(counts["lru"], counts["fifo"])

    def test_opt_refills_an_invalidated_way_without_evicting(self) -> None:
        stream = [1, 2, 3, 4, 9, 1, 3, 4]
        accesses = [(block * BLOCK, False) for block in stream]
        cache = Cache("L1", 4 * BLOCK, BLOCK, 4, policy="opt")
        policy = cache.policy
        assert isinstance(policy, OPTPolicy)
        policy.preload([addr // BLOCK for addr, _ in accesses])
        for block in stream[:4]:
            cache.access(block * BLOCK)
        cache.invalidate(2)
        cache.access(9 * BLOCK)  # takes the freed way, evicts nothing
        self.assertEqual(cache.evictions, 0)
        self.assertEqual({b for _, _, b, _ in cache.lines()}, {1, 3, 4, 9})

    def test_asking_opt_to_run_without_the_future_fails_loudly(self) -> None:
        cache = Cache("L1", 4 * BLOCK, BLOCK, 4, policy="opt")
        with self.assertRaises(RuntimeError) as ctx:
            cache.access(0)
        self.assertIn("preload", str(ctx.exception))
        # Even the victim path, reached only when a set is full, says so.
        policy = cache.policy
        assert isinstance(policy, OPTPolicy)
        self.assertFalse(policy.preloaded)
        with self.assertRaises(RuntimeError):
            policy.victim(0)

    def test_a_stream_that_does_not_match_the_simulation_is_rejected(self) -> None:
        cache = Cache("L1", 4 * BLOCK, BLOCK, 4, policy="opt")
        policy = cache.policy
        assert isinstance(policy, OPTPolicy)
        policy.preload([1, 2, 3])
        cache.access(1 * BLOCK)
        with self.assertRaises(RuntimeError) as ctx:
            cache.access(9 * BLOCK)  # the stream says block 2 comes next
        self.assertIn("expected block 2", str(ctx.exception))

    def test_a_stream_that_runs_out_is_rejected(self) -> None:
        cache = Cache("L1", 4 * BLOCK, BLOCK, 4, policy="opt")
        policy = cache.policy
        assert isinstance(policy, OPTPolicy)
        policy.preload([1])
        cache.access(1 * BLOCK)
        with self.assertRaises(RuntimeError) as ctx:
            cache.access(1 * BLOCK)
        self.assertIn("ran out", str(ctx.exception))


class TestSimulateWithOpt(unittest.TestCase):
    """The two-pass driver for OPT at one level of a hierarchy."""

    def config(self, l2_policy: str) -> dict[str, object]:
        return {
            "memory_access_time": 100,
            "levels": [
                {"name": "L1", "size": 1024, "block_size": 64, "associativity": 2, "hit_time": 4},
                {
                    "name": "L2",
                    "size": 4096,
                    "block_size": 64,
                    "associativity": 4,
                    "hit_time": 12,
                    "policy": l2_policy,
                },
            ],
        }

    def stream(self) -> list[tuple[int, bool]]:
        rng = random.Random(7)
        return [(rng.randrange(300) * 64, rng.random() < 0.3) for _ in range(4000)]

    def test_opt_at_l2_beats_lru_at_l2_and_leaves_l1_untouched(self) -> None:
        accesses = self.stream()
        with_opt = simulate_with_opt(accesses, self.config("opt"))
        with_lru = Hierarchy.from_config(self.config("lru"))
        for addr, is_write in accesses:
            with_lru.access(addr, is_write)

        opt_l1, opt_l2 = (level.cache for level in with_opt.levels)
        lru_l1, lru_l2 = (level.cache for level in with_lru.levels)
        # L1 is above the level whose policy changed, so it is unaffected:
        # this is what makes one recording pass enough.
        self.assertEqual(opt_l1.misses, lru_l1.misses)
        self.assertEqual((opt_l1.hits, opt_l1.evictions), (lru_l1.hits, lru_l1.evictions))
        self.assertEqual(opt_l2.accesses, lru_l2.accesses)
        self.assertEqual(opt_l2.accesses, opt_l1.misses)
        self.assertEqual((lru_l1.misses, lru_l2.misses), (3787, 3168))
        self.assertEqual(opt_l2.misses, 2216)

    def test_opt_at_l1_matches_the_single_level_driver(self) -> None:
        accesses = self.stream()
        config = {
            "memory_access_time": 100,
            "levels": [
                {
                    "name": "L1",
                    "size": 4096,
                    "block_size": 64,
                    "associativity": 4,
                    "hit_time": 4,
                    "policy": "opt",
                }
            ],
        }
        hierarchy = simulate_with_opt(accesses, config)
        alone = run_opt(accesses, 4096, 64, 4)
        self.assertEqual(hierarchy.levels[0].cache.misses, alone.misses)

    def test_a_hierarchy_spec_is_accepted_as_well_as_a_dict(self) -> None:
        accesses = self.stream()
        spec = parse_config(self.config("opt"))
        hierarchy = simulate_with_opt(accesses, spec)
        self.assertEqual(hierarchy.levels[1].cache.misses, 2216)

    def test_opt_at_two_levels_is_rejected(self) -> None:
        config = self.config("opt")
        levels = config["levels"]
        assert isinstance(levels, list)
        levels[0]["policy"] = "opt"
        with self.assertRaises(ConfigError) as ctx:
            parse_config(config)
        self.assertIn("only meaningful at one level", str(ctx.exception))
        with self.assertRaises(ConfigError):
            simulate_with_opt(self.stream(), config)

    def test_a_hierarchy_without_opt_is_rejected(self) -> None:
        with self.assertRaises(ConfigError) as ctx:
            simulate_with_opt(self.stream(), self.config("lru"))
        self.assertIn("no level uses policy 'opt'", str(ctx.exception))


if __name__ == "__main__":
    unittest.main(verbosity=2)


class TestOptThroughTheFrontDoor(unittest.TestCase):
    """A configuration with an OPT level works wherever a configuration does."""

    def setUp(self) -> None:
        import os
        import tempfile

        from cachesim.workloads import write_trace

        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        rng = random.Random(11)
        self.accesses = [(rng.randrange(300) * 64, rng.random() < 0.3) for _ in range(3000)]
        self.trace = os.path.join(self._tmp.name, "t.trace")
        write_trace(self.trace, ((a, "W" if w else "R") for a, w in self.accesses))
        self.dir = self._tmp.name

    def config(self, opt_at: str, **l1_extra: object) -> dict[str, object]:
        l1: dict[str, object] = {
            "name": "L1",
            "size": 1024,
            "block_size": 64,
            "associativity": 2,
            "hit_time": 4,
            "policy": "opt" if opt_at == "L1" else "lru",
            **l1_extra,
        }
        l2: dict[str, object] = {
            "name": "L2",
            "size": 4096,
            "block_size": 64,
            "associativity": 4,
            "hit_time": 12,
            "policy": "opt" if opt_at == "L2" else "lru",
        }
        return {"memory_access_time": 100, "levels": [l1, l2]}

    def test_run_trace_uses_the_two_pass_driver(self) -> None:
        from cachesim import run_trace

        via_run = run_trace(self.trace, self.config("L2"))
        direct = simulate_with_opt(self.accesses, self.config("L2"))
        self.assertEqual(via_run.stats().to_dict(), direct.stats().to_dict())
        self.assertEqual(via_run.levels[1].cache.policy_name, "opt")

    def test_opt_at_l1_equals_the_single_level_optimum(self) -> None:
        """L1 sees the trace itself, so OPT there is exactly run_opt on it."""
        from cachesim import run_trace

        hierarchy = run_trace(self.trace, self.config("L1"))
        alone = run_opt(self.accesses, 1024, 64, 2)
        self.assertEqual(hierarchy.levels[0].cache.misses, alone.misses)
        self.assertGreater(alone.misses, 0)

    def test_warmup_resets_the_counters_but_not_the_future(self) -> None:
        from cachesim import run_trace

        warm = run_trace(self.trace, self.config("L2"), warmup=1000)
        cold = run_trace(self.trace, self.config("L2"))
        self.assertEqual(warm.accesses, 2000)
        self.assertLessEqual(warm.levels[1].cache.misses, cold.levels[1].cache.misses)
        swallowed = run_trace(self.trace, self.config("L2"), warmup=10_000)
        self.assertEqual(swallowed.accesses, 0)

    def test_the_opt_level_keeps_its_other_settings(self) -> None:
        from cachesim import run_trace

        hierarchy = run_trace(self.trace, self.config("L1", victim_cache=4, index="xor"))
        l1 = hierarchy.levels[0].cache
        self.assertIsNotNone(l1.victim)
        self.assertEqual(l1.index_name, "xor")
        self.assertEqual(hierarchy.accesses, 3000)

    def test_inclusion_restrictions_are_rejected_up_front(self) -> None:
        exclusive_opt = self.config("L2")
        exclusive_opt["levels"][1]["inclusion"] = "exclusive"  # type: ignore[index]
        with self.assertRaises(ConfigError) as ctx:
            simulate_with_opt(self.accesses, exclusive_opt)
        self.assertIn("inclusion 'nine'", str(ctx.exception))

        inclusive_below = self.config("L1")
        inclusive_below["levels"][1]["inclusion"] = "inclusive"  # type: ignore[index]
        with self.assertRaises(ConfigError) as ctx:
            simulate_with_opt(self.accesses, inclusive_below)
        self.assertIn("inclusive level", str(ctx.exception))

    def test_cli_run_and_compare_accept_an_opt_config(self) -> None:
        import contextlib
        import io
        import json
        import os

        from cachesim.cli import main

        path = os.path.join(self.dir, "opt.json")
        with open(path, "w") as f:
            json.dump(self.config("L2"), f)
        out = io.StringIO()
        with contextlib.redirect_stdout(out):
            code = main(["run", "--config", path, self.trace])
        self.assertEqual(code, 0)
        self.assertIn("OPT", out.getvalue())

        out = io.StringIO()
        with contextlib.redirect_stdout(out):
            code = main(["compare", self.trace, "--config", "default", "--config", path])
        self.assertEqual(code, 0)
        self.assertIn("opt", out.getvalue())
