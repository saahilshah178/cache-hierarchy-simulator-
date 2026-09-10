"""Differential tests: the real simulator against the naive reference model.

Both implementations are driven with the same seeded streams through the
same configurations, and two things are compared:

* the per-access outcome sequence -- the cycle count returned by each
  access and the cumulative per-level miss vector after it, so a
  divergence is reported at the access where it first appears rather than
  as a difference in the final totals;
* every counter at every level, plus the hierarchy totals, the derived
  rates, and the DRAM write-back accounting after ``flush()``.

The technique is differential testing (W. M. McKeeman, "Differential
Testing for Software", *Digital Technical Journal* 10(1), 1998): two
implementations written from one specification are cross-checked on inputs
neither was written against, so only a bug present in both, in the same
form, can survive.

Each case asserts that the scenario actually exercised what it claims to
(evictions happened, dirty lines were written back, a write-back had to
allocate), so a configuration that degenerated into "everything hits"
fails rather than passing vacuously.
"""

from __future__ import annotations

import random
import unittest
from typing import Any

from cachesim.cache import Cache
from cachesim.config import parse_config
from cachesim.hierarchy import Hierarchy
from cachesim.reference import RefCache, RefHierarchy

Access = tuple[int, bool]
Outcome = tuple[int, tuple[int, ...]]

#: Every per-level counter both implementations maintain.
LEVEL_COUNTERS = (
    "hits",
    "misses",
    "read_hits",
    "read_misses",
    "write_hits",
    "write_misses",
    "fills",
    "evictions",
    "invalidations",
    "writebacks",
    "writebacks_received",
    "writeback_allocations",
    "compulsory_misses",
    "capacity_misses",
    "conflict_misses",
    "shadow_misses",
    "anti_conflict_hits",
    "accesses",
    "capacity_misses_aggregate",
    "conflict_misses_aggregate",
)

#: Hierarchy-wide counters.
HIERARCHY_COUNTERS = ("accesses", "reads", "writes", "dram_reads", "dram_writes", "total_time")


def level(
    name: str,
    size: int,
    assoc: int,
    hit_time: int,
    block_size: int = 64,
    policy: str = "lru",
) -> dict[str, Any]:
    """One level of a configuration dict."""
    return {
        "name": name,
        "size": size,
        "block_size": block_size,
        "associativity": assoc,
        "hit_time": hit_time,
        "policy": policy,
    }


def config(*levels: dict[str, Any], memory_access_time: int = 100) -> dict[str, Any]:
    return {"memory_access_time": memory_access_time, "levels": list(levels)}


def with_policy(cfg: dict[str, Any], policy: str) -> dict[str, Any]:
    """A copy of ``cfg`` with every level switched to ``policy``."""
    return {
        "memory_access_time": cfg["memory_access_time"],
        "levels": [{**lv, "policy": policy} for lv in cfg["levels"]],
    }


# -- geometries ---------------------------------------------------------------
# 64-byte blocks throughout, so a level's block count is size/64 and its set
# count is size/(64*associativity). The set counts are noted because
# non-powers of two exercise the ``block % num_sets`` mapping that a
# bit-field implementation would get wrong.

SINGLE_LEVEL: dict[str, dict[str, Any]] = {
    "direct_mapped_256B": config(level("L1", 256, 1, 4)),  # 4 sets
    "fully_associative_512B": config(level("L1", 512, 8, 4)),  # 1 set, 8 ways
    "three_sets_2way": config(level("L1", 384, 2, 4)),  # 3 sets: not a power of two
    "five_sets_direct": config(level("L1", 320, 1, 4)),  # 5 sets
    "seven_sets_4way": config(level("L1", 1792, 4, 4)),  # 7 sets
    "4KB_4way": config(level("L1", 4096, 4, 4)),  # 16 sets
    "64KB_8way": config(level("L1", 64 * 1024, 8, 4)),  # 128 sets
    "small_blocks_8B": config(level("L1", 256, 2, 4, block_size=8)),  # 16 sets, 8B lines
}

TWO_LEVEL: dict[str, dict[str, Any]] = {
    "tiny_inclusive_shapes": config(level("L1", 256, 1, 4), level("L2", 1024, 2, 12)),
    # L2 no larger than L1 and less associative: it drops lines L1 still
    # holds dirty, which is what forces writeback_allocations.
    "l2_no_bigger_than_l1": config(level("L1", 1024, 4, 4), level("L2", 1024, 1, 12)),
    "non_power_of_two_sets": config(level("L1", 384, 2, 4), level("L2", 1280, 4, 12)),
    "fully_associative_l1": config(level("L1", 512, 8, 4), level("L2", 8192, 8, 12)),
}

THREE_LEVEL: dict[str, dict[str, Any]] = {
    "desktop_shapes": config(
        level("L1", 2048, 4, 4), level("L2", 16384, 8, 12), level("L3", 65536, 16, 40)
    ),
    "narrowing_associativity": config(
        level("L1", 1024, 8, 4), level("L2", 2048, 2, 12), level("L3", 4096, 1, 40)
    ),
    "non_power_of_two_everywhere": config(
        level("L1", 384, 2, 4), level("L2", 1152, 3, 12), level("L3", 5760, 5, 40)
    ),
}


# -- streams ------------------------------------------------------------------


def uniform_stream(n: int, footprint: int, seed: int, write_prob: float = 0.3) -> list[Access]:
    """``n`` word-aligned uniform accesses over ``footprint`` bytes."""
    rng = random.Random(seed)
    return [(rng.randrange(footprint // 8) * 8, rng.random() < write_prob) for _ in range(n)]


def cyclic_stream(n: int, blocks: int, seed: int, write_prob: float = 0.2) -> list[Access]:
    """Walk ``blocks`` distinct 64-byte blocks in a fixed cycle.

    A working set slightly larger than a cache is the classic LRU worst
    case, so this stream keeps the eviction path busy at every level.
    """
    rng = random.Random(seed)
    return [((i % blocks) * 64 + rng.randrange(8) * 8, rng.random() < write_prob) for i in range(n)]


def clustered_stream(n: int, seed: int, write_prob: float = 0.4) -> list[Access]:
    """Alternate between a hot 1 KB region and a cold 8 MB region.

    The cold references evict the hot working set, so the hot ones come
    back as capacity and conflict misses rather than compulsory ones.
    """
    rng = random.Random(seed)
    out: list[Access] = []
    for _ in range(n):
        if rng.random() < 0.7:
            addr = rng.randrange(1024 // 8) * 8
        else:
            addr = rng.randrange((8 << 20) // 8) * 8
        out.append((addr, rng.random() < write_prob))
    return out


STREAMS: dict[str, list[Access]] = {
    "tight_2KB": uniform_stream(1500, 2 * 1024, seed=1),
    "medium_64KB": uniform_stream(1500, 64 * 1024, seed=2),
    "sparse_4MB": uniform_stream(1000, 4 << 20, seed=3),
    "read_only_tight": uniform_stream(1200, 4 * 1024, seed=4, write_prob=0.0),
    "write_heavy": uniform_stream(1200, 8 * 1024, seed=5, write_prob=0.9),
    "cyclic_130_blocks": cyclic_stream(1600, 130, seed=6),
    "clustered": clustered_stream(1200, seed=7),
}


class DifferentialCase(unittest.TestCase):
    """Machinery for running one configuration and one stream through both."""

    def run_both(self, cfg: dict[str, Any], stream: list[Access]) -> tuple[Hierarchy, RefHierarchy]:
        spec = parse_config(cfg)
        real = Hierarchy.from_spec(spec)
        ref = RefHierarchy.from_spec(spec)
        real_outcomes: list[Outcome] = []
        ref_outcomes: list[Outcome] = []
        for addr, is_write in stream:
            cycles = real.access(addr, is_write)
            real_outcomes.append((cycles, tuple(lv.cache.misses for lv in real.levels)))
            cycles = ref.access(addr, is_write)
            ref_outcomes.append((cycles, tuple(lv.cache.misses for lv in ref.levels)))
        self.assert_same_outcomes(real_outcomes, ref_outcomes, stream)
        self.assert_same_counters(real, ref)
        return real, ref

    def assert_same_outcomes(
        self, real: list[Outcome], ref: list[Outcome], stream: list[Access]
    ) -> None:
        """Compare access by access and report the first divergence."""
        for i, (got, want) in enumerate(zip(real, ref, strict=True)):
            if got != want:
                addr, is_write = stream[i]
                self.fail(
                    f"outcome diverged at access {i} (addr 0x{addr:x}, "
                    f"{'W' if is_write else 'R'}): simulator {got}, reference {want}"
                )

    def assert_same_counters(self, real: Hierarchy, ref: RefHierarchy) -> None:
        for name in HIERARCHY_COUNTERS:
            self.assertEqual(getattr(real, name), getattr(ref, name), f"hierarchy.{name}")
        self.assertEqual(real.memory_accesses, ref.memory_accesses)
        self.assertAlmostEqual(real.amat(), ref.amat(), places=12)
        self.assertAlmostEqual(real.measured_amat(), ref.measured_amat(), places=12)
        self.assertEqual(len(real.levels), len(ref.levels))
        for i, (rl, fl) in enumerate(zip(real.levels, ref.levels, strict=True)):
            for name in LEVEL_COUNTERS:
                self.assertEqual(
                    getattr(rl.cache, name),
                    getattr(fl.cache, name),
                    f"{rl.cache.name}.{name}",
                )
            self.assertAlmostEqual(rl.cache.miss_rate, fl.cache.miss_rate, places=12)
            self.assertAlmostEqual(real.global_miss_rate(i), ref.global_miss_rate(i), places=12)
            self.assertEqual(
                sorted((b, d) for _, _, b, d in rl.cache.lines()),
                sorted((b, d) for _, _, b, d in fl.cache.lines()),
                f"{rl.cache.name} residency",
            )

    def assert_flush_matches(self, real: Hierarchy, ref: RefHierarchy) -> None:
        self.assertEqual(real.flush(), ref.flush())
        self.assertEqual(real.dram_writes, ref.dram_writes)
        for rl, fl in zip(real.levels, ref.levels, strict=True):
            self.assertEqual(rl.cache.writebacks, fl.cache.writebacks)
            self.assertEqual(rl.cache.writebacks_received, fl.cache.writebacks_received)
            self.assertEqual(rl.cache.writeback_allocations, fl.cache.writeback_allocations)
            self.assertFalse(any(dirty for *_, dirty in rl.cache.lines()))
            self.assertFalse(any(dirty for *_, dirty in fl.cache.lines()))


class TestSingleLevel(DifferentialCase):
    def test_every_geometry_and_policy(self) -> None:
        exercised_eviction = False
        for geometry, cfg in SINGLE_LEVEL.items():
            for policy in ("lru", "fifo", "random"):
                for stream_name in ("tight_2KB", "sparse_4MB", "write_heavy"):
                    with self.subTest(geometry=geometry, policy=policy, stream=stream_name):
                        real, _ = self.run_both(with_policy(cfg, policy), STREAMS[stream_name])
                        exercised_eviction |= real.levels[0].cache.evictions > 0
        self.assertTrue(exercised_eviction, "no case caused an eviction")

    def test_dirty_evictions_reach_dram_identically(self) -> None:
        for policy in ("lru", "fifo", "random"):
            with self.subTest(policy=policy):
                cfg = with_policy(SINGLE_LEVEL["direct_mapped_256B"], policy)
                real, ref = self.run_both(cfg, STREAMS["write_heavy"])
                self.assertGreater(real.dram_writes, 0)
                self.assertEqual(real.dram_writes, real.levels[0].cache.writebacks)
                self.assert_flush_matches(real, ref)


class TestTwoLevel(DifferentialCase):
    def test_every_geometry_and_policy(self) -> None:
        for geometry, cfg in TWO_LEVEL.items():
            for policy in ("lru", "fifo", "random"):
                for stream_name in ("tight_2KB", "medium_64KB", "clustered"):
                    with self.subTest(geometry=geometry, policy=policy, stream=stream_name):
                        self.run_both(with_policy(cfg, policy), STREAMS[stream_name])

    def test_writeback_allocation_path(self) -> None:
        """L2 smaller and less associative than L1 loses lines L1 still
        holds dirty, so a write-back has to allocate in L2."""
        for policy in ("lru", "fifo", "random"):
            with self.subTest(policy=policy):
                cfg = with_policy(TWO_LEVEL["l2_no_bigger_than_l1"], policy)
                real, ref = self.run_both(cfg, STREAMS["write_heavy"])
                l1, l2 = real.levels[0].cache, real.levels[1].cache
                self.assertGreater(l2.writeback_allocations, 0)
                self.assertEqual(l1.writebacks, l2.writebacks_received)
                self.assertGreater(real.dram_writes, 0)
                self.assert_flush_matches(real, ref)

    def test_flush_after_each_stream(self) -> None:
        for stream_name, stream in STREAMS.items():
            with self.subTest(stream=stream_name):
                real, ref = self.run_both(TWO_LEVEL["tiny_inclusive_shapes"], stream)
                self.assert_flush_matches(real, ref)


class TestThreeLevel(DifferentialCase):
    def test_every_geometry_and_policy(self) -> None:
        for geometry, cfg in THREE_LEVEL.items():
            for policy in ("lru", "fifo", "random"):
                for stream_name in ("tight_2KB", "cyclic_130_blocks", "write_heavy"):
                    with self.subTest(geometry=geometry, policy=policy, stream=stream_name):
                        real, ref = self.run_both(with_policy(cfg, policy), STREAMS[stream_name])
                        self.assert_flush_matches(real, ref)

    def test_writeback_chain_reaches_dram_identically(self) -> None:
        cfg = THREE_LEVEL["narrowing_associativity"]
        real, ref = self.run_both(cfg, STREAMS["write_heavy"])
        l1, l2, l3 = (lv.cache for lv in real.levels)
        self.assertGreater(l3.writebacks, 0)
        self.assertEqual(l1.writebacks, l2.writebacks_received)
        self.assertEqual(l2.writebacks, l3.writebacks_received)
        self.assertEqual(l3.writebacks, real.dram_writes)
        self.assert_flush_matches(real, ref)


class TestPrimitives(unittest.TestCase):
    """The three primitives compared directly, outside a hierarchy.

    ``Hierarchy`` never calls ``invalidate``, so the primitive-level
    behaviour it depends on -- a removed dirty line counting as a
    write-back, an emptied way being refilled before any victim is chosen
    -- would otherwise go uncompared.
    """

    def test_probe_allocate_invalidate_sequences(self) -> None:
        for policy in ("lru", "fifo", "random"):
            for ways, sets in ((1, 8), (4, 2), (8, 1), (2, 3), (3, 5)):
                with self.subTest(policy=policy, ways=ways, sets=sets):
                    self.compare_primitive_stream(policy, ways, sets)

    def compare_primitive_stream(self, policy: str, ways: int, sets: int) -> None:
        size = sets * ways * 64
        real = Cache("C", size, 64, ways, policy=policy, rng_seed=9)
        ref = RefCache("C", size, 64, ways, policy=policy, rng_seed=9)
        rng = random.Random(ways * 100 + sets)
        blocks = sets * ways * 3  # three times the block count: forced turnover
        invalidations = 0
        for step in range(2000):
            block = rng.randrange(blocks)
            roll = rng.random()
            if roll < 0.55:
                is_write = rng.random() < 0.4
                got, want = real.probe(block, is_write), ref.probe(block, is_write)
                self.assertEqual(got, want, f"probe diverged at step {step}")
                if not got:
                    self.assertEqual(
                        real.allocate(block, dirty=is_write),
                        ref.allocate(block, dirty=is_write),
                        f"allocate diverged at step {step}",
                    )
            elif roll < 0.8:
                dirty = rng.random() < 0.5
                self.assertEqual(
                    real.allocate(block, dirty=dirty),
                    ref.allocate(block, dirty=dirty),
                    f"allocate diverged at step {step}",
                )
            else:
                removed = real.invalidate(block)
                self.assertEqual(removed, ref.invalidate(block), f"invalidate at step {step}")
                invalidations += removed is not None
            self.assertEqual(
                sorted(real.lines()), sorted(ref.lines()), f"contents diverged at step {step}"
            )
        self.assertGreater(invalidations, 0, "no line was ever invalidated")
        self.assertGreater(real.writebacks, 0, "no dirty line was ever written back")
        for name in LEVEL_COUNTERS:
            self.assertEqual(getattr(real, name), getattr(ref, name), name)

    def test_single_level_access_convenience_matches(self) -> None:
        real = Cache("C", 512, 64, 2, policy="fifo")
        ref = RefCache("C", 512, 64, 2, policy="fifo")
        for addr, is_write in STREAMS["tight_2KB"]:
            self.assertEqual(real.access(addr, is_write), ref.access(addr, is_write))
        for name in LEVEL_COUNTERS:
            self.assertEqual(getattr(real, name), getattr(ref, name), name)


class TestResetAndReuse(DifferentialCase):
    def test_reset_stats_midstream_matches(self) -> None:
        """A warm-up prefix followed by reset_stats must leave both models
        with the same contents and the same counters afterwards."""
        cfg = TWO_LEVEL["tiny_inclusive_shapes"]
        spec = parse_config(cfg)
        real, ref = Hierarchy.from_spec(spec), RefHierarchy.from_spec(spec)
        stream = STREAMS["tight_2KB"]
        for addr, is_write in stream[:400]:
            real.access(addr, is_write)
            ref.access(addr, is_write)
        real.reset_stats()
        ref.reset_stats()
        for addr, is_write in stream[400:]:
            self.assertEqual(real.access(addr, is_write), ref.access(addr, is_write))
        self.assert_same_counters(real, ref)
        # The reset kept the seen-block sets, so later misses are not all
        # compulsory; the case would be vacuous if they were.
        self.assertGreater(real.levels[0].cache.misses, real.levels[0].cache.compulsory_misses)


if __name__ == "__main__":
    unittest.main(verbosity=2)
