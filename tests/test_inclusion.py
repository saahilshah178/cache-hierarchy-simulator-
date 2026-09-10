"""Known-answer tests for the per-level inclusion policies.

The hand-traced cases use one-set caches so that the replacement order is
the whole state: L1 is 1 set x 2 ways and L2 is 1 set x 4 ways (or 2 ways
where the tighter geometry makes the trace shorter), both LRU, 64 B blocks.
"""

from __future__ import annotations

import random
import unittest

from cachesim.cache import Cache
from cachesim.hierarchy import Hierarchy, Level

BLOCK = 64


def one_set(name: str, ways: int) -> Cache:
    """A cache with a single set of ``ways`` 64-byte blocks."""
    return Cache(name, size=ways * BLOCK, block_size=BLOCK, associativity=ways)


def two_level(inclusion: str, l1_ways: int = 2, l2_ways: int = 4) -> Hierarchy:
    """L1 (1 x l1_ways) over L2 (1 x l2_ways) with the given L2 policy."""
    return Hierarchy(
        [
            Level(one_set("L1", l1_ways), hit_time=4),
            Level(one_set("L2", l2_ways), hit_time=12, inclusion=inclusion),
        ],
        memory_access_time=100,
    )


def run(h: Hierarchy, blocks: list[int], write: set[int] | None = None) -> Hierarchy:
    """Feed a list of block numbers through ``h``; blocks in ``write`` are stores."""
    for i, block in enumerate(blocks):
        h.access(block * BLOCK, is_write=write is not None and i in write)
    return h


class TestNineIsUnchanged(unittest.TestCase):
    """NINE is the default and must behave exactly as it did before."""

    def test_default_is_nine(self) -> None:
        h = two_level("nine")
        self.assertEqual([lv.inclusion for lv in h.levels], ["nine", "nine"])
        self.assertEqual(h.levels[1].inclusion, Level(one_set("X", 1), 1).inclusion)

    def test_lower_level_may_drop_a_block_the_upper_still_holds(self) -> None:
        """0,1,0,2,0,3,0,4,0: block 0 stays hot in L1, so L2 never sees it
        again and ages it out at the 4 -- but under NINE L1 keeps it, and the
        final reference to 0 is an L1 hit."""
        h = run(two_level("nine"), [0, 1, 0, 2, 0, 3, 0, 4, 0])
        l1, l2 = h.levels[0].cache, h.levels[1].cache
        self.assertEqual((l1.hits, l1.misses), (4, 5))
        self.assertEqual(h.dram_reads, 5)
        self.assertTrue(l1.contains(0))
        self.assertFalse(l2.contains(0))  # the invariant an inclusive L2 forbids
        self.assertEqual(h.levels[0].back_invalidations, 0)
        h.check_inclusion()  # NINE constrains nothing


class TestInclusive(unittest.TestCase):
    def test_back_invalidation_costs_an_upper_level_hit(self) -> None:
        """The same 0,1,0,2,0,3,0,4,0 stream against an inclusive L2.

        Filling block 4 makes L2 evict block 0 (its LRU: L1 absorbed every
        later reference to it), which must now be back-invalidated out of
        L1, so the final reference to 0 misses everywhere instead of hitting
        in L1. One eviction, one back-invalidation, one extra DRAM read."""
        h = run(two_level("inclusive"), [0, 1, 0, 2, 0, 3, 0, 4, 0])
        l1, l2 = h.levels[0].cache, h.levels[1].cache
        self.assertEqual(h.levels[0].back_invalidations, 1)
        self.assertEqual(l1.invalidations, 1)
        self.assertEqual((l1.hits, l1.misses), (3, 6))
        self.assertEqual(h.dram_reads, 6)
        self.assertTrue(l1.contains(0))  # refetched by that last reference
        self.assertTrue(l2.contains(0))
        h.check_inclusion()

    def test_a_freed_way_is_reused_by_the_fill_that_caused_it(self) -> None:
        """With equal geometries (1 x 2 over 1 x 2) every L2 eviction hits a
        block L1 still holds. 0,1,0,2: filling 2 evicts block 0 from L2,
        back-invalidates it from L1, and the L1 fill then lands in the way
        that back-invalidation just freed, so block 1 survives."""
        h = run(two_level("inclusive", l1_ways=2, l2_ways=2), [0, 1, 0, 2])
        l1 = h.levels[0].cache
        self.assertEqual(h.levels[0].back_invalidations, 1)
        self.assertEqual(l1.evictions, 0)  # the way was already empty
        self.assertEqual(sorted(b for _, _, b, _ in l1.lines()), [1, 2])
        h.check_inclusion()

    def test_dirty_copy_above_goes_below_the_evicting_level(self) -> None:
        """The evicting level is losing the block too, so a dirty copy found
        above it cannot be parked there: it goes to the next level down --
        here DRAM, since L2 is the last level."""
        h = run(two_level("inclusive"), [0, 1, 0, 2, 0, 3, 0, 4], write={0})
        l1, l2 = h.levels[0].cache, h.levels[1].cache
        self.assertEqual(h.levels[0].back_invalidations, 1)
        self.assertEqual(l1.writebacks, 1)  # the dirty line did travel down
        self.assertEqual(l2.writebacks_received, 0)  # ... but not into L2
        self.assertEqual(h.dram_writes, 1)
        self.assertFalse(l2.is_dirty(0))
        h.check_inclusion()

    def test_writeback_allocation_also_back_invalidates(self) -> None:
        """An inclusive level must back-invalidate whatever it drops, even
        when the eviction was caused by a write-back allocating a line
        rather than by a demand fill."""
        h = two_level("inclusive", l1_ways=2, l2_ways=2)
        l2 = h.levels[1].cache
        h.access(0, is_write=True)  # block 0 dirty in L1, clean in L2
        l2.invalidate(0)  # L2 forgets it (leaving L1's copy stranded)
        h.access(1 * BLOCK)  # block 1 fills both
        h.access(2 * BLOCK)  # evicts dirty 0 from L1 -> L2 must allocate it
        # The write-back allocation evicted a line from L2; whichever it was,
        # inclusion is restored by back-invalidating it out of L1.
        self.assertEqual(l2.writeback_allocations, 1)
        self.assertGreaterEqual(h.levels[0].back_invalidations, 1)
        h.check_inclusion()

    def test_invariant_holds_over_a_random_stream(self) -> None:
        rng = random.Random(11)
        h = Hierarchy(
            [
                Level(Cache("L1", 512, 64, 2), 4),
                Level(Cache("L2", 2048, 64, 4), 12, inclusion="inclusive"),
                Level(Cache("L3", 8192, 64, 8), 40, inclusion="inclusive"),
            ],
            memory_access_time=100,
        )
        for _ in range(20_000):
            h.access(rng.randrange(0, 1 << 16), is_write=rng.random() < 0.3)
            h.check_inclusion()


class TestExclusive(unittest.TestCase):
    def test_a_hit_moves_the_block_up_and_leaves_the_levels_disjoint(self) -> None:
        """0,1,2,3,0 with an exclusive L2. The 2 and the 3 push blocks 0 and
        1 out of L1 and into L2; the final 0 hits in L2, is invalidated
        there, and swaps places with L1's LRU line."""
        h = run(two_level("exclusive"), [0, 1, 2, 3, 0])
        l1, l2 = h.levels[0].cache, h.levels[1].cache
        self.assertEqual((l2.hits, l2.misses), (1, 4))
        self.assertEqual(h.dram_reads, 4)
        self.assertEqual(sorted(b for _, _, b, _ in l1.lines()), [0, 3])
        self.assertEqual(sorted(b for _, _, b, _ in l2.lines()), [1, 2])
        h.check_inclusion()

    def test_a_demand_fetch_does_not_fill_the_exclusive_level(self) -> None:
        h = run(two_level("exclusive"), [0])
        l1, l2 = h.levels[0].cache, h.levels[1].cache
        self.assertTrue(l1.contains(0))
        self.assertFalse(l2.contains(0))
        self.assertEqual((l2.fills, l2.misses), (0, 1))

    def test_capacity_is_the_sum_of_the_two_levels(self) -> None:
        """Cycling 6 blocks through 2 + 4 ways. Exclusive holds all six, so
        only the 6 compulsory misses reach DRAM; NINE duplicates L1's two
        blocks inside L2, holds four, and misses on every single access."""
        cycle = [b for _ in range(3) for b in range(6)]
        excl = run(two_level("exclusive"), list(cycle))
        nine = run(two_level("nine"), list(cycle))
        self.assertEqual(excl.dram_reads, 6)
        self.assertEqual(excl.levels[1].cache.hits, 12)
        self.assertEqual(nine.dram_reads, 18)
        self.assertEqual(nine.levels[1].cache.hits, 0)
        excl.check_inclusion()

    def test_clean_victims_are_passed_down_not_dropped(self) -> None:
        """Under NINE only dirty lines cross the boundary; an exclusive level
        takes every victim, clean or dirty."""
        h = run(two_level("exclusive"), [0, 1, 2])
        l2 = h.levels[1].cache
        self.assertEqual(l2.fills, 1)  # block 0, evicted clean from L1
        self.assertEqual(l2.writebacks_received, 0)
        self.assertTrue(l2.contains(0))

    def test_a_dirty_victim_keeps_its_dirty_bit_across_the_boundary(self) -> None:
        h = two_level("exclusive")
        l1, l2 = h.levels[0].cache, h.levels[1].cache
        h.access(0, is_write=True)
        h.access(1 * BLOCK)
        h.access(2 * BLOCK)  # evicts dirty block 0 from L1 into L2
        self.assertTrue(l2.is_dirty(0))
        self.assertEqual(l2.writebacks_received, 1)
        self.assertEqual(h.dram_writes, 0)
        h.access(0)  # moves back up, still dirty, and L2 must not write it
        self.assertTrue(l1.is_dirty(0))
        self.assertFalse(l2.contains(0))
        self.assertEqual(l2.writebacks, 0)
        self.assertEqual(h.dram_writes, 0)
        self.assertEqual(h.flush(), 1)  # exactly one copy of the data survives

    def test_invariant_holds_over_a_random_stream(self) -> None:
        rng = random.Random(13)
        h = Hierarchy(
            [
                Level(Cache("L1", 512, 64, 2), 4),
                Level(Cache("L2", 2048, 64, 4), 12, inclusion="exclusive"),
                Level(Cache("L3", 8192, 64, 8), 40),
            ],
            memory_access_time=100,
        )
        for _ in range(20_000):
            h.access(rng.randrange(0, 1 << 16), is_write=rng.random() < 0.3)
            h.check_inclusion()

    def test_dirty_data_survives_an_exclusive_boundary(self) -> None:
        written: list[int] = []

        class Recording(Hierarchy):
            def _write_to_memory(self, block: int, charged: bool = False) -> int:
                written.append(block)
                return super()._write_to_memory(block, charged)

        rng = random.Random(17)
        h = Recording(
            [
                Level(Cache("L1", 512, 64, 2), 4),
                Level(Cache("L2", 2048, 64, 4), 12, inclusion="exclusive"),
            ],
            memory_access_time=100,
        )
        stored: set[int] = set()
        for _ in range(5000):
            addr = rng.randrange(0, 1 << 14)
            is_write = rng.random() < 0.3
            if is_write:
                stored.add(addr // BLOCK)
            h.access(addr, is_write)
        h.flush()
        self.assertTrue(stored <= set(written))


class TestInclusionValidation(unittest.TestCase):
    def test_first_level_may_not_declare_an_inclusion_policy(self) -> None:
        with self.assertRaises(ValueError) as ctx:
            Hierarchy([Level(one_set("L1", 2), 4, inclusion="inclusive")], 100)
        self.assertIn("no level above", str(ctx.exception))

    def test_unknown_policy_is_rejected(self) -> None:
        with self.assertRaises(ValueError) as ctx:
            Level(one_set("L2", 2), 4, inclusion="strict")
        self.assertIn("unknown inclusion policy", str(ctx.exception))

    def test_check_inclusion_reports_a_broken_invariant(self) -> None:
        h = two_level("inclusive")
        h.access(0)
        h.levels[1].cache.invalidate(0)  # break it behind the model's back
        with self.assertRaises(AssertionError) as ctx:
            h.check_inclusion()
        self.assertIn("inclusive", str(ctx.exception))

    def test_check_inclusion_reports_a_broken_exclusion(self) -> None:
        h = two_level("exclusive")
        h.access(0)
        h.levels[1].cache.allocate(0)
        with self.assertRaises(AssertionError) as ctx:
            h.check_inclusion()
        self.assertIn("exclusive", str(ctx.exception))


if __name__ == "__main__":
    unittest.main(verbosity=2)
