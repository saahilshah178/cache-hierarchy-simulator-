"""Known-answer tests for the reference model itself.

The reference model is only useful as an oracle if it is independently
correct, so it is pinned here against hand-worked scenarios rather than
against ``cachesim.cache``. The cross-checks between the two live in
``tests/test_differential.py``.
"""

from __future__ import annotations

import unittest
from random import Random

from cachesim.config import DEFAULT_CONFIG, parse_config
from cachesim.reference import RefCache, RefHierarchy, RefLevel


def ref_cache(ways: int, sets: int = 2, block: int = 16, policy: str = "lru") -> RefCache:
    """A deliberately tiny reference cache: sets*ways blocks of 16 bytes."""
    return RefCache(
        "T", size=sets * ways * block, block_size=block, associativity=ways, policy=policy
    )


class TestRefCacheGeometry(unittest.TestCase):
    def test_derived_geometry(self) -> None:
        c = RefCache("L1", size=32 * 1024, block_size=64, associativity=4)
        self.assertEqual((c.num_sets, c.num_blocks), (128, 512))

    def test_block_set_and_tag_arithmetic(self) -> None:
        c = ref_cache(ways=1, sets=4)  # 4 sets of 16-byte blocks
        self.assertEqual(c.block_of(0x25), 2)
        self.assertEqual(c.set_of(6), 2)
        self.assertEqual(c.tag_of(6), 1)

    def test_rejects_degenerate_geometry(self) -> None:
        with self.assertRaises(ValueError):
            RefCache("bad", size=96, block_size=48, associativity=1)  # block not a power of two
        with self.assertRaises(ValueError):
            RefCache("bad", size=100, block_size=64, associativity=1)  # not a whole set count
        with self.assertRaises(ValueError):
            RefCache("bad", size=64, block_size=64, associativity=1, policy="clock")


class TestRefReplacement(unittest.TestCase):
    """One set of two ways; blocks 0, 1, 2 all map to it."""

    def stream(self, policy: str) -> RefCache:
        c = ref_cache(ways=2, sets=1, policy=policy)
        for block in (0, 1, 0, 2):
            c.access(block * 16)
        return c

    def test_lru_evicts_the_least_recently_used_way(self) -> None:
        # 0, 1 fill both ways; the hit on 0 refreshes it; 2 therefore
        # evicts 1, and a later reference to 1 misses.
        c = self.stream("lru")
        self.assertTrue(c.contains(0))
        self.assertFalse(c.contains(1))
        self.assertTrue(c.contains(2))
        self.assertEqual((c.hits, c.misses), (1, 3))
        self.assertFalse(c.access(1 * 16))  # miss

    def test_fifo_ignores_the_hit_and_evicts_the_oldest_fill(self) -> None:
        # Same stream: FIFO does not refresh 0 on the hit, so 2 evicts 0
        # and a later reference to 1 hits.
        c = self.stream("fifo")
        self.assertFalse(c.contains(0))
        self.assertTrue(c.contains(1))
        self.assertTrue(c.contains(2))
        self.assertEqual((c.hits, c.misses), (1, 3))
        self.assertTrue(c.access(1 * 16))  # hit

    def test_empty_ways_are_filled_lowest_first(self) -> None:
        c = ref_cache(ways=4, sets=1)
        c.access(0 * 16)
        c.access(1 * 16)
        self.assertEqual([(w, b) for _, w, b, _ in c.lines()], [(0, 0), (1, 1)])
        c.invalidate(0)
        c.access(2 * 16)  # way 0 is free again and is used first
        self.assertEqual([(w, b) for _, w, b, _ in c.lines()], [(0, 2), (1, 1)])


class TestRefRandomPolicy(unittest.TestCase):
    def test_victim_way_is_one_randrange_draw_per_eviction(self) -> None:
        """The random policy must consume ``Random(rng_seed)`` exactly as
        ``cachesim.policies.RandomPolicy`` does: one ``randrange(ways)``
        call per eviction and none for a fill into a free way.

        The check replays an independently seeded ``Random`` and asserts the
        full way-by-way contents after each eviction, so both the number of
        draws and the way index each one selects are pinned.
        """
        seed, ways = 7, 4
        c = RefCache("R", ways * 16, 16, ways, policy="random", rng_seed=seed)
        oracle = Random(seed)
        for block in range(ways):  # free ways: no draws
            c.access(block * 16)
        blocks_by_way = list(range(ways))
        self.assertEqual([b for _, _, b, _ in c.lines()], blocks_by_way)
        for block in range(ways, ways + 25):
            way = oracle.randrange(ways)
            c.access(block * 16)
            blocks_by_way[way] = block
            self.assertEqual([b for _, _, b, _ in c.lines()], blocks_by_way)
        self.assertEqual(c.evictions, 25)


class TestRefThreeC(unittest.TestCase):
    def test_compulsory_conflict_and_capacity_labels(self) -> None:
        """Direct-mapped, 2 blocks. Shadow is fully-associative LRU, also 2.

        0 -> compulsory; 2 -> compulsory (evicts 0 from set 0);
        0 -> conflict (the shadow still holds it);
        1 -> compulsory (pushes 2 out of the shadow);
        2 -> capacity (the shadow had already evicted it).
        """
        c = ref_cache(ways=1, sets=2)
        for block in (0, 2, 0, 1, 2):
            c.access(block * 16)
        self.assertEqual((c.hits, c.misses), (0, 5))
        self.assertEqual(c.compulsory_misses, 3)
        self.assertEqual(c.conflict_misses, 1)
        self.assertEqual(c.capacity_misses, 1)
        self.assertEqual(c.shadow_misses, 4)
        self.assertEqual(c.anti_conflict_hits, 0)
        self.assertEqual(c.capacity_misses_aggregate, 1)
        self.assertEqual(c.conflict_misses_aggregate, 1)

    def test_anti_conflict_hit_makes_aggregate_conflict_negative(self) -> None:
        """A 2-set, 2-way cache can hold a block that a fully-associative
        LRU cache of the same capacity has already evicted.

        Blocks 1, 3, 5, 7 all land in set 1 and churn there; block 0 sits
        undisturbed in set 0 but falls off the end of the shadow's LRU
        list, so the final reference to 0 hits here and would have missed
        a fully-associative cache.
        """
        c = ref_cache(ways=2, sets=2)
        for block in (0, 1, 3, 5, 7, 0):
            c.access(block * 16)
        self.assertEqual((c.hits, c.misses), (1, 5))
        self.assertEqual(c.compulsory_misses, 5)
        self.assertEqual((c.capacity_misses, c.conflict_misses), (0, 0))
        self.assertEqual(c.anti_conflict_hits, 1)
        self.assertEqual(c.shadow_misses, 6)
        self.assertEqual(c.capacity_misses_aggregate, 1)
        self.assertEqual(c.conflict_misses_aggregate, -1)

    def test_track_3c_off_leaves_the_classifier_idle(self) -> None:
        c = RefCache("T", 64, 16, 1, track_3c=False)  # 4 sets, direct-mapped
        for block in (0, 4, 0):  # 4 aliases 0, so all three miss
            c.access(block * 16)
        self.assertEqual(c.misses, 3)
        self.assertEqual((c.compulsory_misses, c.shadow_misses), (0, 0))


class TestRefThrashing(unittest.TestCase):
    def test_cyclic_working_set_of_capacity_plus_one_never_hits(self) -> None:
        """LRU evicts exactly the block that is needed next, so a cycle one
        block longer than the cache misses on every reference."""
        capacity = 4
        c = ref_cache(ways=capacity, sets=1)  # fully associative, 4 blocks
        for _ in range(10):
            for block in range(capacity + 1):
                c.access(block * 16)
        self.assertEqual(c.accesses, 50)
        self.assertEqual((c.hits, c.misses), (0, 50))
        self.assertEqual(c.miss_rate, 1.0)

    def test_cycle_that_fits_misses_only_once_per_block(self) -> None:
        capacity = 4
        c = ref_cache(ways=capacity, sets=1)
        for _ in range(10):
            for block in range(capacity):
                c.access(block * 16)
        self.assertEqual((c.hits, c.misses), (36, 4))


class TestRefInvalidateAndDirty(unittest.TestCase):
    def test_invalidate_reports_and_counts_a_dirty_line(self) -> None:
        c = ref_cache(ways=2, sets=1)
        c.access(0, is_write=True)
        self.assertTrue(c.is_dirty(0))
        removed = c.invalidate(0)
        self.assertEqual(removed, (0, True))
        self.assertEqual((c.invalidations, c.writebacks), (1, 1))
        self.assertIsNone(c.invalidate(0))

    def test_allocate_on_a_resident_block_only_ors_the_dirty_bit(self) -> None:
        c = ref_cache(ways=2, sets=1)
        c.allocate(0)
        c.allocate(1)
        self.assertIsNone(c.allocate(0, dirty=True))
        self.assertTrue(c.is_dirty(0))
        self.assertEqual(c.fills, 2)  # the third allocate installed nothing
        # It also left the LRU order alone: 0 is still the older way.
        c.allocate(2)
        self.assertFalse(c.contains(0))

    def test_clean_and_mark_dirty_report_residency(self) -> None:
        c = ref_cache(ways=1, sets=2)
        c.allocate(0)
        self.assertTrue(c.mark_dirty(0))
        self.assertTrue(c.clean(0))
        self.assertFalse(c.is_dirty(0))
        self.assertFalse(c.mark_dirty(99))
        self.assertFalse(c.clean(99))

    def test_reset_stats_keeps_contents_and_seen_blocks(self) -> None:
        c = ref_cache(ways=1, sets=2)
        c.access(0)
        c.reset_stats()
        self.assertEqual((c.hits, c.misses, c.compulsory_misses), (0, 0, 0))
        self.assertTrue(c.contains(0))
        c.access(2 * 16)  # evicts block 0 from set 0
        c.access(0)  # a miss, but block 0 was seen before the reset
        self.assertEqual(c.misses, 2)
        self.assertEqual(c.compulsory_misses, 1)  # only block 2 was new


class TestRefHierarchy(unittest.TestCase):
    def make(self) -> RefHierarchy:
        l1 = RefCache("L1", 256, 64, 1)  # 4 sets, direct-mapped
        l2 = RefCache("L2", 1024, 64, 2)  # 8 sets, 2-way
        return RefHierarchy([RefLevel(l1, 4), RefLevel(l2, 12)], memory_access_time=100)

    def test_miss_path_timing(self) -> None:
        h = self.make()
        self.assertEqual(h.access(0x0), 4 + 12 + 100)
        self.assertEqual(h.access(0x0), 4)
        for blk in (4, 12, 20, 28):  # alias block 0 out of L1 but not L2
            h.access(blk * 64)
        self.assertEqual(h.access(0x0), 4 + 12)

    def test_lower_levels_only_see_misses(self) -> None:
        h = self.make()
        for _ in range(10):
            h.access(0x40)
        self.assertEqual(h.levels[0].cache.misses, 1)
        self.assertEqual(h.levels[1].cache.accesses, 1)

    def test_store_dirties_only_the_first_level(self) -> None:
        h = self.make()
        h.access(0x40, is_write=True)
        l1, l2 = h.levels[0].cache, h.levels[1].cache
        self.assertEqual(l1.write_misses, 1)
        self.assertEqual(l2.write_misses + l2.write_hits, 0)
        self.assertEqual(l2.read_misses, 1)
        self.assertTrue(l1.is_dirty(1))
        self.assertFalse(l2.is_dirty(1))

    def test_dirty_eviction_marks_the_level_below(self) -> None:
        h = self.make()
        l1, l2 = h.levels[0].cache, h.levels[1].cache
        h.access(0x0, is_write=True)
        h.access(4 * 64)  # block 4 aliases L1 set 0 and evicts dirty block 0
        self.assertFalse(l1.contains(0))
        self.assertTrue(l2.is_dirty(0))
        self.assertEqual((l1.writebacks, l2.writebacks_received), (1, 1))
        self.assertEqual((l2.writeback_allocations, h.dram_writes), (0, 0))

    def test_writeback_allocates_when_the_copy_below_is_gone(self) -> None:
        h = self.make()
        l1, l2 = h.levels[0].cache, h.levels[1].cache
        h.access(0x0, is_write=True)
        l2.invalidate(0)
        h.access(4 * 64)
        self.assertTrue(l2.is_dirty(0))
        self.assertEqual((l2.writeback_allocations, l1.writebacks), (1, 1))

    def test_writeback_chain_through_three_levels(self) -> None:
        caches = [RefCache(f"L{i}", 64, 64, 1) for i in (1, 2, 3)]  # one block each
        h = RefHierarchy([RefLevel(c, 1) for c in caches], memory_access_time=10)
        h.access(0x0, is_write=True)
        h.access(0x40)
        self.assertTrue(caches[1].is_dirty(0))
        self.assertEqual(h.dram_writes, 0)
        h.access(0x80)
        self.assertTrue(caches[2].is_dirty(0))
        h.access(0xC0)
        self.assertEqual(h.dram_writes, 1)

    def test_flush_writes_every_dirty_line_once(self) -> None:
        h = self.make()
        l1, l2 = h.levels[0].cache, h.levels[1].cache
        for blk in range(4):
            h.access(blk * 64, is_write=True)
        h.access(4 * 64, is_write=True)
        self.assertEqual(h.flush(), 5)
        self.assertEqual(h.dram_writes, 5)
        self.assertFalse(any(d for *_, d in l1.lines()))
        self.assertFalse(any(d for *_, d in l2.lines()))
        self.assertEqual(h.flush(), 0)

    def test_amat_formula_matches_the_measurement(self) -> None:
        h = self.make()
        rng = Random(11)
        for _ in range(2000):
            h.access(rng.randrange(0, 1 << 14), is_write=rng.random() < 0.25)
        self.assertAlmostEqual(h.amat(), h.measured_amat(), places=9)
        self.assertEqual(h.memory_accesses, h.dram_reads + h.dram_writes)

    def test_reset_stats_keeps_contents(self) -> None:
        h = self.make()
        h.access(0x0)
        h.reset_stats()
        self.assertEqual((h.accesses, h.total_time, h.dram_reads), (0, 0, 0))
        self.assertEqual(h.access(0x0), 4)

    def test_from_spec_builds_the_configured_levels(self) -> None:
        h = RefHierarchy.from_spec(parse_config(DEFAULT_CONFIG))
        self.assertEqual([lv.cache.name for lv in h.levels], ["L1", "L2", "L3"])
        self.assertEqual([lv.hit_time for lv in h.levels], [4, 12, 40])
        self.assertEqual(h.memory_access_time, 100)

    def test_rejects_empty_and_mismatched_hierarchies(self) -> None:
        with self.assertRaises(ValueError):
            RefHierarchy([], memory_access_time=1)
        with self.assertRaises(ValueError):
            RefHierarchy(
                [
                    RefLevel(RefCache("L1", 256, 64, 1), 1),
                    RefLevel(RefCache("L2", 256, 32, 1), 2),
                ],
                memory_access_time=1,
            )

    def test_negative_addresses_are_rejected(self) -> None:
        h = self.make()
        with self.assertRaises(ValueError):
            h.access(-1)
        with self.assertRaises(ValueError):
            h.levels[0].cache.access(-1)


if __name__ == "__main__":
    unittest.main(verbosity=2)
