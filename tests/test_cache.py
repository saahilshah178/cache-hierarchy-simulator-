"""Known-answer tests for a single cache level."""

from __future__ import annotations

import random
import unittest
from typing import Any
from unittest import mock

from cachesim.cache import Cache, Evicted
from cachesim.policies import POLICIES, ReplacementPolicy
from tests.helpers import block_addr, tiny_cache


class TestAddressing(unittest.TestCase):
    def test_same_block_hits(self) -> None:
        """All bytes of one block share one miss: spatial locality."""
        c = tiny_cache(ways=1)
        self.assertFalse(c.access(0x100))  # first touch: miss
        self.assertTrue(c.access(0x101))  # same 16-byte block: hit
        self.assertTrue(c.access(0x10F))  # last byte of block: hit
        self.assertFalse(c.access(0x110))  # next block: miss

    def test_geometry(self) -> None:
        c = Cache("g", size=32 * 1024, block_size=64, associativity=4)
        self.assertEqual(c.num_sets, 128)
        self.assertEqual(c.num_blocks, 512)

    def test_bad_geometry_rejected(self) -> None:
        bad: list[tuple[Any, Any, Any]] = [
            (1000, 64, 4),  # size is not a multiple of block_size * ways
            (0, 64, 4),
            (-1024, 64, 4),
            (1024, 0, 4),
            (1024, -64, 4),
            (1024, 48, 1),  # block size is not a power of two
            (1024, 64, 0),
            (1024, 64, -4),
            (1024.0, 64, 4),  # not an int
            (True, 64, 4),
        ]
        for size, block_size, ways in bad:
            with (
                self.subTest(size=size, block_size=block_size, ways=ways),
                self.assertRaises(ValueError),
            ):
                Cache("bad", size=size, block_size=block_size, associativity=ways)

    def test_non_power_of_two_set_count_allowed(self) -> None:
        """Three sets is legal: set index is block modulo num_sets."""
        c = Cache("odd", size=3 * 64 * 2, block_size=64, associativity=2)
        self.assertEqual(c.num_sets, 3)
        for blk in range(12):
            c.access(blk * 64)
        self.assertEqual(c.misses, 12)

    def test_rng_seed_must_be_an_integer(self) -> None:
        with self.assertRaises(ValueError):
            Cache("r", 1024, 64, 2, policy="random", rng_seed=None)  # type: ignore[arg-type]

    def test_rng_seed_changes_random_victims(self) -> None:
        def run(seed: int) -> list[bool]:
            c = Cache("r", 64 * 4, 64, 4, policy="random", rng_seed=seed)
            return [c.access(blk * 64) for blk in [0, 1, 2, 3, 4, 0, 1, 2, 3, 4] * 4]

        self.assertEqual(run(1), run(1))
        self.assertNotEqual(run(1), run(2))

    def test_bad_policy_rejected(self) -> None:
        with self.assertRaises(ValueError):
            tiny_cache(ways=1, policy="clairvoyant")


class TestDirectMappedConflict(unittest.TestCase):
    def test_two_blocks_one_slot_thrash(self) -> None:
        """Two blocks mapped to the same set evict each other forever."""
        c = tiny_cache(ways=1)
        a = block_addr(c, set_idx=0, tag=1)
        b = block_addr(c, set_idx=0, tag=2)
        for _ in range(4):
            self.assertFalse(c.access(a))
            self.assertFalse(c.access(b))
        self.assertEqual(c.hits, 0)
        self.assertEqual(c.misses, 8)

    def test_two_ways_end_the_thrash(self) -> None:
        """The same pattern in a 2-way cache: both blocks coexist."""
        c = tiny_cache(ways=2)
        a = block_addr(c, set_idx=0, tag=1)
        b = block_addr(c, set_idx=0, tag=2)
        for _ in range(4):
            c.access(a)
            c.access(b)
        self.assertEqual(c.misses, 2)  # just the two first touches
        self.assertEqual(c.hits, 6)


class TestThreeCs(unittest.TestCase):
    def test_first_touches_are_compulsory(self) -> None:
        c = tiny_cache(ways=1, sets=4)
        for i in range(4):
            c.access(i * c.block_size)
        self.assertEqual(c.compulsory_misses, 4)
        self.assertEqual(c.capacity_misses, 0)
        self.assertEqual(c.conflict_misses, 0)

    def test_conflict_miss_detected(self) -> None:
        """Fully-associative twin would hit -> classified as conflict."""
        c = tiny_cache(ways=1, sets=2)  # 2 blocks total capacity
        a = block_addr(c, 0, 1)
        b = block_addr(c, 0, 2)  # same set as a, cache half empty
        c.access(a)
        c.access(b)
        c.access(a)  # a was evicted by b ...
        self.assertEqual(c.conflict_misses, 1)  # ... but the FA twin held it

    def test_capacity_miss_detected(self) -> None:
        """Working set bigger than the whole cache -> capacity."""
        c = tiny_cache(ways=2, sets=1)  # fully associative, 2 blocks
        addrs = [block_addr(c, 0, t) for t in (1, 2, 3)]
        for a in addrs:  # touch 3 blocks: 3 compulsory
            c.access(a)
        c.access(addrs[0])  # was evicted; FA twin == cache
        self.assertEqual(c.capacity_misses, 1)
        self.assertEqual(c.conflict_misses, 0)

    def test_three_cs_partition_the_misses(self) -> None:
        """compulsory + capacity + conflict == misses, on a random pattern."""
        rng = random.Random(42)
        c = Cache("p", size=1024, block_size=64, associativity=2)
        for _ in range(5000):
            c.access(rng.randrange(0, 64 * 1024), is_write=rng.random() < 0.3)
        self.assertEqual(c.compulsory_misses + c.capacity_misses + c.conflict_misses, c.misses)
        self.assertEqual(c.hits + c.misses, c.accesses)


class TestWriteback(unittest.TestCase):
    def test_dirty_eviction_counts_as_writeback(self) -> None:
        c = tiny_cache(ways=1)
        a = block_addr(c, 0, 1)
        b = block_addr(c, 0, 2)
        c.access(a, is_write=True)  # a is dirty
        c.access(b)  # evicts dirty a
        self.assertEqual(c.writebacks, 1)
        c.access(a)  # evicts clean b
        self.assertEqual(c.writebacks, 1)  # unchanged


if __name__ == "__main__":
    unittest.main(verbosity=2)


class TestPrimitives(unittest.TestCase):
    """probe / allocate / invalidate, the building blocks of a hierarchy."""

    def test_probe_does_not_fill(self) -> None:
        c = tiny_cache(ways=1)
        self.assertFalse(c.probe(5))
        self.assertFalse(c.contains(5))
        self.assertEqual((c.misses, c.fills), (1, 0))

    def test_allocate_reports_the_victim(self) -> None:
        c = tiny_cache(ways=1)
        self.assertIsNone(c.allocate(0, dirty=True))  # empty way: nothing evicted
        self.assertEqual(c.allocate(2), Evicted(block=0, dirty=True))  # same set (2 sets)
        self.assertEqual((c.evictions, c.writebacks, c.fills), (1, 1, 2))
        self.assertTrue(c.contains(2))
        self.assertFalse(c.contains(0))

    def test_allocate_of_a_resident_block_is_a_noop(self) -> None:
        c = tiny_cache(ways=2)
        c.allocate(0)
        self.assertIsNone(c.allocate(0, dirty=True))
        self.assertEqual(c.fills, 1)
        self.assertTrue(c.is_dirty(0))  # the dirty bit is OR-ed in

    def test_invalidate_frees_the_way_without_calling_victim(self) -> None:
        class NoVictim(ReplacementPolicy):
            def victim(self, set_idx: int) -> int:
                raise AssertionError("victim() called although a way was empty")

        with mock.patch.dict(POLICIES, {"novictim": NoVictim}):
            c = tiny_cache(ways=2, policy="novictim")
            c.allocate(0)
            c.allocate(2, dirty=True)
            self.assertEqual(c.invalidate(2), Evicted(block=2, dirty=True))
            self.assertIsNone(c.invalidate(2))  # already gone
            self.assertFalse(c.contains(2))
            self.assertEqual((c.invalidations, c.evictions, c.writebacks), (1, 0, 1))
            self.assertIsNone(c.allocate(4))  # refills the emptied way
            self.assertEqual(sorted(blk for _, _, blk, _ in c.lines()), [0, 4])

    def test_invalidate_keeps_lru_order_of_the_remaining_ways(self) -> None:
        c = tiny_cache(ways=4, sets=1)
        for blk in (0, 1, 2, 3):
            c.access(blk * c.block_size)
        c.invalidate(1)  # way 1 is empty; LRU order of the rest is 0, 2, 3
        c.access(4 * c.block_size)  # takes the empty way, no eviction
        self.assertEqual(c.evictions, 0)
        c.access(5 * c.block_size)  # set full: evicts block 0, the true LRU
        self.assertFalse(c.contains(0))
        self.assertTrue(c.contains(2))

    def test_mark_dirty_and_lines(self) -> None:
        c = tiny_cache(ways=2)
        c.allocate(1)
        self.assertTrue(c.mark_dirty(1))
        self.assertFalse(c.mark_dirty(9))
        self.assertEqual(list(c.lines()), [(1, 0, 1, True)])

    def test_address_helpers(self) -> None:
        c = Cache("g", size=32 * 1024, block_size=64, associativity=4)
        blk = c.block_of(0x12345)
        self.assertEqual(blk, 0x12345 // 64)
        self.assertEqual(c.set_of(blk), blk % 128)
        self.assertEqual(c.tag_of(blk), blk // 128)

    def test_negative_address_rejected(self) -> None:
        with self.assertRaises(ValueError):
            tiny_cache(ways=1).access(-1)
