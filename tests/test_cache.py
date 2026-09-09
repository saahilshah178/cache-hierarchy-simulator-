"""Known-answer tests for a single cache level."""

from __future__ import annotations

import random
import unittest

from cachesim.cache import Cache
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
        with self.assertRaises(ValueError):
            Cache("bad", size=1000, block_size=64, associativity=4)

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
