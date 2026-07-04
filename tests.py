"""tests.py — known-answer tests for the simulator.

Run with either::

    python3 tests.py
    python3 -m unittest tests -v

Every test is a tiny hand-checkable scenario: a few accesses whose hit/miss
outcome you can verify on paper. If you change cache.py or hierarchy.py and
these still pass, the core model is still right.
"""

import io
import tempfile
import unittest

from cache import Cache, POLICIES, ReplacementPolicy
from hierarchy import Hierarchy, Level
from simulator import parse_trace, run_trace, DEFAULT_CONFIG


def tiny_cache(ways, sets=2, block=16, policy="lru"):
    """A deliberately tiny cache: sets*ways blocks of 16 bytes."""
    return Cache("T", size=sets * ways * block, block_size=block,
                 associativity=ways, policy=policy)


def block_addr(cache, set_idx, tag):
    """Build an address that lands in `set_idx` with `tag`."""
    block = tag * cache.num_sets + set_idx
    return block * cache.block_size


class TestAddressing(unittest.TestCase):
    def test_same_block_hits(self):
        """All bytes of one block share one miss: spatial locality."""
        c = tiny_cache(ways=1)
        self.assertFalse(c.access(0x100))      # first touch: miss
        self.assertTrue(c.access(0x101))       # same 16-byte block: hit
        self.assertTrue(c.access(0x10f))       # last byte of block: hit
        self.assertFalse(c.access(0x110))      # next block: miss

    def test_geometry(self):
        c = Cache("g", size=32 * 1024, block_size=64, associativity=4)
        self.assertEqual(c.num_sets, 128)
        self.assertEqual(c.num_blocks, 512)

    def test_bad_geometry_rejected(self):
        with self.assertRaises(ValueError):
            Cache("bad", size=1000, block_size=64, associativity=4)

    def test_bad_policy_rejected(self):
        with self.assertRaises(ValueError):
            tiny_cache(ways=1, policy="clairvoyant")


class TestDirectMappedConflict(unittest.TestCase):
    def test_two_blocks_one_slot_thrash(self):
        """Two blocks mapped to the same set evict each other forever."""
        c = tiny_cache(ways=1)
        a = block_addr(c, set_idx=0, tag=1)
        b = block_addr(c, set_idx=0, tag=2)
        for _ in range(4):
            self.assertFalse(c.access(a))
            self.assertFalse(c.access(b))
        self.assertEqual(c.hits, 0)
        self.assertEqual(c.misses, 8)

    def test_two_ways_end_the_thrash(self):
        """The same pattern in a 2-way cache: both blocks coexist."""
        c = tiny_cache(ways=2)
        a = block_addr(c, set_idx=0, tag=1)
        b = block_addr(c, set_idx=0, tag=2)
        for _ in range(4):
            c.access(a)
            c.access(b)
        self.assertEqual(c.misses, 2)          # just the two first touches
        self.assertEqual(c.hits, 6)


class TestReplacementPolicies(unittest.TestCase):
    def setUp(self):
        # Three blocks (a, b, c) all in set 0 of a 2-way cache: exactly one
        # of them must be evicted when the third arrives.
        self.c_lru = tiny_cache(ways=2, policy="lru")
        self.c_fifo = tiny_cache(ways=2, policy="fifo")
        self.a = block_addr(self.c_lru, 0, 1)
        self.b = block_addr(self.c_lru, 0, 2)
        self.x = block_addr(self.c_lru, 0, 3)

    def test_lru_keeps_the_recently_hit_block(self):
        c = self.c_lru
        c.access(self.a)                       # fill a
        c.access(self.b)                       # fill b
        c.access(self.a)                       # touch a: b is now LRU
        c.access(self.x)                       # evicts b
        self.assertTrue(c.access(self.a))      # a survived
        self.assertFalse(c.access(self.b))     # b was the victim

    def test_fifo_evicts_the_oldest_fill_even_if_hot(self):
        c = self.c_fifo
        c.access(self.a)                       # fill a (oldest)
        c.access(self.b)                       # fill b
        c.access(self.a)                       # hit does NOT refresh FIFO age
        c.access(self.x)                       # evicts a anyway
        # Check b FIRST: probing a would refill it and evict b (FIFO's next
        # victim), polluting the second check.
        self.assertTrue(c.access(self.b))      # b survived (x still cached)
        self.assertFalse(c.access(self.a))     # a was the victim

    def test_random_is_reproducible(self):
        def run():
            c = tiny_cache(ways=2, policy="random")
            for tag in (1, 2, 3, 1, 2, 3, 1):
                c.access(block_addr(c, 0, tag))
            return (c.hits, c.misses)
        self.assertEqual(run(), run())         # same seed, same outcome

    def test_policies_are_extensible(self):
        """The README promises a subclass + registry entry is all it takes."""
        class EvictWayZero(ReplacementPolicy):
            def victim(self, set_idx):
                return 0
        POLICIES["way0"] = EvictWayZero
        try:
            c = tiny_cache(ways=2, policy="way0")
            c.access(block_addr(c, 0, 1))      # fills way 0
            c.access(block_addr(c, 0, 2))      # fills way 1
            c.access(block_addr(c, 0, 3))      # evicts way 0 (tag 1)
            self.assertTrue(c.access(block_addr(c, 0, 2)))
            self.assertFalse(c.access(block_addr(c, 0, 1)))
        finally:
            del POLICIES["way0"]


class TestThreeCs(unittest.TestCase):
    def test_first_touches_are_compulsory(self):
        c = tiny_cache(ways=1, sets=4)
        for i in range(4):
            c.access(i * c.block_size)
        self.assertEqual(c.compulsory_misses, 4)
        self.assertEqual(c.capacity_misses, 0)
        self.assertEqual(c.conflict_misses, 0)

    def test_conflict_miss_detected(self):
        """Fully-associative twin would hit -> classified as conflict."""
        c = tiny_cache(ways=1, sets=2)         # 2 blocks total capacity
        a = block_addr(c, 0, 1)
        b = block_addr(c, 0, 2)                # same set as a, cache half empty
        c.access(a); c.access(b); c.access(a)  # third access: a was evicted,
        self.assertEqual(c.conflict_misses, 1)  # but FA twin still held it

    def test_capacity_miss_detected(self):
        """Working set bigger than the whole cache -> capacity."""
        c = tiny_cache(ways=2, sets=1)         # fully associative, 2 blocks
        addrs = [block_addr(c, 0, t) for t in (1, 2, 3)]
        for a in addrs:                        # touch 3 blocks: 3 compulsory
            c.access(a)
        c.access(addrs[0])                     # was evicted; FA twin == cache
        self.assertEqual(c.capacity_misses, 1)
        self.assertEqual(c.conflict_misses, 0)

    def test_three_cs_partition_the_misses(self):
        """compulsory + capacity + conflict == misses, on a messy pattern."""
        import random
        rng = random.Random(42)
        c = Cache("p", size=1024, block_size=64, associativity=2)
        for _ in range(5000):
            c.access(rng.randrange(0, 64 * 1024), is_write=rng.random() < .3)
        self.assertEqual(
            c.compulsory_misses + c.capacity_misses + c.conflict_misses,
            c.misses)
        self.assertEqual(c.hits + c.misses, c.accesses)


class TestWriteback(unittest.TestCase):
    def test_dirty_eviction_counts_as_writeback(self):
        c = tiny_cache(ways=1)
        a = block_addr(c, 0, 1)
        b = block_addr(c, 0, 2)
        c.access(a, is_write=True)             # a is dirty
        c.access(b)                            # evicts dirty a
        self.assertEqual(c.writebacks, 1)
        c.access(a)                            # evicts CLEAN b
        self.assertEqual(c.writebacks, 1)      # unchanged


class TestHierarchy(unittest.TestCase):
    def make(self):
        l1 = Cache("L1", 256, 64, 1)           # 4 blocks
        l2 = Cache("L2", 1024, 64, 2)          # 16 blocks
        return Hierarchy([Level(l1, 4), Level(l2, 12)], memory_access_time=100)

    def test_miss_path_timing(self):
        h = self.make()
        self.assertEqual(h.access(0x0), 4 + 12 + 100)   # miss, miss, DRAM
        self.assertEqual(h.access(0x0), 4)              # now an L1 hit
        # Blocks 4, 12, 20, 28: all alias to L1 set 0 (4 sets, direct-mapped)
        # and evict block 0 from L1 — but they land in L2 set 4 (8 sets),
        # leaving block 0 untouched in L2 set 0.
        for blk in (4, 12, 20, 28):
            h.access(blk * 64)
        self.assertEqual(h.access(0x0), 4 + 12)         # L1 evicted it; L2 has it

    def test_l2_only_sees_l1_misses(self):
        h = self.make()
        for _ in range(10):
            h.access(0x40)
        self.assertEqual(h.levels[0].cache.misses, 1)
        self.assertEqual(h.levels[1].cache.accesses, 1)

    def test_writes_dirty_the_first_level_only(self):
        """A store that misses everywhere is a WRITE at L1 but a line-fill
        (read) at L2 — write-back semantics, so writeback stats aren't
        double-counted down the hierarchy."""
        h = self.make()
        h.access(0x40, is_write=True)
        l1, l2 = h.levels[0].cache, h.levels[1].cache
        self.assertEqual(l1.write_misses, 1)
        self.assertEqual(l2.write_misses + l2.write_hits, 0)
        self.assertEqual(l2.read_misses, 1)

    def test_amat_formula_matches_measurement(self):
        import random
        rng = random.Random(7)
        h = Hierarchy.from_config(DEFAULT_CONFIG)
        for _ in range(20000):
            h.access(rng.randrange(0, 1 << 22), is_write=rng.random() < .25)
        self.assertAlmostEqual(h.amat(), h.measured_amat(), places=9)


class TestTraceParsing(unittest.TestCase):
    def parse(self, text):
        with tempfile.NamedTemporaryFile("w", suffix=".trace",
                                         delete=False) as f:
            f.write(text)
            path = f.name
        return list(parse_trace(path))

    def test_valid_lines(self):
        got = self.parse("# comment\n\n0x10 R\n20 w\n0XFF W\n")
        self.assertEqual(got, [(0x10, False), (0x20, True), (0xFF, True)])

    def test_bad_address_rejected(self):
        with self.assertRaises(ValueError):
            self.parse("0xZZ R\n")

    def test_bad_op_rejected(self):
        with self.assertRaises(ValueError):
            self.parse("0x10 READ\n")

    def test_missing_field_rejected(self):
        with self.assertRaises(ValueError):
            self.parse("0x10\n")


if __name__ == "__main__":
    unittest.main(verbosity=2)
