"""Known-answer tests for the multi-level hierarchy."""

from __future__ import annotations

import random
import unittest

from cachesim.cache import Cache
from cachesim.config import DEFAULT_CONFIG
from cachesim.hierarchy import Hierarchy, Level


class TestHierarchy(unittest.TestCase):
    def make(self) -> Hierarchy:
        l1 = Cache("L1", 256, 64, 1)           # 4 blocks
        l2 = Cache("L2", 1024, 64, 2)          # 16 blocks
        return Hierarchy([Level(l1, 4), Level(l2, 12)], memory_access_time=100)

    def test_miss_path_timing(self) -> None:
        h = self.make()
        self.assertEqual(h.access(0x0), 4 + 12 + 100)   # miss, miss, DRAM
        self.assertEqual(h.access(0x0), 4)              # now an L1 hit
        # Blocks 4, 12, 20, 28 all alias to L1 set 0 (4 sets, direct-mapped)
        # and evict block 0 from L1, but they land in L2 set 4 (8 sets),
        # leaving block 0 untouched in L2 set 0.
        for blk in (4, 12, 20, 28):
            h.access(blk * 64)
        self.assertEqual(h.access(0x0), 4 + 12)         # L1 evicted it; L2 has it

    def test_l2_only_sees_l1_misses(self) -> None:
        h = self.make()
        for _ in range(10):
            h.access(0x40)
        self.assertEqual(h.levels[0].cache.misses, 1)
        self.assertEqual(h.levels[1].cache.accesses, 1)

    def test_writes_dirty_the_first_level_only(self) -> None:
        """A store that misses everywhere is a write at L1 but a line fill
        (read) at L2: the data is modified in L1 only."""
        h = self.make()
        h.access(0x40, is_write=True)
        l1, l2 = h.levels[0].cache, h.levels[1].cache
        self.assertEqual(l1.write_misses, 1)
        self.assertEqual(l2.write_misses + l2.write_hits, 0)
        self.assertEqual(l2.read_misses, 1)

    def test_amat_formula_matches_measurement(self) -> None:
        rng = random.Random(7)
        h = Hierarchy.from_config(DEFAULT_CONFIG)
        for _ in range(20000):
            h.access(rng.randrange(0, 1 << 22), is_write=rng.random() < 0.25)
        self.assertAlmostEqual(h.amat(), h.measured_amat(), places=9)


if __name__ == "__main__":
    unittest.main(verbosity=2)
