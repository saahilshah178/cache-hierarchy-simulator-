"""Known-answer tests for the multi-level hierarchy."""

from __future__ import annotations

import glob
import os
import random
import unittest

from cachesim.cache import Cache
from cachesim.config import DEFAULT_CONFIG, load_config
from cachesim.hierarchy import Hierarchy, Level

CONFIG_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "configs")


class TestHierarchy(unittest.TestCase):
    def make(self) -> Hierarchy:
        l1 = Cache("L1", 256, 64, 1)  # 4 blocks
        l2 = Cache("L2", 1024, 64, 2)  # 16 blocks
        return Hierarchy([Level(l1, 4), Level(l2, 12)], memory_access_time=100)

    def test_miss_path_timing(self) -> None:
        h = self.make()
        self.assertEqual(h.access(0x0), 4 + 12 + 100)  # miss, miss, DRAM
        self.assertEqual(h.access(0x0), 4)  # now an L1 hit
        # Blocks 4, 12, 20, 28 all alias to L1 set 0 (4 sets, direct-mapped)
        # and evict block 0 from L1, but they land in L2 set 4 (8 sets),
        # leaving block 0 untouched in L2 set 0.
        for blk in (4, 12, 20, 28):
            h.access(blk * 64)
        self.assertEqual(h.access(0x0), 4 + 12)  # L1 evicted it; L2 has it

    def test_l2_only_sees_l1_misses(self) -> None:
        h = self.make()
        for _ in range(10):
            h.access(0x40)
        self.assertEqual(h.levels[0].cache.misses, 1)
        self.assertEqual(h.levels[1].cache.accesses, 1)

    def test_writes_dirty_the_first_level_only(self) -> None:
        """A store that misses everywhere is a write at L1 but a line fill
        (read) at L2: the data is modified in L1 until L1 evicts it."""
        h = self.make()
        h.access(0x40, is_write=True)
        l1, l2 = h.levels[0].cache, h.levels[1].cache
        self.assertEqual(l1.write_misses, 1)
        self.assertEqual(l2.write_misses + l2.write_hits, 0)
        self.assertEqual(l2.read_misses, 1)
        self.assertTrue(l1.is_dirty(1))
        self.assertFalse(l2.is_dirty(1))

    def test_amat_formula_matches_measurement(self) -> None:
        rng = random.Random(7)
        h = Hierarchy.from_config(DEFAULT_CONFIG)
        for _ in range(20000):
            h.access(rng.randrange(0, 1 << 22), is_write=rng.random() < 0.25)
        self.assertAlmostEqual(h.amat(), h.measured_amat(), places=9)


class TestExampleConfigs(unittest.TestCase):
    """Every config shipped in configs/ must parse and simulate."""

    def config_paths(self) -> list[str]:
        paths = sorted(glob.glob(os.path.join(CONFIG_DIR, "*.json")))
        self.assertTrue(paths, f"no configs found in {CONFIG_DIR}")
        return paths

    def test_every_example_config_parses_and_runs(self) -> None:
        rng = random.Random(5)
        stream = [(rng.randrange(0, 1 << 18), rng.random() < 0.3) for _ in range(3000)]
        for path in self.config_paths():
            with self.subTest(config=os.path.basename(path)):
                h = Hierarchy.from_config(load_config(path))
                for addr, is_write in stream:
                    h.access(addr, is_write)
                h.check_inclusion()
                h.flush()
                stats = h.stats()
                self.assertEqual(stats.accesses, len(stream))
                self.assertGreater(stats.measured_amat, 0.0)
                self.assertTrue(stats.to_dict())

    def test_default_json_matches_the_built_in_default(self) -> None:
        from cachesim.config import parse_config

        on_disk = parse_config(load_config(os.path.join(CONFIG_DIR, "default.json")))
        self.assertEqual(on_disk, parse_config(DEFAULT_CONFIG))


class TestWritebackPropagation(unittest.TestCase):
    """Dirty data must travel down the hierarchy, never vanish."""

    def make(self) -> Hierarchy:
        l1 = Cache("L1", 256, 64, 1)  # 4 sets, direct-mapped
        l2 = Cache("L2", 1024, 64, 2)  # 8 sets, 2-way
        return Hierarchy([Level(l1, 4), Level(l2, 12)], memory_access_time=100)

    def test_dirty_l1_eviction_marks_l2_dirty(self) -> None:
        h = self.make()
        l1, l2 = h.levels[0].cache, h.levels[1].cache
        h.access(0x0, is_write=True)  # block 0 dirty in L1, clean in L2
        h.access(4 * 64)  # block 4 aliases L1 set 0: evicts dirty block 0
        self.assertFalse(l1.contains(0))
        self.assertTrue(l2.is_dirty(0))
        self.assertEqual(l1.writebacks, 1)
        self.assertEqual(l2.writebacks_received, 1)
        self.assertEqual(l2.writeback_allocations, 0)
        self.assertEqual(h.dram_writes, 0)

    def test_writeback_allocates_when_l2_lost_the_line(self) -> None:
        h = self.make()
        l1, l2 = h.levels[0].cache, h.levels[1].cache
        h.access(0x0, is_write=True)  # block 0: L1 set 0, L2 set 0
        # Blocks 8 and 16 map to L1 set 0 and L2 set 0. Touch them via a
        # path that does not evict block 0 from L1 first: L1 is direct
        # mapped, so we cannot; instead evict block 0 from L2 directly.
        l2.invalidate(0)
        self.assertFalse(l2.contains(0))
        h.access(4 * 64)  # evicts dirty block 0 from L1 -> L2 must allocate it
        self.assertTrue(l2.is_dirty(0))
        self.assertEqual(l2.writeback_allocations, 1)
        self.assertEqual(l1.writebacks, 1)

    def test_dirty_last_level_eviction_writes_dram(self) -> None:
        l1 = Cache("L1", 64, 64, 1)  # one block
        h = Hierarchy([Level(l1, 1)], memory_access_time=10)
        h.access(0x0, is_write=True)
        self.assertEqual(h.dram_writes, 0)
        h.access(0x40)  # evicts dirty block 0 straight to DRAM
        self.assertEqual((h.dram_reads, h.dram_writes, h.memory_accesses), (2, 1, 3))
        h.access(0x0)  # block 0 comes back clean
        h.access(0x40)
        self.assertEqual(h.dram_writes, 1)

    def test_writeback_chain_through_three_levels(self) -> None:
        """A dirty line evicted from L1 lands in L2; evicted from L2 it lands
        in L3; evicted from L3 it reaches DRAM."""
        caches = [Cache(f"L{i}", 64, 64, 1) for i in (1, 2, 3)]  # one block each
        h = Hierarchy([Level(c, 1) for c in caches], memory_access_time=10)
        h.access(0x0, is_write=True)
        h.access(0x40)  # block 0 dirty -> L2, block 1 fills all levels
        self.assertTrue(caches[1].is_dirty(0))
        self.assertEqual(h.dram_writes, 0)
        h.access(0x80)  # block 1 evicted from L1 (clean); L2 evicts dirty 0 -> L3
        self.assertTrue(caches[2].is_dirty(0))
        h.access(0xC0)  # L3 evicts dirty block 0 -> DRAM
        self.assertEqual(h.dram_writes, 1)

    def test_flush_writes_every_dirty_line_once(self) -> None:
        h = self.make()
        l1, l2 = h.levels[0].cache, h.levels[1].cache
        for blk in range(4):
            h.access(blk * 64, is_write=True)  # four dirty lines in L1
        h.access(4 * 64, is_write=True)  # block 0 dirty -> L2; block 4 dirty in L1
        self.assertEqual(h.flush(), 5)
        self.assertEqual(h.dram_writes, 5)
        self.assertFalse(any(dirty for *_, dirty in l1.lines()))
        self.assertFalse(any(dirty for *_, dirty in l2.lines()))
        self.assertEqual(h.flush(), 0)  # nothing left to write

    def test_dirty_data_conservation(self) -> None:
        """After a flush every block the CPU ever wrote has reached DRAM."""
        written_to_dram: list[int] = []

        class Recording(Hierarchy):
            def _write_to_memory(self, block: int, charged: bool = False) -> int:
                written_to_dram.append(block)
                return super()._write_to_memory(block, charged)

        rng = random.Random(3)
        l1 = Cache("L1", 512, 64, 2)
        l2 = Cache("L2", 2048, 64, 4)
        h = Recording([Level(l1, 1), Level(l2, 5)], memory_access_time=50)
        cpu_writes: set[int] = set()
        for _ in range(5000):
            addr = rng.randrange(0, 16 * 1024)
            is_write = rng.random() < 0.3
            if is_write:
                cpu_writes.add(addr // 64)
            h.access(addr, is_write)
        h.flush()
        self.assertTrue(cpu_writes <= set(written_to_dram))
        self.assertEqual(len(written_to_dram), h.dram_writes)
        self.assertEqual(l1.writebacks, l2.writebacks_received)


if __name__ == "__main__":
    unittest.main(verbosity=2)


class TestWarmup(unittest.TestCase):
    def test_reset_keeps_contents_and_seen_blocks(self) -> None:
        l1 = Cache("L1", 256, 64, 1)
        h = Hierarchy([Level(l1, 4)], memory_access_time=100)
        h.access(0x0)
        h.access(0x40)
        h.reset_stats()
        self.assertEqual((h.accesses, h.total_time, l1.misses, l1.compulsory_misses), (0, 0, 0, 0))
        self.assertTrue(l1.contains(0))
        self.assertEqual(h.access(0x0), 4)  # still resident: a hit
        self.assertEqual((l1.hits, l1.misses), (1, 0))
        # Evict block 0 with an alias, then touch it again: a miss, but not
        # compulsory, because it was seen before the reset.
        h.access(4 * 64)
        h.access(0x0)
        self.assertEqual(l1.compulsory_misses, 1)  # only block 4 was new
        self.assertEqual(l1.misses, 2)

    def test_run_trace_warmup(self) -> None:
        import os
        import tempfile

        from cachesim import run_trace
        from cachesim.workloads import sequential, write_trace

        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "seq.trace")
            write_trace(path, sequential(buffer_bytes=4096, passes=3))  # 512 accesses/pass
            cold = run_trace(path)
            warm = run_trace(path, warmup=512)
            self.assertEqual(cold.accesses, 1536)
            self.assertEqual(warm.accesses, 1024)
            # The buffer fits in L1, so after the first pass everything hits.
            self.assertEqual(cold.levels[0].cache.misses, 64)
            self.assertEqual(warm.levels[0].cache.misses, 0)
            self.assertEqual(warm.levels[0].cache.compulsory_misses, 0)
