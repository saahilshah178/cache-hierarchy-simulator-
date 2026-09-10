"""Known-answer tests for the multi-level hierarchy."""

from __future__ import annotations

import glob
import os
import random
import unittest

from cachesim.cache import Cache, VictimBuffer
from cachesim.config import DEFAULT_CONFIG, load_config
from cachesim.hierarchy import Hierarchy, Level
from cachesim.workloads import conflict_streams

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


class TestDefaultFastPath(unittest.TestCase):
    """``_access_default`` must be ``access`` with the branches removed.

    The fast path is a specialisation, not a second model, so the check is
    to run the same stream twice through identical hierarchies -- once with
    the fast path enabled and once with the flag forced off, which routes
    every access through the general code -- and compare the whole
    statistics snapshot. That covers every counter the simulator keeps,
    down to the per-level 3-C classification and the byte totals, plus the
    cycle counts and both AMATs, and it compares the cycles each individual
    access reported as well.
    """

    def stream(self, seed: int, length: int = 4000, span: int = 1 << 16) -> list[tuple[int, bool]]:
        rng = random.Random(seed)
        return [(rng.randrange(span), rng.random() < 0.35) for _ in range(length)]

    def both_paths(self, config: dict[str, object], accesses: list[tuple[int, bool]]) -> None:
        fast = Hierarchy.from_config(config)
        self.assertTrue(fast._all_defaults, "this config should qualify for the fast path")
        general = Hierarchy.from_config(config)
        general._all_defaults = False  # force every access through access()
        fast_cycles = [fast.access(addr, w) for addr, w in accesses]
        general_cycles = [general.access(addr, w) for addr, w in accesses]
        self.assertEqual(fast_cycles, general_cycles)
        self.assertEqual(fast.stats().to_dict(), general.stats().to_dict())
        # ... and still equal once the dirty data is accounted for.
        self.assertEqual(fast.flush(), general.flush())
        self.assertEqual(fast.stats().to_dict(), general.stats().to_dict())

    def test_default_config(self) -> None:
        self.both_paths(DEFAULT_CONFIG, self.stream(seed=17))

    def test_a_hierarchy_small_enough_to_thrash(self) -> None:
        # Tiny levels so that the stream evicts, writes back and refills
        # constantly: the paths have to agree about eviction handling, not
        # merely about hits.
        config = {
            "memory_access_time": 100,
            "levels": [
                {"name": "L1", "size": 512, "block_size": 64, "associativity": 2, "hit_time": 4},
                {"name": "L2", "size": 2048, "block_size": 64, "associativity": 4, "hit_time": 12},
            ],
        }
        self.both_paths(config, self.stream(seed=18, span=8192))

    def test_a_single_level(self) -> None:
        config = {
            "memory_access_time": 70,
            "levels": [
                {"name": "L1", "size": 1024, "block_size": 32, "associativity": 4, "hit_time": 2}
            ],
        }
        self.both_paths(config, self.stream(seed=19, span=4096))

    def test_writes_only(self) -> None:
        accesses = [(addr, True) for addr, _ in self.stream(seed=20, span=8192)]
        self.both_paths(DEFAULT_CONFIG, accesses)

    def test_reads_only(self) -> None:
        accesses = [(addr, False) for addr, _ in self.stream(seed=21, span=8192)]
        self.both_paths(DEFAULT_CONFIG, accesses)

    def test_every_extra_leaves_the_fast_path(self) -> None:
        """Anything the fast path does not model must switch it off."""
        base = {
            "name": "L2",
            "size": 2048,
            "block_size": 64,
            "associativity": 4,
            "hit_time": 12,
        }
        extras: list[dict[str, object]] = [
            {"inclusion": "inclusive"},
            {"inclusion": "exclusive"},
            {"write_policy": "write-through"},
            {"write_allocate": False},
            {"bus_width": 16},
            {"prefetcher": "next-line"},
            {"victim_cache": {"entries": 4}},
        ]
        for extra in extras:
            with self.subTest(extra=extra):
                config = {
                    "memory_access_time": 100,
                    "levels": [
                        {
                            "name": "L1",
                            "size": 512,
                            "block_size": 64,
                            "associativity": 2,
                            "hit_time": 4,
                        },
                        {**base, **extra},
                    ],
                }
                self.assertFalse(Hierarchy.from_config(config)._all_defaults)
        # ... and so must a memory whose latency depends on the address.
        row_buffer = {
            "memory": {"type": "row-buffer", "row_size": 8192, "banks": 8},
            "levels": [
                {"name": "L1", "size": 512, "block_size": 64, "associativity": 2, "hit_time": 4}
            ],
        }
        self.assertFalse(Hierarchy.from_config(row_buffer)._all_defaults)

    def test_the_shipped_default_config_takes_the_fast_path(self) -> None:
        path = os.path.join(CONFIG_DIR, "default.json")
        self.assertTrue(Hierarchy.from_config(load_config(path))._all_defaults)


class TestVictimCache(unittest.TestCase):
    """A fully-associative buffer of the lines a level's array has replaced
    (Jouppi, ISCA 1990). A hit there is a hit at the level: the buffer is
    part of it, probed with the tag array and charged the same hit time."""

    def two_block_cache(self, entries: int | None) -> Hierarchy:
        """Direct-mapped, 2 sets: blocks 0 and 2 collide in set 0."""
        cache = Cache("L1", 128, 64, 1, victim_entries=entries)
        return Hierarchy([Level(cache, 4)], memory_access_time=100)

    def test_a_swapping_pair_stops_missing(self) -> None:
        """Blocks 0 and 2 map to the same set and evict each other on every
        reference. With a two-entry buffer, each eviction is caught and the
        next reference swaps the block straight back: two compulsory misses
        and then nothing but hits."""
        plain = self.two_block_cache(None)
        victim = self.two_block_cache(2)
        for _ in range(8):
            for block in (0, 2):
                plain.access(block * 64)
                victim.access(block * 64)
        self.assertEqual(plain.levels[0].cache.misses, 16)
        c = victim.levels[0].cache
        self.assertEqual(c.misses, 2)
        self.assertEqual(c.victim_hits, 14)
        self.assertEqual(c.hits, 14)  # a victim hit is a hit at this level
        self.assertEqual(victim.access(0), 4)  # ... and costs the level's hit time
        self.assertAlmostEqual(victim.amat(), victim.measured_amat(), places=9)

    def test_conflict_streams_reach_the_compulsory_floor(self) -> None:
        """The conflict workload reads four streams whose addresses are 1 MB
        apart, so all four land in one set: a direct-mapped cache misses on
        every single access. Three buffer entries hold the three blocks the
        array cannot, and the miss count drops to the 8,192 compulsory
        misses -- everything a fully-associative cache of any size would
        also have taken. Two entries are one short and change nothing."""
        stream = [(addr, op == "W") for addr, op in conflict_streams()]
        misses = {}
        for entries in (None, 2, 3, 4):
            h = Hierarchy(
                [Level(Cache("L1", 32768, 64, 1, victim_entries=entries), 4)],
                memory_access_time=100,
            )
            for addr, is_write in stream:
                h.access(addr, is_write)
            misses[entries] = h.levels[0].cache.misses
            self.assertEqual(h.levels[0].cache.compulsory_misses, 8192)
        self.assertEqual(misses[None], 65536)
        self.assertEqual(misses[2], 65536)
        self.assertEqual(misses[3], 8192)
        self.assertEqual(misses[4], 8192)

    def test_a_buffered_line_is_still_resident(self) -> None:
        h = self.two_block_cache(2)
        c = h.levels[0].cache
        h.access(0)
        h.access(2 * 64)  # evicts block 0 into the buffer
        self.assertTrue(c.contains(0))
        self.assertEqual(sorted(b for _, _, b, _ in c.lines()), [0, 2])

    def test_a_dirty_line_is_written_back_only_when_it_leaves_the_level(self) -> None:
        h = self.two_block_cache(1)
        c = h.levels[0].cache
        h.access(0, is_write=True)  # block 0 dirty in the array
        h.access(2 * 64)  # array evicts it into the buffer: not a write-back
        self.assertEqual((c.writebacks, h.dram_writes), (0, 0))
        self.assertTrue(c.is_dirty(0))
        h.access(4 * 64)  # block 4 also maps to set 0; block 0 is pushed out
        self.assertEqual((c.writebacks, h.dram_writes), (1, 1))
        self.assertFalse(c.contains(0))

    def test_flush_reaches_lines_in_the_buffer(self) -> None:
        h = self.two_block_cache(2)
        h.access(0, is_write=True)
        h.access(2 * 64, is_write=True)  # block 0, dirty, is now in the buffer
        self.assertEqual(h.flush(), 2)
        self.assertEqual(h.dram_writes, 2)
        self.assertEqual(h.flush(), 0)

    def test_config_accepts_a_count_or_an_object(self) -> None:
        from cachesim.config import ConfigError, parse_config

        def config(victim: object) -> dict[str, object]:
            return {
                "memory_access_time": 10,
                "levels": [
                    {
                        "name": "L1",
                        "size": 1024,
                        "block_size": 64,
                        "associativity": 1,
                        "hit_time": 1,
                        "victim_cache": victim,
                    }
                ],
            }

        self.assertEqual(parse_config(config(4)).levels[0].victim_cache, 4)
        self.assertEqual(parse_config(config({"entries": 4})).levels[0].victim_cache, 4)
        spec = parse_config(config({"entries": 4}))
        self.assertEqual(parse_config(spec.to_dict()), spec)  # round-trips as a count
        for bad, fragment in (
            (0, "must be >= 1"),
            ({"size": 4}, "unknown key(s) 'size'"),
            ({}, "missing required key 'entries'"),
        ):
            with self.subTest(bad=bad), self.assertRaises(ConfigError) as ctx:
                parse_config(config(bad))
            self.assertIn(fragment, str(ctx.exception))

    def test_default_is_no_victim_cache(self) -> None:
        self.assertIsNone(Cache("L1", 1024, 64, 2).victim)
        self.assertIsNone(self.two_block_cache(None).stats().levels[0].victim_cache_entries)

    def test_entries_are_validated(self) -> None:
        for entries in (0, -1, "four"):
            with self.subTest(entries=entries), self.assertRaises(ValueError):
                VictimBuffer(entries)  # type: ignore[arg-type]

    def test_it_composes_with_the_rest_of_the_hierarchy(self) -> None:
        rng = random.Random(37)
        h = Hierarchy(
            [
                Level(Cache("L1", 512, 64, 1, victim_entries=4), 4, prefetcher="next-line"),
                Level(Cache("L2", 4096, 64, 4, victim_entries=2), 12, inclusion="inclusive"),
            ],
            memory_access_time=100,
        )
        for _ in range(5000):
            h.access(rng.randrange(0, 1 << 14), is_write=rng.random() < 0.3)
            h.check_inclusion()
        self.assertGreater(h.levels[0].cache.victim_hits, 0)
        h.flush()


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


class TestPackageSurface(unittest.TestCase):
    """``cachesim`` re-exports the types a caller builds a model out of, so
    the memory models, prefetchers and their specs have to be reachable
    from the package root and not only from their own modules."""

    def test_the_modelling_types_are_importable_from_the_package_root(self) -> None:
        import cachesim

        for name in (
            "INCLUSION_POLICIES",
            "WRITE_POLICIES",
            "PREFETCHERS",
            "PREFETCHER_NAMES",
            "make_prefetcher",
            "Prefetcher",
            "MemoryModel",
            "ConstantMemory",
            "RowBufferMemory",
            "ConstantMemorySpec",
            "RowBufferMemorySpec",
            "MemorySpec",
            "MemoryStats",
            "VictimBuffer",
        ):
            with self.subTest(name=name):
                self.assertIn(name, cachesim.__all__)
                self.assertTrue(hasattr(cachesim, name))

    def test_a_reexport_is_the_object_its_module_defines(self) -> None:
        import cachesim
        import cachesim.dram
        import cachesim.prefetch

        self.assertIs(cachesim.RowBufferMemory, cachesim.dram.RowBufferMemory)
        self.assertIs(cachesim.PREFETCHERS, cachesim.prefetch.PREFETCHERS)
        self.assertIs(cachesim.VictimBuffer, VictimBuffer)

    def test_every_exported_name_resolves_and_is_listed_once(self) -> None:
        import cachesim

        self.assertEqual([n for n in cachesim.__all__ if not hasattr(cachesim, n)], [])
        self.assertEqual(len(set(cachesim.__all__)), len(cachesim.__all__))
