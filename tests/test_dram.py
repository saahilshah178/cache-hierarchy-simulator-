"""Known-answer tests for the main-memory models.

The row-buffer cases use a deliberately small geometry -- 256-byte rows
and 2 banks -- so a hand-written address list crosses rows and banks in a
few steps and every latency can be checked by eye.
"""

from __future__ import annotations

import random
import unittest

from cachesim.cache import Cache
from cachesim.config import (
    ConfigError,
    ConstantMemorySpec,
    RowBufferMemorySpec,
    parse_config,
)
from cachesim.dram import ConstantMemory, RowBufferMemory
from cachesim.hierarchy import Hierarchy, Level

BLOCK = 64


def tiny(row_size: int = 256, banks: int = 2) -> RowBufferMemory:
    """Four 64-byte blocks per row, two banks: rows alternate between them."""
    return RowBufferMemory(row_size=row_size, banks=banks, row_hit=40, row_miss=100)


def one_level(memory: ConstantMemory | RowBufferMemory, size: int = BLOCK) -> Hierarchy:
    """A cache with a single block, so that every access reaches memory."""
    return Hierarchy([Level(Cache("L1", size, BLOCK, 1), 4)], memory=memory)


class TestConstantMemory(unittest.TestCase):
    def test_every_access_costs_the_same(self) -> None:
        m = ConstantMemory(100)
        self.assertEqual(m.read(0), 100)
        self.assertEqual(m.read(1 << 30), 100)
        self.assertEqual(m.write(12345), 100)
        self.assertEqual(m.average_latency(), 100.0)
        self.assertEqual((m.row_hits, m.row_misses), (0, 0))

    def test_average_latency_is_defined_before_any_access(self) -> None:
        self.assertEqual(ConstantMemory(7).average_latency(), 7.0)

    def test_memory_access_time_still_builds_one(self) -> None:
        h = Hierarchy([Level(Cache("L1", 1024, BLOCK, 2), 4)], memory_access_time=100)
        self.assertIsInstance(h.memory, ConstantMemory)
        self.assertEqual(h.memory_access_time, 100)
        self.assertEqual(h.access(0), 4 + 100)

    def test_the_two_forms_are_mutually_exclusive(self) -> None:
        with self.assertRaises(ValueError):
            Hierarchy([Level(Cache("L1", 1024, BLOCK, 2), 4)], 100, memory=ConstantMemory(50))
        with self.assertRaises(ValueError):
            Hierarchy([Level(Cache("L1", 1024, BLOCK, 2), 4)])

    def test_negative_latency_is_rejected(self) -> None:
        with self.assertRaises(ValueError):
            ConstantMemory(-1)


class TestRowBuffer(unittest.TestCase):
    def test_the_first_access_to_a_bank_is_always_a_row_miss(self) -> None:
        m = tiny()
        self.assertEqual(m.read(0), 100)  # row 0, bank 0: closed
        self.assertEqual(m.read(64), 40)  # same row, still open
        self.assertEqual(m.read(192), 40)  # still row 0
        self.assertEqual(m.read(256), 100)  # row 1, bank 1: closed
        self.assertEqual(m.read(300), 40)  # row 1 again
        self.assertEqual((m.row_hits, m.row_misses), (3, 2))

    def test_two_streams_in_different_banks_do_not_evict_each_other(self) -> None:
        """Rows alternate between banks, so a stream in row 0 and one in
        row 1 each keep their own row open."""
        m = tiny()
        for _ in range(4):
            m.read(0)  # row 0, bank 0
            m.read(256)  # row 1, bank 1
        self.assertEqual((m.row_hits, m.row_misses), (6, 2))

    def test_two_streams_in_the_same_bank_thrash(self) -> None:
        """Rows 0 and 2 both map to bank 0 (2 banks), so alternating between
        them re-activates a row on every single access."""
        m = tiny()
        for _ in range(4):
            m.read(0)  # row 0, bank 0
            m.read(512)  # row 2, bank 0
        self.assertEqual((m.row_hits, m.row_misses), (0, 8))

    def test_writes_share_the_row_buffer_with_reads(self) -> None:
        m = tiny()
        m.read(0)
        self.assertEqual(m.write(64), 40)  # the read left row 0 open
        self.assertEqual(m.write(512), 100)  # row 2, same bank: activates
        self.assertEqual(m.read(0), 100)  # ... and the write cost the read its row
        self.assertEqual((m.reads, m.writes), (2, 2))

    def test_only_charged_accesses_enter_the_average(self) -> None:
        m = tiny()
        m.read(0, charged=True)  # 100
        m.read(64, charged=True)  # 40
        m.read(128, charged=False)  # a prefetch: still counted as a row hit
        self.assertEqual((m.row_hits, m.row_misses), (2, 1))
        self.assertAlmostEqual(m.average_latency(), 70.0)  # (100 + 40) / 2

    def test_average_latency_before_any_charged_access(self) -> None:
        self.assertEqual(tiny().average_latency(), 100.0)  # the closed-page cost

    def test_open_rows_are_kept_across_a_stats_reset(self) -> None:
        m = tiny()
        m.read(0)
        m.reset_stats()
        self.assertEqual((m.row_hits, m.row_misses), (0, 0))
        self.assertEqual(m.read(64), 40)  # row 0 is still open
        self.assertEqual(m.open_rows(), [0, -1])

    def test_geometry_is_validated(self) -> None:
        for kwargs in (
            {"row_size": 0},
            {"banks": 0},
            {"row_hit": -1},
            {"row_hit": 90, "row_miss": 50},
            {"banks": "eight"},
        ):
            with self.subTest(**kwargs), self.assertRaises(ValueError):
                RowBufferMemory(**kwargs)


class TestRowBufferInAHierarchy(unittest.TestCase):
    def test_a_sequential_miss_stream_is_almost_all_row_hits(self) -> None:
        """A one-block cache scanned block by block: every access misses, and
        every access but the first of each row finds the row open. With
        4 blocks per row, 64 blocks make 16 row misses and 48 hits."""
        h = one_level(tiny())
        for block in range(64):
            h.access(block * BLOCK)
        m = h.memory
        self.assertEqual(h.dram_reads, 64)
        self.assertEqual((m.row_hits, m.row_misses), (48, 16))
        self.assertAlmostEqual(m.average_latency(), (48 * 40 + 16 * 100) / 64)
        self.assertAlmostEqual(h.amat(), h.measured_amat(), places=9)

    def test_a_random_miss_stream_is_almost_all_row_misses(self) -> None:
        rng = random.Random(211)
        h = one_level(tiny())
        for _ in range(5000):
            h.access(rng.randrange(0, 1 << 20) & ~(BLOCK - 1))
        m = h.memory
        self.assertLess(m.row_buffer_hit_rate, 0.02)
        self.assertGreater(m.average_latency(), 98.0)

    def test_a_pointer_chase_sits_between_the_two(self) -> None:
        """Nodes are one block each and visited in a shuffled order, so
        consecutive hops rarely share a row -- but the region is small
        enough that they sometimes do."""
        rng = random.Random(213)
        nodes = list(range(512))
        rng.shuffle(nodes)
        h = one_level(tiny())
        for node in nodes:
            h.access(node * BLOCK)
        rate = h.memory.row_buffer_hit_rate
        self.assertGreater(rate, 0.0)
        self.assertLess(rate, 0.20)

    def test_analytic_amat_uses_the_measured_average_latency(self) -> None:
        rng = random.Random(217)
        h = Hierarchy(
            [
                Level(Cache("L1", 4096, BLOCK, 4), 4),
                Level(Cache("L2", 32768, BLOCK, 8), 12, bus_width=16),
            ],
            memory=RowBufferMemory(row_size=8192, banks=8, row_hit=40, row_miss=100),
        )
        for _ in range(20_000):
            h.access(rng.randrange(0, 1 << 20), is_write=rng.random() < 0.3)
        self.assertAlmostEqual(h.amat(), h.measured_amat(), places=9)
        self.assertIsNone(h.memory_access_time)  # no single latency to report
        # The analytic figure is built on the measured DRAM average, so it
        # must lie between the row-hit and row-miss costs.
        self.assertLess(40.0, h.memory.average_latency())
        self.assertLess(h.memory.average_latency(), 100.0)

    def test_write_backs_disturb_the_open_rows(self) -> None:
        """A write-back is untimed but still activates a row, so it can cost
        the demand stream the row it was using -- the interference a model
        that ignored write traffic would miss.

        Store to block 0 (row 0, bank 0), then read block 8 (row 2, also
        bank 0). The read activates row 2; the eviction it causes writes
        block 0 back, pulling row 0 into the same bank again. The next
        reference to row 2 therefore has to re-activate it."""
        h = one_level(tiny())
        h.access(0, is_write=True)
        h.access(512)
        m = h.memory
        assert isinstance(m, RowBufferMemory)
        self.assertEqual((h.dram_reads, h.dram_writes), (2, 1))
        self.assertEqual(m.writes, 1)
        self.assertEqual((m.row_hits, m.row_misses), (0, 3))
        self.assertEqual(m.open_rows(), [0, -1])  # the write-back's row, not the read's
        self.assertAlmostEqual(m.average_latency(), 100.0)  # only the 2 reads are charged
        self.assertEqual(h.access(576), 4 + 100)  # row 2 again: it has to re-activate

    def test_a_prefetch_moves_an_open_row_without_being_charged(self) -> None:
        h = Hierarchy(
            [Level(Cache("L1", 4 * BLOCK, BLOCK, 1), 4, prefetcher="next-line")],
            memory=tiny(),
        )
        h.access(0)  # row 0 miss, then prefetch block 1 (row 0, a hit)
        m = h.memory
        self.assertEqual((m.reads, m.row_hits, m.row_misses), (2, 1, 1))
        self.assertAlmostEqual(m.average_latency(), 100.0)  # the prefetch is free
        self.assertEqual(h.dram_prefetch_reads, 1)


class TestMemoryConfig(unittest.TestCase):
    def levels(self) -> list[dict[str, object]]:
        return [{"name": "L1", "size": 1024, "block_size": 64, "associativity": 2, "hit_time": 1}]

    def test_memory_access_time_and_memory_integer_are_equivalent(self) -> None:
        a = parse_config({"memory_access_time": 42, "levels": self.levels()})
        b = parse_config({"memory": 42, "levels": self.levels()})
        self.assertEqual(a, b)
        self.assertEqual(a.memory, ConstantMemorySpec(42))
        self.assertEqual(a.memory_access_time, 42)

    def test_row_buffer_defaults(self) -> None:
        spec = parse_config({"memory": {"type": "row-buffer"}, "levels": self.levels()})
        self.assertEqual(spec.memory, RowBufferMemorySpec(8192, 8, 40, 100))
        self.assertIsNone(spec.memory_access_time)

    def test_row_buffer_round_trips_through_to_dict(self) -> None:
        spec = parse_config(
            {
                "memory": {"type": "row-buffer", "row_size": 1024, "banks": 4},
                "levels": self.levels(),
            }
        )
        self.assertEqual(parse_config(spec.to_dict()), spec)
        self.assertEqual(spec.memory, RowBufferMemorySpec(1024, 4, 40, 100))

    def test_explicit_constant_object(self) -> None:
        spec = parse_config(
            {"memory": {"type": "constant", "latency": 33}, "levels": self.levels()}
        )
        self.assertEqual(spec.memory, ConstantMemorySpec(33))

    def test_bad_values_are_rejected_with_a_path(self) -> None:
        cases: list[tuple[dict[str, object], str]] = [
            (
                {"memory_access_time": 1, "memory": 2},
                "not both",
            ),
            ({"levels": None}, "missing required top-level key(s) 'memory_access_time'"),
            ({"memory": "fast"}, "must be an integer latency or an object"),
            ({"memory": {"type": "hbm"}}, "unknown type 'hbm'"),
            ({"memory": {"type": "row-buffer", "banks": 0}}, "'banks' must be >= 1"),
            ({"memory": {"type": "row-buffer", "rows": 4}}, "unknown key(s) 'rows'"),
            (
                {"memory": {"type": "row-buffer", "row_hit": 90, "row_miss": 50}},
                "must be at least",
            ),
            ({"memory": {"type": "constant"}}, "missing required key 'latency'"),
            ({"memory": {"type": "constant", "latency": -1}}, "must be >= 0"),
        ]
        for fragment_config, fragment in cases:
            config = {"levels": self.levels(), **fragment_config}
            with self.subTest(fragment=fragment), self.assertRaises(ConfigError) as ctx:
                parse_config(config)
            self.assertIn(fragment, str(ctx.exception))

    def test_an_old_config_still_parses(self) -> None:
        """Every configuration written before the memory key existed keeps
        working, and keeps meaning a constant latency."""
        h = Hierarchy.from_config({"memory_access_time": 100, "levels": self.levels()})
        self.assertIsInstance(h.memory, ConstantMemory)
        self.assertEqual(h.memory_access_time, 100)


if __name__ == "__main__":
    unittest.main(verbosity=2)
