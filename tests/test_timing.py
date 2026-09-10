"""Known-answer tests for transfer time, bandwidth accounting, and AMAT.

The transfer term is ``ceil(block_size / bus_width)`` cycles per fill into
a level, charged to the access that caused the fill; ``bus_width: null``
(the default) means an infinitely wide link and no term at all.
"""

from __future__ import annotations

import random
import unittest

from cachesim.cache import Cache
from cachesim.config import ConfigError, parse_config
from cachesim.hierarchy import Hierarchy, Level
from cachesim.workloads import matmul, sequential


def single(bus_width: int | None, block_size: int = 64) -> Hierarchy:
    """One 1 KB 2-way level, hit time 4, DRAM 100."""
    cache = Cache("L1", 1024, block_size, 2)
    return Hierarchy([Level(cache, 4, bus_width=bus_width)], memory_access_time=100)


class TestTransferTime(unittest.TestCase):
    def test_no_bus_width_preserves_the_original_timing(self) -> None:
        h = single(None)
        self.assertEqual(h.levels[0].transfer_cycles, 0)
        self.assertEqual(h.access(0), 4 + 100)
        self.assertEqual(h.access(0), 4)

    def test_every_miss_costs_hit_plus_memory_plus_transfer(self) -> None:
        """64 B blocks over an 8 B/cycle bus: eight cycles per fill."""
        h = single(8)
        self.assertEqual(h.levels[0].transfer_cycles, 8)
        self.assertEqual(h.access(0), 4 + 100 + 8)
        self.assertEqual(h.access(0), 4)  # a hit fills nothing
        self.assertEqual(h.access(64), 4 + 100 + 8)
        self.assertAlmostEqual(h.amat(), h.measured_amat(), places=9)

    def test_transfer_cycles_round_up(self) -> None:
        for block_size, width, expected in [(64, 64, 1), (64, 65, 1), (64, 63, 2), (16, 3, 6)]:
            with self.subTest(block_size=block_size, bus_width=width):
                h = single(width, block_size=block_size)
                self.assertEqual(h.levels[0].transfer_cycles, expected)

    def test_each_level_pays_for_its_own_fill(self) -> None:
        """A miss that reaches DRAM fills L2 and then L1, so it is charged
        both transfer terms; a miss that L2 serves is charged only L1's."""
        h = Hierarchy(
            [
                Level(Cache("L1", 256, 64, 1), 4, bus_width=32),  # 2 cyc
                Level(Cache("L2", 4096, 64, 4), 12, bus_width=8),  # 8 cyc
            ],
            memory_access_time=100,
        )
        self.assertEqual(h.access(0), 4 + 12 + 100 + 8 + 2)
        # Block 4 aliases L1 set 0 (4 sets, direct mapped) but not L2's.
        h.access(4 * 64)
        self.assertEqual(h.access(0), 4 + 12 + 2)  # L2 supplies it: only L1 fills

    def test_analytic_amat_matches_measurement_with_transfers(self) -> None:
        rng = random.Random(41)
        h = Hierarchy(
            [
                Level(Cache("L1", 8192, 64, 4), 4, bus_width=32),
                Level(Cache("L2", 65536, 64, 8), 12, bus_width=16),
                Level(Cache("L3", 262144, 64, 16), 40, bus_width=8),
            ],
            memory_access_time=100,
        )
        for _ in range(30_000):
            h.access(rng.randrange(0, 1 << 20), is_write=rng.random() < 0.3)
        self.assertAlmostEqual(h.amat(), h.measured_amat(), places=9)
        # ... and the formula really is doing something: without the
        # transfer terms it would be lower.
        plain = 4.0
        penalty: float = 100.0
        for level in reversed(h.levels):
            penalty = level.hit_time + level.cache.miss_rate * penalty
        plain = penalty
        self.assertGreater(h.amat(), plain)


class TestReadWriteSplit(unittest.TestCase):
    def test_cycles_are_attributed_to_the_access_type(self) -> None:
        h = single(None)
        h.access(0)  # load miss: 104
        h.access(0, is_write=True)  # store hit: 4
        h.access(64, is_write=True)  # store miss: 104
        self.assertEqual((h.read_cycles, h.write_cycles), (104, 108))
        self.assertEqual(h.read_cycles + h.write_cycles, h.total_time)
        self.assertAlmostEqual(h.read_amat(), 104.0)
        self.assertAlmostEqual(h.write_amat(), 54.0)
        self.assertAlmostEqual(h.measured_amat(), 212 / 3)

    def test_stores_are_cheaper_than_loads_when_they_hit_more(self) -> None:
        """The sequential-scan shape: every block is missed once by whichever
        access touches it first, and the rest hit."""
        h = single(None)
        for i in range(64):  # 4 loads then 4 stores per 64 B block
            h.access(i * 8, is_write=(i % 8) >= 4)
        self.assertEqual(h.reads + h.writes, 64)
        self.assertGreater(h.read_amat(), h.write_amat())  # loads took every miss
        self.assertEqual(h.write_cycles, 32 * 4)


class TestByteCounters(unittest.TestCase):
    def test_a_cold_miss_moves_one_block_up_at_every_level(self) -> None:
        h = Hierarchy(
            [Level(Cache("L1", 256, 64, 1), 4), Level(Cache("L2", 4096, 64, 4), 12)],
            memory_access_time=100,
        )
        h.access(0)
        self.assertEqual(h.levels[0].bytes_read_from_below, 64)
        self.assertEqual(h.levels[1].bytes_read_from_below, 64)
        self.assertEqual((h.dram_bytes_read, h.dram_bytes_written), (64, 0))

    def test_a_writeback_counts_bytes_out_of_the_level_that_lost_it(self) -> None:
        l1 = Cache("L1", 64, 64, 1)  # one block
        h = Hierarchy([Level(l1, 4)], memory_access_time=100)
        h.access(0, is_write=True)
        h.access(64)  # evicts dirty block 0 to DRAM
        self.assertEqual(h.levels[0].bytes_written_below, 64)
        self.assertEqual(h.dram_bytes_written, 64)
        self.assertEqual(h.dram_bytes_read, 128)

    def test_write_through_moves_far_more_data_down(self) -> None:
        """Same stream, two policies: write-back sends one block down per
        dirty eviction, write-through one per store."""
        stream = [(i % 8) * 64 for i in range(200)]
        traffic = {}
        for policy in ("write-back", "write-through"):
            h = Hierarchy(
                [
                    Level(Cache("L1", 1024, 64, 2), 4, write_policy=policy),
                    Level(Cache("L2", 8192, 64, 4), 12),
                ],
                memory_access_time=100,
            )
            for addr in stream:
                h.access(addr, is_write=True)
            traffic[policy] = h.levels[0].bytes_written_below
        self.assertEqual(traffic["write-through"], 200 * 64)  # one per store
        self.assertLess(traffic["write-back"], traffic["write-through"])


class TestBusWidthConfig(unittest.TestCase):
    def base(self, **extra: object) -> dict[str, object]:
        return {
            "memory_access_time": 10,
            "levels": [
                {
                    "name": "L1",
                    "size": 1024,
                    "block_size": 64,
                    "associativity": 2,
                    "hit_time": 1,
                    **extra,
                }
            ],
        }

    def test_default_is_no_transfer_term(self) -> None:
        spec = parse_config(self.base())
        self.assertIsNone(spec.levels[0].bus_width)
        self.assertEqual(Hierarchy.from_spec(spec).levels[0].transfer_cycles, 0)

    def test_explicit_null_is_accepted(self) -> None:
        self.assertIsNone(parse_config(self.base(bus_width=None)).levels[0].bus_width)

    def test_bad_values_are_rejected_with_a_path(self) -> None:
        for value, fragment in [(0, "must be >= 1"), (-8, "must be >= 1"), ("8", "integer")]:
            with self.subTest(value=value), self.assertRaises(ConfigError) as ctx:
                parse_config(self.base(bus_width=value))
            self.assertIn(fragment, str(ctx.exception))
            self.assertIn("bus_width", str(ctx.exception))
            self.assertIn("L1", str(ctx.exception))

    def test_level_rejects_a_non_positive_width_directly(self) -> None:
        with self.assertRaises(ValueError) as ctx:
            Level(Cache("L1", 1024, 64, 2), 4, bus_width=0)
        self.assertIn("bus_width must be positive", str(ctx.exception))


class TestBlockSizeTradeoff(unittest.TestCase):
    """Wider blocks cut the miss rate but cost more to move, so AMAT need
    not fall with them. Which way it goes depends on the workload."""

    SIZES = (16, 32, 64, 128, 256, 512)

    def sweep(self, stream: list[tuple[int, bool]], size: int, ways: int) -> dict[int, float]:
        """AMAT against block size for a single level with a 16 B/cyc bus."""
        amats = {}
        previous = 1.0
        for block_size in self.SIZES:
            h = Hierarchy(
                [Level(Cache("L1", size, block_size, ways), 4, bus_width=16)],
                memory_access_time=100,
            )
            for addr, is_write in stream:
                h.access(addr, is_write)
            self.assertAlmostEqual(h.amat(), h.measured_amat(), places=9)
            # Whatever AMAT does, wider blocks always fetch more of what the
            # program is about to use: the miss rate falls monotonically.
            miss_rate = h.levels[0].cache.miss_rate
            self.assertLess(miss_rate, previous)
            previous = miss_rate
            amats[block_size] = h.measured_amat()
        return amats

    def test_a_perfect_scan_never_pays_for_wider_blocks(self) -> None:
        """A linear scan uses every byte it fetches, so the miss rate falls
        as 1/block while the transfer term grows only as block/width: AMAT
        is 4 + (8/B)*100 + 0.5 and falls monotonically. There is no interior
        minimum here for any bus width, which is exactly why the trade-off
        has to be shown on a workload with imperfect spatial locality."""
        stream = [(addr, False) for _ in range(2) for addr in range(0, 32 * 1024, 8)]
        amats = self.sweep(stream, size=8192, ways=4)
        self.assertEqual(min(amats, key=lambda b: amats[b]), self.SIZES[-1])
        self.assertAlmostEqual(amats[64], 4 + 0.125 * (100 + 4))
        self.assertAlmostEqual(amats[512], 4 + 0.015625 * (100 + 32))

    def test_amat_bottoms_out_and_rises_on_a_matrix_multiply(self) -> None:
        """32x32 naive matmul in an 8 KB 4-way cache: the walk down B's
        columns strides 256 B, so a wider block brings in mostly unused
        bytes, and past 128 B the transfer time costs more than the misses
        it saves."""
        stream = [(addr, op == "W") for addr, op in matmul(n=32)]
        amats = self.sweep(stream, size=8192, ways=4)
        best = min(amats, key=lambda b: amats[b])
        self.assertEqual(best, 128)
        self.assertLess(amats[128], amats[64])
        self.assertLess(amats[128], amats[256])
        self.assertLess(amats[256], amats[512])


class TestSampleTraceBlockSizeTable(unittest.TestCase):
    """Pins the block-size table quoted in ``hierarchy``'s module docstring.

    Geometry: one 32 KB 4-way level, hit time 4, ``bus_width`` 16, DRAM
    100 cycles, block size swept over 16..512 B. The two workloads are the
    ones behind ``traces/sequential.trace`` and ``traces/matmul_naive.trace``::

        block   sequential          matmul_naive        transfer
           16   50.0000% / 54.500    7.1575% / 11.229      1
           32   25.0000% / 29.500    5.8804% /  9.998      2
           64   12.5000% / 17.000    5.2419% /  9.452      4
          128    6.2500% / 10.750    4.9226% /  9.316      8
          256    3.1250% /  7.625    4.0587% /  8.708     16
          512    1.5625% /  6.062    3.9288% /  9.186     32

    Only the second column is a U. The matmul rows are checked around the
    minimum only, which is where the claim lives and where the sweep is
    expensive (532,480 accesses per block size).
    """

    def point(self, stream: list[tuple[int, bool]], block_size: int) -> tuple[float, float]:
        """Run the stream through the table's geometry; (miss rate, AMAT)."""
        h = Hierarchy(
            [Level(Cache("L1", 32 * 1024, block_size, 4), 4, bus_width=16)],
            memory_access_time=100,
        )
        for addr, is_write in stream:
            h.access(addr, is_write)
        self.assertEqual(h.levels[0].transfer_cycles, block_size // 16)
        self.assertAlmostEqual(h.amat(), h.measured_amat(), places=9)
        return h.levels[0].cache.miss_rate, h.measured_amat()

    def test_the_sequential_column_is_the_closed_form_and_never_turns(self) -> None:
        """``sequential`` walks 8-byte words, so a B-byte block absorbs B/8
        of them and the miss rate is exactly 8/B at every size: AMAT is
        ``4 + (8/B)*(100 + B/16) = 4 + 800/B + 0.5``. That falls
        monotonically for any bus width, so this trace can never show the
        block-size U the transfer term is supposed to produce -- which is
        why the docstring table carries a second workload."""
        stream = [(addr, op == "W") for addr, op in sequential()]
        amats = {}
        for block_size in (16, 32, 64, 128, 256, 512):
            miss_rate, amat = self.point(stream, block_size)
            self.assertAlmostEqual(miss_rate, 8 / block_size, places=10)
            self.assertAlmostEqual(amat, 4 + (8 / block_size) * (100 + block_size / 16), places=10)
            amats[block_size] = amat
        self.assertAlmostEqual(amats[16], 54.5)
        self.assertAlmostEqual(amats[512], 6.0625)
        self.assertEqual(sorted(amats, key=lambda b: amats[b])[0], 512)

    def test_the_matmul_naive_column_bottoms_out_at_256_bytes(self) -> None:
        """The same geometry on the 64x64 naive matrix multiply: the miss
        rate keeps falling from 128 B to 512 B, but AMAT turns at 256 B
        because the 32-cycle transfer of a 512 B block costs more than the
        692 misses it saves."""
        stream = [(addr, op == "W") for addr, op in matmul(n=64)]
        rates, amats = {}, {}
        for block_size in (128, 256, 512):
            rates[block_size], amats[block_size] = self.point(stream, block_size)
        for block_size, rate, amat in (
            (128, 0.0492263, 9.316),
            (256, 0.0405874, 8.708),
            (512, 0.0392879, 9.186),
        ):
            self.assertAlmostEqual(rates[block_size], rate, places=6)
            self.assertAlmostEqual(amats[block_size], amat, places=3)
        self.assertLess(rates[512], rates[256])
        self.assertLess(rates[256], rates[128])
        self.assertLess(amats[256], amats[128])
        self.assertLess(amats[256], amats[512])


if __name__ == "__main__":
    unittest.main(verbosity=2)
