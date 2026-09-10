"""Known-answer tests for the per-level prefetchers.

Most cases run a single level big enough to hold the whole working set, so
the numbers isolate the prefetcher from replacement effects.
"""

from __future__ import annotations

import random
import unittest

from cachesim.cache import Cache
from cachesim.config import ConfigError, parse_config
from cachesim.hierarchy import Hierarchy, Level
from cachesim.prefetch import (
    PREFETCHER_NAMES,
    NextLinePrefetcher,
    StridePrefetcher,
    make_prefetcher,
)

BLOCK = 64


def one_level(prefetcher: str, size: int = 4096, ways: int = 4) -> Hierarchy:
    return Hierarchy(
        [Level(Cache("L1", size, BLOCK, ways), 4, prefetcher=prefetcher)],
        memory_access_time=100,
    )


def scan(blocks: int, base: int = 0) -> list[int]:
    """Every 8-byte word of ``blocks`` consecutive 64-byte blocks, in order."""
    return [base + off for off in range(0, blocks * BLOCK, 8)]


class TestNextLine(unittest.TestCase):
    def test_a_scan_takes_one_miss_instead_of_one_per_block(self) -> None:
        """32 blocks scanned word by word. Untagged next-line would halve
        the misses -- prefetch on a miss, hit, miss, prefetch -- but the tag
        keeps the stream running: the first demand hit on each prefetched
        line issues the next prefetch, so after the very first miss the
        prefetcher stays exactly one block ahead and nothing else misses.
        The one wasted prefetch is the block past the end of the scan."""
        plain = one_level("none")
        for addr in scan(32):
            plain.access(addr)
        self.assertEqual(plain.levels[0].cache.misses, 32)

        h = one_level("next-line")
        for addr in scan(32):
            h.access(addr)
        level = h.levels[0]
        self.assertEqual(level.cache.misses, 1)
        self.assertEqual(level.cache.accesses, 32 * 8)
        self.assertEqual(level.prefetches_issued, 32)  # blocks 1..32
        self.assertEqual(level.prefetch_hits, 31)  # block 32 is never used
        self.assertEqual(level.prefetch_evicted_unused, 0)  # it all still fits
        self.assertAlmostEqual(level.prefetch_accuracy, 31 / 32)
        self.assertAlmostEqual(level.prefetch_coverage, 31 / 32)
        self.assertEqual((h.dram_reads, h.dram_prefetch_reads), (1, 32))

    def test_a_prefetched_line_is_credited_once(self) -> None:
        """Only the FIRST demand hit on a prefetched line counts, and only
        it re-arms the prefetcher."""
        h = one_level("next-line")
        h.access(0)  # miss on block 0, prefetch block 1
        level = h.levels[0]
        self.assertEqual(level.prefetches_issued, 1)
        self.assertTrue(level.cache.is_prefetched(1))
        h.access(BLOCK)  # first hit on block 1: useful, prefetch block 2
        self.assertEqual((level.prefetch_hits, level.prefetches_issued), (1, 2))
        self.assertFalse(level.cache.is_prefetched(1))
        h.access(BLOCK + 8)  # same block again: nothing new
        self.assertEqual((level.prefetch_hits, level.prefetches_issued), (1, 2))

    def test_prefetches_are_not_charged_time(self) -> None:
        h = one_level("next-line")
        self.assertEqual(h.access(0), 4 + 100)  # the miss only
        self.assertEqual(h.access(BLOCK), 4)  # the prefetch already landed

    def test_a_resident_block_is_not_re_prefetched(self) -> None:
        h = one_level("next-line")
        h.access(BLOCK)  # miss on block 1, prefetch block 2
        h.access(0)  # miss on block 0; block 1 is already here
        self.assertEqual(h.levels[0].prefetches_issued, 1)

    def test_pollution_is_counted_when_a_prefetch_is_never_used(self) -> None:
        """A one-block cache: every prefetch immediately displaces the line
        the program is actually using, and none is ever hit."""
        h = one_level("next-line", size=BLOCK, ways=1)
        level = h.levels[0]
        for block in (0, 2, 4, 6):  # never touches a prefetched block+1
            h.access(block * BLOCK)
        self.assertEqual(level.prefetches_issued, 4)
        self.assertEqual(level.prefetch_hits, 0)
        self.assertEqual(level.prefetch_evicted_unused, 3)  # the fourth is still resident
        self.assertEqual([b for _, _, b, _ in level.cache.lines()], [7])
        self.assertEqual(level.prefetch_accuracy, 0.0)
        self.assertEqual(level.prefetch_coverage, 0.0)


class TestStride(unittest.TestCase):
    def test_a_128_byte_stride_becomes_almost_all_prefetch_hits(self) -> None:
        """Blocks b, b+2, b+4, ... At the third reference in a region the
        stride has been seen twice, so the prefetcher runs one step ahead
        from there on. A 4 KB region holds 64 blocks = 32 strided
        references, and crossing into the next region resets the slot: the
        prefetch issued by the last reference of a region lands the first
        block of the next one (a hit), then two more references are needed
        to re-establish the stride. So 256 references over 8 regions take
        3 + 7*2 = 17 misses instead of 256."""
        h = one_level("stride", size=64 * 1024, ways=4)
        level = h.levels[0]
        for i in range(256):
            h.access(i * 128)
        self.assertEqual(level.cache.misses, 17)
        self.assertEqual(level.prefetch_hits, 239)
        self.assertEqual(level.prefetches_issued, 240)  # the last one runs past the end
        self.assertAlmostEqual(level.prefetch_accuracy, 239 / 240)
        self.assertAlmostEqual(level.prefetch_coverage, 239 / 256)

    def test_next_line_cannot_follow_a_stride(self) -> None:
        """The same stream through a next-line prefetcher: every prefetch
        lands on the block the stream skips."""
        h = one_level("next-line", size=64 * 1024, ways=4)
        level = h.levels[0]
        for i in range(256):
            h.access(i * 128)
        self.assertEqual(level.cache.misses, 256)
        self.assertEqual(level.prefetch_hits, 0)
        self.assertEqual(level.prefetches_issued, 256)

    def test_a_negative_stride_is_followed_too(self) -> None:
        h = one_level("stride", size=64 * 1024, ways=4)
        start = 63 * BLOCK  # walk backwards inside one region
        for i in range(32):
            h.access(start - i * 2 * BLOCK)
        self.assertGreater(h.levels[0].prefetch_hits, 20)

    def test_a_stride_wider_than_a_region_is_never_learned(self) -> None:
        """Consecutive references land in different table slots, so each
        slot only ever sees its own first reference. This is the main limit
        of indexing by region instead of by program counter."""
        h = one_level("stride", size=64 * 1024, ways=4)
        for i in range(64):
            h.access(i * 8192)  # 8 KB stride = two regions
        self.assertEqual(h.levels[0].prefetches_issued, 0)
        self.assertEqual(h.levels[0].cache.misses, 64)

    def test_random_access_issues_almost_nothing_and_hits_nothing(self) -> None:
        """Deltas inside a region almost never repeat, so the confidence
        counter rarely reaches the threshold: the prefetcher stays quiet
        instead of flooding the cache. Over 20,000 uniformly random
        references across 1 MB it issues 148 prefetches -- 0.7% of the
        references -- of which 7 are ever used; accuracy is 4.7% and
        coverage 0.04%, both effectively zero, and 133 of the 148 are
        evicted having done nothing but displace a useful line."""
        rng = random.Random(101)
        h = one_level("stride", size=64 * 1024, ways=4)
        level = h.levels[0]
        for _ in range(20_000):
            h.access(rng.randrange(0, 1 << 20) & ~7)
        self.assertEqual(level.prefetches_issued, 148)
        self.assertEqual(level.prefetch_hits, 7)
        self.assertEqual(level.prefetch_evicted_unused, 133)
        self.assertLess(level.prefetch_accuracy, 0.05)
        self.assertLess(level.prefetch_coverage, 0.001)

    def test_the_table_is_direct_mapped_and_tagged(self) -> None:
        p = StridePrefetcher(entries=2, region_shift=6)
        # Two regions 2 slots apart alias onto slot 0 and evict each other.
        a, b = 0, 2 * 64 * 2  # region 0 and region 2
        self.assertIsNone(p.predict(a, False, False))
        self.assertIsNone(p.predict(b, False, False))
        self.assertIsNone(p.predict(a + 2, False, False))  # slot lost its history
        self.assertEqual(p.entries, 2)


class TestPrefetchAccounting(unittest.TestCase):
    def test_prefetch_lookups_below_are_not_demand_accesses(self) -> None:
        """L1 prefetches; L2 must report the fetch as a prefetch lookup, not
        as an access, so its miss rate still describes demand traffic."""
        h = Hierarchy(
            [
                Level(Cache("L1", 4096, BLOCK, 4), 4, prefetcher="next-line"),
                Level(Cache("L2", 65536, BLOCK, 8), 12),
            ],
            memory_access_time=100,
        )
        h.access(0)  # demand miss at both levels, then a prefetch of block 1
        l1, l2 = h.levels[0], h.levels[1]
        self.assertEqual(l1.prefetches_issued, 1)
        self.assertEqual(l2.cache.accesses, 1)  # the demand miss alone
        self.assertEqual((l2.prefetch_probes, l2.prefetch_probe_hits), (1, 0))
        self.assertTrue(l2.cache.contains(1))  # the fetch filled it on the way up
        self.assertFalse(l2.cache.is_prefetched(1))  # only the issuing level flags it
        self.assertEqual(h.accesses, 1)  # the CPU issued one reference

    def test_a_prefetch_served_by_the_level_below_is_recorded_as_a_hit(self) -> None:
        h = Hierarchy(
            [
                Level(Cache("L1", BLOCK, BLOCK, 1), 4, prefetcher="next-line"),
                Level(Cache("L2", 65536, BLOCK, 8), 12),
            ],
            memory_access_time=100,
        )
        h.access(BLOCK)  # fills block 1 everywhere, prefetches block 2
        h.access(0)  # L1 miss (one block only), prefetches block 1 -> in L2
        l2 = h.levels[1]
        self.assertEqual(l2.prefetch_probe_hits, 1)
        self.assertEqual(h.dram_prefetch_reads, 1)  # only block 2 came from DRAM

    def test_analytic_and_measured_amat_still_agree(self) -> None:
        rng = random.Random(107)
        h = Hierarchy(
            [
                Level(Cache("L1", 8192, BLOCK, 4), 4, bus_width=32, prefetcher="next-line"),
                Level(Cache("L2", 65536, BLOCK, 8), 12, bus_width=16, prefetcher="stride"),
            ],
            memory_access_time=100,
        )
        for _ in range(20_000):
            h.access(rng.randrange(0, 1 << 18), is_write=rng.random() < 0.3)
        self.assertAlmostEqual(h.amat(), h.measured_amat(), places=9)
        self.assertEqual(h.levels[1].cache.accesses, h.levels[0].cache.misses)

    def test_inclusion_survives_prefetching(self) -> None:
        for inclusion in ("inclusive", "exclusive"):
            with self.subTest(inclusion=inclusion):
                rng = random.Random(109)
                h = Hierarchy(
                    [
                        Level(Cache("L1", 512, BLOCK, 2), 4, prefetcher="next-line"),
                        Level(Cache("L2", 2048, BLOCK, 4), 12, inclusion=inclusion),
                    ],
                    memory_access_time=100,
                )
                for _ in range(5000):
                    h.access(rng.randrange(0, 1 << 14), is_write=rng.random() < 0.3)
                    h.check_inclusion()

    def test_reset_stats_keeps_the_prefetcher_trained(self) -> None:
        h = one_level("next-line")
        for addr in scan(4):
            h.access(addr)
        h.reset_stats()
        level = h.levels[0]
        self.assertEqual((level.prefetches_issued, level.prefetch_hits), (0, 0))
        # Block 4 was prefetched before the reset and its flag survives, so
        # the reference to it is still credited as a useful prefetch.
        h.access(4 * BLOCK)
        self.assertEqual(level.prefetch_hits, 1)


class TestPrefetchConfig(unittest.TestCase):
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

    def test_default_is_none(self) -> None:
        spec = parse_config(self.base())
        self.assertEqual(spec.levels[0].prefetcher, "none")
        self.assertIsNone(Hierarchy.from_spec(spec).levels[0].prefetcher)

    def test_names_are_case_insensitive(self) -> None:
        self.assertEqual(
            parse_config(self.base(prefetcher="Next-Line")).levels[0].prefetcher, "next-line"
        )

    def test_bad_values_are_rejected_with_a_path(self) -> None:
        for value, fragment in [("nextline", "unknown prefetcher"), (2, "must be a string")]:
            with self.subTest(value=value), self.assertRaises(ConfigError) as ctx:
                parse_config(self.base(prefetcher=value))
            self.assertIn(fragment, str(ctx.exception))
            self.assertIn("L1", str(ctx.exception))

    def test_the_registry_matches_the_accepted_names(self) -> None:
        self.assertEqual(PREFETCHER_NAMES, ("none", "next-line", "stride"))
        self.assertIsNone(make_prefetcher("none"))
        self.assertIsInstance(make_prefetcher("next-line"), NextLinePrefetcher)
        self.assertIsInstance(make_prefetcher("stride"), StridePrefetcher)
        with self.assertRaises(ValueError):
            make_prefetcher("magic")

    def test_stride_table_geometry_is_validated(self) -> None:
        with self.assertRaises(ValueError):
            StridePrefetcher(entries=0)
        with self.assertRaises(ValueError):
            StridePrefetcher(region_shift=-1)


if __name__ == "__main__":
    unittest.main(verbosity=2)
