"""Known-answer tests for the per-level write policies.

Geometry throughout: L1 is 1 set x 2 ways and L2 is 1 set x 4 ways, both
LRU with 64 B blocks, hit times 4 and 12, DRAM 100 cycles. The four
combinations of ``write_policy`` x ``write_allocate`` are exercised on the
same two-access stream so their differences are directly comparable.
"""

from __future__ import annotations

import random
import unittest

from cachesim.cache import Cache
from cachesim.config import ConfigError, parse_config
from cachesim.hierarchy import Hierarchy, Level

BLOCK = 64


def one_set(name: str, ways: int) -> Cache:
    return Cache(name, size=ways * BLOCK, block_size=BLOCK, associativity=ways)


def two_level(write_policy: str = "write-back", write_allocate: bool = True) -> Hierarchy:
    """L1 with the given write policy over a plain write-back L2."""
    return Hierarchy(
        [
            Level(
                one_set("L1", 2),
                hit_time=4,
                write_policy=write_policy,
                write_allocate=write_allocate,
            ),
            Level(one_set("L2", 4), hit_time=12),
        ],
        memory_access_time=100,
    )


class TestWriteBackWriteAllocate(unittest.TestCase):
    """The default, unchanged: the store is applied in L1 and stops there."""

    def test_store_to_a_cold_block(self) -> None:
        h = two_level()
        l1, l2 = h.levels[0].cache, h.levels[1].cache
        self.assertEqual(h.access(0, is_write=True), 4 + 12 + 100)
        self.assertEqual((l1.write_misses, l1.fills), (1, 1))
        self.assertEqual((l2.write_misses, l2.read_misses), (0, 1))  # L2 sees a fill
        self.assertTrue(l1.is_dirty(0))
        self.assertFalse(l2.is_dirty(0))
        self.assertEqual((h.dram_reads, h.dram_writes), (1, 0))
        self.assertEqual((h.levels[0].write_throughs, h.levels[0].write_bypasses), (0, 0))


class TestWriteThrough(unittest.TestCase):
    def test_store_leaves_l1_clean_and_dirties_l2(self) -> None:
        h = two_level(write_policy="write-through")
        l1, l2 = h.levels[0].cache, h.levels[1].cache
        self.assertEqual(h.access(0, is_write=True), 4 + 12 + 100)
        self.assertTrue(l1.contains(0))
        self.assertFalse(l1.is_dirty(0))  # a write-through line is never dirty
        self.assertTrue(l2.is_dirty(0))
        self.assertEqual(h.levels[0].write_throughs, 1)
        self.assertEqual(l2.writebacks_received, 1)
        self.assertEqual(h.dram_writes, 0)

    def test_a_write_hit_costs_only_the_hit_time(self) -> None:
        """The duplicate sent down is assumed absorbed by a write buffer, so
        it is counted but never charged."""
        h = two_level(write_policy="write-through")
        h.access(0, is_write=True)
        self.assertEqual(h.access(0, is_write=True), 4)
        l1 = h.levels[0].cache
        self.assertEqual((l1.write_hits, l1.write_misses), (1, 1))
        self.assertFalse(l1.is_dirty(0))
        self.assertEqual(h.levels[0].write_throughs, 2)

    def test_an_evicted_write_through_line_is_never_written_back(self) -> None:
        h = two_level(write_policy="write-through")
        for block in (0, 1, 2):  # the third store evicts the first line
            h.access(block * BLOCK, is_write=True)
        l1 = h.levels[0].cache
        self.assertEqual(l1.evictions, 1)
        self.assertEqual(l1.writebacks, 0)  # nothing dirty ever leaves L1
        self.assertEqual(h.levels[0].write_throughs, 3)
        self.assertEqual(h.flush(), 3)  # the three dirty lines now sit in L2

    def test_write_through_all_the_way_reaches_dram(self) -> None:
        """A store that passes the last level is a DRAM write."""
        h = Hierarchy(
            [
                Level(one_set("L1", 2), 4, write_policy="write-through"),
                Level(one_set("L2", 4), 12, write_policy="write-through"),
            ],
            memory_access_time=100,
        )
        h.access(0, is_write=True)
        self.assertEqual(h.dram_writes, 1)
        self.assertEqual(h.dram_demand_writes, 0)  # both levels did allocate
        self.assertEqual((h.levels[0].write_throughs, h.levels[1].write_throughs), (1, 1))
        self.assertFalse(any(dirty for *_, dirty in h.levels[1].cache.lines()))
        self.assertEqual(h.flush(), 0)


class TestNoWriteAllocate(unittest.TestCase):
    def test_a_written_only_block_never_enters_l1(self) -> None:
        h = two_level(write_allocate=False)
        l1, l2 = h.levels[0].cache, h.levels[1].cache
        self.assertEqual(h.access(0, is_write=True), 4 + 12 + 100)
        self.assertFalse(l1.contains(0))
        self.assertEqual((l1.write_misses, l1.fills), (1, 0))
        self.assertEqual(h.levels[0].write_bypasses, 1)
        # L2 sees the store as a store, allocates for it, and holds the data.
        self.assertEqual(l2.write_misses, 1)
        self.assertTrue(l2.is_dirty(0))
        self.assertEqual((h.dram_reads, h.dram_writes), (1, 0))

    def test_reads_are_unaffected(self) -> None:
        h = two_level(write_allocate=False)
        l1 = h.levels[0].cache
        self.assertEqual(h.access(0), 4 + 12 + 100)
        self.assertTrue(l1.contains(0))
        self.assertEqual(h.access(0), 4)
        self.assertEqual((l1.read_hits, l1.read_misses), (1, 1))

    def test_a_store_no_level_allocates_for_goes_straight_to_dram(self) -> None:
        h = Hierarchy([Level(one_set("L1", 2), 4, write_allocate=False)], memory_access_time=100)
        self.assertEqual(h.access(0, is_write=True), 4 + 100)
        self.assertEqual((h.dram_reads, h.dram_writes, h.dram_demand_writes), (0, 1, 1))
        self.assertFalse(h.levels[0].cache.contains(0))  # nothing was fetched
        self.assertAlmostEqual(h.amat(), h.measured_amat())

    def test_a_write_hit_is_still_applied_here(self) -> None:
        """no-write-allocate changes only what a write MISS does."""
        h = two_level(write_allocate=False)
        h.access(0)  # a load brings the block in
        h.access(0, is_write=True)
        l1 = h.levels[0].cache
        self.assertEqual((l1.write_hits, l1.write_misses), (1, 0))
        self.assertTrue(l1.is_dirty(0))
        self.assertEqual(h.levels[0].write_bypasses, 0)


class TestWriteThroughNoWriteAllocate(unittest.TestCase):
    """Write-around L1: the classic pairing, since a write-through level has
    no reason to fetch a line it is not going to keep modified data in."""

    def test_a_store_miss_bypasses_and_a_store_hit_writes_through(self) -> None:
        h = two_level(write_policy="write-through", write_allocate=False)
        l1, l2 = h.levels[0].cache, h.levels[1].cache
        h.access(0, is_write=True)  # miss: bypassed, not written through
        self.assertEqual((h.levels[0].write_bypasses, h.levels[0].write_throughs), (1, 0))
        self.assertFalse(l1.contains(0))
        self.assertTrue(l2.is_dirty(0))
        h.access(0)  # a load pulls a clean copy up into L1
        self.assertTrue(l1.contains(0))
        h.access(0, is_write=True)  # hit: applied here and duplicated down
        self.assertEqual((h.levels[0].write_bypasses, h.levels[0].write_throughs), (1, 1))
        self.assertFalse(l1.is_dirty(0))
        self.assertTrue(l2.is_dirty(0))


#: Every combination of the two write-policy keys.
COMBOS = [
    ("write-back", True),
    ("write-back", False),
    ("write-through", True),
    ("write-through", False),
]


class RecordingHierarchy(Hierarchy):
    """A hierarchy that remembers every block it writes out to DRAM."""

    def __init__(self, levels: list[Level], memory_access_time: int) -> None:
        super().__init__(levels, memory_access_time)
        self.written: list[int] = []

    def _write_to_memory(self, block: int) -> None:
        super()._write_to_memory(block)
        self.written.append(block)


class TestAllFourCombinations(unittest.TestCase):
    def test_where_a_cold_store_leaves_the_block(self) -> None:
        """One cold store to block 0, and where the only copy of the modified
        data ends up: in L1 only when L1 both allocates and keeps it dirty."""
        expected = {
            # (policy, allocate): (l1 holds, l1 dirty, l2 dirty, dram_reads)
            ("write-back", True): (True, True, False, 1),
            ("write-back", False): (False, False, True, 1),
            ("write-through", True): (True, False, True, 1),
            ("write-through", False): (False, False, True, 1),
        }
        for policy, allocate in COMBOS:
            with self.subTest(write_policy=policy, write_allocate=allocate):
                h = two_level(policy, allocate)
                h.access(0, is_write=True)
                l1, l2 = h.levels[0].cache, h.levels[1].cache
                self.assertEqual(
                    (l1.contains(0), l1.is_dirty(0), l2.is_dirty(0), h.dram_reads),
                    expected[(policy, allocate)],
                )
                # However it got there, exactly one copy of the data is
                # modified, so exactly one block reaches DRAM on a flush.
                self.assertEqual(h.flush(), 1)

    def test_analytic_and_measured_amat_agree_for_every_combination(self) -> None:
        for policy, allocate in COMBOS:
            with self.subTest(write_policy=policy, write_allocate=allocate):
                rng = random.Random(23)
                h = Hierarchy(
                    [
                        Level(
                            Cache("L1", 1024, 64, 2),
                            4,
                            write_policy=policy,
                            write_allocate=allocate,
                        ),
                        Level(Cache("L2", 8192, 64, 4), 12),
                    ],
                    memory_access_time=100,
                )
                for _ in range(20_000):
                    h.access(rng.randrange(0, 1 << 16), is_write=rng.random() < 0.4)
                self.assertAlmostEqual(h.amat(), h.measured_amat(), places=9)
                # Every level's access count is the miss count of the one above.
                self.assertEqual(h.levels[1].cache.accesses, h.levels[0].cache.misses)
                self.assertEqual(h.dram_reads + h.dram_demand_writes, h.levels[1].cache.misses)

    def test_no_dirty_line_ever_sits_at_a_write_through_level(self) -> None:
        rng = random.Random(29)
        h = two_level(write_policy="write-through")
        for _ in range(5000):
            h.access(rng.randrange(0, 1 << 12), is_write=rng.random() < 0.5)
            self.assertFalse(any(dirty for *_, dirty in h.levels[0].cache.lines()))

    def test_modified_data_is_conserved_for_every_combination(self) -> None:
        for policy, allocate in COMBOS:
            with self.subTest(write_policy=policy, write_allocate=allocate):
                rng = random.Random(31)
                h = RecordingHierarchy(
                    [
                        Level(
                            Cache("L1", 512, 64, 2),
                            4,
                            write_policy=policy,
                            write_allocate=allocate,
                        ),
                        Level(Cache("L2", 2048, 64, 4), 12),
                    ],
                    memory_access_time=100,
                )
                stored: set[int] = set()
                for _ in range(5000):
                    addr = rng.randrange(0, 1 << 14)
                    is_write = rng.random() < 0.3
                    if is_write:
                        stored.add(addr // BLOCK)
                    h.access(addr, is_write)
                h.flush()
                self.assertTrue(stored <= set(h.written))
                self.assertEqual(len(h.written), h.dram_writes)


class TestWritePolicyConfig(unittest.TestCase):
    def base(self, **level_extra: object) -> dict[str, object]:
        return {
            "memory_access_time": 10,
            "levels": [
                {
                    "name": "L1",
                    "size": 1024,
                    "block_size": 64,
                    "associativity": 2,
                    "hit_time": 1,
                    **level_extra,
                }
            ],
        }

    def test_defaults_preserve_the_original_behaviour(self) -> None:
        spec = parse_config(self.base())
        self.assertEqual(spec.levels[0].write_policy, "write-back")
        self.assertTrue(spec.levels[0].write_allocate)

    def test_policy_names_are_case_insensitive(self) -> None:
        spec = parse_config(self.base(write_policy="WRITE-THROUGH"))
        self.assertEqual(spec.levels[0].write_policy, "write-through")

    def test_bad_values_are_rejected_with_a_path(self) -> None:
        cases = [
            (self.base(write_policy="writeback"), "unknown write policy"),
            (self.base(write_policy=1), "must be a string"),
            (self.base(write_allocate="yes"), "must be true or false"),
            (self.base(inclusion="inclusive"), "the first level has none"),
            (self.base(inclusion="strict"), "unknown inclusion policy"),
        ]
        for config, fragment in cases:
            with self.subTest(fragment=fragment), self.assertRaises(ConfigError) as ctx:
                parse_config(config)
            self.assertIn(fragment, str(ctx.exception))
            self.assertIn("L1", str(ctx.exception))

    def test_the_hierarchy_rejects_an_unknown_policy_directly(self) -> None:
        with self.assertRaises(ValueError) as ctx:
            Level(one_set("L1", 2), 4, write_policy="write-around")
        self.assertIn("unknown write policy", str(ctx.exception))


if __name__ == "__main__":
    unittest.main(verbosity=2)
