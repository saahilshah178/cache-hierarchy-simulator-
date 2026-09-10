"""Closed-form checks: every reported number derived by hand first.

Each case works the answer out from the geometry and the access pattern,
then asserts that the simulator produces exactly that. The derivations are
in the code and the comments, so a number that changes has to be argued
with rather than re-recorded, and the figures the documentation quotes are
the ones pinned here.

The workload generators are used directly; no trace files are involved.
"""

from __future__ import annotations

import unittest
from collections.abc import Iterable

from cachesim.cache import Cache
from cachesim.config import DEFAULT_CONFIG, parse_config
from cachesim.hierarchy import Hierarchy
from cachesim.invariants import check_invariants
from cachesim.report import format_report
from cachesim.stats import HierarchyStats
from cachesim.workloads import BLOCK, WORD, Access, conflict_streams, sequential

SPEC = parse_config(DEFAULT_CONFIG)
L1_HIT, L2_HIT, L3_HIT = 4, 12, 40
DRAM = 100
#: Cycles for an access that reaches each level, from the constant-latency
#: model: every level probed on the way down is charged once.
L1_COST = L1_HIT
L2_COST = L1_HIT + L2_HIT
L3_COST = L1_HIT + L2_HIT + L3_HIT
DRAM_COST = L3_COST + DRAM  # 156


def simulate(workload: Iterable[Access]) -> Hierarchy:
    """Run one workload generator through the default hierarchy."""
    h = Hierarchy.from_spec(SPEC)
    for addr, op in workload:
        h.access(addr, op == "W")
    return h


class TestSequentialClosedForm(unittest.TestCase):
    """``sequential()`` through DEFAULT_CONFIG, every figure derived.

    The workload scans a 256 KB buffer twice, one 8-byte word at a time,
    with every tenth access a store. The default hierarchy is
    32 KB/4-way L1 (4 cyc), 256 KB/8-way L2 (12 cyc), 2 MB/16-way L3
    (40 cyc), 64-byte blocks, 100-cycle DRAM.
    """

    BUFFER = 256 * 1024
    PASSES = 2
    BASE = 0x0010_0000  # where sequential() puts the buffer

    h: Hierarchy
    stats: HierarchyStats
    words_per_pass: int
    accesses: int
    buffer_blocks: int
    words_per_block: int
    l1_blocks: int
    l2_blocks: int
    l3_blocks: int

    @classmethod
    def setUpClass(cls) -> None:
        cls.h = simulate(sequential(buffer_bytes=cls.BUFFER, passes=cls.PASSES))
        cls.stats = cls.h.stats()

        cls.words_per_pass = cls.BUFFER // WORD  # 32,768
        cls.accesses = cls.PASSES * cls.words_per_pass  # 65,536
        cls.buffer_blocks = cls.BUFFER // BLOCK  # 4,096
        cls.words_per_block = BLOCK // WORD  # 8
        cls.l1_blocks = 32 * 1024 // BLOCK  # 512
        cls.l2_blocks = 256 * 1024 // BLOCK  # 4,096
        cls.l3_blocks = 2 * 1024 * 1024 // BLOCK  # 32,768

    def test_the_trace_is_the_size_the_derivation_assumes(self) -> None:
        self.assertEqual((self.words_per_pass, self.accesses), (32768, 65536))
        self.assertEqual((self.buffer_blocks, self.l1_blocks), (4096, 512))
        self.assertEqual(self.stats.accesses, 65536)

    def test_read_and_write_counts(self) -> None:
        """Every tenth access is a store: indices 9, 19, ... 65,529."""
        writes = sum(1 for i in range(self.accesses) if i % 10 == 9)
        self.assertEqual(writes, 6553)
        self.assertEqual(self.stats.writes, 6553)
        self.assertEqual(self.stats.reads, 65536 - 6553)
        self.assertEqual(self.stats.reads, 58983)

    def test_l1_misses_once_per_block_per_pass(self) -> None:
        """The buffer is 4,096 blocks and L1 holds 512, so a sequential scan
        has evicted a block long before the next pass reaches it: both
        passes miss on all 4,096 blocks and hit on the other 7 words of
        each block."""
        l1 = self.stats.levels[0]
        self.assertGreater(self.buffer_blocks, self.l1_blocks)
        expected_misses = self.PASSES * self.buffer_blocks
        expected_hits = self.accesses - expected_misses
        self.assertEqual((expected_misses, expected_hits), (8192, 57344))
        self.assertEqual((l1.misses, l1.hits), (8192, 57344))
        self.assertEqual(l1.accesses, 65536)
        self.assertEqual(l1.local_miss_rate, 0.125)
        self.assertEqual(l1.global_miss_rate, 0.125)

    def test_no_store_ever_misses_l1(self) -> None:
        """A reference misses only on the first word of a block, i.e. at an
        access index that is a multiple of 8 (even); stores land on indices
        congruent to 9 modulo 10 (odd). The two never coincide, so every
        store is a write hit and every miss is a read miss."""
        self.assertFalse(any(i % 8 == 0 for i in range(self.accesses) if i % 10 == 9))
        l1 = self.stats.levels[0]
        self.assertEqual((l1.write_misses, l1.write_hits), (0, 6553))
        self.assertEqual((l1.read_misses, l1.read_hits), (8192, 50791))

    def test_l1_fills_and_evictions(self) -> None:
        """One fill per miss; the first 512 fills go into empty ways."""
        l1 = self.stats.levels[0]
        self.assertEqual(l1.fills, 8192)
        self.assertEqual(l1.evictions, 8192 - self.l1_blocks)
        self.assertEqual(l1.evictions, 7680)

    def test_l1_writebacks(self) -> None:
        """6,144 dirty lines leave L1, and 409 are still dirty at the end.

        Store positions repeat with period lcm(8 words per block, 10
        accesses per store) = 40 words = 5 blocks, and the four stores in
        each window land in four different blocks, so four blocks in five
        are dirtied. Pass 1 (access indices 0..32,767) dirties 3,276
        blocks; pass 2 starts at index 32,768, which is 8 modulo 40, so its
        stores land one block earlier in each window and it dirties 3,277,
        including the last block of the buffer. No block is stored to twice
        while it stays resident, so the 6,553 stores dirty 6,553 lines. The
        512 lines L1 still holds at the end are the buffer's last 512
        blocks, 409 of them dirty; the other 6,144 were written back.
        """
        pass1 = self.dirtied_blocks(0)
        pass2 = self.dirtied_blocks(self.words_per_pass)
        self.assertEqual((len(pass1), len(pass2)), (3276, 3277))
        self.assertEqual(len(pass1) + len(pass2), self.stats.writes)

        resident = set(range(self.buffer_blocks - self.l1_blocks, self.buffer_blocks))
        still_dirty = pass2 & resident
        self.assertEqual(len(still_dirty), 409)
        self.assertEqual(self.stats.writes - len(still_dirty), 6144)

        l1 = self.stats.levels[0]
        self.assertEqual(l1.writebacks, 6144)
        self.assertEqual(sum(1 for *_, dirty in self.h.levels[0].cache.lines() if dirty), 409)

    def dirtied_blocks(self, start: int) -> set[int]:
        """Blocks of the buffer that receive a store during one pass."""
        return {
            (i - start) // self.words_per_block
            for i in range(start, start + self.words_per_pass)
            if i % 10 == 9
        }

    def test_l2_holds_the_whole_buffer_exactly(self) -> None:
        """L2 is 4,096 blocks and the buffer is 4,096 blocks. It has 512
        sets of 8 ways, the buffer starts at block 16,384 (a multiple of
        512), and 4,096 consecutive blocks put exactly 8 in every set, so
        the fit is exact: pass 1 misses on all of them, pass 2 hits on all
        of them, and nothing is ever evicted."""
        self.assertEqual(self.l2_blocks, self.buffer_blocks)
        self.assertEqual((self.BASE // BLOCK) % 512, 0)
        self.assertEqual(self.buffer_blocks // 512, 8)  # ways per set, exactly

        l2 = self.stats.levels[1]
        self.assertEqual(l2.accesses, 8192)  # L1's misses
        self.assertEqual((l2.misses, l2.hits), (4096, 4096))
        self.assertEqual(l2.local_miss_rate, 0.5)
        self.assertEqual(l2.global_miss_rate, 0.0625)
        self.assertEqual((l2.fills, l2.evictions), (4096, 0))

    def test_l2_receives_every_writeback_without_allocating(self) -> None:
        """L2 never loses a buffer block, so each write-back only marks the
        copy already there."""
        l2 = self.stats.levels[1]
        self.assertEqual(l2.writebacks_received, 6144)
        self.assertEqual(l2.writeback_allocations, 0)
        self.assertEqual(l2.writebacks, 0)

    def test_l3_sees_each_block_once_and_never_hits(self) -> None:
        """L3 only ever sees L2's 4,096 compulsory misses, and holds 32,768
        blocks, so it fills without evicting and hits nothing."""
        l3 = self.stats.levels[2]
        self.assertEqual((l3.accesses, l3.misses, l3.hits), (4096, 4096, 0))
        self.assertEqual(l3.local_miss_rate, 1.0)
        self.assertEqual((l3.fills, l3.evictions, l3.writebacks), (4096, 0, 0))
        self.assertLess(4096, self.l3_blocks)

    def test_dram_traffic(self) -> None:
        """One demand fetch per distinct block; nothing dirty ever leaves
        L3, so no DRAM writes happen during the run."""
        self.assertEqual(self.stats.dram_reads, self.buffer_blocks)
        self.assertEqual((self.stats.dram_reads, self.stats.dram_writes), (4096, 0))

    def test_three_c_classification(self) -> None:
        """Pass 1 is entirely compulsory. Pass 2's L1 misses are capacity,
        not conflict: a fully-associative 512-block LRU cache would have
        evicted those blocks too. Below L1 every miss is a first touch."""
        l1, l2, l3 = self.stats.levels
        assert l1.three_c is not None and l2.three_c is not None and l3.three_c is not None
        self.assertEqual((l1.three_c.compulsory, l1.three_c.capacity), (4096, 4096))
        self.assertEqual(l1.three_c.conflict, 0)
        self.assertEqual(l1.three_c.shadow_misses, 8192)
        self.assertEqual(l1.three_c.anti_conflict_hits, 0)
        self.assertEqual((l1.three_c.capacity_aggregate, l1.three_c.conflict_aggregate), (4096, 0))
        for level in (l2.three_c, l3.three_c):
            self.assertEqual((level.compulsory, level.capacity, level.conflict), (4096, 0, 0))

    def test_total_cycles_and_amat(self) -> None:
        """57,344 L1 hits at 4 cycles, 4,096 L2 hits at 4+12, and 4,096 DRAM
        accesses at 4+12+40+100."""
        expected = 57344 * L1_COST + 4096 * L2_COST + 4096 * DRAM_COST
        self.assertEqual((229376 + 65536 + 638976), 933888)
        self.assertEqual(expected, 933888)
        self.assertEqual(self.stats.total_cycles, 933888)
        self.assertEqual(self.stats.measured_amat, 933888 / 65536)
        self.assertEqual(self.stats.measured_amat, 14.25)
        self.assertAlmostEqual(self.stats.amat, 14.25, places=12)

    def test_flush_writes_the_whole_buffer_back(self) -> None:
        """L2 holds a dirty copy of every block that was ever stored to.
        Flushing pushes L1's 409 remaining dirty lines down, after which
        all 4,096 buffer blocks are dirty in L2 and reach DRAM."""
        h = simulate(sequential(buffer_bytes=self.BUFFER, passes=self.PASSES))
        self.assertEqual(sum(1 for *_, d in h.levels[1].cache.lines() if d), 3993)
        self.assertEqual(h.flush(), 4096)
        self.assertEqual(h.dram_writes, 4096)
        self.assertEqual(h.levels[0].cache.writebacks, 6553)  # 6,144 + the 409 flushed
        self.assertEqual(check_invariants(h, after_flush=True), [])

    def test_the_report_prints_the_derived_numbers(self) -> None:
        text = format_report(self.stats)
        for line in (
            "Total accesses :       65,536   (reads 58,983 / writes 6,553)",
            "  hits        :       57,344   (reads 50,791 / writes 6,553)",
            "  misses      :        8,192   (reads 8,192 / writes 0)",
            "  local miss rate  :  12.50%",
            "  evictions   :        7,680   (writebacks of dirty blocks: 6,144)",
            "  writebacks received :  6,144   (from L1; 0 allocated a line)",
            "AMAT (analytic formula) :   14.250 cycles",
            "AMAT (measured)         :   14.250 cycles   (933,888 cycles / 65,536 accesses)",
        ):
            self.assertIn(line, text)

    def test_the_run_satisfies_every_invariant(self) -> None:
        self.assertEqual(check_invariants(self.h), [])


class TestConflictClosedForm(unittest.TestCase):
    """``conflict_streams()`` through DEFAULT_CONFIG.

    Four arrays 1 MB apart are read in lockstep, one 8-byte word from each
    per step, 16,384 words per array.
    """

    STREAMS = 4
    WORDS_PER_STREAM = 16_384
    ALIGN = 0x0010_0000  # 1 MB = 16,384 blocks between stream bases

    h: Hierarchy
    stats: HierarchyStats
    accesses: int
    blocks_per_stream: int
    distinct_blocks: int

    @classmethod
    def setUpClass(cls) -> None:
        cls.h = simulate(
            conflict_streams(
                streams=cls.STREAMS, words_per_stream=cls.WORDS_PER_STREAM, align=cls.ALIGN
            )
        )
        cls.stats = cls.h.stats()
        cls.accesses = cls.STREAMS * cls.WORDS_PER_STREAM  # 65,536
        cls.blocks_per_stream = cls.WORDS_PER_STREAM // (BLOCK // WORD)  # 2,048
        cls.distinct_blocks = cls.STREAMS * cls.blocks_per_stream  # 8,192

    def test_the_streams_collide_in_every_level(self) -> None:
        """The bases are 16,384 blocks apart, and 16,384 is a multiple of
        each level's set count, so at every step the four streams' blocks
        land in one set at every level."""
        stride_blocks = self.ALIGN // BLOCK
        self.assertEqual(stride_blocks, 16384)
        for level in self.stats.levels:
            self.assertEqual(stride_blocks % level.num_sets, 0, level.name)
        self.assertEqual([lv.num_sets for lv in self.stats.levels], [128, 512, 2048])

    def test_l1_misses_are_one_in_eight_per_stream(self) -> None:
        """L1 is 4-way and exactly four blocks are hot, so the set holds all
        of them: each block misses once and then serves its other seven
        words. 8,192 distinct blocks out of 65,536 accesses is 12.5%."""
        self.assertEqual(self.stats.accesses, 65536)
        self.assertEqual(self.stats.reads, 65536)
        self.assertEqual(self.stats.writes, 0)
        l1 = self.stats.levels[0]
        self.assertGreaterEqual(l1.associativity, self.STREAMS)
        self.assertEqual(l1.misses, self.distinct_blocks)
        self.assertEqual((l1.misses, l1.hits), (8192, 57344))
        self.assertEqual(l1.local_miss_rate, 0.125)
        self.assertEqual(l1.evictions, 8192 - 32 * 1024 // BLOCK)
        self.assertEqual(l1.evictions, 7680)

    def test_lower_levels_never_hit(self) -> None:
        """The streams never revisit a block, so each of the 8,192 blocks
        reaches L2 and L3 exactly once: every reference below L1 is a
        compulsory miss. L2 holds 4,096 blocks and so evicts half of them;
        L3 holds 32,768 and evicts none."""
        l2, l3 = self.stats.levels[1], self.stats.levels[2]
        for level in (l2, l3):
            self.assertEqual((level.accesses, level.misses, level.hits), (8192, 8192, 0))
            self.assertEqual(level.local_miss_rate, 1.0)
            assert level.three_c is not None
            self.assertEqual(level.three_c.compulsory, 8192)
        self.assertEqual(l2.evictions, 8192 - 256 * 1024 // BLOCK)
        self.assertEqual((l2.evictions, l3.evictions), (4096, 0))

    def test_every_miss_is_compulsory_at_every_level(self) -> None:
        for level in self.stats.levels:
            assert level.three_c is not None
            self.assertEqual(level.three_c.compulsory, 8192)
            self.assertEqual((level.three_c.capacity, level.three_c.conflict), (0, 0))

    def test_dram_traffic_and_cycles(self) -> None:
        """57,344 L1 hits at 4 cycles and 8,192 DRAM accesses at 156."""
        self.assertEqual(self.stats.dram_reads, self.distinct_blocks)
        self.assertEqual((self.stats.dram_reads, self.stats.dram_writes), (8192, 0))
        expected = 57344 * L1_COST + 8192 * DRAM_COST
        self.assertEqual(expected, 229376 + 1277952)
        self.assertEqual(expected, 1507328)
        self.assertEqual(self.stats.total_cycles, 1507328)
        self.assertEqual(self.stats.measured_amat, 1507328 / 65536)
        self.assertEqual(self.stats.measured_amat, 23.0)
        self.assertAlmostEqual(self.stats.amat, 23.0, places=12)

    def test_the_run_satisfies_every_invariant(self) -> None:
        self.assertEqual(check_invariants(self.h), [])


class TestCyclicWorkingSet(unittest.TestCase):
    """A cycle one block longer than the cache misses on every reference.

    LRU evicts the block that was used longest ago, which in a cyclic scan
    is precisely the block needed next, so nothing is ever reused. This is
    the standard demonstration that LRU is not optimal (Belady's OPT would
    keep all but one block and miss once per lap).
    """

    LAPS = 20

    def cycle(self, cache: Cache, blocks: int, laps: int | None = None) -> Cache:
        for _ in range(laps if laps is not None else self.LAPS):
            for block in range(blocks):
                cache.access(block * cache.block_size)
        return cache

    def fully_associative(self, capacity: int, policy: str = "lru") -> Cache:
        return Cache("FA", capacity * BLOCK, BLOCK, capacity, policy=policy, rng_seed=1)

    def test_capacity_plus_one_blocks_never_hit(self) -> None:
        for capacity in (1, 2, 4, 8, 16, 64):
            with self.subTest(capacity=capacity):
                c = self.cycle(self.fully_associative(capacity), capacity + 1)
                expected = self.LAPS * (capacity + 1)
                self.assertEqual(c.accesses, expected)
                self.assertEqual(c.misses, expected)
                self.assertEqual((c.hits, c.miss_rate), (0, 1.0))
                # All but the first lap's misses are capacity misses.
                self.assertEqual(c.compulsory_misses, capacity + 1)
                self.assertEqual(c.capacity_misses, expected - (capacity + 1))
                self.assertEqual(c.conflict_misses, 0)

    def test_a_cycle_that_fits_misses_only_once_per_block(self) -> None:
        for capacity in (1, 2, 4, 8, 16, 64):
            with self.subTest(capacity=capacity):
                c = self.cycle(self.fully_associative(capacity), capacity)
                self.assertEqual(c.misses, capacity)
                self.assertEqual(c.hits, self.LAPS * capacity - capacity)
                self.assertEqual(c.compulsory_misses, capacity)

    def test_fifo_thrashes_identically(self) -> None:
        """In a pure cycle nothing is ever hit, so LRU and FIFO see the same
        events and make the same choices."""
        for capacity in (2, 8):
            with self.subTest(capacity=capacity):
                c = self.cycle(self.fully_associative(capacity, "fifo"), capacity + 1)
                self.assertEqual(c.miss_rate, 1.0)

    def test_random_replacement_beats_lru_on_this_pattern(self) -> None:
        """Random replacement often keeps the block that is needed next, so
        the case that is pathological for LRU is ordinary for it: 416 misses
        out of 1,800 references, a 23.1% miss rate against LRU's 100%.

        The exact figure depends on the seeded RNG, so it also pins that the
        policy draws once per eviction and never anywhere else.
        """
        capacity = 8
        c = self.cycle(self.fully_associative(capacity, "random"), capacity + 1, laps=200)
        self.assertEqual(c.accesses, 200 * 9)
        self.assertEqual(c.misses, 416)
        self.assertAlmostEqual(c.miss_rate, 416 / 1800, places=12)
        self.assertLess(c.miss_rate, 0.25)

    def test_set_associative_thrashing_needs_only_one_hot_set(self) -> None:
        """Associativity+1 blocks that map to the same set thrash that set
        while the rest of the cache stays empty."""
        ways, sets = 4, 8
        c = Cache("SA", ways * sets * BLOCK, BLOCK, ways)
        self.assertEqual(c.num_sets, sets)
        for _ in range(self.LAPS):
            for i in range(ways + 1):
                c.access(i * sets * BLOCK)  # blocks 0, 8, 16, 24, 32: all set 0
        self.assertEqual(c.miss_rate, 1.0)
        self.assertEqual(len(list(c.lines())), ways)  # only one set is populated


if __name__ == "__main__":
    unittest.main(verbosity=2)
