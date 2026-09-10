"""Property-based tests: identities that must hold for every input.

Where the known-answer tests pin one scenario to one number and the
differential tests compare two implementations, these state laws the
simulator obeys for *any* geometry and *any* reference stream, and let
hypothesis search for a counterexample (K. Claessen and J. Hughes,
"QuickCheck: A Lightweight Tool for Random Testing of Haskell Programs",
ICFP 2000).

The laws checked here:

* counter algebra -- hits + misses = accesses, reads + writes split both,
  misses never exceed accesses;
* the three-C decomposition -- per-reference labels sum to the misses, the
  Hill-Smith aggregate labels sum to the same total, and the two taxonomies
  differ by exactly the anti-conflict hits;
* compulsory misses = the number of distinct blocks the stream touches, at
  *every* level, because a block's first reference misses all the way down;
* level chaining -- a level sees exactly the misses of the level above it,
  and DRAM sees exactly the misses of the last level;
* the LRU stack property (R. L. Mattson, J. Gecsei, D. R. Slutz and I. L.
  Traiger, "Evaluation techniques for storage hierarchies", *IBM Systems
  Journal* 9(2), 1970): for fully-associative LRU, the contents of a
  smaller cache are always a subset of a larger one's, so miss counts are
  non-increasing in capacity;
* determinism -- the same spec and the same stream give the same counters,
  including under the random replacement policy;
* ``reset_stats`` zeroes the counters and changes nothing else;
* structural invariants of the primitives -- a block has at most one copy
  in a cache, and every valid line is found by ``contains``.

hypothesis is a dev dependency (``pip install 'cachesim[dev]'``); without
it the whole module is skipped.
"""

from __future__ import annotations

import unittest
from typing import Any

try:
    from hypothesis import HealthCheck, given, settings
    from hypothesis import strategies as st
except ImportError as exc:  # pragma: no cover - exercised only without the extra
    raise unittest.SkipTest(
        f"hypothesis is not installed, skipping the property-based tests "
        f"(pip install 'cachesim[dev]'): {exc}"
    ) from exc

from cachesim.cache import Cache
from cachesim.config import CacheSpec, HierarchySpec
from cachesim.hierarchy import Hierarchy

Access = tuple[int, bool]

#: Shared budget: enough examples to be worth running, small enough that the
#: whole module stays under a couple of seconds. Deadlines are disabled
#: because a large geometry can make one example much slower than the rest.
EXAMPLES = settings(
    max_examples=100,
    deadline=None,
    suppress_health_check=[HealthCheck.too_slow, HealthCheck.data_too_large],
)

BLOCK_SIZES = st.sampled_from([8, 16, 32, 64])
POLICIES = st.sampled_from(["lru", "fifo", "random"])


@st.composite
def cache_specs(draw: st.DrawFn, max_ways: int = 8, max_sets: int = 16) -> CacheSpec:
    """One cache level of an arbitrary legal geometry."""
    block_size = draw(BLOCK_SIZES)
    ways = draw(st.integers(min_value=1, max_value=max_ways))
    sets = draw(st.integers(min_value=1, max_value=max_sets))
    return CacheSpec(
        name="L1",
        size=sets * ways * block_size,
        block_size=block_size,
        associativity=ways,
        hit_time=draw(st.integers(min_value=0, max_value=8)),
        policy=draw(POLICIES),
        rng_seed=draw(st.integers(min_value=0, max_value=1000)),
    )


@st.composite
def hierarchy_specs(draw: st.DrawFn, max_levels: int = 3) -> HierarchySpec:
    """One to ``max_levels`` levels sharing a block size, L1 first."""
    block_size = draw(BLOCK_SIZES)
    count = draw(st.integers(min_value=1, max_value=max_levels))
    levels = []
    for i in range(count):
        ways = draw(st.integers(min_value=1, max_value=4))
        sets = draw(st.integers(min_value=1, max_value=8))
        levels.append(
            CacheSpec(
                name=f"L{i + 1}",
                size=sets * ways * block_size,
                block_size=block_size,
                associativity=ways,
                hit_time=4 * (i + 1),
                policy=draw(POLICIES),
                rng_seed=draw(st.integers(min_value=0, max_value=1000)),
            )
        )
    return HierarchySpec(
        levels=tuple(levels),
        memory_access_time=draw(st.integers(min_value=0, max_value=200)),
    )


@st.composite
def streams(draw: st.DrawFn, max_addr: int = 2048, max_size: int = 120) -> list[Access]:
    """An access stream over a small address range, walked one to four times.

    The address range is kept narrow so that blocks are reused rather than
    every reference being compulsory, and repeating the same references is
    what drives capacity and conflict misses: hypothesis on its own favours
    short lists of distinct values, which leave the replacement path barely
    exercised.
    """
    base = draw(
        st.lists(
            st.tuples(st.integers(min_value=0, max_value=max_addr), st.booleans()),
            min_size=1,
            max_size=max_size,
        )
    )
    return base * draw(st.integers(min_value=1, max_value=4))


@st.composite
def write_heavy_streams(draw: st.DrawFn, max_addr: int = 2048) -> list[Access]:
    """A stream in which most references are stores, walked repeatedly.

    Only stores make lines dirty, and only a dirty line evicted from a
    level that the level below has already dropped reaches the write-back
    allocation path, so an evenly split stream rarely gets there.
    """
    base = draw(
        st.lists(
            st.integers(min_value=0, max_value=max_addr),
            min_size=40,  # short streams never reach the allocation path
            max_size=100,
        )
    )
    writes = draw(st.lists(st.booleans(), min_size=1, max_size=4))
    return [(addr, writes[i % len(writes)] or i % 4 != 0) for i, addr in enumerate(base)] * draw(
        st.integers(min_value=1, max_value=4)
    )


def run(spec: HierarchySpec, stream: list[Access]) -> Hierarchy:
    h = Hierarchy.from_spec(spec)
    for addr, is_write in stream:
        h.access(addr, is_write)
    return h


def counters(h: Hierarchy) -> list[Any]:
    """Every counter of a hierarchy, flattened, for equality comparisons."""
    out: list[Any] = [h.accesses, h.reads, h.writes, h.dram_reads, h.dram_writes, h.total_time]
    for level in h.levels:
        c = level.cache
        out.extend(
            [
                c.hits,
                c.misses,
                c.read_hits,
                c.read_misses,
                c.write_hits,
                c.write_misses,
                c.fills,
                c.evictions,
                c.invalidations,
                c.writebacks,
                c.writebacks_received,
                c.writeback_allocations,
                c.compulsory_misses,
                c.capacity_misses,
                c.conflict_misses,
                c.shadow_misses,
                c.anti_conflict_hits,
            ]
        )
    return out


class TestCounterAlgebra(unittest.TestCase):
    @EXAMPLES
    @given(spec=hierarchy_specs(), stream=streams())
    def test_hits_and_misses_partition_the_accesses(
        self, spec: HierarchySpec, stream: list[Access]
    ) -> None:
        h = run(spec, stream)
        self.assertEqual(h.accesses, len(stream))
        self.assertEqual(h.reads + h.writes, h.accesses)
        for level in h.levels:
            c = level.cache
            self.assertEqual(c.hits + c.misses, c.accesses)
            self.assertEqual(c.read_hits + c.write_hits, c.hits)
            self.assertEqual(c.read_misses + c.write_misses, c.misses)
            self.assertLessEqual(c.misses, c.accesses)
            self.assertGreaterEqual(c.misses, 0)
            self.assertEqual(c.fills, c.misses + c.writeback_allocations)
            self.assertLessEqual(c.evictions, c.fills)

    @EXAMPLES
    @given(spec=hierarchy_specs(), stream=streams())
    def test_levels_see_exactly_the_misses_of_the_level_above(
        self, spec: HierarchySpec, stream: list[Access]
    ) -> None:
        """Non-inclusive, non-exclusive with no prefetching: nothing but a
        miss can make a level below do work."""
        h = run(spec, stream)
        self.assertEqual(h.levels[0].cache.accesses, h.accesses)
        for upper, lower in zip(h.levels, h.levels[1:], strict=False):
            self.assertEqual(lower.cache.accesses, upper.cache.misses)
        self.assertEqual(h.dram_reads, h.levels[-1].cache.misses)

    @EXAMPLES
    @given(spec=hierarchy_specs(), stream=streams())
    def test_total_cycles_equal_the_probe_and_memory_charges(
        self, spec: HierarchySpec, stream: list[Access]
    ) -> None:
        h = run(spec, stream)
        expected = sum(lv.hit_time * lv.cache.accesses for lv in h.levels)
        expected += h.memory_access_time * h.dram_reads
        self.assertEqual(h.total_time, expected)
        if h.accesses:
            self.assertAlmostEqual(h.amat(), h.measured_amat(), places=9)


class TestWriteBackConservation(unittest.TestCase):
    """Dirty data is conserved: it moves down the hierarchy, never vanishes.

    The specs here deliberately give each level fewer ways and no more
    capacity than the one above it, so lower levels drop lines the level
    above still holds dirty. That is the only way a write-back has to
    allocate rather than mark, and an ordinary widening hierarchy almost
    never reaches it.
    """

    SHRINKING = HierarchySpec(
        levels=(
            CacheSpec("L1", size=1024, block_size=64, associativity=4, hit_time=4),
            CacheSpec("L2", size=1024, block_size=64, associativity=1, hit_time=12),
            CacheSpec("L3", size=2048, block_size=64, associativity=2, hit_time=40),
        ),
        memory_access_time=100,
    )

    @EXAMPLES
    @given(stream=write_heavy_streams())
    def test_each_level_receives_exactly_what_the_level_above_wrote(
        self, stream: list[Access]
    ) -> None:
        h = run(self.SHRINKING, stream)
        for upper, lower in zip(h.levels, h.levels[1:], strict=False):
            self.assertEqual(lower.cache.writebacks_received, upper.cache.writebacks)
        self.assertEqual(h.dram_writes, h.levels[-1].cache.writebacks)
        for level in h.levels:
            c = level.cache
            self.assertEqual(c.fills, c.misses + c.writeback_allocations)
            self.assertLessEqual(c.writeback_allocations, c.writebacks_received)

    @EXAMPLES
    @given(stream=write_heavy_streams())
    def test_flush_preserves_the_conservation_law(self, stream: list[Access]) -> None:
        h = run(self.SHRINKING, stream)
        h.flush()
        for upper, lower in zip(h.levels, h.levels[1:], strict=False):
            self.assertEqual(lower.cache.writebacks_received, upper.cache.writebacks)
        self.assertEqual(h.dram_writes, h.levels[-1].cache.writebacks)
        for level in h.levels:
            self.assertFalse(any(dirty for *_, dirty in level.cache.lines()))


class TestThreeCDecomposition(unittest.TestCase):
    @EXAMPLES
    @given(spec=hierarchy_specs(), stream=streams())
    def test_both_taxonomies_sum_to_the_misses(
        self, spec: HierarchySpec, stream: list[Access]
    ) -> None:
        h = run(spec, stream)
        for level in h.levels:
            c = level.cache
            self.assertEqual(c.compulsory_misses + c.capacity_misses + c.conflict_misses, c.misses)
            self.assertEqual(
                c.compulsory_misses + c.capacity_misses_aggregate + c.conflict_misses_aggregate,
                c.misses,
            )

    @EXAMPLES
    @given(spec=hierarchy_specs(), stream=streams())
    def test_the_taxonomies_differ_by_the_anti_conflict_hits(
        self, spec: HierarchySpec, stream: list[Access]
    ) -> None:
        """Each reference that hits here but would have missed the
        fully-associative twin moves one miss from aggregate capacity to
        per-reference capacity, and the other way for conflict."""
        h = run(spec, stream)
        for level in h.levels:
            c = level.cache
            self.assertEqual(c.capacity_misses_aggregate, c.capacity_misses + c.anti_conflict_hits)
            self.assertEqual(c.conflict_misses_aggregate, c.conflict_misses - c.anti_conflict_hits)
            self.assertEqual(
                c.shadow_misses,
                c.compulsory_misses + c.capacity_misses + c.anti_conflict_hits,
            )

    @EXAMPLES
    @given(spec=hierarchy_specs(), stream=streams())
    def test_compulsory_misses_count_the_distinct_blocks_at_every_level(
        self, spec: HierarchySpec, stream: list[Access]
    ) -> None:
        """A block's first reference misses every level, so each level
        records exactly one compulsory miss per distinct block."""
        h = run(spec, stream)
        distinct = len({addr // h.block_size for addr, _ in stream})
        for level in h.levels:
            self.assertEqual(level.cache.compulsory_misses, distinct)

    @EXAMPLES
    @given(spec=cache_specs(), stream=streams())
    def test_a_fully_associative_lru_cache_has_no_conflict_misses(
        self, spec: CacheSpec, stream: list[Access]
    ) -> None:
        """With one set and LRU, the shadow cache is the cache, so every
        miss is compulsory or capacity and both aggregates are exact."""
        c = Cache(
            "FA",
            size=spec.size,
            block_size=spec.block_size,
            associativity=spec.size // spec.block_size,
            policy="lru",
        )
        for addr, is_write in stream:
            c.access(addr, is_write)
        self.assertEqual(c.num_sets, 1)
        self.assertEqual(c.conflict_misses, 0)
        self.assertEqual(c.anti_conflict_hits, 0)
        self.assertEqual(c.shadow_misses, c.misses)
        self.assertEqual(c.conflict_misses_aggregate, 0)
        self.assertEqual(c.capacity_misses_aggregate, c.capacity_misses)


class TestLRUStackProperty(unittest.TestCase):
    """LRU is a stack algorithm: a bigger cache holds everything a smaller
    one does, so adding capacity can never add misses (Mattson et al., 1970)."""

    @EXAMPLES
    @given(
        block_size=BLOCK_SIZES,
        small=st.integers(min_value=1, max_value=8),
        extra=st.integers(min_value=0, max_value=8),
        stream=streams(max_addr=512),
    )
    def test_contents_are_nested_at_every_access(
        self, block_size: int, small: int, extra: int, stream: list[Access]
    ) -> None:
        big = small + extra
        a = Cache("small", small * block_size, block_size, small, policy="lru")
        b = Cache("big", big * block_size, block_size, big, policy="lru")
        for addr, _ in stream:
            a.access(addr)
            b.access(addr)
            self.assertLessEqual(
                {block for *_, block, _ in a.lines()},
                {block for *_, block, _ in b.lines()},
            )
        self.assertLessEqual(b.misses, a.misses)

    @EXAMPLES
    @given(block_size=BLOCK_SIZES, stream=streams(max_addr=512, max_size=100))
    def test_misses_are_non_increasing_in_capacity(
        self, block_size: int, stream: list[Access]
    ) -> None:
        misses = []
        for capacity in range(1, 9):
            c = Cache("FA", capacity * block_size, block_size, capacity, policy="lru")
            for addr, _ in stream:
                c.access(addr)
            misses.append(c.misses)
        self.assertEqual(misses, sorted(misses, reverse=True))
        # No capacity can beat the compulsory floor.
        distinct = len({addr // block_size for addr, _ in stream})
        self.assertGreaterEqual(misses[-1], distinct)


class TestDeterminismAndReset(unittest.TestCase):
    @EXAMPLES
    @given(spec=hierarchy_specs(), stream=streams())
    def test_the_same_spec_and_stream_give_the_same_counters(
        self, spec: HierarchySpec, stream: list[Access]
    ) -> None:
        """Including under random replacement: the RNG is seeded per level
        from the spec, so nothing in a run depends on wall time or hashing."""
        self.assertEqual(counters(run(spec, stream)), counters(run(spec, stream)))

    @EXAMPLES
    @given(spec=hierarchy_specs(), stream=streams())
    def test_reset_stats_zeroes_counters_and_keeps_contents(
        self, spec: HierarchySpec, stream: list[Access]
    ) -> None:
        h = run(spec, stream)
        before = [sorted(lv.cache.lines()) for lv in h.levels]
        h.reset_stats()
        self.assertEqual([sorted(lv.cache.lines()) for lv in h.levels], before)
        self.assertEqual(counters(h), [0] * len(counters(h)))

    @EXAMPLES
    @given(spec=hierarchy_specs(), stream=streams())
    def test_flush_cleans_every_line_and_is_idempotent(
        self, spec: HierarchySpec, stream: list[Access]
    ) -> None:
        h = run(spec, stream)
        first_level = sorted((s, w, b) for s, w, b, _ in h.levels[0].cache.lines())
        resident = [sum(1 for _ in lv.cache.lines()) for lv in h.levels]
        h.flush()
        for level in h.levels:
            self.assertFalse(any(dirty for *_, dirty in level.cache.lines()))
        self.assertEqual(h.flush(), 0)
        # Write-backs only travel downwards, so the first level keeps every
        # line in the way it held it; only the dirty bits are cleared.
        self.assertEqual(sorted((s, w, b) for s, w, b, _ in h.levels[0].cache.lines()), first_level)
        # A write-back that has to allocate below either fills a free way or
        # replaces one line with another, so no level can shrink.
        for level, before in zip(h.levels, resident, strict=True):
            self.assertGreaterEqual(sum(1 for _ in level.cache.lines()), before)


class TestPrimitiveInvariants(unittest.TestCase):
    @EXAMPLES
    @given(
        spec=cache_specs(),
        ops=st.lists(
            st.tuples(
                st.sampled_from(["probe", "allocate", "invalidate"]),
                st.integers(min_value=0, max_value=64),
                st.booleans(),
            ),
            min_size=1,
            max_size=120,
        ),
    )
    def test_a_block_has_at_most_one_copy_and_is_always_findable(
        self, spec: CacheSpec, ops: list[tuple[str, int, bool]]
    ) -> None:
        c = Cache(
            spec.name,
            spec.size,
            spec.block_size,
            spec.associativity,
            policy=spec.policy,
            rng_seed=spec.rng_seed,
        )
        for kind, block, flag in ops:
            if kind == "probe":
                if not c.probe(block, flag):
                    c.allocate(block, dirty=flag)
            elif kind == "allocate":
                c.allocate(block, dirty=flag)
            else:
                c.invalidate(block)

            lines = list(c.lines())
            blocks = [b for *_, b, _ in lines]
            self.assertEqual(len(blocks), len(set(blocks)), "a block was duplicated")
            self.assertLessEqual(len(blocks), c.num_blocks)
            for set_idx, _, b, dirty in lines:
                self.assertEqual(set_idx, c.set_of(b), "a line sits in the wrong set")
                self.assertTrue(c.contains(b), "a valid line was not found by contains()")
                self.assertEqual(c.is_dirty(b), dirty)
        # Nothing the primitives did left a phantom block behind.
        resident = {b for *_, b, _ in c.lines()}
        for absent in set(range(65)) - resident:
            self.assertFalse(c.contains(absent))
            self.assertFalse(c.is_dirty(absent))

    @EXAMPLES
    @given(spec=cache_specs(), stream=streams())
    def test_writebacks_never_exceed_dirty_evictions(
        self, spec: CacheSpec, stream: list[Access]
    ) -> None:
        c = Cache(
            spec.name,
            spec.size,
            spec.block_size,
            spec.associativity,
            policy=spec.policy,
            rng_seed=spec.rng_seed,
        )
        writes = 0
        for addr, is_write in stream:
            c.access(addr, is_write)
            writes += is_write
        self.assertLessEqual(c.writebacks, c.evictions + c.invalidations)
        dirty_resident = sum(1 for *_, dirty in c.lines() if dirty)
        # A line can only be dirty because of a store, and each store can
        # account for at most one write-back or one still-dirty line.
        self.assertLessEqual(c.writebacks + dirty_resident, writes)


if __name__ == "__main__":
    unittest.main(verbosity=2)
