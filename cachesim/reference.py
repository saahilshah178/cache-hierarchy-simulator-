"""An independent, deliberately naive re-implementation of the simulator.

This module is a **test oracle**. It exists so that the optimised model in
``cachesim.cache`` and ``cachesim.hierarchy`` can be checked against a
second implementation written from the documented semantics rather than
from the same code (differential testing; W. M. McKeeman, "Differential
Testing for Software", *Digital Technical Journal* 10(1), 1998).

Nothing here is intended for use in a simulation run:

* It shares no algorithmic code with ``cachesim.cache`` or
  ``cachesim.hierarchy``. The only import from the package is the
  ``HierarchySpec`` parameter dataclass, which carries numbers, not
  behaviour.
* Every lookup is a linear scan over a list of line records, every victim
  choice materialises and sorts a list, and the shadow cache is a list
  scanned linearly, so a reference costs O(num_blocks) instead of
  O(associativity). Measured at 3-5x the real model's run time on short
  traces, and the gap widens with the shadow cache's size.
* It has no CLI, no report, and no statistics snapshot.

What it does model, matching the documented semantics of the real code:

* A set-associative cache as ``num_sets`` lists of ``associativity`` slots,
  each slot either empty or a ``RefLine`` record of
  ``(block, dirty, last_used, filled_at)``. A block maps to set
  ``block % num_sets``; ``allocate`` fills the lowest-numbered empty slot
  before it asks the policy for a victim, so slot occupancy is identical to
  the real model's way occupancy.
* LRU (victim = smallest ``last_used``), FIFO (victim = smallest
  ``filled_at``) and random (victim = ``rng.randrange(associativity)``).
  The random policy draws from a ``random.Random(rng_seed)`` created once
  per cache and consumes exactly one value per eviction, which is what
  ``cachesim.policies.RandomPolicy`` does, so identically seeded runs make
  identical replacement decisions.
* The three-C miss classification of Hill and Smith ("Evaluating
  Associativity in CPU Caches", *IEEE Transactions on Computers* 38(12),
  1989), using a plain Python list as the fully-associative LRU shadow
  cache (oldest block first) and a set of every block ever referenced.
* Write-back, write-allocate, non-inclusive non-exclusive multi-level
  behaviour: probe top-down until a level hits, fill bottom-up into every
  level that missed, and propagate a dirty eviction into the level below by
  marking its copy dirty or allocating the line dirty when the copy is
  gone. A dirty line leaving the last level is a DRAM write. Timing charges
  each probed level's hit time plus the memory access time when every level
  missed.

Policy semantics follow J. L. Hennessy and D. A. Patterson, *Computer
Architecture: A Quantitative Approach*, 6th ed., Morgan Kaufmann, 2017,
Chapter 2 and Appendix B.
"""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass
from random import Random
from typing import NamedTuple

from cachesim.config import HierarchySpec

#: Replacement policies the reference model understands.
REF_POLICIES = ("lru", "fifo", "random")


class RefEvicted(NamedTuple):
    """A line removed from a ``RefCache`` by ``allocate`` or ``invalidate``."""

    block: int
    dirty: bool


@dataclass
class RefLine:
    """One resident line: its block, dirty bit, and two event timestamps.

    ``last_used`` is stamped on every hit and on the fill; ``filled_at``
    only on the fill. LRU orders by the former, FIFO by the latter.
    """

    block: int
    dirty: bool
    last_used: int
    filled_at: int


class RefCache:
    """A naive set-associative cache level.

    The constructor takes the same parameters as ``cachesim.cache.Cache``
    and rejects the same degenerate geometries, so a spec that builds one
    builds the other.
    """

    def __init__(
        self,
        name: str,
        size: int,
        block_size: int,
        associativity: int,
        policy: str = "lru",
        track_3c: bool = True,
        rng_seed: int = 0,
    ) -> None:
        if min(size, block_size, associativity) <= 0:
            raise ValueError(f"{name}: geometry must be positive")
        if block_size & (block_size - 1):
            raise ValueError(f"{name}: block_size must be a power of two, got {block_size}")
        if size % (block_size * associativity) != 0:
            raise ValueError(f"{name}: size {size} does not hold a whole number of sets")
        if policy not in REF_POLICIES:
            raise ValueError(f"{name}: unknown replacement policy {policy!r}")

        self.name = name
        self.size = size
        self.block_size = block_size
        self.associativity = associativity
        self.num_sets = size // (block_size * associativity)
        self.num_blocks = size // block_size
        self.policy_name = policy
        self.track_3c = track_3c

        self._sets: list[list[RefLine | None]] = [
            [None] * associativity for _ in range(self.num_sets)
        ]
        self._rng = Random(rng_seed)
        self._clock = 0  # monotonic event counter stamped into RefLine

        self.hits = 0
        self.misses = 0
        self.read_hits = 0
        self.read_misses = 0
        self.write_hits = 0
        self.write_misses = 0
        self.fills = 0
        self.evictions = 0
        self.invalidations = 0
        self.writebacks = 0
        self.writebacks_received = 0
        self.writeback_allocations = 0

        self.compulsory_misses = 0
        self.capacity_misses = 0
        self.conflict_misses = 0
        self.shadow_misses = 0
        self.anti_conflict_hits = 0
        #: Fully-associative LRU shadow cache, least recently used first.
        self._shadow: list[int] = []
        self._seen: set[int] = set()

    # -- address arithmetic --------------------------------------------------

    def block_of(self, addr: int) -> int:
        """The block number containing byte address ``addr``."""
        return addr // self.block_size

    def set_of(self, block: int) -> int:
        """The set index a block maps to."""
        return block % self.num_sets

    def tag_of(self, block: int) -> int:
        """The tag that distinguishes blocks sharing a set."""
        return block // self.num_sets

    # -- primitives ----------------------------------------------------------

    def probe(self, block: int, is_write: bool = False) -> bool:
        """Look ``block`` up, updating statistics and replacement state.

        Returns True on a hit. Never fills; the caller decides what to do
        with a miss.
        """
        for line in self._sets[block % self.num_sets]:
            if line is not None and line.block == block:
                self.hits += 1
                if is_write:
                    self.write_hits += 1
                    line.dirty = True
                else:
                    self.read_hits += 1
                self._clock += 1
                line.last_used = self._clock
                if self.track_3c and not self._reference_shadow(block):
                    self.anti_conflict_hits += 1
                return True

        self.misses += 1
        if is_write:
            self.write_misses += 1
        else:
            self.read_misses += 1
        if self.track_3c:
            self._classify(block)
            self._reference_shadow(block)
        return False

    def allocate(self, block: int, dirty: bool = False) -> RefEvicted | None:
        """Install ``block``, evicting from a full set. Returns the victim.

        A block that is already resident is left in place with its dirty
        bit OR-ed in, and None is returned; that path deliberately does not
        touch either timestamp, so the replacement order is unchanged.
        """
        ways = self._sets[block % self.num_sets]
        free: int | None = None
        for way, line in enumerate(ways):
            if line is not None and line.block == block:
                if dirty:
                    line.dirty = True
                return None
            if line is None and free is None:
                free = way

        evicted: RefEvicted | None = None
        if free is None:
            way = self._choose_victim(ways)
            victim = ways[way]
            if victim is None:  # pragma: no cover - _choose_victim guarantees this
                raise AssertionError("victim slot is empty")
            self.evictions += 1
            evicted = RefEvicted(victim.block, victim.dirty)
            if victim.dirty:
                self.writebacks += 1
        else:
            way = free

        self._clock += 1
        ways[way] = RefLine(block, dirty, last_used=self._clock, filled_at=self._clock)
        self.fills += 1
        return evicted

    def invalidate(self, block: int) -> RefEvicted | None:
        """Remove ``block`` if resident. Returns the removed line, else None."""
        ways = self._sets[block % self.num_sets]
        for way, line in enumerate(ways):
            if line is not None and line.block == block:
                ways[way] = None
                self.invalidations += 1
                if line.dirty:
                    self.writebacks += 1
                return RefEvicted(block, line.dirty)
        return None

    def _choose_victim(self, ways: list[RefLine | None]) -> int:
        """Pick the way to evict from a set in which every way is valid."""
        if self.policy_name == "random":
            return self._rng.randrange(self.associativity)
        keyed: list[tuple[int, int]] = []
        for way, line in enumerate(ways):
            if line is None:
                raise AssertionError("a free way should have been used instead of a victim")
            stamp = line.last_used if self.policy_name == "lru" else line.filled_at
            keyed.append((stamp, way))
        keyed.sort()
        return keyed[0][1]

    # -- inspection ----------------------------------------------------------

    def contains(self, block: int) -> bool:
        """True if ``block`` is currently resident."""
        return any(
            line is not None and line.block == block for line in self._sets[block % self.num_sets]
        )

    def is_dirty(self, block: int) -> bool:
        """True if ``block`` is resident and dirty."""
        for line in self._sets[block % self.num_sets]:
            if line is not None and line.block == block:
                return line.dirty
        return False

    def mark_dirty(self, block: int) -> bool:
        """Set the dirty bit of a resident block. Returns False if absent."""
        for line in self._sets[block % self.num_sets]:
            if line is not None and line.block == block:
                line.dirty = True
                return True
        return False

    def clean(self, block: int) -> bool:
        """Clear the dirty bit of a resident block. Returns False if absent."""
        for line in self._sets[block % self.num_sets]:
            if line is not None and line.block == block:
                line.dirty = False
                return True
        return False

    def lines(self) -> Iterator[tuple[int, int, int, bool]]:
        """Yield ``(set_idx, way, block, dirty)`` for every valid line."""
        for set_idx, ways in enumerate(self._sets):
            for way, line in enumerate(ways):
                if line is not None:
                    yield set_idx, way, line.block, line.dirty

    # -- single-level convenience --------------------------------------------

    def access(self, addr: int, is_write: bool = False) -> bool:
        """Probe, then fill on a miss. Returns True on a hit."""
        if addr < 0:
            raise ValueError(f"{self.name}: address must be non-negative, got {addr}")
        block = addr // self.block_size
        if self.probe(block, is_write):
            return True
        self.allocate(block, dirty=is_write)
        return False

    # -- three-C classification ----------------------------------------------

    def _classify(self, block: int) -> None:
        """Label the miss just counted, before the shadow cache is updated."""
        if block not in self._seen:
            self.compulsory_misses += 1
        elif block in self._shadow:
            self.conflict_misses += 1
        else:
            self.capacity_misses += 1

    def _reference_shadow(self, block: int) -> bool:
        """Feed one reference to the fully-associative LRU shadow cache.

        Returns True if the shadow cache held the block. The shadow is a
        plain list ordered least-recently-used first: a hit removes and
        re-appends the block, a miss appends it and drops element 0 once
        the list exceeds the cache's block count.
        """
        self._seen.add(block)
        if block in self._shadow:
            self._shadow.remove(block)
            self._shadow.append(block)
            return True
        self.shadow_misses += 1
        self._shadow.append(block)
        if len(self._shadow) > self.num_blocks:
            del self._shadow[0]
        return False

    def reset_stats(self) -> None:
        """Zero every counter, keeping contents, timestamps, shadow, and seen set."""
        self.hits = self.misses = 0
        self.read_hits = self.read_misses = 0
        self.write_hits = self.write_misses = 0
        self.fills = self.evictions = self.invalidations = 0
        self.writebacks = self.writebacks_received = self.writeback_allocations = 0
        self.compulsory_misses = self.capacity_misses = self.conflict_misses = 0
        self.shadow_misses = self.anti_conflict_hits = 0

    # -- derived stats -------------------------------------------------------

    @property
    def accesses(self) -> int:
        return self.hits + self.misses

    @property
    def capacity_misses_aggregate(self) -> int:
        """Hill-Smith capacity misses: non-compulsory shadow misses."""
        return self.shadow_misses - self.compulsory_misses

    @property
    def conflict_misses_aggregate(self) -> int:
        """Hill-Smith conflict misses: misses beyond the shadow cache's."""
        return self.misses - self.shadow_misses

    @property
    def miss_rate(self) -> float:
        """Local miss rate: misses / accesses that reached this cache."""
        return self.misses / self.accesses if self.accesses else 0.0


class RefLevel:
    """A ``RefCache`` and the cycles it costs to probe it."""

    def __init__(self, cache: RefCache, hit_time: int) -> None:
        self.cache = cache
        self.hit_time = hit_time


class RefHierarchy:
    """A naive chain of ``RefLevel`` backed by main memory."""

    def __init__(self, levels: list[RefLevel], memory_access_time: int) -> None:
        if not levels:
            raise ValueError("a hierarchy needs at least one cache level")
        if len({level.cache.block_size for level in levels}) != 1:
            raise ValueError("all levels must share one block size")
        self.levels = levels
        self.block_size = levels[0].cache.block_size
        self.memory_access_time = memory_access_time

        self.accesses = 0
        self.reads = 0
        self.writes = 0
        self.dram_reads = 0
        self.dram_writes = 0
        self.total_time = 0

    @classmethod
    def from_spec(cls, spec: HierarchySpec) -> RefHierarchy:
        """Build a reference hierarchy from the same spec the real one uses."""
        levels = [
            RefLevel(
                RefCache(
                    name=level.name,
                    size=level.size,
                    block_size=level.block_size,
                    associativity=level.associativity,
                    policy=level.policy,
                    track_3c=level.track_3c,
                    rng_seed=level.rng_seed,
                ),
                level.hit_time,
            )
            for level in spec.levels
        ]
        return cls(levels, spec.memory_access_time)

    def access(self, addr: int, is_write: bool = False) -> int:
        """Simulate one access; returns the cycles it took.

        Probe top-down until a level hits or DRAM is reached, then fill the
        block into every level that missed, bottom-up.
        """
        if addr < 0:
            raise ValueError(f"address must be non-negative, got {addr}")
        self.accesses += 1
        if is_write:
            self.writes += 1
        else:
            self.reads += 1

        block = addr // self.block_size
        time = 0
        hit_level = len(self.levels)
        for i, level in enumerate(self.levels):
            time += level.hit_time
            # Only L1 sees the store: the levels below supply the line and
            # therefore observe a read.
            if level.cache.probe(block, is_write and i == 0):
                hit_level = i
                break
        if hit_level == len(self.levels):
            time += self.memory_access_time
            self.dram_reads += 1

        for i in range(hit_level - 1, -1, -1):
            evicted = self.levels[i].cache.allocate(block, dirty=is_write and i == 0)
            if evicted is not None and evicted.dirty:
                self._write_back(i + 1, evicted.block)

        self.total_time += time
        return time

    def _write_back(self, level_index: int, block: int) -> None:
        """Deliver a dirty block to ``levels[level_index]``, or to DRAM."""
        if level_index >= len(self.levels):
            self.dram_writes += 1
            return
        cache = self.levels[level_index].cache
        cache.writebacks_received += 1
        if not cache.mark_dirty(block):
            cache.writeback_allocations += 1
            evicted = cache.allocate(block, dirty=True)
            if evicted is not None and evicted.dirty:
                self._write_back(level_index + 1, evicted.block)

    def flush(self) -> int:
        """Write every dirty line back and clean it, top-down.

        Lines stay resident. Returns the DRAM writes the flush caused.
        """
        before = self.dram_writes
        for i, level in enumerate(self.levels):
            cache = level.cache
            for _, _, block, dirty in list(cache.lines()):
                if dirty:
                    cache.clean(block)
                    cache.writebacks += 1
                    self._write_back(i + 1, block)
        return self.dram_writes - before

    def reset_stats(self) -> None:
        """Zero every counter at every level while keeping cache contents."""
        self.accesses = self.reads = self.writes = 0
        self.dram_reads = self.dram_writes = 0
        self.total_time = 0
        for level in self.levels:
            level.cache.reset_stats()

    @property
    def memory_accesses(self) -> int:
        """Total DRAM traffic: demand reads plus write-backs."""
        return self.dram_reads + self.dram_writes

    def global_miss_rate(self, level_index: int) -> float:
        """Misses at this level divided by all CPU accesses."""
        if self.accesses == 0:
            return 0.0
        return self.levels[level_index].cache.misses / self.accesses

    def amat(self) -> float:
        """Analytic AMAT from the nested miss-rate formula, in cycles."""
        penalty: float = self.memory_access_time
        for level in reversed(self.levels):
            penalty = level.hit_time + level.cache.miss_rate * penalty
        return penalty

    def measured_amat(self) -> float:
        """Total simulated cycles divided by accesses."""
        return self.total_time / self.accesses if self.accesses else 0.0
