"""Chains cache levels into a memory hierarchy.

A hierarchy is an ordered list of cache levels (L1 first) with main memory
(DRAM) at the bottom. Every access works its way down:

    L1?  hit -> done (cost: L1 hit time)
    L2?  hit -> done (cost: L1 + L2 hit times)
    L3?  hit -> done (cost: L1 + L2 + L3 hit times)
    DRAM      -> done (cost: all hit times + memory access time)

Modelling choices:

* Allocate-on-miss at every level: when a level misses, the block is
  installed into that level as part of the same access.
* Write-back, write-allocate at every level by default. A store dirties
  the line in L1 only; the levels below supply the line, so they see a
  read. When a dirty line is evicted from level i it is written into
  level i+1: the copy there is marked dirty, or the line is allocated
  dirty if level i+1 no longer holds it (which may evict another line,
  handled the same way). A dirty line evicted from the last level is
  written to DRAM. Each level may instead be write-through and/or
  no-write-allocate (see below).
* Timing charges each level's hit time once per level probed, plus the
  memory access time if everything missed. Traffic generated *behind* a
  satisfied access -- write-back evictions and the duplicate stores a
  write-through level sends down -- is counted but not charged time, as
  if fully absorbed by write buffers.
* Each level declares an ``inclusion`` policy describing its relation to
  the level above it (see below). The default is NINE, which enforces
  nothing.

The model is functional (exact hit/miss/eviction behaviour) with a serial,
fixed-latency timing model; it does not model overlap of misses (MSHRs),
DRAM banks, or bandwidth.

Write policies
--------------

Two orthogonal keys per level (Hennessy and Patterson, "Computer
Architecture: A Quantitative Approach", 6th ed., 2017, Appendix B.1):

``write_policy``
    ``"write-back"`` (default) marks the line dirty and tells the level
    below only when the line is evicted. ``"write-through"`` applies the
    store here *and* sends a duplicate down as a store, which the level
    below handles under its own policy; a write-through level therefore
    never holds a dirty line, and a store that passes the last level is a
    DRAM write. Those duplicates are untimed.

``write_allocate``
    ``true`` (default) fetches the line on a write miss, so the store is
    applied here. ``false`` leaves this level untouched and passes the
    store to the level below, which sees a store rather than a line fill.
    Loads are unaffected.

A store that no level allocates for reaches DRAM directly, without ever
fetching the block. Unlike the untimed traffic above, a store still
looking for a level to take it is the access itself and is charged each
level's hit time (and the memory access time if it gets that far), exactly
as a load miss is. That is what keeps every level's access count equal to
the miss count of the level above it, and hence keeps analytic and
measured AMAT identical.

Inclusion policies
------------------

``inclusion`` is declared on the LOWER level of a pair and names the
relation it maintains with the level above it (Baer and Wang, "On the
Inclusion Properties for Multi-Level Cache Hierarchies", ISCA 1988):

nine
    Non-inclusive non-exclusive: fills install the block in every level
    that missed and nothing is enforced afterwards, so a lower level may
    drop a block an upper level still holds. This is the default.

inclusive
    The level is a superset of the level above. Whenever it drops a block
    -- replacement by a demand fill, replacement by a write-back
    allocation, or an invalidation -- the block is *back-invalidated* out
    of every level above it. A dirty copy found above is written to the
    level BELOW the evicting level (or to DRAM), because the evicting
    level is losing the block too and cannot hold the data. Back-
    invalidation is what makes an inclusive last level able to filter
    coherence traffic on behalf of the whole hierarchy, and it is also why
    an inclusive level that is not much larger than the level above it
    destroys upper-level hits.

exclusive
    The level holds only blocks the level above does not, as a victim
    cache for it:

    * A demand fetch passing through the level does not fill it; only the
      level above is filled.
    * A hit here moves the block UP: it is invalidated here (carrying its
      dirty bit with it) and allocated in the level above.
    * Every line the level above evicts, clean or dirty, is allocated
      here. This replaces the dirty-only write-back across that boundary.
      The insertion may evict a line here, which is handled by this
      level's own rule: written back if dirty, or passed on to the next
      exclusive level.

    Total capacity is the sum of the two levels rather than the larger of
    them, at the cost of moving every victim across the boundary.

``check_inclusion()`` asserts the invariant for every adjacent pair and is
meant to be called from tests after a random access stream.

AMAT (average memory access time) is the headline metric:

    AMAT = L1_hit_time + L1_miss_rate * L1_miss_penalty

where L1's miss penalty is itself the AMAT of the rest of the hierarchy, so
the formula nests. Both the analytic value and the measured average
(total simulated cycles / accesses) are reported. They agree exactly for an
allocate-on-miss NINE hierarchy, because every level's access count is then
the level above's miss count.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from cachesim.cache import Cache, Evicted
from cachesim.config import ConfigError, HierarchySpec, parse_config
from cachesim.stats import HierarchyStats, collect


class Level:
    """A cache, the cycles it costs to probe it, and its relation to the
    level above.

    Parameters
    ----------
    cache          : the cache at this level.
    hit_time       : cycles charged for probing this level, hit or miss.
    inclusion      : "nine" (default), "inclusive", or "exclusive"; see the
                     module docstring. It constrains this level against the
                     one ABOVE it, so the first level must be "nine".
    write_policy   : "write-back" (default) or "write-through".
    write_allocate : whether a write miss fetches the line into this level
                     (default) or passes the store to the level below.
    """

    def __init__(
        self,
        cache: Cache,
        hit_time: int,
        inclusion: str = "nine",
        write_policy: str = "write-back",
        write_allocate: bool = True,
    ) -> None:
        if inclusion not in ("nine", "inclusive", "exclusive"):
            raise ValueError(f"{cache.name}: unknown inclusion policy {inclusion!r}")
        if write_policy not in ("write-back", "write-through"):
            raise ValueError(f"{cache.name}: unknown write policy {write_policy!r}")
        self.cache = cache
        self.hit_time = hit_time
        self.inclusion = inclusion
        #: True if this level must stay a superset of the level above.
        self.inclusive = inclusion == "inclusive"
        #: True if this level must stay disjoint from the level above.
        self.exclusive = inclusion == "exclusive"
        self.write_policy = write_policy
        #: True if every store applied here is also sent to the level below,
        #: so that no line here is ever dirty.
        self.write_through = write_policy == "write-through"
        self.write_allocate = write_allocate

        #: Lines removed from this level because an inclusive level below it
        #: evicted the block (back-invalidations received, not sent).
        self.back_invalidations = 0
        #: Stores duplicated to the level below because this level is
        #: write-through.
        self.write_throughs = 0
        #: Write misses this level declined to allocate for, passing the
        #: store to the level below instead (no-write-allocate).
        self.write_bypasses = 0

    def reset_stats(self) -> None:
        """Zero this level's counters, including the cache's."""
        self.back_invalidations = 0
        self.write_throughs = 0
        self.write_bypasses = 0
        self.cache.reset_stats()


class Hierarchy:
    """An ordered chain of cache levels backed by main memory.

    Parameters
    ----------
    levels             : list of Level, fastest/smallest (L1) first.
    memory_access_time : cycles to fetch from DRAM after the last level misses.
    """

    def __init__(self, levels: list[Level], memory_access_time: int) -> None:
        if not levels:
            raise ValueError("a hierarchy needs at least one cache level")
        block_sizes = {level.cache.block_size for level in levels}
        if len(block_sizes) != 1:
            raise ValueError(f"all levels must share one block size, got {sorted(block_sizes)}")
        if levels[0].inclusion != "nine":
            raise ValueError(
                f"{levels[0].cache.name}: the first level has no level above it, so its "
                f"inclusion policy must be 'nine', got {levels[0].inclusion!r}"
            )
        self.levels = levels
        self.block_size = levels[0].cache.block_size
        self.memory_access_time = memory_access_time
        # Precomputed so the access path can skip work no level asks for.
        self._has_exclusive = any(level.exclusive for level in levels)
        self._has_inclusive = any(level.inclusive for level in levels)

        # --- statistics ---
        self.accesses = 0
        self.reads = 0
        self.writes = 0
        self.dram_reads = 0  # demand fetches that missed every level
        self.dram_writes = 0  # blocks of modified data written out to DRAM
        self.dram_demand_writes = 0  # ... of which were stores no level allocated for
        self.total_time = 0  # simulated cycles spent on all accesses

    # -- construction helper --------------------------------------------------

    @classmethod
    def from_spec(cls, spec: HierarchySpec) -> Hierarchy:
        """Build a hierarchy from a validated ``HierarchySpec``."""
        levels = []
        for level in spec.levels:
            try:
                cache = Cache(
                    name=level.name,
                    size=level.size,
                    block_size=level.block_size,
                    associativity=level.associativity,
                    policy=level.policy,
                    track_3c=level.track_3c,
                    rng_seed=level.rng_seed,
                    index=level.index,
                )
            except ValueError as exc:
                raise ConfigError(str(exc)) from None
            levels.append(
                Level(
                    cache,
                    level.hit_time,
                    inclusion=level.inclusion,
                    write_policy=level.write_policy,
                    write_allocate=level.write_allocate,
                )
            )
        return cls(levels, spec.memory_access_time)

    @classmethod
    def from_config(cls, config: Mapping[str, Any]) -> Hierarchy:
        """Build a hierarchy from a plain dict (e.g. parsed from JSON).

        The dict is validated with ``config.parse_config``; see that module
        for the expected shape. Raises ``ConfigError`` on malformed input.
        """
        return cls.from_spec(parse_config(config))

    # -- the main entry point ---------------------------------------------------

    def access(self, addr: int, is_write: bool = False) -> int:
        """Simulate one memory access through the whole hierarchy.

        Returns the number of cycles this access took.

        The access probes levels top-down until one hits (or DRAM is
        reached), then fills the block into every level that missed,
        bottom-up, handling each eviction those fills cause.

        A store carries its write intent down only until some level takes
        it: that level applies the store (marking the line dirty unless it
        is write-through) and the levels below it see the remainder of the
        access as a plain line fill, exactly as they always did. A level
        that does not allocate on a write miss simply passes the store on,
        so the next level down sees a store rather than a fill. If no level
        takes the store it becomes a DRAM write.

        Exclusive levels are not filled by the fetch; a block that hit in
        one is moved out of it and up instead.
        """
        if addr < 0:
            raise ValueError(f"address must be non-negative, got {addr}")
        self.accesses += 1
        if is_write:
            self.writes += 1
        else:
            self.reads += 1

        block = addr // self.block_size
        levels = self.levels
        nlevels = len(levels)
        time = 0
        hit_level = nlevels
        write = is_write  # the store is still looking for a level to take it
        taken = -1  # the level that took it (-1: a load, or nobody yet)
        fill_from = 0  # topmost level this access fills
        forward_from = -1  # level a write-through duplicate must be sent to
        for i in range(nlevels):
            level = levels[i]
            time += level.hit_time  # pay to probe this level
            if level.cache.probe(block, write):
                hit_level = i
                if write:
                    taken = fill_from = i  # it is already here; nothing to fill
                    if level.write_through:
                        level.cache.clean(block)
                        forward_from = i + 1
                break
            if write:
                if not level.write_allocate:
                    level.write_bypasses += 1
                    continue  # no line here; the store goes on down
                taken = fill_from = i
                write = False  # below this, the access is an ordinary fill
                if level.write_through:
                    forward_from = i + 1
        else:
            if write:
                # No level allocates on a write miss: the store goes to DRAM
                # without ever fetching the block.
                time += self.memory_access_time
                self.dram_demand_writes += 1
                self._write_to_memory(block)
                self.total_time += time
                return time
            # Missed every level: fetch from DRAM.
            time += self.memory_access_time
            self.dram_reads += 1

        dirty = taken >= 0 and not levels[taken].write_through
        if self._has_exclusive and fill_from < hit_level < nlevels:
            # An exclusive level does not keep a block the level above is
            # about to hold: the block moves up, dirty bit and all.
            level = levels[hit_level]
            if level.exclusive:
                removed = level.cache.invalidate(block, count_writeback=False)
                if removed is not None and removed.dirty:
                    dirty = True

        # Fill the block into every level between the one that took the
        # store (level 0 for a load) and the one that supplied it. Exclusive
        # levels are filled only by the evictions of the level above them,
        # unless this very store is what they took.
        for i in range(hit_level - 1, fill_from - 1, -1):
            if self._has_exclusive and levels[i].exclusive and i != taken:
                continue
            evicted = levels[i].cache.allocate(block, dirty=dirty and i == fill_from)
            if evicted is not None:
                self._handle_eviction(i, evicted)

        if forward_from >= 0:
            levels[forward_from - 1].write_throughs += 1
            self._write_back(forward_from, block)

        self.total_time += time
        return time

    def _handle_eviction(self, level_index: int, evicted: Evicted) -> None:
        """React to a line leaving ``levels[level_index]``.

        A line that leaves a level goes somewhere:

        * into the next level down, if that level is exclusive -- clean or
          dirty, because an exclusive level is exactly the victim buffer of
          the level above it;
        * otherwise down as a write-back if it is dirty, and nowhere at all
          if it is clean.

        If this level is inclusive it then back-invalidates the block out of
        every level above it, which may send a further, more recent copy of
        the data down past this level.
        """
        below = level_index + 1
        if self._has_exclusive and below < len(self.levels) and self.levels[below].exclusive:
            level = self.levels[below]
            cache = level.cache
            keep_dirty = evicted.dirty and not level.write_through
            if evicted.dirty:
                cache.writebacks_received += 1
                cache.writeback_allocations += 1
            passed_on = cache.allocate(evicted.block, dirty=keep_dirty)
            if passed_on is not None:
                self._handle_eviction(below, passed_on)
            if evicted.dirty and not keep_dirty:
                # A write-through victim buffer holds no dirty data: the
                # modified block continues on down.
                level.write_throughs += 1
                self._write_back(below + 1, evicted.block)
        elif evicted.dirty:
            self._write_back(below, evicted.block)
        if self.levels[level_index].inclusive:
            self._back_invalidate(level_index, evicted.block)

    def _back_invalidate(self, level_index: int, block: int) -> None:
        """Remove ``block`` from every level above an inclusive ``levels[level_index]``.

        Walked from the level closest to ``level_index`` upwards, so that if
        several levels somehow hold dirty copies the one closest to the core
        -- the most recent -- is written down last and wins. A dirty copy
        goes to the level BELOW the evicting level, since that level is
        losing the block on this same eviction.
        """
        for i in range(level_index - 1, -1, -1):
            level = self.levels[i]
            removed = level.cache.invalidate(block)
            if removed is None:
                continue
            level.back_invalidations += 1
            if removed.dirty:
                self._write_back(level_index + 1, block)

    def _write_back(self, level_index: int, block: int) -> None:
        """Deliver a block of modified data to ``levels[level_index]``.

        This is the one path by which modified data travels downwards,
        whether it came from a write-back eviction above, from a
        write-through duplicate, or from a store no level above allocated
        for. The receiving level applies its own write policy:

        * write-back: mark the copy dirty, or allocate the block dirty if
          the level no longer holds it. An eviction caused by that
          allocation is handled recursively. The data stops here.
        * write-through: the level never holds dirty data, so it allocates
          a clean line (if it allocates on write misses at all) and the
          data continues to the next level.
        * no-write-allocate on a miss: nothing is installed and the data
          continues to the next level.

        Data that runs past the last level is written to DRAM. Nothing on
        this path is charged simulated time: it is assumed to be absorbed
        by write buffers and drained off the critical path.

        A level that allocates for data arriving from above installs the
        block without fetching it from below. The simulator tracks block
        presence rather than bytes, so the partial-line read that real
        hardware would issue has no effect on any counter it keeps.
        """
        levels = self.levels
        nlevels = len(levels)
        while level_index < nlevels:
            level = levels[level_index]
            cache = level.cache
            cache.writebacks_received += 1
            if level.write_through:
                if level.write_allocate and not cache.contains(block):
                    cache.writeback_allocations += 1
                    evicted = cache.allocate(block, dirty=False)
                    if evicted is not None:
                        self._handle_eviction(level_index, evicted)
                level.write_throughs += 1
                level_index += 1
                continue
            if not cache.mark_dirty(block):
                if not level.write_allocate:
                    level.write_bypasses += 1
                    level_index += 1
                    continue
                cache.writeback_allocations += 1
                evicted = cache.allocate(block, dirty=True)
                if evicted is not None:
                    self._handle_eviction(level_index, evicted)
            return
        self._write_to_memory(block)

    def _write_to_memory(self, block: int) -> None:
        """A dirty block leaves the last level: one DRAM write."""
        self.dram_writes += 1

    def flush(self) -> int:
        """Write every dirty line back to DRAM and clean it, top-down.

        Lines stay resident. Returns the number of DRAM writes performed.
        Use it at the end of a run to account for modified data that is
        still in the caches, or to check dirty-data conservation.
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

    def check_inclusion(self) -> None:
        """Assert the inclusion invariant of every adjacent pair of levels.

        Raises ``AssertionError`` naming the offending block if an inclusive
        level is missing a block held above it, or if an exclusive level
        shares a block with the level above it. NINE pairs are unconstrained
        and always pass. Intended for tests: run a random access stream
        through the hierarchy and then call this.
        """
        for i in range(1, len(self.levels)):
            level = self.levels[i]
            if level.inclusion == "nine":
                continue
            upper = self.levels[i - 1].cache
            lower = level.cache
            upper_blocks = {block for _, _, block, _ in upper.lines()}
            if level.inclusive:
                missing = sorted(b for b in upper_blocks if not lower.contains(b))
                if missing:
                    raise AssertionError(
                        f"{lower.name} is inclusive of {upper.name} but does not hold "
                        f"{len(missing)} block(s) it caches, e.g. {missing[:8]}"
                    )
            else:
                shared = sorted(b for b in upper_blocks if lower.contains(b))
                if shared:
                    raise AssertionError(
                        f"{lower.name} is exclusive of {upper.name} but shares "
                        f"{len(shared)} block(s) with it, e.g. {shared[:8]}"
                    )

    def reset_stats(self) -> None:
        """Zero every counter at every level while keeping cache contents.

        Call it after a warm-up prefix of the trace so the reported figures
        describe steady-state behaviour rather than cold caches.
        """
        self.accesses = self.reads = self.writes = 0
        self.dram_reads = self.dram_writes = self.dram_demand_writes = 0
        self.total_time = 0
        for level in self.levels:
            level.reset_stats()

    # -- metrics -------------------------------------------------------------

    def stats(self) -> HierarchyStats:
        """Snapshot every counter and derived metric (see ``cachesim.stats``)."""
        return collect(self)

    @property
    def memory_accesses(self) -> int:
        """Total DRAM traffic: demand reads plus write-backs."""
        return self.dram_reads + self.dram_writes

    def global_miss_rate(self, level_index: int) -> float:
        """Misses at this level / all CPU accesses (not just ones reaching it)."""
        if self.accesses == 0:
            return 0.0
        return self.levels[level_index].cache.misses / self.accesses

    def amat(self) -> float:
        """Analytic AMAT via the nested formula, in cycles.

        Built from the bottom up: the miss penalty of the last level is the
        memory access time; every level above adds
        ``hit_time + miss_rate * penalty_below``.
        """
        penalty: float = self.memory_access_time
        for level in reversed(self.levels):
            penalty = level.hit_time + level.cache.miss_rate * penalty
        return penalty

    def measured_amat(self) -> float:
        """Total simulated cycles / accesses."""
        return self.total_time / self.accesses if self.accesses else 0.0
