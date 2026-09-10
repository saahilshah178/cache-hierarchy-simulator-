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

Transfer time and bandwidth
---------------------------

``bus_width`` on a level is the width in bytes per cycle of the link that
carries blocks from the level below INTO it. A fill then costs
``ceil(block_size / bus_width)`` cycles on top of the latency, charged to
the access that caused it, and ``amat()`` carries the same term (see its
docstring). The default, ``null``, is an infinitely wide link and adds
nothing, so an existing configuration keeps its timing exactly.

A wider block lowers the miss rate but takes longer to move, so AMAT
against block size need not fall with it: past some width the extra
transfer time costs more than the misses it saves. Measured on the two
sample traces through a single 32 KB 4-way level, hit time 4, DRAM 100,
``bus_width`` 16 (miss rate / measured AMAT, transfer cycles per fill)::

    block   sequential          matmul_naive        transfer
       16   50.0000% / 54.500    7.1575% / 11.229      1
       32   25.0000% / 29.500    5.8804% /  9.998      2
       64   12.5000% / 17.000    5.2419% /  9.452      4
      128    6.2500% / 10.750    4.9226% /  9.316      8
      256    3.1250% /  7.625    4.0587% /  8.708     16
      512    1.5625% /  6.062    3.9288% /  9.186     32

Only ``matmul_naive`` shows the U. A linear scan uses every byte it
fetches, so its miss rate falls exactly as 1/block and its AMAT is
``4 + (8/B)*(100 + B/16) = 4 + 800/B + 0.5``, which is monotonically
decreasing for *any* bus width -- ``sequential`` has no interior minimum
to find. The trade-off needs imperfect spatial locality: the naive
matrix multiply strides down B's columns, bottoms out at 256 B and rises
again at 512 B while its miss rate is still falling. The scan column and
the rows around that minimum are pinned in ``tests/test_timing.py``.

Every level also counts the bytes it pulls in from below and pushes down,
and the hierarchy counts DRAM bytes both ways, so a configuration can be
judged on traffic as well as on latency.

Prefetching
-----------

A level may name a ``prefetcher`` (see ``cachesim.prefetch``), which
watches the demand references that reach it and may name one block to
fetch ahead. The block is fetched through the levels below and installed
here with a ``prefetched`` flag, so the first demand hit on it can be
counted as a prefetch that paid off and one that leaves unused can be
counted as pollution.

Two accounting decisions, both of which the alternative would have
undone: prefetch traffic is charged no time (it is assumed to overlap
with useful work), and the lookups it causes at lower levels are counted
as ``prefetch_probes`` rather than as accesses, so every level's miss
rate stays a statement about demand references and the AMAT identity
stays exact. Prefetch lookups also leave the replacement state of the
levels below untouched.

Victim caches
-------------

``victim_cache`` gives a level a small fully-associative buffer of the
lines its array has replaced (Jouppi, ISCA 1990; see
``cachesim.cache.VictimBuffer``). The buffer is part of the level: it is
probed with the tag array, a block found there counts as a hit at that
level and is charged the level's hit time, and only the line the buffer
itself pushes out actually leaves. A handful of entries removes most of
the conflict misses of a direct-mapped cache, which is why the counter to
watch is ``victim_hits`` beside the level's own miss count.

Main memory
-----------

Below the last level sits a ``cachesim.dram.MemoryModel``. The default is
a constant latency -- ``memory_access_time`` -- which is what the model
has always assumed. The alternative is an open-page DRAM with one
activated row per bank, under which the *order* of the misses matters as
much as their number: a sequential miss stream walks a row at a time and
almost always finds it open, while a random one activates a new row
nearly every access.

Write-backs and prefetches occupy a bank and shift its open row, so they
interfere with demand traffic even though they are charged no cycles.

AMAT (average memory access time) is the headline metric:

    AMAT = L1_hit_time + L1_miss_rate * L1_miss_penalty

where L1's miss penalty is itself the AMAT of the rest of the hierarchy, so
the formula nests. Both the analytic value and the measured average
(total simulated cycles / accesses) are reported, along with the measured
averages split by loads and stores. Analytic and measured agree exactly
for an allocate-on-miss NINE hierarchy, because every level's access count
is then the level above's miss count and its fill count is its own miss
count.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from cachesim.cache import Cache, Evicted
from cachesim.config import ConfigError, HierarchySpec, parse_config
from cachesim.dram import ConstantMemory, MemoryModel
from cachesim.prefetch import make_prefetcher
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
    bus_width      : bytes per cycle of the link that carries blocks from
                     the level below into this one. None (the default)
                     models an infinitely wide link and adds no transfer
                     term, preserving the original timing model.
    prefetcher     : name of the hardware prefetcher watching this level,
                     "none" (default), "next-line" or "stride"; see
                     ``cachesim.prefetch``.
    """

    def __init__(
        self,
        cache: Cache,
        hit_time: int,
        inclusion: str = "nine",
        write_policy: str = "write-back",
        write_allocate: bool = True,
        bus_width: int | None = None,
        prefetcher: str = "none",
    ) -> None:
        if inclusion not in ("nine", "inclusive", "exclusive"):
            raise ValueError(f"{cache.name}: unknown inclusion policy {inclusion!r}")
        if write_policy not in ("write-back", "write-through"):
            raise ValueError(f"{cache.name}: unknown write policy {write_policy!r}")
        if bus_width is not None and bus_width < 1:
            raise ValueError(f"{cache.name}: bus_width must be positive, got {bus_width}")
        self.cache = cache
        self.hit_time = hit_time
        self.prefetcher_name = prefetcher
        self.prefetcher = make_prefetcher(prefetcher)
        self.bus_width = bus_width
        #: Cycles a whole block occupies the link from below, ceil(block/width).
        self.transfer_cycles = (
            0 if bus_width is None else -(-cache.block_size // bus_width)  # ceil division
        )
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
        #: Bytes filled into this level from below (demand and prefetch).
        self.bytes_read_from_below = 0
        #: Bytes sent from this level to the level below: write-backs,
        #: victims handed to an exclusive level, and forwarded stores. The
        #: trace carries no access size, so a forwarded store is counted as
        #: a whole block -- an upper bound; the exact transaction counts are
        #: ``write_throughs`` and ``write_bypasses``.
        self.bytes_written_below = 0

        #: Prefetches this level's prefetcher issued.
        self.prefetches_issued = 0
        #: Prefetched lines a demand reference then hit: useful prefetches.
        self.prefetch_hits = 0
        #: Prefetched lines that left this level without a demand hit.
        self.prefetch_evicted_unused = 0
        #: Lookups at this level caused by a prefetch from a level above.
        self.prefetch_probes = 0
        #: ... of which found the block here.
        self.prefetch_probe_hits = 0

    def reset_stats(self) -> None:
        """Zero this level's counters, including the cache's.

        Prediction state is deliberately kept, exactly as cache contents
        and replacement state are: a warm-up should leave the prefetcher
        trained.
        """
        self.back_invalidations = 0
        self.write_throughs = 0
        self.write_bypasses = 0
        self.bytes_read_from_below = 0
        self.bytes_written_below = 0
        self.prefetches_issued = 0
        self.prefetch_hits = 0
        self.prefetch_evicted_unused = 0
        self.prefetch_probes = 0
        self.prefetch_probe_hits = 0
        self.cache.reset_stats()

    @property
    def prefetch_accuracy(self) -> float:
        """Useful prefetches / prefetches issued; 0.0 if none were issued.

        How much of the extra traffic was worth fetching.
        """
        if not self.prefetches_issued:
            return 0.0
        return self.prefetch_hits / self.prefetches_issued

    @property
    def prefetch_coverage(self) -> float:
        """Useful prefetches / (useful prefetches + demand misses).

        The share of the misses this level would otherwise have taken that
        the prefetcher turned into hits.
        """
        total = self.prefetch_hits + self.cache.misses
        return self.prefetch_hits / total if total else 0.0


class Hierarchy:
    """An ordered chain of cache levels backed by main memory.

    Parameters
    ----------
    levels             : list of Level, fastest/smallest (L1) first.
    memory_access_time : cycles to fetch from DRAM after the last level
                         misses. Shorthand for ``memory=ConstantMemory(n)``.
    memory             : a ``cachesim.dram.MemoryModel`` instead, for a
                         latency that depends on the address stream.
                         Exactly one of the two must be given.
    """

    def __init__(
        self,
        levels: list[Level],
        memory_access_time: int | None = None,
        memory: MemoryModel | None = None,
    ) -> None:
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
        if memory is None:
            if memory_access_time is None:
                raise ValueError("a hierarchy needs a memory_access_time or a memory model")
            memory = ConstantMemory(memory_access_time)
        elif memory_access_time is not None:
            raise ValueError("give either memory_access_time or memory, not both")
        self.levels = levels
        self._nlevels = len(levels)
        self.block_size = levels[0].cache.block_size
        self.memory = memory
        #: The fixed DRAM latency, or None if it depends on the address.
        self.memory_access_time = memory.constant_latency
        # When memory answers in a fixed number of cycles there is nothing
        # for it to remember, so the access path adds the latency directly
        # rather than calling into the model on every miss.
        self._constant_latency = memory.constant_latency
        # Precomputed so the access path can skip work no level asks for.
        self._has_exclusive = any(level.exclusive for level in levels)
        self._has_inclusive = any(level.inclusive for level in levels)
        self._prefetch_levels = tuple(
            i for i, level in enumerate(levels) if level.prefetcher is not None
        )
        self._has_victim = any(level.cache.victim is not None for level in levels)
        # True when every optional feature of the access path is switched
        # off, which is the configuration in configs/default.json and the
        # one nearly every run uses. ``_access_default`` then handles the
        # access; see ``access`` for what the two paths share.
        self._all_defaults = (
            self._constant_latency is not None
            and not self._has_exclusive
            and not self._has_inclusive
            and not self._has_victim
            and not self._prefetch_levels
            and all(
                not level.write_through and level.write_allocate and level.transfer_cycles == 0
                for level in levels
            )
        )
        #: DRAM latency as a plain int for the fast path, where it is never
        #: None (``_all_defaults`` requires a constant-latency memory).
        self._default_latency = 0 if self._constant_latency is None else self._constant_latency

        # --- statistics ---
        self.accesses = 0
        self.reads = 0
        self.writes = 0
        self.dram_reads = 0  # demand fetches that missed every level
        self.dram_writes = 0  # blocks of modified data written out to DRAM
        self.dram_demand_writes = 0  # ... of which were stores no level allocated for
        self.dram_prefetch_reads = 0  # prefetches that reached DRAM
        self.dram_bytes_read = 0
        self.dram_bytes_written = 0
        self.read_cycles = 0  # simulated cycles spent on loads
        self.write_cycles = 0  # ... and on stores

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
                    victim_entries=level.victim_cache,
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
                    bus_width=level.bus_width,
                    prefetcher=level.prefetcher,
                )
            )
        return cls(levels, memory=spec.memory.build())

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

        A hierarchy with every optional feature switched off is handled by
        ``_access_default``, which is this method with the branches those
        features need removed. The code below stays the definition of what
        an access means; see ``_access_default`` for the argument that the
        two agree, and ``tests/test_hierarchy.py`` for the check that they
        do, counter by counter.
        """
        if addr < 0:
            raise ValueError(f"address must be non-negative, got {addr}")
        self.accesses += 1
        if self._all_defaults:
            return self._access_default(addr, is_write)

        block_size = self.block_size
        block = addr // block_size
        levels = self.levels
        nlevels = len(levels)
        exclusive_anywhere = self._has_exclusive
        time = 0
        hit_level = nlevels
        write = is_write  # the store is still looking for a level to take it
        taken = -1  # the level that took it (-1: a load, or nobody yet)
        fill_from = 0  # topmost level this access fills
        forward_from = -1  # level a write-through duplicate must be sent to
        for i, level in enumerate(levels):
            time += level.hit_time  # pay to probe this level
            if level.cache.probe(block, write):
                hit_level = i
                if self._has_victim:
                    # A victim-buffer hit swapped a line back into the array
                    # and may have pushed another out of the level.
                    displaced = level.cache.take_pending_eviction()
                    if displaced is not None:
                        self._handle_eviction(i, displaced)
                if write:
                    taken = fill_from = i  # it is already here; nothing to fill
                    if level.write_through:
                        level.cache.clean(block)
                        forward_from = i + 1
                break
            if write:
                if not level.write_allocate:
                    level.write_bypasses += 1
                    level.bytes_written_below += block_size
                    continue  # no line here; the store goes on down
                taken = fill_from = i
                write = False  # below this, the access is an ordinary fill
                if level.write_through:
                    forward_from = i + 1
        else:
            latency = self._constant_latency
            if write:
                # No level allocates on a write miss: the store goes to DRAM
                # without ever fetching the block, so nothing is filled.
                self.dram_demand_writes += 1
                served = self._write_to_memory(block, charged=True)
                if latency is None:
                    latency = served
                fill_from = nlevels
            else:
                # Missed every level: fetch from DRAM.
                self.dram_reads += 1
                self.dram_bytes_read += block_size
                if latency is None:
                    latency = self.memory.read(block * block_size)
            time += latency

        dirty = taken >= 0 and not levels[taken].write_through
        if exclusive_anywhere and fill_from < hit_level < nlevels:
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
            level = levels[i]
            if exclusive_anywhere and level.exclusive and i != taken:
                continue
            time += level.transfer_cycles  # moving the block up occupies the link
            level.bytes_read_from_below += block_size
            evicted = level.cache.allocate(block, dirty=dirty and i == fill_from)
            if evicted is not None:
                self._handle_eviction(i, evicted)

        if forward_from >= 0:
            level = levels[forward_from - 1]
            level.write_throughs += 1
            level.bytes_written_below += block_size
            self._write_back(forward_from, block)

        if self._prefetch_levels:
            self._run_prefetchers(block, hit_level)

        if is_write:
            self.writes += 1
            self.write_cycles += time
        else:
            self.reads += 1
            self.read_cycles += time
        return time

    def _access_default(self, addr: int, is_write: bool) -> int:
        """``access`` for a hierarchy that asks for none of the extras.

        Reached only when ``_all_defaults`` holds: NINE everywhere,
        write-back and write-allocate everywhere, no bus width, no
        prefetcher, no victim buffer, and a constant-latency memory. The
        caller has already rejected a negative address and counted the
        access.

        Under those conditions the general path collapses, term by term:

        * ``exclusive_anywhere`` is false, so the block never moves up out
          of a level and no fill is skipped;
        * every level allocates on a write miss, so the first level takes
          the store -- on a hit at level 0 or on its miss -- and below it
          ``write`` is false. ``taken`` is therefore 0 for a store and -1
          for a load, ``fill_from`` is 0 either way, and ``dirty``, which
          is ``taken >= 0 and not write_through``, is just ``is_write``;
        * no level is write-through, so ``forward_from`` stays -1 and
          nothing is duplicated downwards;
        * the store is always taken, so the loop's ``else`` can only be a
          load that missed everything: one DRAM read;
        * ``transfer_cycles`` is 0 at every level, so the fill loop charges
          no link time;
        * no level has a victim buffer, so a probe never displaces a line,
          and no prefetcher is watching, so nothing is predicted.

        What is left is: probe downwards, fetch, fill upwards. Evictions go
        through ``_handle_eviction`` exactly as they do on the general path,
        so nothing about what happens to a line that leaves a level is
        restated here.

        The first level is then peeled out of the probe loop, because a hit
        there is what nearly every access is -- 94% of the accesses in
        matmul_naive -- and it is the one outcome that fills nothing, writes
        nothing back and reaches no level below. Peeling it means that
        common access never builds the ``enumerate`` or the ``range`` the
        loops would need. The peeled copy is the loop's first iteration
        written out: it probes ``levels[0]`` with the access's own write
        flag, exactly as iteration 0 did, and every later level is probed
        with ``False`` because level 0 has by then taken the store.
        """
        block_size = self.block_size
        block = addr // block_size
        levels = self.levels
        first = levels[0]
        time = first.hit_time
        if first.cache.probe(block, is_write):
            # A hit in the first level: nothing to fetch, fill or evict.
            if is_write:
                self.writes += 1
                self.write_cycles += time
            else:
                self.reads += 1
                self.read_cycles += time
            return time

        nlevels = self._nlevels
        hit_level = nlevels
        for i in range(1, nlevels):
            level = levels[i]
            time += level.hit_time
            # Level 0 has taken the store, so below it this is a plain fill.
            if level.cache.probe(block, False):
                hit_level = i
                break
        else:
            self.dram_reads += 1
            self.dram_bytes_read += block_size
            time += self._default_latency

        for i in range(hit_level - 1, -1, -1):
            level = levels[i]
            level.bytes_read_from_below += block_size
            evicted = level.cache.allocate(block, dirty=is_write and i == 0)
            if evicted is not None:
                self._handle_eviction(i, evicted)

        if is_write:
            self.writes += 1
            self.write_cycles += time
        else:
            self.reads += 1
            self.read_cycles += time
        return time

    # -- prefetching ----------------------------------------------------------

    def _run_prefetchers(self, block: int, hit_level: int) -> None:
        """Let every prefetcher the access reached observe it and predict.

        A prefetcher at level i sees the reference only if the access got
        that far, so an L1 prefetcher watches every reference while an L2
        one watches only L1 misses. A prediction the level already holds is
        dropped, and a prediction that is issued is fetched off the
        critical path: no cycles are charged.
        """
        levels = self.levels
        deepest = min(hit_level, len(levels) - 1)
        for i in self._prefetch_levels:
            if i > deepest:
                break
            level = levels[i]
            prefetcher = level.prefetcher
            assert prefetcher is not None  # _prefetch_levels only lists these
            hit = i == hit_level
            was_prefetched = hit and level.cache.clear_prefetched(block)
            if was_prefetched:
                level.prefetch_hits += 1
            target = prefetcher.predict(block, hit, was_prefetched)
            if target is None or target < 0 or level.cache.contains(target):
                continue
            level.prefetches_issued += 1
            self._prefetch_fill(i, target)

    def _prefetch_fill(self, level_index: int, block: int) -> None:
        """Fetch ``block`` into ``levels[level_index]`` speculatively.

        The lookup walks the levels below without touching their demand
        counters or their replacement state: those probes are recorded as
        ``prefetch_probes`` instead, so every level's miss rate remains a
        statement about demand references alone. The block is then filled
        into the same levels a demand fetch would have filled, and only the
        issuing level marks its copy prefetched -- lines the prefetch
        happened to leave behind at intermediate levels are ordinary fills.
        Nothing here is charged simulated time.
        """
        levels = self.levels
        nlevels = len(levels)
        block_size = self.block_size
        source = nlevels
        for k in range(level_index + 1, nlevels):
            below = levels[k]
            below.prefetch_probes += 1
            if below.cache.contains(block):
                below.prefetch_probe_hits += 1
                source = k
                break
        dirty = False
        if source == nlevels:
            self.dram_prefetch_reads += 1
            self.dram_bytes_read += block_size
            if self._constant_latency is None:
                # Untimed, but it still occupies a bank and moves an open row.
                self.memory.read(block * block_size, charged=False)
        elif levels[source].exclusive:
            # As on the demand path, an exclusive level hands the block up.
            removed = levels[source].cache.invalidate(block, count_writeback=False)
            if removed is not None:
                dirty = removed.dirty
        for k in range(source - 1, level_index - 1, -1):
            level = levels[k]
            if level.exclusive and k != level_index:
                continue
            level.bytes_read_from_below += block_size
            evicted = level.cache.allocate(
                block, dirty=dirty and k == level_index, prefetched=k == level_index
            )
            if evicted is not None:
                self._handle_eviction(k, evicted)

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
        if evicted.prefetched:
            self.levels[level_index].prefetch_evicted_unused += 1
        below = level_index + 1
        if self._has_exclusive and below < len(self.levels) and self.levels[below].exclusive:
            level = self.levels[below]
            cache = level.cache
            keep_dirty = evicted.dirty and not level.write_through
            if evicted.dirty:
                cache.writebacks_received += 1
                cache.writeback_allocations += 1
            self.levels[level_index].bytes_written_below += self.block_size
            passed_on = cache.allocate(evicted.block, dirty=keep_dirty)
            if passed_on is not None:
                self._handle_eviction(below, passed_on)
            if evicted.dirty and not keep_dirty:
                # A write-through victim buffer holds no dirty data: the
                # modified block continues on down.
                level.write_throughs += 1
                level.bytes_written_below += self.block_size
                self._write_back(below + 1, evicted.block)
        elif evicted.dirty:
            self.levels[level_index].bytes_written_below += self.block_size
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
            if removed.prefetched:
                level.prefetch_evicted_unused += 1
            if removed.dirty:
                level.bytes_written_below += self.block_size
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
                level.bytes_written_below += self.block_size
                level_index += 1
                continue
            if not cache.mark_dirty(block):
                if not level.write_allocate:
                    level.write_bypasses += 1
                    level.bytes_written_below += self.block_size
                    level_index += 1
                    continue
                cache.writeback_allocations += 1
                evicted = cache.allocate(block, dirty=True)
                if evicted is not None:
                    self._handle_eviction(level_index, evicted)
            return
        self._write_to_memory(block)

    def _write_to_memory(self, block: int, charged: bool = False) -> int:
        """A block of modified data leaves the last level: one DRAM write.

        Returns the cycles main memory took. ``charged`` says whether the
        access is on the critical path -- true only for a store that no
        cache level allocated for, false for write-backs, which the model
        assumes a write buffer drains. Either way the write occupies a
        bank, so it can shift the open rows under the demand stream.
        """
        self.dram_writes += 1
        self.dram_bytes_written += self.block_size
        if self._constant_latency is None:
            return self.memory.write(block * self.block_size, charged=charged)
        return self._constant_latency

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
                    level.bytes_written_below += self.block_size
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
        self.dram_prefetch_reads = 0
        self.dram_bytes_read = self.dram_bytes_written = 0
        self.read_cycles = self.write_cycles = 0
        # Memory keeps its open rows, as the caches keep their contents.
        self.memory.reset_stats()
        for level in self.levels:
            level.reset_stats()

    # -- metrics -------------------------------------------------------------

    def stats(self) -> HierarchyStats:
        """Snapshot every counter and derived metric (see ``cachesim.stats``)."""
        return collect(self)

    @property
    def total_time(self) -> int:
        """Simulated cycles spent on all accesses: loads plus stores."""
        return self.read_cycles + self.write_cycles

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

        Written out, with levels numbered from 0 and ``transfer_i`` the
        cycles a block takes to cross the link into level i::

            AMAT      = hit_time_0 + miss_rate_0 * penalty_0
            penalty_i = hit_time_{i+1} + transfer_i
                          + miss_rate_{i+1} * penalty_{i+1}
            penalty_last = memory_access_time + transfer_last

        so every level pays for its own fill however deep the data came
        from, and the bottom of the recursion is a DRAM access plus the
        transfer into the last level. It is computed bottom-up: start at
        the memory access time, and for each level from the last upwards
        take ``hit_time + miss_rate * (transfer + penalty_below)``.

        This equals ``measured_amat()`` exactly for an allocate-on-miss
        NINE hierarchy, because each level is then probed once per miss of
        the level above and filled once per miss of its own. Levels that do
        not fill on every miss -- exclusive levels, which are filled by the
        level above instead, and no-write-allocate levels, which decline
        write misses -- still match on the hit-time and memory terms but
        make the transfer term an upper bound.

        Under a memory model whose latency depends on the address stream,
        the DRAM term is the *measured* average latency of the accesses
        that were charged, since no single number describes it in advance.
        The identity therefore still holds, but the analytic figure is then
        analytic only in the miss rates; the measured AMAT is the ground
        truth in both cases.
        """
        penalty: float = self.memory.average_latency()
        for level in reversed(self.levels):
            penalty = level.hit_time + level.cache.miss_rate * (level.transfer_cycles + penalty)
        return penalty

    def measured_amat(self) -> float:
        """Total simulated cycles / accesses."""
        return self.total_time / self.accesses if self.accesses else 0.0

    def read_amat(self) -> float:
        """Measured average cycles per load."""
        return self.read_cycles / self.reads if self.reads else 0.0

    def write_amat(self) -> float:
        """Measured average cycles per store."""
        return self.write_cycles / self.writes if self.writes else 0.0
