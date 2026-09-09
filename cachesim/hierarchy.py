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
* Writes dirty the first level only (write-back semantics): if a store
  misses L1, the levels below are supplying the line, so they see a read.
* Timing charges each level's hit time once per level probed, plus the
  memory access time if everything missed. Write-back traffic is counted
  (see ``Cache.writebacks``) but not charged time.

The model is functional (exact hit/miss/eviction behaviour) with a serial,
fixed-latency timing model; it does not model overlap of misses (MSHRs),
DRAM banks, or bandwidth.

AMAT (average memory access time) is the headline metric:

    AMAT = L1_hit_time + L1_miss_rate * L1_miss_penalty

where L1's miss penalty is itself the AMAT of the rest of the hierarchy, so
the formula nests. Both the analytic value and the measured average
(total simulated cycles / accesses) are reported.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from cachesim.cache import Cache, Evicted
from cachesim.config import ConfigError, HierarchySpec, parse_config


class Level:
    """A cache plus the time (in cycles) it costs to probe it."""

    def __init__(self, cache: Cache, hit_time: int) -> None:
        self.cache = cache
        self.hit_time = hit_time


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
        self.levels = levels
        self.block_size = levels[0].cache.block_size
        self.memory_access_time = memory_access_time

        # --- statistics ---
        self.accesses = 0
        self.reads = 0
        self.writes = 0
        self.memory_accesses = 0  # accesses that fell all the way to DRAM
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
                )
            except ValueError as exc:
                raise ConfigError(str(exc)) from None
            levels.append(Level(cache, level.hit_time))
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
        time = 0
        hit_level = len(levels)
        for i, level in enumerate(levels):
            time += level.hit_time  # pay to probe this level
            # Only the first level sees the write intent: with write-back
            # caches, a store that misses L1 asks the levels below for the
            # line (a read); the data itself is only modified in L1.
            if level.cache.probe(block, is_write and i == 0):
                hit_level = i
                break
        else:
            # Missed every level: fetch from DRAM.
            time += self.memory_access_time
            self.memory_accesses += 1

        # Fill the block into every level above the one that supplied it.
        for i in range(hit_level - 1, -1, -1):
            evicted = levels[i].cache.allocate(block, dirty=is_write and i == 0)
            if evicted is not None:
                self._handle_eviction(i, evicted)

        self.total_time += time
        return time

    def _handle_eviction(self, level_index: int, evicted: Evicted) -> None:
        """React to a line leaving ``levels[level_index]``.

        Write-backs are counted at the evicting level (``Cache.writebacks``)
        and not forwarded further down.
        """

    # -- metrics -------------------------------------------------------------

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
