"""A single, configurable cache level.

One level of a cache (an L1, an L2, ...) is modelled as a grid of
``num_sets`` sets by ``associativity`` ways:

* The cache holds ``size`` bytes, divided into blocks (lines) of
  ``block_size`` bytes. Addresses are converted to block numbers
  (``addr // block_size``) at the boundary; everything inside works on
  block numbers.
* A block maps to exactly one set (``block % num_sets``) and may occupy any
  way of that set.

    - associativity == 1                    -> direct-mapped
    - associativity == blocks in the cache  -> fully associative
    - otherwise                             -> N-way set-associative

* When a set is full, the ``ReplacementPolicy`` chooses the victim.

The level exposes three primitives so that a hierarchy can compose them
into miss handling, write-backs, back-invalidation, and prefetching:

* ``probe(block, is_write)``  -- look the block up, update hit/miss
  statistics and replacement state, and classify a miss. Does not fill.
* ``allocate(block, dirty)``  -- install a block, evicting a victim if the
  set is full; returns the victim as an ``Evicted`` record.
* ``invalidate(block)``       -- remove a block on command; returns the
  removed line so a dirty one can be written back.

``access(addr, is_write)`` is ``probe`` followed by ``allocate`` on a miss,
for single-level use.

Optionally every miss is classified into the three Cs:

* compulsory -- the first reference to this block.
* capacity   -- a fully-associative LRU cache of the same capacity would also
                have missed.
* conflict   -- the fully-associative twin would have hit, so the miss is
                due to the set mapping (or, for non-LRU policies, to the
                replacement choice).

Classification runs a shadow fully-associative LRU cache of identical
capacity alongside the real one and compares outcomes.

Dirty tracking implements write-back semantics: a write marks the line
dirty, and evicting or invalidating a dirty line counts as a write-back.
"""

from __future__ import annotations

import random
from collections import OrderedDict
from collections.abc import Iterator
from typing import NamedTuple

from cachesim.policies import POLICIES, ReplacementPolicy

#: Block-number sentinel for an empty way. Real block numbers are >= 0.
EMPTY = -1


class Evicted(NamedTuple):
    """A line removed from a cache by ``allocate`` or ``invalidate``."""

    block: int
    dirty: bool


def _validate_geometry(name: str, size: int, block_size: int, associativity: int) -> None:
    """Reject geometries that cannot be simulated.

    ``block_size`` must be a power of two so that the block number is a
    bit-field of the address. ``size`` must hold a whole number of sets:
    the set count may be any positive integer (set index = block modulo
    num_sets), although real hardware uses a power of two.
    """
    for label, value in (
        ("size", size),
        ("block_size", block_size),
        ("associativity", associativity),
    ):
        if isinstance(value, bool) or not isinstance(value, int):
            raise ValueError(f"{name}: {label} must be an integer, got {value!r}")
        if value <= 0:
            raise ValueError(f"{name}: {label} must be positive, got {value}")
    if block_size & (block_size - 1):
        raise ValueError(f"{name}: block_size must be a power of two, got {block_size}")
    if size % (block_size * associativity) != 0:
        raise ValueError(
            f"{name}: size {size} must be a multiple of "
            f"block_size*associativity ({block_size * associativity})"
        )


class Cache:
    """One set-associative cache level.

    Parameters
    ----------
    name          : label used in reports ("L1", "L2", ...).
    size          : total capacity in bytes.
    block_size    : bytes per cache block.
    associativity : ways per set; 1 is direct-mapped.
    policy        : replacement policy name (see ``policies.POLICIES``).
    track_3c      : run the shadow fully-associative cache and classify
                    misses into compulsory / capacity / conflict.
    rng_seed      : seed for the random policy.
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
        _validate_geometry(name, size, block_size, associativity)

        self.name = name
        self.size = size
        self.block_size = block_size
        self.associativity = associativity
        self.num_sets = size // (block_size * associativity)
        self.num_blocks = size // block_size
        self.policy_name = policy

        if isinstance(rng_seed, bool) or not isinstance(rng_seed, int):
            raise ValueError(f"{name}: rng_seed must be an integer, got {rng_seed!r}")
        rng = random.Random(rng_seed)
        try:
            self.policy: ReplacementPolicy = POLICIES[policy](self.num_sets, associativity, rng)
        except KeyError:
            raise ValueError(
                f"unknown replacement policy {policy!r}; choose from {sorted(POLICIES)}"
            ) from None

        # Cache contents, indexed [set][way]: the block number held in each
        # way (EMPTY for an invalid way) and its dirty bit.
        self._blocks = [[EMPTY] * associativity for _ in range(self.num_sets)]
        self._dirty = [[False] * associativity for _ in range(self.num_sets)]

        # --- statistics -----------------------------------------------------
        self.hits = 0
        self.misses = 0
        self.read_hits = 0
        self.read_misses = 0
        self.write_hits = 0
        self.write_misses = 0
        self.fills = 0  # blocks installed by allocate()
        self.evictions = 0  # valid lines replaced by allocate()
        self.invalidations = 0  # valid lines removed by invalidate()
        self.writebacks = 0  # dirty lines written down (evicted, invalidated, flushed)
        self.writebacks_received = 0  # dirty lines written into this level from above
        self.writeback_allocations = 0  # ... of which had to be allocated (line was absent)

        # --- 3-C classification machinery ----------------------------------
        self.track_3c = track_3c
        self.compulsory_misses = 0
        self.capacity_misses = 0
        self.conflict_misses = 0
        self._seen_blocks: set[int] = set()  # every block number ever touched
        # Shadow fully-associative LRU cache of the same capacity:
        # an OrderedDict of block numbers, oldest first.
        self._shadow: OrderedDict[int, bool] = OrderedDict()

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

    # -- primitives ------------------------------------------------------------

    def probe(self, block: int, is_write: bool = False) -> bool:
        """Look ``block`` up. Returns True on hit, False on miss.

        Updates hit/miss statistics, replacement state, the dirty bit on a
        write hit, and the 3-C classification on a miss. Does not fill: on a
        miss the caller decides whether and how to ``allocate``.
        """
        set_idx = block % self.num_sets
        blocks = self._blocks[set_idx]
        # At most ``ways`` comparisons, like the parallel tag comparators of
        # real hardware.
        for way in range(self.associativity):
            if blocks[way] == block:
                self.hits += 1
                if is_write:
                    self.write_hits += 1
                    self._dirty[set_idx][way] = True
                else:
                    self.read_hits += 1
                self.policy.on_hit(set_idx, way, block)
                if self.track_3c:
                    self._update_shadow(block)
                return True

        self.misses += 1
        if is_write:
            self.write_misses += 1
        else:
            self.read_misses += 1
        if self.track_3c:
            self._classify_miss(block)
            self._update_shadow(block)
        return False

    def allocate(self, block: int, dirty: bool = False) -> Evicted | None:
        """Install ``block``, evicting a victim if its set is full.

        Returns the evicted line, or None if an empty way was used. If the
        block is already present this is a no-op (the dirty bit is OR-ed in)
        and None is returned.
        """
        set_idx = block % self.num_sets
        blocks = self._blocks[set_idx]
        dirty_bits = self._dirty[set_idx]

        way = EMPTY
        for w in range(self.associativity):
            held = blocks[w]
            if held == block:
                if dirty:
                    dirty_bits[w] = True
                return None
            if held == EMPTY and way == EMPTY:
                way = w
        evicted = None
        if way == EMPTY:
            way = self.policy.victim(set_idx)
            self.evictions += 1
            evicted = Evicted(blocks[way], dirty_bits[way])
            if evicted.dirty:
                self.writebacks += 1

        blocks[way] = block
        dirty_bits[way] = dirty
        self.fills += 1
        self.policy.on_fill(set_idx, way, block)
        return evicted

    def invalidate(self, block: int) -> Evicted | None:
        """Remove ``block`` if present. Returns the removed line, else None.

        Counts as an invalidation, not an eviction; a dirty line still
        counts as a write-back because its data must be written down.
        """
        set_idx = block % self.num_sets
        blocks = self._blocks[set_idx]
        for way in range(self.associativity):
            if blocks[way] == block:
                removed = Evicted(block, self._dirty[set_idx][way])
                blocks[way] = EMPTY
                self._dirty[set_idx][way] = False
                self.invalidations += 1
                if removed.dirty:
                    self.writebacks += 1
                self.policy.on_invalidate(set_idx, way)
                return removed
        return None

    def contains(self, block: int) -> bool:
        """True if ``block`` is currently resident (no statistics updated)."""
        return block in self._blocks[block % self.num_sets]

    def is_dirty(self, block: int) -> bool:
        """True if ``block`` is resident and dirty."""
        set_idx = block % self.num_sets
        blocks = self._blocks[set_idx]
        for way in range(self.associativity):
            if blocks[way] == block:
                return self._dirty[set_idx][way]
        return False

    def mark_dirty(self, block: int) -> bool:
        """Set the dirty bit of a resident block. Returns False if absent."""
        set_idx = block % self.num_sets
        blocks = self._blocks[set_idx]
        for way in range(self.associativity):
            if blocks[way] == block:
                self._dirty[set_idx][way] = True
                return True
        return False

    def clean(self, block: int) -> bool:
        """Clear the dirty bit of a resident block. Returns False if absent."""
        set_idx = block % self.num_sets
        blocks = self._blocks[set_idx]
        for way in range(self.associativity):
            if blocks[way] == block:
                self._dirty[set_idx][way] = False
                return True
        return False

    def lines(self) -> Iterator[tuple[int, int, int, bool]]:
        """Yield ``(set_idx, way, block, dirty)`` for every valid line."""
        for set_idx in range(self.num_sets):
            blocks = self._blocks[set_idx]
            dirty_bits = self._dirty[set_idx]
            for way in range(self.associativity):
                if blocks[way] != EMPTY:
                    yield set_idx, way, blocks[way], dirty_bits[way]

    # -- single-level convenience --------------------------------------------

    def access(self, addr: int, is_write: bool = False) -> bool:
        """Simulate one access at byte address ``addr``: probe, then fill on a miss.

        Returns True on hit, False on miss. Models a single level backed by
        memory: a miss allocates the block, evicting a victim if needed.
        """
        if addr < 0:
            raise ValueError(f"{self.name}: address must be non-negative, got {addr}")
        block = addr // self.block_size
        if self.probe(block, is_write):
            return True
        self.allocate(block, dirty=is_write)
        return False

    # -- 3-C helpers ----------------------------------------------------------

    def _classify_miss(self, block: int) -> None:
        """Assign this miss to compulsory, capacity, or conflict."""
        if block not in self._seen_blocks:
            # Never referenced before: no cache could have held it.
            self.compulsory_misses += 1
        elif block in self._shadow:
            # The fully-associative twin still holds it, so the miss is due
            # to the set mapping, not to capacity.
            self.conflict_misses += 1
        else:
            # Even a fully-associative cache of this capacity had already
            # evicted it: the working set is too large.
            self.capacity_misses += 1

    def _update_shadow(self, block: int) -> None:
        """Feed the same reference to the shadow fully-associative LRU cache."""
        self._seen_blocks.add(block)
        if block in self._shadow:
            self._shadow.move_to_end(block)  # refresh LRU position
        else:
            self._shadow[block] = True
            if len(self._shadow) > self.num_blocks:
                self._shadow.popitem(last=False)  # evict shadow's LRU block

    # -- derived stats ----------------------------------------------------------

    @property
    def accesses(self) -> int:
        return self.hits + self.misses

    @property
    def miss_rate(self) -> float:
        """Local miss rate: misses / accesses that reached this cache."""
        return self.misses / self.accesses if self.accesses else 0.0
