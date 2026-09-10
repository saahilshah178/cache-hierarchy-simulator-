"""A single, configurable cache level.

One level of a cache (an L1, an L2, ...) is modelled as a grid of
``num_sets`` sets by ``associativity`` ways:

* The cache holds ``size`` bytes, divided into blocks (lines) of
  ``block_size`` bytes. Addresses are converted to block numbers
  (``addr // block_size``) at the boundary; everything inside works on
  block numbers.
* A block maps to exactly one set, chosen by the level's index function
  (``block % num_sets`` by default; see ``cachesim.indexing``), and may
  occupy any way of that set.

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

Optionally misses are classified into the three Cs by running a shadow
fully-associative LRU cache of identical capacity alongside the real one.
Two taxonomies are kept, because they differ:

Per-reference labels (``compulsory_misses``, ``capacity_misses``,
``conflict_misses``) classify each miss as it happens:

* compulsory -- the first reference to this block.
* capacity   -- the shadow cache had already evicted the block.
* conflict   -- the shadow cache still holds the block, so the miss is due
                to the set mapping (or, for non-LRU policies, to the
                replacement choice).

The aggregate decomposition of Hill and Smith (1989), also used by Hennessy
and Patterson, is defined on totals rather than individual references:

* capacity (aggregate) = shadow misses - compulsory misses
* conflict (aggregate) = misses - shadow misses

The two agree unless some reference hits the set-associative cache while
missing the shadow cache (``anti_conflict_hits``): each such reference
adds one to per-reference conflict and subtracts one from per-reference
capacity relative to the aggregate figures, and aggregate conflict can be
negative when the set mapping happens to beat fully-associative LRU.

Dirty tracking implements write-back semantics: a write marks the line
dirty, and evicting or invalidating a dirty line counts as a write-back.
"""

from __future__ import annotations

import random
from collections import OrderedDict
from collections.abc import Callable, Iterator
from typing import NamedTuple

from cachesim.indexing import DEFAULT_INDEX, INDEX_FUNCTIONS, IndexFunction, make_index
from cachesim.policies import POLICIES, ReplacementPolicy

#: Block-number sentinel for an empty way. Real block numbers are >= 0.
EMPTY = -1


class Evicted(NamedTuple):
    """A line removed from a cache by ``allocate`` or ``invalidate``.

    ``prefetched`` marks a line that a prefetcher brought in and that no
    demand reference ever hit: such a line was fetched for nothing and,
    worse, displaced something else.
    """

    block: int
    dirty: bool
    prefetched: bool = False


class VictimBuffer:
    """A small fully-associative LRU buffer beside a cache's tag array.

    Every line the array replaces is pushed in here instead of leaving the
    level, and a reference that misses the array but finds its block here
    takes it back. A handful of entries is enough to hold the blocks that a
    direct-mapped or low-associativity cache keeps throwing at each other,
    so the buffer removes conflict misses without the cost of widening
    every set.

    Jouppi, "Improving Direct-Mapped Cache Performance by the Addition of a
    Small Fully-Associative Cache and Prefetch Buffers", ISCA 1990, which
    reports that four entries remove a large fraction of the conflict
    misses of a direct-mapped cache.
    """

    def __init__(self, entries: int) -> None:
        if isinstance(entries, bool) or not isinstance(entries, int):
            raise ValueError(f"victim buffer entries must be an integer, got {entries!r}")
        if entries < 1:
            raise ValueError(f"victim buffer needs at least one entry, got {entries}")
        self.entries = entries
        # Block number -> line, least recently used first.
        self._lines: OrderedDict[int, Evicted] = OrderedDict()

    def __contains__(self, block: int) -> bool:
        return block in self._lines

    def peek(self, block: int) -> Evicted | None:
        """The buffered line, without disturbing the LRU order."""
        return self._lines.get(block)

    def take(self, block: int) -> Evicted | None:
        """Remove and return the buffered line, or None if it is absent."""
        return self._lines.pop(block, None)

    def push(self, line: Evicted) -> Evicted | None:
        """Insert ``line`` as most recently used.

        Returns the line pushed out of the buffer, which is the one that
        actually leaves the cache level, or None if there was room.
        """
        self._lines[line.block] = line
        self._lines.move_to_end(line.block)
        if len(self._lines) > self.entries:
            _, overflow = self._lines.popitem(last=False)
            return overflow
        return None

    def replace(self, line: Evicted) -> bool:
        """Update a buffered line in place; returns False if it is absent."""
        if line.block not in self._lines:
            return False
        self._lines[line.block] = line
        return True

    def lines(self) -> Iterator[Evicted]:
        """Every buffered line, least recently used first."""
        return iter(list(self._lines.values()))


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
    index         : set-index function name (see ``indexing.INDEX_FUNCTIONS``);
                    "modulo" is the plain low-order-bits mapping.
    victim_entries: size of a fully-associative victim buffer beside the
                    array (see ``VictimBuffer``); None or 0 for none.
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
        index: str = DEFAULT_INDEX,
        victim_entries: int | None = None,
    ) -> None:
        _validate_geometry(name, size, block_size, associativity)

        self.name = name
        self.size = size
        self.block_size = block_size
        self.associativity = associativity
        self.num_sets = size // (block_size * associativity)
        self.num_blocks = size // block_size
        self.policy_name = policy
        self.index_name = index

        # The default mapping is inlined in the primitives below, so plain
        # modulo indexing costs no call at all; anything else goes through
        # the function built here.
        if index not in INDEX_FUNCTIONS:
            raise ValueError(
                f"{name}: unknown index function {index!r}; choose from {sorted(INDEX_FUNCTIONS)}"
            )
        self._index_fn: IndexFunction | None = None
        if index != DEFAULT_INDEX:
            try:
                self._index_fn = make_index(index, self.num_sets)
            except ValueError as exc:
                raise ValueError(f"{name}: {exc}") from None

        if isinstance(rng_seed, bool) or not isinstance(rng_seed, int):
            raise ValueError(f"{name}: rng_seed must be an integer, got {rng_seed!r}")
        rng = random.Random(rng_seed)
        try:
            policy_class = POLICIES[policy]
        except KeyError:
            raise ValueError(
                f"unknown replacement policy {policy!r}; choose from {sorted(POLICIES)}"
            ) from None
        try:
            self.policy: ReplacementPolicy = policy_class(self.num_sets, associativity, rng)
        except ValueError as exc:
            # e.g. plru rejecting a non-power-of-two associativity: name the
            # level so the message identifies which one is misconfigured.
            raise ValueError(f"{name}: {exc}") from None
        # Offline policies (OPT) have to see every lookup, not just its
        # outcome. Bind the hook once here so that probe() pays a single
        # ``is None`` test rather than an attribute lookup and a call.
        self._on_probe: Callable[[int, int], None] | None = (
            self.policy.on_probe if self.policy.sees_references else None
        )

        # Cache contents, indexed [set][way]: the block number held in each
        # way (EMPTY for an invalid way), its dirty bit, and whether a
        # prefetcher installed it and no demand reference has hit it yet.
        self._blocks = [[EMPTY] * associativity for _ in range(self.num_sets)]
        self._dirty = [[False] * associativity for _ in range(self.num_sets)]
        self._prefetched = [[False] * associativity for _ in range(self.num_sets)]

        #: Optional fully-associative buffer of lines the array has replaced.
        self.victim = VictimBuffer(victim_entries) if victim_entries else None
        #: Set by ``probe`` when a victim-buffer hit pushed a line out of the
        #: level; a hierarchy drains it with ``take_pending_eviction``.
        self.pending_eviction: Evicted | None = None
        self.victim_hits = 0  # hits the array missed and the buffer served

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
        self.shadow_misses = 0  # misses of the fully-associative LRU twin
        self.anti_conflict_hits = 0  # hits here that the twin would have missed
        self._seen_blocks: set[int] = set()  # every block number ever touched
        # Shadow fully-associative LRU cache of the same capacity:
        # an OrderedDict of block numbers, oldest first.
        self._shadow: OrderedDict[int, bool] = OrderedDict()

    # -- address arithmetic --------------------------------------------------

    def block_of(self, addr: int) -> int:
        """The block number containing byte address ``addr``."""
        return addr // self.block_size

    def set_of(self, block: int) -> int:
        """The set index a block maps to, under this level's index function."""
        index = self._index_fn
        return block % self.num_sets if index is None else index(block)

    def tag_of(self, block: int) -> int:
        """The tag that distinguishes blocks sharing a set.

        Only meaningful for the default modulo mapping: a hashed index is
        not a bit-field of the block number, so hardware using one stores
        the whole block number (or a wider tag) instead.
        """
        return block // self.num_sets

    # -- primitives ------------------------------------------------------------

    def probe(self, block: int, is_write: bool = False) -> bool:
        """Look ``block`` up. Returns True on hit, False on miss.

        Updates hit/miss statistics, replacement state, the dirty bit on a
        write hit, and the 3-C classification on a miss. Does not fill: on a
        miss the caller decides whether and how to ``allocate``.

        A victim buffer is part of the level, and probed with the array: a
        block found there counts as a hit (and as a ``victim_hits``), and is
        swapped back into the array immediately, which may push a line out
        of the level into ``pending_eviction``.
        """
        index = self._index_fn
        set_idx = block % self.num_sets if index is None else index(block)
        on_probe = self._on_probe
        if on_probe is not None:
            on_probe(set_idx, block)
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
                if self.track_3c and not self._update_shadow(block):
                    self.anti_conflict_hits += 1
                return True

        if self.victim is not None:
            line = self.victim.take(block)
            if line is not None:
                self.hits += 1
                self.victim_hits += 1
                if is_write:
                    self.write_hits += 1
                else:
                    self.read_hits += 1
                # Swap: the block goes back into the array and the line it
                # displaces takes its place in the buffer.
                self.pending_eviction = self.allocate(
                    block, dirty=line.dirty or is_write, prefetched=line.prefetched
                )
                if self.track_3c and not self._update_shadow(block):
                    self.anti_conflict_hits += 1
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

    def allocate(self, block: int, dirty: bool = False, prefetched: bool = False) -> Evicted | None:
        """Install ``block``, evicting a victim if its set is full.

        Returns the evicted line, or None if an empty way was used. If the
        block is already present this is a no-op (the dirty bit is OR-ed in)
        and None is returned.

        ``prefetched`` marks the line as speculatively fetched; the flag is
        cleared by ``clear_prefetched`` on the first demand hit, and travels
        with the line into the ``Evicted`` record if it leaves unused.

        With a victim buffer the line the array replaces does not leave the
        level: it is pushed into the buffer, and what this returns is the
        line the buffer pushed out, if any.
        """
        index = self._index_fn
        set_idx = block % self.num_sets if index is None else index(block)
        blocks = self._blocks[set_idx]
        dirty_bits = self._dirty[set_idx]
        prefetched_bits = self._prefetched[set_idx]

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
            evicted = Evicted(blocks[way], dirty_bits[way], prefetched_bits[way])
            if self.victim is not None:
                evicted = self.victim.push(evicted)
            if evicted is not None and evicted.dirty:
                self.writebacks += 1

        blocks[way] = block
        dirty_bits[way] = dirty
        prefetched_bits[way] = prefetched
        self.fills += 1
        self.policy.on_fill(set_idx, way, block)
        return evicted

    def invalidate(self, block: int, count_writeback: bool = True) -> Evicted | None:
        """Remove ``block`` if present. Returns the removed line, else None.

        Counts as an invalidation, not an eviction; a dirty line also counts
        as a write-back, because its data must be written down.

        ``count_writeback=False`` suppresses that write-back count, for the
        one case in which a dirty line leaves a cache without being written
        towards memory: an exclusive hierarchy moving the line *up* into the
        level above, which takes the dirty bit with it.
        """
        set_idx = self.set_of(block)
        blocks = self._blocks[set_idx]
        for way in range(self.associativity):
            if blocks[way] == block:
                removed = Evicted(block, self._dirty[set_idx][way], self._prefetched[set_idx][way])
                blocks[way] = EMPTY
                self._dirty[set_idx][way] = False
                self._prefetched[set_idx][way] = False
                self.invalidations += 1
                if removed.dirty and count_writeback:
                    self.writebacks += 1
                self.policy.on_invalidate(set_idx, way)
                return removed
        if self.victim is not None:
            buffered = self.victim.take(block)
            if buffered is not None:
                self.invalidations += 1
                if buffered.dirty and count_writeback:
                    self.writebacks += 1
                return buffered
        return None

    def take_pending_eviction(self) -> Evicted | None:
        """Return and clear the line a victim-buffer swap pushed out.

        ``probe`` sets it when a hit in the victim buffer put a block back
        into the array and the line it displaced did not fit in the buffer.
        A hierarchy drains it after every probe so the line can be written
        back or handed to the level below.
        """
        pending = self.pending_eviction
        self.pending_eviction = None
        return pending

    def contains(self, block: int) -> bool:
        """True if ``block`` is currently resident (no statistics updated).

        A block sitting in the victim buffer is resident: it is still in
        this level and a reference to it will hit.
        """
        if block in self._blocks[self.set_of(block)]:
            return True
        return self.victim is not None and block in self.victim

    def is_dirty(self, block: int) -> bool:
        """True if ``block`` is resident and dirty."""
        set_idx = self.set_of(block)
        blocks = self._blocks[set_idx]
        for way in range(self.associativity):
            if blocks[way] == block:
                return self._dirty[set_idx][way]
        if self.victim is not None:
            line = self.victim.peek(block)
            if line is not None:
                return line.dirty
        return False

    def mark_dirty(self, block: int) -> bool:
        """Set the dirty bit of a resident block. Returns False if absent."""
        set_idx = self.set_of(block)
        blocks = self._blocks[set_idx]
        for way in range(self.associativity):
            if blocks[way] == block:
                self._dirty[set_idx][way] = True
                return True
        if self.victim is not None:
            line = self.victim.peek(block)
            if line is not None:
                return self.victim.replace(line._replace(dirty=True))
        return False

    def is_prefetched(self, block: int) -> bool:
        """True if ``block`` is resident and no demand reference has hit it
        since a prefetcher installed it."""
        set_idx = self.set_of(block)
        blocks = self._blocks[set_idx]
        for way in range(self.associativity):
            if blocks[way] == block:
                return self._prefetched[set_idx][way]
        if self.victim is not None:
            line = self.victim.peek(block)
            if line is not None:
                return line.prefetched
        return False

    def clear_prefetched(self, block: int) -> bool:
        """Clear ``block``'s prefetched flag; returns whether it was set.

        A True return is a prefetch that paid off: this is the first demand
        reference to reach a line the prefetcher fetched.
        """
        set_idx = self.set_of(block)
        blocks = self._blocks[set_idx]
        prefetched_bits = self._prefetched[set_idx]
        for way in range(self.associativity):
            if blocks[way] == block:
                was_set = prefetched_bits[way]
                prefetched_bits[way] = False
                return was_set
        if self.victim is not None:
            line = self.victim.peek(block)
            if line is not None and line.prefetched:
                self.victim.replace(line._replace(prefetched=False))
                return True
        return False

    def clean(self, block: int) -> bool:
        """Clear the dirty bit of a resident block. Returns False if absent."""
        set_idx = self.set_of(block)
        blocks = self._blocks[set_idx]
        for way in range(self.associativity):
            if blocks[way] == block:
                self._dirty[set_idx][way] = False
                return True
        if self.victim is not None:
            line = self.victim.peek(block)
            if line is not None:
                return self.victim.replace(line._replace(dirty=False))
        return False

    def lines(self) -> Iterator[tuple[int, int, int, bool]]:
        """Yield ``(set_idx, way, block, dirty)`` for every valid line.

        Lines held in the victim buffer are yielded too, with a set index
        and way of -1, since they are as resident as any other.
        """
        for set_idx in range(self.num_sets):
            blocks = self._blocks[set_idx]
            dirty_bits = self._dirty[set_idx]
            for way in range(self.associativity):
                if blocks[way] != EMPTY:
                    yield set_idx, way, blocks[way], dirty_bits[way]
        if self.victim is not None:
            for line in self.victim.lines():
                yield -1, -1, line.block, line.dirty

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

    def _update_shadow(self, block: int) -> bool:
        """Feed the same reference to the shadow fully-associative LRU cache.

        Returns True if the shadow cache hit.
        """
        self._seen_blocks.add(block)
        if block in self._shadow:
            self._shadow.move_to_end(block)  # refresh LRU position
            return True
        self.shadow_misses += 1
        self._shadow[block] = True
        if len(self._shadow) > self.num_blocks:
            self._shadow.popitem(last=False)  # evict shadow's LRU block
        return False

    def reset_stats(self) -> None:
        """Zero every counter, keeping the contents, replacement state, and
        the set of blocks already seen (so a later reference to a block
        touched before the reset is not counted as compulsory)."""
        self.hits = self.misses = 0
        self.read_hits = self.read_misses = 0
        self.write_hits = self.write_misses = 0
        self.fills = self.evictions = self.invalidations = 0
        self.writebacks = self.writebacks_received = self.writeback_allocations = 0
        self.compulsory_misses = self.capacity_misses = self.conflict_misses = 0
        self.shadow_misses = self.anti_conflict_hits = 0
        self.victim_hits = 0

    # -- derived stats ----------------------------------------------------------

    @property
    def accesses(self) -> int:
        return self.hits + self.misses

    @property
    def capacity_misses_aggregate(self) -> int:
        """Hill-Smith capacity misses: shadow (fully-associative LRU) misses
        that are not compulsory."""
        return self.shadow_misses - self.compulsory_misses

    @property
    def conflict_misses_aggregate(self) -> int:
        """Hill-Smith conflict misses: misses beyond those of the
        fully-associative LRU twin. Negative if the set mapping did better."""
        return self.misses - self.shadow_misses

    @property
    def miss_rate(self) -> float:
        """Local miss rate: misses / accesses that reached this cache."""
        return self.misses / self.accesses if self.accesses else 0.0
