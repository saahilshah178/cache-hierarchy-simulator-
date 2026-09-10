"""Replacement policies for a set-associative cache.

A policy decides which way of a full set to evict. Every policy keeps its own
per-set bookkeeping and is driven by hooks that ``Cache`` calls:

* ``on_hit(set_idx, way, block)``  -- an access hit this way.
* ``on_fill(set_idx, way, block)`` -- ``block`` was installed into this way.
* ``on_invalidate(set_idx, way)``  -- the way was emptied on command.
* ``victim(set_idx)``              -- the set is full; return the way to evict.

``victim`` is only called when every way of the set holds valid data: empty
ways (including ways emptied by ``on_invalidate``) are always filled first,
so policies never see invalid ways.

Policies are selected by name through the ``POLICIES`` registry.

Two families live here:

* recency and frequency approximations that real hardware implements --
  ``lru``, ``fifo``, ``plru`` (tree pseudo-LRU), ``nru``, and ``lfu``;
* deliberate contrasts used to bound behaviour -- ``random`` and ``mru``.
"""

from __future__ import annotations

import random


class ReplacementPolicy:
    """Base class. Tracks nothing; subclasses add per-set state."""

    def __init__(self, num_sets: int, num_ways: int, rng: random.Random | None = None) -> None:
        self.num_sets = num_sets
        self.num_ways = num_ways
        self.rng = rng if rng is not None else random.Random(0)

    def on_hit(self, set_idx: int, way: int, block: int) -> None:
        """Called every time an access hits ``way`` in ``set_idx``."""

    def on_fill(self, set_idx: int, way: int, block: int) -> None:
        """Called when ``block`` is installed into ``way`` in ``set_idx``."""

    def on_invalidate(self, set_idx: int, way: int) -> None:
        """Called when ``way`` in ``set_idx`` is emptied by an invalidation."""

    def victim(self, set_idx: int) -> int:
        """Return the way to evict from a completely full set."""
        raise NotImplementedError


class _RecencyOrder(ReplacementPolicy):
    """Shared machinery: each set keeps its ways in an ordered list, victim
    at the front. Subclasses decide which events move a way to the back."""

    def __init__(self, num_sets: int, num_ways: int, rng: random.Random | None = None) -> None:
        super().__init__(num_sets, num_ways, rng)
        self._order = [list(range(num_ways)) for _ in range(num_sets)]

    def _to_back(self, set_idx: int, way: int) -> None:
        order = self._order[set_idx]
        order.remove(way)
        order.append(way)

    def _to_front(self, set_idx: int, way: int) -> None:
        order = self._order[set_idx]
        order.remove(way)
        order.insert(0, way)

    def on_fill(self, set_idx: int, way: int, block: int) -> None:
        self._to_back(set_idx, way)

    def on_invalidate(self, set_idx: int, way: int) -> None:
        # An emptied way is refilled before any victim is chosen; keeping it
        # at the front leaves the relative order of the valid ways intact.
        self._to_front(set_idx, way)

    def victim(self, set_idx: int) -> int:
        return self._order[set_idx][0]


class LRUPolicy(_RecencyOrder):
    """Least Recently Used: evict the way untouched for the longest time.

    Hits and fills move the way to the most-recent end; the victim is the
    way at the least-recent front.
    """

    def on_hit(self, set_idx: int, way: int, block: int) -> None:
        self._to_back(set_idx, way)


class FIFOPolicy(_RecencyOrder):
    """First In, First Out: evict the way that was filled longest ago.

    Hits do not refresh a block's position; only being loaded does.
    """


class MRUPolicy(_RecencyOrder):
    """Most Recently Used: evict the way touched last.

    The adversarial contrast to LRU. On a cyclic scan whose working set is
    one block larger than the set, LRU evicts exactly the block that is
    about to be referenced and misses on every access, while MRU sacrifices
    the same way over and over and keeps the rest of the set resident. It is
    not a serious hardware candidate; it is here to show that "recency is
    good" is a property of the reference stream, not a law, and it is the
    cheapest stand-in for Belady's rule on cyclic patterns.
    """

    def on_hit(self, set_idx: int, way: int, block: int) -> None:
        self._to_back(set_idx, way)

    def victim(self, set_idx: int) -> int:
        return self._order[set_idx][-1]


class RandomPolicy(ReplacementPolicy):
    """Evict a uniformly random way. No bookkeeping.

    Draws from the cache's seeded RNG, so runs are reproducible.
    """

    def victim(self, set_idx: int) -> int:
        return self.rng.randrange(self.num_ways)


class TreePLRUPolicy(ReplacementPolicy):
    """Tree pseudo-LRU: ``num_ways - 1`` bits per set arranged as a binary tree.

    The bits form a complete binary tree stored as a heap: node 0 is the
    root and the children of node ``n`` are ``2n+1`` (left) and ``2n+2``
    (right). A bit points at the half of its subtree that holds the victim:
    0 means "the victim is on the left", 1 means "on the right".

    * touching a way (hit or fill) walks root to leaf along the path to that
      way and sets every bit on the path to point *away* from it;
    * ``victim`` walks root to leaf following the bits.

    The result is exact LRU for two ways and an approximation beyond that:
    one bit per tree node cannot record the full order of a set, so the
    victim is only guaranteed to be one of the ways in the less recently
    used half at every level. It costs ``num_ways - 1`` bits per set instead
    of the ``num_ways * log2(num_ways)`` bits of true LRU, which is why it is
    what real L1/L2 caches implement (see Hennessy and Patterson,
    "Computer Architecture: A Quantitative Approach", 6th ed., 2017, sec.
    B.1 and 2.3).

    Requires a power-of-two associativity, as the tree must be complete.
    """

    def __init__(self, num_sets: int, num_ways: int, rng: random.Random | None = None) -> None:
        super().__init__(num_sets, num_ways, rng)
        if num_ways & (num_ways - 1):
            raise ValueError(f"plru requires a power-of-two associativity, got {num_ways}")
        self._depth = num_ways.bit_length() - 1  # tree levels = log2(num_ways)
        self._tree = [0] * num_sets  # num_ways-1 bits per set, packed into an int

    def _walk(self, set_idx: int, way: int, toward: bool) -> None:
        """Point every bit on the root-to-leaf path at ``way`` toward or away from it."""
        depth = self._depth
        tree = self._tree[set_idx]
        node = 0
        for level in range(depth):
            # Descend using the way index MSB first: bit ``level`` of the
            # path is bit ``depth-1-level`` of the way number.
            right = (way >> (depth - 1 - level)) & 1
            if bool(right) is toward:
                tree |= 1 << node  # point right
            else:
                tree &= ~(1 << node)  # point left
            node = 2 * node + 1 + right
        self._tree[set_idx] = tree

    def on_hit(self, set_idx: int, way: int, block: int) -> None:
        self._walk(set_idx, way, toward=False)

    def on_fill(self, set_idx: int, way: int, block: int) -> None:
        self._walk(set_idx, way, toward=False)

    def on_invalidate(self, set_idx: int, way: int) -> None:
        # Aim the tree at the way that was just emptied. The cache fills
        # empty ways before asking for a victim, so this only matters for
        # keeping the state meaningful, but it is what hardware does: the
        # invalid way is the one you want back first.
        self._walk(set_idx, way, toward=True)

    def victim(self, set_idx: int) -> int:
        tree = self._tree[set_idx]
        node = 0
        way = 0
        for _ in range(self._depth):
            right = (tree >> node) & 1
            way = (way << 1) | right
            node = 2 * node + 1 + right
        return way


class NRUPolicy(ReplacementPolicy):
    """Not Recently Used: one reference bit per way.

    A fill or a hit sets the way's bit. The victim is the lowest-numbered
    way whose bit is clear; if every bit is set, all of them are cleared and
    way 0 is evicted, which is the same choice the scan would make on the
    second pass.

    This is the coarsest useful recency approximation -- one bit per way
    against LRU's full order -- and is what large, highly associative
    last-level caches and TLBs use, sometimes under the name "clock" or
    "second chance" (Silberschatz, Galvin and Gagne, "Operating System
    Concepts", 10th ed., 2018, sec. 10.4.5).
    """

    def __init__(self, num_sets: int, num_ways: int, rng: random.Random | None = None) -> None:
        super().__init__(num_sets, num_ways, rng)
        self._referenced = [[False] * num_ways for _ in range(num_sets)]

    def on_hit(self, set_idx: int, way: int, block: int) -> None:
        self._referenced[set_idx][way] = True

    def on_fill(self, set_idx: int, way: int, block: int) -> None:
        self._referenced[set_idx][way] = True

    def on_invalidate(self, set_idx: int, way: int) -> None:
        self._referenced[set_idx][way] = False

    def victim(self, set_idx: int) -> int:
        bits = self._referenced[set_idx]
        for way in range(self.num_ways):
            if not bits[way]:
                return way
        for way in range(self.num_ways):  # every way claims to be recent:
            bits[way] = False  # forget everything and start again
        return 0


class LFUPolicy(ReplacementPolicy):
    """Least Frequently Used: evict the way with the fewest references.

    A fill starts the counter at 1 (the reference that caused the fill) and
    each hit adds one; ties are broken by the lowest way number. The
    counters do not age, so a block that was hot early keeps its score for
    as long as it stays resident -- the classic weakness of plain LFU, which
    real designs fix by halving all counters periodically or by aging
    (Silberschatz, Galvin and Gagne, "Operating System Concepts", 10th ed.,
    2018, sec. 10.4.4). Nothing here ages, so ``lfu`` is a faithful model of
    the textbook policy, cache pollution included.
    """

    def __init__(self, num_sets: int, num_ways: int, rng: random.Random | None = None) -> None:
        super().__init__(num_sets, num_ways, rng)
        self._count = [[0] * num_ways for _ in range(num_sets)]

    def on_hit(self, set_idx: int, way: int, block: int) -> None:
        self._count[set_idx][way] += 1

    def on_fill(self, set_idx: int, way: int, block: int) -> None:
        self._count[set_idx][way] = 1

    def on_invalidate(self, set_idx: int, way: int) -> None:
        self._count[set_idx][way] = 0

    def victim(self, set_idx: int) -> int:
        counts = self._count[set_idx]
        best = 0
        best_count = counts[0]
        for way in range(1, self.num_ways):
            if counts[way] < best_count:
                best = way
                best_count = counts[way]
        return best


#: Registry of selectable policies, keyed by the name used in configs.
POLICIES: dict[str, type[ReplacementPolicy]] = {
    "lru": LRUPolicy,
    "fifo": FIFOPolicy,
    "plru": TreePLRUPolicy,
    "nru": NRUPolicy,
    "lfu": LFUPolicy,
    "mru": MRUPolicy,
    "random": RandomPolicy,
}
