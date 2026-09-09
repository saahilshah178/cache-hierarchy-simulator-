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


class RandomPolicy(ReplacementPolicy):
    """Evict a uniformly random way. No bookkeeping.

    Draws from the cache's seeded RNG, so runs are reproducible.
    """

    def victim(self, set_idx: int) -> int:
        return self.rng.randrange(self.num_ways)


#: Registry of selectable policies, keyed by the name used in configs.
POLICIES: dict[str, type[ReplacementPolicy]] = {
    "lru": LRUPolicy,
    "fifo": FIFOPolicy,
    "random": RandomPolicy,
}
