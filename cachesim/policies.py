"""Replacement policies for a set-associative cache.

A policy decides which way of a full set to evict. Every policy keeps its own
per-set bookkeeping and is driven by three hooks that ``Cache`` calls:

* ``on_hit(set_idx, way)``  -- an access hit this way.
* ``on_fill(set_idx, way)`` -- a new block was installed into this way.
* ``victim(set_idx)``       -- the set is full; return the way to evict.

``victim`` is only called when every way of the set holds valid data: empty
ways are always filled first, so policies never see invalid ways.

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

    def on_hit(self, set_idx: int, way: int) -> None:
        """Called every time an access hits ``way`` in ``set_idx``."""

    def on_fill(self, set_idx: int, way: int) -> None:
        """Called when a new block is installed into ``way`` in ``set_idx``."""

    def victim(self, set_idx: int) -> int:
        """Return the way to evict from a completely full set."""
        raise NotImplementedError


class LRUPolicy(ReplacementPolicy):
    """Least Recently Used: evict the way untouched for the longest time.

    Each set keeps its ways in a list ordered least- to most-recently used.
    Hits and fills move the way to the most-recent end; the victim is the
    way at the least-recent front.
    """

    def __init__(self, num_sets: int, num_ways: int, rng: random.Random | None = None) -> None:
        super().__init__(num_sets, num_ways, rng)
        self._order = [list(range(num_ways)) for _ in range(num_sets)]

    def _touch(self, set_idx: int, way: int) -> None:
        order = self._order[set_idx]
        order.remove(way)
        order.append(way)

    on_hit = _touch
    on_fill = _touch

    def victim(self, set_idx: int) -> int:
        return self._order[set_idx][0]


class FIFOPolicy(ReplacementPolicy):
    """First In, First Out: evict the way that was filled longest ago.

    Hits do not refresh a block's position; only being loaded does.
    """

    def __init__(self, num_sets: int, num_ways: int, rng: random.Random | None = None) -> None:
        super().__init__(num_sets, num_ways, rng)
        self._order = [list(range(num_ways)) for _ in range(num_sets)]

    def on_fill(self, set_idx: int, way: int) -> None:
        order = self._order[set_idx]
        order.remove(way)
        order.append(way)

    def victim(self, set_idx: int) -> int:
        return self._order[set_idx][0]


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
