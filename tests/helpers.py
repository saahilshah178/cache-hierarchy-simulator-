"""Shared helpers for the test suite."""

from __future__ import annotations

import os

from cachesim.cache import Cache

#: True when the developer asked to skip the slow tests, as in
#: ``CACHESIM_SKIP_SLOW=1 pytest``. Any value but ``""`` or ``"0"`` skips.
#:
#: Defined here rather than in each module so that every test the flag is
#: meant to cover reads the same switch: a second copy of this expression
#: is how a slow test ends up outside a gate that claims to hold it.
SKIP_SLOW = os.environ.get("CACHESIM_SKIP_SLOW", "") not in ("", "0")


def tiny_cache(ways: int, sets: int = 2, block: int = 16, policy: str = "lru") -> Cache:
    """A deliberately tiny cache: sets*ways blocks of 16 bytes."""
    return Cache("T", size=sets * ways * block, block_size=block, associativity=ways, policy=policy)


def block_addr(cache: Cache, set_idx: int, tag: int) -> int:
    """Build an address that lands in ``set_idx`` with ``tag``."""
    block = tag * cache.num_sets + set_idx
    return block * cache.block_size
