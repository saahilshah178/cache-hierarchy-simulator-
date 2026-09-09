"""Shared helpers for the test suite."""

from __future__ import annotations

from cachesim.cache import Cache


def tiny_cache(ways: int, sets: int = 2, block: int = 16, policy: str = "lru") -> Cache:
    """A deliberately tiny cache: sets*ways blocks of 16 bytes."""
    return Cache("T", size=sets * ways * block, block_size=block,
                 associativity=ways, policy=policy)


def block_addr(cache: Cache, set_idx: int, tag: int) -> int:
    """Build an address that lands in ``set_idx`` with ``tag``."""
    block = tag * cache.num_sets + set_idx
    return block * cache.block_size
