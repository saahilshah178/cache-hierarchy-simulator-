"""Known-answer tests for replacement policies."""

from __future__ import annotations

import unittest
from unittest import mock

from cachesim.policies import POLICIES, ReplacementPolicy
from tests.helpers import block_addr, tiny_cache


class TestReplacementPolicies(unittest.TestCase):
    def setUp(self) -> None:
        # Three blocks (a, b, x) all in set 0 of a 2-way cache: exactly one
        # of them must be evicted when the third arrives.
        self.c_lru = tiny_cache(ways=2, policy="lru")
        self.c_fifo = tiny_cache(ways=2, policy="fifo")
        self.a = block_addr(self.c_lru, 0, 1)
        self.b = block_addr(self.c_lru, 0, 2)
        self.x = block_addr(self.c_lru, 0, 3)

    def test_lru_keeps_the_recently_hit_block(self) -> None:
        c = self.c_lru
        c.access(self.a)                       # fill a
        c.access(self.b)                       # fill b
        c.access(self.a)                       # touch a: b is now LRU
        c.access(self.x)                       # evicts b
        self.assertTrue(c.access(self.a))      # a survived
        self.assertFalse(c.access(self.b))     # b was the victim

    def test_fifo_evicts_the_oldest_fill_even_if_hot(self) -> None:
        c = self.c_fifo
        c.access(self.a)                       # fill a (oldest)
        c.access(self.b)                       # fill b
        c.access(self.a)                       # hit does not refresh FIFO age
        c.access(self.x)                       # evicts a anyway
        # Check b first: probing a would refill it and evict b (FIFO's next
        # victim), polluting the second check.
        self.assertTrue(c.access(self.b))      # b survived (x still cached)
        self.assertFalse(c.access(self.a))     # a was the victim

    def test_random_is_reproducible(self) -> None:
        def run() -> tuple[int, int]:
            c = tiny_cache(ways=2, policy="random")
            for tag in (1, 2, 3, 1, 2, 3, 1):
                c.access(block_addr(c, 0, tag))
            return (c.hits, c.misses)
        self.assertEqual(run(), run())         # same seed, same outcome

    def test_policies_are_extensible(self) -> None:
        """A subclass plus a registry entry is all a new policy needs."""
        class EvictWayZero(ReplacementPolicy):
            def victim(self, set_idx: int) -> int:
                return 0

        with mock.patch.dict(POLICIES, {"way0": EvictWayZero}):
            c = tiny_cache(ways=2, policy="way0")
            c.access(block_addr(c, 0, 1))      # fills way 0
            c.access(block_addr(c, 0, 2))      # fills way 1
            c.access(block_addr(c, 0, 3))      # evicts way 0 (tag 1)
            self.assertTrue(c.access(block_addr(c, 0, 2)))
            self.assertFalse(c.access(block_addr(c, 0, 1)))
        self.assertNotIn("way0", POLICIES)


if __name__ == "__main__":
    unittest.main(verbosity=2)
