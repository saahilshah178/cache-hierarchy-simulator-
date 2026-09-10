"""Known-answer tests for replacement policies."""

from __future__ import annotations

import random
import unittest
from unittest import mock

from cachesim.cache import Cache
from cachesim.hierarchy import Hierarchy, Level
from cachesim.policies import POLICIES, DRRIPPolicy, ReplacementPolicy
from tests.helpers import block_addr, tiny_cache


def one_set_cache(ways: int, policy: str, block: int = 16) -> Cache:
    """A cache that is a single set: every block competes with every other."""
    return Cache("S", size=ways * block, block_size=block, associativity=ways, policy=policy)


def resident_blocks(cache: Cache) -> set[int]:
    """The block numbers currently held, ignoring where they sit."""
    return {block for _, _, block, _ in cache.lines()}


def cyclic_scan(cache: Cache, num_blocks: int, passes: int) -> list[int]:
    """Reference blocks 0..num_blocks-1 in order, ``passes`` times.

    Returns the miss count of each pass.
    """
    per_pass = []
    for _ in range(passes):
        before = cache.misses
        for block in range(num_blocks):
            cache.access(block * cache.block_size)
        per_pass.append(cache.misses - before)
    return per_pass


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
        c.access(self.a)  # fill a
        c.access(self.b)  # fill b
        c.access(self.a)  # touch a: b is now LRU
        c.access(self.x)  # evicts b
        self.assertTrue(c.access(self.a))  # a survived
        self.assertFalse(c.access(self.b))  # b was the victim

    def test_fifo_evicts_the_oldest_fill_even_if_hot(self) -> None:
        c = self.c_fifo
        c.access(self.a)  # fill a (oldest)
        c.access(self.b)  # fill b
        c.access(self.a)  # hit does not refresh FIFO age
        c.access(self.x)  # evicts a anyway
        # Check b first: probing a would refill it and evict b (FIFO's next
        # victim), polluting the second check.
        self.assertTrue(c.access(self.b))  # b survived (x still cached)
        self.assertFalse(c.access(self.a))  # a was the victim

    def test_random_is_reproducible(self) -> None:
        def run() -> tuple[int, int]:
            c = tiny_cache(ways=2, policy="random")
            for tag in (1, 2, 3, 1, 2, 3, 1):
                c.access(block_addr(c, 0, tag))
            return (c.hits, c.misses)

        self.assertEqual(run(), run())  # same seed, same outcome

    def test_policies_are_extensible(self) -> None:
        """A subclass plus a registry entry is all a new policy needs."""

        class EvictWayZero(ReplacementPolicy):
            def victim(self, set_idx: int) -> int:
                return 0

        with mock.patch.dict(POLICIES, {"way0": EvictWayZero}):
            c = tiny_cache(ways=2, policy="way0")
            c.access(block_addr(c, 0, 1))  # fills way 0
            c.access(block_addr(c, 0, 2))  # fills way 1
            c.access(block_addr(c, 0, 3))  # evicts way 0 (tag 1)
            self.assertTrue(c.access(block_addr(c, 0, 2)))
            self.assertFalse(c.access(block_addr(c, 0, 1)))
        self.assertNotIn("way0", POLICIES)


class TestTreePLRU(unittest.TestCase):
    """Tree pseudo-LRU: num_ways-1 bits per set, victim follows the bits."""

    def test_plru_and_lru_disagree_on_the_classic_four_way_sequence(self) -> None:
        # 4-way single set, references a, b, c, d, a, e. True LRU evicts b
        # (untouched longest). PLRU evicts c: touching a flips the root to
        # point at the {c,d} half and the {c,d} bit still points at c, so
        # the fact that c was used more recently than b is not recorded.
        sequence = [1, 2, 3, 4, 1, 5]  # tags a, b, c, d, a, e
        plru = one_set_cache(ways=4, policy="plru")
        lru = one_set_cache(ways=4, policy="lru")
        for tag in sequence:
            plru.access(tag * plru.block_size)
            lru.access(tag * lru.block_size)
        self.assertEqual(resident_blocks(plru), {1, 2, 4, 5})  # c (3) evicted
        self.assertEqual(resident_blocks(lru), {1, 3, 4, 5})  # b (2) evicted
        self.assertEqual((plru.hits, plru.misses), (1, 5))
        self.assertEqual((lru.hits, lru.misses), (1, 5))

    def test_plru_fills_ways_in_tree_order_from_the_reset_state(self) -> None:
        # All bits clear means "victim to the left" at every node, so the
        # first victim of a cold, full set is way 0.
        c = one_set_cache(ways=4, policy="plru")
        for tag in (1, 2, 3, 4):  # fills ways 0..3 in order
            c.access(tag * c.block_size)
        # Filling way 3 last leaves the root pointing at the {0,1} half and
        # that half pointing at way 0.
        self.assertEqual(c.policy.victim(0), 0)

    def test_plru_is_exact_lru_at_two_ways(self) -> None:
        # One bit per set is enough to record the full order of two ways, so
        # tree-PLRU degenerates to true LRU. Cross-check on a random stream.
        rng = random.Random(11)
        stream = [rng.randrange(6) for _ in range(400)]
        plru = Cache("P", size=8 * 16, block_size=16, associativity=2, policy="plru")
        lru = Cache("L", size=8 * 16, block_size=16, associativity=2, policy="lru")
        for block in stream:
            plru.access(block * 16)
            lru.access(block * 16)
        self.assertEqual(plru.misses, lru.misses)
        self.assertEqual(resident_blocks(plru), resident_blocks(lru))

    def test_plru_rejects_a_non_power_of_two_associativity(self) -> None:
        with self.assertRaises(ValueError) as ctx:
            Cache("L1", size=3 * 16, block_size=16, associativity=3, policy="plru")
        self.assertIn("power-of-two", str(ctx.exception))

    def test_plru_points_at_an_invalidated_way(self) -> None:
        c = one_set_cache(ways=4, policy="plru")
        for tag in (1, 2, 3, 4):
            c.access(tag * c.block_size)
        c.invalidate(3)  # empties way 2
        self.assertEqual(c.policy.victim(0), 2)


class TestNRU(unittest.TestCase):
    """One reference bit per way; evict the lowest way whose bit is clear."""

    def test_nru_evicts_the_recently_used_block_when_every_bit_is_set(self) -> None:
        # 2-way, references a, b, a. Both bits are set (a by its hit, b by
        # its fill), so c clears them and takes way 0, which holds a. LRU
        # would have evicted b.
        c = one_set_cache(ways=2, policy="nru")
        for tag in (1, 2, 1, 3):
            c.access(tag * c.block_size)
        self.assertEqual(resident_blocks(c), {2, 3})  # a (1) was the victim
        self.assertEqual((c.hits, c.misses), (1, 3))

    def test_nru_prefers_a_way_whose_bit_is_clear(self) -> None:
        # After the sweep that cleared every bit, only the refilled way is
        # marked, so the next victim is the lowest unmarked way.
        c = one_set_cache(ways=4, policy="nru")
        for tag in (1, 2, 3, 4, 5):  # the fifth fill clears all bits, takes way 0
            c.access(tag * c.block_size)
        self.assertEqual(c.policy.victim(0), 1)  # way 0 was just re-referenced
        c.access(2 * c.block_size)  # hit in way 1: sets its bit again
        self.assertEqual(c.policy.victim(0), 2)

    def test_nru_clears_the_bit_of_an_invalidated_way(self) -> None:
        c = one_set_cache(ways=2, policy="nru")
        c.access(1 * c.block_size)
        c.access(2 * c.block_size)
        c.invalidate(1)  # way 0
        self.assertEqual(c.policy.victim(0), 0)


class TestLFU(unittest.TestCase):
    """Per-way reference counters, no aging."""

    def test_lfu_evicts_the_least_referenced_block(self) -> None:
        # 2-way: a is referenced three times (count 3), b once (count 1), so
        # b is the victim even though it is the more recent of the two.
        c = one_set_cache(ways=2, policy="lfu")
        for tag in (1, 1, 1, 2, 3):
            c.access(tag * c.block_size)
        self.assertEqual(resident_blocks(c), {1, 3})  # b (2) was the victim
        self.assertEqual((c.hits, c.misses), (2, 3))

    def test_lfu_breaks_count_ties_with_the_lowest_way(self) -> None:
        c = one_set_cache(ways=4, policy="lfu")
        for tag in (1, 2, 3, 4):  # every count is 1
            c.access(tag * c.block_size)
        self.assertEqual(c.policy.victim(0), 0)

    def test_lfu_counters_do_not_age(self) -> None:
        # The classic weakness: a block that was hot long ago keeps its
        # score and cannot be displaced by a stream of one-touch blocks.
        c = one_set_cache(ways=2, policy="lfu")
        hot = 1
        for _ in range(5):
            c.access(hot * c.block_size)  # count 5
        for tag in range(2, 20):  # each newcomer arrives with count 1
            c.access(tag * c.block_size)
        self.assertIn(hot, resident_blocks(c))


class TestMRU(unittest.TestCase):
    """Evict the most recently used way: the cyclic-scan counter-example."""

    def test_mru_beats_lru_on_a_cyclic_scan_one_block_too_large(self) -> None:
        # C+1 distinct blocks scanned repeatedly over a C-way set. LRU
        # evicts exactly the block that is referenced next, so every access
        # after the first pass misses. MRU keeps C-1 blocks pinned and
        # sacrifices the same way, so it misses about once per pass.
        ways, passes = 4, 10
        blocks = ways + 1
        lru = cyclic_scan(one_set_cache(ways, "lru"), blocks, passes)
        mru = cyclic_scan(one_set_cache(ways, "mru"), blocks, passes)
        self.assertEqual(lru, [5] * passes)  # 100% miss, every pass
        self.assertEqual(mru, [5, 1, 1, 1, 2, 1, 1, 1, 2, 1])
        self.assertEqual((sum(lru), sum(mru)), (50, 16))

    def test_mru_cyclic_scan_matches_its_closed_form(self) -> None:
        # Measured: after the cold pass, MRU misses once per pass except
        # every C-th pass, which misses twice, i.e.
        #     misses(P) = (C+1) + (P-1) + floor((P-1)/C).
        for ways in (2, 3, 4, 8):
            for passes in (1, 2, 5, 9, 10, 17):
                measured = sum(cyclic_scan(one_set_cache(ways, "mru"), ways + 1, passes))
                predicted = (ways + 1) + (passes - 1) + (passes - 1) // ways
                self.assertEqual(measured, predicted, f"ways={ways} passes={passes}")


class TestRRIP(unittest.TestCase):
    """Re-reference interval prediction (Jaleel et al., ISCA 2010)."""

    WAYS = 4
    SCAN_PER_ROUND = 3
    ROUNDS = 10

    def scan_rounds(self, policy: str) -> tuple[Cache, list[int]]:
        """Two hot blocks touched twice, then three one-touch scan blocks.

        Repeated ``ROUNDS`` times over a single 4-way set, with fresh scan
        blocks every round. The hot pair fits in the set with two ways to
        spare, so keeping them is the right decision and every miss on them
        is a policy failure.
        """
        cache = one_set_cache(self.WAYS, policy)
        block_size = cache.block_size
        scan_block = 100
        per_round = []
        for _ in range(self.ROUNDS):
            before = cache.misses
            for _ in range(2):
                cache.access(1 * block_size)
                cache.access(2 * block_size)
            for _ in range(self.SCAN_PER_ROUND):
                cache.access(scan_block * block_size)
                scan_block += 1
            per_round.append(cache.misses - before)
        return cache, per_round

    def test_srrip_survives_a_scan_that_flushes_lru(self) -> None:
        # LRU installs every scan block as most recently used, so the hot
        # pair is evicted every round: 2 compulsory + 3 scan = 5 misses per
        # round for ever. SRRIP inserts scan blocks at RRPV 2 and promotes
        # the hot pair to 0 on their hits, so from round 2 only the three
        # scan blocks miss.
        lru, lru_rounds = self.scan_rounds("lru")
        srrip, srrip_rounds = self.scan_rounds("srrip")
        self.assertEqual(lru_rounds, [5] * self.ROUNDS)
        self.assertEqual(srrip_rounds, [5] + [3] * (self.ROUNDS - 1))
        self.assertEqual((lru.misses, srrip.misses), (50, 32))
        # The hot pair is what survives, not just a lower count.
        self.assertLessEqual({1, 2}, resident_blocks(srrip))
        self.assertNotIn(1, resident_blocks(lru))

    def test_brrip_keeps_a_thrashing_working_set_resident(self) -> None:
        # Cyclic scan of 5 blocks over a 4-way set. LRU and SRRIP both evict
        # the block that is referenced next and miss 100% of the time.
        # BRRIP inserts at RRPV 3, so a new block is the next victim and
        # three of the four ways stay put: two misses per pass.
        passes = 10
        lru = cyclic_scan(one_set_cache(self.WAYS, "lru"), 5, passes)
        srrip = cyclic_scan(one_set_cache(self.WAYS, "srrip"), 5, passes)
        brrip = cyclic_scan(one_set_cache(self.WAYS, "brrip"), 5, passes)
        self.assertEqual(lru, [5] * passes)
        self.assertEqual(srrip, [5] * passes)
        self.assertEqual(brrip, [5] + [2] * (passes - 1))
        self.assertEqual((sum(srrip), sum(brrip)), (50, 23))

    def test_brrip_is_reproducible(self) -> None:
        # The 1/32 insertions come from the cache's seeded RNG.
        runs = [sum(cyclic_scan(one_set_cache(self.WAYS, "brrip"), 5, 40)) for _ in range(2)]
        self.assertEqual(runs[0], runs[1])

    def test_rrip_ages_a_set_until_a_victim_appears(self) -> None:
        # Every way holds a block that was just hit (RRPV 0), so victim()
        # has to increment the whole set three times before any way reaches
        # RRPV 3; the first way to get there is the lowest-numbered one.
        c = one_set_cache(self.WAYS, "srrip")
        for tag in (1, 2, 3, 4):
            c.access(tag * c.block_size)
            c.access(tag * c.block_size)  # promote to RRPV 0
        self.assertEqual(c.policy.victim(0), 0)


class TestDRRIP(unittest.TestCase):
    """Set dueling between SRRIP and BRRIP with a 10-bit PSEL counter."""

    def make_policy(self, num_sets: int, num_ways: int = 4) -> DRRIPPolicy:
        policy = POLICIES["drrip"](num_sets, num_ways)
        assert isinstance(policy, DRRIPPolicy)
        return policy

    def test_leader_sets_are_chosen_by_index(self) -> None:
        p = self.make_policy(256)
        srrip = [s for s in range(256) if p.role_of(s) == "srrip-leader"]
        brrip = [s for s in range(256) if p.role_of(s) == "brrip-leader"]
        self.assertEqual(srrip, [0, 64, 128, 192])
        self.assertEqual(brrip, [32, 96, 160, 224])
        self.assertEqual(p.role_of(1), "follower")

    def test_small_caches_fall_back_to_one_leader_each(self) -> None:
        p = self.make_policy(8)
        expected = ["srrip-leader"] + ["follower"] * 3 + ["brrip-leader"] + ["follower"] * 3
        self.assertEqual([p.role_of(s) for s in range(8)], expected)
        # A single set cannot duel: it follows, and PSEL never moves.
        single = self.make_policy(1)
        self.assertEqual(single.role_of(0), "follower")

    def test_psel_saturates_in_both_directions(self) -> None:
        p = self.make_policy(64)
        self.assertEqual(p.following(), "srrip")  # 511: below the threshold
        for _ in range(2000):  # misses in the SRRIP leader set
            p.on_fill(0, 0, 1)
        self.assertEqual(p.psel, p.PSEL_MAX)
        self.assertEqual(p.following(), "brrip")
        for _ in range(2000):  # misses in the BRRIP leader set
            p.on_fill(32, 0, 1)
        self.assertEqual(p.psel, 0)
        self.assertEqual(p.following(), "srrip")

    def test_drrip_learns_to_use_brrip_on_a_thrashing_workload(self) -> None:
        # 64 sets x 4 ways, with five blocks per set cycled 20 times: the
        # working set is one block per set too large, the case BRRIP exists
        # for. DRRIP pays for the duel on two sets and tracks BRRIP.
        sets, ways, passes = 64, 4, 20
        one_pass = [(s + sets * j) * 64 for s in range(sets) for j in range(ways + 1)]
        stream = one_pass * passes
        caches = {}
        for name in ("srrip", "brrip", "drrip"):
            c = Cache("S", size=sets * ways * 64, block_size=64, associativity=ways, policy=name)
            for addr in stream:
                c.access(addr)
            caches[name] = c
        self.assertEqual(caches["srrip"].misses, len(stream))  # 100% miss: thrashing
        self.assertEqual(caches["brrip"].misses, 2752)
        self.assertEqual(caches["drrip"].misses, 2810)
        policy = caches["drrip"].policy
        assert isinstance(policy, DRRIPPolicy)
        self.assertEqual(policy.following(), "brrip")

    def test_drrip_leaves_psel_alone_when_the_working_set_fits(self) -> None:
        # Four blocks per set in a 4-way cache: nothing is ever evicted, so
        # the two leader sets miss equally and the duel stays a draw.
        sets, ways = 64, 4
        c = Cache("S", size=sets * ways * 64, block_size=64, associativity=ways, policy="drrip")
        for _ in range(20):
            for block in range(sets * ways):
                c.access(block * 64)
        self.assertEqual(c.misses, sets * ways)  # compulsory only
        policy = c.policy
        assert isinstance(policy, DRRIPPolicy)
        self.assertEqual(policy.psel, policy.PSEL_THRESHOLD - 1)
        self.assertEqual(policy.following(), "srrip")

    def test_drrip_degenerates_to_srrip_without_leader_sets(self) -> None:
        srrip = cyclic_scan(one_set_cache(4, "srrip"), 5, 10)
        drrip = cyclic_scan(one_set_cache(4, "drrip"), 5, 10)
        self.assertEqual(drrip, srrip)


class TestEveryPolicy(unittest.TestCase):
    """Invariants that every entry in the registry must satisfy."""

    names = sorted(POLICIES)

    def test_an_invalidated_way_is_refilled_before_any_eviction(self) -> None:
        # Emptying a way must leave the policy consistent: the next fill
        # takes the free way instead of evicting a resident block.
        for name in self.names:
            with self.subTest(policy=name):
                c = one_set_cache(ways=4, policy=name)
                for tag in (1, 2, 3, 4):
                    c.access(tag * c.block_size)
                c.invalidate(2)
                c.access(9 * c.block_size)  # must use the way freed by 2
                self.assertEqual(c.evictions, 0)
                self.assertEqual(resident_blocks(c), {1, 3, 4, 9})

    def test_every_policy_works_at_every_level_of_a_hierarchy(self) -> None:
        rng = random.Random(3)
        stream = [(rng.randrange(400) * 64, rng.random() < 0.3) for _ in range(2000)]
        for name in self.names:
            with self.subTest(policy=name):
                h = Hierarchy(
                    [
                        Level(Cache("L1", 1024, 64, 4, policy=name), 4),
                        Level(Cache("L2", 4096, 64, 8, policy=name), 12),
                    ],
                    memory_access_time=100,
                )
                for addr, is_write in stream:
                    h.access(addr, is_write)
                self.assertEqual(h.accesses, len(stream))
                l1, l2 = h.levels[0].cache, h.levels[1].cache
                self.assertEqual(l1.hits + l1.misses, len(stream))
                self.assertEqual(l2.accesses, l1.misses)
                self.assertEqual(l1.fills, l1.misses)


if __name__ == "__main__":
    unittest.main(verbosity=2)
