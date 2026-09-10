"""Replacement policies for a set-associative cache.

A policy decides which way of a full set to evict. Every policy keeps its own
per-set bookkeeping and is driven by hooks that ``Cache`` calls:

* ``on_hit(set_idx, way, block)``  -- an access hit this way.
* ``on_fill(set_idx, way, block)`` -- ``block`` was installed into this way.
* ``on_invalidate(set_idx, way)``  -- the way was emptied on command.
* ``victim(set_idx)``              -- the set is full; return the way to evict.
* ``on_probe(set_idx, block)``     -- a lookup started, hit or miss unknown.
  Only called for policies that set ``sees_references``, so nothing is paid
  for it on the hot path otherwise. It is the hook a policy needs to count
  *references* to the level: a probe is one reference, whereas ``on_fill``
  also fires for a write-back arriving from the level above, which is not a
  lookup.

``victim`` is only called when every way of the set holds valid data: empty
ways (including ways emptied by ``on_invalidate``) are always filled first,
so policies never see invalid ways.

Policies are selected by name through the ``POLICIES`` registry.

Two families live here:

* recency and frequency approximations that real hardware implements --
  ``lru``, ``fifo``, ``plru`` (tree pseudo-LRU), ``nru``, ``lfu``, and the
  re-reference interval predictors ``srrip``, ``brrip`` and ``drrip``;
* deliberate contrasts used to bound behaviour -- ``random`` and ``mru``.

Belady's optimal policy is offline -- it has to be told the future -- so it
lives in ``cachesim.opt`` and registers itself here under the name ``opt``.
"""

from __future__ import annotations

import random
from collections import OrderedDict
from typing import ClassVar


class ReplacementPolicy:
    """Base class. Tracks nothing; subclasses add per-set state."""

    #: Set by a policy that has to observe every lookup, not just its
    #: outcome; ``Cache`` then calls ``on_probe``. False for every online
    #: policy, so they cost nothing.
    sees_references: ClassVar[bool] = False

    def __init__(self, num_sets: int, num_ways: int, rng: random.Random | None = None) -> None:
        self.num_sets = num_sets
        self.num_ways = num_ways
        self.rng = rng if rng is not None else random.Random(0)

    def on_probe(self, set_idx: int, block: int) -> None:
        """Called at the start of a lookup, before hit or miss is decided.

        Only called when ``sees_references`` is set.
        """

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
    """Shared machinery: each set keeps its ways in recency order, victim at
    the front. Subclasses decide which events move a way to the back.

    The order is an ``OrderedDict`` of ways rather than a list of them.
    Both spell the same sequence and both are ordered by insertion, but a
    list has to find the way before it can move it: ``remove`` plus
    ``append`` is a scan, and it runs on every cache hit, which is the most
    frequent event in the simulator. An ``OrderedDict`` threads a doubly
    linked list through its entries, so ``move_to_end`` is a lookup and two
    pointer swaps whatever the associativity is. Timed on this machine, one
    move to the back:

        ways    list    OrderedDict
           4   33 ns          11 ns
           8   45 ns          11 ns
          16   70 ns          11 ns

    against 8.5 ns and 29 ns respectively for reading the front, which is
    paid only when a set is full and has to name a victim.

    The mapping's values are unused: it is a set of ways that remembers the
    order they were touched in.

    Only two moves are ever made on that order, and both are written out at
    each call site rather than put behind a ``_to_back`` / ``_to_front``
    helper:

    * ``order.move_to_end(way)`` makes ``way`` the MOST recently used of
      its set -- the far end from the victim;
    * ``order.move_to_end(way, last=False)`` makes it the LEAST recently
      used -- the next way to be evicted.

    Naming those two reads better, and costs a Python frame on the most
    frequent event in the simulator. Measured with ``cachesim bench``,
    interleaved best of 3 x ``--repeat 5``, helper against written out:
    matmul_naive 0.248 -> 0.264 s, matmul_blocked 0.211 -> 0.225 s,
    sequential 0.0398 -> 0.0423 s, conflict 0.0460 -> 0.0487 s -- 6% of a
    hit-heavy run for a call that does one thing. So the names live here,
    where they cannot fall out of step with the four statements below, and
    the statements say ``move_to_end`` where the reader can see it.
    """

    def __init__(self, num_sets: int, num_ways: int, rng: random.Random | None = None) -> None:
        super().__init__(num_sets, num_ways, rng)
        #: Ways of each set, least recently used first.
        self._order: list[OrderedDict[int, None]] = [
            OrderedDict.fromkeys(range(num_ways)) for _ in range(num_sets)
        ]

    def on_fill(self, set_idx: int, way: int, block: int) -> None:
        # To the back: a freshly loaded block is the most recently used.
        self._order[set_idx].move_to_end(way)

    def on_invalidate(self, set_idx: int, way: int) -> None:
        # To the front: an emptied way is refilled before any victim is
        # chosen, so parking it there leaves the relative order of the
        # valid ways intact.
        self._order[set_idx].move_to_end(way, last=False)

    def victim(self, set_idx: int) -> int:
        return next(iter(self._order[set_idx]))


class LRUPolicy(_RecencyOrder):
    """Least Recently Used: evict the way untouched for the longest time.

    Hits and fills move the way to the most-recent end; the victim is the
    way at the least-recent front.
    """

    def on_hit(self, set_idx: int, way: int, block: int) -> None:
        # To the back: a referenced block is the most recently used. This
        # is the single most frequently executed function in the simulator
        # -- once per cache hit at every level -- and the reason
        # ``_RecencyOrder`` spells the move out instead of naming it.
        self._order[set_idx].move_to_end(way)


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
        # To the back, exactly as LRU does; only ``victim`` differs.
        self._order[set_idx].move_to_end(way)

    def victim(self, set_idx: int) -> int:
        return next(reversed(self._order[set_idx]))


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
    Concepts", 10th ed., 2018, sec. 10.4.5, "LRU-Approximation Page
    Replacement").
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
    2018, sec. 10.4.6, "Counting-Based Page Replacement", which pairs LFU
    with its mirror image MFU). Nothing here ages, so ``lfu`` is a faithful
    model of the textbook policy, cache pollution included.
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


# -- re-reference interval prediction ------------------------------------------
#
# Jaleel, Theobald, Steely and Emer, "High Performance Cache Replacement Using
# Re-Reference Interval Prediction (RRIP)", ISCA 2010.

#: Bits of re-reference prediction value per way (M in the paper).
RRPV_BITS = 2
#: "Distant" re-reference prediction: the way is the next victim.
RRPV_DISTANT = (1 << RRPV_BITS) - 1  # 3
#: "Long" re-reference prediction: what SRRIP inserts with.
RRPV_LONG = RRPV_DISTANT - 1  # 2
#: "Near-immediate" prediction: what a hit promotes to.
RRPV_NEAR = 0
#: BRRIP inserts with RRPV_LONG once every ``BRRIP_EPSILON`` fills.
BRRIP_EPSILON = 32


class _RRIP(ReplacementPolicy):
    """Shared machinery for the RRIP family (Jaleel et al., ISCA 2010).

    Each way carries a 2-bit re-reference prediction value (RRPV): 0 means
    "will be re-referenced in the near future", 3 means "in the distant
    future", so 3 marks the preferred victim. A hit promotes the way to 0
    (hit priority, the paper's SRRIP-HP). To find a victim the set is
    scanned for the first way at 3; if there is none, every RRPV in the set
    is incremented (ageing the whole set at once) and the scan repeats, so
    at most three increments are ever needed.

    The insertion value is what separates the family members and is chosen
    by ``_insert_rrpv``.
    """

    def __init__(self, num_sets: int, num_ways: int, rng: random.Random | None = None) -> None:
        super().__init__(num_sets, num_ways, rng)
        self._rrpv = [[RRPV_DISTANT] * num_ways for _ in range(num_sets)]

    def _insert_rrpv(self, set_idx: int) -> int:
        raise NotImplementedError

    def on_hit(self, set_idx: int, way: int, block: int) -> None:
        self._rrpv[set_idx][way] = RRPV_NEAR

    def on_fill(self, set_idx: int, way: int, block: int) -> None:
        self._rrpv[set_idx][way] = self._insert_rrpv(set_idx)

    def on_invalidate(self, set_idx: int, way: int) -> None:
        # An empty way is the best possible victim: predict a distant
        # re-reference so the state stays truthful if it is ever scanned.
        self._rrpv[set_idx][way] = RRPV_DISTANT

    def victim(self, set_idx: int) -> int:
        rrpv = self._rrpv[set_idx]
        num_ways = self.num_ways
        while True:
            for way in range(num_ways):
                if rrpv[way] == RRPV_DISTANT:
                    return way
            for way in range(num_ways):
                rrpv[way] += 1


class SRRIPPolicy(_RRIP):
    """Static RRIP: insert every block predicting a long re-reference interval.

    A newly filled block starts at RRPV 2 rather than 0, so it must be
    re-referenced before it is aged to 3 to earn a place. That makes the
    policy scan-resistant: a burst of one-touch blocks cycles through the
    same handful of ways at RRPV 2->3 while blocks that have been hit sit at
    0 and survive, whereas LRU installs every scan block as most recently
    used and flushes the whole set.
    """

    def _insert_rrpv(self, set_idx: int) -> int:
        return RRPV_LONG


class BRRIPPolicy(_RRIP):
    """Bimodal RRIP: insert predicting a *distant* re-reference interval,
    except once every ``BRRIP_EPSILON`` fills.

    A block inserted at RRPV 3 is the next victim, so a thrashing working
    set (one larger than the set) keeps most of its blocks resident instead
    of evicting all of them in turn: the cache preserves a fraction of the
    working set rather than cycling. The occasional insertion at RRPV 2
    (probability 1/32, drawn from the cache's seeded RNG) is what lets the
    resident set change over time.
    """

    def _insert_rrpv(self, set_idx: int) -> int:
        if self.rng.randrange(BRRIP_EPSILON) == 0:
            return RRPV_LONG
        return RRPV_DISTANT


class DRRIPPolicy(_RRIP):
    """Dynamic RRIP: run SRRIP and BRRIP against each other and follow the winner.

    Set dueling (Qureshi, Jaleel, Patt, Steely and Emer, "Adaptive Insertion
    Policies for High Performance Caching", ISCA 2007; reused for RRIP by
    Jaleel et al., ISCA 2010): a few sets always insert the SRRIP way, a few
    always insert the BRRIP way, and a saturating counter records which
    group is missing more. Every other set (the followers) uses whichever
    insertion policy is currently winning, so the cache pays the cost of the
    losing policy on the leader sets only.

    Leader sets, chosen by set index so no state is needed:

    * ``num_sets >= 64``: ``set_idx % 64 == 0`` leads SRRIP and
      ``set_idx % 64 == 32`` leads BRRIP -- 1/32 of the cache duels, spread
      evenly, which is the sampling ratio used in the paper.
    * ``2 <= num_sets < 64``: set 0 leads SRRIP and set ``num_sets // 2``
      leads BRRIP, so a small cache still duels. With two sets there are no
      followers left and the result is simply one set of each.
    * ``num_sets == 1``: dueling is impossible; the single set is a follower
      and PSEL never moves, so DRRIP degenerates to SRRIP.

    PSEL is a 10-bit saturating counter, incremented on a fill into an SRRIP
    leader set and decremented on a fill into a BRRIP leader set (a fill is
    a miss: the line was not resident). High means SRRIP is missing more, so
    followers use BRRIP when the top bit is set. It starts at 511, one below
    the midpoint, so that before any evidence has accumulated the followers
    run SRRIP.
    """

    #: Width of the policy selection counter.
    PSEL_BITS = 10
    PSEL_MAX = (1 << PSEL_BITS) - 1
    #: Followers use BRRIP once PSEL reaches this (the top bit is set).
    PSEL_THRESHOLD = 1 << (PSEL_BITS - 1)
    #: One set in ``LEADER_PERIOD`` leads each policy when there is room.
    LEADER_PERIOD = 64

    _FOLLOWER = 0
    _SRRIP_LEADER = 1
    _BRRIP_LEADER = 2

    def __init__(self, num_sets: int, num_ways: int, rng: random.Random | None = None) -> None:
        super().__init__(num_sets, num_ways, rng)
        self.psel = self.PSEL_THRESHOLD - 1
        self._role = [self._FOLLOWER] * num_sets
        if num_sets >= self.LEADER_PERIOD:
            for set_idx in range(num_sets):
                if set_idx % self.LEADER_PERIOD == 0:
                    self._role[set_idx] = self._SRRIP_LEADER
                elif set_idx % self.LEADER_PERIOD == self.LEADER_PERIOD // 2:
                    self._role[set_idx] = self._BRRIP_LEADER
        elif num_sets >= 2:
            self._role[0] = self._SRRIP_LEADER
            self._role[num_sets // 2] = self._BRRIP_LEADER

    #: Role names, indexed by the internal role code.
    ROLE_NAMES = ("follower", "srrip-leader", "brrip-leader")

    def role_of(self, set_idx: int) -> str:
        """Which side of the duel a set is on: see ``ROLE_NAMES``."""
        return self.ROLE_NAMES[self._role[set_idx]]

    def following(self) -> str:
        """The insertion policy the follower sets are using right now."""
        return "srrip" if self.psel < self.PSEL_THRESHOLD else "brrip"

    def _bimodal(self) -> int:
        if self.rng.randrange(BRRIP_EPSILON) == 0:
            return RRPV_LONG
        return RRPV_DISTANT

    def _insert_rrpv(self, set_idx: int) -> int:
        role = self._role[set_idx]
        if role == self._SRRIP_LEADER:
            # Missing in an SRRIP leader is evidence against SRRIP.
            if self.psel < self.PSEL_MAX:
                self.psel += 1
            return RRPV_LONG
        if role == self._BRRIP_LEADER:
            if self.psel > 0:
                self.psel -= 1
            return self._bimodal()
        if self.psel < self.PSEL_THRESHOLD:
            return RRPV_LONG  # SRRIP is winning
        return self._bimodal()


#: Registry of selectable policies, keyed by the name used in configs.
POLICIES: dict[str, type[ReplacementPolicy]] = {
    "lru": LRUPolicy,
    "fifo": FIFOPolicy,
    "plru": TreePLRUPolicy,
    "nru": NRUPolicy,
    "lfu": LFUPolicy,
    "mru": MRUPolicy,
    "srrip": SRRIPPolicy,
    "brrip": BRRIPPolicy,
    "drrip": DRRIPPolicy,
    "random": RandomPolicy,
}


def online_policy_names() -> list[str]:
    """The registered policies one forward pass can run, sorted by name.

    An offline policy (one whose ``sees_references`` is set, i.e. ``opt``)
    has to be handed the level's whole reference stream before the first
    probe, which only ``cachesim policies`` and the two-pass
    ``cachesim.opt.simulate_with_opt`` driver do. Every command that builds
    a cache and replays a trace straight through -- ``sweep``, ``sets`` --
    offers this list rather than the whole registry, so that naming a
    policy it cannot run is a usage error instead of a RuntimeError raised
    mid-simulation.

    It is a function, not a constant, because ``cachesim.opt`` adds to
    ``POLICIES`` at import time: a tuple frozen at this module's import
    would be read before that registration.
    """
    return sorted(name for name, cls in POLICIES.items() if not cls.sees_references)
