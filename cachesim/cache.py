"""A single, configurable cache level.

One level of a cache (an L1, an L2, ...) is modelled as a grid of
``num_sets`` sets by ``associativity`` ways:

* The cache holds ``size`` bytes, divided into blocks (lines) of
  ``block_size`` bytes.
* A memory address maps to exactly one set (``block % num_sets``) and may
  occupy any way of that set.

    - associativity == 1                    -> direct-mapped
    - associativity == blocks in the cache  -> fully associative
    - otherwise                             -> N-way set-associative

* When a set is full, the ``ReplacementPolicy`` chooses the victim.

Optionally every miss is classified into the three Cs:

* compulsory -- the first reference to this block.
* capacity   -- a fully-associative LRU cache of the same capacity would also
                have missed.
* conflict   -- the fully-associative twin would have hit, so the miss is
                due to the set mapping (or, for non-LRU policies, to the
                replacement choice).

Classification runs a shadow fully-associative LRU cache of identical
capacity alongside the real one and compares outcomes.

Writes use write-back, write-allocate semantics: a write miss loads the block
like a read miss, and dirty blocks are counted as a write-back when evicted.
"""

from __future__ import annotations

import random
from collections import OrderedDict

from cachesim.policies import POLICIES, ReplacementPolicy


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
        if size % (block_size * associativity) != 0:
            raise ValueError(
                f"{name}: size {size} must be a multiple of "
                f"block_size*associativity ({block_size * associativity})"
            )

        self.name = name
        self.size = size
        self.block_size = block_size
        self.associativity = associativity
        self.num_sets = size // (block_size * associativity)
        self.num_blocks = size // block_size
        self.policy_name = policy

        rng = random.Random(rng_seed)
        try:
            self.policy: ReplacementPolicy = POLICIES[policy](self.num_sets, associativity, rng)
        except KeyError:
            raise ValueError(
                f"unknown replacement policy {policy!r}; choose from {sorted(POLICIES)}"
            ) from None

        # Cache contents, indexed [set][way].
        self._valid = [[False] * associativity for _ in range(self.num_sets)]
        self._tags = [[0] * associativity for _ in range(self.num_sets)]
        self._dirty = [[False] * associativity for _ in range(self.num_sets)]

        # --- statistics -----------------------------------------------------
        self.hits = 0
        self.misses = 0
        self.read_hits = 0
        self.read_misses = 0
        self.write_hits = 0
        self.write_misses = 0
        self.evictions = 0
        self.writebacks = 0  # dirty blocks pushed out on eviction

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

    def _split(self, addr: int) -> tuple[int, int, int]:
        """Break an address into (block number, set index, tag).

        block = addr // block_size   -> which block-sized chunk of memory
        set   = block % num_sets     -> which set that chunk may use
        tag   = block // num_sets    -> distinguishes chunks within a set
        """
        block = addr // self.block_size
        return block, block % self.num_sets, block // self.num_sets

    # -- the main entry point --------------------------------------------------

    def access(self, addr: int, is_write: bool = False) -> bool:
        """Simulate one access. Returns True on hit, False on miss.

        On a miss the block is installed into the cache, evicting a victim if
        the set is full: ``access`` models both the lookup and the fill that
        follows once the data arrives from the next level.
        """
        block, set_idx, tag = self._split(addr)

        # Search the set for a matching valid tag (at most ``ways`` compares,
        # matching the parallel tag comparators of real hardware).
        valid = self._valid[set_idx]
        tags = self._tags[set_idx]
        for way in range(self.associativity):
            if valid[way] and tags[way] == tag:
                # HIT ---------------------------------------------------------
                self.hits += 1
                if is_write:
                    self.write_hits += 1
                    self._dirty[set_idx][way] = True
                else:
                    self.read_hits += 1
                self.policy.on_hit(set_idx, way)
                if self.track_3c:
                    self._update_shadow(block)
                return True

        # MISS -----------------------------------------------------------------
        self.misses += 1
        if is_write:
            self.write_misses += 1
        else:
            self.read_misses += 1

        if self.track_3c:
            self._classify_miss(block)
            self._update_shadow(block)

        # Fill: prefer an empty way; otherwise ask the policy for a victim.
        way = next((w for w in range(self.associativity) if not valid[w]), None)
        if way is None:
            way = self.policy.victim(set_idx)
            self.evictions += 1
            if self._dirty[set_idx][way]:
                self.writebacks += 1  # modified data must be written down

        valid[way] = True
        tags[way] = tag
        self._dirty[set_idx][way] = is_write
        self.policy.on_fill(set_idx, way)
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
