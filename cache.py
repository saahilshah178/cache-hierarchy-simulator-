"""cache.py — a single, configurable cache level.

This module models ONE level of a cache (an L1, an L2, ...) as a grid of
"sets" x "ways":

  * The cache holds `size` bytes total, chopped into fixed-size "blocks"
    (also called cache lines) of `block_size` bytes each.
  * Blocks are grouped into `num_sets` sets. A given memory address is only
    allowed to live in exactly one set (picked by simple arithmetic on the
    address), but within that set it may occupy any of `associativity` ways.
      - associativity == 1              -> direct-mapped
      - associativity == blocks in cache -> fully associative
      - anything in between             -> N-way set-associative
  * When a set is full and a new block needs a spot, the ReplacementPolicy
    picks a victim to evict.

The cache also (optionally) classifies every miss into the classic "3 Cs":

  * compulsory — the very first time this block is ever referenced. No cache
    of any size or shape could have hit; the data simply was never loaded.
  * capacity   — a fully-associative cache of the SAME total capacity (with
    LRU) would ALSO have missed. The working set is just too big.
  * conflict   — the fully-associative twin would have hit, so the miss is
    purely an artifact of limited associativity (too many hot blocks mapped
    to the same set).

To do that classification we run a "shadow" fully-associative LRU cache of
identical capacity next to the real one and compare outcomes. The shadow
stores only block numbers, so it is cheap.

Writes use a write-back, write-allocate policy: a write miss loads the block
just like a read miss, and modified ("dirty") blocks are counted as a
write-back when evicted. This affects the eviction/writeback statistics only;
timing is handled by hierarchy.py.
"""

import random
from collections import OrderedDict


# ---------------------------------------------------------------------------
# Replacement policies
# ---------------------------------------------------------------------------
#
# To add a new policy (e.g. RRIP, PLRU, LFU):
#   1. Subclass ReplacementPolicy.
#   2. Implement on_hit / on_fill / victim (whatever the policy needs).
#   3. Register it in the POLICIES dict at the bottom of this section.
# After that it is selectable everywhere by its string name.

class ReplacementPolicy:
    """Base class. Tracks nothing; subclasses add per-set bookkeeping.

    The cache guarantees that `victim()` is only called when every way in the
    set already holds valid data (empty ways are filled first), so policies
    never have to worry about invalid ways.
    """

    def __init__(self, num_sets, num_ways, rng=None):
        self.num_sets = num_sets
        self.num_ways = num_ways
        self.rng = rng or random.Random(0)

    def on_hit(self, set_idx, way):
        """Called every time an access hits `way` in `set_idx`."""

    def on_fill(self, set_idx, way):
        """Called when a new block is installed into `way` in `set_idx`."""

    def victim(self, set_idx):
        """Return the way to evict from a completely full set."""
        raise NotImplementedError


class LRUPolicy(ReplacementPolicy):
    """Least Recently Used: evict the way untouched for the longest time.

    Each set keeps its ways in a list ordered least- -> most-recently used.
    Hits and fills move the way to the "most recent" end; the victim is
    whatever sits at the "least recent" front.
    """

    def __init__(self, num_sets, num_ways, rng=None):
        super().__init__(num_sets, num_ways, rng)
        self._order = [list(range(num_ways)) for _ in range(num_sets)]

    def _touch(self, set_idx, way):
        order = self._order[set_idx]
        order.remove(way)
        order.append(way)

    on_hit = _touch
    on_fill = _touch

    def victim(self, set_idx):
        return self._order[set_idx][0]


class FIFOPolicy(ReplacementPolicy):
    """First In, First Out: evict the way that was FILLED longest ago.

    Unlike LRU, hits do NOT refresh a block's position — only being loaded
    does. Cheaper to build in hardware, usually a little worse.
    """

    def __init__(self, num_sets, num_ways, rng=None):
        super().__init__(num_sets, num_ways, rng)
        self._order = [list(range(num_ways)) for _ in range(num_sets)]

    def on_fill(self, set_idx, way):
        order = self._order[set_idx]
        order.remove(way)
        order.append(way)

    def victim(self, set_idx):
        return self._order[set_idx][0]


class RandomPolicy(ReplacementPolicy):
    """Evict a uniformly random way. Nearly free in hardware, no bookkeeping.

    Seeded through the cache's RNG so runs are reproducible.
    """

    def victim(self, set_idx):
        return self.rng.randrange(self.num_ways)


#: Registry of selectable policies. Add your own here.
POLICIES = {
    "lru": LRUPolicy,
    "fifo": FIFOPolicy,
    "random": RandomPolicy,
}


# ---------------------------------------------------------------------------
# The cache itself
# ---------------------------------------------------------------------------

class Cache:
    """One set-associative cache level.

    Parameters
    ----------
    name          : label used in reports ("L1", "L2", ...).
    size          : total capacity in bytes (e.g. 32768 for 32 KB).
    block_size    : bytes per cache block/line (64 is typical).
    associativity : ways per set. 1 = direct-mapped.
    policy        : "lru", "fifo", or "random" (see POLICIES).
    track_3c      : run the shadow fully-associative cache and classify
                    misses into compulsory/capacity/conflict.
    rng_seed      : seed for the random policy, for reproducibility.
    """

    def __init__(self, name, size, block_size, associativity,
                 policy="lru", track_3c=True, rng_seed=0):
        if size % (block_size * associativity) != 0:
            raise ValueError(
                f"{name}: size {size} must be a multiple of "
                f"block_size*associativity ({block_size * associativity})")

        self.name = name
        self.size = size
        self.block_size = block_size
        self.associativity = associativity
        self.num_sets = size // (block_size * associativity)
        self.num_blocks = size // block_size
        self.policy_name = policy

        rng = random.Random(rng_seed)
        try:
            self.policy = POLICIES[policy](self.num_sets, associativity, rng)
        except KeyError:
            raise ValueError(
                f"unknown replacement policy {policy!r}; "
                f"choose from {sorted(POLICIES)}") from None

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
        self.writebacks = 0          # dirty blocks pushed out on eviction

        # --- 3-C classification machinery ----------------------------------
        self.track_3c = track_3c
        self.compulsory_misses = 0
        self.capacity_misses = 0
        self.conflict_misses = 0
        self._seen_blocks = set()          # every block number ever touched
        # Shadow fully-associative LRU cache of the same capacity:
        # an OrderedDict of block numbers, oldest first.
        self._shadow = OrderedDict()

    # -- address arithmetic --------------------------------------------------

    def _split(self, addr):
        """Break an address into (block number, set index, tag).

        block = addr // block_size   -> which cache-line-sized chunk of memory
        set   = block % num_sets     -> which set that chunk is allowed to use
        tag   = block // num_sets    -> what we store to tell chunks apart
        """
        block = addr // self.block_size
        return block, block % self.num_sets, block // self.num_sets

    # -- the main entry point --------------------------------------------------

    def access(self, addr, is_write=False):
        """Simulate one access. Returns True on hit, False on miss.

        On a miss the block is installed (allocated) into the cache, evicting
        a victim if the set is full — i.e. calling access() also models the
        fill that happens after the data arrives from the next level.
        """
        block, set_idx, tag = self._split(addr)

        # Search the set for a matching, valid tag (at most `ways` compares —
        # exactly what the parallel tag comparators do in real hardware).
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
                self.writebacks += 1   # modified data must be written down

        valid[way] = True
        tags[way] = tag
        self._dirty[set_idx][way] = is_write
        self.policy.on_fill(set_idx, way)
        return False

    # -- 3-C helpers ----------------------------------------------------------

    def _classify_miss(self, block):
        """Assign this miss to compulsory, capacity, or conflict."""
        if block not in self._seen_blocks:
            # Never referenced before: no cache could have held it.
            self.compulsory_misses += 1
        elif block in self._shadow:
            # The fully-associative twin still holds it, so only our limited
            # associativity (set conflicts) caused this miss.
            self.conflict_misses += 1
        else:
            # Even a fully-associative cache of this capacity had already
            # evicted it: the working set is simply too large.
            self.capacity_misses += 1

    def _update_shadow(self, block):
        """Feed the same reference to the shadow fully-associative LRU cache."""
        self._seen_blocks.add(block)
        if block in self._shadow:
            self._shadow.move_to_end(block)        # refresh LRU position
        else:
            self._shadow[block] = True
            if len(self._shadow) > self.num_blocks:
                self._shadow.popitem(last=False)   # evict shadow's LRU block

    # -- derived stats ----------------------------------------------------------

    @property
    def accesses(self):
        return self.hits + self.misses

    @property
    def miss_rate(self):
        """Local miss rate: misses / accesses *that reached this cache*."""
        return self.misses / self.accesses if self.accesses else 0.0
