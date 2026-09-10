"""Belady's MIN (OPT): the offline optimal replacement policy.

Belady, "A study of replacement algorithms for a virtual-storage computer",
IBM Systems Journal 5(2), 1966, proved that evicting the block whose next
reference is farthest in the future minimises the number of misses for a
fully associative cache with demand fetching. No implementable policy can
do better, so OPT is the lower bound every real policy is measured against:
the interesting number about LRU is not its miss rate but its distance from
OPT on the same trace.

OPT is offline -- it needs the whole reference stream up front -- so it is
kept out of ``policies.py`` and driven from here.

Two things live in this module.

``opt_misses(blocks, capacity)``
    The miss count of a fully associative OPT cache of ``capacity`` blocks
    on a block-number stream, computed in O(N log N) from precomputed
    next-use indices and a heap keyed by next use.

``OPTPolicy`` (registered as the policy name ``"opt"``)
    A set-associative Belady policy: within each set it evicts the way
    whose block is referenced farthest in the future. It must be given the
    future with ``preload(blocks)`` before the simulation starts, and
    raises ``RuntimeError`` if it is asked for a victim without one. Note
    that per-set Belady is optimal *given the set mapping*, not globally
    optimal: the true lower bound for a set-associative cache is what
    ``opt_misses`` computes for a fully associative one of the same
    capacity.

Use ``run_opt`` for a single level and ``simulate_with_opt`` for one level
of a hierarchy.
"""

from __future__ import annotations

import random
from collections.abc import Mapping, Sequence
from dataclasses import replace
from heapq import heappop, heappush
from typing import Any

from cachesim.cache import EMPTY, Cache
from cachesim.config import ConfigError, HierarchySpec, parse_config
from cachesim.hierarchy import Hierarchy, Level
from cachesim.policies import POLICIES, ReplacementPolicy

#: The name ``OPTPolicy`` registers itself under in ``POLICIES``.
OPT = "opt"


def next_use_indices(blocks: Sequence[int]) -> list[int]:
    """For every position, the index of the next reference to the same block.

    A block that is never referenced again gets ``len(blocks)``, which
    compares greater than any real index, so "never again" sorts as the
    farthest possible future.
    """
    count = len(blocks)
    never = count
    nxt = [never] * count
    last: dict[int, int] = {}
    for i in range(count - 1, -1, -1):
        block = blocks[i]
        nxt[i] = last.get(block, never)
        last[block] = i
    return nxt


def opt_misses(blocks: Sequence[int], capacity: int) -> int:
    """Misses of a fully associative OPT cache of ``capacity`` blocks.

    ``blocks`` is a stream of block numbers, one per reference. Every miss
    allocates (demand fetching, no bypassing); when the cache is full the
    resident block whose next reference is farthest away is evicted, and
    blocks that are never referenced again are evicted first.

    Implementation: next-use indices are precomputed in one backward pass,
    and the resident set is kept in a max-heap keyed by next use. A heap
    entry becomes stale when its block is referenced again (which gives it
    a later next use), so entries are validated against the current next
    use on the way out -- the standard lazy-deletion heap. O(N log N) time,
    O(N) space.

    Ties can only happen between blocks that are never referenced again,
    and evicting any of those is equally good, so the miss count does not
    depend on how they are broken.
    """
    if capacity < 1:
        raise ValueError(f"capacity must be at least one block, got {capacity}")
    nxt = next_use_indices(blocks)
    resident: dict[int, int] = {}  # block -> its current next-use index
    heap: list[tuple[int, int]] = []  # (-next use, block), lazily invalidated
    misses = 0
    for i, block in enumerate(blocks):
        next_use = nxt[i]
        if block not in resident:
            misses += 1
            if len(resident) == capacity:
                while True:  # pop until a live entry surfaces
                    negated, victim = heappop(heap)
                    if resident.get(victim, -1) == -negated:
                        del resident[victim]
                        break
        resident[block] = next_use
        heappush(heap, (-next_use, block))
    return misses


class OPTPolicy(ReplacementPolicy):
    """Belady's rule inside each set: evict the way used farthest in the future.

    The policy is told the exact sequence of block references the level
    will see (``preload``) and follows it with a cursor: every probe is one
    reference, so ``on_probe`` advances the cursor and refreshes that
    block's next-use time. ``victim`` then evicts the way whose block has
    the largest next-use time, with "never referenced again" as the largest
    value of all.

    Blocks that arrive without being referenced -- write-backs pushed down
    from the level above, which allocate a line but are not lookups -- are
    not in the preloaded stream and therefore count as never referenced
    again, which is exactly right: nothing in the future stream will ask
    this level for them.

    The cursor is checked against the stream on every probe, so a stream
    that does not match the simulation (wrong level, wrong trace, wrong
    order) fails immediately instead of silently producing a wrong bound.
    """

    sees_references = True

    def __init__(self, num_sets: int, num_ways: int, rng: random.Random | None = None) -> None:
        super().__init__(num_sets, num_ways, rng)
        # The block held in each way, mirrored from the fill/invalidate
        # hooks so that victim() can look up next-use times.
        self._ways = [[EMPTY] * num_ways for _ in range(num_sets)]
        self._stream: list[int] | None = None
        self._next_after: list[int] = []
        self._next_use: dict[int, int] = {}
        self._never = 0
        self._cursor = 0

    # -- being told the future -------------------------------------------------

    def preload(self, blocks: Sequence[int]) -> None:
        """Give the policy the reference stream this level is about to see.

        One entry per probe of this level, in order. Resets the cursor, so
        the same policy object can be reused for another run.
        """
        stream = list(blocks)
        self._stream = stream
        self._next_after = next_use_indices(stream)
        self._next_use = {}
        self._never = len(stream)
        self._cursor = 0

    @property
    def preloaded(self) -> bool:
        """True once ``preload`` has supplied a reference stream."""
        return self._stream is not None

    def _no_future(self) -> RuntimeError:
        return RuntimeError(
            "policy 'opt' is offline and has not been given the future: call "
            "preload(blocks) with the reference stream this level will see, or "
            "drive the simulation with cachesim.opt.run_opt / simulate_with_opt"
        )

    # -- hooks -----------------------------------------------------------------

    def on_probe(self, set_idx: int, block: int) -> None:
        stream = self._stream
        if stream is None:
            raise self._no_future()
        cursor = self._cursor
        if cursor >= len(stream):
            raise RuntimeError(
                f"policy 'opt': the preloaded reference stream ran out after "
                f"{len(stream)} references, but the level was probed again for "
                f"block {block}"
            )
        if stream[cursor] != block:
            raise RuntimeError(
                f"policy 'opt': the preloaded reference stream does not match the "
                f"simulation at reference {cursor}: expected block {stream[cursor]}, "
                f"got block {block}"
            )
        self._next_use[block] = self._next_after[cursor]
        self._cursor = cursor + 1

    def on_fill(self, set_idx: int, way: int, block: int) -> None:
        self._ways[set_idx][way] = block

    def on_invalidate(self, set_idx: int, way: int) -> None:
        self._ways[set_idx][way] = EMPTY

    def victim(self, set_idx: int) -> int:
        if self._stream is None:
            raise self._no_future()
        ways = self._ways[set_idx]
        next_use = self._next_use
        never = self._never
        best_way = 0
        farthest = -1
        for way in range(self.num_ways):
            when = next_use.get(ways[way], never)
            if when > farthest:
                farthest = when
                best_way = way
        return best_way


POLICIES[OPT] = OPTPolicy


# -- drivers -------------------------------------------------------------------


def run_opt(
    trace_accesses: Sequence[tuple[int, bool]],
    size: int,
    block_size: int,
    associativity: int,
    track_3c: bool = True,
    index: str = "modulo",
) -> Cache:
    """Simulate one OPT cache level over ``(address, is_write)`` accesses.

    Returns the finished ``Cache``, so its miss count is the set-associative
    Belady bound for that geometry. The whole trace has to be in memory
    anyway; the block numbers are derived from it and preloaded.
    """
    cache = Cache(
        "OPT",
        size,
        block_size,
        associativity,
        policy=OPT,
        track_3c=track_3c,
        index=index,
    )
    policy = cache.policy
    if not isinstance(policy, OPTPolicy):  # pragma: no cover - registry sanity
        raise TypeError(f"policy {OPT!r} is not OPTPolicy but {type(policy).__name__}")
    policy.preload([addr // block_size for addr, _ in trace_accesses])
    for addr, is_write in trace_accesses:
        cache.access(addr, is_write)
    return cache


class _ProbeRecorder(Cache):
    """An LRU cache that also records the block of every probe it receives."""

    def __init__(self, name: str, size: int, block_size: int, associativity: int, index: str):
        super().__init__(
            name,
            size,
            block_size,
            associativity,
            policy="lru",
            track_3c=False,
            index=index,
        )
        self.probed: list[int] = []

    def probe(self, block: int, is_write: bool = False) -> bool:
        self.probed.append(block)
        return super().probe(block, is_write)


def _opt_level(spec: HierarchySpec) -> int:
    """The index of the single level asking for OPT, or a ``ConfigError``."""
    levels = [i for i, level in enumerate(spec.levels) if level.policy == OPT]
    if not levels:
        raise ConfigError(
            f"no level uses policy {OPT!r}; simulate_with_opt exists to drive one "
            "that does -- use Hierarchy.from_config for an OPT-free hierarchy"
        )
    if len(levels) > 1:
        names = ", ".join(spec.levels[i].name for i in levels)
        raise ConfigError(
            f"policy {OPT!r} is only meaningful at one level, but {names} all ask "
            "for it; OPT at one level changes what the level below sees, so the "
            "reference stream a second OPT level would need cannot be recorded in "
            "advance"
        )
    return levels[0]


def simulate_with_opt(
    trace_accesses: Sequence[tuple[int, bool]],
    config: Mapping[str, Any] | HierarchySpec,
) -> Hierarchy:
    """Run a hierarchy with Belady's OPT at the one level that asks for it.

    ``config`` is a configuration dict (or an already parsed
    ``HierarchySpec``) in which exactly one level has ``"policy": "opt"``;
    anything else is a ``ConfigError``. Returns the finished ``Hierarchy``.

    Two passes are needed because OPT has to know the reference stream of
    the level it runs at, and that stream is not the trace: an access
    reaches level L only if every level above it missed.

    Pass 1 runs the hierarchy with LRU substituted at level L and records
    the block of every probe L receives. Pass 2 rebuilds the hierarchy,
    preloads that stream into the OPT policy, and replays the trace.

    The recorded stream is exactly the stream the OPT run sees, not an
    approximation of it, and the reason is the model's structure:

    * Whether an access probes L is decided by the levels *above* L. A
      level's own hit/miss behaviour depends only on the addresses it is
      given and on what it holds, and levels here are non-inclusive and
      non-exclusive: nothing a lower level does can install, evict or
      invalidate a line above it, and every miss is answered with the same
      block no matter which level below supplies it. So the levels above L
      behave identically in both passes, and L is probed for the same
      blocks in the same order.
    * L's replacement policy changes only what L *answers*, which changes
      what L+1 sees -- below the level we are recording, never above it.
    * Write-backs arriving from L-1 are the one thing that reaches L
      without being a reference. They mark or allocate a line but are not
      lookups (they are not counted as accesses, hits or misses and no
      probe hook fires), and they are produced by L-1, so they replay
      identically too. OPT treats such a block as never referenced again,
      which is correct: no future demand reference in the recorded stream
      asks L for it.

    The same argument is why only one level may use OPT: OPT at L changes
    L's misses, hence the stream L+1 sees, so a stream recorded for L+1
    while L ran LRU would be wrong. Two OPT levels would need a fixed
    point, not two passes.
    """
    spec = config if isinstance(config, HierarchySpec) else parse_config(config)
    level_index = _opt_level(spec)
    target = spec.levels[level_index]

    # Pass 1: LRU everywhere, recording what the OPT level is asked for.
    lru_levels = tuple(
        replace(level, policy="lru") if i == level_index else level
        for i, level in enumerate(spec.levels)
    )
    first = Hierarchy.from_spec(replace(spec, levels=lru_levels))
    recorder = _ProbeRecorder(
        target.name, target.size, target.block_size, target.associativity, target.index
    )
    first.levels[level_index] = Level(recorder, target.hit_time)
    for addr, is_write in trace_accesses:
        first.access(addr, is_write)

    # Pass 2: the real thing, with the future in hand.
    hierarchy = Hierarchy.from_spec(spec)
    policy = hierarchy.levels[level_index].cache.policy
    if not isinstance(policy, OPTPolicy):  # pragma: no cover - registry sanity
        raise TypeError(f"policy {OPT!r} is not OPTPolicy but {type(policy).__name__}")
    policy.preload(recorder.probed)
    for addr, is_write in trace_accesses:
        hierarchy.access(addr, is_write)
    return hierarchy
