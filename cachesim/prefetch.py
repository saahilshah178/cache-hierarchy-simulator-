"""Hardware prefetchers attached to a cache level.

A prefetcher watches the demand references that reach its level and, for
each one, may name a single block to fetch in advance (degree 1). The
level then installs that block with a ``prefetched`` flag, so the first
demand hit on it can be recognised as a prefetch that paid off.

Prefetchers are selected by name through the ``PREFETCHERS`` registry and
built with ``make_prefetcher``; ``"none"`` builds nothing.

What the model does and does not capture
----------------------------------------

* Prefetches are off the critical path: they are counted but charged no
  cycles, as if the fetch overlapped with useful work. That is optimistic
  -- a real prefetch competes for MSHRs, bus cycles and DRAM banks --
  which is why the *pollution* cost (a prefetched line evicting a useful
  one) is the part this model gets right, and the *timeliness* question
  (did it arrive before the demand?) the part it cannot ask at all.
* A prefetch is issued only for a block the level does not already hold,
  so a prefetcher never re-fetches a resident block.
* The lookup that fetches the block runs below the issuing level without
  disturbing it: lower levels report those lookups separately from demand
  traffic and do not update their replacement state for them. Keeping
  prefetch traffic out of the demand counters is what keeps every level's
  miss rate a statement about demand references, and so keeps the AMAT
  identity exact.
"""

from __future__ import annotations

from collections.abc import Callable


class Prefetcher:
    """Base class: watches references at one level and predicts one block.

    ``predict`` is called once per demand reference that reaches the level,
    hit or miss, after the reference has been resolved. It returns the
    block number to prefetch, or None. The caller drops predictions the
    level already holds and negative block numbers.
    """

    #: Name under which the prefetcher is selected in a configuration.
    name = "none"

    def predict(self, block: int, hit: bool, was_prefetched: bool) -> int | None:
        """Observe a reference to ``block`` and optionally name a prefetch.

        ``hit`` is whether the reference hit at this level, and
        ``was_prefetched`` whether it was the first demand hit on a line
        this prefetcher had brought in.
        """
        raise NotImplementedError

    def reset(self) -> None:
        """Forget all prediction state."""


class NextLinePrefetcher(Prefetcher):
    """Tagged next-line (one-block-lookahead) prefetching.

    On a demand miss for block b, or on the first demand hit to a line
    that was itself prefetched, fetch b+1. The tag is what makes a stream
    keep running: without it a scan prefetches one block ahead of the
    first miss and then stops, because the subsequent references all hit;
    with it, each prefetched line triggers the next prefetch as the stream
    reaches it, so the prefetcher stays exactly one block ahead.

    It is the cheapest prefetcher there is -- no history, no table -- and
    it captures sequential streams completely and everything else not at
    all. Described by Smith, "Cache Memories", ACM Computing Surveys 14(3),
    1982, section 2.4; the tagged variant is due to Gindele, "Buffer Block
    Prefetching Method", IBM Technical Disclosure Bulletin 20(2), 1977.
    """

    name = "next-line"

    def predict(self, block: int, hit: bool, was_prefetched: bool) -> int | None:
        if hit and not was_prefetched:
            return None
        return block + 1


class StridePrefetcher(Prefetcher):
    """Per-region stride detection with a 2-bit confidence counter.

    A reference table of ``entries`` slots is indexed by the address
    region ``block >> region_shift`` -- 4 KB regions for 64-byte blocks at
    the default shift of 6 -- and each slot holds the region tag, the last
    block referenced in that region, the last delta between consecutive
    references there, and a 2-bit saturating confidence counter. A delta
    that repeats raises the confidence; a different delta replaces the
    recorded stride and drops the confidence back to "seen once". Once the
    same stride has been seen twice (confidence 2 of 3) the prefetcher
    predicts ``block + stride``.

    Limits of indexing by region rather than by program counter, which is
    what the reference-prediction table of Chen and Baer, "Effective
    Hardware-Based Data Prefetching for High-Performance Processors", IEEE
    Transactions on Computers 44(5), 1995, actually uses:

    * An address trace carries no PCs, so there is nothing else to index
      by. Two loops striding through the same 4 KB region interleave into
      one slot and cancel each other out, where a PC-indexed table would
      track them separately.
    * A stride wider than the region is never detected: consecutive
      references land in different slots, each of which sees only its
      first reference. This is the common case for large strided scans,
      and it is why the region shift is a parameter.
    * The table is direct-mapped and tagged, so regions that alias evict
      each other's history, exactly as the hardware structure does.
    """

    name = "stride"

    #: Confidence at which a stride is trusted, out of the 2-bit maximum 3.
    THRESHOLD = 2

    def __init__(self, entries: int = 256, region_shift: int = 6) -> None:
        if entries < 1:
            raise ValueError(f"stride prefetcher needs at least one entry, got {entries}")
        if region_shift < 0:
            raise ValueError(f"region_shift must be non-negative, got {region_shift}")
        self.entries = entries
        self.region_shift = region_shift
        # Per slot: [region tag, last block, last stride, confidence].
        # A tag of -1 marks an empty slot; real regions are >= 0.
        self._table = [[-1, 0, 0, 0] for _ in range(entries)]

    def predict(self, block: int, hit: bool, was_prefetched: bool) -> int | None:
        region = block >> self.region_shift
        slot = self._table[region % self.entries]
        if slot[0] != region:
            slot[0], slot[1], slot[2], slot[3] = region, block, 0, 0
            return None
        stride = block - slot[1]
        slot[1] = block
        if stride == 0:
            return None  # the same block again tells us nothing new
        if stride == slot[2]:
            if slot[3] < 3:
                slot[3] += 1
        else:
            slot[2], slot[3] = stride, 1  # this stride has now been seen once
        if slot[3] < self.THRESHOLD:
            return None
        return block + stride

    def reset(self) -> None:
        for slot in self._table:
            slot[0], slot[1], slot[2], slot[3] = -1, 0, 0, 0


#: Registry of selectable prefetchers, keyed by the name used in configs.
PREFETCHERS: dict[str, Callable[[], Prefetcher]] = {
    "next-line": NextLinePrefetcher,
    "stride": StridePrefetcher,
}

#: Every accepted value of a level's ``prefetcher`` key, ``"none"`` included.
PREFETCHER_NAMES = ("none", *sorted(PREFETCHERS))


def make_prefetcher(name: str) -> Prefetcher | None:
    """Build the named prefetcher, or None for ``"none"``.

    Raises ``ValueError`` for an unknown name.
    """
    if name == "none":
        return None
    try:
        return PREFETCHERS[name]()
    except KeyError:
        raise ValueError(
            f"unknown prefetcher {name!r}; choose from {', '.join(PREFETCHER_NAMES)}"
        ) from None
