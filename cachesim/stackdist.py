"""Mattson stack-distance profiling: every LRU miss curve from one pass.

The *LRU stack distance* of a reference is the number of DISTINCT blocks
referenced since the previous reference to the same block. The first
reference to a block has infinite distance: no cache of any size could have
held it.

Mattson, Gecsei, Slutz and Traiger showed that LRU is a *stack algorithm*:
the contents of an LRU cache of C blocks are always a subset of the contents
of an LRU cache of C+1 blocks fed the same references (the inclusion
property). A reference therefore hits a fully-associative LRU cache of C
blocks exactly when its stack distance is finite and strictly less than C,
and one histogram of stack distances yields the miss count at *every*
capacity::

    misses(C) = (first touches) + (references with distance >= C)

The same argument applied per set gives every associativity at once: a
set-associative LRU cache is a bank of independent fully-associative LRU
caches, one per set, each fed the subsequence of references that map to it.
Profiling those subsequences separately makes ``misses(ways = k)`` exact for
all k from a single traversal.

Computing the distances is the expensive part. The naive "search the LRU
stack" method is O(N * D). This module uses the classic tree method: a
Fenwick tree (binary indexed tree) over reference timestamps holds a 1 at
the timestamp of the most recent reference to each distinct block, so the
number of distinct blocks referenced in a time interval is a range sum, and
the whole profile costs O(N log N) time and O(N) space.

References
----------
* Mattson, R. L., Gecsei, J., Slutz, D. R., Traiger, I. L., "Evaluation
  techniques for storage hierarchies", IBM Systems Journal 9(2):78-117, 1970
  -- stack algorithms, the inclusion property, one-pass success functions.
* Bennett, B. T., Kruskal, V. J., "LRU stack processing", IBM Journal of
  Research and Development 19(4):353-357, 1975 -- the tree formulation.
* Olken, F., "Efficient methods for calculating the success function of
  fixed space replacement policies", Report LBL-12370, Lawrence Berkeley
  Laboratory, 1981 -- O(N log N) stack-distance computation.
* Almasi, G., Cascaval, C., Padua, D. A., "Calculating stack distances
  efficiently", ACM SIGPLAN Workshop on Memory System Performance, 2002.
* Fenwick, P. M., "A new data structure for cumulative frequency tables",
  Software: Practice and Experience 24(3):327-336, 1994 -- the tree itself.
* Hill, M. D., Smith, A. J., "Evaluating associativity in CPU caches", IEEE
  Transactions on Computers 38(12):1612-1630, 1989 -- all-associativity
  simulation from one pass.

What is modelled
----------------
Demand references to a single, uniform block size, LRU replacement, and
write-allocate (a store and a load are the same event here, because both
bring the block into the cache). Write-backs, non-LRU policies, prefetching
and multi-level inclusion are not modelled: use ``cachesim.hierarchy`` for
those. The profile is exact, not sampled.
"""

from __future__ import annotations

import argparse
import csv
import os
import sys
from collections.abc import Iterable, Iterator, Sequence
from dataclasses import dataclass, field
from typing import Any

from cachesim.cliargs import positive_int, read_trace
from cachesim.plot import MatplotlibUnavailable, format_bytes, plot_mrc

#: Stack distance reported for the first reference to a block. Real
#: distances are >= 0, so any negative sentinel is unambiguous.
INFINITE = -1


class _Fenwick:
    """Binary indexed tree of ints over positions ``0 .. size-1``.

    Supports point update and prefix sum in O(log size). Positions are
    0-based on the outside and 1-based inside, which is what makes the
    ``i & -i`` stride work. ``prefix(-1)`` is 0, so an empty range needs no
    special case at the call site.
    """

    __slots__ = ("_size", "_tree")

    def __init__(self, size: int) -> None:
        self._size = size
        self._tree = [0] * (size + 1)

    def add(self, index: int, delta: int) -> None:
        """Add ``delta`` to position ``index``."""
        tree = self._tree
        size = self._size
        i = index + 1
        while i <= size:
            tree[i] += delta
            i += i & -i

    def prefix(self, index: int) -> int:
        """Sum of positions ``0 .. index`` inclusive; 0 when ``index < 0``."""
        tree = self._tree
        total = 0
        i = index + 1
        while i > 0:
            total += tree[i]
            i -= i & -i
        return total

    def range_sum(self, low: int, high: int) -> int:
        """Sum of positions ``low .. high`` inclusive; 0 when ``high < low``."""
        if high < low:
            return 0
        return self.prefix(high) - self.prefix(low - 1)


def block_stream(accesses: Iterable[tuple[int, bool]], block_size: int) -> list[int]:
    """Turn ``(address, is_write)`` pairs into the block-number stream.

    Reads and writes are treated alike: both reference the block. ``block_size``
    must be positive; it need not be a power of two here, although the
    simulator requires that.
    """
    if block_size <= 0:
        raise ValueError(f"block_size must be positive, got {block_size}")
    return [addr // block_size for addr, _ in accesses]


def stack_distances(blocks: Sequence[int]) -> Iterator[int]:
    """Yield the LRU stack distance of every reference in ``blocks``.

    Yields exactly ``len(blocks)`` values: ``INFINITE`` for the first
    reference to a block, otherwise the number of distinct *other* blocks
    referenced since that block was last referenced (so an immediately
    repeated reference has distance 0).

    The tree holds one 1 per distinct block, at the timestamp of its most
    recent reference. The distance for a reference at time ``i`` to a block
    last seen at ``prev`` is then the number of 1s strictly between them:
    exactly the distinct blocks touched in between, because every one of
    them has its single marker somewhere in that interval. The marker is
    moved from ``prev`` to ``i`` afterwards, which restores the invariant.

    Cost: O(N log N) time, O(N) space.
    """
    n = len(blocks)
    if n == 0:
        return
    tree = _Fenwick(n)
    last: dict[int, int] = {}
    for i, block in enumerate(blocks):
        prev = last.get(block, -1)
        if prev < 0:
            yield INFINITE
        else:
            yield tree.range_sum(prev + 1, i - 1)
            tree.add(prev, -1)
        tree.add(i, 1)
        last[block] = i


# -- histograms ---------------------------------------------------------------


@dataclass(frozen=True)
class ReuseBucket:
    """One power-of-two bucket of a reuse-distance histogram.

    Covers stack distances ``low .. high`` inclusive.
    """

    low: int
    high: int
    count: int


@dataclass(frozen=True)
class ReuseHistogram:
    """A stack-distance histogram bucketed by powers of two.

    Bucket 0 holds distance 0 (immediate reuse); bucket k >= 1 holds
    distances ``2**(k-1) .. 2**k - 1``. ``infinite`` counts first touches,
    which belong to no finite bucket.
    """

    buckets: tuple[ReuseBucket, ...]
    infinite: int
    references: int


@dataclass(frozen=True)
class StackDistanceProfile:
    """The stack-distance histogram of one block reference stream.

    ``histogram[d]`` is the number of references whose stack distance was
    exactly ``d``. ``infinite`` counts first touches, so it equals the number
    of distinct blocks in the stream, and::

        references == infinite + sum(histogram)

    The miss curve of a fully-associative LRU cache follows directly:
    ``misses(C) == infinite + (references at distance >= C)``.
    """

    references: int
    infinite: int
    histogram: tuple[int, ...]
    #: ``_tail[d]`` = references at distance >= d, so misses() is O(1).
    _tail: tuple[int, ...] = field(init=False, repr=False, compare=False, default=())

    def __post_init__(self) -> None:
        tail = [0] * (len(self.histogram) + 1)
        running = 0
        for d in range(len(self.histogram) - 1, -1, -1):
            running += self.histogram[d]
            tail[d] = running
        object.__setattr__(self, "_tail", tuple(tail))

    # -- shape ---------------------------------------------------------------

    @property
    def distinct_blocks(self) -> int:
        """Distinct blocks referenced; identical to ``infinite``."""
        return self.infinite

    @property
    def max_distance(self) -> int:
        """Largest finite stack distance seen, or -1 if there was no reuse."""
        return len(self.histogram) - 1

    @property
    def compulsory_ratio(self) -> float:
        """The miss-ratio floor: first touches / references."""
        return self.infinite / self.references if self.references else 0.0

    def reuses_at_least(self, distance: int) -> int:
        """References whose finite stack distance was >= ``distance``."""
        if distance < 0:
            distance = 0
        if distance >= len(self._tail):
            return 0
        return self._tail[distance]

    # -- the miss curve ------------------------------------------------------

    def misses(self, capacity: int) -> int:
        """Misses of a fully-associative LRU cache of ``capacity`` blocks.

        Exact, by the inclusion property: a reference hits iff its distance
        is finite and < ``capacity``. ``capacity == 0`` misses everything.
        """
        if capacity < 0:
            raise ValueError(f"capacity must be non-negative, got {capacity}")
        return self.infinite + self.reuses_at_least(capacity)

    def miss_ratio(self, capacity: int) -> float:
        """``misses(capacity) / references``; 0.0 for an empty stream."""
        return self.misses(capacity) / self.references if self.references else 0.0

    def miss_curve(self, capacities: Sequence[int]) -> list[int]:
        """Miss counts for each capacity in ``capacities``, in that order."""
        return [self.misses(c) for c in capacities]

    # -- derived summaries ---------------------------------------------------

    def working_set_size(self, tolerance: float = 0.01) -> int:
        """The smallest capacity whose miss ratio is within ``tolerance``
        of the compulsory floor.

        ``tolerance`` is an ABSOLUTE difference in miss ratio, so the default
        0.01 means "within one percentage point of the floor". The search
        starts at one block and always terminates: at capacity
        ``max_distance + 1`` every reuse hits and the floor is reached
        exactly. Returns 1 for a stream with no reuse.
        """
        if not 0.0 <= tolerance <= 1.0:
            raise ValueError(f"tolerance must be in [0, 1], got {tolerance}")
        if self.references == 0:
            return 1
        budget = tolerance * self.references
        for capacity in range(1, self.max_distance + 2):
            if self.reuses_at_least(capacity) <= budget:
                return capacity
        return max(self.max_distance + 1, 1)

    def reuse_histogram(self) -> ReuseHistogram:
        """Bucket the distance histogram by powers of two.

        Bucket 0 is distance 0; bucket k covers ``2**(k-1) .. 2**k - 1``.
        Buckets are emitted up to the largest distance observed, so an empty
        trailing bucket never appears.
        """
        counts: list[int] = []
        for d, count in enumerate(self.histogram):
            index = 0 if d == 0 else d.bit_length()
            if index >= len(counts):
                counts.extend([0] * (index + 1 - len(counts)))
            counts[index] += count
        buckets = tuple(
            ReuseBucket(
                low=0 if k == 0 else 1 << (k - 1),
                high=0 if k == 0 else (1 << k) - 1,
                count=count,
            )
            for k, count in enumerate(counts)
        )
        return ReuseHistogram(buckets=buckets, infinite=self.infinite, references=self.references)


def stack_distance_profile(blocks: Sequence[int]) -> StackDistanceProfile:
    """Profile ``blocks`` in one O(N log N) pass."""
    histogram: list[int] = []
    infinite = 0
    references = 0
    for distance in stack_distances(blocks):
        references += 1
        if distance == INFINITE:
            infinite += 1
            continue
        if distance >= len(histogram):
            histogram.extend([0] * (distance + 1 - len(histogram)))
        histogram[distance] += 1
    return StackDistanceProfile(
        references=references, infinite=infinite, histogram=tuple(histogram)
    )


def miss_ratio_curve(blocks: Sequence[int], capacities: Sequence[int]) -> list[int]:
    """Misses of a fully-associative LRU cache at each capacity, in one pass.

    Equivalent to running ``Cache(size=C*block_size, block_size=block_size,
    associativity=C, policy="lru")`` over the same stream once per capacity,
    and equal to ``Cache.shadow_misses`` of any set-associative cache holding
    C blocks, but computed for every capacity in a single traversal.
    """
    return stack_distance_profile(blocks).miss_curve(capacities)


# -- per-set profiling ---------------------------------------------------------


@dataclass(frozen=True)
class SetAssociativeProfile:
    """Within-set stack distances for a fixed number of sets.

    A set-associative LRU cache with ``num_sets`` sets behaves as
    ``num_sets`` independent fully-associative LRU caches of ``ways`` blocks,
    each fed the subsequence of references mapping to it. ``profile`` is the
    distance histogram of all those subsequences merged, which is all the
    miss curve for every associativity needs: a reference hits iff its
    within-set distance is finite and < ``ways``.

    ``per_set_references`` and ``per_set_distinct`` record how the stream
    spread over the sets, which is what a hot-set diagnostic reads.
    """

    num_sets: int
    profile: StackDistanceProfile
    per_set_references: tuple[int, ...]
    per_set_distinct: tuple[int, ...]

    @property
    def references(self) -> int:
        return self.profile.references

    @property
    def infinite(self) -> int:
        """First touches; the compulsory floor, independent of ``ways``."""
        return self.profile.infinite

    def misses(self, ways: int) -> int:
        """Misses of a ``num_sets`` x ``ways`` LRU cache over the stream."""
        return self.profile.misses(ways)

    def miss_ratio(self, ways: int) -> float:
        return self.profile.miss_ratio(ways)

    def miss_curve(self, ways_values: Sequence[int]) -> list[int]:
        """Miss counts for each associativity in ``ways_values``."""
        return [self.misses(w) for w in ways_values]


def per_set_profile(blocks: Sequence[int], num_sets: int) -> SetAssociativeProfile:
    """Profile ``blocks`` per set for a cache with ``num_sets`` sets.

    Blocks map to sets modulo ``num_sets``, exactly as ``Cache`` does. The
    per-set subsequences are profiled independently and their histograms
    merged, giving the miss count for every associativity at once. Total cost
    stays O(N log N); ``num_sets == 1`` reproduces the fully-associative
    profile exactly.
    """
    if num_sets < 1:
        raise ValueError(f"num_sets must be >= 1, got {num_sets}")
    buckets: list[list[int]] = [[] for _ in range(num_sets)]
    for block in blocks:
        buckets[block % num_sets].append(block)

    merged: list[int] = []
    infinite = 0
    references = 0
    distinct: list[int] = []
    for bucket in buckets:
        sub = stack_distance_profile(bucket)
        references += sub.references
        infinite += sub.infinite
        distinct.append(sub.infinite)
        if len(sub.histogram) > len(merged):
            merged.extend([0] * (len(sub.histogram) - len(merged)))
        for d, count in enumerate(sub.histogram):
            merged[d] += count

    return SetAssociativeProfile(
        num_sets=num_sets,
        profile=StackDistanceProfile(
            references=references, infinite=infinite, histogram=tuple(merged)
        ),
        per_set_references=tuple(len(b) for b in buckets),
        per_set_distinct=tuple(distinct),
    )


def power_of_two_capacities(limit: int) -> list[int]:
    """``[1, 2, 4, ...]`` up to and including the first value >= ``limit``.

    Used to pick a default set of capacities for a miss-ratio curve: the last
    point is guaranteed to reach the compulsory floor when ``limit`` is
    ``max_distance + 1``.
    """
    if limit < 1:
        return [1]
    capacities = [1]
    while capacities[-1] < limit:
        capacities.append(capacities[-1] * 2)
    return capacities


# -- the `mrc` command ---------------------------------------------------------


def _load_blocks(parser: argparse.ArgumentParser, path: str, block_size: int) -> list[int]:
    """Read a trace straight into a block stream, without the addresses.

    ``block_stream`` is handed the parse iterator rather than a list, so
    the addresses are converted to block numbers as they are read and only
    the block stream is ever resident: on matmul_naive.trace that is one
    list of 532,480 ints instead of that plus 532,480 pairs.
    """
    return read_trace(parser, path, lambda accesses: block_stream(accesses, block_size))


@dataclass(frozen=True)
class MrcRow:
    """One point of a miss-ratio curve, as tabulated and written to CSV.

    ``ways`` and ``num_sets`` describe the geometry that achieves this
    capacity: a fully-associative point is ``num_sets == 1`` with
    ``ways == capacity``. ``conflict`` is the Hill-Smith aggregate conflict
    count, ``misses`` minus the fully-associative misses at the same
    capacity, and is 0 by construction on a fully-associative row.
    """

    num_sets: int
    ways: int
    capacity: int
    capacity_bytes: int
    misses: int
    miss_ratio: float
    conflict: int


def mrc_rows(
    profile: StackDistanceProfile, capacities: Sequence[int], block_size: int
) -> list[MrcRow]:
    """Tabulate the fully-associative miss curve at ``capacities``."""
    return [
        MrcRow(
            num_sets=1,
            ways=capacity,
            capacity=capacity,
            capacity_bytes=capacity * block_size,
            misses=profile.misses(capacity),
            miss_ratio=profile.miss_ratio(capacity),
            conflict=0,
        )
        for capacity in capacities
    ]


def set_associative_rows(
    per_set: SetAssociativeProfile,
    flat: StackDistanceProfile,
    ways_values: Sequence[int],
    block_size: int,
) -> list[MrcRow]:
    """Tabulate the set-associative miss curve at ``ways_values``.

    Each row is compared against the fully-associative curve at the same
    total capacity, so ``conflict`` isolates the cost of the set mapping.
    """
    rows = []
    for ways in ways_values:
        capacity = per_set.num_sets * ways
        misses = per_set.misses(ways)
        rows.append(
            MrcRow(
                num_sets=per_set.num_sets,
                ways=ways,
                capacity=capacity,
                capacity_bytes=capacity * block_size,
                misses=misses,
                miss_ratio=per_set.miss_ratio(ways),
                conflict=misses - flat.misses(capacity),
            )
        )
    return rows


def write_csv(path: str, rows: Sequence[MrcRow]) -> None:
    """Write miss-curve rows as CSV: one header, one line per point."""
    directory = os.path.dirname(path)
    if directory:
        os.makedirs(directory, exist_ok=True)
    with open(path, "w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "sets",
                "ways",
                "capacity_blocks",
                "capacity_bytes",
                "misses",
                "miss_ratio",
                "conflict",
            ]
        )
        for row in rows:
            writer.writerow(
                [
                    row.num_sets,
                    row.ways,
                    row.capacity,
                    row.capacity_bytes,
                    row.misses,
                    f"{row.miss_ratio:.6f}",
                    row.conflict,
                ]
            )


def _print_curve(rows: Sequence[MrcRow], show_ways: bool) -> None:
    """Print a miss-curve table; ``show_ways`` adds the associativity columns."""
    if show_ways:
        print(
            f"{'ways':>5} {'capacity':>9} {'size':>9} "
            f"{'misses':>12} {'miss rate':>10} {'conflict':>10}"
        )
    else:
        print(f"{'capacity':>9} {'size':>9} {'misses':>12} {'miss rate':>10}")
    for row in rows:
        size = format_bytes(row.capacity_bytes)
        if show_ways:
            print(
                f"{row.ways:>5} {row.capacity:>9,} {size:>9} "
                f"{row.misses:>12,} {row.miss_ratio:>9.2%} {row.conflict:>10,}"
            )
        else:
            print(f"{row.capacity:>9,} {size:>9} {row.misses:>12,} {row.miss_ratio:>9.2%}")


def _print_reuse_histogram(profile: StackDistanceProfile) -> None:
    """Print the power-of-two reuse-distance histogram with a share bar."""
    histogram = profile.reuse_histogram()
    entries: list[tuple[str, int]] = []
    for bucket in histogram.buckets:
        label = f"{bucket.low}" if bucket.high == bucket.low else f"{bucket.low}-{bucket.high}"
        entries.append((label, bucket.count))
    entries.append(("infinite", histogram.infinite))
    widest = max((count for _, count in entries), default=0)
    print(f"{'distance':>12} {'refs':>12} {'share':>8}")
    for label, count in entries:
        share = count / profile.references if profile.references else 0.0
        bar = "#" * round(40 * count / widest) if widest else ""
        print(f"{label:>12} {count:>12,} {share:>7.2%}  {bar}")


def add_arguments(parser: argparse.ArgumentParser) -> None:
    """Add the ``mrc`` options to ``parser``."""
    parser.add_argument("trace", help="trace file (one 'ADDR R|W' per line)")
    parser.add_argument(
        "--block-size", type=positive_int, default=64, help="block size in bytes (default 64)"
    )
    parser.add_argument(
        "--sets",
        type=positive_int,
        default=None,
        metavar="N",
        help="also profile a cache with N sets, giving misses for every associativity",
    )
    parser.add_argument(
        "--max-capacity",
        type=positive_int,
        default=None,
        metavar="BLOCKS",
        help="largest capacity to tabulate, in blocks "
        "(default: the smallest power of two that reaches the compulsory floor)",
    )
    parser.add_argument("--csv", metavar="FILE", help="write the curve(s) to FILE as CSV")
    parser.add_argument("--plot", metavar="FILE", help="write the miss-ratio curve to FILE")


def run(args: argparse.Namespace, parser: argparse.ArgumentParser) -> int:
    """Execute the ``mrc`` command from parsed arguments."""
    blocks = _load_blocks(parser, args.trace, args.block_size)
    profile = stack_distance_profile(blocks)

    limit = profile.max_distance + 1
    if args.max_capacity is not None:
        limit = min(limit, args.max_capacity)
    capacities = [c for c in power_of_two_capacities(limit) if c <= (args.max_capacity or c)]

    print(f"trace: {args.trace}")
    print(
        f"references: {profile.references:,}   "
        f"distinct {args.block_size} B blocks: {profile.distinct_blocks:,}   "
        f"compulsory floor: {profile.compulsory_ratio:.2%}"
    )

    rows = mrc_rows(profile, capacities, args.block_size)
    print("\nfully-associative LRU miss curve")
    _print_curve(rows, show_ways=False)

    working_set = profile.working_set_size()
    print(
        f"\nworking set: {working_set:,} blocks "
        f"({format_bytes(working_set * args.block_size)}) "
        f"-- the smallest capacity within 1 percentage point of the floor"
    )

    print("\nreuse-distance histogram")
    _print_reuse_histogram(profile)

    set_rows: list[MrcRow] = []
    if args.sets is not None:
        per_set = per_set_profile(blocks, args.sets)
        ways_limit = per_set.profile.max_distance + 1
        if args.max_capacity is not None:
            ways_limit = min(ways_limit, max(args.max_capacity // args.sets, 1))
        ways_values = power_of_two_capacities(ways_limit)
        set_rows = set_associative_rows(per_set, profile, ways_values, args.block_size)
        print(f"\nset-associative LRU miss curve, {args.sets:,} sets")
        _print_curve(set_rows, show_ways=True)
        print(
            "conflict = misses minus the fully-associative misses at the same capacity "
            "(Hill-Smith aggregate)"
        )

    if args.csv:
        write_csv(args.csv, rows + set_rows)
        print(f"\nwrote {args.csv}")

    if args.plot:
        extra: list[tuple[str, Sequence[int], Sequence[float]]] = []
        if set_rows:
            extra.append(
                (
                    f"{args.sets:,}-set",
                    [row.capacity for row in set_rows],
                    [row.miss_ratio for row in set_rows],
                )
            )
        try:
            written = plot_mrc(
                [row.capacity for row in rows],
                [row.miss_ratio for row in rows],
                args.plot,
                title=f"Miss-ratio curve (LRU)\n{os.path.basename(args.trace)}, "
                f"{args.block_size} B blocks",
                extra=extra,
            )
        except MatplotlibUnavailable as exc:
            print(f"\n({exc})")
        else:
            print(f"wrote {written}")
    return 0


def register(subparsers: Any) -> None:
    """Register the ``mrc`` subcommand with the ``cachesim`` CLI."""
    parser = subparsers.add_parser(
        "mrc", help="miss-ratio curve: misses at every capacity, from one pass"
    )
    add_arguments(parser)
    parser.set_defaults(func=lambda args: run(args, parser))


def main(argv: Sequence[str] | None = None) -> int:
    """Stand-alone entry point (``python -m cachesim.stackdist``)."""
    parser = argparse.ArgumentParser(
        prog="cachesim mrc",
        description="Miss-ratio curve of a trace, from a Mattson stack-distance profile.",
    )
    add_arguments(parser)
    return run(parser.parse_args(argv), parser)


if __name__ == "__main__":
    sys.exit(main())
