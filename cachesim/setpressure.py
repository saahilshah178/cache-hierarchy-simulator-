"""Per-set diagnostics: which sets take the misses, and when they are crowded.

"The stride hammers a few sets" is the usual explanation for a conflict-miss
pathology, and it is usually checked by looking at a histogram of misses per
set. That check is weaker than it looks. Over a whole trace the hot blocks
move, so the CUMULATIVE per-set miss count of a strided workload is often
close to uniform even when the workload is thrashing badly: at any instant a
few sets are overloaded, but which few changes as the loops advance, and
summing over time averages the evidence away.

This module measures both, and reports them separately:

Cumulative hot sets
    Simulate one cache level and tally misses by set index. Reports min,
    mean and max per set, the ratio of max to mean, and the share of all
    misses that landed in the hottest 5% of sets -- next to the share a
    perfectly uniform distribution would give those same sets, so "uniform"
    is a comparison rather than an impression.

Windowed set pressure
    Slice the trace into windows of W accesses and count, per window, how
    many DISTINCT blocks map to each set. A (window, set) pair is
    *oversubscribed* when that count exceeds the associativity: more blocks
    than ways wanted to be resident in that set at the same time, so at
    least one of them must have been evicted before it was reused. The
    fraction of oversubscribed pairs is the precise, measurable form of the
    informal claim, and it needs no simulation -- it is a property of the
    address stream and the set mapping alone.

The denominator for that fraction is the TOUCHED pairs, those where the set
saw at least one block in that window. Counting untouched sets would make
the number say more about how many sets exist than about how the workload
behaves: widening the cache would improve the figure without changing the
crowding at all.

Windows are consecutive and non-overlapping. A trailing partial window is
kept rather than discarded, so every access is accounted for; it can only
lower the distinct counts, never inflate them.

Both measurements use the same block mapping as ``Cache``: block =
address // block_size, set = block % num_sets.

Usage::

    cachesim sets traces/matmul_naive.trace
    cachesim sets traces/conflict.trace --size 4k --assoc 1
    cachesim sets traces/matmul_naive.trace --window 256 --format json
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

from cachesim.cache import Cache
from cachesim.cliargs import load_trace, non_negative_int, parse_size, positive_int
from cachesim.plot import format_bytes
from cachesim.policies import POLICIES

Accesses = Sequence[tuple[int, bool]]

#: Accesses per window for the pressure measurement.
DEFAULT_WINDOW = 512

#: Share of the sets counted as "the hottest" in the cumulative histogram.
DEFAULT_TOP_FRACTION = 0.05


@dataclass(frozen=True)
class HotSets:
    """Cumulative misses per set for one simulated cache level."""

    num_sets: int
    ways: int
    accesses: int
    misses: int
    per_set_misses: tuple[int, ...]
    top_fraction: float

    @property
    def max_misses(self) -> int:
        return max(self.per_set_misses, default=0)

    @property
    def min_misses(self) -> int:
        return min(self.per_set_misses, default=0)

    @property
    def mean_misses(self) -> float:
        return self.misses / self.num_sets if self.num_sets else 0.0

    @property
    def spread(self) -> float:
        """max / mean. 1.0 is perfectly uniform; 2.0 means the worst set took
        twice its share."""
        mean = self.mean_misses
        return self.max_misses / mean if mean else 0.0

    @property
    def top_sets(self) -> int:
        """How many sets make up the hottest ``top_fraction``, at least one."""
        return max(1, math.ceil(self.top_fraction * self.num_sets))

    @property
    def top_share(self) -> float:
        """Share of all misses that landed in the hottest ``top_sets`` sets."""
        if not self.misses:
            return 0.0
        hottest = sorted(self.per_set_misses, reverse=True)[: self.top_sets]
        return sum(hottest) / self.misses

    @property
    def uniform_share(self) -> float:
        """The share those same sets would hold if misses were spread evenly.

        This is ``top_sets / num_sets``, not ``top_fraction``: rounding up to
        a whole number of sets makes the two differ, and comparing against
        the wrong one would manufacture a small apparent imbalance.
        """
        return self.top_sets / self.num_sets if self.num_sets else 0.0

    def hottest(self, count: int) -> list[tuple[int, int]]:
        """The ``count`` busiest sets as ``(set_index, misses)``, busiest first.

        Ties break on the lower set index, so the output is deterministic.
        """
        order = sorted(range(self.num_sets), key=lambda idx: (-self.per_set_misses[idx], idx))
        return [(idx, self.per_set_misses[idx]) for idx in order[:count]]

    def to_dict(self) -> dict[str, Any]:
        return {
            "num_sets": self.num_sets,
            "ways": self.ways,
            "accesses": self.accesses,
            "misses": self.misses,
            "min": self.min_misses,
            "mean": self.mean_misses,
            "max": self.max_misses,
            "spread": self.spread,
            "top_fraction": self.top_fraction,
            "top_sets": self.top_sets,
            "top_share": self.top_share,
            "uniform_share": self.uniform_share,
        }


@dataclass(frozen=True)
class WindowPressure:
    """Distinct blocks per set within fixed-size windows of the trace."""

    window: int
    num_sets: int
    ways: int
    accesses: int
    windows: int
    touched_pairs: int
    oversubscribed_pairs: int
    windows_oversubscribed: int
    max_distinct: int
    total_distinct: int

    @property
    def oversubscribed_fraction(self) -> float:
        """Oversubscribed (window, set) pairs / TOUCHED (window, set) pairs."""
        return self.oversubscribed_pairs / self.touched_pairs if self.touched_pairs else 0.0

    @property
    def windows_oversubscribed_fraction(self) -> float:
        """Windows in which at least one set was oversubscribed."""
        return self.windows_oversubscribed / self.windows if self.windows else 0.0

    @property
    def mean_distinct(self) -> float:
        """Mean distinct blocks per TOUCHED set per window."""
        return self.total_distinct / self.touched_pairs if self.touched_pairs else 0.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "window": self.window,
            "num_sets": self.num_sets,
            "ways": self.ways,
            "windows": self.windows,
            "touched_pairs": self.touched_pairs,
            "oversubscribed_pairs": self.oversubscribed_pairs,
            "oversubscribed_fraction": self.oversubscribed_fraction,
            "windows_oversubscribed": self.windows_oversubscribed,
            "windows_oversubscribed_fraction": self.windows_oversubscribed_fraction,
            "mean_distinct": self.mean_distinct,
            "max_distinct": self.max_distinct,
        }


def hot_sets(
    accesses: Accesses,
    size: int,
    block_size: int,
    associativity: int,
    policy: str = "lru",
    top_fraction: float = DEFAULT_TOP_FRACTION,
) -> HotSets:
    """Simulate one cache level and tally its misses by set index.

    The level is backed directly by memory: a miss allocates the block, as
    ``Cache.access`` does. ``track_3c`` is off because the shadow cache
    plays no part here and doubles the cost.
    """
    if not 0.0 < top_fraction <= 1.0:
        raise ValueError(f"top_fraction must be in (0, 1], got {top_fraction}")
    cache = Cache(
        name="L1",
        size=size,
        block_size=block_size,
        associativity=associativity,
        policy=policy,
        track_3c=False,
    )
    num_sets = cache.num_sets
    per_set = [0] * num_sets
    for addr, is_write in accesses:
        block = addr // block_size
        if not cache.probe(block, is_write):
            per_set[block % num_sets] += 1
            cache.allocate(block, dirty=is_write)
    return HotSets(
        num_sets=num_sets,
        ways=associativity,
        accesses=cache.accesses,
        misses=cache.misses,
        per_set_misses=tuple(per_set),
        top_fraction=top_fraction,
    )


def window_pressure(
    accesses: Accesses,
    num_sets: int,
    block_size: int,
    ways: int,
    window: int = DEFAULT_WINDOW,
) -> WindowPressure:
    """Count distinct blocks per set within consecutive windows of ``window``
    accesses.

    No cache is simulated: the result depends only on the address stream and
    the set mapping, so it isolates the demand placed on each set from the
    replacement policy's response to it. A set that sees more distinct blocks
    than it has ways within one window cannot have held them all at once.
    """
    if num_sets < 1:
        raise ValueError(f"num_sets must be >= 1, got {num_sets}")
    if ways < 1:
        raise ValueError(f"ways must be >= 1, got {ways}")
    if window < 1:
        raise ValueError(f"window must be >= 1, got {window}")
    if block_size < 1:
        raise ValueError(f"block_size must be >= 1, got {block_size}")

    windows = 0
    touched = 0
    oversubscribed = 0
    windows_over = 0
    max_distinct = 0
    total_distinct = 0
    current: dict[int, set[int]] = {}

    def flush() -> None:
        nonlocal windows, touched, oversubscribed, windows_over, max_distinct, total_distinct
        if not current:
            return
        windows += 1
        any_over = False
        for blocks in current.values():
            distinct = len(blocks)
            touched += 1
            total_distinct += distinct
            if distinct > max_distinct:
                max_distinct = distinct
            if distinct > ways:
                oversubscribed += 1
                any_over = True
        if any_over:
            windows_over += 1
        current.clear()

    count = 0
    for addr, _ in accesses:
        if count and count % window == 0:
            flush()
        block = addr // block_size
        current.setdefault(block % num_sets, set()).add(block)
        count += 1
    flush()

    return WindowPressure(
        window=window,
        num_sets=num_sets,
        ways=ways,
        accesses=count,
        windows=windows,
        touched_pairs=touched,
        oversubscribed_pairs=oversubscribed,
        windows_oversubscribed=windows_over,
        max_distinct=max_distinct,
        total_distinct=total_distinct,
    )


# -- printing ------------------------------------------------------------------


def format_hot_sets(hot: HotSets, top: int) -> str:
    """Render the cumulative per-set miss histogram summary."""
    lines = ["cumulative misses per set"]
    lines.append(f"  misses            {hot.misses:>12,}  of {hot.accesses:,} accesses")
    lines.append(
        f"  per set           min {hot.min_misses:,}   "
        f"mean {hot.mean_misses:,.1f}   max {hot.max_misses:,}   "
        f"(max/mean {hot.spread:.2f})"
    )
    lines.append(
        f"  hottest {hot.top_fraction:.0%} of sets  {hot.top_sets} of {hot.num_sets} sets "
        f"hold {hot.top_share:.2%} of misses "
        f"(uniform would be {hot.uniform_share:.2%})"
    )
    if top and hot.misses:
        busiest = ", ".join(f"{idx} ({count:,})" for idx, count in hot.hottest(top))
        lines.append(f"  busiest sets      {busiest}")
    return "\n".join(lines)


def format_window_pressure(pressure: WindowPressure) -> str:
    """Render the windowed set-pressure summary."""
    lines = [f"windowed set pressure (windows of {pressure.window:,} accesses)"]
    lines.append(f"  windows           {pressure.windows:>12,}")
    lines.append(f"  touched pairs     {pressure.touched_pairs:>12,}  (window, set) pairs in use")
    ways = f"{pressure.ways} way" + ("" if pressure.ways == 1 else "s")
    lines.append(
        f"  oversubscribed    {pressure.oversubscribed_pairs:>12,}  "
        f"= {pressure.oversubscribed_fraction:.2%} of touched pairs "
        f"(distinct blocks > {ways})"
    )
    lines.append(
        f"  windows affected  {pressure.windows_oversubscribed:>12,}  "
        f"= {pressure.windows_oversubscribed_fraction:.2%} of windows"
    )
    lines.append(
        f"  distinct blocks per touched set: mean {pressure.mean_distinct:.2f}, "
        f"max {pressure.max_distinct:,}"
    )
    return "\n".join(lines)


# -- CLI -----------------------------------------------------------------------


def add_arguments(parser: argparse.ArgumentParser) -> None:
    """Add the ``sets`` options to ``parser``."""
    parser.add_argument("trace", help="trace file (one 'ADDR R|W' per line)")
    parser.add_argument(
        "--size", type=parse_size, default=32 * 1024, help="cache size (default 32k)"
    )
    parser.add_argument(
        "--block-size", type=parse_size, default=64, help="block size in bytes (default 64)"
    )
    parser.add_argument("--assoc", type=positive_int, default=4, help="ways per set (default 4)")
    parser.add_argument(
        "--policy", default="lru", choices=sorted(POLICIES), help="replacement policy (default lru)"
    )
    parser.add_argument(
        "--window",
        type=positive_int,
        default=DEFAULT_WINDOW,
        metavar="W",
        help=f"accesses per pressure window (default {DEFAULT_WINDOW})",
    )
    parser.add_argument(
        "--top",
        type=non_negative_int,
        default=8,
        metavar="N",
        help="list the N busiest sets (default 8; 0 to omit)",
    )
    parser.add_argument(
        "--format",
        choices=("text", "json"),
        default="text",
        help="report format: text summary (default) or JSON",
    )


def run(args: argparse.Namespace, parser: argparse.ArgumentParser) -> int:
    """Execute the ``sets`` command from parsed arguments."""
    accesses = load_trace(parser, args.trace)
    try:
        hot = hot_sets(accesses, args.size, args.block_size, args.assoc, args.policy)
    except ValueError as exc:
        parser.error(str(exc))
    pressure = window_pressure(accesses, hot.num_sets, args.block_size, args.assoc, args.window)

    if args.format == "json":
        print(
            json.dumps(
                {
                    "trace": args.trace,
                    "geometry": {
                        "size": args.size,
                        "block_size": args.block_size,
                        "associativity": args.assoc,
                        "num_sets": hot.num_sets,
                        "policy": args.policy,
                    },
                    "hot_sets": hot.to_dict(),
                    "window_pressure": pressure.to_dict(),
                },
                indent=2,
            )
        )
        return 0

    print(f"trace: {args.trace} ({len(accesses):,} accesses)")
    print(
        f"geometry: {format_bytes(args.size)}, {args.block_size} B blocks, "
        f"{args.assoc}-way, {args.policy.upper()} -> {hot.num_sets:,} sets\n"
    )
    print(format_hot_sets(hot, args.top))
    print()
    print(format_window_pressure(pressure))
    return 0


def register(subparsers: Any) -> None:
    """Register the ``sets`` subcommand with the ``cachesim`` CLI."""
    parser = subparsers.add_parser(
        "sets", help="per-set diagnostics: hot sets and windowed set pressure"
    )
    add_arguments(parser)
    parser.set_defaults(func=lambda args: run(args, parser))


def main(argv: Sequence[str] | None = None) -> int:
    """Stand-alone entry point (``python -m cachesim.setpressure``)."""
    parser = argparse.ArgumentParser(
        prog="cachesim sets",
        description="Per-set miss distribution and windowed set pressure for one trace.",
    )
    add_arguments(parser)
    return run(parser.parse_args(argv), parser)


if __name__ == "__main__":
    sys.exit(main())
