"""``cachesim trace-stats``: what a trace contains, before any cache is involved.

The figures here are properties of the address stream alone, so they bound
what any cache can achieve on it:

* the number of distinct blocks is the compulsory-miss floor -- no cache,
  however large or associative, can miss fewer times than that on a cold
  start (M. D. Hill and A. J. Smith, "Evaluating Associativity in CPU
  Caches", IEEE Transactions on Computers 38(12), 1989);
* the same count times the block size is the trace's footprint, which says
  which level of a hierarchy could hold the whole working set;
* the distinct 4 KB pages bound TLB behaviour the same way;
* the first-touch fraction is that floor expressed as a miss rate: a
  simulated miss rate at or near it means the cache is already doing as
  well as any cache could.

Nothing here simulates anything, so it is fast enough to run on a trace
before deciding which geometries are worth sweeping.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Iterable, Sequence
from dataclasses import asdict, dataclass
from typing import Any

from cachesim.cliargs import parse_size
from cachesim.report import format_size
from cachesim.trace import FORMATS, open_trace

#: The page size the page count is reported at: 4 KB, the size on x86-64 and
#: the default on AArch64 Linux.
PAGE_SIZE = 4096


@dataclass(frozen=True)
class TraceStats:
    """Static properties of one trace at a given block size.

    ``first_touches`` counts the accesses that are the first reference to
    their block, which is by definition ``distinct_blocks``; it is reported
    as its own field because ``compulsory_fraction`` is its share of all
    accesses, the lowest miss rate any cache can achieve on this trace from
    a cold start.

    ``max_address`` is the largest address referenced, not the last byte
    touched: access size is not modelled (see ``cachesim.trace``). Both
    bounds are ``None`` for an empty trace.
    """

    accesses: int
    reads: int
    writes: int
    block_size: int
    distinct_blocks: int
    footprint_bytes: int
    page_size: int
    distinct_pages: int
    min_address: int | None
    max_address: int | None
    first_touches: int
    compulsory_fraction: float

    def to_dict(self) -> dict[str, Any]:
        """JSON-compatible dict of every field."""
        return asdict(self)


def analyse_trace(
    accesses: Iterable[tuple[int, bool]],
    block_size: int = 64,
    page_size: int = PAGE_SIZE,
) -> TraceStats:
    """Summarise an access stream without simulating a cache.

    ``accesses`` is the ``(address, is_write)`` stream the trace readers
    yield. Memory use is one entry per distinct block and per distinct page,
    not per access, so a long trace over a small footprint costs little.
    """
    if block_size <= 0 or page_size <= 0:
        raise ValueError(
            f"block_size and page_size must be positive, got {block_size}, {page_size}"
        )
    blocks: set[int] = set()
    pages: set[int] = set()
    count = writes = 0
    low: int | None = None
    high: int | None = None
    for addr, is_write in accesses:
        count += 1
        writes += is_write
        blocks.add(addr // block_size)
        pages.add(addr // page_size)
        if low is None or addr < low:
            low = addr
        if high is None or addr > high:
            high = addr
    return TraceStats(
        accesses=count,
        reads=count - writes,
        writes=writes,
        block_size=block_size,
        distinct_blocks=len(blocks),
        footprint_bytes=len(blocks) * block_size,
        page_size=page_size,
        distinct_pages=len(pages),
        min_address=low,
        max_address=high,
        first_touches=len(blocks),
        compulsory_fraction=len(blocks) / count if count else 0.0,
    )


def format_stats(stats: TraceStats, trace_name: str | None = None) -> str:
    """Render a ``TraceStats`` as the text report ``trace-stats`` prints."""
    s = stats
    span = (
        "n/a"
        if s.min_address is None or s.max_address is None
        else f"{s.max_address - s.min_address:,} B"
    )
    low = "n/a" if s.min_address is None else f"0x{s.min_address:012x}"
    high = "n/a" if s.max_address is None else f"0x{s.max_address:012x}"
    reads_pct = 100.0 * s.reads / s.accesses if s.accesses else 0.0
    writes_pct = 100.0 * s.writes / s.accesses if s.accesses else 0.0
    header = "TRACE STATISTICS"
    if trace_name:
        header += f"  ({trace_name})"
    return "\n".join(
        [
            header,
            f"  accesses        : {s.accesses:>12,}",
            f"  reads           : {s.reads:>12,}   ({reads_pct:5.1f}%)",
            f"  writes          : {s.writes:>12,}   ({writes_pct:5.1f}%)",
            f"  distinct blocks : {s.distinct_blocks:>12,}   "
            f"(footprint {format_size(s.footprint_bytes)} at {s.block_size} B blocks)",
            f"  distinct pages  : {s.distinct_pages:>12,}   ({format_size(s.page_size)} pages)",
            f"  address range   : {low} .. {high}   (span {span})",
            f"  first touches   : {s.first_touches:>12,}   "
            f"({100 * s.compulsory_fraction:5.2f}% of accesses: the compulsory-miss floor)",
        ]
    )


def add_arguments(parser: argparse.ArgumentParser) -> None:
    """Add the ``trace-stats`` options to ``parser``."""
    parser.add_argument("trace", help="trace file to summarise")
    parser.add_argument(
        "--block-size",
        type=parse_size,
        default=64,
        metavar="B",
        help="block size the footprint and first-touch counts assume "
        "(default 64; accepts a k/M suffix, e.g. 1k)",
    )
    parser.add_argument(
        "--trace-format",
        choices=("auto", *FORMATS),
        default="auto",
        help="format of the input trace (default: auto, chosen by extension)",
    )
    parser.add_argument(
        "--format",
        choices=("text", "json"),
        default="text",
        help="report format: human-readable text (default) or JSON",
    )


def run(args: argparse.Namespace) -> int:
    """Execute ``trace-stats``; exits with a message on a bad trace."""
    try:
        stats = analyse_trace(open_trace(args.trace, args.trace_format), block_size=args.block_size)
    except (OSError, ValueError) as exc:
        sys.exit(f"error: {exc}")
    if stats.accesses == 0:
        sys.exit(f"error: no accesses found in {args.trace}")
    if args.format == "json":
        print(json.dumps(stats.to_dict(), indent=2))
    else:
        print(format_stats(stats, trace_name=args.trace))
    return 0


def register(subparsers: Any) -> None:
    """Register the ``trace-stats`` subcommand with the ``cachesim`` CLI."""
    parser = subparsers.add_parser(
        "trace-stats", help="summarise a trace: footprint, pages, compulsory-miss floor"
    )
    add_arguments(parser)
    parser.set_defaults(func=run)


def main(argv: Sequence[str] | None = None) -> int:
    """Stand-alone entry point (``python -m cachesim.tracecmd``)."""
    parser = argparse.ArgumentParser(
        prog="cachesim trace-stats",
        description="Summarise a trace without simulating a cache.",
    )
    add_arguments(parser)
    return run(parser.parse_args(argv))


if __name__ == "__main__":
    sys.exit(main())
