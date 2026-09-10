"""Compares every replacement policy on one trace against Belady's OPT.

A miss rate in isolation says nothing about a policy: 5% may be all the
trace allows, or twice what a better policy would achieve. This subcommand
runs one cache level -- the same geometry for everyone -- through every
policy in the registry plus the offline optimum, and reports how far each
one is from the optimum on that trace:

    cachesim policies traces/matmul_naive.trace --size 32768 --assoc 4

Two bounds are printed. The ``opt`` row is Belady's rule applied inside
each set, i.e. the best any policy could do *with this set mapping*; the
line below the table is the fully associative bound, the best any cache of
this capacity could do at all. The distance between the two is the price of
the set mapping, and no replacement policy can recover it -- more
associativity or a different index function is the only cure.

AMAT is the single-level case of the usual formula,
``hit_time + miss_rate * memory_access_time``.
"""

from __future__ import annotations

import argparse
import csv
import sys
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

from cachesim.cache import Cache
from cachesim.opt import OPT, opt_misses, run_opt
from cachesim.policies import POLICIES
from cachesim.trace import parse_trace


@dataclass(frozen=True)
class PolicyResult:
    """What one policy did on the trace, at one geometry."""

    policy: str
    accesses: int
    misses: int
    miss_rate: float
    amat: float
    gap_to_opt: float  # misses / OPT misses; 1.0 is optimal

    @property
    def row(self) -> tuple[str, int, int, float, float, float]:
        """The CSV row, in header order."""
        return (
            self.policy,
            self.accesses,
            self.misses,
            self.miss_rate,
            self.amat,
            self.gap_to_opt,
        )


CSV_HEADER = ("policy", "accesses", "misses", "miss_rate", "amat", "gap_to_opt")


def _run_policy(
    accesses: Sequence[tuple[int, bool]],
    policy: str,
    size: int,
    block_size: int,
    associativity: int,
) -> Cache:
    """One cache level, one policy, the whole trace. No 3-C shadow: this
    command compares policies, and the shadow would double the work."""
    if policy == OPT:
        return run_opt(accesses, size, block_size, associativity, track_3c=False)
    cache = Cache("L1", size, block_size, associativity, policy=policy, track_3c=False)
    for addr, is_write in accesses:
        cache.access(addr, is_write)
    return cache


def compare_policies(
    accesses: Sequence[tuple[int, bool]],
    size: int,
    block_size: int,
    associativity: int,
    hit_time: int,
    mem_time: int,
    policies: Sequence[str] | None = None,
) -> list[PolicyResult]:
    """Run every named policy (default: all registered) at one geometry.

    Returns the results sorted by miss count, so the best policy is first.
    ``gap_to_opt`` is measured against the ``opt`` row, which is always
    included and is a lower bound for every other row at this geometry:
    sets are independent, and inside a set Belady's rule is optimal.
    """
    names = list(policies) if policies is not None else sorted(POLICIES)
    if OPT not in names:
        names.append(OPT)

    caches = {name: _run_policy(accesses, name, size, block_size, associativity) for name in names}
    optimal = caches[OPT].misses
    results = []
    for name, cache in caches.items():
        miss_rate = cache.miss_rate
        results.append(
            PolicyResult(
                policy=name,
                accesses=cache.accesses,
                misses=cache.misses,
                miss_rate=miss_rate,
                amat=hit_time + miss_rate * mem_time,
                gap_to_opt=cache.misses / optimal if optimal else 1.0,
            )
        )
    # Best first; OPT leads a tie, since a policy that equals the bound is
    # news about that policy, not about the bound.
    results.sort(key=lambda r: (r.misses, r.policy != OPT, r.policy))
    return results


def format_table(
    results: Sequence[PolicyResult],
    fully_associative_bound: int | None = None,
    accesses: int = 0,
) -> str:
    """Render the comparison as a fixed-width table."""
    lines = [f"{'policy':<8} {'misses':>10} {'miss rate':>10} {'AMAT':>8} {'x OPT':>7}"]
    for r in results:
        lines.append(
            f"{r.policy:<8} {r.misses:>10,} {r.miss_rate:>9.2%} {r.amat:>8.2f} {r.gap_to_opt:>7.2f}"
        )
    if fully_associative_bound is not None:
        rate = f" ({fully_associative_bound / accesses:.2%})" if accesses else ""
        lines.append(
            f"\nfully associative Belady bound: {fully_associative_bound:,} misses{rate}"
            "  -- the set mapping costs everything above this"
        )
    return "\n".join(lines)


def write_csv(path: str, results: Sequence[PolicyResult]) -> None:
    """Write the results to ``path`` with a header row."""
    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(CSV_HEADER)
        for result in results:
            writer.writerow(result.row)


# -- command-line plumbing ------------------------------------------------------


def _positive_int(text: str) -> int:
    value = int(text)
    if value <= 0:
        raise argparse.ArgumentTypeError(f"must be a positive integer, got {text}")
    return value


def _non_negative_int(text: str) -> int:
    value = int(text)
    if value < 0:
        raise argparse.ArgumentTypeError(f"must be a non-negative integer, got {text}")
    return value


def add_arguments(parser: argparse.ArgumentParser) -> None:
    """Add the policy-comparison options to ``parser``."""
    parser.add_argument("trace", help="trace file (one 'ADDR R|W' per line)")
    parser.add_argument(
        "--size", type=_positive_int, default=32 * 1024, help="cache size in bytes (default 32768)"
    )
    parser.add_argument(
        "--assoc", type=_positive_int, default=4, help="associativity in ways (default 4)"
    )
    parser.add_argument(
        "--block-size", type=_positive_int, default=64, help="block size in bytes (default 64)"
    )
    parser.add_argument(
        "--hit-time", type=_non_negative_int, default=4, help="cache hit time in cycles (default 4)"
    )
    parser.add_argument(
        "--mem-time",
        type=_non_negative_int,
        default=100,
        help="memory access time in cycles (default 100)",
    )
    parser.add_argument("--csv", metavar="FILE", help="also write the table to FILE as CSV")


def run(args: argparse.Namespace, parser: argparse.ArgumentParser) -> int:
    """Execute a policy comparison from parsed arguments."""
    if args.size % (args.block_size * args.assoc):
        parser.error(
            f"--size {args.size} is not a multiple of --block-size * --assoc "
            f"({args.block_size} * {args.assoc})"
        )
    try:
        accesses = list(parse_trace(args.trace))
    except FileNotFoundError:
        parser.error(
            f"trace not found: {args.trace} (run `cachesim gen-traces` to create the samples)"
        )
    except (OSError, ValueError) as exc:
        parser.error(str(exc))
    if not accesses:
        parser.error(f"no accesses in {args.trace}")

    print(
        f"{args.trace}: {len(accesses):,} accesses through one level of "
        f"{args.size // 1024} KB, {args.assoc}-way, {args.block_size} B blocks "
        f"({args.size // (args.block_size * args.assoc)} sets)"
    )
    results = compare_policies(
        accesses,
        args.size,
        args.block_size,
        args.assoc,
        args.hit_time,
        args.mem_time,
    )
    bound = opt_misses(
        [addr // args.block_size for addr, _ in accesses], args.size // args.block_size
    )
    print(format_table(results, bound, len(accesses)))
    if args.csv:
        write_csv(args.csv, results)
        print(f"\nwrote {args.csv}")
    return 0


def register(subparsers: Any) -> None:
    """Register the ``policies`` subcommand with the ``cachesim`` CLI."""
    parser = subparsers.add_parser(
        "policies", help="compare every replacement policy on one trace against OPT"
    )
    add_arguments(parser)
    parser.set_defaults(func=lambda args: run(args, parser))


def main(argv: Sequence[str] | None = None) -> int:
    """Stand-alone entry point (``python -m cachesim.policycmp``)."""
    parser = argparse.ArgumentParser(
        prog="cachesim policies",
        description="Compare replacement policies on one trace against Belady's OPT.",
    )
    add_arguments(parser)
    return run(parser.parse_args(argv), parser)


if __name__ == "__main__":
    sys.exit(main())
