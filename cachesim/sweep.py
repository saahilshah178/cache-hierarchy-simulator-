"""Sweeps cache parameters and plots miss rate against each.

Two sweeps, both run on a SINGLE cache level backed directly by memory
(so the effect of the knob you are turning isn't blurred by L2/L3):

  1. associativity sweep — fixed cache size, associativity 1,2,4,8,16.
     Watch conflict misses melt away as ways increase, then flatten:
     past the knee, extra associativity buys nothing.
  2. size sweep — fixed associativity, size 1 KB ... 64 KB.
     Watch capacity misses disappear once the cache holds the working set.

With no trace argument, each sweep uses the trace that shows its effect
most clearly: conflict.trace for the associativity knee, matmul_naive.trace
for the size cliff. Pass a trace to run both sweeps on it instead —
NOTE: sweeping associativity on matmul_naive is a great experiment
precisely because it does NOT flatten nicely: its power-of-two stride
(N=64 doubles = 512 B rows) keeps aliasing to the same sets however the
cache is shaped. That pathology is why HPC code avoids power-of-two array
dimensions.

Usage::

    cachesim sweep                              # the two default sweeps
    cachesim sweep traces/pointer_chase.trace   # both sweeps on one trace
    cachesim sweep --size 16384 --policy fifo   # change the fixed knobs

Prints a table for each sweep and saves plots to plots/*.png (matplotlib).
"""

import argparse
import os

from cachesim.cache import Cache
from cachesim.hierarchy import Hierarchy, Level
from cachesim.policies import POLICIES
from cachesim.trace import parse_trace

ASSOCIATIVITIES = [1, 2, 4, 8, 16]
SIZES_KB = [1, 2, 4, 8, 16, 32, 64]


def run_single_level(trace, size, block_size, associativity, policy,
                     hit_time, mem_time):
    """Simulate `trace` (a list of (addr, is_write)) through one cache level.

    Returns (miss_rate, amat, cache) for that configuration.
    """
    cache = Cache("L1", size, block_size, associativity, policy)
    hier = Hierarchy([Level(cache, hit_time)], mem_time)
    for addr, is_write in trace:
        hier.access(addr, is_write)
    return cache.miss_rate, hier.amat(), cache


def sweep_associativity(trace, size, block_size, policy, hit_time, mem_time):
    print(f"\n=== associativity sweep (size fixed at {size // 1024} KB, "
          f"{policy.upper()}) ===")
    print(f"{'ways':>5} {'miss rate':>10} {'AMAT':>8}   "
          f"{'compulsory':>10} {'capacity':>9} {'conflict':>9}")
    results = []
    for ways in ASSOCIATIVITIES:
        miss_rate, amat, c = run_single_level(
            trace, size, block_size, ways, policy, hit_time, mem_time)
        results.append((ways, miss_rate))
        print(f"{ways:>5} {miss_rate:>9.2%} {amat:>8.2f}   "
              f"{c.compulsory_misses:>10,} {c.capacity_misses:>9,} "
              f"{c.conflict_misses:>9,}")
    return results


def sweep_size(trace, associativity, block_size, policy, hit_time, mem_time):
    print(f"\n=== size sweep (associativity fixed at {associativity}-way, "
          f"{policy.upper()}) ===")
    print(f"{'size':>7} {'miss rate':>10} {'AMAT':>8}   "
          f"{'compulsory':>10} {'capacity':>9} {'conflict':>9}")
    results = []
    for kb in SIZES_KB:
        miss_rate, amat, c = run_single_level(
            trace, kb * 1024, block_size, associativity, policy,
            hit_time, mem_time)
        results.append((kb, miss_rate))
        print(f"{kb:>4} KB {miss_rate:>9.2%} {amat:>8.2f}   "
              f"{c.compulsory_misses:>10,} {c.capacity_misses:>9,} "
              f"{c.conflict_misses:>9,}")
    return results


def plot(assoc_results, size_results, assoc_trace, size_trace,
         fixed_size, fixed_assoc, out_dir="plots"):
    """Save the two miss-rate plots. Skipped (with a note) if matplotlib
    is not installed — the tables above still tell the story."""
    try:
        import matplotlib
        matplotlib.use("Agg")           # no display needed
        import matplotlib.pyplot as plt
        from matplotlib.ticker import ScalarFormatter
    except ImportError:
        print("\n(matplotlib not installed — skipping plots. "
              "`pip install matplotlib` to get them.)")
        return

    os.makedirs(out_dir, exist_ok=True)

    def trace_base(path):
        return os.path.splitext(os.path.basename(path))[0]

    for results, xlabel, fixed_desc, trace_name, kind in [
        (assoc_results, "associativity (ways)",
         f"size fixed at {fixed_size // 1024} KB",
         assoc_trace, "associativity"),
        (size_results, "cache size (KB)",
         f"{fixed_assoc}-way fixed",
         size_trace, "size"),
    ]:
        base = trace_base(trace_name)
        xs = [r[0] for r in results]
        ys = [r[1] * 100 for r in results]
        fig, ax = plt.subplots(figsize=(7, 4.5))
        ax.plot(xs, ys, marker="o")
        ax.set_xscale("log", base=2)
        ax.set_xticks(xs)
        ax.get_xaxis().set_major_formatter(ScalarFormatter())
        ax.set_xlabel(xlabel)
        ax.set_ylabel("miss rate (%)")
        ax.set_ylim(bottom=0)
        ax.set_title(f"Miss rate vs {xlabel}\n{base} trace, {fixed_desc}")
        ax.grid(True, alpha=0.3)
        fname = f"miss_rate_vs_{kind}_{base}.png"
        fig.tight_layout()
        path = os.path.join(out_dir, fname)
        fig.savefig(path, dpi=120)
        plt.close(fig)
        print(f"saved {path}")


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(
        description="Sweep cache associativity and size; plot miss rates.")
    parser.add_argument("trace", nargs="?", default=None,
                        help="trace file for BOTH sweeps (default: "
                             "conflict.trace for the associativity sweep, "
                             "matmul_naive.trace for the size sweep)")
    parser.add_argument("--size", type=int, default=8 * 1024,
                        help="fixed size in bytes for the associativity sweep "
                             "(default 8192 = 8 KB)")
    parser.add_argument("--assoc", type=int, default=4,
                        help="fixed associativity for the size sweep (default 4)")
    parser.add_argument("--block-size", type=int, default=64,
                        help="block size in bytes (default 64)")
    parser.add_argument("--policy", default="lru", choices=sorted(POLICIES),
                        help="replacement policy (default lru)")
    parser.add_argument("--hit-time", type=int, default=4,
                        help="cache hit time in cycles (default 4)")
    parser.add_argument("--mem-time", type=int, default=100,
                        help="memory access time in cycles (default 100)")
    args = parser.parse_args(argv)

    # Validate the fixed knobs against every point we are about to sweep,
    # so a bad combination fails with one clear message instead of a
    # traceback halfway through.
    for ways in ASSOCIATIVITIES:
        if args.size % (args.block_size * ways):
            parser.error(
                f"--size {args.size} is not a multiple of "
                f"block_size*ways ({args.block_size}*{ways}); "
                "pick a power-of-two size")
    for kb in SIZES_KB:
        if (kb * 1024) % (args.block_size * args.assoc):
            parser.error(
                f"size {kb} KB is not a multiple of block_size*--assoc "
                f"({args.block_size}*{args.assoc}); "
                "pick a power-of-two associativity")

    assoc_trace_path = args.trace or "traces/conflict.trace"
    size_trace_path = args.trace or "traces/matmul_naive.trace"

    # Parse each trace once; every sweep point replays the same access list.
    assoc_trace = list(parse_trace(assoc_trace_path))
    print(f"associativity sweep trace: {assoc_trace_path} "
          f"({len(assoc_trace):,} accesses)")
    assoc_results = sweep_associativity(
        assoc_trace, args.size, args.block_size, args.policy,
        args.hit_time, args.mem_time)

    size_trace = (assoc_trace if size_trace_path == assoc_trace_path
                  else list(parse_trace(size_trace_path)))
    print(f"\nsize sweep trace: {size_trace_path} "
          f"({len(size_trace):,} accesses)")
    size_results = sweep_size(
        size_trace, args.assoc, args.block_size, args.policy,
        args.hit_time, args.mem_time)

    plot(assoc_results, size_results, assoc_trace_path, size_trace_path,
         args.size, args.assoc)
    return 0


if __name__ == "__main__":
    main()
