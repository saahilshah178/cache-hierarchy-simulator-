"""``cachesim bench``: how fast the simulator itself is on a trace.

Every other subcommand measures the *simulated* machine. This one measures
the simulator: the wall time one pass of a trace through one hierarchy
costs, so that a change to the hot path (``Hierarchy.access``,
``Cache.probe``/``allocate``, the replacement policies) can be shown to have
made it faster rather than merely looking as if it should.

What it does, and why in that shape:

* The trace is parsed **once**, into a list, and the parse is timed
  separately. Parsing is a real cost of a run, but it is not the cost of
  simulating, and it is a third of the wall time of a default
  ``cachesim run``; folding the two together hides which one moved.
* Each repetition builds a **fresh** ``Hierarchy`` and replays the same
  list through it. Fresh, because a warm hierarchy misses less and would
  make repetition after the first measure a different workload; the same
  list, because re-parsing per repetition would drown the signal.
* The headline is the **best** of the repetitions, not the mean. The
  quantity of interest is how long the work takes, and everything the
  machine does besides this process can only add to a sample. The median
  is reported beside it as a check on how noisy the machine was: when the
  two differ by much, the number to distrust is the median.

``--format json`` prints the same figures, plus every individual timing,
for a script that records them across commits.
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from collections.abc import Sequence
from dataclasses import asdict, dataclass, field
from typing import Any

from cachesim.config import DEFAULT_CONFIG, ConfigError, load_config, parse_config
from cachesim.hierarchy import Hierarchy
from cachesim.opt import uses_opt
from cachesim.trace import FORMATS, load_trace

#: Repetitions when ``--repeat`` is not given.
DEFAULT_REPEAT = 3


@dataclass(frozen=True)
class BenchResult:
    """Timings of one benchmark: parsing once, then ``repeat`` simulations.

    ``accesses_per_second`` is derived from ``best_seconds``, so it is the
    throughput of the fastest observed pass -- the same choice, and for the
    same reason, as reporting the best time rather than the mean.
    """

    trace: str
    config: str | None
    accesses: int
    repeat: int
    parse_seconds: float
    best_seconds: float
    median_seconds: float
    accesses_per_second: float
    seconds: list[float] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        """JSON-compatible dict of every field."""
        return asdict(self)


def benchmark(
    accesses: Sequence[tuple[int, bool]],
    config: dict[str, Any],
    repeat: int = DEFAULT_REPEAT,
) -> list[float]:
    """Replay ``accesses`` through a fresh hierarchy ``repeat`` times.

    Returns one wall-clock duration per repetition, in seconds. Building
    the hierarchy is outside the timed region: it is a fixed cost of
    milliseconds that has nothing to do with the per-access hot path.
    """
    if repeat < 1:
        raise ValueError(f"repeat must be at least 1, got {repeat}")
    timings = []
    for _ in range(repeat):
        hierarchy = Hierarchy.from_config(config)
        access = hierarchy.access
        start = time.perf_counter()
        for addr, is_write in accesses:
            access(addr, is_write)
        timings.append(time.perf_counter() - start)
    return timings


def _format_result(result: BenchResult) -> str:
    """Render a ``BenchResult`` as the text report ``bench`` prints."""
    r = result
    per_access = 1e9 * r.best_seconds / r.accesses if r.accesses else 0.0
    return "\n".join(
        [
            f"BENCHMARK  ({r.trace})",
            f"  config          : {r.config or 'built-in default'}",
            f"  accesses        : {r.accesses:>12,}",
            f"  repetitions     : {r.repeat:>12,}",
            f"  parse           : {r.parse_seconds:>12.3f} s   (once, not in the times below)",
            f"  best            : {r.best_seconds:>12.3f} s",
            f"  median          : {r.median_seconds:>12.3f} s",
            f"  throughput      : {r.accesses_per_second:>12,.0f} accesses/s   "
            f"({per_access:.0f} ns/access, best)",
        ]
    )


def add_arguments(parser: argparse.ArgumentParser) -> None:
    """Add the ``bench`` options to ``parser``."""
    parser.add_argument("trace", help="trace file to replay")
    parser.add_argument(
        "--config",
        metavar="FILE",
        help="JSON hierarchy config (default: built-in L1/L2/L3 config, see configs/default.json)",
    )
    parser.add_argument(
        "--repeat",
        type=int,
        default=DEFAULT_REPEAT,
        metavar="N",
        help=f"simulation passes to time, each through a fresh hierarchy "
        f"(default {DEFAULT_REPEAT}); the best is the headline figure",
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
    """Execute ``bench``; exits with a message on a bad trace or config."""
    if args.repeat < 1:
        sys.exit(f"error: --repeat must be at least 1, got {args.repeat}")
    config = DEFAULT_CONFIG
    if args.config:
        try:
            config = load_config(args.config)
        except (OSError, json.JSONDecodeError, ConfigError) as exc:
            sys.exit(f"error: cannot load config {args.config}: {exc}")
    try:
        spec = parse_config(config)
    except ConfigError as exc:
        sys.exit(f"error: invalid config: {exc}")
    if uses_opt(spec):
        # `run` and `compare` simulate OPT by replaying the trace twice, once
        # to collect the future and once to use it. Timing that against an
        # online policy's single pass would compare two different amounts of
        # work, so bench declines rather than quietly measuring the wrong one.
        sys.exit(
            "error: bench times one replay, and the offline 'opt' policy needs two passes "
            "over the trace; its timings would not be comparable with an online policy's. "
            "Use `cachesim run` to simulate this config."
        )
    start = time.perf_counter()
    try:
        accesses = load_trace(args.trace, args.trace_format)
    except (OSError, ValueError) as exc:
        sys.exit(f"error: {exc}")
    parse_seconds = time.perf_counter() - start
    if not accesses:
        sys.exit(f"error: no accesses found in {args.trace}")
    try:
        timings = benchmark(accesses, config, repeat=args.repeat)
    except ConfigError as exc:
        sys.exit(f"error: invalid config: {exc}")
    best = min(timings)
    result = BenchResult(
        trace=args.trace,
        config=args.config,
        accesses=len(accesses),
        repeat=args.repeat,
        parse_seconds=parse_seconds,
        best_seconds=best,
        median_seconds=statistics.median(timings),
        accesses_per_second=len(accesses) / best if best > 0 else float("inf"),
        seconds=timings,
    )
    if args.format == "json":
        print(json.dumps(result.to_dict(), indent=2))
    else:
        print(_format_result(result))
    return 0


def register(subparsers: Any) -> None:
    """Register the ``bench`` subcommand with the ``cachesim`` CLI."""
    parser = subparsers.add_parser(
        "bench", help="time the simulator itself on a trace: accesses per second"
    )
    add_arguments(parser)
    parser.set_defaults(func=run)


def main(argv: Sequence[str] | None = None) -> int:
    """Stand-alone entry point (``python -m cachesim.benchcmd``)."""
    parser = argparse.ArgumentParser(
        prog="cachesim bench",
        description="Measure how fast the simulator replays a trace.",
    )
    add_arguments(parser)
    return run(parser.parse_args(argv))


if __name__ == "__main__":
    sys.exit(main())
